// src/utils/sendApi.ts
import { platformFetch } from './platform';
import { InferenceParams } from '../config/inference-params';
import type { AssistantResponse, ToolCall, WireToolSpec } from '../mcp/types';


/**
 * Handles streaming response from the API
 * @param response The fetch response object
 * @param onStreamChunk Optional callback for each chunk
 * @returns The complete message content
 */
async function handleStreamingResponse(response: Response, onStreamChunk?: (chunk: string) => void, onReasoningChunk?: (chunk: string) => void): Promise<string> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body available for streaming');
  }

  const decoder = new TextDecoder();
  let fullContent = '';
  let buffer = '';

  /** Apply one complete SSE line. Returns true when the stream signalled [DONE]. */
  const consume = (line: string): boolean => {
    if (!line.startsWith('data: ')) return false;
    const data = line.slice(6); // Remove 'data: ' prefix

    if (data === '[DONE]') {
      //console.log('🏁 Stream finished');
      return true;
    }

    try {
      const parsed = JSON.parse(data);
      const content = parsed.choices?.[0]?.delta?.content;
      const reasoning = parsed.choices?.[0]?.delta?.reasoning;

      if (reasoning && onReasoningChunk) {
        onReasoningChunk(reasoning);
      }

      if (content) {
        //console.log('📝 Token:', content);
        fullContent += content;
        // Call the callback with the new content if provided
        if (onStreamChunk) {
          onStreamChunk(content);
        }
      }
    } catch (parseError) {
      //console.warn('Failed to parse streaming chunk:', data);
    }
    return false;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        //console.log('🎯 Stream completed');
        // A stream that ends without a trailing newline leaves its last event buffered.
        buffer += decoder.decode();
        if (buffer.trim() !== '') consume(buffer.replace(/\r$/, '').trim());
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const { lines, rest } = takeCompleteLines(buffer);
      buffer = rest;

      for (const line of lines) {
        if (consume(line)) return fullContent;
      }
    }
  } finally {
    reader.releaseLock();
  }

  return fullContent;
}

/**
 * Split an SSE buffer into COMPLETE lines, leaving any trailing partial line behind.
 *
 * A network read is not aligned to line boundaries: a single `data: {...}` event — Gemini
 * emits whole tool calls in one line, easily several KB — is routinely split across two
 * reads. Parsing each read in isolation makes both halves unparseable JSON, and since the
 * parse error is swallowed the event vanishes: the model's tool call never reaches the
 * runner and the turn looks like the model stopped saying nothing. Keeping the remainder
 * buffered until its newline arrives is what makes the stream lossless.
 */
function takeCompleteLines(buffer: string): { lines: string[]; rest: string } {
  const parts = buffer.split('\n');
  const rest = parts.pop() ?? '';
  // \r survives on CRLF streams and would break the `data: [DONE]` compare.
  const lines = parts.map(l => l.replace(/\r$/, '')).filter(l => l.trim() !== '');
  return { lines, rest };
}

/**
 * Scan from offset and return the end index (exclusive) of the first balanced JSON
 * object/array starting there, accounting for strings and escapes. Returns -1 if there
 * is no complete value (unbalanced).
 */
function endOfFirstJsonValue(s: string, start: number): number {
  let depth = 0;
  let inStr = false;
  let escaped = false;
  for (let i = start; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === '{' || ch === '[') depth++;
    else if (ch === '}' || ch === ']') {
      depth--;
      if (depth === 0) return i + 1;
    }
  }
  return -1;
}

/**
 * Normalize a streamed tool-call `arguments` string into a single valid JSON object string.
 *
 * Gemini's OpenAI-compat streaming repeats the FULL arguments object in multiple delta
 * chunks rather than splitting it into fragments, so naive concatenation yields strings
 * like `{}{}{}` (zero-arg tools) or `{...}{...}` (repeated payloads). Those are invalid
 * JSON, and when such an assistant message is replayed in the next request, Gemini 400s
 * trying to parse the function arguments. We keep concatenation (correct for real OpenAI
 * fragment streams) and repair the result here: if it doesn't parse, take the first
 * complete JSON value. Always returns a parseable object string.
 */
function normalizeToolArguments(raw: string): string {
  const s = (raw || '').trim();
  if (!s) return '{}';
  try {
    JSON.parse(s);
    return s; // already valid (the normal case)
  } catch {
    // fall through to repair
  }
  const end = endOfFirstJsonValue(s, 0);
  if (end > 0) {
    const first = s.slice(0, end);
    try {
      JSON.parse(first);
      return first;
    } catch {
      // ignore — fall through
    }
  }
  return '{}';
}

/**
 * Streaming handler for tool-calling requests. A single turn may interleave prose and
 * tool-call fragments; prose is forwarded via onStreamChunk while tool_calls are
 * accumulated per call (concatenating function.arguments fragments), keyed by `index`
 * when the provider sends one and by `id` when it doesn't (Gemini).
 */
async function handleStreamingResponseWithTools(
  response: Response,
  onStreamChunk?: (chunk: string) => void,
  onReasoningChunk?: (chunk: string) => void,
): Promise<AssistantResponse> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body available for streaming');
  }

  const decoder = new TextDecoder();
  let fullContent = '';
  let finishReason: string | undefined;
  let buffer = '';

  // One accumulator per tool call, in the order the calls first appeared on the stream.
  type CallAccumulator = { id?: string; name?: string; arguments: string; extra_content?: any };
  const calls: CallAccumulator[] = [];
  const callByKey = new Map<string, CallAccumulator>();

  /**
   * Route a tool_call delta to its accumulator. OpenAI keys fragments by `index`; Gemini's
   * compat endpoint omits `index` entirely (it lives on the choice) and instead sends each
   * call whole with a unique `id` — so keying on index alone collapsed two parallel calls
   * into one slot and concatenated their arguments into unparseable JSON.
   */
  const accumulatorFor = (tc: any): CallAccumulator => {
    const key = typeof tc.index === 'number' ? `i:${tc.index}` : tc.id ? `id:${tc.id}` : null;
    if (key === null) {
      // Neither key present: a bare follow-up chunk (Gemini sometimes ships the
      // thought_signature this way) belongs to the call already in flight.
      if (calls.length === 0) {
        const fresh: CallAccumulator = { arguments: '' };
        calls.push(fresh);
        return fresh;
      }
      return calls[calls.length - 1];
    }
    let acc = callByKey.get(key);
    if (!acc) {
      acc = { arguments: '' };
      callByKey.set(key, acc);
      calls.push(acc);
    }
    return acc;
  };

  const finalize = (): AssistantResponse => {
    const tool_calls: ToolCall[] = calls.map((acc, i) => ({
      id: acc.id || `call_${i}`,
      type: 'function' as const,
      function: {
        name: acc.name || '',
        arguments: normalizeToolArguments(acc.arguments),
      },
      // Echo back Gemini's thought_signature (and any other provider passthrough) so the
      // replayed history stays valid for 2.5+ thinking models.
      ...(acc.extra_content ? { extra_content: acc.extra_content } : {}),
    }));
    return {
      content: fullContent,
      tool_calls: tool_calls.length > 0 ? tool_calls : undefined,
      finish_reason: finishReason,
    };
  };

  /** Apply one complete SSE line. Returns true when the stream signalled [DONE]. */
  const consume = (line: string): boolean => {
    if (!line.startsWith('data: ')) return false;
    const data = line.slice(6);
    if (data === '[DONE]') return true;

    try {
      const parsed = JSON.parse(data);
      const choice = parsed.choices?.[0];
      if (!choice) return false;

      const delta = choice.delta || {};

      if (delta.reasoning && onReasoningChunk) {
        onReasoningChunk(delta.reasoning);
      }

      if (delta.content) {
        fullContent += delta.content;
        if (onStreamChunk) onStreamChunk(delta.content);
      }

      if (Array.isArray(delta.tool_calls)) {
        for (const tc of delta.tool_calls) {
          const acc = accumulatorFor(tc);
          if (tc.id) acc.id = tc.id;
          if (tc.function?.name) acc.name = tc.function.name;
          if (tc.function?.arguments) acc.arguments += tc.function.arguments;
          // Gemini 2.5+ may deliver the thought_signature in any chunk for this call
          // (sometimes a content-less one), so capture it whenever it appears.
          if (tc.extra_content) acc.extra_content = tc.extra_content;
        }
      }

      if (choice.finish_reason) finishReason = choice.finish_reason;
    } catch (parseError) {
      // A complete line that still won't parse is a real anomaly now that partial
      // lines are buffered — surface it instead of silently losing the event.
      console.warn('Dropped unparseable stream event:', data.slice(0, 200));
    }
    return false;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        // A stream that ends without a trailing newline leaves its last event buffered.
        buffer += decoder.decode();
        if (buffer.trim() !== '') consume(buffer.replace(/\r$/, '').trim());
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const { lines, rest } = takeCompleteLines(buffer);
      buffer = rest;

      for (const line of lines) {
        if (consume(line)) return finalize();
      }
    }
  } finally {
    reader.releaseLock();
  }

  return finalize();
}

/**
 * Hosted System One call (cloud Jev): POST {server}/v1/systemone. Returns the `answers` map.
 * Same auth and error contract as fetchResponse: 429 → UnauthorizedError.
 */
export async function fetchDecision(
  serverAddress: string,
  request: { model: string; state: unknown; questions: unknown },
  token?: string,
): Promise<Record<string, any>> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (serverAddress.includes('api.observer-ai.com') && token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await platformFetch(`${serverAddress}/v1/systemone`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ model: request.model, state: request.state, questions: request.questions }),
  });

  if (response.status === 429) {
    throw new UnauthorizedError('Access denied. Quota may be exceeded.');
  }
  if (!response.ok) {
    const body = await response.text();
    let detail = '';
    try { detail = JSON.parse(body).detail ?? ''; } catch { if (body.length < 200) detail = body; }
    throw new Error(`API error: ${response.status}${detail ? ` - ${detail}` : ''}`);
  }
  const data = await response.json();
  return data.answers;
}

export class UnauthorizedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UnauthorizedError';
  }
}

/**
 * Direct fetch to a specific server without model discovery
 * @param serverAddress Full server address (e.g., 'https://api.observer-ai.com:443')
 * @param messages Array of OpenAI-format messages with role and content
 * @param modelName Name of the model to use
 * @param token Optional authorization token (used for Observer API only)
 * @param enableStreaming Whether to enable streaming response (default: false)
 * @param onStreamChunk Optional callback for streaming chunks
 * @param inferenceParams Optional inference parameters (temperature, top_p, etc.)
 *        - For Observer API: token param is used for auth
 *        - For custom servers: inferenceParams.customApiKey is used as Bearer token
 * @returns The model's response text
 */
// True when a string is empty or contains only whitespace / zero-width / BOM chars.
// Gemini maps such an assistant message to a Content with no `parts` and 400s with
// "contents.parts must not be empty". Note a zero-width space (U+200B) counts as empty:
// Gemini normalizes Unicode format chars away, so it is NOT a usable non-empty placeholder.
const EMPTYISH = /^[\s\u200B\u200C\u200D\uFEFF]*$/;

/**
 * Make assistant messages API-safe. An assistant turn with no tool_calls MUST carry
 * non-empty text or Gemini's OpenAI-compat endpoint rejects the whole request. Models
 * (e.g. gemini-flash-lite after a tool result) sometimes return empty content; replace
 * that with a single space, which Gemini accepts as a non-empty part. Assistant messages
 * WITH tool_calls are left alone — their functionCall part satisfies the non-empty rule,
 * so null/empty content there is valid.
 */
function sanitizeAssistantContent(messages: Array<{role: string, content: any}>): Array<{role: string, content: any}> {
  return messages.map(msg => {
    const hasToolCalls = Array.isArray((msg as any).tool_calls) && (msg as any).tool_calls.length > 0;
    if (msg.role !== 'assistant' || hasToolCalls) return msg;
    if (typeof msg.content === 'string' && !EMPTYISH.test(msg.content)) return msg;
    // Empty string, null, or only-whitespace/zero-width → unusable for Gemini.
    if (typeof msg.content === 'string' || msg.content == null) {
      return { ...msg, content: ' ' };
    }
    return msg;
  });
}

/**
 * Convert native image format to OpenAI format for external APIs.
 * Native: { type: 'image', image: 'data:...' }
 * OpenAI: { type: 'image_url', image_url: { url: 'data:...' } }
 */
function convertToOpenAIFormat(messages: Array<{role: string, content: any}>): Array<{role: string, content: any}> {
  return sanitizeAssistantContent(messages).map(msg => {
    if (typeof msg.content === 'string' || !Array.isArray(msg.content)) {
      return msg;
    }

    const convertedContent = msg.content.map((part: any) => {
      // Convert native image format to OpenAI format
      if (part.type === 'image' && part.image) {
        return { type: 'image_url', image_url: { url: part.image } };
      }
      return part;
    });

    return { ...msg, content: convertedContent };
  });
}

// Overloads: passing `tools` switches the return type to the full AssistantResponse
// (content + tool_calls + finish_reason); omitting it keeps the back-compat string path.
export async function fetchResponse(
  serverAddress: string,
  messages: Array<{role: string, content: any}>,
  modelName: string,
  token: string | undefined,
  enableStreaming: boolean,
  onStreamChunk: ((chunk: string) => void) | undefined,
  inferenceParams: InferenceParams | undefined,
  onReasoningChunk: ((chunk: string) => void) | undefined,
  tools: WireToolSpec[]
): Promise<AssistantResponse>;
export async function fetchResponse(
  serverAddress: string,
  messages: Array<{role: string, content: any}>,
  modelName: string,
  token?: string,
  enableStreaming?: boolean,
  onStreamChunk?: (chunk: string) => void,
  inferenceParams?: InferenceParams,
  onReasoningChunk?: (chunk: string) => void
): Promise<string>;
export async function fetchResponse(
  serverAddress: string,
  messages: Array<{role: string, content: any}>,
  modelName: string,
  token?: string,
  enableStreaming: boolean = false,
  onStreamChunk?: (chunk: string) => void,
  inferenceParams?: InferenceParams,
  onReasoningChunk?: (chunk: string) => void,
  tools?: WireToolSpec[]
): Promise<string | AssistantResponse> {
  try {
    // External API: convert to OpenAI format
    const apiMessages = convertToOpenAIFormat(messages);
    const url = `${serverAddress}/v1/chat/completions`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    const isObserverApi = serverAddress.includes('api.observer-ai.com');

    if (isObserverApi) {
      // Observer hosted API: use JWT token for auth
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
    } else if (inferenceParams?.customApiKey) {
      // Custom inference server: use user's own API key (BYOK)
      headers['Authorization'] = `Bearer ${inferenceParams.customApiKey}`;
    }

    // Build request body with optional inference parameters
    const requestBodyObj: Record<string, any> = {
      model: modelName,
      messages: apiMessages,
      stream: enableStreaming
    };

    // Apply inference parameters if provided
    if (inferenceParams) {
      // Tier 1 - Universal OpenAI-compatible parameters
      if (inferenceParams.temperature !== undefined) requestBodyObj.temperature = inferenceParams.temperature;
      if (inferenceParams.top_p !== undefined) requestBodyObj.top_p = inferenceParams.top_p;
      if (inferenceParams.max_tokens !== undefined) requestBodyObj.max_tokens = inferenceParams.max_tokens;
      if (inferenceParams.seed !== undefined) requestBodyObj.seed = inferenceParams.seed;
      if (inferenceParams.stop !== undefined) requestBodyObj.stop = inferenceParams.stop;

      // Tier 2 - Common extensions
      if (inferenceParams.frequency_penalty !== undefined) requestBodyObj.frequency_penalty = inferenceParams.frequency_penalty;
      if (inferenceParams.presence_penalty !== undefined) requestBodyObj.presence_penalty = inferenceParams.presence_penalty;
      if (inferenceParams.top_k !== undefined) requestBodyObj.top_k = inferenceParams.top_k;

      // Tier 3 - Thinking/Reasoning control
      if (inferenceParams.reasoning_effort !== undefined) requestBodyObj.reasoning_effort = inferenceParams.reasoning_effort;
      if (inferenceParams.enable_thinking !== undefined) {
        // For Qwen3 via vLLM - passed as chat_template_kwargs
        requestBodyObj.chat_template_kwargs = { enable_thinking: inferenceParams.enable_thinking };
      }
    }

    // Attach function tools when provided (native OpenAI function calling)
    if (tools && tools.length > 0) {
      requestBodyObj.tools = tools;
    }

    const requestBody = JSON.stringify(requestBodyObj);

    const response = await platformFetch(url, {
      method: 'POST',
      headers,
      body: requestBody,
    });

    if (response.status === 429) {
      throw new UnauthorizedError('Access denied. Quota may be exceeded.');
    }

    if (!response.ok) {
        const errorBody = await response.text();
        console.error(`API Error Response Body: ${errorBody}`);

        let errorMessage = `API error: ${response.status}`;
        try {
          const errorData = JSON.parse(errorBody);
          if (errorData.detail) {
            errorMessage += ` - ${errorData.detail}`;
          }
        } catch {
          if (errorBody && errorBody.length < 200) {
            errorMessage += ` - ${errorBody}`;
          }
        }

        throw new Error(errorMessage);
    }

    const usingTools = !!(tools && tools.length > 0);

    if (enableStreaming) {
      if (usingTools) {
        return await handleStreamingResponseWithTools(response, onStreamChunk, onReasoningChunk);
      }
      return await handleStreamingResponse(response, onStreamChunk, onReasoningChunk);
    } else {
      const data = await response.json();
      const message = data.choices?.[0]?.message;

      if (usingTools) {
        // With tools, content may be null when the model only emits tool_calls.
        if (!message) {
          console.error('Unexpected API response structure:', data);
          throw new Error('Unexpected API response structure');
        }
        const result: AssistantResponse = {
          content: message.content ?? '',
          tool_calls: message.tool_calls,
          finish_reason: data.choices?.[0]?.finish_reason,
        };
        return result;
      }

      if (!message || typeof message.content === 'undefined') {
          console.error('Unexpected API response structure:', data);
          throw new Error('Unexpected API response structure');
      }

      return message.content;
    }

  } catch (error) {
    console.error('Error calling API:', error);
    throw error;
  }
}

