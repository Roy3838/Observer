// src/utils/sendApi.ts
import { PreProcessorResult } from './pre-processor';
import { listModels, BROWSER_LOCAL_SENTINEL } from './inferenceServer';
import { GemmaModelManager } from './gemma/GemmaModelManager';
import { platformFetch } from './platform';
import { InferenceParams } from '../config/inference-params';

/**
 * Decrements the quota counter stored in localStorage and dispatches an event.
 * This is an optimistic update for the UI.
 */
const optimisticUpdateQuota = () => {
  try {
    const key = 'observer-quota-remaining';
    const currentQuotaStr = localStorage.getItem(key);

    // Only update if there's an existing number value
    if (currentQuotaStr !== null) {
      const currentQuota = parseInt(currentQuotaStr, 10);
      if (!isNaN(currentQuota)) {
        localStorage.setItem(key, (currentQuota - 1).toString());
        // Dispatch a custom event that the AppHeader can listen to
        window.dispatchEvent(new CustomEvent('quotaUpdated'));
      }
    }
  } catch (error) {
    console.error('Failed to optimistically update quota:', error);
  }
};

/**
 * Handles streaming response from the API
 * @param response The fetch response object
 * @param onStreamChunk Optional callback for each chunk
 * @returns The complete message content
 */
async function handleStreamingResponse(response: Response, onStreamChunk?: (chunk: string) => void): Promise<string> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body available for streaming');
  }

  const decoder = new TextDecoder();
  let fullContent = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        //console.log('🎯 Stream completed');
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n').filter(line => line.trim() !== '');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6); // Remove 'data: ' prefix

          if (data === '[DONE]') {
            //console.log('🏁 Stream finished');
            return fullContent;
          }

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content;

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
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  return fullContent;
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
export async function fetchResponse(
  serverAddress: string,
  messages: Array<{role: string, content: any}>,
  modelName: string,
  token?: string,
  enableStreaming: boolean = false,
  onStreamChunk?: (chunk: string) => void,
  inferenceParams?: InferenceParams
): Promise<string> {
  try {
    if (serverAddress === BROWSER_LOCAL_SENTINEL) {
      const manager = GemmaModelManager.getInstance();
      if (!manager.isReady()) throw new Error('Gemma model not loaded. Please load it from the Add Model panel.');
      return manager.generate(messages, onStreamChunk);
    }

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
      messages: messages,
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

    if (enableStreaming) {
      return await handleStreamingResponse(response, onStreamChunk);
    } else {
      const data = await response.json();

      if (!data.choices || !data.choices[0] || !data.choices[0].message || typeof data.choices[0].message.content === 'undefined') {
          console.error('Unexpected API response structure:', data);
          throw new Error('Unexpected API response structure');
      }

      return data.choices[0].message.content;
    }

  } catch (error) {
    console.error('Error calling API:', error);
    throw error;
  }
}

/**
 * Send a prompt to the API server using OpenAI-compatible v1 chat completions endpoint
 * @param modelName Name of the model to use
 * @param preprocessResult The preprocessed result containing prompt and optional images
 * @param token Optional authorization token
 * @param enableStreaming Whether to enable streaming response (default: true)
 * @param onStreamChunk Optional callback for streaming chunks (only used when streaming is enabled)
 * @param inferenceParams Optional inference parameters (temperature, top_p, etc.)
 * @returns The model's response text
 */
export async function sendPrompt(
  modelName: string,
  preprocessResult: PreProcessorResult,
  token?: string,
  enableStreaming: boolean = false,
  onStreamChunk?: (chunk: string) => void,
  inferenceParams?: InferenceParams
): Promise<string> {
  try {
    // Find the server for this model
    const modelsResponse = listModels();
    const model = modelsResponse.models.find(m => m.name === modelName);

    if (!model) {
      throw new Error(`Model '${modelName}' not found in available models`);
    }

    const serverAddress = model.server;

    // Convert single-turn request to messages array format
    let content: any = preprocessResult.modifiedPrompt;
    const hasImages = preprocessResult.images && preprocessResult.images.length > 0;

    if (hasImages) {
      // Multimodal content with images
      content = [
        { type: "text", text: preprocessResult.modifiedPrompt },
        ...preprocessResult.images!.map(imageBase64Data => ({
          type: "image_url",
          image_url: {
            url: `data:image/png;base64,${imageBase64Data}`
          }
        }))
      ];
    }

    // Build messages array (single user message for agent loop)
    const messages = [
      {
        role: "user",
        content: content
      }
    ];

    if (serverAddress.includes('api.observer-ai.com') && token) {
      optimisticUpdateQuota();
    }

    // Use the new fetchResponse function with messages array
    return await fetchResponse(serverAddress, messages, modelName, token, enableStreaming, onStreamChunk, inferenceParams);

  } catch (error) {
    console.error('Error calling API:', error);
    // Re-throw the error so the calling function knows something went wrong
    throw error;
  }
}
