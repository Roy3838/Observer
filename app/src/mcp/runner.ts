// src/mcp/runner.ts
//
// The agentic loop. FRAMEWORK-AGNOSTIC and pure (no React, no DOM): everything
// browser-bound is injected via deps. This is the portability line — the same loop runs
// in a future MCP server with only `send` and the registry's executors swapped.
//
//   send → tool_calls → run → append results → repeat
// until the model returns a turn with no tool calls, or MAX_ITERATIONS is hit.

import type {
  WireMessage,
  AssistantResponse,
  ToolDefinition,
  ToolContext,
  ToolCallStatus,
  ToolResult,
} from './types';
import { validateArgs } from './validate';

// High enough to cover a full agentic build with the post-start crop-verification loop
// (create → crop → start → get_iteration → re-crop → start → get_iteration → …), which on
// its own can run ~15+ turns. get_iteration blocks while a pass completes; it does NOT burn
// a turn waiting, so this caps model turns, not wall-clock time.
export const MAX_ITERATIONS = 30;

/** A validated, ready-to-run tool call. */
export interface PreparedCall {
  id: string;
  name: string;
  args: any;
  tool: ToolDefinition;
}

export interface RunnerDeps {
  /** Streamed model call. Resolves with the full assistant response for one turn. */
  send: (wire: WireMessage[], onTextDelta?: (chunk: string) => void) => Promise<AssistantResponse>;
  /** Registry lookup (injectable for testing). */
  getTool: (name: string) => ToolDefinition | undefined;
  /** Ambient context passed to executors (e.g. getToken). */
  context: ToolContext;
  /** Aborts the loop. Checked at each safe boundary; an in-flight `send` should also
   *  reject when this fires so the loop unwinds promptly. */
  signal?: AbortSignal;
  /** Called whenever the wire changes (new assistant/tool message appended). */
  onWireUpdate?: (wire: WireMessage[]) => void;
  /** Called with each message the loop appends for the user to read: assistant turns and the
   *  tool-images follow-up (not raw tool results). Remote control mirrors these to the phone. */
  onMessage?: (message: WireMessage) => void;
  /** Streamed assistant prose for the live bubble. */
  onAssistantDelta?: (chunk: string) => void;
  /** Per-tool-call status transitions. */
  onStatus?: (toolCallId: string, status: ToolCallStatus, meta?: { name: string; args: any }) => void;
}

/** Serialize a ToolResult into the string content of a `role: 'tool'` message. */
function toolResultContent(result: ToolResult): string {
  if (result.error) return JSON.stringify({ error: result.error });
  return JSON.stringify(result.data ?? {});
}

const CANCELLED: ToolResult = { error: 'Cancelled by user.' };

/** The text of the user message that carries a turn's tool images (OpenAI tool messages are
 *  text-only). Lets consumers tell it apart from what the user wrote. */
export const TOOL_IMAGES_TEXT = 'Images from the tool result(s) above:';

/**
 * Race a tool's `execute()` against the abort signal, so the loop unblocks the instant a new
 * message interrupts it — even for a tool that never looks at `ctx.signal` itself (e.g.
 * capture_screen's browser picker with nobody there to click it). The underlying call is left
 * to settle in the background; its result is simply discarded, same as the model-call race in
 * useMCP's sendToModel.
 */
function executeOrAbort(call: PreparedCall, context: RunnerDeps['context'], signal?: AbortSignal): Promise<ToolResult> {
  const executePromise = call.tool.execute(call.args, context).catch((e): ToolResult => ({
    error: e instanceof Error ? e.message : String(e),
  }));
  if (!signal) return executePromise;
  if (signal.aborted) return Promise.resolve(CANCELLED);
  return new Promise<ToolResult>(resolve => {
    const onAbort = () => resolve(CANCELLED);
    signal.addEventListener('abort', onAbort, { once: true });
    executePromise.then(result => {
      signal.removeEventListener('abort', onAbort);
      resolve(result);
    });
  });
}

/**
 * Guarantee wire validity after an abort: the OpenAI API rejects a request where an
 * assistant `tool_calls` message isn't immediately followed by a `role: 'tool'` result
 * for every call. If a run is killed mid-turn, the most recent assistant turn can have
 * tool calls with no results — fill those with a synthetic "stopped" result so the
 * conversation can still be continued or cleared without a 400.
 */
export function sealDanglingToolCalls(wire: WireMessage[]): WireMessage[] {
  // Only the most recent assistant turn can be unanswered (the loop's invariant is that
  // every prior turn's calls already have results), so stop at the first assistant we hit.
  for (let i = wire.length - 1; i >= 0; i--) {
    if (wire[i].role !== 'assistant') continue;
    const calls = wire[i].tool_calls;
    if (calls && calls.length > 0) {
      const answered = new Set<string>();
      for (let j = i + 1; j < wire.length; j++) {
        const id = wire[j].tool_call_id;
        if (wire[j].role === 'tool' && id) answered.add(id);
      }
      for (const tc of calls) {
        if (!answered.has(tc.id)) {
          wire.push({ role: 'tool', tool_call_id: tc.id, name: tc.function.name, content: JSON.stringify({ error: 'Stopped by user.' }) });
        }
      }
    }
    return wire;
  }
  return wire;
}

/**
 * Run the agentic loop in place, mutating and returning `wire`.
 */
export async function runConversation(wire: WireMessage[], deps: RunnerDeps): Promise<WireMessage[]> {
  const pushTool = (id: string, result: ToolResult, name?: string) => {
    wire.push({ role: 'tool', tool_call_id: id, name, content: toolResultContent(result) });
    deps.onWireUpdate?.(wire);
  };

  for (let i = 0; i < MAX_ITERATIONS; i++) {
    // Boundary 1: between turns the wire always ends in a user/tool/system message, so
    // it's already valid — just stop.
    if (deps.signal?.aborted) return wire;

    const assistant = await deps.send(wire, deps.onAssistantDelta);

    const toolCalls = assistant.tool_calls ?? [];
    const hasToolCalls = toolCalls.length > 0;
    // Gemini rejects empty-string content on assistant messages — use null when empty.
    // When there are no tool_calls, use a zero-width space so the message isn't blank.
    const rawContent = assistant.content;
    const assistantContent = rawContent
      ? rawContent
      : hasToolCalls ? null : '​';
    const assistantMessage: WireMessage = {
      role: 'assistant',
      content: assistantContent,
      tool_calls: assistant.tool_calls,
    };
    wire.push(assistantMessage);
    deps.onMessage?.(assistantMessage);
    deps.onWireUpdate?.(wire);

    if (!hasToolCalls) {
      // Natural termination → final prose already on the wire.
      return wire;
    }

    // Boundary 2: aborted while the model was responding. We just pushed an assistant turn
    // with tool calls — seal it so the wire stays valid, then stop before running anything.
    if (deps.signal?.aborted) {
      sealDanglingToolCalls(wire);
      deps.onWireUpdate?.(wire);
      return wire;
    }

    // Validate + classify every requested call. Each tool_call MUST get a result.
    const executable: PreparedCall[] = [];
    for (const tc of toolCalls) {
      const tool = deps.getTool(tc.function.name);
      if (!tool) {
        deps.onStatus?.(tc.id, 'error', { name: tc.function.name, args: {} });
        pushTool(tc.id, { error: `Unknown tool: ${tc.function.name}` }, tc.function.name);
        continue;
      }
      const validation = validateArgs(tool, tc.function.arguments);
      if (!validation.ok) {
        deps.onStatus?.(tc.id, 'error', { name: tc.function.name, args: {} });
        pushTool(tc.id, { error: validation.error }, tc.function.name);
        continue;
      }
      executable.push({ id: tc.id, name: tc.function.name, args: validation.value, tool });
    }

    // Collected across all calls this turn; injected as one follow-up user message.
    const pendingImages: string[] = [];

    const runOne = async (call: PreparedCall) => {
      deps.onStatus?.(call.id, 'running', { name: call.name, args: call.args });
      const result = await executeOrAbort(call, deps.context, deps.signal);
      deps.onStatus?.(call.id, result.error ? 'error' : 'done', { name: call.name, args: call.args });
      pushTool(call.id, result, call.name);
      if (result.images && result.images.length > 0) pendingImages.push(...result.images);
    };

    // Every validated call runs immediately, in parallel: the model issued them as one
    // independent batch (the common opener is list_agents + list_models), and nothing here
    // depends on ordering — each runOne appends its own result keyed by tool_call_id (the API
    // only requires every id be answered, not in call order) and the UI keys each chip's
    // status by id too. Results land in completion order, so chips resolve as each finishes.
    await Promise.all(executable.map(c => runOne(c)));

    // Multimodal results: OpenAI tool messages are text-only, so images ride in a
    // follow-up user message with image_url parts.
    if (pendingImages.length > 0) {
      const imagesMessage: WireMessage = {
        role: 'user',
        content: [
          { type: 'text', text: TOOL_IMAGES_TEXT },
          ...pendingImages.map(url => ({ type: 'image_url', image_url: { url } })),
        ],
      };
      wire.push(imagesMessage);
      deps.onMessage?.(imagesMessage);
      deps.onWireUpdate?.(wire);
    }
  }

  // Cap hit.
  const capMessage: WireMessage = {
    role: 'assistant',
    content: 'I\'ve reached the maximum number of steps for this request. Let me know how you\'d like to proceed.',
  };
  wire.push(capMessage);
  deps.onMessage?.(capMessage);
  deps.onWireUpdate?.(wire);
  return wire;
}
