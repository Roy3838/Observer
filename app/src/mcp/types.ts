// src/mcp/types.ts
//
// Transport-agnostic wire + tool types for the Observer MCP core.
//
// These types are intentionally framework-free (no React, no IndexedDB). They model
// the OpenAI function-calling wire format so that the same registry + runner can later
// drop into a real MCP server with only the transport swapped.

import type { LocalModelChoice } from './localModel';

/**
 * A single tool call requested by the model, in OpenAI shape.
 * `arguments` is a raw JSON string (as the API returns it); the runner parses + validates it.
 */
export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
  /**
   * Provider-specific passthrough. Gemini 2.5+ thinking models return an opaque
   * `thought_signature` here on the OpenAI-compat endpoint and REQUIRE it to be echoed
   * back verbatim on this same tool call when the conversation history is replayed —
   * otherwise the next request 400s with "Function call is missing a thought_signature".
   * We capture it from the stream and carry it on the wire so the replay stays valid.
   */
  extra_content?: { google?: { thought_signature?: string } };
}

/**
 * A message on the wire. Mirrors the OpenAI chat-completions message shape:
 * - system / user / assistant text
 * - assistant messages may carry `tool_calls`
 * - tool results are `role: 'tool'` keyed by `tool_call_id`
 *
 * `content` is `any` because user messages may be multimodal content-part arrays
 * (text + image_url parts), matching what the rest of the app already sends.
 */
export interface WireMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: any;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
}

/**
 * The model's response for a single turn. Returned by the tools-aware send path.
 */
export interface AssistantResponse {
  content: string;
  tool_calls?: ToolCall[];
  finish_reason?: string;
}

/**
 * Result of executing a tool. `images` are optional base64 data-URLs that the runner
 * injects as a follow-up multimodal user message (OpenAI tool results are text-only).
 */
export interface ToolResult {
  // Arbitrary JSON-serializable payload (object, array, string, …). Serialized into the
  // `role: 'tool'` message content by the runner.
  data?: any;
  error?: string;
  images?: string[]; // data-URL strings, e.g. "data:image/png;base64,..."
}

/**
 * A JSON-Schema object describing a tool's parameters (the `parameters` field of an
 * OpenAI function definition).
 */
export interface JsonSchema {
  type: 'object';
  properties: Record<string, any>;
  required?: string[];
  additionalProperties?: boolean;
}

/**
 * A tool definition. The `execute` body is the only browser-bound part; everything else
 * (name, description, schema) is pure data that serializes onto the wire.
 */
export interface ToolDefinition {
  name: string;
  description: string;
  parameters: JsonSchema;
  /** Whether this tool may return images in its ToolResult. */
  multimodal: boolean;
  execute: (args: any, ctx: ToolContext) => Promise<ToolResult>;
}

/**
 * The kinds of contact info `ask_user_info` can collect. Mirrors the notification tools
 * available to agent code (sendSms/call/sendWhatsapp, sendEmail, sendTelegram, sendDiscord,
 * sendPushover).
 */
export type UserInfoKind = 'phone' | 'email' | 'telegram' | 'discord' | 'pushover';

/**
 * A request for one piece of contact info, handed from the `ask_user_info` executor up to
 * the React layer, which renders the guided modal.
 */
export interface UserInfoRequest {
  requestId: string;
  /** A contact kind, or 'model': whether to download a local model or run in the cloud. */
  kind: UserInfoKind | 'model';
  /** For `model`: which local model the download would fetch. */
  model?: LocalModelChoice;
  /** For `phone`: which tool the code is for. 'whatsapp' also requires WhatsApp's 24h window. */
  channel?: 'sms' | 'voice' | 'whatsapp';
  /** The model's one-line "why I need this", shown to the user for context. */
  reason?: string;
}

/**
 * The user's answer. `skipped` means they dismissed the modal — the tool reports that back
 * to the model so it can ask in chat instead of hanging.
 */
export interface UserInfoResponse {
  value: string;
  skipped?: boolean;
}

/**
 * Ambient context handed to executors that need it (e.g. an auth token provider for
 * starting agents against the Observer API).
 */
export interface ToolContext {
  getToken?: () => Promise<string | undefined>;
  /** Aborts a long-running/blocking executor (e.g. check_whitelist waiting for the user). */
  signal?: AbortSignal;
  /**
   * Blocks until the user supplies a piece of contact info via the guided modal. Injected by
   * the React layer (useMCP) via a deferred promise. Absent in non-React hosts, where
   * `ask_user_info` degrades to telling the model to ask in chat instead.
   */
  requestUserInfo?: (req: UserInfoRequest) => Promise<UserInfoResponse>;
}

/**
 * Lifecycle status of a single tool call, surfaced to the UI.
 */
export type ToolCallStatus =
  | 'running'   // executor in flight
  | 'done'      // executor resolved
  | 'error';    // executor threw / returned error

/**
 * The OpenAI "function tool" wire shape, built from a ToolDefinition for the request body.
 */
export interface WireToolSpec {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: JsonSchema;
  };
}
