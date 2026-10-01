// src/mcp/useMCP.ts
//
// React glue between the pure runner and the MCP UI. Holds the wire, the per-tool-call
// status map, and the deferred-promise registry for the `ask_user_info` modal.

import { useCallback, useEffect, useRef, useState } from 'react';
import type { TokenProvider } from '@utils/main_loop';
import { ModelManager } from '@utils/ModelManager';
import { Logger } from '@utils/logging';
import type { WireMessage, ToolCallStatus, UserInfoRequest, UserInfoResponse } from './types';
import { getTool, getToolSpecs } from './registry';
import getMcpSystemPrompt from './systemPrompt';
import { runConversation, sealDanglingToolCalls } from './runner';
import {
  type ConversationSummary,
  deleteStoredConversation,
  listConversations,
  loadConversation as loadStoredConversation,
  newConversationId,
  saveConversation,
} from './conversationStore';

export interface ToolStatusEntry {
  status: ToolCallStatus;
  name: string;
  args: any;
}

export interface UseMCPOptions {
  getToken: TokenProvider;
  isUsingObServer: boolean;
  /** Cloud model that drives function calling. */
  modelName?: string;
}

/** Listener fired after any agent-mutating tool completes. Receives the tool name. */
export type MutationListener = (toolName: string, args?: Record<string, any>) => void;

const DEFAULT_MODEL = 'gemini-2.5-flash-lite-free';

/** Logger source for everything the MCP agentic loop emits. */
const LOG_SOURCE = 'MCP';

/** Tools that change the dashboard's agent list / running state. */
const MUTATING_TOOLS = new Set(['create_agent', 'edit_agent', 'start_agent', 'stop_agent']);

/** The last non-empty assistant prose in `messages` (the runner pads empty turns with U+200B). */
function finalAssistantText(messages: WireMessage[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== 'assistant' || typeof m.content !== 'string') continue;
    const text = m.content.replace(/\u200B/g, '').trim();
    if (text) return text;
  }
  return undefined;
}

/** Max images to carry back to a remote reply \u2014 keeps MMS/Telegram sends bounded. */
const MAX_REMOTE_IMAGES = 3;

/**
 * Images the runner collected from this turn's tool results (see runner.ts's
 * "Images from the tool result(s) above:" follow-up user message), most recent last.
 * Lets a remote reply (e.g. capture_screen) carry back what the agent actually saw.
 */
function turnImages(messages: WireMessage[]): string[] {
  const images: string[] = [];
  for (const m of messages) {
    if (m.role !== 'user' || !Array.isArray(m.content)) continue;
    for (const part of m.content) {
      if (part?.type === 'image_url' && part.image_url?.url) images.push(part.image_url.url);
    }
  }
  return images.slice(-MAX_REMOTE_IMAGES);
}

const abortError = () => new DOMException('Stopped by user.', 'AbortError');
const isAbortError = (e: unknown): boolean =>
  e instanceof DOMException ? e.name === 'AbortError' : (e as any)?.name === 'AbortError';

export function useMCP(options: UseMCPOptions) {
  const { getToken, isUsingObServer, modelName = DEFAULT_MODEL } = options;

  // The wire always begins with the (hidden) system message.
  const wireRef = useRef<WireMessage[]>([
    { role: 'system', content: getMcpSystemPrompt() },
  ]);

  // Consumers subscribe to agent-mutation events (create/edit/start/stop) so each screen
  // can react in its own way (refresh the dashboard, close the modal, start a tutorial, …)
  // without the shared hook baking in one caller's callback.
  const mutationListeners = useRef<Set<MutationListener>>(new Set());
  const subscribeMutation = useCallback((listener: MutationListener) => {
    mutationListeners.current.add(listener);
    return () => { mutationListeners.current.delete(listener); };
  }, []);

  const [messages, setMessages] = useState<WireMessage[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string>(newConversationId);
  const [conversations, setConversations] = useState<ConversationSummary[]>(() => listConversations());
  // Set when messages were replaced by a load, so the persist effect doesn't bump its recency.
  const skipSaveRef = useRef(false);
  const [streamingText, setStreamingText] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [toolStatus, setToolStatus] = useState<Map<string, ToolStatusEntry>>(new Map());
  // Requests are queued: the runner executes a turn's tool calls in parallel, so two
  // `ask_user_info` calls arrive together. The modal shows the head; the rest wait their turn.
  const [userInfoQueue, setUserInfoQueue] = useState<UserInfoRequest[]>([]);
  const pendingUserInfo = userInfoQueue[0] ?? null;

  // requestId → settlers for the deferred `ask_user_info` promise. We keep `reject` too so
  // a hard stop can unwind a run parked on the modal.
  const userInfoResolvers = useRef<Map<string, {
    resolve: (r: UserInfoResponse) => void;
    reject: (e: unknown) => void;
  }>>(new Map());

  // Aborts the in-flight run. One controller per send(); stop() fires it.
  const abortRef = useRef<AbortController | null>(null);
  // The currently in-flight run's body, so a new send() can wait for it to actually unwind
  // (not just for isRunning to flip, which lags a tick behind React state) before starting.
  const inFlightRef = useRef<Promise<void> | null>(null);

  const syncMessages = useCallback(() => {
    // Expose everything except the system message; the component decides what to render.
    setMessages(wireRef.current.filter(m => m.role !== 'system'));
  }, []);

  const setStatus = useCallback((id: string, status: ToolCallStatus, meta?: { name: string; args: any }) => {
    setToolStatus(prev => {
      const next = new Map(prev);
      const existing = next.get(id);
      next.set(id, {
        status,
        name: meta?.name ?? existing?.name ?? '',
        args: meta?.args ?? existing?.args ?? {},
      });
      return next;
    });
  }, []);

  /** Blocks the `ask_user_info` executor until the modal resolves. Injected via ToolContext. */
  const requestUserInfo = useCallback((req: UserInfoRequest): Promise<UserInfoResponse> => {
    Logger.info(LOG_SOURCE, `Awaiting user info: ${req.kind}${req.channel ? ` (${req.channel})` : ''}`, {
      requestId: req.requestId,
    });
    return new Promise<UserInfoResponse>((resolve, reject) => {
      userInfoResolvers.current.set(req.requestId, { resolve, reject });
      setUserInfoQueue(q => [...q, req]);
    });
  }, []);

  /** Resolve a pending user-info request from the modal. */
  const resolveUserInfo = useCallback((requestId: string, response: UserInfoResponse) => {
    const settler = userInfoResolvers.current.get(requestId);
    if (settler) {
      userInfoResolvers.current.delete(requestId);
      Logger.info(LOG_SOURCE, `User ${response.skipped ? 'skipped' : 'supplied'} requested info`, { requestId });
      settler.resolve(response);
    }
    setUserInfoQueue(q => q.filter(r => r.requestId !== requestId));
  }, []);

  /** Abort whatever's in flight (a model call or a tool call) and reject any pending
   *  ask_user_info modal, so it isn't left on screen with nothing listening. Doesn't wait for
   *  the run to actually unwind — callers that need that await `inFlightRef.current`. */
  const interrupt = useCallback(() => {
    abortRef.current?.abort();
    for (const [, settler] of userInfoResolvers.current) settler.reject(abortError());
    userInfoResolvers.current.clear();
    setUserInfoQueue([]);
  }, []);

  /** Runs one user turn. Resolves, once the run settles, with the run's final assistant text
   *  and any images the model captured along the way (what remote control sends back to the
   *  phone), or undefined if there was no final text.
   *
   *  A new message always wins over whatever's currently running: rather than queuing behind
   *  a slow or hung tool call (e.g. capture_screen's picker with nobody there to click it),
   *  it interrupts that run — via the same abort path `stop()` uses — and waits for it to
   *  actually unwind before starting. */
  const send = useCallback(async (userText: string, images?: string[]): Promise<{ text: string; images: string[] } | undefined> => {
    if (inFlightRef.current) {
      interrupt();
      await inFlightRef.current.catch(() => {});
    }

    Logger.info(LOG_SOURCE, `User message sent (model: ${modelName})`, {
      imageCount: images?.length ?? 0,
    });

    // Build the user message (multimodal if images were attached).
    let userContent: any = userText;
    if (images && images.length > 0) {
      userContent = [
        { type: 'text', text: userText },
        ...images.map(img => ({
          type: 'image_url',
          image_url: { url: img.startsWith('data:') ? img : `data:image/png;base64,${img}` },
        })),
      ];
    }
    wireRef.current.push({ role: 'user', content: userContent });
    syncMessages();
    const turnStart = wireRef.current.length;

    const controller = new AbortController();
    abortRef.current = controller;
    const { signal } = controller;
    setIsRunning(true);

    const sendToModel = async (
      wire: WireMessage[],
      onTextDelta?: (chunk: string) => void,
    ) => {
      setStreamingText('');
      try {
        if (signal.aborted) throw abortError();
        const token = isUsingObServer ? await getToken() : undefined;
        const modelCall = ModelManager.getInstance().sendToolMessages(
          modelName,
          wire,
          getToolSpecs(),
          token,
          true,
          onTextDelta,
        );
        // Race the streamed call against the abort signal so a hard stop unwinds the loop
        // immediately instead of waiting for the model to finish. The orphaned modelCall
        // settles in the background; its result is discarded (deltas are gated below).
        return await new Promise<Awaited<typeof modelCall>>((resolve, reject) => {
          const onAbort = () => reject(abortError());
          signal.addEventListener('abort', onAbort, { once: true });
          modelCall.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
        });
      } finally {
        setStreamingText('');
      }
    };

    const runPromise = (async () => {
      try {
        await runConversation(wireRef.current, {
          send: sendToModel,
          getTool,
          context: { getToken, signal, requestUserInfo },
          signal,
          onWireUpdate: syncMessages,
          // Drop stream chunks from an orphaned (aborted) model call so a killed run can't
          // keep repopulating the live bubble.
          onAssistantDelta: chunk => { if (!signal.aborted) setStreamingText(prev => prev + chunk); },
          onStatus: (id, status, meta) => {
            setStatus(id, status, meta);
            const toolName = meta?.name ?? 'unknown';
            if (status === 'running') {
              Logger.info(LOG_SOURCE, `Running tool: ${toolName}`, { toolCallId: id, args: meta?.args });
            } else if (status === 'done') {
              Logger.info(LOG_SOURCE, `Tool completed: ${toolName}`, { toolCallId: id });
            } else if (status === 'error') {
              Logger.error(LOG_SOURCE, `Tool failed: ${toolName}`, { toolCallId: id });
            }
            if (status === 'done' && meta && MUTATING_TOOLS.has(meta.name)) {
              mutationListeners.current.forEach(l => l(meta.name, meta.args));
            }
          },
        });
      } catch (err) {
        if (isAbortError(err)) {
          // Interrupted — either a hard stop or a new message taking over. Leave the
          // conversation valid (no dangling tool_calls) and silent; either way there's no
          // error bubble to show.
          Logger.info(LOG_SOURCE, 'Run interrupted');
          sealDanglingToolCalls(wireRef.current);
          syncMessages();
        } else {
          const text = err instanceof Error ? err.message : 'An unknown error occurred.';
          Logger.error(LOG_SOURCE, `Run failed: ${text}`, { error: err });
          wireRef.current.push({ role: 'assistant', content: `Sorry, I ran into an error: ${text}` });
          syncMessages();
        }
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
        setIsRunning(false);
        setStreamingText('');
      }
    })();
    inFlightRef.current = runPromise;
    await runPromise;
    if (inFlightRef.current === runPromise) inFlightRef.current = null;

    const turnSlice = wireRef.current.slice(turnStart);
    const text = finalAssistantText(turnSlice);
    return text ? { text, images: turnImages(turnSlice) } : undefined;
  }, [isUsingObServer, getToken, modelName, syncMessages, setStatus, requestUserInfo, interrupt]);

  /** Hard-stop the in-flight run: kill the loop and abandon any model call. Leaves the
   *  (sealed) conversation intact. */
  const stop = useCallback(() => {
    interrupt();
  }, [interrupt]);

  // Persist once a run settles (not on every streamed wire update).
  useEffect(() => {
    if (isRunning || messages.length === 0) return;
    if (skipSaveRef.current) { skipSaveRef.current = false; return; }
    saveConversation(activeConversationId, messages);
    setConversations(listConversations());
  }, [messages, isRunning, activeConversationId]);

  /** Swap in a wire (fresh, or a stored conversation's messages). No-op while a run is in flight. */
  const resetWire = useCallback((id: string, stored: WireMessage[] = []) => {
    userInfoResolvers.current.clear();
    wireRef.current = [{ role: 'system', content: getMcpSystemPrompt() }, ...stored];
    skipSaveRef.current = stored.length > 0;
    setActiveConversationId(id);
    setMessages(stored);
    setToolStatus(new Map());
    setStreamingText('');
    setUserInfoQueue([]);
  }, []);

  /** Start a fresh conversation (the previous one stays saved). */
  const newConversation = useCallback(() => {
    if (isRunning) return;
    resetWire(newConversationId());
  }, [isRunning, resetWire]);

  const loadConversation = useCallback((id: string) => {
    if (isRunning || id === activeConversationId) return;
    const conv = loadStoredConversation(id);
    if (conv) resetWire(conv.id, conv.messages);
  }, [isRunning, activeConversationId, resetWire]);

  const deleteConversation = useCallback((id: string) => {
    if (isRunning && id === activeConversationId) return;
    deleteStoredConversation(id);
    setConversations(listConversations());
    if (id === activeConversationId) resetWire(newConversationId());
  }, [isRunning, activeConversationId, resetWire]);

  return {
    messages,
    streamingText,
    isRunning,
    toolStatus,
    pendingUserInfo,
    resolveUserInfo,
    subscribeMutation,
    conversations,
    activeConversationId,
    newConversation,
    loadConversation,
    deleteConversation,
    clear: newConversation,
    stop,
    send,
  };
}

export type UseMCPReturn = ReturnType<typeof useMCP>;
