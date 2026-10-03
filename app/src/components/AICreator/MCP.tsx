// src/components/AICreator/MCP.tsx
//
// Native tool-calling agent creator. Replaces MultiAgentCreator: instead of brittle
// $$$/%%%/&&& text delimiters, the model drives agent creation/management through real
// OpenAI function calls (see src/mcp/). This component is pure UI over the useMCP hook.

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Plus, CheckCircle2, XCircle, Loader, Square, Cpu, Mic, ChevronDown, Check, FileDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import type { TokenProvider } from '@utils/main_loop';
import { type ToolStatusEntry } from '../../mcp/useMCP';
import { useMCPContext } from '../../mcp/MCPContext';
import { parseRemotePrompt, type RemoteChannel } from '../../mcp/remote';
import type { ToolCall, WireMessage } from '../../mcp/types';
import { Logger, type WhitelistChannel } from '@utils/logging';
import { StreamManager, StreamState } from '@utils/streamManager';
import type { CompleteAgent } from '@utils/agent_database';
import { useSubscriberText } from '@hooks/useTranscriptionState';
import WhitelistInline from '@components/whitelist/WhitelistInline';
import { SensorSettings } from '@utils/settings';
import { normalizeWhitelistCode } from '@utils/whitelistCode';
import { isTauri } from '@utils/platform';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import { DEFAULT_LLAMACPP_FILES } from '@/mcp/localModel';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import type { GemmaModelState, NativeModelState } from '@utils/localLlm/types';
import { ModelManager, type Model } from '@utils/ModelManager';
import { ModelRow, RowButtonGhost } from '@components/ModelCard/ModelRow';
import RecipeMini from './RecipeMini';
import AgentLiveCard from '@components/Observer/AgentLiveCard';
import { useFloatingAgents } from '@components/Observer/FloatingAgentsContext';

// Synthetic owner id for voice dictation. Mirrors SettingsTab's TEST_AGENT_ID pattern:
// StreamManager treats it like any agent, so transcription routes through the same
// TranscriptionRouter path and honors the user's global mode (cloud / local / self-hosted)
// for free — changing the mode in Settings changes dictation too.
const MCP_MIC_ID = 'mcp-creator-mic';

interface MCPProps {
  getToken: TokenProvider;
  isAuthenticated: boolean;
  isUsingObServer: boolean;
  onSignIn?: () => void;
  onSwitchToObServer?: () => void;
  onUpgrade?: () => void;
  onRefresh?: () => void;
  onSaveComplete?: () => void;
  /** Hide the built-in generic suggestion chips (e.g. when the RecipeBuilder hero is shown above). */
  hideSuggestions?: boolean;
  initialMessage?: string;
  /** Opens the full "When... Then..." recipe builder modal; renders a teaser chip when set. */
  onOpenRecipe?: () => void;
  /** Tailwind height classes for the chat container. Defaults to the hero/sheet sizing. */
  heightClass?: string;
  /** When false, renders unboxed/full-bleed (no bordered card, rounder pill input) for a
   *  permanently docked surface like the Observer tab. Defaults to true so existing callers
   *  (GetStarted, MCPPanel) are visually unaffected. */
  boxed?: boolean;
  /** Docked-surface extras (Observer tab only) — when provided, a `create_agent` tool call
   *  renders a live AgentLiveCard inline in the transcript instead of just a status chip,
   *  which the user can drag out into the floating overlay (see FloatingAgentsContext). */
  agents?: CompleteAgent[];
  runningAgents?: Set<string>;
  startingAgents?: Set<string>;
  onToggleAgent?: (agentId: string, isCurrentlyRunning: boolean) => void;
  onSelectAgent?: (agentId: string) => void;
}

// ===================================================================================
//  MARKDOWN BUBBLE
// ===================================================================================
const Markdown: React.FC<{ text: string }> = ({ text }) => (
  <div className="prose prose-sm max-w-none">
    <ReactMarkdown
      components={{
        ul: ({ children }) => <ul className="list-disc pl-4 space-y-1 mb-2">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal pl-4 space-y-1 mb-2">{children}</ol>,
        li: ({ children }) => <li className="text-inherit">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        code: ({ children }) => <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">{children}</code>,
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
      }}
    >
      {text}
    </ReactMarkdown>
  </div>
);

// ===================================================================================
//  TOOL-CALL STATUS CHIP
// ===================================================================================
const StatusIcon: React.FC<{ status?: string }> = ({ status }) => {
  switch (status) {
    case 'done': return <CheckCircle2 className="h-3 w-3 text-green-600" />;
    case 'error': return <XCircle className="h-3 w-3 text-red-500" />;
    case 'running': return <Loader className="h-3 w-3 text-gray-400 animate-spin" />;
    default: return <CheckCircle2 className="h-3 w-3 text-green-600" />;
  }
};

// Deliberately small and muted — a status caption, not a message. Tool calls aren't
// conversation content, so they shouldn't read like a chat bubble the user is meant to parse.
// Clickable so the raw call (args + result) can be inspected when debugging model confusion.
const ToolChip: React.FC<{ call: ToolCall; status?: ToolStatusEntry; onInspect?: () => void }> = ({ call, status, onInspect }) => (
  <button
    type="button"
    onClick={onInspect}
    className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 text-[10px] font-medium text-gray-500 dark:text-gray-300 hover:border-purple-300 dark:hover:border-purple-500 hover:text-purple-600 dark:hover:text-purple-300 transition-colors"
  >
    <StatusIcon status={status?.status} />
    <span className="font-mono">{call.function.name}</span>
  </button>
);

// Pretty-prints a JSON string (tool arguments, or a `role: 'tool'` result's content) for
// display; falls back to the raw string when it isn't valid JSON (e.g. a plain-text error).
const formatJsonish = (raw: unknown): string => {
  if (raw == null) return '';
  const str = typeof raw === 'string' ? raw : JSON.stringify(raw);
  try {
    return JSON.stringify(JSON.parse(str), null, 2);
  } catch {
    return str;
  }
};

// ===================================================================================
//  TOOL-CALL DETAIL MODAL
// ===================================================================================
// Shows exactly what the model sent (name + parsed arguments) and what came back (the
// paired `role: 'tool'` result message), so you can see why the model got confused —
// e.g. it called create_agent with a malformed config, or the executor returned an error
// the chip alone never surfaces.
const ToolCallDetailModal: React.FC<{
  call: ToolCall;
  status?: ToolStatusEntry;
  resultMsg?: WireMessage;
  onClose: () => void;
}> = ({ call, status, resultMsg, onClose }) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
    <div
      className="w-full max-w-lg max-h-[80vh] overflow-y-auto rounded-lg bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-xl p-4"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <StatusIcon status={status?.status} />
          <span className="font-mono text-sm font-semibold text-gray-800 dark:text-gray-100">{call.function.name}</span>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">
          <XCircle className="h-4 w-4" />
        </button>
      </div>

      <div className="mb-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-gray-400 mb-1">Arguments</div>
        <pre className="text-xs font-mono whitespace-pre-wrap break-all bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-700 rounded p-2 text-gray-700 dark:text-gray-300">
          {formatJsonish(call.function.arguments) || '(none)'}
        </pre>
      </div>

      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wide text-gray-400 mb-1">Result</div>
        {resultMsg ? (
          <pre className="text-xs font-mono whitespace-pre-wrap break-all bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-700 rounded p-2 text-gray-700 dark:text-gray-300">
            {formatJsonish(resultMsg.content)}
          </pre>
        ) : (
          <div className="text-xs text-gray-400 italic">
            {status?.status === 'running' ? 'Still running…' : 'No result recorded.'}
          </div>
        )}
      </div>
    </div>
  </div>
);

// ===================================================================================
//  WORKING GROUP  (Cowork-style collapsed run of tool calls)
// ===================================================================================
// Consecutive tool-only assistant turns (no text) are visually one "the agent is working"
// moment, not a stack of separate messages — so they're merged into a single compact,
// auto-collapsing panel instead of one spaced-out chip row per turn (see grouping logic
// in renderMessages below).
const WorkingGroup: React.FC<{ batches: ToolCall[][]; toolStatus: Map<string, ToolStatusEntry>; messages: WireMessage[] }> = ({ batches, toolStatus, messages }) => {
  const calls = batches.flat();
  const isRunning = calls.some(tc => toolStatus.get(tc.id)?.status === 'running');
  const hasError = calls.some(tc => toolStatus.get(tc.id)?.status === 'error');
  const [inspecting, setInspecting] = useState<ToolCall | null>(null);
  // Auto-collapse shortly after the run finishes, mirroring Cowork's "done, tucked away" feel.
  // Stays expanded while running or right after finishing so the last tool is still visible.
  const [collapsed, setCollapsed] = useState(false);
  const wasRunning = useRef(isRunning);
  useEffect(() => {
    if (wasRunning.current && !isRunning) {
      const t = setTimeout(() => setCollapsed(true), 1200);
      wasRunning.current = isRunning;
      return () => clearTimeout(t);
    }
    wasRunning.current = isRunning;
  }, [isRunning]);

  return (
    <div className="w-1/4 min-w-[220px] mt-1 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50/80 dark:bg-gray-800/60 overflow-hidden">
      <button
        type="button"
        onClick={() => setCollapsed(c => !c)}
        className="w-full flex items-center gap-2 px-2.5 py-1.5 text-left hover:bg-gray-100 dark:hover:bg-gray-700/60 transition-colors"
      >
        <span className="flex-shrink-0 flex items-center justify-center h-4 w-4">
          {isRunning ? (
            <Loader2 className="h-3.5 w-3.5 text-purple-500 animate-spin" />
          ) : hasError ? (
            <XCircle className="h-3.5 w-3.5 text-red-500" />
          ) : (
            <CheckCircle2 className="h-3.5 w-3.5 text-green-600" />
          )}
        </span>
        <span className={`text-xs font-medium ${isRunning ? 'text-gray-600 dark:text-gray-300 animate-pulse' : 'text-gray-500 dark:text-gray-400'}`}>
          {isRunning ? 'Working…' : hasError ? 'Finished with errors' : `Used ${calls.length} tool${calls.length > 1 ? 's' : ''}`}
        </span>
        <ChevronDown className={`h-3 w-3 ml-auto text-gray-400 transition-transform flex-shrink-0 ${collapsed ? '-rotate-90' : ''}`} />
      </button>
      {!collapsed && (
        <div className="px-2.5 pb-2 space-y-1">
          {batches.map((batch, bi) => (
            <div key={bi} className="flex flex-wrap items-center gap-1">
              {batch.map(tc => (
                <ToolChip key={tc.id} call={tc} status={toolStatus.get(tc.id)} onInspect={() => setInspecting(tc)} />
              ))}
              {bi < batches.length - 1 && <span className="text-gray-300 dark:text-gray-600 text-[10px]">·</span>}
            </div>
          ))}
        </div>
      )}
      {inspecting && (
        <ToolCallDetailModal
          call={inspecting}
          status={toolStatus.get(inspecting.id)}
          resultMsg={messages.find(m => m.role === 'tool' && m.tool_call_id === inspecting.id)}
          onClose={() => setInspecting(null)}
        />
      )}
    </div>
  );
};

// How long a `check_whitelist` call must stay 'running' before we show the QR pill. The
// executor's very first poll is often already a hit (number was whitelisted earlier), which
// resolves in one network round-trip — without this grace period the pill would mount then
// immediately unmount, flashing the QR codes for no reason.
const WHITELIST_PILL_DELAY_MS = 600;

/** Gates WhitelistInline behind WHITELIST_PILL_DELAY_MS of sustained 'running' status. */
const CheckWhitelistGate: React.FC<{
  toolCallId: string;
  status?: ToolStatusEntry;
  onCancel: () => void;
}> = ({ toolCallId, status, onCancel }) => {
  const isRunning = status?.status === 'running';
  const [showPill, setShowPill] = useState(false);

  useEffect(() => {
    if (!isRunning) {
      setShowPill(false);
      return;
    }
    const timer = setTimeout(() => setShowPill(true), WHITELIST_PILL_DELAY_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRunning, toolCallId]);

  if (!showPill) return null;
  // The code the executor is actually polling: same arg resolution as check_whitelist, which
  // defaults to the saved code. Anything that isn't a code fails the tool immediately, so
  // there's nothing to pair and no pill.
  const raw: string | undefined = status?.args?.code ?? status?.args?.phone_number;
  const channel = (status?.args?.channel as WhitelistChannel | undefined) ?? 'sms';
  const code = raw ? normalizeWhitelistCode(raw) : SensorSettings.defaultPhoneCode(channel);
  if (!code) return null;
  return <WhitelistInline code={code} channel={channel} onCancel={onCancel} />;
};

// ===================================================================================
//  DOWNLOAD-MODEL PROGRESS  (live bars while download_model runs)
// ===================================================================================
const formatBytes = (bytes: number) => {
  if (!+bytes) return '0 B';
  const k = 1024, sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

// Same per-file row as ModelHub's FileProgressList (AvailableModels.tsx): name, bytes, thin gray bar.
interface FileProgress { file: string; loaded: number; total: number; progress: number; done: boolean }

const FileProgressList: React.FC<{ items: FileProgress[] }> = ({ items }) => (
  <div className="pb-2.5 space-y-1.5">
    {items.map(item => (
      <div key={item.file}>
        <div className="flex justify-between items-center gap-2 text-[11px] text-gray-500 mb-0.5">
          <span className="flex items-center gap-1 min-w-0">
            {item.done
              ? <Check size={11} className="text-gray-800 flex-shrink-0" />
              : <FileDown size={11} className="text-gray-400 flex-shrink-0" />}
            <span className="truncate" title={item.file}>{item.file}</span>
          </span>
          <span className="tabular-nums flex-shrink-0">
            {item.done ? 'Done'
              : item.total > 0 ? `${formatBytes(item.loaded)} / ${formatBytes(item.total)}`
              : `${Math.round(item.progress)}%`}
          </span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-1">
          <div
            className="h-1 rounded-full transition-all duration-300 bg-gray-800"
            style={{ width: `${Math.max(0, Math.min(100, item.progress))}%` }}
          />
        </div>
      </div>
    ))}
  </div>
);

/**
 * Subscribes directly to the local-model managers (the same state the Models tab renders) to show
 * live progress for the in-flight `download_model` tool call, as a ModelHub-style ModelRow.
 * Renders nothing when idle.
 */
const DownloadShell: React.FC<{ meta: React.ReactNode; onCancel?: () => void; children?: React.ReactNode }> = ({ meta, onCancel, children }) => (
  <div className="mt-2 w-full max-w-md">
    <ModelRow
      icon={<Cpu size={16} />}
      name="On-device model"
      meta={meta}
      action={onCancel && <RowButtonGhost onClick={onCancel}>Cancel</RowButtonGhost>}
      detailSlot={children}
    />
  </div>
);

const DownloadModelProgress: React.FC = () => {
  const tauri = isTauri();
  const [gemma, setGemma] = useState<GemmaModelState>(() => GemmaModelManager.getInstance().getState());
  const [native, setNative] = useState<NativeModelState>(() => NativeLlmManager.getInstance().getState());

  useEffect(() => {
    const unsubGemma = GemmaModelManager.getInstance().onStateChange(setGemma);
    if (!tauri) return unsubGemma;
    const unsubNative = NativeLlmManager.getInstance().onStateChange(setNative);
    return () => { unsubGemma(); unsubNative(); };
  }, [tauri]);

  if (tauri) {
    const { status, modelId, downloadProgress, downloadedBytes, totalBytes, error } = native;
    if (status === 'downloading') {
      return (
        <DownloadShell meta="Downloading…" onCancel={() => NativeLlmManager.getInstance().cancelDownload()}>
          <FileProgressList items={(DEFAULT_LLAMACPP_FILES.length ? DEFAULT_LLAMACPP_FILES : [modelId ?? 'model']).map((name, i, all) => {
            const current = Math.max(0, all.indexOf(modelId ?? ''));
            const done = i < current;
            const live = i === current;
            return {
              file: `${name}.gguf`,
              loaded: live ? downloadedBytes : 0,
              total: live ? totalBytes : 0,
              progress: done ? 100 : live ? downloadProgress : 0,
              done,
            };
          })} />
        </DownloadShell>
      );
    }
    if (status === 'loading') {
      return (
        <DownloadShell meta="Loading into memory…" onCancel={() => NativeLlmManager.getInstance().unloadModel()} />
      );
    }
    if (status === 'loaded') {
      return <DownloadShell meta="Ready on your device" />;
    }
    if (status === 'error' && error) {
      return <DownloadShell meta={error} />;
    }
    return null;
  }

  // transformers.js (browser): one shot that both downloads and loads
  if (gemma.status === 'loading') {
    return (
      <DownloadShell meta="Downloading…" onCancel={() => GemmaModelManager.getInstance().unloadModel()}>
        <FileProgressList items={gemma.progress.map(p => ({ file: p.file, loaded: p.loaded, total: p.total, progress: p.progress, done: p.status === 'done' }))} />
      </DownloadShell>
    );
  }
  if (gemma.status === 'loaded') {
    return <DownloadShell meta="Ready in your browser" />;
  }
  if (gemma.status === 'error' && gemma.error) {
    return <DownloadShell meta={gemma.error} />;
  }
  return null;
};

// ===================================================================================
//  MAIN COMPONENT
// ===================================================================================
const MCP: React.FC<MCPProps> = ({
  isAuthenticated,
  isUsingObServer,
  onRefresh,
  onSaveComplete,
  hideSuggestions,
  initialMessage,
  onOpenRecipe,
  heightClass = 'h-[350px] md:h-[450px]',
  boxed = true,
  agents,
  runningAgents,
  startingAgents,
  onToggleAgent,
  onSelectAgent,
}) => {
  // Conversation state lives in the app-level MCPProvider, so it's shared across every
  // place the MCP UI is opened (GetStarted, the modal) and survives this component
  // unmounting mid-run.
  const {
    messages,
    streamingText,
    isRunning,
    toolStatus,
    subscribeMutation,
    stop,
    send,
    modelName,
    setModelName,
  } = useMCPContext();

  const { isFloating, dock, setDockTarget, getSlotSize, flyToCorner } = useFloatingAgents();
  const [liveStreams, setLiveStreams] = useState<StreamState>(StreamManager.getCurrentState());
  useEffect(() => {
    StreamManager.addListener(setLiveStreams);
    return () => StreamManager.removeListener(setLiveStreams);
  }, []);

  // Each screen reacts to agent mutations in its own way; register this screen's reaction.
  useEffect(() => subscribeMutation((toolName, args) => {
    onRefresh?.();
    // A started agent leaves the chat for the top-right corner so it stays visible on scroll.
    if (toolName === 'start_agent' && typeof args?.id === 'string') flyToCorner(args.id);
    // Mirror the old "Save → close modal" UX once an agent is actually persisted.
    if (toolName === 'create_agent' || toolName === 'edit_agent') {
      onSaveComplete?.();
    }
  }), [subscribeMutation, onRefresh, onSaveComplete, flyToCorner]);

  const [userInput, setUserInput] = useState('');
  const [previewImages, setPreviewImages] = useState<string[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const hasInitialMessageSet = useRef(false);

  // --- Voice dictation -------------------------------------------------------
  // Reuses the exact transcription path agents use (StreamManager → TranscriptionRouter),
  // so it automatically honors the transcription mode chosen in Settings. The live
  // transcript streams into the input on top of whatever the user already typed.
  const { fullText: micTranscript } = useSubscriberText(MCP_MIC_ID, 'microphone');
  const [isRecording, setIsRecording] = useState(false);
  const [micStarting, setMicStarting] = useState(false);
  const micBaseRef = useRef('');
  // Bumped on every stop so an in-flight start (e.g. a slow local model load) knows it was
  // cancelled and tears down instead of flipping us back into recording.
  const micGenRef = useRef(0);

  // Mirror the streaming transcript into the input while recording.
  useEffect(() => {
    if (!isRecording) return;
    const base = micBaseRef.current;
    setUserInput(base && micTranscript ? `${base} ${micTranscript}` : base + micTranscript);
  }, [micTranscript, isRecording]);

  // Always tear the mic down on unmount so closing the modal mid-dictation can't leave a
  // hot mic / open WebSocket behind. Release + destroy are safe to call when already idle.
  useEffect(() => () => {
    StreamManager.releaseStreamsForAgent(MCP_MIC_ID);
    StreamManager.destroySubscribersForAgent(MCP_MIC_ID);
  }, []);

  const stopMic = () => {
    // Cancel any in-flight start, then flip the flag first so the reflect effect won't
    // overwrite the dictated text when the released service clears its subscriber.
    micGenRef.current++;
    setIsRecording(false);
    setMicStarting(false);
    StreamManager.releaseStreamsForAgent(MCP_MIC_ID);
    StreamManager.destroySubscribersForAgent(MCP_MIC_ID);
  };

  const startMic = async () => {
    micBaseRef.current = userInput.trim();
    setMicStarting(true);
    const gen = ++micGenRef.current;
    try {
      // Acquires the mic, starts the mode-appropriate service, wires PCM capture, and
      // creates our subscriber — all keyed on MCP_MIC_ID.
      await StreamManager.requestStreamsForAgent(MCP_MIC_ID, ['microphone']);
      // Stopped (or a message was sent) while we were still spinning up: tear down what we
      // just acquired instead of resurrecting recording.
      if (micGenRef.current !== gen) {
        StreamManager.releaseStreamsForAgent(MCP_MIC_ID);
        StreamManager.destroySubscribersForAgent(MCP_MIC_ID);
        return;
      }
      setIsRecording(true);
    } catch (e) {
      Logger.error('MCP', `Voice dictation failed to start: ${e}`);
      stopMic();
    } finally {
      if (micGenRef.current === gen) setMicStarting(false);
    }
  };

  const getCustomServerModels = (): Model[] =>
    ModelManager.getInstance().listModels().models.filter(m =>
      m.server !== ModelManager.BROWSER_LOCAL &&
      m.server !== ModelManager.LLAMA_CPP_LOCAL &&
      m.server !== ModelManager.SKIP_MODEL &&
      !m.server.includes('api.observer-ai.com')
    );

  const [customModels, setCustomModels] = useState<Model[]>(getCustomServerModels);

  useEffect(() => ModelManager.getInstance().onModelsChange(() => {
    setCustomModels(getCustomServerModels());
  }), []);

  useEffect(() => {
    if (customModels.length === 0) return;
    const stillAvailable = customModels.some(m => m.name === modelName);
    if (!stillAvailable && modelName !== 'gemini-2.5-flash-lite-free') {
      setModelName('gemini-2.5-flash-lite-free');
    }
  }, [customModels, modelName, setModelName]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

  useEffect(() => {
    if (initialMessage && !hasInitialMessageSet.current) {
      hasInitialMessageSet.current = true;
      setUserInput(initialMessage);
    }
  }, [initialMessage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // No isRunning gate: send() interrupts whatever's in flight (a slow model call or a hung
    // tool call) and takes over immediately, rather than queuing behind it.
    if (!userInput.trim() && previewImages.length === 0) return;
    const text = userInput.trim() || `[${previewImages.length} image${previewImages.length > 1 ? 's' : ''}]`;
    const images = previewImages;
    // Stop dictation so the reflect effect can't re-populate the box we're about to clear.
    stopMic();
    micBaseRef.current = '';
    setUserInput('');
    setPreviewImages([]);
    await send(text, images.length > 0 ? images : undefined);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        const base64 = result.split(',')[1] || result;
        setPreviewImages(prev => [...prev, base64]);
      };
      reader.readAsDataURL(file);
    });
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Simple two-pass render: turn every message into a small typed block (skipping the noise —
  // tool-result messages, the image-injection user message, and empty no-op turns), then merge
  // any run of consecutive 'tools' blocks into one before rendering. No message-stream state
  // machine — grouping falls out of "these blocks are next to each other in the list."
  type Block =
    | { type: 'user'; text: string; images: any[]; remote?: RemoteChannel }
    | { type: 'text'; content: string }
    | { type: 'tools'; calls: ToolCall[] };

  const renderMessages = () => {
    const blocks: Block[] = [];

    for (const msg of messages) {
      if (msg.role === 'tool') continue;
      if (msg.role === 'user' && Array.isArray(msg.content)
        && msg.content[0]?.text?.startsWith('Images from the tool result')) continue;

      if (msg.role === 'user') {
        const text = typeof msg.content === 'string'
          ? msg.content
          : (msg.content.find((p: any) => p.type === 'text')?.text ?? '');
        const images = Array.isArray(msg.content) ? msg.content.filter((p: any) => p.type === 'image_url') : [];
        // Messages sent from the phone carry a model-facing prefix; show an icon instead.
        const remote = parseRemotePrompt(text);
        blocks.push({ type: 'user', text: remote?.text ?? text, images, remote: remote?.channel });
        continue;
      }

      // assistant
      const content = typeof msg.content === 'string' ? msg.content : '';
      const toolCalls = msg.tool_calls || [];
      if (content) blocks.push({ type: 'text', content });
      if (toolCalls.length > 0) blocks.push({ type: 'tools', calls: toolCalls });
    }

    // Merge consecutive 'tools' blocks.
    const merged: (Block | { type: 'tools'; calls: ToolCall[]; batches: ToolCall[][] })[] = [];
    for (const block of blocks) {
      const prev = merged[merged.length - 1];
      if (block.type === 'tools' && prev && prev.type === 'tools') {
        (prev as any).calls.push(...block.calls);
        (prev as any).batches.push(block.calls);
      } else if (block.type === 'tools') {
        merged.push({ type: 'tools', calls: [...block.calls], batches: [block.calls] });
      } else {
        merged.push(block);
      }
    }

    return merged.map((block, idx) => {
      if (block.type === 'user') {
        return (
          <div key={idx} className="flex flex-col items-end">
            {block.remote && (
              <span className="text-xs text-gray-400 mb-1 mr-1">
                {block.remote === 'whatsapp' ? 'WhatsApp' : 'Telegram'}
              </span>
            )}
            <div className={userBubbleClass}>
              {block.text && <p className="whitespace-pre-wrap">{block.text}</p>}
              {block.images.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                  {block.images.map((p: any, j: number) => (
                    <img key={j} src={p.image_url.url} alt="" className="max-w-[120px] h-auto rounded-lg" />
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      }
      if (block.type === 'text') {
        return (
          <div key={idx} className="flex justify-start">
            <div className={assistantBubbleClass}><Markdown text={block.content} /></div>
          </div>
        );
      }
      // tools
      const { calls, batches } = block as { calls: ToolCall[]; batches: ToolCall[][] };
      return (
        <div key={idx} className="flex flex-col items-start w-full">
          <WorkingGroup batches={batches} toolStatus={toolStatus} messages={messages} />
          {calls.some(tc => tc.function.name === 'download_model') && <DownloadModelProgress />}
          {calls
            .filter(tc => tc.function.name === 'check_whitelist')
            .map(tc => (
              <CheckWhitelistGate key={tc.id} toolCallId={tc.id} status={toolStatus.get(tc.id)} onCancel={stop} />
            ))}
          {agents && calls
            .filter(tc => tc.function.name === 'create_agent')
            .map(tc => {
              const resultMsg = messages.find(m => m.role === 'tool' && m.tool_call_id === tc.id);
              if (!resultMsg) return null; // still running
              let createdId: string | undefined;
              try { createdId = JSON.parse(resultMsg.content as string)?.id; } catch { /* malformed/error result — nothing to show */ }
              if (!createdId) return null;
              const agentObj = agents.find(a => a.id === createdId);
              if (!agentObj) return null; // agents list hasn't refreshed yet

              if (isFloating(createdId)) {
                // Registered as this agent's dock target — dragging the floating card back
                // onto this specific pill (not just "anywhere in the chat") is what re-docks
                // it; see FloatingAgentsContext.tsx for why the hit zone has to be this small.
                // Same footprint as the inline card it stands in for, so the transcript
                // doesn't reflow when the card leaves or comes back.
                const slot = getSlotSize(createdId);
                return (
                  <div key={tc.id} className="mt-2 w-full">
                    <button
                      ref={(el) => setDockTarget(createdId!, el)}
                      onClick={() => dock(createdId!)}
                      style={slot ? { width: slot.width, height: slot.height } : undefined}
                      className="max-w-[350px] w-full min-h-[72px] flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-gray-200 bg-gray-50 text-xs text-gray-500 hover:bg-gray-100 transition-colors px-4 text-center"
                    >
                      <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse flex-shrink-0" />
                      <span><span className="font-medium text-gray-700">{agentObj.name}</span> is floating — drop it here (or tap) to bring back</span>
                    </button>
                  </div>
                );
              }

              return (
                <div key={tc.id} className="mt-2 w-full">
                  <AgentLiveCard
                    agent={agentObj}
                    isRunning={runningAgents?.has(createdId) ?? false}
                    isStarting={startingAgents?.has(createdId) ?? false}
                    streams={liveStreams}
                    mode="inline"
                    onToggle={onToggleAgent ?? (() => {})}
                    onSelectAgent={onSelectAgent}
                  />
                </div>
              );
            })}
        </div>
      );
    });
  };

  const SUGGESTIONS = [
    'Call me when my download is finished',
    'WhatsApp me when my video finishes rendering',
    'Log my screen activity every hour',
    'Notify me when my battery is low',
  ];

  // Not gated on isRunning: typing/sending is always available — a send interrupts whatever
  // tool/model call is in flight instead of waiting for it.
  const isInputDisabled = isUsingObServer && !isAuthenticated;
  const isTextEmpty = !userInput.trim() && previewImages.length === 0;
  const isSendDisabled = isInputDisabled || isTextEmpty;
  const showSuggestions = !hideSuggestions && messages.length === 0 && !isRunning;

  const getPlaceholder = () => {
    if (isUsingObServer && !isAuthenticated) return 'Enable Ob-Server and log in to use Observer';
    return 'Describe what you want monitored…';
  };

  // "Girth" pass for the unboxed (boxed=false) surface: rounder, roomier bubbles and a plain
  // background instead of the default boxed card's tighter, bordered look. Existing callers
  // never pass boxed=false, so their rendering is untouched.
  const assistantBubbleClass = boxed
    ? 'max-w-xs md:max-w-md p-2 md:p-3 rounded-lg text-sm md:text-base bg-gray-100 text-gray-800 shadow-sm'
    : 'max-w-md md:max-w-2xl p-3 md:p-4 rounded-2xl text-sm md:text-base bg-gray-100 text-gray-800';
  const userBubbleClass = boxed
    ? 'max-w-xs md:max-w-md p-2 md:p-3 rounded-lg text-sm md:text-base bg-gray-700 text-white'
    : 'max-w-md md:max-w-2xl p-3 md:p-4 rounded-2xl text-sm md:text-base bg-gray-700 text-white';

  return (
    <div className={`flex flex-col ${heightClass} relative ${boxed ? 'bg-white rounded-lg border border-purple-200' : 'bg-transparent'}`}>
      {/* Chat Messages */}
      <div className={boxed ? 'flex-1 p-3 md:p-4 space-y-3 md:space-y-4 overflow-y-auto' : 'flex-1 p-4 md:p-6 space-y-3 md:space-y-4 overflow-y-auto'}>
        {messages.length === 0 && (
          <div className="flex justify-start">
            <div className={assistantBubbleClass}>
              <Markdown text={`Hi! I'm **Observer**! I can create and run micro-agents. `} />
            </div>
          </div>
        )}

        {renderMessages()}

        {streamingText && (
          <div className="flex justify-start">
            <div className={`${assistantBubbleClass} animate-pulse`}>
              <Markdown text={streamingText} />
            </div>
          </div>
        )}

        {isRunning && !streamingText && (
          <div className="flex justify-start">
            <div className={`text-gray-800 p-2 md:p-3 inline-flex items-center ${boxed ? 'bg-gray-100 rounded-lg shadow-sm' : 'bg-gray-100 rounded-2xl'}`}>
              <Loader2 className="h-4 w-4 md:h-5 md:w-5 animate-spin" />
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Image previews */}
      {previewImages.length > 0 && (
        <div className="px-3 pb-1 flex flex-wrap gap-2">
          {previewImages.map((img, i) => (
            <div key={i} className="relative">
              <img src={`data:image/png;base64,${img}`} alt="" className="h-12 w-12 object-cover rounded border" />
              <button
                onClick={() => setPreviewImages(prev => prev.filter((_, idx) => idx !== i))}
                className="absolute -top-1.5 -right-1.5 bg-gray-700 text-white rounded-full p-0.5"
              >
                <XCircle className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Suggestion chips */}
      {showSuggestions && (
        <div className="px-3 pt-2 pb-1 flex flex-wrap justify-center gap-2">
          {onOpenRecipe && <RecipeMini onClick={onOpenRecipe} />}
          {SUGGESTIONS.map(s => (
            <button
              key={s}
              type="button"
              onClick={() => send(s)}
              className="text-xs px-3 py-1.5 rounded-lg border border-gray-200 bg-white text-gray-500 hover:text-gray-700 hover:border-gray-300 transition-colors whitespace-nowrap"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input Area */}
      <div className={boxed ? 'p-2 border-t border-purple-200 bg-white/80 backdrop-blur-sm rounded-b-lg' : 'p-3 md:p-4'}>
        {customModels.length > 0 && (
          <div className="flex items-center gap-1.5 px-1 pb-1.5">
            <Cpu className="h-3.5 w-3.5 text-purple-400 flex-shrink-0" />
            <select
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              disabled={isRunning}
              className="flex-1 text-xs text-gray-600 bg-transparent border-0 focus:ring-0 focus:outline-none cursor-pointer disabled:cursor-not-allowed truncate"
            >
              <option value="gemini-2.5-flash-lite-free">Default (cloud)</option>
              {customModels.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
          </div>
        )}
        {(isRecording || micStarting) && (
          <div className="flex items-center gap-2 px-1 pb-1.5 text-xs font-medium text-red-600">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
            </span>
            <span>{micStarting ? 'Starting microphone…' : 'Listening… tap the mic to stop'}</span>
          </div>
        )}
        <form onSubmit={handleSubmit} className="flex items-center gap-1.5 md:gap-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder={getPlaceholder()}
            disabled={isInputDisabled}
            className={`flex-1 min-w-0 text-sm md:text-base text-gray-700 disabled:bg-gray-100 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
              boxed
                ? 'p-2 md:p-3 border border-gray-200 rounded-lg'
                : 'p-3 md:p-4 border border-gray-200 rounded-full bg-white shadow-sm'
            }`}
          />

          <input ref={fileInputRef} type="file" accept="image/*" multiple onChange={handleFileChange} className="hidden" />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isInputDisabled}
            className={`bg-gray-700 text-white hover:bg-gray-800 disabled:bg-gray-300 transition-colors flex items-center flex-shrink-0 ${boxed ? 'p-2 md:p-3 rounded-md' : 'p-3 md:p-4 rounded-full'}`}
            title="Upload Image"
          >
            <Plus className="h-5 w-5" />
          </button>

          <button
            type="button"
            onClick={() => (isRecording ? stopMic() : startMic())}
            disabled={micStarting || (!isRecording && isInputDisabled)}
            className={`flex items-center justify-center flex-shrink-0 transition-colors ${boxed ? 'p-2 md:p-3 rounded-lg' : 'p-3 md:p-4 rounded-full'} ${
              isRecording
                ? 'bg-red-600 text-white hover:bg-red-700 animate-pulse'
                : 'bg-gray-700 text-white hover:bg-gray-800 disabled:bg-gray-300 disabled:cursor-not-allowed'
            }`}
            title={isRecording ? 'Stop voice input' : 'Speak to fill the message'}
            aria-label={isRecording ? 'Stop voice input' : 'Start voice input'}
          >
            {micStarting
              ? <Loader2 className="h-5 w-5 animate-spin" />
              : <Mic className="h-5 w-5" />}
          </button>

          {isRunning && isTextEmpty ? (
            // Nothing typed yet: offer a plain stop. As soon as they type, this swaps to Send,
            // which interrupts the run and takes over instead of stopping it outright.
            <button
              type="button"
              onClick={stop}
              className={`bg-red-600 text-white hover:bg-red-700 transition-colors flex items-center flex-shrink-0 ${boxed ? 'p-2 md:p-3 rounded-lg' : 'p-3 md:p-4 rounded-full'}`}
              title="Stop"
            >
              <Square className="h-4 w-4" fill="currentColor" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={isSendDisabled}
              className={`bg-gray-700 text-white hover:bg-gray-800 disabled:bg-gray-300 transition-colors flex items-center flex-shrink-0 ${boxed ? 'p-2 md:p-3 rounded-lg' : 'p-3 md:p-4 rounded-full'}`}
              title="Send"
            >
              <Send className="h-4 w-4" />
            </button>
          )}
        </form>
      </div>
    </div>
  );
};

export default MCP;
