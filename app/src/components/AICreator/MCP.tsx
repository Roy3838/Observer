// src/components/AICreator/MCP.tsx
//
// Native tool-calling agent creator. Replaces MultiAgentCreator: instead of brittle
// $$$/%%%/&&& text delimiters, the model drives agent creation/management through real
// OpenAI function calls (see src/mcp/). This component is pure UI over the useMCP hook.

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Users, Plus, CheckCircle2, XCircle, Loader, Play, Square, Save, Download, Cpu, Sparkles, StopCircle, Mic } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import type { TokenProvider } from '@utils/main_loop';
import { type ToolStatusEntry } from '../../mcp/useMCP';
import { useMCPContext } from '../../mcp/MCPContext';
import type { WireMessage, ToolCall } from '../../mcp/types';
import { Logger, type WhitelistChannel } from '@utils/logging';
import { StreamManager } from '@utils/streamManager';
import { useSubscriberText } from '@hooks/useTranscriptionState';
import WhitelistInline from '@components/whitelist/WhitelistInline';
import { isTauri } from '@utils/platform';
import { tauriStreamCapture, type CaptureTarget } from '@utils/tauriStreamCapture';
import { Monitor } from 'lucide-react';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import type { GemmaModelState, NativeModelState } from '@utils/localLlm/types';
import { ModelManager, type Model } from '@utils/ModelManager';
import RecipeMini from './RecipeMini';

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
    case 'done': return <CheckCircle2 className="h-4 w-4 text-green-600" />;
    case 'error':
    case 'denied': return <XCircle className="h-4 w-4 text-red-500" />;
    case 'running': return <Loader className="h-4 w-4 text-purple-600 animate-spin" />;
    default: return <Loader className="h-4 w-4 text-gray-400 animate-spin" />;
  }
};

const ToolChip: React.FC<{ call: ToolCall; status?: ToolStatusEntry }> = ({ call, status }) => (
  <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-gray-100 border border-gray-200 text-xs font-medium text-gray-700 mr-1.5 mt-1.5">
    <StatusIcon status={status?.status} />
    <span className="font-mono">{call.function.name}</span>
  </div>
);

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
  const phoneNumber: string | undefined = status?.args?.phone_number;
  if (!phoneNumber) return null;
  const channel = status?.args?.channel as WhitelistChannel | undefined;
  return <WhitelistInline phoneNumber={phoneNumber} channel={channel} onCancel={onCancel} />;
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

const Bar: React.FC<{ pct: number; done?: boolean }> = ({ pct, done }) => (
  <div className="w-full bg-gray-200 rounded-full h-1.5">
    <div
      className={`h-1.5 rounded-full transition-all duration-300 ${done ? 'bg-green-500' : 'bg-purple-600'}`}
      style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
    />
  </div>
);

/**
 * Subscribes directly to the local-model managers (the same state ModelHub renders) to show
 * live progress for the in-flight `download_model` tool call. Renders nothing when idle.
 */
const DownloadShell: React.FC<{ icon: React.ReactNode; children: React.ReactNode; onCancel?: () => void }> = ({ icon, children, onCancel }) => (
  <div className="mt-2 w-full max-w-md p-3 rounded-lg border border-purple-200 bg-white/70">
    <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-gray-700">
      {icon}<span>On-device model</span>
      {onCancel && (
        <button onClick={onCancel} className="ml-auto flex items-center gap-1 text-red-500 hover:text-red-700 font-medium">
          <StopCircle size={11} /> Cancel
        </button>
      )}
    </div>
    {children}
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
        <DownloadShell icon={<Download className="h-4 w-4 text-purple-600 animate-bounce" />} onCancel={() => NativeLlmManager.getInstance().cancelDownload()}>
          <div className="flex justify-between text-[11px] text-gray-600 mb-1">
            <span className="truncate max-w-[60%]">{modelId ?? 'model'}.gguf</span>
            <span className="font-medium">
              {totalBytes > 0 ? `${formatBytes(downloadedBytes)} / ${formatBytes(totalBytes)}` : `${Math.round(downloadProgress)}%`}
            </span>
          </div>
          <Bar pct={downloadProgress} />
        </DownloadShell>
      );
    }
    if (status === 'loading') {
      return (
        <DownloadShell icon={<Cpu className="h-4 w-4 text-purple-600 animate-pulse" />} onCancel={() => NativeLlmManager.getInstance().unloadModel()}>
          <p className="text-xs text-gray-600">Loading model into memory…</p>
        </DownloadShell>
      );
    }
    if (status === 'loaded') {
      return <DownloadShell icon={<CheckCircle2 className="h-4 w-4 text-green-600" />}><p className="text-xs text-gray-600">Model ready on your device.</p></DownloadShell>;
    }
    if (status === 'error' && error) {
      return <DownloadShell icon={<XCircle className="h-4 w-4 text-red-500" />}><p className="text-xs text-red-600">{error}</p></DownloadShell>;
    }
    return null;
  }

  // transformers.js (browser): one shot that both downloads and loads
  if (gemma.status === 'loading') {
    return (
      <DownloadShell icon={<Sparkles className="h-4 w-4 text-purple-600 animate-pulse" />} onCancel={() => GemmaModelManager.getInstance().unloadModel()}>
        {gemma.progress.length > 0 ? (
          <div className="space-y-1.5">
            {gemma.progress.map(item => (
              <div key={item.file}>
                <div className="flex justify-between text-[11px] text-gray-600 mb-1">
                  <span className="truncate max-w-[60%]">{item.file}</span>
                  <span className="font-medium">
                    {item.status === 'done' ? 'Done' : item.total > 0 ? `${formatBytes(item.loaded)} / ${formatBytes(item.total)}` : `${Math.round(item.progress)}%`}
                  </span>
                </div>
                <Bar pct={item.progress} done={item.status === 'done'} />
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-gray-600">Downloading & loading…</p>
        )}
      </DownloadShell>
    );
  }
  if (gemma.status === 'loaded') {
    return <DownloadShell icon={<CheckCircle2 className="h-4 w-4 text-green-600" />}><p className="text-xs text-gray-600">Model ready in your browser.</p></DownloadShell>;
  }
  if (gemma.status === 'error' && gemma.error) {
    return <DownloadShell icon={<XCircle className="h-4 w-4 text-red-500" />}><p className="text-xs text-red-600">{gemma.error}</p></DownloadShell>;
  }
  return null;
};

// ===================================================================================
//  AGENT APPROVAL CARD  (the human gate — a tool whose promise resolves from the UI)
// ===================================================================================
interface ApprovalCall { id: string; name: string; args: any; }

/**
 * Preview for select_screen_target / set_screen_crop calls: shows the chosen monitor/window
 * thumbnail (fetched live from the same source of truth the capture path reads) with the
 * proposed crop drawn as an overlay box, so the user sees exactly what will be captured.
 */
const ScreenCaptureApproval: React.FC<{
  targetId?: string;
  crop?: { x: number; y: number; width: number; height: number };
  clearCrop?: boolean;
}> = ({ targetId, crop, clearCrop }) => {
  const [target, setTarget] = useState<CaptureTarget | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    tauriStreamCapture.getTargets(true)
      .then(targets => { if (!cancelled) setTarget(targets.find(t => t.id === targetId) ?? null); })
      .catch(e => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, [targetId]);

  const thumb = target?.thumbnail
    ? (target.thumbnail.startsWith('data:') ? target.thumbnail : `data:image/png;base64,${target.thumbnail}`)
    : null;

  // Crop overlay expressed as % of the target's pixel space, so it tracks the scaled thumbnail.
  const overlay = crop && target
    ? {
        left: `${(crop.x / target.width) * 100}%`,
        top: `${(crop.y / target.height) * 100}%`,
        width: `${(crop.width / target.width) * 100}%`,
        height: `${(crop.height / target.height) * 100}%`,
      }
    : null;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3">
      <div className="flex items-center mb-2">
        <Monitor className="h-4 w-4 text-blue-600 mr-2" />
        <span className="text-sm text-gray-800">
          Capture {target ? <strong>{target.name}</strong> : <code className="font-mono text-xs">{targetId}</code>}
          {target?.appName && target.appName !== target.name ? ` (${target.appName})` : ''}
          {clearCrop ? ' · full screen' : crop ? ` · cropped to ${crop.width}×${crop.height}` : ''}
        </span>
      </div>
      {error && <p className="text-xs text-red-500">{error}</p>}
      {thumb && (
        <div className="relative inline-block max-w-full rounded overflow-hidden border border-gray-200">
          <img src={thumb} alt={target?.name ?? 'screen'} className="block max-h-48 w-auto" />
          {overlay && (
            <div
              className="absolute border-2 border-blue-500 bg-blue-500/20"
              style={overlay}
            />
          )}
        </div>
      )}
    </div>
  );
};

const AgentApprovalCard: React.FC<{
  calls: ApprovalCall[];
  onDecision: (approved: boolean) => void;
}> = ({ calls, onDecision }) => {
  const agentBuilds = calls.filter(c => c.name === 'create_agent' || c.name === 'edit_agent');
  const lifecycle = calls.filter(c => c.name === 'start_agent' || c.name === 'stop_agent');
  const targetSelect = calls.find(c => c.name === 'select_screen_target');
  const cropCalls = calls.filter(c => c.name === 'set_screen_crop');
  // Pair a crop with its target so the overlay can be drawn on the right thumbnail.
  const cropForTarget = cropCalls.find(c => !c.args.clear)?.args;
  const showScreenCapture = !!targetSelect || cropCalls.length > 0;

  return (
    <div className="w-full bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-4 md:p-6">
      <div className="flex items-center justify-center mb-4">
        <Users className="h-6 w-6 text-purple-600 mr-2" />
        <h3 className="text-lg font-bold text-purple-800">
          {agentBuilds.length > 0
            ? `Review ${agentBuilds.length} proposed agent${agentBuilds.length === 1 ? '' : 's'}`
            : 'Confirm action'}
        </h3>
      </div>

      {agentBuilds.length > 0 && (
        <div className="space-y-4 mb-4">
          {agentBuilds.map((c, index) => (
            <div key={c.id} className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center mb-3">
                <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center mr-3">
                  <span className="text-purple-600 font-bold text-sm">{index + 1}</span>
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">{c.args.name || c.args.id}</h4>
                  <p className="text-sm text-gray-600">{c.args.description}</p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {c.name === 'edit_agent' ? 'edit' : 'create'} · {c.args.model_name} · every {c.args.loop_interval_seconds ?? 30}s
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div>
                  <h5 className="font-medium text-gray-700 mb-2">System Prompt:</h5>
                  <div className="bg-gray-50 border rounded p-2 max-h-24 overflow-y-auto">
                    <code className="text-gray-600 whitespace-pre-wrap text-xs">{c.args.system_prompt}</code>
                  </div>
                </div>
                <div>
                  <h5 className="font-medium text-gray-700 mb-2">Code:</h5>
                  <div className="bg-gray-50 border rounded p-2 max-h-24 overflow-y-auto">
                    <code className="text-gray-600 whitespace-pre-wrap text-xs">{c.args.code}</code>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {showScreenCapture && (
        <div className="space-y-2 mb-4">
          {targetSelect ? (
            <ScreenCaptureApproval
              targetId={targetSelect.args.target_id}
              crop={cropForTarget ? { x: cropForTarget.x, y: cropForTarget.y, width: cropForTarget.width, height: cropForTarget.height } : undefined}
            />
          ) : (
            // Crop without a target in the same batch (e.g. re-cropping a running agent).
            cropCalls.map(c => (
              <div key={c.id} className="bg-white border border-gray-200 rounded-lg p-3 flex items-center">
                <Monitor className="h-4 w-4 text-blue-600 mr-2" />
                <span className="text-sm text-gray-800">
                  {c.args.clear
                    ? <>Remove screen crop for <code className="font-mono">{c.args.agent_id}</code></>
                    : <>Crop <code className="font-mono">{c.args.agent_id}</code> to {c.args.width}×{c.args.height} at ({c.args.x}, {c.args.y})</>}
                </span>
              </div>
            ))
          )}
        </div>
      )}

      {lifecycle.length > 0 && (
        <div className="space-y-2 mb-4">
          {lifecycle.map(c => (
            <div key={c.id} className="bg-white border border-gray-200 rounded-lg p-3 flex items-center">
              {c.name === 'start_agent'
                ? <Play className="h-4 w-4 text-green-600 mr-2" />
                : <Square className="h-4 w-4 text-red-500 mr-2" />}
              <span className="text-sm text-gray-800">
                {c.name === 'start_agent' ? 'Start' : 'Stop'} agent <code className="font-mono">{c.args.id}</code>
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center justify-center gap-3">
        <button
          onClick={() => onDecision(false)}
          className="px-5 py-2.5 text-base bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 font-medium transition-colors"
        >
          Deny
        </button>
        <button
          onClick={() => onDecision(true)}
          className="px-6 py-2.5 text-base bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg hover:from-purple-700 hover:to-indigo-700 font-medium transition-colors flex items-center shadow-lg"
        >
          <Save className="h-5 w-5 mr-2" />
          Approve
        </button>
      </div>
    </div>
  );
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
}) => {
  // Conversation state lives in the app-level MCPProvider, so it's shared across every
  // place the MCP UI is opened (GetStarted, the modal) and survives this component
  // unmounting mid-run.
  const {
    messages,
    streamingText,
    isRunning,
    toolStatus,
    pendingApproval,
    resolveInteraction,
    subscribeMutation,
    stop,
    send,
    modelName,
    setModelName,
  } = useMCPContext();

  // Each screen reacts to agent mutations in its own way; register this screen's reaction.
  useEffect(() => subscribeMutation((toolName) => {
    onRefresh?.();
    // Mirror the old "Save → close modal" UX once an agent is actually persisted.
    if (toolName === 'create_agent' || toolName === 'edit_agent') {
      onSaveComplete?.();
    }
  }), [subscribeMutation, onRefresh, onSaveComplete]);

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
  }, [messages, streamingText, pendingApproval]);

  useEffect(() => {
    if (initialMessage && !hasInitialMessageSet.current) {
      hasInitialMessageSet.current = true;
      setUserInput(initialMessage);
    }
  }, [initialMessage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!userInput.trim() && previewImages.length === 0) || isRunning) return;
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

  const handleApproval = (approved: boolean) => {
    if (!pendingApproval) return;
    // onSaveComplete / onRefresh fire from useMCP's onAgentMutated once the save resolves.
    resolveInteraction(pendingApproval.batchId, { approved });
  };

  const renderMessage = (msg: WireMessage, idx: number) => {
    // Hide internal tool-result messages and the runner's image-injection user message.
    if (msg.role === 'tool') return null;
    if (msg.role === 'user' && Array.isArray(msg.content)
      && msg.content[0]?.text?.startsWith('Images from the tool result')) {
      return null;
    }

    if (msg.role === 'user') {
      const text = typeof msg.content === 'string'
        ? msg.content
        : (msg.content.find((p: any) => p.type === 'text')?.text ?? '');
      const imageParts = Array.isArray(msg.content)
        ? msg.content.filter((p: any) => p.type === 'image_url')
        : [];
      return (
        <div key={idx} className="flex justify-end">
          <div className="max-w-xs md:max-w-md p-2 md:p-3 rounded-lg text-sm md:text-base bg-purple-600 text-white">
            {text && <p className="whitespace-pre-wrap">{text}</p>}
            {imageParts.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2">
                {imageParts.map((p: any, i: number) => (
                  <img key={i} src={p.image_url.url} alt="" className="max-w-[120px] h-auto rounded-lg" />
                ))}
              </div>
            )}
          </div>
        </div>
      );
    }

    // assistant
    const content = typeof msg.content === 'string' ? msg.content : '';
    const toolCalls = msg.tool_calls || [];
    if (!content && toolCalls.length === 0) return null;
    return (
      <div key={idx} className="flex justify-start">
        <div className="max-w-xs md:max-w-md p-2 md:p-3 rounded-lg text-sm md:text-base bg-gradient-to-br from-purple-50 to-indigo-50 text-gray-800 shadow-sm">
          {content && <Markdown text={content} />}
          {toolCalls.length > 0 && (
            <div className="flex flex-wrap">
              {toolCalls.map(tc => (
                <ToolChip key={tc.id} call={tc} status={toolStatus.get(tc.id)} />
              ))}
            </div>
          )}
          {toolCalls.some(tc => tc.function.name === 'download_model') && <DownloadModelProgress />}
          {toolCalls
            .filter(tc => tc.function.name === 'check_whitelist')
            .map(tc => (
              <CheckWhitelistGate key={tc.id} toolCallId={tc.id} status={toolStatus.get(tc.id)} onCancel={stop} />
            ))}
        </div>
      </div>
    );
  };

  const SUGGESTIONS = [
    'Call me when my download is finished',
    'WhatsApp me when my video finishes rendering',
    'Log my screen activity every hour',
    'Notify me when my battery is low',
  ];

  const isInputDisabled = isRunning || (isUsingObServer && !isAuthenticated) || !!pendingApproval;
  const isSendDisabled = isInputDisabled || (!userInput.trim() && previewImages.length === 0);
  const showSuggestions = !hideSuggestions && messages.length === 0 && !isRunning && !pendingApproval;

  const getPlaceholder = () => {
    if (isUsingObServer && !isAuthenticated) return 'Enable Ob-Server and log in to use Observer';
    if (pendingApproval) return 'Approve or deny the proposed action above…';
    return 'Describe what you want monitored…';
  };

  return (
    <div className={`flex flex-col ${heightClass} bg-white rounded-lg border border-purple-200 relative`}>
      {/* Chat Messages */}
      <div className="flex-1 p-3 md:p-4 space-y-3 md:space-y-4 overflow-y-auto">
        {messages.length === 0 && (
          <div className="flex justify-start">
            <div className="max-w-xs md:max-w-md p-2 md:p-3 rounded-lg text-sm md:text-base bg-gradient-to-br from-purple-50 to-indigo-50 text-gray-800 shadow-sm">
              <Markdown text={`Hi! I'm **Observer**! I can create and run micro-agents. `} />
            </div>
          </div>
        )}

        {messages.map(renderMessage)}

        {streamingText && (
          <div className="flex justify-start">
            <div className="max-w-xs md:max-w-md p-2 md:p-3 rounded-lg text-sm md:text-base bg-gradient-to-br from-purple-50 to-indigo-50 text-gray-800 shadow-sm animate-pulse">
              <Markdown text={streamingText} />
            </div>
          </div>
        )}

        {pendingApproval && (
          <div className="flex justify-start w-full">
            <AgentApprovalCard calls={pendingApproval.calls} onDecision={handleApproval} />
          </div>
        )}

        {isRunning && !streamingText && !pendingApproval && (
          <div className="flex justify-start">
            <div className="bg-gradient-to-br from-purple-50 to-indigo-50 text-gray-800 p-2 md:p-3 rounded-lg inline-flex items-center shadow-sm">
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
      <div className="p-2 border-t border-purple-200 bg-white/80 backdrop-blur-sm rounded-b-lg">
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
            className="flex-1 min-w-0 p-2 md:p-3 border border-purple-300 rounded-lg text-sm md:text-base text-gray-700 disabled:bg-gray-100 disabled:cursor-not-allowed focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
          />

          <input ref={fileInputRef} type="file" accept="image/*" multiple onChange={handleFileChange} className="hidden" />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isInputDisabled}
            className="p-2 md:p-3 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-300 transition-colors flex items-center flex-shrink-0"
            title="Upload Image"
          >
            <Plus className="h-5 w-5" />
          </button>

          <button
            type="button"
            onClick={() => (isRecording ? stopMic() : startMic())}
            disabled={micStarting || (!isRecording && isInputDisabled)}
            className={`p-2 md:p-3 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors ${
              isRecording
                ? 'bg-red-600 text-white hover:bg-red-700 animate-pulse'
                : 'bg-purple-600 text-white hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed'
            }`}
            title={isRecording ? 'Stop voice input' : 'Speak to fill the message'}
            aria-label={isRecording ? 'Stop voice input' : 'Start voice input'}
          >
            {micStarting
              ? <Loader2 className="h-5 w-5 animate-spin" />
              : <Mic className="h-5 w-5" />}
          </button>

          {isRunning ? (
            <button
              type="button"
              onClick={stop}
              className="p-2 md:p-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center flex-shrink-0"
              title="Stop"
            >
              <Square className="h-4 w-4" fill="currentColor" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={isSendDisabled}
              className="p-2 md:p-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-300 transition-colors flex items-center flex-shrink-0"
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
