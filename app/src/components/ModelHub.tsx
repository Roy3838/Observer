// components/ModelHub.tsx
//
// Acquisition + hardware config panel for local AI models, custom servers, and Ob-Server cloud.
// Primary path: curated model catalog. Advanced: raw GGUF download, custom servers, sampler settings.

import React, { useState, useEffect, useRef } from 'react';
import Modal from '@components/EditAgent/Modal';
import {
  Download, CheckCircle, AlertTriangle, X, StopCircle, FileDown, Cpu, Trash2,
  AlertCircle, Settings2, RefreshCw, Plus, Cloud, Server, Zap,
  Sparkles, Package, ChevronDown, Terminal, Play, Eye,
} from 'lucide-react';
import { Logger, LogEntry, LogLevel } from '@utils/logging';
import pullModelManager, { PullState } from '@utils/pullModelManager';
import { platformFetch, isTauri, isWeb } from '@utils/platform';
import { invoke } from '@tauri-apps/api/core';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import type { CustomServer } from '@utils/inferenceServer';
import {
  GemmaModelState,
  GemmaDevice,
  GemmaDtype,
  GemmaImageTokenBudget,
  GemmaModelId,
  LocalModelEntry,
  GgufFileInfo,
  NativeModelState,
  SamplerParams,
  DEFAULT_SAMPLER_PARAMS,
  ContextParams,
  DEFAULT_CONTEXT_PARAMS,
} from '@utils/localLlm/types';
import { MODEL_PRESETS, type ModelPreset } from '@utils/modelPresets';
import LocalServerSetupDialog from '@components/LocalServerSetupDialog';

type QuotaInfo = {
  used: number;
  remaining: number;
  limit: number;
  tier: string;
} | null;

type TabId = 'llamacpp' | 'transformers' | 'servers';
type TabColor = 'green' | 'purple' | 'blue' | 'orange';

interface ModelHubProps {
  isOpen: boolean;
  onClose: () => void;
  onPullComplete?: () => void;
  autoDownloadPreset?: ModelPreset;
  ollamaServers?: string[];
  isUsingObServer?: boolean;
  handleToggleObServer?: () => void;
  showLoginMessage?: boolean;
  isAuthenticated?: boolean;
  quotaInfo?: QuotaInfo;
  renderQuotaStatus?: () => React.ReactNode;
  localServerOnline?: boolean;
  checkLocalServer?: () => void;
  customServers?: CustomServer[];
  onAddCustomServer?: (address: string) => void;
  onRemoveCustomServer?: (address: string) => void;
  onToggleCustomServer?: (address: string) => void;
  onCheckCustomServer?: (address: string) => void;
  appInferenceUrl?: string | null;
  onSetAppInferenceUrl?: (url: string) => void;
}

const SUGGESTED_OLLAMA_MODELS = [
  'gemma3:4b', 'gemma3:12b', 'gemma3:27b', 'gemma3:27b-it-qat',
  'qwen2.5vl:3b', 'qwen2.5vl:7b', 'llava:7b', 'llava:13b',
];

const formatBytes = (bytes: number, decimals = 2) => {
  if (!+bytes) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};

const TAB_COLORS: Record<TabColor, { active: string; dot: string }> = {
  green:  { active: 'bg-white text-gray-800 shadow-sm',   dot: 'bg-gray-500' },
  purple: { active: 'bg-white text-yellow-800 shadow-sm', dot: 'bg-yellow-400' },
  blue:   { active: 'bg-white text-blue-700 shadow-sm',   dot: 'bg-blue-500' },
  orange: { active: 'bg-white text-orange-700 shadow-sm', dot: 'bg-orange-500' },
};

const TabButton: React.FC<{
  active: boolean;
  onClick: () => void;
  color: TabColor;
  icon: React.ReactNode;
  label: string;
  dimmed?: boolean;
}> = ({ active, onClick, color, icon, label, dimmed }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg transition-all whitespace-nowrap flex-1 justify-center ${
      active ? TAB_COLORS[color].active : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
    } ${dimmed ? 'opacity-60' : ''}`}
  >
    <span className={`w-1.5 h-1.5 rounded-full ${active ? TAB_COLORS[color].dot : 'bg-gray-300'}`} />
    {icon}
    <span className="hidden sm:inline">{label}</span>
    {dimmed && <AlertCircle size={11} className="text-gray-400" />}
  </button>
);

const SamplerSlider: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  hint?: string;
  format: (v: number) => string;
  onChange: (v: number) => void;
}> = ({ label, value, min, max, step, disabled, hint, format, onChange }) => (
  <div>
    <div className="flex justify-between items-center mb-1">
      <label className="text-xs font-medium text-gray-600">{label}</label>
      <span className="text-xs text-gray-500 font-mono">{format(value)}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      disabled={disabled}
      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-green-600"
    />
    {hint && <p className="text-xs text-gray-400 mt-0.5">{hint}</p>}
  </div>
);

const ModelHub: React.FC<ModelHubProps> = ({
  isOpen,
  onClose,
  onPullComplete,
  autoDownloadPreset,
  ollamaServers,
  isUsingObServer = false,
  handleToggleObServer,
  showLoginMessage = false,
  isAuthenticated = false,
  quotaInfo,
  renderQuotaStatus,
  localServerOnline = false,
  checkLocalServer,
  customServers = [],
  onAddCustomServer,
  onRemoveCustomServer,
  onToggleCustomServer,
  onCheckCustomServer,
  appInferenceUrl,
  onSetAppInferenceUrl,
}) => {
  const isTauriApp = isTauri();

  // ── Ollama pull state
  const [modelToPull, setModelToPull] = useState('');
  const [downloadState, setDownloadState] = useState<PullState>(pullModelManager.getInitialState());
  const [detectedServers, setDetectedServers] = useState<string[]>([]);
  const availableServers = ollamaServers || detectedServers;
  const [selectedServer, setSelectedServer] = useState<string>(availableServers[0] || '');

  // ── Tabs + Advanced panel
  const [activeTab, setActiveTab] = useState<TabId>(isTauriApp ? 'llamacpp' : 'transformers');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // ── Transformers.js (Gemma) state
  const [gemmaState, setGemmaState] = useState<GemmaModelState>(GemmaModelManager.getInstance().getState());
  const [transformersModels, setTransformersModels] = useState<LocalModelEntry[]>(GemmaModelManager.getInstance().listLocalModels());
  const [gemmaDevice, setGemmaDevice] = useState<GemmaDevice>('webgpu');
  const [gemmaDtype, setGemmaDtype] = useState<GemmaDtype>('q4');
  const [gemmaTokenBudget, setGemmaTokenBudget] = useState<GemmaImageTokenBudget>(70);
  const [gemmaEnableThinking, setGemmaEnableThinking] = useState(false);
  const [customOnnxModelId, setCustomOnnxModelId] = useState('');

  // ── llama.cpp (Native) state
  const [nativeState, setNativeState] = useState<NativeModelState>(NativeLlmManager.getInstance().getState());
  const [ggufFiles, setGgufFiles] = useState<GgufFileInfo[]>(() => NativeLlmManager.getInstance().getCachedGgufFiles());
  const [mmprojAssignments, setMmprojAssignments] = useState<Record<string, string>>(() => NativeLlmManager.getInstance().getMmprojAssignments());
  const [ggufUrl, setGgufUrl] = useState('');
  const [samplerParams, setSamplerParams] = useState<SamplerParams>({ ...DEFAULT_SAMPLER_PARAMS });
  const [showSamplerSettings, setShowSamplerSettings] = useState(false);
  const [contextParams, setContextParams] = useState<ContextParams>({ ...DEFAULT_CONTEXT_PARAMS });
  const [showContextSettings, setShowContextSettings] = useState(false);
  const [useGpu, setUseGpu] = useState<boolean>(() => {
    if (isWeb()) return true;
    return NativeLlmManager.getInstance().getPersistedUseGpu();
  });

  // ── Engine init state
  const [engineInitStatus, setEngineInitStatus] = useState<'idle' | 'loading' | 'ok' | 'error'>('idle');
  const [engineInitError, setEngineInitError] = useState<string | null>(null);
  const [engineLogs, setEngineLogs] = useState<LogEntry[]>([]);
  const engineLogsEndRef = useRef<HTMLDivElement>(null);

  // ── Preset download state
  const [downloadingPreset, setDownloadingPreset] = useState<ModelPreset | null>(null);
  const [presetDownloadStep, setPresetDownloadStep] = useState<'gguf' | 'mmproj' | null>(null);

  // ── System memory
  const [memInfo, setMemInfo] = useState<{ totalBytes: number; usedBytes: number; availableBytes: number } | null>(null);

  // ── First-visit setup dialog
  const [showLocalSetup, setShowLocalSetup] = useState(false);

  // ── Custom server state
  const [isAddingServer, setIsAddingServer] = useState(false);
  const [newServerAddress, setNewServerAddress] = useState('');
  const [addError, setAddError] = useState('');
  const [inferenceUrlInput, setInferenceUrlInput] = useState(appInferenceUrl || 'http://localhost:11434');

  useEffect(() => {
    if (!isOpen) return;
    setActiveTab(isTauriApp ? 'llamacpp' : 'transformers');
    if (!localStorage.getItem('observer_local_setup_seen')) setShowLocalSetup(true);
  }, [isOpen, isTauriApp]);

  // Subscribe to LLM logs while the Advanced panel is open on the llama.cpp tab
  useEffect(() => {
    if (!isOpen || !showAdvanced || activeTab !== 'llamacpp') return;

    const existing = Logger.getFilteredLogs({ source: ['NativeLlmManager', 'LlmEngine'] }).slice(-200);
    setEngineLogs(existing);

    const listener = (entry: LogEntry) => {
      if (entry.source === 'NativeLlmManager' || entry.source === 'LlmEngine') {
        setEngineLogs(prev => [...prev, entry].slice(-200));
      }
    };
    Logger.addListener(listener);
    return () => Logger.removeListener(listener);
  }, [isOpen, showAdvanced, activeTab]);

  useEffect(() => {
    engineLogsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [engineLogs]);

  useEffect(() => {
    if (!isOpen || ollamaServers) return;
    const checkLocalhost = async () => {
      try {
        const response = await platformFetch('http://localhost:3838/api/tags', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });
        if (response.ok) setDetectedServers(['http://localhost:3838']);
      } catch {
        setDetectedServers([]);
      }
    };
    checkLocalhost();
  }, [isOpen, ollamaServers]);

  useEffect(() => {
    if (!isOpen) return;
    return pullModelManager.subscribe((newState) => {
      setDownloadState(newState);
      if (newState.status === 'success' && onPullComplete) onPullComplete();
    });
  }, [isOpen, onPullComplete]);

  useEffect(() => {
    if (!isOpen) return;
    const manager = GemmaModelManager.getInstance();
    const unsubscribe = manager.onStateChange((state) => {
      setGemmaState(state);
      setTransformersModels(manager.listLocalModels());
    });
    const currentState = manager.getState();
    setGemmaState(currentState);
    setTransformersModels(manager.listLocalModels());
    const runtime = manager.getRuntimeSettings();
    setGemmaDevice(runtime.device);
    setGemmaTokenBudget(runtime.imageTokenBudget);
    setGemmaEnableThinking(runtime.enableThinking);
    // dtype comes from the currently loaded model (if any), otherwise leave at default
    if (currentState.loadSettings?.dtype) {
      setGemmaDtype(currentState.loadSettings.dtype);
    }
    return unsubscribe;
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !isTauriApp) return;
    const unsubscribe = NativeLlmManager.getInstance().onStateChange(setNativeState);
    setNativeState(NativeLlmManager.getInstance().getState());
    return unsubscribe;
  }, [isOpen, isTauriApp]);

  useEffect(() => {
    if (!isOpen || !isTauriApp) return;
    NativeLlmManager.getInstance().listGgufFiles().then(setGgufFiles);
    setMmprojAssignments(NativeLlmManager.getInstance().getMmprojAssignments());
  }, [isOpen, isTauriApp, nativeState.status]);

  useEffect(() => {
    if (!isOpen || !isTauriApp) return;
    if (nativeState.status === 'loading' || nativeState.status === 'unloaded') {
      setSamplerParams({ ...DEFAULT_SAMPLER_PARAMS });
    } else if (nativeState.status === 'loaded') {
      NativeLlmManager.getInstance().getDebugInfo().then(info => {
        if (info.engine.samplerParams) setSamplerParams(info.engine.samplerParams);
        if (info.engine.contextParams) setContextParams(info.engine.contextParams);
      }).catch(() => {});
    }
  }, [isOpen, isTauriApp, nativeState.status]);

  useEffect(() => {
    if (appInferenceUrl) setInferenceUrlInput(appInferenceUrl);
  }, [appInferenceUrl]);

  useEffect(() => {
    if (availableServers.length > 0 && !selectedServer) {
      setSelectedServer(availableServers[0]);
    }
  }, [availableServers, selectedServer]);

  useEffect(() => {
    if (!isOpen || !isTauriApp) return;
    const poll = async () => {
      try {
        const info = await invoke<{ totalBytes: number; usedBytes: number; availableBytes: number }>('get_memory_info');
        setMemInfo(info);
      } catch {}
    };
    poll();
    const id = setInterval(poll, 1000);
    return () => clearInterval(id);
  }, [isOpen, isTauriApp]);

  // Auto-start download when opened via LocalServerSetupDialog
  useEffect(() => {
    if (!isOpen || !autoDownloadPreset || isPresetInstalled(autoDownloadPreset)) return;
    handleDownloadPreset(autoDownloadPreset);
  }, [isOpen, autoDownloadPreset]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clear preset download state once native download finishes
  useEffect(() => {
    if (nativeState.status !== 'downloading' && downloadingPreset?.engine === 'llamacpp') {
      if (presetDownloadStep === 'gguf' && downloadingPreset.mmprojUrl) {
        // gguf done, kick off mmproj
        setPresetDownloadStep('mmproj');
        NativeLlmManager.getInstance().downloadModel(downloadingPreset.mmprojUrl)
          .then(() => {
            const ggufFilename = downloadingPreset.ggufUrl!.split('/').pop()!;
            const mmprojFilename = downloadingPreset.mmprojUrl!.split('/').pop()!;
            NativeLlmManager.getInstance().setMmprojAssignment(ggufFilename, mmprojFilename);
            setMmprojAssignments(NativeLlmManager.getInstance().getMmprojAssignments());
            onPullComplete?.();
          })
          .catch(() => {})
          .finally(() => {
            setDownloadingPreset(null);
            setPresetDownloadStep(null);
          });
      }
    }
  }, [nativeState.status]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Handlers ─────────────────────────────────────────────

  const handleStartPull = () => {
    if (modelToPull.trim() && selectedServer) {
      pullModelManager.pullModel(modelToPull.trim(), selectedServer);
    }
  };

  const handleCancelPull = () => pullModelManager.cancelPull();

  const handleDone = () => {
    if (downloadState.status === 'success' || downloadState.status === 'error') {
      pullModelManager.resetState();
    }
    onClose();
  };

  const handleDownloadGguf = async () => {
    if (!ggufUrl.trim()) return;
    try {
      await NativeLlmManager.getInstance().downloadModel(ggufUrl.trim());
      setGgufUrl('');
    } catch { /* surfaced via nativeState.error */ }
  };

  const handleMmprojAssignment = (modelFilename: string, mmprojFilename: string | null) => {
    NativeLlmManager.getInstance().setMmprojAssignment(modelFilename, mmprojFilename);
    setMmprojAssignments(NativeLlmManager.getInstance().getMmprojAssignments());
  };

  const handleSamplerParamChange = async (key: keyof SamplerParams, value: number) => {
    setSamplerParams(prev => ({ ...prev, [key]: value }));
    if (nativeState.status === 'loaded') {
      try { await NativeLlmManager.getInstance().setSamplerParams({ [key]: value }); } catch {}
    }
  };

  const handleResetSamplerParams = async () => {
    setSamplerParams({ ...DEFAULT_SAMPLER_PARAMS });
    if (nativeState.status === 'loaded') {
      try { await NativeLlmManager.getInstance().setSamplerParams(DEFAULT_SAMPLER_PARAMS); } catch {}
    }
  };

  const handleContextParamChange = async (key: keyof ContextParams, value: number) => {
    setContextParams(prev => ({ ...prev, [key]: value }));
    try { await NativeLlmManager.getInstance().setContextParams({ [key]: value }); } catch {}
  };

  const handleResetContextParams = async () => {
    setContextParams({ ...DEFAULT_CONTEXT_PARAMS });
    try { await NativeLlmManager.getInstance().setContextParams(DEFAULT_CONTEXT_PARAMS); } catch {}
  };

  const handleToggleUnifiedGpu = async (enabled: boolean) => {
    setUseGpu(enabled);
    setGemmaDevice(enabled ? 'webgpu' : 'wasm');
    try { await NativeLlmManager.getInstance().setUseGpu(enabled); } catch {}
  };

  const handleDownloadPreset = async (preset: ModelPreset) => {
    if (preset.engine === 'transformers') {
      GemmaModelManager.getInstance().loadModelWithSettings(
        preset.hfModelId! as GemmaModelId,
        useGpu ? 'webgpu' : 'wasm',
        gemmaDtype,
        gemmaTokenBudget,
        gemmaEnableThinking,
      );
      return;
    }
    // llamacpp — fire gguf download; mmproj is chained in the effect above
    setDownloadingPreset(preset);
    setPresetDownloadStep('gguf');
    try {
      await NativeLlmManager.getInstance().downloadModel(preset.ggufUrl!);
      if (!preset.mmprojUrl) {
        onPullComplete?.();
        setDownloadingPreset(null);
        setPresetDownloadStep(null);
      }
      // if mmprojUrl exists, the effect handles chaining
    } catch {
      setDownloadingPreset(null);
      setPresetDownloadStep(null);
    }
  };

  const handleAddServer = () => {
    setAddError('');
    if (!newServerAddress.trim()) {
      setAddError('Please enter a server address');
      return;
    }
    if (!newServerAddress.match(/^https?:\/\//)) {
      setAddError('URL must start with http:// or https://');
      return;
    }
    try {
      new URL(newServerAddress);
      onAddCustomServer?.(newServerAddress);
      setNewServerAddress('');
      setIsAddingServer(false);
    } catch {
      setAddError('Invalid URL format');
    }
  };

  const handleLoadCustomOnnx = () => {
    if (customOnnxModelId.trim()) {
      GemmaModelManager.getInstance().loadModelWithSettings(
        customOnnxModelId.trim() as GemmaModelId,
        gemmaDevice,
        gemmaDtype,
        gemmaTokenBudget,
        gemmaEnableThinking,
      );
    }
  };

  // ── Derived ──────────────────────────────────────────────

  const { status, progress, statusText, errorText, completedBytes, totalBytes } = downloadState;
  const isPulling = status === 'pulling';
  const isFinished = status === 'success' || status === 'error';
  const isNativeDownloading = nativeState.status === 'downloading';
  const isAnyNativeBusy = nativeState.status === 'loading' || nativeState.status === 'unloading' || isNativeDownloading;

  const ollamaServersFromCustom = customServers
    .filter(s => s.enabled && s.status === 'online')
    .map(s => s.address);
  const allOllamaServers = [...new Set([...availableServers, ...ollamaServersFromCustom])];

  const ggufModels = ggufFiles.filter(f => !f.filename.toLowerCase().includes('mmproj'));
  const ggufProjectors = ggufFiles.filter(f => f.filename.toLowerCase().includes('mmproj'));
  const installedCount = ggufFiles.length + transformersModels.length;

  const isPresetInstalled = (preset: ModelPreset) => {
    if (preset.engine === 'llamacpp') {
      const filename = preset.ggufUrl?.split('/').pop();
      return ggufFiles.some(f => f.filename === filename);
    }
    return transformersModels.some(m => m.id === preset.hfModelId);
  };

  const isPresetDownloading = (preset: ModelPreset) => {
    if (preset.engine === 'llamacpp') return downloadingPreset?.name === preset.name;
    return gemmaState.modelId === preset.hfModelId && gemmaState.status === 'loading';
  };

  return (
    <>
    {showLocalSetup && (
      <LocalServerSetupDialog
        onDismiss={() => {
          localStorage.setItem('observer_local_setup_seen', '1');
          setShowLocalSetup(false);
        }}
      />
    )}
    <Modal open={isOpen} onClose={handleDone} className="w-full max-w-3xl">
      <div className="p-6 sm:p-8 overflow-y-auto" style={{ maxHeight: 'calc(88vh - env(safe-area-inset-top) - env(safe-area-inset-bottom))' }}>

        {/* Header */}
        <div className="flex justify-between items-start mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Model Hub</h2>
          </div>
          <button
            onClick={handleDone}
            className="text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full p-2 transition-colors"
            aria-label="Close"
          >
            <X size={22} />
          </button>
        </div>

        {/* ── Ob-Server card ───────────────────────────────── */}
        <section className="mb-5">
          <div className={`p-4 rounded-xl border transition-all ${
            isUsingObServer
              ? 'bg-gradient-to-r from-indigo-50 to-purple-50 border-indigo-200'
              : 'bg-gray-50 border-gray-200'
          }`}>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                  isUsingObServer ? 'bg-indigo-200' : 'bg-gray-200'
                }`}>
                  <Cloud size={20} className={isUsingObServer ? 'text-indigo-600' : 'text-gray-500'} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-semibold text-gray-900">Ob-Server Cloud</span>
                    {isUsingObServer && quotaInfo?.tier && (
                      <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                        quotaInfo.tier === 'max' ? 'bg-green-100 text-green-700' :
                        quotaInfo.tier === 'pro' ? 'bg-purple-100 text-purple-700' :
                        quotaInfo.tier === 'plus' ? 'bg-blue-100 text-blue-700' :
                        'bg-gray-100 text-gray-700'
                      }`}>
                        {quotaInfo.tier.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5 truncate">
                    {!isUsingObServer
                      ? 'Disabled — fully offline mode'
                      : !isAuthenticated
                        ? 'Login required'
                        : quotaInfo && typeof quotaInfo.remaining === 'number' && typeof quotaInfo.limit === 'number'
                          ? `${quotaInfo.remaining} of ${quotaInfo.limit} credits remaining`
                          : 'Cloud inference enabled'
                    }
                  </p>
                </div>
              </div>
              <button
                onClick={handleToggleObServer}
                className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors focus:outline-none flex-shrink-0 ${
                  isUsingObServer ? 'bg-indigo-600' : 'bg-gray-300'
                }`}
                aria-label={isUsingObServer ? 'Disable Ob-Server' : 'Enable Ob-Server'}
              >
                <span className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
                  isUsingObServer ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>

            {isUsingObServer && isAuthenticated && quotaInfo && typeof quotaInfo.remaining === 'number' && typeof quotaInfo.limit === 'number' && (
              <div className="mt-3 pt-3 border-t border-indigo-100">
                <div className="w-full bg-white/70 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      quotaInfo.remaining <= 10 ? 'bg-red-500' :
                      quotaInfo.remaining <= quotaInfo.limit * 0.3 ? 'bg-orange-500' :
                      'bg-indigo-600'
                    }`}
                    style={{ width: `${Math.max(0, Math.min(100, ((quotaInfo.limit - quotaInfo.remaining) / quotaInfo.limit) * 100))}%` }}
                  />
                </div>
                {renderQuotaStatus && (
                  <div className="text-xs text-gray-600 mt-1.5">{renderQuotaStatus()}</div>
                )}
              </div>
            )}

            {showLoginMessage && (
              <div className="mt-3 p-2.5 bg-red-50 border border-red-200 rounded-lg text-red-700 text-xs font-medium">
                Login required to use Ob-Server Cloud
              </div>
            )}
          </div>
        </section>



        {/* ── Installed Models ─────────────────────────────── */}
        <section className="mb-5">
          <div className="flex items-center gap-2 mb-3">
            <Package size={16} className="text-gray-500" />
            <h3 className="text-sm font-semibold text-gray-700">Installed Models</h3>
            {installedCount > 0 && (
              <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
                {installedCount}
              </span>
            )}
          </div>

          {installedCount === 0 ? (
            <div className="border border-dashed border-gray-200 rounded-xl p-6 text-center">
              <p className="text-sm text-gray-500">No local models installed yet.</p>
              <p className="text-xs text-gray-400 mt-1">Download one from the catalog below.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {/* llama.cpp model files */}
              {ggufModels.map((file) => {
                const modelId = file.filename.replace(/\.gguf$/i, '');
                const isThisModel = nativeState.modelId === modelId;
                const isLoaded = isThisModel && nativeState.status === 'loaded';
                const isLoading = isThisModel && nativeState.status === 'loading';
                const isUnloading = isThisModel && nativeState.status === 'unloading';
                const assignedMmproj = mmprojAssignments[file.filename] ?? null;
                const isMultimodal = isThisModel
                  ? NativeLlmManager.getInstance().isMultimodal()
                  : !!assignedMmproj;

                return (
                  <div
                    key={file.filename}
                    className={`border rounded-xl p-3 transition-all ${
                      isLoaded ? 'border-gray-400 bg-gray-50' : 'border-gray-200 bg-white hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${
                          isLoaded ? 'bg-gray-200' : 'bg-gray-100'
                        }`}>
                          <Cpu size={18} className={isLoaded ? 'text-gray-800' : 'text-gray-500'} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="font-medium text-gray-900 truncate">{modelId}</span>
                            <span className="text-[10px] font-semibold text-gray-700 bg-gray-200 px-1.5 py-0.5 rounded">llama.cpp</span>
                            {isMultimodal && (
                              <span className="text-[10px] font-semibold text-purple-700 bg-purple-100 px-1.5 py-0.5 rounded flex items-center gap-0.5"><Eye size={9} /></span>
                            )}
                          </div>
                          <div className="flex items-center gap-2 mt-1">
                            <p className="text-xs text-gray-500">{formatBytes(file.sizeBytes)}</p>
                            {ggufProjectors.length > 0 && (
                              <select
                                value={assignedMmproj ?? ''}
                                onChange={e => handleMmprojAssignment(file.filename, e.target.value || null)}
                                disabled={isLoaded || isLoading}
                                className="text-xs border border-gray-200 rounded px-1.5 py-0.5 bg-white text-gray-600 focus:ring-1 focus:ring-purple-400 disabled:opacity-50 max-w-[160px] truncate"
                                title="Assign a vision projector"
                              >
                                <option value="">No projector</option>
                                {ggufProjectors.map(p => (
                                  <option key={p.filename} value={p.filename}>{p.filename.replace(/\.gguf$/i, '')}</option>
                                ))}
                              </select>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        {isUnloading ? (
                          <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-200 text-gray-600 rounded-lg font-medium">
                            <Cpu size={12} className="animate-pulse" /> Unloading
                          </span>
                        ) : isLoaded ? (
                          <button
                            onClick={() => NativeLlmManager.getInstance().unloadModel()}
                            className="group flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg font-medium transition-colors bg-gray-200 text-gray-800 hover:bg-red-100 hover:text-red-700"
                          >
                            <span className="group-hover:hidden flex items-center gap-1.5"><CheckCircle size={12} /> Ready</span>
                            <span className="hidden group-hover:flex items-center gap-1.5"><X size={12} /> Unload</span>
                          </button>
                        ) : isLoading ? (
                          <button
                            onClick={() => NativeLlmManager.getInstance().unloadModel()}
                            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-200 text-gray-600 rounded-lg font-medium"
                          >
                            <Cpu size={12} className="animate-pulse" /> Loading
                          </button>
                        ) : (
                          <>
                            <button
                              disabled={isAnyNativeBusy}
                              onClick={() => NativeLlmManager.getInstance().loadModel(file.filename)}
                              className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 text-white rounded-lg hover:bg-gray-900 disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-sm"
                            >
                              <Cpu size={12} /> Load
                            </button>
                            <button
                              disabled={isAnyNativeBusy}
                              onClick={async () => {
                                const mmproj = mmprojAssignments[file.filename] ?? null;
                                setGgufFiles(prev => prev.filter(f => f.filename !== file.filename && f.filename !== mmproj));
                                await NativeLlmManager.getInstance().deleteModel(file.filename);
                                if (mmproj) {
                                  const stillUsed = ggufModels.some(
                                    m => m.filename !== file.filename && mmprojAssignments[m.filename] === mmproj
                                  );
                                  if (!stillUsed) await NativeLlmManager.getInstance().deleteModel(mmproj);
                                }
                              }}
                              className="p-1.5 text-gray-400 hover:text-red-600 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                              title="Delete"
                            >
                              <Trash2 size={14} />
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}


              {/* Transformers.js installed models */}
              {transformersModels.map((model) => {
                const isThisModel = gemmaState.modelId === model.id;
                const tStatus = isThisModel ? gemmaState.status : model.status;
                const isLoaded = tStatus === 'loaded';
                const isLoading = tStatus === 'loading';
                const isError = tStatus === 'error';
                const loadSettings = isThisModel ? gemmaState.loadSettings : null;

                return (
                  <div key={model.id} className={`border rounded-xl p-3 transition-all ${
                    isLoaded ? 'border-gray-400 bg-gray-50' : 'border-gray-200 bg-white'
                  }`}>
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${
                          isLoaded ? 'bg-gray-200' : 'bg-gray-100'
                        }`}>
                          <Sparkles size={18} className={isLoaded ? 'text-gray-800' : 'text-gray-500'} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="font-medium text-gray-900 truncate">{model.name}</span>
                            <span className="text-[10px] font-semibold text-yellow-800 bg-yellow-300 px-1.5 py-0.5 rounded">Transformers.js</span>
                          </div>
                          <p className="text-xs text-gray-500 mt-0.5 flex items-center gap-1.5 flex-wrap">
                            {loadSettings ? (
                              <>{loadSettings.device} · {loadSettings.dtype} · {loadSettings.imageTokenBudget} tokens · </>
                            ) : null}
                            {gemmaEnableThinking ? (
                              <span className="text-[10px] font-semibold text-purple-700 bg-purple-100 px-1.5 py-0.5 rounded">Thinking on</span>
                            ) : (
                              <span className="text-[10px] font-medium text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">Thinking off</span>
                            )}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        {isLoaded ? (
                          <button
                            onClick={() => GemmaModelManager.getInstance().unloadModel()}
                            className="group flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg font-medium transition-colors bg-gray-200 text-gray-800 hover:bg-red-100 hover:text-red-700"
                          >
                            <span className="group-hover:hidden flex items-center gap-1.5"><CheckCircle size={12} /> Ready</span>
                            <span className="hidden group-hover:flex items-center gap-1.5"><X size={12} /> Unload</span>
                          </button>
                        ) : isLoading ? (
                          <button
                            onClick={() => GemmaModelManager.getInstance().unloadModel()}
                            className="group flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-200 text-gray-600 hover:bg-red-100 hover:text-red-700 rounded-lg font-medium transition-colors"
                          >
                            <span className="group-hover:hidden flex items-center gap-1.5"><Cpu size={12} className="animate-pulse" /> Loading</span>
                            <span className="hidden group-hover:flex items-center gap-1.5"><StopCircle size={12} /> Cancel</span>
                          </button>
                        ) : isError ? (
                          <span className="flex items-center gap-1 text-xs font-semibold text-red-700 bg-red-100 px-2 py-1 rounded-full">
                            <AlertTriangle size={12} /> Error
                          </span>
                        ) : (
                          <>
                            <button
                              onClick={() => {
                                const installedDtype = GemmaModelManager.getInstance().getSettingsForModel(model.id as GemmaModelId).dtype;
                                GemmaModelManager.getInstance().loadModelWithSettings(model.id as GemmaModelId, gemmaDevice, installedDtype, gemmaTokenBudget, gemmaEnableThinking);
                              }}
                              className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 text-white rounded-lg hover:bg-gray-900 font-medium shadow-sm"
                            >
                              <Sparkles size={12} /> Load
                            </button>
                            <button
                              onClick={() => { setTransformersModels(prev => prev.filter(m => m.id !== model.id)); GemmaModelManager.getInstance().deleteModel(model.id as GemmaModelId); }}
                              className="p-1.5 text-gray-400 hover:text-red-600 rounded transition-colors"
                              title="Delete model"
                            >
                              <Trash2 size={14} />
                            </button>
                          </>
                        )}
                      </div>
                    </div>

                    {isThisModel && isLoading && gemmaState.progress.length > 0 && (
                      <div className="space-y-1.5 mt-3">
                        {gemmaState.progress.map((item) => (
                          <div key={item.file}>
                            <div className="flex justify-between items-center text-[11px] mb-1">
                              <span className="text-gray-600 flex items-center gap-1.5 truncate max-w-[55%]">
                                {item.status === 'done'
                                  ? <CheckCircle className="h-3 w-3 text-green-500 flex-shrink-0" />
                                  : <FileDown className="h-3 w-3 text-purple-400 flex-shrink-0" />
                                }
                                {item.file}
                              </span>
                              <span className="font-medium text-gray-500 flex-shrink-0">
                                {item.status === 'done'
                                  ? 'Done'
                                  : item.total > 0
                                    ? `${formatBytes(item.loaded)} / ${formatBytes(item.total)}`
                                    : `${Math.round(item.progress)}%`
                                }
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5">
                              <div
                                className={`h-1.5 rounded-full transition-all duration-300 ${item.status === 'done' ? 'bg-green-500' : 'bg-purple-600'}`}
                                style={{ width: `${item.progress}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {isThisModel && isError && (
                      <p className="mt-2 text-xs text-red-600 bg-red-50 px-3 py-2 rounded-lg">{gemmaState.error}</p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* ── GPU + RAM ────────────────────────────────────── */}
        <section className="mb-5">
          <div className="p-3 border border-gray-200 rounded-xl bg-white space-y-2.5">
            {/* GPU row */}
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-sm font-semibold text-gray-800 whitespace-nowrap">GPU</span>
                <span className="text-xs text-gray-500 truncate">
                  {useGpu ? 'WebGPU / Metal' : 'CPU only'}
                </span>
              </div>
              <button
                onClick={() => handleToggleUnifiedGpu(!useGpu)}
                disabled={nativeState.status === 'loading' || nativeState.status === 'loaded'}
                className={`relative inline-flex h-6 w-10 items-center rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0 ${
                  useGpu ? 'bg-green-600' : 'bg-gray-300'
                }`}
                title={nativeState.status === 'loaded' ? 'Unload current model to change' : undefined}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
                  useGpu ? 'translate-x-5' : 'translate-x-1'
                }`} />
              </button>
            </div>

            {/* RAM row */}
            {isTauriApp && memInfo && (() => {
              const pct = memInfo.usedBytes / memInfo.totalBytes;
              return (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-semibold text-gray-800">RAM</span>
                    <span className="text-xs font-mono text-gray-500">
                      {formatBytes(memInfo.usedBytes)} / {formatBytes(memInfo.totalBytes)}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div
                      className={`h-1.5 rounded-full transition-all duration-500 ${
                        pct > 0.9 ? 'bg-red-500' : pct > 0.7 ? 'bg-orange-400' : 'bg-green-500'
                      }`}
                      style={{ width: `${Math.min(100, pct * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })()}
          </div>
        </section>

        {/* ── Download Models (catalog) ─────────────────────── */}
        <section className="mb-5">
          <div className="flex items-center gap-2 mb-3">
            <Download size={16} className="text-gray-500" />
            <h3 className="text-sm font-semibold text-gray-700">Download Models</h3>
          </div>

          <div className="space-y-2">
            {MODEL_PRESETS.map(preset => {
              const isLlamaCpp = preset.engine === 'llamacpp';
              const unavailable = isLlamaCpp && !isTauriApp;
              const installed = isPresetInstalled(preset);
              const thisDownloading = isPresetDownloading(preset);
              const downloadBlocked = isLlamaCpp
                ? (isAnyNativeBusy || (downloadingPreset !== null && downloadingPreset.name !== preset.name))
                : gemmaState.status === 'loading' && gemmaState.modelId !== preset.hfModelId;

              return (
                <div
                  key={preset.name}
                  className={`border rounded-xl p-3 transition-all ${
                    unavailable ? 'opacity-50 bg-gray-50' :
                    installed ? 'border-gray-400 bg-gray-50' :
                    'border-gray-200 bg-white hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="font-medium text-gray-900">{preset.name}</span>
                        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                          isLlamaCpp ? 'bg-gray-200 text-gray-700' : 'bg-yellow-300 text-yellow-800'
                        }`}>
                          {isLlamaCpp ? 'llama.cpp' : 'Transformers.js'}
                        </span>
                        {(preset.mmprojUrl || preset.hfModelId) && (
                          <span className="text-[10px] font-semibold text-purple-700 bg-purple-100 px-1.5 py-0.5 rounded flex items-center gap-0.5"><Eye size={9} /></span>
                        )}
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5">{preset.sizeLabel}</p>
                    </div>

                    <div className="flex-shrink-0">
                      {unavailable ? (
                        <span className="text-xs text-gray-400" title="Install the desktop app to use llama.cpp">App only</span>
                      ) : installed ? (
                        <span className="flex items-center gap-1 text-xs text-green-700 font-medium">
                          <CheckCircle size={12} /> Installed
                        </span>
                      ) : thisDownloading ? (
                        <span className="flex items-center gap-1.5 text-xs text-blue-600 font-medium">
                          <Download size={12} className="animate-bounce" />
                          {isLlamaCpp
                            ? (presetDownloadStep === 'gguf' ? 'Model…' : 'Vision…')
                            : 'Downloading…'
                          }
                          <button
                            onClick={isLlamaCpp
                              ? () => { NativeLlmManager.getInstance().cancelDownload(); setDownloadingPreset(null); setPresetDownloadStep(null); }
                              : () => GemmaModelManager.getInstance().unloadModel()
                            }
                            className="ml-1 p-0.5 text-red-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                            title="Cancel"
                          >
                            <StopCircle size={12} />
                          </button>
                        </span>
                      ) : (
                        <button
                          onClick={() => handleDownloadPreset(preset)}
                          disabled={downloadBlocked}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-sm"
                        >
                          <Download size={12} />
                          Download
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Inline progress for llamacpp preset download */}
                  {thisDownloading && isLlamaCpp && (
                    <div className="mt-3 space-y-2">
                      {/* Model (gguf) bar */}
                      <div>
                        <div className="flex justify-between text-[11px] mb-1">
                          <span className="text-gray-500 flex items-center gap-1.5">
                            {presetDownloadStep === 'mmproj'
                              ? <CheckCircle className="h-3 w-3 text-green-500" />
                              : <FileDown className="h-3 w-3 text-blue-400" />
                            }
                            Model (.gguf)
                          </span>
                          <span className="font-mono text-gray-500">
                            {presetDownloadStep === 'mmproj'
                              ? 'Done'
                              : nativeState.totalBytes > 0
                                ? `${formatBytes(nativeState.downloadedBytes)} / ${formatBytes(nativeState.totalBytes)}`
                                : `${nativeState.downloadProgress}%`
                            }
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div
                            className={`h-1.5 rounded-full transition-all duration-300 ${presetDownloadStep === 'mmproj' ? 'bg-green-500' : 'bg-blue-600'}`}
                            style={{ width: presetDownloadStep === 'mmproj' ? '100%' : `${nativeState.downloadProgress}%` }}
                          />
                        </div>
                      </div>
                      {/* Vision projector (mmproj) bar — only shown if preset has one */}
                      {preset.mmprojUrl && (
                        <div>
                          <div className="flex justify-between text-[11px] mb-1">
                            <span className="text-gray-500 flex items-center gap-1.5">
                              <FileDown className={`h-3 w-3 ${presetDownloadStep === 'mmproj' ? 'text-purple-400' : 'text-gray-300'}`} />
                              Vision projector (.gguf)
                            </span>
                            <span className="font-mono text-gray-500">
                              {presetDownloadStep === 'mmproj'
                                ? nativeState.totalBytes > 0
                                  ? `${formatBytes(nativeState.downloadedBytes)} / ${formatBytes(nativeState.totalBytes)}`
                                  : `${nativeState.downloadProgress}%`
                                : 'Pending'
                              }
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-1.5">
                            <div
                              className="h-1.5 rounded-full bg-purple-500 transition-all duration-300"
                              style={{ width: presetDownloadStep === 'mmproj' ? `${nativeState.downloadProgress}%` : '0%' }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Inline progress for transformers preset */}
                  {thisDownloading && !isLlamaCpp && gemmaState.progress.length > 0 && (
                    <div className="space-y-1.5 mt-3">
                      {gemmaState.progress.map((item) => (
                        <div key={item.file}>
                          <div className="flex justify-between items-center text-[11px] mb-1">
                            <span className="text-gray-600 flex items-center gap-1.5 truncate max-w-[55%]">
                              {item.status === 'done'
                                ? <CheckCircle className="h-3 w-3 text-green-500 flex-shrink-0" />
                                : <FileDown className="h-3 w-3 text-purple-400 flex-shrink-0" />
                              }
                              {item.file}
                            </span>
                            <span className="font-medium text-gray-500 flex-shrink-0">
                              {item.status === 'done'
                                ? 'Done'
                                : item.total > 0
                                  ? `${formatBytes(item.loaded)} / ${formatBytes(item.total)}`
                                  : `${Math.round(item.progress)}%`
                              }
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-1.5">
                            <div
                              className={`h-1.5 rounded-full transition-all duration-300 ${item.status === 'done' ? 'bg-green-500' : 'bg-purple-600'}`}
                              style={{ width: `${item.progress}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {isNativeDownloading && !downloadingPreset && (
            <div className="mt-3 border border-blue-200 bg-blue-50 rounded-xl p-3">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-semibold text-gray-700 flex items-center gap-1.5 truncate">
                  <FileDown size={13} className="text-blue-500" /> {nativeState.modelId}
                </span>
                <button
                  onClick={() => NativeLlmManager.getInstance().cancelDownload()}
                  className="flex items-center gap-1 px-2 py-1 text-xs bg-red-100 text-red-600 hover:bg-red-200 rounded-lg font-medium"
                >
                  <StopCircle size={11} /> Cancel
                </button>
              </div>
              <div className="w-full bg-blue-200 rounded-full h-1.5">
                <div className="h-1.5 rounded-full bg-blue-600 transition-all" style={{ width: `${nativeState.downloadProgress}%` }} />
              </div>
            </div>
          )}

          {nativeState.status === 'error' && (
            <div className="mt-3 border border-red-200 bg-red-50 rounded-lg p-3">
              <p className="text-xs text-red-700">{nativeState.error}</p>
            </div>
          )}
        </section>

        {/* ── Advanced collapsible ─────────────────────────── */}
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          <button
            onClick={() => setShowAdvanced(v => !v)}
            className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors"
          >
            <span className="text-sm font-medium text-gray-600">Advanced</span>
            <ChevronDown size={16} className={`text-gray-400 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
          </button>

          {showAdvanced && (
            <div className="border-t border-gray-200">
              {/* Tab bar */}
              <div className="flex gap-1 m-4 mb-0 p-1 bg-gray-100 rounded-xl overflow-x-auto">
                <TabButton
                  active={activeTab === 'llamacpp'}
                  onClick={() => setActiveTab('llamacpp')}
                  color="green"
                  icon={<Cpu size={14} />}
                  label="llama.cpp"
                  dimmed={!isTauriApp}
                />
                <TabButton
                  active={activeTab === 'transformers'}
                  onClick={() => setActiveTab('transformers')}
                  color="purple"
                  icon={<Sparkles size={14} />}
                  label="Transformers.js"
                />
                <TabButton
                  active={activeTab === 'servers'}
                  onClick={() => setActiveTab('servers')}
                  color="blue"
                  icon={<Server size={14} />}
                  label="Servers"
                />
              </div>

              <div className="p-4">

                {/* ── llama.cpp tab ─────────────────────────── */}
                {activeTab === 'llamacpp' && (
                  <div className="space-y-4">
                    {!isTauriApp ? (
                      <div className="text-center py-8 border border-gray-200 rounded-xl bg-gray-50">
                        <Cpu size={28} className="text-gray-300 mx-auto mb-2" />
                        <h3 className="text-base font-semibold text-gray-700 mb-1">Native app required</h3>
                        <p className="text-sm text-gray-500 max-w-sm mx-auto">
                          llama.cpp runs natively with GPU acceleration. Install the desktop or mobile app to use this engine.
                        </p>
                      </div>
                    ) : (
                      <>
                        {/* Engine initialization */}
                        <div className="border border-gray-200 rounded-xl overflow-hidden">
                          <div className="flex items-center justify-between px-4 py-3 bg-gray-50">
                            <span className="flex items-center gap-2 text-sm font-medium text-gray-700">
                              <Play size={14} className="text-green-500" /> Engine
                            </span>
                            <div className="flex items-center gap-2">
                              {engineInitStatus === 'ok' && (
                                <span className="text-xs text-green-600 font-medium">Initialized</span>
                              )}
                              {engineInitStatus === 'error' && (
                                <span className="text-xs text-red-600 font-medium">Error</span>
                              )}
                              <button
                                onClick={async () => {
                                  setEngineInitStatus('loading');
                                  setEngineInitError(null);
                                  try {
                                    await NativeLlmManager.getInstance().initEngine();
                                    setEngineInitStatus('ok');
                                  } catch (e) {
                                    setEngineInitStatus('error');
                                    setEngineInitError(e instanceof Error ? e.message : String(e));
                                  }
                                }}
                                disabled={engineInitStatus === 'loading'}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 text-white rounded-lg hover:bg-gray-900 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                              >
                                <RefreshCw size={12} className={engineInitStatus === 'loading' ? 'animate-spin' : ''} />
                                {engineInitStatus === 'loading' ? 'Initializing...' : 'Init Engine'}
                              </button>
                            </div>
                          </div>
                          {engineInitError && (
                            <div className="px-4 py-2 text-xs text-red-700 bg-red-50 border-t border-red-200">
                              {engineInitError}
                            </div>
                          )}
                          {/* Live logs */}
                          <div className="border-t border-gray-200">
                            <div className="flex items-center justify-between px-4 py-2">
                              <span className="flex items-center gap-1.5 text-xs font-medium text-gray-600">
                                <Terminal size={12} /> Logs
                              </span>
                              <button
                                onClick={() => setEngineLogs([])}
                                className="text-xs text-gray-400 hover:text-red-500 transition-colors"
                              >
                                Clear
                              </button>
                            </div>
                            <div className="h-36 overflow-y-auto bg-gray-900 mx-3 mb-3 rounded p-2 font-mono text-[10px] leading-tight">
                              {engineLogs.length === 0 ? (
                                <span className="text-gray-500">No logs yet — init the engine or load a model</span>
                              ) : (
                                engineLogs.map(log => (
                                  <div
                                    key={log.id}
                                    className={
                                      log.level === LogLevel.ERROR ? 'text-red-400' :
                                      log.level === LogLevel.WARNING ? 'text-yellow-400' :
                                      'text-gray-300'
                                    }
                                  >
                                    <span className="text-gray-500">{log.timestamp.toLocaleTimeString()} </span>
                                    {log.message}
                                  </div>
                                ))
                              )}
                              <div ref={engineLogsEndRef} />
                            </div>
                          </div>
                        </div>

                        {/* Custom GGUF download */}
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">Download GGUF from URL</label>
                          <p className="text-xs text-gray-400 mb-2">Works for model files and vision projectors alike.</p>
                          <div className="flex gap-2">
                            <input
                              type="text"
                              value={ggufUrl}
                              onChange={(e) => setGgufUrl(e.target.value)}
                              placeholder="https://huggingface.co/.../resolve/main/model.gguf"
                              disabled={isNativeDownloading}
                              className="flex-grow p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 disabled:opacity-50 text-sm"
                            />
                            <button
                              onClick={handleDownloadGguf}
                              disabled={!ggufUrl.trim() || isNativeDownloading}
                              className="flex items-center gap-1.5 px-4 py-2.5 bg-gray-700 text-white rounded-lg hover:bg-gray-900 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
                            >
                              <Download size={14} /> Download
                            </button>
                          </div>
                        </div>

                        {/* Advanced GGUF download progress */}
                        {isNativeDownloading && !downloadingPreset && (
                          <div className="border border-blue-200 bg-blue-50 rounded-xl p-3">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-xs font-semibold text-gray-700 flex items-center gap-1.5 truncate">
                                <FileDown size={13} className="text-blue-500" /> {nativeState.modelId}
                              </span>
                              <button
                                onClick={() => NativeLlmManager.getInstance().cancelDownload()}
                                className="flex items-center gap-1 px-2 py-1 text-xs bg-red-100 text-red-600 hover:bg-red-200 rounded-lg font-medium"
                              >
                                <StopCircle size={11} /> Cancel
                              </button>
                            </div>
                            <div className="flex justify-between text-[11px] mb-1">
                              <span className="text-gray-500">
                                {nativeState.totalBytes > 0
                                  ? `${formatBytes(nativeState.downloadedBytes)} / ${formatBytes(nativeState.totalBytes)}`
                                  : `${nativeState.downloadProgress}%`
                                }
                              </span>
                            </div>
                            <div className="w-full bg-blue-200 rounded-full h-2">
                              <div
                                className="h-2 rounded-full bg-blue-600 transition-all duration-300"
                                style={{ width: `${nativeState.downloadProgress}%` }}
                              />
                            </div>
                          </div>
                        )}

                        {nativeState.status === 'error' && (
                          <div className="border border-red-200 bg-red-50 rounded-lg p-3">
                            <p className="text-xs text-red-700">{nativeState.error}</p>
                          </div>
                        )}

                        {/* Thinking mode toggle */}
                        <div className="flex items-center justify-between gap-3 p-3 border border-gray-200 rounded-xl bg-white">
                          <div className="min-w-0">
                            <p className="text-sm font-medium text-gray-800">Thinking Mode</p>
                            <p className="text-xs text-gray-500 mt-0.5">
                              Model reasons step-by-step before answering (requires a reasoning model).
                            </p>
                          </div>
                          <button
                            onClick={() => NativeLlmManager.getInstance().setEnableThinking(!nativeState.enableThinking)}
                            className={`relative inline-flex h-6 w-10 flex-shrink-0 items-center rounded-full transition-colors ${
                              nativeState.enableThinking ? 'bg-purple-600' : 'bg-gray-300'
                            }`}
                            aria-label={nativeState.enableThinking ? 'Disable thinking' : 'Enable thinking'}
                          >
                            <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
                              nativeState.enableThinking ? 'translate-x-5' : 'translate-x-1'
                            }`} />
                          </button>
                        </div>

                        {/* Sampler settings */}
                        <div className="border border-gray-200 rounded-xl overflow-hidden">
                          <button
                            onClick={() => setShowSamplerSettings(!showSamplerSettings)}
                            className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors"
                          >
                            <span className="flex items-center gap-2 text-sm font-medium text-gray-700">
                              <Settings2 size={14} /> Generation Settings
                            </span>
                            <span className={`text-gray-400 transition-transform text-xs ${showSamplerSettings ? 'rotate-180' : ''}`}>▼</span>
                          </button>
                          {showSamplerSettings && (
                            <div className="p-4 space-y-4 border-t border-gray-200">
                              {nativeState.status !== 'loaded' && (
                                <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                                  Load a model to configure generation settings.
                                </p>
                              )}
                              <SamplerSlider
                                label="Temperature"
                                value={samplerParams.temperature}
                                min={0} max={2} step={0.05}
                                disabled={nativeState.status !== 'loaded'}
                                hint="Higher = more creative, lower = more focused"
                                format={(v) => v.toFixed(2)}
                                onChange={(v) => handleSamplerParamChange('temperature', v)}
                              />
                              <SamplerSlider
                                label="Top P (nucleus sampling)"
                                value={samplerParams.topP}
                                min={0} max={1} step={0.05}
                                disabled={nativeState.status !== 'loaded'}
                                format={(v) => v.toFixed(2)}
                                onChange={(v) => handleSamplerParamChange('topP', v)}
                              />
                              <SamplerSlider
                                label="Top K"
                                value={samplerParams.topK}
                                min={1} max={100} step={1}
                                disabled={nativeState.status !== 'loaded'}
                                format={(v) => v.toString()}
                                onChange={(v) => handleSamplerParamChange('topK', v)}
                              />
                              <SamplerSlider
                                label="Repeat Penalty"
                                value={samplerParams.repeatPenalty}
                                min={1} max={2} step={0.05}
                                disabled={nativeState.status !== 'loaded'}
                                hint="Discourages repetitive text"
                                format={(v) => v.toFixed(2)}
                                onChange={(v) => handleSamplerParamChange('repeatPenalty', v)}
                              />
                              <div>
                                <label className="text-xs font-medium text-gray-600">Seed</label>
                                <input
                                  type="number"
                                  value={samplerParams.seed}
                                  onChange={(e) => handleSamplerParamChange('seed', parseInt(e.target.value) || 0)}
                                  disabled={nativeState.status !== 'loaded'}
                                  className="w-full mt-1 p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 disabled:opacity-50 font-mono"
                                  placeholder="42"
                                />
                                <p className="text-xs text-gray-400 mt-0.5">Use -1 for random seed</p>
                              </div>
                              <button
                                onClick={handleResetSamplerParams}
                                disabled={nativeState.status !== 'loaded'}
                                className="text-xs text-gray-500 hover:text-gray-700 underline disabled:opacity-50 disabled:no-underline"
                              >
                                Reset to defaults
                              </button>
                            </div>
                          )}
                        </div>

                        {/* Context / inference parameters */}
                        <div className="border border-gray-200 rounded-xl overflow-hidden">
                          <button
                            onClick={() => setShowContextSettings(!showContextSettings)}
                            className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors"
                          >
                            <span className="flex items-center gap-2 text-sm font-medium text-gray-700">
                              <Cpu size={14} /> Context &amp; Memory
                            </span>
                            <span className={`text-gray-400 text-xs transition-transform ${showContextSettings ? 'rotate-180' : ''}`}>▼</span>
                          </button>
                          {showContextSettings && (
                            <div className="p-4 space-y-4 border-t border-gray-200">
                              <p className="text-xs text-gray-500">
                                Changes take effect on the next generation. <span className="font-medium text-amber-700">GPU Layers</span> requires reloading the model.
                              </p>

                              {/* Context window */}
                              <div className="grid grid-cols-2 gap-3">
                                <div>
                                  <label className="text-xs font-medium text-gray-600">Context (text)</label>
                                  <p className="text-[10px] text-gray-400 mb-1">Max tokens in memory</p>
                                  <input
                                    type="number"
                                    value={contextParams.nCtx}
                                    min={512} max={32768} step={512}
                                    onChange={(e) => handleContextParamChange('nCtx', parseInt(e.target.value) || DEFAULT_CONTEXT_PARAMS.nCtx)}
                                    className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 font-mono"
                                  />
                                </div>
                                <div>
                                  <label className="text-xs font-medium text-gray-600">Context (multimodal)</label>
                                  <p className="text-[10px] text-gray-400 mb-1">When images are present</p>
                                  <input
                                    type="number"
                                    value={contextParams.nCtxMultimodal}
                                    min={512} max={32768} step={512}
                                    onChange={(e) => handleContextParamChange('nCtxMultimodal', parseInt(e.target.value) || DEFAULT_CONTEXT_PARAMS.nCtxMultimodal)}
                                    className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 font-mono"
                                  />
                                </div>
                              </div>

                              {/* Batch sizes */}
                              <div className="grid grid-cols-2 gap-3">
                                <div>
                                  <label className="text-xs font-medium text-gray-600">Batch size (text)</label>
                                  <p className="text-[10px] text-gray-400 mb-1">Prompt eval chunk size</p>
                                  <input
                                    type="number"
                                    value={contextParams.nBatch}
                                    min={64} max={2048} step={64}
                                    onChange={(e) => handleContextParamChange('nBatch', parseInt(e.target.value) || DEFAULT_CONTEXT_PARAMS.nBatch)}
                                    className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 font-mono"
                                  />
                                </div>
                                <div>
                                  <label className="text-xs font-medium text-gray-600">Batch size (multimodal)</label>
                                  <p className="text-[10px] text-gray-400 mb-1">Smaller = less Metal memory</p>
                                  <input
                                    type="number"
                                    value={contextParams.nBatchMultimodal}
                                    min={64} max={2048} step={64}
                                    onChange={(e) => handleContextParamChange('nBatchMultimodal', parseInt(e.target.value) || DEFAULT_CONTEXT_PARAMS.nBatchMultimodal)}
                                    className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 font-mono"
                                  />
                                </div>
                              </div>

                              {/* Image token budget */}
                              <div>
                                <label className="text-xs font-medium text-gray-600">Image Token Budget</label>
                                <p className="text-[10px] text-gray-400 mb-1">Visual tokens per image — lower = faster, less memory (Gemma 4)</p>
                                <select
                                  value={contextParams.imageMaxTokens}
                                  onChange={(e) => {
                                    const v = parseInt(e.target.value);
                                    setContextParams(prev => ({ ...prev, imageMinTokens: v, imageMaxTokens: v }));
                                    NativeLlmManager.getInstance().setContextParams({ imageMinTokens: v, imageMaxTokens: v }).catch(() => {});
                                  }}
                                  className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 font-mono"
                                >
                                  <option value={-1}>Default (model decides)</option>
                                  <option value={70}>70 — fastest, low detail</option>
                                  <option value={140}>140 — fast</option>
                                  <option value={280}>280 — balanced</option>
                                  <option value={560}>560 — detailed</option>
                                  <option value={1120}>1120 — max detail (OCR/documents)</option>
                                </select>
                              </div>

                              {/* Threads + GPU layers */}
                              <div className="grid grid-cols-2 gap-3">
                                <div>
                                  <label className="text-xs font-medium text-gray-600">CPU Threads</label>
                                  <p className="text-[10px] text-gray-400 mb-1">0 = auto-detect</p>
                                  <input
                                    type="number"
                                    value={contextParams.nThreads}
                                    min={0} max={32} step={1}
                                    onChange={(e) => handleContextParamChange('nThreads', parseInt(e.target.value) || 0)}
                                    className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 font-mono"
                                  />
                                </div>
                                <div>
                                  <label className="text-xs font-medium text-gray-600 text-amber-700">GPU Layers</label>
                                  <p className="text-[10px] text-gray-400 mb-1">-1 = follow GPU toggle; 0 = CPU; 99 = all</p>
                                  <input
                                    type="number"
                                    value={contextParams.nGpuLayers}
                                    min={-1} max={999} step={1}
                                    onChange={(e) => handleContextParamChange('nGpuLayers', parseInt(e.target.value) ?? -1)}
                                    className="w-full p-2 text-sm border border-amber-200 rounded-md bg-amber-50 focus:ring-2 focus:ring-amber-400 font-mono"
                                  />
                                </div>
                              </div>

                              <button
                                onClick={handleResetContextParams}
                                className="text-xs text-gray-500 hover:text-gray-700 underline"
                              >
                                Reset to defaults
                              </button>
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                )}

                {/* ── Transformers.js tab ───────────────────── */}
                {activeTab === 'transformers' && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-3">
                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-1">Device</label>
                        <select
                          value={gemmaDevice}
                          onChange={e => setGemmaDevice(e.target.value as GemmaDevice)}
                          disabled={gemmaState.status === 'loading'}
                          className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
                        >
                          <option value="webgpu">WebGPU (GPU)</option>
                          <option value="wasm">WASM (CPU)</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-1">Precision</label>
                        <select
                          value={gemmaDtype}
                          onChange={e => setGemmaDtype(e.target.value as GemmaDtype)}
                          disabled={gemmaState.status === 'loading'}
                          className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
                        >
                          <option value="q4f16">q4f16 (4-bit + f16)</option>
                          <option value="q4">q4 (4-bit)</option>
                          <option value="q8">q8 (8-bit INT8)</option>
                          <option value="fp16">fp16 (half)</option>
                          <option value="fp32">fp32 (full)</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-1">Image Tokens</label>
                        <select
                          value={gemmaTokenBudget}
                          onChange={e => setGemmaTokenBudget(Number(e.target.value) as GemmaImageTokenBudget)}
                          disabled={gemmaState.status === 'loading'}
                          className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
                        >
                          <option value={70}>70 (fastest)</option>
                          <option value={140}>140</option>
                          <option value={280}>280</option>
                          <option value={560}>560</option>
                          <option value={1120}>1120 (OCR)</option>
                        </select>
                      </div>
                    </div>

                    {/* Thinking toggle */}
                    <div className="flex items-center justify-between gap-3 p-3 border border-gray-200 rounded-xl bg-white">
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-gray-800">Thinking Mode</p>
                        <p className="text-xs text-gray-500 mt-0.5">
                          Model reasons step-by-step before answering.
                        </p>
                      </div>
                      <button
                        onClick={() => setGemmaEnableThinking(v => !v)}
                        disabled={gemmaState.status === 'loading'}
                        className={`relative inline-flex h-6 w-10 flex-shrink-0 items-center rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                          gemmaEnableThinking ? 'bg-purple-600' : 'bg-gray-300'
                        }`}
                        aria-label={gemmaEnableThinking ? 'Disable thinking' : 'Enable thinking'}
                      >
                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
                          gemmaEnableThinking ? 'translate-x-5' : 'translate-x-1'
                        }`} />
                      </button>
                    </div>

                    {gemmaState.status === 'loaded' && (
                      <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                        Unload the current model to apply settings changes.
                      </p>
                    )}

                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Custom Hugging Face Model ID</label>
                      <p className="text-xs text-gray-500 mb-2">
                        e.g. <code className="bg-gray-100 px-1 py-0.5 rounded">onnx-community/gemma-3n-E2B-it-ONNX</code>
                      </p>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={customOnnxModelId}
                          onChange={(e) => setCustomOnnxModelId(e.target.value)}
                          placeholder="onnx-community/model-name"
                          disabled={gemmaState.status === 'loading'}
                          className="flex-grow p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 disabled:opacity-50 text-sm"
                        />
                        <button
                          onClick={handleLoadCustomOnnx}
                          disabled={!customOnnxModelId.trim() || gemmaState.status === 'loading'}
                          className="flex items-center gap-1.5 px-4 py-2.5 bg-gray-700 text-white rounded-lg hover:bg-gray-900 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
                        >
                          <Cpu size={14} /> Load
                        </button>
                      </div>

                      {/* Custom ONNX load progress */}
                      {gemmaState.status === 'loading' && gemmaState.modelId === customOnnxModelId.trim() && gemmaState.progress.length > 0 && (
                        <div className="space-y-1.5">
                          {gemmaState.progress.map((item) => (
                            <div key={item.file}>
                              <div className="flex justify-between items-center text-[11px] mb-1">
                                <span className="text-gray-600 flex items-center gap-1.5 truncate max-w-[55%]">
                                  {item.status === 'done'
                                    ? <CheckCircle className="h-3 w-3 text-green-500 flex-shrink-0" />
                                    : <FileDown className="h-3 w-3 text-purple-400 flex-shrink-0" />
                                  }
                                  {item.file}
                                </span>
                                <span className="font-medium text-gray-500 flex-shrink-0">
                                  {item.status === 'done'
                                    ? 'Done'
                                    : item.total > 0
                                      ? `${formatBytes(item.loaded)} / ${formatBytes(item.total)}`
                                      : `${Math.round(item.progress)}%`
                                  }
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-1.5">
                                <div
                                  className={`h-1.5 rounded-full transition-all duration-300 ${item.status === 'done' ? 'bg-green-500' : 'bg-purple-600'}`}
                                  style={{ width: `${item.progress}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      {gemmaState.status === 'error' && gemmaState.modelId === customOnnxModelId.trim() && (
                        <p className="text-xs text-red-600 bg-red-50 px-3 py-2 rounded-lg">{gemmaState.error}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* ── Servers tab ───────────────────────────── */}
                {activeTab === 'servers' && (
                  <div className="space-y-4">
                    {isTauriApp && (
                      <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <Server size={16} className="text-blue-600" />
                            <span className="font-semibold text-gray-800 text-sm">Local Server</span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                              localServerOnline ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                            }`}>
                              {localServerOnline ? 'Online' : 'Offline'}
                            </span>
                          </div>
                          <button
                            onClick={checkLocalServer}
                            className="p-1.5 hover:bg-white rounded transition-colors"
                            title="Re-check status"
                          >
                            <RefreshCw className="h-3.5 w-3.5 text-blue-600" />
                          </button>
                        </div>
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={inferenceUrlInput}
                            onChange={(e) => setInferenceUrlInput(e.target.value)}
                            placeholder="http://localhost:11434"
                            className="flex-grow p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
                          />
                          <button
                            onClick={() => onSetAppInferenceUrl?.(inferenceUrlInput)}
                            className="px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium text-sm"
                          >
                            Save
                          </button>
                        </div>
                        <p className="text-xs text-gray-500 mt-2">Set your local Ollama or compatible server</p>
                      </div>
                    )}

                    <div className="border border-gray-200 rounded-xl p-4">
                      <h3 className="text-sm font-semibold text-gray-700 mb-3">Custom Servers</h3>
                      {!isAddingServer ? (
                        <button
                          onClick={() => setIsAddingServer(true)}
                          className="w-full py-2.5 px-4 border-2 border-dashed border-gray-300 rounded-xl hover:border-blue-400 hover:bg-blue-50 flex items-center justify-center text-sm text-gray-600 hover:text-blue-600 transition-all"
                        >
                          <Plus className="h-4 w-4 mr-2" /> Add Server
                        </button>
                      ) : (
                        <div className="space-y-2">
                          <input
                            type="text"
                            value={newServerAddress}
                            onChange={(e) => { setNewServerAddress(e.target.value); setAddError(''); }}
                            placeholder="http://192.168.1.100:8080"
                            className="w-full px-3 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            autoFocus
                          />
                          {addError && <p className="text-xs text-red-500">{addError}</p>}
                          <div className="flex gap-2">
                            <button
                              onClick={handleAddServer}
                              className="flex-1 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium text-sm"
                            >
                              Add
                            </button>
                            <button
                              onClick={() => { setIsAddingServer(false); setNewServerAddress(''); setAddError(''); }}
                              className="flex-1 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 font-medium text-sm"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      )}

                      {customServers.length > 0 && (
                        <div className="space-y-2 mt-3">
                          {customServers.map(server => (
                            <div key={server.address} className="border border-gray-200 rounded-lg p-3 hover:border-gray-300 transition-all">
                              <div className="flex items-center justify-between gap-2">
                                <div className="flex-1 min-w-0">
                                  <p className="font-medium text-gray-800 text-sm truncate">{server.address}</p>
                                  <div className="flex items-center gap-1.5 mt-0.5">
                                    <span className={`text-[10px] font-semibold ${
                                      server.status === 'online' ? 'text-green-600' :
                                      server.status === 'offline' ? 'text-red-500' : 'text-gray-400'
                                    }`}>
                                      {server.status === 'online' ? 'Online' : server.status === 'offline' ? 'Offline' : 'Unchecked'}
                                    </span>
                                    <button
                                      onClick={() => onCheckCustomServer?.(server.address)}
                                      className="p-0.5 hover:bg-gray-100 rounded transition-colors"
                                      title="Re-check"
                                    >
                                      <RefreshCw className="h-3 w-3 text-gray-500" />
                                    </button>
                                  </div>
                                </div>
                                <div className="flex items-center gap-2 flex-shrink-0">
                                  <button
                                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                                      server.enabled ? 'bg-blue-500' : 'bg-gray-300'
                                    }`}
                                    onClick={() => onToggleCustomServer?.(server.address)}
                                    aria-label={server.enabled ? 'Disable server' : 'Enable server'}
                                  >
                                    <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${
                                      server.enabled ? 'translate-x-5' : 'translate-x-1'
                                    }`} />
                                  </button>
                                  <button
                                    onClick={() => onRemoveCustomServer?.(server.address)}
                                    className="p-1 hover:bg-red-50 rounded text-gray-400 hover:text-red-500 transition-colors"
                                    title="Remove server"
                                  >
                                    <Trash2 className="h-3.5 w-3.5" />
                                  </button>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {allOllamaServers.length > 0 && (
                      <div className="border border-blue-200 bg-blue-50/50 rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <Zap size={16} className="text-blue-600" />
                          <h3 className="text-sm font-semibold text-gray-700">Pull Ollama Model</h3>
                        </div>
                        <p className="text-xs text-gray-600 mb-3">
                          Download from the Ollama library (e.g. <code className="bg-blue-100 px-1.5 py-0.5 rounded text-xs font-medium">gemma3:4b</code>).
                        </p>
                        {allOllamaServers.length > 1 && (
                          <div className="mb-3">
                            <label className="block text-xs font-medium text-gray-600 mb-1">Server</label>
                            <select
                              value={selectedServer}
                              onChange={(e) => setSelectedServer(e.target.value)}
                              className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white text-sm"
                            >
                              {allOllamaServers.map(s => <option key={s} value={s}>{s}</option>)}
                            </select>
                          </div>
                        )}
                        {!isPulling && !isFinished && (
                          <div className="flex gap-2">
                            <input
                              type="text"
                              list="ollama-model-suggestions-hub"
                              value={modelToPull}
                              onChange={(e) => setModelToPull(e.target.value)}
                              placeholder="Enter model name..."
                              className="flex-grow p-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
                            />
                            <datalist id="ollama-model-suggestions-hub">
                              {SUGGESTED_OLLAMA_MODELS.map(m => <option key={m} value={m} />)}
                            </datalist>
                            <button
                              onClick={handleStartPull}
                              disabled={!selectedServer || !modelToPull.trim()}
                              className="flex items-center gap-1.5 px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
                            >
                              <Download size={14} /> Pull
                            </button>
                          </div>
                        )}

                        {isPulling && (
                          <div className="space-y-2 p-3 bg-blue-100 border border-blue-200 rounded-lg">
                            <div className="flex justify-between items-center">
                              <p className="text-sm font-semibold text-gray-800 truncate">{statusText}</p>
                              <p className="text-sm font-bold text-blue-600 flex-shrink-0">{progress}%</p>
                            </div>
                            <div className="w-full bg-blue-200 rounded-full h-2">
                              <div className="bg-blue-600 h-2 rounded-full transition-all" style={{ width: `${progress}%` }} />
                            </div>
                            <div className="flex justify-between items-center">
                              {totalBytes > 0
                                ? <p className="text-xs text-gray-500 font-mono">{formatBytes(completedBytes)} / {formatBytes(totalBytes)}</p>
                                : <div />}
                              <button
                                onClick={handleCancelPull}
                                className="flex items-center gap-1 px-2.5 py-1 text-xs text-red-600 bg-red-100 hover:bg-red-200 rounded-lg font-semibold transition-colors"
                              >
                                <StopCircle size={12} /> Cancel
                              </button>
                            </div>
                          </div>
                        )}

                        {status === 'success' && (
                          <div className="p-3 bg-green-100 border border-green-200 text-green-800 rounded-lg flex items-center gap-2">
                            <CheckCircle size={18} className="text-green-600" />
                            <div>
                              <span className="font-semibold text-sm">Download complete</span>
                              <p className="text-xs text-green-700">{statusText}</p>
                            </div>
                          </div>
                        )}

                        {status === 'error' && (
                          <div className="p-3 bg-red-100 border border-red-200 text-red-800 rounded-lg flex items-center gap-2">
                            <AlertTriangle size={18} className="text-red-600" />
                            <div>
                              <span className="font-semibold text-sm">Error</span>
                              <p className="text-xs text-red-700">{errorText}</p>
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {customServers.length === 0 && allOllamaServers.length === 0 && !isTauriApp && (
                      <div className="text-center py-6 border border-dashed border-gray-200 rounded-xl">
                        <Server size={28} className="text-gray-300 mx-auto mb-2" />
                        <p className="text-sm text-gray-500">No servers configured</p>
                        <p className="text-xs text-gray-400 mt-1">Add a custom server to connect to Ollama or OpenAI-compatible endpoints.</p>
                      </div>
                    )}
                  </div>
                )}

              </div>
            </div>
          )}
        </div>

      </div>
    </Modal>
    </>
  );
};

export default ModelHub;
