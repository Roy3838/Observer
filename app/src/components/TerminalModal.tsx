// components/TerminalModal.tsx

import React, { useState, useEffect } from 'react';
import Modal from '@components/EditAgent/Modal';
import { Download, CheckCircle, AlertTriangle, X, StopCircle, FileDown, Cpu, Trash2, BarChart3, AlertCircle, Settings2 } from 'lucide-react';
import pullModelManager, { PullState } from '@utils/pullModelManager';
import { platformFetch, isTauri, isMobile } from '@utils/platform';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import BenchmarkPanel from '@components/BenchmarkPanel';
import {
  GemmaModelId,
  GemmaModelState,
  GemmaDevice,
  GemmaDtype,
  GemmaImageTokenBudget,
  NativeModelInfo,
  NativeModelState,
  SamplerParams,
  DEFAULT_SAMPLER_PARAMS,
} from '@utils/localLlm/types';

// Preset GGUF models for llama.cpp simple mode
const LLAMA_PRESETS = {
  'gemma-4-E2B': {
    label: 'Gemma 4 E2B',
    size: '~1.5 GB',
    gguf: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q3_K_S.gguf',
    mmproj: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf',
  },
  'gemma-4-E4B': {
    label: 'Gemma 4 E4B',
    size: '~3 GB',
    gguf: 'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q3_K_S.gguf',
    mmproj: 'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf',
  },
} as const;

type LlamaPresetId = keyof typeof LLAMA_PRESETS;
type ViewMode = 'simple' | 'advanced';

interface TerminalModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPullComplete?: () => void;
  noModels?: boolean;
  ollamaServers?: string[];
}

const suggestedModels = [
  'gemma3:4b',
  'gemma3:12b',
  'gemma3:27b',
  'gemma3:27b-it-qat',
  'qwen2.5vl:3b',
  'qwen2.5vl:7b',
  'llava:7b',
  'llava:13b'
];

const GEMMA_CARDS: { modelId: GemmaModelId; label: string; size: string }[] = [
  { modelId: 'onnx-community/gemma-4-E2B-it-ONNX', label: 'Gemma 4 E2B', size: '~1.5 GB' },
  { modelId: 'onnx-community/gemma-4-E4B-it-ONNX', label: 'Gemma 4 E4B', size: '~3 GB' },
];

const formatBytes = (bytes: number, decimals = 2) => {
  if (!+bytes) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};

const TerminalModal: React.FC<TerminalModalProps> = ({ isOpen, onClose, onPullComplete, noModels = false, ollamaServers }) => {
  const [modelToPull, setModelToPull] = useState('');
  const [downloadState, setDownloadState] = useState<PullState>(pullModelManager.getInitialState());
  const [showWelcomeScreen, setShowWelcomeScreen] = useState(noModels);
  const [detectedServers, setDetectedServers] = useState<string[]>([]);
  const availableServers = ollamaServers || detectedServers;
  const [selectedServer, setSelectedServer] = useState<string>(availableServers[0] || '');

  const hasOllama = availableServers.length > 0;
  const isTauriApp = isTauri();
  const isMobileDevice = isMobile();
  const [activeTab, setActiveTab] = useState<'transformers' | 'llamacpp' | 'benchmark' | 'ollama'>('transformers');

  // Simple/Advanced view mode for llama.cpp and Transformers.js tabs
  const [llamaViewMode, setLlamaViewMode] = useState<ViewMode>('simple');
  const [transformersViewMode, setTransformersViewMode] = useState<ViewMode>('simple');

  // Track which preset is currently downloading (for sequential GGUF + mmproj download)
  const [downloadingPreset, setDownloadingPreset] = useState<LlamaPresetId | null>(null);
  const [presetDownloadStep, setPresetDownloadStep] = useState<'gguf' | 'mmproj' | null>(null);

  const [gemmaState, setGemmaState] = useState<GemmaModelState>(GemmaModelManager.getInstance().getState());
  // Default values match GemmaModelManager.DEFAULT_SETTINGS
  const [gemmaDevice, setGemmaDevice] = useState<GemmaDevice>('webgpu');
  const [gemmaDtype, setGemmaDtype] = useState<GemmaDtype>('q4');
  const [gemmaTokenBudget, setGemmaTokenBudget] = useState<GemmaImageTokenBudget>(70);

  // Native LLM state (llama.cpp on Tauri - iOS/macOS)
  const [nativeState, setNativeState] = useState<NativeModelState>(NativeLlmManager.getInstance().getState());
  const [nativeModels, setNativeModels] = useState<NativeModelInfo[]>([]);
  const [ggufUrl, setGgufUrl] = useState('');
  const [mmprojUrl, setMmprojUrl] = useState('');
  const [currentDownload, setCurrentDownload] = useState<'gguf' | 'mmproj' | null>(null);

  // Sampler parameters for llama.cpp advanced settings
  const [samplerParams, setSamplerParams] = useState<SamplerParams>({ ...DEFAULT_SAMPLER_PARAMS });
  const [showSamplerSettings, setShowSamplerSettings] = useState(false);

  // Custom ONNX model input for Transformers.js advanced mode
  const [customOnnxModelId, setCustomOnnxModelId] = useState('');

  // GPU acceleration setting for llama.cpp (defaults to CPU for compatibility)
  const [useGpu, setUseGpu] = useState<boolean>(() => NativeLlmManager.getInstance().getPersistedUseGpu());

  useEffect(() => {
    if (!isOpen) return;
    // Default to Ollama if available, otherwise llama.cpp on Tauri or Transformers.js on web
    if (hasOllama) {
      setActiveTab('ollama');
    } else if (isTauriApp) {
      setActiveTab('llamacpp');
    } else {
      setActiveTab('transformers');
    }
  }, [isOpen, hasOllama, isTauriApp]);

  useEffect(() => {
    if (isOpen && !ollamaServers && noModels) {
      const checkLocalhost = async () => {
        try {
          const response = await platformFetch('http://localhost:3838/api/tags', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
          });
          if (response.ok) {
            setDetectedServers(['http://localhost:3838']);
          }
        } catch {
          setDetectedServers([]);
        }
      };
      checkLocalhost();
    }
  }, [isOpen, ollamaServers, noModels]);

  useEffect(() => {
    if (!isOpen) return;
    const unsubscribe = pullModelManager.subscribe((newState) => {
      setDownloadState(newState);
      if (newState.status === 'success' && onPullComplete) {
        onPullComplete();
      }
    });
    return unsubscribe;
  }, [isOpen, onPullComplete]);

  useEffect(() => {
    if (!isOpen) return;
    const manager = GemmaModelManager.getInstance();
    const unsubscribe = manager.onStateChange(setGemmaState);
    const currentState = manager.getState();
    setGemmaState(currentState);

    // Initialize dropdown values from loaded model's settings, or first model's saved settings
    if (currentState.loadSettings) {
      setGemmaDevice(currentState.loadSettings.device);
      setGemmaDtype(currentState.loadSettings.dtype);
      setGemmaTokenBudget(currentState.loadSettings.imageTokenBudget);
    } else {
      // Use saved settings from first available model
      const defaultModelId = GEMMA_CARDS[0].modelId;
      const savedSettings = manager.getSettingsForModel(defaultModelId);
      setGemmaDevice(savedSettings.device);
      setGemmaDtype(savedSettings.dtype);
      setGemmaTokenBudget(savedSettings.imageTokenBudget);
    }

    return unsubscribe;
  }, [isOpen]);

  // Tauri: Subscribe to native LLM state changes
  useEffect(() => {
    if (!isOpen || !isTauriApp) return;
    const unsubscribe = NativeLlmManager.getInstance().onStateChange(setNativeState);
    setNativeState(NativeLlmManager.getInstance().getState());
    return unsubscribe;
  }, [isOpen, isTauriApp]);

  // Tauri: Fetch available native models whenever status changes
  useEffect(() => {
    if (!isOpen || !isTauriApp) return;
    NativeLlmManager.getInstance().listModels().then(setNativeModels);
  }, [isOpen, isTauriApp, nativeState.status]);

  // Tauri: Reset sampler params to defaults when model starts loading, fetch actual values when loaded
  useEffect(() => {
    if (!isOpen || !isTauriApp) return;

    if (nativeState.status === 'loading') {
      // Reset to defaults when a new model starts loading
      setSamplerParams({ ...DEFAULT_SAMPLER_PARAMS });
    } else if (nativeState.status === 'loaded') {
      // Fetch current params from the backend once loaded
      NativeLlmManager.getInstance().getDebugInfo().then(info => {
        if (info.engine.samplerParams) {
          setSamplerParams(info.engine.samplerParams);
        }
      }).catch(() => {
        // Ignore errors, use defaults
      });
    } else if (nativeState.status === 'unloaded') {
      // Reset when unloaded
      setSamplerParams({ ...DEFAULT_SAMPLER_PARAMS });
    }
  }, [isOpen, isTauriApp, nativeState.status]);

  const handleStartPull = () => {
    if (modelToPull.trim() && selectedServer) {
      pullModelManager.pullModel(modelToPull.trim(), selectedServer);
    }
  };

  const handlePullModelClick = () => {
    setModelToPull('gemma3:4b');
    setShowWelcomeScreen(false);
    if (selectedServer) {
      pullModelManager.pullModel('gemma3:4b', selectedServer);
    }
  };

  const handleSkipForNow = () => {
    setShowWelcomeScreen(false);
  };

  const handleCancelPull = () => {
    pullModelManager.cancelPull();
  };

  const handleDone = () => {
    if (downloadState.status === 'success' || downloadState.status === 'error') {
      pullModelManager.resetState();
    }
    onClose();
  };

  const handleDownloadGguf = async () => {
    if (!ggufUrl.trim()) return;
    setCurrentDownload('gguf');
    try {
      await NativeLlmManager.getInstance().downloadModel(ggufUrl.trim());
      setGgufUrl('');
    } catch {
      // Error shown via nativeState.error
    } finally {
      setCurrentDownload(null);
    }
  };

  const handleDownloadMmproj = async () => {
    if (!mmprojUrl.trim()) return;
    setCurrentDownload('mmproj');
    try {
      await NativeLlmManager.getInstance().downloadModel(mmprojUrl.trim());
      setMmprojUrl('');
    } catch {
      // Error shown via nativeState.error
    } finally {
      setCurrentDownload(null);
    }
  };

  // Update a single sampler parameter and sync to backend
  const handleSamplerParamChange = async (key: keyof SamplerParams, value: number) => {
    const newParams = { ...samplerParams, [key]: value };
    setSamplerParams(newParams);
    if (nativeState.status === 'loaded') {
      try {
        await NativeLlmManager.getInstance().setSamplerParams({ [key]: value });
      } catch {
        // Ignore errors, state is still updated locally
      }
    }
  };

  // Reset sampler params to defaults
  const handleResetSamplerParams = async () => {
    setSamplerParams({ ...DEFAULT_SAMPLER_PARAMS });
    if (nativeState.status === 'loaded') {
      try {
        await NativeLlmManager.getInstance().setSamplerParams(DEFAULT_SAMPLER_PARAMS);
      } catch {
        // Ignore errors
      }
    }
  };

  // Toggle GPU acceleration for llama.cpp
  const handleToggleGpu = async (enabled: boolean) => {
    setUseGpu(enabled);
    try {
      await NativeLlmManager.getInstance().setUseGpu(enabled);
    } catch {
      // Ignore errors, setting is still persisted locally
    }
  };

  // Sequential download for preset models (GGUF first, then mmproj automatically)
  const handleDownloadPreset = async (presetId: LlamaPresetId) => {
    const preset = LLAMA_PRESETS[presetId];
    setDownloadingPreset(presetId);

    try {
      // Step 1: Download GGUF
      setPresetDownloadStep('gguf');
      await NativeLlmManager.getInstance().downloadModel(preset.gguf);

      // Step 2: Download mmproj automatically
      setPresetDownloadStep('mmproj');
      await NativeLlmManager.getInstance().downloadModel(preset.mmproj);

    } catch {
      // Error shown via nativeState.error
    } finally {
      setDownloadingPreset(null);
      setPresetDownloadStep(null);
    }
  };

  // Get native model matching a preset
  const getPresetModel = (presetId: LlamaPresetId): NativeModelInfo | undefined => {
    const preset = LLAMA_PRESETS[presetId];
    const ggufFilename = preset.gguf.split('/').pop()?.replace('.gguf', '') || '';
    return nativeModels.find(m => m.id.includes(ggufFilename.split('-Q')[0]));
  };

  useEffect(() => {
    if (isOpen) setShowWelcomeScreen(noModels);
  }, [isOpen, noModels]);

  useEffect(() => {
    if (availableServers.length > 0 && !selectedServer) {
      setSelectedServer(availableServers[0]);
    }
  }, [availableServers, selectedServer]);

  const { status, progress, statusText, errorText, completedBytes, totalBytes } = downloadState;
  const isPulling = status === 'pulling';
  const isFinished = status === 'success' || status === 'error';
  const isNativeDownloading = nativeState.status === 'downloading';

  return (
    <Modal open={isOpen} onClose={handleDone} className="w-full max-w-3xl">
      <div className="p-8 max-h-[85vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Model Hub</h2>
            <p className="text-sm text-gray-500 mt-1">Download and manage local AI models</p>
          </div>
          <button onClick={handleDone} className="text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full p-2 transition-colors">
            <X size={22} />
          </button>
        </div>

        {/* Tab bar - always show all 4 tabs */}
        {!showWelcomeScreen && (
          <div className="flex gap-2 mb-6 p-1 bg-gray-100 rounded-xl overflow-x-auto">
            <button
              onClick={() => setActiveTab('transformers')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg transition-all whitespace-nowrap ${
                activeTab === 'transformers'
                  ? 'bg-white text-purple-700 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${activeTab === 'transformers' ? 'bg-purple-500' : 'bg-gray-300'}`} />
              Transformers.js
            </button>
            <button
              onClick={() => setActiveTab('llamacpp')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg transition-all whitespace-nowrap ${
                activeTab === 'llamacpp'
                  ? 'bg-white text-green-700 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              } ${!isTauriApp ? 'opacity-60' : ''}`}
            >
              <span className={`w-2 h-2 rounded-full ${activeTab === 'llamacpp' ? 'bg-green-500' : 'bg-gray-300'}`} />
              <Cpu size={14} />
              llama.cpp
              {!isTauriApp && <AlertCircle size={12} className="text-gray-400" />}
            </button>
            <button
              onClick={() => setActiveTab('benchmark')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg transition-all whitespace-nowrap ${
                activeTab === 'benchmark'
                  ? 'bg-white text-orange-700 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${activeTab === 'benchmark' ? 'bg-orange-500' : 'bg-gray-300'}`} />
              <BarChart3 size={14} />
              Benchmark
            </button>
            {hasOllama && (
            <button
              onClick={() => setActiveTab('ollama')}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg transition-all whitespace-nowrap ${
                activeTab === 'ollama'
                  ? 'bg-white text-blue-700 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              } ${!hasOllama ? 'opacity-60' : ''}`}
            >
              <span className={`w-2 h-2 rounded-full ${activeTab === 'ollama' ? 'bg-blue-500' : 'bg-gray-300'}`} />
              Ollama
              {!hasOllama && <AlertCircle size={12} className="text-gray-400" />}
            </button>
            )}
          </div>
        )}

        {/* ── OLLAMA TAB ── */}
        {activeTab === 'ollama' && !showWelcomeScreen && (
          <div className="space-y-5">
            {!hasOllama ? (
              /* Ollama unavailable state */
              <div className="text-center py-10">
                <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gray-100 mb-5">
                  <AlertCircle size={40} className="text-gray-400" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-2">No Ollama Server Detected</h3>
                <p className="text-gray-500 text-sm mb-5 max-w-md mx-auto">
                  Ollama must be running locally to download and run models. Start Ollama or check your connection.
                </p>
                <a
                  href="https://ollama.ai"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium text-sm"
                >
                  Get Ollama →
                </a>
                <div className="mt-8 opacity-50">
                  <div className="flex flex-col sm:flex-row gap-3">
                    <input
                      type="text"
                      placeholder="Enter model name..."
                      disabled
                      className="flex-grow p-3 border border-gray-300 rounded-xl bg-gray-50 cursor-not-allowed"
                    />
                    <button
                      disabled
                      className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-3 bg-blue-600 text-white rounded-xl opacity-50 cursor-not-allowed font-medium"
                    >
                      <Download size={18} />
                      <span>Download</span>
                    </button>
                  </div>
                </div>
              </div>
            ) : !isFinished && !isPulling ? (
              <>
                <p className="text-gray-600 text-sm">
                  Enter a model name from the Ollama library (e.g., <code className="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-medium">gemma3:4b</code>).
                </p>
                {availableServers.length > 1 && (
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Select Ollama Server</label>
                    <select
                      value={selectedServer}
                      onChange={(e) => setSelectedServer(e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 bg-white"
                    >
                      {availableServers.map((server) => (
                        <option key={server} value={server}>{server}</option>
                      ))}
                    </select>
                  </div>
                )}
                <div className="flex flex-col sm:flex-row gap-3">
                  <input
                    type="text" list="ollama-model-suggestions" value={modelToPull}
                    onChange={(e) => setModelToPull(e.target.value)}
                    placeholder="Enter model name..."
                    className="flex-grow p-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500"
                  />
                  <datalist id="ollama-model-suggestions">
                    {suggestedModels.map(model => <option key={model} value={model} />)}
                  </datalist>
                  <button
                    onClick={handleStartPull}
                    disabled={!hasOllama}
                    className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-sm"
                  >
                    <Download size={18} />
                    <span>Download</span>
                  </button>
                </div>
              </>
            ) : null}

            {isPulling && (
              <div className="space-y-4 p-5 bg-blue-50 border border-blue-200 rounded-xl">
                <div className="flex justify-between items-center">
                  <p className="text-sm font-semibold text-gray-800">{statusText}</p>
                  <p className="text-sm font-bold text-blue-600">{progress}%</p>
                </div>
                <div className="w-full bg-blue-200 rounded-full h-2.5">
                  <div className="bg-blue-600 h-2.5 rounded-full transition-all duration-150" style={{ width: `${progress}%` }} />
                </div>
                <div className="flex justify-between items-center">
                  {totalBytes > 0 ? (
                    <p className="text-xs text-gray-500 font-mono">{formatBytes(completedBytes)} / {formatBytes(totalBytes)}</p>
                  ) : <div />}
                  <button onClick={handleCancelPull} className="flex items-center gap-1.5 px-4 py-2 text-xs text-red-600 bg-red-100 hover:bg-red-200 rounded-lg font-semibold transition-colors">
                    <StopCircle size={14} /> Cancel
                  </button>
                </div>
              </div>
            )}

            {status === 'success' && (
              <div className="p-6 bg-green-50 border border-green-200 text-green-800 rounded-xl flex flex-col items-center gap-4 text-center">
                <div className="w-16 h-16 rounded-full bg-green-200 flex items-center justify-center">
                  <CheckCircle size={32} className="text-green-600" />
                </div>
                <div>
                  <h3 className="font-bold text-lg">Download Complete!</h3>
                  <p className="text-sm text-green-700 mt-1">{statusText}</p>
                </div>
              </div>
            )}

            {status === 'error' && (
              <div className="p-6 bg-red-50 border border-red-200 text-red-800 rounded-xl flex flex-col items-center gap-4 text-center">
                <div className="w-16 h-16 rounded-full bg-red-200 flex items-center justify-center">
                  <AlertTriangle size={32} className="text-red-600" />
                </div>
                <div>
                  <h3 className="font-bold text-lg">An Error Occurred</h3>
                  <p className="text-sm text-red-700 mt-1">{errorText}</p>
                </div>
              </div>
            )}

            {isFinished && (
              <div className="mt-6 flex justify-end">
                <button onClick={handleDone} className="px-6 py-2.5 bg-gray-800 text-white rounded-xl hover:bg-gray-900 font-medium shadow-sm transition-colors">
                  Done
                </button>
              </div>
            )}
          </div>
        )}

        {/* Welcome screen (always shown regardless of tab when noModels) */}
        {showWelcomeScreen && (
          <div>
            <div className="flex items-center gap-3 mb-4">
              <Download className="h-7 w-7 text-green-500 flex-shrink-0" />
              <h2 className="text-xl sm:text-2xl font-semibold">Let's Get Your First Model</h2>
            </div>
            <p className="text-gray-700 mb-6">
              Your local server is running, but it looks like you don't have any AI models installed yet. Models are the "brains" that power your agents.
            </p>
            <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
              <h3 className="font-semibold text-green-800 mb-2 text-lg">Recommended Model: Gemma3 4B</h3>
              <button
                onClick={handlePullModelClick}
                className="w-full sm:w-auto px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium text-base shadow-sm hover:shadow-md"
              >
                Pull Your First Model!
              </button>
            </div>
            <div className="mt-8 flex flex-col-reverse sm:flex-row justify-between items-center gap-4">
              <button onClick={handleSkipForNow} className="text-sm text-gray-600 hover:underline">
                Choose a different model
              </button>
              <button onClick={handleDone} className="text-sm text-gray-600 hover:underline">
                I'll do this later
              </button>
            </div>
          </div>
        )}


        {/* ── LLAMA.CPP TAB ── */}
        {activeTab === 'llamacpp' && !showWelcomeScreen && (
          <div className="space-y-5">
            {!isTauriApp ? (
              /* llama.cpp unavailable - not in Tauri */
              <div className="text-center py-10">
                <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gray-100 mb-5">
                  <Cpu size={40} className="text-gray-400" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-2">Native App Required</h3>
                <p className="text-gray-500 text-sm mb-5 max-w-md mx-auto">
                  llama.cpp runs natively with GPU acceleration. Install the native app for the best local AI experience.
                </p>
                <div className="mt-8 opacity-50">
                  <div className="space-y-3">
                    {Object.entries(LLAMA_PRESETS).map(([id, preset]) => (
                      <div key={id} className="border border-gray-200 rounded-xl p-4 bg-gray-50">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-gray-200 flex items-center justify-center">
                              <Cpu size={20} className="text-gray-400" />
                            </div>
                            <div>
                              <span className="font-semibold text-gray-700">{preset.label}</span>
                              <span className="block text-xs text-gray-400">{preset.size}</span>
                            </div>
                          </div>
                          <button
                            disabled
                            className="px-4 py-2 text-sm bg-green-600 text-white rounded-lg opacity-50 cursor-not-allowed font-medium"
                          >
                            Download
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <>
                {/* Simple/Advanced toggle */}
                <div className="flex items-center justify-between">
                  <p className="text-gray-500 text-sm">
                    Run models on your device.
                  </p>
                  <div className="flex rounded-xl border border-gray-200 overflow-hidden bg-gray-100 p-0.5">
                    <button
                      onClick={() => setLlamaViewMode('simple')}
                      className={`px-4 py-1.5 text-xs font-medium transition-all rounded-lg ${
                        llamaViewMode === 'simple'
                          ? 'bg-white text-green-700 shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      Simple
                    </button>
                    <button
                      onClick={() => setLlamaViewMode('advanced')}
                      className={`px-4 py-1.5 text-xs font-medium transition-all rounded-lg ${
                        llamaViewMode === 'advanced'
                          ? 'bg-white text-green-700 shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      Advanced
                    </button>
                  </div>
                </div>

                {llamaViewMode === 'simple' ? (
                  /* Simple mode: Preset cards */
                  <div className="space-y-3">
                    {/* GPU Toggle */}
                    <div className="flex items-center justify-between p-4 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-gray-800">GPU Acceleration</span>
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${useGpu ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-700'}`}>
                            {useGpu ? 'On' : 'Off'}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
                          {useGpu
                            ? 'Hardware-accelerated for faster performance. May cause instability.'
                            : 'CPU mode for broader compatibility.'}
                        </p>
                      </div>
                      <button
                        onClick={() => handleToggleGpu(!useGpu)}
                        disabled={nativeState.status === 'loading' || nativeState.status === 'loaded'}
                        className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed ${
                          useGpu ? 'bg-green-600' : 'bg-gray-300'
                        }`}
                        title={nativeState.status === 'loaded' ? 'Unload model to change this setting' : undefined}
                      >
                        <span
                          className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
                            useGpu ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>

                    {(Object.entries(LLAMA_PRESETS) as [LlamaPresetId, typeof LLAMA_PRESETS[LlamaPresetId]][]).map(([presetId, preset]) => {
                      const presetModel = getPresetModel(presetId);
                      const isDownloaded = !!presetModel;
                      const isThisDownloading = downloadingPreset === presetId;
                      const isThisModelLoaded = presetModel && nativeState.modelId === presetModel.id && nativeState.status === 'loaded';
                      const isThisModelLoading = presetModel && nativeState.modelId === presetModel.id && nativeState.status === 'loading';
                      const isThisModelUnloading = presetModel && nativeState.modelId === presetModel.id && nativeState.status === 'unloading';
                      const isAnyModelBusy = nativeState.status === 'loading' || nativeState.status === 'unloading' || isNativeDownloading;

                      return (
                        <div key={presetId} className={`border rounded-xl p-4 transition-all ${isThisModelLoaded ? 'border-green-300 bg-green-50/50' : 'border-gray-200 bg-white hover:border-gray-300'}`}>
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${isThisModelLoaded ? 'bg-green-200' : 'bg-gray-100'}`}>
                                <Cpu size={20} className={isThisModelLoaded ? 'text-green-700' : 'text-gray-500'} />
                              </div>
                              <div>
                                <span className="font-semibold text-gray-900">{preset.label}</span>
                                <div className="flex items-center gap-2 mt-0.5">
                                  <span className="text-xs text-gray-500">{preset.size}</span>
                                  {isDownloaded && presetModel?.isMultimodal && (
                                    <span className="text-xs text-purple-600 font-medium bg-purple-100 px-1.5 py-0.5 rounded">Vision</span>
                                  )}
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              {isThisModelUnloading ? (
                                <span className="flex items-center gap-1.5 px-4 py-2 text-sm bg-amber-100 text-amber-700 rounded-lg font-medium">
                                  <Cpu size={14} className="animate-pulse" /> Unloading…
                                </span>
                              ) : isThisModelLoaded ? (
                                <button
                                  onClick={() => NativeLlmManager.getInstance().unloadModel()}
                                  className="group flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg font-medium transition-colors bg-green-200 text-green-800 hover:bg-red-100 hover:text-red-600"
                                >
                                  <span className="group-hover:hidden flex items-center gap-1.5">
                                    <CheckCircle size={14} /> Ready
                                  </span>
                                  <span className="hidden group-hover:flex items-center gap-1.5">
                                    <X size={14} /> Unload
                                  </span>
                                </button>
                              ) : isThisModelLoading ? (
                                <button
                                  onClick={() => NativeLlmManager.getInstance().unloadModel()}
                                  className="flex items-center gap-1.5 px-4 py-2 text-sm bg-gray-200 text-gray-600 rounded-lg font-medium"
                                >
                                  <Cpu size={14} className="animate-pulse" /> Loading…
                                </button>
                              ) : isDownloaded ? (
                                <button
                                  disabled={isAnyModelBusy}
                                  onClick={() => NativeLlmManager.getInstance().loadModel(presetModel!.filename, presetModel!.mmprojFilename)}
                                  className="flex items-center gap-1.5 px-4 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-sm"
                                >
                                  <Cpu size={14} /> Load
                                </button>
                              ) : isThisDownloading ? (
                                <button
                                  onClick={async () => {
                                    await NativeLlmManager.getInstance().cancelDownload();
                                    setDownloadingPreset(null);
                                    setPresetDownloadStep(null);
                                  }}
                                  className="flex items-center gap-1.5 px-4 py-2 text-sm bg-red-100 text-red-600 hover:bg-red-200 rounded-lg font-medium transition-colors"
                                >
                                  <StopCircle size={14} /> Cancel
                                </button>
                              ) : (
                                <button
                                  disabled={isNativeDownloading || !!downloadingPreset}
                                  onClick={() => handleDownloadPreset(presetId)}
                                  className="flex items-center gap-1.5 px-4 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-sm"
                                >
                                  <Download size={14} /> Download
                                </button>
                              )}
                            </div>
                          </div>

                          {/* Download progress for this preset - show both bars */}
                          {isThisDownloading && (
                            <div className="mt-4 space-y-3">
                              {/* GGUF Model Progress */}
                              <div>
                                <div className="flex justify-between items-center text-xs mb-1.5">
                                  <span className="text-gray-600 flex items-center gap-1.5">
                                    {presetDownloadStep === 'mmproj' ? (
                                      <CheckCircle className="h-3.5 w-3.5 text-green-500" />
                                    ) : (
                                      <FileDown className="h-3.5 w-3.5 text-blue-500" />
                                    )}
                                    Model (.gguf)
                                  </span>
                                  <span className="font-medium text-gray-500">
                                    {presetDownloadStep === 'mmproj'
                                      ? 'Done'
                                      : nativeState.totalBytes > 0
                                        ? `${formatBytes(nativeState.downloadedBytes)} / ${formatBytes(nativeState.totalBytes)}`
                                        : `${nativeState.downloadProgress}%`
                                    }
                                  </span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                  <div
                                    className={`h-2 rounded-full transition-all duration-300 ${presetDownloadStep === 'mmproj' ? 'bg-green-500' : 'bg-blue-600'}`}
                                    style={{ width: presetDownloadStep === 'mmproj' ? '100%' : `${nativeState.downloadProgress}%` }}
                                  />
                                </div>
                              </div>

                              {/* mmproj Vision Projector Progress */}
                              <div>
                                <div className="flex justify-between items-center text-xs mb-1.5">
                                  <span className="text-gray-600 flex items-center gap-1.5">
                                    {presetDownloadStep === 'mmproj' ? (
                                      <FileDown className="h-3.5 w-3.5 text-purple-500" />
                                    ) : (
                                      <FileDown className="h-3.5 w-3.5 text-gray-300" />
                                    )}
                                    Vision projector (.gguf)
                                  </span>
                                  <span className="font-medium text-gray-500">
                                    {presetDownloadStep === 'mmproj'
                                      ? nativeState.totalBytes > 0
                                        ? `${formatBytes(nativeState.downloadedBytes)} / ${formatBytes(nativeState.totalBytes)}`
                                        : `${nativeState.downloadProgress}%`
                                      : 'Pending'
                                    }
                                  </span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                  <div
                                    className={`h-2 rounded-full transition-all duration-300 ${presetDownloadStep === 'mmproj' ? 'bg-purple-500' : 'bg-gray-200'}`}
                                    style={{ width: presetDownloadStep === 'mmproj' ? `${nativeState.downloadProgress}%` : '0%' }}
                                  />
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  /* Advanced mode: Custom URLs + downloaded models list */
                  <>
                    {/* GGUF URL input */}
                    <div className="space-y-2">
                      <label className="block text-xs font-medium text-gray-600">Model (.gguf)</label>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={ggufUrl}
                          onChange={(e) => setGgufUrl(e.target.value)}
                          placeholder="https://huggingface.co/.../resolve/main/model.gguf"
                          disabled={isNativeDownloading}
                          className="flex-grow p-2.5 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 disabled:opacity-50 text-sm"
                        />
                        <button
                          onClick={handleDownloadGguf}
                          disabled={!ggufUrl.trim() || isNativeDownloading}
                          className="flex items-center gap-1.5 px-3 py-2.5 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
                        >
                          <Download size={16} />
                          Download
                        </button>
                      </div>
                    </div>

                    {/* mmproj URL input */}
                    <div className="space-y-2">
                      <label className="block text-xs font-medium text-gray-600">
                        Vision projector — mmproj (.gguf) <span className="text-gray-400 font-normal">optional</span>
                      </label>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={mmprojUrl}
                          onChange={(e) => setMmprojUrl(e.target.value)}
                          placeholder="https://huggingface.co/.../resolve/main/mmproj.gguf"
                          disabled={isNativeDownloading}
                          className="flex-grow p-2.5 border border-gray-300 rounded-md focus:ring-2 focus:ring-purple-500 disabled:opacity-50 text-sm"
                        />
                        <button
                          onClick={handleDownloadMmproj}
                          disabled={!mmprojUrl.trim() || isNativeDownloading}
                          className="flex items-center gap-1.5 px-3 py-2.5 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
                        >
                          <Download size={16} />
                          Download
                        </button>
                      </div>
                    </div>

                    {/* Download progress */}
                    {isNativeDownloading && !downloadingPreset && (
                      <div className="border border-blue-200 bg-blue-50 rounded-xl p-4">
                        <div className="flex justify-between items-center mb-3">
                          <span className="text-sm text-gray-800 font-semibold">
                            Downloading {currentDownload === 'mmproj' ? 'vision projector' : 'model'}
                          </span>
                          <button
                            onClick={async () => {
                              await NativeLlmManager.getInstance().cancelDownload();
                              setCurrentDownload(null);
                            }}
                            className="flex items-center gap-1 px-3 py-1.5 text-xs bg-red-100 text-red-600 hover:bg-red-200 rounded-lg font-medium transition-colors"
                          >
                            <StopCircle size={12} /> Cancel
                          </button>
                        </div>
                        <div>
                          <div className="flex justify-between items-center text-xs mb-1.5">
                            <span className="text-gray-600 flex items-center gap-1.5">
                              <FileDown className={`h-3.5 w-3.5 ${currentDownload === 'mmproj' ? 'text-purple-500' : 'text-blue-500'}`} />
                              {nativeState.modelId}
                            </span>
                            <span className="font-medium text-gray-500">
                              {nativeState.totalBytes > 0
                                ? `${formatBytes(nativeState.downloadedBytes)} / ${formatBytes(nativeState.totalBytes)}`
                                : `${nativeState.downloadProgress}%`
                              }
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full transition-all duration-300 ${currentDownload === 'mmproj' ? 'bg-purple-500' : 'bg-blue-600'}`}
                              style={{ width: `${nativeState.downloadProgress}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Error state */}
                    {nativeState.status === 'error' && (
                      <div className="border border-red-200 bg-red-50 rounded-lg p-3">
                        <p className="text-xs text-red-600">{nativeState.error}</p>
                      </div>
                    )}

                    {/* Downloaded models list */}
                    {nativeModels.length > 0 && (
                      <div className="space-y-2">
                        <h3 className="text-sm font-medium text-gray-700">Downloaded Models</h3>
                        {nativeModels.map((model) => {
                          const isThisModel = nativeState.modelId === model.id;
                          const isLoaded = isThisModel && nativeState.status === 'loaded';
                          const isLoading = isThisModel && nativeState.status === 'loading';
                          const isUnloading = isThisModel && nativeState.status === 'unloading';
                          const isAnyModelBusy = nativeState.status === 'loading' || nativeState.status === 'unloading' || isNativeDownloading;

                          return (
                            <div key={model.filename} className="border border-gray-200 rounded-lg p-3">
                              <div className="flex items-center justify-between">
                                <div className="flex-1 min-w-0">
                                  <span className="font-medium text-gray-900 truncate block">{model.name}</span>
                                  <span className="text-xs text-gray-500">
                                    {formatBytes(model.sizeBytes)}
                                    {model.isMultimodal && (
                                      <span className="ml-2 text-purple-600 font-medium">· Vision ({model.mmprojFilename})</span>
                                    )}
                                  </span>
                                </div>
                                <div className="flex items-center gap-2 ml-2">
                                  {isUnloading ? (
                                    <span className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-amber-100 text-amber-700 rounded-lg font-medium">
                                      <Cpu size={14} className="animate-pulse" /> Unloading…
                                    </span>
                                  ) : isLoaded ? (
                                    <>
                                      <span className="flex items-center gap-1 text-xs font-semibold text-green-700 bg-green-100 px-2 py-1 rounded-full">
                                        <CheckCircle size={12} /> Ready
                                      </span>
                                      <button
                                        onClick={() => NativeLlmManager.getInstance().unloadModel()}
                                        className="p-1.5 text-gray-400 hover:text-red-600 rounded transition-colors"
                                        title="Unload model"
                                      >
                                        <X size={16} />
                                      </button>
                                    </>
                                  ) : isLoading ? (
                                    <button
                                      onClick={() => NativeLlmManager.getInstance().unloadModel()}
                                      className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-gray-200 text-gray-600 rounded-lg font-medium"
                                    >
                                      <Cpu size={14} className="animate-pulse" /> Loading…
                                    </button>
                                  ) : (
                                    <>
                                      <button
                                        disabled={isAnyModelBusy}
                                        onClick={() => NativeLlmManager.getInstance().loadModel(model.filename, model.mmprojFilename)}
                                        className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                                      >
                                        <Cpu size={14} /> Load
                                      </button>
                                      <button
                                        disabled={isAnyModelBusy}
                                        onClick={() => NativeLlmManager.getInstance().deleteModel(model.filename)}
                                        className="p-1.5 text-gray-400 hover:text-red-600 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                        title="Delete model"
                                      >
                                        <Trash2 size={16} />
                                      </button>
                                    </>
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {nativeModels.length === 0 && !isNativeDownloading && (
                      <p className="text-gray-500 text-sm text-center py-4">
                        No models downloaded yet. Paste a HuggingFace GGUF URL above to get started.
                      </p>
                    )}

                    {/* Sampler Settings - collapsible panel */}
                    <div className="border border-gray-200 rounded-lg overflow-hidden">
                      <button
                        onClick={() => setShowSamplerSettings(!showSamplerSettings)}
                        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors"
                      >
                        <span className="flex items-center gap-2 text-sm font-medium text-gray-700">
                          <Settings2 size={16} />
                          Generation Settings
                        </span>
                        <span className={`text-gray-400 transition-transform ${showSamplerSettings ? 'rotate-180' : ''}`}>
                          ▼
                        </span>
                      </button>
                      {showSamplerSettings && (
                        <div className="p-4 space-y-4 border-t border-gray-200">
                          {nativeState.status !== 'loaded' && (
                            <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded px-2 py-1.5">
                              Load a model to configure generation settings.
                            </p>
                          )}

                          {/* Temperature */}
                          <div>
                            <div className="flex justify-between items-center mb-1">
                              <label className="text-xs font-medium text-gray-600">Temperature</label>
                              <span className="text-xs text-gray-500 font-mono">{samplerParams.temperature.toFixed(2)}</span>
                            </div>
                            <input
                              type="range"
                              min="0"
                              max="2"
                              step="0.05"
                              value={samplerParams.temperature}
                              onChange={(e) => handleSamplerParamChange('temperature', parseFloat(e.target.value))}
                              disabled={nativeState.status !== 'loaded'}
                              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-green-600"
                            />
                            <p className="text-xs text-gray-400 mt-0.5">Higher = more creative, lower = more focused</p>
                          </div>

                          {/* Top P */}
                          <div>
                            <div className="flex justify-between items-center mb-1">
                              <label className="text-xs font-medium text-gray-600">Top P (nucleus sampling)</label>
                              <span className="text-xs text-gray-500 font-mono">{samplerParams.topP.toFixed(2)}</span>
                            </div>
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={samplerParams.topP}
                              onChange={(e) => handleSamplerParamChange('topP', parseFloat(e.target.value))}
                              disabled={nativeState.status !== 'loaded'}
                              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-green-600"
                            />
                          </div>

                          {/* Top K */}
                          <div>
                            <div className="flex justify-between items-center mb-1">
                              <label className="text-xs font-medium text-gray-600">Top K</label>
                              <span className="text-xs text-gray-500 font-mono">{samplerParams.topK}</span>
                            </div>
                            <input
                              type="range"
                              min="1"
                              max="100"
                              step="1"
                              value={samplerParams.topK}
                              onChange={(e) => handleSamplerParamChange('topK', parseInt(e.target.value))}
                              disabled={nativeState.status !== 'loaded'}
                              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-green-600"
                            />
                          </div>

                          {/* Repeat Penalty */}
                          <div>
                            <div className="flex justify-between items-center mb-1">
                              <label className="text-xs font-medium text-gray-600">Repeat Penalty</label>
                              <span className="text-xs text-gray-500 font-mono">{samplerParams.repeatPenalty.toFixed(2)}</span>
                            </div>
                            <input
                              type="range"
                              min="1.0"
                              max="2.0"
                              step="0.05"
                              value={samplerParams.repeatPenalty}
                              onChange={(e) => handleSamplerParamChange('repeatPenalty', parseFloat(e.target.value))}
                              disabled={nativeState.status !== 'loaded'}
                              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-green-600"
                            />
                            <p className="text-xs text-gray-400 mt-0.5">Discourages repetitive text</p>
                          </div>

                          {/* Seed */}
                          <div>
                            <div className="flex justify-between items-center mb-1">
                              <label className="text-xs font-medium text-gray-600">Seed</label>
                            </div>
                            <input
                              type="number"
                              value={samplerParams.seed}
                              onChange={(e) => handleSamplerParamChange('seed', parseInt(e.target.value) || 0)}
                              disabled={nativeState.status !== 'loaded'}
                              className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 disabled:opacity-50 font-mono"
                              placeholder="42"
                            />
                            <p className="text-xs text-gray-400 mt-0.5">Use -1 for random seed</p>
                          </div>

                          {/* Reset button */}
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
                  </>
                )}
              </>
            )}
          </div>
        )}

        {/* ── TRANSFORMERS.JS TAB ── */}
        {activeTab === 'transformers' && !showWelcomeScreen && (
          <div className="space-y-5">
            {/* Simple/Advanced toggle */}
            <div className="flex items-center justify-between">
              <p className="text-gray-500 text-sm">
                Run models in-browser using WebGPU.
              </p>
              <div className="flex rounded-xl border border-gray-200 overflow-hidden bg-gray-100 p-0.5">
                <button
                  onClick={() => setTransformersViewMode('simple')}
                  className={`px-4 py-1.5 text-xs font-medium transition-all rounded-lg ${
                    transformersViewMode === 'simple'
                      ? 'bg-white text-purple-700 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Simple
                </button>
                <button
                  onClick={() => setTransformersViewMode('advanced')}
                  className={`px-4 py-1.5 text-xs font-medium transition-all rounded-lg ${
                    transformersViewMode === 'advanced'
                      ? 'bg-white text-purple-700 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Advanced
                </button>
              </div>
            </div>

            {/* Mobile warning */}
            {isMobileDevice && (
              <div className="flex items-center gap-2 text-amber-700 text-xs bg-amber-50 border border-amber-200 rounded-xl px-4 py-3">
                <AlertCircle size={16} className="text-amber-500 flex-shrink-0" />
                <span>Mobile devices have limited memory. Models may fail to load or run slowly.</span>
              </div>
            )}

            {transformersViewMode === 'simple' ? (
              /* Simple mode: Device toggle + model cards */
              <div className="space-y-3">
                {/* Device Toggle */}
                <div className="flex items-center justify-between p-4 bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-xl">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-gray-800">Processing</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${gemmaDevice === 'webgpu' ? 'bg-purple-200 text-purple-800' : 'bg-gray-200 text-gray-700'}`}>
                        {gemmaDevice === 'webgpu' ? 'WebGPU' : 'WASM'}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {gemmaDevice === 'webgpu'
                        ? 'GPU-accelerated for faster performance.'
                        : 'CPU mode.'}
                    </p>
                  </div>
                  <button
                    onClick={() => setGemmaDevice(gemmaDevice === 'webgpu' ? 'wasm' : 'webgpu')}
                    disabled={gemmaState.status === 'loading' || gemmaState.status === 'loaded'}
                    className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed ${
                      gemmaDevice === 'webgpu' ? 'bg-purple-600' : 'bg-gray-300'
                    }`}
                    title={gemmaState.status === 'loaded' ? 'Unload model to change this setting' : undefined}
                  >
                    <span
                      className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
                        gemmaDevice === 'webgpu' ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
                {gemmaState.status === 'loaded' && (
                  <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                    Unload the current model to change processing settings.
                  </p>
                )}

                {GEMMA_CARDS.map(({ modelId, label, size }) => {
                  const isThisModel = gemmaState.modelId === modelId;
                  const isLoaded = isThisModel && gemmaState.status === 'loaded';
                  const isLoading = isThisModel && gemmaState.status === 'loading';
                  const hasError = isThisModel && gemmaState.status === 'error';

                  return (
                    <div key={modelId} className={`border rounded-xl p-4 transition-all ${isLoaded ? 'border-purple-300 bg-purple-50/50' : 'border-gray-200 bg-white hover:border-gray-300'}`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${isLoaded ? 'bg-purple-200' : 'bg-gray-100'}`}>
                            <Cpu size={20} className={isLoaded ? 'text-purple-700' : 'text-gray-500'} />
                          </div>
                          <div>
                            <span className="font-semibold text-gray-900">{label}</span>
                            <div className="flex items-center gap-2 mt-0.5">
                              <span className="text-xs text-gray-500">{size}</span>
                              <span className="text-xs text-purple-600 font-medium bg-purple-100 px-1.5 py-0.5 rounded">Vision</span>
                            </div>
                          </div>
                        </div>
                        {isLoaded ? (
                          <button
                            onClick={() => GemmaModelManager.getInstance().unloadModel()}
                            className="group flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg font-medium transition-colors bg-purple-200 text-purple-800 hover:bg-red-100 hover:text-red-600"
                          >
                            <span className="group-hover:hidden flex items-center gap-1.5">
                              <CheckCircle size={14} /> Ready
                            </span>
                            <span className="hidden group-hover:flex items-center gap-1.5">
                              <X size={14} /> Unload
                            </span>
                          </button>
                        ) : isLoading ? (
                          <button
                            onClick={() => GemmaModelManager.getInstance().unloadModel()}
                            className="group flex items-center gap-1.5 px-4 py-2 text-sm bg-gray-200 text-gray-600 hover:bg-red-100 hover:text-red-600 rounded-lg font-medium transition-colors"
                          >
                            <span className="group-hover:hidden flex items-center gap-1.5"><Cpu size={14} className="animate-pulse" /> Loading…</span>
                            <span className="hidden group-hover:flex items-center gap-1.5"><StopCircle size={14} /> Cancel</span>
                          </button>
                        ) : (
                          <button
                            disabled={gemmaState.status === 'loading'}
                            onClick={() => GemmaModelManager.getInstance().loadModelWithSettings(modelId, gemmaDevice, gemmaDtype, gemmaTokenBudget)}
                            className="flex items-center gap-1.5 px-4 py-2 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-sm"
                          >
                            <Cpu size={14} /> Load
                          </button>
                        )}
                      </div>

                      {/* Loading progress */}
                      {isLoading && gemmaState.progress.length > 0 && (
                        <div className="space-y-2 mt-4">
                          {gemmaState.progress.map((item) => (
                            <div key={item.file}>
                              <div className="flex justify-between items-center text-xs mb-1.5">
                                <span className="text-gray-600 flex items-center gap-1.5 truncate max-w-[55%]">
                                  {item.status === 'done'
                                    ? <CheckCircle className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
                                    : <FileDown className="h-3.5 w-3.5 text-purple-400 flex-shrink-0" />
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
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full transition-all duration-300 ${item.status === 'done' ? 'bg-green-500' : 'bg-purple-600'}`}
                                  style={{ width: `${item.progress}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      {hasError && (
                        <p className="mt-3 text-xs text-red-600 bg-red-50 px-3 py-2 rounded-lg">{gemmaState.error}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              /* Advanced mode: Settings dropdowns + model cards */
              <>
                <div className="flex gap-3">
                  <div className="flex-1">
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
                  <div className="flex-1">
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
                  <div className="flex-1">
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
                      <option value={1120}>1120 (OCR/detail)</option>
                    </select>
                  </div>
                </div>

                {/* Custom ONNX model input */}
                <div className="border border-purple-200 bg-purple-50/50 rounded-xl p-4 space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Import Custom ONNX Model</label>
                    <p className="text-xs text-gray-500 mb-2">
                      Enter a Hugging Face model ID (e.g., <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">onnx-community/Florence-2-base-ft</code>)
                    </p>
                  </div>
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
                      onClick={() => {
                        if (customOnnxModelId.trim()) {
                          GemmaModelManager.getInstance().loadModelWithSettings(
                            customOnnxModelId.trim() as GemmaModelId,
                            gemmaDevice,
                            gemmaDtype,
                            gemmaTokenBudget
                          );
                        }
                      }}
                      disabled={!customOnnxModelId.trim() || gemmaState.status === 'loading'}
                      className="flex items-center gap-1.5 px-4 py-2.5 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
                    >
                      <Cpu size={16} />
                      Load
                    </button>
                  </div>
                  {/* Show loading/loaded state for custom model */}
                  {gemmaState.modelId && !GEMMA_CARDS.some(c => c.modelId === gemmaState.modelId) && (
                    <div className="border border-purple-200 bg-white rounded-lg p-3 mt-2">
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <span className="font-medium text-gray-900 truncate block">{gemmaState.modelId}</span>
                          {gemmaState.loadSettings && (
                            <span className="text-xs text-gray-500">
                              {gemmaState.loadSettings.device} · {gemmaState.loadSettings.dtype} · {gemmaState.loadSettings.imageTokenBudget}tok
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2 ml-2">
                          {gemmaState.status === 'loaded' ? (
                            <>
                              <span className="flex items-center gap-1 text-xs font-semibold text-green-700 bg-green-100 px-2 py-1 rounded-full">
                                <CheckCircle size={12} /> Ready
                              </span>
                              <button
                                onClick={() => GemmaModelManager.getInstance().unloadModel()}
                                className="p-1.5 text-gray-400 hover:text-red-600 rounded transition-colors"
                                title="Unload model"
                              >
                                <X size={16} />
                              </button>
                            </>
                          ) : gemmaState.status === 'loading' ? (
                            <button
                              onClick={() => GemmaModelManager.getInstance().unloadModel()}
                              className="group flex items-center gap-1.5 px-3 py-1.5 text-sm bg-gray-200 text-gray-600 hover:bg-red-100 hover:text-red-600 rounded-lg font-medium transition-colors"
                            >
                              <span className="group-hover:hidden flex items-center gap-1.5"><Cpu size={14} className="animate-pulse" /> Loading…</span>
                              <span className="hidden group-hover:flex items-center gap-1.5"><StopCircle size={14} /> Cancel</span>
                            </button>
                          ) : gemmaState.status === 'error' ? (
                            <span className="flex items-center gap-1 text-xs font-semibold text-red-700 bg-red-100 px-2 py-1 rounded-full">
                              <AlertTriangle size={12} /> Error
                            </span>
                          ) : null}
                        </div>
                      </div>
                      {/* Loading progress for custom model */}
                      {gemmaState.status === 'loading' && gemmaState.progress.length > 0 && (
                        <div className="space-y-2 mt-3">
                          {gemmaState.progress.map((item) => (
                            <div key={item.file}>
                              <div className="flex justify-between items-center text-xs mb-1">
                                <span className="text-gray-600 flex items-center gap-1 truncate max-w-[55%]">
                                  {item.status === 'done'
                                    ? <CheckCircle className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
                                    : <FileDown className="h-3.5 w-3.5 text-gray-400 flex-shrink-0" />
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
                      {gemmaState.status === 'error' && (
                        <p className="mt-2 text-xs text-red-600">{gemmaState.error}</p>
                      )}
                    </div>
                  )}
                </div>

                {/* Preset models divider */}
                <div className="flex items-center gap-3 pt-2">
                  <div className="flex-1 border-t border-gray-200" />
                  <span className="text-xs text-gray-400 font-medium">Preset Models</span>
                  <div className="flex-1 border-t border-gray-200" />
                </div>

                {GEMMA_CARDS.map(({ modelId, label, size }) => {
                  const isThisModel = gemmaState.modelId === modelId;
                  const isLoaded = isThisModel && gemmaState.status === 'loaded';
                  const isLoading = isThisModel && gemmaState.status === 'loading';
                  const hasError = isThisModel && gemmaState.status === 'error';

                  return (
                    <div key={modelId} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <span className="font-medium text-gray-900">{label}</span>
                          <span className="ml-2 text-xs text-gray-500">{size}</span>
                        </div>
                        {isLoaded ? (
                          <div className="flex items-center gap-2">
                            {gemmaState.loadSettings && (
                              <span className="text-xs text-gray-500">
                                {gemmaState.loadSettings.device} · {gemmaState.loadSettings.dtype} · {gemmaState.loadSettings.imageTokenBudget}tok
                              </span>
                            )}
                            <span className="flex items-center gap-1 text-xs font-semibold text-green-700 bg-green-100 px-2 py-1 rounded-full">
                              <CheckCircle size={12} /> Ready
                            </span>
                          </div>
                        ) : isLoading ? (
                          <button
                            onClick={() => GemmaModelManager.getInstance().unloadModel()}
                            className="group flex items-center gap-1.5 px-3 py-1.5 text-sm bg-gray-200 text-gray-600 hover:bg-red-100 hover:text-red-600 rounded-lg font-medium transition-colors"
                          >
                            <span className="group-hover:hidden flex items-center gap-1.5"><Cpu size={14} /> Loading…</span>
                            <span className="hidden group-hover:flex items-center gap-1.5"><StopCircle size={14} /> Cancel</span>
                          </button>
                        ) : (
                          <button
                            disabled={gemmaState.status === 'loading'}
                            onClick={() => GemmaModelManager.getInstance().loadModelWithSettings(modelId, gemmaDevice, gemmaDtype, gemmaTokenBudget)}
                            className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                          >
                            <Cpu size={14} /> Load
                          </button>
                        )}
                      </div>

                      {isLoading && gemmaState.progress.length > 0 && (
                        <div className="space-y-2 mt-3">
                          {gemmaState.progress.map((item) => (
                            <div key={item.file}>
                              <div className="flex justify-between items-center text-xs mb-1">
                                <span className="text-gray-600 flex items-center gap-1 truncate max-w-[55%]">
                                  {item.status === 'done'
                                    ? <CheckCircle className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
                                    : <FileDown className="h-3.5 w-3.5 text-gray-400 flex-shrink-0" />
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

                      {hasError && (
                        <p className="mt-2 text-xs text-red-600">{gemmaState.error}</p>
                      )}
                    </div>
                  );
                })}
              </>
            )}
          </div>
        )}

        {/* ── BENCHMARK TAB ── */}
        {activeTab === 'benchmark' && !showWelcomeScreen && (
          <BenchmarkPanel isVisible={activeTab === 'benchmark'} />
        )}
      </div>
    </Modal>
  );
};

export default TerminalModal;
