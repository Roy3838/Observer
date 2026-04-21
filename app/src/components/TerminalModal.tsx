// components/TerminalModal.tsx

import React, { useState, useEffect } from 'react';
import Modal from '@components/EditAgent/Modal';
import { Download, CheckCircle, AlertTriangle, X, StopCircle, FileDown, Cpu, Trash2 } from 'lucide-react';
import pullModelManager, { PullState } from '@utils/pullModelManager';
import { platformFetch, isIOS } from '@utils/platform';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import {
  GemmaModelId,
  GemmaModelState,
  GemmaDevice,
  GemmaDtype,
  GemmaImageTokenBudget,
  NativeModelInfo,
  NativeModelState,
} from '@utils/localLlm/types';

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
  const [activeTab, setActiveTab] = useState<'ollama' | 'ondevice'>(hasOllama ? 'ollama' : 'ondevice');

  const [gemmaState, setGemmaState] = useState<GemmaModelState>(GemmaModelManager.getInstance().getState());
  // Default values match GemmaModelManager.DEFAULT_SETTINGS
  const [gemmaDevice, setGemmaDevice] = useState<GemmaDevice>('webgpu');
  const [gemmaDtype, setGemmaDtype] = useState<GemmaDtype>('q4');
  const [gemmaTokenBudget, setGemmaTokenBudget] = useState<GemmaImageTokenBudget>(70);

  // iOS native LLM state
  const isIOSPlatform = isIOS();
  const [nativeState, setNativeState] = useState<NativeModelState>(NativeLlmManager.getInstance().getState());
  const [nativeModels, setNativeModels] = useState<NativeModelInfo[]>([]);
  const [ggufUrl, setGgufUrl] = useState('');

  useEffect(() => {
    if (!isOpen) return;
    setActiveTab(hasOllama ? 'ollama' : 'ondevice');
  }, [isOpen, hasOllama]);

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

  // iOS: Subscribe to native LLM state changes
  useEffect(() => {
    if (!isOpen || !isIOSPlatform) return;
    const unsubscribe = NativeLlmManager.getInstance().onStateChange(setNativeState);
    setNativeState(NativeLlmManager.getInstance().getState());
    return unsubscribe;
  }, [isOpen, isIOSPlatform]);

  // iOS: Fetch available native models
  useEffect(() => {
    if (!isOpen || !isIOSPlatform) return;
    const fetchNativeModels = async () => {
      const models = await NativeLlmManager.getInstance().listModels();
      setNativeModels(models);
    };
    fetchNativeModels();
  }, [isOpen, isIOSPlatform, nativeState.status]);

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

  return (
    <Modal open={isOpen} onClose={handleDone} className="w-full max-w-xl">
      <div className="p-6 max-h-[80vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-800">Add Model</h2>
          <button onClick={handleDone} className="text-gray-400 hover:text-gray-600 rounded-full p-1">
            <X size={20} />
          </button>
        </div>

        {/* Tab bar — only show when both options are relevant */}
        {hasOllama && !showWelcomeScreen && (
          <div className="flex border-b border-gray-200 mb-5">
            <button
              onClick={() => setActiveTab('ollama')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'ollama'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Ollama
            </button>
            <button
              onClick={() => setActiveTab('ondevice')}
              className={`flex items-center gap-1.5 px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'ondevice'
                  ? 'border-green-600 text-green-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <Cpu size={14} />
              {isIOSPlatform ? 'On-Device (Metal)' : 'On-Device (WebGPU)'}
            </button>
          </div>
        )}

        {/* ── OLLAMA TAB ── */}
        {(activeTab === 'ollama' || !hasOllama) && !showWelcomeScreen && activeTab !== 'ondevice' && (
          <>
            {showWelcomeScreen ? (
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <Download className="h-7 w-7 text-green-500 flex-shrink-0" />
                  <h2 className="text-xl sm:text-2xl font-semibold">Let's Get Your First Model</h2>
                </div>
                <p className="text-gray-700 mb-6">
                  Your local server is running, but it looks like you don't have any AI models installed yet.
                </p>
                <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
                  <h3 className="font-semibold text-green-800 mb-2 text-lg">Recommended Model: Gemma3 4B</h3>
                  <button
                    onClick={handlePullModelClick}
                    className="w-full sm:w-auto px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium text-base"
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
            ) : !isFinished && !isPulling ? (
              <>
                <p className="text-gray-600 mb-5">
                  Enter a model name from the Ollama library (e.g., <code className="bg-gray-100 px-1 rounded text-sm">gemma3:4b</code>).
                </p>
                {availableServers.length > 1 && (
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-2">Select Ollama Server:</label>
                    <select
                      value={selectedServer}
                      onChange={(e) => setSelectedServer(e.target.value)}
                      className="w-full p-2.5 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 bg-white"
                    >
                      {availableServers.map((server) => (
                        <option key={server} value={server}>{server}</option>
                      ))}
                    </select>
                  </div>
                )}
                <div className="flex flex-col sm:flex-row gap-2">
                  <input
                    type="text" list="ollama-model-suggestions" value={modelToPull}
                    onChange={(e) => setModelToPull(e.target.value)}
                    placeholder="Enter model name..."
                    className="flex-grow p-2.5 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                  />
                  <datalist id="ollama-model-suggestions">
                    {suggestedModels.map(model => <option key={model} value={model} />)}
                  </datalist>
                  <button
                    onClick={handleStartPull}
                    className="w-full sm:w-auto flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
                  >
                    <Download size={18} />
                    <span>Start Download</span>
                  </button>
                </div>
              </>
            ) : null}

            {isPulling && (
              <div className="mt-6 space-y-3">
                <div className="flex justify-between items-baseline">
                  <p className="text-sm font-medium text-gray-700">{statusText}</p>
                  <p className="text-sm font-semibold text-blue-600">{progress}%</p>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div className="bg-blue-500 h-2.5 rounded-full transition-all duration-150" style={{ width: `${progress}%` }} />
                </div>
                <div className="flex justify-between items-center">
                  {totalBytes > 0 ? (
                    <p className="text-xs text-gray-500 font-mono">{formatBytes(completedBytes)} / {formatBytes(totalBytes)}</p>
                  ) : <div />}
                  <button onClick={handleCancelPull} className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-red-600 bg-red-50 hover:bg-red-100 rounded-md font-semibold">
                    <StopCircle size={14} /> Cancel
                  </button>
                </div>
              </div>
            )}

            {status === 'success' && (
              <div className="mt-4 p-4 bg-green-50 border border-green-200 text-green-800 rounded-md flex flex-col items-center gap-3 text-center">
                <CheckCircle size={32} className="text-green-500" />
                <div>
                  <h3 className="font-semibold">Download Complete!</h3>
                  <p className="text-sm">{statusText}</p>
                </div>
              </div>
            )}

            {status === 'error' && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 text-red-800 rounded-md flex flex-col items-center gap-3 text-center">
                <AlertTriangle size={32} className="text-red-500" />
                <div>
                  <h3 className="font-semibold">An Error Occurred</h3>
                  <p className="text-sm">{errorText}</p>
                </div>
              </div>
            )}

            {isFinished && (
              <div className="mt-6 flex justify-end">
                <button onClick={handleDone} className="px-5 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-800 font-medium">
                  Done
                </button>
              </div>
            )}
          </>
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

        {/* ── ON-DEVICE TAB ── */}
        {activeTab === 'ondevice' && !showWelcomeScreen && (
          <div className="space-y-4">
            {isIOSPlatform ? (
              /* iOS: Native GGUF models with llama.cpp + Metal */
              <>
                <p className="text-gray-600 text-sm">
                  Download any GGUF model from HuggingFace. Models run natively with Metal GPU acceleration.
                </p>
                <p className="text-green-600 text-xs bg-green-50 border border-green-200 rounded-md px-3 py-2">
                  Paste a HuggingFace GGUF URL (e.g., https://huggingface.co/.../model.gguf)
                </p>

                {/* URL Input for downloading new models */}
                <div className="flex flex-col gap-2">
                  <input
                    type="text"
                    value={ggufUrl}
                    onChange={(e) => setGgufUrl(e.target.value)}
                    placeholder="https://huggingface.co/.../resolve/main/model.gguf"
                    disabled={nativeState.status === 'downloading'}
                    className="flex-grow p-2.5 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 disabled:opacity-50 text-sm"
                  />
                  <button
                    onClick={async () => {
                      if (ggufUrl.trim()) {
                        try {
                          await NativeLlmManager.getInstance().downloadModel(ggufUrl.trim());
                          setGgufUrl('');
                        } catch (e) {
                          // Error is shown via nativeState.error
                        }
                      }
                    }}
                    disabled={!ggufUrl.trim() || nativeState.status === 'downloading'}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                  >
                    <Download size={18} />
                    <span>Download GGUF Model</span>
                  </button>
                </div>

                {/* Download Progress */}
                {nativeState.status === 'downloading' && (
                  <div className="border border-blue-200 bg-blue-50 rounded-lg p-4">
                    <div className="flex justify-between items-center text-xs mb-1">
                      <span className="text-gray-700 font-medium">Downloading {nativeState.modelId}...</span>
                      <span className="font-medium text-blue-600">{nativeState.downloadProgress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="h-2 rounded-full transition-all duration-300 bg-blue-600"
                        style={{ width: `${nativeState.downloadProgress}%` }}
                      />
                    </div>
                    {nativeState.totalBytes > 0 && (
                      <p className="text-xs text-gray-500 mt-1">
                        {formatBytes(nativeState.downloadedBytes)} / {formatBytes(nativeState.totalBytes)}
                      </p>
                    )}
                  </div>
                )}

                {/* Error State */}
                {nativeState.status === 'error' && (
                  <div className="border border-red-200 bg-red-50 rounded-lg p-3">
                    <p className="text-xs text-red-600">{nativeState.error}</p>
                  </div>
                )}

                {/* Downloaded Models List */}
                {nativeModels.length > 0 && (
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-gray-700">Downloaded Models</h3>
                    {nativeModels.map((model) => {
                      const isThisModel = nativeState.modelId === model.id;
                      const isLoaded = isThisModel && nativeState.status === 'loaded';
                      const isLoading = isThisModel && nativeState.status === 'loading';

                      return (
                        <div key={model.filename} className="border border-gray-200 rounded-lg p-3">
                          <div className="flex items-center justify-between">
                            <div className="flex-1 min-w-0">
                              <span className="font-medium text-gray-900 truncate block">{model.name}</span>
                              <span className="text-xs text-gray-500">{formatBytes(model.sizeBytes)}</span>
                            </div>
                            <div className="flex items-center gap-2 ml-2">
                              {isLoaded ? (
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
                                  <Cpu size={14} /> Loading…
                                </button>
                              ) : (
                                <>
                                  <button
                                    disabled={nativeState.status === 'loading' || nativeState.status === 'downloading'}
                                    onClick={() => NativeLlmManager.getInstance().loadModel(model.filename)}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                                  >
                                    <Cpu size={14} /> Load
                                  </button>
                                  <button
                                    onClick={() => NativeLlmManager.getInstance().deleteModel(model.filename)}
                                    className="p-1.5 text-gray-400 hover:text-red-600 rounded transition-colors"
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

                {nativeModels.length === 0 && nativeState.status !== 'downloading' && (
                  <p className="text-gray-500 text-sm text-center py-4">
                    No models downloaded yet. Paste a HuggingFace GGUF URL above to get started.
                  </p>
                )}
              </>
            ) : (
              /* Web/Desktop: ONNX models with WebGPU/WASM */
              <>
                <p className="text-gray-600 text-sm">
                  These models run in-browser. Weights are downloaded once and cached locally.
                </p>
                <p className="text-amber-600 text-xs bg-amber-50 border border-amber-200 rounded-md px-3 py-2">
                  Desktop only. Mobile browsers have memory limits that prevent loading these models.
                </p>

                <div className="flex gap-3 mt-3">
                  <div className="flex-1">
                    <label className="block text-xs font-medium text-gray-600 mb-1">Device</label>
                    <select
                      value={gemmaDevice}
                      onChange={e => setGemmaDevice(e.target.value as GemmaDevice)}
                      disabled={gemmaState.status === 'loading'}
                      className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 disabled:opacity-50"
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
                      className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                    >
                      <option value="q4f16">q4f16 (4-bit + f16)</option>
                      <option value="q4">q4 (4-bit)</option>
                      <option value="q8">q8 (8-bit INT8)</option>
                      <option value="quantized">quantized (INT8 alt)</option>
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
                      className="w-full p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                    >
                      <option value={70}>70 (fastest)</option>
                      <option value={140}>140</option>
                      <option value={280}>280 (default)</option>
                      <option value={560}>560</option>
                      <option value={1120}>1120 (OCR/detail)</option>
                    </select>
                  </div>
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
                            className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
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
                                  className={`h-1.5 rounded-full transition-all duration-300 ${item.status === 'done' ? 'bg-green-500' : 'bg-blue-600'}`}
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
      </div>
    </Modal>
  );
};

export default TerminalModal;
