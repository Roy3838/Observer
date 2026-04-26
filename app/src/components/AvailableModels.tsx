import React, { useState, useEffect } from 'react';
import { fetchModels as fetchAllModels, Model } from '@utils/inferenceServer';
import {
  Cpu, RefreshCw, Eye, CheckCircle, X, StopCircle, Sparkles,
  AlertTriangle, Trash2, Settings2, BarChart3, FileDown, Cpu as CpuIcon, Cloud,
} from 'lucide-react';
import { BROWSER_LOCAL_SENTINEL, LLAMA_CPP_LOCAL_SENTINEL } from '@utils/inferenceServer';
import { Logger } from '@utils/logging';
import ModelHub from '@components/ModelHub';
import BenchmarkPanel from '@components/BenchmarkPanel';
import Modal from '@components/EditAgent/Modal';
import LlamaCppSamplerPanel from '@components/ModelCard/LlamaCppSamplerPanel';
import RemoteInferenceParamsPanel from '@components/ModelCard/RemoteInferenceParamsPanel';
import { platformFetch, isTauri } from '@utils/platform';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import type { CustomServer } from '@utils/inferenceServer';
import {
  NativeModelState,
  GemmaModelState,
  GemmaModelId,
  GgufFileInfo,
  SamplerParams,
  DEFAULT_SAMPLER_PARAMS,
} from '@utils/localLlm/types';

type QuotaInfo = {
  used: number;
  remaining: number;
  limit: number;
  tier: string;
} | null;

interface AvailableModelsProps {
  isProUser?: boolean;
  // Ob-Server + server props threaded to ModelHub
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

const formatBytes = (bytes: number, decimals = 2) => {
  if (!+bytes) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};

const AvailableModels: React.FC<AvailableModelsProps> = ({
  isProUser = false,
  isUsingObServer,
  handleToggleObServer,
  showLoginMessage,
  isAuthenticated,
  quotaInfo,
  renderQuotaStatus,
  localServerOnline,
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

  // ── Remote model list
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [ollamaServers, setOllamaServers] = useState<string[]>([]);

  // ── Local model state (llama.cpp)
  const [nativeState, setNativeState] = useState<NativeModelState>(NativeLlmManager.getInstance().getState());
  const [ggufFiles, setGgufFiles] = useState<GgufFileInfo[]>([]);
  const [mmprojAssignments, setMmprojAssignments] = useState<Record<string, string>>({});
  const [samplerParams, setSamplerParams] = useState<SamplerParams>({ ...DEFAULT_SAMPLER_PARAMS });

  // ── Local model state (Transformers.js)
  const [gemmaState, setGemmaState] = useState<GemmaModelState>(GemmaModelManager.getInstance().getState());

  // ── UI state
  const [expandedSettings, setExpandedSettings] = useState<string | null>(null);
  const [showBenchmark, setShowBenchmark] = useState(false);
  const [showModelHub, setShowModelHub] = useState(false);

  // ── Native subscriptions
  useEffect(() => {
    if (!isTauriApp) return;
    const unsub = NativeLlmManager.getInstance().onStateChange(state => {
      setNativeState(state);
      NativeLlmManager.getInstance().listGgufFiles().then(setGgufFiles);
      setMmprojAssignments(NativeLlmManager.getInstance().getMmprojAssignments());
    });
    setNativeState(NativeLlmManager.getInstance().getState());
    NativeLlmManager.getInstance().listGgufFiles().then(setGgufFiles);
    setMmprojAssignments(NativeLlmManager.getInstance().getMmprojAssignments());
    return unsub;
  }, [isTauriApp]);

  // ── Sampler params sync
  useEffect(() => {
    if (!isTauriApp) return;
    if (nativeState.status === 'loading' || nativeState.status === 'unloaded') {
      setSamplerParams({ ...DEFAULT_SAMPLER_PARAMS });
    } else if (nativeState.status === 'loaded') {
      NativeLlmManager.getInstance().getDebugInfo().then(info => {
        if (info.engine.samplerParams) setSamplerParams(info.engine.samplerParams);
      }).catch(() => {});
    }
  }, [isTauriApp, nativeState.status]);

  // ── Gemma subscription
  useEffect(() => {
    const unsub = GemmaModelManager.getInstance().onStateChange(setGemmaState);
    setGemmaState(GemmaModelManager.getInstance().getState());
    return unsub;
  }, []);

  // ── Remote model fetching
  const checkOllamaSupport = async (address: string): Promise<boolean> => {
    try {
      const response = await platformFetch(`${address}/api/tags`, { method: 'GET', headers: { 'Content-Type': 'application/json' } });
      return response.ok;
    } catch { return false; }
  };

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await fetchAllModels();
      if (response.error) throw new Error(response.error);
      setModels(response.models);
      Logger.info('MODELS', `Loaded ${response.models.length} models`);
    } catch (err) {
      Logger.error('MODELS', `Failed to fetch models: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const detectOllamaServers = async () => {
    const { getInferenceAddresses } = await import('@utils/inferenceServer');
    const addresses = getInferenceAddresses();
    const checks = await Promise.all(addresses.map(async addr => ({ addr, ok: await checkOllamaSupport(addr) })));
    setOllamaServers(checks.filter(c => c.ok).map(c => c.addr));
  };

  useEffect(() => {
    fetchModels();
    detectOllamaServers();
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchModels();
    detectOllamaServers();
  };

  // ── Sampler handlers
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

  const isAnyNativeBusy = nativeState.status === 'loading' || nativeState.status === 'unloading' || nativeState.status === 'downloading';
  const ggufModels = ggufFiles.filter(f => !f.filename.toLowerCase().includes('mmproj'));
  const ggufProjectors = ggufFiles.filter(f => f.filename.toLowerCase().includes('mmproj'));

  // Determine if a model has configurable settings
  const hasSettings = (model: Model) => {
    if (model.server === LLAMA_CPP_LOCAL_SENTINEL) return nativeState.status === 'loaded';
    if (model.server === BROWSER_LOCAL_SENTINEL) return false;
    if (model.server.includes('api.observer-ai.com')) return false;
    return true; // custom/Ollama server models get InferenceParams
  };

  const toggleSettings = (modelName: string) => {
    setExpandedSettings(prev => prev === modelName ? null : modelName);
  };

  if (loading && !refreshing) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <div className="animate-spin mb-4"><Cpu className="h-8 w-8 text-blue-500" /></div>
        <p className="text-gray-600">Loading available models...</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold text-gray-800">Models</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowBenchmark(b => !b)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              showBenchmark ? 'bg-orange-100 text-orange-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
            title="Performance Benchmark"
          >
            <BarChart3 className="h-4 w-4" />
            <span className="hidden sm:inline">Benchmark</span>
          </button>
          <button
            onClick={() => setShowModelHub(true)}
            className="px-3 py-2 rounded-md bg-green-50 text-green-700 hover:bg-green-100 text-sm font-medium"
          >
            Add Model
          </button>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-md text-sm ${
              refreshing ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 'bg-blue-50 text-blue-600 hover:bg-blue-100'
            }`}
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            <span className="hidden sm:inline">{refreshing ? 'Refreshing...' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {/* llama.cpp local model cards */}
      {isTauriApp && ggufModels.length > 0 && (
        <div className="space-y-2 mb-4">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Local — llama.cpp</p>
          {ggufModels.map(file => {
            const modelId = file.filename.replace(/\.gguf$/i, '');
            const isThisModel = nativeState.modelId === modelId;
            const isLoaded = isThisModel && nativeState.status === 'loaded';
            const isLoading = isThisModel && nativeState.status === 'loading';
            const isUnloading = isThisModel && nativeState.status === 'unloading';
            const assignedMmproj = mmprojAssignments[file.filename] ?? null;
            const isMultimodal = isThisModel ? NativeLlmManager.getInstance().isMultimodal() : !!assignedMmproj;
            const settingsOpen = expandedSettings === modelId;

            return (
              <div key={file.filename} className={`border rounded-xl transition-all ${
                isLoaded ? 'border-gray-400 bg-gray-50' : 'border-gray-200 bg-white'
              }`}>
                <div className="flex items-center justify-between gap-3 p-3">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${
                      isLoaded ? 'bg-gray-200' : 'bg-gray-100'
                    }`}>
                      <CpuIcon size={18} className={isLoaded ? 'text-gray-800' : 'text-gray-500'} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="font-medium text-gray-900 truncate">{modelId}</span>
                        <span className="text-[10px] font-semibold text-green-700 bg-green-100 px-1.5 py-0.5 rounded">llama.cpp</span>
                        {isMultimodal && <span className="text-[10px] font-semibold text-purple-700 bg-purple-100 px-1.5 py-0.5 rounded">Vision</span>}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <p className="text-xs text-gray-500">{formatBytes(file.sizeBytes)}</p>
                        {ggufProjectors.length > 0 && (
                          <select
                            value={assignedMmproj ?? ''}
                            onChange={e => {
                              NativeLlmManager.getInstance().setMmprojAssignment(file.filename, e.target.value || null);
                              setMmprojAssignments(NativeLlmManager.getInstance().getMmprojAssignments());
                            }}
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
                    {/* Settings gear — only when loaded */}
                    {isLoaded && (
                      <button
                        onClick={() => toggleSettings(modelId)}
                        className={`p-1.5 rounded transition-colors ${settingsOpen ? 'text-gray-800 bg-gray-200' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'}`}
                        title="Generation settings"
                      >
                        <Settings2 size={14} />
                      </button>
                    )}
                    {isUnloading ? (
                      <span className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-200 text-gray-600 rounded-lg font-medium">
                        <CpuIcon size={12} className="animate-pulse" /> Unloading
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
                        <CpuIcon size={12} className="animate-pulse" /> Loading
                      </button>
                    ) : (
                      <>
                        <button
                          disabled={isAnyNativeBusy}
                          onClick={() => NativeLlmManager.getInstance().loadModel(file.filename)}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 text-white rounded-lg hover:bg-gray-900 disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-sm"
                        >
                          <CpuIcon size={12} /> Load
                        </button>
                        <button
                          disabled={isAnyNativeBusy}
                          onClick={() => NativeLlmManager.getInstance().deleteModel(file.filename)}
                          className="p-1.5 text-gray-400 hover:text-red-600 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                          title="Delete"
                        >
                          <Trash2 size={14} />
                        </button>
                      </>
                    )}
                  </div>
                </div>
                {settingsOpen && isLoaded && (
                  <LlamaCppSamplerPanel
                    nativeStatus={nativeState.status}
                    samplerParams={samplerParams}
                    onParamChange={handleSamplerParamChange}
                    onReset={handleResetSamplerParams}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Transformers.js local model cards */}
      {gemmaState.modelId && (
        <div className="space-y-2 mb-4">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Local — Transformers.js</p>
          {GemmaModelManager.getInstance().listLocalModels().map(model => {
            const isThisModel = gemmaState.modelId === model.id;
            const status = isThisModel ? gemmaState.status : model.status;
            const isLoaded = status === 'loaded';
            const isLoading = status === 'loading';
            const isError = status === 'error';
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
                      <CpuIcon size={18} className={isLoaded ? 'text-gray-800' : 'text-gray-500'} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="font-medium text-gray-900 truncate">{model.name}</span>
                        <span className="text-[10px] font-semibold text-yellow-800 bg-yellow-300 px-1.5 py-0.5 rounded">Transformers.js</span>
                      </div>
                      {loadSettings && (
                        <p className="text-xs text-gray-500 mt-0.5">
                          {loadSettings.device} · {loadSettings.dtype} · {loadSettings.imageTokenBudget} tokens
                        </p>
                      )}
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
                        <span className="group-hover:hidden flex items-center gap-1.5"><CpuIcon size={12} className="animate-pulse" /> Loading</span>
                        <span className="hidden group-hover:flex items-center gap-1.5"><StopCircle size={12} /> Cancel</span>
                      </button>
                    ) : isError ? (
                      <span className="flex items-center gap-1 text-xs font-semibold text-red-700 bg-red-100 px-2 py-1 rounded-full">
                        <AlertTriangle size={12} /> Error
                      </span>
                    ) : (
                      <>
                        <button
                          onClick={() => GemmaModelManager.getInstance().loadModel(model.id as GemmaModelId)}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-700 text-white rounded-lg hover:bg-gray-900 font-medium shadow-sm"
                        >
                          <Sparkles size={12} /> Load
                        </button>
                        <button
                          onClick={() => GemmaModelManager.getInstance().deleteModel(model.id as GemmaModelId)}
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
                    {gemmaState.progress.map(item => (
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
                            {item.status === 'done' ? 'Done'
                              : item.total > 0 ? `${formatBytes(item.loaded)} / ${formatBytes(item.total)}`
                              : `${Math.round(item.progress)}%`}
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

      {/* Remote model cards */}
      {models.length > 0 && (
        <div className="space-y-2 mb-4">
          {models.some(m => !m.server.includes('api.observer-ai.com') && m.server !== LLAMA_CPP_LOCAL_SENTINEL && m.server !== BROWSER_LOCAL_SENTINEL) && (
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Remote Servers</p>
          )}
          {models
            .filter(m => m.server !== LLAMA_CPP_LOCAL_SENTINEL && m.server !== BROWSER_LOCAL_SENTINEL)
            .map(model => {
              const isObServer = model.server.includes('api.observer-ai.com');
              const settingsOpen = expandedSettings === model.name;
              const canConfigure = hasSettings(model);

              return (
                <div key={model.name} className={`border rounded-xl transition-all ${
                  model.pro && !isProUser ? 'opacity-60' : ''
                } ${settingsOpen ? 'border-blue-200' : 'border-gray-200 bg-white hover:border-gray-300'}`}>
                  <div className="flex items-center gap-3 p-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 bg-gray-100">
                      <CpuIcon size={18} className="text-gray-500" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="font-medium text-gray-900 truncate">{model.name}</span>
                        {model.pro && !isProUser && (
                          <span className="text-[10px] font-semibold text-purple-700 bg-purple-100 px-1.5 py-0.5 rounded">PRO</span>
                        )}
                        {model.multimodal && (
                          <span className="text-[10px] font-semibold text-purple-700 bg-purple-100 px-1.5 py-0.5 rounded flex items-center gap-0.5">
                            <Eye size={9} />
                          </span>
                        )}
                        {isObServer ? (
                          <span className="text-[10px] font-semibold text-indigo-700 bg-indigo-100 px-1.5 py-0.5 rounded flex items-center gap-0.5">
                            <Cloud size={9} />
                          </span>
                        ) : (
                          <span className="text-[10px] font-semibold text-gray-600 bg-gray-100 px-1.5 py-0.5 rounded">Server</span>
                        )}
                      </div>
                      {model.parameterSize && model.parameterSize !== 'N/A' && (
                        <p className="text-xs text-gray-500 mt-0.5">{model.parameterSize}</p>
                      )}
                    </div>
                    {canConfigure && (
                      <button
                        onClick={() => toggleSettings(model.name)}
                        className={`p-1.5 rounded transition-colors flex-shrink-0 ${
                          settingsOpen ? 'text-blue-700 bg-blue-100' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                        }`}
                        title="Inference settings"
                      >
                        <Settings2 size={14} />
                      </button>
                    )}
                  </div>
                  {settingsOpen && canConfigure && (
                    <RemoteInferenceParamsPanel modelName={model.name} />
                  )}
                </div>
              );
            })}
        </div>
      )}

      {models.length === 0 && ggufModels.length === 0 && !gemmaState.modelId && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 mb-6">
          <p className="text-yellow-700">No models found.</p>
          <p className="text-sm text-yellow-600 mt-1">
            Use "Add Model" to download a local model or connect to a server.
          </p>
        </div>
      )}

      {/* Benchmark modal */}
      <Modal open={showBenchmark} onClose={() => setShowBenchmark(false)} className="w-full max-w-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b border-orange-200 bg-orange-50">
          <span className="text-sm font-semibold text-orange-800 flex items-center gap-2">
            <BarChart3 size={15} /> Performance Benchmark
          </span>
          <button
            onClick={() => setShowBenchmark(false)}
            className="p-1 text-orange-400 hover:text-orange-700 rounded transition-colors"
          >
            <X size={16} />
          </button>
        </div>
        <div className="p-4 overflow-y-auto" style={{ maxHeight: 'calc(88vh - env(safe-area-inset-top) - env(safe-area-inset-bottom))' }}>
          <BenchmarkPanel isVisible={true} />
        </div>
      </Modal>

      <ModelHub
        isOpen={showModelHub}
        onClose={() => setShowModelHub(false)}
        onPullComplete={handleRefresh}
        ollamaServers={ollamaServers}
        isUsingObServer={isUsingObServer}
        handleToggleObServer={handleToggleObServer}
        showLoginMessage={showLoginMessage}
        isAuthenticated={isAuthenticated}
        quotaInfo={quotaInfo}
        renderQuotaStatus={renderQuotaStatus}
        localServerOnline={localServerOnline}
        checkLocalServer={checkLocalServer}
        customServers={customServers}
        onAddCustomServer={onAddCustomServer}
        onRemoveCustomServer={onRemoveCustomServer}
        onToggleCustomServer={onToggleCustomServer}
        onCheckCustomServer={onCheckCustomServer}
        appInferenceUrl={appInferenceUrl}
        onSetAppInferenceUrl={onSetAppInferenceUrl}
      />
    </div>
  );
};

export default AvailableModels;
