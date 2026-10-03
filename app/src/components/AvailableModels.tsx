import React, { useState, useEffect } from 'react';
import { fetchModels as fetchAllModels, Model } from '@utils/inferenceServer';
import {
  Cpu, RefreshCw, X, Trash2, Settings2, BarChart3,
  Cloud, MinusCircle, Server, Check, FileDown, Download, ChevronDown, StopCircle, Plus, Zap, Play,
} from 'lucide-react';
import { BROWSER_LOCAL_SENTINEL, LLAMA_CPP_LOCAL_SENTINEL, SKIP_MODEL_SENTINEL } from '@utils/inferenceServer';
import { Logger, LogEntry, LogLevel } from '@utils/logging';
import BenchmarkPanel from '@components/BenchmarkPanel';
import Modal from '@components/EditAgent/Modal';
import LlamaCppSamplerPanel from '@components/ModelCard/LlamaCppSamplerPanel';
import RemoteInferenceParamsPanel from '@components/ModelCard/RemoteInferenceParamsPanel';
import { ModelRow, RowButtonPrimary, RowButtonGhost, RowIconButton } from '@components/ModelCard/ModelRow';
import pullModelManager, { PullState } from '@utils/pullModelManager';
import { platformFetch, isTauri, isWeb } from '@utils/platform';
import { invoke } from '@tauri-apps/api/core';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import type { CustomServer } from '@utils/inferenceServer';
import {
  NativeModelState,
  GemmaModelState,
  GemmaModelId,
  GemmaDevice,
  GemmaDtype,
  GemmaImageTokenBudget,
  GgufFileInfo,
  NativeLocalModel,
  LocalModelEntry,
  SamplerParams,
  DEFAULT_SAMPLER_PARAMS,
  ContextParams,
  DEFAULT_CONTEXT_PARAMS,
} from '@utils/localLlm/types';
import { MODEL_CATALOG, MODEL_PRESETS, EXTENDED_PRESETS, type ModelPreset } from '@utils/modelPresets';
import { remaining as remainingOf, type QuotaInfo as QuotaInfoBase, type QuotaBlock } from '@/types/quota';

type QuotaInfo = QuotaInfoBase | null;

const SUGGESTED_OLLAMA_MODELS = [
  'gemma3:4b', 'gemma3:12b', 'gemma3:27b', 'gemma3:27b-it-qat',
  'qwen2.5vl:3b', 'qwen2.5vl:7b', 'llava:7b', 'llava:13b',
];

// Matched against LocalOnboardingTutorial's `download-gemma` step — it spotlights
// whichever engine's default preset applies (native on desktop, ONNX on web) and
// polls `[data-tutorial-gemma-state]` for 'loaded' or 'installed' to advance.
const TUTORIAL_MODEL_NAME = 'Gemma 4 E2B';

interface AvailableModelsProps {
  isProUser?: boolean;
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

interface FileProgress { file: string; loaded: number; total: number; progress: number; done: boolean }

// Per-file download breakdown (name, bytes loaded / total, thin bar) so it's clear
// exactly what is being fetched — used for both Transformers.js and llama.cpp.
const FileProgressList: React.FC<{ items: FileProgress[] }> = ({ items }) => {
  if (items.length === 0) return null;
  return (
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
                : item.total > 0 ? `${formatBytes(item.loaded, 1)} / ${formatBytes(item.total, 1)}`
                : `${Math.round(item.progress)}%`}
            </span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-1">
            <div
              className={`h-1 rounded-full transition-all duration-300 bg-gray-800`}
              style={{ width: `${Math.max(0, Math.min(100, item.progress))}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
};

const gemmaFileProgress = (state: GemmaModelState): FileProgress[] =>
  state.progress.map(p => ({ file: p.file, loaded: p.loaded, total: p.total, progress: p.progress, done: p.status === 'done' }));

const QuotaBar: React.FC<{ label: string; block: QuotaBlock }> = ({ label, block }) => {
  const rem = remainingOf(block);
  const pct = block.limit > 0 ? Math.min(100, (block.used / block.limit) * 100) : 0;
  const barColor = rem <= block.limit * 0.1 ? 'bg-red-500' : rem <= block.limit * 0.3 ? 'bg-orange-400' : 'bg-indigo-500';
  return (
    <div className="flex-1 min-w-0">
      <div className="flex items-center justify-between text-[11px] text-gray-400 mb-0.5">
        <span>{label}</span>
        <span className="tabular-nums">{rem}/{block.limit}</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-1">
        <div className={`h-1 rounded-full transition-all ${barColor}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
};

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => {
  if (React.Children.toArray(children).every(c => !c)) return null;
  return (
    <section className="mb-6">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-1">{title}</h3>
      <div className="divide-y divide-gray-100">{children}</div>
    </section>
  );
};

type HfFile = { path: string; size: number };
// Quant list for a GGUF repo, read live from the HF API so new quants show up in the dropdown without an app update.
const fetchGgufFiles = async (repo: string): Promise<HfFile[]> => {
  const res = await platformFetch(`https://huggingface.co/api/models/${repo}/tree/main`, { method: 'GET' });
  if (!res.ok) return [];
  const items = await res.json() as { type: string; path: string; size?: number; lfs?: { size: number } }[];
  return items
    .filter(f => f.type === 'file' && /\.gguf$/i.test(f.path))
    .map(f => ({ path: f.path, size: f.lfs?.size ?? f.size ?? 0 }));
};

// Quant name from a GGUF filename (`gemma-4-E2B-it-UD-Q3_K_XL.gguf` -> `UD-Q3_K_XL`). Files without one
// (e.g. `mtp-gemma-4-E2B-it.gguf`, a speculative-decoding draft head) aren't standalone models: null.
type GgufOption = { file: string; size: number };
const QUANT_RE = /(?:^|[-_.])((?:UD-)?(?:IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))$/i;
const quantLabelOf = (file: string): string | null =>
  file.replace(/\.gguf$/i, '').match(QUANT_RE)?.[1] ?? null;

async function deleteNativeModelCascade(model: NativeLocalModel, allModels: NativeLocalModel[]) {
  const manager = NativeLlmManager.getInstance();
  await manager.deleteModel(model.id);
  if (model.projectorFilename) {
    const stillUsed = allModels.some(m => m.id !== model.id && m.projectorFilename === model.projectorFilename);
    if (!stillUsed) await manager.deleteModel(model.projectorFilename);
  }
}

type AdvancedTab = 'hardware' | 'download' | 'servers';

const AvailableModels: React.FC<AvailableModelsProps> = ({
  isProUser = false,
  isUsingObServer,
  handleToggleObServer,
  showLoginMessage,
  isAuthenticated,
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

  // ── Remote/cloud model list (fetched from configured inference addresses) ──
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // ── Local model state (llama.cpp) ──
  const [nativeState, setNativeState] = useState<NativeModelState>(NativeLlmManager.getInstance().getState());
  const [nativeModels, setNativeModels] = useState<NativeLocalModel[]>(() => NativeLlmManager.getInstance().listNativeModels());
  const [orphanProjectors, setOrphanProjectors] = useState<GgufFileInfo[]>(() => NativeLlmManager.getInstance().listOrphanProjectors());
  const [samplerParams, setSamplerParams] = useState<SamplerParams>({ ...DEFAULT_SAMPLER_PARAMS });
  const [contextParams, setContextParams] = useState<ContextParams>({ ...DEFAULT_CONTEXT_PARAMS });
  const [useGpu, setUseGpu] = useState<boolean>(() => (isWeb() ? true : NativeLlmManager.getInstance().getPersistedUseGpu()));

  // ── Local model state (Transformers.js) ──
  const [gemmaState, setGemmaState] = useState<GemmaModelState>(GemmaModelManager.getInstance().getState());
  const [transformersModels, setTransformersModels] = useState<LocalModelEntry[]>(GemmaModelManager.getInstance().listLocalModels());
  const [gemmaDevice, setGemmaDevice] = useState<GemmaDevice>('webgpu');
  const [gemmaDtype, setGemmaDtype] = useState<GemmaDtype>('q4');
  const [gemmaTokenBudget, setGemmaTokenBudget] = useState<GemmaImageTokenBudget>(70);
  const [gemmaEnableThinking, setGemmaEnableThinking] = useState(false);
  const [customOnnxModelId, setCustomOnnxModelId] = useState('');

  // ── Preset download chaining (gguf → mmproj) ──
  const [downloadingPreset, setDownloadingPreset] = useState<ModelPreset | null>(null);
  const [presetDownloadStep, setPresetDownloadStep] = useState<'gguf' | 'mmproj' | null>(null);
  const [ggufUrl, setGgufUrl] = useState('');
  // Quant dropdown: gguf files per HF repo (fetched from the HF API) and the user's pick per model
  const [repoGgufs, setRepoGgufs] = useState<Record<string, HfFile[]>>({});
  const [selectedGguf, setSelectedGguf] = useState<Record<string, string>>({});

  // ── UI state ──
  const [expandedSettings, setExpandedSettings] = useState<string | null>(null);
  const [showBenchmark, setShowBenchmark] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advancedTab, setAdvancedTab] = useState<AdvancedTab>('hardware');
  const [showMoreQuants, setShowMoreQuants] = useState(false);
  const [showEngineLogs, setShowEngineLogs] = useState(false);

  // ── System memory (Tauri) ──
  const [memInfo, setMemInfo] = useState<{ totalBytes: number; usedBytes: number; availableBytes: number } | null>(null);

  // ── Engine debug ──
  const [engineInitStatus, setEngineInitStatus] = useState<'idle' | 'loading' | 'ok' | 'error'>('idle');
  const [engineInitError, setEngineInitError] = useState<string | null>(null);
  const [engineLogs, setEngineLogs] = useState<LogEntry[]>([]);

  // ── Custom servers (Servers tab) ──
  const [isAddingServer, setIsAddingServer] = useState(false);
  const [newServerAddress, setNewServerAddress] = useState('');
  const [addError, setAddError] = useState('');
  const [inferenceUrlInput, setInferenceUrlInput] = useState(appInferenceUrl || 'http://localhost:11434');

  // ── Ollama pull ──
  const [modelToPull, setModelToPull] = useState('');
  const [downloadState, setDownloadState] = useState<PullState>(pullModelManager.getInitialState());
  const [detectedServers, setDetectedServers] = useState<string[]>([]);
  const availableServers = detectedServers;
  const [selectedServer, setSelectedServer] = useState<string>('');

  // ── Subscriptions ──────────────────────────────────────────

  useEffect(() => {
    const unsub = NativeLlmManager.getInstance().onStateChange(state => {
      setNativeState(state);
      setNativeModels(NativeLlmManager.getInstance().listNativeModels());
      setOrphanProjectors(NativeLlmManager.getInstance().listOrphanProjectors());
    });
    NativeLlmManager.getInstance().listGgufFiles();
    return unsub;
  }, []);

  useEffect(() => {
    if (nativeState.status === 'loading' || nativeState.status === 'unloaded') {
      setSamplerParams({ ...DEFAULT_SAMPLER_PARAMS });
    } else if (nativeState.status === 'loaded') {
      NativeLlmManager.getInstance().getDebugInfo().then(info => {
        if (info.engine.samplerParams) setSamplerParams(info.engine.samplerParams);
        if (info.engine.contextParams) setContextParams(info.engine.contextParams);
      }).catch(() => {});
    }
  }, [nativeState.status]);

  useEffect(() => {
    const manager = GemmaModelManager.getInstance();
    const unsub = manager.onStateChange((state) => {
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
    if (currentState.loadSettings?.dtype) setGemmaDtype(currentState.loadSettings.dtype);
    return unsub;
  }, []);

  useEffect(() => {
    if (!showAdvanced || advancedTab !== 'hardware' || !showEngineLogs) return;
    const existing = Logger.getFilteredLogs({ source: ['NativeLlmManager', 'LlmEngine'] }).slice(-200);
    setEngineLogs(existing);
    const listener = (entry: LogEntry) => {
      if (entry.source === 'NativeLlmManager' || entry.source === 'LlmEngine') {
        setEngineLogs(prev => [...prev, entry].slice(-200));
      }
    };
    Logger.addListener(listener);
    return () => Logger.removeListener(listener);
  }, [showAdvanced, advancedTab, showEngineLogs]);

  useEffect(() => {
    if (!isTauriApp) return;
    const poll = async () => {
      try {
        const info = await invoke<{ totalBytes: number; usedBytes: number; availableBytes: number }>('get_memory_info');
        setMemInfo(info);
      } catch {}
    };
    poll();
    const id = setInterval(poll, 1000);
    return () => clearInterval(id);
  }, [isTauriApp]);

  // List each llama.cpp model's quants straight from its HF repo (the only live call; sizes of everything else are fixed).
  useEffect(() => {
    if (!isTauriApp) return;
    let cancelled = false;
    for (const { llamacpp } of MODEL_CATALOG) {
      if (!llamacpp) continue;
      const repo = llamacpp.repo;
      fetchGgufFiles(repo)
        .then(files => { if (!cancelled && files.length > 0) setRepoGgufs(prev => ({ ...prev, [repo]: files })); })
        .catch(() => {}); // offline / rate-limited: rows keep their default quant, no size shown
    }
    return () => { cancelled = true; };
  }, [isTauriApp]);

  useEffect(() => {
    return pullModelManager.subscribe((newState) => {
      setDownloadState(newState);
      if (newState.status === 'success') { fetchModelsList(); detectOllamaServers(); }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (availableServers.length > 0 && !selectedServer) setSelectedServer(availableServers[0]);
  }, [availableServers, selectedServer]);

  useEffect(() => {
    if (appInferenceUrl) setInferenceUrlInput(appInferenceUrl);
  }, [appInferenceUrl]);

  // Clear preset download state once the native download finishes
  useEffect(() => {
    if (nativeState.status !== 'downloading' && downloadingPreset?.engine === 'llamacpp') {
      if (presetDownloadStep === 'gguf' && downloadingPreset.mmprojUrl) {
        // gguf done, kick off mmproj. Pre-assign before downloading so the
        // projector renders inside the model's card during download (orphans
        // are otherwise hidden until assigned).
        setPresetDownloadStep('mmproj');
        const ggufFilename = downloadingPreset.ggufUrl!.split('/').pop()!;
        const mmprojFilename = downloadingPreset.mmprojUrl!.split('/').pop()!;
        NativeLlmManager.getInstance().setMmprojAssignment(ggufFilename, mmprojFilename);
        NativeLlmManager.getInstance().downloadModel(downloadingPreset.mmprojUrl)
          .catch(() => {})
          .finally(() => {
            setDownloadingPreset(null);
            setPresetDownloadStep(null);
          });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nativeState.status]);

  const checkOllamaSupport = async (address: string): Promise<boolean> => {
    try {
      const response = await platformFetch(`${address}/api/tags`, { method: 'GET', headers: { 'Content-Type': 'application/json' } });
      return response.ok;
    } catch { return false; }
  };

  const fetchModelsList = async () => {
    setLoading(true);
    try {
      const response = await fetchAllModels();
      if (response.error) throw new Error(response.error);
      setModels(response.models);
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
    setDetectedServers(checks.filter(c => c.ok).map(c => c.addr));
  };

  useEffect(() => { fetchModelsList(); detectOllamaServers(); }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchModelsList();
    detectOllamaServers();
  };

  // ── Sampler / context handlers ──────────────────────────────

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

  // ── Download handlers ────────────────────────────────────────

  const handleDownloadPreset = async (preset: ModelPreset) => {
    if (preset.engine === 'transformers') {
      // Some repos don't publish every dtype (declared via preset.dtypes): fall back to one they do.
      const dtype = !preset.dtypes || preset.dtypes.includes(gemmaDtype) ? gemmaDtype : preset.dtypes[0];
      GemmaModelManager.getInstance().loadModelWithSettings(
        preset.hfModelId! as GemmaModelId,
        useGpu ? 'webgpu' : 'wasm',
        dtype,
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
        setDownloadingPreset(null);
        setPresetDownloadStep(null);
      }
      // if mmprojUrl exists, the effect handles chaining
    } catch {
      setDownloadingPreset(null);
      setPresetDownloadStep(null);
    }
  };

  const handleCancelNativeDownload = () => {
    NativeLlmManager.getInstance().cancelDownload();
    setDownloadingPreset(null);
    setPresetDownloadStep(null);
  };

  const handleDownloadGguf = async () => {
    if (!ggufUrl.trim()) return;
    try {
      await NativeLlmManager.getInstance().downloadModel(ggufUrl.trim());
      setGgufUrl('');
    } catch {}
  };

  const handleLoadCustomOnnx = () => {
    if (customOnnxModelId.trim()) {
      GemmaModelManager.getInstance().loadModelWithSettings(
        customOnnxModelId.trim() as GemmaModelId, gemmaDevice, gemmaDtype, gemmaTokenBudget, gemmaEnableThinking,
      );
    }
  };

  // ── Servers ──────────────────────────────────────────────────

  const handleAddServer = () => {
    setAddError('');
    if (!newServerAddress.trim()) { setAddError('Please enter a server address'); return; }
    if (!newServerAddress.match(/^https?:\/\//)) { setAddError('URL must start with http:// or https://'); return; }
    try {
      new URL(newServerAddress);
      onAddCustomServer?.(newServerAddress);
      setNewServerAddress('');
      setIsAddingServer(false);
    } catch {
      setAddError('Invalid URL format');
    }
  };

  const handleStartPull = () => {
    if (modelToPull.trim() && selectedServer) pullModelManager.pullModel(modelToPull.trim(), selectedServer);
  };
  const handleCancelPull = () => pullModelManager.cancelPull();

  // ── Toggle inline settings ──────────────────────────────────

  const toggleSettings = (id: string) => setExpandedSettings(prev => (prev === id ? null : id));

  const hasRemoteSettings = (model: Model) => {
    if (model.server === SKIP_MODEL_SENTINEL) return false;
    if (model.server === LLAMA_CPP_LOCAL_SENTINEL) return false;
    if (model.server === BROWSER_LOCAL_SENTINEL) return false;
    if (model.server.includes('api.observer-ai.com')) return false;
    return true;
  };

  const isAnyNativeBusy = nativeState.status === 'loading' || nativeState.status === 'unloading' || nativeState.status === 'downloading';

  // ── Unified local model list: one row per catalog model, one control per engine ───────

  const catalogRows = MODEL_CATALOG.map(model => {
    const basePreset = MODEL_PRESETS.find(p => p.name === model.name && p.engine === 'llamacpp') ?? null;
    const transformersPreset = MODEL_PRESETS.find(p => p.name === model.name && p.engine === 'transformers') ?? null;

    // llama.cpp: apply the dropdown's quant pick on top of the catalog default
    const options: GgufOption[] = (basePreset?.repo ? repoGgufs[basePreset.repo] ?? [] : [])
      .filter(f => !/mmproj/i.test(f.path) && !/-\d{5}-of-\d{5}/.test(f.path) && quantLabelOf(f.path) !== null)
      .map(f => ({ file: f.path, size: f.size }))
      .sort((a, b) => a.size - b.size);
    const defaultFile = basePreset?.ggufUrl?.split('/').pop() ?? null;
    const selectedFile = selectedGguf[model.name] ?? defaultFile;
    const nativePreset = basePreset && selectedFile
      ? { ...basePreset, ggufUrl: basePreset.ggufUrl!.replace(/[^/]+$/, selectedFile) }
      : basePreset;
    // Installed = the picked quant, else any other quant of this repo already on disk
    const repoFiles = new Set([...(defaultFile ? [defaultFile] : []), ...options.map(o => o.file)]);
    const nativeInstalled = nativeModels.find(m => m.id === selectedFile)
      ?? nativeModels.find(m => repoFiles.has(m.id)) ?? null;
    const transformersInstalled = transformersPreset
      ? transformersModels.find(m => m.id === transformersPreset.hfModelId) ?? null
      : null;
    return { model, nativePreset, transformersPreset, nativeInstalled, transformersInstalled, options, selectedFile };
  });
  const nativeExtraModels = nativeModels.filter(
    m => !catalogRows.some(r => r.nativeInstalled?.id === m.id)
  );
  const transformersExtraModels = transformersModels.filter(
    m => !catalogRows.some(r => r.transformersInstalled?.id === m.id)
  );

  const remoteModels = models.filter(m =>
    m.server !== LLAMA_CPP_LOCAL_SENTINEL && m.server !== BROWSER_LOCAL_SENTINEL &&
    m.server !== SKIP_MODEL_SENTINEL && !m.server.includes('api.observer-ai.com')
  );
  const cloudModels = models.filter(m => m.server.includes('api.observer-ai.com'));

  // ── Engine views ──────────────────────────────────────────────
  // Each engine (llama.cpp / Transformers.js) builds the pieces of its half of a row; the row
  // renderers below either combine two views (catalog models) or show one (extra installed files).

  interface EngineView {
    idle: boolean;            // not installed and not downloading: just a download button
    meta?: React.ReactNode;   // status text, only for non-idle views
    action: React.ReactNode;
    detailSlot?: React.ReactNode;
    settingsSlot?: React.ReactNode;
  }

  const nativeInstalledView = (model: NativeLocalModel, isTutorialTarget = false): EngineView => {
    const isLoaded = model.runtime === 'loaded';
    const isLoading = model.runtime === 'loading';
    const isUnloading = nativeState.modelId === model.name && nativeState.status === 'unloading';
    const modelFile = model.modelFile;
    const projectorFile = model.projectorFile;
    const isModelDownloading = modelFile.kind === 'partial' && modelFile.downloading;
    const isProjectorDownloading = projectorFile.kind === 'partial' && projectorFile.downloading;
    const isDownloading = isModelDownloading || isProjectorDownloading;
    const isFullyOnDisk = modelFile.kind === 'complete' &&
      (projectorFile.kind === 'complete' || projectorFile.kind === 'absent');
    const settingsOpen = expandedSettings === model.id;

    // A preset download in flight owns this row: list gguf + mmproj up front, even
    // though the .part file has already replaced the preset row with this one.
    const activePreset = downloadingPreset?.mmprojUrl && downloadingPreset.ggufUrl!.split('/').pop() === model.id
      ? downloadingPreset : null;
    const fileProgress: FileProgress[] = [];
    if (activePreset) {
      fileProgress.push(...presetFileProgress(activePreset));
    } else if (modelFile.kind === 'partial' && modelFile.downloading) {
      fileProgress.push({ file: model.id, loaded: modelFile.downloadedBytes ?? modelFile.bytes, total: modelFile.totalBytes ?? 0, progress: modelFile.progress ?? 0, done: false });
    }
    if (activePreset) {
      // already covered by presetFileProgress
    } else if (projectorFile.kind === 'partial' && projectorFile.downloading) {
      fileProgress.push({ file: model.projectorFilename ?? 'vision projector', loaded: projectorFile.downloadedBytes ?? projectorFile.bytes, total: projectorFile.totalBytes ?? 0, progress: projectorFile.progress ?? 0, done: false });
    } else if (isModelDownloading && projectorFile.kind === 'complete') {
      fileProgress.push({ file: model.projectorFilename ?? 'vision projector', loaded: projectorFile.bytes, total: projectorFile.bytes, progress: 100, done: true });
    }

    const sizeText = modelFile.kind === 'complete' ? formatBytes(modelFile.bytes)
      : isModelDownloading ? 'Downloading model…'
      : isProjectorDownloading ? 'Downloading vision projector…'
      : modelFile.kind === 'partial' ? 'Paused — resume from Download'
      : 'Missing';

    const action = isDownloading ? (
      <RowIconButton onClick={handleCancelNativeDownload} title="Cancel download"><StopCircle size={14} /></RowIconButton>
    ) : isUnloading ? (
      <span className="text-xs text-gray-400">Unloading…</span>
    ) : isLoaded ? (
      <>
        <RowIconButton onClick={() => toggleSettings(model.id)} title="Generation settings" className={settingsOpen ? 'text-gray-700 bg-gray-100' : ''}>
          <Settings2 size={14} />
        </RowIconButton>
        <button
          onClick={() => NativeLlmManager.getInstance().unloadModel()}
          className="group px-2.5 py-1 text-xs font-medium rounded-md bg-gray-100 text-gray-700 hover:bg-red-50 hover:text-red-600 transition-colors"
        >
          <span className="group-hover:hidden">Ready</span>
          <span className="hidden group-hover:inline">Unload</span>
        </button>
      </>
    ) : isLoading ? (
      <RowButtonGhost onClick={() => NativeLlmManager.getInstance().unloadModel()}>Loading…</RowButtonGhost>
    ) : (
      <>
        <RowButtonPrimary
          disabled={isAnyNativeBusy || !isFullyOnDisk}
          onClick={() => NativeLlmManager.getInstance().loadModel(model.id, undefined, contextParams.imageMinTokens, contextParams.imageMaxTokens)}
        >
          Load
        </RowButtonPrimary>
        <RowIconButton disabled={isAnyNativeBusy} onClick={() => deleteNativeModelCascade(model, nativeModels)} title="Delete">
          <Trash2 size={14} />
        </RowIconButton>
      </>
    );

    const meta = (
      <span className="flex items-center gap-1.5">
        {sizeText}{model.isMultimodal ? ' · Vision' : ''}
        {isTutorialTarget && (
          <span className="hidden" data-tutorial-gemma-state={modelFile.kind === 'complete' ? 'installed' : isModelDownloading ? 'downloading' : undefined} />
        )}
        {(orphanProjectors.length > 0 || model.projectorFilename !== null) && (
          <select
            value={model.projectorFilename ?? ''}
            onChange={e => NativeLlmManager.getInstance().setMmprojAssignment(model.id, e.target.value || null)}
            disabled={isLoaded || isLoading || isDownloading}
            onClick={e => e.stopPropagation()}
            className="text-xs border border-gray-200 rounded px-1 py-0.5 bg-white text-gray-500 disabled:opacity-50 max-w-[140px] truncate"
            title="Assign a vision projector"
          >
            <option value="">No projector</option>
            {model.projectorFilename && (
              <option value={model.projectorFilename}>{model.projectorFilename.replace(/\.gguf$/i, '')}</option>
            )}
            {orphanProjectors.map(p => (
              <option key={p.filename} value={p.filename}>{p.filename.replace(/\.gguf$/i, '')}</option>
            ))}
          </select>
        )}
      </span>
    );

    return {
      idle: false,
      meta,
      action,
      detailSlot: <FileProgressList items={fileProgress} />,
      settingsSlot: isLoaded && settingsOpen && (
        <LlamaCppSamplerPanel
          nativeStatus={nativeState.status}
          samplerParams={samplerParams}
          onParamChange={handleSamplerParamChange}
          onReset={handleResetSamplerParams}
        />
      ),
    };
  };

  // Both files (gguf + mmproj) are listed from the first click so the user sees the
  // full download up front; the inactive one waits at 0% / shows Done.
  const presetFileProgress = (preset: ModelPreset): FileProgress[] => {
    const ggufName = preset.ggufUrl!.split('/').pop()!;
    const live = { loaded: nativeState.downloadedBytes, total: nativeState.totalBytes, progress: nativeState.downloadProgress };
    const idle = { loaded: 0, total: 0, progress: 0 };
    const inMmproj = presetDownloadStep === 'mmproj';
    const items: FileProgress[] = [inMmproj
      ? { file: ggufName, loaded: 1, total: 1, progress: 100, done: true }
      : { file: ggufName, ...live, done: false }];
    if (preset.mmprojUrl) {
      items.push({ file: preset.mmprojUrl.split('/').pop()!, ...(inMmproj ? live : idle), done: false });
    }
    return items;
  };

  const nativePresetView = (preset: ModelPreset, isTutorialTarget = false, quantSelect: React.ReactNode = null): EngineView => {
    const isThisDownloading = downloadingPreset?.name === preset.name && isAnyNativeBusy;
    const blocked = isAnyNativeBusy && !isThisDownloading;
    if (isThisDownloading) {
      return {
        idle: false,
        meta: 'Downloading…',
        detailSlot: <FileProgressList items={presetFileProgress(preset)} />,
        action: (
          <RowIconButton onClick={handleCancelNativeDownload} title="Cancel download" {...(isTutorialTarget ? { 'data-tutorial-gemma-state': 'downloading' } : {})}><StopCircle size={14} /></RowIconButton>
        ),
      };
    }
    return {
      idle: true,
      action: (
        <>
          {quantSelect}
          <RowButtonPrimary
            disabled={blocked}
            onClick={() => handleDownloadPreset(preset)}
            title="Download for llama.cpp (runs natively in the app)"
            {...(isTutorialTarget ? { 'data-tutorial-gemma-e2b': true } : {})}
          >
            <Download size={12} className="inline -mt-0.5 mr-1" />App
          </RowButtonPrimary>
        </>
      ),
    };
  };

  const transformersInstalledView = (model: LocalModelEntry, isTutorialTarget = false): EngineView => {
    const isThisModel = gemmaState.modelId === model.id;
    const status = isThisModel ? gemmaState.status : model.status;
    const isLoaded = status === 'loaded';
    const isLoading = status === 'loading';
    const isError = status === 'error';
    const loadSettings = isThisModel ? gemmaState.loadSettings : null;
    const fileProgress = isThisModel && isLoading ? gemmaFileProgress(gemmaState) : [];
    const tutorialAttrs = (state: string) => (isTutorialTarget ? { 'data-tutorial-gemma-state': state } : {});

    const meta = isError ? (isThisModel ? gemmaState.error ?? 'Error' : 'Error')
      : loadSettings ? `${loadSettings.device} · ${loadSettings.dtype} · ${loadSettings.imageTokenBudget} tokens`
      : isLoading ? 'Downloading…' : 'Downloaded';

    const action = isLoaded ? (
      <button
        onClick={() => GemmaModelManager.getInstance().unloadModel()}
        className="group px-2.5 py-1 text-xs font-medium rounded-md bg-gray-100 text-gray-700 hover:bg-red-50 hover:text-red-600 transition-colors"
        {...tutorialAttrs('loaded')}
      >
        <span className="group-hover:hidden">Ready</span>
        <span className="hidden group-hover:inline">Unload</span>
      </button>
    ) : isLoading ? (
      <RowButtonGhost onClick={() => GemmaModelManager.getInstance().unloadModel()} {...tutorialAttrs('loading')}>Cancel</RowButtonGhost>
    ) : isError ? (
      <RowButtonGhost onClick={() => GemmaModelManager.getInstance().deleteModel(model.id as GemmaModelId)} {...tutorialAttrs('error')}>Remove</RowButtonGhost>
    ) : (
      <>
        <RowButtonPrimary onClick={() => GemmaModelManager.getInstance().loadModel(model.id as GemmaModelId)} {...(isTutorialTarget ? { 'data-tutorial-gemma-e2b': true, ...tutorialAttrs('load') } : {})}>Load</RowButtonPrimary>
        <RowIconButton onClick={() => GemmaModelManager.getInstance().deleteModel(model.id as GemmaModelId)} title="Delete"><Trash2 size={14} /></RowIconButton>
      </>
    );

    return { idle: false, meta, action, detailSlot: <FileProgressList items={fileProgress} /> };
  };

  const transformersPresetView = (preset: ModelPreset, isTutorialTarget = false): EngineView => {
    const isThisDownloading = gemmaState.modelId === preset.hfModelId && gemmaState.status === 'loading';
    const blocked = gemmaState.status === 'loading' && gemmaState.modelId !== preset.hfModelId;
    if (isThisDownloading) {
      return {
        idle: false,
        meta: 'Downloading…',
        detailSlot: <FileProgressList items={gemmaFileProgress(gemmaState)} />,
        action: (
          <RowButtonGhost onClick={() => GemmaModelManager.getInstance().unloadModel()} {...(isTutorialTarget ? { 'data-tutorial-gemma-state': 'downloading' } : {})}>Cancel</RowButtonGhost>
        ),
      };
    }
    return {
      idle: true,
      action: (
        <RowButtonPrimary
          disabled={blocked}
          onClick={() => handleDownloadPreset(preset)}
          title="Download for Transformers.js (runs in the webview via WebGPU/WASM)"
          {...(isTutorialTarget ? { 'data-tutorial-gemma-e2b': true } : {})}
        >
          <Download size={12} className="inline -mt-0.5 mr-1" />Web
        </RowButtonPrimary>
      ),
    };
  };

  // ── Row renderers ─────────────────────────────────────────────

  // Table columns shared by the header and every row so the Transformers.js / llama.cpp cells line up.
  const TF_COL = 'md:w-[104px]';
  const NATIVE_COL = 'w-[148px]';
  // Fixed width below md only matters when two cells stack; web has just the one
  const ACTION_COL = isTauriApp ? 'w-[148px]' : 'w-auto';

  // Below md the two engine cells stack ([Web] above [App]) in one fixed-width column so the name keeps
  // the room; from md up they sit side by side as table columns under the header.
  const engineCells = (transformersCell: React.ReactNode, nativeCell: React.ReactNode) => (
    <div className={`${ACTION_COL} flex flex-col items-start gap-1.5 md:w-auto md:flex-row md:items-center md:gap-0`}>
      <div className={`${TF_COL} flex items-center gap-1`}>{transformersCell}</div>
      {isTauriApp && <div className={`${NATIVE_COL} flex items-center gap-1`}>{nativeCell}</div>}
    </div>
  );

  // Installed files that don't match a catalog model (custom GGUFs, retired presets, custom ONNX ids).
  const renderNativeInstalledRow = (model: NativeLocalModel) => {
    const v = nativeInstalledView(model);
    return <ModelRow key={model.id} icon={<Cpu size={16} />} name={model.name} tag="llama.cpp" meta={v.meta} detailSlot={v.detailSlot} action={engineCells(null, v.action)} settingsSlot={v.settingsSlot} />;
  };

  const renderTransformersInstalledRow = (model: LocalModelEntry) => {
    const v = transformersInstalledView(model);
    return <ModelRow key={model.id} icon={<Cpu size={16} />} name={model.name} tag="Transformers.js" meta={v.meta} detailSlot={v.detailSlot} action={engineCells(v.action, null)} />;
  };

  const renderCatalogRow = ({ model, nativePreset, transformersPreset, nativeInstalled, transformersInstalled, options, selectedFile }: typeof catalogRows[number]) => {
    // The onboarding tutorial targets one engine per platform: llama.cpp in the app, Transformers.js on web.
    const isTutorialModel = model.name === TUTORIAL_MODEL_NAME;
    const engines: { label: string; view: EngineView }[] = [];
    let transformersView: EngineView | null = null;
    let nativeView: EngineView | null = null;

    if (transformersPreset) {
      const tutorialTarget = isTutorialModel && !isTauriApp;
      transformersView = transformersInstalled
        ? transformersInstalledView(transformersInstalled, tutorialTarget)
        : transformersPresetView(transformersPreset, tutorialTarget);
      engines.push({ label: 'Transformers.js', view: transformersView });
    }

    if (nativePreset && isTauriApp) {
      // Styled like the Download button; the closed state shows only the quant (e.g. "Q3_K_S")
      const quantSelect = options.length > 1 ? (
        <span className="relative inline-flex items-center">
          <select
            value={selectedFile ?? ''}
            onChange={e => setSelectedGguf(prev => ({ ...prev, [model.name]: e.target.value }))}
            disabled={isAnyNativeBusy}
            title="Quantization (from the HuggingFace repo)"
            className="appearance-none cursor-pointer w-[76px] truncate pl-2.5 pr-5 py-1 text-xs font-medium bg-gray-900 text-white rounded-md hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {options.map(o => (
              <option key={o.file} value={o.file} className="bg-white text-gray-900">{quantLabelOf(o.file)}</option>
            ))}
          </select>
          <ChevronDown size={11} className="pointer-events-none absolute right-1.5 text-white/70" />
        </span>
      ) : null;
      nativeView = nativeInstalled
        ? nativeInstalledView(nativeInstalled, isTutorialModel)
        : nativePresetView(nativePreset, isTutorialModel, quantSelect);
      engines.push({ label: 'llama.cpp', view: nativeView });
    }

    // "2B · 3.4 GB", plus status text (downloading, load settings, projector picker) only while an engine is active
    return (
      <ModelRow
        key={model.name}
        icon={<Cpu size={16} />}
        name={model.name}
        meta={
          <span className="flex items-center gap-x-1.5">
            <span>{model.params} · {model.size}</span>
            {engines.filter(e => !e.view.idle).map(e => (
              <React.Fragment key={e.label}><span>·</span>{e.view.meta}</React.Fragment>
            ))}
          </span>
        }
        detailSlot={engines.map(e => <React.Fragment key={e.label}>{e.view.detailSlot}</React.Fragment>)}
        settingsSlot={engines.map(e => <React.Fragment key={e.label}>{e.view.settingsSlot}</React.Fragment>)}
        action={engineCells(transformersView?.action, nativeView?.action)}
      />
    );
  };

  // Column headers; mirrors ModelRow's layout (icon + flexible name + action cells).
  const localModelsHeader = (
    <div className="flex items-center gap-3 pb-1.5 text-[11px] font-medium text-gray-400">
      <div className="w-8 flex-shrink-0" />
      <div className="flex-1 min-w-0" />
      {/* md and up: side-by-side column headers */}
      <div className="hidden md:flex items-center flex-shrink-0">
        <div className={TF_COL}>Transformers.js</div>
        {isTauriApp && <div className={NATIVE_COL}>llama.cpp</div>}
      </div>
      {/* below md: stacked to match the stacked [Web] / [App] buttons */}
      <div className={`${ACTION_COL} md:hidden flex flex-col items-start leading-tight flex-shrink-0`}>
        <span>Transformers.js{isTauriApp ? ' /' : ''}</span>
        {isTauriApp && <span>llama.cpp</span>}
      </div>
    </div>
  );

  if (loading && !refreshing) {
    return (
      <div className="flex flex-col items-center justify-center py-16">
        <div className="animate-spin mb-3"><Cpu className="h-6 w-6 text-gray-400" /></div>
        <p className="text-sm text-gray-500">Loading models…</p>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">Models</h2>
        <div className="flex items-center gap-1">
          <RowIconButton onClick={() => setShowBenchmark(v => !v)} title="Benchmark" className={showBenchmark ? 'text-gray-900 bg-gray-100' : ''}>
            <BarChart3 size={16} />
          </RowIconButton>
          <RowIconButton onClick={handleRefresh} disabled={refreshing} title="Refresh">
            <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
          </RowIconButton>
        </div>
      </div>

      {/* Cloud credits — small pill, not a card */}
      <div className="px-3.5 py-2.5 mb-5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50/60 dark:bg-gray-800/60 space-y-2">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <Cloud size={15} className={isUsingObServer ? 'text-purple-500' : 'text-gray-400'} />
            <span className="text-sm text-gray-700 dark:text-gray-200">Cloud inference</span>
            {isUsingObServer && !isAuthenticated && (
              <span className="text-xs text-gray-400">Sign in to use</span>
            )}
            {isUsingObServer && isAuthenticated && (!quotaInfo || quotaInfo.tier === 'max' || !quotaInfo.daily || !quotaInfo.monthly) && renderQuotaStatus && (
              <span className="text-xs">{renderQuotaStatus()}</span>
            )}
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {showLoginMessage && <span className="text-xs text-red-500">Login required</span>}
            <button
              onClick={handleToggleObServer}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${isUsingObServer ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'}`}
              aria-label={isUsingObServer ? 'Disable cloud inference' : 'Enable cloud inference'}
            >
              <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${isUsingObServer ? 'translate-x-5' : 'translate-x-1'}`} />
            </button>
          </div>
        </div>

        {isUsingObServer && isAuthenticated && quotaInfo && quotaInfo.tier !== 'max' && quotaInfo.daily && quotaInfo.monthly && (
          <div className="flex items-center gap-4">
            <QuotaBar label="Today" block={quotaInfo.daily} />
            <QuotaBar label="This month" block={quotaInfo.monthly} />
          </div>
        )}
      </div>

      {/* Local models: one row per model, a download button per engine */}
      {/* Heuristic for mobile: below Tailwind's md breakpoint */}
      <div className="md:hidden mb-3 px-3 py-2 rounded-lg border border-red-200 bg-red-50 text-xs text-red-600">
        Transformers.js models will crash on mobile.
      </div>
      <Section title="Local Models">
        {localModelsHeader}
        {catalogRows.map(renderCatalogRow)}
        {nativeExtraModels.map(renderNativeInstalledRow)}
        {transformersExtraModels.map(renderTransformersInstalledRow)}
      </Section>

      {/* Cloud Models */}
      <Section title="Cloud Models">
        {cloudModels.map(model => {
          const settingsOpen = expandedSettings === model.name;
          const canConfigure = hasRemoteSettings(model);
          const locked = model.pro && !isProUser;
          return (
            <ModelRow
              key={model.name}
              icon={<Cloud size={16} />}
              name={model.name}
              tag={model.pro ? 'PRO' : undefined}
              dimmed={locked}
              meta={[model.parameterSize && model.parameterSize !== 'N/A' ? model.parameterSize : null, model.multimodal ? 'Vision' : null].filter(Boolean).join(' · ') || undefined}
              action={canConfigure && (
                <RowIconButton onClick={() => toggleSettings(model.name)} title="Inference settings" className={settingsOpen ? 'text-gray-700 bg-gray-100' : ''}>
                  <Settings2 size={14} />
                </RowIconButton>
              )}
              settingsSlot={settingsOpen && canConfigure && <RemoteInferenceParamsPanel modelName={model.name} />}
            />
          );
        })}
      </Section>

      {/* Custom Server Models */}
      <Section title="Custom Server Models">
        {remoteModels.map(model => {
          const settingsOpen = expandedSettings === model.name;
          const canConfigure = hasRemoteSettings(model);
          return (
            <ModelRow
              key={model.name}
              icon={<Server size={16} />}
              name={model.name}
              meta={[model.parameterSize && model.parameterSize !== 'N/A' ? model.parameterSize : null, model.multimodal ? 'Vision' : null].filter(Boolean).join(' · ') || undefined}
              action={canConfigure && (
                <RowIconButton onClick={() => toggleSettings(model.name)} title="Inference settings" className={settingsOpen ? 'text-gray-700 bg-gray-100' : ''}>
                  <Settings2 size={14} />
                </RowIconButton>
              )}
              settingsSlot={settingsOpen && canConfigure && <RemoteInferenceParamsPanel modelName={model.name} />}
            />
          );
        })}
      </Section>

      <Section title="Other">
        <ModelRow
          icon={<MinusCircle size={16} />}
          name="Skip Model Call"
          meta="Runs the agent loop with no AI model — response is always empty"
          action={<span className="text-xs text-gray-400">Always ready</span>}
        />
      </Section>

      {/* Advanced */}
      <div className="mt-6 border-t border-gray-100 pt-4">
        <button onClick={() => setShowAdvanced(v => !v)} className="flex items-center gap-1.5 text-xs font-medium text-gray-500 hover:text-gray-700">
          <ChevronDown size={14} className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
          Advanced
        </button>

        {showAdvanced && (
          <div className="mt-3">
            <div className="flex gap-4 border-b border-gray-100 mb-4 text-sm">
              {(['hardware', 'download', 'servers'] as AdvancedTab[]).map(tab => (
                <button
                  key={tab}
                  onClick={() => setAdvancedTab(tab)}
                  className={`pb-2 -mb-px border-b-2 transition-colors ${
                    advancedTab === tab ? 'border-gray-800 text-gray-900 font-medium' : 'border-transparent text-gray-400 hover:text-gray-600'
                  }`}
                >
                  {tab === 'hardware' ? 'Hardware' : tab === 'download' ? 'Download' : 'Servers'}
                </button>
              ))}
            </div>

            {advancedTab === 'hardware' && (
              <div className="space-y-5 text-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-800">GPU</p>
                    <p className="text-xs text-gray-400">{useGpu ? 'WebGPU / Metal' : 'CPU only'}</p>
                  </div>
                  <button
                    onClick={() => handleToggleUnifiedGpu(!useGpu)}
                    disabled={nativeState.status === 'loading' || nativeState.status === 'loaded'}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors disabled:opacity-50 ${useGpu ? 'bg-gray-800' : 'bg-gray-300'}`}
                  >
                    <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${useGpu ? 'translate-x-5' : 'translate-x-1'}`} />
                  </button>
                </div>

                {isTauriApp && memInfo && (
                  <div>
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                      <span className="font-medium text-gray-800 text-sm">RAM</span>
                      <span className="font-mono">{formatBytes(memInfo.usedBytes)} / {formatBytes(memInfo.totalBytes)}</span>
                    </div>
                    <div className="w-full bg-gray-100 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full transition-all ${memInfo.usedBytes / memInfo.totalBytes > 0.9 ? 'bg-red-500' : 'bg-gray-800'}`}
                        style={{ width: `${Math.min(100, (memInfo.usedBytes / memInfo.totalBytes) * 100)}%` }}
                      />
                    </div>
                  </div>
                )}

                {isTauriApp && (
                  <div className="pt-1 border-t border-gray-100 space-y-4">
                    <div className="flex items-center justify-between">
                      <p className="text-gray-700">Thinking mode (llama.cpp)</p>
                      <button
                        onClick={() => NativeLlmManager.getInstance().setEnableThinking(!nativeState.enableThinking)}
                        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${nativeState.enableThinking ? 'bg-gray-800' : 'bg-gray-300'}`}
                      >
                        <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${nativeState.enableThinking ? 'translate-x-5' : 'translate-x-1'}`} />
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-gray-500">Context (text)</label>
                        <input type="number" value={contextParams.nCtx} min={512} max={32768} step={512}
                          onChange={(e) => handleContextParamChange('nCtx', parseInt(e.target.value) || DEFAULT_CONTEXT_PARAMS.nCtx)}
                          className="w-full mt-0.5 p-1.5 text-sm border border-gray-200 rounded-md font-mono" />
                      </div>
                      <div>
                        <label className="text-xs text-gray-500">Context (multimodal)</label>
                        <input type="number" value={contextParams.nCtxMultimodal} min={512} max={32768} step={512}
                          onChange={(e) => handleContextParamChange('nCtxMultimodal', parseInt(e.target.value) || DEFAULT_CONTEXT_PARAMS.nCtxMultimodal)}
                          className="w-full mt-0.5 p-1.5 text-sm border border-gray-200 rounded-md font-mono" />
                      </div>
                      <div>
                        <label className="text-xs text-gray-500">CPU threads (0 = auto)</label>
                        <input type="number" value={contextParams.nThreads} min={0} max={32}
                          onChange={(e) => handleContextParamChange('nThreads', parseInt(e.target.value) || 0)}
                          className="w-full mt-0.5 p-1.5 text-sm border border-gray-200 rounded-md font-mono" />
                      </div>
                      <div>
                        <label className="text-xs text-amber-700">GPU layers (-1 auto, 0 CPU, 99 all)</label>
                        <input type="number" value={contextParams.nGpuLayers} min={-1} max={999}
                          onChange={(e) => handleContextParamChange('nGpuLayers', parseInt(e.target.value) ?? -1)}
                          className="w-full mt-0.5 p-1.5 text-sm border border-amber-200 bg-amber-50 rounded-md font-mono" />
                      </div>
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Image token budget</label>
                      <select
                        value={contextParams.imageMaxTokens}
                        onChange={(e) => {
                          const v = parseInt(e.target.value);
                          setContextParams(prev => ({ ...prev, imageMinTokens: v, imageMaxTokens: v }));
                          NativeLlmManager.getInstance().setContextParams({ imageMinTokens: v, imageMaxTokens: v }).catch(() => {});
                        }}
                        className="w-full mt-0.5 p-1.5 text-sm border border-gray-200 rounded-md font-mono"
                      >
                        <option value={-1}>Default (model decides)</option>
                        <option value={70}>70 — fastest, low detail</option>
                        <option value={140}>140 — fast</option>
                        <option value={280}>280 — balanced</option>
                        <option value={560}>560 — detailed</option>
                        <option value={1120}>1120 — max detail (OCR/documents)</option>
                      </select>
                    </div>
                    <button onClick={handleResetContextParams} className="text-xs text-gray-400 hover:text-gray-600 underline">Reset to defaults</button>
                  </div>
                )}

                <div className="pt-1 border-t border-gray-100 space-y-3">
                  <p className="text-gray-700">Transformers.js runtime</p>
                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <label className="text-xs text-gray-500">Device</label>
                      <select value={gemmaDevice} onChange={e => setGemmaDevice(e.target.value as GemmaDevice)} disabled={gemmaState.status === 'loading'}
                        className="w-full mt-0.5 p-1.5 text-sm border border-gray-200 rounded-md disabled:opacity-50">
                        <option value="webgpu">WebGPU</option>
                        <option value="wasm">WASM</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Precision</label>
                      <select value={gemmaDtype} onChange={e => setGemmaDtype(e.target.value as GemmaDtype)} disabled={gemmaState.status === 'loading'}
                        className="w-full mt-0.5 p-1.5 text-sm border border-gray-200 rounded-md disabled:opacity-50">
                        <option value="q4f16">q4f16</option>
                        <option value="q4">q4</option>
                        <option value="q8">q8</option>
                        <option value="fp16">fp16</option>
                        <option value="fp32">fp32</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Image tokens</label>
                      <select value={gemmaTokenBudget} onChange={e => setGemmaTokenBudget(Number(e.target.value) as GemmaImageTokenBudget)} disabled={gemmaState.status === 'loading'}
                        className="w-full mt-0.5 p-1.5 text-sm border border-gray-200 rounded-md disabled:opacity-50">
                        <option value={70}>70</option>
                        <option value={140}>140</option>
                        <option value={280}>280</option>
                        <option value={560}>560</option>
                        <option value={1120}>1120</option>
                      </select>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <p className="text-gray-700">Thinking mode</p>
                    <button
                      onClick={() => setGemmaEnableThinking(v => !v)}
                      disabled={gemmaState.status === 'loading'}
                      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors disabled:opacity-50 ${gemmaEnableThinking ? 'bg-gray-800' : 'bg-gray-300'}`}
                    >
                      <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${gemmaEnableThinking ? 'translate-x-5' : 'translate-x-1'}`} />
                    </button>
                  </div>
                </div>

                {isTauriApp && (
                  <div className="pt-1 border-t border-gray-100">
                    <button onClick={() => setShowEngineLogs(v => !v)} className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-600">
                      <ChevronDown size={12} className={`transition-transform ${showEngineLogs ? 'rotate-180' : ''}`} /> Engine logs
                    </button>
                    {showEngineLogs && (
                      <div className="mt-2 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1.5 text-xs text-gray-500"><Play size={12} /> Engine {engineInitStatus === 'ok' ? '· initialized' : engineInitStatus === 'error' ? '· error' : ''}</span>
                          <button
                            onClick={async () => {
                              setEngineInitStatus('loading'); setEngineInitError(null);
                              try { await NativeLlmManager.getInstance().initEngine(); setEngineInitStatus('ok'); }
                              catch (e) { setEngineInitStatus('error'); setEngineInitError(e instanceof Error ? e.message : String(e)); }
                            }}
                            disabled={engineInitStatus === 'loading'}
                            className="text-xs text-gray-600 hover:text-gray-900 underline disabled:opacity-50"
                          >
                            {engineInitStatus === 'loading' ? 'Initializing…' : 'Init engine'}
                          </button>
                        </div>
                        {engineInitError && <p className="text-xs text-red-600">{engineInitError}</p>}
                        <div className="h-40 overflow-y-auto bg-gray-900 rounded p-2 font-mono text-[10px] leading-tight">
                          {engineLogs.length === 0 ? (
                            <span className="text-gray-500">No logs yet</span>
                          ) : engineLogs.map(log => (
                            <div key={log.id} className={log.level === LogLevel.ERROR ? 'text-red-400' : log.level === LogLevel.WARNING ? 'text-yellow-400' : 'text-gray-300'}>
                              <span className="text-gray-500">{log.timestamp.toLocaleTimeString()} </span>{log.message}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {advancedTab === 'download' && (
              <div className="space-y-4 text-sm">
                {isTauriApp && (
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">GGUF from URL</label>
                    <div className="flex gap-2">
                      <input type="text" value={ggufUrl} onChange={(e) => setGgufUrl(e.target.value)}
                        placeholder="https://huggingface.co/.../model.gguf" disabled={isAnyNativeBusy}
                        className="flex-1 min-w-0 p-2 text-sm border border-gray-200 rounded-md disabled:opacity-50" />
                      <RowButtonPrimary onClick={handleDownloadGguf} disabled={!ggufUrl.trim() || isAnyNativeBusy} className="px-3 py-2">Download</RowButtonPrimary>
                    </div>
                  </div>
                )}
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Hugging Face ONNX model ID</label>
                  <div className="flex gap-2">
                    <input type="text" value={customOnnxModelId} onChange={(e) => setCustomOnnxModelId(e.target.value)}
                      placeholder="onnx-community/model-name" disabled={gemmaState.status === 'loading'}
                      className="flex-1 min-w-0 p-2 text-sm border border-gray-200 rounded-md disabled:opacity-50" />
                    <RowButtonPrimary onClick={handleLoadCustomOnnx} disabled={!customOnnxModelId.trim() || gemmaState.status === 'loading'} className="px-3 py-2">Load</RowButtonPrimary>
                  </div>
                </div>

                {isTauriApp && (
                  <div className="pt-2 border-t border-gray-100">
                    <button onClick={() => setShowMoreQuants(v => !v)} className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-600">
                      <ChevronDown size={12} className={`transition-transform ${showMoreQuants ? 'rotate-180' : ''}`} /> More quantizations (Gemma 4 E2B)
                    </button>
                    {showMoreQuants && (
                      <div className="divide-y divide-gray-100 mt-2">
                        {EXTENDED_PRESETS.map(preset => {
                          const filename = preset.ggufUrl?.split('/').pop();
                          const installed = filename ? nativeModels.find(m => m.id === filename && m.modelFile.kind === 'complete') : undefined;
                          const isThisDownloading = downloadingPreset?.name === preset.name;
                          return (
                            <ModelRow
                              key={preset.name}
                              icon={<Cpu size={16} />}
                              name={preset.name}
                              meta={preset.sizeLabel}
                              action={installed ? (
                                <span className="text-xs text-gray-400">Installed</span>
                              ) : isThisDownloading ? (
                                <RowIconButton onClick={handleCancelNativeDownload} title="Cancel"><StopCircle size={14} /></RowIconButton>
                              ) : (
                                <RowButtonPrimary disabled={isAnyNativeBusy && !isThisDownloading} onClick={() => handleDownloadPreset(preset)}>Download</RowButtonPrimary>
                              )}
                            />
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {advancedTab === 'servers' && (
              <div className="space-y-5 text-sm">
                {isTauriApp && (
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-medium text-gray-800">Local server</span>
                      <span className="flex items-center gap-1.5">
                        <span className={`text-xs ${localServerOnline ? 'text-green-600' : 'text-gray-400'}`}>{localServerOnline ? 'Online' : 'Offline'}</span>
                        <RowIconButton onClick={checkLocalServer} title="Re-check"><RefreshCw size={13} /></RowIconButton>
                      </span>
                    </div>
                    <div className="flex gap-2">
                      <input type="text" value={inferenceUrlInput} onChange={(e) => setInferenceUrlInput(e.target.value)}
                        placeholder="http://localhost:11434" className="flex-1 min-w-0 p-2 text-sm border border-gray-200 rounded-md" />
                      <RowButtonPrimary onClick={() => onSetAppInferenceUrl?.(inferenceUrlInput)} className="px-3 py-2">Save</RowButtonPrimary>
                    </div>
                  </div>
                )}

                <div>
                  <p className="font-medium text-gray-800 mb-1">Custom servers</p>
                  <p className="text-xs text-gray-400 mb-2">Direct connection — may fail if the server blocks cross-origin requests</p>
                  {!isAddingServer ? (
                    <button onClick={() => setIsAddingServer(true)} className="w-full py-2 border border-dashed border-gray-200 rounded-md text-xs text-gray-500 hover:border-gray-300 hover:text-gray-700 flex items-center justify-center gap-1.5">
                      <Plus size={13} /> Add server
                    </button>
                  ) : (
                    <div className="space-y-2">
                      <input type="text" value={newServerAddress} onChange={(e) => { setNewServerAddress(e.target.value); setAddError(''); }}
                        placeholder="http://192.168.1.100:8080" autoFocus className="w-full p-2 text-sm border border-gray-200 rounded-md" />
                      {addError && <p className="text-xs text-red-500">{addError}</p>}
                      <div className="flex gap-2">
                        <RowButtonPrimary onClick={handleAddServer} className="flex-1 py-1.5">Add</RowButtonPrimary>
                        <RowButtonGhost onClick={() => { setIsAddingServer(false); setNewServerAddress(''); setAddError(''); }} className="flex-1 py-1.5">Cancel</RowButtonGhost>
                      </div>
                    </div>
                  )}
                  {customServers.length > 0 && (
                    <div className="divide-y divide-gray-100 mt-2">
                      {customServers.map(server => (
                        <div key={server.address} className="flex items-center justify-between gap-2 py-2">
                          <div className="min-w-0">
                            <p className="text-sm text-gray-800 truncate">{server.address}</p>
                            <div className="flex items-center gap-1.5">
                              <span className={`text-xs ${server.status === 'online' ? 'text-green-600' : server.status === 'offline' ? 'text-red-500' : 'text-gray-400'}`}>
                                {server.status === 'online' ? 'Online' : server.status === 'offline' ? 'Offline' : 'Unchecked'}
                              </span>
                              <RowIconButton onClick={() => onCheckCustomServer?.(server.address)} title="Re-check"><RefreshCw size={12} /></RowIconButton>
                            </div>
                          </div>
                          <div className="flex items-center gap-1.5 flex-shrink-0">
                            <button
                              onClick={() => onToggleCustomServer?.(server.address)}
                              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${server.enabled ? 'bg-gray-800' : 'bg-gray-300'}`}
                            >
                              <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${server.enabled ? 'translate-x-5' : 'translate-x-1'}`} />
                            </button>
                            <RowIconButton onClick={() => onRemoveCustomServer?.(server.address)} title="Remove"><Trash2 size={14} /></RowIconButton>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {availableServers.length > 0 && (
                  <div>
                    <p className="font-medium text-gray-800 mb-1 flex items-center gap-1.5"><Zap size={14} /> Pull Ollama model</p>
                    {availableServers.length > 1 && (
                      <select value={selectedServer} onChange={(e) => setSelectedServer(e.target.value)} className="w-full mb-2 p-2 text-sm border border-gray-200 rounded-md">
                        {availableServers.map(s => <option key={s} value={s}>{s}</option>)}
                      </select>
                    )}
                    {downloadState.status !== 'pulling' && downloadState.status !== 'success' && downloadState.status !== 'error' && (
                      <div className="flex gap-2">
                        <input type="text" list="ollama-model-suggestions" value={modelToPull} onChange={(e) => setModelToPull(e.target.value)}
                          placeholder="Model name…" className="flex-1 min-w-0 p-2 text-sm border border-gray-200 rounded-md" />
                        <datalist id="ollama-model-suggestions">
                          {SUGGESTED_OLLAMA_MODELS.map(m => <option key={m} value={m} />)}
                        </datalist>
                        <RowButtonPrimary disabled={!selectedServer || !modelToPull.trim()} onClick={handleStartPull} className="px-3 py-2">Pull</RowButtonPrimary>
                      </div>
                    )}
                    {downloadState.status === 'pulling' && (
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-xs text-gray-600">
                          <span className="truncate">{downloadState.statusText}</span>
                          <span className="font-medium">{downloadState.progress}%</span>
                        </div>
                        <div className="w-full bg-gray-100 rounded-full h-1.5">
                          <div className="h-1.5 rounded-full bg-gray-800 transition-all" style={{ width: `${downloadState.progress}%` }} />
                        </div>
                        <RowButtonGhost onClick={handleCancelPull}>Cancel</RowButtonGhost>
                      </div>
                    )}
                    {downloadState.status === 'success' && <p className="text-xs text-green-600">{downloadState.statusText}</p>}
                    {downloadState.status === 'error' && <p className="text-xs text-red-600">{downloadState.errorText}</p>}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Benchmark modal */}
      <Modal open={showBenchmark} onClose={() => setShowBenchmark(false)} className="w-full max-w-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
          <span className="text-sm font-semibold text-gray-800 flex items-center gap-2"><BarChart3 size={15} /> Performance Benchmark</span>
          <button onClick={() => setShowBenchmark(false)} className="p-1 text-gray-400 hover:text-gray-700 rounded transition-colors"><X size={16} /></button>
        </div>
        <div className="p-4 overflow-y-auto" style={{ maxHeight: 'calc(88vh - env(safe-area-inset-top) - env(safe-area-inset-bottom))' }}>
          <BenchmarkPanel isVisible={true} />
        </div>
      </Modal>
    </div>
  );
};

export default AvailableModels;
