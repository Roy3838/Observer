import { GemmaModelId, GemmaDevice, GemmaDtype, GemmaImageTokenBudget, GemmaModelState, GemmaProgressItem, GemmaMessage, GemmaLoadSettings, LocalModelEntry, GEMMA_DISPLAY_NAMES } from './types';
import { Logger } from '../logging';

// Tracks which models have been downloaded, keyed by modelId → dtype.
// dtype is the only setting that determines which weight files are cached.
const GEMMA_INSTALL_KEY = 'observer-gemma-installs';
type InstallRecord = { dtype: GemmaDtype };
type InstallMap = { [modelId: string]: InstallRecord };

// Runtime settings apply at inference time and are independent of which weights
// are cached. Stored globally (not per-model) so changes take effect on the
// next load without re-downloading.
const GEMMA_RUNTIME_KEY = 'observer-gemma-runtime';
type RuntimeSettings = { device: GemmaDevice; imageTokenBudget: GemmaImageTokenBudget; enableThinking: boolean };

// Legacy keys — read once for migration, then removed.
const LEGACY_V2_KEY  = 'observer-gemma-model-settings-v2';
const LEGACY_V1_KEY  = 'observer-gemma-model-settings';

const DEFAULT_DTYPE:   GemmaDtype            = 'q4';
const DEFAULT_RUNTIME: RuntimeSettings       = { device: 'webgpu', imageTokenBudget: 70, enableThinking: false };

export class GemmaModelManager {
  private static instance: GemmaModelManager | null = null;
  private worker: Worker | null = null;
  private state: GemmaModelState = {
    status: 'unloaded',
    modelId: null,
    progress: [],
    error: null,
    loadSettings: null,
  };
  private stateChangeListeners: Array<(state: GemmaModelState) => void> = [];
  private pendingGenerations = new Map<number, { resolve: (text: string) => void; reject: (err: Error) => void; onToken?: (t: string) => void; onReasoningToken?: (t: string) => void }>();
  private nextGenerationId = 0;
  private autoLoadTriggered = false;
  private currentLoadSettings: GemmaLoadSettings | null = null;

  private constructor() {}

  public static getInstance(): GemmaModelManager {
    if (!GemmaModelManager.instance) {
      GemmaModelManager.instance = new GemmaModelManager();
    }
    return GemmaModelManager.instance;
  }

  public getState(): GemmaModelState {
    return { ...this.state };
  }

  public onStateChange(listener: (state: GemmaModelState) => void): () => void {
    this.stateChangeListeners.push(listener);
    return () => {
      this.stateChangeListeners = this.stateChangeListeners.filter(l => l !== listener);
    };
  }

  private setState(updates: Partial<GemmaModelState>): void {
    this.state = { ...this.state, ...updates };
    this.stateChangeListeners.forEach(l => l(this.getState()));
  }

  /**
   * Load a model using saved settings (or defaults if none saved).
   * This is the primary API - settings are automatically fetched from storage.
   */
  public async loadModel(modelId: GemmaModelId): Promise<void> {
    const dtype = this.getInstalledDtype(modelId);
    const runtime = this.getRuntimeSettings();
    return this.loadModelWithSettings(modelId, runtime.device, dtype, runtime.imageTokenBudget, runtime.enableThinking);
  }

  /**
   * Load a model with explicit settings. Saves settings for future loads.
   * Use this when the user explicitly changes settings in the UI.
   */
  public async loadModelWithSettings(
    modelId: GemmaModelId,
    device: GemmaDevice,
    dtype: GemmaDtype,
    imageTokenBudget: GemmaImageTokenBudget,
    enableThinking: boolean = false,
  ): Promise<void> {
    if (this.state.status === 'loading') {
      Logger.warn('GemmaModelManager', 'Model already loading');
      return;
    }

    if (this.state.status === 'loaded' && this.state.modelId === modelId) {
      Logger.warn('GemmaModelManager', 'Model already loaded');
      return;
    }

    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }

    const loadSettings: GemmaLoadSettings = { device, dtype, imageTokenBudget, enableThinking };
    this.setState({ status: 'loading', modelId, progress: [], error: null, loadSettings });
    this.currentLoadSettings = loadSettings;
    Logger.info('GemmaModelManager', `Loading model: ${modelId} (device: ${device}, dtype: ${dtype}, imageTokenBudget: ${imageTokenBudget}), thinking: ${enableThinking}`);

    this.worker = new Worker(new URL('./gemma.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = this.handleWorkerMessage.bind(this);
    this.worker.onerror = this.handleWorkerError.bind(this);

    this.worker.postMessage({ type: 'load', data: { modelId, device, dtype, imageTokenBudget } });
  }

  public unloadModel(): void {
    if (this.state.status === 'unloaded') return;

    this.pendingGenerations.forEach(({ reject }) => reject(new Error('Model unloaded')));
    this.pendingGenerations.clear();

    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }

    // Note: We don't clear persisted settings on unload - they're per-model preferences
    this.currentLoadSettings = null;
    this.setState({ status: 'unloaded', modelId: null, progress: [], error: null, loadSettings: null });
  }

  public async generate(
    messages: GemmaMessage[],
    onToken?: (token: string) => void,
    enableThinking?: boolean,
    onReasoningToken?: (token: string) => void,
  ): Promise<string> {
    if (this.state.status !== 'loaded' || !this.worker) {
      throw new Error('Gemma model not loaded');
    }

    const generationId = this.nextGenerationId++;

    return new Promise((resolve, reject) => {
      this.pendingGenerations.set(generationId, { resolve, reject, onToken, onReasoningToken });
      this.worker!.postMessage({ type: 'generate', data: { messages, generationId, enableThinking: !!enableThinking } });
    });
  }

  private handleWorkerMessage(event: MessageEvent): void {
    const { type, data } = event.data;

    switch (type) {
      case 'progress':
        this.handleProgress(data);
        break;

      case 'ready':
        Logger.info('GemmaModelManager', 'Model loaded successfully');
        if (this.state.modelId && this.currentLoadSettings) {
          // Record install and runtime settings BEFORE notifying listeners so that
          // listLocalModels() reflects the new model when the UI re-renders.
          this.recordInstall(this.state.modelId as GemmaModelId, this.currentLoadSettings.dtype);
          this.saveRuntimeSettings({
            device: this.currentLoadSettings.device,
            imageTokenBudget: this.currentLoadSettings.imageTokenBudget,
            enableThinking: this.currentLoadSettings.enableThinking ?? false,
          });
        }
        this.setState({ status: 'loaded', progress: [] });
        break;

      case 'generation-token': {
        const pending = this.pendingGenerations.get(data.generationId);
        if (pending?.onToken) pending.onToken(data.token);
        break;
      }

      case 'reasoning-token': {
        const pending = this.pendingGenerations.get(data.generationId);
        if (pending?.onReasoningToken) pending.onReasoningToken(data.token);
        break;
      }

      case 'generation-complete': {
        const pending = this.pendingGenerations.get(data.generationId);
        if (pending) {
          this.pendingGenerations.delete(data.generationId);
          pending.resolve(data.text);
        }
        break;
      }

      case 'error': {
        const msg = data.message as string;
        if (data.generationId !== undefined && this.pendingGenerations.has(data.generationId)) {
          const pending = this.pendingGenerations.get(data.generationId)!;
          this.pendingGenerations.delete(data.generationId);
          pending.reject(new Error(msg));
        } else {
          Logger.error('GemmaModelManager', `Worker error: ${msg}`);
          this.setState({ status: 'error', error: msg });
        }
        break;
      }

      default:
        Logger.warn('GemmaModelManager', `Unknown worker message: ${type}`);
    }
  }

  private handleProgress(info: any): void {
    // Transformers.js from_pretrained emits { status, file, progress, loaded, total }
    // Skip non-file events (initiate, ready, etc.)
    if (!info.file) return;

    const item: GemmaProgressItem = {
      file: info.file,
      progress: info.progress ?? (info.status === 'done' ? 100 : 0),
      loaded: info.loaded ?? 0,
      total: info.total ?? 0,
      status: info.status === 'done' ? 'done' : 'progress',
    };

    const existing = this.state.progress.findIndex(p => p.file === item.file);
    const updated = [...this.state.progress];
    if (existing !== -1) {
      updated[existing] = item;
    } else {
      updated.push(item);
    }
    this.setState({ progress: updated });
  }

  private handleWorkerError(error: ErrorEvent): void {
    const msg = `Worker error: ${error.message}`;
    Logger.error('GemmaModelManager', msg);
    this.pendingGenerations.forEach(({ reject }) => reject(new Error(msg)));
    this.pendingGenerations.clear();
    this.setState({ status: 'error', error: msg });
  }

  public isReady(): boolean { return this.state.status === 'loaded'; }
  public isLoading(): boolean { return this.state.status === 'loading'; }
  public hasError(): boolean { return this.state.status === 'error'; }
  public getError(): string | null { return this.state.error; }

  // ============================================================================
  // Install records  (per-model, dtype only)
  // ============================================================================

  private getInstallMap(): InstallMap {
    try {
      const stored = localStorage.getItem(GEMMA_INSTALL_KEY);
      if (stored) return JSON.parse(stored) as InstallMap;

      // Migrate from v2 format: extract dtype per model, extract runtime from first entry
      const v2 = localStorage.getItem(LEGACY_V2_KEY);
      if (v2) {
        const v2map = JSON.parse(v2) as Record<string, { dtype?: GemmaDtype; device?: GemmaDevice; imageTokenBudget?: GemmaImageTokenBudget }>;
        const installs: InstallMap = {};
        let migratedRuntime: Partial<RuntimeSettings> = {};
        for (const [id, s] of Object.entries(v2map)) {
          installs[id] = { dtype: s.dtype ?? DEFAULT_DTYPE };
          if (!migratedRuntime.device && s.device) migratedRuntime = { device: s.device, imageTokenBudget: s.imageTokenBudget ?? DEFAULT_RUNTIME.imageTokenBudget, enableThinking: false };
        }
        localStorage.setItem(GEMMA_INSTALL_KEY, JSON.stringify(installs));
        if (migratedRuntime.device) localStorage.setItem(GEMMA_RUNTIME_KEY, JSON.stringify({ ...DEFAULT_RUNTIME, ...migratedRuntime }));
        localStorage.removeItem(LEGACY_V2_KEY);
        Logger.info('GemmaModelManager', 'Migrated v2 settings to install/runtime split');
        return installs;
      }

      // Migrate from v1 format
      const v1 = localStorage.getItem(LEGACY_V1_KEY);
      if (v1) {
        const s = JSON.parse(v1) as { modelId?: GemmaModelId; dtype?: GemmaDtype; device?: GemmaDevice; imageTokenBudget?: GemmaImageTokenBudget };
        const installs: InstallMap = s.modelId ? { [s.modelId]: { dtype: s.dtype ?? DEFAULT_DTYPE } } : {};
        localStorage.setItem(GEMMA_INSTALL_KEY, JSON.stringify(installs));
        if (s.device) localStorage.setItem(GEMMA_RUNTIME_KEY, JSON.stringify({ device: s.device, imageTokenBudget: s.imageTokenBudget ?? DEFAULT_RUNTIME.imageTokenBudget, enableThinking: false }));
        localStorage.removeItem(LEGACY_V1_KEY);
        return installs;
      }
    } catch {
      Logger.warn('GemmaModelManager', 'Failed to read install map');
    }
    return {};
  }

  private saveInstallMap(map: InstallMap): void {
    localStorage.setItem(GEMMA_INSTALL_KEY, JSON.stringify(map));
  }

  private getInstalledDtype(modelId: GemmaModelId): GemmaDtype {
    return this.getInstallMap()[modelId]?.dtype ?? DEFAULT_DTYPE;
  }

  private recordInstall(modelId: GemmaModelId, dtype: GemmaDtype): void {
    const map = this.getInstallMap();
    map[modelId] = { dtype };
    this.saveInstallMap(map);
  }

  public deleteModel(modelId: GemmaModelId): void {
    try {
      if (this.state.modelId === modelId) this.unloadModel();
      const map = this.getInstallMap();
      delete map[modelId];
      this.saveInstallMap(map);
      Logger.info('GemmaModelManager', `Deleted model from storage: ${modelId}`);
      this.stateChangeListeners.forEach(l => l(this.getState()));
    } catch {
      Logger.warn('GemmaModelManager', `Failed to delete model ${modelId}`);
    }
  }

  // ============================================================================
  // Runtime settings  (global, not tied to a specific model download)
  // ============================================================================

  public getRuntimeSettings(): RuntimeSettings {
    try {
      const stored = localStorage.getItem(GEMMA_RUNTIME_KEY);
      if (stored) return { ...DEFAULT_RUNTIME, ...JSON.parse(stored) };
    } catch {}
    return { ...DEFAULT_RUNTIME };
  }

  public saveRuntimeSettings(settings: RuntimeSettings): void {
    localStorage.setItem(GEMMA_RUNTIME_KEY, JSON.stringify(settings));
  }

  // ============================================================================
  // Combined view (for UI / ModelManager)
  // ============================================================================

  /** Full settings for a model: install dtype + current runtime prefs. */
  public getSettingsForModel(modelId: GemmaModelId): GemmaLoadSettings {
    const dtype = this.getInstalledDtype(modelId);
    return { ...this.getRuntimeSettings(), dtype };
  }

  public getLastLoadedModelId(): GemmaModelId | null {
    const ids = Object.keys(this.getInstallMap()) as GemmaModelId[];
    return ids.length > 0 ? ids[0] : null;
  }

  public listLocalModels(): LocalModelEntry[] {
    const entries: LocalModelEntry[] = [];
    for (const modelId of Object.keys(this.getInstallMap())) {
      const displayName = GEMMA_DISPLAY_NAMES[modelId as GemmaModelId] ?? modelId;
      const isCurrentModel = this.state.modelId === modelId;
      let status: LocalModelEntry['status'] = 'unloaded';
      if (isCurrentModel) status = this.state.status === 'error' ? 'error' : this.state.status;
      entries.push({ id: modelId, name: displayName, status, isMultimodal: true });
    }
    return entries;
  }

  public tryAutoLoad(): void {
    if (this.autoLoadTriggered) return;
    if (this.state.status !== 'unloaded') return;
    const modelId = this.getLastLoadedModelId();
    if (modelId) {
      this.autoLoadTriggered = true;
      Logger.info('GemmaModelManager', `Auto-loading persisted model: ${modelId}`);
      this.loadModel(modelId);
    }
  }

  public clearAllPersistedSettings(): void {
    try {
      localStorage.removeItem(GEMMA_INSTALL_KEY);
      localStorage.removeItem(GEMMA_RUNTIME_KEY);
      localStorage.removeItem(LEGACY_V2_KEY);
      localStorage.removeItem(LEGACY_V1_KEY);
      Logger.info('GemmaModelManager', 'Cleared all persisted model settings');
    } catch {
      Logger.warn('GemmaModelManager', 'Failed to clear persisted settings');
    }
  }
}
