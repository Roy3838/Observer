import { GemmaModelId, GemmaDevice, GemmaDtype, GemmaImageTokenBudget, GemmaModelState, GemmaProgressItem, GemmaMessage, GemmaLoadSettings } from './types';
import { Logger } from '../logging';

const GEMMA_STORAGE_KEY = 'observer-gemma-model-settings-v2';

// Per-model settings map
type PersistedGemmaSettingsMap = {
  [modelId: string]: GemmaLoadSettings;
};

// Legacy format for migration
interface LegacyPersistedSettings {
  modelId: GemmaModelId;
  device: GemmaDevice;
  dtype: GemmaDtype;
  imageTokenBudget: GemmaImageTokenBudget;
}
const LEGACY_STORAGE_KEY = 'observer-gemma-model-settings';

const DEFAULT_SETTINGS: GemmaLoadSettings = {
  device: 'webgpu',
  dtype: 'q4',
  imageTokenBudget: 70,
};

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
  private pendingGenerations = new Map<number, { resolve: (text: string) => void; reject: (err: Error) => void; onToken?: (t: string) => void }>();
  private nextGenerationId = 0;
  private autoLoadTriggered = false;
  private currentLoadSettings: { device: GemmaDevice; dtype: GemmaDtype; imageTokenBudget: GemmaImageTokenBudget } | null = null;

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
    const settings = this.getSettingsForModel(modelId);
    return this.loadModelWithSettings(modelId, settings.device, settings.dtype, settings.imageTokenBudget);
  }

  /**
   * Load a model with explicit settings. Saves settings for future loads.
   * Use this when the user explicitly changes settings in the UI.
   */
  public async loadModelWithSettings(
    modelId: GemmaModelId,
    device: GemmaDevice,
    dtype: GemmaDtype,
    imageTokenBudget: GemmaImageTokenBudget
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

    const loadSettings: GemmaLoadSettings = { device, dtype, imageTokenBudget };
    this.setState({ status: 'loading', modelId, progress: [], error: null, loadSettings });
    this.currentLoadSettings = loadSettings;
    Logger.info('GemmaModelManager', `Loading model: ${modelId} (device: ${device}, dtype: ${dtype}, imageTokenBudget: ${imageTokenBudget})`);

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
    onToken?: (token: string) => void
  ): Promise<string> {
    if (this.state.status !== 'loaded' || !this.worker) {
      throw new Error('Gemma model not loaded');
    }

    const generationId = this.nextGenerationId++;

    return new Promise((resolve, reject) => {
      this.pendingGenerations.set(generationId, { resolve, reject, onToken });
      this.worker!.postMessage({ type: 'generate', data: { messages, generationId } });
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
        this.setState({ status: 'loaded', progress: [] });
        // Persist settings for this model
        if (this.state.modelId && this.currentLoadSettings) {
          this.persistSettingsForModel(this.state.modelId, this.currentLoadSettings);
        }
        break;

      case 'generation-token': {
        const pending = this.pendingGenerations.get(data.generationId);
        if (pending?.onToken) pending.onToken(data.token);
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
  // Per-model settings persistence
  // ============================================================================

  private getAllPersistedSettings(): PersistedGemmaSettingsMap {
    try {
      // Try new format first
      const stored = localStorage.getItem(GEMMA_STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored) as PersistedGemmaSettingsMap;
      }

      // Migrate from legacy format if exists
      const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);
      if (legacy) {
        const legacySettings = JSON.parse(legacy) as LegacyPersistedSettings;
        const migrated: PersistedGemmaSettingsMap = {
          [legacySettings.modelId]: {
            device: legacySettings.device,
            dtype: legacySettings.dtype,
            imageTokenBudget: legacySettings.imageTokenBudget,
          },
        };
        // Save in new format and remove legacy
        localStorage.setItem(GEMMA_STORAGE_KEY, JSON.stringify(migrated));
        localStorage.removeItem(LEGACY_STORAGE_KEY);
        Logger.info('GemmaModelManager', 'Migrated legacy settings to per-model format');
        return migrated;
      }
    } catch (error) {
      Logger.warn('GemmaModelManager', 'Failed to read persisted settings');
    }
    return {};
  }

  private persistSettingsForModel(modelId: GemmaModelId, settings: GemmaLoadSettings): void {
    try {
      const all = this.getAllPersistedSettings();
      all[modelId] = settings;
      localStorage.setItem(GEMMA_STORAGE_KEY, JSON.stringify(all));
      Logger.info('GemmaModelManager', `Persisted settings for ${modelId}`);
    } catch (error) {
      Logger.warn('GemmaModelManager', 'Failed to persist settings to localStorage');
    }
  }

  /**
   * Get saved settings for a model, or defaults if none saved.
   * Useful for populating UI dropdowns.
   */
  public getSettingsForModel(modelId: GemmaModelId): GemmaLoadSettings {
    const all = this.getAllPersistedSettings();
    return all[modelId] ?? { ...DEFAULT_SETTINGS };
  }

  /**
   * Get the last loaded model ID (for auto-load on app restart).
   * Returns the model that was most recently persisted.
   */
  public getLastLoadedModelId(): GemmaModelId | null {
    // For auto-load, we check which models have saved settings
    // Since we persist on successful load, any model with settings was loaded before
    const all = this.getAllPersistedSettings();
    const modelIds = Object.keys(all) as GemmaModelId[];
    // Return the first one (could enhance to track "last used" timestamp)
    return modelIds.length > 0 ? modelIds[0] : null;
  }

  public tryAutoLoad(): void {
    if (this.autoLoadTriggered) return;
    if (this.state.status !== 'unloaded') return;

    const modelId = this.getLastLoadedModelId();
    if (modelId) {
      this.autoLoadTriggered = true;
      Logger.info('GemmaModelManager', `Auto-loading persisted model: ${modelId}`);
      this.loadModel(modelId); // Will auto-fetch saved settings
    }
  }

  /**
   * Clear all persisted settings (for testing/reset).
   */
  public clearAllPersistedSettings(): void {
    try {
      localStorage.removeItem(GEMMA_STORAGE_KEY);
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      Logger.info('GemmaModelManager', 'Cleared all persisted model settings');
    } catch (error) {
      Logger.warn('GemmaModelManager', 'Failed to clear persisted settings');
    }
  }
}
