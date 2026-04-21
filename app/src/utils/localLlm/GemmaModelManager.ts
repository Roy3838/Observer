import { GemmaModelId, GemmaDevice, GemmaDtype, GemmaImageTokenBudget, GemmaModelState, GemmaProgressItem, GemmaMessage } from './types';
import { Logger } from '../logging';

const GEMMA_STORAGE_KEY = 'observer-gemma-model-settings';

interface PersistedGemmaSettings {
  modelId: GemmaModelId;
  device: GemmaDevice;
  dtype: GemmaDtype;
  imageTokenBudget: GemmaImageTokenBudget;
}

export class GemmaModelManager {
  private static instance: GemmaModelManager | null = null;
  private worker: Worker | null = null;
  private state: GemmaModelState = {
    status: 'unloaded',
    modelId: null,
    progress: [],
    error: null,
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

  public async loadModel(
    modelId: GemmaModelId,
    device: GemmaDevice = 'webgpu',
    dtype: GemmaDtype = 'q4f16',
    imageTokenBudget: GemmaImageTokenBudget = 280
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

    this.setState({ status: 'loading', modelId, progress: [], error: null });
    this.currentLoadSettings = { device, dtype, imageTokenBudget };
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

    this.clearPersistedSettings();
    this.currentLoadSettings = null;
    this.setState({ status: 'unloaded', modelId: null, progress: [], error: null });
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
        // Persist settings for auto-load on next app start
        if (this.state.modelId && this.currentLoadSettings) {
          this.persistSettings(
            this.state.modelId,
            this.currentLoadSettings.device,
            this.currentLoadSettings.dtype,
            this.currentLoadSettings.imageTokenBudget
          );
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

  // Persistence methods for auto-load on app restart
  private persistSettings(modelId: GemmaModelId, device: GemmaDevice, dtype: GemmaDtype, imageTokenBudget: GemmaImageTokenBudget): void {
    try {
      const settings: PersistedGemmaSettings = { modelId, device, dtype, imageTokenBudget };
      localStorage.setItem(GEMMA_STORAGE_KEY, JSON.stringify(settings));
      Logger.info('GemmaModelManager', 'Persisted model settings to localStorage');
    } catch (error) {
      Logger.warn('GemmaModelManager', 'Failed to persist settings to localStorage');
    }
  }

  private clearPersistedSettings(): void {
    try {
      localStorage.removeItem(GEMMA_STORAGE_KEY);
      Logger.info('GemmaModelManager', 'Cleared persisted model settings');
    } catch (error) {
      Logger.warn('GemmaModelManager', 'Failed to clear persisted settings');
    }
  }

  public getPersistedSettings(): PersistedGemmaSettings | null {
    try {
      const stored = localStorage.getItem(GEMMA_STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored) as PersistedGemmaSettings;
      }
    } catch (error) {
      Logger.warn('GemmaModelManager', 'Failed to read persisted settings');
    }
    return null;
  }

  public tryAutoLoad(): void {
    if (this.autoLoadTriggered) return;
    if (this.state.status !== 'unloaded') return;

    const settings = this.getPersistedSettings();
    if (settings) {
      this.autoLoadTriggered = true;
      Logger.info('GemmaModelManager', `Auto-loading persisted model: ${settings.modelId}`);
      this.loadModel(settings.modelId, settings.device, settings.dtype, settings.imageTokenBudget);
    }
  }
}
