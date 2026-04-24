// NativeLlmManager.ts - iOS native LLM inference via Tauri commands + llama.cpp
// This manager mirrors the GemmaModelManager API for platform abstraction.
// Supports downloading any GGUF model from HuggingFace URLs.

import { invoke, Channel } from '@tauri-apps/api/core';
import { Logger } from '../logging';
import {
  NativeModelInfo,
  NativeModelState,
  NativeProgressEvent,
  LocalLlmMessage,
  LlmDebugInfo,
  SamplerParams,
  GenerationMetrics,
  LocalModelEntry,
} from './types';

const NATIVE_LLM_STORAGE_KEY = 'observer-native-llm-settings';
const NATIVE_LLM_MODELS_CACHE_KEY = 'observer-native-llm-models-cache';

interface PersistedNativeSettings {
  filename: string;  // Filename of the loaded model
}

export class NativeLlmManager {
  private static instance: NativeLlmManager | null = null;
  private state: NativeModelState = {
    status: 'unloaded',
    modelId: null,
    downloadProgress: 0,
    downloadedBytes: 0,
    totalBytes: 0,
    error: null,
  };
  private stateChangeListeners: Array<(state: NativeModelState) => void> = [];
  private autoLoadTriggered = false;
  private loadedFilename: string | null = null;
  private multimodalAvailable = false; // Whether loaded model supports vision
  private currentDownloadFilename: string | null = null; // Track filename being downloaded
  private cachedModels: NativeModelInfo[] = []; // Cached list of downloaded models

  private constructor() {
    // Load cached models from localStorage for instant UI on startup
    this.loadCachedModelsFromStorage();
  }

  /**
   * Load cached models from localStorage (for instant UI on startup)
   */
  private loadCachedModelsFromStorage(): void {
    try {
      const stored = localStorage.getItem(NATIVE_LLM_MODELS_CACHE_KEY);
      if (stored) {
        this.cachedModels = JSON.parse(stored) as NativeModelInfo[];
        Logger.info('NativeLlmManager', `Loaded ${this.cachedModels.length} models from cache`);
      }
    } catch (error) {
      Logger.warn('NativeLlmManager', 'Failed to load cached models from storage');
      this.cachedModels = [];
    }
  }

  /**
   * Save cached models to localStorage
   */
  private saveCachedModelsToStorage(): void {
    try {
      localStorage.setItem(NATIVE_LLM_MODELS_CACHE_KEY, JSON.stringify(this.cachedModels));
    } catch (error) {
      Logger.warn('NativeLlmManager', 'Failed to save cached models to storage');
    }
  }

  public static getInstance(): NativeLlmManager {
    if (!NativeLlmManager.instance) {
      NativeLlmManager.instance = new NativeLlmManager();
    }
    return NativeLlmManager.instance;
  }

  public getState(): NativeModelState {
    return { ...this.state };
  }

  public onStateChange(listener: (state: NativeModelState) => void): () => void {
    this.stateChangeListeners.push(listener);
    return () => {
      this.stateChangeListeners = this.stateChangeListeners.filter(l => l !== listener);
    };
  }

  private setState(updates: Partial<NativeModelState>): void {
    this.state = { ...this.state, ...updates };
    this.stateChangeListeners.forEach(l => l(this.getState()));
  }

  /**
   * List downloaded models from the models directory (async, updates cache)
   */
  public async listModels(): Promise<NativeModelInfo[]> {
    try {
      const models = await invoke<NativeModelInfo[]>('llm_list_models');
      this.cachedModels = models;
      // Persist to localStorage for instant UI on next app startup
      this.saveCachedModelsToStorage();
      // Trigger state change so UI components re-render with updated models
      this.stateChangeListeners.forEach(l => l(this.getState()));
      return models;
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to list models: ${error}`);
      return [];
    }
  }

  /**
   * Get cached list of downloaded models (sync, for use in listModels())
   */
  public getCachedModels(): NativeModelInfo[] {
    return [...this.cachedModels];
  }

  /**
   * List all local models with unified status tracking.
   * Returns cached models with their current status.
   */
  public listLocalModels(): LocalModelEntry[] {
    return this.cachedModels.map(model => {
      // Check if this model is currently loaded OR being loaded/unloaded
      const isThisModelLoaded = this.loadedFilename === model.filename;
      const modelIdFromFilename = model.filename.replace('.gguf', '').replace('.GGUF', '');
      const isThisModelActive = isThisModelLoaded || this.state.modelId === modelIdFromFilename;

      let status: LocalModelEntry['status'] = 'unloaded';
      if (isThisModelActive) {
        if (this.state.status === 'loaded') status = 'loaded';
        else if (this.state.status === 'loading') status = 'loading';
        else if (this.state.status === 'error') status = 'error';
        // 'unloading' and 'downloading' map to 'unloaded' for the unified interface
      }

      return {
        id: model.filename,
        name: model.name,
        status,
        sizeBytes: model.sizeBytes,
        isMultimodal: isThisModelLoaded ? this.multimodalAvailable : model.isMultimodal,
      };
    });
  }

  /**
   * Refresh the cached models list (call on app start, after downloads, etc.)
   */
  public async refreshModelCache(): Promise<void> {
    await this.listModels();
  }

  /**
   * Download a GGUF model from a HuggingFace URL with progress reporting
   * @param url Full URL to the GGUF file (e.g., https://huggingface.co/user/repo/resolve/main/model.gguf)
   * @returns The filename of the downloaded model
   */
  public async downloadModel(url: string): Promise<string> {
    if (this.state.status === 'downloading') {
      throw new Error('Already downloading a model');
    }

    // Extract filename from URL for display
    const filename = url.split('/').pop() || 'model.gguf';
    this.currentDownloadFilename = filename;

    this.setState({
      status: 'downloading',
      modelId: filename.replace('.gguf', '').replace('.GGUF', '') as any,
      downloadProgress: 0,
      downloadedBytes: 0,
      totalBytes: 0,
      error: null,
    });

    Logger.info('NativeLlmManager', `Starting download: ${url}`);

    try {
      const progressChannel = new Channel<NativeProgressEvent>();

      progressChannel.onmessage = (event: NativeProgressEvent) => {
        if (event.status === 'downloading') {
          this.setState({
            downloadProgress: event.progress,
            downloadedBytes: event.downloadedBytes,
            totalBytes: event.totalBytes,
          });
        } else if (event.status === 'complete') {
          Logger.info('NativeLlmManager', `Download complete: ${filename}`);
          this.currentDownloadFilename = null;
          this.setState({
            status: 'unloaded',
            downloadProgress: 100,
          });
          // Refresh model cache so the new model shows up in listModels()
          this.refreshModelCache();
        } else if (event.status === 'error') {
          Logger.error('NativeLlmManager', `Download error: ${event.error}`);
          this.currentDownloadFilename = null;
          this.setState({
            status: 'error',
            error: event.error || 'Download failed',
          });
        }
      };

      const resultFilename = await invoke<string>('llm_download_model', {
        url,
        onProgress: progressChannel,
      });

      return resultFilename;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      Logger.error('NativeLlmManager', `Download failed: ${msg}`);
      this.currentDownloadFilename = null;
      this.setState({ status: 'error', error: msg });
      throw error;
    }
  }

  /**
   * Cancel an ongoing download - signals Rust to stop and delete partial file
   */
  public async cancelDownload(): Promise<void> {
    if (this.state.status === 'downloading') {
      Logger.info('NativeLlmManager', `Cancelling download: ${this.currentDownloadFilename}`);

      try {
        // Signal Rust to cancel the download (it will delete the partial file)
        await invoke('llm_cancel_download');
        Logger.info('NativeLlmManager', 'Cancel signal sent to backend');
      } catch (error) {
        Logger.warn('NativeLlmManager', `Cancel command failed: ${error}`);
      }

      // Reset local state
      this.currentDownloadFilename = null;
      this.setState({
        status: 'unloaded',
        modelId: null,
        downloadProgress: 0,
        downloadedBytes: 0,
        totalBytes: 0,
        error: null,
      });
    }
  }

  /**
   * Delete a downloaded model by filename
   */
  public async deleteModel(filename: string): Promise<void> {
    try {
      await invoke('llm_delete_model', { filename });
      Logger.info('NativeLlmManager', `Deleted model: ${filename}`);

      // If this was the loaded model, update state
      if (this.loadedFilename === filename) {
        this.setState({ status: 'unloaded', modelId: null });
        this.loadedFilename = null;
        this.clearPersistedSettings();
      }

      // Refresh model cache so the deleted model is removed from listModels()
      this.refreshModelCache();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      Logger.error('NativeLlmManager', `Failed to delete model: ${msg}`);
      throw error;
    }
  }

  /**
   * Load a model into memory for inference by filename.
   * Pass mmprojFilename explicitly to avoid ambiguity when multiple mmproj files exist.
   */
  public async loadModel(filename: string, mmprojFilename?: string): Promise<void> {
    if (this.state.status === 'loading') {
      Logger.warn('NativeLlmManager', 'Already loading a model');
      return;
    }

    if (this.state.status === 'loaded' && this.loadedFilename === filename) {
      Logger.warn('NativeLlmManager', 'Model already loaded');
      return;
    }

    const modelId = filename.replace('.gguf', '').replace('.GGUF', '');
    this.setState({ status: 'loading', modelId: modelId as any, error: null });
    Logger.info('NativeLlmManager', `Loading model: ${filename}, mmproj: ${mmprojFilename ?? 'auto-detect'}`);

    try {
      // Apply GPU setting before loading (must be set before load_model)
      await this.applyPersistedGpuSetting();

      await invoke('llm_load_model', { filename, mmprojFilename: mmprojFilename ?? null });
      Logger.info('NativeLlmManager', 'Model loaded successfully');
      this.loadedFilename = filename;
      this.setState({ status: 'loaded' });
      this.persistSettings(filename);

      // Refresh model cache so listModels() returns updated state
      this.refreshModelCache();

      // Check if multimodal is available after loading
      try {
        this.multimodalAvailable = await invoke<boolean>('llm_is_multimodal');
        Logger.info('NativeLlmManager', `Multimodal available: ${this.multimodalAvailable}`);
      } catch {
        this.multimodalAvailable = false;
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      Logger.error('NativeLlmManager', `Failed to load model: ${msg}`);
      this.setState({ status: 'error', error: msg });
      throw error;
    }
  }

  /**
   * Unload the current model to free memory
   */
  public async unloadModel(): Promise<void> {
    if (this.state.status === 'unloaded' || this.state.status === 'unloading') return;

    const previousModelId = this.state.modelId;
    this.setState({ status: 'unloading' });
    Logger.info('NativeLlmManager', `Unloading model: ${previousModelId}`);

    try {
      await invoke('llm_unload_model');
      Logger.info('NativeLlmManager', 'Model unloaded');
      this.loadedFilename = null;
      this.multimodalAvailable = false;
      // Don't clear persisted settings - keep them so model shows as "unloaded" in listModels()
      this.setState({ status: 'unloaded', modelId: null, error: null });

      // Refresh model cache so listModels() returns updated state
      this.refreshModelCache();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      Logger.error('NativeLlmManager', `Failed to unload model: ${msg}`);
      // Revert to loaded state on error
      this.setState({ status: 'loaded', error: msg });
    }
  }

  /**
   * Generate a response from chat messages with optional streaming
   * Supports multimodal messages if the loaded model has an mmproj file.
   *
   * Message content can be:
   * - A simple string for text-only messages
   * - An array of content parts for multimodal messages:
   *   [{ type: 'text', text: '...' }, { type: 'image', image: 'base64...' }]
   */
  public async generate(
    messages: LocalLlmMessage[],
    onToken?: (token: string) => void
  ): Promise<string> {
    if (this.state.status !== 'loaded') {
      throw new Error('Native model not loaded');
    }

    Logger.info('NativeLlmManager', `Generating response for ${messages.length} messages`);

    try {
      const tokenChannel = new Channel<string>();

      if (onToken) {
        tokenChannel.onmessage = onToken;
      }

      const result = await invoke<string>('llm_generate', {
        messages,
        onToken: tokenChannel,
      });

      Logger.info('NativeModelInfo', `Generated response from llama.cpp ${result}`);

      return result;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      Logger.error('NativeLlmManager', `Generation failed: ${msg}`);
      throw error;
    }
  }

  // State query methods
  public isReady(): boolean { return this.state.status === 'loaded'; }
  public isLoading(): boolean { return this.state.status === 'loading'; }
  public isDownloading(): boolean { return this.state.status === 'downloading'; }
  public hasError(): boolean { return this.state.status === 'error'; }
  public getError(): string | null { return this.state.error; }

  /**
   * Check if the loaded model supports multimodal (vision) input
   * Returns true if model has an associated mmproj file loaded
   */
  public isMultimodal(): boolean { return this.multimodalAvailable; }

  /**
   * Get the filename of the loaded model
   */
  public getLoadedFilename(): string | null {
    return this.loadedFilename;
  }

  /**
   * Get display name for the loaded model
   */
  public getLoadedModelName(): string | null {
    if (this.loadedFilename) {
      return this.loadedFilename.replace('.gguf', '').replace('.GGUF', '');
    }
    return null;
  }

  // Persistence methods
  private persistSettings(filename: string): void {
    try {
      const settings: PersistedNativeSettings = { filename };
      localStorage.setItem(NATIVE_LLM_STORAGE_KEY, JSON.stringify(settings));
      Logger.info('NativeLlmManager', 'Persisted model settings');
    } catch (error) {
      Logger.warn('NativeLlmManager', 'Failed to persist settings');
    }
  }

  private clearPersistedSettings(): void {
    try {
      localStorage.removeItem(NATIVE_LLM_STORAGE_KEY);
      Logger.info('NativeLlmManager', 'Cleared persisted settings');
    } catch (error) {
      Logger.warn('NativeLlmManager', 'Failed to clear settings');
    }
  }

  public getPersistedSettings(): PersistedNativeSettings | null {
    try {
      const stored = localStorage.getItem(NATIVE_LLM_STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored) as PersistedNativeSettings;
      }
    } catch (error) {
      Logger.warn('NativeLlmManager', 'Failed to read persisted settings');
    }
    return null;
  }

  public async tryAutoLoad(): Promise<void> {
    if (this.autoLoadTriggered) return;
    this.autoLoadTriggered = true;

    // Always refresh model cache on startup so downloaded models show up
    const models = await this.listModels();
    Logger.info('NativeLlmManager', `Found ${models.length} downloaded models`);

    if (this.state.status !== 'unloaded') return;

    const settings = this.getPersistedSettings();
    if (settings) {
      Logger.info('NativeLlmManager', `Auto-loading persisted model: ${settings.filename}`);

      try {
        // Apply GPU setting before loading
        await this.applyPersistedGpuSetting();

        // Check if model file still exists
        const model = models.find(m => m.filename === settings.filename);
        if (model) {
          await this.loadModel(settings.filename, model.mmprojFilename);
        } else {
          Logger.warn('NativeLlmManager', 'Persisted model not found, clearing settings');
          this.clearPersistedSettings();
        }
      } catch (error) {
        Logger.error('NativeLlmManager', `Auto-load failed: ${error}`);
      }
    }
  }

  /**
   * Get detailed debug state from the native LLM engine
   * Useful for debugging when logs aren't accessible
   */
  public async getDebugState(): Promise<{
    modelsDir: string;
    modelsDirExists: boolean;
    modelFiles: string[];
    engine: {
      initialized: boolean;
      isLoaded: boolean;
      loadedModelId: string | null;
      isMultimodal: boolean;
      error?: string;
    };
  }> {
    try {
      const state = await invoke<any>('llm_debug_state');
      Logger.info('NativeLlmManager', `Debug state: ${JSON.stringify(state)}`);
      return state;
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to get debug state: ${error}`);
      throw error;
    }
  }

  /**
   * Get comprehensive debug info including sampler params and metrics
   */
  public async getDebugInfo(): Promise<LlmDebugInfo> {
    try {
      const info = await invoke<LlmDebugInfo>('llm_get_debug_info');
      Logger.info('NativeLlmManager', `Debug info retrieved`);
      return info;
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to get debug info: ${error}`);
      throw error;
    }
  }

  /**
   * Set sampler parameters for text generation
   */
  public async setSamplerParams(params: Partial<SamplerParams>): Promise<void> {
    try {
      await invoke('llm_set_sampler_params', {
        temperature: params.temperature,
        topP: params.topP,
        topK: params.topK,
        seed: params.seed,
        repeatPenalty: params.repeatPenalty,
      });
      Logger.info('NativeLlmManager', `Sampler params updated: ${JSON.stringify(params)}`);
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to set sampler params: ${error}`);
      throw error;
    }
  }

  /**
   * Set whether to use GPU acceleration (Metal)
   * Must be called BEFORE loading a model to take effect.
   * Setting is persisted to localStorage.
   * @param useGpu true for GPU (faster but may cause issues on some hardware), false for CPU (safer)
   */
  public async setUseGpu(useGpu: boolean): Promise<void> {
    try {
      await invoke('llm_set_use_gpu', { useGpu });
      // Persist the setting
      localStorage.setItem('observer-native-llm-use-gpu', JSON.stringify(useGpu));
      Logger.info('NativeLlmManager', `GPU mode set to: ${useGpu}`);
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to set GPU mode: ${error}`);
      throw error;
    }
  }

  /**
   * Get whether GPU acceleration is enabled
   */
  public async getUseGpu(): Promise<boolean> {
    try {
      const useGpu = await invoke<boolean>('llm_get_use_gpu');
      return useGpu;
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to get GPU mode: ${error}`);
      return false; // Default to CPU on error
    }
  }

  /**
   * Get persisted GPU setting from localStorage
   * Returns false (CPU mode) if not set
   */
  public getPersistedUseGpu(): boolean {
    try {
      const stored = localStorage.getItem('observer-native-llm-use-gpu');
      if (stored) {
        return JSON.parse(stored) as boolean;
      }
    } catch (error) {
      Logger.warn('NativeLlmManager', 'Failed to read persisted GPU setting');
    }
    return false; // Default to CPU mode for maximum compatibility
  }

  /**
   * Apply persisted GPU setting to the engine
   * Call this before loading a model
   */
  public async applyPersistedGpuSetting(): Promise<void> {
    const useGpu = this.getPersistedUseGpu();
    await this.setUseGpu(useGpu);
  }

  /**
   * Test generation with a simple prompt, returns response and metrics
   */
  public async testGenerate(
    prompt: string,
    maxTokens?: number,
    onToken?: (token: string) => void
  ): Promise<{ response: string; metrics: GenerationMetrics | null }> {
    if (this.state.status !== 'loaded') {
      throw new Error('Native model not loaded');
    }

    Logger.info('NativeLlmManager', `Test generate: "${prompt.substring(0, 50)}..."`);

    try {
      const tokenChannel = new Channel<string>();

      if (onToken) {
        tokenChannel.onmessage = onToken;
      }

      const result = await invoke<{ response: string; metrics: GenerationMetrics | null }>('llm_test_generate', {
        prompt,
        maxTokens: maxTokens ?? 256,
        onToken: tokenChannel,
      });

      Logger.info('NativeLlmManager', `Test generate complete: ${result.metrics?.tokensGenerated ?? 0} tokens`);
      return result;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      Logger.error('NativeLlmManager', `Test generate failed: ${msg}`);
      throw error;
    }
  }
}
