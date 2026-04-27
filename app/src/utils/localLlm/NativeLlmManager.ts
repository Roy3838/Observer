// NativeLlmManager.ts - iOS native LLM inference via Tauri commands + llama.cpp
// This manager mirrors the GemmaModelManager API for platform abstraction.
// Supports downloading any GGUF model from HuggingFace URLs.

import { invoke, Channel } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Logger, LogLevel } from '../logging';
import {
  GgufFileInfo,
  NativeModelState,
  NativeProgressEvent,
  LocalLlmMessage,
  LlmDebugInfo,
  SamplerParams,
  GenerationMetrics,
  LocalModelEntry,
} from './types';

const NATIVE_LLM_STORAGE_KEY = 'observer-native-llm-settings';
const NATIVE_LLM_GGUF_CACHE_KEY = 'observer-native-llm-gguf-cache';
const NATIVE_LLM_MMPROJ_ASSIGNMENTS_KEY = 'observer-native-llm-mmproj-assignments';

interface PersistedNativeSettings {
  filename: string;
  enableThinking: boolean;
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
    enableThinking: true,
  };
  private stateChangeListeners: Array<(state: NativeModelState) => void> = [];
  private autoLoadTriggered = false;
  private loadedFilename: string | null = null;
  private loadedMmprojFilename: string | null = null;
  private multimodalAvailable = false;
  private currentDownloadFilename: string | null = null;
  private cachedGgufFiles: GgufFileInfo[] = [];

  private constructor() {
    this.loadGgufCacheFromStorage();
    this.subscribeToEngineEvents();
  }

  private subscribeToEngineEvents(): void {
    listen<{ level: string; message: string }>('llm-log', (event) => {
      const { level, message } = event.payload;
      const logLevel =
        level === 'error' ? LogLevel.ERROR :
        level === 'warn'  ? LogLevel.WARNING :
        LogLevel.INFO;
      Logger.log(logLevel, 'LlmEngine', message);
    }).catch(() => {
      // Not in a Tauri context (e.g. browser dev) — silently ignore
    });
  }

  private loadGgufCacheFromStorage(): void {
    try {
      const stored = localStorage.getItem(NATIVE_LLM_GGUF_CACHE_KEY);
      if (stored) {
        this.cachedGgufFiles = JSON.parse(stored) as GgufFileInfo[];
      }
    } catch {
      this.cachedGgufFiles = [];
    }
  }

  private saveGgufCacheToStorage(): void {
    try {
      localStorage.setItem(NATIVE_LLM_GGUF_CACHE_KEY, JSON.stringify(this.cachedGgufFiles));
    } catch {
      Logger.warn('NativeLlmManager', 'Failed to save GGUF cache to storage');
    }
  }

  // ── mmproj assignment map ──────────────────────────────────────────────────
  // Persisted map of { [modelFilename]: mmprojFilename }.
  // The frontend owns this relationship; the backend receives it explicitly on load.

  public getMmprojAssignments(): Record<string, string> {
    try {
      const stored = localStorage.getItem(NATIVE_LLM_MMPROJ_ASSIGNMENTS_KEY);
      return stored ? JSON.parse(stored) : {};
    } catch {
      return {};
    }
  }

  public setMmprojAssignment(modelFilename: string, mmprojFilename: string | null): void {
    const assignments = this.getMmprojAssignments();
    if (mmprojFilename === null) {
      delete assignments[modelFilename];
    } else {
      assignments[modelFilename] = mmprojFilename;
    }
    try {
      localStorage.setItem(NATIVE_LLM_MMPROJ_ASSIGNMENTS_KEY, JSON.stringify(assignments));
    } catch {
      Logger.warn('NativeLlmManager', 'Failed to save mmproj assignments');
    }
    this.stateChangeListeners.forEach(l => l(this.getState()));
  }

  public getMmprojAssignment(modelFilename: string): string | null {
    return this.getMmprojAssignments()[modelFilename] ?? null;
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
   * List all GGUF files in the models directory (async, updates cache).
   * Both model files and projector files are returned — no filtering.
   */
  public async listGgufFiles(): Promise<GgufFileInfo[]> {
    try {
      const files = await invoke<GgufFileInfo[]>('llm_list_gguf');
      this.cachedGgufFiles = files;
      this.saveGgufCacheToStorage();
      this.stateChangeListeners.forEach(l => l(this.getState()));
      return files;
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to list GGUF files: ${error}`);
      return [];
    }
  }

  /** Sync access to the last-fetched GGUF file list. */
  public getCachedGgufFiles(): GgufFileInfo[] {
    return [...this.cachedGgufFiles];
  }

  /**
   * Convenience: all files whose filename does NOT suggest they are a projector.
   * Heuristic is filename-based and used for display only — not load logic.
   */
  public getCachedModelFiles(): GgufFileInfo[] {
    return this.cachedGgufFiles.filter(
      f => !f.filename.toLowerCase().includes('mmproj')
    );
  }

  /**
   * Convenience: all files that look like projectors (mmproj in filename).
   * Heuristic for display only.
   */
  public getCachedProjectorFiles(): GgufFileInfo[] {
    return this.cachedGgufFiles.filter(
      f => f.filename.toLowerCase().includes('mmproj')
    );
  }

  /**
   * List all local models with unified status tracking.
   * Returns model-looking files (non-mmproj) with their current status.
   */
  public listLocalModels(): LocalModelEntry[] {
    return this.getCachedModelFiles().map(file => {
      const isThisFile = this.loadedFilename === file.filename;
      const modelIdFromFilename = file.filename.replace('.gguf', '').replace('.GGUF', '');
      const isThisModelActive = isThisFile || this.state.modelId === modelIdFromFilename;

      let status: LocalModelEntry['status'] = 'unloaded';
      if (isThisModelActive) {
        if (this.state.status === 'loaded') status = 'loaded';
        else if (this.state.status === 'loading') status = 'loading';
        else if (this.state.status === 'error') status = 'error';
      }

      return {
        id: file.filename,
        name: file.filename.replace(/\.gguf$/i, ''),
        status,
        sizeBytes: file.sizeBytes,
        isMultimodal: isThisFile ? this.multimodalAvailable : false,
      };
    });
  }

  public async refreshGgufCache(): Promise<void> {
    await this.listGgufFiles();
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
          // Refresh GGUF file list so the new file appears
          this.refreshGgufCache();
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

      // Refresh GGUF file list so the deleted file disappears
      this.refreshGgufCache();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      Logger.error('NativeLlmManager', `Failed to delete model: ${msg}`);
      throw error;
    }
  }

  /**
   * Load a model into memory for inference.
   * If mmprojFilename is omitted, the persisted assignment map is checked.
   * If neither is set, the model loads as text-only — no auto-detection.
   */
  public async loadModel(filename: string, mmprojFilename?: string): Promise<void> {
    if (this.state.status === 'downloading') {
      throw new Error('Cannot load a model while a download is in progress');
    }

    if (this.state.status === 'loading') {
      Logger.warn('NativeLlmManager', 'Already loading a model');
      return;
    }

    if (this.state.status === 'loaded' && this.loadedFilename === filename) {
      Logger.warn('NativeLlmManager', 'Model already loaded');
      return;
    }

    const modelId = filename.replace('.gguf', '').replace('.GGUF', '');
    const resolvedMmproj = mmprojFilename ?? this.getMmprojAssignment(filename) ?? undefined;
    this.setState({ status: 'loading', modelId: modelId as any, error: null });
    Logger.info('NativeLlmManager', `Loading model: ${filename}, mmproj: ${resolvedMmproj ?? 'none'}`);

    try {
      await this.applyPersistedGpuSetting();

      await invoke('llm_load_model', { filename, mmprojFilename: resolvedMmproj ?? null });
      Logger.info('NativeLlmManager', 'Model loaded successfully');
      this.loadedFilename = filename;
      this.loadedMmprojFilename = resolvedMmproj ?? null;
      this.setState({ status: 'loaded' });
      this.persistSettings(filename);

      this.refreshGgufCache();

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
      this.loadedMmprojFilename = null;
      this.multimodalAvailable = false;
      this.setState({ status: 'unloaded', modelId: null, error: null });

      // Refresh GGUF file list so UI reflects current state
      this.refreshGgufCache();
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
  public setEnableThinking(value: boolean): void {
    this.setState({ enableThinking: value });
    const settings = this.getPersistedSettings();
    if (settings) {
      this.persistSettings(settings.filename);
    }
  }

  public async generate(
    messages: LocalLlmMessage[],
    onToken?: (token: string) => void,
    _onReasoningToken?: (token: string) => void,
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

      const result = await invoke<{ response: string; metrics: GenerationMetrics | null }>('llm_generate', {
        messages,
        enableThinking: this.state.enableThinking,
        onToken: tokenChannel,
      });

      Logger.info('NativeLlmManager', `Generated response from llama.cpp ${result.response}`);

      return result.response;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      Logger.error('NativeLlmManager', `Generation failed: ${msg}`);
      throw error;
    }
  }

  public async cancelGeneration(): Promise<void> {
    await invoke('llm_cancel_generation');
  }

  /**
   * Initialize the llama.cpp backend engine explicitly.
   * Idempotent — safe to call multiple times. On first call this triggers
   * Metal shader compilation (iOS/macOS) which can take several seconds on a
   * cold device. Call this when the user opens the model screen so the cost
   * is paid with visible UI feedback rather than silently during model load.
   */
  public async initEngine(): Promise<void> {
    await invoke('llm_init_engine');
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

  public getLoadedFilename(): string | null { return this.loadedFilename; }
  public getLoadedMmprojFilename(): string | null { return this.loadedMmprojFilename; }

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
      const settings: PersistedNativeSettings = { filename, enableThinking: this.state.enableThinking };
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

    const files = await this.listGgufFiles();
    Logger.info('NativeLlmManager', `Found ${files.length} GGUF files`);

    if (this.state.status !== 'unloaded') return;

    const settings = this.getPersistedSettings();
    if (settings) {
      Logger.info('NativeLlmManager', `Auto-loading persisted model: ${settings.filename}`);
      if (settings.enableThinking) {
        this.setState({ enableThinking: settings.enableThinking });
      }
      try {
        const exists = files.some(f => f.filename === settings.filename);
        if (exists) {
          await this.loadModel(settings.filename);
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

      const messages = [{ role: 'user', content: prompt }];
      const result = await invoke<{ response: string; metrics: GenerationMetrics | null }>('llm_generate', {
        messages,
        enableThinking: this.state.enableThinking,
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
