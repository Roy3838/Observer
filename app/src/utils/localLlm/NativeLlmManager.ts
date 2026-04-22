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
} from './types';

const NATIVE_LLM_STORAGE_KEY = 'observer-native-llm-settings';

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

  private constructor() {}

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
   * List downloaded models from the models directory
   */
  public async listModels(): Promise<NativeModelInfo[]> {
    try {
      const models = await invoke<NativeModelInfo[]>('llm_list_models');
      return models;
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to list models: ${error}`);
      return [];
    }
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
          this.setState({
            status: 'unloaded',
            downloadProgress: 100,
          });
        } else if (event.status === 'error') {
          Logger.error('NativeLlmManager', `Download error: ${event.error}`);
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
      this.setState({ status: 'error', error: msg });
      throw error;
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
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      Logger.error('NativeLlmManager', `Failed to delete model: ${msg}`);
      throw error;
    }
  }

  /**
   * Load a model into memory for inference by filename
   */
  public async loadModel(filename: string): Promise<void> {
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
    Logger.info('NativeLlmManager', `Loading model: ${filename}`);

    try {
      await invoke('llm_load_model', { filename });
      Logger.info('NativeLlmManager', 'Model loaded successfully');
      this.loadedFilename = filename;
      this.setState({ status: 'loaded' });
      this.persistSettings(filename);

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
    if (this.state.status === 'unloaded') return;

    try {
      await invoke('llm_unload_model');
      Logger.info('NativeLlmManager', 'Model unloaded');
      this.loadedFilename = null;
      this.multimodalAvailable = false;
      this.clearPersistedSettings();
      this.setState({ status: 'unloaded', modelId: null, error: null });
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to unload model: ${error}`);
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
    if (this.state.status !== 'unloaded') return;

    const settings = this.getPersistedSettings();
    if (settings) {
      this.autoLoadTriggered = true;
      Logger.info('NativeLlmManager', `Auto-loading persisted model: ${settings.filename}`);

      try {
        // Check if model file still exists
        const models = await this.listModels();
        const model = models.find(m => m.filename === settings.filename);
        if (model) {
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
}
