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
  ContextParams,
  GenerationMetrics,
  LocalModelEntry,
  LocalFileState,
  NativeLocalModel,
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
    enableThinking: false,
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

  /**
   * Recover from a cancelled projector download: unassign the projector
   * from the model and delete its .part file from disk. Called from the UI
   * when a partial projector blocks loading. If the assigned projector is
   * already complete (.gguf on disk), nothing is deleted — the dropdown's
   * "No projector" option handles that case.
   */
  public async discardPartialProjector(modelFilename: string): Promise<void> {
    const assignment = this.getMmprojAssignment(modelFilename);
    if (!assignment) return;

    const partFilename = `${assignment}.part`;
    const hasPart = this.cachedGgufFiles.some(f => f.filename === partFilename);

    this.setMmprojAssignment(modelFilename, null);

    if (hasPart) {
      try {
        await invoke('llm_delete_model', { filename: partFilename });
        Logger.info('NativeLlmManager', `Discarded partial projector: ${partFilename}`);
      } catch (error) {
        Logger.warn('NativeLlmManager', `Failed to delete partial projector ${partFilename}: ${error}`);
      }
      await this.refreshGgufCache();
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
   * Logical-model view of the registry.
   *
   * One entry per non-mmproj file (model). Each entry pairs that file with its
   * assigned projector (if any). The view is fully derived — no extra storage.
   *
   * Sources of truth:
   *   - cachedGgufFiles      : on-disk .gguf and .gguf.part files
   *   - mmprojAssignments    : user-defined model→projector pairing
   *   - state                : in-flight download progress + runtime status
   *   - loadedFilename       : currently loaded model
   */
  public listNativeModels(): NativeLocalModel[] {
    const assignments = this.getMmprojAssignments();

    // Group on-disk files by canonical name (strip .part). When both a .gguf
    // and a .gguf.part exist for the same name, prefer the complete one.
    const filesByCanonical = new Map<string, { sizeBytes: number; isPart: boolean }>();
    for (const file of this.cachedGgufFiles) {
      const isPart = file.filename.endsWith('.part');
      const canonical = isPart ? file.filename.slice(0, -'.part'.length) : file.filename;
      const existing = filesByCanonical.get(canonical);
      if (!existing || (existing.isPart && !isPart)) {
        filesByCanonical.set(canonical, { sizeBytes: file.sizeBytes, isPart });
      }
    }

    const isMmproj = (name: string) => name.toLowerCase().includes('mmproj');

    const resolveFileState = (canonicalName: string): LocalFileState => {
      const entry = filesByCanonical.get(canonicalName);
      const isActiveDownload =
        this.state.status === 'downloading' &&
        this.currentDownloadFilename === canonicalName;

      if (!entry) {
        if (isActiveDownload) {
          return {
            kind: 'partial',
            bytes: 0,
            downloading: true,
            progress: this.state.downloadProgress,
            downloadedBytes: this.state.downloadedBytes,
            totalBytes: this.state.totalBytes,
          };
        }
        return { kind: 'absent' };
      }
      if (!entry.isPart) return { kind: 'complete', bytes: entry.sizeBytes };
      return isActiveDownload
        ? {
            kind: 'partial',
            bytes: entry.sizeBytes,
            downloading: true,
            progress: this.state.downloadProgress,
            downloadedBytes: this.state.downloadedBytes,
            totalBytes: this.state.totalBytes,
          }
        : { kind: 'partial', bytes: entry.sizeBytes, downloading: false };
    };

    const models: NativeLocalModel[] = [];
    const seen = new Set<string>();

    for (const canonical of filesByCanonical.keys()) {
      if (isMmproj(canonical)) continue;
      seen.add(canonical);

      const projectorFilename = assignments[canonical] ?? null;
      const projectorFile: LocalFileState = projectorFilename
        ? resolveFileState(projectorFilename)
        : { kind: 'absent' };

      const canonicalModelId = canonical.replace(/\.gguf$/i, '');
      const isThisModelRuntime =
        this.loadedFilename === canonical ||
        (this.state.status !== 'downloading' && this.state.modelId === canonicalModelId);

      let runtime: NativeLocalModel['runtime'] = 'unloaded';
      if (isThisModelRuntime) {
        if (this.state.status === 'loaded') runtime = 'loaded';
        else if (this.state.status === 'loading') runtime = 'loading';
        else if (this.state.status === 'error') runtime = 'error';
      }

      models.push({
        id: canonical,
        name: canonicalModelId,
        modelFile: resolveFileState(canonical),
        projectorFilename,
        projectorFile,
        runtime,
        errorMessage: runtime === 'error' ? this.state.error ?? undefined : undefined,
        isMultimodal:
          projectorFile.kind === 'complete' && runtime === 'loaded' && this.multimodalAvailable,
      });
    }

    // In-flight download not yet on disk (the .part file usually appears
    // subsecond, so this branch is brief — but keep it so the card shows up
    // immediately when the user clicks download).
    if (
      this.state.status === 'downloading' &&
      this.currentDownloadFilename &&
      !isMmproj(this.currentDownloadFilename) &&
      !seen.has(this.currentDownloadFilename)
    ) {
      const canonical = this.currentDownloadFilename;
      const projectorFilename = assignments[canonical] ?? null;
      const projectorFile: LocalFileState = projectorFilename
        ? resolveFileState(projectorFilename)
        : { kind: 'absent' };

      models.push({
        id: canonical,
        name: canonical.replace(/\.gguf$/i, ''),
        modelFile: resolveFileState(canonical),
        projectorFilename,
        projectorFile,
        runtime: 'unloaded',
        isMultimodal: false,
      });
    }

    return models;
  }

  /**
   * Projector files (mmproj) that aren't paired with any model.
   * Used to populate the "assign projector" dropdown.
   * Orphans don't appear as their own LocalModel entries — only here.
   */
  public listOrphanProjectors(): GgufFileInfo[] {
    const assignments = this.getMmprojAssignments();
    const assigned = new Set(Object.values(assignments));
    const seen = new Set<string>();
    const orphans: GgufFileInfo[] = [];

    for (const file of this.cachedGgufFiles) {
      const isPart = file.filename.endsWith('.part');
      const canonical = isPart ? file.filename.slice(0, -'.part'.length) : file.filename;
      if (!canonical.toLowerCase().includes('mmproj')) continue;
      if (assigned.has(canonical)) continue;
      if (seen.has(canonical)) continue;
      seen.add(canonical);
      orphans.push({ filename: canonical, sizeBytes: file.sizeBytes });
    }
    return orphans;
  }

  /**
   * Backwards-compatible flat view for the cross-manager interface.
   * Native UI should prefer listNativeModels().
   */
  public listLocalModels(): LocalModelEntry[] {
    return this.listNativeModels().map(m => {
      const status: LocalModelEntry['status'] =
        m.runtime === 'loaded' ? 'loaded' :
        m.runtime === 'loading' ? 'loading' :
        m.runtime === 'error' ? 'error' :
        'unloaded';
      return {
        id: m.id,
        name: m.name,
        status,
        sizeBytes: m.modelFile.kind === 'absent' ? undefined : m.modelFile.bytes,
        isMultimodal: m.isMultimodal,
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
      let firstDownloadingEvent = true;

      progressChannel.onmessage = (event: NativeProgressEvent) => {
        if (event.status === 'downloading') {
          if (firstDownloadingEvent) {
            firstDownloadingEvent = false;
            // Pick up the .part file as soon as the OS creates it, so the
            // model entry transitions from "absent" to "partial+downloading".
            this.refreshGgufCache();
          }
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
      const isCancelled = msg.includes('cancelled') || msg.includes('Cancel');
      Logger.error('NativeLlmManager', `Download failed: ${msg}`);
      this.currentDownloadFilename = null;
      this.setState({ status: isCancelled ? 'unloaded' : 'error', error: isCancelled ? null : msg });
      // Refresh so any .part file appears in the file list
      this.refreshGgufCache();
      if (!isCancelled) throw error;
      return '';
    }
  }

  /**
   * Cancel an ongoing download - signals Rust to stop and delete partial file
   */
  public async cancelDownload(): Promise<void> {
    if (this.state.status === 'downloading') {
      Logger.info('NativeLlmManager', `Cancelling download: ${this.currentDownloadFilename}`);

      try {
        // Signal Rust to cancel the download (partial .part file is kept for resume)
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
  public async loadModel(filename: string, mmprojFilename?: string, imageMinTokens?: number, imageMaxTokens?: number): Promise<void> {
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
    let resolvedMmproj = mmprojFilename ?? this.getMmprojAssignment(filename) ?? undefined;

    // Defensive: never pass a missing or partial (.part) projector to the
    // backend — llama.cpp will fail trying to open it. Pre-assignment for
    // chained preset downloads makes this reachable when a projector
    // download is cancelled mid-flight.
    if (resolvedMmproj) {
      const completeOnDisk = this.cachedGgufFiles.some(f => f.filename === resolvedMmproj);
      if (!completeOnDisk) {
        Logger.warn('NativeLlmManager', `Projector ${resolvedMmproj} is not present as a complete file; loading text-only`);
        resolvedMmproj = undefined;
      }
    }

    this.setState({ status: 'loading', modelId: modelId as any, error: null });
    Logger.info('NativeLlmManager', `Loading model: ${filename}, mmproj: ${resolvedMmproj ?? 'none'}`);

    try {
      await this.applyPersistedGpuSetting();

      await invoke('llm_load_model', {
        filename,
        mmprojFilename: resolvedMmproj ?? null,
        imageMinTokens: imageMinTokens ?? null,
        imageMaxTokens: imageMaxTokens ?? null,
      });
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

  public async getContextParams(): Promise<ContextParams> {
    try {
      return await invoke<ContextParams>('llm_get_context_params');
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to get context params: ${error}`);
      throw error;
    }
  }

  public async setContextParams(params: Partial<ContextParams>): Promise<void> {
    try {
      await invoke('llm_set_context_params', {
        nCtx: params.nCtx,
        nCtxMultimodal: params.nCtxMultimodal,
        nBatch: params.nBatch,
        nBatchMultimodal: params.nBatchMultimodal,
        nUbatch: params.nUbatch,
        nThreads: params.nThreads,
        nGpuLayers: params.nGpuLayers,
        imageMinTokens: params.imageMinTokens,
        imageMaxTokens: params.imageMaxTokens,
      });
    } catch (error) {
      Logger.error('NativeLlmManager', `Failed to set context params: ${error}`);
      throw error;
    }
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
