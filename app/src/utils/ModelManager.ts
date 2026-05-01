// ModelManager.ts - Unified singleton for all model sources
// Manages remote models from inference servers and local models from GemmaModelManager/NativeLlmManager
// This is the SINGLE SOURCE OF TRUTH for all model state - components should only interact with this class

import { platformFetch, isTauri } from './platform';
import { NativeLlmManager } from './localLlm/NativeLlmManager';
import { GemmaModelManager } from './localLlm/GemmaModelManager';
import { GemmaModelId, LocalLlmMessage } from './localLlm/types';
import { InferenceParams, DEFAULT_INFERENCE_PARAMS } from '../config/inference-params';
import { PreProcessorResult } from './pre-processor';
import { UnauthorizedError } from './sendApi';

export { UnauthorizedError };

/**
 * Model entry representing a model from any source (remote or local)
 */
export interface Model {
  name: string;
  parameterSize?: string;
  multimodal?: boolean;
  pro?: boolean;
  server: string;
  ownedBy?: string;
  status?: 'loaded' | 'loading' | 'unloaded' | 'unloading' | 'error';  // For local models
  localModelId?: string;  // For loading unloaded models (e.g., GemmaModelId or filename)
}

/**
 * Unified progress item for local model loading
 * Used by transformers.js models; llama.cpp models don't have file-level progress
 */
export interface LocalModelProgressItem {
  file: string;
  progress: number;  // 0-100
  loaded: number;    // bytes
  total: number;     // bytes
  done: boolean;
}

/**
 * Unified state for local model loading/inference
 * Abstracts away the differences between GemmaModelManager and NativeLlmManager
 */
export interface LocalModelState {
  status: 'unloaded' | 'loading' | 'loaded' | 'unloading' | 'downloading' | 'error';
  modelId: string | null;
  error: string | null;
  // Progress info (only for transformers.js models)
  progress: LocalModelProgressItem[];
  // Engine-specific settings (only for transformers.js)
  engineInfo?: {
    type: 'transformers.js' | 'llama.cpp';
    device?: string;       // 'webgpu' | 'wasm' for transformers.js
    dtype?: string;        // quantization type
    enableThinking?: boolean;
  };
}

/**
 * Custom server configuration
 */
export interface CustomServer {
  address: string;
  enabled: boolean;
  status: 'unchecked' | 'online' | 'offline';
}

interface ModelsResponse {
  models: Model[];
  error?: string;
}

const CUSTOM_SERVERS_KEY = 'observer-custom-servers';
const MODEL_PARAMS_PREFIX = 'observer-ai:inference:model:';

/**
 * ModelManager - Single source of truth for all model state
 *
 * Manages:
 * - Remote models from inference server addresses
 * - Local models via LocalModelManager (GemmaModelManager or NativeLlmManager)
 */
// Gemma 4 GGUF thinking delimiters (observed from llama.cpp token stream)
const LLAMA_THINK_START = '<|channel>thought';
const LLAMA_THINK_END   = '<channel|>';

/**
 * Returns a single onToken callback that splits the raw llama.cpp token stream
 * into reasoning tokens and answer tokens based on Gemma 4's thinking delimiters.
 */
function makeLlamaCppThinkingRouter(
  onToken?: (t: string) => void,
  onReasoningToken?: (t: string) => void,
): ((token: string) => void) & { flush: () => void } {
  type State = 'scanning' | 'thinking' | 'answering';
  let state: State = 'scanning';
  let buf = '';
  const MAX_MARKER = Math.max(LLAMA_THINK_START.length, LLAMA_THINK_END.length);

  const callback = (token: string) => {
    buf += token;

    while (buf.length > 0) {
      if (state === 'scanning' || state === 'answering') {
        const idx = buf.indexOf(LLAMA_THINK_START);
        if (idx !== -1) {
          // Emit everything before the marker as answer text
          const before = buf.slice(0, idx);
          if (before) onToken?.(before);
          buf = buf.slice(idx + LLAMA_THINK_START.length);
          state = 'thinking';
        } else {
          // No marker yet — keep a tail in case the marker is split across tokens
          const safe = buf.length > MAX_MARKER ? buf.slice(0, buf.length - MAX_MARKER) : '';
          if (safe) onToken?.(safe);
          buf = buf.slice(safe.length);
          break;
        }
      } else {
        // state === 'thinking'
        const idx = buf.indexOf(LLAMA_THINK_END);
        if (idx !== -1) {
          const reasoning = buf.slice(0, idx);
          if (reasoning) onReasoningToken?.(reasoning);
          buf = buf.slice(idx + LLAMA_THINK_END.length);
          state = 'answering';
        } else {
          const safe = buf.length > MAX_MARKER ? buf.slice(0, buf.length - MAX_MARKER) : '';
          if (safe) onReasoningToken?.(safe);
          buf = buf.slice(safe.length);
          break;
        }
      }
    }
  };

  callback.flush = () => {
    if (!buf) return;
    if (state === 'thinking') onReasoningToken?.(buf);
    else onToken?.(buf);
    buf = '';
  };

  return callback;
}

export class ModelManager {
  private static instance: ModelManager | null = null;

  // Sentinel values for local model servers
  static readonly BROWSER_LOCAL = 'browser_local';
  static readonly LLAMA_CPP_LOCAL = 'llama_cpp_local';
  static readonly SKIP_MODEL = 'skip_model';

  // State
  private inferenceAddresses: string[] = [];
  private remoteModels: Model[] = [];
  private customServers: CustomServer[] = [];
  private listeners: Array<(models: Model[]) => void> = [];

  private constructor() {
    // Load custom servers from localStorage on init
    this.loadCustomServersFromStorage();

    // Subscribe to underlying model managers and forward state changes
    // This makes ModelManager the single source of truth for all model state
    GemmaModelManager.getInstance().onStateChange(() => {
      this.notifyListeners();
    });

    if (isTauri()) {
      NativeLlmManager.getInstance().onStateChange(() => {
        this.notifyListeners();
      });
    }
  }

  public static getInstance(): ModelManager {
    if (!ModelManager.instance) {
      ModelManager.instance = new ModelManager();
    }
    return ModelManager.instance;
  }

  // ===========================================================================
  // Model Listing
  // ===========================================================================

  /**
   * List all models from all sources (remote + local)
   */
  public listModels(): ModelsResponse {
    const localModels = this.getLocalModels();

    // Sort: local models first, then user-managed servers, then Observer cloud last
    const sortedModels = [...this.remoteModels, ...localModels].sort((a, b) => {
      const isLocalA = this.isLocalModel(a.server);
      const isLocalB = this.isLocalModel(b.server);
      const aScore = isLocalA ? 0
        : a.server.includes('api.observer-ai.com') ? 2
        : 1;
      const bScore = isLocalB ? 0
        : b.server.includes('api.observer-ai.com') ? 2
        : 1;
      return aScore - bScore;
    });

    return { models: sortedModels };
  }

  /**
   * Get local models converted to the unified Model format.
   * On Tauri: returns both llama.cpp models AND transformers.js models.
   * On web: returns only transformers.js models.
   */
  private getLocalModels(): Model[] {
    const models: Model[] = [];

    // Skip Model Call — always available, returns empty string
    models.push({
      name: 'Skip Model Call',
      server: ModelManager.SKIP_MODEL,
      multimodal: false,
      status: 'loaded',
    });

    // Transformers.js models (available on all platforms)
    const gemmaModels = GemmaModelManager.getInstance().listLocalModels();
    for (const entry of gemmaModels) {
      models.push({
        name: entry.name,
        server: ModelManager.BROWSER_LOCAL,
        multimodal: entry.isMultimodal,
        status: entry.status,
        localModelId: entry.id,
      });
    }

    // llama.cpp models (Tauri only)
    if (isTauri()) {
      const nativeModels = NativeLlmManager.getInstance().listLocalModels();
      for (const entry of nativeModels) {
        models.push({
          name: entry.name,
          server: ModelManager.LLAMA_CPP_LOCAL,
          multimodal: entry.isMultimodal,
          status: entry.status,
          localModelId: entry.id,
        });
      }
    }

    return models;
  }

  /**
   * Fetch models from all sources (remote servers + refresh local cache)
   */
  public async fetchModels(): Promise<ModelsResponse> {
    try {
      const allModels: Model[] = [];

      // Fetch from all inference addresses
      for (const address of this.inferenceAddresses) {
        const models = await this.fetchFromAddress(address);
        allModels.push(...models);
      }

      // Update remote models state
      this.remoteModels = allModels;

      // Refresh local model caches
      // llama.cpp models (Tauri only)
      if (isTauri()) {
        await NativeLlmManager.getInstance().refreshGgufCache();
      }
      // Note: GemmaModelManager doesn't have a refresh - it reads from localStorage synchronously

      this.notifyListeners();
      return this.listModels();
    } catch (error) {
      return {
        models: [],
        error: `Could not retrieve models: ${error instanceof Error ? error.message : String(error)}`
      };
    }
  }

  /**
   * Fetch models from a single address
   */
  private async fetchFromAddress(address: string): Promise<Model[]> {
    try {
      const response = await platformFetch(`${address}/v1/models`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        return [];
      }

      const data = await response.json();
      const modelData = data.data || [];

      if (!Array.isArray(modelData)) {
        return [];
      }

      return modelData.map((model: any) => ({
        name: model.id,
        parameterSize: model.parameter_size,
        multimodal: model.multimodal ?? false,
        pro: model.pro ?? false,
        server: address,
        ownedBy: model.owned_by
      }));
    } catch (error) {
      return [];
    }
  }

  // ===========================================================================
  // Server Management
  // ===========================================================================

  /**
   * Add an inference server address
   */
  public addServer(address: string): void {
    if (!this.inferenceAddresses.includes(address)) {
      this.inferenceAddresses.push(address);
    }
  }

  /**
   * Remove an inference server address
   */
  public removeServer(address: string): void {
    this.inferenceAddresses = this.inferenceAddresses.filter(addr => addr !== address);
  }

  /**
   * Get all inference server addresses
   */
  public getServers(): string[] {
    return [...this.inferenceAddresses];
  }

  /**
   * Clear all inference server addresses
   */
  public clearServers(): void {
    this.inferenceAddresses = [];
  }

  // ===========================================================================
  // Custom Server Management
  // ===========================================================================

  private loadCustomServersFromStorage(): void {
    try {
      const stored = localStorage.getItem(CUSTOM_SERVERS_KEY);
      if (stored) {
        this.customServers = JSON.parse(stored);
      }
    } catch (error) {
      console.error('Failed to load custom servers:', error);
    }
  }

  private saveCustomServersToStorage(): void {
    localStorage.setItem(CUSTOM_SERVERS_KEY, JSON.stringify(this.customServers));
  }

  public getCustomServers(): CustomServer[] {
    return [...this.customServers];
  }

  public addCustomServer(address: string): CustomServer[] {
    const normalizedAddress = address.trim();

    if (this.customServers.some(s => s.address === normalizedAddress)) {
      return this.customServers;
    }

    const newServer: CustomServer = {
      address: normalizedAddress,
      enabled: true,
      status: 'unchecked'
    };

    this.customServers.push(newServer);
    this.saveCustomServersToStorage();

    return [...this.customServers];
  }

  public removeCustomServer(address: string): CustomServer[] {
    this.customServers = this.customServers.filter(s => s.address !== address);
    this.saveCustomServersToStorage();
    this.removeServer(address);

    return [...this.customServers];
  }

  public toggleCustomServer(address: string): CustomServer[] {
    const server = this.customServers.find(s => s.address === address);
    if (server) {
      server.enabled = !server.enabled;
      this.saveCustomServersToStorage();

      if (server.enabled && server.status === 'online') {
        this.addServer(address);
      } else {
        this.removeServer(address);
      }
    }

    return [...this.customServers];
  }

  public updateCustomServerStatus(address: string, status: 'online' | 'offline'): CustomServer[] {
    const server = this.customServers.find(s => s.address === address);
    if (server) {
      server.status = status;
      this.saveCustomServersToStorage();

      if (status === 'online' && server.enabled) {
        this.addServer(address);
      } else {
        this.removeServer(address);
      }
    }

    return [...this.customServers];
  }

  public async checkCustomServer(address: string): Promise<{ status: 'online' | 'offline'; error?: string }> {
    const result = await this.checkServer(address);
    this.updateCustomServerStatus(address, result.status);
    return result;
  }

  public async checkServer(address: string): Promise<{ status: 'online' | 'offline'; error?: string }> {
    try {
      const response = await platformFetch(`${address}/v1/models`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (response.ok) {
        return { status: 'online' };
      }

      return {
        status: 'offline',
        error: `Server responded with status ${response.status}`
      };
    } catch (error) {
      return {
        status: 'offline',
        error: 'Could not connect to server'
      };
    }
  }

  // ===========================================================================
  // Local Model Management (unified interface)
  // ===========================================================================

  /**
   * Get unified state for a local model by server type.
   * Returns the current loading/loaded state for the specified engine.
   */
  public getLocalModelState(server: string): LocalModelState {
    if (server === ModelManager.BROWSER_LOCAL) {
      const gemmaState = GemmaModelManager.getInstance().getState();
      return {
        status: gemmaState.status === 'unloaded' ? 'unloaded'
          : gemmaState.status === 'loading' ? 'loading'
          : gemmaState.status === 'loaded' ? 'loaded'
          : 'error',
        modelId: gemmaState.modelId,
        error: gemmaState.error,
        progress: gemmaState.progress.map(p => ({
          file: p.file,
          progress: p.progress,
          loaded: p.loaded,
          total: p.total,
          done: p.status === 'done',
        })),
        engineInfo: gemmaState.loadSettings ? {
          type: 'transformers.js',
          device: gemmaState.loadSettings.device,
          dtype: gemmaState.loadSettings.dtype,
          enableThinking: gemmaState.loadSettings.enableThinking ?? false,
        } : { type: 'transformers.js' },
      };
    }

    if (server === ModelManager.LLAMA_CPP_LOCAL && isTauri()) {
      const nativeState = NativeLlmManager.getInstance().getState();
      return {
        status: nativeState.status,
        modelId: nativeState.modelId,
        error: nativeState.error,
        progress: [],  // llama.cpp doesn't have file-level progress
        engineInfo: { type: 'llama.cpp' },
      };
    }

    // Default state for unknown server
    return {
      status: 'unloaded',
      modelId: null,
      error: null,
      progress: [],
    };
  }

  /**
   * Load a local model by its ID and server type.
   * Routes to the appropriate underlying manager based on the server sentinel.
   */
  public async loadLocalModel(localModelId: string, server: string): Promise<void> {
    if (server === ModelManager.BROWSER_LOCAL) {
      await GemmaModelManager.getInstance().loadModel(localModelId as GemmaModelId);
    } else if (server === ModelManager.LLAMA_CPP_LOCAL && isTauri()) {
      await NativeLlmManager.getInstance().loadModel(localModelId);
    } else {
      throw new Error(`Cannot load model: unsupported server type "${server}"`);
    }
  }

  /**
   * Unload the currently loaded local model for a given server type.
   */
  public async unloadLocalModel(server: string): Promise<void> {
    if (server === ModelManager.BROWSER_LOCAL) {
      GemmaModelManager.getInstance().unloadModel();
    } else if (server === ModelManager.LLAMA_CPP_LOCAL && isTauri()) {
      await NativeLlmManager.getInstance().unloadModel();
    }
  }

  /**
   * Check if a local model is ready for inference.
   */
  public isLocalModelReady(server: string): boolean {
    if (server === ModelManager.SKIP_MODEL) return true;
    if (server === ModelManager.BROWSER_LOCAL) {
      return GemmaModelManager.getInstance().isReady();
    }
    if (server === ModelManager.LLAMA_CPP_LOCAL && isTauri()) {
      return NativeLlmManager.getInstance().isReady();
    }
    return false;
  }

  /**
   * Generate a response using a local model.
   * Routes to the appropriate underlying manager based on the server sentinel.
   */
  public async generateWithLocalModel(
    server: string,
    messages: LocalLlmMessage[],
    onToken?: (token: string) => void,
    onReasoningToken?: (token: string) => void,
  ): Promise<string> {
    if (server === ModelManager.SKIP_MODEL) {
      return '';
    }

    if (server === ModelManager.BROWSER_LOCAL) {
      const manager = GemmaModelManager.getInstance();
      if (!manager.isReady()) {
        throw new Error('Transformers.js model not loaded');
      }
      const enableThinking = manager.getState().loadSettings?.enableThinking ?? false;
      return manager.generate(messages, onToken, enableThinking, onReasoningToken);
    }

    if (server === ModelManager.LLAMA_CPP_LOCAL && isTauri()) {
      const manager = NativeLlmManager.getInstance();
      if (!manager.isReady()) {
        throw new Error('llama.cpp model not loaded');
      }
      const enableThinking = manager.getState().enableThinking;
      if (!enableThinking || !onReasoningToken) {
        return manager.generate(messages, onToken);
      }
      // Route thinking tokens to onReasoningToken, answer tokens to onToken.
      // Accumulate answer tokens — manager.generate returns the raw backend string which
      // still contains the thinking delimiters, so we can't use it directly.
      let answer = '';
      const collectAndForward = (t: string) => { answer += t; onToken?.(t); };
      const router = makeLlamaCppThinkingRouter(collectAndForward, onReasoningToken);
      await manager.generate(messages, router);
      router.flush();
      return answer;
    }

    throw new Error(`Cannot generate: unsupported server type "${server}"`);
  }

  // ===========================================================================
  // Per-Model Inference Params (remote models only — local models use their own settings)
  // ===========================================================================

  public getModelParams(modelName: string): Partial<InferenceParams> {
    try {
      const stored = localStorage.getItem(`${MODEL_PARAMS_PREFIX}${modelName}`);
      if (stored) return JSON.parse(stored);
    } catch {}
    return {};
  }

  public setModelParams(modelName: string, params: Partial<InferenceParams>): void {
    const cleaned: Partial<InferenceParams> = {};
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) (cleaned as any)[k] = v;
    }
    if (Object.keys(cleaned).length === 0) {
      localStorage.removeItem(`${MODEL_PARAMS_PREFIX}${modelName}`);
    } else {
      localStorage.setItem(`${MODEL_PARAMS_PREFIX}${modelName}`, JSON.stringify(cleaned));
    }
  }

  public clearModelParams(modelName: string): void {
    localStorage.removeItem(`${MODEL_PARAMS_PREFIX}${modelName}`);
  }

  public hasModelParams(modelName: string): boolean {
    return Object.keys(this.getModelParams(modelName)).length > 0;
  }

  // ===========================================================================
  // Central prompt router
  // ===========================================================================

  /**
   * Send a prompt through the appropriate backend for the given model.
   * - Local models (browser_local, llama_cpp_local): routed directly, no inference params
   * - Remote models: fetches per-model params and delegates to sendApi.fetchResponse
   */
  public async sendPrompt(
    modelName: string,
    preprocessResult: PreProcessorResult,
    token?: string,
    enableStreaming: boolean = false,
    onStreamChunk?: (chunk: string) => void,
    onReasoningChunk?: (chunk: string) => void
  ): Promise<string> {
    // Resolve model → server
    let modelsResponse = this.listModels();
    let model = modelsResponse.models.find(m => m.name === modelName);
    if (!model) {
      modelsResponse = await this.fetchModels();
      model = modelsResponse.models.find(m => m.name === modelName);
    }
    if (!model) throw new Error(`Model '${modelName}' not found in available models`);

    const serverAddress = model.server;

    // Build messages from preprocessResult
    const hasImages = preprocessResult.images && preprocessResult.images.length > 0;
    let content: any = preprocessResult.modifiedPrompt;
    if (hasImages) {
      content = [
        { type: 'text', text: preprocessResult.modifiedPrompt },
        ...preprocessResult.images!.map(img => ({
          type: 'image',
          image: `data:image/png;base64,${img}`,
        })),
      ];
    }
    const messages = [{ role: 'user', content }];

    // Local models: route directly, they manage their own settings
    if (this.isLocalModel(serverAddress)) {
      if (!this.isLocalModelReady(serverAddress)) {
        throw new Error('Local model not loaded. Please load it from the Add Model panel.');
      }
      return this.generateWithLocalModel(serverAddress, messages, onStreamChunk, onReasoningChunk);
    }

    // Remote models: attach per-model inference params
    if (serverAddress.includes('api.observer-ai.com') && token) {
      this.optimisticUpdateQuota();
    }
    const { fetchResponse } = await import('./sendApi');
    const params = { ...DEFAULT_INFERENCE_PARAMS, ...this.getModelParams(modelName) };
    return fetchResponse(serverAddress, messages, modelName, token, enableStreaming, onStreamChunk, params, onReasoningChunk);
  }

  // ===========================================================================
  // Utilities
  // ===========================================================================

  /**
   * Check if a server string represents a local model
   */
  public isLocalModel(server: string): boolean {
    return server === ModelManager.BROWSER_LOCAL
      || server === ModelManager.LLAMA_CPP_LOCAL
      || server === ModelManager.SKIP_MODEL;
  }

  /**
   * Subscribe to model list changes
   */
  public onModelsChange(listener: (models: Model[]) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notifyListeners(): void {
    const models = this.listModels().models;
    this.listeners.forEach(l => l(models));
  }

  /**
   * Send messages to a model by name. Routes local models directly, remote models via fetchResponse.
   */
  public async sendMessages(
    modelName: string,
    messages: Array<{ role: string; content: any }>,
    token?: string,
    enableStreaming: boolean = false,
    onStreamChunk?: (chunk: string) => void,
    onReasoningChunk?: (chunk: string) => void
  ): Promise<string> {
    let modelsResponse = this.listModels();
    let model = modelsResponse.models.find(m => m.name === modelName);
    if (!model) {
      modelsResponse = await this.fetchModels();
      model = modelsResponse.models.find(m => m.name === modelName);
    }
    // Ghost models (agent-creator-only) aren't returned by listModels but are valid on the
    // Observer API — route them there when a token is present.
    if (!model && !token) throw new Error(`Model '${modelName}' not found in available models`);

    const serverAddress = model ? model.server : 'https://api.observer-ai.com:443';

    if (this.isLocalModel(serverAddress)) {
      if (!this.isLocalModelReady(serverAddress)) {
        throw new Error('Local model not loaded. Please load it from the Add Model panel.');
      }
      return this.generateWithLocalModel(serverAddress, messages, onStreamChunk, onReasoningChunk);
    }

    if (serverAddress.includes('api.observer-ai.com') && token) {
      this.optimisticUpdateQuota();
    }
    const { fetchResponse } = await import('./sendApi');
    const params = { ...DEFAULT_INFERENCE_PARAMS, ...this.getModelParams(modelName) };
    return fetchResponse(serverAddress, messages, modelName, token, enableStreaming, onStreamChunk, params, onReasoningChunk);
  }

  private optimisticUpdateQuota(): void {
    try {
      const key = 'observer-quota-remaining';
      const current = localStorage.getItem(key);
      if (current !== null) {
        const n = parseInt(current, 10);
        if (!isNaN(n)) {
          localStorage.setItem(key, (n - 1).toString());
          window.dispatchEvent(new CustomEvent('quotaUpdated'));
        }
      }
    } catch {}
  }
}
