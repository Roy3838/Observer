// src/utils/inferenceServer.ts
import { platformFetch } from './platform';
import { usesNativeLlm } from './localLlm/LocalModelManager';
import { GEMMA_DISPLAY_NAMES, GemmaModelId } from './localLlm/types';
import { GemmaModelManager } from './localLlm/GemmaModelManager';
import { NativeLlmManager } from './localLlm/NativeLlmManager';

interface ServerResponse {
  status: 'online' | 'offline';
  error?: string;
}

export interface Model {
  name: string;
  parameterSize?: string;
  multimodal?: boolean;
  pro?: boolean;
  server: string;
  ownedBy?: string;
  status?: 'loaded' | 'loading' | 'unloaded' | 'unloading';  // For local models
  localModelId?: string;  // For loading unloaded models (e.g., GemmaModelId or filename)
}

export interface CustomServer {
  address: string;
  enabled: boolean;
  status: 'unchecked' | 'online' | 'offline';
}

// Global state for inference addresses
let inferenceAddresses: string[] = [];

// Global state for models (updated by fetchModels, read by listModels)
let availableModels: Model[] = [];

// Global state for custom servers
let customServers: CustomServer[] = [];

// LocalStorage key
const CUSTOM_SERVERS_KEY = 'observer-custom-servers';

export const BROWSER_LOCAL_SENTINEL = 'browser_local';

interface ModelsResponse {
  models: Model[];
  error?: string;
}

// Global state management functions
export function addInferenceAddress(address: string): void {
  if (!inferenceAddresses.includes(address)) {
    inferenceAddresses.push(address);
  }
}

export function removeInferenceAddress(address: string): void {
  inferenceAddresses = inferenceAddresses.filter(addr => addr !== address);
}

export function getInferenceAddresses(): string[] {
  return [...inferenceAddresses];
}

export function clearInferenceAddresses(): void {
  inferenceAddresses = [];
}

// Custom server management functions
export function loadCustomServers(): CustomServer[] {
  try {
    const stored = localStorage.getItem(CUSTOM_SERVERS_KEY);
    if (stored) {
      customServers = JSON.parse(stored);
      return customServers;
    }
  } catch (error) {
    console.error('Failed to load custom servers:', error);
  }
  return [];
}

export function getCustomServers(): CustomServer[] {
  return [...customServers];
}

export function addCustomServer(address: string): CustomServer[] {
  // Normalize the address (trim whitespace)
  const normalizedAddress = address.trim();

  // Check if already exists
  if (customServers.some(s => s.address === normalizedAddress)) {
    return customServers;
  }

  const newServer: CustomServer = {
    address: normalizedAddress,
    enabled: true,
    status: 'unchecked'
  };

  customServers.push(newServer);
  localStorage.setItem(CUSTOM_SERVERS_KEY, JSON.stringify(customServers));

  return [...customServers];
}

export function removeCustomServer(address: string): CustomServer[] {
  customServers = customServers.filter(s => s.address !== address);
  localStorage.setItem(CUSTOM_SERVERS_KEY, JSON.stringify(customServers));

  // Also remove from inference addresses if present
  removeInferenceAddress(address);

  return [...customServers];
}

export function toggleCustomServer(address: string): CustomServer[] {
  const server = customServers.find(s => s.address === address);
  if (server) {
    server.enabled = !server.enabled;
    localStorage.setItem(CUSTOM_SERVERS_KEY, JSON.stringify(customServers));

    // Update inference addresses based on enabled state
    if (server.enabled && server.status === 'online') {
      addInferenceAddress(address);
    } else {
      removeInferenceAddress(address);
    }
  }

  return [...customServers];
}

export function updateCustomServerStatus(address: string, status: 'online' | 'offline'): CustomServer[] {
  const server = customServers.find(s => s.address === address);
  if (server) {
    server.status = status;
    localStorage.setItem(CUSTOM_SERVERS_KEY, JSON.stringify(customServers));

    // Update inference addresses based on status and enabled state
    if (status === 'online' && server.enabled) {
      addInferenceAddress(address);
    } else {
      removeInferenceAddress(address);
    }
  }

  return [...customServers];
}

export async function checkCustomServer(address: string): Promise<ServerResponse> {
  const result = await checkInferenceServer(address);

  // Update the custom server status
  updateCustomServerStatus(address, result.status);

  return result;
}

export async function checkInferenceServer(address: string): Promise<ServerResponse> {
  try {
    const response = await platformFetch(`${address}/v1/models`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
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

async function listModelsFromAddress(address: string): Promise<Model[]> {
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

// Local getter function - returns the current model list
export function listModels(): ModelsResponse {
  const localModels: Model[] = [];

  if (usesNativeLlm()) {
    // iOS: Use NativeLlmManager
    const nativeManager = NativeLlmManager.getInstance();
    // Don't auto-load - let user trigger load from UI

    const nativeState = nativeManager.getState();
    if ((nativeState.status === 'loaded' || nativeState.status === 'loading' || nativeState.status === 'unloading') && nativeState.modelId) {
      // Use the loaded model name (derived from filename)
      const modelName = nativeManager.getLoadedModelName() || nativeState.modelId;
      const filename = nativeManager.getLoadedFilename();
      const displayStatus = nativeState.status === 'loaded' ? 'loaded' : nativeState.status === 'unloading' ? 'unloading' : 'loading';
      localModels.push({
        name: modelName,
        server: BROWSER_LOCAL_SENTINEL,
        multimodal: false, // GGUF models are text-only for now
        status: displayStatus,
        localModelId: filename || undefined,
      });
    } else {
      // Check persisted settings - model is downloaded but not loaded
      const persistedSettings = nativeManager.getPersistedSettings();
      if (persistedSettings?.filename) {
        const modelName = persistedSettings.filename.replace('.gguf', '').replace('.GGUF', '');
        localModels.push({
          name: modelName,
          server: BROWSER_LOCAL_SENTINEL,
          multimodal: false,
          status: 'unloaded',
          localModelId: persistedSettings.filename,
        });
      }
    }
  } else {
    // Web/Desktop: Use GemmaModelManager
    const gemmaManager = GemmaModelManager.getInstance();
    // Don't auto-load - let user trigger load from UI

    const gemmaState = gemmaManager.getState();
    // Include models that are loaded OR loading
    if ((gemmaState.status === 'loaded' || gemmaState.status === 'loading') && gemmaState.modelId) {
      localModels.push({
        name: GEMMA_DISPLAY_NAMES[gemmaState.modelId as GemmaModelId],
        server: BROWSER_LOCAL_SENTINEL,
        multimodal: true,
        parameterSize: gemmaState.modelId.includes('E2B') ? '2B' : '4B',
        status: gemmaState.status === 'loaded' ? 'loaded' : 'loading',
        localModelId: gemmaState.modelId,
      });
    } else {
      // Check for previously loaded model - show as unloaded
      const lastModelId = gemmaManager.getLastLoadedModelId();
      if (lastModelId) {
        localModels.push({
          name: GEMMA_DISPLAY_NAMES[lastModelId as GemmaModelId],
          server: BROWSER_LOCAL_SENTINEL,
          multimodal: true,
          parameterSize: lastModelId.includes('E2B') ? '2B' : '4B',
          status: 'unloaded',
          localModelId: lastModelId,
        });
      }
    }
  }

  // Sort: browser_local first, then user-managed servers, then Observer cloud last
  const sortedModels = [...availableModels, ...localModels].sort((a, b) => {
    const aScore = a.server === BROWSER_LOCAL_SENTINEL ? 0
      : a.server.includes('api.observer-ai.com') ? 2
      : 1;
    const bScore = b.server === BROWSER_LOCAL_SENTINEL ? 0
      : b.server.includes('api.observer-ai.com') ? 2
      : 1;
    return aScore - bScore;
  });

  return { models: sortedModels };
}

// Fetch function - called by AppHeader to update the model list
export async function fetchModels(): Promise<ModelsResponse> {
  try {
    const allModels: Model[] = [];

    for (const address of inferenceAddresses) {
      const models = await listModelsFromAddress(address);
      allModels.push(...models);
    }

    // Update the global state
    availableModels = allModels;

    return { models: allModels };
  } catch (error) {
    return {
      models: [],
      error: `Could not retrieve models: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}
