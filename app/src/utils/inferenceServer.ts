// src/utils/inferenceServer.ts
// Backward-compatible wrapper that delegates to ModelManager

import { ModelManager, Model, CustomServer } from './ModelManager';

// Re-export types from ModelManager for backward compatibility
export type { Model, CustomServer };

// Re-export sentinel constants from ModelManager for backward compatibility
export const BROWSER_LOCAL_SENTINEL = ModelManager.BROWSER_LOCAL;
export const LLAMA_CPP_LOCAL_SENTINEL = ModelManager.LLAMA_CPP_LOCAL;
export const SKIP_MODEL_SENTINEL = ModelManager.SKIP_MODEL;

interface ServerResponse {
  status: 'online' | 'offline';
  error?: string;
}

interface ModelsResponse {
  models: Model[];
  error?: string;
}

// ===========================================================================
// Inference Address Management (delegates to ModelManager)
// ===========================================================================

export function addInferenceAddress(address: string): void {
  ModelManager.getInstance().addServer(address);
}

export function removeInferenceAddress(address: string): void {
  ModelManager.getInstance().removeServer(address);
}

export function getInferenceAddresses(): string[] {
  return ModelManager.getInstance().getServers();
}

export function clearInferenceAddresses(): void {
  ModelManager.getInstance().clearServers();
}

// ===========================================================================
// Custom Server Management (delegates to ModelManager)
// ===========================================================================

export function loadCustomServers(): CustomServer[] {
  // Custom servers are loaded on ModelManager construction
  return ModelManager.getInstance().getCustomServers();
}

export function getCustomServers(): CustomServer[] {
  return ModelManager.getInstance().getCustomServers();
}

export function addCustomServer(address: string): CustomServer[] {
  return ModelManager.getInstance().addCustomServer(address);
}

export function removeCustomServer(address: string): CustomServer[] {
  return ModelManager.getInstance().removeCustomServer(address);
}

export function toggleCustomServer(address: string): CustomServer[] {
  return ModelManager.getInstance().toggleCustomServer(address);
}

export function updateCustomServerStatus(address: string, status: 'online' | 'offline'): CustomServer[] {
  return ModelManager.getInstance().updateCustomServerStatus(address, status);
}

export async function checkCustomServer(address: string): Promise<ServerResponse> {
  return ModelManager.getInstance().checkCustomServer(address);
}

export async function checkInferenceServer(address: string): Promise<ServerResponse> {
  return ModelManager.getInstance().checkServer(address);
}

// ===========================================================================
// Model Listing (delegates to ModelManager)
// ===========================================================================

/**
 * List all available models from all sources (remote + local)
 * This is a synchronous getter that returns the current cached model list.
 */
export function listModels(): ModelsResponse {
  return ModelManager.getInstance().listModels();
}

/**
 * Fetch models from all configured servers and refresh local model cache.
 * Call this to update the model list from remote servers.
 */
export async function fetchModels(): Promise<ModelsResponse> {
  return ModelManager.getInstance().fetchModels();
}
