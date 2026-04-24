// LocalModelManager.ts - Platform-aware abstraction for local LLM inference
// Returns the appropriate manager based on platform:
// - Tauri (iOS + desktop): NativeLlmManager (llama.cpp with Metal/CPU)
// - Web browser: GemmaModelManager (transformers.js in Web Worker)

import { isTauri } from '../platform';
import { GemmaModelManager } from './GemmaModelManager';
import { NativeLlmManager } from './NativeLlmManager';
import { LocalLlmMessage, LocalModelEntry } from './types';

/**
 * Common interface for local model managers
 * Both GemmaModelManager and NativeLlmManager implement these methods
 */
export interface LocalModelManagerInterface {
  isReady(): boolean;
  isLoading(): boolean;
  hasError(): boolean;
  getError(): string | null;
  generate(messages: LocalLlmMessage[], onToken?: (token: string) => void): Promise<string>;
  unloadModel(): void | Promise<void>;
  onStateChange(listener: (state: any) => void): () => void;
  getState(): any;
  listLocalModels(): LocalModelEntry[];
}

/**
 * Type for the manager - either GemmaModelManager or NativeLlmManager
 */
export type LocalModelManager = GemmaModelManager | NativeLlmManager;

/**
 * Get the appropriate local model manager for the current platform.
 *
 * Usage:
 * ```typescript
 * const manager = getLocalModelManager();
 * if (manager.isReady()) {
 *   const response = await manager.generate(messages, onToken);
 * }
 * ```
 */
export function getLocalModelManager(): LocalModelManager {
  if (isTauri()) {
    return NativeLlmManager.getInstance();
  }
  return GemmaModelManager.getInstance();
}

/**
 * Check if the current platform uses native LLM inference (Tauri: iOS + desktop)
 */
export function usesNativeLlm(): boolean {
  return isTauri();
}

/**
 * Check if the current platform uses web-based LLM inference (transformers.js)
 */
export function usesWebLlm(): boolean {
  return !isTauri();
}

/**
 * Get the GemmaModelManager (for web browser only)
 * Throws if called in Tauri
 */
export function getGemmaManager(): GemmaModelManager {
  if (isTauri()) {
    throw new Error('GemmaModelManager not available in Tauri. Use getLocalModelManager() instead.');
  }
  return GemmaModelManager.getInstance();
}

/**
 * Get the NativeLlmManager (for Tauri: iOS + desktop)
 * Throws if called in web browser
 */
export function getNativeManager(): NativeLlmManager {
  if (!isTauri()) {
    throw new Error('NativeLlmManager only available in Tauri. Use getLocalModelManager() instead.');
  }
  return NativeLlmManager.getInstance();
}

/**
 * Try to auto-load a persisted model on app startup.
 * Call this early in app initialization.
 */
export function tryAutoLoadLocalModel(): void {
  const manager = getLocalModelManager();

  if ('tryAutoLoad' in manager) {
    if (isTauri()) {
      // NativeLlmManager.tryAutoLoad is async
      (manager as NativeLlmManager).tryAutoLoad();
    } else {
      // GemmaModelManager.tryAutoLoad is sync
      (manager as GemmaModelManager).tryAutoLoad();
    }
  }
}
