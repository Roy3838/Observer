// LocalModelManager.ts - Platform-aware abstraction for local LLM inference
// Returns the appropriate manager based on platform:
// - iOS: NativeLlmManager (llama.cpp with Metal)
// - Web/Desktop: GemmaModelManager (transformers.js in Web Worker)

import { isIOS } from '../platform';
import { GemmaModelManager } from './GemmaModelManager';
import { NativeLlmManager } from './NativeLlmManager';
import { LocalLlmMessage } from './types';

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
  if (isIOS()) {
    return NativeLlmManager.getInstance();
  }
  return GemmaModelManager.getInstance();
}

/**
 * Check if the current platform uses native LLM inference (iOS)
 */
export function usesNativeLlm(): boolean {
  return isIOS();
}

/**
 * Check if the current platform uses web-based LLM inference (transformers.js)
 */
export function usesWebLlm(): boolean {
  return !isIOS();
}

/**
 * Get the GemmaModelManager (for web/desktop platforms)
 * Throws if called on iOS
 */
export function getGemmaManager(): GemmaModelManager {
  if (isIOS()) {
    throw new Error('GemmaModelManager not available on iOS. Use getLocalModelManager() instead.');
  }
  return GemmaModelManager.getInstance();
}

/**
 * Get the NativeLlmManager (for iOS platform)
 * Throws if called on non-iOS platforms
 */
export function getNativeManager(): NativeLlmManager {
  if (!isIOS()) {
    throw new Error('NativeLlmManager only available on iOS. Use getLocalModelManager() instead.');
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
    if (isIOS()) {
      // NativeLlmManager.tryAutoLoad is async
      (manager as NativeLlmManager).tryAutoLoad();
    } else {
      // GemmaModelManager.tryAutoLoad is sync
      (manager as GemmaModelManager).tryAutoLoad();
    }
  }
}
