// localLlm module exports
// Platform-aware local LLM inference for Observer

// Platform abstraction (main entry point)
export {
  getLocalModelManager,
  usesNativeLlm,
  usesWebLlm,
  getGemmaManager,
  getNativeManager,
  tryAutoLoadLocalModel,
  type LocalModelManager,
  type LocalModelManagerInterface,
} from './LocalModelManager';

// Individual managers
export { GemmaModelManager } from './GemmaModelManager';
export { NativeLlmManager } from './NativeLlmManager';

// Types
export * from './types';
