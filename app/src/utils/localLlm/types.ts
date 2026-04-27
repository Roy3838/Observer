export type GemmaModelId = 'onnx-community/gemma-4-E2B-it-ONNX' | 'onnx-community/gemma-4-E4B-it-ONNX';

export const GEMMA_DISPLAY_NAMES: Record<GemmaModelId, string> = {
  'onnx-community/gemma-4-E2B-it-ONNX': 'gemma-4-E2B ONNX',
  'onnx-community/gemma-4-E4B-it-ONNX': 'gemma-4-E4B ONNX',
};

export type GemmaDevice = 'webgpu' | 'wasm';
export type GemmaDtype = 'q4f16' | 'q4' | 'fp16' | 'fp32' | 'q8';
export type GemmaImageTokenBudget = 70 | 140 | 280 | 560 | 1120;
export type GemmaStatus = 'unloaded' | 'loading' | 'loaded' | 'error';

// Multimodal content types
export type GemmaTextContent = { type: 'text'; text: string };
export type GemmaImageContent = { type: 'image'; image?: string | Blob }; // URL, data URL, or Blob
export type GemmaContentPart = GemmaTextContent | GemmaImageContent;

export interface GemmaMessage {
  role: string;
  content: string | GemmaContentPart[];
}

export interface GemmaProgressItem {
  file: string;
  progress: number;
  loaded: number;
  total: number;
  status: 'progress' | 'done';
}

export interface GemmaLoadSettings {
  device: GemmaDevice;
  dtype: GemmaDtype;
  imageTokenBudget: GemmaImageTokenBudget;
  enableThinking?: boolean;
}

export interface GemmaModelState {
  status: GemmaStatus;
  modelId: GemmaModelId | null;
  progress: GemmaProgressItem[];
  error: string | null;
  loadSettings: GemmaLoadSettings | null;
}

// ============================================================================
// Native LLM Types (iOS llama.cpp)
// ============================================================================

// A single GGUF file on disk — could be a model or a projector (mmproj).
// The frontend decides how each file is used; no auto-detection on the backend.
export interface GgufFileInfo {
  filename: string;   // Full filename on disk (e.g. "gemma-4-E2B-Q4.gguf")
  sizeBytes: number;
}

export type NativeModelStatus = 'unloaded' | 'loading' | 'loaded' | 'downloading' | 'unloading' | 'error';

export interface NativeModelState {
  status: NativeModelStatus;
  modelId: string | null;  // Currently loaded/loading model ID
  downloadProgress: number;
  downloadedBytes: number;
  totalBytes: number;
  error: string | null;
  enableThinking: boolean;
}

export interface NativeProgressEvent {
  status: 'downloading' | 'complete' | 'error';
  progress: number;
  downloadedBytes: number;
  totalBytes: number;
  filename?: string;
  error?: string;
}

// Multimodal content types for native LLM (matches Gemma pattern)
export type LocalLlmTextContent = { type: 'text'; text: string };
export type LocalLlmImageContent = { type: 'image'; image: string }; // base64 data URL
export type LocalLlmContentPart = LocalLlmTextContent | LocalLlmImageContent;

// Generic message type for both managers
export interface LocalLlmMessage {
  role: string;
  content: string | LocalLlmContentPart[];
}

// ============================================================================
// Debug Panel Types (iOS llama.cpp)
// ============================================================================

export interface GenerationMetrics {
  tokensGenerated: number;
  promptTokens: number;
  timeToFirstTokenMs: number;
  totalGenerationTimeMs: number;
  tokensPerSecond: number;
}

export interface SamplerParams {
  temperature: number;
  topP: number;
  topK: number;
  seed: number;
  repeatPenalty: number;
}

export interface LlmDebugInfo {
  modelsDir: string;
  engine: {
    initialized: boolean;
    isLoaded: boolean;
    loadedModelId: string | null;
    isMultimodal: boolean;
    modelPath: string | null;
    mmprojPath: string | null;
    samplerParams: SamplerParams | null;
    lastMetrics: GenerationMetrics | null;
  };
}

export const DEFAULT_SAMPLER_PARAMS: SamplerParams = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  seed: 42,
  repeatPenalty: 1.1,
};

// ============================================================================
// Unified Local Model Entry (for ModelManager)
// ============================================================================

/**
 * Represents a local model from either GemmaModelManager or NativeLlmManager.
 * Used by ModelManager to provide a unified view of all local models.
 */
export interface LocalModelEntry {
  id: string;                                       // Unique identifier (filename or GemmaModelId)
  name: string;                                     // Display name
  status: 'loaded' | 'loading' | 'unloaded' | 'error';
  sizeBytes?: number;                               // For downloaded GGUF models
  isMultimodal?: boolean;
}
