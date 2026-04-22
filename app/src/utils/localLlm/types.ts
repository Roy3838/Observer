export type GemmaModelId = 'onnx-community/gemma-4-E2B-it-ONNX' | 'onnx-community/gemma-4-E4B-it-ONNX';

export const GEMMA_DISPLAY_NAMES: Record<GemmaModelId, string> = {
  'onnx-community/gemma-4-E2B-it-ONNX': 'gemma-4-E2B',
  'onnx-community/gemma-4-E4B-it-ONNX': 'gemma-4-E4B',
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

// Native models are identified by their filename (without .gguf extension)
export interface NativeModelInfo {
  id: string;           // Model ID (filename without extension)
  name: string;         // Display name
  filename: string;     // Full filename on disk
  sizeBytes: number;    // File size in bytes
  hfUrl?: string;       // Original HuggingFace URL if known
  mmprojFilename?: string;  // Associated mmproj file if multimodal
  isMultimodal: boolean;    // Whether model supports vision/multimodal
}

export type NativeModelStatus = 'unloaded' | 'loading' | 'loaded' | 'downloading' | 'error';

export interface NativeModelState {
  status: NativeModelStatus;
  modelId: string | null;  // Currently loaded/loading model ID
  downloadProgress: number;
  downloadedBytes: number;
  totalBytes: number;
  error: string | null;
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
