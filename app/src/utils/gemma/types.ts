export type GemmaModelId = 'onnx-community/gemma-4-E2B-it-ONNX' | 'onnx-community/gemma-4-E4B-it-ONNX';
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

export interface GemmaModelState {
  status: GemmaStatus;
  modelId: GemmaModelId | null;
  progress: GemmaProgressItem[];
  error: string | null;
}
