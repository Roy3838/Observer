export type ModelSize = 'tiny' | 'base' | 'small' | 'medium' | 'large' | 'large-v2' | 'large-v3';
export type WhisperTask = 'transcribe' | 'translate';
export type ModelState = 'unloaded' | 'loading' | 'loaded' | 'error';
export type LanguageCode = 'en' | 'zh' | 'de' | 'es' | 'ru' | 'ko' | 'fr' | 'ja' | 'pt' | 'tr' | 'pl' | 'ca' | 'nl' | 'ar' | 'sv' | 'it' | 'id' | 'hi' | 'fi' | 'vi' | 'he' | 'uk' | 'el' | 'ms' | 'cs' | 'ro' | 'da' | 'hu' | 'ta' | 'no' | 'th' | 'ur' | 'hr' | 'bg' | 'lt' | 'la' | 'mi' | 'ml' | 'cy' | 'sk' | 'te' | 'fa' | 'lv' | 'bn' | 'sr' | 'az' | 'sl' | 'kn' | 'et' | 'mk' | 'br' | 'eu' | 'is' | 'hy' | 'ne' | 'mn' | 'bs' | 'kk' | 'sq' | 'sw' | 'gl' | 'mr' | 'pa' | 'si' | 'km' | 'sn' | 'yo' | 'so' | 'af' | 'oc' | 'ka' | 'be' | 'tg' | 'sd' | 'gu' | 'am' | 'yi' | 'lo' | 'uz' | 'fo' | 'ht' | 'ps' | 'tk' | 'nn' | 'mt' | 'sa' | 'lb' | 'my' | 'bo' | 'tl' | 'mg' | 'as' | 'tt' | 'haw' | 'ln' | 'ha' | 'ba' | 'jw' | 'su';

export interface WhisperModelConfig {
  modelId: string;
  task?: WhisperTask;
  language?: LanguageCode | string;
  quantized: boolean;
}

export interface WhisperSettings {
  modelId: string;
  task?: WhisperTask;
  language?: LanguageCode | string;
  quantized: boolean;
  chunkDurationMs: number;
}

// Legacy interface for backward compatibility
export interface ModelConfig {
  modelSize: ModelSize;
  language: string;
  quantized: boolean;
  modelId: string;
}

export interface ProgressItem {
  file: string;
  progress: number;
  loaded: number;
  total: number;
  status: 'progress' | 'done';
  name?: string;
}

export interface WhisperModelState {
  status: ModelState;
  config: ModelConfig | null;
  progress: ProgressItem[];
  error: string | null;
}

export interface TranscriptionChunk {
  id: number;
  blob: Blob;
  text: string;
}

export interface WorkerMessage {
  type: 'configure' | 'transcribe' | 'progress' | 'ready' | 'error' | 'transcription-complete';
  data?: any;
}

export interface WorkerConfigureMessage extends WorkerMessage {
  type: 'configure';
  data: WhisperModelConfig;
}

export interface WorkerTranscribeMessage extends WorkerMessage {
  type: 'transcribe';
  data: {
    audio: Float32Array;
    chunkId: number;
  };
}

export interface WorkerProgressMessage extends WorkerMessage {
  type: 'progress';
  data: ProgressItem;
}

export interface WorkerReadyMessage extends WorkerMessage {
  type: 'ready';
}

export interface WorkerErrorMessage extends WorkerMessage {
  type: 'error';
  data: {
    message: string;
    chunkId?: number;
  };
}

export interface WorkerTranscriptionCompleteMessage extends WorkerMessage {
  type: 'transcription-complete';
  data: {
    text: string;
    chunkId: number;
  };
}

export type PendingTranscription = {
  resolve: (result: { text: string }) => void;
  reject: (error: Error) => void;
  timestamp: number;
};