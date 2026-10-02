import type { GemmaDtype } from './localLlm/types';

export type ModelPreset = {
  name: string;
  sizeLabel: string;
  engine: 'llamacpp' | 'transformers';
  ggufUrl?: string;
  mmprojUrl?: string;
  hfModelId?: string;
  // transformers only: dtypes with files in the repo. Omit when all are published.
  dtypes?: GemmaDtype[];
};

// Extended quant ladder for testing across devices.
// Shared mmprojUrl per repo — only the main GGUF changes per quant.
const UNSLOTH_E2B_MMPROJ = 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf';
const UNSLOTH_E2B_BASE = 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-';
const GGML_E2B_MMPROJ = 'https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/mmproj-gemma-4-E2B-it-Q8_0.gguf';
const GGML_E2B_BASE = 'https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-';

export const EXTENDED_PRESETS: ModelPreset[] = [
  // ── Unsloth quantizations (lightest → heaviest) ──────────────────────────
  { name: 'Unsloth E2B UD-IQ2_M',   sizeLabel: '~2.3 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}UD-IQ2_M.gguf`,   mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B UD-IQ3_XXS', sizeLabel: '~2.4 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}UD-IQ3_XXS.gguf`, mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B UD-Q2_K_XL', sizeLabel: '~2.4 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}UD-Q2_K_XL.gguf`, mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q3_K_S',     sizeLabel: '~2.5 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q3_K_S.gguf`,     mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q3_K_M',     sizeLabel: '~2.5 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q3_K_M.gguf`,     mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B UD-Q3_K_XL', sizeLabel: '~2.9 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}UD-Q3_K_XL.gguf`, mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B IQ4_XS',     sizeLabel: '~3.0 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}IQ4_XS.gguf`,    mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B IQ4_NL',     sizeLabel: '~3.0 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}IQ4_NL.gguf`,    mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q4_0',       sizeLabel: '~3.0 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q4_0.gguf`,      mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q4_K_S',     sizeLabel: '~3.0 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q4_K_S.gguf`,    mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q4_K_M',     sizeLabel: '~3.1 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q4_K_M.gguf`,    mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q4_1',       sizeLabel: '~3.2 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q4_1.gguf`,      mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B UD-Q4_K_XL', sizeLabel: '~3.2 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}UD-Q4_K_XL.gguf`, mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q5_K_S',     sizeLabel: '~3.3 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q5_K_S.gguf`,    mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q5_K_M',     sizeLabel: '~3.4 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q5_K_M.gguf`,    mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B UD-Q5_K_XL', sizeLabel: '~4.3 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}UD-Q5_K_XL.gguf`, mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q6_K',       sizeLabel: '~4.5 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q6_K.gguf`,      mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B UD-Q6_K_XL', sizeLabel: '~4.7 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}UD-Q6_K_XL.gguf`, mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B Q8_0',       sizeLabel: '~5.1 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}Q8_0.gguf`,      mmprojUrl: UNSLOTH_E2B_MMPROJ },
  { name: 'Unsloth E2B UD-Q8_K_XL', sizeLabel: '~5.3 GB', engine: 'llamacpp', ggufUrl: `${UNSLOTH_E2B_BASE}UD-Q8_K_XL.gguf`, mmprojUrl: UNSLOTH_E2B_MMPROJ },
  // ── ggml-org quantizations ───────────────────────────────────────────────
  { name: 'ggml E2B Q8_0',  sizeLabel: '~5.0 GB', engine: 'llamacpp', ggufUrl: `${GGML_E2B_BASE}Q8_0.gguf`,  mmprojUrl: GGML_E2B_MMPROJ },
  { name: 'ggml E2B BF16',  sizeLabel: '~9.3 GB', engine: 'llamacpp', ggufUrl: `${GGML_E2B_BASE}bf16.gguf`,  mmprojUrl: GGML_E2B_MMPROJ },
];

export const MODEL_PRESETS: ModelPreset[] = [
  {
    name: 'Gemma 4 E2B',
    sizeLabel: '~3 GB',
    engine: 'llamacpp',
    ggufUrl: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q3_K_S.gguf',
    mmprojUrl: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf',
  },
  {
    name: 'gemma-4-E2B',
    sizeLabel: '~3 GB will crash on mobile',
    engine: 'transformers',
    hfModelId: 'onnx-community/gemma-4-E2B-it-ONNX',
  },
  {
    name: 'Qwen3.5-0.8B',
    sizeLabel: '~1 GB',
    engine: 'transformers',
    hfModelId: 'onnx-community/Qwen3.5-0.8B-ONNX',
  },
  {
    name: 'Qwen3-VL-2B',
    sizeLabel: '~2 GB',
    engine: 'transformers',
    hfModelId: 'onnx-community/Qwen3-VL-2B-Instruct-ONNX',
  },
  {
    name: 'Gemma 4 E4B',
    sizeLabel: '~5 GB',
    engine: 'llamacpp',
    ggufUrl: 'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q3_K_M.gguf',
    mmprojUrl: 'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf',
  },
  {
    name: 'Dots OCR',
    sizeLabel: '~5 GB',
    engine: 'llamacpp',
    ggufUrl: 'https://huggingface.co/ggml-org/dots.ocr-GGUF/resolve/main/dots.ocr-Q8_0.gguf',
    mmprojUrl: 'https://huggingface.co/ggml-org/dots.ocr-GGUF/resolve/main/mmproj-dots.ocr-Q8_0.gguf',
  },
  {
    name: 'Gemma 4 E4B',
    sizeLabel: '~5 GB will crash on mobile',
    engine: 'transformers',
    hfModelId: 'onnx-community/gemma-4-E4B-it-ONNX',
  },
  {
    name: 'LFM2.5-VL-450M',
    sizeLabel: '~0.5 GB, hallucinates — best for simple describer agents',
    engine: 'transformers',
    hfModelId: 'onnx-community/LFM2.5-VL-450M-ONNX',
  },
  {
    name: 'FastVLM 0.5B',
    sizeLabel: '~0.7 GB, hallucinates — best for simple describer agents',
    engine: 'transformers',
    hfModelId: 'onnx-community/FastVLM-0.5B-ONNX',
  },
];
