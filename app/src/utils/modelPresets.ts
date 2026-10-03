import type { GemmaDtype } from './localLlm/types';

export type ModelPreset = {
  name: string;
  sizeLabel: string;
  engine: 'llamacpp' | 'transformers';
  ggufUrl?: string;
  mmprojUrl?: string;
  // llamacpp only: HF repo the gguf comes from; the UI lists its other quants from the HF API.
  repo?: string;
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

// `size` is a hardcoded estimate (default quant gguf + mmproj, from HF file sizes on 2026-10-02): it's only a label.
// One entry per model, listing every engine it can run on. The Models UI renders one row per entry
// with a download button for each engine; MODEL_PRESETS below flattens it for the other consumers.
// NOTE: the app stores downloads by URL basename, so every gguf/mmproj filename must be unique
// across entries (generic names like unsloth's `mmproj-F16.gguf` would overwrite each other).
export type CatalogModel = {
  name: string;
  params: string;   // parameter count shown under the name, e.g. "2B"
  size: string;     // hardcoded estimate of the default download (default quant gguf + mmproj), shown next to it
  llamacpp?: { repo: string; ggufUrl: string; mmprojUrl?: string };
  transformers?: { hfModelId: string; dtypes?: GemmaDtype[] };
};

export const MODEL_CATALOG: CatalogModel[] = [
  {
    name: 'Gemma 4 E2B',
    params: '2B',
    size: '3.4 GB',
    llamacpp: {
      repo: 'unsloth/gemma-4-E2B-it-GGUF',
      ggufUrl: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q3_K_S.gguf',
      mmprojUrl: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf',
    },
    transformers: { hfModelId: 'onnx-community/gemma-4-E2B-it-ONNX' },
  },
  {
    name: 'Qwen3.5',
    params: '0.8B',
    size: '0.6 GB',
    llamacpp: {
      repo: 'unsloth/Qwen3.5-0.8B-GGUF',
      ggufUrl: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q3_K_S.gguf',
      // unsloth's projectors are all named `mmproj-*.gguf` and would overwrite the Gemma ones on disk,
      // so use bartowski's uniquely named f16 projector (same base model).
      mmprojUrl: 'https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/mmproj-Qwen_Qwen3.5-0.8B-f16.gguf',
    },
    transformers: { hfModelId: 'onnx-community/Qwen3.5-0.8B-ONNX' },
  },
  {
    name: 'Qwen3-VL',
    params: '2B',
    size: '1.9 GB',
    llamacpp: {
      repo: 'Qwen/Qwen3-VL-2B-Instruct-GGUF',
      ggufUrl: 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf',
      mmprojUrl: 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-F16.gguf',
    },
    transformers: { hfModelId: 'onnx-community/Qwen3-VL-2B-Instruct-ONNX' },
  },
  {
    name: 'Gemma 4 E4B',
    params: '4B',
    size: '5.0 GB',
    llamacpp: {
      repo: 'unsloth/gemma-4-E4B-it-GGUF',
      ggufUrl: 'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q3_K_M.gguf',
      mmprojUrl: 'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf',
    },
    transformers: { hfModelId: 'onnx-community/gemma-4-E4B-it-ONNX' },
  },
  {
    name: 'LFM2.5-VL',
    params: '450M',
    size: '0.6 GB',
    llamacpp: {
      repo: 'LiquidAI/LFM2.5-VL-450M-GGUF',
      ggufUrl: 'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-Q8_0.gguf',
      mmprojUrl: 'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf',
    },
    transformers: { hfModelId: 'onnx-community/LFM2.5-VL-450M-ONNX' },
  },
];

export const MODEL_PRESETS: ModelPreset[] = MODEL_CATALOG.flatMap(m => [
  ...(m.llamacpp ? [{ name: m.name, sizeLabel: m.size, engine: 'llamacpp' as const, ...m.llamacpp }] : []),
  ...(m.transformers ? [{ name: m.name, sizeLabel: m.size, engine: 'transformers' as const, ...m.transformers }] : []),
]);
