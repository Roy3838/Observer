export type ModelPreset = {
  name: string;
  sizeLabel: string;
  engine: 'llamacpp' | 'transformers';
  ggufUrl?: string;
  mmprojUrl?: string;
  hfModelId?: string;
};

export const MODEL_PRESETS: ModelPreset[] = [
  {
    name: 'Gemma 4 E2B',
    sizeLabel: '~3 GB',
    engine: 'llamacpp',
    ggufUrl: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q3_K_S.gguf',
    mmprojUrl: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf',
  },
  {
    name: 'Gemma 4 E2B ONNX',
    sizeLabel: '~3 GB',
    engine: 'transformers',
    hfModelId: 'onnx-community/gemma-4-E2B-it-ONNX',
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
    sizeLabel: '~1 GB',
    engine: 'llamacpp',
    ggufUrl: 'https://huggingface.co/ggml-org/dots.ocr-GGUF/resolve/main/dots.ocr-Q8_0.gguf',
    mmprojUrl: 'https://huggingface.co/ggml-org/dots.ocr-GGUF/resolve/main/mmproj-dots.ocr-Q8_0.gguf',
  },
  {
    name: 'Gemma 4 E4B ONNX',
    sizeLabel: '~5 GB',
    engine: 'transformers',
    hfModelId: 'onnx-community/gemma-4-E4B-it-ONNX',
  },
];
