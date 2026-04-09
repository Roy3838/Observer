// src/config/inference-params.ts
// Types and defaults for inference parameters passed to LLM endpoints

/**
 * Inference parameters that can be passed to v1/chat/completions endpoints.
 * These are supported by most OpenAI-compatible inference engines
 * (Ollama, vLLM, llama.cpp, LMStudio, etc.)
 */
export interface InferenceParams {
  // Tier 1 - Universal OpenAI-compatible parameters
  /** Controls randomness. 0 = deterministic, 2 = very random. Default: 0.7 */
  temperature?: number;
  /** Nucleus sampling. Only consider tokens with cumulative probability <= top_p. Default: 0.9 */
  top_p?: number;
  /** Maximum tokens to generate. -1 for model default/unlimited. */
  max_tokens?: number;
  /** Random seed for reproducible outputs. */
  seed?: number;
  /** Stop sequences - generation stops when these are encountered. */
  stop?: string[];

  // Tier 2 - Common extensions (supported by most engines)
  /** Penalize tokens based on frequency in output so far. Range: -2.0 to 2.0 */
  frequency_penalty?: number;
  /** Penalize tokens that have appeared at all. Range: -2.0 to 2.0 */
  presence_penalty?: number;
  /** Top-k sampling. Only consider top k tokens. (vLLM, llama.cpp, Ollama native) */
  top_k?: number;

  // Tier 3 - Thinking/Reasoning control
  /**
   * Controls reasoning effort for thinking models.
   * Supported by Ollama (with reasoning models) and some cloud providers.
   * 'none' disables reasoning/thinking.
   */
  reasoning_effort?: 'none' | 'low' | 'medium' | 'high';
  /**
   * Explicitly enable/disable thinking mode for Qwen3 models via vLLM.
   * Passed as chat_template_kwargs.enable_thinking
   */
  enable_thinking?: boolean;

  // Tier 4 - Custom API Configuration (Power Users)
  /**
   * Custom API key for BYOK (Bring Your Own Key) scenarios.
   * When set, this key is used instead of the default API key.
   * Passed as Bearer token in Authorization header.
   */
  customApiKey?: string;
}

/**
 * Default inference parameters used when no overrides are specified.
 * These are conservative defaults that work well across most models.
 */
export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  // Intentionally minimal - let the model/engine use its own defaults
  // unless the user explicitly configures something
};

/**
 * Labels and constraints for UI rendering
 */
export const INFERENCE_PARAM_METADATA: Record<keyof InferenceParams, {
  label: string;
  description: string;
  min?: number;
  max?: number;
  step?: number;
  type: 'number' | 'boolean' | 'string[]' | 'select' | 'password';
  options?: string[];
}> = {
  temperature: {
    label: 'Temperature',
    description: 'Controls randomness. Lower = more focused, higher = more creative.',
    min: 0,
    max: 2,
    step: 0.1,
    type: 'number',
  },
  top_p: {
    label: 'Top P (Nucleus Sampling)',
    description: 'Only consider tokens within this cumulative probability.',
    min: 0,
    max: 1,
    step: 0.05,
    type: 'number',
  },
  max_tokens: {
    label: 'Max Tokens',
    description: 'Maximum tokens to generate. -1 for unlimited.',
    min: -1,
    max: 32000,
    step: 100,
    type: 'number',
  },
  seed: {
    label: 'Seed',
    description: 'Random seed for reproducible outputs.',
    min: 0,
    max: 2147483647,
    step: 1,
    type: 'number',
  },
  stop: {
    label: 'Stop Sequences',
    description: 'Generation stops when these strings are encountered.',
    type: 'string[]',
  },
  frequency_penalty: {
    label: 'Frequency Penalty',
    description: 'Penalize tokens based on how often they appear.',
    min: -2,
    max: 2,
    step: 0.1,
    type: 'number',
  },
  presence_penalty: {
    label: 'Presence Penalty',
    description: 'Penalize tokens that have appeared at all.',
    min: -2,
    max: 2,
    step: 0.1,
    type: 'number',
  },
  top_k: {
    label: 'Top K',
    description: 'Only consider the top K most likely tokens.',
    min: 1,
    max: 100,
    step: 1,
    type: 'number',
  },
  reasoning_effort: {
    label: 'Reasoning Effort',
    description: 'Controls thinking depth for reasoning models. "none" disables thinking.',
    type: 'select',
    options: ['none', 'low', 'medium', 'high'],
  },
  enable_thinking: {
    label: 'Enable Thinking (Qwen3)',
    description: 'Enable/disable thinking mode for Qwen3 models via vLLM.',
    type: 'boolean',
  },
  customApiKey: {
    label: 'Custom API Key',
    description: 'Your own API key for BYOK. Used as Bearer token in Authorization header.',
    type: 'password',
  },
};
