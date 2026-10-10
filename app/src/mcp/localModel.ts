// src/mcp/localModel.ts
//
// One-call acquisition of Observer's default *local* model, mirroring the primary
// path of AvailableModels.tsx's curated catalog:
//   - Browser  → the Qwen3.5 0.8B ONNX transformers.js preset (download + load in one shot)
//   - Desktop  → the Qwen3.5 0.8B llama.cpp preset (download gguf + mmproj, then load)
//
// The MCP exposes this as the `download_model` tool. We block until the model is actually
// loaded so the agentic loop can go straight to create_agent with a real, ready model
// name — no race against a half-downloaded file.
//
// `download_model({ model: 'onejev' })` instead fetches OneJev, the decision model for
// dial watchers (a percentage, timer or progress bar). It only exists for transformers.js, which runs on every platform.

import { isTauri } from '@utils/platform';
import { ModelManager } from '@utils/ModelManager';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import { MODEL_PRESETS } from '@utils/modelPresets';
import type { GemmaModelId, GemmaImageTokenBudget } from '@utils/localLlm/types';

/** Result mirrors a row from ModelManager.listModels(): the name to use in create_agent + its sentinel server. */
export interface DownloadedLocalModel {
  model_name: string;
  server: string;
  loaded: boolean;
  decision: boolean;   // decision model: agent code reads `decision`, not `response`
}

/** Models `download_model` can fetch. 'default' is the platform's general-purpose LLM. */
export type LocalModelChoice = 'default' | 'onejev';
export const LOCAL_MODEL_CHOICES: LocalModelChoice[] = ['default', 'onejev'];

// Single source of truth: pull the two default presets straight out of the catalog.
const TRANSFORMERS_PRESET = MODEL_PRESETS.find(
  p => p.engine === 'transformers' && p.hfModelId?.includes('Qwen3.5-0.8B'),
);
const LLAMACPP_PRESET = MODEL_PRESETS.find(
  p => p.engine === 'llamacpp' && p.ggufUrl?.includes('Qwen3.5-0.8B'),
);
const ONEJEV_PRESET = MODEL_PRESETS.find(
  p => p.engine === 'transformers' && p.hfModelId?.includes('OneJev'),
);

// OneJev reads small text badly below this many vision tokens per image (70 is the global default).
const ONEJEV_MIN_IMAGE_TOKEN_BUDGET: GemmaImageTokenBudget = 280;

/** Files the desktop default model downloads, in order (gguf, then vision projector) — lets the
 *  progress UI list both from the first byte. Names are extension-less, matching NativeModelState.modelId. */
export const DEFAULT_LLAMACPP_FILES: string[] = LLAMACPP_PRESET
  ? [LLAMACPP_PRESET.ggufUrl, LLAMACPP_PRESET.mmprojUrl]
      .filter((u): u is string => !!u)
      .map(u => u.split('/').pop()!.replace(/\.gguf$/i, ''))
  : [];

/** Resolve once the Gemma worker reaches `loaded` for `modelId` (or reject on `error`). */
function awaitGemmaLoaded(modelId: GemmaModelId): Promise<void> {
  const mgr = GemmaModelManager.getInstance();
  return new Promise<void>((resolve, reject) => {
    const check = (s = mgr.getState()) => {
      if (s.status === 'loaded' && s.modelId === modelId) { unsub(); resolve(); }
      else if (s.status === 'error') { unsub(); reject(new Error(s.error || 'Model failed to load.')); }
      else if (s.status === 'unloaded') { unsub(); reject(new Error('Download cancelled.')); }
    };
    const unsub = mgr.onStateChange(check);
    check(); // settle synchronously if it's already loaded
  });
}

/** transformers.js download+load is a single atomic call. */
async function downloadTransformers(
  preset: typeof TRANSFORMERS_PRESET,
  minImageTokenBudget?: GemmaImageTokenBudget,
): Promise<DownloadedLocalModel> {
  if (!preset?.hfModelId) throw new Error('No transformers.js preset configured.');
  const modelId = preset.hfModelId as GemmaModelId;
  const mgr = GemmaModelManager.getInstance();
  const runtime = mgr.getRuntimeSettings();
  const imageTokenBudget = minImageTokenBudget && runtime.imageTokenBudget < minImageTokenBudget
    ? minImageTokenBudget
    : runtime.imageTokenBudget;

  const state = mgr.getState();
  const alreadyReady = mgr.isReady() && state.modelId === modelId
    && (state.loadSettings?.imageTokenBudget ?? 0) >= imageTokenBudget;
  if (!alreadyReady) {
    // Some repos don't publish every dtype (preset.dtypes): fall back to one they do, as the Models UI does.
    const saved = mgr.getSettingsForModel(modelId).dtype;
    const dtype = !preset.dtypes || preset.dtypes.includes(saved) ? saved : preset.dtypes[0];
    // Reloading the same model with a new budget needs an unload first (load is a no-op when loaded).
    if (state.modelId === modelId) mgr.unloadModel();
    // loadModelWithSettings is fire-and-forget (it posts to a worker); await the state.
    mgr.loadModelWithSettings(modelId, runtime.device, dtype, imageTokenBudget, runtime.enableThinking);
    await awaitGemmaLoaded(modelId);
  }

  const entry = mgr.listLocalModels().find(e => e.id === modelId);
  return { model_name: entry?.name ?? preset.name, server: ModelManager.BROWSER_LOCAL, loaded: true, decision: !!entry?.isDecision };
}

/** Desktop path: download the gguf + vision projector, assign it, then load. */
async function downloadLlamaCpp(): Promise<DownloadedLocalModel> {
  if (!LLAMACPP_PRESET?.ggufUrl) throw new Error('No llama.cpp preset configured.');
  const mgr = NativeLlmManager.getInstance();
  const ggufUrl = LLAMACPP_PRESET.ggufUrl;
  const mmprojUrl = LLAMACPP_PRESET.mmprojUrl;
  const ggufFilename = ggufUrl.split('/').pop()!;
  const mmprojFilename = mmprojUrl?.split('/').pop();

  const onDisk = await mgr.listGgufFiles();
  const hasComplete = (fn: string) => onDisk.some(f => f.filename === fn);

  // Model file
  if (!hasComplete(ggufFilename)) {
    const result = await mgr.downloadModel(ggufUrl);
    if (!result) throw new Error('Download cancelled.');
  }
  // Vision projector — assign before downloading so it renders inside the model's card,
  // then fetch it if it isn't already on disk.
  if (mmprojUrl && mmprojFilename) {
    mgr.setMmprojAssignment(ggufFilename, mmprojFilename);
    if (!hasComplete(mmprojFilename)) {
      await mgr.downloadModel(mmprojUrl);
    }
  }

  // downloadModel refreshes the gguf cache fire-and-forget; loadModel checks that cache to see
  // whether the projector is complete, so without a fresh listing it would load text-only.
  await mgr.listGgufFiles();

  // loadModel awaits the native llm_load_model call, so this resolves only when loaded.
  await mgr.loadModel(ggufFilename);

  const name = mgr.listNativeModels().find(e => e.id === ggufFilename)?.name ?? ggufFilename;
  return { model_name: name, server: ModelManager.LLAMA_CPP_LOCAL, loaded: true, decision: false };
}

/**
 * Download (and load) a local model. 'default' is this platform's general-purpose model;
 * 'onejev' is the decision model (transformers.js on every platform). Blocks until the model
 * is ready for inference. Progress is broadcast via the underlying managers' state, which
 * the MCP UI subscribes to for live progress bars.
 */
export async function downloadLocalModel(choice: LocalModelChoice = 'default'): Promise<DownloadedLocalModel> {
  if (choice === 'onejev') return downloadTransformers(ONEJEV_PRESET, ONEJEV_MIN_IMAGE_TOKEN_BUDGET);
  return isTauri() ? downloadLlamaCpp() : downloadTransformers(TRANSFORMERS_PRESET);
}
