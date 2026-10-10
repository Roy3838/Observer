import { env } from '@huggingface/transformers';
import { GemmaModelId, GemmaDevice, GemmaDtype, GemmaImageTokenBudget, isDecisionModelId } from './types';
import { LETTERS } from './systemOne';

// Enable browser Cache API for model persistence
env.useBrowserCache = true;
env.cacheKey = 'observer-transformers-cache';

let processor: any = null;
let model: any = null;
let TextStreamer: any = null;
let load_image: any = null;
let transformersModule: any = null;
let currentImageTokenBudget: GemmaImageTokenBudget = 280;
let currentFamily: ModelFamily | null = null;
let slotTokenIds: number[] | null = null;  // token ids of the option letters A..Z, for decision models

// Per-architecture differences, keyed by `config.model_type` from the model's config.json.
// Model + processor classes are resolved by transformers.js itself (AutoModelForImageTextToText /
// AutoProcessor); only the processor call signature and optional features differ per family.
type ModelFamily = {
  supportsThinking: boolean;
  // Set for templates that expect plain-string content with an inline image token
  // (e.g. FastVLM's `<image>`) instead of structured `{ type: 'image' }` parts.
  inlineImageToken?: string;
  // Applied once after the processor loads, to map Observer's image token budget onto the
  // model's own resolution limits (only needed when the model's default is unbounded).
  configureProcessor?: (proc: any, imageTokenBudget: number) => void;
  buildImageInputs: (proc: any, prompt: string, images: any[], imageTokenBudget: number) => Promise<any>;
};

// Qwen2_5_VLProcessor._call(text, images) — image placeholders come from the chat template.
// Shared by qwen3_vl and qwen3_5 (both use Qwen3VLProcessor + Qwen2VLImageProcessor).
const QWEN_VL_FAMILY: ModelFamily = {
  supportsThinking: false,
  // preprocessor_config.json allows up to 16.7M pixels (~16k vision tokens) per image, which
  // blows past WebGPU memory on a full screenshot. One vision token covers a
  // (patch_size * merge_size)^2 = 32x32 px area, so budget * 1024 caps the token count.
  configureProcessor: (proc, imageTokenBudget) => {
    const ip = proc.image_processor;
    const factor = (ip.patch_size ?? 16) * (ip.merge_size ?? 2);
    ip.max_pixels = imageTokenBudget * factor * factor;
    ip.min_pixels = Math.min(ip.min_pixels ?? 0, ip.max_pixels);
  },
  buildImageInputs: (proc, prompt, images) => proc(prompt, images),
};

const MODEL_FAMILIES: Record<string, ModelFamily> = {
  // Gemma4Processor._call(text, images, audio, options)
  gemma4: {
    supportsThinking: true,
    buildImageInputs: (proc, prompt, images, imageTokenBudget) =>
      // Processor expects a single RawImage here
      proc(prompt, images[0], null, { add_special_tokens: false, max_soft_tokens: imageTokenBudget }),
  },
  // Lfm2VlProcessor._call(images, text, kwargs)
  lfm2_vl: {
    supportsThinking: false,
    buildImageInputs: (proc, prompt, images) =>
      proc(images, prompt, { add_special_tokens: false }),
  },
  qwen3_vl: QWEN_VL_FAMILY,
  qwen3_5: QWEN_VL_FAMILY,
  // LlavaProcessor._call(images, text, kwargs) — chat template takes string content with `<image>`;
  // the processor only expands the first `<image>`, so a single image is supported.
  llava_qwen2: {
    supportsThinking: false,
    inlineImageToken: '<image>',
    buildImageInputs: (proc, prompt, images) =>
      proc(images[0], prompt, { add_special_tokens: false }),
  },
};

async function loadTransformers() {
  if (!transformersModule) {
    transformersModule = await import('@huggingface/transformers');
    TextStreamer = transformersModule.TextStreamer;
    load_image = transformersModule.load_image;
  }
  return transformersModule;
}

// Extract images from multimodal message content
// Supports both Gemma format ({ type: 'image', image: url })
// and OpenAI format ({ type: 'image_url', image_url: { url: ... } })
async function extractImages(messages: Array<{ role: string; content: any }>): Promise<any[]> {
  const images: any[] = [];

  for (const msg of messages) {
    if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        let imageSource: string | Blob | null = null;

        // Gemma format: { type: 'image', image: url }
        if (part.type === 'image' && part.image) {
          imageSource = part.image;
        }
        // OpenAI format: { type: 'image_url', image_url: { url: ... } }
        else if (part.type === 'image_url' && part.image_url?.url) {
          imageSource = part.image_url.url;
        }

        if (imageSource) {
          console.log('[Gemma Worker] Loading image, source length:', typeof imageSource === 'string' ? imageSource.length : 'Blob');
          const img = await load_image(imageSource);
          console.log('[Gemma Worker] Image loaded successfully');
          images.push(img);
        }
      }
    }
  }

  console.log('[Gemma Worker] Total images extracted:', images.length);
  return images;
}

// Chat messages → model inputs (images, chat template, processor call for this family)
async function buildInputs(messages: Array<{ role: string; content: any }>, family: ModelFamily, thinking: boolean) {
  // Extract images from multimodal messages
  const images = await extractImages(messages);

  // Transform messages for chat template:
  // Replace image_url/image content with simple { type: "image" } placeholders
  const templateMessages = messages.map((msg: { role: string; content: any }) => {
    if (Array.isArray(msg.content) && family.inlineImageToken) {
      // Flatten to a string: image token (first image only) followed by the text parts
      let imageEmitted = false;
      const content = msg.content.map((part: any) => {
        if (part.type === 'image_url' || part.type === 'image') {
          if (imageEmitted) return '';
          imageEmitted = true;
          return family.inlineImageToken;
        }
        return part.text ?? '';
      }).join('');
      return { ...msg, content };
    }
    if (Array.isArray(msg.content)) {
      return {
        ...msg,
        content: msg.content.map((part: any) => {
          // Convert image_url or image parts to simple placeholder
          if (part.type === 'image_url' || part.type === 'image') {
            return { type: 'image' };
          }
          return part;
        })
      };
    }
    return msg;
  });

  console.log('[Gemma Worker] Template messages:', JSON.stringify(templateMessages, null, 2).slice(0, 500));
  console.log('[Gemma Worker] Images extracted:', images.length);

  const prompt = processor.apply_chat_template(templateMessages, {
    enable_thinking: thinking,
    add_generation_prompt: true,
  });

  console.log('[Gemma Worker] Generated prompt length:', prompt.length);

  // For multimodal, the processor call signature depends on the model family
  // For text-only, use tokenizer directly
  if (images.length > 0) {
    console.log('[Gemma Worker] Processing with image, token budget:', currentImageTokenBudget);
    return family.buildImageInputs(processor, prompt, images, currentImageTokenBudget);
  }
  console.log('[Gemma Worker] Processing text-only...');
  return processor.tokenizer(prompt, { add_special_tokens: false, return_tensors: 'pt' });
}

// Token ids for the option letters, each of which must be a single token that decodes back to
// itself (qev engine._verify_slots) — otherwise the letter logits don't mean what we read.
function getSlotTokenIds(): number[] {
  if (slotTokenIds) return slotTokenIds;
  const tokenizer = processor.tokenizer;
  const ids = LETTERS.map(letter => {
    const enc: number[] = tokenizer.encode(letter, { add_special_tokens: false });
    if (enc.length !== 1 || tokenizer.decode(enc) !== letter) {
      throw new Error(`Option letter '${letter}' is not a single token in this tokenizer`);
    }
    return enc[0];
  });
  if (new Set(ids).size !== ids.length) throw new Error('Option letter tokens collide');
  slotTokenIds = ids;
  return ids;
}

self.onmessage = async (event: MessageEvent) => {
  const { type, data } = event.data;

  try {
    switch (type) {
      case 'load': {
        const modelId = data.modelId as GemmaModelId;
        const device = (data.device ?? 'webgpu') as GemmaDevice;
        const dtype = (data.dtype ?? 'q4f16') as GemmaDtype;
        currentImageTokenBudget = (data.imageTokenBudget ?? 280) as GemmaImageTokenBudget;
        processor = null;
        model = null;
        currentFamily = null;
        slotTokenIds = null;

        console.log('[Gemma Worker] Loading model:', modelId, 'device:', device, 'dtype:', dtype, 'imageTokenBudget:', currentImageTokenBudget);

        const { AutoConfig, AutoProcessor, AutoModelForImageTextToText } = await loadTransformers();

        const progressCallback = (info: any) => {
          // "done" status with no download = loaded from cache
          if (info.status === 'done' && info.loaded === 0) {
            console.log(`[Gemma Worker] Loaded from cache: ${info.file}`);
          }
          self.postMessage({ type: 'progress', data: info });
        };

        // config.json is the source of truth for which architecture (and therefore which ONNX
        // sessions: vision_encoder, audio_encoder, ...) transformers.js will load.
        const config = await AutoConfig.from_pretrained(modelId, { progress_callback: progressCallback });
        const family = MODEL_FAMILIES[config.model_type];
        if (!family) {
          throw new Error(`Unsupported model_type '${config.model_type}' for ${modelId}. Supported: ${Object.keys(MODEL_FAMILIES).join(', ')}`);
        }
        console.log('[Gemma Worker] model_type:', config.model_type);

        processor = await AutoProcessor.from_pretrained(modelId, {
          progress_callback: progressCallback,
        });

        family.configureProcessor?.(processor, currentImageTokenBudget);

        // OneJev only publishes f16 weights below fp32, and an f16 vision encoder on WebGPU leaves the
        // model blind (same answers for any image, on an Intel Gen 9 GPU). An fp32 vision encoder
        // (~400 MB) with the f16 text model matches qev's reference answers.
        const sessionDtype = isDecisionModelId(modelId) && dtype !== 'fp32' && device === 'webgpu'
          ? { embed_tokens: dtype, decoder_model_merged: dtype, vision_encoder: 'fp32' }
          : dtype;

        model = await AutoModelForImageTextToText.from_pretrained(modelId, {
          config,
          dtype: sessionDtype,
          device,
          progress_callback: progressCallback,
        });

        currentFamily = family;

        self.postMessage({ type: 'ready' });
        break;
      }

      case 'generate': {
        const { messages, generationId, enableThinking = false } = data;

        if (!processor || !model || !currentFamily) {
          throw new Error('Model not loaded');
        }
        const family = currentFamily;

        console.log('[Gemma Worker] Received messages:', JSON.stringify(messages, null, 2).slice(0, 500));

        const thinking = enableThinking && family.supportsThinking;
        const inputs = await buildInputs(messages, family, thinking);

        let fullText = '';

        // When thinking is enabled we keep special tokens so we can detect the
        // <|channel>thought\n … <channel|> delimiters that wrap the reasoning.
        //
        // State machine:
        //   'scanning'  – buffering output until we know if thinking block starts
        //   'thinking'  – inside the thinking block, routing to reasoning-token
        //   'answering' – past the thinking block, routing to generation-token
        const THINK_PREFIX = '<|channel>thought\n';
        const THINK_END    = '<channel|>';
        // Known extra special tokens to silently drop when skip_special_tokens is off
        const DROP_TOKENS  = new Set(['<bos>', '<eos>', '<|end_of_turn|>']);

        type ThinkState = 'scanning' | 'thinking' | 'answering';
        let thinkState: ThinkState = thinking ? 'scanning' : 'answering';
        let scanBuf = '';   // accumulator for prefix detection / partial-end detection

        const handleToken = (raw: string) => {
          if (DROP_TOKENS.has(raw)) return;

          if (thinkState === 'answering') {
            fullText += raw;
            self.postMessage({ type: 'generation-token', data: { token: raw, generationId } });
            return;
          }

          // --- scanning: try to match THINK_PREFIX ---
          if (thinkState === 'scanning') {
            scanBuf += raw;
            if (THINK_PREFIX.startsWith(scanBuf)) {
              if (scanBuf === THINK_PREFIX) {
                thinkState = 'thinking';
                scanBuf = '';
              }
              // else still building prefix – don't emit yet
              return;
            }
            // Mismatch: not a thinking response; emit buffered content as answer
            const flushed = scanBuf;
            scanBuf = '';
            thinkState = 'answering';
            fullText += flushed;
            self.postMessage({ type: 'generation-token', data: { token: flushed, generationId } });
            return;
          }

          // --- thinking: look for THINK_END, keep a tail buffer for partial matches ---
          if (thinkState === 'thinking') {
            const combined = scanBuf + raw;
            const endIdx = combined.indexOf(THINK_END);
            if (endIdx !== -1) {
              const thinkPart  = combined.slice(0, endIdx);
              const answerPart = combined.slice(endIdx + THINK_END.length);
              scanBuf = '';
              thinkState = 'answering';
              if (thinkPart) {
                self.postMessage({ type: 'reasoning-token', data: { token: thinkPart, generationId } });
              }
              if (answerPart) {
                fullText += answerPart;
                self.postMessage({ type: 'generation-token', data: { token: answerPart, generationId } });
              }
              return;
            }

            // Check if tail of combined could be the start of THINK_END
            let keepLen = 0;
            for (let pl = Math.min(THINK_END.length - 1, combined.length); pl > 0; pl--) {
              if (THINK_END.startsWith(combined.slice(-pl))) {
                keepLen = pl;
                break;
              }
            }
            const toEmit = combined.slice(0, combined.length - keepLen);
            scanBuf = combined.slice(combined.length - keepLen);
            if (toEmit) {
              self.postMessage({ type: 'reasoning-token', data: { token: toEmit, generationId } });
            }
          }
        };

        const streamer = new TextStreamer(processor.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: !thinking,
          callback_function: handleToken,
        });

        await model.generate({
          ...inputs,
          max_new_tokens: 2048,
          do_sample: false,
          streamer,
        });

        // Flush any remaining scan buffer as answer (shouldn't normally happen)
        if (scanBuf) {
          fullText += scanBuf;
          self.postMessage({ type: 'generation-token', data: { token: scanBuf, generationId } });
        }

        self.postMessage({ type: 'generation-complete', data: { text: fullText, generationId } });
        break;
      }

      // Decision models: one forward pass per branch (question), reading the logits of the option
      // letters at the answer position. Generating a single token lets transformers.js build the
      // multimodal position ids; a logits processor captures the raw scores before sampling.
      case 'decide': {
        const { branches, generationId } = data as {
          branches: Array<{ messages: Array<{ role: string; content: any }>; nSlots: number }>;
          generationId: number;
        };

        if (!processor || !model || !currentFamily) {
          throw new Error('Model not loaded');
        }
        const family = currentFamily;
        const slots = getSlotTokenIds();
        const { LogitsProcessor, LogitsProcessorList } = await loadTransformers();

        const logits: number[][] = [];
        for (const branch of branches) {
          if (branch.nSlots > slots.length) {
            throw new Error(`Question has ${branch.nSlots} options, maximum is ${slots.length}`);
          }
          const branchSlots = slots.slice(0, branch.nSlots);
          const capture: { logits: number[] | null } = { logits: null };

          class CaptureSlotLogits extends LogitsProcessor {
            _call(_inputIds: bigint[][], scores: any) {
              // scores: [batch=1, vocab] float32
              const row = scores.data as Float32Array;
              capture.logits = branchSlots.map(id => row[id]);
              return scores;
            }
          }
          const processors = new LogitsProcessorList();
          processors.push(new CaptureSlotLogits());

          const inputs = await buildInputs(branch.messages, family, false);
          await model.generate({
            ...inputs,
            max_new_tokens: 1,
            do_sample: false,
            repetition_penalty: 1.0,  // the prompt itself contains the letters; never penalise them
            logits_processor: processors,
          });

          if (!capture.logits) throw new Error('Decision forward pass produced no logits');
          logits.push(capture.logits);
        }

        self.postMessage({ type: 'decision-complete', data: { logits, generationId } });
        break;
      }

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      data: {
        message: error instanceof Error ? error.message : String(error),
        generationId: data?.generationId,
      },
    });
  }
};
