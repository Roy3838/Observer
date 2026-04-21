import { env } from '@huggingface/transformers';
import { GemmaModelId, GemmaDevice, GemmaDtype, GemmaImageTokenBudget } from './types';

// Enable browser Cache API for model persistence
env.useBrowserCache = true;
env.cacheKey = 'observer-transformers-cache';

let processor: any = null;
let model: any = null;
let TextStreamer: any = null;
let load_image: any = null;
let transformersModule: any = null;
let currentImageTokenBudget: GemmaImageTokenBudget = 280;

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

        console.log('[Gemma Worker] Loading model:', modelId, 'device:', device, 'dtype:', dtype, 'imageTokenBudget:', currentImageTokenBudget);

        const { AutoProcessor, Gemma4ForConditionalGeneration } = await loadTransformers();

        const progressCallback = (info: any) => {
          // "done" status with no download = loaded from cache
          if (info.status === 'done' && info.loaded === 0) {
            console.log(`[Gemma Worker] Loaded from cache: ${info.file}`);
          }
          self.postMessage({ type: 'progress', data: info });
        };

        processor = await AutoProcessor.from_pretrained(modelId, {
          progress_callback: progressCallback,
        });

        model = await Gemma4ForConditionalGeneration.from_pretrained(modelId, {
          dtype,
          device,
          progress_callback: progressCallback,
        });

        self.postMessage({ type: 'ready' });
        break;
      }

      case 'generate': {
        const { messages, generationId } = data;

        if (!processor || !model) {
          throw new Error('Model not loaded');
        }

        console.log('[Gemma Worker] Received messages:', JSON.stringify(messages, null, 2).slice(0, 500));

        // Extract images from multimodal messages
        const images = await extractImages(messages);

        // Transform messages for chat template:
        // Replace image_url/image content with simple { type: "image" } placeholders
        const templateMessages = messages.map((msg: { role: string; content: any }) => {
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
          enable_thinking: false,
          add_generation_prompt: true,
        });

        console.log('[Gemma Worker] Generated prompt length:', prompt.length);

        // For multimodal, pass first image (processor expects single RawImage)
        // For text-only, use tokenizer directly
        let inputs;
        if (images.length > 0) {
          console.log('[Gemma Worker] Processing with image, token budget:', currentImageTokenBudget);
          inputs = await processor(prompt, images[0], null, {
            add_special_tokens: false,
            max_soft_tokens: currentImageTokenBudget,
          });
          console.log('[Gemma Worker] Inputs created with image');
        } else {
          console.log('[Gemma Worker] Processing text-only...');
          inputs = processor.tokenizer(prompt, { add_special_tokens: false, return_tensors: 'pt' });
          console.log('[Gemma Worker] Inputs created text-only');
        }

        let fullText = '';

        const streamer = new TextStreamer(processor.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (token: string) => {
            fullText += token;
            self.postMessage({ type: 'generation-token', data: { token, generationId } });
          },
        });

        await model.generate({
          ...inputs,
          max_new_tokens: 2048,
          do_sample: false,
          streamer,
        });

        self.postMessage({ type: 'generation-complete', data: { text: fullText, generationId } });
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
