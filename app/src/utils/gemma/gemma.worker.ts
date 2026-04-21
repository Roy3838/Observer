import { GemmaModelId, GemmaDevice, GemmaDtype } from './types';

let processor: any = null;
let model: any = null;
let TextStreamer: any = null;
let load_image: any = null;
let transformersModule: any = null;

async function loadTransformers() {
  if (!transformersModule) {
    transformersModule = await import('@huggingface/transformers');
    TextStreamer = transformersModule.TextStreamer;
    load_image = transformersModule.load_image;
  }
  return transformersModule;
}

// Extract images from multimodal message content
async function extractImages(messages: Array<{ role: string; content: any }>): Promise<any[]> {
  const images: any[] = [];

  for (const msg of messages) {
    if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (part.type === 'image' && part.image) {
          // part.image can be a URL string or a Blob
          const img = await load_image(part.image);
          images.push(img);
        }
      }
    }
  }

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
        processor = null;
        model = null;

        const { AutoProcessor, Gemma4ForConditionalGeneration } = await loadTransformers();

        const progressCallback = (info: any) => {
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

        // Extract images from multimodal messages
        const images = await extractImages(messages);

        const prompt = processor.apply_chat_template(messages, {
          enable_thinking: false,
          add_generation_prompt: true,
        });

        // Use processor directly for multimodal: processor(prompt, image, audio, options)
        // Pass null for unused modalities
        const inputs = images.length > 0
          ? await processor(prompt, images, null, { add_special_tokens: false })
          : await processor(prompt, null, null, { add_special_tokens: false });

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
