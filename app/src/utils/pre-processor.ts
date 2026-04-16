// src/utils/pre-processor.ts

import { Logger } from './logging';
import { getAgentMemory, getAgentImageMemory } from './agent_database';
import { captureScreenImage } from './screenCapture';
import { captureCameraImage } from './cameraCapture';
import { StreamManager } from './streamManager';


// Define the result structure
export interface PreProcessorResult {
  modifiedPrompt: string;  // The text prompt with placeholders removed
  images?: string[];       // Base64 encoded images for the API
  imageSources?: {
    screen?: string;       // Single base64 from $SCREEN_64
    camera?: string;       // Single base64 from $CAMERA
    imemory?: string[];    // Array from $IMEMORY@agentid
  };
  microphone?: string;
  screenAudio?: string;
  allAudio?: string;
}

// Map of processor functions
type ProcessorFunction = (agentId: string, prompt: string, match: RegExpExecArray, iterationId?: string) => Promise<{
  replacementText?: string;
  images?: string[];
}>;

// Simple map of placeholder patterns to handler functions
const processors: Record<string, { regex: RegExp, handler: ProcessorFunction }> = {
  // Screen OCR processor
  'SCREEN_OCR': {
    regex: /\$SCREEN_OCR/g,
    handler: async (agentId: string, _prompt: string, _match: RegExpExecArray, iterationId?: string) => {
      try {
        // Lazy load OCR functionality - only loads when OCR is actually used!
        const { captureFrameAndOCR } = await import('./screenOCR');

        const { screenVideoStream } = StreamManager.getCurrentState();
        if (!screenVideoStream) throw new Error('Screen stream not available for OCR.');

        const ocrResult = await captureFrameAndOCR(screenVideoStream, agentId, 'screen'); 

        if (ocrResult.success && ocrResult.text) {
          Logger.debug(agentId, `OCR successful, text injected into prompt`);
          // Enhanced logging: Log the OCR text separately for better debugging
          Logger.info(agentId, `Screen OCR detected: ${ocrResult.text.length} characters`, { 
            logType: 'sensor-ocr', 
            iterationId,
            content: ocrResult.text 
          });
          return { replacementText: ocrResult.text };
        }
        Logger.error(agentId, `OCR failed: ${ocrResult.error || 'Unknown error'}`);
        return { replacementText: '[Error performing OCR]' };
      } catch (error) {
        Logger.error(agentId, `Error with screen capture for OCR: ${error instanceof Error ? error.message : String(error)}`);
        return { replacementText: '[Error with screen capture]' };
      }
    }
  },
  
  // Memory processor
  'MEMORY': {
    regex: /\$MEMORY(?:@([a-zA-Z0-9_]+))?/g,
    handler: async (_agentId: string, _prompt: string, match: RegExpExecArray) => {
      try {
        // match[1] will be undefined if no @agentId provided, so default to current agent
        const referencedAgentId = match[1] || _agentId;
        const memory = await getAgentMemory(referencedAgentId);
        return { replacementText: memory };
      } catch (error) {
        Logger.error(_agentId, `Error with memory retrieval: ${error instanceof Error ? error.message : String(error)}`);
        return { replacementText: `[Error with memory retrieval]` };
      }
    }
  },

  'SCREEN_64': {
    regex: /\$SCREEN_64/g,
    handler: async (agentId: string, _prompt: string, _match: RegExpExecArray, iterationId?: string) => {
      try {
        const { screenVideoStream } = StreamManager.getCurrentState();
        if (!screenVideoStream) throw new Error('Screen stream not available for image capture.');

        // Pass the existing stream to the utility
        const base64Image = await captureScreenImage(screenVideoStream, agentId, 'screen');

        if (base64Image) {
          // Basic check for data URI prefix, then the base64 part
          const parts = base64Image.split(',');
          const b64data = parts.length > 1 ? parts[1] : parts[0];
          if (!/^[A-Za-z0-9+/=]+$/.test(b64data) || b64data.length % 4 !== 0) { // Added length % 4 check
            Logger.error(agentId, `Invalid base64 image data`);
            return { replacementText: '[Error: Invalid image data]' };
          }
          Logger.debug(agentId, `Base64 image captured (length: ${base64Image.length})`);
          // Enhanced logging: Log screenshot capture separately
          Logger.info(agentId, `Screenshot captured (${Math.round(base64Image.length/1024)}KB)`, { 
            logType: 'sensor-screenshot', 
            iterationId,
            content: { size: base64Image.length, timestamp: Date.now() }
          });
          return { replacementText: '', images: [base64Image] };
        }
        Logger.error(agentId, `Screen capture for image failed`);
        return { replacementText: '[Error capturing screen]' };
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        Logger.error(agentId, `Error capturing screen: ${errorMessage}`);
        return { replacementText: '[Error with screen capture]' };
      }
    }
  },

  'CLIPBOARD': {
    regex: /\$CLIPBOARD/g,
    handler: async (agentId: string, _prompt: string, _match: RegExpExecArray, iterationId?: string) => {
      try {
        if (typeof navigator !== 'undefined' && navigator.clipboard && typeof navigator.clipboard.readText === 'function') {
          const clipboardText = await navigator.clipboard.readText();
          Logger.debug(agentId, `Retrieved clipboard text: "${clipboardText}"`);
          // Enhanced logging: Log clipboard content separately
          Logger.info(agentId, `Clipboard accessed (${clipboardText.length} characters)`, { 
            logType: 'sensor-clipboard', 
            iterationId,
            content: clipboardText 
          });
          return { replacementText: clipboardText };
        }
        Logger.warn(agentId, `navigator.clipboard.readText is not available for CLIPBOARD_TEXT.`);
        return { replacementText: '[Error: Clipboard API not available or permission denied]' };
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        Logger.error(agentId, `Error retrieving clipboard text: ${errorMessage}`);
        if (errorMessage.includes('NotAllowedError') || errorMessage.includes('Must be handling a user gesture') || errorMessage.includes('Read permission denied')) {
            return { replacementText: '[Error: Clipboard access denied or requires user interaction]' };
        }
        return { replacementText: `[Error retrieving clipboard text: ${errorMessage}]` };
      }
    }
  },

  'MICROPHONE': {
    regex: /\$MICROPHONE/g,
    handler: async (agentId: string, _prompt: string, _match: RegExpExecArray, iterationId?: string) => {
      try {
        // Get or create subscriber for this agent - accumulates transcript independently
        const subscriber = StreamManager.getOrCreateSubscriber(agentId, 'microphone');
        const transcript = subscriber.getTranscript();
        Logger.debug(agentId, `Retrieved microphone transcript via subscriber: "${transcript}"`);
        // Enhanced logging: Log audio transcript separately
        Logger.info(agentId, `Microphone transcript (${transcript.length} characters)`, {
          logType: 'sensor-audio',
          iterationId,
          content: { source: 'microphone', transcript: transcript }
        });
        return { replacementText: transcript };
      } catch (error: any) {
        Logger.error(agentId, `Error retrieving microphone transcript: ${error.message}`);
        return { replacementText: `[Error processing microphone input: ${error.message}]` };
      }
    }
  },


  'CAMERA': {
    regex: /\$CAMERA/g,
    handler: async (agentId: string, _prompt: string, _match: RegExpExecArray, iterationId?: string) => {
      try {
        const { cameraStream } = StreamManager.getCurrentState();
        if (!cameraStream) throw new Error('Camera stream not available for image capture.');
        
        // Pass the existing stream to the utility
        const base64Image = await captureCameraImage(cameraStream, agentId);

        if (base64Image) {
          // You can add the same base64 validation as SCREEN_64 if you like
          Logger.debug(agentId, `Base64 camera image captured (length: ${base64Image.length})`);
          // Enhanced logging: Log camera capture separately
          Logger.info(agentId, `Camera image captured (${Math.round(base64Image.length/1024)}KB)`, { 
            logType: 'sensor-camera', 
            iterationId,
            content: { size: base64Image.length, timestamp: Date.now() }
          });
          // Return an empty string to remove the placeholder, and provide the image data
          return { replacementText: '', images: [base64Image] };
        }
        
        Logger.error(agentId, `Camera capture for image failed`);
        return { replacementText: '[Error capturing camera image]' };
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        Logger.error(agentId, `Error capturing camera image: ${errorMessage}`);
        return { replacementText: `[Error with camera capture: ${errorMessage}]` };
      }
    }
  },

  'SCREEN_AUDIO': {
    regex: /\$SCREEN_AUDIO/g,
    handler: async (agentId: string, _prompt: string, _match: RegExpExecArray, iterationId?: string) => {
      try {
        // Get or create subscriber for this agent - accumulates transcript independently
        const subscriber = StreamManager.getOrCreateSubscriber(agentId, 'screenAudio');
        const transcript = subscriber.getTranscript();
        Logger.debug(agentId, `Retrieved system audio transcript via subscriber: "${transcript}"`);
        // Enhanced logging: Log audio transcript separately
        Logger.info(agentId, `Screen audio transcript (${transcript.length} characters)`, {
          logType: 'sensor-audio',
          iterationId,
          content: { source: 'screenAudio', transcript: transcript }
        });
        return { replacementText: transcript };
      } catch (error: any) {
        Logger.error(agentId, `Error retrieving system audio transcript: ${error.message}`);
        return { replacementText: `[Error processing system audio: ${error.message}]` };
      }
    }
  },

  // Handler for the combined audio stream
  'ALL_AUDIO': {
    regex: /\$ALL_AUDIO/g,
    handler: async (agentId: string, _prompt: string, _match: RegExpExecArray, iterationId?: string) => {
      try {
        // Get or create subscriber for this agent - accumulates transcript independently
        const subscriber = StreamManager.getOrCreateSubscriber(agentId, 'allAudio');
        const transcript = subscriber.getTranscript();
        Logger.debug(agentId, `Retrieved combined audio transcript via subscriber: "${transcript}"`);
        // Enhanced logging: Log audio transcript separately
        Logger.info(agentId, `Combined audio transcript (${transcript.length} characters)`, {
          logType: 'sensor-audio',
          iterationId,
          content: { source: 'allAudio', transcript: transcript }
        });
        return { replacementText: transcript };
      } catch (error: any) {
        Logger.error(agentId, `Error retrieving combined audio transcript: ${error.message}`);
        return { replacementText: `[Error processing combined audio: ${error.message}]` };
      }
    }
  },

  // Image memory processor
  'IMEMORY': {
    regex: /\$IMEMORY(?:@([a-zA-Z0-9_]+))?/g,
    handler: async (agentId: string, _prompt: string, match: RegExpExecArray, iterationId?: string) => {
      try {
        // match[1] will be undefined if no @agentId provided, so default to current agent
        const referencedAgentId = match[1] || agentId;
        const imageMemory = await getAgentImageMemory(referencedAgentId);
        Logger.debug(agentId, `Retrieved image memory for agent ${referencedAgentId}: ${imageMemory.length} images`);
        // Enhanced logging: Log image memory access separately
        Logger.info(agentId, `Image memory accessed (${imageMemory.length} images from ${referencedAgentId})`, {
          logType: 'sensor-imemory',
          iterationId,
          content: { sourceAgent: referencedAgentId, imageCount: imageMemory.length }
        });
        return { replacementText: '', images: imageMemory };
      } catch (error) {
        Logger.error(agentId, `Error with image memory retrieval: ${error instanceof Error ? error.message : String(error)}`);
        return { replacementText: '[Error with image memory retrieval]' };
      }
    }
  },

};

export async function preProcess(agentId: string, systemPrompt: string, iterationId?: string): Promise<PreProcessorResult> {
  let modifiedPrompt = systemPrompt;
  const result: PreProcessorResult = {
    modifiedPrompt,
    images: [],
    imageSources: {},
    microphone: "",
    screenAudio: "",
    allAudio: ""
  };
  
  try {
    Logger.debug(agentId, 'Starting prompt pre-processing');

    //if (/\$MICROPHONE/.test(systemPrompt)) {
    //    Logger.debug(agentId, '$MICROPHONE placeholder detected, ensuring recognition is started.');
    //    try {
    //        await ensureRecognitionStarted(agentId);
    //    } catch (speechError: any) {
    //        Logger.error(agentId, `Failed to ensure speech recognition active in preProcess: ${speechError.message}`);
    //        modifiedPrompt = modifiedPrompt.replace(/\$MICROPHONE/g, `[Speech input unavailable: ${speechError.message}]`);
    //    }
    //}

    for (const [key, processor] of Object.entries(processors)) {
      //if (key === '$MICROPHONE' && modifiedPrompt.includes('[Speech input unavailable:')) {
      //    continue;
      //}

      processor.regex.lastIndex = 0; 
      let match;
      
      // Need to re-construct the string in a way that doesn't invalidate ongoing regex exec
      let currentSearchIndex = 0;
      let tempPrompt = "";

      while ((match = processor.regex.exec(modifiedPrompt)) !== null) {
        const placeholder = match[0];
        const matchIndex = match.index;

        // Append the part of the string before the current match
        tempPrompt += modifiedPrompt.substring(currentSearchIndex, matchIndex);
        
        Logger.debug(agentId, `Processing placeholder: ${placeholder} (at index ${matchIndex} using ${key})`);
        const processorResult = await processor.handler(agentId, modifiedPrompt, match, iterationId);
        
        let replacement = placeholder; // Default to keeping the placeholder if no replacementText
        if (processorResult.replacementText !== undefined) {
          replacement = processorResult.replacementText;
        }
        tempPrompt += replacement;

        currentSearchIndex = matchIndex + placeholder.length;
        processor.regex.lastIndex = currentSearchIndex; // Ensure regex continues after this placeholder

        if (processorResult.images && processorResult.images.length > 0) {
          result.images = [...(result.images || []), ...processorResult.images];
          
          // Track image sources for agent context
          if (key === 'SCREEN_64' && processorResult.images[0]) {
            result.imageSources!.screen = processorResult.images[0];
          } else if (key === 'CAMERA' && processorResult.images[0]) {
            result.imageSources!.camera = processorResult.images[0];
          } else if (key === 'IMEMORY') {
            result.imageSources!.imemory = processorResult.images;
          }
        }

           
        if (key === 'MICROPHONE'){
          result.microphone = processorResult.replacementText;
        } else if (key === 'SCREEN_AUDIO') {
          result.screenAudio = processorResult.replacementText;
        } else if (key === 'ALL_AUDIO'){
          result.allAudio = processorResult.replacementText;
        }

        // Safety break for empty placeholder matches to prevent infinite loops
        // if somehow regex.lastIndex isn't advanced by the above.
        if (matchIndex === processor.regex.lastIndex && placeholder.length === 0) {
            processor.regex.lastIndex++;
        }
        if (processor.regex.lastIndex > modifiedPrompt.length) { // Boundary check
            break;
        }

      }
      // Append any remaining part of the string
      tempPrompt += modifiedPrompt.substring(currentSearchIndex);
      modifiedPrompt = tempPrompt;
    }
    result.modifiedPrompt = modifiedPrompt;
    Logger.debug(agentId, 'Completed prompt pre-processing. Final prompt:', result.modifiedPrompt);
    return result;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    Logger.error(agentId, `Error in pre-processing: ${errorMessage}`);
    return { modifiedPrompt: systemPrompt, images: [] };
  }
}
