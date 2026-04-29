import { SensorSettings } from './settings';
import { StreamManager } from './streamManager';

export interface OCRResult {
  success?: boolean;
  text?: string;
  confidence?: number;
  error?: string;
}

/**
 * Captures a frame from the stream and performs OCR on it.
 * Uses StreamManager's persistent video element for instant capture.
 * This eliminates the race condition where onloadedmetadata fires before the first frame is rendered.
 *
 * @param stream The active MediaStream (kept for API compatibility, not used internally)
 * @param agentId Optional agent ID for crop configuration
 * @param streamType The type of stream ('camera' or 'screen'), defaults to 'screen'
 * @returns OCR result with text, confidence, and success status
 */
export async function captureFrameAndOCR(_stream: MediaStream, agentId?: string, streamType?: 'camera' | 'screen'): Promise<OCRResult> {
  const type = streamType || 'screen';
  const imageData = await StreamManager.captureFrame(type, agentId);

  if (!imageData) {
    return { error: 'Failed to capture frame: video not ready' };
  }

  return performOCR(imageData);
}

export async function performOCROnBase64(imageData: string): Promise<OCRResult> {
  return performOCR(imageData);
}

// Function to perform OCR on image data
async function performOCR(imageData: string): Promise<OCRResult> {
  console.log('Starting OCR processing...');

  try {
    // Lazy load Tesseract.js - only loads when OCR is actually used!
    const { createWorker } = await import('tesseract.js');

    // Initialize worker
    const worker = await createWorker(SensorSettings.getOcrLanguage(), 1, {
      workerPath: SensorSettings.getOcrWorkerPath(),
      langPath: SensorSettings.getOcrLangPath(),
      corePath: SensorSettings.getOcrCorePath(),
      logger: m => console.log('[Tesseract]', m)
    });

    // Recognize text
    const result = await worker.recognize(`data:image/png;base64,${imageData}`);

    // Terminate worker
    await worker.terminate();

    console.log(`OCR processing complete. Confidence: ${result.data.confidence}`);
    return {
      success: true,
      text: result.data.text,
      confidence: result.data.confidence
    };


  } catch (error) {
    console.error('OCR processing error:', error);
    return {
      error: `OCR processing failed: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}
