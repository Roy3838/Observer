import { StreamManager } from './streamManager';
import { Logger } from './logging';

/**
 * Captures a frame from the camera stream.
 * Uses StreamManager's persistent video element for instant capture.
 *
 * @param stream The active camera MediaStream (kept for API compatibility, not used internally)
 * @param agentId Optional agent ID for crop configuration
 * @returns The raw Base64 string (without the data URI prefix) or null on failure.
 */
export async function captureCameraImage(_stream: MediaStream, agentId?: string): Promise<string | null> {
  const base64 = await StreamManager.captureFrame('camera', agentId);
  if (base64) {
    Logger.debug(agentId || 'camera', `Camera frame captured (${base64.length} bytes)`);
  }
  return base64;
}
