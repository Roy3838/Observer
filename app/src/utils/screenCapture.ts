import { StreamManager, resolveCrop } from './streamManager';

/**
 * A crop region stored as fractions of the captured frame (0–1), NOT pixels.
 * Keeping it resolution-independent is what lets the same crop apply correctly
 * regardless of the live capture's actual pixel size — which differs from the
 * "target" dimensions reported by the OS (especially on the macOS/Tauri path,
 * where ScreenCaptureKit's frame size is unrelated to xcap's logical dims).
 * It is resolved to pixels against the real frame at capture time.
 */
export interface CropConfig {
  x: number;      // left edge, 0–1 of frame width
  y: number;      // top edge, 0–1 of frame height
  width: number;  // 0–1 of frame width
  height: number; // 0–1 of frame height
}

export interface AgentCropConfig {
  camera?: CropConfig;
  screen?: CropConfig;
}

// Per-agent crop storage
const agentCropConfigs = new Map<string, AgentCropConfig>();

// --- Crop Management API ---
export function setAgentCrop(agentId: string, streamType: 'camera' | 'screen', cropConfig: CropConfig | null): void {
  if (!agentCropConfigs.has(agentId)) {
    agentCropConfigs.set(agentId, {});
  }

  const agentConfig = agentCropConfigs.get(agentId)!;
  if (streamType === 'camera') {
    agentConfig.camera = cropConfig || undefined;
  } else {
    agentConfig.screen = cropConfig || undefined;
  }

  console.log(`Set ${streamType} crop for agent '${agentId}':`, cropConfig);
}

export function getAgentCrop(agentId: string, streamType: 'camera' | 'screen'): CropConfig | null {
  const agentConfig = agentCropConfigs.get(agentId);
  if (!agentConfig) return null;
  return streamType === 'camera' ? agentConfig.camera || null : agentConfig.screen || null;
}

/**
 * Apply a crop to a full-frame image (data URL), exactly as the agent loop will on each
 * iteration (same resolveCrop). Used to preview what a crop will send to the model.
 */
export async function cropDataUrl(
  dataUrl: string,
  crop: CropConfig,
): Promise<{ dataUrl: string; width: number; height: number }> {
  const bitmap = await createImageBitmap(await (await fetch(dataUrl)).blob());
  const px = resolveCrop(crop, bitmap.width, bitmap.height);
  const canvas = document.createElement('canvas');
  canvas.width = px.width;
  canvas.height = px.height;
  canvas.getContext('2d')!.drawImage(bitmap, px.x, px.y, px.width, px.height, 0, 0, px.width, px.height);
  bitmap.close();
  return { dataUrl: canvas.toDataURL('image/png'), width: px.width, height: px.height };
}

export function removeAgentCrops(agentId: string): void {
  agentCropConfigs.delete(agentId);
  console.log(`Removed all crop configs for agent '${agentId}'`);
}

/**
 * Captures a frame from the screen stream.
 * Uses raw frame bytes from Tauri channel (reliable even when backgrounded on iOS)
 * or persistent video element for browser/camera.
 *
 * @param stream The active screen MediaStream (kept for API compatibility, not used internally)
 * @param agentId Optional agent ID for crop configuration
 * @param streamType The type of stream ('camera' or 'screen'), defaults to 'screen'
 * @returns The raw Base64 string (without the data URI prefix) or null on failure.
 */
export async function captureScreenImage(_stream: MediaStream, agentId?: string, streamType?: 'camera' | 'screen'): Promise<string | null> {
  const type = streamType || 'screen';
  return StreamManager.captureFrame(type, agentId);
}
