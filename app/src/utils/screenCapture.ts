import { StreamManager } from './streamManager';

export interface CropConfig {
  x: number;
  y: number;
  width: number;
  height: number;
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
