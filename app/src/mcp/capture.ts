// src/mcp/capture.ts
//
// One preview frame for the MCP, leaving a live stream that start_agent reuses. Shared by the
// capture_screen / capture_camera tools and the ask_user_info 'screen' / 'camera' modal: the
// modal calls the same functions from its button, and that click is the user gesture Safari
// requires for getDisplayMedia (an MCP tool call never is one).
//
// The platform split mirrors StreamManager (streamManager.ts): isWeb() → browser capture;
// Tauri → the native plugin, which handles iOS/Android behind one acquireMasterStream.

import type { ToolResult } from './types';
import { browserStreamCapture, mockCameraStreams } from '@utils/browserStreamCapture';
import { tauriStreamCapture } from '@utils/tauriStreamCapture';
import { isWeb } from '@utils/platform';
import { StreamManager, type PseudoStreamType } from '@utils/streamManager';

/** Thrown by getDisplayMedia outside a click; the cue to fall back to ask_user_info kind='screen'. */
const NEEDS_GESTURE = /user gesture/i;

/** Holds the screen stream during a turn, so stopping and restarting an agent there reuses it. */
export const TURN_HOLDER = '__mcp_turn__';

// The MCP subscribes to a stream the way an agent does, so the agent card can show it and it
// survives between turns ("How should I tell you?" sits between capture and start_agent).
// One subscriber per stream, so letting go of one leaves the other open.
const HELD = ['screenVideo', 'camera'] as const;
type HeldStream = typeof HELD[number];
const holderFor = (type: HeldStream) => `__mcp_capture__:${type}`;
const HOLD_TTL_MS = 10 * 60 * 1000;
const expiry = new Map<HeldStream, ReturnType<typeof setTimeout>>();

/** Unsubscribe the MCP once an agent owns these streams or the build is abandoned. */
export function releaseCaptureHolds(types: readonly PseudoStreamType[] = HELD): void {
  for (const type of HELD) {
    if (!types.includes(type)) continue;
    clearTimeout(expiry.get(type));
    StreamManager.releaseStreamsForAgent(holderFor(type));
  }
}

/**
 * Subscribe the MCP to a stream, opening it if needed. Each call restarts the expiry, so a
 * walked-away build can't leave a stream open. If it fires mid-turn nothing breaks: the turn's
 * own hold keeps the screen alive, and the camera reopens without a gesture.
 */
async function holdCapture(type: HeldStream): Promise<void> {
  await StreamManager.requestStreamsForAgent(holderFor(type), [type]);
  if (type === 'screenVideo') StreamManager.holdScreen(holderFor(type));
  clearTimeout(expiry.get(type));
  expiry.set(type, setTimeout(() => releaseCaptureHolds([type]), HOLD_TTL_MS));
}

/**
 * Drop the MCP's subscriptions to the screen, so the next capture can open on a new target.
 * Returns false if the stream stayed open, which means someone else is still using it.
 */
export function letGoOfScreen(): boolean {
  releaseCaptureHolds(['screenVideo']);
  StreamManager.releaseStreamsForAgent(TURN_HOLDER);
  const closed = !tauriStreamCapture.isStreamAvailable('screenVideo');
  StreamManager.holdScreen(TURN_HOLDER);
  return closed;
}

/** Draw the current frame of a live video stream to a JPEG data URL via a hidden <video>. */
async function grabVideoFrame(stream: MediaStream): Promise<{ dataUrl: string; width: number; height: number }> {
  const video = document.createElement('video');
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error('Video metadata load failed'));
    video.play().catch(reject);
  });
  // Give the first frame a moment to render.
  await new Promise(r => setTimeout(r, 150));

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  canvas.getContext('2d')!.drawImage(video, 0, 0);
  video.pause();
  video.srcObject = null;
  return { dataUrl: canvas.toDataURL('image/jpeg', 0.85), width: canvas.width, height: canvas.height };
}

/** Web / mobile web: getDisplayMedia picker, then one frame. */
async function captureScreenWeb(): Promise<ToolResult> {
  if (!navigator?.mediaDevices?.getDisplayMedia) {
    return { error: "Screen capture is not supported in this browser. Call ask_user_info kind='screen' once; if that fails too, tell the user to open Observer in Chrome, Edge or Firefox." };
  }
  try {
    // getDisplayMedia is the first await inside, so a click handler calling this keeps its gesture.
    await holdCapture('screenVideo');
    const videoTrack = browserStreamCapture.getStreams().screenVideoStream?.getVideoTracks()[0];
    if (!videoTrack) return { error: 'No video track acquired from screen share.' };

    const frame = await grabVideoFrame(new MediaStream([videoTrack]));
    const settings = videoTrack.getSettings();
    return {
      data: {
        captured: true,
        width: settings.width ?? frame.width,
        height: settings.height ?? frame.height,
        note: 'Stream is live and will be reused by start_agent — no second picker prompt.',
      },
      images: [frame.dataUrl],
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (NEEDS_GESTURE.test(msg)) {
      return { error: "This browser only opens the screen picker from a click. Call ask_user_info kind='screen': its button opens the picker and returns the frame." };
    }
    if (/Permission denied|NotAllowed/i.test(msg)) {
      return { error: "Screen share was cancelled or denied. Call ask_user_info kind='screen' so the user can share it with one click." };
    }
    return { error: msg };
  }
}

/**
 * Mobile app (Tauri iOS/Android): trigger the native picker via tauriStreamCapture — the
 * same singleton the agent loop acquires, so the live stream is reused by start_agent.
 */
function captureScreenTauri(): Promise<ToolResult> {
  return grabTauriFrame(
    { captured: true, note: 'Native screen capture is live (whole screen) and will be reused by start_agent — no second prompt.' },
    'Screen capture started but no frame has arrived yet. On iOS, make sure you tapped "Start Broadcast" in the system sheet, then call capture_screen again.',
  );
}

/** Web and mobile app; the desktop app captures a target_id instead (see registry.ts). */
export const captureScreen = (): Promise<ToolResult> => (isWeb() ? captureScreenWeb() : captureScreenTauri());

/**
 * Acquire tauriStreamCapture's display stream and return its newest frame. Frames arrive
 * asynchronously (on iOS the user taps "Start Broadcast" + a short countdown), so poll the
 * latest raw frame until one lands or we time out, rather than grabbing synchronously like
 * the browser path.
 */
export async function grabTauriFrame(data: Record<string, unknown>, noFrameError: string): Promise<ToolResult> {
  try {
    await holdCapture('screenVideo');
  } catch (e) {
    // Android surfaces a real rejection on denial; iOS dismissal just yields no frames.
    const msg = e instanceof Error ? e.message : String(e);
    if (/denied|cancel|NotAllowed/i.test(msg)) {
      return { error: "Screen capture was cancelled or denied. Call ask_user_info kind='screen' so the user can allow it." };
    }
    return { error: msg };
  }
  return waitForTauriFrame(data, noFrameError);
}

/** The newest frame of the live Tauri display stream, without taking a hold on it. */
export async function waitForTauriFrame(data: Record<string, unknown>, noFrameError: string): Promise<ToolResult> {
  const FRAME_TIMEOUT_MS = 15000;
  const POLL_MS = 200;
  const deadline = Date.now() + FRAME_TIMEOUT_MS;
  let raw: string | null = null;
  while (Date.now() < deadline) {
    raw = tauriStreamCapture.getLatestBase64Frame();
    if (raw) break;
    await new Promise(r => setTimeout(r, POLL_MS));
  }

  if (!raw) {
    return { error: noFrameError };
  }

  return {
    data,
    images: [`data:image/jpeg;base64,${raw}`],
  };
}

/**
 * Open the camera and return one frame; the stream stays live for start_agent. Both capture
 * layers swap in a placeholder stream when access is denied (so a running agent never
 * crashes); here that placeholder is torn down and reported, so the MCP asks for permission
 * instead of building an agent on top of it.
 */
export async function captureCamera(cameraId?: string): Promise<ToolResult> {
  if (!navigator?.mediaDevices?.getUserMedia) {
    return { error: 'Camera capture is not supported here. Tell the user to open Observer in Chrome, Edge or Firefox.' };
  }
  const impl = isWeb() ? browserStreamCapture : tauriStreamCapture;
  try {
    if (cameraId) {
      const cameras = await StreamManager.getAvailableCameraDevices();
      if (!cameras.some(c => c.deviceId === cameraId)) {
        return { error: `Camera '${cameraId}' is not available (it may have been unplugged). Re-run list_cameras.` };
      }
      // The live stream is reused as-is by acquireMasterStream, so switching means re-seating
      // it; switchCameraDevice also moves any running $CAMERA agents onto the new camera.
      const live = impl.getStreams().cameraStream?.getVideoTracks()[0]?.getSettings().deviceId;
      if (live !== cameraId) await StreamManager.switchCameraDevice(cameraId);
    }
    await holdCapture('camera');
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }

  const stream = impl.getStreams().cameraStream;
  if (!stream || mockCameraStreams.has(stream)) {
    releaseCaptureHolds(['camera']);
    return { error: "Camera access was denied or no camera was found. Call ask_user_info kind='camera' so the user can allow it with one click." };
  }

  try {
    const frame = await grabVideoFrame(stream);
    return {
      data: {
        captured: true,
        camera_id: stream.getVideoTracks()[0]?.getSettings().deviceId,
        width: frame.width,
        height: frame.height,
        note: 'Camera is live and will be reused by start_agent — no second prompt.',
      },
      images: [frame.dataUrl],
    };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
}

/**
 * The cameras capture_camera can open, by deviceId. Browsers blank the labels until camera
 * permission is granted, so the model is told to capture once first when they're all empty.
 */
export async function listCameras(): Promise<ToolResult> {
  if (!navigator?.mediaDevices?.enumerateDevices) {
    return { error: 'Camera listing is not supported here. Tell the user to open Observer in Chrome, Edge or Firefox.' };
  }
  const devices = await StreamManager.getAvailableCameraDevices();
  const preferred = StreamManager.getPreferredCameraDevice();
  const live = (isWeb() ? browserStreamCapture : tauriStreamCapture)
    .getStreams().cameraStream?.getVideoTracks()[0]?.getSettings().deviceId;
  const cameras = devices.map((d, i) => ({
    id: d.deviceId,
    label: d.label || `Camera ${i + 1}`,
    preferred: d.deviceId === preferred,
    live: d.deviceId === live,
  }));
  const unlabeled = devices.length > 0 && devices.every(d => !d.label);
  return {
    data: {
      cameras,
      ...(cameras.length === 0 && { note: "No camera found. Call ask_user_info kind='camera'." }),
      ...(unlabeled && { note: 'Camera names are hidden until permission is granted: call capture_camera once (no camera_id), then list_cameras again.' }),
    },
  };
}
