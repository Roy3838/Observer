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

/** Thrown by getDisplayMedia outside a click; the cue to fall back to ask_user_info kind='screen'. */
const NEEDS_GESTURE = /user gesture/i;

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
    await browserStreamCapture.acquireMasterStream('display');
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
    await tauriStreamCapture.acquireMasterStream('display');
  } catch (e) {
    // Android surfaces a real rejection on denial; iOS dismissal just yields no frames.
    const msg = e instanceof Error ? e.message : String(e);
    if (/denied|cancel|NotAllowed/i.test(msg)) {
      return { error: "Screen capture was cancelled or denied. Call ask_user_info kind='screen' so the user can allow it." };
    }
    return { error: msg };
  }

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
export async function captureCamera(): Promise<ToolResult> {
  if (!navigator?.mediaDevices?.getUserMedia) {
    return { error: 'Camera capture is not supported here. Tell the user to open Observer in Chrome, Edge or Firefox.' };
  }
  const impl = isWeb() ? browserStreamCapture : tauriStreamCapture;
  try {
    await impl.acquireMasterStream('camera');
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }

  const stream = impl.getStreams().cameraStream;
  if (!stream || mockCameraStreams.has(stream)) {
    impl.teardownMasterStream('camera');
    return { error: "Camera access was denied or no camera was found. Call ask_user_info kind='camera' so the user can allow it with one click." };
  }

  try {
    const frame = await grabVideoFrame(stream);
    return {
      data: {
        captured: true,
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
