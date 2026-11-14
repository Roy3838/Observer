// src/utils/statusPip.ts
//
// The mobile "Run in Background" window: Picture-in-Picture of a status canvas, not of the
// screen. Keeping a PiP video playing is what keeps the app alive while the user is in another
// app, so the window should earn its spot: it shows what the MCP is doing and every running
// agent (what it sees, its last word, its countdown).
//
// One app-level <video> fed by canvas.captureStream(), so it belongs to no agent card: agents
// can start, stop and be deleted under it. requestPictureInPicture needs a user gesture, so
// prepareStatusPip() sets the video up ahead of time and startStatusPip() is called straight from
// a tap (the agent card's button, or ask_user_info kind='background').
//
// The canvas is redrawn slowly (1 fps) from setup on, and at FPS while the window is open:
// WebKit's captureStream only emits a frame when the canvas is painted after the stream exists,
// so a canvas drawn once has no frames, the video never loads, and PiP can't open.

import { IterationStore } from './IterationStore';
import { imageDataUrl } from './imageDataUrl';
import { StreamManager } from './streamManager';
import { isPipSupported } from './pictureInPicture';
import { Logger } from './logging';
import { isTauri } from './platform';

// Portrait 3:4, drawn at 2x so text stays sharp in a small window. Change W/H to try other shapes.
const W = 540;
const H = 720;
const SCALE = 2;
const FPS = 4;
const IDLE_FPS = 1;
/** While the MCP works the logo animates, and its blink is ~190ms: 4 fps would skip it. */
const ANIM_FPS = 24;

const PAD = 24;
const FONT = '-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif';
const MONO = 'ui-monospace, "SF Mono", Menlo, monospace';

// The app's palette (Tailwind), so the window reads like the agent cards it mirrors.
const C = {
  page: '#f9fafb',      // gray-50
  card: '#ffffff',
  border: '#e5e7eb',    // gray-200
  track: '#e5e7eb',     // gray-200
  text: '#1f2937',      // gray-800
  strong: '#111827',    // gray-900
  body: '#4b5563',      // gray-600
  muted: '#9ca3af',     // gray-400
  green: '#22c55e',     // green-500
  greenText: '#16a34a', // green-600
  greenTint: '#f0fdf4', // green-50
  blue: '#3b82f6',      // blue-500
  blueSoft: '#60a5fa',  // blue-400
  orange: '#fb923c',    // orange-400
  thumbBg: '#f3f4f6',   // gray-100
};

// Same words as AgentLiveCard's status line.
const STATUS_LABEL: Record<AgentStatus, string> = {
  STARTING: 'Starting…',
  CAPTURING: 'Capturing…',
  THINKING: 'Thinking…',
  RESPONDING: 'Responding…',
  WAITING: 'Waiting…',
  SKIPPED: 'Skipped (no change)',
  SLEEPING: 'Sleeping',
  IDLE: 'Idle',
};
const OVERRUN_LABEL: Partial<Record<AgentStatus, string>> = {
  CAPTURING: 'Still capturing…',
  THINKING: 'Still thinking…',
  RESPONDING: 'Still responding…',
};

// ---- What the window shows ------------------------------------------------------------------
//
// Not tracked here: StatusPipBridge (components/Observer) hands over the same state the agent
// cards render (App's agents and running set, AgentLiveStateProvider's live state, the MCP
// context) with setStatusSnapshot() on every change. The window only draws the latest one.

/** AgentLiveStateContext's AgentLiveStatus. */
export type AgentStatus = 'STARTING' | 'CAPTURING' | 'THINKING' | 'RESPONDING' | 'WAITING' | 'SKIPPED' | 'SLEEPING' | 'IDLE';

export interface StatusAgent {
  id: string;
  name: string;
  status: AgentStatus;
  lastWord: string;
  /** 0–100 through the current loop interval. */
  progress: number;
  /** Until the next look (or the end of a sleep); null before the first iteration. */
  remainingMs: number | null;
  isSleeping: boolean;
  isOverrun: boolean;
}

export interface StatusMcp {
  running: boolean;
  /** The step in progress: "Thinking…", "Cropping Finish Line"… */
  activity: string;
  /** Set when an agent's mcp() started the turn. */
  wokenBy: string | null;
  /** The last answer and when its turn ended, shown for a little while after. */
  lastReply: string;
  endedAt: number;
}

let snapshot: { mcp: StatusMcp; agents: StatusAgent[] } = {
  mcp: { running: false, activity: '', wokenBy: null, lastReply: '', endedAt: 0 },
  agents: [],
};

export function setStatusSnapshot(next: typeof snapshot): void {
  snapshot = next;
}

/** What the agent saw on its last iteration, as the model got it: the iteration's first image,
 *  decoded once per new image so the old one stays up until the new one is ready. */
const thumbs = new Map<string, { src: string; img: HTMLImageElement | null }>();
function thumbFor(agentId: string): HTMLImageElement | null {
  const iterations = IterationStore.getIterationsForAgent(agentId);
  let src: string | undefined;
  for (let i = iterations.length - 1; i >= 0 && !src; i--) src = iterations[i].modelImages?.[0];
  const cached = thumbs.get(agentId);
  if (src && cached?.src !== src) {
    const entry = { src, img: cached?.img ?? null };
    thumbs.set(agentId, entry);
    const img = new Image();
    img.src = imageDataUrl(src);
    img.decode().then(() => { if (thumbs.get(agentId) === entry) entry.img = img; }).catch(() => {});
  }
  return thumbs.get(agentId)?.img ?? null;
}

/** How long the MCP's last answer stays up after a turn. */
const REPLY_MS = 20_000;

// Lucide icons (24×24, stroked), as SVG path data for Path2D.
const ICON: Record<string, string[]> = {
  check: ['M20 6 9 17l-5-5'],
};

/**
 * Two ways to feed the window, side by side while we debug PiP on iOS (PipDebugPanel):
 *   canvas  the status alone, painted at FPS (IDLE_FPS while closed)
 *   video   the same status over a live video (the screen if it's shared, else the camera),
 *           painted at VIDEO_FPS like a real video
 * Same drawing, same <video> + captureStream setup; only what's under the status differs.
 */
export type PipMode = 'canvas' | 'video';
const VIDEO_FPS = 15;

interface Pip {
  mode: PipMode;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  video: HTMLVideoElement;
  timer: ReturnType<typeof setInterval> | null;
  active: boolean;
  /** video mode: the camera, opened when no screen is shared. */
  camera: HTMLVideoElement | null;
  lastError: string;
  lastDraw: number;
}

const pips: Partial<Record<PipMode, Pip>> = {};
const listeners = new Set<(active: boolean) => void>();

export const isStatusPipSupported = (): boolean => isPipSupported();
export const isStatusPipActive = (): boolean => Object.values(pips).some(p => p?.active);

/** Follow the window opening and closing (for the "Running in background" button state). */
export function onStatusPipChange(listener: (active: boolean) => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** Create the canvas and its playing <video> ahead of the tap that opens the window. */
export function prepareStatusPip(mode: PipMode = 'canvas'): HTMLVideoElement {
  const existing = pips[mode];
  if (existing) return existing.video;

  const canvas = document.createElement('canvas');
  canvas.width = W * SCALE;
  canvas.height = H * SCALE;
  const ctx = canvas.getContext('2d')!;

  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.autoplay = true;
  video.setAttribute('playsinline', '');
  // In the document (PiP needs that) but out of sight.
  Object.assign(video.style, {
    position: 'fixed', left: '0', bottom: '0', width: '2px', height: '2px',
    opacity: '0.01', pointerEvents: 'none', zIndex: '-1',
  });
  video.srcObject = canvas.captureStream(mode === 'video' ? VIDEO_FPS : ANIM_FPS);
  document.body.appendChild(video);

  const pip: Pip = { mode, canvas, ctx, video, timer: null, active: false, camera: null, lastError: '', lastDraw: 0 };
  pips[mode] = pip;
  video.play().catch(e => report(pip, `play() failed: ${errorText(e)}`));
  video.addEventListener('enterpictureinpicture', () => setActive(pip, true));
  video.addEventListener('leavepictureinpicture', () => setActive(pip, false));
  if (mode === 'video') openCameraIfNoScreen(pip);
  setActive(pip, false);
  return video;
}

/** Open the window. Call it straight from a tap: browsers only allow PiP from a user gesture,
 *  so nothing is awaited before requestPictureInPicture. */
export async function startStatusPip(mode: PipMode = 'canvas'): Promise<void> {
  const v = prepareStatusPip(mode);
  const pip = pips[mode]!;
  if (document.pictureInPictureElement === v) return;
  if (v.readyState < HTMLMediaElement.HAVE_METADATA) {
    report(pip, `not ready (readyState ${v.readyState}, paused ${v.paused})`);
    throw new Error('The status window is still loading. Try again in a second.');
  }
  try {
    await v.requestPictureInPicture();
    pip.lastError = '';
  } catch (e) {
    report(pip, `requestPictureInPicture failed: ${errorText(e)}`);
    throw e;
  }
}

export async function stopStatusPip(): Promise<void> {
  const open = Object.values(pips).find(p => p && document.pictureInPictureElement === p.video);
  if (open) await document.exitPictureInPicture();
}

/** For PipDebugPanel: the mode's <video> and what state it's in. */
export function statusPipDiagnostics(mode: PipMode) {
  const pip = pips[mode];
  if (!pip) return null;
  const v = pip.video;
  const bg = mode === 'video' ? backgroundFor(pip) : null;
  return {
    video: v,
    readyState: v.readyState,
    paused: v.paused,
    size: `${v.videoWidth}x${v.videoHeight}`,
    active: pip.active,
    background: mode === 'video' ? (bg ? (bg === pip.camera ? 'camera' : 'screen') : 'none yet') : 'none',
    lastError: pip.lastError,
  };
}

function setActive(pip: Pip, next: boolean): void {
  const changed = pip.active !== next;
  pip.active = next;
  if (pip.timer) clearInterval(pip.timer);
  const fps = pip.mode === 'video' ? VIDEO_FPS : next ? ANIM_FPS : IDLE_FPS;
  pip.timer = setInterval(() => {
    // Full speed only while the logo animates (the MCP is working); FPS is plenty otherwise.
    if (pip.mode === 'canvas' && next && !snapshot.mcp.running && Date.now() - pip.lastDraw < 1000 / FPS) return;
    draw(pip);
  }, 1000 / fps);
  draw(pip);
  if (changed) listeners.forEach(l => l(isStatusPipActive()));
}

function openCameraIfNoScreen(pip: Pip): void {
  if (StreamManager.getVideoElement('screen')?.videoWidth) return;
  navigator.mediaDevices?.getUserMedia({ video: { facingMode: 'user' }, audio: false })
    .then(stream => {
      const cam = document.createElement('video');
      cam.muted = true;
      cam.playsInline = true;
      cam.setAttribute('playsinline', '');
      cam.srcObject = stream;
      cam.play().catch(e => report(pip, `camera play() failed: ${errorText(e)}`));
      pip.camera = cam;
    })
    .catch(e => report(pip, `camera failed: ${errorText(e)}`));
}

function backgroundFor(pip: Pip): HTMLVideoElement | null {
  const screen = StreamManager.getVideoElement('screen');
  if (screen?.videoWidth) return screen;
  return pip.camera?.videoWidth ? pip.camera : null;
}

const errorText = (e: unknown): string => (e instanceof Error ? `${e.name}: ${e.message}` : String(e));

/** To the app's Logger, and to the native log so it shows up in the device's syslog. */
function report(pip: Pip, message: string): void {
  pip.lastError = message;
  Logger.error('PiP', `[${pip.mode}] ${message}`);
  if (isTauri()) import('@tauri-apps/plugin-log').then(log => log.error(`[PiP ${pip.mode}] ${message}`)).catch(() => {});
}

// ---- Drawing --------------------------------------------------------------------------------

function draw(pip: Pip): void {
  const c = pip.ctx;
  const now = Date.now();
  pip.lastDraw = now;
  c.setTransform(SCALE, 0, 0, SCALE, 0, 0);
  c.textBaseline = 'middle';
  c.textAlign = 'left';
  const bg = pip.mode === 'video' ? backgroundFor(pip) : null;
  if (bg) {
    // Cover the window with the video, then wash it out so the cards stay readable.
    const k = Math.max(W / bg.videoWidth, H / bg.videoHeight);
    const dw = bg.videoWidth * k;
    const dh = bg.videoHeight * k;
    try { c.drawImage(bg, (W - dw) / 2, (H - dh) / 2, dw, dh); } catch { /* frame not ready */ }
    c.fillStyle = 'rgba(249, 250, 251, 0.72)';
  } else {
    c.fillStyle = C.page;
  }
  c.fillRect(0, 0, W, H);

  let y = PAD;
  const agents = snapshot.agents;
  y = drawHeader(c, y, agents.length, now);
  y = drawMcp(c, y, now);
  drawAgents(c, agents, y, H - PAD);
}

/** A white card with the app's border and a soft shadow. */
function card(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, fill = C.card, stroke = C.border): void {
  c.save();
  c.shadowColor = 'rgba(17, 24, 39, 0.06)';
  c.shadowBlur = 12;
  c.shadowOffsetY = 3;
  roundRect(c, x, y, w, h, 18);
  c.fillStyle = fill;
  c.fill();
  c.restore();
  roundRect(c, x, y, w, h, 18);
  c.strokeStyle = stroke;
  c.lineWidth = 1.5;
  c.stroke();
}

function icon(c: CanvasRenderingContext2D, name: string, cx: number, cy: number, size: number, color: string): void {
  const paths = ICON[name];
  if (!paths) return;
  c.save();
  c.translate(cx - size / 2, cy - size / 2);
  c.scale(size / 24, size / 24);
  c.strokeStyle = color;
  c.lineWidth = 2;
  c.lineCap = 'round';
  c.lineJoin = 'round';
  for (const d of paths) c.stroke(new Path2D(d));
  c.restore();
}

// ---- Observer's logo, animated like ObserverEye ---------------------------------------------
//
// The same moves as ObserverEye.tsx / index.css (observer-eye-*), ported keyframe for keyframe:
// one 2.4s move at a time, a different one each cycle, over a 1.2s opacity breath. Units are the
// logo's own (viewBox -50..50), like the CSS px on the SVG. Keep in sync with index.css.

const MOVE_MS = 2400;
const BREATH_MS = 1200;

/** [percent, values]: a CSS keyframe. "0%, 8%" is two keyframes with the same values. */
type Keyframes = [number, number[]][];

const EYE_MOVES: Record<string, { lid?: Keyframes; pupil?: Keyframes }> = {
  // peek: pupil glances up-left, then up-right. pupil = [x, y]
  peek: {
    pupil: [[0, [0, 0]], [8, [0, 0]], [18, [-14, -5]], [38, [-14, -5]], [50, [14, -5]], [70, [14, -5]], [84, [0, 0]], [100, [0, 0]]],
  },
  // blink: squishy double blink. lid = [translateY, scaleX, scaleY]
  blink: {
    lid: [[0, [0, 1, 1]], [20, [0, 1, 1]], [28, [0, 1.1, 0.06]], [36, [0, 0.95, 1.07]], [43, [0, 1, 1]], [48, [0, 1, 1]],
      [56, [0, 1.1, 0.06]], [64, [0, 0.95, 1.07]], [72, [0, 1, 1]], [100, [0, 1, 1]]],
  },
  // hop: crouch, jump looking up, squash on landing
  hop: {
    lid: [[0, [0, 1, 1]], [10, [0, 1, 1]], [20, [0, 1.12, 0.84]], [38, [-16, 0.92, 1.1]], [54, [0, 1.14, 0.84]],
      [66, [0, 0.97, 1.04]], [76, [0, 1, 1]], [100, [0, 1, 1]]],
    pupil: [[0, [0, 0]], [18, [0, 0]], [30, [0, -15]], [46, [0, -15]], [62, [0, 0]], [100, [0, 0]]],
  },
};
const EYE_MOVE_NAMES = Object.keys(EYE_MOVES);

/** CSS cubic-bezier(x1, y1, x2, y2) as a function of progress 0..1. */
function cubicBezier(x1: number, y1: number, x2: number, y2: number): (t: number) => number {
  const bx = (t: number) => 3 * x1 * t * (1 - t) ** 2 + 3 * x2 * t * t * (1 - t) + t ** 3;
  const by = (t: number) => 3 * y1 * t * (1 - t) ** 2 + 3 * y2 * t * t * (1 - t) + t ** 3;
  return x => {
    if (x <= 0 || x >= 1) return x <= 0 ? 0 : 1;
    let lo = 0, hi = 1, t = x;
    for (let i = 0; i < 24; i++) { // bisection: x(t) is monotonic
      if (bx(t) < x) lo = t; else hi = t;
      t = (lo + hi) / 2;
    }
    return by(t);
  };
}
const MOVE_EASE = cubicBezier(0.45, 0, 0.55, 1);
const BREATH_EASE = cubicBezier(0.42, 0, 0.58, 1); // CSS ease-in-out

/** The values at `percent`, eased per segment the way CSS applies a timing function. */
function sample(frames: Keyframes, percent: number): number[] {
  for (let i = 0; i < frames.length - 1; i++) {
    const [a, from] = frames[i];
    const [b, to] = frames[i + 1];
    if (percent > b) continue;
    const t = b === a ? 1 : MOVE_EASE((percent - a) / (b - a));
    return from.map((v, j) => v + (to[j] - v) * t);
  }
  return frames[frames.length - 1][1];
}

const eye = { move: '', start: 0 };

function drawLogo(c: CanvasRenderingContext2D, cx: number, cy: number, size: number, working: boolean, now: number): void {
  let lid = [0, 1, 1];
  let pupil = [0, 0];
  let alpha = 1;
  if (working) {
    // Next move on the 2.4s grid, never the same one twice in a row.
    if (!eye.move || now - eye.start >= MOVE_MS) {
      const options = EYE_MOVE_NAMES.filter(m => m !== eye.move);
      eye.move = options[Math.floor(Math.random() * options.length)];
      eye.start = eye.start && now - eye.start < MOVE_MS * 2 ? eye.start + MOVE_MS : now;
    }
    const percent = ((now - eye.start) / MOVE_MS) * 100;
    const move = EYE_MOVES[eye.move];
    if (move.lid) lid = sample(move.lid, percent);
    if (move.pupil) pupil = sample(move.pupil, percent);
    // Breath: 1 → 0.7 → 1, alternating every 1.2s.
    let phase = (now % (BREATH_MS * 2)) / BREATH_MS;
    if (phase > 1) phase = 2 - phase;
    alpha = 1 - 0.3 * BREATH_EASE(phase);
  } else {
    eye.move = '';
    eye.start = 0;
  }

  c.save();
  c.globalAlpha = alpha;
  c.translate(cx, cy);
  c.scale(size / 100, size / 100);
  // The lid group: its transform scales the ring and carries the pupil, as in the SVG.
  c.translate(0, lid[0]);
  c.scale(lid[1], lid[2]);
  c.strokeStyle = C.strong;
  c.fillStyle = C.strong;
  c.lineWidth = 10;
  c.beginPath();
  c.arc(0, 0, 45, 0, Math.PI * 2);
  c.stroke();
  c.beginPath();
  c.arc(pupil[0], pupil[1], 12, 0, Math.PI * 2);
  c.fill();
  c.restore();
}

function drawHeader(c: CanvasRenderingContext2D, y: number, count: number, now: number): number {
  const cy = y + 20;
  drawLogo(c, PAD + 18, cy, 36, snapshot.mcp.running, now);

  c.fillStyle = C.strong;
  c.font = `700 28px ${FONT}`;
  c.fillText('Observer', PAD + 50, cy);

  c.font = `500 20px ${FONT}`;
  c.textAlign = 'right';
  c.fillStyle = C.muted;
  c.fillText(count ? `${count} running` : 'Idle', W - PAD, cy);
  c.textAlign = 'left';
  return y + 60;
}

/** The MCP's step while it works, then its answer for a little while. Nothing when idle. */
function drawMcp(c: CanvasRenderingContext2D, y: number, now: number): number {
  const mcp = snapshot.mcp;
  const showReply = !mcp.running && mcp.lastReply && now - mcp.endedAt < REPLY_MS;
  if (!mcp.running && !showReply) return y;

  const textX = PAD + 64;
  const textW = W - PAD - textX - 20;
  const lines = wrap(c, mcp.running ? mcp.activity : mcp.lastReply, textW, `600 24px ${FONT}`, mcp.running ? 2 : 3);
  const label = mcp.wokenBy ? `Woken by ${mcp.wokenBy}` : mcp.running ? 'Working' : 'Done';
  const h = 24 + 26 + lines.length * 32 + 8;
  card(c, PAD, y, W - PAD * 2, h);

  const cx = PAD + 34;
  const cy = y + h / 2;
  if (mcp.running) {
    // Like lucide's Loader2: an arc spinning on a light track.
    c.lineWidth = 4;
    c.lineCap = 'round';
    c.strokeStyle = C.track;
    c.beginPath();
    c.arc(cx, cy, 14, 0, Math.PI * 2);
    c.stroke();
    c.strokeStyle = C.blue;
    const a = (now / 150) % (Math.PI * 2);
    c.beginPath();
    c.arc(cx, cy, 14, a, a + Math.PI * 1.2);
    c.stroke();
  } else {
    c.beginPath();
    c.arc(cx, cy, 17, 0, Math.PI * 2);
    c.fillStyle = C.greenTint;
    c.fill();
    icon(c, 'check', cx, cy, 20, C.greenText);
  }

  c.fillStyle = C.muted;
  c.font = `500 18px ${FONT}`;
  c.fillText(fit(c, label, textW), textX, y + 26);
  c.fillStyle = mcp.running ? C.text : C.body;
  c.font = `600 24px ${FONT}`;
  let ty = y + 26 + 32;
  for (const line of lines) {
    c.fillText(line, textX, ty);
    ty += 32;
  }
  return y + h + 16;
}

// Compact rows, one card for the whole list, so every running agent fits.
const ROW_H = 76;
const THUMB = 52;

function drawAgents(c: CanvasRenderingContext2D, rows: StatusAgent[], top: number, bottom: number): void {
  if (!rows.length) {
    c.fillStyle = C.muted;
    c.font = `500 22px ${FONT}`;
    c.fillText('No agents running', PAD + 4, top + 20);
    return;
  }
  // Rows shrink (down to 56px) before any get cut; past that, the last line says how many more.
  const space = bottom - top;
  const rowH = Math.max(56, Math.min(ROW_H, space / rows.length));
  const fits = Math.floor(space / rowH);
  const shown = rows.length > fits ? rows.slice(0, fits - 1) : rows;
  const extra = rows.length - shown.length;
  const h = (shown.length + (extra ? 1 : 0)) * rowH;
  card(c, PAD, top, W - PAD * 2, h);

  shown.forEach((row, i) => {
    const y = top + i * rowH;
    if (i > 0) {
      c.fillStyle = C.border;
      c.fillRect(PAD + 16, y, W - PAD * 2 - 32, 1);
    }
    drawAgentRow(c, row, y, rowH);
  });
  if (extra) {
    c.fillStyle = C.muted;
    c.font = `600 20px ${FONT}`;
    c.fillText(`+${extra} more`, PAD + 18, top + shown.length * rowH + rowH / 2);
  }
}

/** [thumb] Name / Status · 11s ........ LASTWORD (ring) */
function drawAgentRow(c: CanvasRenderingContext2D, row: StatusAgent, y: number, rowH: number): void {
  const cy = y + rowH / 2;
  const thumb = Math.min(THUMB, rowH - 16);

  // What the agent saw on its last iteration, as the model got it.
  const tx = PAD + 14;
  const ty = cy - thumb / 2;
  roundRect(c, tx, ty, thumb, thumb, 10);
  c.fillStyle = C.thumbBg;
  c.fill();
  const img = thumbFor(row.id);
  if (img) drawContain(c, img, tx, ty, thumb);
  roundRect(c, tx, ty, thumb, thumb, 10);
  c.strokeStyle = C.border;
  c.lineWidth = 1.5;
  c.stroke();

  // Countdown ring, like the card's PieTimer: green, blue while sleeping.
  const ringR = 15;
  const rx = W - PAD - 16 - ringR;
  const remaining = row.remainingMs;
  c.lineWidth = 3.5;
  c.strokeStyle = C.track;
  c.beginPath();
  c.arc(rx, cy, ringR, 0, Math.PI * 2);
  c.stroke();
  if (remaining !== null) {
    // Like the card's PieTimer: fills through the loop, green (orange on overrun), blue asleep.
    const frac = row.isSleeping ? 1 : row.progress / 100;
    c.strokeStyle = row.isSleeping ? C.blue : row.isOverrun ? C.orange : C.green;
    c.lineCap = 'round';
    c.beginPath();
    c.arc(rx, cy, ringR, -Math.PI / 2, -Math.PI / 2 + Math.max(0.02, frac) * Math.PI * 2);
    c.stroke();
  }

  // The last word (a decision model's noul), mono like the card, right-aligned by the ring.
  const wordRight = rx - ringR - 12;
  c.font = `700 24px ${MONO}`;
  c.fillStyle = row.lastWord ? C.body : C.muted;
  c.textAlign = 'right';
  const word = fit(c, row.lastWord || '—', 150);
  c.fillText(word, wordRight, cy);
  c.textAlign = 'left';

  const x = tx + thumb + 14;
  const textW = wordRight - c.measureText(word).width - 14 - x;
  c.fillStyle = C.text;
  c.font = `600 21px ${FONT}`;
  c.fillText(fit(c, row.name, textW), x, cy - 12);
  const time = remaining !== null ? ` · ${formatSeconds(remaining)}` : '';
  const label = row.isSleeping ? 'Sleeping' : (row.isOverrun && OVERRUN_LABEL[row.status]) || STATUS_LABEL[row.status];
  c.fillStyle = row.isSleeping ? C.blue : row.isOverrun ? C.orange : C.muted;
  c.font = `500 16px ${FONT}`;
  c.fillText(fit(c, label + time, textW), x, cy + 14);
}

function drawContain(c: CanvasRenderingContext2D, img: HTMLImageElement, x: number, y: number, size: number): void {
  const k = Math.min(size / img.naturalWidth, size / img.naturalHeight);
  const dw = img.naturalWidth * k;
  const dh = img.naturalHeight * k;
  c.save();
  roundRect(c, x, y, size, size, 10);
  c.clip();
  c.drawImage(img, x + (size - dw) / 2, y + (size - dh) / 2, dw, dh);
  c.restore();
}

// ---- Helpers --------------------------------------------------------------------------------

function roundRect(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

/** Cut text to fit maxWidth with an ellipsis, in the current font. */
function fit(c: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (c.measureText(text).width <= maxWidth) return text;
  let s = text;
  while (s.length > 1 && c.measureText(s + '…').width > maxWidth) s = s.slice(0, -1);
  return s + '…';
}

/** Word-wrap into at most maxLines lines, the last one cut with an ellipsis. */
function wrap(c: CanvasRenderingContext2D, text: string, maxWidth: number, font: string, maxLines: number): string[] {
  c.font = font;
  const words = text.replace(/\s+/g, ' ').replace(/[*_`#]/g, '').trim().split(' ');
  const lines: string[] = [];
  let line = '';
  for (let i = 0; i < words.length; i++) {
    const next = line ? `${line} ${words[i]}` : words[i];
    if (c.measureText(next).width <= maxWidth || !line) {
      line = next;
      continue;
    }
    if (lines.length === maxLines - 1) {
      lines.push(fit(c, `${line} ${words.slice(i).join(' ')}`, maxWidth));
      return lines;
    }
    lines.push(line);
    line = words[i];
  }
  if (line) lines.push(fit(c, line, maxWidth));
  return lines;
}

function formatSeconds(ms: number): string {
  const s = Math.ceil(ms / 1000);
  return s >= 60 ? `${Math.floor(s / 60)}m` : `${s}s`;
}
