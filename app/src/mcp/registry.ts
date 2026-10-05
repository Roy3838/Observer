// src/mcp/registry.ts
//
// The v1 Observer MCP tool set. Each tool is a JSON-Schema parameter spec plus a pure
// executor over existing app utilities. Executors are React-free and only touch the
// data layer (agent_database, IterationStore, main_loop, ModelManager, local model managers),
// so the same registry can later be served from a real MCP server with the transport swapped.

import type { ToolDefinition, ToolResult, WireToolSpec, UserInfoKind } from './types';
import {
  listAgents,
  getAgent,
  getAgentCode,
  saveAgent,
  type CompleteAgent,
} from '@utils/agent_database';
import {
  startAgentLoop,
  stopAgentLoop,
  getRunningAgentIds,
  isAgentLoopRunning,
} from '@utils/main_loop';
import { IterationStore, type IterationData } from '@utils/IterationStore';
import { ModelManager } from '@utils/ModelManager';
import type { TokenProvider } from '@utils/main_loop';
import { checkPhoneWhitelist } from '@utils/pre-flight';
import { downloadDefaultLocalModel } from './localModel';
import { tauriStreamCapture } from '@utils/tauriStreamCapture';
import { setAgentCrop } from '@utils/screenCapture';
import { isDesktop, isWeb } from '@utils/platform';
import { browserStreamCapture } from '@utils/browserStreamCapture';
import { tutorialStreamCapture } from '@utils/tutorialStreamCapture';
import { SensorSettings } from '@utils/settings';
import { normalizeWhitelistCode } from '@utils/whitelistCode';
import { fetchStatus } from './remote';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Resolve after `ms`, or reject as soon as `signal` aborts (so a blocking wait is cancellable). */
function sleepOrAbort(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('Aborted', 'AbortError'));
    const onAbort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
    const timer = setTimeout(() => { signal?.removeEventListener('abort', onAbort); resolve(); }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Count images attached to an iteration without surfacing any base64 payloads. */
function countIterationImages(it: IterationData): number {
  let count = it.modelImages?.length || 0;
  for (const sensor of it.sensors) {
    if (sensor.type === 'screenshot' || sensor.type === 'camera') count += 1;
  }
  return count;
}

/** Base64-free summary of a single iteration. */
function summarizeIteration(it: IterationData) {
  return {
    id: it.id,
    sessionId: it.sessionId,
    sessionIterationNumber: it.sessionIterationNumber,
    startTime: it.startTime,
    duration: it.duration,
    modelResponse: it.modelResponse,
    tools: it.tools.map(t => ({ name: t.name, status: t.status })),
    hasError: it.hasError,
    isSkipped: it.isSkipped ?? false,
    sensorTypes: Array.from(new Set(it.sensors.map(s => s.type))),
    imageCount: countIterationImages(it),
  };
}

/** Find a full iteration by id across the in-memory current session and persisted history. */
async function findIteration(iterationId: string, agentId?: string): Promise<IterationData | undefined> {
  const inMemory = IterationStore.getIteration(iterationId);
  if (inMemory) return inMemory;

  // Search persisted history. If agentId is known, restrict to that agent; otherwise
  // we can't enumerate every agent cheaply, so require agentId for historical lookups.
  if (agentId) {
    const sessions = await IterationStore.getHistoricalSessions(agentId);
    for (const session of sessions) {
      const found = session.iterations.find(i => i.id === iterationId);
      if (found) return found;
    }
  }
  return undefined;
}

/**
 * An iteration is "settled" once the micro-agent finished a full pass — the model has
 * responded (or the pass was skipped). Until then its screenshot/response may be missing,
 * so get_iteration waits for this before returning (e.g. when verifying a crop right after
 * start_agent, where the first pass hasn't completed yet).
 */
function isIterationSettled(it: IterationData): boolean {
  return it.modelResponse !== undefined || it.isSkipped === true;
}

/** Build the full multimodal ToolResult (summary + prompt + sensor list + images) for one iteration. */
function iterationToResult(it: IterationData): ToolResult {
  // Collect images (model images + screenshot/camera sensors) as data-URLs.
  const images: string[] = [];
  const toDataUrl = (raw: string) => raw.startsWith('data:') ? raw : `data:image/png;base64,${raw}`;
  if (it.modelImages) images.push(...it.modelImages.map(toDataUrl));
  for (const sensor of it.sensors) {
    if (sensor.type === 'screenshot' || sensor.type === 'camera') {
      const content: any = sensor.content;
      if (typeof content === 'string') images.push(toDataUrl(content));
      else if (content?.data) images.push(toDataUrl(content.data));
    }
  }

  return {
    data: {
      ...summarizeIteration(it),
      modelPrompt: it.modelPrompt,
      sensors: it.sensors.map(s => ({ type: s.type, timestamp: s.timestamp, source: s.source })),
    },
    images,
  };
}

/**
 * start_agent pre-flight, the MCP-path twin of the one in the UI: every 4-word code the agent
 * code sends to (phone tools and sendTelegram) must be connected. Returns a descriptive error
 * telling the model to run ask_user_info, or null when everything checks out / nothing to check.
 */
async function checkContactCodes(agentCode: string, getToken?: TokenProvider): Promise<string | null> {
  const problems: string[] = [];

  // Each use carries its own tool's channel: SMS and WhatsApp have separate codes.
  const { phoneNumbers } = await checkPhoneWhitelist(agentCode, getToken);
  for (const p of phoneNumbers) {
    if (p.isWhitelisted) continue;
    problems.push(p.isCode
      ? `'${p.number}' is not whitelisted for ${p.channel}: the user has not connected this 4-word code for it. Call ask_user_info kind='phone' channel='${p.channel}' so they connect it, then edit_agent if the code changed and start_agent again.`
      : `'${p.number}' is not a 4-word Observer code and phone numbers are never accepted. Call ask_user_info kind='phone' channel='${p.channel}' to get the user's code, replace it in the agent code with edit_agent, then start_agent again.`);
  }

  const telegramRegex = /\bsendTelegram\(\s*["']([^"']+)["']/g;
  const telegramArgs = [...new Set(Array.from(agentCode.matchAll(telegramRegex), m => m[1]))];
  for (const arg of telegramArgs) {
    const code = normalizeWhitelistCode(arg);
    if (!code) {
      problems.push(`'${arg}' is not a 4-word Telegram code (numeric chat_ids are never accepted). Call ask_user_info kind='telegram' to get the user's code, replace it in the agent code with edit_agent, then start_agent again.`);
    } else if (!(await fetchStatus('telegram', code)).linked) {
      problems.push(`Telegram code '${code}' is not whitelisted: no Telegram chat is linked to it. Call ask_user_info kind='telegram' so the user links it, then edit_agent if the code changed and start_agent again.`);
    }
  }

  return problems.length ? `start_agent blocked by pre-flight: ${problems.join(' ')}` : null;
}

// ---------------------------------------------------------------------------
// capture_screen helpers — return one preview frame and leave a live stream that
// start_agent reuses. The platform split mirrors StreamManager (streamManager.ts):
// isWeb() → browser getDisplayMedia; Tauri → the native screen-capture plugin, which
// is itself platform-agnostic (iOS broadcast vs Android MediaProjection handled behind
// one `start_capture_cmd`), so this layer needs no iOS/Android branching.
// ---------------------------------------------------------------------------

/** Web / mobile web: getDisplayMedia picker, grab one frame via a hidden <video>. */
async function captureScreenWeb(): Promise<ToolResult> {
  if (!navigator?.mediaDevices?.getDisplayMedia) {
    return { error: 'Screen capture is not supported in this browser. Screen monitoring agents cannot be created here.' };
  }
  try {
    await browserStreamCapture.acquireMasterStream('display');
    const streams = browserStreamCapture.getStreams();
    const videoTrack = streams.screenVideoStream?.getVideoTracks()[0];
    if (!videoTrack) return { error: 'No video track acquired from screen share.' };

    // Grab one frame via a hidden video element (works across all browsers).
    const video = document.createElement('video');
    video.srcObject = streams.screenVideoStream!;
    video.muted = true;
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

    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    const settings = videoTrack.getSettings();
    return {
      data: {
        captured: true,
        width: settings.width ?? canvas.width,
        height: settings.height ?? canvas.height,
        note: 'Stream is live and will be reused by start_agent — no second picker prompt.',
      },
      images: [dataUrl],
    };
  } catch (e) {
    // User cancelled the picker or permission was denied.
    const msg = e instanceof Error ? e.message : String(e);
    // Safari rejects getDisplayMedia outside a user gesture, which an MCP tool call never is.
    // Same message as main_loop.ts so the model relays it instead of suggesting a retry.
    if (e instanceof Error && msg.includes('getDisplayMedia must be called from a user gesture handler')) {
      return { error: "Safari is bad at screen sharing and can't start automatically, click on the Observer App Icon on the top left to enter the Sensor Permissions Menu and ask for screen sharing manually. Tell user Safari won't capture System Audio, and they have to setup screen sharing manually, and to use chrome, firefox or edge for the best experience." };
    }
    if (msg.includes('Permission denied') || msg.includes('NotAllowed') || msg.includes('user gesture')) {
      return { error: 'Screen share was cancelled or denied. Ask the user if they want to try again.' };
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

/**
 * Acquire tauriStreamCapture's display stream and return its newest frame. Frames arrive
 * asynchronously (on iOS the user taps "Start Broadcast" + a short countdown), so poll the
 * latest raw frame until one lands or we time out, rather than grabbing synchronously like
 * the browser path.
 */
async function grabTauriFrame(data: Record<string, unknown>, noFrameError: string): Promise<ToolResult> {
  try {
    await tauriStreamCapture.acquireMasterStream('display');
  } catch (e) {
    // Android surfaces a real rejection on denial; iOS dismissal just yields no frames.
    const msg = e instanceof Error ? e.message : String(e);
    if (/denied|cancel|NotAllowed/i.test(msg)) {
      return { error: 'Screen capture was cancelled or denied. Ask the user if they want to try again.' };
    }
    return { error: msg };
  }

  // Onboarding tutorial: the stream is a synthetic canvas, not the native plugin, so no raw
  // frames ever arrive for getLatestBase64Frame() — serve a still of the fake screen instead.
  if (tauriStreamCapture.isTutorialDisplayActive()) {
    return {
      data,
      images: [`data:image/jpeg;base64,${tutorialStreamCapture.captureStillFrame()}`],
    };
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

// Sentinel target id used to stand in for a real screen/window during the RecipeSplash
// onboarding tutorial (see SensorSettings.isMcpTutorialMode / tutorialStreamCapture).
const TUTORIAL_TARGET_ID = 'tutorial';

/**
 * Desktop app: start the live stream on a target from list_screen_targets with no selector
 * window, by seating it as the preselected target before acquiring. The frame comes from
 * the same capture pipeline the agent uses (unlike see_screen_target's small xcap
 * thumbnail), so a box_2d read off it lines up with the agent's crop. There is only one
 * master display stream: if one is already live (e.g. a running agent's), its frame is
 * returned as-is and target_id is not applied.
 */
async function captureScreenDesktop(targetId: unknown): Promise<ToolResult> {
  if (typeof targetId !== 'string' || !targetId) {
    return { error: 'On desktop, capture_screen needs a target_id from list_screen_targets.' };
  }

  if (tauriStreamCapture.isStreamAvailable('screenVideo')) {
    return grabTauriFrame(
      {
        captured: true,
        note: `A screen stream was already live (only one can run at a time), so this frame is from that stream and target '${targetId}' was NOT applied. Check the image is what you expected; to switch targets, stop the agents using the screen first.`,
      },
      'The screen stream is live but no frame has arrived yet. Call capture_screen again.',
    );
  }

  let target: Record<string, unknown> = { id: TUTORIAL_TARGET_ID, kind: 'window', name: 'Downloader.app', width: 960, height: 540 };
  // The tutorial target needs no seating: the synthetic stream is substituted inside
  // tauriStreamCapture the moment the display stream is acquired.
  if (targetId !== TUTORIAL_TARGET_ID) {
    try {
      const targets = await tauriStreamCapture.getTargets(false);
      const match = targets.find(t => t.id === targetId);
      if (!match) {
        return { error: `Target '${targetId}' is no longer available (window may have closed). Re-run list_screen_targets.` };
      }
      target = { id: match.id, kind: match.kind, name: match.name, width: match.width, height: match.height };
      tauriStreamCapture.setPreselectedTarget(match.id);
    } catch (e) {
      return { error: e instanceof Error ? e.message : String(e) };
    }
  }

  return grabTauriFrame(
    { captured: true, ...target, note: 'Stream is live on this target and will be reused by start_agent — no selector, no select_screen_target needed.' },
    'Screen capture started but no frame has arrived yet. Call capture_screen again.',
  );
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

export const TOOLS: ToolDefinition[] = [
  // ---- READ TOOLS ----------------------------------------------------------
  {
    name: 'list_agents',
    description: 'List all Observer agents the user has saved, with their basic config (id, name, description, model, loop interval).',
    parameters: { type: 'object', properties: {} },
    multimodal: false,
    execute: async (): Promise<ToolResult> => {
      const agents = await listAgents();
      return {
        data: agents.map(a => ({
          id: a.id,
          name: a.name,
          description: a.description,
          model_name: a.model_name,
          loop_interval_seconds: a.loop_interval_seconds,
          running: isAgentLoopRunning(a.id),
        })),
      };
    },
  },
  {
    name: 'get_agent',
    description: 'Get the full configuration of a single agent by id, including its system prompt and JavaScript code.',
    parameters: {
      type: 'object',
      properties: { id: { type: 'string', description: 'The agent id.' } },
      required: ['id'],
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      const agent = await getAgent(args.id);
      if (!agent) return { error: `Agent '${args.id}' not found.` };
      const code = await getAgentCode(args.id);
      return { data: { agent, code: code ?? '' } };
    },
  },
  {
    name: 'get_status',
    description: 'Get which agents are currently running. Pass an id to check a single agent, or omit it to list all running agents.',
    parameters: {
      type: 'object',
      properties: { id: { type: 'string', description: 'Optional agent id to check.' } },
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      if (args.id) {
        return { data: { id: args.id, running: isAgentLoopRunning(args.id) } };
      }
      return { data: { runningAgentIds: getRunningAgentIds() } };
    },
  },
  {
    name: 'get_runs',
    description: 'Get a summary of an agent\'s recent runs (iterations) across all sessions. Returns metadata only — model responses, tool calls, errors, sensor types, and image counts — with NO image data. Call get_iteration to actually view a screenshot.',
    parameters: {
      type: 'object',
      properties: {
        agent_id: { type: 'string', description: 'The agent id.' },
        limit: { type: 'integer', description: 'Max number of recent iterations to return (default 20).' },
      },
      required: ['agent_id'],
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      const limit = typeof args.limit === 'number' && args.limit > 0 ? args.limit : 20;

      // Current (in-memory) session iterations + persisted history.
      const current = IterationStore.getIterationsForAgent(args.agent_id);
      const historicalSessions = await IterationStore.getHistoricalSessions(args.agent_id);
      const historical = historicalSessions.flatMap(s => s.iterations);

      // De-dupe by id (current session may also appear in history), newest first.
      const byId = new Map<string, IterationData>();
      for (const it of [...historical, ...current]) byId.set(it.id, it);
      const all = Array.from(byId.values()).sort(
        (a, b) => new Date(b.startTime).getTime() - new Date(a.startTime).getTime()
      );

      return {
        data: {
          agent_id: args.agent_id,
          total: all.length,
          iterations: all.slice(0, limit).map(summarizeIteration),
        },
      };
    },
  },
  {
    name: 'get_iteration',
    description: 'Get the full detail of one agent iteration, INCLUDING the screenshot/camera images it captured — use it to actually see what an agent saw. Two ways to call it: (1) pass iteration_id (from get_runs) to fetch that specific iteration, plus agent_id so historical iterations can be located; (2) pass ONLY agent_id to get that agent\'s most recent completed iteration. In mode (2), if the agent was just started and has not finished its first pass yet, this BLOCKS (polling) until the first iteration completes, then returns it — so you can call it right after start_agent to verify a crop without racing the first run. It waits up to 2 minutes; if nothing has completed by then it returns an error and you can get_status / stop_agent to investigate a stuck agent.',
    parameters: {
      type: 'object',
      properties: {
        iteration_id: { type: 'string', description: 'The iteration id (from get_runs). Omit to get the agent\'s latest completed iteration, waiting for the first one if the agent was just started.' },
        agent_id: { type: 'string', description: 'The owning agent id. Required when iteration_id is omitted; also needed to locate historical iterations.' },
      },
    },
    multimodal: true,
    execute: async (args, ctx): Promise<ToolResult> => {
      if (!args.iteration_id && !args.agent_id) {
        return { error: 'Provide iteration_id, or agent_id to wait for and return the agent\'s latest iteration.' };
      }

      // The micro-agent's first pass can take a while (model load, slow remote model, long
      // loop_interval), so block here until an iteration settles — but cap the wait so the
      // MCP loop isn't blocked forever on a stuck agent (it can then get_status / stop_agent).
      const POLL_MS = 2000;
      const WAIT_MS = 2 * 60 * 1000;
      const deadline = Date.now() + WAIT_MS;

      while (true) {
        if (ctx.signal?.aborted) return { error: 'Cancelled.' };

        if (args.iteration_id) {
          const it = await findIteration(args.iteration_id, args.agent_id);
          // A genuinely unknown id won't appear later — fail fast rather than waiting it out.
          if (!it) return { error: `Iteration '${args.iteration_id}' not found. For historical iterations, include agent_id.` };
          if (isIterationSettled(it) || Date.now() >= deadline) return iterationToResult(it);
        } else {
          // agent_id only → newest settled iteration of the current session; wait for the first.
          const current = IterationStore.getIterationsForAgent(args.agent_id);
          const settled = [...current].reverse().find(isIterationSettled);
          if (settled) return iterationToResult(settled);
          if (Date.now() >= deadline) {
            return { error: `No completed iteration for agent '${args.agent_id}' after waiting 2 min. The agent may be stuck or its loop_interval may be longer than that — call get_status, then stop_agent if it's wedged.` };
          }
        }

        try {
          await sleepOrAbort(POLL_MS, ctx.signal);
        } catch {
          return { error: 'Cancelled.' };
        }
      }
    },
  },
  {
    name: 'list_models',
    description: 'List the inference models available to power agents, with their server and multimodal capability.',
    parameters: { type: 'object', properties: {} },
    multimodal: false,
    execute: async (): Promise<ToolResult> => {
      const { models } = ModelManager.getInstance().listModels();
      return {
        data: models.map(m => ({
          name: m.name,
          multimodal: m.multimodal ?? false,
          pro: m.pro ?? false,
          server: m.server,
        })),
      };
    },
  },

  // ---- WRITE TOOLS ---------------------------------------------------------
  {
    name: 'create_agent',
    description: 'Create a new Observer agent (or overwrite one with the same id). The `code` field is JavaScript run after each model call; it uses the SEPARATE agent-code API (sendEmail, appendMemory, $SCREEN sensors in the system_prompt, etc.) — these are NOT function tools you can call here.',
    parameters: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Unique id letters, numbers, and underscores only, no dashes.' },
        name: { type: 'string', description: 'Human-readable agent name.' },
        description: { type: 'string', description: 'Short description of what the agent does.' },
        model_name: { type: 'string', description: 'Model to power the agent (see list_models).' },
        system_prompt: { type: 'string', description: 'The system prompt, including any $SENSOR placeholders ($SCREEN, $MEMORY, $CLIPBOARD, ...).' },
        loop_interval_seconds: { type: 'number', description: 'Seconds between agent iterations. Optional — defaults to 30. Use shorter (~5–15s) for live screen/camera watchers, longer for periodic checks.' },
        code: { type: 'string', description: 'JavaScript run after each model response (agent-code API: response, sendEmail(), appendMemory(), overlay(), startAgent(), stopAgent(), ...).' },
      },
      required: ['id', 'name', 'model_name', 'system_prompt', 'code'],
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      if (!/\$(?:SCREEN|CAMERA|MICROPHONE|CLIPBOARD|MEMORY|LOCATION)\b/.test(args.system_prompt)) {
        return { error: 'The system_prompt must include at least one sensor placeholder (e.g. $SCREEN, $CAMERA, $MICROPHONE, $CLIPBOARD, $MEMORY, $LOCATION). Agents without a sensor have no perception — add one and try again.' };
      }
      const agent: CompleteAgent = {
        id: args.id,
        name: args.name,
        description: args.description ?? '',
        model_name: args.model_name,
        system_prompt: args.system_prompt,
        loop_interval_seconds: args.loop_interval_seconds ?? 30,
      };
      try {
        const saved = await saveAgent(agent, args.code);
        return { data: { saved: true, id: saved.id, name: saved.name } };
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
    },
  },
  {
    name: 'edit_agent',
    description: 'Edit an existing agent by id. Provide the complete updated configuration (same shape as create_agent). Fails if the agent does not exist.',
    parameters: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'The id of the existing agent to edit.' },
        name: { type: 'string', description: 'Human-readable agent name.' },
        description: { type: 'string', description: 'Short description of what the agent does.' },
        model_name: { type: 'string', description: 'Model to power the agent.' },
        system_prompt: { type: 'string', description: 'The system prompt.' },
        loop_interval_seconds: { type: 'number', description: 'Seconds between agent iterations. Optional — omit to keep the agent\'s current interval.' },
        code: { type: 'string', description: 'JavaScript run after each model response (agent-code API).' },
      },
      required: ['id', 'name', 'model_name', 'system_prompt', 'code'],
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      const existing = await getAgent(args.id);
      if (!existing) return { error: `Agent '${args.id}' does not exist. Use create_agent to make a new one.` };
      const agent: CompleteAgent = {
        id: args.id,
        name: args.name,
        description: args.description ?? '',
        model_name: args.model_name,
        system_prompt: args.system_prompt,
        loop_interval_seconds: args.loop_interval_seconds ?? existing.loop_interval_seconds ?? 30,
      };
      try {
        const saved = await saveAgent(agent, args.code);
        return { data: { saved: true, id: saved.id, name: saved.name } };
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
    },
  },
  {
    name: 'check_whitelist',
    description: "Pre-flight check for the phone notification tools (sendSms, call, sendWhatsapp). They only send to the user's 4-word Observer code once it is connected for that tool — the SMS code (texted to Observer) for sendSms, the WhatsApp code (sent on WhatsApp) for sendWhatsapp, either for call — and start_agent FAILS otherwise, so call this BEFORE start_agent for any agent that uses a phone tool. Pass the code the agent sends to (defaults to the user's saved code for that channel) and the channel. This does NOT block and shows the user nothing: it returns {whitelisted:true} if the code is connected, and FAILS if it isn't. On failure, call ask_user_info kind='phone' (same channel) to get the user connected, then call check_whitelist again (or just start_agent). Phone numbers are never accepted: if an agent uses one, replace it with the user's code.",
    parameters: {
      type: 'object',
      properties: {
        code: { type: 'string', description: "The 4-word Observer code the agent passes to the phone tool (e.g. \"tree-book-shower-golden\"). Omit to use the user's saved code for the channel." },
        channel: { type: 'string', enum: ['sms', 'voice', 'whatsapp'], description: "Which tool the code is used with: sms (sendSms), voice (call), or whatsapp (sendWhatsapp). WhatsApp also needs WhatsApp's 24h window to be open. Defaults to whatsapp." },
      },
    },
    multimodal: false,
    execute: async (args, ctx): Promise<ToolResult> => {
      // `phone_number` is what this tool took before codes-only; a chat started under the
      // old prompt may still pass it.
      const raw: string | undefined = args.code ?? args.phone_number;
      const code = raw ? normalizeWhitelistCode(raw) : SensorSettings.defaultPhoneCode(args.channel ?? 'sms');
      if (!code) {
        const saved = SensorSettings.defaultPhoneCode(args.channel ?? 'sms');
        return {
          error: `'${raw}' is not an Observer code. The phone tools only send to the user's 4-word code${saved ? ` ('${saved}')` : ''}, never to a phone number: put the code in the agent instead${saved ? '' : " (ask_user_info kind='phone' gets it)"}, then check it.`,
        };
      }

      // Funnel through the same checkPhoneWhitelist gate start_agent uses by handing it a
      // one-line snippet — keeps channel handling + the API call in a single place.
      const fn = args.channel === 'whatsapp' ? 'sendWhatsapp'
        : args.channel === 'voice' ? 'call'
        : 'sendSms';
      const snippet = `${fn}("${code}")`;
      const channel = args.channel ?? 'sms';

      try {
        const { phoneNumbers } = await checkPhoneWhitelist(snippet, ctx.getToken);
        if (phoneNumbers.length > 0 && phoneNumbers.every(p => p.isWhitelisted)) {
          return { data: { code, channel, whitelisted: true } };
        }
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
      return {
        error: `Code '${code}' is not connected for ${channel}. Call ask_user_info kind='phone' channel='${channel}' so the user can connect it, then check again.`,
      };
    },
  },
  {
    name: 'ask_user_info',
    description: "Ask the user for a piece of contact info needed by a notification tool, via a guided modal. Use this INSTEAD of asking for a phone number / chat_id / webhook URL in chat prose — the modal walks the user through actually obtaining the value (QR codes, deep links, step-by-step instructions) and prefills anything they've given before. Call it BEFORE create_agent, once per piece of info you need. This BLOCKS until the user confirms. For kind='phone' the value is ALWAYS the user's 4-word Observer code for that channel (e.g. \"tree-book-shower-golden\"): the SMS code for sms, the WhatsApp code for whatsapp, either for voice. Never a phone number. Each channel has its own code, so ask again (with the right channel) when an agent uses both sendSms and sendWhatsapp. The modal only returns once it's connected, so you do NOT need a separate check_whitelist call for it; pass it verbatim as the phone_number/number argument to sendSms/sendWhatsapp/call. Observer never sends to raw phone numbers. For kind='telegram' the value is the user's separate 4-word Telegram code (not the phone code, never a numeric chat_id), already linked; pass it verbatim as sendTelegram's first argument. If the result is {skipped:true}, the user declined — ask them about it in chat rather than calling this again. Do not narrate the modal or tell the user to fill it in; they can see it.",
    parameters: {
      type: 'object',
      properties: {
        kind: {
          type: 'string',
          enum: ['phone', 'email', 'telegram', 'discord', 'pushover'],
          description: "Which piece of info to collect: 'phone' for sendSms/call/sendWhatsapp, 'email' for sendEmail, 'telegram' for sendTelegram's code, 'discord' for sendDiscord's webhook URL, 'pushover' for sendPushover's user key.",
        },
        channel: {
          type: 'string',
          enum: ['sms', 'voice', 'whatsapp'],
          description: "Only for kind='phone': which tool the code will be used with — sms (sendSms), voice (call), or whatsapp (sendWhatsapp). WhatsApp also needs WhatsApp's 24h window to be open. Defaults to sms.",
        },
        reason: {
          type: 'string',
          description: 'A short, friendly one-line explanation of what this will be used for, shown in the modal (e.g. "so I can text you when your render finishes").',
        },
      },
      required: ['kind'],
    },
    multimodal: false,
    execute: async (args, ctx): Promise<ToolResult> => {
      const kind = args.kind as UserInfoKind;
      if (!kind) return { error: 'Provide a `kind` to ask for.' };

      // Non-React host (or a future MCP server transport): no modal to show, so tell the
      // model to fall back to plain conversation rather than silently hanging.
      if (!ctx.requestUserInfo) {
        return { error: 'No interactive UI is available here. Ask the user for this value in chat instead.' };
      }

      const channel = kind === 'phone'
        ? (args.channel === 'sms' ? 'sms' : args.channel === 'voice' ? 'voice' : 'whatsapp')
        : undefined;

      let response;
      try {
        response = await ctx.requestUserInfo({
          requestId: `info_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          kind,
          channel,
          reason: typeof args.reason === 'string' ? args.reason : undefined,
        });
      } catch {
        return { error: 'Cancelled.' };
      }

      if (response.skipped) {
        return { data: { kind, skipped: true, note: 'The user dismissed the prompt without providing a value. Ask them about it in chat; do not call ask_user_info again for this.' } };
      }

      const value = (response.value ?? '').trim();
      if (!value) return { data: { kind, skipped: true } };

      // The modal only resolves a phone once the code is confirmed connected, so a returned
      // code is always ready to use — no separate check_whitelist needed.
      return { data: { kind, value, ...(kind === 'phone' ? { channel, whitelisted: true } : {}) } };
    },
  },
  {
    name: 'start_agent',
    description: 'Start an agent\'s run loop. This runs the agent\'s sandboxed code on a schedule (which may send emails/SMS, click, etc.).',
    parameters: {
      type: 'object',
      properties: { id: { type: 'string', description: 'The agent id to start.' } },
      required: ['id'],
    },
    multimodal: false,
    execute: async (args, ctx): Promise<ToolResult> => {
      const agent = await getAgent(args.id);
      if (!agent) return { error: `Agent '${args.id}' not found.` };
      try {
        const blocked = await checkContactCodes((await getAgentCode(args.id)) ?? '', ctx.getToken);
        if (blocked) return { error: blocked };
        await startAgentLoop(args.id, ctx.getToken);
        return { data: { started: true, id: args.id } };
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
    },
  },
  {
    name: 'stop_agent',
    description: 'Stop a running agent\'s loop. Always safe and reversible.',
    parameters: {
      type: 'object',
      properties: { id: { type: 'string', description: 'The agent id to stop.' } },
      required: ['id'],
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      await stopAgentLoop(args.id);
      return { data: { stopped: true, id: args.id } };
    },
  },
  {
    name: 'list_screen_targets',
    description: 'List the screens (monitors) and windows available to capture for a $SCREEN agent, as a text-only catalog (NO images — a desktop can have many windows, so thumbnails are fetched one at a time with see_screen_target). Call this on desktop BEFORE create_agent for any agent whose system_prompt uses $SCREEN: read the list, see_screen_target the few that plausibly match what the user wants to watch, then capture_screen the best one. On the web/mobile app this returns a note instead — there the OS picker appears automatically when the agent starts, so just go straight to start_agent. Each target has an id (for see_screen_target / capture_screen / select_screen_target), kind (monitor/window), name, appName, and width/height in pixels (for context; set_screen_crop takes a normalized box_2d, not these pixels).',
    parameters: { type: 'object', properties: {} },
    multimodal: false,
    execute: async (): Promise<ToolResult> => {
      if (!isDesktop()) {
        return {
          data: {
            platform: 'web',
            targets: [],
            note: 'You are on Observer web — there is nothing to list or select here. Screen selection is handled by the OS picker, which pops automatically at start_agent. Do NOT stop, summarize, or wait for the user: continue the flow in this same run and proceed to start_agent now (it will surface its own approval card). Skip see_screen_target / select_screen_target / set_screen_crop entirely.',
          },
        };
      }
      // Onboarding tutorial: hand back exactly one synthetic target instead of enumerating
      // real windows. Peek only (never consumes the flag) — list/see can be called freely;
      // the flag is actually spent later, at the moment a stream is really acquired.
      if (SensorSettings.isMcpTutorialMode()) {
        return {
          data: {
            platform: 'desktop',
            targets: [{ id: TUTORIAL_TARGET_ID, kind: 'window', name: 'Downloader.app', appName: 'Downloader', width: 960, height: 540, isPrimary: false }],
          },
        };
      }
      try {
        const targets = await tauriStreamCapture.getTargets(false);
        return {
          data: {
            platform: 'desktop',
            targets: targets.map(t => ({
              id: t.id,
              kind: t.kind,
              name: t.name,
              appName: t.appName,
              width: t.width,
              height: t.height,
              isPrimary: t.isPrimary,
            })),
          },
        };
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
    },
  },
  {
    name: 'see_screen_target',
    description: 'Fetch a small, low-resolution thumbnail of ONE capture target so you can tell which monitor/window it is. Pass a target_id from list_screen_targets. Desktop only. Use this to check the one or few candidates that match what the user wants to watch, then capture_screen the right one. The thumbnail is for IDENTIFYING the target only — it is too small and not framed exactly like the live capture, so never read a set_screen_crop box_2d off it; read the crop off capture_screen\'s frame instead. Starts no stream. If the target has since closed, this fails; re-run list_screen_targets.',
    parameters: {
      type: 'object',
      properties: {
        target_id: { type: 'string', description: 'The id of the target to preview (from list_screen_targets).' },
      },
      required: ['target_id'],
    },
    multimodal: true,
    execute: async (args): Promise<ToolResult> => {
      if (!isDesktop()) {
        return { error: 'see_screen_target is desktop-only; on web the OS picker shows the screens at start_agent.' };
      }
      if (args.target_id === TUTORIAL_TARGET_ID) {
        return {
          data: { id: TUTORIAL_TARGET_ID, kind: 'window', name: 'Downloader.app', appName: 'Downloader', width: 960, height: 540, hasThumbnail: true },
          images: [`data:image/jpeg;base64,${tutorialStreamCapture.captureStillFrame()}`],
        };
      }
      try {
        const targets = await tauriStreamCapture.getTargets(true);
        const match = targets.find(t => t.id === args.target_id);
        if (!match) {
          return { error: `Target '${args.target_id}' is no longer available (window may have closed). Re-run list_screen_targets.` };
        }
        const raw = match.thumbnail;
        const url = !raw ? undefined : raw.startsWith('data:') ? raw : `data:image/png;base64,${raw}`;
        return {
          data: {
            id: match.id,
            kind: match.kind,
            name: match.name,
            appName: match.appName,
            width: match.width,
            height: match.height,
            hasThumbnail: !!url,
          },
          images: url ? [url] : undefined,
        };
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
    },
  },
  {
    name: 'select_screen_target',
    description: 'Pre-select which screen or window a $SCREEN agent will capture, so start_agent runs without popping the desktop screen-selector — WITHOUT starting the stream or returning an image. Pass a target_id from list_screen_targets. Desktop only. Not needed after capture_screen (which already starts the stream on its target); use this only to seat a target you do not need to look at. Call it right before start_agent. If the chosen window has since closed, this fails — re-run list_screen_targets and pick again.',
    parameters: {
      type: 'object',
      properties: {
        target_id: { type: 'string', description: 'The id of the target to capture (from list_screen_targets).' },
      },
      required: ['target_id'],
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      if (!isDesktop()) {
        return { error: 'select_screen_target is desktop-only; on web the OS picker handles selection at start_agent.' };
      }
      if (args.target_id === TUTORIAL_TARGET_ID) {
        // Nothing to pre-seat: the tutorial stream is substituted automatically, inside
        // tauriStreamCapture, the moment start_agent actually acquires the display stream.
        return { data: { selected: true, id: TUTORIAL_TARGET_ID, kind: 'window', name: 'Downloader.app', width: 960, height: 540 } };
      }
      try {
        const targets = await tauriStreamCapture.getTargets(false);
        const match = targets.find(t => t.id === args.target_id);
        if (!match) {
          return { error: `Target '${args.target_id}' is no longer available (window may have closed). Re-run list_screen_targets and pick again.` };
        }
        tauriStreamCapture.setPreselectedTarget(match.id);
        return { data: { selected: true, id: match.id, kind: match.kind, name: match.name, width: match.width, height: match.height } };
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
    },
  },
  {
    name: 'set_screen_crop',
    description: 'OPTIONAL. Crop a $SCREEN agent\'s capture to a rectangular sub-region of its target, so the model only sees (and only spends tokens on) the part that matters — e.g. a download progress bar. Skip this entirely to watch the whole screen/window (the default). Give the region as box_2d = [ymin, xmin, ymax, xmax] in the SAME normalized 0–1000 coordinate space you use for object detection: top-left origin, y first, every value 0–1000 regardless of the screen\'s real resolution. Read box_2d straight off the frame capture_screen returned (or an iteration image from get_iteration) — never off a see_screen_target thumbnail, and do NOT convert to pixels yourself, and do NOT pass the target\'s resolution; the crop is stored normalized and resolved against the live frame at capture time. Pass clear:true to remove an existing crop and capture the full target.',
    parameters: {
      type: 'object',
      properties: {
        agent_id: { type: 'string', description: 'The agent whose screen capture to crop.' },
        box_2d: {
          type: 'array',
          items: { type: 'number' },
          minItems: 4,
          maxItems: 4,
          description: 'The crop region as [ymin, xmin, ymax, xmax], normalized to 0–1000 (top-left origin, y first) — the same format you emit for bounding boxes.',
        },
        clear: { type: 'boolean', description: 'If true, remove any existing crop and capture the full target. Ignores box_2d.' },
      },
      required: ['agent_id'],
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      if (args.clear) {
        setAgentCrop(args.agent_id, 'screen', null);
        return { data: { agent_id: args.agent_id, cropped: false } };
      }

      const box = args.box_2d;
      if (!Array.isArray(box) || box.length !== 4 || box.some((v: unknown) => typeof v !== 'number')) {
        return { error: 'Provide box_2d as four numbers [ymin, xmin, ymax, xmax] normalized 0–1000 (or clear:true to remove the crop).' };
      }
      const [ymin, xmin, ymax, xmax] = box as number[];
      if ([ymin, xmin, ymax, xmax].some(v => v < 0 || v > 1000)) {
        return { error: 'box_2d values must be normalized to 0–1000.' };
      }
      if (xmax <= xmin || ymax <= ymin) {
        return { error: 'box_2d must satisfy xmax > xmin and ymax > ymin (it is [ymin, xmin, ymax, xmax]).' };
      }

      // Store the crop as fractions of the frame (0–1) — resolution-independent.
      // It is resolved to pixels against the real capture frame at capture time, so it
      // stays correct regardless of the target's resolution or the capture pipeline's
      // scaling (the previous descale-to-target-pixels step mismatched the actual frame).
      const crop = {
        x: xmin / 1000,
        y: ymin / 1000,
        width: (xmax - xmin) / 1000,
        height: (ymax - ymin) / 1000,
      };

      setAgentCrop(args.agent_id, 'screen', crop);
      return { data: { agent_id: args.agent_id, cropped: true, box_2d: [ymin, xmin, ymax, xmax], crop } };
    },
  },
  {
    name: 'capture_screen',
    description: 'Start the live screen stream a $SCREEN agent will use and return one frame from it, so you can SEE exactly what will be monitored before building the agent. The frame comes from the real capture pipeline, so it is the image to read a set_screen_crop box_2d from. On the desktop app pass target_id (from list_screen_targets): the stream starts on that screen/window with NO selector popping up; only one screen stream runs at a time, so if one is already live (e.g. a running agent\'s) you get its frame and target_id is not applied. On web / mobile web (no target_id) this opens the browser screen-share picker (pick a screen, window, or tab). On the mobile app (no target_id) it triggers the OS screen-capture picker and captures the WHOLE screen (iOS broadcast / Android screen-record permission): the user must approve the system prompt, and on iOS there can be a few seconds of delay before the first frame — if this returns a "no frame yet" message, just call it again. The stream stays live — start_agent reuses it without prompting again. Call it BEFORE create_agent for any agent whose system_prompt uses $SCREEN.',
    parameters: {
      type: 'object',
      properties: {
        target_id: { type: 'string', description: 'Desktop app only (required there): the id of the target to capture, from list_screen_targets. Omit on web / mobile.' },
      },
    },
    multimodal: true,
    // Same platform split as StreamManager: browser → getDisplayMedia; Tauri → native plugin
    // (desktop seats the target up front so no selector opens).
    execute: async (args): Promise<ToolResult> => isWeb()
      ? captureScreenWeb()
      : isDesktop() ? captureScreenDesktop(args.target_id) : captureScreenTauri(),
  },
  {
    name: 'download_model',
    description: 'Download and load the default on-device model so agents can run locally with NO cloud and NO API key. Takes no arguments — Observer picks the right Qwen3.5 0.8B build for the platform (a transformers.js ONNX model in the browser, a llama.cpp GGUF in the desktop app). This BLOCKS while it downloads (under 1 GB) and loads; progress bars are shown to the user. When it resolves, the returned `model_name` is immediately usable as a `create_agent` model_name. Only one local model is needed; call list_models afterward to confirm.',
    parameters: { type: 'object', properties: {} },
    multimodal: false,
    execute: async (): Promise<ToolResult> => {
      try {
        const result = await downloadDefaultLocalModel();
        return { data: result };
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
    },
  },
];

// ---------------------------------------------------------------------------
// Registry lookups
// ---------------------------------------------------------------------------

/** Desktop-only screen tools — on all other surfaces the OS/browser picker inside capture_screen does the choosing. */
const DESKTOP_ONLY_TOOLS = new Set(['list_screen_targets', 'see_screen_target', 'select_screen_target']);

function getPlatformTools(): ToolDefinition[] {
  if (isDesktop()) return TOOLS;
  // Web, mobile app, mobile web — sub-agentic: browser picker flow.
  return TOOLS.filter(t => !DESKTOP_ONLY_TOOLS.has(t.name));
}

export function getTool(name: string): ToolDefinition | undefined {
  return getPlatformTools().find(t => t.name === name);
}

/** Build the OpenAI `tools` request payload for the current platform. */
export function getToolSpecs(): WireToolSpec[] {
  return getPlatformTools().map(t => ({
    type: 'function' as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    },
  }));
}
