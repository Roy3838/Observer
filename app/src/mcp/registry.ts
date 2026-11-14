// src/mcp/registry.ts
//
// The v1 Observer MCP tool set. Each tool is a JSON-Schema parameter spec plus a pure
// executor over existing app utilities. Executors are React-free and only touch the
// data layer (agent_database, IterationStore, main_loop, ModelManager, local model managers),
// so the same registry can later be served from a real MCP server with the transport swapped.

import type { ToolDefinition, ToolResult, WireToolSpec, NotifyMethod, UserInfoRequestKind } from './types';
import { captureScreen, captureCamera, listCameras, grabTauriFrame, waitForTauriFrame, releaseCaptureHolds, letGoOfScreen } from './capture';
import { isStatusPipActive, isStatusPipSupported, prepareStatusPip } from '@utils/statusPip';
import { StreamManager } from '@utils/streamManager';
import { imageDataUrl } from '@utils/imageDataUrl';
import {
  listAgents,
  getAgent,
  getAgentCode,
  getAgentMemory,
  saveAgent,
  deleteAgent,
  type CompleteAgent,
} from '@utils/agent_database';
import {
  startAgentLoop,
  stopAgentLoop,
  getRunningAgentIds,
  getRequiredStreamsForPrompt,
  isAgentLoopRunning,
} from '@utils/main_loop';
import { IterationStore, type IterationData } from '@utils/IterationStore';
import { ModelManager } from '@utils/ModelManager';
import type { TokenProvider } from '@utils/main_loop';
import { checkPhoneWhitelist } from '@utils/pre-flight';
import { downloadLocalModel, LOCAL_MODEL_CHOICES, type LocalModelChoice } from './localModel';
import { tauriStreamCapture } from '@utils/tauriStreamCapture';
import { setAgentCrop, getAgentCrop, cropDataUrl, type CropConfig } from '@utils/screenCapture';
import { isDesktop, isMobile } from '@utils/platform';
import { getPhoneLockStatus, phoneLockPermissionName, browserNotifyReady } from '@utils/handlers/utils';
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
  if (it.modelImages) images.push(...it.modelImages.map(imageDataUrl));
  for (const sensor of it.sensors) {
    if (sensor.type === 'screenshot' || sensor.type === 'camera') {
      const content: any = sensor.content;
      if (typeof content === 'string') images.push(imageDataUrl(content));
      else if (content?.data) images.push(imageDataUrl(content.data));
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
      ? `'${p.number}' is not whitelisted for ${p.channel}: the user has not connected this 4-word code for it. Call ask_user_info kind='method' channel='${p.channel}' so they connect it, then edit_agent if the code changed and start_agent again.`
      : `'${p.number}' is not a 4-word Observer code and phone numbers are never accepted. Call ask_user_info kind='method' channel='${p.channel}' to get the user's code, replace it in the agent code with edit_agent, then start_agent again.`);
  }

  const telegramRegex = /\bsendTelegram\(\s*["']([^"']+)["']/g;
  const telegramArgs = [...new Set(Array.from(agentCode.matchAll(telegramRegex), m => m[1]))];
  for (const arg of telegramArgs) {
    const code = normalizeWhitelistCode(arg);
    if (!code) {
      problems.push(`'${arg}' is not a 4-word Telegram code (numeric chat_ids are never accepted). Call ask_user_info kind='method' channel='telegram' to get the user's code, replace it in the agent code with edit_agent, then start_agent again.`);
    } else if (!(await fetchStatus('telegram', code)).linked) {
      problems.push(`Telegram code '${code}' is not whitelisted: no Telegram chat is linked to it. Call ask_user_info kind='method' channel='telegram' so the user links it, then edit_agent if the code changed and start_agent again.`);
    }
  }

  return problems.length ? `start_agent blocked by pre-flight: ${problems.join(' ')}` : null;
}

/**
 * start_agent pre-flight for lockPhone(): the permission has to be granted from a tap while the
 * user is in the app, because the agent itself runs in the background where the prompt can't
 * show. Returns an error pointing at ask_user_info kind='lock_phone', or null.
 */
async function checkPhoneLock(agentCode: string): Promise<string | null> {
  if (!isMobile() || !/\blockPhone\s*\(/.test(agentCode)) return null;
  const status = await getPhoneLockStatus();
  if (!status.supported) return 'start_agent blocked by pre-flight: lockPhone() needs iOS 16 or later. Tell the user, and remove lockPhone() with edit_agent.';
  if (status.authorized) return null;
  return `start_agent blocked by pre-flight: lockPhone() needs ${phoneLockPermissionName()}. Call ask_user_info kind='lock_phone' so the user can allow it, then start_agent again.`;
}

/**
 * start_agent pre-flight for notify() / system_notify(): the user allows notifications from a
 * tap, by accepting a test one in ask_user_info, so the browser / OS prompt never pops up the
 * first time the agent fires (or gets silently dropped because no gesture was behind it).
 */
function checkNotifications(agentCode: string): string | null {
  if (/\bnotify\s*\(/.test(agentCode) && !browserNotifyReady()) {
    return "start_agent blocked by pre-flight: notify() needs notification permission. Call ask_user_info kind='method' channel='browser' so the user accepts a test notification, then start_agent again.";
  }
  if (isDesktop() && /\bsystem_notify\s*\(/.test(agentCode) && !SensorSettings.getSystemNotifyConfirmed()) {
    return "start_agent blocked by pre-flight: system_notify() hasn't been tested on this computer. Call ask_user_info kind='method' channel='system' so the user accepts a test notification, then start_agent again.";
  }
  return null;
}

/** The iteration get_iteration last returned per agent (agent_id mode), so a repeat call waits for a newer one. */
const lastSeenIteration = new Map<string, string>();

const NOTIFY_METHODS: NotifyMethod[] = ['browser', 'system', 'email', 'whatsapp', 'sms', 'voice', 'telegram', 'discord', 'pushover'];

/** How the agent code uses what kind='method' returned, spelled out so the model can't mix codes up. */
const METHOD_USAGE: Record<NotifyMethod, (v: string) => string> = {
  browser: () => 'notify(title, message)',
  system: () => 'system_notify(body, title?)',
  email: v => `sendEmail("${v}", message, images?)`,
  whatsapp: v => `sendWhatsapp("${v}", message, images?)`,
  sms: v => `sendSms("${v}", message, images?)`,
  voice: v => `call("${v}", message)`,
  telegram: v => `sendTelegram("${v}", message, images?)`,
  discord: v => `sendDiscord("${v}", message, images?)`,
  pushover: v => `sendPushover("${v}", message, images?)`,
};

// ---------------------------------------------------------------------------
// capture_screen on desktop — web and mobile capture lives in capture.ts, shared with the
// ask_user_info 'screen' / 'camera' modal.
// ---------------------------------------------------------------------------

/**
 * Desktop app: start the live stream on a target from list_screen_targets with no selector
 * window, by seating it as the preselected target before acquiring. The frame comes from
 * the same capture pipeline the agent uses (unlike see_screen_target's small xcap
 * thumbnail), so a box_2d read off it lines up with the agent's crop. There is only one
 * master display stream: a different target_id restarts it on that target, unless something
 * running depends on it. Then its frame is returned as-is and target_id is not applied.
 */
async function captureScreenDesktop(targetId: unknown): Promise<ToolResult> {
  const live = tauriStreamCapture.isStreamAvailable('screenVideo');
  const wanted = typeof targetId === 'string' && targetId ? targetId : null;
  const noFrameYet = 'The screen stream is live but no frame has arrived yet. Call capture_screen again.';

  // A live stream is reused as-is, so no target_id is needed to look at it again (agent_id previews).
  if (live && (!wanted || wanted === tauriStreamCapture.getLiveTargetId())) {
    return grabTauriFrame(
      { captured: true, note: 'This frame is from the screen stream that was already live. Check the image is what you expected; to switch, pass a different target_id.' },
      noFrameYet,
    );
  }

  if (!wanted) {
    return { error: 'On desktop, capture_screen needs a target_id from list_screen_targets.' };
  }

  let target: Record<string, unknown>;
  try {
    const targets = await tauriStreamCapture.getTargets(false);
    const match = targets.find(t => t.id === wanted);
    if (!match) {
      return { error: `Target '${wanted}' is no longer available (window may have closed). Re-run list_screen_targets.` };
    }
    target = { id: match.id, kind: match.kind, name: match.name, width: match.width, height: match.height };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }

  // Switching would silently point a running agent at another window, so if the stream outlives
  // the MCP letting go of it, it stays. On macOS system audio shares the capture session, which
  // would stay on its current target even with no video stream live.
  const audioPinsTarget = /mac/i.test(navigator.platform) && !!StreamManager.getCurrentState().screenAudioStream;
  if (audioPinsTarget || (live && !letGoOfScreen())) {
    const notApplied = `Target '${wanted}' was NOT applied: the screen or system audio is in use, usually by a running agent, and only one screen stream can run at a time. To switch targets, stop_agent those agents, then call capture_screen again. If none are running, tell the user another part of the app is holding the screen.`;
    return live ? waitForTauriFrame({ captured: true, note: notApplied }, noFrameYet) : { error: notApplied };
  }

  tauriStreamCapture.setPreselectedTarget(wanted);

  return grabTauriFrame(
    { captured: true, ...target, note: 'Stream is live on this target and will be reused by start_agent — no selector, no select_screen_target needed.' },
    'Screen capture started but no frame has arrived yet. Call capture_screen again.',
  );
}

/** Set an agent's crop and let an open agent card redraw it. */
function setCropAndNotify(agentId: string, stream: 'screen' | 'camera', crop: CropConfig | null): void {
  setAgentCrop(agentId, stream, crop);
  window.dispatchEvent(new CustomEvent('agent-crop-update', { detail: { agentId } }));
}

/** set_screen_crop / set_camera_crop: validate box_2d and store it on the agent. */
function applyCropArgs(stream: 'screen' | 'camera', args: Record<string, any>): ToolResult {
  if (args.clear) {
    setCropAndNotify(args.agent_id, stream, null);
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

  setCropAndNotify(args.agent_id, stream, crop);
  return { data: { agent_id: args.agent_id, cropped: true, box_2d: [ymin, xmin, ymax, xmax], crop } };
}

/** capture_screen / capture_camera with agent_id: the same fractions and resolveCrop its loop applies. */
async function previewAgentCrop(result: ToolResult, stream: 'screen' | 'camera', agentId: unknown): Promise<ToolResult> {
  if (typeof agentId !== 'string' || !agentId || result.error || !result.images?.[0]) return result;

  const crop = getAgentCrop(agentId, stream);
  if (!crop) return { ...result, data: { ...(result.data as object), note: `Agent '${agentId}' has no crop set: this is the full frame it will receive.` } };
  const cropped = await cropDataUrl(result.images[0], crop);
  return {
    data: { agent_id: agentId, cropped: true, crop_px: [cropped.width, cropped.height], note: 'This is exactly what the model of the agent receives each iteration.' },
    images: [cropped.dataUrl],
  };
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
    description: 'Get the full configuration of a single agent by id, including its system prompt, JavaScript code and memory.',
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
      const memory = await getAgentMemory(args.id);
      return { data: { agent, code: code ?? '', memory } };
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
    description: 'Get the full detail of one agent iteration, INCLUDING the screenshot/camera images it captured — use it to actually see what an agent saw. Two ways to call it: (1) pass iteration_id (from get_runs) to fetch that specific iteration, plus agent_id so historical iterations can be located; (2) pass ONLY agent_id to get that agent\'s most recent completed iteration. In mode (2) this BLOCKS (polling) until there is an iteration you have not seen yet: right after start_agent it waits for the first one, and calling it again waits for the NEXT one, so two calls always give you two different moments to compare. It waits up to 2 minutes; if nothing has completed by then it returns an error and you can get_status / stop_agent to investigate a stuck agent.',
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
          // A repeat call waits for a newer iteration: the second look must be a later moment.
          if (settled && settled.id !== lastSeenIteration.get(args.agent_id)) {
            lastSeenIteration.set(args.agent_id, settled.id);
            return iterationToResult(settled);
          }
          // Stopped (its code called stopAgent(), or an error stopped it): the session was
          // moved to IndexedDB, so nothing new will appear in memory. Return its last iteration.
          if (!isAgentLoopRunning(args.agent_id)) {
            const [session] = await IterationStore.getHistoricalSessions(args.agent_id);
            const last = session && [...session.iterations].reverse().find(isIterationSettled);
            if (last) {
              const result = iterationToResult(last);
              const stopped = `The agent is STOPPED (since ${session.endTime ?? 'its last iteration'}): its code called stopAgent() or an error stopped it. This is its last iteration. If it stopped on the very first iterations while the job is still going, its condition is wrong: fix it before start_agent again.`;
              return { ...result, data: { ...result.data, stopped } };
            }
          }
          if (Date.now() >= deadline) {
            if (settled) return { error: `Agent '${args.agent_id}' finished no new iteration in 2 min after the one you already saw (${settled.id}). Its loop_interval may be longer than that — call get_status, then stop_agent if it's wedged.` };
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
    description: 'List the inference models available to power agents, with their server and multimodal capability. `decision: true` marks a decision model (OneJev): it answers a yes/no question instead of writing text, so its agent code reads `decision.noul`, never `response`.',
    parameters: { type: 'object', properties: {} },
    multimodal: false,
    execute: async (): Promise<ToolResult> => {
      const { models } = ModelManager.getInstance().listModels();
      return {
        data: models.map(m => ({
          name: m.name,
          multimodal: m.multimodal ?? false,
          decision: m.decision ?? false,
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
        code: { type: 'string', description: 'JavaScript run after each model response (agent-code API: response, or decision.noul for a decision model, sendEmail(), appendMemory(), overlay(), startAgent(), stopAgent(), ...).' },
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
      if (isAgentLoopRunning(args.id)) {
        return { error: `Agent '${args.id}' is running: its old prompt and code would keep acting while you edit. stop_agent first, then edit_agent, then start_agent again.` };
      }
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
    description: "Pre-flight check for the phone notification tools (sendSms, call, sendWhatsapp). They only send to the user's 4-word Observer code once it is connected for that tool — the SMS code (texted to Observer) for sendSms, the WhatsApp code (sent on WhatsApp) for sendWhatsapp, either for call — and start_agent FAILS otherwise, so call this BEFORE start_agent for any agent that uses a phone tool. Pass the code the agent sends to (defaults to the user's saved code for that channel) and the channel. This does NOT block and shows the user nothing: it returns {whitelisted:true} if the code is connected, and FAILS if it isn't. On failure, call ask_user_info kind='method' (same channel) to get the user connected, then call check_whitelist again (or just start_agent). Phone numbers are never accepted: if an agent uses one, replace it with the user's code.",
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
          error: `'${raw}' is not an Observer code. The phone tools only send to the user's 4-word code${saved ? ` ('${saved}')` : ''}, never to a phone number: put the code in the agent instead${saved ? '' : " (ask_user_info kind='method' gets it)"}, then check it.`,
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
        error: `Code '${code}' is not connected for ${channel}. Call ask_user_info kind='method' channel='${channel}' so the user can connect it, then check again.`,
      };
    },
  },
  {
    name: 'ask_user_info',
    description: "Ask the user for something only they can give, via a modal. This BLOCKS until they answer. Never ask for these in chat, and never narrate the modal: they can see it. kind='screen' / kind='camera': ONLY after capture_screen / capture_camera failed; the modal's button opens the share picker / camera permission (a click is what browsers like Safari require) and returns the same frame those tools return. kind='method': how the agent tells the user; pass `channel` when the user already named one (e.g. 'send me a WhatsApp' → channel='whatsapp'), omit it to let them pick. It returns {channel, value}: value is what the agent code passes to that channel's tool, already connected and verified (the user's 4-word code for whatsapp/sms/voice/telegram, their email, webhook or Pushover key; '' for browser and system). For browser and system the modal pushes a test notification so the browser / OS asks for permission now and the user confirms they saw it; call it before start_agent for any agent that uses notify() or system_notify(). A returning user with a working connection is not shown anything. kind='model' (`model`: 'onejev' for a decision model, 'default' for an LLM): local or cloud; returns {choice:'local'} (call download_model with that `model` next) or {choice:'cloud'}. Ask it LAST, right before download_model. Omit `model` to ask only local vs cloud before you know which model fits (a Scout on the phone); pick download_model's `model` once you've seen the frame. kind='lock_phone' (mobile app only): the permission lockPhone() needs; call it before start_agent for any agent that uses lockPhone(). Its button opens the Screen Time / \"Display over other apps\" prompt and it returns {authorized:true}; if the permission is already granted the user sees nothing. If the result is {skipped:true}, the user declined: ask them about it in chat instead of calling this again. kind='background' (mobile app only): a button that opens Observer's status window (Picture-in-Picture), which keeps Observer running while the user is in another app; call it before telling the user to leave Observer. Returns {active:true}, at once if the window is already open.",
    parameters: {
      type: 'object',
      properties: {
        kind: {
          type: 'string',
          enum: ['screen', 'camera', 'method', 'model', 'lock_phone', 'background'],
          description: "'screen' / 'camera' for capture permission, 'method' for how to notify the user, 'model' for local vs cloud, 'lock_phone' for the permission lockPhone() needs, 'background' to keep Observer running while the user is in another app (mobile app).",
        },
        channel: {
          type: 'string',
          enum: NOTIFY_METHODS,
          description: "Only for kind='method': the channel the user already asked for. browser → notify(), system → system_notify() (desktop app), whatsapp → sendWhatsapp, sms → sendSms, voice → call, telegram → sendTelegram, email → sendEmail, discord → sendDiscord, pushover → sendPushover. Omit to show the picker.",
        },
        model: {
          type: 'string',
          enum: LOCAL_MODEL_CHOICES,
          description: "Only for kind='model': the local model the download would fetch — 'onejev' for a decision model, 'default' for an LLM. Omit it to ask only local vs cloud, before you know which.",
        },
        reason: {
          type: 'string',
          description: 'A short, friendly one-line explanation shown in the modal (e.g. "so I can watch the GitHub Actions tab we just opened").',
        },
      },
      required: ['kind'],
    },
    multimodal: true,
    execute: async (args, ctx): Promise<ToolResult> => {
      let kind = args.kind as UserInfoRequestKind;
      if (!kind) return { error: 'Provide a `kind` to ask for.' };

      // Non-React host (or a future MCP server transport): no modal to show, so tell the
      // model to fall back to plain conversation rather than silently hanging.
      if (!ctx.requestUserInfo) {
        return { error: 'No interactive UI is available here. Ask the user for this value in chat instead.' };
      }
      if (kind === 'screen' && isDesktop()) {
        return { error: 'The desktop app needs no permission modal: use list_screen_targets, then capture_screen with a target_id.' };
      }

      // The old contact kinds are a method with its channel fixed.
      let method = NOTIFY_METHODS.includes(args.channel) ? (args.channel as NotifyMethod) : undefined;
      if (kind === 'phone') {
        method = args.channel === 'sms' || args.channel === 'voice' ? args.channel : 'whatsapp';
        kind = 'method';
      } else if (kind !== 'screen' && kind !== 'camera' && kind !== 'method' && kind !== 'model' && kind !== 'lock_phone' && kind !== 'background') {
        method = kind;
        kind = 'method';
      }
      if (method === 'system' && !isDesktop()) method = 'browser';

      // The status window: only ask when it isn't open yet. The video is set up now so the
      // modal's tap can open it straight away (PiP needs a user gesture).
      if (kind === 'background') {
        if (!isMobile()) return { error: "kind='background' is only for the mobile app: Observer keeps running on its own here." };
        if (!isStatusPipSupported()) return { error: "This phone can't open Observer's status window. Tell the user to keep Observer open on screen." };
        if (isStatusPipActive()) return { data: { kind, active: true } };
        prepareStatusPip();
      }

      // lockPhone() permission: only ask when there's something to grant.
      if (kind === 'lock_phone') {
        if (!isMobile()) return { error: 'lockPhone() only works in the Observer mobile app (iOS 16+ or Android). Tell the user, and build the agent without it.' };
        const status = await getPhoneLockStatus();
        if (!status.supported) return { error: 'lockPhone() needs iOS 16 or later. Tell the user, and build the agent without it.' };
        if (status.authorized) return { data: { kind, authorized: true } };
      }

      // Nothing to ask: no modal at all (the e2e suite, which can't answer one, relies on this).
      if (kind === 'method' && ((method === 'system' && SensorSettings.getSystemNotifyConfirmed())
        || (method === 'browser' && browserNotifyReady()))) {
        return { data: { kind, channel: method, value: '', use: METHOD_USAGE[method]('') } };
      }

      let response;
      try {
        response = await ctx.requestUserInfo({
          requestId: `info_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          kind,
          method: kind === 'method' ? method : undefined,
          model: kind === 'model' && (args.model === 'onejev' || args.model === 'default') ? args.model : undefined,
          reason: typeof args.reason === 'string' ? args.reason : undefined,
        });
      } catch {
        return { error: 'Cancelled.' };
      }

      // Dismissing the model prompt means "don't download anything": run in the cloud.
      if (kind === 'model') {
        const picked = args.model === 'onejev' || args.model === 'default' ? args.model : null;
        if (response.value !== 'local') return { data: { kind, choice: 'cloud', note: 'Use the cloud model; do not call download_model.' } };
        return picked
          ? { data: { kind, choice: 'local', note: `Call download_model model='${picked}' now, then create_agent with the model_name it returns.` } }
          : { data: { kind, choice: 'local', note: "Once you've seen the frame, call download_model with 'onejev' (decision model) or 'default' (LLM), then create_agent with the model_name it returns." } };
      }

      if (response.skipped) {
        return { data: { kind, skipped: true, note: 'The user dismissed the prompt. Ask them about it in chat; do not call ask_user_info again for this.' } };
      }
      if (kind === 'background') return { data: { kind, active: true } };

      // The modal asked for the permission itself, from its button click.
      if (kind === 'lock_phone') {
        return { data: { kind, authorized: true } };
      }

      // The modal ran the capture itself, from its button click.
      if (kind === 'screen' || kind === 'camera') {
        return response.capture ?? { error: 'No frame was captured.' };
      }

      const channel = response.method;
      if (!channel) return { data: { kind, skipped: true } };
      return { data: { kind, channel, value: response.value, use: METHOD_USAGE[channel](response.value) } };
    },
  },
  {
    name: 'capture_camera',
    description: "Open the camera a $CAMERA agent will use and return one frame from it, so you can SEE what will be watched before building the agent. The camera stays live and start_agent reuses it. Call it BEFORE create_agent for any agent whose system_prompt uses $CAMERA. Without camera_id it opens the user's last-chosen camera (or the default). If the user has more than one camera, named a specific one, or the frame shows the wrong view, call list_cameras and pass camera_id: there is one camera stream, so switching also moves any running $CAMERA agents to it. If it fails (no permission, no camera), call ask_user_info kind='camera'. Pass agent_id (after set_camera_crop) to see the frame CROPPED exactly as that agent model will receive it on every iteration, with the crop pixel size: do this before start_agent to check the crop.",
    parameters: {
      type: 'object',
      properties: {
        camera_id: { type: 'string', description: 'Optional camera id from list_cameras. Omit to use the current camera.' },
        agent_id: { type: 'string', description: 'Return the frame cropped by the set_camera_crop of this agent instead of the full frame.' },
      },
    },
    multimodal: true,
    execute: async (args): Promise<ToolResult> => {
      const result = await captureCamera(typeof args.camera_id === 'string' && args.camera_id ? args.camera_id : undefined);
      return previewAgentCrop(result, 'camera', args.agent_id);
    },
  },
  {
    name: 'list_cameras',
    description: "List the cameras available to a $CAMERA agent, as text (id, label, preferred = the user's saved choice, live = the one open now). Call it when the user mentions a specific camera (\"my webcam\", \"the USB camera\", \"the back camera\") or capture_camera shows the wrong view, then capture_camera with the matching camera_id to switch and check it. Labels are hidden until camera permission is granted; if so, call capture_camera once first.",
    parameters: { type: 'object', properties: {} },
    multimodal: false,
    execute: () => listCameras(),
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
        const code = (await getAgentCode(args.id)) ?? '';
        const blocked = await checkContactCodes(code, ctx.getToken) ?? await checkPhoneLock(code) ?? checkNotifications(code);
        if (blocked) return { error: blocked };
        await startAgentLoop(args.id, ctx.getToken);
        // The agent owns its streams now, so the share ends when it stops.
        releaseCaptureHolds(getRequiredStreamsForPrompt(agent.system_prompt));
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
      // A capture_screen / capture_camera while it ran subscribed the MCP; don't outlive the agent.
      const agent = await getAgent(args.id);
      if (agent) releaseCaptureHolds(getRequiredStreamsForPrompt(agent.system_prompt));
      return { data: { stopped: true, id: args.id } };
    },
  },
  {
    name: 'delete_agent',
    description: "Delete an agent with its code and memory. Refuses a running one: stop_agent first. Use it to clean up your scratchpad agents; never delete the user's agents unless they ask.",
    parameters: {
      type: 'object',
      properties: { id: { type: 'string', description: 'The agent id to delete.' } },
      required: ['id'],
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      if (!(await getAgent(args.id))) return { error: `Agent '${args.id}' not found.` };
      if (isAgentLoopRunning(args.id)) return { error: `Agent '${args.id}' is running: stop_agent first, then delete_agent.` };
      try {
        await deleteAgent(args.id);
        return { data: { deleted: true, id: args.id } };
      } catch (e) {
        return { error: e instanceof Error ? e.message : String(e) };
      }
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
      try {
        const targets = await tauriStreamCapture.getTargets(true);
        const match = targets.find(t => t.id === args.target_id);
        if (!match) {
          return { error: `Target '${args.target_id}' is no longer available (window may have closed). Re-run list_screen_targets.` };
        }
        const raw = match.thumbnail;
        const url = !raw ? undefined : imageDataUrl(raw);
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
    execute: async (args): Promise<ToolResult> => applyCropArgs('screen', args),
  },
  {
    name: 'set_camera_crop',
    description: 'OPTIONAL. Crop a $CAMERA agent\'s capture to a rectangular sub-region of the camera view, so the model only sees (and only spends tokens on) the part that matters — e.g. a doorway or a printer bed. Skip this entirely to watch the whole view (the default). Give the region as box_2d = [ymin, xmin, ymax, xmax] in the SAME normalized 0–1000 coordinate space you use for object detection: top-left origin, y first, every value 0–1000 regardless of the camera\'s real resolution. Read box_2d straight off the frame capture_camera returned (or an iteration image from get_iteration), and do NOT convert to pixels yourself; the crop is stored normalized and resolved against the live frame at capture time. Pass clear:true to remove an existing crop and capture the full view.',
    parameters: {
      type: 'object',
      properties: {
        agent_id: { type: 'string', description: 'The agent whose camera capture to crop.' },
        box_2d: {
          type: 'array',
          items: { type: 'number' },
          minItems: 4,
          maxItems: 4,
          description: 'The crop region as [ymin, xmin, ymax, xmax], normalized to 0–1000 (top-left origin, y first) — the same format you emit for bounding boxes.',
        },
        clear: { type: 'boolean', description: 'If true, remove any existing crop and capture the full view. Ignores box_2d.' },
      },
      required: ['agent_id'],
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => applyCropArgs('camera', args),
  },
  {
    name: 'capture_screen',
    description: 'Start the live screen stream a $SCREEN agent will use and return one frame from it, so you can SEE exactly what will be monitored before building the agent. The frame comes from the real capture pipeline, so it is the image to read a set_screen_crop box_2d from. On the desktop app pass target_id (from list_screen_targets): the stream starts on that screen/window with NO selector popping up; only one screen stream runs at a time, so a different target_id restarts it on that target, unless a running agent is using the screen or system audio (then you get its frame and target_id is not applied: stop_agent first). On web / mobile web (no target_id) this opens the browser screen-share picker (pick a screen, window, or tab). On the mobile app (no target_id) it triggers the OS screen-capture picker and captures the WHOLE screen (iOS broadcast / Android screen-record permission): the user must approve the system prompt, and on iOS there can be a few seconds of delay before the first frame — if this returns a "no frame yet" message, just call it again. The stream stays live — start_agent reuses it without prompting again. Call it BEFORE create_agent for any agent whose system_prompt uses $SCREEN. Pass agent_id (after set_screen_crop) to see the frame CROPPED exactly as that agent model will receive it on every iteration, with the crop pixel size: do this before every start_agent to check the crop (the whole status text inside it with a margin, the label included, nothing misleading). The stream must already be live from an earlier capture_screen.',
    parameters: {
      type: 'object',
      properties: {
        target_id: { type: 'string', description: 'Desktop app only (required there): the id of the target to capture, from list_screen_targets. Omit on web / mobile.' },
        agent_id: { type: 'string', description: 'Return the frame cropped by the set_screen_crop of this agent instead of the full frame. The target is already seated, so target_id is not needed.' },
      },
    },
    multimodal: true,
    // Same platform split as StreamManager: browser → getDisplayMedia; Tauri → native plugin
    // (desktop seats the target up front so no selector opens).
    execute: async (args): Promise<ToolResult> => {
      const result = isDesktop() ? await captureScreenDesktop(args.target_id) : await captureScreen();
      return previewAgentCrop(result, 'screen', args.agent_id);
    },
  },
  {
    name: 'download_model',
    description: 'Download and load an on-device model so agents can run locally with NO cloud and NO API key. `model`: "onejev" (OneJev 0.8B, the decision model for dial watchers: one cropped percentage, progress bar, timer or counter; returns `decision: true`) or "default" (Qwen3.5 0.8B, a general LLM for everything else). Observer picks a transformers.js ONNX build in the browser, a llama.cpp GGUF in the desktop app. This BLOCKS while it downloads (about 1 GB) and loads; progress bars are shown to the user. When it resolves, the returned `model_name` is immediately usable as a `create_agent` model_name. If the download fails it returns `fallback: "cloud"` with the cloud model_name instead: use that. Only one on-device model is loaded at a time, so loading one unloads the other.',
    parameters: {
      type: 'object',
      properties: {
        model: { type: 'string', enum: LOCAL_MODEL_CHOICES, description: '"onejev" for dial watchers (a percentage, progress bar, timer or counter), "default" for everything else. Defaults to "default".' },
      },
    },
    multimodal: false,
    execute: async (args): Promise<ToolResult> => {
      const choice: LocalModelChoice = args.model ?? 'default';
      if (!LOCAL_MODEL_CHOICES.includes(choice)) return { error: `Unknown model '${args.model}'. Use one of: ${LOCAL_MODEL_CHOICES.join(', ')}.` };
      try {
        const result = await downloadLocalModel(choice);
        return { data: result };
      } catch (e) {
        // A failed download falls back to the cloud counterpart instead of stalling the build.
        const cloud = choice === 'onejev' ? 'clef-flash' : 'gemma-3-12b-it';
        return { data: {
          model_name: cloud,
          fallback: 'cloud',
          download_error: e instanceof Error ? e.message : String(e),
          note: `The local download failed, so use the cloud model '${cloud}' instead. Tell the user in one line when you're done.`,
        } };
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
