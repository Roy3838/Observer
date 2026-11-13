// src/utils/statusBoard.ts
//
// What Observer is doing right now, in one place, for the mobile status PiP (statusPip.ts):
// the MCP's current step and every running agent (status, last word, countdown, what it saw).
//
// No list of agents of its own: which agents run comes from main_loop (getRunningAgentIds, read
// on every call), their names from the agent database. What's kept here is only what neither
// has: each agent's live state (status, last word, countdown, last image), taken from the Logger
// so it doesn't depend on any card being mounted. The MCP has no log for "which step am I on", so
// useMCP reports that with the mcp* methods below.
//
// The last-word rule mirrors AgentLiveStateContext: the last word of the model's answer, which
// for a decision model is its noul value ("noul 0.943" → "0.943").

import { Logger, type LogEntry } from './logging';
import { getAgent } from './agent_database';
import { getRunningAgentIds } from './main_loop';
import { imageDataUrl } from './imageDataUrl';

export type AgentStatus = 'STARTING' | 'CAPTURING' | 'THINKING' | 'RESPONDING' | 'WAITING' | 'SKIPPED' | 'SLEEPING';

export interface AgentRow {
  id: string;
  name: string;
  status: AgentStatus;
  lastWord: string;
  iterationStart: number;
  intervalMs: number;
  sleepUntil: number | null;
  /** The image the model got on the last iteration (already cropped), decoded for drawing. */
  thumb: HTMLImageElement | null;
}

export interface McpStatus {
  running: boolean;
  /** The step in progress: "Thinking…", "Cropping Finish Line"… */
  activity: string;
  /** Set when an agent's mcp() started this turn. */
  wokenBy: string | null;
  /** The MCP's last answer and when it finished, shown for a little while after a turn. */
  lastReply: string;
  endedAt: number;
}


const lastWordOf = (text: string): string => {
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.length ? words[words.length - 1] : '';
};

type LiveState = Omit<AgentRow, 'id' | 'name'>;

const FRESH: LiveState = { status: 'STARTING', lastWord: '', iterationStart: 0, intervalMs: 0, sleepUntil: null, thumb: null };

// Log types only agents write: any of them means the log's source is an agent id.
const AGENT_LOGS = new Set(['iteration-start', 'model-prompt', 'stream-start', 'stream-chunk', 'model-response', 'iteration-skipped', 'agent-sleep-start', 'agent-sleep-end']);

class StatusBoard {
  mcp: McpStatus = { running: false, activity: '', wokenBy: null, lastReply: '', endedAt: 0 };

  private live = new Map<string, LiveState>();
  /** Names from the agent database, plus agents the MCP is creating right now. */
  private names = new Map<string, string>();
  private streamBuffers = new Map<string, string>();
  /** The MCP's tool calls in flight, in start order. */
  private mcpTools = new Map<string, string>();

  constructor() {
    Logger.addListener(this.onLog);
  }

  /** Every running agent, in start order, with its live state. */
  runningAgents(): AgentRow[] {
    const ids = getRunningAgentIds();
    for (const id of this.live.keys()) if (!ids.includes(id)) this.live.delete(id);
    return ids.map(id => ({ id, name: this.nameOf(id), ...(this.live.get(id) ?? FRESH) }));
  }

  nameOf(id: string): string {
    const name = this.names.get(id);
    if (name) return name;
    if (id) {
      this.names.set(id, id); // until the database answers
      getAgent(id).then(agent => { if (agent) this.names.set(id, agent.name); }).catch(() => {});
    }
    return id;
  }

  /** Seconds until the agent's next iteration (or the end of its sleep). */
  remainingMs(row: AgentRow, now = Date.now()): number | null {
    if (row.sleepUntil) return Math.max(0, row.sleepUntil - now);
    if (!row.iterationStart || !row.intervalMs) return null;
    // main_loop ticks on a fixed grid from the first start.
    return row.intervalMs - ((now - row.iterationStart) % row.intervalMs);
  }

  // ---- MCP ----------------------------------------------------------------

  mcpStart(wokenBy: string | null): void {
    this.mcpTools.clear();
    this.mcp = { ...this.mcp, running: true, activity: 'Thinking…', wokenBy };
  }

  mcpToolStart(callId: string, name: string, args: unknown): void {
    const parsed = parseArgs(args);
    if ((name === 'create_agent' || name === 'edit_agent') && parsed.id && parsed.name) {
      this.names.set(String(parsed.id), String(parsed.name));
    }
    this.mcpTools.set(callId, this.describeTool(name, parsed));
    this.mcp = { ...this.mcp, activity: this.mcpTools.get(callId)! };
  }

  mcpToolEnd(callId: string): void {
    this.mcpTools.delete(callId);
    const running = [...this.mcpTools.values()];
    this.mcp = { ...this.mcp, activity: running.length ? running[running.length - 1] : 'Thinking…' };
  }

  mcpWriting(): void {
    if (this.mcpTools.size === 0) this.mcp = { ...this.mcp, activity: 'Writing…' };
  }

  mcpEnd(lastReply: string): void {
    this.mcpTools.clear();
    this.mcp = { running: false, activity: '', wokenBy: this.mcp.wokenBy, lastReply, endedAt: Date.now() };
  }

  private describeTool(name: string, args: Record<string, any>): string {
    const agent = () => this.nameOf(String(args.agent_id ?? args.id ?? ''));
    switch (name) {
      case 'list_agents': case 'get_agent': case 'get_status': case 'list_models': return 'Reading your agents';
      case 'create_agent': return `Creating ${args.name ?? 'an agent'}`;
      case 'edit_agent': return `Editing ${args.name ?? agent()}`;
      case 'start_agent': return `Starting ${agent()}`;
      case 'stop_agent': return `Stopping ${agent()}`;
      case 'delete_agent': return `Removing ${agent()}`;
      case 'get_runs': return `Reading ${agent()}'s history`;
      case 'get_iteration': return `Checking ${agent()}`;
      case 'list_screen_targets': case 'see_screen_target': return 'Finding the right window';
      case 'capture_screen': return args.agent_id ? `Checking what ${agent()} sees` : 'Looking at your screen';
      case 'capture_camera': return args.agent_id ? `Checking what ${agent()} sees` : 'Looking at your camera';
      case 'set_screen_crop': case 'set_camera_crop': return `Cropping ${agent()}`;
      case 'ask_user_info': return 'Waiting for you';
      case 'download_model': return 'Downloading a model';
      default: return 'Working';
    }
  }

  // ---- Agents -------------------------------------------------------------

  private onLog = (log: LogEntry): void => {
    const type = log.details?.logType;
    if (type === 'agent-status-changed') {
      // A fresh start (or a stop) forgets the last run's state.
      const id = log.details.content?.agentId;
      if (id) { this.live.delete(id); this.streamBuffers.delete(id); }
      return;
    }
    if (!type || !AGENT_LOGS.has(type)) return;

    const id = log.source;
    let row = this.live.get(id);
    if (!row) { row = { ...FRESH }; this.live.set(id, row); }

    switch (type) {
      case 'iteration-start':
        row.status = 'CAPTURING';
        row.sleepUntil = null;
        row.iterationStart = log.details.content?.iterationStartTime ?? Date.now();
        row.intervalMs = log.details.content?.intervalMs ?? row.intervalMs;
        break;
      case 'model-prompt': {
        row.status = 'THINKING';
        // What the agent sees, exactly as the model gets it. Decoded off to the side so the
        // old thumbnail stays up until the new one is ready.
        const first = log.details.content?.images?.[0];
        if (typeof first === 'string' && first) {
          const img = new Image();
          img.src = imageDataUrl(first);
          const target = row;
          img.decode().then(() => { target.thumb = img; }).catch(() => {});
        }
        break;
      }
      case 'stream-start':
        row.status = 'RESPONDING';
        this.streamBuffers.set(id, '');
        break;
      case 'stream-chunk': {
        const buffer = (this.streamBuffers.get(id) ?? '') + (log.details.content?.chunk ?? '');
        this.streamBuffers.set(id, buffer);
        const word = lastWordOf(buffer);
        if (word) row.lastWord = word;
        break;
      }
      case 'model-response': {
        row.status = 'WAITING';
        const word = lastWordOf(typeof log.details.content === 'string' ? log.details.content : '');
        if (word) row.lastWord = word;
        break;
      }
      case 'iteration-skipped':
        row.status = 'SKIPPED';
        break;
      case 'agent-sleep-start':
        row.status = 'SLEEPING';
        row.sleepUntil = Date.now() + (log.details.content?.durationMs ?? 0);
        break;
      case 'agent-sleep-end':
        row.status = 'WAITING';
        row.sleepUntil = null;
        break;
    }
  };
}

function parseArgs(args: unknown): Record<string, any> {
  if (typeof args === 'string') {
    try { return JSON.parse(args) ?? {}; } catch { return {}; }
  }
  return args && typeof args === 'object' ? (args as Record<string, any>) : {};
}

export const statusBoard = new StatusBoard();
