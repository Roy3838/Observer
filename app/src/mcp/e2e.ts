// src/mcp/e2e.ts
//
// Test hooks for the MCP end-to-end suite (app/e2e). Only loaded when the app is served with
// VITE_E2E=1: MCPContext imports this module dynamically behind that flag, so normal builds
// never include it.
//
// The suite is black-box. It sends the MCP a prompt and judges only what the agents it built
// did: WHEN they notified the user (mcp.spec.ts), or what they wrote to memory and when they
// marked clips (patterns.spec.ts), never which model or tools the MCP chose. Every agent-code tool already
// logs `{ logType: 'tool-success' | 'tool-error', content: { tool } }` (handlers/javascript.ts),
// so one Logger listener sees every notification on every channel. Failed sends count too:
// the suite's contact codes are fake, and what matters is that the agent tried, and when.

import { Logger } from '@utils/logging';
import { getRunningAgentIds, stopAgentLoop, type TokenProvider } from '@utils/main_loop';
import { listAgents, deleteAgent, getAgentCode, getAgentMemory, getAllMemories, deleteMemory } from '@utils/agent_database';
import { IterationStore } from '@utils/IterationStore';
import { ModelManager } from '@utils/ModelManager';
import { listConversations, loadConversation } from './conversationStore';
import type { WireMessage } from './types';
import type { UseMCPReturn } from './useMCP';

// Agent-code tools that reach the user. Memory, clips, timing and agent control don't count.
const NOTIFY_TOOLS = new Set([
  'ask', 'call', 'message', 'notify', 'overlay', 'sound', 'system_notify',
  'sendDiscord', 'sendEmail', 'sendGotify', 'sendPushover', 'sendSms', 'sendTelegram', 'sendWhatsapp',
]);

// Agent-code tools that leave a record instead of reaching the user: the pattern scenarios
// (patterns.spec.ts) score what agents wrote to memory and when they marked clips.
const MEMORY_TOOLS = new Set(['setMemory', 'appendMemory']);
const CLIP_TOOLS = new Set(['startClip', 'stopClip', 'markClip', 'getVideo']);

export type E2EEvent =
  | { type: 'notification'; agentId: string; tool: string; ok: boolean; params: any }
  | { type: 'ask_user_info'; message: string }
  /** A memory write, with the target's whole memory right after it. */
  | { type: 'memory'; agentId: string; tool: string; ok: boolean; targetId: string; memory: string }
  | { type: 'clip'; agentId: string; tool: string; ok: boolean; params: any };

export interface E2EHandles {
  mcp: () => UseMCPReturn & { modelName: string };
  getToken: () => TokenProvider;
}

declare global {
  interface Window {
    /** Exposed by Playwright; receives every event the suite times. */
    __e2eReport?: (event: E2EEvent) => void;
    __observerE2E?: ReturnType<typeof createApi>;
  }
}

/** Text-only view of the conversation for the report: tool calls with their args, no images. */
function transcript(messages: WireMessage[]) {
  return messages.map(m => ({
    role: m.role,
    text: typeof m.content === 'string'
      ? m.content
      : Array.isArray(m.content)
        ? m.content.filter((p: any) => p.type === 'text').map((p: any) => p.text).join('\n')
        : '',
    toolCalls: m.tool_calls?.map(c => ({ name: c.function.name, args: c.function.arguments })),
  }));
}

// The provider can re-mount (it does after login), so the hooks always go through the
// latest mounted one: an unmounted provider's send() still runs, but its state never updates.
let handles: E2EHandles;
let listening = false;
// How many times a provider mounted since the page loaded. Above 1, the provider that ran the
// last send() may not be the one the hooks now reach.
let mounts = 0;

/** What each agent was built as, and what it did on every iteration. `lastImage` is the
 *  last image an agent's model saw, to check its crop. */
async function agentReport(since = 0) {
  const agents = await Promise.all((await listAgents()).map(async agent => {
    // stopAgent() moves the session's iterations to IndexedDB: read them back from there.
    const saved = (await IterationStore.getHistoricalSessions(agent.id)).find(s => new Date(s.startTime).getTime() >= since);
    const live = IterationStore.getIterationsForAgent(agent.id);
    const iterations = live.length ? live : saved?.iterations ?? [];
    return {
      id: agent.id,
      model_name: agent.model_name,
      loop_interval_seconds: agent.loop_interval_seconds,
      only_on_significant_change: agent.only_on_significant_change,
      system_prompt: agent.system_prompt,
      code: await getAgentCode(agent.id),
      memory: await getAgentMemory(agent.id),
      iterations: iterations.map(it => ({
        n: it.sessionIterationNumber,
        startTime: it.startTime,
        durationMs: it.duration,
        response: it.modelResponse?.slice(0, 400),
        tools: it.tools.map(t => `${t.name}:${t.status}${t.error ? ` (${t.error})` : ''}`),
        hasError: it.hasError,
        skipped: it.isSkipped ? it.skipReason : undefined,
        image: it.modelImages?.[0],
        prompt: it.modelPrompt,
      })),
      lastImage: [...iterations].reverse().find(it => it.modelImages?.length)?.modelImages?.[0],
    };
  }));
  return agents;
}

function createApi(h: () => E2EHandles) {
  return {
    isAuthenticated: async () => !!(await h().getToken()().catch(() => undefined)),
    /** One user turn. Resolves when the MCP's run settles, with its final text. */
    send: async (prompt: string) => (await h().mcp().send(prompt))?.text ?? '',
    stop: () => h().mcp().stop(),
    runningAgentIds: () => getRunningAgentIds(),
    /** The live conversation; if it's empty (the provider re-mounted mid-run, so its state is
     *  fresh), the most recently saved one, if it was saved after `since`. */
    transcript: (since = 0) => {
      const live = h().mcp().messages;
      const latest = listConversations()[0];
      const saved = !live.length && latest && latest.updatedAt >= since ? loadConversation(latest.id) : null;
      return {
        modelName: h().mcp().modelName,
        source: live.length ? 'live' : saved ? 'saved' : 'none',
        mounts,
        messages: transcript(live.length ? live : saved?.messages ?? []),
      };
    },
    agentReport,
    /** Send the same prompt + image to a model `n` times: is the answer the same every time? */
    replay: async (modelName: string, modifiedPrompt: string, image: string, n: number) => {
      const token = await h().getToken()().catch(() => undefined);
      const out: (number | string)[] = [];
      for (let i = 0; i < n; i++) {
        const r = await ModelManager.getInstance().sendPrompt(modelName, { modifiedPrompt, images: [image] } as any, token);
        out.push((r.decision as any)?.noul ?? 'n/a');
      }
      return out;
    },
    /** Every memory slot, including ones written for an id that isn't an agent. */
    memories: () => getAllMemories(),
    /** Stop and delete every agent so the next scenario starts from an empty Observer.
     *  The suite reloads the page afterwards for a fresh conversation and stream state. */
    reset: async () => {
      h().mcp().stop();
      await Promise.all(getRunningAgentIds().map(id => stopAgentLoop(id)));
      for (const agent of await listAgents()) await deleteAgent(agent.id);
      for (const { id } of await getAllMemories()) await deleteMemory(id);
    },
  };
}

export function installE2EHooks(h: E2EHandles): void {
  handles = h; // called on every provider mount; the latest one wins
  mounts++;
  if (listening) return;
  listening = true;
  window.__observerE2E = createApi(() => handles);
  Logger.addListener(entry => {
    const d = entry.details;
    if ((d?.logType === 'tool-success' || d?.logType === 'tool-error') && NOTIFY_TOOLS.has(d.content?.tool)) {
      window.__e2eReport?.({
        type: 'notification',
        agentId: entry.source,
        tool: d.content.tool,
        ok: d.logType === 'tool-success',
        params: d.content.params,
      });
    }
    if ((d?.logType === 'tool-success' || d?.logType === 'tool-error') && MEMORY_TOOLS.has(d.content?.tool)) {
      const targetId: string = d.content.params?.targetId ?? entry.source;
      getAgentMemory(targetId).catch(() => '').then(memory => window.__e2eReport?.({
        type: 'memory', agentId: entry.source, tool: d.content.tool, ok: d.logType === 'tool-success', targetId, memory,
      }));
    }
    if ((d?.logType === 'tool-success' || d?.logType === 'tool-error') && CLIP_TOOLS.has(d.content?.tool)) {
      window.__e2eReport?.({
        type: 'clip', agentId: entry.source, tool: d.content.tool, ok: d.logType === 'tool-success', params: d.content.params,
      });
    }
    // Nobody will fill the modal, so the run would hang: report it and unwind the run.
    if (entry.source === 'MCP' && entry.message.startsWith('Awaiting user info')) {
      window.__e2eReport?.({ type: 'ask_user_info', message: entry.message });
      setTimeout(() => handles.mcp().stop(), 0);
    }
  });
  Logger.info('E2E', 'MCP end-to-end hooks installed');
}
