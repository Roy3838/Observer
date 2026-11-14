// src/components/Observer/StatusPipBridge.tsx
//
// Feeds the mobile status window (utils/statusPip.ts) the same state the agent cards render, so
// the window keeps no registry of its own:
//   - which agents, and their names: App's agents (the agent database) and running set
//   - each agent's status, last word and countdown: AgentLiveStateProvider
//   - what the MCP is doing: the MCP context (isRunning, toolStatus, messages)
// Mounted inside ObserverTab's LiveAgentTracking, which tracks every running agent. Renders
// nothing; it hands statusPip a fresh snapshot on every change.

import React, { useEffect, useRef } from 'react';
import type { CompleteAgent } from '@utils/agent_database';
import { setStatusSnapshot } from '@utils/statusPip';
import { useMCPContext } from '../../mcp/MCPContext';
import { parseAgentPrompt } from '../../mcp/agentWake';
import type { ToolStatusEntry } from '../../mcp/useMCP';
import type { WireMessage } from '../../mcp/types';
import { useAgentLiveStates } from './AgentLiveStateContext';

const textOf = (m: WireMessage | undefined): string =>
  !m ? '' : typeof m.content === 'string' ? m.content
    : Array.isArray(m.content) ? (m.content.find((p: any) => p?.type === 'text')?.text ?? '') : '';

/** The MCP's step, as the window says it. */
function describeTool(name: string, args: Record<string, any>, nameOf: (id: string) => string): string {
  const agent = () => nameOf(String(args.agent_id ?? args.id ?? ''));
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

function mcpActivity(toolStatus: Map<string, ToolStatusEntry>, streamingText: string, nameOf: (id: string) => string): string {
  const running = [...toolStatus.values()].filter(t => t.status === 'running');
  const latest = running[running.length - 1];
  if (latest) {
    let args = latest.args;
    if (typeof args === 'string') { try { args = JSON.parse(args); } catch { args = {}; } }
    return describeTool(latest.name, args ?? {}, nameOf);
  }
  return streamingText ? 'Writing…' : 'Thinking…';
}

const StatusPipBridge: React.FC<{
  agents: CompleteAgent[];
  runningAgents: Set<string>;
  startingAgents: Set<string>;
}> = ({ agents, runningAgents, startingAgents }) => {
  const live = useAgentLiveStates();
  const { isRunning, toolStatus, messages, streamingText } = useMCPContext();

  // When a turn ends, its answer stays up in the window for a little while.
  const reply = useRef({ text: '', endedAt: 0 });
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !isRunning) {
      const lastUser = messages.map(m => m.role).lastIndexOf('user');
      const answer = [...messages.slice(lastUser + 1)].reverse().find(m => m.role === 'assistant' && textOf(m).trim());
      reply.current = { text: textOf(answer), endedAt: Date.now() };
    }
    wasRunning.current = isRunning;
  }, [isRunning, messages]);

  useEffect(() => {
    const names = new Map(agents.map(a => [a.id, a.name]));
    const nameOf = (id: string) => names.get(id) ?? id;
    const lastUser = [...messages].reverse().find(m => m.role === 'user');

    const ids = [...new Set([...runningAgents, ...startingAgents])];
    setStatusSnapshot({
      mcp: {
        running: isRunning,
        activity: isRunning ? mcpActivity(toolStatus, streamingText, nameOf) : '',
        wokenBy: parseAgentPrompt(textOf(lastUser))?.agentName ?? null,
        lastReply: reply.current.text,
        endedAt: reply.current.endedAt,
      },
      agents: ids.map(id => {
        const s = live[id];
        const remainingMs = !s ? null
          : s.isSleeping ? s.sleepRemainingMs
          : s.durationMs ? s.durationMs * (1 - s.progress / 100) : null;
        return {
          id,
          name: nameOf(id),
          status: !runningAgents.has(id) ? 'STARTING' : s?.liveStatus ?? 'STARTING',
          lastWord: s?.lastWord ?? '',
          progress: s?.progress ?? 0,
          remainingMs,
          isSleeping: s?.isSleeping ?? false,
          isOverrun: s?.isOverrun ?? false,
        };
      }),
    });
  });

  return null;
};

export default StatusPipBridge;
