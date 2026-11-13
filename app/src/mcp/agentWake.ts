// src/mcp/agentWake.ts
//
// mcp(message) in agent code: an agent wakes the MCP with a message, as a new turn in whatever
// MCP conversation is open. The MCPProvider registers the inbox on mount; with no provider
// mounted, mcp() throws so the agent's run log says why nothing happened.
//
// Unlike remote messages (useRemoteControl), an agent's message never interrupts a run: it
// waits until the MCP is idle, so a Scout that fires while the MCP is still checking it doesn't
// cut that turn short. The MCP answers in the chat; reaching the user is up to the MCP (an agent
// it builds, or a Scratchpad).

import { useEffect, useRef, useState } from 'react';
import { Logger } from '@utils/logging';
import type { UseMCPReturn } from './useMCP';

export interface AgentWake {
  agentId: string;
  agentName: string;
  message: string;
}

let inbox: ((wake: AgentWake) => void) | null = null;

/** Queue a message for the MCP. Throws when no MCP is mounted to receive it. */
export function wakeMCP(wake: AgentWake): void {
  if (!inbox) throw new Error('No MCP is open in Observer to receive this message.');
  inbox(wake);
}

/** What the model sees for an agent's message: who sent it, and that the user may be away. */
export function agentPrompt(wake: AgentWake): string {
  return (
    `[Sent by your agent "${wake.agentName}" (id: ${wake.agentId}), not by the user. They may be away from ` +
    `this chat: don't call ask_user_info (nobody is there to answer it), and reach them through an agent if you need to.]\n\n` +
    wake.message
  );
}

const AGENT_PREFIX = /^\[Sent by your agent "([^"]*)" \(id: [^)]*\)[^\]]*\]\n\n/;

/** Splits an agentPrompt back into the agent's name and its message; null for other messages. */
export function parseAgentPrompt(text: string): { agentName: string; text: string } | null {
  const match = AGENT_PREFIX.exec(text);
  if (!match) return null;
  return { agentName: match[1], text: text.slice(match[0].length) };
}

/** Mounted once, by MCPProvider: receives agents' messages and sends them one at a time when idle. */
export function useAgentWake(mcp: Pick<UseMCPReturn, 'send' | 'isRunning'>): void {
  const queue = useRef<AgentWake[]>([]);
  const draining = useRef(false);
  const [wake, setWake] = useState(0);

  useEffect(() => {
    inbox = w => {
      Logger.info('MCP', `Woken by agent ${w.agentId}`);
      queue.current.push(w);
      setWake(n => n + 1);
    };
    return () => { inbox = null; };
  }, []);

  useEffect(() => {
    if (draining.current || mcp.isRunning || queue.current.length === 0) return;
    const next = queue.current.shift()!;
    draining.current = true;
    mcp.send(agentPrompt(next))
      .catch(error => Logger.error('MCP', `Agent wake failed: ${error instanceof Error ? error.message : String(error)}`))
      .finally(() => {
        draining.current = false;
        setWake(n => n + 1);
      });
  }, [mcp.isRunning, mcp.send, wake]);
}
