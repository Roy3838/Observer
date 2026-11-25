// src/components/Observer/AgentsFooter.tsx
//
// The Observer tab's status footer: one pill under the composer that says how many
// micro-agents exist and which are running, and opens a list to start/stop them. It reads
// App's agent list and running set, so it's right no matter how an agent was started (this
// chat, another chat, the Micro Agents tab, the user's phone). Live detail (countdown, last
// word, video) belongs to the agent's card in the chat (AgentChatCard), not here.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUpRight, ChevronUp, Loader2, Play, Square } from 'lucide-react';
import type { CompleteAgent } from '@utils/agent_database';

interface AgentsFooterProps {
  agents: CompleteAgent[];
  runningAgents: Set<string>;
  startingAgents: Set<string>;
  onToggle: (agentId: string, isCurrentlyRunning: boolean) => void;
  onOpenMicroAgents?: () => void;
}

const StatusDot: React.FC<{ running: boolean }> = ({ running }) => (
  <span className="relative flex h-2 w-2 flex-shrink-0">
    {running && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-60" />}
    <span className={`relative inline-flex h-2 w-2 rounded-full ${running ? 'bg-green-500' : 'border border-gray-400'}`} />
  </span>
);

const AgentsFooter: React.FC<AgentsFooterProps> = ({ agents, runningAgents, startingAgents, onToggle, onOpenMicroAgents }) => {
  const [isOpen, setIsOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const running = useMemo(() => agents.filter(a => runningAgents.has(a.id)), [agents, runningAgents]);

  // Running first, as of when the list was opened: re-sorting on every start/stop would move
  // another agent's button under the finger that just tapped.
  const [liveAtOpen, setLiveAtOpen] = useState<Set<string>>(new Set());
  const rows = useMemo(
    () => [...agents.filter(a => liveAtOpen.has(a.id)), ...agents.filter(a => !liveAtOpen.has(a.id))],
    [agents, liveAtOpen]
  );
  const toggleOpen = () => {
    if (!isOpen) setLiveAtOpen(new Set([...runningAgents, ...startingAgents]));
    setIsOpen(!isOpen);
  };

  useEffect(() => {
    if (!isOpen) return;
    const onPointerDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setIsOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => { if (e.key === 'Escape') setIsOpen(false); };
    window.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [isOpen]);

  if (agents.length === 0) return null;

  const label = running.length === 1
    ? running[0].name
    : running.length > 1
      ? `${running.length} running`
      : `${agents.length} micro-agent${agents.length === 1 ? '' : 's'}`;

  return (
    <div ref={rootRef} className="relative z-30 flex-shrink-0 flex justify-center px-3 pb-1">
      {isOpen && (
        <div
          role="region"
          aria-label="Micro-agents"
          className="absolute bottom-full mb-2 w-[calc(100%-1.5rem)] max-w-md flex flex-col max-h-[min(50vh,22rem)] bg-white border border-gray-200 rounded-2xl shadow-xl overflow-hidden animate-sheet-in"
        >
          <div className="flex items-center justify-between px-4 pt-3 pb-1.5 flex-shrink-0">
            <span className="text-xs font-medium text-gray-500">Micro-agents</span>
            {onOpenMicroAgents && (
              <button
                type="button"
                onClick={() => { setIsOpen(false); onOpenMicroAgents(); }}
                className="inline-flex items-center gap-0.5 text-xs text-gray-500 hover:text-gray-800 transition-colors"
              >
                View all <ArrowUpRight className="w-3 h-3" />
              </button>
            )}
          </div>
          <ul className="overflow-y-auto overscroll-contain px-1.5 pb-1.5">
            {rows.map(agent => {
              const isRunning = runningAgents.has(agent.id);
              const isStarting = !isRunning && startingAgents.has(agent.id);
              return (
                <li key={agent.id} className="flex items-center gap-3 pl-2.5 pr-1 py-1 rounded-xl">
                  <StatusDot running={isRunning} />
                  <span className={`flex-1 min-w-0 truncate text-sm ${isRunning ? 'font-medium text-gray-800' : 'text-gray-600'}`}>{agent.name}</span>
                  <span className={`text-xs flex-shrink-0 ${isRunning ? 'text-green-600' : 'text-gray-500'}`}>
                    {isRunning ? 'Running' : isStarting ? 'Starting…' : 'Stopped'}
                  </span>
                  <button
                    type="button"
                    onClick={() => onToggle(agent.id, isRunning)}
                    disabled={isStarting}
                    aria-label={`${isRunning ? 'Stop' : 'Start'} ${agent.name}`}
                    className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 text-gray-500 hover:bg-gray-100 hover:text-gray-800 disabled:opacity-60 transition-colors"
                  >
                    {isStarting
                      ? <Loader2 className="w-4 h-4 animate-spin" />
                      : isRunning
                        ? <Square className="w-3 h-3" fill="currentColor" />
                        : <Play className="w-3.5 h-3.5" fill="currentColor" />}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <button
        type="button"
        onClick={toggleOpen}
        aria-expanded={isOpen}
        className="inline-flex items-center gap-2 max-w-full h-8 px-3 rounded-full border border-gray-200 bg-white text-xs text-gray-600 hover:bg-gray-100 transition-colors"
      >
        <StatusDot running={running.length > 0} />
        <span className="truncate">{label}</span>
        {running.length > 0 && agents.length > running.length && (
          <span className="text-gray-500 flex-shrink-0">· {agents.length - running.length} stopped</span>
        )}
        <ChevronUp className={`w-3.5 h-3.5 text-gray-500 flex-shrink-0 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>
    </div>
  );
};

export default AgentsFooter;
