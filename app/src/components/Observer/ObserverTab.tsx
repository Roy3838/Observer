// src/components/Observer/ObserverTab.tsx
//
// Top-level "Observer" tab — the app's default landing view. A minimal ChatGPT/Claude
// Cowork-style splash (ObserverHero) until the first message is sent, then docks into a
// normal scrolling transcript with the input pinned to the bottom (MCP, unboxed). A strip
// of currently running agents overlays both states so a user never has to open the My
// Agents grid just to see what's active.
//
// The My Agents tab (grid, GetStarted, MCPPanel, minimize system) is untouched by this file.

import React, { useMemo } from 'react';
import type { TokenProvider } from '@utils/main_loop';
import type { CompleteAgent } from '@utils/agent_database';
import { useMCPContext } from '../../mcp/MCPContext';
import MCP from '@components/AICreator/MCP';
import ObserverHero from './ObserverHero';
import RunningAgentsStrip from './RunningAgentsStrip';
import TutorialDemoBanner from './TutorialDemoBanner';
import { FloatingAgentsProvider, useFloatingAgents } from './FloatingAgentsContext';
import { AgentLiveStateProvider } from './AgentLiveStateContext';

// Tracks live status/progress/last-word for every agent id that currently needs it —
// running, starting, or popped out into the floating overlay (a floating card still needs
// live data even if its agent isn't running). Must live inside FloatingAgentsProvider since
// it reads the floating id set from it.
const LiveAgentTracking: React.FC<{
  runningAgents: Set<string>;
  startingAgents: Set<string>;
  children: React.ReactNode;
}> = ({ runningAgents, startingAgents, children }) => {
  const { floating } = useFloatingAgents();
  const agentIds = useMemo(
    () => Array.from(new Set([...runningAgents, ...startingAgents, ...Object.keys(floating)])),
    [runningAgents, startingAgents, floating]
  );
  return (
    <AgentLiveStateProvider agentIds={agentIds} runningAgents={runningAgents} startingAgents={startingAgents}>
      {children}
    </AgentLiveStateProvider>
  );
};

interface ObserverTabProps {
  getToken: TokenProvider;
  isAuthenticated: boolean;
  isUsingObServer: boolean;
  onSignIn?: () => void;
  onSwitchToObServer?: () => void;
  onUpgrade?: () => void;
  onRefresh?: () => void;
  agents: CompleteAgent[];
  runningAgents: Set<string>;
  startingAgents: Set<string>;
  onToggleAgent: (agentId: string, isCurrentlyRunning: boolean) => void;
  onOpenMicroAgents?: () => void;
  onShowTutorial?: () => void;
}

const ObserverTab: React.FC<ObserverTabProps> = ({
  getToken,
  isAuthenticated,
  isUsingObServer,
  onSignIn,
  onSwitchToObServer,
  onUpgrade,
  onRefresh,
  agents,
  runningAgents,
  startingAgents,
  onToggleAgent,
  onOpenMicroAgents,
  onShowTutorial,
}) => {
  const { messages, isRunning } = useMCPContext();
  const isEmpty = messages.length === 0 && !isRunning;

  return (
    <FloatingAgentsProvider>
      <LiveAgentTracking runningAgents={runningAgents} startingAgents={startingAgents}>
        <div className="flex flex-col h-full">
          {/* Portaled to document.body and viewport-fixed — floats over the whole app
              (including the ProgressBar) without participating in this flex layout. */}
          <RunningAgentsStrip
            agents={agents}
            runningAgents={runningAgents}
            startingAgents={startingAgents}
            onToggle={onToggleAgent}
            onSelectAgent={onOpenMicroAgents}
          />
          <TutorialDemoBanner />
          {isEmpty ? (
            <ObserverHero onShowTutorial={onShowTutorial} />
          ) : (
            <div className="flex-1 min-h-0">
              <MCP
                boxed={false}
                heightClass="h-full"
                getToken={getToken}
                isAuthenticated={isAuthenticated}
                isUsingObServer={isUsingObServer}
                onSignIn={onSignIn}
                onSwitchToObServer={onSwitchToObServer}
                onUpgrade={onUpgrade}
                onRefresh={onRefresh}
                agents={agents}
                runningAgents={runningAgents}
                startingAgents={startingAgents}
                onToggleAgent={onToggleAgent}
                onSelectAgent={onOpenMicroAgents}
              />
            </div>
          )}
        </div>
      </LiveAgentTracking>
    </FloatingAgentsProvider>
  );
};

export default ObserverTab;
