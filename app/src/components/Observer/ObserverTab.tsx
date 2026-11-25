// src/components/Observer/ObserverTab.tsx
//
// Top-level "Observer" tab — the app's default landing view. A minimal ChatGPT/Claude
// Cowork-style splash (ObserverHero) until the first message is sent, then docks into a
// normal scrolling transcript with the input pinned to the bottom (MCP, unboxed). A footer
// lists the user's micro-agents in both states so they never have to open the My Agents grid
// just to see what exists and what's running.
//
// The My Agents tab (grid, GetStarted, MCPPanel, minimize system) is untouched by this file.

import React, { useMemo } from 'react';
import type { TokenProvider } from '@utils/main_loop';
import type { CompleteAgent } from '@utils/agent_database';
import { useMCPContext } from '../../mcp/MCPContext';
import MCP from '@components/AICreator/MCP';
import ObserverHero from './ObserverHero';
import AgentsFooter from './AgentsFooter';
import TutorialDemoBanner from './TutorialDemoBanner';
import { AgentLiveStateProvider } from './AgentLiveStateContext';
import StatusPipBridge from './StatusPipBridge';
import { isMobile } from '@utils/platform';

// Tracks live status/progress/last-word for every running or starting agent.
const LiveAgentTracking: React.FC<{
  runningAgents: Set<string>;
  startingAgents: Set<string>;
  children: React.ReactNode;
}> = ({ runningAgents, startingAgents, children }) => {
  const agentIds = useMemo(
    () => Array.from(new Set([...runningAgents, ...startingAgents])),
    [runningAgents, startingAgents]
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
    <LiveAgentTracking runningAgents={runningAgents} startingAgents={startingAgents}>
      <div className="flex flex-col h-full">
        {/* Mobile: the status window (Run in Background) shows these same agents. */}
        {isMobile() && <StatusPipBridge agents={agents} runningAgents={runningAgents} startingAgents={startingAgents} />}
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
        <AgentsFooter
          agents={agents}
          runningAgents={runningAgents}
          startingAgents={startingAgents}
          onToggle={onToggleAgent}
          onOpenMicroAgents={onOpenMicroAgents}
        />
      </div>
    </LiveAgentTracking>
  );
};

export default ObserverTab;
