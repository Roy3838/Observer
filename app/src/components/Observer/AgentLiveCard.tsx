// src/components/Observer/AgentLiveCard.tsx
//
// An agent's live card in the Observer chat: status, countdown, last word and sensor preview.
// `pinned` folds it to a one-line bar with a video thumbnail; see AgentChatCard.tsx for when.

import React, { useEffect, useRef } from 'react';
import { Square, Play, Loader2, ArrowUpRight, ChevronDown } from 'lucide-react';
import type { CompleteAgent } from '@utils/agent_database';
import { StreamState } from '@utils/streamManager';
import PieTimer from '@components/AgentCard/PieTimer';
import SensorPreviewPanel from '@components/AgentCard/SensorPreviewPanel';
import { agentHasCameraSensor, agentHasScreenSensor } from '@components/AgentCard/agentCapabilities';
import { useTutorialFlow, tutorialFlow } from '@utils/tutorialFlow';
import { useAgentLiveStateFor, type AgentLiveStatus } from './AgentLiveStateContext';

const STATUS_LABEL: Record<AgentLiveStatus, string> = {
  STARTING: 'Starting…',
  CAPTURING: 'Capturing…',
  THINKING: 'Thinking…',
  RESPONDING: 'Responding…',
  WAITING: 'Waiting…',
  SKIPPED: 'Skipped (no change)',
  SLEEPING: 'Sleeping',
  IDLE: 'Idle',
};

// While the loop interval has elapsed but the model is still working.
const OVERRUN_LABEL: Partial<Record<AgentLiveStatus, string>> = {
  CAPTURING: 'Still capturing…',
  THINKING: 'Still thinking…',
  RESPONDING: 'Still responding…',
};

function formatCountdown(ms: number) {
  const totalSeconds = Math.max(0, Math.ceil(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}:${seconds.toString().padStart(2, '0')}` : `${seconds}s`;
}

/** The stream a pinned card shows in miniature, if the agent watches one. */
function previewStream(agent: CompleteAgent, streams: StreamState): MediaStream | null {
  if (agentHasScreenSensor(agent.system_prompt) && streams.screenVideoStream) return streams.screenVideoStream;
  if (agentHasCameraSensor(agent.system_prompt) && streams.cameraStream) return streams.cameraStream;
  return null;
}

const VideoThumbnail: React.FC<{ stream: MediaStream }> = ({ stream }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (videoRef.current) videoRef.current.srcObject = stream;
  }, [stream]);
  return <video ref={videoRef} muted autoPlay playsInline className="w-14 h-8 rounded-md bg-black object-cover flex-shrink-0" />;
};

interface AgentLiveCardProps {
  agent: CompleteAgent;
  isRunning: boolean;
  isStarting: boolean;
  streams: StreamState;
  onToggle: (agentId: string, isCurrentlyRunning: boolean) => void;
  onSelectAgent?: (agentId: string) => void;
  /** Stuck to the top of the transcript: folded to one line until tapped. */
  pinned: boolean;
  /** Unfolded by the user while pinned. */
  unfolded: boolean;
  onUnfold: (unfolded: boolean) => void;
}

const AgentLiveCard: React.FC<AgentLiveCardProps> = ({ agent, isRunning, isStarting, streams, onToggle, onSelectAgent, pinned, unfolded, onUnfold }) => {
  const { liveStatus, lastWord, progress, durationMs, isSleeping, sleepRemainingMs, isOverrun } = useAgentLiveStateFor(agent.id);
  const tutorial = useTutorialFlow();
  useEffect(() => {
    if (tutorial.phase === 'building' && isRunning) tutorialFlow.attachAgent(agent.id);
  }, [tutorial.phase, isRunning, agent.id]);
  // The demo agent being stopped ends the demo (nothing left to wait on).
  useEffect(() => {
    if (tutorial.agentId === agent.id && !isRunning && tutorial.phase === 'monitoring') tutorialFlow.reset();
  }, [tutorial.phase, tutorial.agentId, isRunning, agent.id]);
  const isTutorialAgent = tutorial.agentId === agent.id;
  const monitoringCopy = isTutorialAgent && tutorial.phase === 'monitoring'
    ? "This is your micro agent! It checks the demo tab on a loop" : null;

  const folded = pinned && !unfolded;

  const isLive = isRunning || isStarting;
  const overrun = isLive && isOverrun;
  const sleeping = isLive && isSleeping;
  const statusLabel = !isLive ? 'Stopped'
    : sleeping ? 'Sleeping'
    : (overrun && OVERRUN_LABEL[liveStatus]) || STATUS_LABEL[liveStatus];
  const timeLabel = sleeping
    ? formatCountdown(sleepRemainingMs)
    : isRunning && durationMs
      ? formatCountdown(durationMs * (1 - progress / 100))
      : null;

  const thumbnail = folded ? previewStream(agent, streams) : null;
  const opensAgent = !pinned && !!onSelectAgent;
  const handleHeaderClick = pinned ? () => onUnfold(!unfolded) : opensAgent ? () => onSelectAgent!(agent.id) : undefined;

  return (
    <div className="relative w-full bg-white border border-gray-200 rounded-xl shadow-lg">
      {monitoringCopy && (
        <div className={`absolute left-1/2 -translate-x-1/2 w-64 max-w-full z-20 select-none pointer-events-none flex items-center ${pinned ? 'top-full mt-2 flex-col-reverse' : 'bottom-full mb-2 flex-col'}`}>
          <div className="bg-slate-900 text-white rounded-2xl px-4 py-3 shadow-lg text-sm font-medium text-center">{monitoringCopy}</div>
          <div className={`w-3 h-3 bg-slate-900 rotate-45 ${pinned ? '-mb-1.5' : '-mt-1.5'}`} />
        </div>
      )}
      <div className={`flex items-center gap-2 pr-3 ${pinned ? 'h-[52px]' : 'h-16'} ${folded ? '' : 'border-b border-gray-100'}`}>
        <button
          type="button"
          onClick={handleHeaderClick}
          disabled={!handleHeaderClick}
          aria-expanded={pinned ? unfolded : undefined}
          title={pinned ? (unfolded ? 'Hide preview' : 'Show preview') : opensAgent ? `Open "${agent.name}" in Micro Agents` : undefined}
          className={`h-full min-w-0 flex-1 flex items-center gap-3 text-left ${pinned ? 'pl-2.5' : 'pl-4'}`}
        >
          <span className="w-9 h-9 flex-shrink-0 flex items-center justify-center">
            {isRunning && !sleeping ? (
              <PieTimer progress={progress} color={overrun ? 'orange' : 'green'} totalDurationMs={durationMs || undefined} isFilling size={pinned ? 30 : 36} />
            ) : (
              <span className={`w-2.5 h-2.5 rounded-full ${isLive ? 'bg-blue-400' : 'border border-gray-400'}`} />
            )}
          </span>

          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-1.5">
              <span className={`font-semibold text-gray-800 truncate ${pinned ? 'text-sm' : 'text-base'}`}>{agent.name}</span>
              {opensAgent && <ArrowUpRight className="w-3.5 h-3.5 text-gray-300 flex-shrink-0" />}
              {pinned && <ChevronDown className={`w-3.5 h-3.5 text-gray-400 flex-shrink-0 transition-transform duration-200 ${unfolded ? 'rotate-180' : ''}`} />}
            </span>
            {/* status | time | last word: one glance line, e.g. "Sleeping · 12s · PERSON_DETECTED" */}
            <span className={`flex items-center gap-1.5 text-xs min-w-0 ${overrun ? 'text-orange-500' : 'text-gray-400'}`}>
              <span className="flex-shrink-0">{statusLabel}</span>
              {timeLabel && <><span className="flex-shrink-0">·</span><span className="flex-shrink-0">{timeLabel}</span></>}
              <span className="flex-shrink-0">·</span>
              <span key={lastWord} className="font-mono font-semibold text-gray-600 truncate animate-fade-in">{lastWord || '—'}</span>
            </span>
          </span>
        </button>

        {thumbnail && <VideoThumbnail stream={thumbnail} />}

        <button
          type="button"
          onClick={() => onToggle(agent.id, isRunning)}
          disabled={isStarting}
          className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 transition-colors disabled:text-gray-400 ${isRunning ? 'text-red-500 hover:bg-red-50' : 'text-green-600 hover:bg-green-50'}`}
          title={isStarting ? 'Starting…' : isRunning ? 'Stop agent' : 'Start agent'}
        >
          {isStarting
            ? <Loader2 className="w-4 h-4 animate-spin" />
            : isRunning
              ? <Square className="w-3.5 h-3.5" fill="currentColor" />
              : <Play className="w-3.5 h-3.5" fill="currentColor" />}
        </button>
      </div>

      {/* Folded, not unmounted, so the preview keeps its video. Animates only while pinned:
          returning to the transcript must restore the full height at once (AgentChatCard
          measures it). */}
      <div className={`grid ${folded ? 'grid-rows-[0fr]' : 'grid-rows-[1fr]'} ${pinned ? 'transition-[grid-template-rows] duration-200 ease-out' : ''}`}>
        <div className={`min-h-0 ${folded ? 'overflow-hidden invisible' : pinned ? 'max-h-[40vh] overflow-y-auto overscroll-contain' : 'overflow-hidden'}`}>
          <div className="px-4 pt-3">
            <SensorPreviewPanel agentId={agent.id} streams={streams} systemPrompt={agent.system_prompt} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentLiveCard;
