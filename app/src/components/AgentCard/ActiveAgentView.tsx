// components/AgentCard/ActiveAgentView.tsx
import React, { useMemo, useEffect, useState } from 'react';
import { Clock, Power, Activity, Eye, Moon, Sun } from 'lucide-react';
import { wakeAgentLoop } from '@utils/main_loop';
import { StreamState } from '@utils/streamManager';
import { CompleteAgent } from '@utils/agent_database';
import { useLastTools, useLastCapture } from '@hooks/useIterations';
import { useElementWidth } from '@hooks/useElementWidth';
import { CARD_LAYOUT_BREAKPOINT_PX } from './layoutBreakpoint';
import { DetectionMode } from '@utils/change_detector';
import ToolStatus from '@components/AgentCard/ToolStatus';
import SensorPreviewPanel from './SensorPreviewPanel';
import ChangeDetectionIndicator from './ChangeDetectionIndicator';
import PieTimer from './PieTimer';
import ChangeDetectionSettings from '@components/ChangeDetectionSettings';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';
import 'katex/dist/katex.min.css';

export type AgentLiveStatus = 'STARTING' | 'CAPTURING' | 'THINKING' | 'RESPONDING' | 'WAITING' | 'SKIPPED' | 'SLEEPING' | 'IDLE';

// Define the type for change detection data
interface ChangeDetectionData {
  agentId: string;
  isFirstIteration?: boolean;
  textChanged?: boolean;
  imagesChanged?: boolean;
  isSignificant?: boolean;
  detectionMode?: DetectionMode;
  thresholds?: {
    textSimilarity: number;
    dhashSimilarity: number;
    pixelSimilarity: number;
    suspiciousSimilarity: number;
  };
  imageDetails?: Array<{
    dhashSimilarity?: number;
    pixelSimilarity?: number;
    triggeredPixelCheck: boolean;
    contentType: 'camera' | 'ui' | 'unknown';
  }>;
}

// --- Helper Components ---

const SleepIcon: React.FC<{ onWake?: () => void }> = ({ onWake }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <button
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onWake}
      className="p-0.5 rounded hover:bg-blue-100 transition-colors cursor-pointer"
      title="Agent is sleeping to avoid spamming notifications. Click to wake up early."
    >
      {isHovered ? (
        <Sun className="w-5 h-5 text-yellow-500" />
      ) : (
        <Moon className="w-5 h-5" />
      )}
    </button>
  );
};

const ModelStatePanel: React.FC<{
  status: AgentLiveStatus;
  capture?: { image?: string; prompt?: string };
  changeDetectionData?: ChangeDetectionData | null;
  onSettingsClick?: (threshold: 'text' | 'dhash' | 'pixel' | 'suspicious') => void;
  onWake?: () => void;
  onClick?: () => void;
  loopProgress?: number;
  sleepProgress?: number;
  loopDurationMs?: number;
  sleepDurationMs?: number;
  skipReason?: 'same_inputs' | 'network_error' | null;
  isOverrun?: boolean;
}> = ({ status, capture, changeDetectionData, onSettingsClick, onWake, onClick, loopProgress, sleepProgress, loopDurationMs, sleepDurationMs, skipReason, isOverrun }) => {
  const statusInfo = useMemo(() => {
    switch (status) {
      case 'STARTING': return { icon: <Power className="w-5 h-5" />, text: 'Agent is starting...', color: 'text-yellow-600' };
      case 'CAPTURING': return { icon: <Eye className="w-5 h-5 animate-subtle-pulse" />, text: 'Capturing Inputs...', color: 'text-cyan-600' };
      case 'THINKING': return { icon: <Activity className="w-5 h-5" />, text: 'Model is thinking...', color: 'text-purple-600' };
      case 'RESPONDING': return { icon: <Activity className="w-5 h-5 animate-pulse" />, text: 'Model is responding...', color: 'text-blue-600' };
      case 'SLEEPING': return { icon: null, text: 'Sleeping...', color: 'text-blue-600' }; // Icon handled separately
      case 'SKIPPED': return skipReason === 'network_error'
        ? { icon: <Clock className="w-5 h-5" />, text: 'Skipped due to Network Error, Waiting...', color: 'text-red-500' }
        : { icon: <Clock className="w-5 h-5" />, text: 'Reused Model Call, Waiting...', color: 'text-gray-500' };
      case 'WAITING': return { icon: <Clock className="w-5 h-5" />, text: 'Waiting for next cycle...', color: 'text-gray-500' };
      default: return { icon: <div />, text: 'Idle', color: 'text-gray-400' };
    }
  }, [status, skipReason]);
  // modelImages are stored as raw base64; add the data URL prefix unless already present
  // (mirrors toDataUrl in mcp/registry.ts). Empty for text-only sensors / before first capture.
  const src = capture?.image
    ? (capture.image.startsWith('data:') ? capture.image : `data:image/png;base64,${capture.image}`)
    : undefined;

  return (
    <div
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } } : undefined}
      title={onClick ? 'View activity log' : undefined}
      className={`flex items-start gap-3 rounded-lg border border-gray-200 bg-gray-50 p-2 ${onClick ? 'cursor-pointer hover:bg-gray-100 transition-colors' : ''}`}
    >
      {/* Left: the still frame the model actually reasoned about (contrasts with the live
          feed on the left of the card — shows the agent snapshots a frame, not video). */}
      {src && (
        <img
          src={src}
          alt="Frame sent to the model"
          className="h-16 w-16 rounded object-cover flex-shrink-0 border border-gray-200"
        />
      )}
      {/* Right: live status, then the prompt that went with the frame. */}
      <div className="flex-1 min-w-0">
        <div className={`flex items-center gap-2 ${statusInfo.color}`}>
          {/* Sleep/wake is interactive — don't let it bubble up to the activity modal. */}
          <div className="flex-shrink-0" onClick={status === 'SLEEPING' ? (e) => e.stopPropagation() : undefined}>
            {status === 'SLEEPING' ? <SleepIcon onWake={onWake} /> : statusInfo.icon}
          </div>
          <span className="font-medium text-sm">{statusInfo.text}</span>
          {/* Pie timer for WAITING, SLEEPING, or overrun states */}
          {((status === 'SLEEPING' && !!sleepProgress) ||
            ((status === 'WAITING' || isOverrun) && !!loopProgress)) && (
            <PieTimer
              progress={status === 'SLEEPING' ? (sleepProgress || 0) : (loopProgress || 0)}
              color={status === 'SLEEPING' ? 'blue' : isOverrun ? 'orange' : 'green'}
              totalDurationMs={status === 'SLEEPING' ? sleepDurationMs : loopDurationMs}
              isFilling={status !== 'SLEEPING'}
              size={20}
            />
          )}
          {changeDetectionData && (
            // Settings indicator opens its own modal — stop it from also firing the activity modal.
            <span onClick={(e) => e.stopPropagation()}>
              <ChangeDetectionIndicator
                data={changeDetectionData}
                onSettingsClick={onSettingsClick}
              />
            </span>
          )}
          {(status === 'THINKING' || status === 'RESPONDING' || status === 'STARTING') && <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin ml-auto" />}
        </div>
        {src && capture?.prompt && (
          <p className="text-xs text-gray-500 line-clamp-3 mt-1.5">{capture.prompt}</p>
        )}
      </div>
    </div>
  );
};

const MarkdownRenderer: React.FC<{ content: string }> = ({ content }) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeRaw, rehypeKatex]}
      components={{
        code({ node, inline, className, children, ...props }: any) {
          const match = /language-(\w+)/.exec(className || '');
          return !inline && match ? (
            <SyntaxHighlighter
              style={dracula}
              language={match[1]}
              PreTag="div"
              customStyle={{
                margin: '0.5rem 0',
                borderRadius: '0.375rem',
                fontSize: '0.75rem',
              }}
              {...props}
            >
              {String(children).replace(/\n$/, '')}
            </SyntaxHighlighter>
          ) : (
            <code className="bg-gray-100 text-gray-800 px-1 py-0.5 rounded text-xs" {...props}>
              {children}
            </code>
          );
        },
        table({ children }) {
          return (
            <div className="overflow-x-auto my-2">
              <table className="min-w-full divide-y divide-gray-200 border border-gray-300 text-xs">
                {children}
              </table>
            </div>
          );
        },
        thead({ children }) {
          return <thead className="bg-gray-50">{children}</thead>;
        },
        th({ children }) {
          return (
            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700 border-b border-gray-300">
              {children}
            </th>
          );
        },
        td({ children }) {
          return (
            <td className="px-3 py-2 text-xs text-gray-700 border-b border-gray-200">
              {children}
            </td>
          );
        },
        p({ children }) {
          return <p className="mb-2 last:mb-0">{children}</p>;
        },
        ul({ children }) {
          return <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>;
        },
        ol({ children }) {
          return <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>;
        },
        h1({ children }) {
          return <h1 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h1>;
        },
        h2({ children }) {
          return <h2 className="text-sm font-bold mb-2 mt-2 first:mt-0">{children}</h2>;
        },
        h3({ children }) {
          return <h3 className="text-sm font-semibold mb-1 mt-2 first:mt-0">{children}</h3>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

const ThinkingBox: React.FC<{
  content: string;
  isExpanded: boolean;
  onToggle: () => void;
}> = ({ content, isExpanded, onToggle }) => {
  const scrollRef = React.useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isExpanded && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [content, isExpanded]);

  const lastWords = content.trimEnd().split(/\s+/).slice(-10).join(' ');

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg shadow-sm text-xs text-gray-500">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-100 rounded-lg transition-colors text-left"
      >
        <span className="flex-shrink-0">💭</span>
        {isExpanded ? (
          <span className="italic flex-1 truncate">Thinking...</span>
        ) : (
          <span className="italic flex-1 truncate text-gray-400">...{lastWords}</span>
        )}
        <span className="flex-shrink-0 text-gray-400">{isExpanded ? '▲' : '▼'}</span>
      </button>
      {isExpanded && (
        <div
          ref={scrollRef}
          className="px-3 pb-2 max-h-32 overflow-y-auto scrollbar-thin italic text-gray-400 whitespace-pre-wrap"
        >
          {content}
        </div>
      )}
    </div>
  );
};

const LastResponse: React.FC<{ response: string, responseKey: number }> = ({ response, responseKey }) => {
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = React.useState(true);
  const prevScrollHeightRef = React.useRef<number>(0);

  // Handle scroll events to detect user interaction
  const handleScroll = () => {
    if (!scrollRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const isAtBottom = scrollTop + clientHeight >= scrollHeight - 5;

    // Enable auto-scroll if user scrolled to bottom, disable if scrolled up
    setAutoScroll(isAtBottom);
  };

  // Auto-scroll effect when content changes
  useEffect(() => {
    if (!scrollRef.current || !autoScroll) return;

    const { scrollHeight } = scrollRef.current;

    // Only scroll if content actually changed (new content added)
    if (scrollHeight !== prevScrollHeightRef.current) {
      scrollRef.current.scrollTop = scrollHeight;
      prevScrollHeightRef.current = scrollHeight;
    }
  }, [response, autoScroll]);

  // Reset auto-scroll when responseKey changes (new response started)
  useEffect(() => {
    setAutoScroll(true);
  }, [responseKey]);

  return (
    <div key={responseKey} className="bg-white border border-gray-200 rounded-lg shadow-sm animate-fade-in min-h-0">
      <h4 className="text-xs font-semibold text-gray-500 mb-1 px-3 pt-2">Last Response</h4>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="text-sm text-gray-700 max-h-40 overflow-y-auto scrollbar-thin px-3 pb-2"
      >
        <MarkdownRenderer content={response} />
      </div>
    </div>
  );
};


// --- Main Component ---

interface ActiveAgentViewProps {
    streams: StreamState;
    liveStatus: AgentLiveStatus;
    lastResponse: string;
    responseKey: number;
    agentId: string;
    agent: CompleteAgent;
    code?: string;
    loopProgress?: number;
    sleepProgress?: number;
    loopDurationMs?: number;
    sleepDurationMs?: number;
    skipReason?: 'same_inputs' | 'network_error' | null;
    isOverrun?: boolean;
    onActivity?: (agentId: string) => void;
}

const ActiveAgentView: React.FC<ActiveAgentViewProps> = ({
    streams,
    liveStatus,
    lastResponse,
    responseKey,
    agentId,
    agent,
    loopProgress,
    sleepProgress,
    loopDurationMs,
    sleepDurationMs,
    skipReason,
    isOverrun,
    onActivity
}) => {
    const [streamingResponse, setStreamingResponse] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [thinkingContent, setThinkingContent] = useState('');
    const [isThinkingExpanded, setIsThinkingExpanded] = useState(false);
    const lastTools = useLastTools(agentId);
    const lastCapture = useLastCapture(agentId);
    const [changeDetectionData, setChangeDetectionData] = useState<ChangeDetectionData | null>(null);
    const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
    const [focusedThreshold, setFocusedThreshold] = useState<'text' | 'dhash' | 'pixel' | 'suspicious' | undefined>(undefined);

    // ESC key to close modal
    useEffect(() => {
        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && isSettingsModalOpen) {
                setIsSettingsModalOpen(false);
                setFocusedThreshold(undefined);
            }
        };
        window.addEventListener('keydown', handleEscape);
        return () => window.removeEventListener('keydown', handleEscape);
    }, [isSettingsModalOpen]);

    useEffect(() => {
        const handleStreamStart = (event: CustomEvent) => {
            if (event.detail.agentId === agentId) {
                setStreamingResponse('');
                setIsStreaming(true);
            }
        };

        const handleStreamChunk = (event: CustomEvent) => {
            if (event.detail.agentId === agentId) {
                setStreamingResponse(prev => prev + event.detail.chunk);
            }
        };

        const handleIterationStart = (event: CustomEvent) => {
            if (event.detail.agentId === agentId) {
                setIsStreaming(false);
                setStreamingResponse('');
                setThinkingContent('');
                setIsThinkingExpanded(false);
            }
        };

        const handleThinkingChunk = (event: CustomEvent) => {
            if (event.detail.agentId === agentId) {
                setThinkingContent(prev => prev + event.detail.chunk);
            }
        };

        const handleChangeDetection = (event: CustomEvent) => {
            if (event.detail.agentId === agentId) {
                // Only store if not first iteration
                if (!event.detail.isFirstIteration) {
                    setChangeDetectionData(event.detail);
                } else {
                    setChangeDetectionData(null);
                }
            }
        };

        window.addEventListener('agentStreamStart', handleStreamStart as EventListener);
        window.addEventListener('agentResponseChunk', handleStreamChunk as EventListener);
        window.addEventListener('agentIterationStart', handleIterationStart as EventListener);
        window.addEventListener('agentChangeDetectionResult', handleChangeDetection as EventListener);
        window.addEventListener('agentThinkingChunk', handleThinkingChunk as EventListener);

        return () => {
            window.removeEventListener('agentStreamStart', handleStreamStart as EventListener);
            window.removeEventListener('agentResponseChunk', handleStreamChunk as EventListener);
            window.removeEventListener('agentIterationStart', handleIterationStart as EventListener);
            window.removeEventListener('agentChangeDetectionResult', handleChangeDetection as EventListener);
            window.removeEventListener('agentThinkingChunk', handleThinkingChunk as EventListener);
        };
    }, [agentId]);

    // Responsive to the card's own rendered width (ResizeObserver), not the
    // viewport — this card lives in a resizable tiling grid, so a "desktop"
    // viewport says nothing about how wide this particular tile is.
    const { ref: gridRef, width: gridWidth } = useElementWidth<HTMLDivElement>(750);
    const isWide = gridWidth >= CARD_LAYOUT_BREAKPOINT_PX;

    return (
        <>
            <div
                ref={gridRef}
                className={`grid animate-fade-in overflow-visible ${isWide ? 'grid-cols-2 gap-6' : 'grid-cols-1'}`}
            >
                {/* Left Column: Sensor Previews */}
                <SensorPreviewPanel
                    agentId={agentId}
                    streams={streams}
                    systemPrompt={agent.system_prompt}
                />

            {/* Right Column: Status and Response */}
            <div className="space-y-4 flex flex-col justify-start overflow-visible">
                <ModelStatePanel
                    status={liveStatus}
                    capture={lastCapture}
                    onClick={onActivity ? () => onActivity(agentId) : undefined}
                    changeDetectionData={changeDetectionData}
                    onSettingsClick={(threshold) => {
                        setFocusedThreshold(threshold);
                        setIsSettingsModalOpen(true);
                    }}
                    onWake={() => wakeAgentLoop(agentId)}
                    loopProgress={loopProgress}
                    sleepProgress={sleepProgress}
                    loopDurationMs={loopDurationMs}
                    sleepDurationMs={sleepDurationMs}
                    skipReason={skipReason}
                    isOverrun={isOverrun}
                />
                {thinkingContent.length > 0 && (
                    <ThinkingBox
                        content={thinkingContent}
                        isExpanded={isThinkingExpanded}
                        onToggle={() => setIsThinkingExpanded(prev => !prev)}
                    />
                )}
                <LastResponse
                    response={isStreaming ? streamingResponse : lastResponse}
                    responseKey={isStreaming ? -1 : responseKey}
                />
                {/* Tool Status - Show below last response only if there are tools */}
                {lastTools.length > 0 && (
                    <ToolStatus
                        tools={lastTools}
                        variant="compact"
                    />
                )}
            </div>

            {/* Change Detection Settings Modal */}
            {isSettingsModalOpen && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[70]"
                    onClick={() => {
                        setIsSettingsModalOpen(false);
                        setFocusedThreshold(undefined);
                    }}
                >
                    <div
                        className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-y-auto m-4"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between">
                            <h2 className="text-xl font-bold text-gray-900">Change Detection Settings</h2>
                            <button
                                onClick={() => {
                                    setIsSettingsModalOpen(false);
                                    setFocusedThreshold(undefined);
                                }}
                                className="text-gray-400 hover:text-gray-600 transition-colors"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                        <div className="p-6">
                            <ChangeDetectionSettings
                                compact={true}
                                focusedThreshold={focusedThreshold}
                            />
                        </div>
                    </div>
                </div>
            )}
            </div>
        </>
    );
};

export default ActiveAgentView;
