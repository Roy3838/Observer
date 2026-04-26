// src/components/shared/ToolStatus.tsx
import React, { useState } from 'react';
import {
  CheckCircle, XCircle, Send, MessageSquare, MessageSquarePlus, MessageSquareQuote,
  MessageCircle, Mail, Bell, Save, SquarePen, PlayCircle, StopCircle, Hourglass,
  Video, VideoOff, Hammer, Tag, AlertTriangle, HelpCircle, Phone, MousePointerClick, PartyPopper
} from 'lucide-react';
import { ToolCall } from '@utils/IterationStore';

// Tool icon mapping
const getToolIcon = (toolName: string) => {
  const iconMap: Record<string, React.ElementType> = {
    'code-execution': AlertTriangle,
    'unknown': HelpCircle,
    sendDiscordBot: () => <Send className="w-4 h-4" />,
    sendWhatsapp: () => <MessageSquare className="w-4 h-4" />,
    sendSms: () => <MessageSquarePlus className="w-4 h-4" />,
    sendPushover: () => <Send className="w-4 h-4" />,
    sendTelegram: MessageCircle,
    sendEmail: Mail,
    call: Phone,
    notify: Bell,
    system_notify: Bell,
    getMemory: Save,
    setMemory: SquarePen,
    appendMemory: SquarePen,
    startAgent: PlayCircle,
    stopAgent: StopCircle,
    time: Hourglass,
    startClip: Video,
    stopClip: VideoOff,
    markClip: Tag,
    ask: MessageSquareQuote,
    message: MessageSquare,
    click: MousePointerClick,
    celebrate: PartyPopper,
  };
  return iconMap[toolName] || HelpCircle;
};

interface ToolStatusProps {
  tools: ToolCall[];
  variant?: 'compact' | 'full' | 'inline';
  showTooltip?: boolean;
  maxTools?: number;
  className?: string;
}

const ToolStatus: React.FC<ToolStatusProps> = ({
  tools,
  variant = 'inline',
  showTooltip = true,
  maxTools,
  className = ''
}) => {
  const [hoveredTool, setHoveredTool] = useState<{ index: string; name: string; status: string; error?: string } | null>(null);

  // Limit tools if maxTools is specified
  const displayTools = maxTools ? tools.slice(-maxTools) : tools;

  if (displayTools.length === 0) {
    if (variant === 'compact') {
      return (
        <div className={`flex items-center gap-2 bg-gray-50 px-3 py-2 rounded-lg border border-gray-200 flex-1 min-w-0 ${className}`}>
          <Hammer className="w-4 h-4 text-gray-500 flex-shrink-0" />
          <div className="text-xs font-medium text-gray-500">Tools</div>
          <span className="text-gray-500 text-sm italic">No recent actions</span>
        </div>
      );
    }

    return (
      <div className={`flex items-center gap-2 text-gray-400 text-sm bg-gray-50 px-3 py-2 rounded-lg border border-gray-200 ${className}`}>
        <span className="italic">No tools used</span>
      </div>
    );
  }

  // Compact variant for ActiveAgentView
  if (variant === 'compact') {
    // Check if any tools have errors to determine section color
    const hasErrors = displayTools.some(tool => tool.status === 'error');
    const sectionColorClass = hasErrors
      ? 'bg-red-50 border-red-200'
      : 'bg-green-50 border-green-200';
    const iconColorClass = hasErrors ? 'text-red-600' : 'text-green-600';
    const labelColorClass = hasErrors ? 'text-red-600' : 'text-green-600';
    const countColorClass = hasErrors ? 'text-red-500' : 'text-green-500';

    return (
      <div className={`${sectionColorClass} px-3 py-2 rounded-lg border flex-1 min-w-0 overflow-visible ${className}`}>
        <div className="flex items-center gap-2 mb-2">
          <Hammer className={`w-4 h-4 ${iconColorClass} flex-shrink-0`} />
          <div className={`text-xs font-medium ${labelColorClass}`}>Tools</div>
          <span className={`text-xs ${countColorClass}`}>({displayTools.length} tool{displayTools.length !== 1 ? 's' : ''})</span>
        </div>
        <div className="flex gap-2 flex-wrap overflow-visible">
          {displayTools.map((tool, index) => {
            const isSuccess = tool.status === 'success';
            const ToolIcon = getToolIcon(tool.name);
            const toolId = `tool-${tool.name}-${index}-${tool.timestamp}`;

            return (
              <div key={index} className="relative">
                <div
                  className={`flex items-center gap-1 px-2 py-1 rounded border text-xs ${
                    isSuccess
                      ? 'bg-green-50 border-green-200 hover:bg-green-100'
                      : 'bg-red-50 border-red-200 hover:bg-red-100'
                  } ${showTooltip ? 'cursor-help' : ''}`}
                  onMouseEnter={showTooltip ? () => setHoveredTool({
                    index: toolId,
                    name: tool.name,
                    status: isSuccess ? 'Success' : 'Failed',
                    error: tool.error
                  }) : undefined}
                  onMouseLeave={showTooltip ? () => setHoveredTool(null) : undefined}
                >
                  <ToolIcon className={`w-3 h-3 ${isSuccess ? 'text-green-600' : 'text-red-600'}`} />
                  {isSuccess ? (
                    <CheckCircle className="w-3 h-3 text-green-600" />
                  ) : (
                    <XCircle className="w-3 h-3 text-red-600" />
                  )}
                </div>

                {/* Tooltip */}
                {showTooltip && hoveredTool?.index === toolId && (
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded shadow-lg z-10 max-w-xs">
                    <div className="font-medium whitespace-nowrap">
                      {hoveredTool.name} - {hoveredTool.status}
                    </div>
                    {hoveredTool.error && (
                      <>
                        <div className="border-t border-gray-700 my-1"></div>
                        <div className="text-gray-300 break-words">
                          {hoveredTool.error.length > 120
                            ? `${hoveredTool.error.substring(0, 120)}...`
                            : hoveredTool.error}
                        </div>
                      </>
                    )}
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Full variant for detailed views
  if (variant === 'full') {
    return (
      <div className={`space-y-3 ${className}`}>
        {displayTools.map((tool, index) => {
          const isSuccess = tool.status === 'success';
          const ToolIcon = getToolIcon(tool.name);

          return (
            <div key={index} className="bg-gray-50 border rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <ToolIcon className={`w-4 h-4 ${
                  isSuccess ? 'text-green-600' : 'text-red-600'
                }`} />
                <span className="font-medium text-gray-900">{tool.name}</span>
                {isSuccess ? (
                  <CheckCircle className="w-4 h-4 text-green-600" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-600" />
                )}
              </div>

              {tool.error && (
                <div className="text-sm text-red-600 mt-2">
                  {tool.error}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  }

  // Inline variant for iteration cards (default)
  return (
    <div className={`flex items-center gap-3 relative overflow-visible ${className}`}>
      {displayTools.map((tool, index) => {
        const isSuccess = tool.status === 'success';
        const ToolIcon = getToolIcon(tool.name);
        const toolId = `tool-${tool.name}-${index}-${tool.timestamp}`;

        return (
          <div key={index} className="relative">
            <div
              className={`flex items-center gap-1 px-2 py-1 rounded border ${
                isSuccess
                  ? 'bg-green-50 border-green-200 hover:bg-green-100'
                  : 'bg-red-50 border-red-200 hover:bg-red-100'
              } ${showTooltip ? 'cursor-help' : ''}`}
              onMouseEnter={showTooltip ? () => setHoveredTool({
                index: toolId,
                name: tool.name,
                status: isSuccess ? 'Success' : 'Failed',
                error: tool.error
              }) : undefined}
              onMouseLeave={showTooltip ? () => setHoveredTool(null) : undefined}
            >
              <ToolIcon className={`w-3 h-3 ${isSuccess ? 'text-green-600' : 'text-red-600'}`} />
              {isSuccess ? (
                <CheckCircle className="w-3 h-3 text-green-600" />
              ) : (
                <XCircle className="w-3 h-3 text-red-600" />
              )}
            </div>

            {/* Tooltip */}
            {showTooltip && hoveredTool?.index === toolId && (
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded shadow-lg z-10 max-w-xs">
                <div className="font-medium whitespace-nowrap">
                  {hoveredTool.name} - {hoveredTool.status}
                </div>
                {hoveredTool.error && (
                  <>
                    <div className="border-t border-gray-700 my-1"></div>
                    <div className="text-gray-300 break-words">
                      {hoveredTool.error.length > 120
                        ? `${hoveredTool.error.substring(0, 120)}...`
                        : hoveredTool.error}
                    </div>
                  </>
                )}
                <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default ToolStatus;
export { getToolIcon };
