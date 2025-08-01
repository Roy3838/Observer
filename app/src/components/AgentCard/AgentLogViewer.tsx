// src/components/AgentLogViewer.tsx
import React, { useState, useEffect } from 'react';
import { 
  Brain, HelpCircle, 
  Monitor, Clipboard, Camera, Mic, CheckCircle, XCircle,
  ScanText, Bell, Mail, Send, MessageSquare, MessageSquarePlus, 
  MessageSquareQuote, PlayCircle, StopCircle, Video, VideoOff, Tag, SquarePen, Hourglass,
  ArrowRight, Clock
} from 'lucide-react';
import { IterationStore, IterationData, SensorData, ToolCall, AgentSession } from '../../utils/IterationStore';

// Lazy Image Loading Component
interface LazyImageProps {
  src: string;
  alt: string;
  className: string;
  fallbackSrc: string;
  imageId: string;
  setLoadedImages: React.Dispatch<React.SetStateAction<Set<string>>>;
}

const LazyImage: React.FC<LazyImageProps> = ({ 
  src, alt, className, fallbackSrc, imageId, setLoadedImages 
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState(false);
  const [shouldLoad, setShouldLoad] = useState(false);

  useEffect(() => {
    // Add a small delay to prevent all images from loading at once
    const timer = setTimeout(() => {
      setShouldLoad(true);
    }, Math.random() * 500); // Random delay up to 500ms

    return () => clearTimeout(timer);
  }, []);

  const handleLoad = () => {
    setIsLoaded(true);
    setLoadedImages(prev => new Set(prev).add(imageId));
  };

  const handleError = () => {
    if (!error && fallbackSrc && fallbackSrc !== src) {
      setError(true);
      // Try fallback format (JPEG instead of PNG)
    } else {
      // Give up and show placeholder
      setIsLoaded(true);
    }
  };

  if (!shouldLoad) {
    // Show placeholder while waiting to load
    return (
      <div className={`${className} bg-gray-200 flex items-center justify-center`}>
        <Monitor className="w-6 h-6 text-gray-400" />
      </div>
    );
  }

  return (
    <>
      {!isLoaded && (
        <div className={`${className} bg-gray-200 flex items-center justify-center animate-pulse`}>
          <Monitor className="w-6 h-6 text-gray-400" />
        </div>
      )}
      <img
        src={error ? fallbackSrc : src}
        alt={alt}
        className={`${className} ${isLoaded ? 'block' : 'hidden'}`}
        onLoad={handleLoad}
        onError={handleError}
      />
    </>
  );
};

// Simple icon components for tools
const getToolIcon = (toolName: string) => {
  const iconMap: Record<string, React.ElementType> = {
    sendDiscordBot: () => <Send className="w-4 h-4" />,
    sendWhatsapp: () => <MessageSquare className="w-4 h-4" />,
    sendSms: () => <MessageSquarePlus className="w-4 h-4" />,
    sendPushover: () => <Send className="w-4 h-4" />,
    sendEmail: Mail,
    notify: Bell,
    system_notify: Bell,
    getMemory: Brain,
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
  };
  return iconMap[toolName] || CheckCircle;
};

// Simple sensor icon mapping
const getSensorIcon = (sensorType: string) => {
  const iconMap: Record<string, React.ElementType> = {
    screenshot: Monitor,
    camera: Camera,
    ocr: ScanText,
    audio: Mic,
    clipboard: Clipboard,
    memory: Brain,
  };
  return iconMap[sensorType] || Monitor;
};

interface AgentLogViewerProps {
  agentId: string;
  maxEntries?: number;
  maxHeight?: string;
  getToken?: () => Promise<string | undefined>;
  isAuthenticated?: boolean;
}

const AgentLogViewer: React.FC<AgentLogViewerProps> = ({
  agentId,
  maxEntries = 50,
  maxHeight = '400px',
}) => {
  const [currentIterations, setCurrentIterations] = useState<IterationData[]>([]);
  const [historicalSessions, setHistoricalSessions] = useState<AgentSession[]>([]);
  const [hoveredTool, setHoveredTool] = useState<{ index: string; name: string; status: string } | null>(null);
  const [, setLoadedImages] = useState<Set<string>>(new Set());
  const [selectedIteration, setSelectedIteration] = useState<IterationData | null>(null);
  const [scrollPosition, setScrollPosition] = useState(0);

  useEffect(() => {
    // Get initial data
    const loadData = async () => {
      const initialIterations = IterationStore.getIterationsForAgent(agentId);
      const historicalData = await IterationStore.getHistoricalSessions(agentId);
      
      setCurrentIterations(initialIterations);
      setHistoricalSessions(historicalData);
    };
    
    loadData();

    // Subscribe to updates
    const unsubscribe = IterationStore.subscribe(async () => {
      const updatedIterations = IterationStore.getIterationsForAgent(agentId);
      const updatedHistorical = await IterationStore.getHistoricalSessions(agentId);
      
      setCurrentIterations(updatedIterations);
      setHistoricalSessions(updatedHistorical);
    });

    return unsubscribe;
  }, [agentId]);

  // Render sensor previews
  const renderSensorPreviews = (sensors: SensorData[], modelImages?: string[], iterationId?: string) => {
    const allPreviews: React.ReactNode[] = [];
    let imageIndex = 0;

    // Process sensor data and match with corresponding images
    sensors.forEach((sensor, index) => {
      const SensorIcon = getSensorIcon(sensor.type);
      
      if (sensor.type === 'screenshot' || sensor.type === 'camera') {
        // Show image thumbnail with sensor type
        const hasImage = modelImages && imageIndex < modelImages.length;
        const imageData = hasImage ? modelImages[imageIndex] : null;
        
        allPreviews.push(
          <div key={index} className="flex items-center gap-2 bg-gray-50 px-3 py-2 rounded-lg border border-gray-200 w-fit">
            <SensorIcon className="w-4 h-4 text-blue-600 flex-shrink-0" />
            {imageData ? (
              <LazyImage 
                src={`data:image/png;base64,${imageData}`}
                alt={`${sensor.type} ${imageIndex + 1}`}
                className="w-20 h-20 rounded border border-gray-200 object-cover"
                fallbackSrc={`data:image/jpeg;base64,${imageData}`}
                imageId={`${iterationId}-${sensor.type}-${imageIndex}`}
                setLoadedImages={setLoadedImages}
              />
            ) : (
              <span className="text-gray-600 text-sm capitalize font-medium">{sensor.type}</span>
            )}
          </div>
        );
        
        if (hasImage) imageIndex++;
      } else if (sensor.type === 'ocr' || sensor.type === 'clipboard') {
        const preview = typeof sensor.content === 'string' 
          ? sensor.content.slice(0, 30) + (sensor.content.length > 30 ? '...' : '')
          : 'text';
        allPreviews.push(
          <div key={index} className="flex items-center gap-2 bg-gray-50 px-3 py-2 rounded-lg border border-gray-200 w-fit">
            <SensorIcon className="w-4 h-4 text-gray-600 flex-shrink-0" />
            <span className="text-gray-700 text-sm truncate">{preview}</span>
          </div>
        );
      } else if (sensor.type === 'audio') {
        const transcript = sensor.content?.transcript || '';
        const preview = transcript.slice(0, 30) + (transcript.length > 30 ? '...' : '');
        const source = sensor.source ? `(${sensor.source})` : '';
        allPreviews.push(
          <div key={index} className="flex items-center gap-2 bg-gray-50 px-3 py-2 rounded-lg border border-gray-200 w-fit">
            <SensorIcon className="w-4 h-4 text-gray-600 flex-shrink-0" />
            <span className="text-gray-700 text-sm truncate">{preview} {source}</span>
          </div>
        );
      } else {
        allPreviews.push(
          <div key={index} className="flex items-center gap-2 bg-gray-50 px-3 py-2 rounded-lg border border-gray-200 w-fit">
            <SensorIcon className="w-4 h-4 text-gray-600 flex-shrink-0" />
            <span className="text-sm text-gray-600 capitalize font-medium">{sensor.type}</span>
          </div>
        );
      }
    });
    
    if (allPreviews.length === 0) {
      return (
        <div className="flex items-center gap-2 text-gray-400 text-sm bg-gray-50 px-3 py-2 rounded-lg border border-gray-200">
          <HelpCircle className="w-4 h-4" />
          <span>No sensor data</span>
        </div>
      );
    }

    return (
      <div className="flex flex-wrap gap-2">
        {allPreviews}
      </div>
    );
  };

  // Render model response preview
  const renderModelPreview = (response?: string) => {
    if (!response) {
      return (
        <div className="flex items-center gap-2 text-gray-400 text-sm bg-gray-50 px-3 py-2 rounded-lg border border-gray-200">
          <Brain className="w-4 h-4" />
          <span className="italic">No response</span>
        </div>
      );
    }
    
    const preview = response.slice(0, 100);
    return (
      <div className="flex items-center gap-2 bg-green-50 px-3 py-2 rounded-lg border border-green-200">
        <Brain className="w-4 h-4 text-green-600 flex-shrink-0" />
        <span className="text-sm text-green-700 truncate">
          "{preview}{response.length > 100 ? '...' : ''}"
        </span>
      </div>
    );
  };

  // Render tool indicators
  const renderToolIndicators = (tools: ToolCall[], iterationId: string) => {
    if (tools.length === 0) {
      return (
        <div className="flex items-center gap-2 text-gray-400 text-sm bg-gray-50 px-3 py-2 rounded-lg border border-gray-200">
          <span className="italic">No tools used</span>
        </div>
      );
    }
    
    return (
      <div className="flex items-center gap-3 relative">
        {tools.map((tool, index) => {
          const isSuccess = tool.status === 'success';
          const ToolIcon = getToolIcon(tool.name);
          const toolId = `${iterationId}-${tool.name}-${index}`;
          
          return (
            <div key={index} className="relative">
              <div 
                className={`flex items-center gap-1 px-2 py-1 rounded border cursor-help ${
                  isSuccess 
                    ? 'bg-green-50 border-green-200 hover:bg-green-100' 
                    : 'bg-red-50 border-red-200 hover:bg-red-100'
                }`}
                onMouseEnter={() => setHoveredTool({ 
                  index: toolId, 
                  name: tool.name, 
                  status: isSuccess ? 'Success' : 'Failed' 
                })}
                onMouseLeave={() => setHoveredTool(null)}
              >
                <ToolIcon className={`w-3 h-3 ${isSuccess ? 'text-green-600' : 'text-red-600'}`} />
                {isSuccess ? (
                  <CheckCircle className="w-3 h-3 text-green-600" />
                ) : (
                  <XCircle className="w-3 h-3 text-red-600" />
                )}
              </div>
              
              {/* Custom Tooltip */}
              {hoveredTool?.index === toolId && (
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap z-10">
                  {hoveredTool.name} - {hoveredTool.status}
                  <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  // Handle card click
  const handleCardClick = (iteration: IterationData, scrollContainer: HTMLElement) => {
    setScrollPosition(scrollContainer.scrollTop);
    setSelectedIteration(iteration);
  };

  // Handle back to list
  const handleBackToList = () => {
    setSelectedIteration(null);
    // Restore scroll position after component updates
    setTimeout(() => {
      const scrollContainer = document.querySelector('[data-scroll-container]') as HTMLElement;
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollPosition;
      }
    }, 0);
  };

  // Helper function to render iterations
  const renderIterations = (iterations: IterationData[], sessionLabel?: string) => {
    return iterations.map((iteration) => {
      const iterationNumber = iteration.sessionIterationNumber;
      
      return (
        <div 
          key={iteration.id} 
          className={`border rounded-lg p-4 transition-all duration-200 shadow-sm cursor-pointer hover:shadow-md hover:scale-[1.01] ${
            iteration.hasError ? 'border-red-200 bg-red-50 hover:bg-red-100' : 'border-gray-200 bg-white hover:bg-gray-50'
          }`}
          onClick={(e) => {
            const scrollContainer = e.currentTarget.closest('[data-scroll-container]') as HTMLElement;
            handleCardClick(iteration, scrollContainer);
          }}
        >
          {/* Information Flow: Inputs → Response/Actions */}
          <div className="flex gap-6">
            {/* Header + Inputs - Half the card */}
            <div className="flex-1 min-w-0">
              {/* Compact Header with Inputs Label */}
              <div className="flex items-center gap-2 mb-2">
                <div className={`w-1.5 h-1.5 rounded-full ${
                  iteration.hasError ? 'bg-red-500' : 'bg-blue-500'
                }`}></div>
                <span className="text-sm font-medium text-gray-700">
                  #{iterationNumber}
                </span>
                {sessionLabel && (
                  <span className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">
                    {sessionLabel}
                  </span>
                )}
                <span className="text-xs font-medium text-gray-500">Inputs</span>
              </div>
              {renderSensorPreviews(iteration.sensors, iteration.modelImages, iteration.id)}
            </div>
            
            {/* Flow Arrow */}
            <div className="flex flex-col items-center justify-center pt-8">
              {iteration.duration && (
                <>
                  <Clock className="w-4 h-4 text-gray-500" />
                  <div className="text-xs text-gray-500 mb-2">
                    {iteration.duration.toFixed(2)}s
                  </div>
                </>
              )}
              <ArrowRight className="w-5 h-5 text-gray-400" />
            </div>
            
            {/* Response and Actions - Half the card, stacked */}
            <div className="flex-1 min-w-0 space-y-4">
              {/* Processing (Model Response) */}
              <div>
                <div className="text-xs font-medium text-gray-500 mb-2">Response</div>
                {renderModelPreview(iteration.modelResponse)}
              </div>
              
              {/* Actions (Tools) */}
              <div>
                <div className="text-xs font-medium text-gray-500 mb-2">Actions</div>
                {renderToolIndicators(iteration.tools, iteration.id)}
              </div>
            </div>
          </div>
        </div>
      );
    });
  };

  // Main render
  if (currentIterations.length === 0 && historicalSessions.length === 0) {
    return (
      <div className="p-4 text-center text-gray-500">
        <HelpCircle className="mx-auto h-8 w-8 text-gray-400 mb-2" />
        <p className="font-medium">No activity yet.</p>
        <p className="text-sm">Start the agent to see its activity log here.</p>
      </div>
    );
  }

  // If an iteration is selected, show detailed view
  if (selectedIteration) {
    return (
      <div className="h-full flex flex-col">
        {/* Back button at very top left */}
        <div className="px-6 pt-2 pb-1">
          <button
            onClick={handleBackToList}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 text-sm"
          >
            <ArrowRight className="w-4 h-4 rotate-180" />
            Back
          </button>
        </div>

        {/* Detailed content */}
        <div className="flex-1 overflow-y-auto px-6 pb-6">
          <div className="max-w-4xl mx-auto space-y-8">
            {/* Prompt */}
            <div>
              <h2 className="text-xl font-medium text-gray-900 mb-4">Prompt</h2>
              
              {selectedIteration.modelPrompt && (
                <div className="bg-gray-50 border rounded-lg p-4 mb-6">
                  <pre className="text-sm text-gray-700 whitespace-pre-wrap">
                    {selectedIteration.modelPrompt}
                  </pre>
                </div>
              )}
              
              {selectedIteration.modelImages && selectedIteration.modelImages.length > 0 && (
                <div className="space-y-4">
                  {selectedIteration.modelImages.map((image, index) => (
                    <img 
                      key={index}
                      src={`data:image/png;base64,${image}`}
                      alt={`Image ${index + 1}`}
                      className="w-full rounded border object-contain"
                      onError={(e) => {
                        const imgElement = e.target as HTMLImageElement;
                        imgElement.src = `data:image/jpeg;base64,${image}`;
                      }}
                    />
                  ))}
                </div>
              )}
            </div>

            {/* Response */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-medium text-gray-900">Response</h2>
                {selectedIteration.duration && (
                  <div className="flex items-center gap-1 text-sm text-gray-500">
                    <Clock className="w-4 h-4" />
                    {selectedIteration.duration.toFixed(2)}s
                  </div>
                )}
              </div>
              <div className="bg-gray-50 border rounded-lg p-4">
                {selectedIteration.modelResponse ? (
                  <div className="text-sm text-gray-700 whitespace-pre-wrap">
                    {selectedIteration.modelResponse}
                  </div>
                ) : (
                  <div className="text-sm text-gray-500 italic">No response available</div>
                )}
              </div>
            </div>

            {/* Tools */}
            <div>
              <h2 className="text-xl font-medium text-gray-900 mb-4">Tools</h2>
              {selectedIteration.tools.length > 0 ? (
                <div className="space-y-3">
                  {selectedIteration.tools.map((tool, index) => {
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
              ) : (
                <div className="bg-gray-50 border rounded-lg p-4">
                  <div className="text-sm text-gray-500 italic">No tools used</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div
        data-scroll-container
        className="overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100"
        style={{ maxHeight }}
      >
        <div className="space-y-4 p-2">
          {/* Current Session */}
          {currentIterations.length > 0 && (
            <>
              <div className="flex items-center gap-2 mb-4">
                <div className="h-px bg-green-200 flex-grow"></div>
                <span className="text-sm font-medium text-green-700 bg-green-50 px-3 py-1 rounded-full">
                  Current Run
                </span>
                <div className="h-px bg-green-200 flex-grow"></div>
              </div>
              <div className="space-y-4">
                {renderIterations(currentIterations.slice(0, maxEntries).reverse())}
              </div>
            </>
          )}

          {/* Historical Sessions */}
          {historicalSessions.map((session, sessionIndex) => {
            const sessionDate = new Date(session.startTime);
            const sessionLabel = `Run ${sessionDate.toLocaleDateString()} ${sessionDate.toLocaleTimeString()}`;
            
            // Create relative time label
            let relativeLabel = '';
            if (sessionIndex === 0) {
              relativeLabel = 'Last run';
            } else if (sessionIndex === 1) {
              relativeLabel = 'Two runs ago';
            } else if (sessionIndex === 2) {
              relativeLabel = 'Three runs ago';
            } else {
              relativeLabel = `${sessionIndex + 1} runs ago`;
            }
            
            return (
              <div key={session.sessionId}>
                {/* Session Separator */}
                <div className="flex items-center gap-2 my-6">
                  <div className="h-px bg-gray-200 flex-grow"></div>
                  <span className="text-sm font-medium text-gray-500 bg-gray-50 px-3 py-1 rounded-full">
                    {relativeLabel} - {sessionLabel}
                  </span>
                  <div className="h-px bg-gray-200 flex-grow"></div>
                </div>
                
                {/* Session Iterations */}
                <div className="space-y-4">
                  {renderIterations(session.iterations.slice(0, maxEntries).reverse())}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default AgentLogViewer;
