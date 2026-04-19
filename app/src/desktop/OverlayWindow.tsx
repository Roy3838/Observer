import { useState, useEffect } from 'react'
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface OverlayMessage {
  id: string;
  content: string;
  timestamp: number;
}

function useOverlaySetup() {
  const [messages, setMessages] = useState<OverlayMessage[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeShortcuts, setActiveShortcuts] = useState<string[]>([]);
  const [pulsingShortcut, setPulsingShortcut] = useState<string | null>(null);

  // Set transparent background on html/body/root for this window
  useEffect(() => {
    document.documentElement.style.backgroundColor = 'transparent';
    document.body.style.backgroundColor = 'transparent';
    const root = document.getElementById('root');
    if (root) {
      root.style.backgroundColor = 'transparent';
    }
    return () => {
      document.documentElement.style.backgroundColor = '';
      document.body.style.backgroundColor = '';
      if (root) {
        root.style.backgroundColor = '';
      }
    };
  }, []);

  useEffect(() => {
    // Initial data fetch
    Promise.all([
      invoke<OverlayMessage[]>('get_overlay_messages'),
      invoke<string[]>('get_registered_shortcuts')
    ]).then(([messages, shortcuts]) => {
      setMessages(messages);
      setActiveShortcuts(shortcuts);
      setIsLoading(false);
    }).catch(error => {
      console.error('Failed to fetch initial data:', error);
      setIsLoading(false);
    });

    // Set up event listeners
    const setupListeners = async () => {
      const messageUnlisten = await listen<OverlayMessage[]>('overlay-messages-updated', (event) => {
        setMessages(event.payload);
        setIsLoading(false);
      });
      
      const shortcutUnlisten = await listen<string>('shortcut-pressed', (event) => {
        setPulsingShortcut(event.payload);
        setTimeout(() => setPulsingShortcut(null), 250);
      });
      
      return () => {
        messageUnlisten();
        shortcutUnlisten();
      };
    };
    
    let cleanup: (() => void) | null = null;
    setupListeners().then(cleanupFn => {
      cleanup = cleanupFn;
    });
    
    return () => cleanup?.();
  }, []);

  return { messages, isLoading, activeShortcuts, pulsingShortcut };
}

// Enhanced markdown renderer with proper code block support
function renderMarkdown(text: string): React.ReactNode {
  // Split text by code blocks first
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(text)) !== null) {
    // Add text before code block
    if (match.index > lastIndex) {
      const beforeText = text.slice(lastIndex, match.index);
      parts.push(renderInlineMarkdown(beforeText));
    }

    // Add code block
    const language = match[1] || 'text';
    const code = match[2];
    parts.push(
      <div key={match.index} className="my-2">
        <SyntaxHighlighter
          language={language}
          style={dracula}
          customStyle={{
            margin: 0,
            padding: '0.5rem',
            borderRadius: '0.375rem',
            fontSize: '0.75rem',
            backgroundColor: 'rgba(40, 42, 54, 0.8)',
            maxWidth: '100%',
            overflow: 'auto'
          }}
          wrapLongLines={true}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    );

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    const remainingText = text.slice(lastIndex);
    parts.push(renderInlineMarkdown(remainingText));
  }

  return parts.length > 1 ? <div>{parts}</div> : parts[0] || renderInlineMarkdown(text);
}

// Helper function for inline markdown (no code blocks)
function renderInlineMarkdown(text: string): React.ReactNode {
  if (!text) return null;
  
  const html = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
    .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
    .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>') // Inline code
    .replace(/\n/g, '<br>'); // Line breaks

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function OverlayWindow() {
  const { messages, isLoading, activeShortcuts, pulsingShortcut } = useOverlaySetup();

  // Helper function to check if a shortcut should pulse
  const shouldPulse = (shortcutKey: string) => {
    if (!pulsingShortcut) return false;
    // Match the raw shortcut key with the pulsing one
    return shortcutKey === pulsingShortcut;
  };

  // Helper function to get pulse classes
  const getPulseClasses = (shortcutKey: string, baseClasses: string) => {
    const isPulsing = shouldPulse(shortcutKey);
    return isPulsing 
      ? `${baseClasses} scale-110 shadow-lg shadow-white/50 border-white/60 transition-all duration-200 ease-out`
      : `${baseClasses} transition-all duration-200 ease-out`;
  };

  return (
    <>
      {/* CSS for inline code styling */}
      <style>
        {`
          .inline-code {
            background-color: rgba(40, 42, 54, 0.6);
            color: #f8f8f2;
            padding: 0.125rem 0.25rem;
            border-radius: 0.25rem;
            font-family: 'Fira Code', 'Courier New', monospace;
            font-size: 0.875em;
          }
        `}
      </style>
      <div 
        className="fixed inset-0 pointer-events-none"
        style={{ 
          backgroundColor: 'rgba(0, 0, 0, 0)',
          WebkitAppRegion: 'no-drag' 
        } as React.CSSProperties}
      >
        {/* Minimal Observer header with shortcuts - only show when there are messages */}
        {messages.length > 0 && (
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
            <div className="bg-black/70 backdrop-blur-xl rounded-md px-3 py-1.5 border border-white/20">
              <div className="text-white/70 text-xs font-mono font-medium flex items-center gap-2">
                <span className="text-white/90 font-bold">Observer</span>
                <span className="text-white/40">•</span>
                {(() => {
                  const moveShortcuts = activeShortcuts.filter(s => s.includes('move'));
                  const resizeShortcuts = activeShortcuts.filter(s => s.includes('resize'));
                  const toggleShortcuts = activeShortcuts.filter(s => s.includes('toggle') && !s.includes('agent'));
                  const agentShortcuts = activeShortcuts.filter(s => s.includes('toggle agent'));
                  
                  const formatKey = (keyPart: string) => keyPart
                    .replace('Cmd+', '⌘')
                    .replace('Alt+', '⌥')
                    .replace('Ctrl+', '^')
                    .replace('Arrow', '')
                    .replace('Up', '↑')
                    .replace('Down', '↓')
                    .replace('Left', '←')
                    .replace('Right', '→')
                    .replace('Shift+', '⇧');
                  
                  const isMovePulsing = moveShortcuts.some(s => shouldPulse(s.split(' ')[0]));
                  const isResizePulsing = resizeShortcuts.some(s => shouldPulse(s.split(' ')[0]));
                  const isTogglePulsing = toggleShortcuts.some(s => shouldPulse(s.split(' ')[0]));
                  
                  return (
                    <>
                      <span className={isMovePulsing 
                        ? "text-white/90 scale-110 transition-all duration-200 ease-out" 
                        : "text-white/60 transition-all duration-200 ease-out"}
                      >
                        ⌘↑↓←→ move
                      </span>
                      <span className="text-white/40">•</span>
                      <span className={isResizePulsing 
                        ? "text-white/90 scale-110 transition-all duration-200 ease-out" 
                        : "text-white/60 transition-all duration-200 ease-out"}
                      >
                        ⇧⌘↑↓←→ resize
                      </span>
                      <span className="text-white/40">•</span>
                      <span className={isTogglePulsing 
                        ? "text-white/90 scale-110 transition-all duration-200 ease-out" 
                        : "text-white/60 transition-all duration-200 ease-out"}
                      >
                        ⌘B toggle
                      </span>
                      {agentShortcuts.map((shortcut, _) => {
                        const parts = shortcut.split(' ');
                        const keyPart = parts[0] || '';
                        const formattedKey = formatKey(keyPart);
                        const isPulsing = shouldPulse(keyPart);
                        
                        return (
                          <>
                            <span className="text-white/40">•</span>
                            <span className={isPulsing 
                              ? "text-white/90 scale-110 transition-all duration-200 ease-out" 
                              : "text-white/60 transition-all duration-200 ease-out"}
                            >
                              {formattedKey}
                            </span>
                          </>
                        );
                      })}
                    </>
                  );
                })()}
              </div>
            </div>
          </div>
        )}

      <div className="h-full w-full flex items-start justify-start p-4 pt-16">
        {/* Compact Messages Container */}
        <div className="min-w-0 w-full max-w-[calc(100vw-8rem)]">
          {isLoading ? (
            <div className="bg-black/70 backdrop-blur-xl rounded-lg px-4 py-3 border border-white/20 shadow-xl">
              <div className="text-white/70 text-sm">Loading...</div>
            </div>
          ) : messages.length === 0 ? (
            // Welcome state - clean design with transparent black background
            <div className="fixed inset-0 bg-black/60 flex items-center justify-center p-8 z-10">
              <div className="text-center max-w-4xl w-full">
                {/* Big title */}
                <div className="mb-12">
                  <h1 className="text-6xl font-bold text-white tracking-tight mb-4">Observer Overlay</h1>
                  <p className="text-xl text-white/70 max-w-md mx-auto">
                    Your AI agents will display messages here
                  </p>
                </div>

                {/* Commands section */}
                <div className="space-y-8">
                  <div>
                    <h2 className="text-2xl font-semibold text-white mb-8">Available Commands</h2>
                    {activeShortcuts.length > 0 ? (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-5xl mx-auto">
                        {(() => {
                          // Group shortcuts by category
                          const overlayShortcuts = activeShortcuts.filter(s => s.includes('toggle') && !s.includes('agent'));
                          const moveShortcuts = activeShortcuts.filter(s => s.includes('move'));
                          const resizeShortcuts = activeShortcuts.filter(s => s.includes('resize'));
                          const agentShortcuts = activeShortcuts.filter(s => s.includes('toggle agent'));
                          
                          const formatKey = (keyPart: string) => keyPart
                            .replace('Cmd+', '⌘')
                            .replace('Alt+', '⌥')
                            .replace('Ctrl+', '^')
                            .replace('Arrow', '')
                            .replace('Up', '↑')
                            .replace('Down', '↓')
                            .replace('Left', '←')
                            .replace('Right', '→')
                            .replace('Shift+', '⇧');
                          
                          const renderWelcomeGroup = (shortcuts: string[], groupName: string) => {
                            if (shortcuts.length === 0) return null;
                            
                            if (groupName === 'Move' || groupName === 'Resize') {
                              // Show grouped format for move/resize - check if any of the group shortcuts are pulsing
                              const baseKey = shortcuts[0]?.split(' ')[0]?.replace(/Arrow(Up|Down|Left|Right)/, '') || '';
                              const formattedBase = formatKey(baseKey);
                              const isGroupPulsing = shortcuts.some(s => shouldPulse(s.split(' ')[0]));
                              
                              return (
                                <div className={isGroupPulsing 
                                  ? "text-center p-6 bg-black/30 rounded-lg border border-transparent scale-110 shadow-lg shadow-white/50 border-white/60 transition-all duration-200 ease-out"
                                  : "text-center p-6 bg-black/30 rounded-lg border border-transparent transition-all duration-200 ease-out"}>
                                  <div className="text-white text-2xl font-mono font-bold mb-3">{formattedBase}↑↓←→</div>
                                  <div className="text-white/80 text-base font-medium">{groupName} Overlay</div>
                                </div>
                              );
                            } else {
                              // Show individual shortcuts for toggle and agents
                              return shortcuts.map((shortcut, index) => {
                                const parts = shortcut.split(' ');
                                const keyPart = parts[0] || '';
                                const descriptionPart = parts.slice(1).join(' ') || '';
                                const formattedKey = formatKey(keyPart);
                                const cardClasses = getPulseClasses(keyPart, "text-center p-6 bg-black/30 rounded-lg border border-transparent");
                                
                                return (
                                  <div key={index} className={cardClasses}>
                                    <div className="text-white text-2xl font-mono font-bold mb-3">{formattedKey}</div>
                                    <div className="text-white/80 text-base font-medium capitalize">
                                      {descriptionPart.replace('toggle ', '').replace('toggle', 'Show/Hide')}
                                    </div>
                                  </div>
                                );
                              });
                            }
                          };
                          
                          return (
                            <>
                              {renderWelcomeGroup(overlayShortcuts, 'Toggle')}
                              {renderWelcomeGroup(moveShortcuts, 'Move')}
                              {renderWelcomeGroup(resizeShortcuts, 'Resize')}
                              {renderWelcomeGroup(agentShortcuts, 'Agents')}
                            </>
                          );
                        })()}
                      </div>
                    ) : (
                      <div className="text-white/50 text-lg">Loading commands...</div>
                    )}
                  </div>
                </div>

              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {/* Latest message (largest) */}
              <div className="bg-black/70 backdrop-blur-xl rounded-lg px-4 py-3 border border-white/20 animate-in slide-in-from-bottom-2 duration-300">
                <div className="flex items-start justify-between mb-1">
                  <div className="text-white/40 text-xs font-mono">
                    {formatTime(messages[messages.length - 1].timestamp)}
                  </div>
                  <div className="flex items-center space-x-2">
                    {messages.length > 1 && (
                      <div className="text-white/30 text-xs bg-white/5 px-1.5 py-0.5 rounded text-center min-w-[1rem]">
                        {messages.length}
                      </div>
                    )}
                  </div>
                </div>
                <div className="text-white/90 text-xs leading-relaxed max-w-none">
                  {renderMarkdown(messages[messages.length - 1].content)}
                </div>
              </div>

              {/* Previous messages (smaller, if any) */}
              {messages.length > 1 && (
                <div className="space-y-1 max-h-32 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                  {messages.slice(-6, -1).reverse().map((message) => (
                    <div
                      key={message.id}
                      className="bg-black/50 backdrop-blur-lg rounded-md px-2 py-1 border border-white/5 opacity-75 hover:opacity-100 transition-opacity"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-white/30 text-xs font-mono">
                          {formatTime(message.timestamp)}
                        </div>
                      </div>
                      <div className="text-white/70 text-xs leading-tight line-clamp-2">
                        {renderMarkdown(message.content.slice(0, 100) + (message.content.length > 100 ? '...' : ''))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      </div>
    </>
  );
}
