// src/components/AICreator/MultiAgentCreator.tsx

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Save, Users, Cpu, Plus, FlaskConical} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { ModelManager } from '@utils/ModelManager';
import { CompleteAgent, updateAgentImageMemory, saveAgent } from '@utils/agent_database';
import {
  extractMultipleAgentConfigs,
  parseAgentResponse,
  extractImageRequest,
  extractDetectionModeTemplate,
  extractAgentReferencesWithPositions,
  extractAgentReferences,
  fetchAgentReferenceData,
  formatAgentReferenceContext,
  AgentReferenceData
} from '@utils/agentParser';
import MediaUploadMessage from '../MediaUploadMessage';
import getMultiAgentSystemPrompt from '@utils/multi_agent_creator';
import type { TokenProvider } from '@utils/main_loop';
import AgentReferenceModal from './AgentReferenceModal';
import { AgentAutocompleteInput } from './AgentAutocompleteInput';
import LocalWarning from './LocalWarning';
import { DetectionModePanel, type DetectionCategory, type DetectionModeInitialData, type ClassifiedImage } from './DetectionMode';

// ===================================================================================
//  MULTI-AGENT PREVIEW COMPONENT
// ===================================================================================
interface MultiAgentPreviewProps {
  configsJson: string;
  onSaveAll: (configsJson: string) => void;
}

const MultiAgentPreview: React.FC<MultiAgentPreviewProps> = ({ configsJson, onSaveAll }) => {
  try {
    const agentConfigs = JSON.parse(configsJson) as string[];
    const previews = agentConfigs.map((config, _) => {
      const parsed = parseAgentResponse(config);
      return parsed ? { agent: parsed.agent, code: parsed.code } : null;
    }).filter(Boolean);

    if (previews.length === 0) {
      return (
        <div className="w-full bg-red-50 border border-red-200 rounded-lg p-3 md:p-4 text-center">
          <p className="text-red-800 font-medium">Error parsing agent configurations</p>
        </div>
      );
    }

    return (
      <div className="w-full bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-4 md:p-6">
        <div className="text-center mb-4">
          <div className="flex items-center justify-center mb-2">
            <Users className="h-6 w-6 text-purple-600 mr-2" />
            <h3 className="text-lg font-bold text-purple-800">
              🎉 I've created {previews.length} coordinated agents!
            </h3>
          </div>
          <p className="text-purple-700 text-sm">
            These agents will work together to accomplish your task
          </p>
        </div>

        <div className="space-y-4 mb-6">
          {previews.map((preview, index) => (
            <div key={index} className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center mb-3">
                <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center mr-3">
                  <span className="text-purple-600 font-bold text-sm">{index + 1}</span>
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900">{preview!.agent.name}</h4>
                  <p className="text-sm text-gray-600">{preview!.agent.description}</p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div>
                  <h5 className="font-medium text-gray-700 mb-2">System Prompt:</h5>
                  <div className="bg-gray-50 border rounded p-2 max-h-20 overflow-y-auto">
                    <code className="text-gray-600 whitespace-pre-wrap text-xs">
                      {preview!.agent.system_prompt.substring(0, 200)}
                      {preview!.agent.system_prompt.length > 200 ? '...' : ''}
                    </code>
                  </div>
                </div>
                <div>
                  <h5 className="font-medium text-gray-700 mb-2">Code:</h5>
                  <div className="bg-gray-50 border rounded p-2 max-h-20 overflow-y-auto">
                    <code className="text-gray-600 whitespace-pre-wrap text-xs">
                      {preview!.code.substring(0, 200)}
                      {preview!.code.length > 200 ? '...' : ''}
                    </code>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="text-center">
          <button
            onClick={() => onSaveAll(configsJson)}
            className="px-6 py-3 text-base bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg hover:from-purple-700 hover:to-indigo-700 font-medium transition-colors flex items-center mx-auto shadow-lg"
          >
            <Save className="h-5 w-5 mr-2" />
            {previews.length === 1 ? 'Save Agent' : `Save All ${previews.length} Agents`}
          </button>
        </div>
      </div>
    );
  } catch (error) {
    return (
      <div className="w-full bg-red-50 border border-red-200 rounded-lg p-3 md:p-4 text-center">
        <p className="text-red-800 font-medium">Error parsing agent configurations</p>
      </div>
    );
  }
};


// ===================================================================================
//  MESSAGE CONTENT WITH AGENT BADGES
// ===================================================================================
interface MessageContentProps {
  text: string;
  onAgentClick: (agentId: string) => void;
}

const MessageContent: React.FC<MessageContentProps> = ({ text, onAgentClick }) => {
  const agentRefs = extractAgentReferencesWithPositions(text);

  if (agentRefs.length === 0) {
    return (
      <div className="prose prose-sm max-w-none">
        <ReactMarkdown
          components={{
            ul: ({children}) => <ul className="list-disc pl-4 space-y-1 mb-2">{children}</ul>,
            ol: ({children}) => <ol className="list-decimal pl-4 space-y-1 mb-2">{children}</ol>,
            li: ({children}) => <li className="text-inherit">{children}</li>,
            strong: ({children}) => <strong className="font-semibold">{children}</strong>,
            code: ({children}) => <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">{children}</code>,
            p: ({children}) => <p className="mb-2 last:mb-0">{children}</p>
          }}
        >
          {text}
        </ReactMarkdown>
      </div>
    );
  }

  // Replace @references with agent badges
  return (
    <div className="prose prose-sm max-w-none">
      <div className="mb-2 last:mb-0">
        {renderTextWithAgentBadges(text, agentRefs, onAgentClick)}
      </div>
    </div>
  );
};

function renderTextWithAgentBadges(
  text: string,
  agentRefs: Array<{
    reference: { agentId: string; runCount: number };
    start: number;
    end: number;
    fullMatch: string;
  }>,
  onAgentClick: (agentId: string) => void
) {
  const elements: React.ReactNode[] = [];
  let lastIndex = 0;

  agentRefs.forEach((ref, i) => {
    // Add text before this reference (with markdown parsing)
    if (ref.start > lastIndex) {
      const beforeText = text.substring(lastIndex, ref.start);
      elements.push(
        <ReactMarkdown
          key={`text-${i}`}
          components={{
            p: ({children}) => <span>{children}</span>,
            strong: ({children}) => <strong className="font-semibold">{children}</strong>,
            code: ({children}) => <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">{children}</code>
          }}
        >
          {beforeText}
        </ReactMarkdown>
      );
    }

    // Add clickable agent badge
    elements.push(
      <button
        key={`agent-${i}`}
        onClick={() => onAgentClick(ref.reference.agentId)}
        className="inline-flex items-center px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-sm font-medium hover:bg-purple-200 transition-colors mx-1"
      >
        <Users className="h-4 w-4 mr-1" />
        {ref.reference.agentId}
        {ref.reference.runCount !== 3 && (
          <span className="ml-1 text-xs opacity-75">#{ref.reference.runCount}</span>
        )}
      </button>
    );

    lastIndex = ref.end;
  });

  // Add remaining text (with markdown parsing)
  if (lastIndex < text.length) {
    const remainingText = text.substring(lastIndex);
    elements.push(
      <ReactMarkdown
        key="text-end"
        components={{
          p: ({children}) => <span>{children}</span>,
          strong: ({children}) => <strong className="font-semibold">{children}</strong>,
          code: ({children}) => <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">{children}</code>
        }}
      >
        {remainingText}
      </ReactMarkdown>
    );
  }

  return elements;
}

// ===================================================================================
//  MAIN MULTI-AGENT CREATOR COMPONENT
// ===================================================================================
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'ai' | 'system' | 'image-request' | 'multi-agent-system' | 'detection-mode';
  imageDatas?: string[];
  isStreaming?: boolean;
  detectionModeData?: DetectionModeInitialData;
}

interface MultiAgentCreatorProps {
  getToken: TokenProvider;
  isAuthenticated: boolean;
  isUsingObServer: boolean;
  isPro?: boolean;
  onSignIn?: () => void;
  onSwitchToObServer?: () => void;
  onUpgrade?: () => void;
  onRefresh?: () => void;
  onSaveComplete?: () => void; // Called after agents are saved and refreshed
  initialMessage?: string;
}

const MultiAgentCreator: React.FC<MultiAgentCreatorProps> = ({
  getToken,
  isAuthenticated,
  isUsingObServer,
  isPro = false,
  onSignIn,
  onSwitchToObServer,
  onUpgrade,
  onRefresh,
  onSaveComplete,
  initialMessage
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      sender: 'ai',
      text: `Hi there! I'm Observer's **Multi-Agent Builder**. I specialize in creating teams of coordinated agents that work together to accomplish complex tasks. For example:

* **Monitor & Document** 🤖 - One agent watches your screen, another documents processes
* **Extract & Solve** 🔍 - One agent reads problems from screen, another solves them
* **Watch & Guide** 👀 - One agent observes, another provides step-by-step guidance

What kind of agent team would you like me to create today?`
    }
  ]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const hasInitialMessageSent = useRef(false);

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Set initial message in input without auto-sending
  useEffect(() => {
    if (initialMessage && !hasInitialMessageSent.current) {
      hasInitialMessageSent.current = true;
      setUserInput(initialMessage);
    }
  }, []); // Empty dependency array - only run once on mount

  // Modal state for agent reference details
  const [selectedAgentModal, setSelectedAgentModal] = useState<{
    isOpen: boolean;
    agentData: AgentReferenceData | null;
  }>({ isOpen: false, agentData: null });

  // --- STATE FOR MODAL AND LOCAL MODEL SELECTION ---
  const [isLocalModalOpen, setIsLocalModalOpen] = useState(false);
  const [selectedLocalModel, setSelectedLocalModel] = useState('');
  const [previewImages, setPreviewImages] = useState<string[]>([]);

  const sendConversation = async (allMessages: Message[]) => {
    setIsLoading(true);

    // Build proper OpenAI messages array
    const openaiMessages: Array<{role: string, content: any}> = [
      // System message first - static, no agent context
      { role: "system", content: getMultiAgentSystemPrompt() }
    ];

    // Convert conversation messages to OpenAI format with per-message agent references
    for (const msg of allMessages) {
      // Skip non-conversational message types
      if (msg.sender === 'system' || msg.sender === 'image-request' || msg.sender === 'multi-agent-system') {
        continue;
      }

      // Map sender to OpenAI role
      const role = msg.sender === 'user' ? 'user' : 'assistant';

      // For user messages, check for agent references and append context
      let messageContent = msg.text;
      let messageImages: string[] = [];

      if (msg.sender === 'user') {
        // Extract agent references from this specific message
        const referencesInMessage = extractAgentReferences(msg.text);

        if (referencesInMessage.length > 0) {
          try {
            // Fetch agent data for references in this message
            const referenceData = await fetchAgentReferenceData(referencesInMessage);

            // Append agent context to the message
            const agentContext = formatAgentReferenceContext(referenceData);
            messageContent = msg.text + agentContext;

            // Also collect iteration images from referenced agents
            referenceData.forEach(data => {
              data.recentRuns.forEach(run => {
                // Add model images (base64 images sent to model)
                if (run.modelImages) {
                  messageImages.push(...run.modelImages);
                }

                // Add screenshot and camera sensor images
                run.sensors.forEach(sensor => {
                  if ((sensor.type === 'screenshot' || sensor.type === 'camera') && sensor.content) {
                    // Handle both base64 and raw image data
                    if (typeof sensor.content === 'string') {
                      messageImages.push(sensor.content);
                    } else if (sensor.content.data) {
                      messageImages.push(sensor.content.data);
                    }
                  }
                });
              });
            });
          } catch (error) {
            console.error('Failed to fetch agent references:', error);
          }
        }
      }

      // Build content with images if present
      if ((msg.imageDatas && msg.imageDatas.length > 0) || messageImages.length > 0) {
        const contentParts: any[] = [
          { type: "text", text: messageContent }
        ];

        // Add user-uploaded images
        if (msg.imageDatas) {
          msg.imageDatas.forEach(imageData => {
            contentParts.push({
              type: "image_url",
              image_url: {
                url: `data:image/png;base64,${imageData}`
              }
            });
          });
        }

        // Add iteration images from referenced agents
        messageImages.forEach(imageBase64Data => {
          contentParts.push({
            type: "image_url",
            image_url: {
              url: `data:image/png;base64,${imageBase64Data}`
            }
          });
        });

        openaiMessages.push({
          role: role,
          content: contentParts
        });
      } else {
        // Text-only message
        openaiMessages.push({
          role: role,
          content: messageContent
        });
      }
    }

    const streamingMessageId = Date.now() + Math.random();
    const streamingMessage: Message = {
      id: streamingMessageId,
      text: '',
      sender: 'ai',
      isStreaming: true
    };
    setMessages(prev => [...prev, streamingMessage]);

    let accumulatedResponse = '';

    try {
      let responseText: string;
      if (isUsingObServer) {
        // --- CLOUD PATH ---
        const token = await getToken();
        if (!token) throw new Error("Authentication failed.");

        responseText = await ModelManager.getInstance().sendMessages(
          'gemini-2.5-flash-lite',
          openaiMessages,
          token,
          true,
          (chunk: string) => {
            accumulatedResponse += chunk;
            setMessages(prev => prev.map(msg =>
              msg.id === streamingMessageId
                ? { ...msg, text: accumulatedResponse }
                : msg
            ));
          }
        );
      } else {
        // --- LOCAL PATH ---
        responseText = await ModelManager.getInstance().sendMessages(
          selectedLocalModel,
          openaiMessages,
          undefined,
          true,
          (chunk: string) => {
            accumulatedResponse += chunk;
            setMessages(prev => prev.map(msg =>
              msg.id === streamingMessageId
                ? { ...msg, text: accumulatedResponse }
                : msg
            ));
          }
        );
      }

      // Convert streaming message to final message
      setMessages(prev => prev.map(msg =>
        msg.id === streamingMessageId
          ? { ...msg, isStreaming: false }
          : msg
      ));

      // Check for multi-agent configs
      const agentConfigs = extractMultipleAgentConfigs(responseText);
      if (agentConfigs.length > 0) {
        const textOutsideBlocks = responseText.replace(/\$\$\$\s*\n?[\s\S]*?\n?\$\$\$/g, '').trim();

        if (textOutsideBlocks) {
          setMessages(prev => prev.map(msg =>
            msg.id === streamingMessageId
              ? { ...msg, text: textOutsideBlocks }
              : msg
          ));
        } else {
          setMessages(prev => prev.filter(msg => msg.id !== streamingMessageId));
        }

        setMessages(prev => [...prev, {
          id: Date.now() + Math.random() * 1000,
          text: JSON.stringify(agentConfigs),
          sender: 'multi-agent-system',
        }]);
      } else {
        // Check for image request
        const imageRequest = extractImageRequest(responseText);
        if (imageRequest) {
          const textOutsideBlocks = responseText.replace(/%%%\s*\n?[\s\S]*?\n?%%%/g, '').trim();

          if (textOutsideBlocks) {
            setMessages(prev => prev.map(msg =>
              msg.id === streamingMessageId
                ? { ...msg, text: textOutsideBlocks }
                : msg
            ));
          } else {
            setMessages(prev => prev.filter(msg => msg.id !== streamingMessageId));
          }

          setMessages(prev => [...prev, {
            id: Date.now() + Math.random() * 1000,
            text: imageRequest,
            sender: 'image-request',
          }]);
        } else {
          // Check for detection mode template
          const detectionTemplate = extractDetectionModeTemplate(responseText);
          if (detectionTemplate) {
            const textOutsideBlocks = responseText.replace(/&&&\s*\n?[\s\S]*?\n?&&&/g, '').trim();

            if (textOutsideBlocks) {
              setMessages(prev => prev.map(msg =>
                msg.id === streamingMessageId
                  ? { ...msg, text: textOutsideBlocks }
                  : msg
              ));
            } else {
              setMessages(prev => prev.filter(msg => msg.id !== streamingMessageId));
            }

            // Find the most recent user message with @reference to get the agent ID
            const recentUserMessages = allMessages.filter(m => m.sender === 'user').reverse();
            let sourceAgentId = '';
            let messageImages: string[] = [];

            for (const userMsg of recentUserMessages) {
              const references = extractAgentReferences(userMsg.text);
              if (references.length > 0) {
                sourceAgentId = references[0].agentId;

                // Fetch images from the referenced agent
                try {
                  const referenceData = await fetchAgentReferenceData(references);
                  referenceData.forEach(data => {
                    data.recentRuns.forEach(run => {
                      // Add model images
                      if (run.modelImages) {
                        messageImages.push(...run.modelImages);
                      }
                      // Add screenshot and camera sensor images
                      run.sensors.forEach(sensor => {
                        if ((sensor.type === 'screenshot' || sensor.type === 'camera') && sensor.content) {
                          if (typeof sensor.content === 'string') {
                            messageImages.push(sensor.content);
                          } else if (sensor.content.data) {
                            messageImages.push(sensor.content.data);
                          }
                        }
                      });
                    });
                  });
                } catch (error) {
                  console.error('Failed to fetch agent reference images:', error);
                }
                break;
              }
            }

            // Build DetectionCategory[] from the template and collected images
            const categoryColors = ['cat1', 'cat2', 'cat3', 'cat4'];
            const categories: DetectionCategory[] = detectionTemplate.categories.map((label, idx) => {
              const categoryId = categoryColors[idx] || `cat${idx + 1}`;
              const imageIndices = detectionTemplate.categorization[label] || [];

              // Map 1-based indices to images
              const categoryImages: ClassifiedImage[] = imageIndices
                .filter(index => index >= 1 && index <= messageImages.length)
                .map((index, imgIdx) => ({
                  id: `${categoryId}-img-${imgIdx}-${Date.now()}`,
                  data: messageImages[index - 1], // Convert 1-based to 0-based
                  source: 'history' as const,
                  sourceAgentId: sourceAgentId,
                  timestamp: new Date().toISOString()
                }));

              return {
                id: categoryId,
                label: label.toUpperCase(),
                images: categoryImages
              };
            });

            const detectionModeData: DetectionModeInitialData = {
              categories,
              sourceAgentId
            };

            setMessages(prev => [...prev, {
              id: Date.now() + Math.random() * 1000,
              text: '',
              sender: 'detection-mode',
              detectionModeData
            }]);
          }
        }
      }

    } catch (err) {
      setMessages(prev => prev.filter(msg => msg.id !== streamingMessageId));
      const errorText = err instanceof Error ? err.message : 'An unknown error occurred.';
      setMessages(prev => [...prev, { id: Date.now() + Math.random() * 1000, sender: 'ai', text: `Sorry, I ran into an error: ${errorText}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userInput.trim() && previewImages.length === 0) return; // Allow send with just images
    if (isLoading) return;

    // Guard against submission if local model isn't selected in local mode
    if (!isUsingObServer && !selectedLocalModel) {
        setIsLocalModalOpen(true); // Prompt user to select a model
        return;
    }

    // Create message with optional image data
    const newUserMessage: Message = {
      id: Date.now() + Math.random() * 1000,
      sender: 'user',
      text: userInput.trim() || (previewImages.length > 0 ? `[${previewImages.length} image${previewImages.length > 1 ? 's' : ''}]` : ''),
      ...(previewImages.length > 0 && { imageDatas: previewImages })
    };

    setMessages(prev => [...prev, newUserMessage]);
    setUserInput('');
    setPreviewImages([]); // Clear previews after sending

    const allMessages = [...messages, newUserMessage];
    await sendConversation(allMessages);
  };

  const handleManualImageUpload = () => {
    const imageRequestMessage: Message = {
      id: Date.now() + Math.random() * 1000,
      text: "Please upload an image to help create this agent",
      sender: 'image-request',
    };
    setMessages(prev => [...prev, imageRequestMessage]);
  };

  const handleOpenDetectionMode = () => {
    const detectionMessage: Message = {
      id: Date.now() + Math.random() * 1000,
      text: '',
      sender: 'detection-mode',
    };
    setMessages(prev => [...prev, detectionMessage]);
  };

  const handleDetectionModeClose = (messageId: number) => {
    setMessages(prev => prev.filter(msg => msg.id !== messageId));
  };

  const handleDetectionModeComplete = (messageId: number, agentBlock: string) => {
    // Remove the detection mode panel
    setMessages(prev => prev.filter(msg => msg.id !== messageId));

    // Show in multi-agent-system message (reuses existing preview/save flow)
    setMessages(prev => [...prev, {
      id: Date.now() + Math.random() * 1000,
      text: JSON.stringify([agentBlock]),  // Wrap in array like extractMultipleAgentConfigs returns
      sender: 'multi-agent-system',
    }]);
  };

  const handleSaveAllAgents = async (configsJson: string) => {
    try {
      const agentConfigs = JSON.parse(configsJson) as string[];
      const savedAgents: CompleteAgent[] = [];

      for (const configText of agentConfigs) {
        const parsed = parseAgentResponse(configText);
        if (parsed) {
          try {
            const savedAgent = await saveAgent(parsed.agent, parsed.code);
            savedAgents.push(savedAgent);

            const images = messages
              .filter(msg => msg.imageDatas && msg.imageDatas.length > 0)
              .flatMap(msg => msg.imageDatas!);

            if (images.length > 0) {
              await updateAgentImageMemory(parsed.agent.id, images);
            }
          } catch (saveError) {
            console.error(`Failed to save agent ${parsed.agent.id}:`, saveError);
            setMessages(prev => [...prev, {
              id: Date.now() + Math.random() * 1000,
              sender: 'ai',
              text: `Error saving agent "${parsed.agent.name}": ${saveError instanceof Error ? saveError.message : 'Unknown error'}`
            }]);
            return;
          }
        }
      }

      if (savedAgents.length > 0) {
        setMessages(prev => [...prev, {
          id: Date.now() + Math.random() * 1000,
          sender: 'ai',
          text: `🎉 Successfully saved ${savedAgents.length} ${savedAgents.length === 1 ? 'agent' : 'agents'}: ${savedAgents.map(a => a.name).join(', ')}!\n\nYour agent ${savedAgents.length === 1 ? 'is' : 'team is'} ready to work together!`
        }]);

        // Refresh the agent list
        if (onRefresh) {
          await onRefresh();
        }

        // Notify parent that save is complete (modal will close, App.tsx handles tutorial)
        if (onSaveComplete) {
          onSaveComplete();
        }
      } else {
        setMessages(prev => [...prev, { id: Date.now() + Math.random() * 1000, sender: 'ai', text: "I'm sorry, there was an error parsing the agents. Could you try again?" }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { id: Date.now() + Math.random() * 1000, sender: 'ai', text: "I'm sorry, there was an error processing the agents. Could you try again?" }]);
    }
  };

  const handleAgentBadgeClick = async (agentId: string) => {
    // Fetch the agent data
    let agentData: AgentReferenceData;
    try {
      const references = [{ agentId, runCount: 3 }];
      const referenceData = await fetchAgentReferenceData(references);
      agentData = referenceData[0];
    } catch (error) {
      console.error('Failed to fetch agent data for modal:', error);
      // Create a placeholder for missing agent
      agentData = {
        reference: { agentId, runCount: 3 },
        agent: null,
        code: null,
        memory: '',
        recentRuns: []
      };
    }

    setSelectedAgentModal({ isOpen: true, agentData });
  };

  const handleMediaResponse = async (messageId: number, result: string | { type: 'image', data: string } | null) => {
    if (result === null) {
      setMessages(prev => prev.filter(msg => msg.id !== messageId));
      return;
    }
    const newUserMessage: Message = typeof result === 'string'
      ? { id: Date.now() + Math.random() * 1000, sender: 'user', text: result }
      : { id: Date.now() + Math.random() * 1000, sender: 'user', text: '[Image uploaded]', imageDatas: [result.data] };

    setMessages(prev => [...prev.filter(msg => msg.id !== messageId), newUserMessage]);

    const currentMessages = messages.filter(msg => msg.id !== messageId);
    const allMessages = [...currentMessages, newUserMessage];
    await sendConversation(allMessages);
  };

  const getPlaceholderText = () => {
    if (!isPro && isUsingObServer) {
      return "Upgrade to Pro to use AI-Studio";
    }
    if (isUsingObServer) {
      return isAuthenticated ? "Describe the agent team you want to build..." : "Enable Ob-Server and log in to use Multi-Agent Builder";
    }
    return selectedLocalModel ? "Describe the agent team to build with your model..." : "Click the CPU icon to select a local model";
  };

  const isInputDisabled = (isUsingObServer && !isPro) || isLoading || (isUsingObServer ? !isAuthenticated : !selectedLocalModel);
  const isSendDisabled = isInputDisabled || (!userInput.trim() && previewImages.length === 0);

  return (
    <>
      <div className={`flex flex-col h-[350px] md:h-[450px] bg-white rounded-lg border border-purple-200 relative`}>
        {/* Greyed out background layer */}
        {!isPro && isUsingObServer && (
          <div className="absolute inset-0 z-[5] bg-white opacity-60 pointer-events-none" />
        )}

        {/* Pro Feature Overlay */}
        {!isPro && isUsingObServer && (
          <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none">
            <div className="bg-white rounded-lg shadow-2xl p-6 max-w-md mx-4 border-2 border-purple-300 pointer-events-auto">
              <div className="text-center">
                <div className="flex items-center justify-center mb-3">
                  <div className="bg-purple-100 rounded-full p-3">
                    <Users className="h-8 w-8 text-purple-600" />
                  </div>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">
                  🔒 AI-Studio is a Pro Feature
                </h3>
                <p className="text-gray-600 mb-4">
                  Upgrade to Pro to create and edit coordinated agent teams with AI collaboration
                </p>
                <button
                  onClick={onUpgrade}
                  className="w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg hover:from-purple-700 hover:to-indigo-700 font-medium transition-colors shadow-lg"
                >
                  Upgrade to Pro
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Chat Messages Area */}
        <div className="flex-1 p-3 md:p-4 space-y-3 md:space-y-4 overflow-y-auto">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.sender === 'multi-agent-system' ? (
                <MultiAgentPreview configsJson={msg.text} onSaveAll={handleSaveAllAgents} />
              ) : msg.sender === 'image-request' ? (
                <div className="w-full">
                  <MediaUploadMessage
                    requestText={msg.text}
                    onResponse={(result) => handleMediaResponse(msg.id, result)}
                  />
                </div>
              ) : msg.sender === 'detection-mode' ? (
                <div className="w-full">
                  <DetectionModePanel
                    onClose={() => handleDetectionModeClose(msg.id)}
                    onComplete={(agentBlock) => handleDetectionModeComplete(msg.id, agentBlock)}
                    initialData={msg.detectionModeData}
                    getToken={getToken}
                    isUsingObServer={isUsingObServer}
                    selectedLocalModel={selectedLocalModel}
                  />
                </div>
              ) : (
                <div className={`max-w-xs md:max-w-md p-2 md:p-3 rounded-lg text-sm md:text-base ${
                  msg.sender === 'user'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gradient-to-br from-purple-50 to-indigo-50 text-gray-800 shadow-sm'
                } ${msg.isStreaming ? 'animate-pulse' : ''}`}>
                  {msg.imageDatas && msg.imageDatas.length > 0 ? (
                    <div className="space-y-2">
                      <MessageContent text={msg.text} onAgentClick={handleAgentBadgeClick} />
                      <div className="flex flex-wrap gap-2">
                        {msg.imageDatas.map((imageData, idx) => (
                          <img
                            key={idx}
                            src={`data:image/png;base64,${imageData}`}
                            alt={`Uploaded image ${idx + 1}`}
                            className="max-w-full h-auto rounded-lg"
                          />
                        ))}
                      </div>
                    </div>
                  ) : (
                    <MessageContent text={msg.text} onAgentClick={handleAgentBadgeClick} />
                  )}
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gradient-to-br from-purple-50 to-indigo-50 text-gray-800 p-2 md:p-3 rounded-lg text-sm md:text-base inline-flex items-center shadow-sm">
                <Loader2 className="h-4 w-4 md:h-5 md:w-5 animate-spin"/>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-2 border-t border-purple-200 bg-white/80 backdrop-blur-sm rounded-b-lg">
          <form onSubmit={handleSubmit} className="flex items-center gap-2">
            <AgentAutocompleteInput
              value={userInput}
              onChange={setUserInput}
              placeholder={getPlaceholderText()}
              disabled={isInputDisabled}
              onImagePaste={(image) => setPreviewImages(prev => [...prev, image])}
              previewImages={previewImages}
              onRemovePreview={(index) => setPreviewImages(prev => prev.filter((_, i) => i !== index))}
              className="flex-1 p-2 md:p-3 border border-purple-300 rounded-lg text-sm md:text-base text-gray-700 disabled:bg-gray-100 disabled:cursor-not-allowed focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            />

            <button
              type="button"
              onClick={handleManualImageUpload}
              disabled={isLoading}
              className="p-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-300 transition-colors flex items-center"
              title="Upload Image"
            >
              <Plus className="h-5 w-5" />
            </button>

            <button
              type="button"
              onClick={handleOpenDetectionMode}
              disabled={isLoading}
              className="p-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-300 transition-colors flex items-center"
              title="Detection Mode"
            >
              <FlaskConical className="h-5 w-5" />
            </button>

            {!isUsingObServer && (
              <button
                type="button"
                onClick={() => setIsLocalModalOpen(true)}
                className="p-2 bg-slate-700 text-white rounded-md hover:bg-slate-800 transition-colors flex items-center"
                title="Configure Local Model"
              >
                <Cpu className="h-5 w-5" />
              </button>
            )}

            <button
              type="submit"
              className="p-2 md:p-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-300 transition-colors flex items-center flex-shrink-0"
              disabled={isSendDisabled}
              title="Send"
            >
              <Send className="h-4 w-4" />
            </button>
          </form>
        </div>
      </div>

      <AgentReferenceModal
        isOpen={selectedAgentModal.isOpen}
        onClose={() => setSelectedAgentModal({ isOpen: false, agentData: null })}
        agentData={selectedAgentModal.agentData}
      />

      <LocalWarning
        isOpen={isLocalModalOpen}
        onClose={() => setIsLocalModalOpen(false)}
        currentModel={selectedLocalModel}
        onSelectModel={setSelectedLocalModel}
        onSignIn={onSignIn}
        onSwitchToObServer={onSwitchToObServer}
        isAuthenticated={isAuthenticated}
        featureType="multiagent"
      />
    </>
  );
};

export default MultiAgentCreator;
