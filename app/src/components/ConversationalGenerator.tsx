// src/components/ConversationalGenerator.tsx

import React, { useState, useRef } from 'react';
import { Send, Loader2, Save, Clipboard } from 'lucide-react'; // Added Clipboard
import { sendPrompt } from '@utils/sendApi';
import { CompleteAgent } from '@utils/agent_database';
import { extractAgentConfig, parseAgentResponse } from '@utils/agentParser';
import getConversationalSystemPrompt from '@utils/conversational_system_prompt';
import type { TokenProvider } from '@utils/main_loop';

// Define the shape of a message
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'ai' | 'system';
}

interface ConversationalGeneratorProps {
  onAgentGenerated: (agent: CompleteAgent, code: string) => void;
  getToken: TokenProvider;
  isAuthDisabled: boolean; 
}

const ConversationalGenerator: React.FC<ConversationalGeneratorProps> = ({ 
  onAgentGenerated, 
  getToken, 
  isAuthDisabled 
}) => {
  // We only need this state for the local/disabled mode
  const [copyButtonText, setCopyButtonText] = useState('Copy System Prompt');

  if (isAuthDisabled) {
    const handleCopyPrompt = async () => {
      const promptText = getConversationalSystemPrompt();
      try {
        await navigator.clipboard.writeText(promptText);
        setCopyButtonText('Copied to Clipboard!');
        setTimeout(() => setCopyButtonText('Copy System Prompt'), 2000); // Reset after 2 seconds
      } catch (err) {
        console.error('Failed to copy text: ', err);
        setCopyButtonText('Copy Failed!');
        setTimeout(() => setCopyButtonText('Copy System Prompt'), 2000);
      }
    };
    
    // Render a special view for local users
    return (
      <div className="flex flex-col h-[450px] bg-white rounded-b-xl border-x border-b border-indigo-200 shadow-md p-6 justify-center items-center text-center">
        <div className="max-w-md">
          <h4 className="text-lg font-semibold text-gray-800 mb-2">Manual Agent Generation</h4>
          <p className="text-sm text-gray-600 bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
            This AI builder uses a powerful cloud model. For local setups, we recommend copying the system prompt and using a large model (like Llama3:70b or via a web UI like ChatGPT/Claude/Gemini) to generate the agent configuration.
          </p>
          <button
            onClick={handleCopyPrompt}
            className="w-full px-4 py-2 bg-gradient-to-r from-indigo-500 to-blue-500 text-white rounded-lg hover:from-indigo-600 hover:to-blue-600 font-medium transition-all flex items-center justify-center shadow-sm"
          >
            <Clipboard className="h-4 w-4 mr-2" />
            {copyButtonText}
          </button>
        </div>
      </div>
    );
  }

  // --- ORIGINAL COMPONENT LOGIC FOR PRODUCTION/AUTHENTICATED MODE ---
  // This part only runs if isAuthDisabled is false.

  const [messages, setMessages] = useState<Message[]>([
    { id: 1, sender: 'ai', text: "Hi there! I'm Observer's agent builder. What would you like to create today?" }
  ]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const newUserMessage: Message = { id: Date.now(), sender: 'user', text: userInput };
    const updatedMessages = [...messages, newUserMessage];
    setMessages(updatedMessages);
    setUserInput('');
    setIsLoading(true);

    const conversationHistory = updatedMessages.map(msg => `${msg.sender}: ${msg.text}`).join('\n');
    const fullPrompt = `${getConversationalSystemPrompt()}\n${conversationHistory}\nai:`;

    try {
      const token = await getToken();
      const responseText = await sendPrompt(
          'api.observer-ai.com', 
          '443', 
          'gemini-2.0-flash-lite',
          { modifiedPrompt: fullPrompt, images: [] },
          token
      );

      const agentConfig = extractAgentConfig(responseText);
      
      if (agentConfig) {
        setMessages(prev => [...prev, {
          id: Date.now() + 1,
          sender: 'system',
          text: agentConfig
        }]);
      } else {
        setMessages(prev => [...prev, { 
          id: Date.now() + 1, 
          sender: 'ai', 
          text: responseText 
        }]);
      }
    } catch (err) {
      const errorText = err instanceof Error ? err.message : 'An unknown error occurred.';
      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'ai', text: `Sorry, I ran into an error: ${errorText}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConfigureAndSave = (configText: string) => {
    const parsed = parseAgentResponse(configText);
    if (parsed) {
      onAgentGenerated(parsed.agent, parsed.code);
    } else {
      setMessages(prev => [...prev, { id: Date.now(), sender: 'ai', text: "I'm sorry, there was an error parsing the final configuration. Could you try describing your agent again?" }]);
    }
  };

  return (
    <div className="flex flex-col h-[450px] bg-white rounded-b-xl border-x border-b border-indigo-200 shadow-md">
      {/* Chat Messages */}
      <div className="flex-1 p-4 space-y-4 overflow-y-auto">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.sender === 'system' ? (
              <div className="w-full bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-center">
                <p className="text-indigo-800 font-medium mb-3">I've generated your agent blueprint!</p>
                <button
                  onClick={() => handleConfigureAndSave(msg.text)}
                  className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg hover:from-green-600 hover:to-emerald-700 font-medium transition-colors flex items-center mx-auto"
                >
                  <Save className="h-4 w-4 mr-2" />
                  Configure & Save Agent
                </button>
              </div>
            ) : (
              <div className={`max-w-md p-3 rounded-lg ${msg.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'}`}>
                {msg.text}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
             <div className="bg-gray-200 text-gray-800 p-3 rounded-lg inline-flex items-center">
                <Loader2 className="h-5 w-5 animate-spin"/>
             </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-gray-200 bg-gray-50 flex items-center">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Type your message..."
          className="flex-1 p-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-700"
          disabled={isLoading}
        />
        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded-r-md hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
          disabled={isLoading || !userInput.trim()}
        >
          <Send className="h-5 w-5" />
        </button>
      </form>
    </div>
  );
};

export default ConversationalGenerator;
