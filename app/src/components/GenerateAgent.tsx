// src/components/GenerateAgent.tsx
import React, { useState } from 'react';
import { Loader2, Zap, XCircle, Save } from 'lucide-react';
import { ModelManager } from '@utils/ModelManager';
import { CompleteAgent, saveAgent } from '@utils/agent_database';
import EditAgentModal from './EditAgent/EditAgentModal';
import getSystemPrompt from '@utils/system_prompt';
import getPythonSystemPrompt from '@utils/python_system_prompt';
// Removed getOllamaServerAddress import - no longer needed
import type { TokenProvider } from '@utils/main_loop';

const PrettyAgentResponse: React.FC<{ responseText: string; isLoading: boolean }> = ({
  responseText,
  isLoading
}) => {
  const [formattedResponse, setFormattedResponse] = useState<React.ReactNode>("");

  React.useEffect(() => {
    if (!responseText) return;
    const agentFileRegex = /\$\$\$\s*\n?([\s\S]*?)\n?\$\$\$/;
    const match = responseText.match(agentFileRegex);
    if (match && match[1] && match.index !== undefined) {
      const beforeContent = responseText.slice(0, match.index).trim();
      const agentFileContent = match[1];
      const afterContent = responseText.slice(match.index + match[0].length).trim();
      const formattedAgentContent = formatAgentFile(agentFileContent);
      setFormattedResponse(
        <>
          {beforeContent && <div className="mb-4 text-gray-700">{beforeContent}</div>}
          <div className="agent-file-container">{formattedAgentContent}</div>
          {afterContent && <div className="mt-4 text-gray-700">{afterContent}</div>}
        </>
      );
    } else {
      setFormattedResponse(<div className="text-gray-700">{responseText}</div>);
    }
  }, [responseText]);

  const formatAgentFile = (content: string) => {
    const lines = content.split('\n');
    return (
      <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg border border-gray-200 shadow-sm overflow-hidden">
        <div className="p-4">
          {lines.map((line, index) => {
            const keyValueMatch = line.match(/^([a-z_]+):\s*(.+)$/);
            if (keyValueMatch && keyValueMatch.length >= 3) {
              const key = keyValueMatch[1];
              const value = keyValueMatch[2];
              return (
                <div key={index} className="flex items-start mb-2">
                  <div className="w-48 font-medium text-gray-600">{key}:</div>
                  <div className="flex-1 text-gray-800">{value}</div>
                </div>
              );
            }
            const sectionMatch = line.match(/^([a-z_]+):\s*\|$/);
            if (sectionMatch && sectionMatch.length >= 2) {
              const key = sectionMatch[1];
              return (
                <div key={index} className="flex flex-col mb-2 mt-4">
                  <div className="w-full font-medium text-gray-600 border-b border-gray-300 pb-1 mb-2">{key}:</div>
                </div>
              );
            }
            if (line.trim() === "") {
              return <div key={index} className="h-2"></div>;
            }
            return (
              <div key={index} className="pl-6 text-gray-700 whitespace-pre-wrap mb-1">
                {line}
              </div>
            );
          })}
        </div>
        {isLoading && (
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 h-1 w-full animate-pulse"></div>
        )}
      </div>
    );
  };
  return <div className="agent-response w-full">{formattedResponse}</div>;
};

function parseAgentResponse(responseText: string, agentType: 'browser' | 'python'): { agent: CompleteAgent, code: string } | null {
  try {
    const agentBlockRegex = /\$\$\$\s*\n?([\s\S]*?)\n?\$\$\$/;
    const agentBlockMatch = responseText.match(agentBlockRegex);
    const relevantText = agentBlockMatch && agentBlockMatch[1] ? agentBlockMatch[1].trim() : responseText.trim();
    const codeMatch = relevantText.match(/code:\s*\|\s*\n([\s\S]*?)(?=\nmemory:)/);
    const systemPromptMatch = relevantText.match(/system_prompt:\s*\|\s*\n([\s\S]*?)(?=\ncode:)/);
    if (!codeMatch || !codeMatch[1] || !systemPromptMatch || !systemPromptMatch[1]) {
      return null;
    }
    let codeSection = codeMatch[1];
    if (agentType === 'python') {
      const lines = codeSection.split('\n');
      let minIndent = Infinity;
      for (const line of lines) {
        if (line.trim() !== '') {
          const spaceMatch = line.match(/^\s*/);
          const leadingSpaces = spaceMatch ? spaceMatch[0].length : 0;
          minIndent = Math.min(minIndent, leadingSpaces);
        }
      }
      if (minIndent < Infinity && minIndent > 0) {
        codeSection = lines.map(line => line.trim() === '' ? '' : line.slice(minIndent)).join('\n');
      }
    }
    const getId = (field: string) => {
      const match = relevantText.match(new RegExp(`${field}:\\s*([^\\n]+)`));
      return match && match[1] ? match[1].trim() : '';
    };
    const agent: CompleteAgent = {
      id: getId('id'),
      name: getId('name'),
      description: getId('description'),
      model_name: getId('model_name'), // This will be overridden by the selected model later
      system_prompt: systemPromptMatch[1].trimEnd(),
      loop_interval_seconds: parseFloat(getId('loop_interval_seconds')),
    };
    return { agent, code: codeSection };
  } catch (error) {
    console.error("Error parsing agent response:", error);
    return null;
  }
}


interface GenerateAgentProps {
  agentType: 'browser' | 'python';
  modelName: string; // Added modelName prop
  getToken: TokenProvider;
}

const GenerateAgent: React.FC<GenerateAgentProps> = ({ agentType, modelName, getToken }) => {
  const [userInput, setUserInput] = useState<string>('');
  const [visibleResponse, setVisibleResponse] = useState<string>('');
  const [fullResponse, setFullResponse] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isFakeStreaming, setIsFakeStreaming] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);

  const [showEditModal, setShowEditModal] = useState<boolean>(false);
  const [parsedAgent, setParsedAgent] = useState<CompleteAgent | null>(null);
  const [parsedCode, setParsedCode] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!userInput.trim() || isLoading || isFakeStreaming || !modelName) { // Added !modelName check
      if (!modelName) setError("Please select a model first.");
      return;
    }

    setVisibleResponse('');
    setFullResponse('');
    setIsLoading(true);
    setError(null);
    setSaveSuccess(false);

    const systemPrompt = agentType === 'browser' ? getSystemPrompt() : getPythonSystemPrompt();

    try {
      // Build proper OpenAI messages array
      const messages = [
        { role: "system", content: systemPrompt },
        { role: "user", content: userInput }
      ];

      const response = await ModelManager.getInstance().sendMessages(
        'gemini-2.0-flash-lite-free',
        messages,
        await getToken()
      );
      setFullResponse(response);
      setIsFakeStreaming(true);
      setIsLoading(false);

      let currentIndex = 0;
      const streamingSpeed = 15;
      const streamInterval = setInterval(() => {
        currentIndex += streamingSpeed;
        if (currentIndex >= response.length) {
          setVisibleResponse(response);
          setIsFakeStreaming(false);
          clearInterval(streamInterval);
        } else {
          setVisibleResponse(response.substring(0, currentIndex));
        }
      }, 25);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Failed to generate: ${errorMessage}`);
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setUserInput('');
    setVisibleResponse('');
    setFullResponse('');
    setError(null);
    setSaveSuccess(false);
  };

  const handleSaveAgent = async () => {
    setIsSaving(true);
    setError(null);

    try {
      const parsed = parseAgentResponse(fullResponse, agentType);
      if (!parsed) {
        throw new Error("Failed to parse agent response. Please check the format.");
      }

      const { agent, code } = parsed;
      
      // Use the model_name from the agent definition, not the generation model
      const agentToSave: CompleteAgent = {
        ...agent
      };

      setParsedAgent(agentToSave);
      setParsedCode(code);
      setShowEditModal(true);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error parsing agent: ${errorMessage}`);
      setIsSaving(false);
    }
  };

  const handleSaveModalChanges = async (agent: CompleteAgent, code: string) => {
    try {
      // The agent object from the modal already has the correct model_name
      await saveAgent(agent, code);
      setSaveSuccess(true);
      setTimeout(() => {
        setSaveSuccess(false);
        setVisibleResponse('');
        setFullResponse('');
        setUserInput('');
      }, 3000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error saving agent: ${errorMessage}`);
    }
    setShowEditModal(false);
    setIsSaving(false);
  };

  return (
    <div className="w-full">
      {!fullResponse ? (
        <form onSubmit={handleSubmit} className="flex">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder={`Describe what you want the ${agentType} agent to do...`}
            className="flex-1 p-3 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-700"
            disabled={isLoading || !modelName} // Disable if no model selected
            autoFocus
          />
          <button
            type="submit"
            className="px-5 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-r-md hover:from-green-600 hover:to-emerald-700 font-medium transition-colors flex items-center"
            disabled={isLoading || !userInput.trim() || !modelName} // Disable if no model selected
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <>
                <Zap className="h-5 w-5 mr-1" />
                Generate
              </>
            )}
          </button>
        </form>
      ) : (
        <div>
          <div className="mb-4 max-h-72 overflow-y-auto">
            <PrettyAgentResponse responseText={visibleResponse} isLoading={isLoading || isFakeStreaming} />
          </div>
          <div className="flex justify-end space-x-3">
            {!isLoading && !isFakeStreaming && fullResponse && (
              <button
                onClick={handleSaveAgent}
                disabled={isSaving}
                className={`px-4 py-2 ${isSaving ? 'bg-gray-400' : 'bg-green-600 hover:bg-green-700'} text-white rounded-md transition-colors flex items-center`}
              >
                {isSaving ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    Parsing...
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4 mr-1" />
                    Configure & Save
                  </>
                )}
              </button>
            )}
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Create Another
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 text-red-700 p-3 my-4 text-sm rounded-md border border-red-200 flex items-start">
          <XCircle className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {saveSuccess && (
        <div className="bg-green-50 text-green-700 p-3 my-4 text-sm rounded-md border border-green-200 flex items-start">
          <Save className="h-5 w-5 mr-2 flex-shrink-0 mt-0.5" />
          <span>Agent saved successfully! You can now find it in your agents list.</span>
        </div>
      )}

      {showEditModal && parsedAgent && (
        <EditAgentModal
          isOpen={showEditModal}
          onClose={() => {
            setShowEditModal(false);
            setIsSaving(false);
          }}
          createMode={true}
          agent={parsedAgent}
          code={parsedCode}
          onSave={handleSaveModalChanges}
          getToken={getToken}
        />
      )}
    </div>
  );
};

export default GenerateAgent;
