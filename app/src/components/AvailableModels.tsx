import React, { useState, useEffect } from 'react';
import { listModels, Model } from '@utils/inferenceServer'; // Import updated Model interface
import { Cpu, RefreshCw, Eye, Server } from 'lucide-react';
import { BROWSER_LOCAL_SENTINEL, LLAMA_CPP_LOCAL_SENTINEL } from '@utils/inferenceServer';
import { Logger } from '@utils/logging';
import { getInferenceAddresses } from '@utils/inferenceServer';
import TerminalModal from '@components/TerminalModal';
import { platformFetch } from '@utils/platform';

// No need to redefine Model interface here if imported correctly

interface AvailableModelsProps {
  isProUser?: boolean;
}

const AvailableModels: React.FC<AvailableModelsProps> = ({ isProUser = false }) => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [showTerminal, setShowTerminal] = useState(false);
  const [ollamaServers, setOllamaServers] = useState<string[]>([]);

  // Check if a server supports Ollama by probing /api/tags endpoint
  const checkOllamaSupport = async (address: string): Promise<boolean> => {
    try {
      const response = await platformFetch(`${address}/api/tags`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      return response.ok;
    } catch (error) {
      return false;
    }
  };

  // Detect which addresses support Ollama
  const detectOllamaServers = async () => {
    const addresses = getInferenceAddresses();
    const ollamaChecks = await Promise.all(
      addresses.map(async (addr) => ({
        address: addr,
        isOllama: await checkOllamaSupport(addr)
      }))
    );
    const detectedServers = ollamaChecks
      .filter(check => check.isOllama)
      .map(check => check.address);

    setOllamaServers(detectedServers);
    Logger.info('MODELS', `Detected Ollama servers: ${detectedServers.join(', ')}`);
  };

  const fetchModels = async () => {
    setLoading(true);
    setError(null);

    try {
      Logger.info('MODELS', 'Fetching available models from server');
      const addresses = getInferenceAddresses();
      Logger.info('MODELS', `Using inference addresses: ${addresses.join(', ')}`);

      const response = listModels(); // Uses updated listModels

      if (response.error) {
        throw new Error(response.error);
      }

      setModels(response.models);
      Logger.info('MODELS', `Successfully loaded ${response.models.length} models`);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(errorMessage);
      Logger.error('MODELS', `Failed to fetch models: ${errorMessage}`);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    // This runs on mount
    fetchModels();
    detectOllamaServers();
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchModels();
    detectOllamaServers();
  };

  if (loading && !refreshing) {
    // ... (loading state remains the same)
    return (
        <div className="flex flex-col items-center justify-center py-12">
          <div className="animate-spin mb-4">
            <Cpu className="h-8 w-8 text-blue-500" />
          </div>
          <p className="text-gray-600">Loading available models...</p>
        </div>
      );
  }

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold text-gray-800">Available Models</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTerminal(true)}
            className="px-3 py-2 rounded-md bg-green-50 text-green-600 hover:bg-green-100"
          >
            Add Model
          </button>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className={`flex items-center space-x-2 px-3 py-2 rounded-md ${
              refreshing
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-blue-50 text-blue-600 hover:bg-blue-100'
            }`}
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            <span>{refreshing ? 'Refreshing...' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {error ? (
        // ... (error display remains the same)
         <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
           <p className="text-red-700">Error: {error}</p>
           <p className="text-sm text-red-600 mt-1">
             Check that your server is running properly and try again.
           </p>
         </div>
      ) : models.length === 0 ? (
        // ... (no models display remains the same)
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <p className="text-yellow-700">No models found on the server.</p>
          <p className="text-sm text-yellow-600 mt-1">
            Ensure that your server is properly configured and has models available.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => (
            <div
              key={model.name}
              className={`bg-white border border-gray-200 rounded-lg p-5 shadow-sm hover:shadow-md transition-shadow ${
                model.pro && !isProUser ? 'opacity-50 grayscale select-none' : ''
              }`}
            >
              <div className="flex items-start mb-2">
                <Cpu className="h-5 w-5 text-blue-500 mr-2 mt-1 flex-shrink-0" />
                <div className="flex-grow">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-medium text-gray-900 break-words">{model.name}</h3>
                    {model.pro && !isProUser && (
                      <span className="text-xs font-bold text-purple-600 bg-purple-100 px-2 py-1 rounded-full">
                        PRO
                      </span>
                    )}
                  </div>
                  {/* Container for parameter size, multimodal icon, and local tag */}
                  <div className="flex items-center flex-wrap mt-1">
                    {model.parameterSize && model.parameterSize !== "N/A" && (
                      <span className="inline-block mr-2 text-xs font-medium text-gray-500 bg-gray-100 px-2 py-1 rounded">
                        {model.parameterSize}
                      </span>
                    )}
                    {/* Conditionally render the Eye icon if multimodal is true */}
                    {model.multimodal && (
                      <span title="Supports Multimodal Input" className="inline-block mr-2 text-xs font-medium text-purple-600 bg-purple-100 px-2 py-1 rounded">
                        <Eye className="h-3.5 w-3.5 inline-block mr-1 -mt-px" />
                        Vision
                      </span>
                    )}
                    {model.server === BROWSER_LOCAL_SENTINEL ? (
                      <span title="Runs in-browser via WebGPU" className="inline-block text-xs font-medium text-green-700 bg-green-100 px-2 py-1 rounded">
                        <Cpu className="h-3.5 w-3.5 inline-block mr-1 -mt-px" />
                        On-Browser
                      </span>
                    ) : model.server === LLAMA_CPP_LOCAL_SENTINEL ? (
                      <span title="Runs locally via llama.cpp" className="inline-block text-xs font-medium text-emerald-700 bg-emerald-100 px-2 py-1 rounded">
                        <Cpu className="h-3.5 w-3.5 inline-block mr-1 -mt-px" />
                        Local
                      </span>
                    ) : !model.server.includes('api.observer-ai.com') && (
                      <span title="Running on local server" className="inline-block text-xs font-medium text-gray-600 bg-gray-100 px-2 py-1 rounded">
                        <Server className="h-3.5 w-3.5 inline-block mr-1 -mt-px" />
                        Server
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ... (footer text remains the same) ... */}
       <div className="mt-6 text-sm text-gray-500">
         <p>
           These models are available on your configured model server.
           You can use them in your agents by specifying their name.
         </p>
      </div>
      <TerminalModal
      isOpen={showTerminal}
      onClose={() => setShowTerminal(false)}
      onPullComplete={handleRefresh}
      ollamaServers={ollamaServers}
      />
    </div>
  );
};

export default AvailableModels;
