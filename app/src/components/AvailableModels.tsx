import React, { useState, useEffect } from 'react';
import { listModels } from '@utils/ollamaServer';
import { Cpu, RefreshCw } from 'lucide-react';
import { Logger } from '@utils/logging';
import { getOllamaServerAddress } from '@utils/main_loop';

interface Model {
  name: string;
  parameterSize?: string;
}

const AvailableModels: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    
    try {
      Logger.info('MODELS', 'Fetching available models from server');
      const { host, port } = getOllamaServerAddress();
      Logger.info('MODELS', `Using server address: ${host}:${port}`);
      
      const response = await listModels(host, port);
      
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
    fetchModels();
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchModels();
  };

  if (loading && !refreshing) {
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
      
      {error ? (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
          <p className="text-red-700">Error: {error}</p>
          <p className="text-sm text-red-600 mt-1">
            Check that your server is running properly and try again.
          </p>
        </div>
      ) : models.length === 0 ? (
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
              className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="flex items-start mb-2">
                <Cpu className="h-5 w-5 text-blue-500 mr-2 mt-1" />
                <div>
                  <h3 className="font-medium text-gray-900">{model.name}</h3>
                  {model.parameterSize && (
                    <span className="inline-block mt-1 text-xs font-medium text-gray-500 bg-gray-100 px-2 py-1 rounded">
                      {model.parameterSize}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
      
      <div className="mt-6 text-sm text-gray-500">
        <p>
          These models are available on your configured model server. 
          You can use them in your agents by specifying their name.
        </p>
      </div>
    </div>
  );
};

export default AvailableModels;
