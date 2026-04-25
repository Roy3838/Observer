import React, { useState, useCallback } from 'react';
import { RotateCcw } from 'lucide-react';
import { InferenceParams } from '../../config/inference-params';
import { inferenceConfigStore } from '@utils/inferenceConfigStore';
import InferenceParamsEditor from '@components/InferenceParamsEditor';

interface RemoteInferenceParamsPanelProps {
  modelName: string;
}

const RemoteInferenceParamsPanel: React.FC<RemoteInferenceParamsPanelProps> = ({ modelName }) => {
  const [params, setParams] = useState<Partial<InferenceParams>>(
    () => inferenceConfigStore.getModelParams(modelName)
  );

  const handleChange = useCallback((updated: Partial<InferenceParams>) => {
    setParams(updated);
    inferenceConfigStore.setModelParams(modelName, updated);
  }, [modelName]);

  const handleClearAll = useCallback(() => {
    inferenceConfigStore.clearModelParams(modelName);
    setParams({});
  }, [modelName]);

  const hasParams = Object.keys(params).length > 0;

  return (
    <div className="p-4 border-t border-gray-200 bg-gray-50/50">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Inference Parameters</span>
        {hasParams && (
          <button
            onClick={handleClearAll}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-red-500 transition-colors"
          >
            <RotateCcw size={11} /> Reset all
          </button>
        )}
      </div>
      <InferenceParamsEditor
        params={params}
        onChange={handleChange}
        isAgentOverride={false}
      />
    </div>
  );
};

export default RemoteInferenceParamsPanel;
