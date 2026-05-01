import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { X, AlertTriangle, Clipboard, Check, Zap, Server } from 'lucide-react';
import { listModels, Model, SKIP_MODEL_SENTINEL } from '@utils/inferenceServer';
import getConversationalSystemPrompt from '@utils/conversational_system_prompt';
import getMultiAgentSystemPrompt from '@utils/multi_agent_creator';

interface LocalWarningProps {
  isOpen: boolean;
  onClose: () => void;
  currentModel: string;
  onSelectModel: (modelName: string) => void;
  onSignIn?: () => void;
  onSwitchToObServer?: () => void;
  isAuthenticated: boolean;
  featureType: 'conversational' | 'multiagent';
}

const MIN_PARAMS = { conversational: '50B+', multiagent: '100B+' };

const LocalWarning: React.FC<LocalWarningProps> = ({
  isOpen,
  onClose,
  currentModel,
  onSelectModel,
  onSignIn,
  onSwitchToObServer,
  isAuthenticated,
  featureType,
}) => {
  const [localModels, setLocalModels] = useState<Model[]>([]);
  const [isFetchingModels, setIsFetchingModels] = useState(false);
  const [localModelError, setLocalModelError] = useState<string | null>(null);
  const [isCopied, setIsCopied] = useState(false);
  const [selected, setSelected] = useState<string>(currentModel);

  useEffect(() => {
    if (!isOpen) return;
    setIsFetchingModels(true);
    setLocalModelError(null);
    const result = listModels();
    if (result.error) {
      setLocalModelError(`Connection failed: ${result.error}`);
    } else if (result.models.length === 0) {
      setLocalModelError('No models found. Ensure your local server is running.');
    } else {
      setLocalModels(result.models.filter(m => m.server !== SKIP_MODEL_SENTINEL));
    }
    setIsFetchingModels(false);
  }, [isOpen]);

  const handleCopyPrompt = () => {
    const text = featureType === 'conversational'
      ? getConversationalSystemPrompt()
      : getMultiAgentSystemPrompt();
    navigator.clipboard.writeText(text).then(() => {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    });
  };

  const handleUseLocal = () => {
    if (selected) {
      onSelectModel(selected);
      onClose();
    }
  };

  const handleObServer = () => {
    if (isAuthenticated) {
      onSwitchToObServer?.();
    } else {
      onSignIn?.();
    }
    onClose();
  };

  if (!isOpen) return null;

  const modal = (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
      style={{ backgroundColor: 'rgba(0,0,0,0.55)' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-sm overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <span className="text-sm font-semibold text-gray-800">Choose AI Backend</span>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors">
            <X className="h-4 w-4 text-gray-500" />
          </button>
        </div>

        <div className="p-5 space-y-3">
          {/* Recommended: ObServer */}
          <button
            onClick={handleObServer}
            className="w-full flex items-start gap-3 p-4 rounded-xl border-2 border-blue-500 bg-blue-50 hover:bg-blue-100 transition-colors text-left group"
          >
            <div className="mt-0.5 flex-shrink-0 bg-blue-500 text-white rounded-lg p-1.5">
              <Zap className="h-4 w-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-blue-700">
                  {isAuthenticated ? 'Use ObServer' : 'Sign in to use ObServer'}
                </span>
                <span className="text-[10px] font-bold uppercase tracking-wide bg-blue-500 text-white px-1.5 py-0.5 rounded-full">
                  Recommended
                </span>
              </div>
              <p className="text-xs text-blue-600 mt-0.5">
                Powerful cloud models optimised for this feature.
              </p>
            </div>
          </button>

          {/* Divider */}
          <div className="flex items-center gap-2">
            <div className="flex-1 h-px bg-gray-200" />
            <span className="text-xs text-gray-400 flex-shrink-0">or use a local model</span>
            <div className="flex-1 h-px bg-gray-200" />
          </div>

          {/* Warning banner */}
          <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2.5 text-xs text-amber-800">
            <AlertTriangle className="h-3.5 w-3.5 mt-0.5 flex-shrink-0 text-amber-500" />
            <span>
              Local models require <strong>{MIN_PARAMS[featureType]} parameters</strong> for reliable results.
              Small models will likely produce poor output.
            </span>
          </div>

          {/* Local model picker */}
          <div className="space-y-2">
            {isFetchingModels ? (
              <div className="flex items-center gap-2 px-3 py-4 text-sm text-gray-400">
                <Server className="h-4 w-4 animate-pulse" />
                Connecting to local server…
              </div>
            ) : localModelError ? (
              <div className="text-xs text-red-600 px-1">{localModelError}</div>
            ) : (
              <div className="max-h-40 overflow-y-auto space-y-1 pr-1">
                {localModels.map((model) => (
                  <button
                    key={model.name}
                    onClick={() => setSelected(model.name)}
                    className={`w-full flex items-center justify-between gap-2 px-3 py-2.5 rounded-lg border text-left text-sm transition-colors ${
                      selected === model.name
                        ? 'border-gray-400 bg-gray-100 font-medium text-gray-900'
                        : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50 text-gray-700'
                    }`}
                  >
                    <span className="truncate">{model.name}</span>
                    {selected === model.name && <Check className="h-3.5 w-3.5 flex-shrink-0 text-gray-600" />}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Actions row */}
          <div className="flex items-center gap-2 pt-1">
            <button
              onClick={handleUseLocal}
              disabled={!selected || !!localModelError || isFetchingModels}
              className="flex-1 px-4 py-2.5 text-sm font-medium rounded-xl bg-gray-900 text-white hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Use local model
            </button>
            <button
              onClick={handleCopyPrompt}
              title={isCopied ? 'Copied!' : 'Copy system prompt'}
              className="p-2.5 rounded-xl border border-gray-200 hover:bg-gray-50 transition-colors text-gray-600"
            >
              {isCopied ? <Check className="h-4 w-4 text-green-500" /> : <Clipboard className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  return createPortal(modal, document.body);
};

export default LocalWarning;
