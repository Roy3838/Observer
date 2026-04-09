import React, { useState } from 'react';
import { InferenceParams, INFERENCE_PARAM_METADATA } from '../config/inference-params';
import { RotateCcw, Eye, EyeOff, Info, X } from 'lucide-react';

interface InferenceParamsEditorProps {
  params: Partial<InferenceParams>;
  onChange: (params: Partial<InferenceParams>) => void;
  /** If true, shows "Clear All" button and "inherited" indicators */
  isAgentOverride?: boolean;
  onClearAll?: () => void;
  /** Global defaults to compare against (for showing inherited values) */
  globalDefaults?: InferenceParams;
}

/**
 * Reusable editor for inference parameters.
 * Used in SettingsTab for global defaults and EditAgentModal for per-agent overrides.
 */
const InferenceParamsEditor: React.FC<InferenceParamsEditorProps> = ({
  params,
  onChange,
  isAgentOverride = false,
  onClearAll,
  globalDefaults = {},
}) => {
  const [showPassword, setShowPassword] = useState<Record<string, boolean>>({});
  const [openTooltip, setOpenTooltip] = useState<string | null>(null);

  const toggleTooltip = (key: string) => {
    setOpenTooltip(openTooltip === key ? null : key);
  };

  const InfoTooltip: React.FC<{ paramKey: string; description: string }> = ({ paramKey, description }) => (
    <div className="relative inline-flex items-center ml-1.5">
      <button
        type="button"
        onClick={() => toggleTooltip(paramKey)}
        className="text-gray-400 hover:text-gray-600 transition-colors"
        title="Show description"
      >
        <Info className="h-3.5 w-3.5" />
      </button>
      {openTooltip === paramKey && (
        <div className="absolute left-0 top-full mt-1 z-10 w-64 p-2 bg-gray-800 text-white text-xs rounded-md shadow-lg">
          <div className="flex justify-between items-start gap-2">
            <span>{description}</span>
            <button
              type="button"
              onClick={() => setOpenTooltip(null)}
              className="flex-shrink-0 text-gray-400 hover:text-white"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
          <div className="absolute -top-1 left-2 w-2 h-2 bg-gray-800 rotate-45" />
        </div>
      )}
    </div>
  );

  const handleParamChange = (key: keyof InferenceParams, value: any) => {
    // If value is empty/undefined, remove the key
    if (value === '' || value === undefined || value === null) {
      const newParams = { ...params };
      delete newParams[key];
      onChange(newParams);
    } else {
      onChange({ ...params, [key]: value });
    }
  };

  const clearParam = (key: keyof InferenceParams) => {
    const newParams = { ...params };
    delete newParams[key];
    onChange(newParams);
  };

  const renderNumberInput = (key: keyof InferenceParams, meta: typeof INFERENCE_PARAM_METADATA[keyof InferenceParams]) => {
    const value = params[key] as number | undefined;
    const inheritedValue = globalDefaults[key] as number | undefined;
    const isSet = value !== undefined;
    const displayValue = value ?? inheritedValue ?? '';

    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <label className="text-sm font-medium text-gray-700">{meta.label}</label>
            <InfoTooltip paramKey={key} description={meta.description} />
          </div>
          {isAgentOverride && isSet && (
            <button
              onClick={() => clearParam(key)}
              className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1"
              title="Reset to global default"
            >
              <RotateCcw className="h-3 w-3" />
            </button>
          )}
        </div>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={meta.min}
            max={meta.max}
            step={meta.step}
            value={displayValue || meta.min || 0}
            onChange={(e) => handleParamChange(key, parseFloat(e.target.value))}
            className="flex-grow h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <input
            type="number"
            min={meta.min}
            max={meta.max}
            step={meta.step}
            value={displayValue}
            onChange={(e) => {
              const val = e.target.value === '' ? undefined : parseFloat(e.target.value);
              handleParamChange(key, val);
            }}
            placeholder={isAgentOverride && inheritedValue !== undefined ? String(inheritedValue) : 'Default'}
            className={`w-20 px-2 py-1 text-sm border rounded-md ${
              isAgentOverride && !isSet ? 'text-gray-400 border-gray-200' : 'border-gray-300'
            }`}
          />
        </div>
      </div>
    );
  };

  const renderBooleanInput = (key: keyof InferenceParams, meta: typeof INFERENCE_PARAM_METADATA[keyof InferenceParams]) => {
    const value = params[key] as boolean | undefined;
    const inheritedValue = globalDefaults[key] as boolean | undefined;
    const isSet = value !== undefined;
    const displayValue = value ?? inheritedValue ?? false;

    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center">
              <label className="text-sm font-medium text-gray-700">{meta.label}</label>
              <InfoTooltip paramKey={key} description={meta.description} />
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={displayValue}
                onChange={(e) => handleParamChange(key, e.target.checked)}
                className="sr-only peer"
              />
              <div className={`w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600 ${
                isAgentOverride && !isSet ? 'opacity-50' : ''
              }`}></div>
            </label>
          </div>
          {isAgentOverride && isSet && (
            <button
              onClick={() => clearParam(key)}
              className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1"
              title="Reset to global default"
            >
              <RotateCcw className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
    );
  };

  const renderSelectInput = (key: keyof InferenceParams, meta: typeof INFERENCE_PARAM_METADATA[keyof InferenceParams]) => {
    const value = params[key] as string | undefined;
    const inheritedValue = globalDefaults[key] as string | undefined;
    const isSet = value !== undefined;

    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <label className="text-sm font-medium text-gray-700">{meta.label}</label>
            <InfoTooltip paramKey={key} description={meta.description} />
          </div>
          {isAgentOverride && isSet && (
            <button
              onClick={() => clearParam(key)}
              className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1"
              title="Reset to global default"
            >
              <RotateCcw className="h-3 w-3" />
            </button>
          )}
        </div>
        <select
          value={value ?? ''}
          onChange={(e) => handleParamChange(key, e.target.value || undefined)}
          className={`w-full px-3 py-2 text-sm border rounded-md ${
            isAgentOverride && !isSet ? 'text-gray-400 border-gray-200' : 'border-gray-300'
          }`}
        >
          <option value="">
            {isAgentOverride && inheritedValue ? `Inherit (${inheritedValue})` : 'Default'}
          </option>
          {meta.options?.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    );
  };

  const renderStringArrayInput = (key: keyof InferenceParams, meta: typeof INFERENCE_PARAM_METADATA[keyof InferenceParams]) => {
    const value = params[key] as string[] | undefined;
    const inheritedValue = globalDefaults[key] as string[] | undefined;
    const isSet = value !== undefined;
    const displayValue = value?.join('\n') ?? '';

    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <label className="text-sm font-medium text-gray-700">{meta.label}</label>
            <InfoTooltip paramKey={key} description={meta.description} />
          </div>
          {isAgentOverride && isSet && (
            <button
              onClick={() => clearParam(key)}
              className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1"
              title="Reset to global default"
            >
              <RotateCcw className="h-3 w-3" />
            </button>
          )}
        </div>
        <textarea
          value={displayValue}
          onChange={(e) => {
            const val = e.target.value.trim();
            if (val === '') {
              handleParamChange(key, undefined);
            } else {
              handleParamChange(key, val.split('\n').map(s => s.trim()).filter(Boolean));
            }
          }}
          placeholder={isAgentOverride && inheritedValue?.length ? `Inherited: ${inheritedValue.join(', ')}` : 'One per line'}
          rows={2}
          className={`w-full px-3 py-2 text-sm border rounded-md ${
            isAgentOverride && !isSet ? 'text-gray-400 border-gray-200' : 'border-gray-300'
          }`}
        />
      </div>
    );
  };

  const renderPasswordInput = (key: keyof InferenceParams, meta: typeof INFERENCE_PARAM_METADATA[keyof InferenceParams]) => {
    const value = params[key] as string | undefined;
    const inheritedValue = globalDefaults[key] as string | undefined;
    const isSet = value !== undefined && value !== '';
    const isVisible = showPassword[key] ?? false;

    // Mask the value for display (show last 4 chars if long enough)
    const getMaskedValue = (val: string) => {
      if (val.length <= 8) return '••••••••';
      return '••••••••' + val.slice(-4);
    };

    return (
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <label className="text-sm font-medium text-gray-700">{meta.label}</label>
            <InfoTooltip paramKey={key} description={meta.description} />
          </div>
          {isAgentOverride && isSet && (
            <button
              onClick={() => clearParam(key)}
              className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1"
              title="Reset to global default"
            >
              <RotateCcw className="h-3 w-3" />
            </button>
          )}
        </div>
        <div className="relative">
          <input
            type={isVisible ? 'text' : 'password'}
            value={value ?? ''}
            onChange={(e) => handleParamChange(key, e.target.value || undefined)}
            placeholder={isAgentOverride && inheritedValue ? getMaskedValue(inheritedValue) : 'sk-...'}
            className={`w-full px-3 py-2 pr-10 text-sm border rounded-md font-mono ${
              isAgentOverride && !isSet ? 'text-gray-400 border-gray-200' : 'border-gray-300'
            }`}
          />
          <button
            type="button"
            onClick={() => setShowPassword({ ...showPassword, [key]: !isVisible })}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            {isVisible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        </div>
      </div>
    );
  };

  const renderParam = (key: keyof InferenceParams) => {
    const meta = INFERENCE_PARAM_METADATA[key];
    switch (meta.type) {
      case 'number':
        return renderNumberInput(key, meta);
      case 'boolean':
        return renderBooleanInput(key, meta);
      case 'select':
        return renderSelectInput(key, meta);
      case 'string[]':
        return renderStringArrayInput(key, meta);
      case 'password':
        return renderPasswordInput(key, meta);
      default:
        return null;
    }
  };

  // Group params by tier for better organization
  const tier1Keys: (keyof InferenceParams)[] = ['temperature', 'top_p', 'max_tokens', 'seed', 'stop'];
  const tier2Keys: (keyof InferenceParams)[] = ['frequency_penalty', 'presence_penalty', 'top_k'];
  const tier3Keys: (keyof InferenceParams)[] = ['reasoning_effort', 'enable_thinking'];
  const tier4Keys: (keyof InferenceParams)[] = ['customApiKey'];

  return (
    <div className="space-y-6">
      {/* Header with Clear All button for agent overrides */}
      {isAgentOverride && onClearAll && (
        <div className="flex justify-between items-center">
          <button
            onClick={onClearAll}
            className="text-xs text-red-500 hover:text-red-700 flex items-center gap-1"
          >
            <RotateCcw className="h-3 w-3" />
            Clear All Overrides
          </button>
        </div>
      )}

      {/* Tier 1 - Core Parameters */}
      <div>
        <h4 className="text-sm font-semibold text-gray-800 mb-3 pb-1 border-b">
          Core Parameters
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {tier1Keys.map((key) => (
            <div key={key}>{renderParam(key)}</div>
          ))}
        </div>
      </div>

      {/* Tier 2 - Penalties & Sampling */}
      <div>
        <h4 className="text-sm font-semibold text-gray-800 mb-3 pb-1 border-b">
          Penalties & Sampling
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {tier2Keys.map((key) => (
            <div key={key}>{renderParam(key)}</div>
          ))}
        </div>
      </div>

      {/* Tier 3 - Reasoning/Thinking */}
      <div>
        <h4 className="text-sm font-semibold text-gray-800 mb-3 pb-1 border-b">
          Reasoning & Thinking
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {tier3Keys.map((key) => (
            <div key={key}>{renderParam(key)}</div>
          ))}
        </div>
      </div>

      {/* Tier 4 - Custom API (Power Users) */}
      <div>
        <h4 className="text-sm font-semibold text-gray-800 mb-3 pb-1 border-b">
          Custom API 
        </h4>
        <div className="grid grid-cols-1 gap-4">
          {tier4Keys.map((key) => (
            <div key={key}>{renderParam(key)}</div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default InferenceParamsEditor;
