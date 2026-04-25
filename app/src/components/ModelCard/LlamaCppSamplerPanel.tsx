import React from 'react';
import { Settings2 } from 'lucide-react';
import { SamplerParams, DEFAULT_SAMPLER_PARAMS, NativeModelState } from '@utils/localLlm/types';

const SamplerSlider: React.FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  hint?: string;
  format: (v: number) => string;
  onChange: (v: number) => void;
}> = ({ label, value, min, max, step, disabled, hint, format, onChange }) => (
  <div>
    <div className="flex justify-between items-center mb-1">
      <label className="text-xs font-medium text-gray-600">{label}</label>
      <span className="text-xs text-gray-500 font-mono">{format(value)}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      disabled={disabled}
      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50 accent-green-600"
    />
    {hint && <p className="text-xs text-gray-400 mt-0.5">{hint}</p>}
  </div>
);

interface LlamaCppSamplerPanelProps {
  nativeStatus: NativeModelState['status'];
  samplerParams: SamplerParams;
  onParamChange: (key: keyof SamplerParams, value: number) => void;
  onReset: () => void;
}

const LlamaCppSamplerPanel: React.FC<LlamaCppSamplerPanelProps> = ({
  nativeStatus,
  samplerParams,
  onParamChange,
  onReset,
}) => {
  const disabled = nativeStatus !== 'loaded';

  return (
    <div className="p-4 space-y-4 border-t border-gray-200 bg-gray-50/50">
      <div className="flex items-center gap-2 mb-2">
        <Settings2 size={13} className="text-gray-500" />
        <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Generation Settings</span>
      </div>

      {disabled && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          Load a model to configure generation settings.
        </p>
      )}

      <SamplerSlider
        label="Temperature"
        value={samplerParams.temperature}
        min={0} max={2} step={0.05}
        disabled={disabled}
        hint="Higher = more creative, lower = more focused"
        format={(v) => v.toFixed(2)}
        onChange={(v) => onParamChange('temperature', v)}
      />
      <SamplerSlider
        label="Top P (nucleus sampling)"
        value={samplerParams.topP}
        min={0} max={1} step={0.05}
        disabled={disabled}
        format={(v) => v.toFixed(2)}
        onChange={(v) => onParamChange('topP', v)}
      />
      <SamplerSlider
        label="Top K"
        value={samplerParams.topK}
        min={1} max={100} step={1}
        disabled={disabled}
        format={(v) => v.toString()}
        onChange={(v) => onParamChange('topK', v)}
      />
      <SamplerSlider
        label="Repeat Penalty"
        value={samplerParams.repeatPenalty}
        min={1} max={2} step={0.05}
        disabled={disabled}
        hint="Discourages repetitive text"
        format={(v) => v.toFixed(2)}
        onChange={(v) => onParamChange('repeatPenalty', v)}
      />

      <div>
        <label className="text-xs font-medium text-gray-600">Seed</label>
        <input
          type="number"
          value={samplerParams.seed}
          onChange={(e) => onParamChange('seed', parseInt(e.target.value) || 0)}
          disabled={disabled}
          className="w-full mt-1 p-2 text-sm border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-green-500 disabled:opacity-50 font-mono"
          placeholder="42"
        />
        <p className="text-xs text-gray-400 mt-0.5">Use -1 for random seed</p>
      </div>

      <button
        onClick={onReset}
        disabled={disabled}
        className="text-xs text-gray-500 hover:text-gray-700 underline disabled:opacity-50 disabled:no-underline"
      >
        Reset to defaults
      </button>
    </div>
  );
};

export { DEFAULT_SAMPLER_PARAMS };
export default LlamaCppSamplerPanel;
