// components/LlmDebugPanel.tsx
// Debug panel for iOS llama.cpp - exposes engine state, metrics, sampler params, and logs

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity,
  Cpu,
  Settings,
  Terminal,
  Play,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Trash2,
  Zap,
  Camera,
  X,
  Image as ImageIcon,
} from 'lucide-react';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import { Logger, LogEntry, LogLevel } from '@utils/logging';
import {
  LlmDebugInfo,
  SamplerParams,
  GenerationMetrics,
  DEFAULT_SAMPLER_PARAMS,
} from '@utils/localLlm/types';

interface LlmDebugPanelProps {
  isVisible: boolean;
}

interface CollapsibleSectionProps {
  title: string;
  icon: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  icon,
  defaultOpen = true,
  children,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
      >
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {icon}
        <span className="font-medium text-sm text-gray-700">{title}</span>
      </button>
      {isOpen && <div className="p-3 space-y-2">{children}</div>}
    </div>
  );
};

const LlmDebugPanel: React.FC<LlmDebugPanelProps> = ({ isVisible }) => {
  const [debugInfo, setDebugInfo] = useState<LlmDebugInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sampler params state
  const [samplerParams, setSamplerParams] = useState<SamplerParams>(DEFAULT_SAMPLER_PARAMS);
  const [samplerDirty, setSamplerDirty] = useState(false);

  // Test generation state
  const [testPrompt, setTestPrompt] = useState('Hello, how are you today?');
  const [testResponse, setTestResponse] = useState('');
  const [testMetrics, setTestMetrics] = useState<GenerationMetrics | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null); // base64 data URL
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Console logs state
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Fetch debug info
  const fetchDebugInfo = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const info = await NativeLlmManager.getInstance().getDebugInfo();
      setDebugInfo(info);
      if (info.engine.samplerParams) {
        setSamplerParams(info.engine.samplerParams);
        setSamplerDirty(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch and log subscription
  useEffect(() => {
    if (!isVisible) return;

    fetchDebugInfo();

    // Subscribe to LLM logs
    const listener = (entry: LogEntry) => {
      if (entry.source === 'LlmEngine' || entry.source === 'NativeLlmManager') {
        setLogs(prev => [...prev, entry].slice(-100));
      }
    };
    Logger.addListener(listener);

    // Load existing logs
    const existingLogs = Logger.getFilteredLogs({
      source: ['LlmEngine', 'NativeLlmManager'],
    }).slice(-100);
    setLogs(existingLogs);

    return () => Logger.removeListener(listener);
  }, [isVisible, fetchDebugInfo]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Suggest a vision prompt when image is first attached
  useEffect(() => {
    if (capturedImage && testPrompt === 'Hello, how are you today?') {
      setTestPrompt('What do you see in this image? Describe it briefly.');
    }
  }, [capturedImage]);

  // Apply sampler params
  const applySamplerParams = async () => {
    try {
      await NativeLlmManager.getInstance().setSamplerParams(samplerParams);
      setSamplerDirty(false);
      await fetchDebugInfo();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  // Handle image capture from camera
  const handleImageCapture = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setCapturedImage(dataUrl);
    };
    reader.readAsDataURL(file);

    // Reset input so same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Remove captured image
  const removeImage = () => {
    setCapturedImage(null);
  };

  // Test generation (supports text-only or multimodal)
  const runTestGeneration = async () => {
    if (!testPrompt.trim() || isGenerating) return;

    setIsGenerating(true);
    setTestResponse('');
    setTestMetrics(null);

    try {
      if (capturedImage) {
        // Multimodal generation with image
        const messages = [{
          role: 'user',
          content: [
            { type: 'image' as const, image: capturedImage },
            { type: 'text' as const, text: testPrompt },
          ],
        }];

        await NativeLlmManager.getInstance().generate(
          messages,
          (token) => setTestResponse(prev => prev + token)
        );

        // Fetch metrics after generation
        const info = await NativeLlmManager.getInstance().getDebugInfo();
        setTestMetrics(info.engine.lastMetrics);
      } else {
        // Text-only generation
        const result = await NativeLlmManager.getInstance().testGenerate(
          testPrompt,
          256,
          (token) => setTestResponse(prev => prev + token)
        );
        setTestMetrics(result.metrics);
      }
    } catch (e) {
      setTestResponse(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setIsGenerating(false);
      await fetchDebugInfo(); // Refresh metrics
    }
  };

  // Clear logs
  const clearLogs = () => setLogs([]);

  if (!isVisible) return null;

  const engine = debugInfo?.engine;

  return (
    <div className="space-y-3">
      {/* Header with refresh */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
          <Zap size={16} className="text-orange-500" />
          llama.cpp Debug Panel
        </h3>
        <button
          onClick={fetchDebugInfo}
          disabled={loading}
          className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded transition-colors disabled:opacity-50"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2">
          {error}
        </div>
      )}

      {/* Engine Status */}
      <CollapsibleSection title="Engine Status" icon={<Cpu size={14} className="text-blue-500" />}>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-500">Backend:</span>
            <span className={engine?.initialized ? 'text-green-600 font-medium' : 'text-red-600'}>
              {engine?.initialized ? 'Initialized' : 'Not initialized'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Model:</span>
            <span className={engine?.isLoaded ? 'text-green-600 font-medium' : 'text-gray-400'}>
              {engine?.isLoaded ? 'Loaded' : 'Not loaded'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Multimodal:</span>
            <span className={engine?.isMultimodal ? 'text-purple-600 font-medium' : 'text-gray-400'}>
              {engine?.isMultimodal ? 'Yes' : 'No'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Model ID:</span>
            <span className="text-gray-700 truncate max-w-[120px]" title={engine?.loadedModelId || ''}>
              {engine?.loadedModelId || '-'}
            </span>
          </div>
        </div>
        {engine?.modelPath && (
          <div className="mt-2 text-xs">
            <span className="text-gray-500">Path: </span>
            <span className="text-gray-600 break-all">{engine.modelPath}</span>
          </div>
        )}
        {engine?.mmprojPath && (
          <div className="text-xs">
            <span className="text-gray-500">mmproj: </span>
            <span className="text-gray-600 break-all">{engine.mmprojPath}</span>
          </div>
        )}
      </CollapsibleSection>

      {/* Last Metrics */}
      <CollapsibleSection title="Last Metrics" icon={<Activity size={14} className="text-green-500" />}>
        {engine?.lastMetrics ? (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-500">Tokens/sec:</span>
              <span className="text-green-600 font-mono font-medium">
                {engine.lastMetrics.tokensPerSecond.toFixed(1)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">TTFT:</span>
              <span className="text-gray-700 font-mono">
                {engine.lastMetrics.timeToFirstTokenMs.toFixed(0)}ms
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Generated:</span>
              <span className="text-gray-700 font-mono">
                {engine.lastMetrics.tokensGenerated} tokens
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Prompt:</span>
              <span className="text-gray-700 font-mono">
                {engine.lastMetrics.promptTokens} tokens
              </span>
            </div>
            <div className="col-span-2 flex justify-between">
              <span className="text-gray-500">Total time:</span>
              <span className="text-gray-700 font-mono">
                {(engine.lastMetrics.totalGenerationTimeMs / 1000).toFixed(2)}s
              </span>
            </div>
          </div>
        ) : (
          <p className="text-xs text-gray-400 text-center py-2">No metrics yet - run a generation</p>
        )}
      </CollapsibleSection>

      {/* Test Generation */}
      <CollapsibleSection title="Test Generation" icon={<Play size={14} className="text-indigo-500" />}>
        <div className="space-y-2">
          {/* Image capture section */}
          <div className="flex items-center gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleImageCapture}
              className="hidden"
              id="camera-input"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isGenerating}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded font-medium transition-colors ${
                capturedImage
                  ? 'bg-purple-100 text-purple-700 border border-purple-300'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-200'
              } disabled:opacity-50`}
            >
              <Camera size={14} />
              {capturedImage ? 'Change Image' : 'Add Camera Image'}
            </button>
            {capturedImage && (
              <button
                onClick={removeImage}
                disabled={isGenerating}
                className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors disabled:opacity-50"
                title="Remove image"
              >
                <X size={14} />
              </button>
            )}
            {engine?.isMultimodal && (
              <span className="text-[10px] text-purple-500 ml-auto">Vision enabled</span>
            )}
          </div>

          {/* Image preview */}
          {capturedImage && (
            <div className="relative">
              <img
                src={capturedImage}
                alt="Captured"
                className="w-full h-32 object-cover rounded border border-purple-200"
              />
              <div className="absolute bottom-1 right-1 flex items-center gap-1 px-1.5 py-0.5 bg-black/50 rounded text-[10px] text-white">
                <ImageIcon size={10} />
                Image attached
              </div>
            </div>
          )}

          <textarea
            value={testPrompt}
            onChange={(e) => setTestPrompt(e.target.value)}
            placeholder={capturedImage ? "Describe what you want to know about the image..." : "Enter test prompt..."}
            rows={2}
            disabled={isGenerating}
            className="w-full p-2 text-xs border border-gray-200 rounded resize-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-50"
          />
          <button
            onClick={runTestGeneration}
            disabled={isGenerating || !engine?.isLoaded}
            className={`w-full flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs text-white rounded disabled:opacity-50 disabled:cursor-not-allowed font-medium ${
              capturedImage
                ? 'bg-purple-600 hover:bg-purple-700'
                : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
          >
            {capturedImage && <ImageIcon size={12} />}
            <Play size={12} />
            {isGenerating ? 'Generating...' : capturedImage ? 'Generate with Image' : 'Generate'}
          </button>

          {/* Warning if image but no multimodal */}
          {capturedImage && !engine?.isMultimodal && (
            <div className="text-[10px] text-amber-600 bg-amber-50 border border-amber-200 rounded px-2 py-1">
              Warning: Model may not support images (no mmproj loaded)
            </div>
          )}

          {testResponse && (
            <div className="mt-2">
              <div className="text-xs text-gray-500 mb-1">Response:</div>
              <div className="p-2 bg-gray-50 border border-gray-200 rounded text-xs text-gray-700 max-h-32 overflow-y-auto whitespace-pre-wrap font-mono">
                {testResponse}
              </div>
            </div>
          )}
          {testMetrics && (
            <div className="flex gap-3 text-xs text-gray-500 mt-1">
              <span>{testMetrics.tokensGenerated} tokens</span>
              <span>{testMetrics.tokensPerSecond.toFixed(1)} tok/s</span>
              <span>TTFT: {testMetrics.timeToFirstTokenMs.toFixed(0)}ms</span>
            </div>
          )}
        </div>
      </CollapsibleSection>

      {/* Sampler Params */}
      <CollapsibleSection title="Sampler Params" icon={<Settings size={14} className="text-amber-500" />} defaultOpen={false}>
        <div className="space-y-3">
          {/* Temperature */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-600">Temperature</span>
              <span className="font-mono text-gray-700">{samplerParams.temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              value={samplerParams.temperature}
              onChange={(e) => {
                setSamplerParams(p => ({ ...p, temperature: parseFloat(e.target.value) }));
                setSamplerDirty(true);
              }}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          {/* Top P */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-600">Top P</span>
              <span className="font-mono text-gray-700">{samplerParams.topP.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={samplerParams.topP}
              onChange={(e) => {
                setSamplerParams(p => ({ ...p, topP: parseFloat(e.target.value) }));
                setSamplerDirty(true);
              }}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          {/* Top K */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-600">Top K</span>
              <span className="font-mono text-gray-700">{samplerParams.topK}</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={samplerParams.topK}
              onChange={(e) => {
                setSamplerParams(p => ({ ...p, topK: parseInt(e.target.value) }));
                setSamplerDirty(true);
              }}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          {/* Repeat Penalty */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-600">Repeat Penalty</span>
              <span className="font-mono text-gray-700">{samplerParams.repeatPenalty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="1"
              max="2"
              step="0.05"
              value={samplerParams.repeatPenalty}
              onChange={(e) => {
                setSamplerParams(p => ({ ...p, repeatPenalty: parseFloat(e.target.value) }));
                setSamplerDirty(true);
              }}
              className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          {/* Seed */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-600">Seed</span>
            </div>
            <input
              type="number"
              min="0"
              value={samplerParams.seed}
              onChange={(e) => {
                setSamplerParams(p => ({ ...p, seed: parseInt(e.target.value) || 0 }));
                setSamplerDirty(true);
              }}
              className="w-full p-1.5 text-xs border border-gray-200 rounded font-mono focus:ring-2 focus:ring-amber-500"
            />
          </div>

          {/* Apply button */}
          <button
            onClick={applySamplerParams}
            disabled={!samplerDirty}
            className="w-full px-3 py-1.5 text-xs bg-amber-500 text-white rounded hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            {samplerDirty ? 'Apply Changes' : 'No Changes'}
          </button>
        </div>
      </CollapsibleSection>

      {/* Console Logs */}
      <CollapsibleSection title="Console Logs" icon={<Terminal size={14} className="text-gray-500" />} defaultOpen={false}>
        <div className="space-y-2">
          <div className="flex justify-end">
            <button
              onClick={clearLogs}
              className="flex items-center gap-1 px-2 py-1 text-xs text-gray-500 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
            >
              <Trash2 size={12} />
              Clear
            </button>
          </div>
          <div className="h-40 overflow-y-auto bg-gray-900 rounded p-2 font-mono text-[10px] leading-tight">
            {logs.length === 0 ? (
              <span className="text-gray-500">No logs yet...</span>
            ) : (
              logs.map((log) => (
                <div
                  key={log.id}
                  className={`${
                    log.level === LogLevel.ERROR
                      ? 'text-red-400'
                      : log.level === LogLevel.WARNING
                      ? 'text-yellow-400'
                      : 'text-gray-300'
                  }`}
                >
                  <span className="text-gray-500">
                    {log.timestamp.toLocaleTimeString()}
                  </span>{' '}
                  {log.message}
                </div>
              ))
            )}
            <div ref={logsEndRef} />
          </div>
        </div>
      </CollapsibleSection>

      {/* Models directory info */}
      {debugInfo?.modelsDir && (
        <div className="text-[10px] text-gray-400 px-1">
          Models: {debugInfo.modelsDir}
        </div>
      )}
    </div>
  );
};

export default LlmDebugPanel;
