// components/BenchmarkPanel.tsx
// Universal benchmark panel for testing any loaded local model (llama.cpp, Transformers.js, or Ollama)

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Camera,
  X,
  Image as ImageIcon,
  AlertCircle,
  Cpu,
  StopCircle,
  CheckCircle,
} from 'lucide-react';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import { isTauri } from '@utils/platform';
import { GenerationMetrics, GEMMA_DISPLAY_NAMES, GemmaModelId } from '@utils/localLlm/types';

interface BenchmarkPanelProps {
  isVisible: boolean;
}

type ActiveBackend = 'llamacpp' | 'transformers' | 'ollama' | null;

interface BackendInfo {
  backend: ActiveBackend;
  modelName: string;
  isMultimodal: boolean;
}

const BenchmarkPanel: React.FC<BenchmarkPanelProps> = ({ isVisible }) => {
  // Test generation state
  const [testPrompt, setTestPrompt] = useState('Hello, how are you today?');
  const [testResponse, setTestResponse] = useState('');
  const [testMetrics, setTestMetrics] = useState<GenerationMetrics | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Backend detection state
  const [backendInfo, setBackendInfo] = useState<BackendInfo>({ backend: null, modelName: '', isMultimodal: false });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Detect which backend has a loaded model
  const detectActiveBackend = useCallback((): BackendInfo => {
    // Check llama.cpp (Tauri only)
    if (isTauri()) {
      const nativeState = NativeLlmManager.getInstance().getState();
      if (nativeState.status === 'loaded' && nativeState.modelId) {
        return {
          backend: 'llamacpp',
          modelName: nativeState.modelId,
          isMultimodal: true, // Assume multimodal if mmproj might be loaded
        };
      }
    }

    // Check Transformers.js
    const gemmaState = GemmaModelManager.getInstance().getState();
    if (gemmaState.status === 'loaded' && gemmaState.modelId) {
      return {
        backend: 'transformers',
        modelName: GEMMA_DISPLAY_NAMES[gemmaState.modelId as GemmaModelId] || gemmaState.modelId,
        isMultimodal: true,
      };
    }

    // TODO: Check Ollama when we have an OllamaManager

    return { backend: null, modelName: '', isMultimodal: false };
  }, []);

  // Subscribe to state changes from both managers
  useEffect(() => {
    if (!isVisible) return;

    // Initial detection
    setBackendInfo(detectActiveBackend());

    // Get initial states
    setBackendInfo(detectActiveBackend());

    // Subscribe to llama.cpp state changes
    const unsubNative = isTauri()
      ? NativeLlmManager.getInstance().onStateChange((_state) => {
          setBackendInfo(detectActiveBackend());
        })
      : () => {};

    // Subscribe to Transformers.js state changes
    const unsubGemma = GemmaModelManager.getInstance().onStateChange(() => {
      setBackendInfo(detectActiveBackend());
    });

    return () => {
      unsubNative();
      unsubGemma();
    };
  }, [isVisible, detectActiveBackend]);

  // Suggest a vision prompt when image is first attached
  useEffect(() => {
    if (capturedImage && testPrompt === 'Hello, how are you today?') {
      setTestPrompt('What do you see in this image? Describe it briefly.');
    }
  }, [capturedImage, testPrompt]);

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

  // Stop ongoing generation
  const stopGeneration = useCallback(() => {
    if (backendInfo.backend === 'llamacpp') {
      NativeLlmManager.getInstance().cancelGeneration();
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
  }, [backendInfo.backend]);

  // Run test generation using the active backend
  const runTestGeneration = async () => {
    if (!testPrompt.trim() || isGenerating || !backendInfo.backend) return;

    // Create abort controller for this generation
    abortControllerRef.current = new AbortController();

    setIsGenerating(true);
    setTestResponse('');
    setTestMetrics(null);
    setError(null);

    try {
      if (backendInfo.backend === 'llamacpp') {
        // llama.cpp generation
        if (capturedImage) {
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
          const result = await NativeLlmManager.getInstance().testGenerate(
            testPrompt,
            (token) => setTestResponse(prev => prev + token)
          );
          setTestMetrics(result.metrics);
        }
      } else if (backendInfo.backend === 'transformers') {
        // Transformers.js generation
        const startTime = performance.now();
        let tokensGenerated = 0;
        let firstTokenTime = 0;

        if (capturedImage) {
          const messages = [{
            role: 'user',
            content: [
              { type: 'image' as const, image: capturedImage },
              { type: 'text' as const, text: testPrompt },
            ],
          }];
          await GemmaModelManager.getInstance().generate(messages, (token) => {
            if (tokensGenerated === 0) {
              firstTokenTime = performance.now() - startTime;
            }
            tokensGenerated++;
            setTestResponse(prev => prev + token);
          });
        } else {
          const messages = [{ role: 'user', content: testPrompt }];
          await GemmaModelManager.getInstance().generate(messages, (token) => {
            if (tokensGenerated === 0) {
              firstTokenTime = performance.now() - startTime;
            }
            tokensGenerated++;
            setTestResponse(prev => prev + token);
          });
        }

        const totalTime = performance.now() - startTime;
        setTestMetrics({
          tokensGenerated,
          promptTokens: 0, // Not available from Transformers.js
          timeToFirstTokenMs: firstTokenTime,
          totalGenerationTimeMs: totalTime,
          tokensPerSecond: tokensGenerated / (totalTime / 1000),
        });
      }
      // TODO: Add Ollama support
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      if (errorMessage !== 'Aborted') {
        setError(errorMessage);
        setTestResponse(`Error: ${errorMessage}`);
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  if (!isVisible) return null;

  return (
    <div className="space-y-5">
      {/* Active model info or no model warning */}
      {backendInfo.backend ? (
        <div className="flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl">
          <div className="w-10 h-10 rounded-lg bg-green-200 flex items-center justify-center">
            <Cpu size={20} className="text-green-700" />
          </div>
          <div className="flex-1">
            <span className="font-semibold text-gray-900">{backendInfo.modelName}</span>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-xs text-gray-500">
                {backendInfo.backend === 'llamacpp' ? 'llama.cpp' : 'Transformers.js'}
              </span>
              {backendInfo.isMultimodal && (
                <span className="text-xs text-purple-600 font-medium bg-purple-100 px-1.5 py-0.5 rounded">Vision</span>
              )}
            </div>
          </div>
          <span className="text-xs font-semibold text-green-700 bg-green-200 px-3 py-1 rounded-full flex items-center gap-1">
            <CheckCircle size={12} /> Ready
          </span>
        </div>
      ) : (
        <div className="flex items-center gap-3 px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl">
          <div className="w-10 h-10 rounded-lg bg-gray-200 flex items-center justify-center">
            <AlertCircle size={20} className="text-gray-400" />
          </div>
          <div className="flex-1">
            <span className="font-medium text-gray-700">No model loaded</span>
            <p className="text-xs text-gray-500 mt-0.5">Load a model from llama.cpp or Transformers.js tabs to run benchmarks.</p>
          </div>
        </div>
      )}

      {/* Test Generation Section */}
      <div className="space-y-4 border border-gray-200 rounded-xl p-5 bg-white">
        {/* Image capture section */}
        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleImageCapture}
            className="hidden"
            id="benchmark-camera-input"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isGenerating || !backendInfo.backend}
            className={`flex items-center gap-1.5 px-3 py-2 text-xs rounded-lg font-medium transition-colors ${
              capturedImage
                ? 'bg-purple-100 text-purple-700 border border-purple-300'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-200'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <Camera size={14} />
            {capturedImage ? 'Change Image' : 'Add Image'}
          </button>
          {capturedImage && (
            <button
              onClick={removeImage}
              disabled={isGenerating}
              className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
              title="Remove image"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Image preview */}
        {capturedImage && (
          <div className="relative">
            <img
              src={capturedImage}
              alt="Captured"
              className="w-full h-36 object-cover rounded-xl border border-purple-200"
            />
            <div className="absolute bottom-2 right-2 flex items-center gap-1.5 px-2 py-1 bg-black/60 backdrop-blur-sm rounded-lg text-xs text-white">
              <ImageIcon size={12} />
              Image attached
            </div>
          </div>
        )}

        {/* Prompt input */}
        <textarea
          value={testPrompt}
          onChange={(e) => setTestPrompt(e.target.value)}
          placeholder={capturedImage ? "Describe what you want to know about the image..." : "Enter a test prompt..."}
          rows={3}
          disabled={isGenerating || !backendInfo.backend}
          className="w-full p-3 text-sm border border-gray-200 rounded-xl resize-none focus:ring-2 focus:ring-orange-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
        />

        {/* Run/Stop button */}
        {isGenerating ? (
          <button
            onClick={stopGeneration}
            className="w-full flex items-center justify-center gap-1.5 px-4 py-2.5 text-sm text-white rounded-xl font-medium bg-red-500 hover:bg-red-600 transition-colors"
          >
            <StopCircle size={16} />
            Stop Generation
          </button>
        ) : (
          <button
            onClick={runTestGeneration}
            disabled={!backendInfo.backend || !testPrompt.trim()}
            className={`w-full flex items-center justify-center gap-1.5 px-4 py-2.5 text-sm text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors ${
              capturedImage
                ? 'bg-purple-600 hover:bg-purple-700'
                : 'bg-orange-500 hover:bg-orange-600'
            }`}
          >
            {capturedImage && <ImageIcon size={14} />}
            <Play size={14} />
            Run Benchmark
          </button>
        )}

        {/* Error display */}
        {error && (
          <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-3">
            {error}
          </div>
        )}

        {/* Response output */}
        {testResponse && (
          <div>
            <div className="text-xs text-gray-500 mb-2 font-semibold uppercase tracking-wide">Response</div>
            <div className="p-4 bg-gray-50 border border-gray-200 rounded-xl text-sm text-gray-700 max-h-48 overflow-y-auto whitespace-pre-wrap font-mono leading-relaxed">
              {testResponse}
              {isGenerating && <span className="inline-block w-2 h-4 bg-orange-500 ml-0.5 animate-pulse" />}
            </div>
          </div>
        )}

        {/* Metrics display */}
        {testMetrics && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs bg-orange-50 border border-orange-200 rounded-xl px-4 py-3">
            <div className="flex items-center gap-1">
              <span className="text-gray-500">Tokens/sec:</span>
              <span className="font-mono font-semibold text-orange-700">{testMetrics.tokensPerSecond.toFixed(1)}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-gray-500">TTFT:</span>
              <span className="font-mono text-gray-700">{testMetrics.timeToFirstTokenMs.toFixed(0)}ms</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-gray-500">Tokens:</span>
              <span className="font-mono text-gray-700">{testMetrics.tokensGenerated}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-gray-500">Total:</span>
              <span className="font-mono text-gray-700">{(testMetrics.totalGenerationTimeMs / 1000).toFixed(2)}s</span>
            </div>
          </div>
        )}
      </div>

    </div>
  );
};

export default BenchmarkPanel;
