// components/BenchmarkPanel.tsx
// Universal benchmark panel for testing any loaded local model (llama.cpp, Transformers.js, or Ollama)

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Send,
  Trash2,
  Camera,
  X,
  AlertCircle,
  Cpu,
  StopCircle,
  CheckCircle,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
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

const mdComponents = {
  p: ({ children }: any) => <p className="mb-2 last:mb-0">{children}</p>,
  ul: ({ children }: any) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
  ol: ({ children }: any) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
  h1: ({ children }: any) => <h1 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h1>,
  h2: ({ children }: any) => <h2 className="text-sm font-bold mb-2 mt-2 first:mt-0">{children}</h2>,
  h3: ({ children }: any) => <h3 className="text-sm font-semibold mb-1 mt-2 first:mt-0">{children}</h3>,
  pre: ({ children }: any) => <pre className="bg-gray-800 text-gray-100 rounded-md p-2 my-2 overflow-x-auto text-xs">{children}</pre>,
  code: ({ className, children }: any) =>
    className ? <code className={className}>{children}</code>
      : <code className="bg-gray-200 text-gray-800 px-1 py-0.5 rounded text-xs">{children}</code>,
  table: ({ children }: any) => <div className="overflow-x-auto my-2"><table className="min-w-full border border-gray-300 text-xs">{children}</table></div>,
  th: ({ children }: any) => <th className="px-2 py-1 text-left font-semibold border-b border-gray-300 bg-gray-50">{children}</th>,
  td: ({ children }: any) => <td className="px-2 py-1 border-b border-gray-200">{children}</td>,
};

const BenchmarkPanel: React.FC<BenchmarkPanelProps> = ({ isVisible }) => {
  // Conversation state
  interface ChatMessage {
    role: 'user' | 'assistant';
    text: string;
    image?: string;
    metrics?: GenerationMetrics | null;
    error?: boolean;
  }
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const clearChat = () => {
    if (isGenerating) stopGeneration();
    setMessages([]);
  };

  // Send the whole conversation to the active backend; the reply streams into the last message
  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isGenerating || !backendInfo.backend) return;

    const userMsg: ChatMessage = { role: 'user', text, image: capturedImage ?? undefined };
    const history = [...messages.filter(m => !m.error), userMsg];
    const payload = history.map(m => ({
      role: m.role,
      content: m.image
        ? [{ type: 'image' as const, image: m.image }, { type: 'text' as const, text: m.text }]
        : m.text,
    }));

    setMessages([...messages, userMsg, { role: 'assistant', text: '' }]);
    setInput('');
    setCapturedImage(null);
    setIsGenerating(true);
    abortControllerRef.current = new AbortController();

    const patchReply = (patch: (m: ChatMessage) => ChatMessage) =>
      setMessages(prev => prev.map((m, i) => (i === prev.length - 1 ? patch(m) : m)));
    const onToken = (token: string) => patchReply(m => ({ ...m, text: m.text + token }));

    try {
      if (backendInfo.backend === 'llamacpp') {
        await NativeLlmManager.getInstance().generate(payload, onToken);
        const info = await NativeLlmManager.getInstance().getDebugInfo();
        patchReply(m => ({ ...m, metrics: info.engine.lastMetrics }));
      } else if (backendInfo.backend === 'transformers') {
        const startTime = performance.now();
        let tokensGenerated = 0;
        let firstTokenTime = 0;
        await GemmaModelManager.getInstance().generate(payload, (token) => {
          if (tokensGenerated === 0) firstTokenTime = performance.now() - startTime;
          tokensGenerated++;
          onToken(token);
        });
        const totalTime = performance.now() - startTime;
        patchReply(m => ({
          ...m,
          metrics: {
            tokensGenerated,
            promptTokens: 0, // Not available from Transformers.js
            timeToFirstTokenMs: firstTokenTime,
            totalGenerationTimeMs: totalTime,
            tokensPerSecond: tokensGenerated / (totalTime / 1000),
          },
        }));
      }
      // TODO: Add Ollama support
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      if (errorMessage !== 'Aborted') {
        patchReply(m => ({ ...m, text: `Error: ${errorMessage}`, error: true }));
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  if (!isVisible) return null;

  const disabled = !backendInfo.backend;

  return (
    <div className="space-y-4">
      {/* Active model info or no model warning */}
      {backendInfo.backend ? (
        <div className="flex items-center gap-3 px-4 py-3 bg-green-50 border border-green-200 rounded-xl">
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
            <p className="text-xs text-gray-500 mt-0.5">Load a local model to chat with it and see its speed.</p>
          </div>
        </div>
      )}

      {/* Conversation */}
      <div ref={scrollRef} className="border border-gray-200 rounded-xl bg-white p-3 h-80 overflow-y-auto space-y-3">
        {messages.length === 0 && (
          <div className="h-full flex items-center justify-center text-sm text-gray-400">
            Say hi to your model 👋
          </div>
        )}
        {messages.map((m, i) => {
          const isUser = m.role === 'user';
          const isLast = i === messages.length - 1;
          return (
            <div key={i} className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
              <div
                className={`max-w-[85%] px-3 py-2 rounded-2xl text-sm leading-relaxed ${isUser || m.error ? 'whitespace-pre-wrap' : ''} break-words ${
                  isUser
                    ? 'bg-purple-600 text-white'
                    : m.error
                      ? 'bg-red-50 border border-red-200 text-red-600'
                      : 'bg-gray-100 text-gray-800'
                }`}
              >
                {m.image && <img src={m.image} alt="Attached" className="mb-2 max-h-40 rounded-lg" />}
                {isUser || m.error ? m.text : (
                  <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={mdComponents}>
                    {m.text}
                  </ReactMarkdown>
                )}
                {!isUser && isLast && isGenerating && (
                  <span className="inline-block w-2 h-4 bg-purple-600 ml-0.5 align-middle animate-pulse" />
                )}
              </div>
              {m.metrics && (
                <div className="mt-1 flex flex-wrap gap-x-3 text-[11px] text-gray-400 font-mono">
                  <span><span className="text-purple-600 font-semibold">{m.metrics.tokensPerSecond.toFixed(1)}</span> tok/s</span>
                  <span>TTFT {m.metrics.timeToFirstTokenMs.toFixed(0)}ms</span>
                  <span>{m.metrics.tokensGenerated} tokens</span>
                  <span>{(m.metrics.totalGenerationTimeMs / 1000).toFixed(2)}s</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Composer */}
      <div className="space-y-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleImageCapture}
          className="hidden"
        />
        {capturedImage && (
          <div className="relative inline-block">
            <img src={capturedImage} alt="Captured" className="h-20 rounded-lg border border-purple-200 object-cover" />
            <button
              onClick={removeImage}
              disabled={isGenerating}
              className="absolute -top-1.5 -right-1.5 p-0.5 bg-white border border-gray-200 rounded-full text-gray-500 hover:text-red-500"
              title="Remove image"
            >
              <X size={12} />
            </button>
          </div>
        )}
        <div className="flex items-end gap-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isGenerating || disabled}
            className={`p-2.5 rounded-xl border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              capturedImage ? 'bg-purple-100 text-purple-700 border-purple-300' : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border-gray-200'
            }`}
            title="Attach image"
          >
            <Camera size={16} />
          </button>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder={capturedImage ? 'Ask about the image...' : 'Type a message...'}
            rows={1}
            disabled={disabled}
            className="flex-1 p-2.5 text-sm border border-gray-200 rounded-xl resize-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          />
          {isGenerating ? (
            <button onClick={stopGeneration} className="p-2.5 text-white rounded-xl bg-red-500 hover:bg-red-600 transition-colors" title="Stop">
              <StopCircle size={16} />
            </button>
          ) : (
            <button
              onClick={sendMessage}
              disabled={disabled || !input.trim()}
              className="p-2.5 text-white rounded-xl bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              title="Send"
            >
              <Send size={16} />
            </button>
          )}
          <button
            onClick={clearChat}
            disabled={messages.length === 0}
            className="p-2.5 rounded-xl text-gray-400 hover:text-red-500 hover:bg-red-50 disabled:opacity-30 transition-colors"
            title="Clear conversation"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default BenchmarkPanel;
