// src/components/AICreator/DetectionMode/DetectionModePanel.tsx

import React, { useState, useRef, useEffect } from 'react';
import { X, Upload, Clock, Sparkles, ChevronDown, AlertTriangle, Check, Loader2, XCircle, FileText, ChevronRight } from 'lucide-react';
import ClassificationColumns from './ClassificationColumns';
import { listAgents, getAgent, type CompleteAgent } from '@utils/agent_database';
import { IterationStore } from '@utils/IterationStore';
import {
  finetuneLoop,
  getServerForModel,
  type TestCase,
  type FinetuneConfig,
  type FinetuneProgress
} from '@utils/finetuneClassifier';
import type { DetectionCategory, ClassifiedImage, DetectionModeInitialData, FinetuneState } from './types';
import type { TokenProvider } from '@utils/main_loop';

interface DetectionModePanelProps {
  onClose: () => void;
  onComplete: (agentBlock: string) => void;
  initialData?: DetectionModeInitialData;
  getToken: TokenProvider;
  isUsingObServer: boolean;
  selectedLocalModel: string;
}

const DEFAULT_CATEGORIES: DetectionCategory[] = [
  { id: 'cat1', label: 'POSITIVE', images: [] },
  { id: 'cat2', label: 'NEGATIVE', images: [] }
];

const MAX_ITERATIONS = 5;
const DEFAULT_FINETUNER_MODEL = 'gemini-2.5-flash-lite';

const DetectionModePanel: React.FC<DetectionModePanelProps> = ({
  onClose,
  onComplete,
  initialData,
  getToken,
  isUsingObServer,
  selectedLocalModel
}) => {
  const [categories, setCategories] = useState<DetectionCategory[]>(
    initialData?.categories ?? DEFAULT_CATEGORIES
  );

  // History state
  const [agents, setAgents] = useState<CompleteAgent[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(
    initialData?.sourceAgentId ?? null
  );
  const [historyImages, setHistoryImages] = useState<ClassifiedImage[]>([]);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Finetuning state
  const [finetuneState, setFinetuneState] = useState<FinetuneState | null>(null);
  const [sourceAgent, setSourceAgent] = useState<CompleteAgent | null>(null);
  const [isPromptExpanded, setIsPromptExpanded] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load agents list on mount
  useEffect(() => {
    const loadAgents = async () => {
      try {
        const agentList = await listAgents();
        setAgents(agentList);
      } catch (err) {
        console.error('Failed to load agents:', err);
      }
    };
    loadAgents();
  }, []);

  // Load source agent when selectedAgentId changes
  useEffect(() => {
    if (!selectedAgentId) {
      setSourceAgent(null);
      return;
    }

    const loadSourceAgent = async () => {
      try {
        const agent = await getAgent(selectedAgentId);
        setSourceAgent(agent);
      } catch (err) {
        console.error('Failed to load source agent:', err);
        setSourceAgent(null);
      }
    };
    loadSourceAgent();
  }, [selectedAgentId]);

  // Load history images when agent is selected
  useEffect(() => {
    if (!selectedAgentId) {
      setHistoryImages([]);
      return;
    }

    const loadHistory = async () => {
      setIsLoadingHistory(true);
      try {
        const sessions = await IterationStore.getHistoricalSessions(selectedAgentId);
        const images: ClassifiedImage[] = [];

        sessions.forEach(session => {
          session.iterations.forEach(iteration => {
            // Get screenshot/camera sensors
            iteration.sensors.forEach(sensor => {
              if ((sensor.type === 'screenshot' || sensor.type === 'camera') && sensor.content) {
                const imageData = typeof sensor.content === 'string'
                  ? sensor.content
                  : sensor.content.data;

                if (imageData) {
                  images.push({
                    id: `${iteration.id}-${sensor.type}-${sensor.timestamp}`,
                    data: imageData,
                    source: 'history',
                    sourceAgentId: selectedAgentId,
                    timestamp: sensor.timestamp
                  });
                }
              }
            });

            // Also get model images if available
            if (iteration.modelImages) {
              iteration.modelImages.forEach((imgData, idx) => {
                images.push({
                  id: `${iteration.id}-model-${idx}`,
                  data: imgData,
                  source: 'history',
                  sourceAgentId: selectedAgentId,
                  timestamp: iteration.startTime
                });
              });
            }
          });
        });

        setHistoryImages(images);
      } catch (err) {
        console.error('Failed to load history:', err);
      } finally {
        setIsLoadingHistory(false);
      }
    };

    loadHistory();
  }, [selectedAgentId]);

  // Image handlers
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach(file => {
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const result = e.target?.result as string;
          const base64Data = result.split(',')[1];

          const newImage: ClassifiedImage = {
            id: `upload-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            data: base64Data,
            source: 'upload'
          };

          // Add to first category by default
          setCategories(prev => prev.map((cat, idx) =>
            idx === 0
              ? { ...cat, images: [...cat.images, newImage] }
              : cat
          ));
        };
        reader.readAsDataURL(file);
      }
    });

    // Clear input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAddHistoryImage = (image: ClassifiedImage) => {
    // Add to first category by default (click behavior)
    setCategories(prev => prev.map((cat, idx) =>
      idx === 0
        ? { ...cat, images: [...cat.images, { ...image, id: `${image.id}-${Date.now()}` }] }
        : cat
    ));
  };

  const handleAddImageToCategory = (image: ClassifiedImage, categoryId: string) => {
    // Add image to specific category (drag-drop behavior)
    setCategories(prev => prev.map(cat =>
      cat.id === categoryId
        ? { ...cat, images: [...cat.images, image] }
        : cat
    ));
  };

  const handleHistoryDragStart = (e: React.DragEvent, image: ClassifiedImage) => {
    e.dataTransfer.setData('application/json', JSON.stringify({
      fromCategoryId: 'history',
      imageData: image.data
    }));
    e.dataTransfer.effectAllowed = 'copy';
  };

  const handleMoveImage = (imageId: string, fromCategoryId: string, toCategoryId: string) => {
    setCategories(prev => {
      let movedImage: ClassifiedImage | null = null;

      // Find and remove from source
      const updated = prev.map(cat => {
        if (cat.id === fromCategoryId) {
          const image = cat.images.find(img => img.id === imageId);
          if (image) movedImage = image;
          return { ...cat, images: cat.images.filter(img => img.id !== imageId) };
        }
        return cat;
      });

      // Add to target
      if (movedImage) {
        return updated.map(cat =>
          cat.id === toCategoryId
            ? { ...cat, images: [...cat.images, movedImage!] }
            : cat
        );
      }

      return updated;
    });
  };

  const handleRemoveImage = (imageId: string, categoryId: string) => {
    setCategories(prev => prev.map(cat =>
      cat.id === categoryId
        ? { ...cat, images: cat.images.filter(img => img.id !== imageId) }
        : cat
    ));
  };

  const handleUpdateLabel = (categoryId: string, newLabel: string) => {
    setCategories(prev => prev.map(cat =>
      cat.id === categoryId
        ? { ...cat, label: newLabel }
        : cat
    ));
  };

  const handleStartFinetune = async () => {
    if (!sourceAgent) {
      console.error('No source agent selected');
      return;
    }

    // Build test cases from categories
    const testCases: TestCase[] = [];
    categories.forEach(cat => {
      cat.images.forEach(img => {
        testCases.push({
          imageData: img.data,
          expectedLabel: cat.label
        });
      });
    });

    if (testCases.length === 0) {
      console.error('No test cases');
      return;
    }

    // Get server address
    const serverAddress = getServerForModel(
      sourceAgent.model_name,
      isUsingObServer
    );

    // Get token if using ObServer
    let token: string | undefined;
    if (isUsingObServer) {
      token = await getToken();
      if (!token) {
        console.error('Failed to get authentication token');
        return;
      }
    }

    // Determine finetuner model
    const finetunerModel = isUsingObServer
      ? DEFAULT_FINETUNER_MODEL
      : selectedLocalModel || sourceAgent.model_name;

    const config: FinetuneConfig = {
      testModel: sourceAgent.model_name,
      finetunerModel,
      serverAddress,
      token,
      maxIterations: MAX_ITERATIONS,
      categories: categories.map(c => c.label)
    };

    // Initialize finetuning state
    setFinetuneState({
      phase: 'testing',
      iteration: 1,
      maxIterations: MAX_ITERATIONS,
      currentPrompt: sourceAgent.system_prompt,
      testResults: [],
      currentTestIndex: 0,
      totalTests: testCases.length
    });

    // Create abort controller
    abortControllerRef.current = new AbortController();

    try {
      const result = await finetuneLoop(
        sourceAgent.system_prompt,
        testCases,
        config,
        (iteration, results, newPrompt, phase) => {
          setFinetuneState(prev => prev ? {
            ...prev,
            iteration,
            testResults: results,
            previousPrompt: phase === 'improving' ? prev.currentPrompt : prev.previousPrompt,
            currentPrompt: newPrompt,
            phase: phase === 'improving' ? 'improving' : 'testing'
          } : null);
        },
        (progress: FinetuneProgress) => {
          setFinetuneState(prev => prev ? {
            ...prev,
            phase: progress.phase,
            iteration: progress.iteration,
            currentTestIndex: progress.currentTestIndex ?? prev.currentTestIndex,
            totalTests: progress.totalTests ?? prev.totalTests,
            // Update test results in real-time as they come in
            testResults: progress.allResults ?? prev.testResults
          } : null);
        },
        abortControllerRef.current.signal
      );

      // Update state with final result
      setFinetuneState(prev => prev ? {
        ...prev,
        phase: result.success ? 'complete' : 'failed',
        testResults: result.finalResults,
        currentPrompt: result.finalPrompt,
        iteration: result.iterations
      } : null);

    } catch (error) {
      console.error('Finetuning error:', error);
      setFinetuneState(prev => prev ? {
        ...prev,
        phase: 'failed'
      } : null);
    }
  };

  const handleAbort = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setFinetuneState(null);
  };

  const handleUseBestPrompt = () => {
    if (!finetuneState || !sourceAgent) return;
    generateAndCompleteAgent(finetuneState.currentPrompt);
  };

  const handleRetry = () => {
    setFinetuneState(null);
  };

  const handleSaveAgent = () => {
    if (!finetuneState || !sourceAgent) return;
    generateAndCompleteAgent(finetuneState.currentPrompt);
  };

  const generateAndCompleteAgent = (systemPrompt: string) => {
    if (!sourceAgent) return;

    const categoryLabels = categories.map(c => c.label).join(', ');

    // Generate code
    const codeChecks = categories.map((cat, idx) => {
      if (idx === 0) {
        return `if (response.includes("${cat.label}")) {
  // Handle ${cat.label} detection
  console.log("Detected: ${cat.label}");
}`;
      }
      return `else if (response.includes("${cat.label}")) {
  // Handle ${cat.label} detection
  console.log("Detected: ${cat.label}");
}`;
    }).join(' ');

    const agentId = `${sourceAgent.id}`;
    const agentName = `${sourceAgent.name}`;

    // Build the $$$ block
    const agentBlock = `$$$
id: ${agentId}
name: ${agentName}
description: Classifies images into: ${categoryLabels}
model_name: ${sourceAgent.model_name}
loop_interval_seconds: ${sourceAgent.loop_interval_seconds}
system_prompt: |
${systemPrompt.split('\n').map(line => '  ' + line).join('\n')}
code: |
${codeChecks.split('\n').map(line => '  ' + line).join('\n')}
memory: ""
$$$`;

    onComplete(agentBlock);
  };

  const canFinetune = categories.every(cat => cat.images.length > 0) && sourceAgent !== null;

  // Calculate test stats
  const passedTests = finetuneState?.testResults.filter(r => r.passed).length ?? 0;
  const totalTests = finetuneState?.testResults.length ?? 0;
  const passRate = totalTests > 0 ? Math.round((passedTests / totalTests) * 100) : 0;

  // Render finetuning UI
  if (finetuneState && finetuneState.phase !== 'idle') {
    return (
      <div className="w-full bg-gradient-to-br from-purple-50 to-indigo-50 border border-purple-200 rounded-xl p-6 relative shadow-sm">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-3 right-3 p-2 rounded-full hover:bg-white/50 transition-colors group"
        >
          <X className="h-5 w-5 text-purple-600 group-hover:text-purple-800" />
        </button>

        <div className="pr-10">
          {/* Testing or Improving Phase */}
          {(finetuneState.phase === 'testing' || finetuneState.phase === 'improving') && (
            <>
              <h3 className="text-purple-900 font-semibold text-lg mb-4">
                Finetuning Classifier (Attempt {finetuneState.iteration}/{finetuneState.maxIterations})
              </h3>

              {/* System Prompt Preview - Collapsible */}
              <div className="mb-4">
                <button
                  onClick={() => setIsPromptExpanded(!isPromptExpanded)}
                  className="flex items-center text-sm font-medium text-purple-700 hover:text-purple-900 mb-2"
                >
                  <ChevronRight className={`h-4 w-4 mr-1 transition-transform ${isPromptExpanded ? 'rotate-90' : ''}`} />
                  <FileText className="h-4 w-4 mr-1" />
                  {finetuneState.phase === 'improving' ? 'Editing System Prompt...' : 'Current System Prompt'}
                </button>
                {isPromptExpanded && (
                  <div className="bg-white border border-purple-200 rounded-lg p-3 max-h-40 overflow-y-auto">
                    <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                      {finetuneState.currentPrompt}
                    </pre>
                  </div>
                )}
              </div>

              {/* Image test results grid */}
              {finetuneState.phase === 'testing' && (
                <div className="mb-4">
                  <div className="flex flex-wrap gap-3 mb-3">
                    {categories.flatMap((cat, catIdx) =>
                      cat.images.map((img, imgIdx) => {
                        const globalIdx = categories.slice(0, catIdx).reduce((sum, c) => sum + c.images.length, 0) + imgIdx;
                        const result = finetuneState.testResults[globalIdx];
                        const isCurrent = globalIdx === finetuneState.currentTestIndex - 1 && finetuneState.testResults.length <= globalIdx;
                        const isTested = globalIdx < finetuneState.testResults.length;

                        return (
                          <div
                            key={img.id}
                            className="flex flex-col items-center"
                          >
                            <div
                              className={`relative w-16 h-16 rounded-lg overflow-hidden border-2 ${
                                isCurrent ? 'border-blue-400 ring-2 ring-blue-300' :
                                result?.passed ? 'border-green-400' :
                                result && !result.passed ? 'border-red-400' :
                                'border-gray-200'
                              }`}
                            >
                              <img
                                src={`data:image/png;base64,${img.data}`}
                                alt={`Test ${globalIdx + 1}`}
                                className="w-full h-full object-cover"
                              />
                              {isTested && (
                                <div className={`absolute bottom-0 right-0 p-0.5 rounded-tl ${
                                  result?.passed ? 'bg-green-500' : 'bg-red-500'
                                }`}>
                                  {result?.passed ?
                                    <Check className="h-3 w-3 text-white" /> :
                                    <XCircle className="h-3 w-3 text-white" />
                                  }
                                </div>
                              )}
                              {!isTested && globalIdx === finetuneState.testResults.length && (
                                <div className="absolute inset-0 bg-blue-500/20 flex items-center justify-center">
                                  <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
                                </div>
                              )}
                            </div>
                            {/* Show model response under each image */}
                            {isTested && (
                              <div className={`mt-1 px-1.5 py-0.5 rounded text-xs font-medium max-w-[70px] truncate ${
                                result?.passed
                                  ? 'bg-green-100 text-green-700'
                                  : 'bg-red-100 text-red-700'
                              }`} title={`Model: "${result?.actual}" | Expected: "${result?.expected}"`}>
                                {result?.actual}
                              </div>
                            )}
                            {!isTested && (
                              <div className="mt-1 px-1.5 py-0.5 rounded text-xs text-gray-400 max-w-[70px] truncate">
                                {cat.label}
                              </div>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>

                  <p className="text-sm text-gray-600 mb-2">
                    Testing image {Math.min(finetuneState.testResults.length + 1, finetuneState.totalTests)}/{finetuneState.totalTests}...
                  </p>

                  {/* Progress bar */}
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
                    <div
                      className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${passRate}%` }}
                    />
                  </div>

                  <p className="text-sm font-medium text-gray-700 mb-2">
                    {passRate}% passing ({passedTests}/{totalTests})
                  </p>

                  {/* Show failures in real-time */}
                  {finetuneState.testResults.filter(r => !r.passed).length > 0 && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-2 max-h-24 overflow-y-auto">
                      {finetuneState.testResults.filter(r => !r.passed).map(result => (
                        <p key={result.imageIndex} className="text-xs text-red-600">
                          Image {result.imageIndex + 1}: expected "{result.expected}" got "{result.actual}"
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Improving phase */}
              {finetuneState.phase === 'improving' && (
                <div className="mb-4">
                  <p className="text-sm text-gray-600 mb-2">
                    Improving prompt based on {finetuneState.testResults.filter(r => !r.passed).length} failed tests...
                  </p>

                  {/* Show what failed */}
                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-2 mb-3 max-h-24 overflow-y-auto">
                    <p className="text-xs font-medium text-amber-700 mb-1">Failures to fix:</p>
                    {finetuneState.testResults.filter(r => !r.passed).map(result => (
                      <p key={result.imageIndex} className="text-xs text-amber-600">
                        Image {result.imageIndex + 1}: expected "{result.expected}" got "{result.actual}"
                      </p>
                    ))}
                  </div>

                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-purple-600 h-2 rounded-full animate-pulse" style={{ width: '60%' }} />
                  </div>
                </div>
              )}

              {/* Abort button */}
              <button
                onClick={handleAbort}
                className="px-4 py-2 text-sm font-medium text-red-700 bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 transition-colors"
              >
                Abort
              </button>
            </>
          )}

          {/* Success Phase */}
          {finetuneState.phase === 'complete' && (
            <>
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center mr-3">
                  <Check className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h3 className="text-green-800 font-semibold text-lg">
                    Finetuning Complete!
                  </h3>
                  <p className="text-green-700 text-sm">
                    All {totalTests}/{totalTests} tests passing after {finetuneState.iteration} iteration{finetuneState.iteration > 1 ? 's' : ''}
                  </p>
                </div>
              </div>

              {/* Final System Prompt Preview */}
              <div className="mb-4">
                <button
                  onClick={() => setIsPromptExpanded(!isPromptExpanded)}
                  className="flex items-center text-sm font-medium text-green-700 hover:text-green-900 mb-2"
                >
                  <ChevronRight className={`h-4 w-4 mr-1 transition-transform ${isPromptExpanded ? 'rotate-90' : ''}`} />
                  <FileText className="h-4 w-4 mr-1" />
                  Final System Prompt
                </button>
                {isPromptExpanded && (
                  <div className="bg-white border border-green-200 rounded-lg p-3 max-h-40 overflow-y-auto">
                    <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                      {finetuneState.currentPrompt}
                    </pre>
                  </div>
                )}
              </div>

              <button
                onClick={handleSaveAgent}
                className="flex items-center px-5 py-2 text-sm font-semibold text-white bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 rounded-lg transition-all shadow-md"
              >
                <Sparkles className="h-4 w-4 mr-2" />
                Save Agent
              </button>
            </>
          )}

          {/* Failed Phase (max iterations reached) */}
          {finetuneState.phase === 'failed' && (
            <>
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-amber-100 rounded-full flex items-center justify-center mr-3">
                  <AlertTriangle className="h-6 w-6 text-amber-600" />
                </div>
                <div>
                  <h3 className="text-amber-800 font-semibold text-lg">
                    Best Result After {finetuneState.iteration} Attempts
                  </h3>
                  <p className="text-amber-700 text-sm">
                    {passedTests}/{totalTests} tests passing
                  </p>
                </div>
              </div>

              {/* Best System Prompt Preview */}
              <div className="mb-4">
                <button
                  onClick={() => setIsPromptExpanded(!isPromptExpanded)}
                  className="flex items-center text-sm font-medium text-amber-700 hover:text-amber-900 mb-2"
                >
                  <ChevronRight className={`h-4 w-4 mr-1 transition-transform ${isPromptExpanded ? 'rotate-90' : ''}`} />
                  <FileText className="h-4 w-4 mr-1" />
                  Best System Prompt
                </button>
                {isPromptExpanded && (
                  <div className="bg-white border border-amber-200 rounded-lg p-3 max-h-40 overflow-y-auto">
                    <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                      {finetuneState.currentPrompt}
                    </pre>
                  </div>
                )}
              </div>

              {/* Show failures */}
              <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-2 max-h-24 overflow-y-auto">
                <p className="text-xs font-medium text-red-700 mb-1">Still failing:</p>
                {finetuneState.testResults.filter(r => !r.passed).map(result => (
                  <p key={result.imageIndex} className="text-xs text-red-600">
                    Image {result.imageIndex + 1}: expected "{result.expected}" got "{result.actual}"
                  </p>
                ))}
              </div>

              <div className="flex gap-3">
                <button
                  onClick={handleUseBestPrompt}
                  className="flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 rounded-lg transition-all shadow-md"
                >
                  Use This Prompt
                </button>
                <button
                  onClick={handleRetry}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Try Again
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  // Normal classification UI
  return (
    <div className="w-full bg-gradient-to-br from-purple-50 to-indigo-50 border border-purple-200 rounded-xl p-6 relative shadow-sm">
      {/* Close button */}
      <button
        onClick={onClose}
        className="absolute top-3 right-3 p-2 rounded-full hover:bg-white/50 transition-colors group"
      >
        <X className="h-5 w-5 text-purple-600 group-hover:text-purple-800" />
      </button>

      <div className="pr-10">
        <h3 className="text-purple-900 font-semibold text-lg mb-4">
          Classification Agent Builder
        </h3>

        {/* Agent selector (shown when no source agent from initial data) */}
        {!initialData?.sourceAgentId && (
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Source Agent (for model & prompt)
            </label>
            <select
              value={selectedAgentId || ''}
              onChange={(e) => setSelectedAgentId(e.target.value || null)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              <option value="">Select an agent...</option>
              {agents.map(agent => (
                <option key={agent.id} value={agent.id}>
                  {agent.name} ({agent.model_name})
                </option>
              ))}
            </select>
            {!selectedAgentId && (
              <p className="text-xs text-amber-600 mt-1">
                Select a source agent to use its model and prompt as the starting point
              </p>
            )}
          </div>
        )}

        {/* Classification Columns */}
        <div className="mb-4">
          <p className="text-sm text-gray-600 mb-2">
            Drag images between columns to classify them:
          </p>
          <ClassificationColumns
            categories={categories}
            onMoveImage={handleMoveImage}
            onRemoveImage={handleRemoveImage}
            onUpdateLabel={handleUpdateLabel}
            onAddImage={handleAddImageToCategory}
          />
        </div>

        {/* Action Bar */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Upload Button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg hover:from-purple-700 hover:to-indigo-700 transition-all shadow-md"
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload
          </button>

          {/* History Dropdown */}
          <div className="relative">
            <button
              onClick={() => setIsHistoryOpen(!isHistoryOpen)}
              className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <Clock className="h-4 w-4 mr-2" />
              History
              <ChevronDown className={`h-4 w-4 ml-2 transition-transform ${isHistoryOpen ? 'rotate-180' : ''}`} />
            </button>

            {isHistoryOpen && (
              <div className="absolute top-full left-0 mt-1 w-72 bg-white rounded-lg shadow-xl border border-gray-200 z-10 max-h-80 overflow-hidden">
                {/* Agent Selector */}
                <div className="p-2 border-b border-gray-100">
                  <select
                    value={selectedAgentId || ''}
                    onChange={(e) => setSelectedAgentId(e.target.value || null)}
                    className="w-full px-2 py-1.5 text-sm border border-gray-200 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="">Select an agent...</option>
                    {agents.map(agent => (
                      <option key={agent.id} value={agent.id}>
                        {agent.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* History Images Grid */}
                <div className="p-2 max-h-52 overflow-y-auto">
                  {isLoadingHistory ? (
                    <p className="text-sm text-gray-500 text-center py-4">Loading...</p>
                  ) : !selectedAgentId ? (
                    <p className="text-sm text-gray-500 text-center py-4">Select an agent to view history</p>
                  ) : historyImages.length === 0 ? (
                    <p className="text-sm text-gray-500 text-center py-4">No images in history</p>
                  ) : (
                    <div className="grid grid-cols-4 gap-2">
                      {historyImages.map(image => (
                        <div
                          key={image.id}
                          draggable
                          onDragStart={(e) => handleHistoryDragStart(e, image)}
                          onClick={() => handleAddHistoryImage(image)}
                          className="w-14 h-14 rounded overflow-hidden border-2 border-transparent hover:border-purple-400 transition-colors cursor-grab active:cursor-grabbing"
                          title="Click to add or drag to a category"
                        >
                          <img
                            src={`data:image/png;base64,${image.data}`}
                            alt="History"
                            className="w-full h-full object-cover pointer-events-none"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Finetune Button */}
          <button
            onClick={handleStartFinetune}
            disabled={!canFinetune}
            className={`flex items-center px-5 py-2 text-sm font-semibold rounded-lg transition-all shadow-md ${
              canFinetune
                ? 'text-white bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700'
                : 'text-gray-400 bg-gray-200 cursor-not-allowed'
            }`}
            title={canFinetune ? 'Start finetuning' : 'Select an agent and add images to each category'}
          >
            <Sparkles className="h-4 w-4 mr-2" />
            Finetune Agent
          </button>
        </div>

        {!canFinetune && (
          <p className="text-xs text-gray-500 mt-2">
            {!sourceAgent
              ? 'Select a source agent to use its model and prompt'
              : 'Add at least 1 image to each category to finetune the agent'}
          </p>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />
    </div>
  );
};

export default DetectionModePanel;
