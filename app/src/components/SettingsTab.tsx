import React, { useState, useEffect, useRef } from 'react';
import { Settings, TestTube2, Loader2, FileDown, CheckCircle2, Database, Trash2, Cloud, Server, Cpu, Mic, Monitor, Play, Square, Volume2, Sliders, RotateCcw } from 'lucide-react';
import { SensorSettings } from '../utils/settings';
import { StreamManager } from '../utils/streamManager';
import { inferenceConfigStore } from '../utils/inferenceConfigStore';
import { InferenceParams } from '../config/inference-params';
import InferenceParamsEditor from './InferenceParamsEditor';

// Whisper imports
import { WhisperModelManager } from '../utils/whisper/WhisperModelManager';
import { TranscriptionRouter } from '../utils/whisper/TranscriptionRouter';
import { WhisperModelState, TranscriptionMode } from '../utils/whisper/types';
import { useSubscriberText } from '../hooks/useTranscriptionState';
import { SUGGESTED_MODELS, LANGUAGE_NAMES } from '../config/whisper-models';

import { AVAILABLE_OCR_LANGUAGES } from '../config/ocr-languages';

// Change Detection component
import ChangeDetectionSettings from './ChangeDetectionSettings';




// Reusable Card Component (Your existing component)
const SettingsCard: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="bg-white shadow-md rounded-lg mb-6">
    <div className="p-4 border-b">
      <h3 className="text-lg font-semibold flex items-center">
        <Settings className="h-5 w-5 mr-2 text-gray-500" />
        {title}
      </h3>
    </div>
    <div className="p-6">{children}</div>
  </div>
);

const SettingsTab = () => {

  // --- OCR State Management (Existing) ---
  const [ocrLang, setOcrLang] = useState(SensorSettings.getOcrLanguage());
  const [ocrConfidence, setOcrConfidence] = useState(SensorSettings.getOcrConfidenceThreshold());

  // --- Inference Parameters State ---
  const [globalInferenceParams, setGlobalInferenceParams] = useState<Partial<InferenceParams>>(
    inferenceConfigStore.getGlobalDefaults()
  );

  const handleGlobalInferenceParamsChange = (params: Partial<InferenceParams>) => {
    setGlobalInferenceParams(params);
    inferenceConfigStore.setGlobalDefaults(params);
  };

  const handleResetInferenceParams = () => {
    inferenceConfigStore.resetGlobalDefaults();
    setGlobalInferenceParams(inferenceConfigStore.getGlobalDefaults());
  };


  // --- OCR Handler Functions (Existing) ---
  const handleOcrLangChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newLang = e.target.value;
    setOcrLang(newLang);
    SensorSettings.setOcrLanguage(newLang);
  };

  const handleOcrConfidenceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newConfidence = parseInt(e.target.value, 10);
    setOcrConfidence(newConfidence);
    SensorSettings.setOcrConfidenceThreshold(newConfidence);
  };


  // --- NEW WHISPER STATE ---
  const [whisperSettings, setWhisperSettings] = useState(SensorSettings.getWhisperSettings());
  const [modelState, setModelState] = useState<WhisperModelState | null>(null);
  const [isTestRunning, setIsTestRunning] = useState(false);
  const [transcriptionMode, setTranscriptionModeState] = useState<TranscriptionMode>(
    TranscriptionRouter.getInstance().getMode()
  );
  const [selfHostedUrl, setSelfHostedUrl] = useState(SensorSettings.getSelfHostedWhisperUrl());

  // --- AUDIO TEST STATE ---
  type AudioTestSource = 'microphone' | 'screenAudio' | 'allAudio';
  interface TranscriptionRecord {
    id: string;
    transcript: string;
    audioUrl: string | null;
    timestamp: Date;
    source: AudioTestSource;
  }
  const [audioTestSource, setAudioTestSource] = useState<AudioTestSource>('microphone');
  const [transcriptionHistory, setTranscriptionHistory] = useState<TranscriptionRecord[]>([]);
  const [playingRecordId, setPlayingRecordId] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const currentTestIdRef = useRef<string | null>(null);
  const TEST_AGENT_ID = 'settings-audio-test';

  // Use subscriber text hook - shows what this test agent is subscribed to
  const transcriptionStreamType = audioTestSource === 'microphone' ? 'microphone' : 'screenAudio';
  const { committedText, interimText } = useSubscriberText(TEST_AGENT_ID, transcriptionStreamType);

  // Model manager instance
  const modelManager = WhisperModelManager.getInstance();

  // Subscribe to model state changes
  useEffect(() => {
    const unsubscribe = modelManager.onStateChange(setModelState);
    setModelState(modelManager.getState());
    return unsubscribe;
  }, [modelManager]);


  // --- TRANSCRIPTION MODE HANDLER ---
  const handleTranscriptionModeChange = (mode: TranscriptionMode) => {
    if (isTestRunning) {
      handleStopTest();
    }
    TranscriptionRouter.getInstance().setMode(mode);
    setTranscriptionModeState(mode);
  };

  const handleSelfHostedUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelfHostedUrl(e.target.value);
    SensorSettings.setSelfHostedWhisperUrl(e.target.value);
  };

  // --- SIMPLE WHISPER HANDLERS ---

  const handleModelIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newSettings = { ...whisperSettings, modelId: e.target.value };
    setWhisperSettings(newSettings);
    SensorSettings.setWhisperModelId(e.target.value);
  };

  const handleTaskChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const task = e.target.value || undefined;
    const newSettings = { ...whisperSettings, task: task as any };
    setWhisperSettings(newSettings);
    SensorSettings.setWhisperTask(task as any);
  };

  const handleLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const language = e.target.value || undefined;
    const newSettings = { ...whisperSettings, language };
    setWhisperSettings(newSettings);
    SensorSettings.setWhisperLanguage(language);
  };

  const handleQuantizedChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newSettings = { ...whisperSettings, quantized: e.target.checked };
    setWhisperSettings(newSettings);
    SensorSettings.setWhisperQuantized(e.target.checked);
  };

  const handleChunkDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newDuration = parseInt(e.target.value, 10);
    const newSettings = { ...whisperSettings, chunkDurationMs: newDuration };
    setWhisperSettings(newSettings);
    SensorSettings.setWhisperChunkDuration(newDuration);
  };

  const handleLoadModel = async () => {
    try {
      await modelManager.loadModel();
    } catch (error) {
      console.error('Failed to load model:', error);
      alert(`Failed to load model: ${error}`);
    }
  };

  const handleUnloadModel = () => {
    if (isTestRunning) {
      handleStopTest();
    }
    modelManager.unloadModel();
  };


  const handleStartTest = async () => {
    // For local mode, ensure model is loaded
    if (transcriptionMode === 'local' && !modelManager.isReady()) {
      alert('Please load a model first');
      return;
    }

    // For self-hosted mode, ensure URL is configured
    if (transcriptionMode === 'self-hosted' && !selfHostedUrl.trim()) {
      alert('Please enter a Whisper server URL');
      return;
    }

    try {
      // Create new test ID and clear recording chunks
      currentTestIdRef.current = `test-${Date.now()}`;
      recordedChunksRef.current = [];

      // Map audio source to stream types
      const streamTypeMap: Record<AudioTestSource, ('microphone' | 'screenAudio')[]> = {
        'microphone': ['microphone'],
        'screenAudio': ['screenAudio'],
        'allAudio': ['microphone', 'screenAudio'],
      };
      const requiredStreams = streamTypeMap[audioTestSource];

      // Use StreamManager to acquire streams
      await StreamManager.requestStreamsForAgent(TEST_AGENT_ID, requiredStreams);
      const streams = StreamManager.getCurrentState();

      // Get the audio stream for MediaRecorder (for playback)
      let audioStream: MediaStream | null = null;

      if (audioTestSource === 'microphone') {
        audioStream = streams.microphoneStream;
      } else if (audioTestSource === 'screenAudio') {
        audioStream = streams.screenAudioStream;
      } else if (audioTestSource === 'allAudio') {
        // For allAudio, use screenAudio for recording (or fallback to mic)
        audioStream = streams.screenAudioStream || streams.microphoneStream;
      }

      if (!audioStream) {
        throw new Error(`Failed to acquire ${audioTestSource} stream`);
      }

      // Set up MediaRecorder for recording (for playback in history)
      try {
        const recorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm;codecs=opus' });
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            recordedChunksRef.current.push(e.data);
          }
        };
        recorder.start(1000); // Collect chunks every second
        mediaRecorderRef.current = recorder;
      } catch (recorderError) {
        console.warn('MediaRecorder not supported, playback will not be available:', recorderError);
      }

      // Transcription is already set up by StreamManager.requestStreamsForAgent()
      // via TranscriptionRouter.acquireService() - no need to create our own
      setIsTestRunning(true);
    } catch (error) {
      console.error('Failed to start transcription test:', error);
      StreamManager.releaseStreamsForAgent(TEST_AGENT_ID);
      alert(`Failed to start test: ${error}`);
    }
  };

  const handleStopTest = () => {
    // Capture current transcript before stopping (using deduplicated values)
    const currentTranscript = committedText + (committedText && interimText ? ' ' : '') + interimText;
    const testId = currentTestIdRef.current;
    const testSource = audioTestSource;

    // Stop MediaRecorder and save to history
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.onstop = () => {
        let audioUrl: string | null = null;
        if (recordedChunksRef.current.length > 0) {
          const blob = new Blob(recordedChunksRef.current, { type: 'audio/webm' });
          audioUrl = URL.createObjectURL(blob);
        }

        // Add to history (newest first) - include empty transcripts too
        if (testId) {
          setTranscriptionHistory(prev => [{
            id: testId,
            transcript: currentTranscript,
            audioUrl,
            timestamp: new Date(),
            source: testSource,
          }, ...prev]);
        }
      };
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    } else if (testId) {
      // No recording, but still save transcript (even if empty)
      setTranscriptionHistory(prev => [{
        id: testId,
        transcript: currentTranscript,
        audioUrl: null,
        timestamp: new Date(),
        source: testSource,
      }, ...prev]);
    }

    // Release streams
    StreamManager.releaseStreamsForAgent(TEST_AGENT_ID);
    setIsTestRunning(false);
    currentTestIdRef.current = null;
  };

  // Handle audio playback for a specific record
  const handlePlayRecording = (record: TranscriptionRecord) => {
    if (!record.audioUrl) return;

    // If clicking the same record that's playing, stop it
    if (playingRecordId === record.id && audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
      setPlayingRecordId(null);
      return;
    }

    // Stop any currently playing audio
    if (audioElementRef.current) {
      audioElementRef.current.pause();
    }

    // Play the new recording
    audioElementRef.current = new Audio(record.audioUrl);
    audioElementRef.current.onended = () => setPlayingRecordId(null);
    audioElementRef.current.play();
    setPlayingRecordId(record.id);
  };

  // Cleanup audio URLs on unmount
  useEffect(() => {
    return () => {
      // Cleanup all audio URLs in history
      transcriptionHistory.forEach(record => {
        if (record.audioUrl) {
          URL.revokeObjectURL(record.audioUrl);
        }
      });
      if (audioElementRef.current) {
        audioElementRef.current.pause();
        audioElementRef.current = null;
      }
    };
  }, []);

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold text-gray-800">Application Settings</h1>

      {/* --- Change Detection Settings Card --- */}
      <SettingsCard title="Change Detection Settings">
        <ChangeDetectionSettings compact={false} />
      </SettingsCard>

      {/* --- Existing Screen OCR Settings Card --- */}
      <SettingsCard title="Screen OCR Settings">
        <div className="space-y-4">
          <div>
            <label htmlFor="ocr-lang" className="block text-sm font-medium text-gray-700">Recognition Language</label>
            <select id="ocr-lang" value={ocrLang} onChange={handleOcrLangChange} className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md">
            {AVAILABLE_OCR_LANGUAGES.map(lang => <option key={lang.code} value={lang.code}>{lang.label}</option>)}
            </select>
          </div>
          <div>
            <label htmlFor="ocr-confidence" className="block text-sm font-medium text-gray-700">Minimum Confidence ({ocrConfidence}%)</label>
            <input type="range" id="ocr-confidence" min="0" max="100" value={ocrConfidence} onChange={handleOcrConfidenceChange} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
          </div>
        </div>
      </SettingsCard>

      {/* --- LLM Inference Parameters Card --- */}
      <div className="bg-white shadow-md rounded-lg mb-6">
        <div className="p-4 border-b flex justify-between items-center">
          <h3 className="text-lg font-semibold flex items-center">
            <Sliders className="h-5 w-5 mr-2 text-gray-500" />
            LLM Inference Parameters
          </h3>
          <button
            onClick={handleResetInferenceParams}
            className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
          >
            <RotateCcw className="h-4 w-4" />
            Reset to Defaults
          </button>
        </div>
        <div className="p-6">
          <InferenceParamsEditor
            params={globalInferenceParams}
            onChange={handleGlobalInferenceParamsChange}
          />
        </div>
      </div>

      {/* --- NEW Whisper Model Management Card --- */}
      <SettingsCard title="Whisper Speech Recognition">
        <div className="space-y-6">
          {/* Transcription Mode Toggle */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Transcription Mode
            </label>
            <div className="flex space-x-3">
              <button
                onClick={() => handleTranscriptionModeChange('cloud')}
                className={`flex-1 p-3 rounded-lg border-2 transition-all ${
                  transcriptionMode === 'cloud'
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                }`}
              >
                <Cloud className="h-4 w-4 mx-auto mb-1" />
                <div className="font-medium text-sm">Cloud</div>
                <div className="text-xs mt-1 opacity-75">Real-time, low overhead</div>
              </button>
              <button
                onClick={() => handleTranscriptionModeChange('self-hosted')}
                className={`flex-1 p-3 rounded-lg border-2 transition-all ${
                  transcriptionMode === 'self-hosted'
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                }`}
              >
                <Server className="h-4 w-4 mx-auto mb-1" />
                <div className="font-medium text-sm">Self-Hosted</div>
                <div className="text-xs mt-1 opacity-75">Your own Whisper server</div>
              </button>
              <button
                onClick={() => handleTranscriptionModeChange('local')}
                className={`flex-1 p-3 rounded-lg border-2 transition-all ${
                  transcriptionMode === 'local'
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                }`}
              >
                <Cpu className="h-4 w-4 mx-auto mb-1" />
                <div className="font-medium text-sm">Browser</div>
                <div className="text-xs mt-1 opacity-75">Offline, uses CPU</div>
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {transcriptionMode === 'cloud'
                ? 'Audio is streamed to Observer servers for real-time transcription.'
                : transcriptionMode === 'self-hosted'
                ? 'Audio is sent to your own Whisper-compatible server for processing.'
                : 'Audio is processed locally in your browser using transformers.js Very CPU intensive!'}
            </p>
          </div>

          {/* Self-Hosted URL input */}
          {transcriptionMode === 'self-hosted' && (
            <div className="bg-gray-50 p-4 rounded-lg border">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Whisper Server URL
              </label>
              <input
                type="url"
                placeholder="http://localhost:8000"
                value={selfHostedUrl}
                onChange={handleSelfHostedUrlChange}
                className="block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
              <p className="text-xs text-gray-500 mt-2">
                OpenAI-compatible endpoint (faster-whisper, whisper.cpp, speaches, etc.)
              </p>
            </div>
          )}

          {/* Local Mode: Model Configuration */}
          {transcriptionMode === 'local' && (
          <>
          <div>
            <label htmlFor="model-id" className="block text-sm font-medium text-gray-700 mb-2">
              Model ID
            </label>
            <input
              type="text"
              id="model-id"
              value={whisperSettings.modelId}
              onChange={handleModelIdChange}
              placeholder="Enter any HuggingFace model ID"
              list="model-suggestions"
              disabled={modelState?.status === 'loading' || modelState?.status === 'loaded'}
              className="block w-full px-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md disabled:bg-gray-100"
            />
            <datalist id="model-suggestions">
              {SUGGESTED_MODELS.map(model => (
                <option key={model} value={model} />
              ))}
            </datalist>
            <p className="text-xs text-gray-500 mt-1">
              Examples: Xenova/whisper-small.en (English only), Xenova/whisper-small (multilingual)
            </p>
          </div>

          {/* Responsive Options - Only show for multilingual models */}
          {!whisperSettings.modelId.endsWith('.en') && (
            <div className="bg-gray-50 p-4 rounded-lg border">
              <h4 className="text-sm font-medium text-gray-900 mb-3">Multilingual Options</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="task" className="block text-xs font-medium text-gray-700 mb-1">
                    Task
                  </label>
                  <select
                    id="task"
                    value={whisperSettings.task || ''}
                    onChange={handleTaskChange}
                    className="block w-full px-3 py-2 text-sm border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Default (transcribe)</option>
                    <option value="transcribe">Transcribe</option>
                    <option value="translate">Translate to English</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="language" className="block text-xs font-medium text-gray-700 mb-1">
                    Language
                  </label>
                  <select
                    id="language"
                    value={whisperSettings.language || ''}
                    onChange={handleLanguageChange}
                    className="block w-full px-3 py-2 text-sm border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Auto-detect</option>
                    {Object.entries(LANGUAGE_NAMES).map(([code, name]) => (
                      <option key={code} value={code}>{name}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* Quantized Option */}
          <div className="flex items-center">
            <input
              type="checkbox"
              id="quantized"
              checked={whisperSettings.quantized}
              onChange={handleQuantizedChange}
              className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
            <label htmlFor="quantized" className="ml-2 text-sm font-medium text-gray-700">
              Quantized (smaller file sizes, faster loading)
            </label>
          </div>
          </>
          )}

          {/* Chunk Duration - Shared by both modes */}
          {transcriptionMode !== 'cloud' && (
          <div>
            <label htmlFor="chunk-duration" className="block text-sm font-medium text-gray-700 mb-2">
              Chunk Duration ({Math.round(whisperSettings.chunkDurationMs / 1000)}s)
            </label>
            <input
              type="range"
              id="chunk-duration"
              min="1000"
              max="60000"
              step="1000"
              value={whisperSettings.chunkDurationMs}
              onChange={handleChunkDurationChange}
              disabled={isTestRunning}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>1s</span>
              <span>30s</span>
              <span>60s</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Each agent accumulates transcripts during its loop window and clears them after processing.
            </p>
          </div>
          )}
          
          {/* Local Mode: Model Management Buttons */}
          {transcriptionMode === 'local' && (
            <div className="flex items-center space-x-4">
              <button
                onClick={handleLoadModel}
                disabled={modelState?.status === 'loading' || modelState?.status === 'loaded'}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center transition-all"
              >
                {modelState?.status === 'loading' ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : modelState?.status === 'loaded' ? (
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                ) : (
                  <Database className="mr-2 h-4 w-4" />
                )}
                {modelState?.status === 'loading' ? 'Loading...' : modelState?.status === 'loaded' ? 'Model Loaded' : 'Load Model'}
              </button>

              {modelState?.status === 'loaded' && (
                <button
                  onClick={handleUnloadModel}
                  className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 flex items-center transition-all"
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Unload Model
                </button>
              )}
            </div>
          )}

          {/* Audio Source Toggle + Test Button */}
          <div className="bg-gray-50 p-4 rounded-lg border">
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Test Audio Source
            </label>
            <div className="flex items-center space-x-3 mb-4">
              <button
                onClick={() => setAudioTestSource('microphone')}
                disabled={isTestRunning}
                className={`flex items-center px-3 py-2 rounded-lg border-2 transition-all ${
                  audioTestSource === 'microphone'
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <Mic className="h-4 w-4 mr-2" />
                <span className="text-sm font-medium">Microphone</span>
              </button>
              <button
                onClick={() => setAudioTestSource('screenAudio')}
                disabled={isTestRunning}
                className={`flex items-center px-3 py-2 rounded-lg border-2 transition-all ${
                  audioTestSource === 'screenAudio'
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <Monitor className="h-4 w-4 mr-2" />
                <span className="text-sm font-medium">Screen Audio</span>
              </button>
              <button
                onClick={() => setAudioTestSource('allAudio')}
                disabled={isTestRunning}
                className={`flex items-center px-3 py-2 rounded-lg border-2 transition-all ${
                  audioTestSource === 'allAudio'
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <Volume2 className="h-4 w-4 mr-2" />
                <span className="text-sm font-medium">All Audio</span>
              </button>
            </div>

            <div className="flex items-center space-x-4">
              <button
                onClick={isTestRunning ? handleStopTest : handleStartTest}
              disabled={
                (transcriptionMode === 'local' && modelState?.status !== 'loaded') ||
                (transcriptionMode === 'self-hosted' && !selfHostedUrl.trim())
              }
              className={`px-4 py-2 rounded-md text-white flex items-center transition-all disabled:bg-gray-400 disabled:cursor-not-allowed ${
                isTestRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              <TestTube2 className="mr-2 h-4 w-4" />
              {isTestRunning ? 'Stop Test' : 'Start Test'}
            </button>
            {transcriptionMode === 'cloud' && (
              <span className="text-sm text-green-600 flex items-center">
                <CheckCircle2 className="h-4 w-4 mr-1" />
                Cloud Ready
              </span>
            )}
            {transcriptionMode === 'self-hosted' && selfHostedUrl.trim() && (
              <span className="text-sm text-green-600 flex items-center">
                <CheckCircle2 className="h-4 w-4 mr-1" />
                Server Configured
              </span>
            )}
            {transcriptionMode === 'self-hosted' && !selfHostedUrl.trim() && (
              <span className="text-sm text-amber-600 flex items-center">
                Enter server URL to enable
              </span>
            )}
            </div>
          </div>

          {/* Local Mode: Model Loading Progress */}
          {transcriptionMode === 'local' && modelState?.status === 'loading' && modelState.progress.length > 0 && (
            <div className="space-y-3 pt-2">
              <h4 className="text-md font-semibold text-gray-700">
                Loading Model: {modelState.config?.modelId}
              </h4>
              {modelState.progress.map((item) => (
                <div key={item.file}>
                  <div className="flex justify-between items-center text-sm mb-1">
                    <span className="text-gray-600 flex items-center">
                      {item.status === 'done' 
                        ? <CheckCircle2 className="h-4 w-4 text-green-500 mr-2"/>
                        : <FileDown className="h-4 w-4 text-gray-400 mr-2"/>
                      }
                      {item.file}
                    </span>
                    <span className="font-medium text-gray-500">
                      {item.status === 'done' ? 'Done' : `${Math.round(item.progress)}%`}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${
                        item.status === 'done' ? 'bg-green-500' : 'bg-blue-600'
                      }`}
                      style={{ width: `${item.progress}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Local Mode: Error Display */}
          {transcriptionMode === 'local' && modelState?.error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-3">
              <p className="text-sm text-red-800">
                <strong>Error:</strong> {modelState.error}
              </p>
            </div>
          )}

          {/* Transcription Results */}
          <div>
            <h4 className="text-md font-semibold text-gray-700 mb-2">Live Transcription</h4>
            <div className="border rounded-lg bg-gray-50 max-h-96 overflow-y-auto">
              {/* Currently running test */}
              {isTestRunning && (
                <div className="p-3 border-b bg-blue-50">
                  <div className="flex items-start gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
                        <span className="text-xs font-medium text-blue-600">
                          Recording ({audioTestSource === 'microphone' ? 'Mic' : audioTestSource === 'screenAudio' ? 'Screen' : 'All'})
                        </span>
                      </div>
                      <p className="text-gray-800 text-sm leading-relaxed whitespace-pre-wrap">
                        {(committedText || interimText) ? (
                          <>
                            <span>{committedText}</span>
                            {committedText && interimText && ' '}
                            <span className="text-gray-500 italic">{interimText}</span>
                          </>
                        ) : (
                          <span className="text-gray-400 italic">
                            {audioTestSource === 'microphone' ? 'Speak into your microphone...' : 'Play some audio on your device...'}
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Transcription history */}
              {transcriptionHistory.map((record) => (
                <div key={record.id} className="p-3 border-b last:border-b-0 hover:bg-gray-100 transition-colors">
                  <div className="flex items-start gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs text-gray-500">
                          {record.source === 'microphone' ? (
                            <Mic className="h-3 w-3 inline mr-1" />
                          ) : record.source === 'screenAudio' ? (
                            <Monitor className="h-3 w-3 inline mr-1" />
                          ) : (
                            <Volume2 className="h-3 w-3 inline mr-1" />
                          )}
                          {record.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-gray-800 text-sm leading-relaxed whitespace-pre-wrap">
                        {record.transcript}
                      </p>
                    </div>
                    {record.audioUrl && (
                      <button
                        onClick={() => handlePlayRecording(record)}
                        className={`flex-shrink-0 p-2 rounded-full transition-all ${
                          playingRecordId === record.id
                            ? 'bg-red-100 text-red-600 hover:bg-red-200'
                            : 'bg-blue-100 text-blue-600 hover:bg-blue-200'
                        }`}
                        title={playingRecordId === record.id ? 'Stop' : 'Play recording'}
                      >
                        {playingRecordId === record.id ? (
                          <Square className="h-4 w-4" />
                        ) : (
                          <Play className="h-4 w-4" />
                        )}
                      </button>
                    )}
                  </div>
                </div>
              ))}

              {/* Empty state */}
              {!isTestRunning && transcriptionHistory.length === 0 && (
                <p className="text-sm text-gray-500 text-center py-8">
                  Start a test to see transcription results here.
                </p>
              )}
            </div>
          </div>
        </div>
      </SettingsCard>
    </div>
  );
};

export default SettingsTab;
