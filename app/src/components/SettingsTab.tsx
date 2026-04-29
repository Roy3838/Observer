import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Settings, TestTube2, Loader2, FileDown, CheckCircle2, Database, Trash2, Cloud, Server, Cpu, Mic, Monitor, Play, Square, Volume2, Keyboard, Check, AlertTriangle, Eye, EyeOff, Layers, Move, Maximize2, Zap, ChevronDown, ChevronRight } from 'lucide-react';
import { SensorSettings } from '../utils/settings';
import { StreamManager } from '../utils/streamManager';
import { isDesktop } from '../utils/platform';

// Whisper imports
import { WhisperModelManager } from '../utils/whisper/WhisperModelManager';
import { TranscriptionRouter } from '../utils/whisper/TranscriptionRouter';
import { WhisperModelState, TranscriptionMode } from '../utils/whisper/types';
import { useSubscriberText } from '../hooks/useTranscriptionState';
import { SUGGESTED_MODELS, LANGUAGE_NAMES } from '../config/whisper-models';

import { AVAILABLE_OCR_LANGUAGES } from '../config/ocr-languages';

// Change Detection component
import ChangeDetectionSettings from './ChangeDetectionSettings';

// Helper function to format bytes
const formatBytes = (bytes: number, decimals = 1) => {
  if (!+bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
};

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

  // --- OCR State Management ---
  const [ocrLang, setOcrLang] = useState(SensorSettings.getOcrLanguage());

  // --- OCR Handler Functions ---
  const handleOcrLangChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newLang = e.target.value;
    setOcrLang(newLang);
    SensorSettings.setOcrLanguage(newLang);
  };


  // --- NEW WHISPER STATE ---
  const [whisperSettings, setWhisperSettings] = useState(SensorSettings.getWhisperSettings());
  const [modelState, setModelState] = useState<WhisperModelState | null>(null);
  const [isTestRunning, setIsTestRunning] = useState(false);
  const [transcriptionMode, setTranscriptionModeState] = useState<TranscriptionMode>(
    TranscriptionRouter.getInstance().getMode()
  );
  const [selfHostedUrl, setSelfHostedUrl] = useState(SensorSettings.getSelfHostedWhisperUrl());

  // --- KEYBOARD SHORTCUTS STATE (Desktop Only) ---
  const [overlayShortcuts, setOverlayShortcuts] = useState({
    toggle: '',
    move_up: '',
    move_down: '',
    move_left: '',
    move_right: '',
    resize_up: '',
    resize_down: '',
    resize_left: '',
    resize_right: ''
  });
  const [availableAgents, setAvailableAgents] = useState<Array<{id: string, name: string}>>([]);
  const [agentShortcuts, setAgentShortcuts] = useState<Record<string, string>>({});
  const [activeShortcuts, setActiveShortcuts] = useState<string[]>([]);
  const [shortcutFeedback, setShortcutFeedback] = useState<{ message: string, type: 'success' | 'error' } | null>(null);
  const [capturingFor, setCapturingFor] = useState<string | null>(null);
  const [newAgentId, setNewAgentId] = useState('');
  const [showMoveShortcuts, setShowMoveShortcuts] = useState(false);
  const [showResizeShortcuts, setShowResizeShortcuts] = useState(false);

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

  // --- KEYBOARD SHORTCUTS FUNCTIONS (Desktop Only) ---
  const buildKeyCombo = (event: KeyboardEvent): string => {
    const modifiers = [];
    const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;

    if (event.metaKey) modifiers.push(isMac ? 'Cmd' : 'Super');
    if (event.ctrlKey) modifiers.push('Ctrl');
    if (event.altKey) modifiers.push('Alt');
    if (event.shiftKey) modifiers.push('Shift');

    let key = event.key;
    if (key === ' ') key = 'Space';
    if (key.length === 1) key = key.toUpperCase();

    if (['Meta', 'Control', 'Alt', 'Shift'].includes(event.key)) {
      return '';
    }

    return modifiers.length > 0 ? `${modifiers.join('+')}+${key}` : key;
  };

  const handleAddAgent = useCallback(() => {
    if (!newAgentId.trim()) return;

    const agentId = newAgentId.trim();
    if (availableAgents.find(a => a.id === agentId)) {
      setShortcutFeedback({ message: `Agent "${agentId}" already exists`, type: 'error' });
      return;
    }

    setAvailableAgents(prev => [...prev, { id: agentId, name: agentId }]);
    setNewAgentId('');
    setShortcutFeedback({ message: `Agent "${agentId}" added`, type: 'success' });
    setTimeout(() => setShortcutFeedback(null), 2000);
  }, [newAgentId, availableAgents]);

  const loadAllShortcuts = useCallback(async () => {
    if (!isDesktop()) return;

    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const [unifiedConfig, registeredShortcuts] = await Promise.all([
        invoke<any>('get_shortcut_config'),
        invoke<string[]>('get_registered_shortcuts')
      ]);

      setOverlayShortcuts({
        toggle: unifiedConfig.overlay_toggle || '',
        move_up: unifiedConfig.overlay_move_up || '',
        move_down: unifiedConfig.overlay_move_down || '',
        move_left: unifiedConfig.overlay_move_left || '',
        move_right: unifiedConfig.overlay_move_right || '',
        resize_up: unifiedConfig.overlay_resize_up || '',
        resize_down: unifiedConfig.overlay_resize_down || '',
        resize_left: unifiedConfig.overlay_resize_left || '',
        resize_right: unifiedConfig.overlay_resize_right || ''
      });

      setAgentShortcuts(unifiedConfig.agent_shortcuts || {});
      const agentIds = Object.keys(unifiedConfig.agent_shortcuts || {});
      const agents = agentIds.map(id => ({ id, name: id }));
      setAvailableAgents(agents);
      setActiveShortcuts(registeredShortcuts);
    } catch (error) {
      console.error('Failed to load shortcuts:', error);
    }
  }, []);

  const validateAllShortcuts = (): string | null => {
    const usedShortcuts = new Set<string>();
    const conflicts: string[] = [];

    for (const [_, shortcut] of Object.entries(overlayShortcuts)) {
      if (shortcut && shortcut.trim()) {
        if (usedShortcuts.has(shortcut)) {
          conflicts.push(shortcut);
        } else {
          usedShortcuts.add(shortcut);
        }
      }
    }

    for (const [_, shortcut] of Object.entries(agentShortcuts)) {
      if (shortcut && shortcut.trim()) {
        if (usedShortcuts.has(shortcut)) {
          conflicts.push(shortcut);
        } else {
          usedShortcuts.add(shortcut);
        }
      }
    }

    if (conflicts.length > 0) {
      return `Duplicate shortcuts detected: ${conflicts.join(', ')}`;
    }

    return null;
  };

  const handleSaveAllShortcuts = useCallback(async () => {
    const validationError = validateAllShortcuts();
    if (validationError) {
      setShortcutFeedback({ message: validationError, type: 'error' });
      return;
    }

    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const unifiedConfigToSave = {
        overlay_toggle: overlayShortcuts.toggle.trim() || null,
        overlay_move_up: overlayShortcuts.move_up.trim() || null,
        overlay_move_down: overlayShortcuts.move_down.trim() || null,
        overlay_move_left: overlayShortcuts.move_left.trim() || null,
        overlay_move_right: overlayShortcuts.move_right.trim() || null,
        overlay_resize_up: overlayShortcuts.resize_up.trim() || null,
        overlay_resize_down: overlayShortcuts.resize_down.trim() || null,
        overlay_resize_left: overlayShortcuts.resize_left.trim() || null,
        overlay_resize_right: overlayShortcuts.resize_right.trim() || null,
        agent_shortcuts: agentShortcuts
      };

      await invoke('set_shortcut_config', { config: unifiedConfigToSave });

      setShortcutFeedback({
        message: 'All shortcuts saved! Restart the app to activate overlay shortcuts.',
        type: 'success'
      });

      setTimeout(() => setShortcutFeedback(null), 4000);
    } catch (error) {
      console.error('Failed to save shortcuts:', error);
      setShortcutFeedback({ message: `Error saving shortcuts: ${error}`, type: 'error' });
    }
  }, [overlayShortcuts, agentShortcuts]);

  // Load shortcuts on mount (desktop only)
  useEffect(() => {
    loadAllShortcuts();
  }, [loadAllShortcuts]);

  // Key capture effect (desktop only)
  useEffect(() => {
    if (!capturingFor || !isDesktop()) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      event.preventDefault();
      event.stopPropagation();

      if (event.key === 'Escape') {
        setCapturingFor(null);
        return;
      }

      const combo = buildKeyCombo(event);
      if (combo) {
        if (capturingFor.startsWith('overlay_')) {
          const overlayKey = capturingFor.replace('overlay_', '');
          setOverlayShortcuts(prev => ({
            ...prev,
            [overlayKey]: combo
          }));
        } else {
          setAgentShortcuts(prev => ({
            ...prev,
            [capturingFor]: combo
          }));
        }
        setCapturingFor(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown, true);
    return () => window.removeEventListener('keydown', handleKeyDown, true);
  }, [capturingFor]);

  // --- OVERLAY CONTROL HANDLERS (Desktop Only) ---
  const handleShowOverlay = useCallback(async () => {
    if (!isDesktop()) return;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('show_overlay');
    } catch (error) {
      console.error('Failed to show overlay:', error);
    }
  }, []);

  const handleHideOverlay = useCallback(async () => {
    if (!isDesktop()) return;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('hide_overlay');
    } catch (error) {
      console.error('Failed to hide overlay:', error);
    }
  }, []);

  const handleClearOverlay = useCallback(async () => {
    if (!isDesktop()) return;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('clear_overlay_messages');
    } catch (error) {
      console.error('Failed to clear overlay:', error);
    }
  }, []);

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

      {/* --- Desktop Only Settings --- */}
      {isDesktop() && (
        <>
          {/* --- Overlay Controls Card --- */}
          <div className="bg-white shadow-md rounded-lg mb-6">
            <div className="p-4 border-b">
              <h3 className="text-lg font-semibold flex items-center">
                <Layers className="h-5 w-5 mr-2 text-purple-500" />
                Overlay Controls
              </h3>
            </div>
            <div className="p-6">
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={handleShowOverlay}
                  className="flex items-center px-4 py-2.5 bg-green-50 text-green-700 border border-green-200 rounded-lg hover:bg-green-100 font-medium text-sm transition-all"
                >
                  <Eye className="h-4 w-4 mr-2" />
                  Show Overlay
                </button>
                <button
                  onClick={handleHideOverlay}
                  className="flex items-center px-4 py-2.5 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-200 font-medium text-sm transition-all"
                >
                  <EyeOff className="h-4 w-4 mr-2" />
                  Hide Overlay
                </button>
                <button
                  onClick={handleClearOverlay}
                  className="flex items-center px-4 py-2.5 bg-red-50 text-red-700 border border-red-200 rounded-lg hover:bg-red-100 font-medium text-sm transition-all"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear Messages
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-3">
                Control the overlay window visibility or clear all displayed messages.
              </p>
            </div>
          </div>

          {/* --- Keyboard Shortcuts Card --- */}
          <div className="bg-white shadow-md rounded-lg mb-6">
            <div className="p-4 border-b flex justify-between items-center">
              <h3 className="text-lg font-semibold flex items-center">
                <Keyboard className="h-5 w-5 mr-2 text-indigo-500" />
                Keyboard Shortcuts
              </h3>
              {activeShortcuts.length > 0 && (
                <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full font-medium">
                  {activeShortcuts.length} active
                </span>
              )}
            </div>
            <div className="p-6 space-y-6">

              {/* Feedback Messages - Show at top */}
              {shortcutFeedback && (
                <div className={`flex items-center text-sm p-3 rounded-lg ${
                  shortcutFeedback.type === 'success'
                    ? 'bg-green-50 text-green-700 border border-green-200'
                    : 'bg-red-50 text-red-700 border border-red-200'
                }`}>
                  {shortcutFeedback.type === 'success' ? <Check className="h-4 w-4 mr-2" /> : <AlertTriangle className="h-4 w-4 mr-2" />}
                  {shortcutFeedback.message}
                </div>
              )}

              {/* Toggle Overlay Shortcut - Primary */}
              <div className="bg-gradient-to-r from-purple-50 to-indigo-50 p-4 rounded-lg border border-purple-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Eye className="h-5 w-5 mr-3 text-purple-600" />
                    <div>
                      <span className="text-sm font-semibold text-gray-800">Toggle Overlay</span>
                      <p className="text-xs text-gray-500">Show or hide the overlay window</p>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <button
                      onClick={() => setCapturingFor('overlay_toggle')}
                      disabled={capturingFor === 'overlay_toggle'}
                      className={`px-4 py-2 text-sm rounded-lg font-mono transition-all min-w-[140px] text-center ${
                        capturingFor === 'overlay_toggle'
                          ? 'bg-orange-100 text-orange-700 border-2 border-orange-400 animate-pulse'
                          : overlayShortcuts.toggle
                          ? 'bg-white text-purple-700 border-2 border-purple-300 hover:border-purple-400 shadow-sm'
                          : 'bg-white text-gray-500 border-2 border-dashed border-gray-300 hover:border-purple-300'
                      }`}
                    >
                      {capturingFor === 'overlay_toggle' ? 'Press keys...' : overlayShortcuts.toggle || 'Click to set'}
                    </button>
                    {overlayShortcuts.toggle && capturingFor !== 'overlay_toggle' && (
                      <button
                        onClick={() => setOverlayShortcuts(prev => ({ ...prev, toggle: '' }))}
                        className="ml-2 p-1 text-gray-400 hover:text-red-500 transition-colors"
                        title="Clear shortcut"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Move Shortcuts - Collapsible */}
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <button
                  onClick={() => setShowMoveShortcuts(!showMoveShortcuts)}
                  className="w-full flex items-center justify-between p-3 bg-gray-50 hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center">
                    <Move className="h-4 w-4 mr-2 text-blue-500" />
                    <span className="text-sm font-medium text-gray-700">Move Overlay</span>
                    <span className="ml-2 text-xs text-gray-400">
                      ({Object.entries(overlayShortcuts).filter(([k, v]) => k.startsWith('move_') && v).length}/4 set)
                    </span>
                  </div>
                  {showMoveShortcuts ? <ChevronDown className="h-4 w-4 text-gray-400" /> : <ChevronRight className="h-4 w-4 text-gray-400" />}
                </button>
                {showMoveShortcuts && (
                  <div className="p-4 grid grid-cols-2 gap-3 bg-white">
                    {(['move_up', 'move_down', 'move_left', 'move_right'] as const).map((key) => (
                      <div key={key} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-600 capitalize">{key.replace('move_', '').replace('_', ' ')}</span>
                        <div className="flex items-center">
                          <button
                            onClick={() => setCapturingFor(`overlay_${key}`)}
                            disabled={capturingFor === `overlay_${key}`}
                            className={`px-3 py-1.5 text-xs rounded font-mono transition-all min-w-[80px] text-center ${
                              capturingFor === `overlay_${key}`
                                ? 'bg-orange-100 text-orange-700 border border-orange-300 animate-pulse'
                                : overlayShortcuts[key]
                                ? 'bg-blue-50 text-blue-700 border border-blue-200'
                                : 'bg-white text-gray-400 border border-dashed border-gray-300'
                            }`}
                          >
                            {capturingFor === `overlay_${key}` ? '...' : overlayShortcuts[key] || 'Set'}
                          </button>
                          {overlayShortcuts[key] && (
                            <button
                              onClick={() => setOverlayShortcuts(prev => ({ ...prev, [key]: '' }))}
                              className="ml-1 p-1 text-gray-300 hover:text-red-500"
                            >
                              <Trash2 className="h-3 w-3" />
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Resize Shortcuts - Collapsible */}
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <button
                  onClick={() => setShowResizeShortcuts(!showResizeShortcuts)}
                  className="w-full flex items-center justify-between p-3 bg-gray-50 hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center">
                    <Maximize2 className="h-4 w-4 mr-2 text-green-500" />
                    <span className="text-sm font-medium text-gray-700">Resize Overlay</span>
                    <span className="ml-2 text-xs text-gray-400">
                      ({Object.entries(overlayShortcuts).filter(([k, v]) => k.startsWith('resize_') && v).length}/4 set)
                    </span>
                  </div>
                  {showResizeShortcuts ? <ChevronDown className="h-4 w-4 text-gray-400" /> : <ChevronRight className="h-4 w-4 text-gray-400" />}
                </button>
                {showResizeShortcuts && (
                  <div className="p-4 grid grid-cols-2 gap-3 bg-white">
                    {(['resize_up', 'resize_down', 'resize_left', 'resize_right'] as const).map((key) => (
                      <div key={key} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-600 capitalize">{key.replace('resize_', '').replace('_', ' ')}</span>
                        <div className="flex items-center">
                          <button
                            onClick={() => setCapturingFor(`overlay_${key}`)}
                            disabled={capturingFor === `overlay_${key}`}
                            className={`px-3 py-1.5 text-xs rounded font-mono transition-all min-w-[80px] text-center ${
                              capturingFor === `overlay_${key}`
                                ? 'bg-orange-100 text-orange-700 border border-orange-300 animate-pulse'
                                : overlayShortcuts[key]
                                ? 'bg-green-50 text-green-700 border border-green-200'
                                : 'bg-white text-gray-400 border border-dashed border-gray-300'
                            }`}
                          >
                            {capturingFor === `overlay_${key}` ? '...' : overlayShortcuts[key] || 'Set'}
                          </button>
                          {overlayShortcuts[key] && (
                            <button
                              onClick={() => setOverlayShortcuts(prev => ({ ...prev, [key]: '' }))}
                              className="ml-1 p-1 text-gray-300 hover:text-red-500"
                            >
                              <Trash2 className="h-3 w-3" />
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Agent Shortcuts Section */}
              <div className="border-t border-gray-200 pt-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <Zap className="h-5 w-5 mr-2 text-amber-500" />
                    <span className="text-sm font-semibold text-gray-800">Agent Shortcuts</span>
                  </div>
                  <span className="text-xs text-gray-400">{availableAgents.length} configured</span>
                </div>

                {/* Add New Agent */}
                <div className="flex items-center gap-2 mb-4">
                  <input
                    type="text"
                    value={newAgentId}
                    onChange={(e) => setNewAgentId(e.target.value)}
                    placeholder="Enter agent ID..."
                    className="flex-grow px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                    onKeyDown={(e) => e.key === 'Enter' && handleAddAgent()}
                  />
                  <button
                    onClick={handleAddAgent}
                    disabled={!newAgentId.trim()}
                    className="px-4 py-2 bg-amber-500 text-white text-sm rounded-lg hover:bg-amber-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
                  >
                    Add
                  </button>
                </div>

                {/* Agent List */}
                {availableAgents.length > 0 ? (
                  <div className="space-y-2">
                    {availableAgents.map((agent) => {
                      const shortcut = agentShortcuts[agent.id];
                      const isDuplicate = shortcut && Object.entries(agentShortcuts)
                        .filter(([id, s]) => id !== agent.id && s === shortcut).length > 0;

                      return (
                        <div key={agent.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                          <span className="text-sm text-gray-700 font-medium">{agent.name}</span>
                          <div className="flex items-center">
                            <button
                              onClick={() => setCapturingFor(agent.id)}
                              disabled={capturingFor === agent.id}
                              className={`px-3 py-1.5 text-xs rounded-lg font-mono transition-all min-w-[120px] text-center ${
                                capturingFor === agent.id
                                  ? 'bg-orange-100 text-orange-700 border-2 border-orange-400 animate-pulse'
                                  : isDuplicate
                                  ? 'bg-red-50 text-red-700 border border-red-300'
                                  : agentShortcuts[agent.id]
                                  ? 'bg-amber-50 text-amber-700 border border-amber-200'
                                  : 'bg-white text-gray-400 border border-dashed border-gray-300'
                              }`}
                            >
                              {capturingFor === agent.id ? 'Press keys...' : agentShortcuts[agent.id] || 'Click to set'}
                              {isDuplicate && ' ⚠️'}
                            </button>
                            <button
                              onClick={() => {
                                setAvailableAgents(prev => prev.filter(a => a.id !== agent.id));
                                setAgentShortcuts(prev => {
                                  const updated = { ...prev };
                                  delete updated[agent.id];
                                  return updated;
                                });
                              }}
                              className="ml-2 p-1 text-gray-300 hover:text-red-500 transition-colors"
                              title="Remove agent"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-center py-6 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200">
                    <Zap className="h-8 w-8 mx-auto text-gray-300 mb-2" />
                    <p className="text-sm text-gray-500">No agent shortcuts configured</p>
                    <p className="text-xs text-gray-400 mt-1">Add an agent ID above to create a shortcut</p>
                  </div>
                )}
              </div>

              {/* Save Button & Help */}
              <div className="border-t border-gray-200 pt-6 space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-xs text-gray-500">
                    Click any button and press your desired key combination. Press Escape to cancel.
                  </p>
                  <button
                    onClick={handleSaveAllShortcuts}
                    className="px-6 py-2.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 text-sm font-semibold shadow-sm transition-all"
                  >
                    Save Shortcuts
                  </button>
                </div>
                <p className="text-xs text-gray-400">
                  Overlay shortcuts require an app restart to take effect. Agent shortcuts are applied immediately.
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* --- Change Detection Settings Card --- */}
      <SettingsCard title="Change Detection Settings">
        <ChangeDetectionSettings compact={false} />
      </SettingsCard>

      {/* --- Existing Screen OCR Settings Card --- */}
      <SettingsCard title="OCR Settings">
        <div className="space-y-4">
          <div>
            <label htmlFor="ocr-lang" className="block text-sm font-medium text-gray-700">Recognition Language</label>
            <select id="ocr-lang" value={ocrLang} onChange={handleOcrLangChange} className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md">
            {AVAILABLE_OCR_LANGUAGES.map(lang => <option key={lang.code} value={lang.code}>{lang.label}</option>)}
            </select>
          </div>
        </div>
      </SettingsCard>

      {/* --- Whisper Model Management Card --- */}
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
            <div className="flex flex-wrap gap-2 mb-4">
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
                    <span className="text-gray-600 flex items-center truncate max-w-[60%]">
                      {item.status === 'done'
                        ? <CheckCircle2 className="h-4 w-4 text-green-500 mr-2 flex-shrink-0"/>
                        : <FileDown className="h-4 w-4 text-gray-400 mr-2 flex-shrink-0"/>
                      }
                      <span className="truncate">{item.file}</span>
                    </span>
                    <span className="font-medium text-gray-500 flex-shrink-0">
                      {item.status === 'done'
                        ? 'Done'
                        : item.total > 0
                          ? `${formatBytes(item.loaded)} / ${formatBytes(item.total)}`
                          : `${Math.round(item.progress)}%`
                      }
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
