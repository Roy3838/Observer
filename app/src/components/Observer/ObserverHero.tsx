// src/components/Observer/ObserverHero.tsx
//
// The Observer tab's empty-conversation splash — minimal, centered, front-and-center
// input, matching Claude Cowork/ChatGPT's own "new chat" screen. Deliberately small and
// self-contained: it does not reuse or refactor MCP.tsx's internal input, so GetStarted.tsx
// (which embeds MCP.tsx as-is for the My Agents tab) is completely unaffected by this file.
//
// RecipeInline's wheels are pinned at a fixed spot behind the textarea (absolutely
// positioned, not in its flow) so they never shift as the textarea grows with wrapped text.
// Until the user types their own words, the textarea just mirrors whatever sentence the
// wheels are currently spelling out — spinning a wheel visibly grows/edits the input live.

import React, { useEffect, useRef, useState } from 'react';
import { Send, Loader2, Info, Mic, Cpu } from 'lucide-react';
import { useMCPContext } from '../../mcp/MCPContext';
import { useAuth } from '@contexts/AuthContext';
import { SensorSettings } from '@utils/settings';
import { Analytics } from '@utils/analytics';
import { tutorialFlow } from '@utils/tutorialFlow';
import { StreamManager } from '@utils/streamManager';
import { useSubscriberText } from '@hooks/useTranscriptionState';
import { Logger } from '@utils/logging';
import { ModelManager, type Model } from '@utils/ModelManager';
import TutorialBubble from './TutorialBubble';
import TutorialSpotlight from './TutorialSpotlight';
import RecipeInline, { type TutorialStep } from '../AICreator/RecipeInline';

// Synthetic owner id for voice dictation on the hero splash screen — mirrors MCP.tsx's
// MCP_MIC_ID so it routes through the same StreamManager / TranscriptionRouter path.
const HERO_MIC_ID = 'observer-hero-mic';
const DEFAULT_CLOUD_MODEL = 'gemini-2.5-flash-lite-free';

const ObserverHero: React.FC = () => {
  const { send, isRunning, modelName, setModelName } = useMCPContext();
  const { isAuthenticated } = useAuth();
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // --- Voice dictation --------------------------------------------------------
  const { fullText: micTranscript } = useSubscriberText(HERO_MIC_ID, 'microphone');
  const [isRecording, setIsRecording] = useState(false);
  const [micStarting, setMicStarting] = useState(false);
  const micBaseRef = useRef('');
  const micGenRef = useRef(0);

  useEffect(() => {
    if (!isRecording) return;
    const base = micBaseRef.current;
    setValue(base && micTranscript ? `${base} ${micTranscript}` : base + micTranscript);
    setUserEdited(true);
  }, [micTranscript, isRecording]);

  useEffect(() => () => {
    StreamManager.releaseStreamsForAgent(HERO_MIC_ID);
    StreamManager.destroySubscribersForAgent(HERO_MIC_ID);
  }, []);

  const stopMic = () => {
    micGenRef.current++;
    setIsRecording(false);
    setMicStarting(false);
    StreamManager.releaseStreamsForAgent(HERO_MIC_ID);
    StreamManager.destroySubscribersForAgent(HERO_MIC_ID);
  };

  const startMic = async () => {
    micBaseRef.current = value.trim();
    setMicStarting(true);
    const gen = ++micGenRef.current;
    try {
      await StreamManager.requestStreamsForAgent(HERO_MIC_ID, ['microphone']);
      if (micGenRef.current !== gen) {
        StreamManager.releaseStreamsForAgent(HERO_MIC_ID);
        StreamManager.destroySubscribersForAgent(HERO_MIC_ID);
        return;
      }
      setIsRecording(true);
    } catch (e) {
      Logger.error('ObserverHero', `Voice dictation failed to start: ${e}`);
      stopMic();
    } finally {
      if (micGenRef.current === gen) setMicStarting(false);
    }
  };
  // --- Model selection (same custom-server list as MCP.tsx's input) --------------
  const getCustomServerModels = (): Model[] =>
    ModelManager.getInstance().listModels().models.filter(m =>
      m.server !== ModelManager.BROWSER_LOCAL &&
      m.server !== ModelManager.LLAMA_CPP_LOCAL &&
      m.server !== ModelManager.SKIP_MODEL &&
      !m.server.includes('api.observer-ai.com')
    );
  const [customModels, setCustomModels] = useState<Model[]>(getCustomServerModels);

  useEffect(() => ModelManager.getInstance().onModelsChange(() => {
    setCustomModels(getCustomServerModels());
  }), []);

  // Signed-out users can't use the cloud default, so pick their first own-server model.
  useEffect(() => {
    if (!isAuthenticated && customModels.length > 0 && !customModels.some(m => m.name === modelName)) {
      setModelName(customModels[0].name);
    }
  }, [isAuthenticated, customModels, modelName, setModelName]);

  // Once the user types their own text, wheel spins stop overwriting it. Clearing the box
  // (back to empty) hands control back to the wheels.
  const [userEdited, setUserEdited] = useState(false);

  // First-run guided demo, layered on the wheels: ends in building a REAL agent that watches
  // a synthetic progress bar (SensorSettings.mcpTutorialMode / tutorialStreamCapture).
  const { user } = useAuth();
  const tutorialKey = user && 'sub' in user && user.sub ? `observer_tutorial_seen_${user.sub}` : null;
  const [tutorialStep, setTutorialStep] = useState<TutorialStep | null>(null);

  useEffect(() => {
    if (tutorialKey && !localStorage.getItem(tutorialKey)) setTutorialStep('hello');
  }, [tutorialKey]);

  const endTutorial = () => {
    setTutorialStep(null);
    if (tutorialKey) localStorage.setItem(tutorialKey, 'true');
  };

  const skipTutorial = () => { Analytics.tutorialSkipped(); SensorSettings.setMcpTutorialMode(false); endTutorial(); };
  const replayTutorial = () => {
    Analytics.tutorialStarted();
    setValue('');
    setUserEdited(false);
    setTutorialStep('hello');
  };

  // Clicking anything in the sidebar means the user is leaving — drop the tutorial rather than
  // leave the app greyed out behind them (the hero stays mounted, just hidden, on tab switch).
  useEffect(() => {
    if (!tutorialStep) return;
    const onPointerDown = (e: PointerEvent) => {
      if ((e.target as Element | null)?.closest?.('[data-app-sidebar]')) skipTutorial();
    };
    document.addEventListener('pointerdown', onPointerDown, true);
    return () => document.removeEventListener('pointerdown', onPointerDown, true);
    // skipTutorial is recreated each render but only touches stable setters/storage.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tutorialStep, tutorialKey]);

  const tutorial = {
    step: tutorialStep,
    onOkay: () => { Analytics.tutorialStarted(); SensorSettings.setMcpTutorialMode(true); setTutorialStep('notify'); },
    onSkip: skipTutorial,
    onActionPicked: () => setTutorialStep('ready'),
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = value.trim();
    if (!text || isRunning) return;
    if (isRecording) stopMic();
    if (tutorialStep) {
      // Only the 'ready' step should build the demo agent; anything earlier is a custom send.
      if (tutorialStep === 'ready') tutorialFlow.start();
      else SensorSettings.setMcpTutorialMode(false);
      endTutorial();
    }
    setValue('');
    setUserEdited(false);
    void send(text);
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const next = e.target.value;
    setValue(next);
    setUserEdited(next.length > 0);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Grow the textarea to fit wrapped/typed content — a plain <textarea> never resizes
  // itself, so height must be recalculated from scrollHeight on every value change
  // (including the live wheel-composed prompt, not just direct typing).
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    // Empty: leave at rows=1 so the placeholder isn't padded into a taller, top-aligned box.
    if (value) el.style.height = `${el.scrollHeight}px`;
  }, [value]);

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 relative">
      {!tutorialStep && (
        <button
          onClick={replayTutorial}
          className="absolute top-4 right-4 text-slate-400 hover:text-slate-700 transition-colors"
          aria-label="Show tutorial"
          title="Show tutorial"
        >
          <Info className="h-6 w-6" />
        </button>
      )}
      <TutorialSpotlight spot={tutorialStep} />
      <h1 className="text-2xl md:text-3xl font-semibold text-gray-800 mb-6 text-center">
        What do you want me to Observe?
      </h1>
      {(isRecording || micStarting) && (
        <div className="w-full max-w-2xl flex items-center gap-2 px-1 pb-1.5 text-xs font-medium text-red-600 relative z-10">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
          </span>
          <span>{micStarting ? 'Starting microphone…' : 'Listening… tap the mic to stop'}</span>
        </div>
      )}
      {customModels.length > 0 && (
        <div className="w-full max-w-2xl flex items-center gap-1.5 px-1 pb-1.5 relative z-10">
          <Cpu className="h-3.5 w-3.5 text-purple-400 flex-shrink-0" />
          <select
            value={modelName}
            onChange={e => setModelName(e.target.value)}
            disabled={isRunning}
            className="flex-1 text-xs text-gray-600 bg-transparent border-0 focus:ring-0 focus:outline-none cursor-pointer disabled:cursor-not-allowed truncate"
          >
            {isAuthenticated && <option value={DEFAULT_CLOUD_MODEL}>Default (cloud)</option>}
            {customModels.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
          </select>
        </div>
      )}
      <form onSubmit={handleSubmit} className={`w-full max-w-2xl flex items-center gap-2 relative z-10`}>
        <textarea
          ref={textareaRef}
          rows={1}
          autoFocus
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="Describe what to monitor…"
          disabled={isRunning}
          className={`flex-1 min-w-0 p-4 md:p-5 text-left text-sm md:text-lg text-gray-700 bg-white border border-gray-200 rounded-3xl shadow-sm disabled:bg-gray-100 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none leading-snug max-h-56 overflow-y-auto`}
        />
        <button
          type="button"
          onClick={() => (isRecording ? stopMic() : startMic())}
          disabled={micStarting || (!isRecording && isRunning)}
          className={`p-4 md:p-5 rounded-full transition-colors flex items-center flex-shrink-0 ${
            isRecording
              ? 'bg-red-600 text-white hover:bg-red-700 animate-pulse'
              : 'bg-gray-700 text-white hover:bg-gray-800 disabled:bg-gray-300 disabled:cursor-not-allowed'
          }`}
          title={isRecording ? 'Stop voice input' : 'Speak to fill the message'}
          aria-label={isRecording ? 'Stop voice input' : 'Start voice input'}
        >
          {micStarting ? <Loader2 className="h-5 w-5 animate-spin" /> : <Mic className="h-5 w-5" />}
        </button>
        <button
          type="submit"
          disabled={isRunning || !value.trim()}
          className={`p-4 md:p-5 bg-gray-700 text-white rounded-full hover:bg-gray-800 disabled:bg-gray-300 transition-colors flex items-center flex-shrink-0 ${tutorialStep === 'ready' ? 'animate-nudge-soft' : ''}`}
          data-spot="ready"
          title="Send"
        >
          {isRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
        </button>
        {tutorialStep === 'ready' && (
          <TutorialBubble placement="above-right" anchorClass="absolute top-0 right-0">
            <div className="w-64 flex flex-col items-end">
              <div className="bg-slate-900 text-white rounded-2xl px-4 py-3 shadow-lg text-sm font-medium text-center">
                Perfect! I have everything I need.
              </div>
              <div className="w-3 h-3 bg-slate-900 rotate-45 -mt-1.5 mr-6" />
            </div>
          </TutorialBubble>
        )}
      </form>

      {/* Pinned behind the form at a fixed spot — absolutely positioned so it never moves
          when the textarea above grows with wrapped text. */}
      <div className={`absolute left-1/2 -translate-x-1/2 top-[72%] md:top-[64%] scale-90 md:scale-100 origin-top pointer-events-auto ${tutorialStep ? 'z-20' : 'z-0'}`}>
        <RecipeInline tutorial={tutorial} onPromptChange={prompt => { if (!userEdited) setValue(prompt); }} />
      </div>
    </div>
  );
};

export default ObserverHero;
