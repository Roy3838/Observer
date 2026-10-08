// src/components/AICreator/RecipeInline.tsx
//
// A pair of decorative "When [wheel] then [wheel]" wheels pinned in the background of the
// Observer hero, behind/below the big input box. No card, no tooltip, no toggle, no button —
// spinning a wheel just live-composes the sentence into the parent's textarea via
// `onPromptChange`, so watching the wheels spin IS the input growing. Position is owned by
// the caller (ObserverHero): this component is pinned at a fixed spot and never reflows when
// the input box grows, so it must not be placed in the same flow as that box.
//
// Also hosts the first-run guided demo (owned by ObserverHero): while a `tutorial` step is
// active the trigger wheel is frozen on "my download is finished" and speech bubbles walk
// the user through picking a notification method.

import React, { useEffect, useMemo, useState } from 'react';
import { useAuth } from '@contexts/AuthContext';
import EditableWheel from './EditableWheel';
import TutorialBubble from '../Observer/TutorialBubble';
import { TRIGGERS, ACTIONS, composeRecipePrompt } from './RecipeSplash';

export type TutorialStep = 'hello' | 'notify' | 'ready';

export interface RecipeTutorial {
  step: TutorialStep | null;
  onOkay: () => void;
  onSkip: () => void;
  onActionPicked: () => void;
}

// Light-theme speech bubble hanging BELOW the wheel (the textarea sits above the wheels).
// Portaled above the grey mask (see TutorialBubble) rather than cut out of it.
const bubble = (text: string, buttons?: React.ReactNode) => (
  <TutorialBubble placement="below-center" anchorClass="absolute top-1/2 mt-6 left-1/2">
    <div className="flex flex-col items-center w-56 md:w-72">
      <div className="w-3 h-3 bg-slate-900 rotate-45 -mb-1.5" />
      <div className="bg-slate-900 text-white rounded-2xl px-4 py-3 shadow-lg text-center flex flex-col items-center gap-2.5">
        <span className="text-sm font-medium">{text}</span>
        {buttons}
      </div>
    </div>
  </TutorialBubble>
);

interface RecipeInlineProps {
  tutorial?: RecipeTutorial;
  /** Called whenever the composed sentence changes, so the caller can live-mirror it into its own input. */
  onPromptChange: (prompt: string) => void;
}

const RecipeInline: React.FC<RecipeInlineProps> = ({ onPromptChange, tutorial }) => {
  const step = tutorial?.step ?? null;
  const { user } = useAuth();
  const authEmail = user?.email ?? '';

  const [triggerId, setTriggerId] = useState(TRIGGERS[0].id);
  const [actionId, setActionId] = useState(ACTIONS[0].id);
  const [triggerOverrides, setTriggerOverrides] = useState<Record<string, string>>({});
  const [actionOverrides, setActionOverrides] = useState<Record<string, string>>({});

  const triggerOptions = useMemo(
    () => TRIGGERS.map(t => triggerOverrides[t.id]
      ? { ...t, label: triggerOverrides[t.id], promptFragment: triggerOverrides[t.id], sensor: '$UNKNOWN' as const }
      : t),
    [triggerOverrides],
  );
  const actionOptions = useMemo(
    () => ACTIONS.map(a => actionOverrides[a.id]
      ? { ...a, label: actionOverrides[a.id], actionFragment: actionOverrides[a.id] }
      : a),
    [actionOverrides],
  );

  const trigger = useMemo(() => triggerOptions.find(t => t.id === triggerId), [triggerOptions, triggerId]);
  const action = useMemo(() => actionOptions.find(a => a.id === actionId), [actionOptions, actionId]);

  // The wheels auto-cycle on their own before anyone touches them (decorative). Only start
  // mirroring the composed sentence into the caller's input once the user has actually
  // spun/clicked a wheel — otherwise the box fills itself while sitting untouched.
  const [interacted, setInteracted] = useState(false);
  const markInteracted = () => setInteracted(true);

  // Whenever a tutorial (re)starts, put everything back to the demo's starting state: the
  // download trigger, untouched wheels (no leftover typed-in rows or earlier spins), and no
  // mirroring into the input until the user picks an action again.
  useEffect(() => {
    if (step !== 'hello') return;
    setTriggerId(TRIGGERS[0].id);
    setActionId(ACTIONS[0].id);
    setTriggerOverrides({});
    setActionOverrides({});
    setInteracted(false);
  }, [step]);

  // The wheel is locked for the whole tutorial, so the only thing that can move the trigger
  // is a glide that was already in flight when it started (the auto-cycle, or a spin the user
  // just made): it settles AFTER the reset above and reports its row, overwriting it. Pin it
  // back; the wheel then spins to the download row once it is idle.
  useEffect(() => {
    if (step && triggerId !== TRIGGERS[0].id) setTriggerId(TRIGGERS[0].id);
  }, [step, triggerId]);

  useEffect(() => {
    if (!interacted) return;
    onPromptChange(composeRecipePrompt(trigger, action, authEmail, 'cloud'));
    // onPromptChange is a fresh closure each render — only re-fire when the composed
    // sentence's actual inputs change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trigger, action, authEmail, interacted]);

  return (
    <div className="flex items-center justify-center gap-1.5 md:gap-2">
      <div data-spot="hello" className="flex items-center gap-1.5 md:gap-2">
      <span className="text-xs md:text-base font-semibold text-slate-400 select-none pointer-events-none shrink-0">When</span>
      <EditableWheel
        options={triggerOptions}
        value={triggerId}
        onChange={setTriggerId}
        onCustom={text => { markInteracted(); setTriggerOverrides(prev => ({ ...prev, [triggerId]: text })); }}
        onInteract={markInteracted}
        ariaLabel="Choose a trigger"
        locked={!!step}
        spinOnExternalChange
        tooltip={step === 'hello' ? bubble(
          "Hi! I'm Observer, Let's do a quick demo. I'll monitor a download so you don't have to.",
          <div className="flex items-center gap-3 pt-0.5">
            <button onClick={tutorial!.onSkip} className="text-slate-400 hover:text-white text-sm font-medium transition-colors">Skip</button>
            <button onClick={tutorial!.onOkay} className="px-4 py-1.5 rounded-full bg-white text-slate-900 text-sm font-semibold hover:bg-slate-200 transition-colors">Okay!</button>
          </div>,
        ) : undefined}
        widthClass="w-[9rem] md:w-[15rem]"
        textClass="text-[10px] md:text-sm"
        dark={false}
      />
      </div>
      <div data-spot="notify ready" className="flex items-center gap-1.5 md:gap-2">
      <span className="text-xs md:text-base font-semibold text-slate-400 select-none pointer-events-none shrink-0">then</span>
      <EditableWheel
        options={actionOptions}
        value={actionId}
        onChange={setActionId}
        onCustom={text => { markInteracted(); setActionOverrides(prev => ({ ...prev, [actionId]: text })); }}
        onInteract={() => { markInteracted(); if (step === 'notify') tutorial!.onActionPicked(); }}
        ariaLabel="Choose an action"
        tooltip={step === 'notify' ? bubble('Which way should I notify you?') : undefined}
        pulseChevrons={step === 'notify'}
        widthClass="w-[6.5rem] md:w-[12rem]"
        textClass="text-[10px] md:text-sm"
        dark={false}
      />
      </div>
    </div>
  );
};

export default RecipeInline;
