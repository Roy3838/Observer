// src/components/AICreator/RecipeInline.tsx
//
// A pair of decorative "When [wheel] then [wheel]" wheels pinned in the background of the
// Observer hero, behind/below the big input box. No card, no tooltip, no toggle, no button —
// spinning a wheel just live-composes the sentence into the parent's textarea via
// `onPromptChange`, so watching the wheels spin IS the input growing. Position is owned by
// the caller (ObserverHero): this component is pinned at a fixed spot and never reflows when
// the input box grows, so it must not be placed in the same flow as that box.

import React, { useEffect, useMemo, useState } from 'react';
import { useAuth } from '@contexts/AuthContext';
import EditableWheel from './EditableWheel';
import { TRIGGERS, ACTIONS, composeRecipePrompt } from './RecipeSplash';

interface RecipeInlineProps {
  /** Called whenever the composed sentence changes, so the caller can live-mirror it into its own input. */
  onPromptChange: (prompt: string) => void;
}

const RecipeInline: React.FC<RecipeInlineProps> = ({ onPromptChange }) => {
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

  useEffect(() => {
    if (!interacted) return;
    onPromptChange(composeRecipePrompt(trigger, action, authEmail, 'cloud'));
    // onPromptChange is a fresh closure each render — only re-fire when the composed
    // sentence's actual inputs change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trigger, action, authEmail, interacted]);

  return (
    <div className="flex items-center justify-center gap-1.5 md:gap-2">
      <div className="flex items-center gap-1.5 md:gap-2">
      <span className="text-xs md:text-base font-semibold text-slate-400 select-none pointer-events-none shrink-0">When</span>
      <EditableWheel
        options={triggerOptions}
        value={triggerId}
        onChange={setTriggerId}
        onCustom={text => { markInteracted(); setTriggerOverrides(prev => ({ ...prev, [triggerId]: text })); }}
        onInteract={markInteracted}
        ariaLabel="Choose a trigger"
        widthClass="w-[9rem] md:w-[15rem]"
        textClass="text-[10px] md:text-sm"
        dark={false}
      />
      </div>
      <div className="flex items-center gap-1.5 md:gap-2">
      <span className="text-xs md:text-base font-semibold text-slate-400 select-none pointer-events-none shrink-0">then</span>
      <EditableWheel
        options={actionOptions}
        value={actionId}
        onChange={setActionId}
        onCustom={text => { markInteracted(); setActionOverrides(prev => ({ ...prev, [actionId]: text })); }}
        onInteract={markInteracted}
        ariaLabel="Choose an action"
        widthClass="w-[6.5rem] md:w-[12rem]"
        textClass="text-[10px] md:text-sm"
        dark={false}
      />
      </div>
    </div>
  );
};

export default RecipeInline;
