// src/components/AICreator/EditableWheel.tsx
//
// Wraps OptionWheel with click-to-type: clicking the wheel's center row (the option
// currently in focus) swaps the wheel for a plain autofocused input, styled to match the
// wheel's own row exactly so the swap reads as "the text became editable" rather than a
// mode change. On commit, typed text that doesn't match another preset overwrites the
// CURRENT row's own label in place (via onCustom) — so the wheel comes back showing the
// edited text at that same position, and spinning carries it along like any other row.
// Typing another preset's exact label jumps the wheel to that row instead. Esc reverts to
// the wheel untouched.

import React, { useEffect, useState } from 'react';
import OptionWheel, { type WheelOption } from './OptionWheel';

interface EditableWheelProps<T extends WheelOption> {
  options: T[];
  value: string;
  onChange: (id: string) => void;
  onCustom: (text: string) => void;
  onInteract?: () => void;
  paused?: boolean;
  ariaLabel: string;
  widthClass?: string;
  tooltip?: React.ReactNode;
  /** Freezes the wheel (no spin, no click-to-type) — see OptionWheel's `locked`. */
  locked?: boolean;
  /** Glide (rather than jump) when `value` is changed from outside. */
  spinOnExternalChange?: boolean;
  /** Light-on-white palette for the inline (Cowork-hero) widget. Defaults to the original white-on-dark splash look. */
  dark?: boolean;
  /** Tailwind text-size classes for the row labels. Defaults to the original splash size. */
  textClass?: string;
  /** Pulse the wheel's up/down chevrons — see OptionWheel's `pulseChevrons`. */
  pulseChevrons?: boolean;
}

function EditableWheel<T extends WheelOption>({
  options, value, onChange, onCustom, onInteract, paused, ariaLabel, widthClass, tooltip, locked, spinOnExternalChange, dark = true, textClass = 'text-lg md:text-xl', pulseChevrons,
}: EditableWheelProps<T>) {
  const displayText = options.find(o => o.id === value)?.label ?? '';
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(displayText);
  // Sticks once the row is edited, even across the OptionWheel remount below (entering edit
  // mode unmounts it, so its own internal "interacted" flag can't survive the round trip).
  const [settled, setSettled] = useState(false);

  useEffect(() => { if (!editing) setDraft(displayText); }, [displayText, editing]);

  const startEdit = () => { setDraft(displayText); setEditing(true); setSettled(true); };

  const commit = () => {
    const trimmed = draft.trim();
    if (!trimmed || trimmed.toLowerCase() === displayText.toLowerCase()) { setEditing(false); return; }
    const match = options.find(o => o.id !== value && o.label.toLowerCase() === trimmed.toLowerCase());
    if (match) onChange(match.id);
    else onCustom(trimmed);
    setEditing(false);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') { e.preventDefault(); setEditing(false); return; }
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
  };

  if (!editing) {
    return (
      <OptionWheel
        options={options}
        value={value}
        onChange={onChange}
        autoCycle={!settled}
        onInteract={() => { setSettled(true); onInteract?.(); }}
        paused={paused}
        ariaLabel={ariaLabel}
        widthClass={widthClass}
        tooltip={tooltip}
        onLabelClick={locked ? undefined : startEdit}
        locked={locked}
        spinOnExternalChange={spinOnExternalChange}
        dark={dark}
        textClass={textClass}
        pulseChevrons={pulseChevrons}
      />
    );
  }

  return (
    <div className={`flex items-center gap-1.5 md:gap-2 ${widthClass ?? ''}`}>
      <input
        autoFocus
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onKeyDown={onKeyDown}
        onFocus={e => e.currentTarget.select()}
        onBlur={commit}
        aria-label={ariaLabel}
        className={`w-full bg-transparent border-0 border-b-2 text-center ${textClass} font-medium outline-none pb-0.5 transition-colors ${
          dark ? 'border-white/40 focus:border-white/80 text-white' : 'border-slate-300 dark:border-slate-600 focus:border-purple-500 text-slate-800 dark:text-slate-100'
        }`}
      />
    </div>
  );
}

export default EditableWheel;
