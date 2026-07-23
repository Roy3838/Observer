// src/components/AICreator/OptionWheel.tsx
//
// A vertical "slot machine" picker used by the RecipeSplash onboarding.
//
// One pixel-offset model drives every interaction:
//   • auto-cycle / chevrons / clicking a neighbor → a `motion` glide of one row that
//     commits the index on transition-end and snaps back seamlessly.
//   • click-and-drag → the column tracks the pointer live via window listeners (so the
//     drag continues anywhere on screen and only ends on pointer-up), then settles to the
//     nearest row on release. Uses absolute clientY, so the wheel's bounding box is
//     irrelevant.
//
// The option list loops (`at()` wraps via mod) and we render a wide window (±HALF rows),
// so a drag can never run off the end. Emphasis/fade is a center-peaked mask gradient
// (see `.wheel-mask` in index.css): text scrolls THROUGH it, so opacity changes smoothly
// with motion — no keyframes. Fixed width + row height keep neighbors from shifting.

import React, { useEffect, useRef, useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

export interface WheelOption {
  id: string;
  label: string;
}

interface OptionWheelProps {
  options: WheelOption[];
  value: string;
  onChange: (id: string) => void;
  autoCycle?: boolean;
  /** Fires once, on the user's first interaction (stops auto-cycling). */
  onInteract?: () => void;
  ariaLabel: string;
  /** Tailwind width classes for the scroll column. Defaults to the original size. */
  widthClass?: string;
  /** Rendered absolutely, centered over the scroll column only (excludes the chevrons). */
  tooltip?: React.ReactNode;
}

const CYCLE_MS = 2100;         // auto-cycle cadence
const ANIM_MS = 480;           // auto-cycle glide duration (gentle)
const ARROW_MS = 120;          // chevron/click glide duration (snappy)
const ROW_REM = 2.5;
const ROW_PX = ROW_REM * 16;   // 40px
const VISIBLE = 5;             // rows shown in the viewport
const HALF = 20;               // render ±20 rows (looping) — drag can't run out
const RENDER = Array.from({ length: 2 * HALF + 1 }, (_, i) => i - HALF);
const BASE_PX = ROW_PX * ((VISIBLE - 1) / 2 - HALF); // centers offset 0 in the viewport
const DRAG_THRESHOLD = 4;

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const mod = (n: number, m: number) => ((n % m) + m) % m;

const OptionWheel: React.FC<OptionWheelProps> = ({
  options,
  value,
  onChange,
  autoCycle = true,
  onInteract,
  ariaLabel,
  widthClass = 'w-[13rem] md:w-[16rem]',
  tooltip,
}) => {
  const startIndex = Math.max(0, options.findIndex(o => o.id === value));
  const [index, setIndex] = useState(startIndex);
  const [motion, setMotion] = useState(0);        // px glide offset (chevron/auto/click)
  const [drag, setDrag] = useState(0);            // px live drag offset
  const [dragging, setDragging] = useState(false);
  const [instant, setInstant] = useState(false);  // suppress transition for seamless reset
  const [interacted, setInteracted] = useState(false);
  const [animMs, setAnimMs] = useState(ANIM_MS);  // current glide duration (slow auto vs fast click)

  const instantRef = useRef(false); instantRef.current = instant;
  // `instant` (the transition-less reset window) must count as busy: starting a glide there
  // would move the transform with transitions off, so no transitionend fires and `motion`
  // gets stuck ≠ 0 — freezing the wheel while its index keeps changing.
  const busyRef = useRef(false); busyRef.current = motion !== 0 || dragging || instant;
  const startYRef = useRef(0);
  const dragRef = useRef(0);
  const movedRef = useRef(false);

  const reduce = prefersReducedMotion();
  const len = options.length;
  const at = (offset: number) => options[mod(index + offset, len)];

  const markInteracted = () => { if (!interacted) onInteract?.(); setInteracted(true); };

  // Sync to external value changes (only while fully idle).
  useEffect(() => {
    if (busyRef.current) return;
    const i = options.findIndex(o => o.id === value);
    if (i >= 0 && i !== index) setIndex(i);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  // Auto-cycle until first interaction.
  useEffect(() => {
    if (!autoCycle || interacted || reduce) return;
    const timer = setInterval(() => {
      if (instantRef.current || busyRef.current) return;
      setAnimMs(ANIM_MS);
      setMotion(-ROW_PX); // glide up one row
    }, CYCLE_MS);
    return () => clearInterval(timer);
  }, [autoCycle, interacted, reduce]);

  // Commit a settled step. onChange is called OUTSIDE the setIndex updater — calling a
  // parent setState inside an updater runs during render and triggers React's
  // "cannot update a component while rendering a different component" warning.
  const commit = (delta: number) => {
    const n = mod(index + delta, len);
    setInstant(true);
    setIndex(n);
    onChange(options[n].id);
    setMotion(0);
    setDrag(0);
    requestAnimationFrame(() => requestAnimationFrame(() => setInstant(false)));
  };

  const glide = (delta: number) => {
    if (busyRef.current) return;
    markInteracted();
    if (reduce) { commit(delta); return; }
    setAnimMs(ARROW_MS); // snappy for chevron / neighbor clicks
    setMotion(-delta * ROW_PX);
  };

  const handleTransitionEnd = () => {
    if (motion === 0) return; // ignore the drag-settle transition (motion already 0)
    commit(-motion / ROW_PX);
  };

  // ---- Drag: track the pointer anywhere on screen until release --------------
  const onPointerDown = (e: React.PointerEvent) => {
    if (busyRef.current) return;
    markInteracted();
    startYRef.current = e.clientY;
    dragRef.current = 0;
    movedRef.current = false;
    setDragging(true);
  };

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: PointerEvent) => {
      const dy = e.clientY - startYRef.current;
      if (Math.abs(dy) > DRAG_THRESHOLD) movedRef.current = true;
      dragRef.current = dy;
      setDrag(dy);
    };
    const onUp = () => {
      setDragging(false);
      if (!movedRef.current) { setDrag(0); return; } // a tap — let the row onClick handle it
      const d = dragRef.current;
      const steps = Math.round(d / ROW_PX);          // dragging down (positive) = earlier items
      const residual = d - steps * ROW_PX;
      // Commit the whole-row shift instantly (pixel-continuous with the finger), then ease the
      // sub-row remainder to center. onChange is called outside any updater (see `commit`).
      const n = mod(index - steps, len);
      setInstant(true);
      setIndex(n);
      onChange(options[n].id);
      setDrag(residual);
      requestAnimationFrame(() => requestAnimationFrame(() => { setInstant(false); setDrag(0); }));
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging]);

  const translateY = BASE_PX + motion + drag;

  return (
    <div className="flex items-center gap-1.5 md:gap-2" aria-label={ariaLabel} role="listbox">
      <div className="flex flex-col">
        <button type="button" onClick={() => glide(-1)} className="p-0.5 text-white/40 hover:text-white transition-colors" aria-label="Previous">
          <ChevronUp className="h-5 w-5" />
        </button>
        <button type="button" onClick={() => glide(1)} className="p-0.5 text-white/40 hover:text-white transition-colors" aria-label="Next">
          <ChevronDown className="h-5 w-5" />
        </button>
      </div>

      <div className={`relative ${widthClass}`}>
        {tooltip}
        <div
          className="wheel-mask relative overflow-hidden w-full touch-none select-none cursor-grab active:cursor-grabbing"
          style={{ height: `${ROW_REM * VISIBLE}rem` }}
          onPointerDown={onPointerDown}
        >
          <div
            className="absolute inset-x-0 top-0 flex flex-col will-change-transform"
            style={{
              transform: `translateY(${translateY}px)`,
              transition: instant || dragging || reduce ? 'none' : `transform ${animMs}ms cubic-bezier(0.22, 1, 0.36, 1)`,
            }}
            onTransitionEnd={handleTransitionEnd}
          >
            {RENDER.map(offset => (
              <div
                key={offset}
                onClick={() => { if (Math.abs(offset) === 1) glide(offset); }}
                className={`flex items-center justify-center text-center px-2 text-lg md:text-xl font-medium text-white truncate ${offset !== 0 ? 'cursor-pointer' : ''}`}
                style={{ height: `${ROW_REM}rem` }}
              >
                {at(offset).label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default OptionWheel;
