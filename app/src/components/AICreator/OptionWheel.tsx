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
//   • wheel / two-finger scroll → accumulates raw scroll distance, spends it a row at a
//     time, and keeps the remainder as a live offset; settles once the scroll stops. The
//     listener is native (not React's onWheel) so it can preventDefault and stop the page
//     scrolling behind the wheel.
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
  /**
   * Temporarily halts auto-cycling without counting as an interaction. RecipeSplash sets this
   * when the pointer reaches "Build it": `commit()` only runs on transitionend, so a click
   * landing mid-glide would otherwise build the row BEFORE the one the user sees.
   */
  paused?: boolean;
  ariaLabel: string;
  /** Tailwind width classes for the scroll column. Defaults to the original size. */
  widthClass?: string;
  /** Rendered absolutely, centered over the scroll column only (excludes the chevrons). */
  tooltip?: React.ReactNode;
  /** Center-row click, distinct from the neighbor-row glide clicks. Used to enter edit mode. */
  onLabelClick?: () => void;
  /**
   * Fully freezes the wheel — no auto-cycle, no drag/scroll/click/chevron interaction —
   * while still rendering normally. Used by the onboarding tutorial to pin the trigger to
   * "my download is finished" until the user opts out.
   */
  locked?: boolean;
  /** When `value` changes from outside, glide there (shortest way round) instead of jumping. */
  spinOnExternalChange?: boolean;
  /** Light-on-white palette for the inline (Cowork-hero) widget. Defaults to the original white-on-dark splash look. */
  dark?: boolean;
  /** Tailwind text-size classes for the row labels. Defaults to the original splash size. */
  textClass?: string;
  /** Grow/shrink the up/down chevrons to cue the user to click them (onboarding tutorial). */
  pulseChevrons?: boolean;
}

const CYCLE_MS = 2100;         // auto-cycle cadence
const SPIN_MS = 800;           // programmatic spin-to-value glide
const ANIM_MS = 480;           // auto-cycle glide duration (gentle)
const ARROW_MS = 220;          // chevron/click glide + drag/wheel settle duration
// Row/viewport geometry is smaller on mobile so the wheels don't dominate the stacked
// layout. Computed in JS (not just CSS) because the drag/wheel handlers map real screen
// pixels 1:1 to row pixels — a CSS `scale()` would desync finger movement from the glide.
const ROW_REM_DESKTOP = 2.5;
const ROW_REM_COMPACT = 2;
const VISIBLE_DESKTOP = 5;     // rows shown in the viewport
const VISIBLE_COMPACT = 4;
const HALF = 20;               // render ±20 rows (looping) — drag can't run out
const RENDER = Array.from({ length: 2 * HALF + 1 }, (_, i) => i - HALF);
const DRAG_THRESHOLD = 4;
const COMPACT_QUERY = '(max-width: 767px)'; // matches Tailwind's md breakpoint
// How long after the last wheel event we consider the gesture over and ease the sub-row
// remainder to center. Trackpads emit a dense stream of small deltas, so this has to be
// longer than the inter-event gap of a continuous two-finger scroll but short enough to
// still feel like a release.
const WHEEL_SETTLE_MS = 120;

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
  paused = false,
  ariaLabel,
  widthClass = 'w-[13rem] md:w-[16rem]',
  tooltip,
  onLabelClick,
  locked = false,
  spinOnExternalChange = false,
  dark = true,
  textClass = 'text-lg md:text-xl',
  pulseChevrons = false,
}) => {
  const startIndex = Math.max(0, options.findIndex(o => o.id === value));
  const [index, setIndex] = useState(startIndex);
  const [motion, setMotion] = useState(0);        // px glide offset (chevron/auto/click)
  const [drag, setDrag] = useState(0);            // px live drag offset
  const [dragging, setDragging] = useState(false);
  const [instant, setInstant] = useState(false);  // suppress transition for seamless reset
  const [interacted, setInteracted] = useState(false);
  const [animMs, setAnimMs] = useState(ANIM_MS);  // current glide duration (slow auto vs fast click)

  const [wheeling, setWheeling] = useState(false);  // a wheel/trackpad gesture is in flight

  const [compact, setCompact] = useState(
    () => typeof window !== 'undefined' && window.matchMedia?.(COMPACT_QUERY).matches,
  );
  useEffect(() => {
    const mql = window.matchMedia(COMPACT_QUERY);
    const onChange = () => setCompact(mql.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);

  const rowRem = compact ? ROW_REM_COMPACT : ROW_REM_DESKTOP;
  const visible = compact ? VISIBLE_COMPACT : VISIBLE_DESKTOP;
  const rowPx = rowRem * 16;
  const basePx = rowPx * ((visible - 1) / 2 - HALF); // centers offset 0 in the viewport
  // Native listeners below close over refs, not render-scoped values.
  const rowPxRef = useRef(rowPx); rowPxRef.current = rowPx;
  const visibleRef = useRef(visible); visibleRef.current = visible;
  // The row height in effect when a glide *started* — a resize crossing the compact
  // breakpoint mid-glide must not change how `motion` gets converted back into a step
  // count when it settles, or the settle math (motion / rowPx) drifts off the row grid.
  const glideRowPxRef = useRef(rowPx);

  const instantRef = useRef(false); instantRef.current = instant;
  // `instant` (the transition-less reset window) must count as busy: starting a glide there
  // would move the transform with transitions off, so no transitionend fires and `motion`
  // gets stuck ≠ 0 — freezing the wheel while its index keeps changing.
  // A wheel gesture counts as busy too, so the external-value sync can't yank the index out
  // from under the user's fingers mid-scroll.
  const glidingRef = useRef(false); glidingRef.current = motion !== 0 || dragging || instant;
  const busyRef = useRef(false); busyRef.current = glidingRef.current || wheeling;
  const startYRef = useRef(0);
  const dragRef = useRef(0);
  const movedRef = useRef(false);

  // The wheel listener is attached natively (see below) and therefore closes over its first
  // render. These refs keep it reading current values instead of stale ones.
  const viewportRef = useRef<HTMLDivElement>(null);
  const wheelAccumRef = useRef(0);      // unspent scroll distance, in px
  const wheelTimerRef = useRef(0);
  const indexRef = useRef(index); indexRef.current = index;
  const optionsRef = useRef(options); optionsRef.current = options;
  const onChangeRef = useRef(onChange); onChangeRef.current = onChange;

  const reduce = prefersReducedMotion();
  const len = options.length;
  const at = (offset: number) => options[mod(index + offset, len)];

  const markInteracted = () => { if (!interacted) onInteract?.(); setInteracted(true); };

  // Sync to external value changes (only while fully idle). Also retried whenever a
  // glide/drag/wheel gesture settles — if `value` changed while busy, this effect bails and
  // would otherwise never get another chance (deps only include `value`), leaving the wheel
  // visually stuck on a stale row even though the caller's state already moved on.
  useEffect(() => {
    if (busyRef.current) return;
    const i = options.findIndex(o => o.id === value);
    if (i < 0 || i === index) return;
    if (spinOnExternalChange && !reduce) {
      // Shortest way round the loop. Bypasses glide(): programmatic spins must work while
      // `locked` and must not count as a user interaction.
      let d = mod(i - index, len);
      if (d > len / 2) d -= len;
      setAnimMs(SPIN_MS);
      glideRowPxRef.current = rowPx;
      setMotion(-d * rowPx);
      return;
    }
    setIndex(i);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, motion, dragging, instant, wheeling]);

  // Auto-cycle until first interaction.
  useEffect(() => {
    if (!autoCycle || interacted || reduce || paused || locked) return;
    const timer = setInterval(() => {
      if (instantRef.current || busyRef.current) return;
      setAnimMs(ANIM_MS);
      glideRowPxRef.current = rowPxRef.current;
      setMotion(-rowPxRef.current); // glide up one row
    }, CYCLE_MS);
    return () => clearInterval(timer);
  }, [autoCycle, interacted, reduce, paused, locked]);

  // Commit a settled step. onChange is called OUTSIDE the setIndex updater — calling a
  // parent setState inside an updater runs during render and triggers React's
  // "cannot update a component while rendering a different component" warning.
  const commit = (delta: number) => {
    // `delta` may be a non-integer number of rows if `rowPx` changed (e.g. the compact
    // breakpoint flipped) between when the glide started and when its transition ended —
    // round it so we never index into `options` with a fractional/out-of-range value.
    const n = mod(index + Math.round(delta), len);
    setInstant(true);
    setIndex(n);
    onChange(options[n].id);
    setMotion(0);
    setDrag(0);
    requestAnimationFrame(() => requestAnimationFrame(() => setInstant(false)));
  };

  const glide = (delta: number) => {
    if (busyRef.current || locked) return;
    markInteracted();
    if (reduce) { commit(delta); return; }
    setAnimMs(ARROW_MS); // snappy for chevron / neighbor clicks
    glideRowPxRef.current = rowPx;
    setMotion(-delta * rowPx);
  };

  const handleTransitionEnd = () => {
    if (motion === 0) return; // ignore the drag-settle transition (motion already 0)
    commit(-motion / glideRowPxRef.current);
  };

  // ---- Drag: track the pointer anywhere on screen until release --------------
  const onPointerDown = (e: React.PointerEvent) => {
    if (busyRef.current || locked) return;
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
      const rp = rowPxRef.current;
      const steps = Math.round(d / rp);          // dragging down (positive) = earlier items
      const residual = d - steps * rp;
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

  // ---- Wheel / two-finger scroll -------------------------------------------
  // Same pixel-offset model as the drag: accumulate raw scroll distance, spend it a whole
  // row at a time, and keep the remainder as a live offset so the column tracks the gesture
  // continuously. Settles to the nearest row once the scroll stops.
  //
  // Attached natively with { passive: false } because React's synthetic onWheel can't
  // preventDefault — without that, scrolling the wheel would also scroll the page behind it.
  const markInteractedRef = useRef(markInteracted); markInteractedRef.current = markInteracted;
  const lockedRef = useRef(locked); lockedRef.current = locked;

  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      if (lockedRef.current) return;
      // Let a genuinely horizontal gesture (or a shift-scroll) pass through untouched.
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
      e.preventDefault();
      // Mark first, then bail: even an event we can't spend means the user is driving, so
      // the auto-cycle should stop rather than keep spinning under them.
      markInteractedRef.current();
      // Don't fight an in-flight glide: its transitionend still owes us a commit.
      if (glidingRef.current) return;

      // deltaMode 1 is lines (Firefox mouse wheels), 2 is pages.
      const rp = rowPxRef.current;
      const unit = e.deltaMode === 1 ? rp : e.deltaMode === 2 ? rp * visibleRef.current : 1;
      wheelAccumRef.current += e.deltaY * unit;

      // Scrolling down (positive deltaY) advances toward later options, matching the
      // chevron-down direction.
      const steps = Math.trunc(wheelAccumRef.current / rp);
      if (steps !== 0) {
        wheelAccumRef.current -= steps * rp;
        const n = mod(indexRef.current + steps, optionsRef.current.length);
        indexRef.current = n;
        setIndex(n);
        onChangeRef.current(optionsRef.current[n].id);
      }
      // Content moves up as the index grows, hence the negation.
      setDrag(-wheelAccumRef.current);
      setWheeling(true);

      clearTimeout(wheelTimerRef.current);
      wheelTimerRef.current = window.setTimeout(() => {
        wheelAccumRef.current = 0;
        setAnimMs(ARROW_MS);   // snappy settle, like a chevron press
        setWheeling(false);    // re-enables the transition so the remainder eases home
        setDrag(0);
      }, WHEEL_SETTLE_MS);
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', onWheel);
      clearTimeout(wheelTimerRef.current);
    };
  }, []);

  const translateY = basePx + motion + drag;

  return (
    <div className="flex items-center gap-1.5 md:gap-2" aria-label={ariaLabel} role="listbox">
      <div className="flex flex-col">
        <button type="button" disabled={locked} onClick={() => glide(-1)} className={`p-0.5 transition-colors disabled:opacity-20 ${dark ? 'text-white/40 hover:text-white disabled:hover:text-white/40' : 'text-slate-400 hover:text-slate-800 dark:hover:text-slate-100 disabled:hover:text-slate-400'}`} aria-label="Previous">
          <ChevronUp className={`h-5 w-5 ${pulseChevrons ? 'animate-nudge text-slate-800' : ''}`} />
        </button>
        <button type="button" disabled={locked} onClick={() => glide(1)} className={`p-0.5 transition-colors disabled:opacity-20 ${dark ? 'text-white/40 hover:text-white disabled:hover:text-white/40' : 'text-slate-400 hover:text-slate-800 dark:hover:text-slate-100 disabled:hover:text-slate-400'}`} aria-label="Next">
          <ChevronDown className={`h-5 w-5 ${pulseChevrons ? 'animate-nudge text-slate-800' : ''}`} />
        </button>
      </div>

      <div className={`relative ${widthClass}`}>
        {tooltip}
        <div
          ref={viewportRef}
          className={`wheel-mask relative overflow-hidden w-full touch-none select-none ${locked ? '' : 'cursor-grab active:cursor-grabbing'}`}
          style={{ height: `${rowRem * visible}rem` }}
          onPointerDown={onPointerDown}
        >
          <div
            className="absolute inset-x-0 top-0 flex flex-col will-change-transform"
            style={{
              transform: `translateY(${translateY}px)`,
              transition: instant || dragging || wheeling || reduce ? 'none' : `transform ${animMs}ms cubic-bezier(0.22, 1, 0.36, 1)`,
            }}
            onTransitionEnd={handleTransitionEnd}
          >
            {RENDER.map(offset => (
              <div
                key={offset}
                onClick={() => {
                  if (locked) return;
                  if (Math.abs(offset) === 1) glide(offset);
                  else if (offset === 0 && onLabelClick) onLabelClick();
                }}
                title={offset === 0 && onLabelClick && !locked ? 'Click to type' : undefined}
                className={`flex items-center justify-center text-center px-2 ${textClass} font-medium truncate ${dark ? 'text-white' : 'text-slate-800 dark:text-slate-100'} ${
                  locked ? '' : offset !== 0 ? 'cursor-pointer' : onLabelClick ? `cursor-text transition-colors ${dark ? 'hover:text-cyan-200' : 'hover:text-purple-600 dark:hover:text-purple-300'}` : ''
                }`}
                style={{ height: `${rowRem}rem` }}
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
