// src/components/Observer/FloatingAgentsContext.tsx
//
// Shared "docked vs floating" state for agent cards spawned inline in the Observer chat
// (see MCP.tsx's 'agent-card' block) and rendered loose over the page (RunningAgentsStrip).
//
// A card starts docked — an ordinary chat message. Dragging its header past a small
// threshold pops it into the floating overlay; dragging a floating card back onto its own
// inline placeholder pill re-docks it (a per-agent dock target, not "anywhere over the chat
// pane" — that pane fills almost the whole screen in the unboxed Observer layout, so treating
// the whole thing as a drop zone meant nearly every release re-docked the card instead of
// letting it stay wherever it was dropped). Only *membership* (which ids are floating) and
// each card's *initial* pop-out position live here — that's state a handful of components
// need to agree on. Position during an active drag is local state inside whichever
// AgentLiveCard instance is currently being dragged (see dragSessionRef below), so a drag
// never re-renders the rest of the chat: only the one card that's actually moving repaints
// per pointermove.
//
// Handing a single continuous drag off between two different component instances (the
// inline card, then the floating card that replaces it the instant it pops out) is the
// tricky part — `dragSessionRef` is how: the window-level pointermove/up listeners that
// drive the gesture always call through this ref, and the floating card overwrites it with
// its own handlers as soon as it mounts, so the same drag just keeps going in a new pair of
// hands without missing a frame.

import React, { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';

export interface FloatingPos {
  x: number;
  y: number;
  /** Set when the card was sent to the corner programmatically (flyToCorner): the inline
   *  card's on-screen top-left, so the floating card can animate from there to x/y. */
  from?: { x: number; y: number };
}

export interface SlotSize { width: number; height: number }

const CORNER_MARGIN = 16;
const CORNER_TOP = 72; // clears the top bar
const CORNER_CASCADE = 24;
const CARD_MAX_WIDTH = 350;

export interface DragSession {
  /** Called on every pointermove once the gesture is past the drag threshold, with the
   *  card's would-be top-left (pointer position minus the original grab offset). */
  onMove: (cardX: number, cardY: number) => void;
  /** Called once on pointerup. `clientX`/`clientY` are the raw pointer position (for the
   *  drop-zone hit test); `cardX`/`cardY` mirror onMove's last values. */
  onEnd: (cardX: number, cardY: number, clientX: number, clientY: number) => void;
}

interface FloatingAgentsValue {
  floating: Record<string, FloatingPos>;
  isFloating: (agentId: string) => boolean;
  /** Pop an agent out of the chat flow into the floating overlay at this card position. */
  popOut: (agentId: string, cardX: number, cardY: number) => void;
  /** Remove an agent from the floating set — it reappears inline where its chat message is. */
  dock: (agentId: string) => void;
  /** Registers (or, with `el: null`, unregisters) the DOM node of an agent's inline
   *  placeholder pill as its dock target — call from the pill's `ref`. */
  setDockTarget: (agentId: string, el: HTMLElement | null) => void;
  /** True while `clientX`/`clientY` sit over this agent's own registered dock target
   *  (measured live, so scrolling the transcript while dragging still hit-tests correctly). */
  isOverDockTarget: (agentId: string, clientX: number, clientY: number) => boolean;
  /** Registers (or unregisters, with null) the DOM node of an agent's inline card. */
  setCardEl: (agentId: string, el: HTMLElement | null) => void;
  /** Records an agent's inline card size so its dock placeholder can match it exactly. */
  measureSlot: (agentId: string) => void;
  /** Size of the agent's inline card the last time it left the chat (null if never measured). */
  getSlotSize: (agentId: string) => SlotSize | null;
  /** Animate an agent's inline card into the top-right corner of the screen, floating. */
  flyToCorner: (agentId: string) => void;
  /** The in-progress drag's current handlers — see file header. Null when nothing is being dragged. */
  dragSessionRef: React.MutableRefObject<DragSession | null>;
}

// MCP.tsx (the inline agent-card renderer) is shared by callers that don't spawn agents
// inline at all (GetStarted, MCPPanel) and so never wrap themselves in
// FloatingAgentsProvider. Rather than force every caller to provide one, the context
// defaults to an inert no-op implementation: nothing is ever "floating" and pop-out/dock
// are no-ops, so those callers behave exactly as before without needing to know this
// feature exists.
function createDefaultValue(): FloatingAgentsValue {
  return {
    floating: {},
    isFloating: () => false,
    popOut: () => {},
    dock: () => {},
    setDockTarget: () => {},
    isOverDockTarget: () => false,
    setCardEl: () => {},
    measureSlot: () => {},
    getSlotSize: () => null,
    flyToCorner: () => {},
    dragSessionRef: { current: null },
  };
}

const FloatingAgentsContext = createContext<FloatingAgentsValue>(createDefaultValue());

export const FloatingAgentsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [floating, setFloating] = useState<Record<string, FloatingPos>>({});
  const dockTargetsRef = useRef<Record<string, HTMLElement | null>>({});
  const dragSessionRef = useRef<DragSession | null>(null);
  const cardElsRef = useRef<Record<string, HTMLElement | null>>({});
  const slotSizesRef = useRef<Record<string, SlotSize>>({});

  const isFloating = useCallback((agentId: string) => Object.prototype.hasOwnProperty.call(floating, agentId), [floating]);

  const popOut = useCallback((agentId: string, cardX: number, cardY: number) => {
    setFloating(prev => ({ ...prev, [agentId]: { x: cardX, y: cardY } }));
  }, []);

  const dock = useCallback((agentId: string) => {
    setFloating(prev => {
      if (!(agentId in prev)) return prev;
      const next = { ...prev };
      delete next[agentId];
      return next;
    });
  }, []);

  const setDockTarget = useCallback((agentId: string, el: HTMLElement | null) => {
    dockTargetsRef.current[agentId] = el;
  }, []);

  const isOverDockTarget = useCallback((agentId: string, clientX: number, clientY: number) => {
    const el = dockTargetsRef.current[agentId];
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
  }, []);

  const setCardEl = useCallback((agentId: string, el: HTMLElement | null) => {
    cardElsRef.current[agentId] = el;
  }, []);

  const measureSlot = useCallback((agentId: string) => {
    const el = cardElsRef.current[agentId];
    if (!el) return;
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) slotSizesRef.current[agentId] = { width: r.width, height: r.height };
  }, []);

  const getSlotSize = useCallback((agentId: string) => slotSizesRef.current[agentId] ?? null, []);

  const flyToCorner = useCallback((agentId: string) => {
    const el = cardElsRef.current[agentId];
    if (!el) return; // no inline card on screen (e.g. conversation scrolled away/unmounted) — leave it docked
    measureSlot(agentId);
    const r = el.getBoundingClientRect();
    setFloating(prev => {
      if (agentId in prev) return prev;
      const width = Math.min(CARD_MAX_WIDTH, r.width);
      return {
        ...prev,
        [agentId]: {
          x: Math.max(CORNER_MARGIN, window.innerWidth - width - CORNER_MARGIN),
          y: CORNER_TOP + Object.keys(prev).length * CORNER_CASCADE,
          from: { x: r.left, y: r.top },
        },
      };
    });
  }, [measureSlot]);

  const value = useMemo<FloatingAgentsValue>(() => ({
    floating, isFloating, popOut, dock, setDockTarget, isOverDockTarget,
    setCardEl, measureSlot, getSlotSize, flyToCorner, dragSessionRef,
  }), [floating, isFloating, popOut, dock, setDockTarget, isOverDockTarget, setCardEl, measureSlot, getSlotSize, flyToCorner]);

  return <FloatingAgentsContext.Provider value={value}>{children}</FloatingAgentsContext.Provider>;
};

export function useFloatingAgents(): FloatingAgentsValue {
  return useContext(FloatingAgentsContext);
}
