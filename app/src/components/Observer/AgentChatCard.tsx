// src/components/Observer/AgentChatCard.tsx
//
// Places an agent's live card in the chat transcript, and keeps it in the chat window once
// the conversation scrolls past it: the card sticks to the top of the transcript as a one-line
// bar (status, countdown, last word, video thumbnail) and unfolds again on tap, or when the
// user scrolls back up to it.
//
// Two siblings, both direct children of the scroll container (sticky only holds within its
// parent, so a wrapper would unstick the card as soon as that wrapper scrolled away):
//   - a zero-height sticky anchor the card overflows from, so folding it moves nothing
//   - a spacer holding the card's full height in the flow, so the transcript never jumps

import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { CompleteAgent } from '@utils/agent_database';
import type { StreamState } from '@utils/streamManager';
import AgentLiveCard from './AgentLiveCard';

/** Height a pinned (folded) card takes, with its gap, when several are stacked. */
export const PINNED_CARD_HEIGHT = 58;

interface AgentChatCardProps {
  agent: CompleteAgent;
  isRunning: boolean;
  isStarting: boolean;
  streams: StreamState;
  onToggle: (agentId: string, isCurrentlyRunning: boolean) => void;
  onSelectAgent?: (agentId: string) => void;
  scrollerRef: React.RefObject<HTMLElement>;
  /** Place in the pinned stack, or null to let the card scroll away (e.g. it's stopped). */
  pinIndex: number | null;
}

const AgentChatCard: React.FC<AgentChatCardProps> = ({ scrollerRef, pinIndex, ...card }) => {
  const anchorRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const spacerRef = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState(false);
  const pinnedRef = useRef(false);
  const [unfolded, setUnfolded] = useState(false);
  const canPin = pinIndex !== null;

  // Pinning again starts folded.
  useEffect(() => { if (!pinned) setUnfolded(false); }, [pinned]);

  // Unpinned, the anchor sits exactly on the spacer; stuck, it's below it.
  const update = useCallback(() => {
    const anchor = anchorRef.current;
    const spacer = spacerRef.current;
    if (!anchor || !spacer) return;
    const next = canPin && anchor.getBoundingClientRect().top > spacer.getBoundingClientRect().top + 1;
    pinnedRef.current = next;
    setPinned(next);
  }, [canPin]);

  // Sticky measures from inside the scroller's padding; pin flush with its top edge instead,
  // so no line of text shows above the card.
  const [scrollerPadding, setScrollerPadding] = useState(0);

  // The transcript can change without a scroll event, so re-check after every render too.
  useLayoutEffect(update);

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    scroller.addEventListener('scroll', update, { passive: true });
    const observer = new ResizeObserver(() => {
      setScrollerPadding(parseFloat(getComputedStyle(scroller).paddingTop) || 0);
      update();
    });
    observer.observe(scroller);
    return () => {
      scroller.removeEventListener('scroll', update);
      observer.disconnect();
    };
  }, [scrollerRef, update]);

  // The spacer follows the card's unfolded height only (0 = the tab is hidden).
  useEffect(() => {
    const el = cardRef.current;
    const spacer = spacerRef.current;
    if (!el || !spacer) return;
    const measure = () => {
      if (!pinnedRef.current && el.offsetHeight > 0) spacer.style.height = `${el.offsetHeight}px`;
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <>
      <div
        ref={anchorRef}
        className={`h-0 pointer-events-none ${!canPin ? '' : unfolded ? 'sticky z-[21]' : 'sticky z-20'}`}
        style={canPin ? { top: (pinIndex ?? 0) * PINNED_CARD_HEIGHT - scrollerPadding } : undefined}
      >
        {/* Pinned on mobile, the card clears the fixed menu button (AppHeader). */}
        <div ref={cardRef} className={`pointer-events-auto max-w-[350px] ${pinned ? 'max-md:ml-12' : ''}`}>
          <AgentLiveCard {...card} pinned={pinned} unfolded={unfolded} onUnfold={setUnfolded} />
        </div>
      </div>
      <div ref={spacerRef} className="!mt-0" aria-hidden />
    </>
  );
};

export default AgentChatCard;
