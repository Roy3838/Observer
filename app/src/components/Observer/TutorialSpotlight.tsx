// src/components/Observer/TutorialSpotlight.tsx
//
// Greys out the whole screen except the elements tagged `data-spot="<spot>"`, using the same
// dark-mask look as EditAgent/SimpleCreatorTutorial. Rendered through a portal so it covers the
// entire app (sidebar/header included). Every tagged element gets its own hole (rounded rect),
// so a bubble and the thing it points at can both stay lit. Pointer events pass straight
// through — it only dims.

import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

interface Props {
  /** A word in the (space-separated) `data-spot` attribute to leave lit; null/undefined = no mask. */
  spot: string | null | undefined;
  pad?: number;
  radius?: number;
}

type Box = { x: number; y: number; w: number; h: number };

const sameBoxes = (a: Box[], b: Box[]) =>
  a.length === b.length && a.every((r, i) => r.x === b[i].x && r.y === b[i].y && r.w === b[i].w && r.h === b[i].h);

const roundedRect = ({ x, y, w, h }: Box, r: number) => {
  const k = Math.min(r, w / 2, h / 2);
  return `M${x + k} ${y}H${x + w - k}A${k} ${k} 0 0 1 ${x + w} ${y + k}V${y + h - k}A${k} ${k} 0 0 1 ${x + w - k} ${y + h}` +
    `H${x + k}A${k} ${k} 0 0 1 ${x} ${y + h - k}V${y + k}A${k} ${k} 0 0 1 ${x + k} ${y}Z`;
};

const TutorialSpotlight: React.FC<Props> = ({ spot, pad = 8, radius = 16 }) => {
  const [boxes, setBoxes] = useState<Box[]>([]);

  // Measure every frame: the targets animate (wheels glide, bubbles mount, pulses scale) and
  // the window can resize.
  useEffect(() => {
    if (!spot) { setBoxes([]); return; }
    let raf: number;
    const loop = () => {
      // Zero-size = the tab is hidden (display:none), so there's nothing to light.
      const rects = Array.from(document.querySelectorAll(`[data-spot~="${spot}"]`))
        .map(el => el.getBoundingClientRect())
        .filter(r => r.width > 0 && r.height > 0);
      const next = rects.map(r => {
        return { x: Math.round(r.left - pad), y: Math.round(r.top - pad), w: Math.round(r.width + pad * 2), h: Math.round(r.height + pad * 2) };
      });
      setBoxes(prev => (sameBoxes(prev, next) ? prev : next));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [spot, pad]);

  if (!spot || boxes.length === 0) return null;

  const path = `M0 0H${window.innerWidth}V${window.innerHeight}H0Z` + boxes.map(b => roundedRect(b, radius)).join('');

  return createPortal(
    <div
      className="fixed inset-0 z-[200] pointer-events-none"
      style={{ background: 'rgba(0,0,0,0.45)', clipPath: `path(evenodd, '${path}')` }}
    />,
    document.body,
  );
};

export default TutorialSpotlight;
