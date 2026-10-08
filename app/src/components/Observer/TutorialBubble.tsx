// src/components/Observer/TutorialBubble.tsx
//
// Tutorial speech bubble that floats ABOVE TutorialSpotlight's grey mask. The bubble itself
// is portaled to <body> (z-[201], mask is z-[200]) so no ancestor stacking context can bury
// it; a zero-size anchor left in the normal tree (positioned by the caller via `anchorClass`)
// tells us where it belongs, and we follow it every frame.

import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

interface Props {
  /** below-center: hangs under the anchor, centered on it. above-right: sits over the anchor, right edges aligned. */
  placement: 'below-center' | 'above-right';
  /** Tailwind classes placing the zero-size anchor inside its `relative` parent. */
  anchorClass: string;
  children: React.ReactNode;
}

const TutorialBubble: React.FC<Props> = ({ placement, anchorClass, children }) => {
  const anchorRef = useRef<HTMLSpanElement>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    let raf: number;
    const loop = () => {
      const r = anchorRef.current?.getBoundingClientRect();
      if (r) {
        const x = Math.round(r.left), y = Math.round(r.top);
        setPos(prev => (prev && prev.x === x && prev.y === y ? prev : { x, y }));
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  const style: React.CSSProperties | null = !pos ? null
    : placement === 'below-center'
      ? { left: pos.x, top: pos.y, transform: 'translateX(-50%)' }
      : { right: window.innerWidth - pos.x, bottom: window.innerHeight - pos.y + 12 };

  return (
    <>
      <span ref={anchorRef} aria-hidden className={`${anchorClass} w-0 h-0 pointer-events-none`} />
      {style && createPortal(
        <div className="fixed z-[201] select-none pointer-events-auto" style={style}>{children}</div>,
        document.body,
      )}
    </>
  );
};

export default TutorialBubble;
