import { useState, useEffect, useCallback, useRef } from 'react';

const EyeLogo = ({
  getTarget,
  size = 300
}: {
  // Returns the current point (in viewport coords) the eye should look at, or null.
  getTarget: () => { x: number; y: number } | null;
  size?: number;
}) => {
  const scale = size / 150; // Base size is 150
  const eyeRadius = 67 * scale;
  const pupilRadius = 18 * scale;
  const strokeWidth = 16 * scale;
  const maxPupilOffset = eyeRadius - pupilRadius - (7 * scale);

  const pupilRef = useRef<SVGCircleElement>(null);
  const posRef = useRef({ x: 0, y: 0 }); // current rendered pupil offset

  // Single rAF loop: lerp the pupil toward its target every frame and write
  // straight to the DOM. No React state, no CSS transition => continuous 60fps.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const target = getTarget();
      let destX = 0;
      let destY = 0;

      if (target) {
        const rect = document.getElementById('eye-container')?.getBoundingClientRect();
        if (rect) {
          const eyeCenterX = rect.left + rect.width / 2;
          const eyeCenterY = rect.top + rect.height / 2;
          const dx = target.x - eyeCenterX;
          const dy = target.y - eyeCenterY;
          const angle = Math.atan2(dy, dx);
          const distance = Math.min(maxPupilOffset, Math.hypot(dx, dy) / 8);
          destX = Math.cos(angle) * distance;
          destY = Math.sin(angle) * distance;
        }
      }

      // Exponential smoothing toward the destination.
      const cur = posRef.current;
      cur.x += (destX - cur.x) * 0.18;
      cur.y += (destY - cur.y) * 0.18;

      if (pupilRef.current) {
        pupilRef.current.setAttribute('cx', String(cur.x));
        pupilRef.current.setAttribute('cy', String(cur.y));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [getTarget, maxPupilOffset]);

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox={`${-75 * scale} ${-75 * scale} ${150 * scale} ${150 * scale}`}
      className="w-full h-full"
    >
      <circle
        cx="0"
        cy="0"
        r={eyeRadius}
        fill="none"
        stroke="currentColor"
        strokeWidth={strokeWidth}
      />
      <circle ref={pupilRef} cx="0" cy="0" r={pupilRadius} fill="currentColor" />
    </svg>
  );
};

const TAGLINE_LINE_1 = 'Local open-source micro-agents that monitor your screen,';
const TAGLINE_LINE_2 = "so you don't have to.";
const TAGLINE_FULL = TAGLINE_LINE_1 + '\n' + TAGLINE_LINE_2;

const ObserverLanding = () => {
  const mouseTargetRef = useRef<{ x: number; y: number } | null>(null);
  const [scrollProgress, setScrollProgress] = useState(0);
  const heroRef = useRef<HTMLDivElement>(null);

  // Terminal-cursor typing effect for the tagline
  const [typedCount, setTypedCount] = useState(0);
  const [typingActive, setTypingActive] = useState(false);
  const cursorRef = useRef<HTMLSpanElement>(null);
  const typingActiveRef = useRef(false);
  const hasTypedRef = useRef(false);

  const typedLine1 = TAGLINE_FULL.slice(0, typedCount).split('\n')[0];
  const typedLine2 = TAGLINE_FULL.slice(0, typedCount).split('\n')[1] ?? null;
  const typingDone = typedCount >= TAGLINE_FULL.length;

  // Mouse tracking - just stash the latest position in a ref (cheap, no re-render).
  // The eye's own rAF loop reads this each frame.
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseTargetRef.current = { x: e.clientX, y: e.clientY };
    };
    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // What the eye should look at: the typing cursor while the tagline types itself,
  // otherwise the mouse. Read live from refs so it never triggers a re-render.
  const getEyeTarget = useCallback(() => {
    if (typingActiveRef.current && cursorRef.current) {
      const r = cursorRef.current.getBoundingClientRect();
      return { x: r.left, y: r.top + r.height / 2 };
    }
    return mouseTargetRef.current;
  }, []);

  // Scroll tracking
  useEffect(() => {
    const handleScroll = () => {
      if (!heroRef.current) return;

      const scrollY = window.scrollY;
      const vh = window.innerHeight;
      // Animation completes at 50% of viewport scroll
      const progress = Math.min(1, Math.max(0, scrollY / (vh * 0.5)));

      setScrollProgress(progress);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Calculate animated values based on scroll
  const eyeSizeMax = 280;
  const eyeSizeMin = 80;
  const eyeSize = eyeSizeMax - scrollProgress * (eyeSizeMax - eyeSizeMin); // 280px -> 80px
  // The eye is centered on the row origin, so its right edge sits at eyeSize/2.
  // The "bserver" offset below is tuned for the final (small) eye, so add back the
  // extra half-radius the eye currently has. Goes to 0 as the animation completes.
  const eyeEdgeCompensation = (eyeSize - eyeSizeMin) / 2;
  const textOpacity = Math.max(0, (scrollProgress - 0.3) / 0.7); // Fade in after 30%

  // Kick off typing once the tagline is essentially visible
  useEffect(() => {
    if (hasTypedRef.current || textOpacity < 0.85) return;
    hasTypedRef.current = true;
    typingActiveRef.current = true;
    setTypingActive(true);
  }, [textOpacity]);

  // Advance the typed characters
  useEffect(() => {
    if (!typingActive) return;
    if (typedCount >= TAGLINE_FULL.length) {
      const t = setTimeout(() => {
        typingActiveRef.current = false;
        setTypingActive(false);
      }, 900);
      return () => clearTimeout(t);
    }
    const ch = TAGLINE_FULL[typedCount];
    const delay = ch === '\n' ? 260 : ch === ',' || ch === '.' ? 140 : 34 + Math.random() * 40;
    const t = setTimeout(() => setTypedCount((c) => c + 1), delay);
    return () => clearTimeout(t);
  }, [typingActive, typedCount]);

  // Final text size when fully scrolled
  const finalTextSize = 72;

  // Calculate offset: at scroll=0, eye is centered. At scroll=1, "Observer AI" is centered.
  const bserverWidth = finalTextSize * 5.4;
  const eyeTranslateX = scrollProgress * -(bserverWidth / 2);

  // Once animation is complete (scrollProgress >= 1), switch to absolute positioning
  const isAnimationComplete = scrollProgress >= 1;

  return (
    // Hero is 150vh: 100vh visible + 50vh for the animation scroll distance
    <div ref={heroRef} className="relative h-[150vh]">
      {/* Container switches from fixed to absolute when animation completes */}
      <div
        className={`${isAnimationComplete ? 'absolute' : 'fixed'} left-0 right-0 flex flex-col items-center justify-center pointer-events-none`}
        style={{
          zIndex: 10,
          // When fixed: fill viewport and center content
          // When absolute: position so content center is at 100vh from doc top (same visual position)
          top: isAnimationComplete ? '50vh' : 0,
          bottom: 0,
          height: isAnimationComplete ? undefined : '100vh',
        }}
      >
        {/* Logo row: Eye + "bserver" */}
        <div className="relative flex items-center justify-center mb-8">
          {/* Eye container */}
          <div
            id="eye-container"
            className="text-white flex-shrink-0"
            style={{
              width: `${eyeSize}px`,
              height: `${eyeSize}px`,
              transform: `translateX(${eyeTranslateX + scrollProgress * 20}px)`,
            }}
          >
            <EyeLogo getTarget={getEyeTarget} size={eyeSize} />
          </div>

          {/* "bserver" text */}
          <div
            className="absolute flex items-center"
            style={{
              left: '50%',
              transform: `translateX(${eyeTranslateX + eyeEdgeCompensation + 40 - 4 + scrollProgress * 24}px)`,
              opacity: textOpacity,
            }}
          >
            <h1
              className="text-white font-golos font-bold tracking-tight whitespace-nowrap"
              style={{
                fontSize: `${finalTextSize}px`,
                lineHeight: 1,
              }}
            >
              <span className="sr-only">Observer AI</span>
              <span aria-hidden="true">bserver AI</span>
            </h1>
          </div>
        </div>

        {/* Tagline - fades in with scroll */}
        <div className="relative font-mono text-lg md:text-xl leading-relaxed px-6 text-left">
          {/* Invisible full copy reserves the final layout box so typed text lands in place */}
          <p aria-hidden="true" className="invisible whitespace-pre-wrap">
            {TAGLINE_LINE_1}
            {'\n'}
            {TAGLINE_LINE_2}
          </p>
          {/* Typed overlay, only present once typing begins */}
          {(typingActive || typedCount > 0) && (
            <p className="absolute inset-0 px-6 text-[#8899A6] whitespace-pre-wrap">
              {typedLine1}
              {typedLine2 !== null && (
                <>
                  {'\n'}
                  <span className="text-[#5C6975]">{typedLine2}</span>
                </>
              )}
              <span
                ref={cursorRef}
                aria-hidden="true"
                className="inline-block w-[0.55em] h-[1.1em] -mb-[0.15em] ml-[1px] bg-[#8899A6] align-baseline"
                style={{
                  animation: typingDone ? 'observer-cursor-blink 1s step-end infinite' : undefined,
                }}
              />
            </p>
          )}
        </div>

        {/* CTA - visible from start */}
        <div className="mt-8 flex gap-4">
          <a
            href="https://app.observer-ai.com"
            className="inline-block px-8 py-4 rounded-full font-semibold text-white border border-white/20 hover:bg-white/10 transition-all duration-200 pointer-events-auto"
          >
            Try it
          </a>
          <a
            href="#downloads"
            className="inline-block px-8 py-4 rounded-full font-semibold text-white border border-white/20 hover:bg-white/10 transition-all duration-200 pointer-events-auto"
          >
            Download
          </a>
        </div>

        {/* Scroll indicator - only visible when not scrolled */}
        <div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          style={{
            opacity: 1 - scrollProgress * 3,
          }}
        >
          <div className="flex flex-col items-center gap-2">
            <span className="text-white/40 text-sm">scroll</span>
            <svg
              className="w-5 h-5 text-white/40 animate-bounce"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObserverLanding;
