// ObserverEye — the "Observer is thinking" indicator. The Observer logo (ring + pupil, same
// geometry as /eye-logo-black.svg) brought to life. It plays one short "move" at a time
// (peek, blink, sleepy, hop), picking a different one each cycle so it never feels
// looped, while the whole icon breathes between 100% and 70% opacity to read as "loading".
// Keyframes live in index.css (observer-eye-*) and only use transform/opacity, so they run
// the same in WebKit and Chromium.
import React, { useEffect, useState } from 'react';

const MOVES = ['peek', 'blink', 'sleepy', 'hop'] as const;
type Move = typeof MOVES[number];

// Must match the animation duration of the .observer-eye--<move> rules in index.css.
const MOVE_MS = 2400;

const pickNext = (current?: Move): Move => {
  const options = MOVES.filter(m => m !== current);
  return options[Math.floor(Math.random() * options.length)];
};

interface ObserverEyeProps {
  className?: string;
  color?: string;
  title?: string;
}

const ObserverEye: React.FC<ObserverEyeProps> = ({ className = 'h-5 w-5', color = '#000', title = 'Observer is thinking' }) => {
  const [move, setMove] = useState<Move>(() => pickNext());
  // Bumped every cycle so the <g> remounts and the CSS animation restarts.
  const [cycle, setCycle] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => {
      setMove(m => pickNext(m));
      setCycle(c => c + 1);
    }, MOVE_MS);
    return () => window.clearInterval(id);
  }, []);

  return (
    <svg viewBox="-50 -50 100 100" fill="none" className={`observer-eye ${className}`} role="img" aria-label={title} overflow="visible">
      <title>{title}</title>
      <g key={cycle} className={`observer-eye-lid observer-eye--${move}`}>
        <circle cx="0" cy="0" r="45" stroke={color} strokeWidth="10" />
        <circle className="observer-eye-pupil" cx="0" cy="0" r="12" fill={color} />
      </g>
    </svg>
  );
};

export default ObserverEye;
