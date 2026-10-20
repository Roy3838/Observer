// src/components/AICreator/RecipeMini.tsx
//
// A decorative "recipe" chip, styled to match MCP's plain suggestion chips exactly, that
// cycles through example "When X, then Y" phrases instead of showing one static suggestion.
// It's a teaser, not a control — clicking it opens the real builder (RecipeSplash). Keeps
// separation of concerns: this component only renders text and fires onClick.

import React, { useEffect, useState } from 'react';
import { Sparkles } from 'lucide-react';
import { TRIGGERS, ACTIONS } from './RecipeSplash';

interface RecipeMiniProps {
  onClick: () => void;
}

const CYCLE_MS = 2400;

const RecipeMini: React.FC<RecipeMiniProps> = ({ onClick }) => {
  const [i, setI] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setI(n => n + 1), CYCLE_MS);
    return () => clearInterval(timer);
  }, []);

  const trigger = TRIGGERS[i % TRIGGERS.length];
  const action = ACTIONS[i % ACTIONS.length];

  return (
    <button
      type="button"
      onClick={onClick}
      title="Build an agent in one line"
      className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-purple-200 bg-purple-50 text-purple-700 hover:bg-purple-100 hover:border-purple-300 transition-colors whitespace-nowrap overflow-hidden"
    >
      <Sparkles className="w-3.5 h-3.5 shrink-0" />
      <span key={i} className="recipe-mini-fade">
        When {trigger.label}, then {action.label}
      </span>
      <style>{`
        @keyframes recipe-mini-fade-in { from { opacity: 0; } to { opacity: 1; } }
        .recipe-mini-fade { animation: recipe-mini-fade-in 300ms ease-out; }
      `}</style>
    </button>
  );
};

export default RecipeMini;
