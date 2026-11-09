// src/components/TutorialDoneModal.tsx
//
// Shown when the first-run demo's notification lands: a short celebration, no upsell.
// Same sheet look as the ask_user_info modals (UserInfoModal.tsx).

import React, { useEffect } from 'react';
import ReactDOM from 'react-dom';
import { X, ArrowRight } from 'lucide-react';

interface TutorialDoneModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const TutorialDoneModal: React.FC<TutorialDoneModalProps> = ({ isOpen, onClose }) => {
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return ReactDOM.createPortal(
    <div
      className="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center bg-zinc-950/30 backdrop-blur-[2px] animate-sheet-fade"
      onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Notification sent"
        className="relative w-full sm:max-w-md sm:mx-4 max-h-[92vh] overflow-y-auto rounded-t-[28px] sm:rounded-[28px] bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 shadow-[0_24px_80px_-16px_rgba(0,0,0,0.35)] ring-1 ring-black/5 dark:ring-white/10 animate-sheet-in"
        onMouseDown={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-end px-5 pt-5">
          <button
            onClick={onClose}
            aria-label="Close"
            className="flex items-center justify-center h-8 w-8 rounded-full text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 dark:hover:text-zinc-200 dark:hover:bg-zinc-800 transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-7 pb-7 -mt-2 flex flex-col items-center text-center">
          {/* Drawn check, same as the "connected" state in the ask_user_info sheets */}
          <svg viewBox="0 0 52 52" className="h-14 w-14">
            <circle cx="26" cy="26" r="24" className="fill-emerald-500/10 stroke-emerald-500" strokeWidth="2" />
            <path d="M15 27 l7 7 l15 -16" fill="none" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"
              className="stroke-emerald-500 animate-check-draw" style={{ strokeDasharray: 40, strokeDashoffset: 40 }} />
          </svg>

          <h2 className="mt-4 text-[19px] font-semibold tracking-tight">Notification sent</h2>
          <p className="mt-3 text-sm font-medium text-zinc-900 dark:text-zinc-100">Welcome to Observer</p>
          <p className="mt-1 text-sm leading-relaxed text-zinc-500 dark:text-zinc-400 max-w-xs">
            Hope it gives you back a little time every day :)
          </p>

          <button
            onClick={onClose}
            className="mt-6 w-full inline-flex items-center justify-center gap-2 h-11 rounded-full bg-zinc-900 text-white dark:bg-white dark:text-zinc-900 text-sm font-medium hover:bg-zinc-800 dark:hover:bg-zinc-200 active:scale-[0.99] transition-all"
          >
            Start exploring <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>,
    document.getElementById('modal-root') ?? document.body,
  );
};
