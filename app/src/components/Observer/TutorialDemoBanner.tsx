// src/components/Observer/TutorialDemoBanner.tsx
//
// While the first-run demo is in flight (tutorialFlow `building` / `monitoring`), tells the
// user where the demo tab is and what to share, since the demo runs outside Observer. Also
// the way back to the tab if the browser blocked it or the user closed it.

import React from 'react';
import { X } from 'lucide-react';
import { useTutorialFlow, tutorialFlow } from '@utils/tutorialFlow';
import { demoTabTitle, demoUrl } from '@utils/tutorialDemos';
import { isTauri, openExternal } from '@utils/platform';

const TutorialDemoBanner: React.FC = () => {
  const { phase, demo } = useTutorialFlow();
  if (!demo || (phase !== 'building' && phase !== 'monitoring')) return null;

  const body = phase === 'monitoring' ? (
    <><strong>Observer is watching the demo.</strong> You'll get your notification when it finishes.</>
  ) : isTauri() ? (
    <><strong>Your demo is open in your browser.</strong> Keep it visible on screen; Observer will watch that window.</>
  ) : (
    <><strong>Your demo is open in a new tab.</strong> When Observer asks to share your screen, choose <strong>Chrome Tab → "{demoTabTitle(demo)}"</strong>.</>
  );

  return (
    <div className="mx-auto mt-3 w-[calc(100%-2rem)] max-w-2xl flex items-start gap-3 rounded-2xl bg-slate-900 text-white px-4 py-3 shadow-lg text-sm relative z-30">
      <div className="flex-1 min-w-0">
        <p>{body}</p>
        <button onClick={() => void openExternal(demoUrl(demo))} className="mt-1 text-xs text-white/60 hover:text-white underline underline-offset-2 transition-colors">
          Tab didn't open? Open demo
        </button>
      </div>
      <button onClick={tutorialFlow.reset} className="text-white/50 hover:text-white transition-colors flex-shrink-0" aria-label="Dismiss">
        <X className="h-4 w-4" />
      </button>
    </div>
  );
};

export default TutorialDemoBanner;
