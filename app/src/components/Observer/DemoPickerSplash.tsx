// src/components/Observer/DemoPickerSplash.tsx
//
// The first-run tutorial: RecipeSplash's full-screen "When [wheel] then [wheel]" sentence, but
// the trigger wheel picks one of the e2e demo pages (tutorialDemos.ts), with a big preview of
// the one in focus. "Open demo" opens that demo in a new tab, where it runs on its own, and
// sends the composed prompt to the MCP — from there Observer runs exactly as it would for any
// real screen, share picker included. TutorialDemoBanner tells the user which tab to share.

import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { X, ArrowRight } from 'lucide-react';
import { useMCPContext } from '../../mcp/MCPContext';
import { useAuth } from '@contexts/AuthContext';
import { Analytics } from '@utils/analytics';
import { openExternal } from '@utils/platform';
import { tutorialFlow } from '@utils/tutorialFlow';
import { TUTORIAL_DEMOS, demoTabTitle, demoUrl } from '@utils/tutorialDemos';
import OptionWheel from '../AICreator/OptionWheel';
import { ACTIONS, composeRecipePrompt } from '../AICreator/RecipeSplash';
import { isMobile, isWeb } from '@utils/platform';

// The tutorial has the user share a demo tab, which phones can't do: not the mobile apps, and
// not mobile browsers (no getDisplayMedia). Desktop apps capture natively, so they're fine.
export const TUTORIAL_SUPPORTED = !isMobile() && !(isWeb() && !navigator.mediaDevices?.getDisplayMedia);

// Previews are full-size screenshots, so rather than download all of them up front, fetch the
// ones this many rows either side of the wheel's focus. The wheel loops and auto-cycles one row
// every ~2 s, so the next preview is nearly always cached before it reaches the middle.
const PREFETCH_ROWS = 3;

// "Log it" writes memory, which agents also do on every loop, so the tutorial can't tell it
// apart from the moment it's waiting for.
const DEMO_ACTIONS = ACTIONS.filter(a => a.contact !== 'none');

interface DemoPickerSplashProps {
  isOpen: boolean;
  /** `started` is false when the user skipped or closed it. */
  onClose: (started: boolean) => void;
}

const DemoPickerSplash: React.FC<DemoPickerSplashProps> = ({ isOpen, onClose }) => {
  const { send } = useMCPContext();
  const { user } = useAuth();
  const authEmail = user?.email ?? '';

  const [demoId, setDemoId] = useState(TUTORIAL_DEMOS[0].id);
  const [actionId, setActionId] = useState(DEMO_ACTIONS[0].id);
  const [demoChosen, setDemoChosen] = useState(false);
  // Settle the wheels as the pointer reaches the button (see RecipeSplash): a click landing
  // mid-glide would otherwise open the demo BEFORE the one on screen.
  const [aiming, setAiming] = useState(false);

  // Each opening starts from the top of the list, auto-cycling again.
  useEffect(() => {
    if (!isOpen) return;
    setDemoId(TUTORIAL_DEMOS[0].id);
    setActionId(DEMO_ACTIONS[0].id);
    setDemoChosen(false);
    setAiming(false);
  }, [isOpen]);

  const demo = useMemo(() => TUTORIAL_DEMOS.find(d => d.id === demoId)!, [demoId]);
  const action = useMemo(() => DEMO_ACTIONS.find(a => a.id === actionId)!, [actionId]);

  useEffect(() => {
    if (!isOpen) return;
    const n = TUTORIAL_DEMOS.length;
    const i = TUTORIAL_DEMOS.findIndex(d => d.id === demoId);
    for (let d = -PREFETCH_ROWS; d <= PREFETCH_ROWS; d++) {
      new Image().src = TUTORIAL_DEMOS[(((i + d) % n) + n) % n].preview;
    }
  }, [isOpen, demoId]);

  const handleOpen = () => {
    // First, while still inside the click: a popup blocker only allows window.open here.
    void openExternal(demoUrl(demo));
    Analytics.tutorialStarted(demo.id, action.id);
    const prompt = `${composeRecipePrompt(demo, action, authEmail, 'cloud')} It's open in my browser in a tab titled "${demoTabTitle(demo)}".`;
    tutorialFlow.start(demo);
    onClose(true);
    void send(prompt);
  };

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[10000] bg-slate-950/70 backdrop-blur-md font-golos overflow-y-auto">
      {/* min-h-full + justify-center (rather than centering the scroller itself) so content
          taller than a phone screen scrolls instead of being clipped at the top. */}
      <div className="min-h-full flex flex-col items-center justify-center gap-5 md:gap-8 px-4 py-12">
        <button
          onClick={() => onClose(false)}
          style={{ top: 'calc(1rem + env(safe-area-inset-top))' }}
          className="absolute right-4 text-white/50 hover:text-white transition-colors"
          aria-label="Close"
        >
          <X className="h-6 w-6" />
        </button>

        <div className="flex flex-col items-center gap-2 select-none px-10 text-center">
          <p className="text-white/50 text-[10px] md:text-sm tracking-[0.2em] md:tracking-[0.3em] uppercase">
            Welcome to Observer!
          </p>
          <h1 className="text-white text-2xl md:text-4xl font-bold tracking-tight">
            Which demo would you like to try?
          </h1>
          <p className="text-white/50 text-xs md:text-base max-w-md">
            I'll open it in a new tab. You share that tab with me, and I'll tell you when it's done.
          </p>
        </div>

        {/* The demo in focus, big. Follows the wheel as it spins. */}
        <div className="w-[min(32rem,85vw,30vh)] md:w-[min(32rem,85vw,48vh)] aspect-[16/10] rounded-xl overflow-hidden ring-1 ring-white/20 shadow-[0_0_60px_-12px_rgba(255,255,255,0.35)] bg-slate-900">
          <img key={demo.id} src={demo.preview} alt={demo.title} className="w-full h-full object-cover object-top animate-fade-in" />
        </div>

        <div className="flex flex-col md:flex-row md:flex-nowrap items-center justify-center gap-x-3 gap-y-2">
          <span className="text-2xl md:text-5xl font-bold text-white tracking-tight select-none pointer-events-none">When</span>
          <OptionWheel
            options={TUTORIAL_DEMOS}
            value={demoId}
            onChange={setDemoId}
            onInteract={() => setDemoChosen(true)}
            paused={aiming}
            ariaLabel="Choose a demo"
            widthClass="w-[18rem] md:w-[26rem]"
            textClass="text-sm md:text-lg"
            tooltip={!demoChosen && (
              <div className="hidden md:block absolute -top-9 left-1/2 -translate-x-1/2 select-none pointer-events-none z-10">
                <div className="flex flex-col items-center animate-bounce">
                  <span className="whitespace-nowrap text-xs md:text-sm font-semibold text-slate-900 bg-white rounded-full px-4 py-1.5 shadow-[0_0_20px_-4px_rgba(255,255,255,0.7)]">
                    Spin to pick a demo
                  </span>
                  <div className="w-2.5 h-2.5 bg-white rotate-45 -mt-1.5" />
                </div>
              </div>
            )}
          />
          <span className="text-2xl md:text-5xl font-bold text-white tracking-tight select-none pointer-events-none">then</span>
          <OptionWheel
            options={DEMO_ACTIONS}
            value={actionId}
            onChange={setActionId}
            autoCycle={false}
            paused={aiming}
            ariaLabel="Choose an action"
            widthClass="w-[14rem] md:w-[13rem]"
            textClass="text-sm md:text-lg"
          />
        </div>

        <div className="flex flex-col items-center gap-3">
          <button
            onClick={handleOpen}
            onPointerDown={() => setAiming(true)}
            onMouseEnter={() => setAiming(true)}
            onMouseLeave={() => setAiming(false)}
            onFocus={() => setAiming(true)}
            onBlur={() => setAiming(false)}
            className="inline-flex items-center gap-3 px-10 py-4 rounded-full bg-white text-slate-900 font-bold text-xl md:text-2xl shadow-[0_0_40px_-8px_rgba(255,255,255,0.6)] hover:shadow-[0_0_60px_-6px_rgba(255,255,255,0.8)] hover:scale-[1.02] transition-all"
          >
            Open demo
            <ArrowRight className="h-6 w-6" />
          </button>
          <button onClick={() => onClose(false)} className="text-white/30 hover:text-white/60 text-xs transition-colors">
            Skip for now
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
};

export default DemoPickerSplash;
