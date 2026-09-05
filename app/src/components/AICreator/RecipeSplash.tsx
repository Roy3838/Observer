// src/components/AICreator/RecipeSplash.tsx
//
// The cosmic onboarding "splash": a full-screen blurred portal shown right after ToS.
// Minimal, modern, space-y. The whole builder is one sentence —
//   "When [wheel]  then [wheel]"
// — where each choice is an auto-cycling OptionWheel.
//
// "Build it" is NEVER disabled. The wheels always display a valid trigger/action pair, so
// whatever is on screen is a real, buildable agent — a user who touches nothing and clicks
// straight through gets exactly what they were looking at. This is deliberate: the old
// version gated the button on picking a trigger AND filling in a contact field, which read
// as a broken grey button and pushed people into "Skip for now". Contact info is now
// collected downstream by the MCP's `ask_user_info` tool, which pops a guided modal at the
// moment the value is actually needed.
//
// A single "master edit" ✏️ swaps the wheels for a textarea of the actual composed MCP
// message (with an × to revert), so power users can tweak the prompt directly.
//
// "Build it" composes / takes that message and one-shots it into the MCP via
// useMCPContext().send(), then closes — so the user watches it build.

import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { X, ArrowRight } from 'lucide-react';
import { useMCPContext } from '../../mcp/MCPContext';
import { useAuth } from '@contexts/AuthContext';
import { SensorSettings } from '@utils/settings';
import { Analytics } from '@utils/analytics';
import type { WhitelistChannel } from '@utils/logging';
import EditableWheel from './EditableWheel';
import type { WheelOption } from './OptionWheel';

export type ContactKind = 'phone' | 'email' | 'telegram' | 'discord' | 'none';

export interface TriggerOption extends WheelOption {
  sensor: '$SCREEN' | '$CAMERA';
  promptFragment: string;
}

export interface ActionOption extends WheelOption {
  contact: ContactKind;
  actionFragment: string;
  /** For phone contacts: which whitelist QR to show. */
  channel?: WhitelistChannel;
}

export const TRIGGERS: TriggerOption[] = [
  { id: 'download_done',   label: 'my download is finished',    sensor: '$SCREEN', promptFragment: 'my download finishes' },
  { id: 'person_camera',   label: 'a person is on camera',      sensor: '$CAMERA', promptFragment: 'a person appears on my camera' },
  { id: 'render_fails',    label: 'my render fails',            sensor: '$SCREEN', promptFragment: 'my render fails or errors out' },
  { id: 'minecraft_death', label: 'my Minecraft character dies', sensor: '$SCREEN', promptFragment: 'my Minecraft character dies' },
  { id: 'steam_ready',     label: 'my Steam game is ready',     sensor: '$SCREEN', promptFragment: 'my Steam game finishes downloading and is ready to play' },
  { id: 'distracted',      label: "I'm distracted",             sensor: '$SCREEN', promptFragment: 'I get distracted (e.g. social media or YouTube on screen)' },
  { id: 'printer_fails',   label: 'my 3D printer fails',        sensor: '$CAMERA', promptFragment: 'my 3D print fails (spaghetti, detached print, or a clog)' },
  { id: 'battery_low',     label: 'my battery is low',          sensor: '$SCREEN', promptFragment: 'my battery indicator shows low battery' },
  { id: 'hour_passes',     label: 'an hour passes',             sensor: '$SCREEN', promptFragment: 'an hour passes (log what is on my screen)' },
  { id: 'tickets_available', label: 'tickets are available',    sensor: '$SCREEN', promptFragment: 'tickets become available on the page I have open' },
];

export const ACTIONS: ActionOption[] = [
  { id: 'email',    label: 'send me an email',    contact: 'email',    actionFragment: 'send me an email' },
  { id: 'whatsapp', label: 'send me a WhatsApp',  contact: 'phone',    actionFragment: 'send me a WhatsApp message', channel: 'whatsapp' },
  { id: 'sms',      label: 'send me an SMS',      contact: 'phone',    actionFragment: 'send me an SMS', channel: 'sms' },
  { id: 'call',     label: 'call me',             contact: 'phone',    actionFragment: 'call me with a phone call', channel: 'voice' },
  { id: 'telegram', label: 'send me a Telegram',  contact: 'telegram', actionFragment: 'send me a Telegram message' },
  { id: 'discord',  label: 'ping my Discord',     contact: 'discord',  actionFragment: 'send a message to my Discord' },
  { id: 'log',      label: 'log it',              contact: 'none',     actionFragment: 'log it to memory' },
];

/**
 * Composes the one-sentence MCP prompt from a trigger/action combo.
 *
 * Deliberately emits the notification *intent* only ("send me an SMS") and never a phone
 * number / chat_id / webhook: the MCP collects those itself via `ask_user_info`, which can
 * also guide the user through obtaining them. Email is the one exception — the Auth0
 * address needs no user input, so it's inlined here.
 */
export type ModelMode = 'cloud' | 'local';

export function composeRecipePrompt(
  trigger: TriggerOption | undefined,
  action: ActionOption | undefined,
  authEmail: string,
  mode: ModelMode = 'cloud',
): string {
  const sensor = trigger?.sensor ?? '$SCREEN';
  const watchWhat = sensor === '$CAMERA' ? 'my camera' : 'my screen';
  const triggerFrag = trigger?.promptFragment ?? '';
  const actionFrag = action?.actionFragment ?? '';
  const phrase = action?.contact === 'email' && authEmail ? ` at ${authEmail}` : '';
  const modelClause = mode === 'local' ? 'Use a local model.' : 'Use a cloud model.';
  return `Watch ${watchWhat}. When ${triggerFrag}, ${actionFrag}${phrase}. ${modelClause}`;
}

interface RecipeSplashProps {
  isOpen: boolean;
  onClose: () => void;
}

const RecipeSplash: React.FC<RecipeSplashProps> = ({ isOpen, onClose }) => {
  const { send } = useMCPContext();
  const { user } = useAuth();
  const authEmail = user?.email ?? '';

  const [triggerId, setTriggerId] = useState(TRIGGERS[0].id);
  const [actionId, setActionId] = useState(ACTIONS[0].id);
  const [modelMode, setModelMode] = useState<ModelMode>('cloud');
  // Free text typed onto a preset's row, keyed by that preset's id. Overwrites the row's
  // label/fragment in place — spinning the wheel carries it along like any other row, and
  // spinning back to that id later still shows the edited text (until edited again).
  const [triggerOverrides, setTriggerOverrides] = useState<Record<string, string>>({});
  const [actionOverrides, setActionOverrides] = useState<Record<string, string>>({});
  // Purely cosmetic now: drives the "Spin to pick" tooltip. Does not gate Build it.
  const [triggerChosen, setTriggerChosen] = useState(false);
  // Settle the wheels as the pointer reaches "Build it". commit() runs on transitionend, so
  // a click landing mid-glide would otherwise build the row BEFORE the one on screen.
  const [aiming, setAiming] = useState(false);

  useEffect(() => { if (isOpen) Analytics.recipeShown(); }, [isOpen]);

  const triggerOptions = useMemo(
    () => TRIGGERS.map(t => triggerOverrides[t.id]
      ? { ...t, label: triggerOverrides[t.id], promptFragment: triggerOverrides[t.id] }
      : t),
    [triggerOverrides],
  );
  const actionOptions = useMemo(
    () => ACTIONS.map(a => actionOverrides[a.id]
      ? { ...a, label: actionOverrides[a.id], actionFragment: actionOverrides[a.id] }
      : a),
    [actionOverrides],
  );

  const trigger = useMemo(() => triggerOptions.find(t => t.id === triggerId), [triggerOptions, triggerId]);
  const action = useMemo(() => actionOptions.find(a => a.id === actionId), [actionOptions, actionId]);

  const composePrompt = (): string => composeRecipePrompt(trigger, action, authEmail, modelMode);

  // No gating, by design. A slot always holds a valid trigger/action pair — a preset or
  // an edited-in-place one — so there is nothing left to wait for.
  const handleBuild = () => {
    const prompt = composePrompt();
    Analytics.recipeBuilt(triggerOverrides[triggerId] ? 'custom' : triggerId, actionOverrides[actionId] ? 'custom' : actionId, modelMode);
    // First-run agents should just run — nobody's here yet to click through tool
    // confirmations, so building from the splash turns on yolo mode.
    SensorSettings.setMcpYoloMode(true);
    send(prompt);
    onClose();
  };

  if (!isOpen) return null;

  // Compact so it sits inline next to the header title on every breakpoint, rather than
  // stacking below it (mobile has enough vertical stuff going on already).
  const modelToggle = (
    <div className="relative inline-flex items-center rounded-full bg-white/10 border border-white/15 p-0.5 backdrop-blur-sm shadow-inner shrink-0">
      <div
        className="absolute top-0.5 bottom-0.5 w-10 md:w-12 rounded-full bg-white shadow-[0_0_20px_-4px_rgba(255,255,255,0.7)] transition-transform duration-300 ease-out"
        style={{ transform: modelMode === 'cloud' ? 'translateX(0%)' : 'translateX(100%)' }}
      />
      <button
        type="button"
        onClick={() => setModelMode('cloud')}
        className={`relative z-10 w-10 md:w-12 py-1 text-[9px] md:text-xs font-semibold rounded-full transition-colors duration-300 ${
          modelMode === 'cloud' ? 'text-slate-900' : 'text-white/60 hover:text-white/80'
        }`}
      >
        Cloud
      </button>
      <button
        type="button"
        onClick={() => setModelMode('local')}
        className={`relative z-10 w-10 md:w-12 py-1 text-[9px] md:text-xs font-semibold rounded-full transition-colors duration-300 ${
          modelMode === 'local' ? 'text-slate-900' : 'text-white/60 hover:text-white/80'
        }`}
      >
        Local
      </button>
    </div>
  );

  return createPortal(
    <div className="fixed inset-0 z-[10000] bg-slate-950/70 backdrop-blur-md font-golos flex flex-col items-center justify-center p-4">
      {/* Close */}
      <button
        onClick={onClose}
        style={{ top: 'calc(1rem + env(safe-area-inset-top))' }}
        className="absolute right-4 text-white/50 hover:text-white transition-colors"
        aria-label="Close"
      >
        <X className="h-6 w-6" />
      </button>

      {/* Header — title (free to wrap, so it never fights the close X) with the
          Local/Cloud toggle right beneath it. Title runs smaller on mobile so the
          pair stays compact and doesn't crowd the wheels below. */}
      <div
        style={{ top: 'calc(1.25rem + env(safe-area-inset-top))' }}
        className="absolute left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5 select-none px-12"
      >
        <p className="text-white/50 text-[10px] md:text-sm tracking-[0.2em] md:tracking-[0.3em] uppercase text-center">
          Welcome to Observer!   Build your first agent:
        </p>
        {modelToggle}
      </div>

      {/* Centered builder: "When [wheel]  then [wheel]" — click the row itself to type. */}
      <div className="w-full max-w-6xl flex items-center justify-center">
        <div className="flex flex-col md:flex-row md:flex-nowrap items-center justify-center gap-x-3 gap-y-4">
          <div className="relative">
            <span className="text-3xl md:text-6xl font-bold text-white tracking-tight select-none pointer-events-none">When</span>
            {/* Mobile (stacked layout): anchor to "When" itself, which is already
                centered correctly — simpler than chasing the wheel's own offset. */}
            {!triggerChosen && (
              <div className="md:hidden absolute -top-11 left-1/2 -translate-x-1/2 select-none pointer-events-none z-10">
                <div className="flex flex-col items-center animate-bounce">
                  <span className="whitespace-nowrap text-xs font-semibold text-slate-900 bg-white rounded-full px-4 py-1.5 shadow-[0_0_20px_-4px_rgba(255,255,255,0.7)]">
                    Spin to pick what to detect
                  </span>
                  <div className="w-2.5 h-2.5 bg-white rotate-45 -mt-1.5" />
                </div>
              </div>
            )}
          </div>
          <div>
            <EditableWheel
              options={triggerOptions}
              value={triggerId}
              onChange={setTriggerId}
              onCustom={text => setTriggerOverrides(prev => ({ ...prev, [triggerId]: text }))}
              onInteract={() => setTriggerChosen(true)}
              paused={aiming}
              ariaLabel="Choose a trigger"
              widthClass="w-[15rem] md:w-[19rem]"
              tooltip={!triggerChosen && (
                <div className="hidden md:block absolute -top-11 left-1/2 -translate-x-1/2 select-none pointer-events-none z-10">
                  <div className="flex flex-col items-center animate-bounce">
                    <span className="whitespace-nowrap text-sm font-semibold text-slate-900 bg-white rounded-full px-4 py-1.5 shadow-[0_0_20px_-4px_rgba(255,255,255,0.7)]">
                      Spin to pick what to detect
                    </span>
                    <div className="w-2.5 h-2.5 bg-white rotate-45 -mt-1.5" />
                  </div>
                </div>
              )}
            />
          </div>
          <span className="text-3xl md:text-6xl font-bold text-white tracking-tight select-none pointer-events-none">then</span>
          <EditableWheel
            options={actionOptions}
            value={actionId}
            onChange={setActionId}
            onCustom={text => setActionOverrides(prev => ({ ...prev, [actionId]: text }))}
            paused={aiming}
            ariaLabel="Choose an action"
            widthClass="w-[15rem] md:w-[13rem]"
          />
        </div>
      </div>

      {/* Bottom cluster — Build it (always live, always pinned) */}
      <div className="absolute bottom-0 inset-x-0 flex flex-col items-center gap-3 pb-8 px-4">
        {/* Never disabled: whatever the wheels show is buildable. */}
        <button
          onClick={handleBuild}
          // Pointer-down covers touch, where there's no hover to settle the wheels on.
          onPointerDown={() => setAiming(true)}
          onMouseEnter={() => setAiming(true)}
          onMouseLeave={() => setAiming(false)}
          onFocus={() => setAiming(true)}
          onBlur={() => setAiming(false)}
          className="inline-flex items-center gap-3 px-10 py-4 rounded-full bg-white text-slate-900 font-bold text-xl md:text-2xl shadow-[0_0_40px_-8px_rgba(255,255,255,0.6)] hover:shadow-[0_0_60px_-6px_rgba(255,255,255,0.8)] hover:scale-[1.02] transition-all"
        >
          Build it
          <ArrowRight className="h-6 w-6" />
        </button>

        {/* The exit, not the alternative — deliberately quieter than the primary action. */}
        <button onClick={onClose} className="text-white/30 hover:text-white/60 text-xs transition-colors">
          Skip for now
        </button>
      </div>
    </div>,
    document.body,
  );
};

export default RecipeSplash;
