// src/components/AICreator/RecipeSplash.tsx
//
// The cosmic onboarding "splash": a full-screen blurred portal shown right after ToS.
// Minimal, modern, space-y. The whole builder is one sentence —
//   "When [wheel]  then [wheel]"
// — where each choice is an auto-cycling OptionWheel. Picking a notification channel
// reveals a Message-setup block (QR for phone, chat_id for Telegram, webhook for
// Discord). Email silently uses the Auth0 user's email (the backend only sends from it).
//
// A single "master edit" ✏️ swaps the wheels for a textarea of the actual composed MCP
// message (with an × to revert), so power users can tweak the prompt directly.
//
// "Build it" composes / takes that message and one-shots it into the MCP via
// useMCPContext().send(), then closes — so the user watches it build.

import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { X, ArrowRight, Pencil } from 'lucide-react';
import { useMCPContext } from '../../mcp/MCPContext';
import { useAuth } from '@contexts/AuthContext';
import { Analytics } from '@utils/analytics';
import type { WhitelistChannel } from '@utils/logging';
import WhitelistInline from '@components/whitelist/WhitelistInline';
import OptionWheel, { type WheelOption } from './OptionWheel';

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

export const CONTACT_PLACEHOLDER: Record<ContactKind, string> = {
  phone: '+1 555 123 4567',
  email: 'you@email.com',
  telegram: 'Telegram chat_id',
  discord: 'https://discord.com/api/webhooks/…',
  none: '',
};

export function contactValid(kind: ContactKind, value: string): boolean {
  const v = value.trim();
  switch (kind) {
    case 'none':
    case 'email': return true; // email uses the Auth0 address; no input to validate
    case 'phone': return /^\+?[0-9][0-9\s()-]{6,}$/.test(v);
    case 'telegram': return v.length > 0;
    case 'discord': return /^https?:\/\/.+/.test(v);
  }
}

/** Composes the one-sentence MCP prompt from a trigger/action/contact combo. Shared by
 *  RecipeSplash (full onboarding) and RecipeMini (inline builder in the chat suggestions). */
export function composeRecipePrompt(
  trigger: TriggerOption | undefined,
  action: ActionOption | undefined,
  contact: string,
  authEmail: string,
): string {
  const sensor = trigger?.sensor ?? '$SCREEN';
  const watchWhat = sensor === '$CAMERA' ? 'my camera' : 'my screen';
  const triggerFrag = trigger?.promptFragment ?? '';
  const actionFrag = action?.actionFragment ?? '';
  const contactKind = action?.contact ?? 'none';
  const v = contact.trim();
  let phrase = '';
  if (contactKind === 'phone') phrase = ` at ${v}`;
  else if (contactKind === 'email') phrase = authEmail ? ` at ${authEmail}` : '';
  else if (contactKind === 'telegram') phrase = ` to Telegram chat_id ${v}`;
  else if (contactKind === 'discord') phrase = ` via the Discord webhook ${v}`;
  return `Watch ${watchWhat}. When ${triggerFrag}, ${actionFrag}${phrase}. Use a cloud model.`;
}

interface RecipeSplashProps {
  isOpen: boolean;
  onClose: () => void;
}

const RecipeSplash: React.FC<RecipeSplashProps> = ({ isOpen, onClose }) => {
  const { send } = useMCPContext();
  const { user, getAccessToken } = useAuth();
  const authEmail = user?.email ?? '';

  const [triggerId, setTriggerId] = useState(TRIGGERS[0].id);
  const [actionId, setActionId] = useState(ACTIONS[0].id);
  const [triggerChosen, setTriggerChosen] = useState(false); // gates Build it until user picks a trigger
  const [actionChosen, setActionChosen] = useState(false); // gates the Message-setup block
  const [contact, setContact] = useState('');
  const [phoneVerified, setPhoneVerified] = useState(false);
  const [editingMessage, setEditingMessage] = useState(false);
  const [messageDraft, setMessageDraft] = useState('');

  useEffect(() => { if (isOpen) Analytics.recipeShown(); }, [isOpen]);

  // A changed number (or action) invalidates any prior verification.
  useEffect(() => { setPhoneVerified(false); }, [contact, actionId]);

  const trigger = useMemo(() => TRIGGERS.find(t => t.id === triggerId), [triggerId]);
  const action = useMemo(() => ACTIONS.find(a => a.id === actionId), [actionId]);
  const contactKind: ContactKind = action?.contact ?? 'none';
  const showSetup = !editingMessage && actionChosen &&
    (contactKind === 'phone' || contactKind === 'telegram' || contactKind === 'discord');

  const composePrompt = (): string => composeRecipePrompt(trigger, action, contact, authEmail);

  const openEditor = () => { setMessageDraft(composePrompt()); setEditingMessage(true); };
  const revertEditor = () => setEditingMessage(false);

  // Phone channels must be whitelisted before building; other channels just need a value.
  const needsPhone = !editingMessage && contactKind === 'phone';
  const canBuild = editingMessage
    ? messageDraft.trim().length > 0
    : !triggerChosen
      ? false
      : needsPhone
        ? contactValid('phone', contact) && phoneVerified
        : contactValid(contactKind, contact);

  const handleBuild = () => {
    if (!canBuild) return;
    const prompt = editingMessage ? messageDraft.trim() : composePrompt();
    Analytics.recipeBuilt(editingMessage ? 'custom' : triggerId, editingMessage ? 'custom' : actionId);
    send(prompt);
    onClose();
  };

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[10000] bg-slate-950/70 backdrop-blur-md font-golos flex flex-col items-center justify-center p-4">
      {/* Close */}
      <button onClick={onClose} className="absolute top-4 right-4 text-white/50 hover:text-white transition-colors" aria-label="Close">
        <X className="h-6 w-6" />
      </button>

      {/* Header label */}
      <p className="absolute top-6 left-1/2 -translate-x-1/2 text-white/50 text-xs md:text-sm tracking-[0.3em] uppercase select-none">
        Welcome to Observer!   Build your first agent:
      </p>

      {/* Centered builder (wheels OR the master-edit textarea) */}
      <div className="w-full max-w-6xl flex items-center justify-center">
        {editingMessage ? (
          <div className="w-full max-w-2xl relative">
            <textarea
              value={messageDraft}
              onChange={e => setMessageDraft(e.target.value)}
              rows={4}
              autoFocus
              className="w-full bg-white/5 border border-white/20 rounded-2xl p-5 pr-12 text-white text-lg md:text-xl leading-relaxed focus:outline-none focus:border-white/50 resize-none"
            />
            <button
              onClick={revertEditor}
              className="absolute top-3 right-3 text-white/50 hover:text-white transition-colors"
              title="Back to the slots"
              aria-label="Back to the slots"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col md:flex-row md:flex-nowrap items-center justify-center gap-x-3 gap-y-3">
            <div className="relative">
              <span className="text-4xl md:text-6xl font-bold text-white tracking-tight select-none pointer-events-none">When</span>
              {/* Mobile (stacked layout): anchor to "When" itself, which is already
                  centered correctly — simpler than chasing the wheel's own offset. */}
              {!triggerChosen && (
                <div className="md:hidden absolute -top-11 left-1/2 -translate-x-1/2 select-none pointer-events-none z-10">
                  <div className="flex flex-col items-center animate-bounce">
                    <span className="whitespace-nowrap text-xs font-semibold text-slate-900 bg-white rounded-full px-4 py-1.5 shadow-[0_0_20px_-4px_rgba(255,255,255,0.7)]">
                      Select what you want to detect
                    </span>
                    <div className="w-2.5 h-2.5 bg-white rotate-45 -mt-1.5" />
                  </div>
                </div>
              )}
            </div>
            <div>
              <OptionWheel
                options={TRIGGERS}
                value={triggerId}
                onChange={setTriggerId}
                onInteract={() => setTriggerChosen(true)}
                ariaLabel="Choose a trigger"
                widthClass="w-[15rem] md:w-[19rem]"
                tooltip={!triggerChosen && (
                  <div className="hidden md:block absolute -top-11 left-1/2 -translate-x-1/2 select-none pointer-events-none z-10">
                    <div className="flex flex-col items-center animate-bounce">
                      <span className="whitespace-nowrap text-sm font-semibold text-slate-900 bg-white rounded-full px-4 py-1.5 shadow-[0_0_20px_-4px_rgba(255,255,255,0.7)]">
                        Select what you want to detect
                      </span>
                      <div className="w-2.5 h-2.5 bg-white rotate-45 -mt-1.5" />
                    </div>
                  </div>
                )}
              />
            </div>
            <span className="text-4xl md:text-6xl font-bold text-white tracking-tight select-none pointer-events-none">then</span>
            <OptionWheel
              options={ACTIONS}
              value={actionId}
              onChange={setActionId}
              onInteract={() => setActionChosen(true)}
              ariaLabel="Choose an action"
              widthClass="w-[11rem] md:w-[13rem]"
            />
            <button
              onClick={openEditor}
              title="Edit the full message"
              aria-label="Edit the full message"
              className="p-2 rounded-full text-white/40 hover:text-white hover:bg-white/10 transition-colors md:self-center"
            >
              <Pencil className="h-5 w-5" />
            </button>
          </div>
        )}
      </div>

      {/* Bottom cluster — message setup + edit toggle + Build it (always pinned) */}
      <div className="absolute bottom-0 inset-x-0 flex flex-col items-center gap-4 pb-8 px-4">
        {showSetup && (
          <div className="w-full max-w-md rounded-2xl border border-slate-700 bg-slate-800 p-4 md:p-5 text-left shadow-xl">
            <p className="text-white/70 text-sm font-medium mb-3">Message setup</p>
            <input
              type={contactKind === 'phone' ? 'tel' : 'text'}
              value={contact}
              onChange={e => setContact(e.target.value)}
              placeholder={CONTACT_PLACEHOLDER[contactKind]}
              className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-2.5 text-white placeholder-white/30 focus:outline-none focus:border-white/50 transition-colors"
            />
            {contactKind === 'phone' && action?.channel && contactValid('phone', contact) && (
              <div className="mt-3">
                <WhitelistInline
                  phoneNumber={contact.trim()}
                  channel={action.channel}
                  getToken={getAccessToken}
                  onWhitelisted={() => setPhoneVerified(true)}
                />
              </div>
            )}
            {contactKind === 'telegram' && (
              <p className="text-xs text-white/50 mt-2">
                Message <span className="font-mono text-white/70">@observer_notification_bot</span> to get your chat_id.
              </p>
            )}
            {contactKind === 'discord' && (
              <p className="text-xs text-white/50 mt-2">
                In Discord: Server Settings → Integrations → Webhooks → New Webhook → Copy URL.
              </p>
            )}
          </div>
        )}

        <button
          onClick={handleBuild}
          disabled={!canBuild}
          className="inline-flex items-center gap-3 px-10 py-4 rounded-full bg-white text-slate-900 font-bold text-xl md:text-2xl shadow-[0_0_40px_-8px_rgba(255,255,255,0.6)] hover:shadow-[0_0_60px_-6px_rgba(255,255,255,0.8)] hover:scale-[1.02] transition-all disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none disabled:hover:scale-100"
        >
          Build it
          <ArrowRight className="h-6 w-6" />
        </button>

        <button onClick={onClose} className="text-white/40 hover:text-white/70 text-sm transition-colors">
          Skip for now
        </button>
      </div>
    </div>,
    document.body,
  );
};

export default RecipeSplash;
