// src/components/AICreator/UserInfoModal.tsx
//
// The modal the `ask_user_info` MCP tool blocks on: everything only the user can give, in one
// sheet with one look.
//
//   screen / camera — a button that opens the share picker / camera permission. That click is
//                     the user gesture Safari requires (an MCP tool call never is one), and it
//                     brings the user back to Observer before the picker opens.
//   method          — how the agent tells the user. A channel the user already has working
//                     resolves silently with no sheet at all; otherwise the sheet sets it up
//                     (QR for phone/Telegram, permission for browser, a field for the rest) and
//                     only returns once it works. The MCP never handles codes itself.
//   model           — local or cloud.
//
// Observer never sends to raw phone numbers: phone channels resolve to the user's 4-word code.

import React, { useCallback, useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom';
import {
  Bell, Mail, Hash, Phone, MessageSquare, Monitor, MonitorUp, Camera, Cloud, HardDrive,
  ChevronLeft, Loader2, RefreshCw, X, ArrowRight,
} from 'lucide-react';
import { FaWhatsapp, FaTelegramPlane } from 'react-icons/fa';
import WhitelistQR from '@components/whitelist/WhitelistQR';
import TelegramQR from '@components/whitelist/TelegramQR';
import { useWhitelistPolling, useTelegramStatus, checkNumber } from '@components/whitelist/shared';
import { useAuth } from '@contexts/AuthContext';
import { SensorSettings } from '@utils/settings';
import { isDesktop } from '@utils/platform';
import { captureScreen, captureCamera } from '../../mcp/capture';
import { fetchStatus } from '../../mcp/remote';
import type { NotifyMethod, UserInfoRequest, UserInfoResponse } from '../../mcp/types';
import type { PhoneChannel } from '@utils/whitelistCode';
import { CONTACT_LABEL, CONTACT_PLACEHOLDER, contactError, contactValid, normalizeContact } from '@utils/contactInfo';

interface UserInfoModalProps {
  req: UserInfoRequest;
  onResolve: (requestId: string, response: UserInfoResponse) => void;
}

type ContactMethod = 'discord' | 'pushover';
type PhoneMethod = 'whatsapp' | 'sms' | 'voice';
const isPhone = (m: NotifyMethod): m is PhoneMethod => m === 'whatsapp' || m === 'sms' || m === 'voice';

const METHODS: Record<NotifyMethod, { label: string; hint: string; icon: React.ReactNode }> = {
  browser: { label: 'Browser', hint: 'Notification in this browser', icon: <Bell className="h-5 w-5" /> },
  system: { label: 'System', hint: 'Notification on this computer', icon: <Monitor className="h-5 w-5" /> },
  whatsapp: { label: 'WhatsApp', hint: 'Message to your phone', icon: <FaWhatsapp className="h-5 w-5" /> },
  sms: { label: 'SMS', hint: 'Text to your phone', icon: <MessageSquare className="h-5 w-5" /> },
  voice: { label: 'Call', hint: 'Phone call that reads it out', icon: <Phone className="h-5 w-5" /> },
  telegram: { label: 'Telegram', hint: 'Message from the Observer bot', icon: <FaTelegramPlane className="h-5 w-5" /> },
  email: { label: 'Email', hint: 'To your inbox', icon: <Mail className="h-5 w-5" /> },
  discord: { label: 'Discord', hint: 'Post to a channel', icon: <Hash className="h-5 w-5" /> },
  pushover: { label: 'Pushover', hint: 'Push to the Pushover app', icon: <Bell className="h-5 w-5" /> },
};

// ---- Shell --------------------------------------------------------------------------------

/** The one card every kind uses: centered on desktop, a bottom sheet on phones. */
const Sheet: React.FC<{
  icon: React.ReactNode;
  title: string;
  subtitle?: string;
  onClose: () => void;
  onBack?: () => void;
  children: React.ReactNode;
}> = ({ icon, title, subtitle, onClose, onBack, children }) => {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return ReactDOM.createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-end sm:items-center justify-center bg-zinc-950/30 backdrop-blur-[2px] animate-sheet-fade"
      onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="relative w-full sm:max-w-md sm:mx-4 max-h-[92vh] overflow-y-auto rounded-t-[28px] sm:rounded-[28px] bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 shadow-[0_24px_80px_-16px_rgba(0,0,0,0.35)] ring-1 ring-black/5 dark:ring-white/10 animate-sheet-in"
        onMouseDown={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 pt-5">
          {onBack ? (
            <IconButton label="Back" onClick={onBack}><ChevronLeft className="h-4 w-4" /></IconButton>
          ) : <span className="h-8 w-8" />}
          <IconButton label="Close" onClick={onClose}><X className="h-4 w-4" /></IconButton>
        </div>
        <div className="px-7 pb-7 -mt-2 flex flex-col items-center text-center">
          <div className="flex items-center justify-center h-12 w-12 rounded-2xl bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-100">
            {icon}
          </div>
          <h2 className="mt-4 text-[19px] font-semibold tracking-tight">{title}</h2>
          {subtitle && <p className="mt-1.5 text-sm leading-relaxed text-zinc-500 dark:text-zinc-400 max-w-xs">{subtitle}</p>}
          <div className="mt-6 w-full">{children}</div>
        </div>
      </div>
    </div>,
    document.getElementById('modal-root') ?? document.body,
  );
};

const IconButton: React.FC<{ label: string; onClick: () => void; children: React.ReactNode }> = ({ label, onClick, children }) => (
  <button
    onClick={onClick}
    aria-label={label}
    className="flex items-center justify-center h-8 w-8 rounded-full text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 dark:hover:text-zinc-200 dark:hover:bg-zinc-800 transition-colors"
  >
    {children}
  </button>
);

const PrimaryButton: React.FC<{ onClick: () => void; disabled?: boolean; busy?: boolean; children: React.ReactNode }> = ({ onClick, disabled, busy, children }) => (
  <button
    onClick={onClick}
    disabled={disabled || busy}
    className="w-full inline-flex items-center justify-center gap-2 h-11 rounded-full bg-zinc-900 text-white dark:bg-white dark:text-zinc-900 text-sm font-medium hover:bg-zinc-800 dark:hover:bg-zinc-200 active:scale-[0.99] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
  >
    {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : children}
  </button>
);

const TextButton: React.FC<{ onClick: () => void; children: React.ReactNode }> = ({ onClick, children }) => (
  <button
    onClick={onClick}
    className="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 transition-colors"
  >
    {children}
  </button>
);

const Waiting: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="flex items-center justify-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
    <span className="relative flex h-2 w-2">
      <span className="absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-60 animate-ping" />
      <span className="relative inline-flex h-2 w-2 rounded-full bg-purple-500" />
    </span>
    {children}
  </div>
);

const Spinner: React.FC<{ label: string }> = ({ label }) => (
  <div className="flex flex-col items-center gap-3 py-6 text-sm text-zinc-500 dark:text-zinc-400">
    <Loader2 className="h-5 w-5 animate-spin" />
    {label}
  </div>
);

/** Drawn check mark, then `onFinish` after a beat: nothing is left to decide. */
const Done: React.FC<{ label: string; onFinish: () => void }> = ({ label, onFinish }) => {
  const finish = useRef(onFinish);
  finish.current = onFinish;
  useEffect(() => {
    const t = setTimeout(() => finish.current(), 1200);
    return () => clearTimeout(t);
  }, []);
  return (
    <div className="flex flex-col items-center gap-3 py-4">
      <svg viewBox="0 0 52 52" className="h-14 w-14">
        <circle cx="26" cy="26" r="24" className="fill-emerald-500/10 stroke-emerald-500" strokeWidth="2" />
        <path d="M15 27 l7 7 l15 -16" fill="none" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"
          className="stroke-emerald-500 animate-check-draw" style={{ strokeDasharray: 40, strokeDashoffset: 40 }} />
      </svg>
      <p className="text-sm font-medium">{label}</p>
    </div>
  );
};

/** A small pill in the corner while a saved connection is checked: usually it's all the user sees. */
const CheckingPill: React.FC<{ label: string }> = ({ label }) => ReactDOM.createPortal(
  <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[100] flex items-center gap-2 px-4 h-9 rounded-full bg-white dark:bg-zinc-900 text-xs font-medium text-zinc-600 dark:text-zinc-300 shadow-lg ring-1 ring-black/5 dark:ring-white/10 animate-sheet-in">
    <Loader2 className="h-3.5 w-3.5 animate-spin" />
    {label}
  </div>,
  document.getElementById('modal-root') ?? document.body,
);

// ---- screen / camera ------------------------------------------------------------------------

const CaptureModal: React.FC<UserInfoModalProps> = ({ req, onResolve }) => {
  const camera = req.kind === 'camera';
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Called straight from the click, before any await, so the browser sees a user gesture.
  const share = () => {
    setBusy(true);
    setError(null);
    (camera ? captureCamera() : captureScreen()).then(result => {
      setBusy(false);
      if (result.error) setError(camera ? 'The camera is blocked. Allow it in the address bar, then try again.' : 'Nothing was shared. Pick a tab, window or screen to continue.');
      else onResolve(req.requestId, { value: '', capture: result });
    });
  };

  return (
    <Sheet
      icon={camera ? <Camera className="h-6 w-6" /> : <MonitorUp className="h-6 w-6" />}
      title={camera ? 'Turn on your camera' : 'Share what I should watch'}
      subtitle={req.reason ?? (camera
        ? 'Observer looks at your camera to build the agent. Frames stay on this device unless you pick a cloud model.'
        : 'Pick the tab, window or screen in the browser picker. I\'ll look at it once to set things up.')}
      onClose={() => onResolve(req.requestId, { value: '', skipped: true })}
    >
      <PrimaryButton onClick={share} busy={busy}>
        {camera ? 'Turn on camera' : 'Share screen'} <ArrowRight className="h-4 w-4" />
      </PrimaryButton>
      {error && <p className="mt-3 text-xs text-red-500">{error}</p>}
    </Sheet>
  );
};

// ---- method ---------------------------------------------------------------------------------

/**
 * The value for `method` if it already works, without showing anything: a connected phone or
 * Telegram code, a contact the user confirmed before, granted notification permission.
 * null means it needs setting up.
 */
async function silentValue(method: NotifyMethod, getToken: () => Promise<string | undefined>): Promise<string | null> {
  switch (method) {
    case 'system':
      return '';
    case 'browser':
      return typeof Notification === 'undefined' || Notification.permission === 'granted' ? '' : null;
    case 'whatsapp':
    case 'sms':
    case 'voice': {
      const order: PhoneChannel[] = method === 'sms' ? ['sms'] : method === 'voice' ? ['whatsapp', 'sms'] : ['whatsapp'];
      const codes = order.map(c => SensorSettings.getPhoneCode(c)).filter((c): c is string => !!c);
      if (codes.length === 0) return null;
      const token = await getToken();
      if (!token) return null;
      for (const code of codes) {
        if ((await checkNumber(code, token, method)).isWhitelisted) return code;
      }
      return null;
    }
    case 'telegram': {
      const code = SensorSettings.getTelegramCode();
      if (!code) return null;
      const status = await fetchStatus('telegram', code).catch(() => null);
      return status?.linked ? code : null;
    }
    case 'email':
      // Always shown: the user sees which address it goes to before agreeing.
      return null;
    default:
      return SensorSettings.getNotificationContact(method) || null;
  }
}

const MethodModal: React.FC<UserInfoModalProps> = ({ req, onResolve }) => {
  const { getAccessToken } = useAuth();
  const [method, setMethod] = useState<NotifyMethod | null>(req.method ?? null);
  // 'checking' a saved connection, 'setup' to connect one, 'done' to show the check mark.
  const [phase, setPhase] = useState<'pick' | 'checking' | 'setup' | 'done'>(req.method ? 'checking' : 'pick');
  const [value, setValue] = useState('');

  const resolve = useCallback((m: NotifyMethod, v: string) => onResolve(req.requestId, { value: v, method: m }), [onResolve, req.requestId]);
  const skip = () => onResolve(req.requestId, { value: '', skipped: true });

  // Whenever a method is chosen, try it silently first.
  useEffect(() => {
    if (phase !== 'checking' || !method) return;
    let cancelled = false;
    silentValue(method, getAccessToken).then(v => {
      if (cancelled) return;
      if (v !== null) resolve(method, v);
      else setPhase('setup');
    });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, method]);

  const pick = (m: NotifyMethod) => {
    // Ask for notification permission inside the click: browsers only allow it from a gesture.
    if (m === 'browser' && typeof Notification !== 'undefined' && Notification.permission === 'default') {
      Notification.requestPermission().finally(() => { setMethod(m); setPhase('checking'); });
      return;
    }
    setMethod(m);
    setPhase('checking');
  };

  const connected = (v: string) => { setValue(v); setPhase('done'); };
  const choosePick = () => { setMethod(null); setPhase('pick'); };

  // A channel the user named, being checked: just the pill. Most returning users see nothing else.
  if (phase === 'checking' && req.method) return <CheckingPill label={`Checking ${METHODS[req.method].label}…`} />;

  const back = !req.method && phase !== 'pick' && phase !== 'done' ? () => { setMethod(null); setPhase('pick'); } : undefined;

  if (phase === 'pick' || !method) {
    const options = (Object.keys(METHODS) as NotifyMethod[]).filter(m => m !== 'system' || isDesktop());
    return (
      <Sheet icon={<Bell className="h-6 w-6" />} title="How should I tell you?" subtitle={req.reason ?? 'Pick where the alert goes. You can change it later.'} onClose={skip}>
        <div className="grid grid-cols-3 gap-2">
          {options.map(m => (
            <button
              key={m}
              onClick={() => pick(m)}
              title={METHODS[m].hint}
              className="group flex flex-col items-center gap-2 py-4 rounded-2xl ring-1 ring-zinc-200 dark:ring-zinc-800 text-zinc-500 dark:text-zinc-400 hover:ring-zinc-900 dark:hover:ring-zinc-200 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-50 dark:hover:bg-zinc-800/60 transition-all"
            >
              {METHODS[m].icon}
              <span className="text-xs font-medium">{METHODS[m].label}</span>
            </button>
          ))}
        </div>
      </Sheet>
    );
  }

  const meta = METHODS[method];
  return (
    <Sheet icon={meta.icon} title={phase === 'done' ? `${meta.label} is ready` : `Get alerts by ${meta.label}`} subtitle={phase === 'setup' ? req.reason : undefined} onClose={skip} onBack={back}>
      {phase === 'checking' && <Spinner label={`Checking ${meta.label}…`} />}
      {phase === 'done' && <Done label={`${meta.label} connected`} onFinish={() => resolve(method, value)} />}
      {phase === 'setup' && (
        isPhone(method) ? <PhoneSetup method={method} getToken={getAccessToken} onConnected={connected} />
          : method === 'telegram' ? <TelegramSetup onConnected={connected} />
            : method === 'browser' ? <BrowserSetup onConnected={() => connected('')} />
              : method === 'email' ? <EmailSetup onConnected={connected} onOther={choosePick} />
              : method === 'system' ? null
                : <ContactSetup kind={method} onConnected={connected} />
      )}
    </Sheet>
  );
};

/** QR for the user's code; texting / messaging it connects the phone. Polls until it does. */
const PhoneSetup: React.FC<{ method: PhoneMethod; getToken: () => Promise<string | undefined>; onConnected: (code: string) => void }> = ({ method, getToken, onConnected }) => {
  const [code, setCode] = useState(() => SensorSettings.defaultPhoneCode(method));
  const pairing = SensorSettings.pairingChannelFor(code, method);
  const { allWhitelisted } = useWhitelistPolling([{ number: code, isWhitelisted: false }], getToken, method, true);

  useEffect(() => { if (allWhitelisted) onConnected(code); }, [allWhitelisted, code, onConnected]);

  return (
    <div className="flex flex-col items-center gap-5">
      {method === 'voice' && (
        <Segmented
          value={pairing}
          onChange={c => setCode(SensorSettings.ensurePhoneCode(c))}
          options={[['whatsapp', 'WhatsApp', <FaWhatsapp key="w" className="h-3.5 w-3.5" />], ['sms', 'SMS', <MessageSquare key="s" className="h-3.5 w-3.5" />]]}
        />
      )}
      <div className="w-full rounded-3xl bg-zinc-50 dark:bg-zinc-800/50 p-5">
        <WhitelistQR code={code} channel={pairing} />
      </div>
      <Waiting>Waiting for your message — this continues on its own</Waiting>
      <TextButton onClick={() => setCode(SensorSettings.rotatePhoneCode(pairing))}>
        <RefreshCw className="h-3 w-3" /> Can't connect? Use a new code
      </TextButton>
    </div>
  );
};

const TelegramSetup: React.FC<{ onConnected: (code: string) => void }> = ({ onConnected }) => {
  const { code, linked, rotate } = useTelegramStatus();
  useEffect(() => { if (linked) onConnected(code); }, [linked, code, onConnected]);
  return (
    <div className="flex flex-col items-center gap-5">
      <div className="w-full rounded-3xl bg-zinc-50 dark:bg-zinc-800/50 p-5">
        <TelegramQR code={code} />
      </div>
      <Waiting>Waiting for Telegram — this continues on its own</Waiting>
      <TextButton onClick={rotate}><RefreshCw className="h-3 w-3" /> Can't connect? Use a new code</TextButton>
    </div>
  );
};

const BrowserSetup: React.FC<{ onConnected: () => void }> = ({ onConnected }) => {
  const [denied, setDenied] = useState(typeof Notification !== 'undefined' && Notification.permission === 'denied');
  const allow = () => {
    Notification.requestPermission().then(p => (p === 'granted' ? onConnected() : setDenied(true)));
  };
  return (
    <div className="flex flex-col items-center gap-4">
      {denied ? (
        <>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Notifications are blocked for this site. Click the icon left of the address bar, allow notifications, then continue.
          </p>
          <PrimaryButton onClick={allow}>I've allowed them</PrimaryButton>
        </>
      ) : (
        <PrimaryButton onClick={allow}>Allow notifications <ArrowRight className="h-4 w-4" /></PrimaryButton>
      )}
    </div>
  );
};

/**
 * The backend only sends to the signed-in user's own address, so there is nothing to type:
 * show it, confirm it, or pick another method.
 */
const EmailSetup: React.FC<{ onConnected: (value: string) => void; onOther: () => void }> = ({ onConnected, onOther }) => {
  const { user } = useAuth();
  const email = user?.email ?? '';
  return (
    <div className="flex flex-col items-center gap-4">
      {email ? (
        <>
          <div className="w-full flex items-center gap-3 px-4 h-12 rounded-2xl bg-zinc-50 dark:bg-zinc-800/50 ring-1 ring-zinc-200 dark:ring-zinc-800">
            <Mail className="h-4 w-4 flex-shrink-0 text-zinc-400" />
            <span className="text-sm font-medium truncate">{email}</span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Alerts can only go to the email on your Observer account.</p>
          <PrimaryButton onClick={() => onConnected(email)}>Send alerts here <ArrowRight className="h-4 w-4" /></PrimaryButton>
        </>
      ) : (
        <p className="text-sm text-zinc-500 dark:text-zinc-400">Sign in to get alerts by email: they go to the address on your Observer account.</p>
      )}
      <TextButton onClick={onOther}><ChevronLeft className="h-3 w-3" /> Choose another method</TextButton>
    </div>
  );
};

const CONTACT_STEPS: Partial<Record<ContactMethod, React.ReactNode[]>> = {
  discord: [
    <>In Discord, open <b>Server Settings → Integrations</b>.</>,
    <>Click <b>Webhooks → New Webhook</b>, pick a channel.</>,
    <>Click <b>Copy Webhook URL</b> and paste it below.</>,
  ],
  pushover: [
    <>Sign in at <a href="https://pushover.net" target="_blank" rel="noreferrer" className="underline">pushover.net</a>.</>,
    <>Copy the <b>User Key</b> on your dashboard and paste it below.</>,
  ],
};

const ContactSetup: React.FC<{ kind: ContactMethod; onConnected: (value: string) => void }> = ({ kind, onConnected }) => {
  const [value, setValue] = useState('');
  const [touched, setTouched] = useState(false);
  const valid = contactValid(kind, value);
  const error = touched ? contactError(kind, value) : null;

  const confirm = () => {
    if (!valid) { setTouched(true); return; }
    const normalized = normalizeContact(kind, value);
    SensorSettings.setNotificationContact(kind, normalized);
    onConnected(normalized);
  };

  return (
    <div className="flex flex-col gap-4 text-left">
      {CONTACT_STEPS[kind] && (
        <ol className="space-y-2.5">
          {CONTACT_STEPS[kind]!.map((s, i) => (
            <li key={i} className="flex gap-3 text-sm text-zinc-600 dark:text-zinc-300">
              <span className="flex-shrink-0 flex items-center justify-center h-5 w-5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-[11px] font-semibold">{i + 1}</span>
              <span>{s}</span>
            </li>
          ))}
        </ol>
      )}
      <div>
        <label className="block text-xs font-medium text-zinc-500 mb-1.5">{CONTACT_LABEL[kind]}</label>
        <input
          type="text"
          value={value}
          onChange={e => setValue(e.target.value)}
          onBlur={() => setTouched(true)}
          onKeyDown={e => { if (e.key === 'Enter') confirm(); }}
          placeholder={CONTACT_PLACEHOLDER[kind]}
          autoFocus
          className={`w-full h-11 px-4 rounded-xl bg-zinc-50 dark:bg-zinc-800 text-sm placeholder-zinc-400 outline-none ring-1 transition-shadow ${
            error ? 'ring-red-400' : 'ring-zinc-200 dark:ring-zinc-700 focus:ring-2 focus:ring-zinc-900 dark:focus:ring-zinc-200'
          }`}
        />
        {error && <p className="text-xs text-red-500 mt-1.5">{error}</p>}
      </div>
      <PrimaryButton onClick={confirm} disabled={!value}>Use this <ArrowRight className="h-4 w-4" /></PrimaryButton>
    </div>
  );
};

function Segmented<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: [T, string, React.ReactNode][] }) {
  return (
    <div className="inline-flex p-1 rounded-full bg-zinc-100 dark:bg-zinc-800">
      {options.map(([v, label, icon]) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          aria-pressed={value === v}
          className={`inline-flex items-center gap-1.5 px-4 h-8 rounded-full text-xs font-medium transition-all ${
            value === v ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200'
          }`}
        >
          {icon}{label}
        </button>
      ))}
    </div>
  );
}

// ---- model ----------------------------------------------------------------------------------

/** kind='model': download a local model now, or run in the cloud. Dismissing means cloud. */
const ModelModal: React.FC<UserInfoModalProps> = ({ req, onResolve }) => {
  const isDecision = req.model === 'onejev';
  const localName = isDecision ? 'OneJev' : 'Qwen3.5 0.8B';
  const cloudName = isDecision ? 'clef-flash' : 'gemma-3-12b-it';
  const choose = (value: 'local' | 'cloud') => onResolve(req.requestId, { value });

  const option = (value: 'local' | 'cloud', icon: React.ReactNode, title: string, detail: string) => (
    <button
      onClick={() => choose(value)}
      className="w-full flex items-start gap-4 p-4 rounded-2xl ring-1 ring-zinc-200 dark:ring-zinc-800 text-left hover:ring-zinc-900 dark:hover:ring-zinc-200 hover:bg-zinc-50 dark:hover:bg-zinc-800/60 transition-all"
    >
      <span className="flex-shrink-0 flex items-center justify-center h-9 w-9 rounded-xl bg-zinc-100 dark:bg-zinc-800">{icon}</span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold">{title}</span>
        <span className="block text-xs text-zinc-500 dark:text-zinc-400 mt-0.5 leading-5">{detail}</span>
      </span>
    </button>
  );

  return (
    <Sheet icon={<HardDrive className="h-6 w-6" />} title="Where should it run?" subtitle={req.reason} onClose={() => choose('cloud')}>
      <div className="flex flex-col gap-2.5">
        {option('local', <HardDrive className="h-4 w-4" />, `On this computer · ${localName}`,
          'Private and free. About 1 GB, so the download can take 10+ minutes — you can walk away, the agent starts on its own.')}
        {option('cloud', <Cloud className="h-4 w-4" />, `In the cloud · ${cloudName}`, 'Starts right away. Frames are sent to the cloud model.')}
      </div>
    </Sheet>
  );
};

// ---------------------------------------------------------------------------------------------

const UserInfoModal: React.FC<UserInfoModalProps> = props => {
  switch (props.req.kind) {
    case 'screen':
    case 'camera':
      return <CaptureModal {...props} />;
    case 'model':
      return <ModelModal {...props} />;
    default:
      return <MethodModal {...props} />;
  }
};

export default UserInfoModal;
