// Front-and-center card on the Settings home: the SMS, WhatsApp and Telegram codes + saved
// notification contacts collected by the `ask_user_info` MCP tool.
import React, { useState } from 'react';
import { Eye, EyeOff, Copy, Check, QrCode, RefreshCw, Pencil, CheckCircle2, XCircle } from 'lucide-react';
import { useAuth } from '@contexts/AuthContext';
import { SensorSettings } from '@utils/settings';
import { useTelegramStatus, useWhitelistPolling } from '@components/whitelist/shared';
import WhitelistQR from '@components/whitelist/WhitelistQR';
import TelegramQR from '@components/whitelist/TelegramQR';
import Switch from './Switch';
import { useMCPContext } from '../../mcp/MCPContext';
import type { UserInfoKind } from '../../mcp/types';
import {
  CONTACT_LABEL, CONTACT_PLACEHOLDER, contactError, contactValid, normalizeContact,
} from '@utils/contactInfo';

const KINDS: UserInfoKind[] = ['discord', 'pushover'];
const SECRET_KINDS: UserInfoKind[] = ['discord', 'pushover'];

const ContactRow: React.FC<{ kind: UserInfoKind }> = ({ kind }) => {
  const [value, setValue] = useState(SensorSettings.getNotificationContact(kind));
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const err = contactError(kind, draft);

  const save = () => {
    const next = draft.trim() ? normalizeContact(kind, draft) : '';
    SensorSettings.setNotificationContact(kind, next);
    setValue(next);
    setEditing(false);
  };

  return (
    <div className="flex items-center justify-between gap-3 px-5 py-3">
      <div className="min-w-0 flex-1">
        <p className="text-xs text-gray-500 dark:text-gray-400">{CONTACT_LABEL[kind]}</p>
        {editing ? (
          <div className="mt-1">
            <input
              autoFocus
              value={draft}
              onChange={e => setDraft(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && (!draft.trim() || contactValid(kind, draft)) && save()}
              placeholder={CONTACT_PLACEHOLDER[kind]}
              className="block w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            {err && <p className="text-xs text-red-500 mt-1">{err}</p>}
          </div>
        ) : (
          <p className="text-sm text-gray-900 dark:text-gray-100 truncate">
            {value ? (SECRET_KINDS.includes(kind) ? '••••••••' + value.slice(-4) : value) : <span className="text-gray-400">Not set</span>}
          </p>
        )}
      </div>
      {editing ? (
        <div className="flex gap-2 flex-shrink-0">
          <button onClick={() => setEditing(false)} className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300">Cancel</button>
          <button
            onClick={save}
            disabled={!!draft.trim() && !contactValid(kind, draft)}
            className="text-xs font-medium text-purple-600 disabled:opacity-40"
          >Save</button>
        </div>
      ) : (
        <button
          onClick={() => { setDraft(value); setEditing(true); }}
          className="p-1.5 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 flex-shrink-0"
          aria-label={`Edit ${CONTACT_LABEL[kind]}`}
        >
          <Pencil className="h-4 w-4" />
        </button>
      )}
    </div>
  );
};

/**
 * Chatting with Observer from a paired phone/chat — same codes, same card, just a different
 * channel capability. `status` comes from the API (only the server knows which phone/chat
 * paired); `listening` is this tab holding the session open — with it off, the bot tells the
 * phone no session is active.
 */
const RemoteControlStatus: React.FC = () => {
  const { remote } = useMCPContext();
  const linkedChannels = (['whatsapp', 'telegram'] as const).filter(c => remote.status[c]?.linked);

  const status = linkedChannels.length === 0
    ? 'Not chatting yet — link WhatsApp or Telegram above'
    : `Chat linked via ${linkedChannels.map(c => (c === 'whatsapp' ? 'WhatsApp' : 'Telegram')).join(' · ')}${
        remote.enabled ? (remote.listening ? ' · listening' : ' · connecting…') : ''
      }`;

  return (
    <div className="flex items-center justify-between gap-3 px-5 py-3">
      <p className="text-xs text-gray-500 dark:text-gray-400 min-w-0">{status}</p>
      <Switch
        checked={remote.enabled}
        onChange={() => remote.setEnabled(!remote.enabled)}
        label="Toggle remote control"
      />
    </div>
  );
};

/** One channel's code: reveal / copy / QR / rotate, plus who it's connected to. */
const CodeRow: React.FC<{
  title: string;
  description: string;
  code: string | null;
  connected: boolean;
  name?: string | null;
  qr: (code: string) => React.ReactNode;
  rotateConfirm: string;
  onRotate: () => void;
  setup?: React.ReactNode;
}> = ({ title, description, code, connected, name, qr, rotateConfirm, onRotate, setup }) => {
  const [revealed, setRevealed] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showQR, setShowQR] = useState(false);

  const copy = () => {
    if (!code) return;
    navigator.clipboard.writeText(code).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); });
  };

  const rotate = () => {
    if (!window.confirm(rotateConfirm)) return;
    onRotate();
    setRevealed(true);
    setShowQR(true);
  };

  return (
    <div className="px-5 py-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">{title}</h2>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{description}</p>
        </div>
        {code && (
          <span className={`inline-flex items-center gap-1 text-xs font-medium flex-shrink-0 ${connected ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'}`}>
            {connected ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
            {connected ? (name ? `Connected to ${name}` : 'Connected') : 'Not connected'}
          </span>
        )}
      </div>

      {code ? (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            onClick={() => setRevealed(r => !r)}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700"
            title={revealed ? 'Hide' : 'Reveal'}
          >
            <span className={`font-mono text-sm font-semibold text-gray-900 dark:text-gray-100 transition ${revealed ? '' : 'blur-sm select-none'}`}>{code}</span>
            {revealed ? <EyeOff className="h-4 w-4 text-gray-400" /> : <Eye className="h-4 w-4 text-gray-400" />}
          </button>
          <button onClick={copy} className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800" aria-label={`Copy ${title}`}>
            {copied ? <Check className="h-4 w-4 text-green-600" /> : <Copy className="h-4 w-4" />}
          </button>
          <button onClick={() => setShowQR(q => !q)} className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800" aria-label="Show QR">
            <QrCode className="h-4 w-4" />
          </button>
          <button onClick={rotate} className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800" aria-label={`Generate new ${title}`}>
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      ) : setup}

      {code && showQR && (
        <div className="mt-4 flex flex-col items-center rounded-xl bg-white border border-gray-200 py-4">
          {qr(code)}
        </div>
      )}
    </div>
  );
};

const ROTATE_CONFIRM = 'Generate a new code? Pair it from your phone/chat to switch over: that disconnects the current code, and agents built with it stop sending.';

const UserInfoCard: React.FC = () => {
  const { getAccessToken, isAuthenticated, login } = useAuth();
  const { remote } = useMCPContext();

  // SMS and WhatsApp: tied to the account, so they need sign-in. The whitelist poll also
  // claims each code and keeps its pairing window open while this card is visible.
  const [smsCode, setSmsCode] = useState<string | null>(SensorSettings.getPhoneCode('sms'));
  const smsEntries = smsCode ? [{ number: smsCode, isWhitelisted: false }] : [];
  const { allWhitelisted: smsConnected } = useWhitelistPolling(smsEntries, getAccessToken, 'sms', !!smsCode && isAuthenticated);

  const [waCode, setWaCode] = useState<string | null>(SensorSettings.getWhitelistCode());
  const waEntries = waCode ? [{ number: waCode, isWhitelisted: false }] : [];
  const { allWhitelisted: waConnected } = useWhitelistPolling(waEntries, getAccessToken, undefined, !!waCode && isAuthenticated);

  // Telegram: its own code, no account needed.
  const telegram = useTelegramStatus();

  return (
    <section className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 divide-y divide-gray-100 dark:divide-gray-800 overflow-hidden">
      <CodeRow
        title="SMS code"
        description="Text it to Observer for SMS and call alerts."
        code={isAuthenticated ? smsCode : null}
        connected={smsConnected}
        qr={code => <WhitelistQR code={code} channel="sms" />}
        rotateConfirm={ROTATE_CONFIRM}
        onRotate={() => setSmsCode(SensorSettings.rotatePhoneCode('sms'))}
        setup={isAuthenticated ? (
          <button
            onClick={() => setSmsCode(SensorSettings.ensurePhoneCode('sms'))}
            className="mt-3 px-4 py-2 bg-purple-600 text-white text-sm font-medium rounded-lg hover:bg-purple-700"
          >
            Set up SMS
          </button>
        ) : (
          <button
            onClick={login}
            className="mt-3 px-4 py-2 bg-purple-600 text-white text-sm font-medium rounded-lg hover:bg-purple-700"
          >
            Sign in to use SMS
          </button>
        )}
      />

      <CodeRow
        title="WhatsApp code"
        description="Link it on WhatsApp for WhatsApp and call alerts, and to chat with Observer."
        code={isAuthenticated ? waCode : null}
        connected={waConnected}
        name={remote.status.whatsapp?.name}
        qr={code => <WhitelistQR code={code} channel="whatsapp" />}
        rotateConfirm={ROTATE_CONFIRM}
        onRotate={() => setWaCode(SensorSettings.rotateWhitelistCode())}
        setup={isAuthenticated ? (
          <button
            onClick={() => setWaCode(SensorSettings.ensureWhitelistCode())}
            className="mt-3 px-4 py-2 bg-purple-600 text-white text-sm font-medium rounded-lg hover:bg-purple-700"
          >
            Set up WhatsApp
          </button>
        ) : (
          <button
            onClick={login}
            className="mt-3 px-4 py-2 bg-purple-600 text-white text-sm font-medium rounded-lg hover:bg-purple-700"
          >
            Sign in to use WhatsApp
          </button>
        )}
      />

      <CodeRow
        title="Telegram code"
        description="Link it on Telegram for Telegram alerts, and to chat with Observer. No account needed."
        code={telegram.code}
        connected={telegram.linked}
        name={telegram.status?.name}
        qr={code => <TelegramQR code={code} />}
        rotateConfirm={ROTATE_CONFIRM}
        onRotate={telegram.rotate}
      />

      <RemoteControlStatus />

      {KINDS.map(kind => <ContactRow key={kind} kind={kind} />)}
    </section>
  );
};

export default UserInfoCard;
