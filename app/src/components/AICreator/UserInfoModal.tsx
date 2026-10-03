// src/components/AICreator/UserInfoModal.tsx
//
// The guided prompt the `ask_user_info` MCP tool blocks on. Instead of the builder asking
// "what's your Discord webhook?" in chat and hoping the user knows how to find one, this
// walks them through actually obtaining the value, then hands it back to the run.
//
// For phones there is nothing to type: the user's 4-word code is shown as a WhatsApp QR,
// and the tool only returns once that code is connected, which is what lets `ask_user_info`
// replace a separate `check_whitelist` call. Observer never sends to raw phone numbers.
//
// Values are remembered (SensorSettings.getNotificationContact) so a returning user gets a
// prefilled field and a one-click confirm rather than re-hunting a webhook URL.

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Phone, Mail, Send, Hash, Bell, Check, Pencil, X,
  CheckCircle2, Loader, RefreshCw, XCircle, KeyRound,
} from 'lucide-react';
import Modal from '@components/EditAgent/Modal';
import WhitelistQR from '@components/whitelist/WhitelistQR';
import TelegramQR from '@components/whitelist/TelegramQR';
import {
  useWhitelistPolling, useTelegramStatus, checkNumber,
} from '@components/whitelist/shared';
import { useAuth } from '@contexts/AuthContext';
import { SensorSettings } from '@utils/settings';
import * as toolUtils from '@utils/handlers/utils';
import type { UserInfoKind, UserInfoRequest, UserInfoResponse } from '../../mcp/types';
import {
  CONTACT_LABEL,
  CONTACT_PLACEHOLDER,
  contactError,
  contactValid,
  normalizeContact,
} from '@utils/contactInfo';

/** The channel to actually test send through — defaults to WhatsApp when unspecified. */
const CHANNEL_TEST_LABEL: Record<'sms' | 'voice' | 'whatsapp', string> = {
  whatsapp: 'Test WhatsApp',
  sms: 'Test SMS',
  voice: 'Test call',
};

interface UserInfoModalProps {
  req: UserInfoRequest;
  onResolve: (requestId: string, response: UserInfoResponse) => void;
}

const KIND_ICON: Record<UserInfoKind, React.ReactNode> = {
  phone: <Phone className="h-5 w-5" />,
  email: <Mail className="h-5 w-5" />,
  telegram: <Send className="h-5 w-5" />,
  discord: <Hash className="h-5 w-5" />,
  pushover: <Bell className="h-5 w-5" />,
};

const KIND_TITLE: Record<UserInfoKind, string> = {
  phone: 'Set up notifications',
  email: 'Confirm your email',
  telegram: 'Connect Telegram',
  discord: 'Connect Discord',
  pushover: 'Connect Pushover',
};

const CHANNEL_TITLE: Record<string, string> = {
  sms: 'Set up SMS',
  voice: 'Set up phone calls',
  whatsapp: 'Set up WhatsApp',
};

/** One numbered step in the guided setup lists. */
const Step: React.FC<{ n: number; children: React.ReactNode }> = ({ n, children }) => (
  <li className="flex gap-2.5">
    <span className="flex-shrink-0 flex items-center justify-center h-5 w-5 rounded-full bg-purple-100 text-purple-700 text-xs font-bold">
      {n}
    </span>
    <span className="text-sm text-gray-700 leading-5">{children}</span>
  </li>
);

/**
 * Phone setup: one big WhatsApp QR for the user's code, no typing. Purpose-built for the
 * modal's roomy layout rather than reusing WhitelistInline's compact chat-pill chrome (that
 * component is check_whitelist's inline gate).
 */
const GoldenPathPanel: React.FC<{
  code: string;
  channel?: UserInfoRequest['channel'];
  getToken: () => Promise<string | undefined>;
  onWhitelisted: () => void;
}> = ({ code, channel, getToken, onWhitelisted }) => {
  const { allWhitelisted } = useWhitelistPolling([{ number: code, isWhitelisted: false }], getToken, channel, true);
  const [countdown, setCountdown] = useState(3);

  useEffect(() => { if (allWhitelisted) onWhitelisted(); }, [allWhitelisted, onWhitelisted]);

  // Visible 3…2…1… tick matching the parent's 3s auto-close timer.
  useEffect(() => {
    if (!allWhitelisted) return;
    setCountdown(3);
    const interval = setInterval(() => setCountdown(c => Math.max(c - 1, 0)), 1000);
    return () => clearInterval(interval);
  }, [allWhitelisted]);

  if (allWhitelisted) {
    return (
      <div className="flex flex-col items-center gap-2 py-6 text-green-700">
        <CheckCircle2 className="h-8 w-8" />
        <p className="text-sm font-medium">You're all set — phone connected.</p>
        <p className="text-xs text-gray-400 tabular-nums">Continuing in {countdown}…</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-4 py-2">
      <WhitelistQR code={code} />

      <div className="flex items-center gap-1.5 text-[11px] text-purple-600">
        <Loader className="h-3 w-3 animate-spin" />
        <span>Waiting — this continues automatically.</span>
      </div>
    </div>
  );
};

/**
 * Golden path for Telegram, mirroring the phone one: the bot deep link carries the user's
 * Telegram code (`/start <code>`), which pairs the chat with it server-side. The code is then
 * what agent code passes as sendTelegram's chat_id, and the chat can also talk to the MCP.
 * No sign-in needed: the Telegram code is its own credential.
 */
const TelegramCodePanel: React.FC<{ onLinked: (code: string) => void }> = ({ onLinked }) => {
  const { code, status, linked, rotate } = useTelegramStatus();

  useEffect(() => { if (linked) onLinked(code); }, [linked, code, onLinked]);

  if (linked) {
    return (
      <div className="flex flex-col items-center gap-2 py-6 text-green-700">
        <CheckCircle2 className="h-8 w-8" />
        <p className="text-sm font-medium">
          You're all set — Telegram connected{status?.name ? ` to ${status.name}` : ''}.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-4 py-2">
      <TelegramQR code={code} />
      <div className="flex items-center gap-1.5 text-[11px] text-purple-600">
        <Loader className="h-3 w-3 animate-spin" />
        <span>Waiting — this continues automatically.</span>
      </div>
      <RotateCodeButton onClick={rotate} />
    </div>
  );
};

/** Escape hatch for a code that won't pair: mint a fresh one right here instead of going to Settings. */
const RotateCodeButton: React.FC<{ onClick: () => void }> = ({ onClick }) => (
  <button
    onClick={onClick}
    className="inline-flex items-center gap-1 text-xs font-medium text-gray-500 hover:text-gray-700 transition-colors"
  >
    <RefreshCw className="h-3 w-3" /> Can't connect? Generate a new code
  </button>
);

type TestState = 'idle' | 'testing' | 'success' | 'failure';

/**
 * Shown when the saved code is already connected — instead of silently reusing it, let the
 * user confirm/test it, or rotate to a new phone, before committing.
 */
const ConfirmExistingCodePanel: React.FC<{
  code: string;
  channel?: UserInfoRequest['channel'];
  getToken: () => Promise<string | undefined>;
  onUse: () => void;
  onRotate: () => void;
}> = ({ code, channel, getToken, onUse, onRotate }) => {
  const [whitelistTest, setWhitelistTest] = useState<TestState>('idle');
  const [toolTest, setToolTest] = useState<TestState>('idle');
  const [toolTestError, setToolTestError] = useState('');

  const testChannel: 'sms' | 'voice' | 'whatsapp' = channel ?? 'whatsapp';

  const testWhitelist = async () => {
    setWhitelistTest('testing');
    const token = await getToken();
    if (!token) { setWhitelistTest('failure'); return; }
    const result = await checkNumber(code, token, channel);
    setWhitelistTest(result.isWhitelisted ? 'success' : 'failure');
  };

  const testTool = async () => {
    setToolTest('testing');
    setToolTestError('');
    try {
      const token = await getToken();
      if (!token) throw new Error('Failed to retrieve authentication token');
      const message = 'This is a test from Observer!';
      if (testChannel === 'whatsapp') await toolUtils.sendWhatsapp(message, code, token);
      else if (testChannel === 'sms') await toolUtils.sendSms(message, code, token);
      else await toolUtils.call(message, code, token);
      setToolTest('success');
    } catch (err) {
      setToolTest('failure');
      setToolTestError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="flex flex-col items-center gap-4 py-2">
      <div className="flex items-center justify-center h-12 w-12 rounded-full bg-purple-100 text-purple-700">
        <KeyRound className="h-6 w-6" />
      </div>

      <p className="text-sm text-gray-700 text-center">
        Use the contact info of{' '}
        <span className="font-mono font-semibold text-gray-900">{code}</span>?
      </p>

      <div className="flex flex-wrap items-center justify-center gap-2">
        <TestButton state={whitelistTest} label="Check connection" onClick={testWhitelist} />
        <TestButton state={toolTest} label={CHANNEL_TEST_LABEL[testChannel]} onClick={testTool} />
      </div>
      {toolTest === 'failure' && toolTestError && (
        <p className="text-xs text-red-600 text-center max-w-xs">{toolTestError}</p>
      )}

      <button
        onClick={onUse}
        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-purple-600 text-white font-medium text-sm hover:bg-purple-700 transition-colors"
      >
        <Check className="h-4 w-4" /> Use this
      </button>

      <div className="flex items-center gap-4 text-xs">
        <button
          onClick={onRotate}
          className="inline-flex items-center gap-1 font-medium text-gray-500 hover:text-gray-700 transition-colors"
        >
          <RefreshCw className="h-3 w-3" /> Rotate key to a new phone
        </button>
      </div>
    </div>
  );
};

const TestButton: React.FC<{ state: TestState; label: string; onClick: () => void }> = ({ state, label, onClick }) => (
  <button
    onClick={onClick}
    disabled={state === 'testing'}
    className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
      state === 'success'
        ? 'bg-green-100 text-green-700'
        : state === 'failure'
          ? 'bg-red-100 text-red-700'
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
    }`}
  >
    {state === 'testing' ? (
      <Loader className="h-3 w-3 animate-spin" />
    ) : state === 'success' ? (
      <CheckCircle2 className="h-3 w-3" />
    ) : state === 'failure' ? (
      <XCircle className="h-3 w-3" />
    ) : null}
    {label}
  </button>
);

const UserInfoModal: React.FC<UserInfoModalProps> = ({ req, onResolve }) => {
  const { user, getAccessToken } = useAuth();
  const { kind, channel, requestId } = req;

  const remembered = useMemo(() => SensorSettings.getNotificationContact(kind), [kind, requestId]);
  // Email needs no hunting — the signed-in address is almost always the answer.
  const initial = kind === 'email' ? (remembered || user?.email || '') : remembered;

  const [value, setValue] = useState(initial);
  // A remembered value collapses the guided steps into a one-click confirm; "Change" expands.
  const [editing, setEditing] = useState(!initial);

  const valid = contactValid(kind, value);
  const error = editing ? contactError(kind, value) : null;
  const needsWhitelist = kind === 'phone';

  // Phone: a persisted code + WhatsApp QR, no typing. The code is generated once and only
  // changes when the user rotates it, so agent code that bakes it in
  // (sendWhatsapp("tree-book-shower-golden", ...)) never goes stale: pairing is permanent,
  // and a closed WhatsApp window is reopened by sending anything, the code included.
  //
  // Steps for phone: 'checking' (silently test the saved code), 'confirm' (it's connected —
  // confirm/test/rotate instead of assuming), 'qr' (not connected yet, or just rotated).
  const hadExistingCode = useMemo(() => (kind === 'phone' ? !!SensorSettings.getWhitelistCode() : false), [kind]);
  const [code, setCode] = useState(() => (kind === 'phone' ? SensorSettings.ensureWhitelistCode() : ''));
  const [phoneStep, setPhoneStep] = useState<'checking' | 'confirm' | 'qr'>(
    kind === 'phone' && hadExistingCode ? 'checking' : 'qr',
  );
  const [codeVerified, setCodeVerified] = useState(false);

  // Telegram: its own persisted code via the bot's /start deep link. Nothing to type.
  const useTelegramCode = kind === 'telegram';
  const useCodePath = needsWhitelist && phoneStep === 'qr';

  const rotateCode = () => {
    setCode(SensorSettings.rotateWhitelistCode());
    setCodeVerified(false);
    setPhoneStep('qr');
  };

  // On open, silently check the saved code instead of trusting it blindly. If it isn't ready
  // (never paired, or WhatsApp's 24h window closed), show the QR for the SAME code: sending
  // it fixes both. Never rotate here — agents already built with this code would stop
  // reaching the phone.
  useEffect(() => {
    if (phoneStep !== 'checking') return;
    let cancelled = false;
    (async () => {
      const token = await getAccessToken();
      if (!token) { if (!cancelled) setPhoneStep('qr'); return; }
      const result = await checkNumber(code, token, channel);
      if (cancelled) return;
      setPhoneStep(result.isWhitelisted ? 'confirm' : 'qr');
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phoneStep]);

  const canConfirm = useCodePath ? codeVerified : valid;

  const confirm = () => {
    if (!canConfirm) return;
    if (useCodePath) {
      onResolve(requestId, { value: code });
      return;
    }
    const normalized = normalizeContact(kind, value);
    SensorSettings.setNotificationContact(kind, normalized);
    onResolve(requestId, { value: normalized });
  };

  // Once the QR path verifies, auto-close after a beat instead of waiting on a click —
  // there's nothing left to decide, the success state already says "you're all set".
  useEffect(() => {
    if (!useCodePath || !codeVerified) return;
    const timer = setTimeout(() => onResolve(requestId, { value: code }), 3000);
    return () => clearTimeout(timer);
  }, [useCodePath, codeVerified, code, requestId]);

  const skip = () => onResolve(requestId, { value: '', skipped: true });
  const resolveTelegram = useCallback(
    (telegramCode: string) => onResolve(requestId, { value: telegramCode }),
    [onResolve, requestId],
  );

  const title = kind === 'phone' && channel ? CHANNEL_TITLE[channel] ?? KIND_TITLE.phone : KIND_TITLE[kind];

  return (
    <Modal open onClose={skip} className="w-full max-w-xl mx-4">
      {/* Header */}
      <div className="flex items-start gap-3 px-6 py-4 rounded-t-lg border-b border-gray-100">
        <div className="flex-shrink-0 mt-0.5 flex items-center justify-center h-9 w-9 rounded-full bg-gray-100 text-gray-700">
          {KIND_ICON[kind]}
        </div>
        <div className="min-w-0 flex-1">
          <h2 className="text-lg font-semibold leading-tight text-gray-900">{title}</h2>
          {req.reason && <p className="text-sm text-gray-500 mt-0.5">{req.reason}</p>}
        </div>
        <button
          onClick={skip}
          aria-label="Skip"
          title="Skip — I'll do this later"
          className="flex-shrink-0 p-1 rounded text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      <div className="p-6 space-y-4 max-h-[70vh] overflow-y-auto">
        {/* Silently re-verify a previously-saved code before trusting it. */}
        {needsWhitelist && phoneStep === 'checking' && (
          <div className="flex flex-col items-center gap-3 py-8 text-gray-500">
            <Loader className="h-6 w-6 animate-spin text-purple-600" />
            <p className="text-sm">Checking existing key…</p>
          </div>
        )}

        {/* Returning user: confirm/test/rotate the already-persisted code instead of assuming it. */}
        {needsWhitelist && phoneStep === 'confirm' && (
          <ConfirmExistingCodePanel
            code={code}
            channel={channel}
            getToken={getAccessToken}
            onUse={() => onResolve(requestId, { value: code })}
            onRotate={rotateCode}
          />
        )}

        {useTelegramCode && (
          <TelegramCodePanel onLinked={resolveTelegram} />
        )}

        {/* Phone: one big WhatsApp QR + code, no typing required. */}
        {useCodePath && (
          <GoldenPathPanel
            code={code}
            channel={channel}
            getToken={getAccessToken}
            onWhitelisted={() => setCodeVerified(true)}
          />
        )}
        {useCodePath && !codeVerified && (
          <div className="flex justify-center">
            <RotateCodeButton onClick={rotateCode} />
          </div>
        )}

        {!needsWhitelist && !useTelegramCode && (
        <>
        {/* Remembered value — one-click confirm instead of retyping. */}
        {!editing && (
          <div className="flex items-center gap-3 p-3 rounded-lg border border-purple-200 bg-purple-50">
            <Check className="h-4 w-4 text-purple-600 flex-shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="text-xs text-gray-500">{CONTACT_LABEL[kind]}</p>
              <p className="text-sm font-mono text-gray-900 truncate">{value}</p>
            </div>
            <button
              onClick={() => setEditing(true)}
              className="flex-shrink-0 inline-flex items-center gap-1 text-xs font-medium text-purple-700 hover:text-purple-900 transition-colors"
            >
              <Pencil className="h-3 w-3" /> Change
            </button>
          </div>
        )}

        {editing && (
          <>
            {/* Guided steps — the part that makes this worth a modal. */}
            {kind === 'discord' && (
              <ol className="space-y-2.5">
                <Step n={1}>In Discord, open <span className="font-medium">Server Settings → Integrations</span>.</Step>
                <Step n={2}>Click <span className="font-medium">Webhooks → New Webhook</span>.</Step>
                <Step n={3}>Pick the channel you want alerts in, then <span className="font-medium">Copy Webhook URL</span>.</Step>
                <Step n={4}>Paste it below.</Step>
              </ol>
            )}

            {kind === 'pushover' && (
              <ol className="space-y-2.5">
                <Step n={1}>
                  Open{' '}
                  <a href="https://pushover.net" target="_blank" rel="noreferrer" className="text-purple-700 underline">
                    pushover.net
                  </a>{' '}
                  and sign in.
                </Step>
                <Step n={2}>Copy the <span className="font-medium">User Key</span> shown on your dashboard.</Step>
              </ol>
            )}

            {/* Input */}
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1.5">{CONTACT_LABEL[kind]}</label>
              <input
                type={kind === 'email' ? 'email' : 'text'}
                value={value}
                onChange={e => setValue(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && canConfirm) confirm(); }}
                placeholder={CONTACT_PLACEHOLDER[kind]}
                autoFocus
                className={`w-full px-3 py-2.5 border rounded-lg text-gray-900 placeholder-gray-400 focus:outline-none transition-colors ${
                  error ? 'border-red-300 focus:border-red-500' : 'border-gray-300 focus:border-purple-500'
                }`}
              />
              {error && <p className="text-xs text-red-600 mt-1.5">{error}</p>}
            </div>
          </>
        )}

        </>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between gap-3 px-6 py-4 bg-gray-50 rounded-b-lg border-t border-gray-200">
        <button onClick={skip} className="text-sm text-gray-500 hover:text-gray-700 transition-colors">
          Skip for now
        </button>
        {phoneStep !== 'confirm' && !useTelegramCode && (
          <button
            onClick={confirm}
            disabled={!canConfirm}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-purple-600 text-white font-medium text-sm hover:bg-purple-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {useCodePath && !codeVerified ? 'Waiting for your phone…' : 'Confirm'}
            {canConfirm && <Check className="h-4 w-4" />}
          </button>
        )}
      </div>
    </Modal>
  );
};

export default UserInfoModal;
