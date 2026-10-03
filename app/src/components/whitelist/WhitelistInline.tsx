// src/components/whitelist/WhitelistInline.tsx
//
// Compact, non-modal pairing prompt the MCP renders under a `check_whitelist` tool call
// while that gate is BLOCKING (status: 'running'). Expanded by default; click to collapse.
// Purely presentational in the MCP flow: the check_whitelist executor does the polling and
// resolves once the code is paired, which unmounts this pill and lets the run continue
// straight to start_agent — no messages, no manual resume.
//
// Shows the pairing the code needs: texting an SMS code connects SMS and call alerts,
// sending a WhatsApp code on WhatsApp connects WhatsApp and call alerts.

import React, { useEffect, useState } from 'react';
import { ChevronRight, MessageCircle, MessageSquare, ExternalLink, AlertTriangle, Loader, X, CheckCircle2 } from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';
import type { WhitelistChannel } from '@utils/logging';
import { SensorSettings } from '@utils/settings';
import {
  whatsappCodeQRValue,
  openWhatsApp,
  OBSERVER_WHATSAPP,
  smsCodeQRValue,
  openSms,
  OBSERVER_SMS,
  SMS_CONSENT,
  useWhitelistPolling,
} from './shared';

interface WhitelistInlineProps {
  /** The 4-word whitelist code to pair. */
  code: string;
  channel?: WhitelistChannel;
  onCancel?: () => void;
  /**
   * When provided, this pill polls the whitelist API itself and reflects pairing live.
   * Leave undefined in the MCP flow, where the `check_whitelist` executor is the sole poller.
   */
  getToken?: () => Promise<string | undefined>;
  /** Fired once the code is paired (only when self-polling via getToken). */
  onWhitelisted?: () => void;
}

const NO_TOKEN = async () => undefined;

const WhitelistInline: React.FC<WhitelistInlineProps> = ({ code, channel, onCancel, getToken, onWhitelisted }) => {
  const [expanded, setExpanded] = useState(true);

  // Self-poll only when a token provider is supplied. In the MCP flow getToken is undefined,
  // so `enabled` is false and the executor remains the single source of polling.
  const selfPolling = !!getToken;
  const { allWhitelisted } = useWhitelistPolling(
    [{ number: code, isWhitelisted: false }],
    getToken ?? NO_TOKEN,
    channel,
    selfPolling,
  );

  useEffect(() => {
    if (selfPolling && allWhitelisted) onWhitelisted?.();
  }, [selfPolling, allWhitelisted, onWhitelisted]);

  const verified = selfPolling && allWhitelisted;
  const sms = SensorSettings.pairingChannelFor(code, channel) === 'sms';

  return (
    <div className="mt-2 w-full rounded-lg border border-purple-200 bg-white overflow-hidden shadow-sm">
      {/* Header — icon + two-line title/code so a long code never forces overflow, plus
          icon-only toggle/cancel buttons to keep the row usable at chat-bubble widths. */}
      <div className="flex items-start gap-2 px-3 py-2">
        <button
          onClick={() => setExpanded(e => !e)}
          className="flex-1 min-w-0 flex items-start gap-2 text-left hover:opacity-80 transition-opacity"
        >
          <AlertTriangle className="h-4 w-4 text-purple-600 flex-shrink-0 mt-0.5" />
          <span className="min-w-0">
            <span className="block text-sm font-medium text-gray-900">
              {sms ? 'Text this code to Observer to connect' : 'Send this code on WhatsApp to connect'}
            </span>
            <span className="block text-xs font-mono text-gray-500 truncate">{code}</span>
          </span>
        </button>
        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            onClick={() => setExpanded(e => !e)}
            title={expanded ? 'Hide' : 'Show how'}
            className="p-1 rounded text-purple-600 hover:bg-purple-50 transition-colors"
          >
            <ChevronRight className={`h-4 w-4 transition-transform ${expanded ? 'rotate-90' : ''}`} />
          </button>
          {onCancel && (
            <button
              onClick={onCancel}
              title="Cancel"
              className="p-1 rounded text-gray-500 hover:bg-gray-100 border border-gray-300 transition-colors"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      {expanded && !verified && (
        <div className="px-3 pb-3 pt-1 space-y-2.5 border-t border-gray-200">
          <p className="text-xs text-gray-600">
            {sms
              ? 'Scan the QR or text the code below to Observer. You only need to do this once: it connects SMS and call alerts to your phone.'
              : channel === 'whatsapp'
                ? 'Scan the QR or send the code below on WhatsApp. If you connected before, send any message to turn WhatsApp alerts back on (WhatsApp pauses them 24 hours after your last message).'
                : 'Scan the QR or send the code below on WhatsApp. You only need to do this once: it connects WhatsApp and call alerts to your phone.'}
          </p>

          <p className="text-center font-mono text-sm font-semibold text-purple-700 bg-purple-50 border border-purple-200 rounded-md py-1.5 select-all">
            {code}
          </p>

          <div className="border border-purple-200 rounded-md p-2 bg-white">
            <div className="flex items-center gap-1.5 mb-1.5 text-xs font-semibold text-gray-800">
              {sms
                ? <><MessageSquare className="h-4 w-4 text-purple-600 flex-shrink-0" /><span>SMS</span></>
                : <><MessageCircle className="h-4 w-4 text-green-600 flex-shrink-0" /><span>WhatsApp</span></>}
            </div>
            <div className="hidden sm:flex justify-center mb-1.5">
              <div className="bg-white p-1.5 rounded border border-gray-200">
                <QRCodeSVG value={sms ? smsCodeQRValue(code) : whatsappCodeQRValue(code)} size={88} level="M" includeMargin={false} />
              </div>
            </div>
            <button
              onClick={sms ? () => openSms(code) : openWhatsApp}
              className="w-full px-2 py-1.5 bg-gray-900 text-white rounded text-xs font-medium hover:bg-black transition-colors flex items-center justify-center gap-1.5"
            >
              <span>{sms ? 'Open Messages' : 'Open WhatsApp'}</span>
              <ExternalLink className="h-3 w-3 flex-shrink-0" />
            </button>
            <p className="text-[10px] text-gray-500 text-center mt-1.5 font-mono">{sms ? OBSERVER_SMS : OBSERVER_WHATSAPP}</p>
            {sms && <p className="text-[10px] text-gray-400 text-center mt-1 leading-snug">{SMS_CONSENT}</p>}
          </div>
        </div>
      )}

      {/* Footer status. When self-polling, flips to green once the code is paired. */}
      {verified ? (
        <div className="flex items-center gap-1.5 px-3 pb-2 pt-1 text-[11px] font-medium text-green-600">
          <CheckCircle2 className="h-3.5 w-3.5 flex-shrink-0" />
          <span>Phone connected — you're all set.</span>
        </div>
      ) : (
        <div className="flex items-center gap-1.5 px-3 pb-2 text-[11px] text-purple-600">
          <Loader className="h-3 w-3 animate-spin flex-shrink-0" />
          <span>Waiting — continues automatically once you've sent it.</span>
        </div>
      )}
    </div>
  );
};

export default WhitelistInline;
