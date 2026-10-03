import React from 'react';
import Modal from '@components/EditAgent/Modal';
import { Phone, X, CheckCircle, AlertTriangle } from 'lucide-react';
import type { WhitelistChannel } from '@utils/logging';
import WhitelistInline from '@components/whitelist/WhitelistInline';
import { SensorSettings } from '@utils/settings';
import { normalizeWhitelistCode } from '@utils/whitelistCode';

interface WhitelistModalProps {
  /** Every literal the agent passes to a phone tool (see pre-flight.ts). */
  phoneNumbers: Array<{
    number: string;
    isWhitelisted: boolean;
    /** The tool it's passed to; falls back to `channel`. */
    channel?: WhitelistChannel;
  }>;
  onClose: () => void;
  onStartAnyway?: () => void;
  onStartAgent?: () => void;
  getToken: () => Promise<string | undefined>;
  channel?: WhitelistChannel; // 'whatsapp' | 'sms' | 'voice'
}

/**
 * Shown when start_agent's phone pre-flight fails. The phone tools only send to the user's
 * 4-word code once it's paired for that tool (by SMS for sendSms, on WhatsApp for sendWhatsapp,
 * either for call), so what the user has to do depends on what the agent actually sends to:
 * a raw phone number has to be replaced with their code (it can never be sent to), and a code
 * has to be paired, or its WhatsApp window reopened.
 */
const WhitelistModal: React.FC<WhitelistModalProps> = ({ phoneNumbers, onClose, onStartAnyway, onStartAgent, getToken, channel }) => {
  const savedCode = React.useMemo(() => SensorSettings.defaultPhoneCode(channel), [channel]);

  const rawNumbers = [...new Set(phoneNumbers.filter(p => !normalizeWhitelistCode(p.number)).map(p => p.number))];
  // Each code once per tool it's used with: the same code can be ready for a call but not SMS.
  const uses = new Map<string, { code: string; channel?: WhitelistChannel }>();
  for (const p of phoneNumbers) {
    const code = normalizeWhitelistCode(p.number);
    const ch = p.channel ?? channel;
    if (code) uses.set(`${ch}:${code}`, { code, channel: ch });
  }
  if (uses.size === 0 && rawNumbers.length > 0) uses.set(`${channel}:${savedCode}`, { code: savedCode, channel });

  const [paired, setPaired] = React.useState<Set<string>>(new Set());
  const markPaired = React.useCallback((key: string) => {
    setPaired(prev => (prev.has(key) ? prev : new Set(prev).add(key)));
  }, []);

  const success = rawNumbers.length === 0 && uses.size > 0 && [...uses.keys()].every(k => paired.has(k));

  return (
    <Modal open={true} onClose={onClose} className="w-full max-w-lg md:max-w-2xl">
      {/* Header */}
      <div className="flex justify-between items-center p-6 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-t-lg">
        <div className="flex items-center space-x-3">
          <Phone className="h-6 w-6" />
          <div>
            <h2 className="text-xl font-semibold">Connect your phone</h2>
            <p className="text-sm text-blue-100">One-time setup</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-full hover:bg-blue-700 hover:bg-opacity-50 text-blue-100 hover:text-white transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Content */}
      <div className="p-6 space-y-4">
        {success && onStartAgent ? (
          <div className="py-8">
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
                <CheckCircle className="h-8 w-8 text-green-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">All Set!</h2>
              <p className="text-gray-600">Your phone is connected and ready to go.</p>
            </div>
            <button
              onClick={() => {
                onStartAgent();
                onClose();
              }}
              className="w-full px-6 py-4 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold text-lg shadow-lg transition-colors"
            >
              Start Agent
            </button>
          </div>
        ) : (
          <>
            {rawNumbers.length > 0 && (
              <div className="flex gap-3 p-4 rounded-lg border border-amber-300 bg-amber-50">
                <AlertTriangle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-amber-900 space-y-2">
                  <p>
                    This agent sends to{' '}
                    {rawNumbers.map((n, i) => (
                      <React.Fragment key={n}>
                        {i > 0 && ', '}
                        <span className="font-mono font-semibold">{n}</span>
                      </React.Fragment>
                    ))}
                    . Observer no longer sends to phone numbers directly.
                  </p>
                  <p>
                    Edit the agent's code and replace {rawNumbers.length > 1 ? 'them' : 'it'} with your code{' '}
                    <span className="font-mono font-semibold select-all">{savedCode}</span>, then connect that code below.
                  </p>
                </div>
              </div>
            )}

            {uses.size === 0 && (
              <p className="text-sm text-orange-700">
                ⚠️ This agent uses a phone tool, but passes it a value that can't be checked before starting.
                It has to be your 4-word code (<span className="font-mono">{savedCode}</span>), connected to your phone.
              </p>
            )}

            {[...uses].map(([key, use]) => (
              <WhitelistInline
                key={key}
                code={use.code}
                channel={use.channel}
                getToken={getToken}
                onWhitelisted={() => markPaired(key)}
              />
            ))}
          </>
        )}
      </div>

      {/* Footer */}
      <div className="flex justify-end items-center px-6 py-4 border-t border-gray-200 bg-gray-50 rounded-b-lg space-x-3">
        {success && onStartAgent ? (
          <button
            onClick={onClose}
            className="px-5 py-2 bg-gray-200 text-gray-700 rounded-md text-sm font-medium hover:bg-gray-300 transition-colors"
          >
            Close
          </button>
        ) : onStartAnyway ? (
          <>
            <button
              onClick={onClose}
              className="px-5 py-2 bg-gray-200 text-gray-700 rounded-md text-sm font-medium hover:bg-gray-300 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={onStartAnyway}
              className="px-5 py-2 border-2 border-orange-500 text-orange-700 rounded-md text-sm font-medium hover:bg-orange-50 transition-colors"
            >
              ⚠️ Start Anyway (Will Fail)
            </button>
          </>
        ) : (
          <button
            onClick={onClose}
            className="px-5 py-2 bg-indigo-600 text-white rounded-md text-sm font-medium hover:bg-indigo-700 transition-colors"
          >
            Got it
          </button>
        )}
      </div>
    </Modal>
  );
};

export default WhitelistModal;
