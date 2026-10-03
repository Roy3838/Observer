// Shared by the ask_user_info modal and Settings: QR, copyable code, open-app button.
// Each phone pairing has its own code and is its own opt-in: texting the SMS code to
// Observer's number enables SMS and call alerts, sending the WhatsApp code to the bot
// enables WhatsApp and call alerts.
import React, { useState } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { Check, Copy, MessageSquare } from 'lucide-react';
import { FaWhatsapp } from 'react-icons/fa';
import type { PhoneChannel } from '@utils/whitelistCode';
import {
  whatsappCodeQRValue, openWhatsApp, smsCodeQRValue, openSms, OBSERVER_SMS, SMS_CONSENT,
} from './shared';

const WhitelistQR: React.FC<{ code: string; channel: PhoneChannel }> = ({ code, channel }) => {
  const [copied, setCopied] = useState(false);
  const sms = channel === 'sms';

  const copyCode = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="flex flex-col items-center gap-4">
    <div className={`bg-white p-3 rounded-xl border shadow-sm ${sms ? 'border-purple-300' : 'border-[#25D366]/30'}`}>
      <QRCodeSVG
        value={sms ? smsCodeQRValue(code) : whatsappCodeQRValue(code)}
        size={168}
        level="H"
        includeMargin={false}
        fgColor="#111827"
      />
    </div>

    <button
      onClick={copyCode}
      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-gray-50 border border-gray-200 hover:bg-gray-100 transition-colors"
      title="Copy code"
    >
      <span className="font-mono text-sm font-semibold text-gray-900">{code}</span>
      {copied ? <Check className="h-3.5 w-3.5 text-green-600" /> : <Copy className="h-3.5 w-3.5 text-gray-400" />}
    </button>

    {sms ? (
      <>
        <p className="text-xs text-gray-500 text-center max-w-xs">
          Scan the QR with your phone, or text that code to Observer at{' '}
          <span className="font-mono whitespace-nowrap">{OBSERVER_SMS}</span>.
          It connects SMS and call alerts.
        </p>
        <p className="text-[10px] text-gray-400 text-center max-w-xs leading-snug">{SMS_CONSENT}</p>
      </>
    ) : (
      <p className="text-xs text-gray-500 text-center max-w-xs">
        Scan the QR with your phone, or send that code to Observer yourself on WhatsApp.
        It connects WhatsApp and call alerts.
      </p>
    )}

    <button
      onClick={sms ? () => openSms(code) : openWhatsApp}
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors text-white ${
        sms ? 'bg-purple-600 hover:bg-purple-700' : 'bg-[#25D366] hover:bg-[#1ebe57]'
      }`}
    >
      {sms ? <MessageSquare className="h-3.5 w-3.5" /> : <FaWhatsapp className="h-3.5 w-3.5" />}
      {sms ? 'Open Messages' : 'Open WhatsApp'}
    </button>
    </div>
  );
};

export default WhitelistQR;
