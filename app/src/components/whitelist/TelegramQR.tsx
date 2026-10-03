// The Telegram counterpart of WhitelistQR: a QR for the bot's `/start <code>` deep link, which
// pairs the chat with the user's Telegram code. Shared by the ask_user_info modal and Settings.
// Laid out exactly like WhitelistQR: QR, copyable code, explainer, open-app button.
import React, { useState } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { Check, Copy } from 'lucide-react';
import { FaTelegramPlane } from 'react-icons/fa';
import { TELEGRAM_BOT, telegramCodeLink } from '@utils/contactInfo';

const TelegramQR: React.FC<{ code: string }> = ({ code }) => {
  const [copied, setCopied] = useState(false);
  const link = telegramCodeLink(code);

  const copyCode = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="bg-white p-3 rounded-xl border shadow-sm border-[#229ED9]/30">
        <QRCodeSVG value={link} size={168} level="H" includeMargin={false} fgColor="#111827" />
      </div>

      <button
        onClick={copyCode}
        className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-gray-50 border border-gray-200 hover:bg-gray-100 transition-colors"
        title="Copy code"
      >
        <span className="font-mono text-sm font-semibold text-gray-900">{code}</span>
        {copied ? <Check className="h-3.5 w-3.5 text-green-600" /> : <Copy className="h-3.5 w-3.5 text-gray-400" />}
      </button>

      <p className="text-xs text-gray-500 text-center max-w-xs">
        Scan the QR with your phone, or open the link and tap <span className="font-medium">Start</span> in the
        chat with @{TELEGRAM_BOT}. Your agents can then alert that chat, and you can message Observer from it.
      </p>

      <a
        href={link}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors bg-[#229ED9] text-white hover:bg-[#1c8bbf]"
      >
        <FaTelegramPlane className="h-3.5 w-3.5" />
        Open Telegram
      </a>
    </div>
  );
};

export default TelegramQR;
