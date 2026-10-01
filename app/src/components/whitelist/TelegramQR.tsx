// The Telegram counterpart of WhitelistQR: a QR for the bot's `/start <code>` deep link, which
// pairs the chat with the user's Telegram code. Shared by the ask_user_info modal and Settings.
import React from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { ExternalLink } from 'lucide-react';
import { TELEGRAM_BOT, telegramCodeLink } from '@utils/contactInfo';

const TelegramQR: React.FC<{ code: string }> = ({ code }) => {
  const link = telegramCodeLink(code);
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="bg-white p-3 rounded-xl border border-gray-200 shadow-sm">
        <QRCodeSVG value={link} size={168} level="H" includeMargin={false} fgColor="#111827" />
      </div>
      <a
        href={link}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-gray-900 text-white rounded text-xs font-medium hover:bg-black transition-colors"
      >
        Open in Telegram <ExternalLink className="h-3 w-3" />
      </a>
      <p className="text-xs text-gray-500 text-center max-w-xs">
        Scan or open, then tap <span className="font-medium">Start</span> in the chat with @{TELEGRAM_BOT}.
        Your agents can then alert that chat, and you can message Observer from it.
      </p>
    </div>
  );
};

export default TelegramQR;
