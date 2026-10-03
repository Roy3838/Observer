// Shared by the ask_user_info modal and Settings: QR, copyable code, open-app button.
// WhatsApp is the only way to pair a phone: sending the code there enables WhatsApp, SMS
// and call alerts to that phone.
import React, { useState } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { Check, Copy } from 'lucide-react';
import { FaWhatsapp } from 'react-icons/fa';
import { whatsappCodeQRValue, openWhatsApp } from './shared';

const WhitelistQR: React.FC<{ code: string }> = ({ code }) => {
  const [copied, setCopied] = useState(false);

  const copyCode = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="flex flex-col items-center gap-4">
    <div className="bg-white p-3 rounded-xl border shadow-sm border-[#25D366]/30">
      <QRCodeSVG
        value={whatsappCodeQRValue(code)}
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

    <p className="text-xs text-gray-500 text-center max-w-xs">
      Scan the QR with your phone, or send that code to Observer yourself on WhatsApp.
      It connects WhatsApp, SMS and call alerts.
    </p>

    <button
      onClick={openWhatsApp}
      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors bg-[#25D366] text-white hover:bg-[#1ebe57]"
    >
      <FaWhatsapp className="h-3.5 w-3.5" />
      Open WhatsApp
    </button>
    </div>
  );
};

export default WhitelistQR;
