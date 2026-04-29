import React from 'react';

const SENSOR_COLORS: Record<string, string> = {
  // Vision/Multimodal Sensors (Purple Spectrum)
  CAMERA: 'text-purple-700 bg-purple-100',
  SCREEN: 'text-purple-700 bg-purple-100',

  // Visual Memory Hybrid (Purple-Teal)
  IMEMORY: 'text-purple-700 bg-teal-50',

  // Text Extraction/Data (Blue Spectrum)
  SCREEN_OCR: 'text-blue-700 bg-blue-100',
  CAMERA_OCR: 'text-blue-700 bg-blue-100',
  IMEMORY_OCR: 'text-blue-700 bg-teal-50',

  // Audio Sensors (Amber Spectrum)
  MICROPHONE: 'text-amber-700 bg-amber-100',
  SCREEN_AUDIO: 'text-amber-700 bg-amber-100',
  ALL_AUDIO: 'text-orange-700 bg-orange-100',

  // Text Memory (Emerald)
  MEMORY: 'text-emerald-700 bg-emerald-100',

  // Input Sensors (Sky Blue)
  CLIPBOARD_TEXT: 'text-sky-700 bg-sky-100',
  CLIPBOARD: 'text-sky-700 bg-sky-100',
};

const highlightPrompt = (text: string) => {
  const parts = text.split(/(\$[A-Z0-9_@]+)/g);
  return parts.map((part, i) => {
    const match = part.match(/^\$([A-Z0-9_@]+)/);
    if (match) {
      const sensorName = match[1].split('@')[0];
      const colorClass = SENSOR_COLORS[sensorName] || 'text-green-500 bg-green-50';
      return <span key={i} className={`font-semibold ${colorClass}`}>{part}</span>;
    }
    return part;
  });
};

interface SensorInputTextProps {
  value: string;
  onChange: (value: string) => void;
  textareaRef?: React.RefObject<HTMLTextAreaElement>;
  className?: string;
  placeholder?: string;
  rows?: number;
}

const SensorInputText: React.FC<SensorInputTextProps> = ({
  value,
  onChange,
  textareaRef,
  className = '',
  placeholder = '',
  rows,
}) => {
  const overlayRef = React.useRef<HTMLDivElement>(null);

  const handleScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    if (overlayRef.current) {
      overlayRef.current.scrollTop = e.currentTarget.scrollTop;
      overlayRef.current.scrollLeft = e.currentTarget.scrollLeft;
    }
  };

  return (
    <div className={`relative ${className}`}>
      <div
        ref={overlayRef}
        className="absolute inset-0 p-4 font-mono text-sm text-transparent whitespace-pre-wrap pointer-events-none leading-relaxed overflow-hidden"
        aria-hidden="true"
      >
        {highlightPrompt(value)}
      </div>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onScroll={handleScroll}
        rows={rows}
        className="w-full h-full p-4 bg-transparent text-gray-900 caret-blue-500 border border-gray-300 rounded-lg font-mono text-sm resize-none focus:ring-2 focus:ring-blue-500 leading-relaxed"
        placeholder={placeholder}
      />
    </div>
  );
};

export default SensorInputText;
