import React from 'react';

interface PieTimerProps {
  progress: number;      // 0-100
  color: 'green' | 'blue' | 'orange';
  totalDurationMs?: number;
  isFilling?: boolean;   // true = filling (WAITING), false = draining (SLEEPING)
  size?: number;         // default: 20
}

const PieTimer: React.FC<PieTimerProps> = ({ progress, color, totalDurationMs, isFilling = true, size = 20 }) => {
  const strokeWidth = 2;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (progress / 100) * circumference;

  const strokeColor = color === 'green' ? 'stroke-green-500' : color === 'orange' ? 'stroke-orange-400' : 'stroke-blue-500';
  const textColor = color === 'green' ? 'fill-green-600' : color === 'orange' ? 'fill-orange-500' : 'fill-blue-600';

  let timeDisplay = '';
  if (totalDurationMs) {
    let remainingSeconds = 0;
    if (isFilling) {
      remainingSeconds = Math.ceil((totalDurationMs * (100 - progress)) / 100 / 1000);
    } else {
      remainingSeconds = Math.ceil((totalDurationMs * progress) / 100 / 1000);
    }
    const minutes = Math.floor(remainingSeconds / 60);
    const seconds = remainingSeconds % 60;
    timeDisplay = minutes > 0 ? `${minutes}:${seconds.toString().padStart(2, '0')}` : `${seconds}`;
  }

  return (
    <svg width={size} height={size} className="flex-shrink-0 -rotate-90">
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="currentColor" strokeWidth={strokeWidth} className="stroke-gray-200" />
      <circle
        cx={size / 2} cy={size / 2} r={radius} fill="none"
        strokeWidth={strokeWidth} strokeDasharray={circumference} strokeDashoffset={offset}
        strokeLinecap="round" className={strokeColor}
        style={{ transition: 'stroke-dashoffset 0.1s linear' }}
      />
      {totalDurationMs && timeDisplay && (
        <text x={size / 2} y={size / 2} textAnchor="middle" dominantBaseline="central"
          className={`text-[7px] font-semibold ${textColor} rotate-90`}
          style={{ transform: 'rotate(90deg)', transformOrigin: 'center' }}>
          {timeDisplay}
        </text>
      )}
    </svg>
  );
};

export default PieTimer;
