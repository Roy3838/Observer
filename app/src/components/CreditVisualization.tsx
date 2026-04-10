// src/components/CreditVisualization.tsx

import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import { Info, X, Clock, Zap, TrendingUp } from 'lucide-react';

interface CreditVisualizationProps {
  /** Number of daily credits for this tier */
  dailyCredits: number;
  /** Tier name for display */
  tierName?: string;
  /** Compact mode for inline display */
  compact?: boolean;
}

// Anchor points for piecewise linear interpolation
const INTERVAL_ANCHORS = [30, 60, 180, 1800]; // seconds
const ANCHOR_LABELS = ['30s', '1 min', '3 min', '30 min'];
const SLIDER_MAX = 300; // 100 units per segment for smooth dragging

// Convert slider position (0-300) to seconds using piecewise linear interpolation
const sliderToSeconds = (sliderValue: number): number => {
  const segmentSize = SLIDER_MAX / (INTERVAL_ANCHORS.length - 1); // 100
  const segmentIndex = Math.min(
    Math.floor(sliderValue / segmentSize),
    INTERVAL_ANCHORS.length - 2
  );
  const segmentProgress = (sliderValue - segmentIndex * segmentSize) / segmentSize;

  const startVal = INTERVAL_ANCHORS[segmentIndex];
  const endVal = INTERVAL_ANCHORS[segmentIndex + 1];

  return startVal + (endVal - startVal) * segmentProgress;
};

// Format seconds to human-readable string
const formatInterval = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3600) {
    const mins = seconds / 60;
    return mins === Math.floor(mins) ? `${mins} min` : `${mins.toFixed(1)} min`;
  }
  return `${(seconds / 60).toFixed(0)} min`;
};

export const CreditVisualization: React.FC<CreditVisualizationProps> = ({
  dailyCredits,
  tierName = 'Your plan',
  compact = false,
}) => {
  const [sliderValue, setSliderValue] = useState(100); // Start at 1 min (first segment boundary)

  const intervalSeconds = sliderToSeconds(sliderValue);
  const monitoringSeconds = dailyCredits * intervalSeconds;
  const monitoringHours = monitoringSeconds / 3600;

  // Format hours nicely
  const formatHours = (hours: number): string => {
    if (hours >= 24) return '24+ hours (all day!)';
    if (hours === Math.floor(hours)) return `${hours} hour${hours !== 1 ? 's' : ''}`;
    return `${hours.toFixed(1)} hours`;
  };

  // Calculate progress bar width (capped at 24 hours = 100%)
  const progressPercent = Math.min((monitoringHours / 24) * 100, 100);

  if (compact) {
    return (
      <div className="text-xs text-gray-600">
        <span className="font-medium">{dailyCredits} credits</span>
        <span className="mx-1">=</span>
        <span>{formatHours(monitoringHours)} at {formatInterval(intervalSeconds)} intervals</span>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-xl p-4 sm:p-5 border border-gray-200">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <Zap className="h-5 w-5 text-purple-500" />
        <h3 className="font-semibold text-gray-800">How Cloud Monitoring Works</h3>
      </div>

      {/* Explanation */}
      <p className="text-sm text-gray-600 mb-4">
        <strong>{tierName}</strong> includes <strong>{dailyCredits} credits/day</strong>.
        Each credit = 1 agent loop. Adjust your loop interval to control monitoring duration:
      </p>

      {/* Interval Slider */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Loop Interval</span>
          <span className="text-sm font-bold text-purple-600 bg-purple-100 px-2 py-0.5 rounded">
            {formatInterval(intervalSeconds)}
          </span>
        </div>

        <input
          type="range"
          min={0}
          max={SLIDER_MAX}
          value={sliderValue}
          onChange={(e) => setSliderValue(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
        />

        <div className="flex justify-between text-xs text-gray-500 mt-1">
          {ANCHOR_LABELS.map((label, idx) => {
            const anchorSliderValue = idx * (SLIDER_MAX / (INTERVAL_ANCHORS.length - 1));
            const isNearAnchor = Math.abs(sliderValue - anchorSliderValue) < 10;
            return (
              <span
                key={label}
                className={isNearAnchor ? 'text-purple-600 font-medium' : ''}
              >
                {label}
              </span>
            );
          })}
        </div>
      </div>

      {/* Result Display */}
      <div className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-blue-500" />
            <span className="text-sm text-gray-600">Daily Monitoring Time</span>
          </div>
          <span className="text-lg font-bold text-gray-800">
            {formatHours(monitoringHours)}
          </span>
        </div>

        {/* Progress bar showing portion of 24-hour day */}
        <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full transition-all duration-300"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0h</span>
          <span>12h</span>
          <span>24h</span>
        </div>
      </div>

      {/* Tip */}
      <div className="mt-4 flex items-start gap-2 text-xs text-gray-500">
        <TrendingUp className="h-4 w-4 text-green-500 flex-shrink-0 mt-0.5" />
        <span>
          <strong className="text-gray-700">Pro tip:</strong> Longer intervals = more coverage.
          Drag the slider to find your ideal balance between frequency and monitoring duration.
        </span>
      </div>
    </div>
  );
};

// ============================================
// Info Icon Button with Popover/Modal
// ============================================

interface CreditInfoButtonProps {
  /** Number of daily credits */
  dailyCredits: number;
  /** Tier name */
  tierName?: string;
  /** Size of the info icon */
  size?: 'sm' | 'md';
  /** Additional class for positioning */
  className?: string;
}

export const CreditInfoButton: React.FC<CreditInfoButtonProps> = ({
  dailyCredits,
  tierName,
  size = 'sm',
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const iconSize = size === 'sm' ? 'h-3.5 w-3.5' : 'h-4 w-4';

  return (
    <>
      {/* Info Icon Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={`inline-flex items-center justify-center text-gray-400 hover:text-purple-600 transition-colors ${className}`}
        aria-label="Learn how credits work"
      >
        <Info className={iconSize} />
      </button>

      {/* Modal Overlay - rendered via Portal to escape stacking contexts */}
      {isOpen && createPortal(
        <div
          className="fixed inset-0 bg-black/40 flex items-center justify-center z-[10001] backdrop-blur-sm p-4"
          onClick={() => setIsOpen(false)}
        >
          <div
            className="relative bg-white rounded-2xl shadow-xl max-w-md w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={() => setIsOpen(false)}
              className="absolute top-3 right-3 text-gray-400 hover:text-gray-700 transition-colors z-10"
            >
              <X className="h-5 w-5" />
            </button>

            {/* Content */}
            <div className="p-5">
              <CreditVisualization
                dailyCredits={dailyCredits}
                tierName={tierName}
              />
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
};

export default CreditVisualization;
