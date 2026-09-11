// src/components/SeatGrid.tsx
//
// One square per purchased seat — filled (occupied) vs open. Seats are
// binary (a seat is either claimed or not), unlike usage-based metrics,
// so this is a simple occupancy grid, not a heatmap.

import React from 'react';

interface SeatGridProps {
  filled: number;
  total: number;
}

export const SeatGrid: React.FC<SeatGridProps> = ({ filled, total }) => {
  if (total <= 0) return null;
  const seats = Array.from({ length: total }, (_, i) => i < filled);

  return (
    <div className="flex flex-wrap gap-1.5">
      {seats.map((isFilled, i) => (
        <div
          key={i}
          title={isFilled ? `Seat ${i + 1}: filled` : `Seat ${i + 1}: open`}
          className={`h-3 w-3 rounded-sm ${
            isFilled
              ? 'bg-gray-900 dark:bg-gray-100'
              : 'bg-gray-100 dark:bg-gray-800'
          }`}
        />
      ))}
    </div>
  );
};

export default SeatGrid;
