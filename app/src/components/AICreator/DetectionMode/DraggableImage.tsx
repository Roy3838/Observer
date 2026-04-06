// src/components/AICreator/DetectionMode/DraggableImage.tsx

import React, { useState } from 'react';
import { X } from 'lucide-react';
import type { DraggableImageProps } from './types';

const DraggableImage: React.FC<DraggableImageProps> = ({
  image,
  categoryId,
  onRemove,
  onDragStart
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragStart = (e: React.DragEvent) => {
    setIsDragging(true);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('application/json', JSON.stringify({
      imageId: image.id,
      fromCategoryId: categoryId
    }));
    onDragStart(e, image.id, categoryId);
  };

  const handleDragEnd = () => {
    setIsDragging(false);
  };

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`relative w-20 h-20 rounded-lg overflow-hidden border-2 transition-all duration-150 ${
        isDragging
          ? 'opacity-50 border-purple-400 scale-95'
          : 'border-gray-200 hover:border-purple-300 cursor-grab active:cursor-grabbing'
      }`}
    >
      <img
        src={`data:image/png;base64,${image.data}`}
        alt="Classification sample"
        className="w-full h-full object-cover"
        draggable={false}
      />

      {/* X button overlay */}
      {isHovered && !isDragging && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove(image.id);
          }}
          className="absolute top-1 right-1 w-5 h-5 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center shadow-md transition-colors"
          title="Remove image"
        >
          <X className="w-3 h-3" />
        </button>
      )}

      {/* Source indicator */}
      {image.source === 'history' && (
        <div className="absolute bottom-1 left-1 px-1.5 py-0.5 bg-black/50 rounded text-[10px] text-white">
          history
        </div>
      )}
    </div>
  );
};

export default DraggableImage;
