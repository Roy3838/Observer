// src/components/AICreator/DetectionMode/ClassificationColumns.tsx

import React, { useState } from 'react';
import { Edit2, Check } from 'lucide-react';
import DraggableImage from './DraggableImage';
import type { ClassificationColumnsProps, DetectionCategory } from './types';

interface ColumnProps {
  category: DetectionCategory;
  colorClass: string;
  onDrop: (imageId: string, fromCategoryId: string) => void;
  onAddImage?: (imageData: string) => void;  // For drops from history
  onRemoveImage: (imageId: string) => void;
  onUpdateLabel: (newLabel: string) => void;
  onDragStart: (e: React.DragEvent, imageId: string, categoryId: string) => void;
}

const Column: React.FC<ColumnProps> = ({
  category,
  colorClass,
  onDrop,
  onAddImage,
  onRemoveImage,
  onUpdateLabel,
  onDragStart
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isEditingLabel, setIsEditingLabel] = useState(false);
  const [labelValue, setLabelValue] = useState(category.label);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    // Only set to false if we're leaving the column entirely
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragOver(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    try {
      const data = JSON.parse(e.dataTransfer.getData('application/json'));

      // Check if this is a drop from history (has imageData but fromCategoryId is 'history')
      if (data.fromCategoryId === 'history' && data.imageData && onAddImage) {
        onAddImage(data.imageData);
      } else if (data.imageId && data.fromCategoryId !== category.id) {
        // Normal move between categories
        onDrop(data.imageId, data.fromCategoryId);
      }
    } catch (err) {
      console.error('Drop parse error:', err);
    }
  };

  const handleLabelSubmit = () => {
    if (labelValue.trim()) {
      onUpdateLabel(labelValue.trim().toUpperCase());
    } else {
      setLabelValue(category.label);
    }
    setIsEditingLabel(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleLabelSubmit();
    } else if (e.key === 'Escape') {
      setLabelValue(category.label);
      setIsEditingLabel(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col">
      {/* Column Header */}
      <div className={`flex items-center justify-center gap-2 px-3 py-2 rounded-t-lg ${colorClass}`}>
        {isEditingLabel ? (
          <div className="flex items-center gap-1">
            <input
              type="text"
              value={labelValue}
              onChange={(e) => setLabelValue(e.target.value)}
              onBlur={handleLabelSubmit}
              onKeyDown={handleKeyDown}
              className="w-24 px-2 py-0.5 text-sm font-semibold text-center rounded border-2 border-white/50 bg-white/20 text-white placeholder-white/50 focus:outline-none focus:border-white"
              autoFocus
            />
            <button
              onClick={handleLabelSubmit}
              className="p-0.5 rounded hover:bg-white/20"
            >
              <Check className="w-4 h-4 text-white" />
            </button>
          </div>
        ) : (
          <>
            <span className="font-semibold text-white text-sm">{category.label}</span>
            <button
              onClick={() => setIsEditingLabel(true)}
              className="p-0.5 rounded hover:bg-white/20"
              title="Edit label"
            >
              <Edit2 className="w-3 h-3 text-white/70 hover:text-white" />
            </button>
          </>
        )}
        <span className="text-white/70 text-xs">({category.images.length})</span>
      </div>

      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`flex-1 min-h-[140px] p-3 rounded-b-lg border-2 transition-all duration-150 ${
          isDragOver
            ? 'border-solid border-purple-400 bg-purple-50'
            : category.images.length === 0
            ? 'border-dashed border-gray-300 bg-gray-50'
            : 'border-solid border-gray-200 bg-white'
        }`}
      >
        {category.images.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <p className="text-gray-400 text-sm text-center">
              {isDragOver ? 'Drop here' : 'Drag images here'}
            </p>
          </div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {category.images.map((image) => (
              <DraggableImage
                key={image.id}
                image={image}
                categoryId={category.id}
                onRemove={onRemoveImage}
                onDragStart={onDragStart}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const ClassificationColumns: React.FC<ClassificationColumnsProps> = ({
  categories,
  onMoveImage,
  onRemoveImage,
  onUpdateLabel,
  onAddImage
}) => {
  const colorClasses = [
    'bg-green-500', // First category (e.g., CLEAN)
    'bg-red-500',   // Second category (e.g., SPAGHETTI)
    'bg-blue-500',  // Third category if added
    'bg-yellow-500' // Fourth category if added
  ];

  const handleDragStart = (_e: React.DragEvent, _imageId: string, _categoryId: string) => {
    // Optional: Add visual feedback or tracking
  };

  return (
    <div className="flex gap-4">
      {categories.map((category, index) => (
        <Column
          key={category.id}
          category={category}
          colorClass={colorClasses[index % colorClasses.length]}
          onDrop={(imageId, fromCategoryId) => onMoveImage(imageId, fromCategoryId, category.id)}
          onAddImage={onAddImage ? (imageData) => onAddImage({
            id: `history-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            data: imageData,
            source: 'history'
          }, category.id) : undefined}
          onRemoveImage={(imageId) => onRemoveImage(imageId, category.id)}
          onUpdateLabel={(newLabel) => onUpdateLabel(category.id, newLabel)}
          onDragStart={handleDragStart}
        />
      ))}
    </div>
  );
};

export default ClassificationColumns;
