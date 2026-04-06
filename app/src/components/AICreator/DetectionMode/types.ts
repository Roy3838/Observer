// src/components/AICreator/DetectionMode/types.ts

import type { TokenProvider } from '@utils/main_loop';

export interface ClassifiedImage {
  id: string;
  data: string; // base64 image data
  source: 'upload' | 'history';
  sourceAgentId?: string; // If from history, which agent
  timestamp?: string;
}

export interface DetectionCategory {
  id: string;
  label: string; // e.g., "CLEAN", "SPAGHETTI"
  images: ClassifiedImage[];
}

export interface DetectionModeState {
  categories: DetectionCategory[];
  selectedHistoryAgentId: string | null;
}

export interface DetectionModeInitialData {
  categories: DetectionCategory[];
  sourceAgentId: string;
}

// Re-export TestResult from finetuneClassifier for convenience
export type { TestResult } from '@utils/finetuneClassifier';

export interface FinetuneState {
  phase: 'idle' | 'testing' | 'improving' | 'complete' | 'failed';
  iteration: number;
  maxIterations: number;
  currentPrompt: string;
  previousPrompt?: string;  // For showing what changed during improvement
  testResults: Array<{
    imageIndex: number;
    expected: string;
    actual: string;
    passed: boolean;
  }>;
  currentTestIndex: number;
  totalTests: number;
}

export interface DetectionModePanelProps {
  onClose: () => void;
  onComplete: (agentBlock: string) => void;  // Returns $$$ formatted block
  initialData?: DetectionModeInitialData;
  // Props for API access
  getToken: TokenProvider;
  isUsingObServer: boolean;
  selectedLocalModel: string;
}

export interface ClassificationColumnsProps {
  categories: DetectionCategory[];
  onMoveImage: (imageId: string, fromCategoryId: string, toCategoryId: string) => void;
  onRemoveImage: (imageId: string, categoryId: string) => void;
  onUpdateLabel: (categoryId: string, newLabel: string) => void;
  onAddImage?: (image: ClassifiedImage, toCategoryId: string) => void;  // For drops from outside (history)
}

export interface DraggableImageProps {
  image: ClassifiedImage;
  categoryId: string;
  onRemove: (imageId: string) => void;
  onDragStart: (e: React.DragEvent, imageId: string, categoryId: string) => void;
}
