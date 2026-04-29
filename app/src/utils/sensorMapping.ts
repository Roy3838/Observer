import { PseudoStreamType } from './streamManager';

// --- Sensor to Stream Mapping ---
// This mapping defines which streams are required for each sensor placeholder
// Used by both main_loop.ts (for agents) and SharingPermissionsModal (for UI)

export type SensorPlaceholder = '$SCREEN' | '$SCREEN_64' | '$SCREEN_OCR' | '$CAMERA' | '$CAMERA_OCR' | '$SCREEN_AUDIO' | '$MICROPHONE' | '$ALL_AUDIO';

export const SENSOR_STREAM_MAP: Record<SensorPlaceholder, PseudoStreamType> = {
  '$SCREEN': 'screenVideo',
  '$SCREEN_64': 'screenVideo',
  '$SCREEN_OCR': 'screenVideo',
  '$CAMERA': 'camera',
  '$CAMERA_OCR': 'camera',
  '$SCREEN_AUDIO': 'screenAudio',
  '$MICROPHONE': 'microphone',
  '$ALL_AUDIO': 'allAudio'
};

// Human-readable descriptions for UI
export const SENSOR_DESCRIPTIONS: Record<SensorPlaceholder, { name: string; description: string; icon: string }> = {
  '$SCREEN': {
    name: 'Screen Capture',
    description: 'Captures screenshots as images for multimodal models',
    icon: 'Monitor'
  },
  '$SCREEN_64': {
    name: 'Screen Capture',
    description: 'Captures screenshots as images for multimodal models (legacy name)',
    icon: 'Monitor'
  },
  '$SCREEN_OCR': {
    name: 'Screen Text (OCR)',
    description: 'Extracts text from screen using optical character recognition',
    icon: 'FileText'
  },
  '$CAMERA': {
    name: 'Camera',
    description: 'Captures video from your camera/webcam',
    icon: 'Camera'
  },
  '$CAMERA_OCR': {
    name: 'Camera Text (OCR)',
    description: 'Extracts text from camera feed using optical character recognition',
    icon: 'ScanText'
  },
  '$SCREEN_AUDIO': {
    name: 'System Audio',
    description: 'Captures audio from system (apps, browser, etc.)',
    icon: 'Volume2'
  },
  '$MICROPHONE': {
    name: 'Microphone',
    description: 'Captures audio from your microphone',
    icon: 'Mic'
  },
  '$ALL_AUDIO': {
    name: 'Mixed Audio',
    description: 'Combines system audio + microphone into single stream',
    icon: 'Headphones'
  }
};

/**
 * Extract required streams from sensor placeholders
 * Used by main_loop.ts for agent stream requirements
 */
export function getRequiredStreamsFromSensors(sensors: SensorPlaceholder[]): PseudoStreamType[] {
  return sensors.map(sensor => SENSOR_STREAM_MAP[sensor]);
}

/**
 * Extract sensor placeholders from agent system prompt
 * Used by main_loop.ts to determine agent stream requirements
 */
export function extractSensorsFromPrompt(systemPrompt: string): SensorPlaceholder[] {
  return Object.keys(SENSOR_STREAM_MAP)
    .filter(placeholder => new RegExp(`\\${placeholder}(?![A-Z_])`).test(systemPrompt)) as SensorPlaceholder[];
}