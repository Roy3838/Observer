// src/utils/settings.ts

// NOTE: No imports are needed from your config files anymore.
import { WhisperSettings } from './whisper/types';
import { getDefaultWhisperSettings } from '../config/whisper-models';

class SettingsManager {
    // --- PRIVATE CONSTANTS FOR LOCALSTORAGE KEYS ---
    private readonly OCR_CONFIDENCE_KEY = 'observer-ai:settings:ocrConfidenceThreshold';
    private readonly OCR_WORKER_PATH_KEY = 'observer-ai:settings:ocrWorkerPath';
    private readonly OCR_LANG_PATH_KEY = 'observer-ai:settings:ocrLangPath';
    private readonly OCR_CORE_PATH_KEY = 'observer-ai:settings:ocrCorePath';
    private readonly OCR_LANGUAGE_KEY = 'observer-ai:settings:ocrLanguage';
    
    // --- WHISPER SETTINGS KEYS ---
    private readonly WHISPER_SETTINGS_KEY = 'observer-ai:settings:whisperSettings';

    // --- SENSIBLE DEFAULTS ---
    private readonly DEFAULTS = {
        ocrConfidenceThreshold: 50,
        ocrWorkerPath: 'https://unpkg.com/tesseract.js@6.0.0/dist/worker.min.js',
        ocrLangPath: 'https://tessdata.projectnaptha.com/4.0.0',
        ocrCorePath: 'https://unpkg.com/tesseract.js-core@4.0.2/tesseract-core.wasm.js',
        ocrLanguage: 'eng',
        whisperSettings: getDefaultWhisperSettings()
    };

    // --- GETTER AND SETTER FUNCTIONS ---

    // Tesseract.js - OCR
    public getOcrConfidenceThreshold(): number {
        const stored = localStorage.getItem(this.OCR_CONFIDENCE_KEY);
        return stored ? parseFloat(stored) : this.DEFAULTS.ocrConfidenceThreshold;
    }
    public setOcrConfidenceThreshold(value: number): void {
        localStorage.setItem(this.OCR_CONFIDENCE_KEY, String(value));
    }
    public getOcrWorkerPath(): string {
        return localStorage.getItem(this.OCR_WORKER_PATH_KEY) ?? this.DEFAULTS.ocrWorkerPath;
    }
    public setOcrWorkerPath(value: string): void {
        localStorage.setItem(this.OCR_WORKER_PATH_KEY, value);
    }
    public getOcrLangPath(): string {
        return localStorage.getItem(this.OCR_LANG_PATH_KEY) ?? this.DEFAULTS.ocrLangPath;
    }
    public setOcrLangPath(value: string): void {
        localStorage.setItem(this.OCR_LANG_PATH_KEY, value);
    }
    public getOcrCorePath(): string {
        return localStorage.getItem(this.OCR_CORE_PATH_KEY) ?? this.DEFAULTS.ocrCorePath;
    }
    public setOcrCorePath(value: string): void {
        localStorage.setItem(this.OCR_CORE_PATH_KEY, value);
    }
    public getOcrLanguage(): string {
        return localStorage.getItem(this.OCR_LANGUAGE_KEY) ?? this.DEFAULTS.ocrLanguage;
    }
    public setOcrLanguage(value: string): void {
        localStorage.setItem(this.OCR_LANGUAGE_KEY, value);
    }

    // Whisper Settings
    public getWhisperSettings(): WhisperSettings {
        const stored = localStorage.getItem(this.WHISPER_SETTINGS_KEY);
        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                // Migrate old settings format if needed
                if (parsed.modelSize && parsed.language && !parsed.modelId) {
                    return this.migrateOldWhisperSettings(parsed);
                }
                return { ...this.DEFAULTS.whisperSettings, ...parsed };
            } catch (error) {
                console.warn('Failed to parse whisper settings, using defaults:', error);
                return this.DEFAULTS.whisperSettings;
            }
        }
        return this.DEFAULTS.whisperSettings;
    }

    private migrateOldWhisperSettings(oldSettings: any): WhisperSettings {
        // Migrate old format to new direct configuration
        const isEnglishOnly = oldSettings.language === 'en';
        const modelId = isEnglishOnly 
            ? `Xenova/whisper-${oldSettings.modelSize}.en`
            : `Xenova/whisper-${oldSettings.modelSize}`;
        
        const newSettings: WhisperSettings = {
            modelId,
            quantized: oldSettings.quantized || true,
            chunkDurationMs: oldSettings.chunkDurationMs || 15000
        };
        
        // Add task/language for multilingual models
        if (!isEnglishOnly) {
            newSettings.task = 'transcribe';
        }
        
        // Save migrated settings
        this.setWhisperSettings(newSettings);
        console.info('Migrated old whisper settings to new format:', modelId);
        
        return newSettings;
    }

    public setWhisperSettings(settings: WhisperSettings): void {
        localStorage.setItem(this.WHISPER_SETTINGS_KEY, JSON.stringify(settings));
    }

    public setWhisperModelId(modelId: string): void {
        const settings = this.getWhisperSettings();
        settings.modelId = modelId;
        this.setWhisperSettings(settings);
    }

    public setWhisperTask(task?: 'transcribe' | 'translate'): void {
        const settings = this.getWhisperSettings();
        if (task) {
            settings.task = task;
        } else {
            delete settings.task;
        }
        this.setWhisperSettings(settings);
    }

    public setWhisperLanguage(language?: string): void {
        const settings = this.getWhisperSettings();
        if (language) {
            settings.language = language;
        } else {
            delete settings.language;
        }
        this.setWhisperSettings(settings);
    }

    public setWhisperQuantized(quantized: boolean): void {
        const settings = this.getWhisperSettings();
        settings.quantized = quantized;
        this.setWhisperSettings(settings);
    }


    public setWhisperChunkDuration(durationMs: number): void {
        if (durationMs < 5000 || durationMs > 60000) {
            throw new Error('Chunk duration must be between 5-60 seconds');
        }
        const settings = this.getWhisperSettings();
        settings.chunkDurationMs = durationMs;
        this.setWhisperSettings(settings);
    }
}

// Export a single instance
export const SensorSettings = new SettingsManager();
