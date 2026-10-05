// src/utils/settings.ts

// NOTE: No imports are needed from your config files anymore.
import { WhisperSettings, TranscriptionMode, WhisperDevice } from './whisper/types';
import { getDefaultWhisperSettings, migrateWhisperModelId } from '../config/whisper-models';
import { generateWhitelistCode, type PhoneChannel } from './whitelistCode';
import type { WhitelistChannel } from './logging';

class SettingsManager {
    // --- PRIVATE CONSTANTS FOR LOCALSTORAGE KEYS ---
    private readonly OCR_WORKER_PATH_KEY = 'observer-ai:settings:ocrWorkerPath';
    private readonly OCR_LANG_PATH_KEY = 'observer-ai:settings:ocrLangPath';
    private readonly OCR_CORE_PATH_KEY = 'observer-ai:settings:ocrCorePath';
    private readonly OCR_LANGUAGE_KEY = 'observer-ai:settings:ocrLanguage';
    
    // --- WHISPER SETTINGS KEYS ---
    private readonly WHISPER_SETTINGS_KEY = 'observer-ai:settings:whisperSettings';
    private readonly TRANSCRIPTION_MODE_KEY = 'observer-ai:settings:transcriptionMode';
    private readonly SELF_HOSTED_WHISPER_URL_KEY = 'observer-ai:settings:selfHostedWhisperUrl';

    // --- SENSIBLE DEFAULTS ---
    private readonly DEFAULTS = {
        ocrWorkerPath: 'https://unpkg.com/tesseract.js@6.0.0/dist/worker.min.js',
        ocrLangPath: 'https://tessdata.projectnaptha.com/4.0.0',
        ocrCorePath: 'https://unpkg.com/tesseract.js-core@4.0.2/tesseract-core.wasm.js',
        ocrLanguage: 'eng',
        whisperSettings: getDefaultWhisperSettings(),
        transcriptionMode: 'cloud' as TranscriptionMode
    };

    // --- GETTER AND SETTER FUNCTIONS ---

    // Tesseract.js - OCR
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
                const settings = { ...this.DEFAULTS.whisperSettings, ...parsed };
                // Migrate legacy Xenova/* model IDs to their onnx-community/* equivalents
                const migratedModelId = migrateWhisperModelId(settings.modelId);
                if (migratedModelId !== settings.modelId) {
                    settings.modelId = migratedModelId;
                    this.setWhisperSettings(settings);
                }
                return settings;
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
            ? `onnx-community/whisper-${oldSettings.modelSize}.en`
            : `onnx-community/whisper-${oldSettings.modelSize}`;

        const newSettings: WhisperSettings = {
            modelId,
            quantized: oldSettings.quantized || true,
            chunkDurationMs: oldSettings.chunkDurationMs || 5000,
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

    public setWhisperDevice(device: WhisperDevice): void {
        const settings = this.getWhisperSettings();
        settings.device = device;
        this.setWhisperSettings(settings);
    }


    public setWhisperChunkDuration(durationMs: number): void {
        if (durationMs < 1000 || durationMs > 60000) {
            throw new Error('Chunk duration must be between 1-60 seconds');
        }
        const settings = this.getWhisperSettings();
        settings.chunkDurationMs = durationMs;
        this.setWhisperSettings(settings);
    }

    // Transcription Mode (cloud vs self-hosted vs local)
    public getTranscriptionMode(): TranscriptionMode {
        const stored = localStorage.getItem(this.TRANSCRIPTION_MODE_KEY);
        if (stored === 'cloud' || stored === 'self-hosted' || stored === 'local') {
            return stored;
        }
        return this.DEFAULTS.transcriptionMode;
    }

    public setTranscriptionMode(mode: TranscriptionMode): void {
        localStorage.setItem(this.TRANSCRIPTION_MODE_KEY, mode);
    }

    // Self-Hosted Whisper URL
    public getSelfHostedWhisperUrl(): string {
        return localStorage.getItem(this.SELF_HOSTED_WHISPER_URL_KEY) ?? '';
    }

    public setSelfHostedWhisperUrl(url: string): void {
        localStorage.setItem(this.SELF_HOSTED_WHISPER_URL_KEY, url);
    }

    // MCP "tutorial mode" — set right before a single send() from the RecipeSplash tutorial
    // path so capture_screen / list_screen_targets hand the model a synthetic progress-bar
    // stream instead of opening the real screen picker. In-memory only (not localStorage):
    // it must never survive a reload or leak into a later, unrelated chat turn, so each
    // read consumes it (see isMcpTutorialMode).
    private mcpTutorialMode = false;

    public setMcpTutorialMode(value: boolean): void {
        this.mcpTutorialMode = value;
    }

    public isMcpTutorialMode(): boolean {
        return this.mcpTutorialMode;
    }

    // One-shot consume: call once a tool has committed to using (or not using) the
    // tutorial stream, so the flag can't leak into the next request from the same session.
    public consumeMcpTutorialMode(): boolean {
        const value = this.mcpTutorialMode;
        this.mcpTutorialMode = false;
        return value;
    }

    // --- DESKTOP SCREEN CAPTURE QUALITY ---
    // Tunable max width / JPEG quality / FPS, pushed to the Rust capture backend
    // (sc_set_capture_config) right before each capture starts. Defaults = "Low" tier:
    // light and fast, sharper-per-pixel than the old build thanks to pixel-correct sizing.
    private readonly CAPTURE_QUALITY_KEY = 'observer-ai:settings:captureQuality';
    private readonly CAPTURE_QUALITY_DEFAULTS = { maxWidth: 1280, jpegQuality: 55, fps: 10 };

    public getCaptureQuality(): { maxWidth: number; jpegQuality: number; fps: number } {
        const stored = localStorage.getItem(this.CAPTURE_QUALITY_KEY);
        if (!stored) return { ...this.CAPTURE_QUALITY_DEFAULTS };
        try {
            return { ...this.CAPTURE_QUALITY_DEFAULTS, ...JSON.parse(stored) };
        } catch {
            return { ...this.CAPTURE_QUALITY_DEFAULTS };
        }
    }

    public setCaptureQuality(value: { maxWidth: number; jpegQuality: number; fps: number }): void {
        localStorage.setItem(this.CAPTURE_QUALITY_KEY, JSON.stringify(value));
    }

    // --- NOTIFICATION CONTACTS ---
    // Remembered so the `ask_user_info` modal can prefill instead of making the user
    // re-hunt their Discord webhook / Telegram chat_id on every agent they build. Stored
    // per-kind under one JSON blob. These same values already end up baked into agent code
    // in IndexedDB, and code_sanitizer.ts redacts them when an agent is shared.
    private readonly NOTIFICATION_CONTACTS_KEY = 'observer-ai:settings:notificationContacts';

    private readAllContacts(): Record<string, string> {
        const stored = localStorage.getItem(this.NOTIFICATION_CONTACTS_KEY);
        if (!stored) return {};
        try {
            const parsed = JSON.parse(stored);
            return parsed && typeof parsed === 'object' ? parsed : {};
        } catch {
            return {};
        }
    }

    /** Last value the user confirmed for this contact kind, or '' if none. */
    public getNotificationContact(kind: string): string {
        const v = this.readAllContacts()[kind];
        return typeof v === 'string' ? v : '';
    }

    public setNotificationContact(kind: string, value: string): void {
        const all = this.readAllContacts();
        const v = value.trim();
        if (v) all[kind] = v; else delete all[kind];
        localStorage.setItem(this.NOTIFICATION_CONTACTS_KEY, JSON.stringify(all));
    }

    public clearNotificationContacts(): void {
        localStorage.removeItem(this.NOTIFICATION_CONTACTS_KEY);
    }

    // --- CONTACT CODES ---
    // Stable codes (e.g. "tree-book-shower-golden") shown in the golden-path QRs, one per
    // channel, because pairing each one is its own opt-in. The SMS code (texted to our number)
    // reaches the phone by SMS and calls; the WhatsApp code by WhatsApp and calls. Both are tied
    // to the account. The Telegram code reaches the chat and works without one. Generated once
    // and only replaced on rotate: agent code bakes them in literally, so if one changed, every
    // agent built against the old code would silently stop working. Pairing a new code from
    // the same phone/chat revokes the old one server-side.
    private readonly WHITELIST_CODE_KEY = 'observer-ai:settings:whitelistCode';
    private readonly SMS_CODE_KEY = 'observer-ai:settings:smsCode';
    private readonly TELEGRAM_CODE_KEY = 'observer-ai:settings:telegramCode';

    private ensureCode(key: string): string {
        return localStorage.getItem(key) ?? this.rotateCode(key);
    }

    private rotateCode(key: string): string {
        const code = generateWhitelistCode();
        localStorage.setItem(key, code);
        return code;
    }

    /** The persisted WhatsApp code, generating and storing one on first use. */
    public ensureWhitelistCode(): string { return this.ensureCode(this.WHITELIST_CODE_KEY); }

    /** The persisted WhatsApp code, or null if one hasn't been generated yet. */
    public getWhitelistCode(): string | null { return localStorage.getItem(this.WHITELIST_CODE_KEY); }

    /** Mints a fresh WhatsApp code and overwrites the stored one, for "rotate to a new contact". */
    public rotateWhitelistCode(): string { return this.rotateCode(this.WHITELIST_CODE_KEY); }

    private phoneCodeKey(channel: PhoneChannel): string {
        return channel === 'sms' ? this.SMS_CODE_KEY : this.WHITELIST_CODE_KEY;
    }

    /** The persisted SMS or WhatsApp code, generating and storing one on first use. */
    public ensurePhoneCode(channel: PhoneChannel): string { return this.ensureCode(this.phoneCodeKey(channel)); }

    /** The persisted SMS or WhatsApp code, or null if one hasn't been generated yet. */
    public getPhoneCode(channel: PhoneChannel): string | null { return localStorage.getItem(this.phoneCodeKey(channel)); }

    /** Mints a fresh SMS or WhatsApp code and overwrites the stored one. */
    public rotatePhoneCode(channel: PhoneChannel): string { return this.rotateCode(this.phoneCodeKey(channel)); }

    /**
     * Which pairing a phone tool's code needs: sendSms pairs by SMS, sendWhatsapp on WhatsApp,
     * and a call takes either, so it goes by which saved code `code` is (WhatsApp by default).
     */
    public pairingChannelFor(code: string | null | undefined, channel?: WhitelistChannel): PhoneChannel {
        if (channel === 'sms' || channel === 'whatsapp') return channel;
        return code && code === this.getPhoneCode('sms') ? 'sms' : 'whatsapp';
    }

    /**
     * The user's code for a phone tool (whatsapp when unspecified), generating one if needed. A call
     * reuses whichever code they already have, WhatsApp first, before minting a new WhatsApp one.
     */
    public defaultPhoneCode(channel?: WhitelistChannel): string {
        if (channel === 'sms') return this.ensurePhoneCode('sms');
        if (channel === 'voice') {
            return this.getPhoneCode('whatsapp') ?? this.getPhoneCode('sms') ?? this.ensurePhoneCode('whatsapp');
        }
        return this.ensurePhoneCode('whatsapp');
    }

    /** The persisted Telegram code, generating and storing one on first use. */
    public ensureTelegramCode(): string { return this.ensureCode(this.TELEGRAM_CODE_KEY); }

    /** The persisted Telegram code, or null if one hasn't been generated yet. */
    public getTelegramCode(): string | null { return localStorage.getItem(this.TELEGRAM_CODE_KEY); }

    /** Mints a fresh Telegram code, e.g. after the chat sent /stop and revoked the old one. */
    public rotateTelegramCode(): string { return this.rotateCode(this.TELEGRAM_CODE_KEY); }

    // --- REMOTE CONTROL ---
    // Whether this browser listens for messages the user sends from their linked
    // WhatsApp/Telegram. On by default: linking a phone is already an explicit action.
    private readonly REMOTE_CONTROL_KEY = 'observer-ai:settings:remoteControl';

    public isRemoteControlEnabled(): boolean {
        return localStorage.getItem(this.REMOTE_CONTROL_KEY) !== 'off';
    }

    public setRemoteControlEnabled(enabled: boolean): void {
        localStorage.setItem(this.REMOTE_CONTROL_KEY, enabled ? 'on' : 'off');
    }
}

// Export a single instance
export const SensorSettings = new SettingsManager();
