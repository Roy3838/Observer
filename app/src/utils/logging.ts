// src/utils/logging.ts

// Define log levels
export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3
}

// Channel types for whitelist and notifications
export type NotificationChannel = 'whatsapp' | 'sms' | 'voice' | 'email' | 'pushover' | 'discord' | 'telegram';
export type WhitelistChannel = 'whatsapp' | 'sms' | 'voice';

// Define log entry structure
export interface LogEntry {
  id: string;           // Unique ID for the log entry
  timestamp: Date;      // When the log occurred
  level: LogLevel;      // Severity level
  source: string;       // Source of the log (agentId, service name, etc.)
  message: string;      // Main log message
  details?: any;        // Optional additional data
}

// Define log listener type
export type LogListener = (entry: LogEntry) => void;

/**
 * Logger service for capturing and managing application logs
 */
class LoggingService {
  private logs: LogEntry[] = [];
  private listeners: LogListener[] = [];
  private maxLogEntries: number = 1000; // Maximum number of logs to keep in memory
  private nextLogId: number = 1;

  constructor() {
    // Initialize with any saved logs if needed
  }

  /**
   * Add a new log entry
   */
  public log(
    level: LogLevel,
    source: string,
    message: string,
    details?: any
  ): LogEntry {
    const entry: LogEntry = {
      id: `log-${this.nextLogId++}`,
      timestamp: new Date(),
      level,
      source,
      message,
      details
    };

    // Add to log array
    this.logs.push(entry);
    
    // Trim logs if we exceed max size
    if (this.logs.length > this.maxLogEntries) {
      this.logs = this.logs.slice(-this.maxLogEntries);
    }

    // Notify all listeners
    this.notifyListeners(entry);

    return entry;
  }

  /**
   * Convenience method for debug logs
   */
  public debug(source: string, message: string, details?: any): LogEntry {
    return this.log(LogLevel.DEBUG, source, message, details);
  }

  /**
   * Convenience method for info logs
   */
  public info(source: string, message: string, details?: any): LogEntry {
    return this.log(LogLevel.INFO, source, message, details);
  }

  /**
   * Convenience method for warning logs
   */
  public warn(source: string, message: string, details?: any): LogEntry {
    return this.log(LogLevel.WARNING, source, message, details);
  }

  /**
   * Convenience method for error logs
   */
  public error(source: string, message: string, details?: any): LogEntry {
    return this.log(LogLevel.ERROR, source, message, details);
  }

  /**
   * Get all stored logs
   */
  public getLogs(): LogEntry[] {
    return [...this.logs];
  }

  /**
   * Get logs filtered by criteria
   */
  public getFilteredLogs(
    filters: {
      level?: LogLevel,
      source?: string | string[],
      since?: Date,
      until?: Date,
      searchText?: string
    }
  ): LogEntry[] {
    return this.logs.filter(entry => {
      // Filter by log level
      if (filters.level !== undefined && entry.level < filters.level) {
        return false;
      }

      // Filter by source
      if (filters.source !== undefined) {
        if (Array.isArray(filters.source)) {
          if (!filters.source.includes(entry.source)) {
            return false;
          }
        } else if (entry.source !== filters.source) {
          return false;
        }
      }

      // Filter by start date
      if (filters.since && entry.timestamp < filters.since) {
        return false;
      }

      // Filter by end date
      if (filters.until && entry.timestamp > filters.until) {
        return false;
      }

      // Filter by text search
      if (filters.searchText) {
        const searchLower = filters.searchText.toLowerCase();
        if (!entry.message.toLowerCase().includes(searchLower)) {
          return false;
        }
      }

      return true;
    });
  }

  /**
   * Get logs for a specific agent
   */
  public getAgentLogs(agentId: string): LogEntry[] {
    return this.getFilteredLogs({ source: agentId });
  }

  /**
   * Clear all logs
   */
  public clearLogs(): void {
    this.logs = [];
    // Optionally notify listeners about clear
  }

  /**
   * Register a listener for new log entries
   */
  public addListener(listener: LogListener): void {
    this.listeners.push(listener);
  }

  /**
   * Remove a previously registered listener
   */
  public removeListener(listener: LogListener): void {
    this.listeners = this.listeners.filter(l => l !== listener);
  }

  /**
   * Notify all listeners about a new log entry
   */
  private notifyListeners(entry: LogEntry): void {
    this.listeners.forEach(listener => {
      try {
        listener(entry);
      } catch (error) {
        console.error('Error in log listener:', error);
      }
    });

    // Dispatch window events for certain logTypes (unified event system)
    this.dispatchWindowEventIfNeeded(entry);
  }

  /**
   * Dispatch window events based on logType - unified event system
   * Logger is the single source of truth; events are notifications
   */
  private dispatchWindowEventIfNeeded(entry: LogEntry): void {
    const logType = entry.details?.logType;
    if (!logType) return;

    try {
      switch (logType) {
        case 'agent-status-changed':
          window.dispatchEvent(
            new CustomEvent('agentStatusChanged', {
              detail: entry.details.content
            })
          );
          break;

        case 'iteration-start':
          window.dispatchEvent(
            new CustomEvent('agentIterationStart', {
              detail: {
                agentId: entry.source,
                intervalMs: entry.details.content?.intervalMs,
                iterationStartTime: entry.details.content?.iterationStartTime,
                iterationId: entry.details.iterationId
              }
            })
          );
          break;

        case 'iteration-end':
        case 'iteration-skipped':
          // These are logged but don't need immediate UI events
          // UI can track completion through log listener
          break;

        case 'stream-start':
          window.dispatchEvent(
            new CustomEvent('agentStreamStart', {
              detail: {
                agentId: entry.source,
                iterationId: entry.details.iterationId
              }
            })
          );
          break;

        case 'stream-chunk':
          window.dispatchEvent(
            new CustomEvent('agentResponseChunk', {
              detail: {
                agentId: entry.source,
                chunk: entry.details.content?.chunk,
                iterationId: entry.details.iterationId
              }
            })
          );
          break;

        case 'thinking-chunk':
          window.dispatchEvent(
            new CustomEvent('agentThinkingChunk', {
              detail: {
                agentId: entry.source,
                chunk: entry.details.content?.chunk,
                iterationId: entry.details.iterationId
              }
            })
          );
          break;

        case 'agent-error':
          window.dispatchEvent(
            new CustomEvent('agentRuntimeError', {
              detail: {
                agentId: entry.source,
                error: entry.details.content?.error
              }
            })
          );
          break;

        case 'quota-exceeded':
          window.dispatchEvent(
            new CustomEvent('quotaExceeded', {
              detail: {
                agentId: entry.source,
                quotaType: entry.details?.quotaType || 'monitor'
              }
            })
          );
          break;

        case 'change-detection-result':
          window.dispatchEvent(
            new CustomEvent('agentChangeDetectionResult', {
              detail: entry.details.content
            })
          );
          break;

        case 'whitelist-required':
          window.dispatchEvent(
            new CustomEvent('whitelistRequired', {
              detail: {
                phoneNumber: entry.details.content?.phoneNumber,
                toolName: entry.details.content?.toolName,
                channel: entry.details.content?.channel as WhitelistChannel
              }
            })
          );
          break;

        case 'agent-sleep-start':
          window.dispatchEvent(
            new CustomEvent('agentSleepStart', {
              detail: {
                agentId: entry.source,
                durationMs: entry.details.content?.durationMs
              }
            })
          );
          break;

        case 'agent-sleep-end':
          window.dispatchEvent(
            new CustomEvent('agentSleepEnd', {
              detail: {
                agentId: entry.source,
                wokeEarly: entry.details.content?.wokeEarly
              }
            })
          );
          break;
      }
    } catch (error) {
      console.error('Error dispatching window event from logger:', error);
    }
  }
}

// Export a singleton instance of the service
export const Logger = new LoggingService();

// Create a React context provider for the Logger (to be used later)
export const createLoggerDecorator = (source: string) => {
  return {
    debug: (message: string, details?: any) => Logger.debug(source, message, details),
    info: (message: string, details?: any) => Logger.info(source, message, details),
    warn: (message: string, details?: any) => Logger.warn(source, message, details),
    error: (message: string, details?: any) => Logger.error(source, message, details)
  };
};
