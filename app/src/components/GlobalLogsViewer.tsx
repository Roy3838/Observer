// src/components/GlobalLogsViewer.tsx
import React, { useState, useEffect, useRef } from 'react';
import { LogEntry, LogLevel, Logger } from '../utils/logging';
import { X, RefreshCw, Download, Filter } from 'lucide-react';

interface GlobalLogsViewerProps {
  isOpen: boolean;
  onClose: () => void;
}

const GlobalLogsViewer: React.FC<GlobalLogsViewerProps> = ({ isOpen, onClose }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState('');
  const [minLevel, setMinLevel] = useState<LogLevel>(LogLevel.INFO);
  const [isAutoScroll, setIsAutoScroll] = useState(true);
  const [filterSource, setFilterSource] = useState<string | null>(null);
  const [uniqueSources, setUniqueSources] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  
  // Load logs when the component mounts
  useEffect(() => {
    if (!isOpen) return;
    
    // Get initial logs
    const allLogs = Logger.getFilteredLogs({ level: minLevel });
    setLogs(allLogs);
    
    // Get unique sources
    const sources = new Set<string>();
    allLogs.forEach(log => sources.add(log.source));
    setUniqueSources(Array.from(sources));

    // Subscribe to new logs
    const handleNewLog = (log: LogEntry) => {
      if (log.level < minLevel) return;
      
      setLogs(prevLogs => [...prevLogs, log]);
      
      // Update unique sources if needed
      if (!uniqueSources.includes(log.source)) {
        setUniqueSources(prev => [...prev, log.source]);
      }
    };

    Logger.addListener(handleNewLog);
    
    return () => {
      Logger.removeListener(handleNewLog);
    };
  }, [isOpen, minLevel]);

  // Scroll to bottom when new logs come in (if auto-scroll is enabled)
  useEffect(() => {
    if (isAutoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isAutoScroll]);

  // Filter logs based on user input
  const filteredLogs = logs.filter(log => {
    // Apply minimum level filter
    if (log.level < minLevel) return false;
    
    // Apply source filter if selected
    if (filterSource && log.source !== filterSource) return false;
    
    // Apply text filter
    if (filter && !log.message.toLowerCase().includes(filter.toLowerCase())) return false;
    
    return true;
  });

  // Format timestamp
  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      hour12: false
    });
  };

  // Download logs as JSON
  const downloadLogs = () => {
    const logsJson = JSON.stringify(filteredLogs, null, 2);
    const blob = new Blob([logsJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `observer-logs-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Clear logs
  const clearLogs = () => {
    setLogs([]);
  };
  
  // Helper function to get color for log level
  const getLevelColor = (level: LogLevel): string => {
    switch (level) {
      case LogLevel.DEBUG:
        return 'text-gray-500 dark:text-gray-400';
      case LogLevel.INFO:
        return 'text-blue-600 dark:text-blue-400';
      case LogLevel.WARNING:
        return 'text-amber-600 dark:text-amber-400';
      case LogLevel.ERROR:
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-gray-800 dark:text-gray-200';
    }
  };

  // Helper function to get text for log level
  const getLevelText = (level: LogLevel): string => {
    switch (level) {
      case LogLevel.DEBUG:
        return 'DEBUG';
      case LogLevel.INFO:
        return 'INFO';
      case LogLevel.WARNING:
        return 'WARN';
      case LogLevel.ERROR:
        return 'ERROR';
      default:
        return 'UNKNOWN';
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-600 shadow-lg z-40 h-64 overflow-hidden flex flex-col">
      {/* Header */}
      <div className="bg-gray-100 dark:bg-gray-700 px-4 py-2 flex justify-between items-center border-b border-gray-200 dark:border-gray-600">
        <div className="flex items-center">
          <h3 className="font-semibold text-gray-900 dark:text-white">System Logs</h3>
          <span className="ml-2 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 text-xs py-1 px-2 rounded-full">
            {filteredLogs.length} entries
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={downloadLogs}
            className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded text-gray-700 dark:text-gray-300"
            title="Download logs"
          >
            <Download size={18} />
          </button>
          <button
            onClick={clearLogs}
            className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded text-gray-700 dark:text-gray-300"
            title="Clear logs"
          >
            <RefreshCw size={18} />
          </button>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded text-gray-700 dark:text-gray-300"
            title="Close log viewer"
          >
            <X size={18} />
          </button>
        </div>
      </div>
      
      {/* Filters */}
      <div className="bg-gray-50 dark:bg-gray-700 px-4 py-2 flex items-center space-x-4 border-b border-gray-200 dark:border-gray-600">
        <div className="flex items-center">
          <label htmlFor="log-level" className="text-sm mr-2 text-gray-700 dark:text-gray-300">Level:</label>
          <select
            id="log-level"
            value={minLevel}
            onChange={(e) => setMinLevel(Number(e.target.value) as LogLevel)}
            className="text-sm border border-gray-300 dark:border-gray-600 rounded p-1 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          >
            <option value={LogLevel.DEBUG}>Debug & Above</option>
            <option value={LogLevel.INFO}>Info & Above</option>
            <option value={LogLevel.WARNING}>Warning & Above</option>
            <option value={LogLevel.ERROR}>Errors Only</option>
          </select>
        </div>
        
        <div className="flex items-center">
          <label htmlFor="log-source" className="text-sm mr-2 text-gray-700 dark:text-gray-300">Source:</label>
          <select
            id="log-source"
            value={filterSource || ''}
            onChange={(e) => setFilterSource(e.target.value || null)}
            className="text-sm border border-gray-300 dark:border-gray-600 rounded p-1 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          >
            <option value="">All Sources</option>
            {uniqueSources.map(source => (
              <option key={source} value={source}>
                {source}
              </option>
            ))}
          </select>
        </div>
        
        <div className="flex items-center flex-grow">
          <label htmlFor="log-filter" className="text-sm mr-2 text-gray-700 dark:text-gray-300">
            <Filter size={16} />
          </label>
          <input
            id="log-filter"
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter logs..."
            className="text-sm border rounded p-1 flex-grow"
          />
        </div>
        
        <div className="flex items-center">
          <label htmlFor="auto-scroll" className="text-sm mr-2 text-gray-700 dark:text-gray-300">Auto-scroll:</label>
          <input
            id="auto-scroll"
            type="checkbox"
            checked={isAutoScroll}
            onChange={() => setIsAutoScroll(!isAutoScroll)}
          />
        </div>
      </div>
      
      {/* Log content */}
      <div className="flex-grow overflow-y-auto p-2 font-mono text-sm bg-white dark:bg-gray-800">
        {filteredLogs.length === 0 ? (
          <div className="text-gray-500 dark:text-gray-400 italic text-center mt-4">
            No logs to display
          </div>
        ) : (
          <div className="space-y-1">
            {filteredLogs.map((log) => (
              <div key={log.id} className="hover:bg-gray-50 dark:hover:bg-gray-700 p-1">
                <div className="flex">
                  <span className="text-gray-500 dark:text-gray-400 w-20 shrink-0">{formatTime(log.timestamp)}</span>
                  <span className={`w-14 shrink-0 ${getLevelColor(log.level)}`}>{getLevelText(log.level)}</span>
                  <span className="w-24 shrink-0 text-gray-700 dark:text-gray-300 truncate">{log.source}</span>
                  <span className="flex-grow text-gray-900 dark:text-gray-100">{log.message}</span>
                </div>
                {log.details && (
                  <div className="ml-34 pl-34 text-xs bg-gray-50 dark:bg-gray-700 p-1 mt-1 rounded border-l-2 border-gray-300 dark:border-gray-600 overflow-x-auto text-gray-700 dark:text-gray-300">
                    {typeof log.details === 'string' 
                      ? log.details 
                      : JSON.stringify(log.details, null, 2)}
                  </div>
                )}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>
    </div>
  );
};

export default GlobalLogsViewer;
