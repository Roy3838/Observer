import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { emit } from '@tauri-apps/api/event';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { Monitor, AppWindow, RefreshCw, X, CheckCircle, ImageOff, Loader } from 'lucide-react';

interface CaptureTarget {
  id: string;
  kind: 'monitor' | 'window';
  name: string;
  appName?: string;
  thumbnail?: string;
  width: number;
  height: number;
  isPrimary: boolean;
  x: number;
  y: number;
}

export default function ScreenSelectorWindow() {
  const [targets, setTargets] = useState<CaptureTarget[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  // Load targets - only called when window becomes visible
  const loadTargets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const targetsResult = await invoke<CaptureTarget[]>('plugin:screen-capture|get_capture_targets_cmd', {
        includeThumbnails: true
      });
      setTargets(targetsResult);
    } catch (e) {
      console.error('Failed to load capture targets:', e);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Only load targets when window becomes visible (not on mount)
  // This prevents permission prompts at app startup
  useEffect(() => {
    let mounted = true;

    const checkVisibilityAndLoad = async () => {
      try {
        const currentWindow = getCurrentWindow();
        const isVisible = await currentWindow.isVisible();
        if (isVisible && mounted) {
          loadTargets();
        }
      } catch (e) {
        console.error('Failed to check window visibility:', e);
      }
    };

    // Check immediately
    checkVisibilityAndLoad();

    // Also listen for window focus events (when window is shown)
    const setupListener = async () => {
      const currentWindow = getCurrentWindow();
      const unlisten = await currentWindow.onFocusChanged(({ payload: focused }) => {
        if (focused && mounted) {
          loadTargets();
        }
      });
      return unlisten;
    };

    const unlistenPromise = setupListener();

    return () => {
      mounted = false;
      unlistenPromise.then(unlisten => unlisten());
    };
  }, [loadTargets]);

  // Group targets by type
  const monitors = targets.filter(t => t.kind === 'monitor');
  const windows = targets.filter(t => t.kind === 'window');

  // Group windows by app
  const windowsByApp = windows.reduce<Record<string, CaptureTarget[]>>((acc, w) => {
    const app = w.appName || 'Unknown';
    if (!acc[app]) acc[app] = [];
    acc[app].push(w);
    return acc;
  }, {});

  const handleSelect = async (targetId: string) => {
    setSelectedTarget(targetId);
    setStarting(true);
    try {
      // Emit event with selected target - the main window will handle starting capture
      await emit('screen-capture-target-selected', { targetId });
      // Close the selector window
      const currentWindow = getCurrentWindow();
      await currentWindow.hide();
    } catch (e) {
      console.error('Failed to emit target selection:', e);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const handleCancel = async () => {
    // Emit cancel event so the main window knows selection was cancelled
    await emit('screen-capture-target-cancelled', {});
    const currentWindow = getCurrentWindow();
    await currentWindow.hide();
  };

  const handleRefresh = () => {
    loadTargets();
  };

  return (
    <div className="h-screen bg-slate-50 text-slate-800 overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-slate-800">Select Screen or Window</h1>
            <p className="text-slate-500 text-sm mt-1">
              Choose what you want to share
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 rounded-lg text-sm transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Loading...' : 'Refresh'}
            </button>
            <button
              onClick={handleCancel}
              className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 rounded-lg text-sm transition-colors flex items-center gap-2"
            >
              <X className="w-4 h-4" />
              Cancel
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <Loader className="w-8 h-8 text-blue-500 animate-spin" />
          </div>
        )}

        {/* Content */}
        {!loading && (
          <div className="space-y-6">
            {/* Monitors Section */}
            {monitors.length > 0 && (
              <section>
                <h2 className="text-lg font-semibold text-slate-700 mb-3 flex items-center gap-2">
                  <Monitor className="w-5 h-5" />
                  Screens ({monitors.length})
                </h2>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                  {monitors.map((monitor) => (
                    <TargetCard
                      key={monitor.id}
                      target={monitor}
                      selected={selectedTarget === monitor.id}
                      disabled={starting}
                      onSelect={() => handleSelect(monitor.id)}
                    />
                  ))}
                </div>
              </section>
            )}

            {/* Windows Section */}
            {Object.keys(windowsByApp).length > 0 && (
              <section>
                <h2 className="text-lg font-semibold text-slate-700 mb-3 flex items-center gap-2">
                  <AppWindow className="w-5 h-5" />
                  Windows ({windows.length})
                </h2>
                <div className="space-y-4">
                  {Object.entries(windowsByApp).map(([appName, appWindows]) => (
                    <div key={appName}>
                      <h3 className="text-sm font-medium text-slate-500 mb-2">{appName}</h3>
                      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                        {appWindows.map((window) => (
                          <TargetCard
                            key={window.id}
                            target={window}
                            selected={selectedTarget === window.id}
                            disabled={starting}
                            onSelect={() => handleSelect(window.id)}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Empty State */}
            {monitors.length === 0 && windows.length === 0 && !error && (
              <div className="text-center py-12 text-slate-400">
                <Monitor className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p>No screens or windows found</p>
                <p className="text-sm mt-1">Try clicking Refresh</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface TargetCardProps {
  target: CaptureTarget;
  selected: boolean;
  disabled: boolean;
  onSelect: () => void;
}

function TargetCard({ target, selected, disabled, onSelect }: TargetCardProps) {
  const displayName = target.kind === 'monitor'
    ? `${target.name}${target.isPrimary ? ' (Primary)' : ''}`
    : target.name;

  return (
    <button
      onClick={onSelect}
      disabled={disabled}
      className={`
        relative overflow-hidden rounded-xl border-2 transition-all text-left
        ${selected
          ? 'border-blue-500 bg-blue-50'
          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50 shadow-sm'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
      `}
    >
      {/* Thumbnail */}
      <div className="aspect-video bg-slate-100 flex items-center justify-center overflow-hidden">
        {target.thumbnail ? (
          <img
            src={`data:image/jpeg;base64,${target.thumbnail}`}
            alt={displayName}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="text-slate-400">
            <ImageOff className="w-12 h-12" />
          </div>
        )}
      </div>

      {/* Info */}
      <div className="p-3">
        <p className="font-medium text-slate-800 truncate" title={displayName}>
          {displayName}
        </p>
        <p className="text-xs text-slate-500 mt-0.5">
          {target.width} x {target.height}
        </p>
      </div>

      {/* Selected Indicator */}
      {selected && (
        <div className="absolute top-2 right-2 text-blue-500">
          <CheckCircle className="w-6 h-6 fill-blue-500 stroke-white" />
        </div>
      )}
    </button>
  );
}
