// src/components/debug/PipDebugPanel.tsx
//
// Dev-only, mobile-only: the status PiP fed two ways, side by side, to find out why "Run in
// Background" doesn't open on iOS (utils/statusPip.ts). Both buttons look and act like the real
// one; only what feeds the window differs: the status alone (canvas) or the status over a live
// video (video). Each column previews its stream (does it have frames?) and prints its state and
// last error, since the webview console isn't visible on the phone.

import React, { useEffect, useRef, useState } from 'react';
import { Smartphone, X } from 'lucide-react';
import { prepareStatusPip, startStatusPip, stopStatusPip, statusPipDiagnostics, type PipMode } from '@utils/statusPip';

type Diagnostics = ReturnType<typeof statusPipDiagnostics>;

const Column: React.FC<{ mode: PipMode; label: string }> = ({ mode, label }) => {
  const previewRef = useRef<HTMLVideoElement>(null);
  const [diag, setDiag] = useState<Diagnostics>(null);
  const [tapError, setTapError] = useState('');

  useEffect(() => {
    const video = prepareStatusPip(mode);
    // A second <video> on the same stream: the PiP one stays hidden and untouched.
    if (previewRef.current) {
      previewRef.current.srcObject = video.srcObject;
      previewRef.current.play().catch(() => {});
    }
    const tick = setInterval(() => setDiag(statusPipDiagnostics(mode)), 500);
    return () => clearInterval(tick);
  }, [mode]);

  // Straight from the tap: no await before startStatusPip.
  const toggle = () => {
    setTapError('');
    if (diag?.active) {
      stopStatusPip().catch(e => setTapError(String(e)));
      return;
    }
    startStatusPip(mode).catch(e => setTapError(e instanceof Error ? `${e.name}: ${e.message}` : String(e)));
  };

  return (
    <div className="flex-1 min-w-0 flex flex-col gap-2">
      {!diag?.active ? (
        <button
          onClick={toggle}
          className="w-full flex items-center justify-center gap-2 px-3 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors text-sm"
        >
          <Smartphone className="w-4 h-4" />
          <span>Run in Background</span>
        </button>
      ) : (
        <button onClick={toggle} className="w-full bg-green-50 border border-green-200 rounded-lg p-2.5 hover:bg-green-100 transition-colors">
          <div className="flex items-center justify-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-sm text-green-800 font-medium">Running</span>
          </div>
        </button>
      )}
      <p className="text-xs font-semibold text-gray-700 text-center">{label}</p>
      <video ref={previewRef} muted playsInline autoPlay className="w-full aspect-[3/4] rounded-md bg-gray-900 object-contain" />
      <pre className="text-[10px] leading-4 text-gray-600 whitespace-pre-wrap break-all">
        {diag
          ? `readyState ${diag.readyState}  paused ${diag.paused}\nsize ${diag.size}\nbg ${diag.background}  pip ${diag.active}`
          : 'not prepared'}
        {(tapError || diag?.lastError) && <span className="text-red-600">{`\n${tapError || diag?.lastError}`}</span>}
      </pre>
    </div>
  );
};

const PipDebugPanel: React.FC = () => {
  const [open, setOpen] = useState(true);
  if (!open) {
    return (
      <button onClick={() => setOpen(true)} className="fixed bottom-3 right-3 z-[10000] px-3 py-1.5 rounded-full bg-gray-900 text-white text-xs shadow-lg">
        PiP debug
      </button>
    );
  }
  return (
    <div className="fixed bottom-3 left-3 right-3 z-[10000] rounded-xl bg-white shadow-2xl ring-1 ring-gray-200 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-gray-900">PiP debug</span>
        <button onClick={() => setOpen(false)} className="p-1 text-gray-400 hover:text-gray-700"><X className="w-4 h-4" /></button>
      </div>
      <div className="flex gap-3">
        <Column mode="canvas" label="Canvas only" />
        <Column mode="video" label="Video + overlay" />
      </div>
    </div>
  );
};

export default PipDebugPanel;
