import { createPortal } from 'react-dom';
import { Cpu, CheckCircle2, Download, Sparkles } from 'lucide-react';
import { isTauri } from '@utils/platform';
import { MODEL_PRESETS } from '@utils/modelPresets';

const _p = MODEL_PRESETS[0];

interface LocalServerSetupDialogProps {
  onDismiss: () => void;
  onModelLoaded?: () => void;
}

const LocalServerSetupDialog = ({ onDismiss }: LocalServerSetupDialogProps) => {
  if (!isTauri()) return null;

  const handleDownloadModel = () => {
    window.dispatchEvent(new CustomEvent('openModelHub', {
      detail: { autoDownloadPreset: _p },
    }));
    onDismiss();
  };

  return createPortal(
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[110] backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full mx-4 overflow-hidden">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-green-100 to-emerald-100 mb-4">
            <Sparkles className="h-8 w-8 text-green-600" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to Observer!</h2>
          <p className="text-gray-600">
            Let's get you started with a local AI model that runs entirely on your device.
          </p>
        </div>

        <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-5 mb-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg bg-green-200 flex items-center justify-center">
              <Cpu size={20} className="text-green-700" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">{_p.name}</h3>
              <p className="text-sm text-gray-600">{_p.sizeLabel} - Vision capable</p>
            </div>
          </div>
          <ul className="space-y-2 text-sm text-gray-700">
            <li className="flex items-center gap-2">
              <CheckCircle2 size={16} className="text-green-600 flex-shrink-0" />
              <span>Runs 100% on your device - no internet needed</span>
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 size={16} className="text-green-600 flex-shrink-0" />
              <span>Understands images and text</span>
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle2 size={16} className="text-green-600 flex-shrink-0" />
              <span>Fast and efficient - optimized for your hardware</span>
            </li>
          </ul>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onDismiss}
            className="flex-1 py-3 px-4 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 transition-colors font-medium"
          >
            Skip for Now
          </button>
          <button
            onClick={handleDownloadModel}
            className="flex-1 py-3 px-4 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-colors flex items-center justify-center gap-2 font-medium shadow-sm"
          >
            <Download size={18} />
            Download Model
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
};

export default LocalServerSetupDialog;
