import React from 'react';
import { Terminal, Cloud, Server } from 'lucide-react';
import { getOllamaServerAddress } from '@utils/main_loop';

interface StartupDialogProps {
  onDismiss: () => void;
  onLogin?: () => void;
  setUseObServer?: (value: boolean) => void;
  isAuthenticated: boolean;
}

const StartupDialog: React.FC<StartupDialogProps> = ({
  onDismiss,
  onLogin,
  setUseObServer,
  isAuthenticated
}) => {
  const handleObServerStart = () => {
    if (!isAuthenticated) {
      // Save the choice even when redirecting to login
      localStorage.setItem('observer-startup-completed', 'true');
      localStorage.setItem('observer-server-choice', 'cloud');
      if (setUseObServer) setUseObServer(true);
      if (onLogin) onLogin();
      onDismiss();
    } else {
      if (setUseObServer) setUseObServer(true);
      // Save the user's choice to localStorage
      localStorage.setItem('observer-startup-completed', 'true');
      localStorage.setItem('observer-server-choice', 'cloud');
      onDismiss();
    }
  };

  const handleSetupLocal = () => {
    if (setUseObServer) setUseObServer(false);
    // Save the user's choice to localStorage
    localStorage.setItem('observer-startup-completed', 'true');
    localStorage.setItem('observer-server-choice', 'local');
    onDismiss();
  };

  const handleAcceptCertClick = (e: React.MouseEvent) => {
    e.preventDefault();
    const { host, port } = getOllamaServerAddress();
    const url = `${host}:${port}`;
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm p-4">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-4 sm:p-6 md:p-8 max-w-3xl w-full">
        <div className="flex items-center gap-3 mb-4 sm:mb-6">
          <Terminal className="h-7 w-7 sm:h-8 sm:w-8 text-blue-500" />
          <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 dark:text-white">Welcome to Observer</h2>
        </div>
        <p className="text-gray-600 dark:text-gray-300 mb-6 text-sm sm:text-base">Choose how you want to get started:</p>
        
        <div className="grid md:grid-cols-2 gap-4 md:gap-6 mb-6">
          {/* Ob-Server Cloud Card */}
          <div className="border rounded-lg p-4 sm:p-5 shadow-sm hover:shadow-md transition-shadow bg-blue-50 dark:bg-blue-900/20 border-blue-100 dark:border-blue-800 flex flex-col justify-between h-full">
            <div>
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-medium text-blue-700 dark:text-blue-300">Ob-Server Cloud</h3>
                <Cloud className="h-5 w-5 sm:h-6 sm:w-6 text-blue-500" />
              </div>
              
              {/* --- MODIFIED: Desktop View (hidden on mobile) --- */}
              <ul className="space-y-2 text-sm hidden sm:block">
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0"></div><span className="text-gray-700 dark:text-gray-300">No installation needed</span></li>
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0"></div><span className="text-gray-700 dark:text-gray-300">Easy to use</span></li>
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0"></div><span className="text-gray-700 dark:text-gray-300">Privacy respecting</span></li>
              </ul>

              {/* --- MODIFIED: Mobile View (hidden on desktop) --- */}
              <p className="text-sm text-blue-800/80 dark:text-blue-300 block sm:hidden">
                Easy · No Install · Privacy Respecting
              </p>
            </div>
            <div className="mt-6">
              <button 
                onClick={handleObServerStart} 
                className="w-full px-4 py-2.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium text-sm"
              >
                {isAuthenticated ? 'Start with Ob-Server' : 'Log In to Get Started'}
              </button>
            </div>
          </div>
          
          {/* Local Server Card */}
          <div className="border rounded-lg p-4 sm:p-5 shadow-sm hover:shadow-md transition-shadow bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-600 flex flex-col justify-between h-full">
            <div>
                <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-medium text-slate-800 dark:text-gray-200">Local Server</h3>
                <Server className="h-5 w-5 sm:h-6 sm:w-6 text-slate-500 dark:text-gray-400" />
                </div>

                {/* --- MODIFIED: Desktop View (hidden on mobile) --- */}
                <ul className="space-y-2 text-sm hidden sm:block">
                  <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-slate-500 flex-shrink-0"></div><span className="text-gray-700 dark:text-gray-300">Full Control</span></li>
                  <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-slate-500 flex-shrink-0"></div><span className="text-gray-700 dark:text-gray-300">Use your own hardware</span></li>
                  <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-slate-500 flex-shrink-0"></div><span className="text-gray-700 dark:text-gray-300">Complete privacy</span></li>
                </ul>

                {/* --- MODIFIED: Mobile View (hidden on desktop) --- */}
                <p className="text-sm text-slate-600 dark:text-gray-300 block sm:hidden">
                    Full Control · Complete Privacy
                </p>
            </div>
            <div className="mt-6">
                <p className="text-center text-xs text-gray-600 mb-2 leading-relaxed">
                  Run <a href="https://github.com/Roy3838/Observer" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">observer-ollama</a> and <a href="#" onClick={handleAcceptCertClick} className="text-blue-600 hover:underline">check server</a>.
                </p>
                <button onClick={handleSetupLocal} className="w-full px-4 py-2.5 bg-slate-700 text-white rounded-lg hover:bg-slate-800 transition-colors font-medium text-sm">
                    Use Local Server
                </button>
            </div>
          </div>
        </div>
        
        {/* --- PRODUCT HUNT BANNER --- */}
        <div className="mt-6 p-3 sm:p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-700 rounded-lg text-center shadow-sm">
            <h3 className="font-semibold text-md sm:text-lg text-purple-800 dark:text-purple-300 mb-2">
                🚀 Observer is Live on Product Hunt!
            </h3>
            <p className="text-xs sm:text-sm text-purple-700 dark:text-purple-300 mb-3 sm:mb-4">
                Your support means the world. Please consider upvoting the project!
            </p>
            <a 
                href="https://www.producthunt.com/products/observer-ai?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-observer&#0045;ai"
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-block bg-orange-500 hover:bg-orange-600 text-white font-bold py-1.5 px-4 sm:py-2 sm:px-5 rounded-lg transition-colors text-sm"
            >
                Support on Product Hunt
            </a>
        </div>

        <div className="text-center text-xs sm:text-sm text-gray-500 mt-6">
          You can switch between options anytime from the app header.
        </div>
      </div>
    </div>
  );
};

export default StartupDialog;
