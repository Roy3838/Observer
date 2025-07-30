import React, { useState, useEffect } from 'react';
import { Terminal, Cloud, Server, AlertTriangle, Download } from 'lucide-react';
import TerminalModal from '@components/TerminalModal';

interface StartupDialogProps {
  onDismiss: () => void;
  onLogin?: () => void;
  setUseObServer?: (value: boolean) => void;
  isAuthenticated: boolean;
  hostingContext: 'official-web' | 'self-hosted' | 'tauri';
}

const LOCAL_STORAGE_KEY = 'observer_local_server_address';
const SERVER_CHOICE_KEY = 'observer_server_choice';
const DEFAULT_SERVER_ADDRESS = 'http://localhost:3838';

const StartupDialog: React.FC<StartupDialogProps> = ({
  onDismiss,
  onLogin,
  setUseObServer,
  isAuthenticated,
  hostingContext
}) => {
  const [view, setView] = useState<'initial' | 'local-warning' | 'no-models'>('initial');
  const [hasNoModels, setHasNoModels] = useState(false);
  const [isCheckingModels, setIsCheckingModels] = useState(true);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);

  useEffect(() => {
    const savedChoice = localStorage.getItem(SERVER_CHOICE_KEY);
    if (savedChoice) {
      const useObServer = savedChoice === 'cloud';
      if (setUseObServer) setUseObServer(useObServer);
      onDismiss();
      return;
    }
    // Only run this check if in a local context
    if (hostingContext === 'self-hosted' || hostingContext === 'tauri') {
      const checkLocalModels = async () => {
        // 1. Create an AbortController to manage the fetch request
        const controller = new AbortController();
        const signal = controller.signal;

        // 2. Set a 1-second timer. If it fires, it aborts the fetch request.
        const timeoutId = setTimeout(() => {
          controller.abort();
        }, 1000); // 1000 milliseconds = 1 second

        try {
          const serverAddress = localStorage.getItem(LOCAL_STORAGE_KEY) || DEFAULT_SERVER_ADDRESS;

          // Don't bother checking if the address is the official cloud server
          if (new URL(serverAddress).hostname.includes('api.observer-ai.com')) {
              clearTimeout(timeoutId); // Clear the timeout as we are not fetching
              setIsCheckingModels(false);
              return;
          }

          // 3. Make the fetch request with the abort signal
          const response = await fetch(`${serverAddress}/api/tags`, { signal });
          
          // 4. If the fetch completes in time, clear the timeout
          clearTimeout(timeoutId);

          if (!response.ok) {
            throw new Error(`Server not reachable (status: ${response.status})`);
          }

          const data = await response.json();
          if (data.models && data.models.length === 0) {
            setHasNoModels(true);
          }
        } catch (error: any) {
          // 5. If the fetch was aborted, it throws an 'AbortError'. We catch it here.
          if (error.name === 'AbortError') {
            console.error("Local model check timed out after 1 second. Assuming local server is not running.");
          } else {
            console.error("Could not check for local models:", error);
          }
        } finally {
          // 6. Always ensure the loading state is turned off
          setIsCheckingModels(false);
        }
      };

      checkLocalModels();
    } else {
        // If not in a local context, just disable the loading state immediately
        setIsCheckingModels(false);
    }
  }, [hostingContext, onDismiss, setUseObServer]);

  const handleObServerStart = () => {
    localStorage.setItem(SERVER_CHOICE_KEY, 'cloud');
    if (!isAuthenticated) {
      if (onLogin) onLogin();
    } else {
      if (setUseObServer) setUseObServer(true);
      onDismiss();
    }
  };

  // --- BUG FIX 1 
  const handleSetupLocalClick = () => {
    if (hostingContext === 'official-web') {
      setView('local-warning');
    } else if (hasNoModels) {
      // Set the view AND set the server mode to local
      setView('no-models');
      if (setUseObServer) setUseObServer(false);
    } else {
      localStorage.setItem(SERVER_CHOICE_KEY, 'local');
      if (setUseObServer) setUseObServer(false);
      onDismiss();
    }
  };
  
  // --- BUG FIX 2 
  const handleProceedWithLocal = () => {
    localStorage.setItem(SERVER_CHOICE_KEY, 'local');
    if (hasNoModels) {
        // Set the view AND set the server mode to local
        setView('no-models');
        if (setUseObServer) setUseObServer(false);
    } else {
        // Proceed as normal
        if (setUseObServer) setUseObServer(false);
        onDismiss();
    }
  };
  
  const handlePullModelClick = () => {
    setIsTerminalOpen(true);
  };

  const handleTerminalClose = () => {
    localStorage.setItem(SERVER_CHOICE_KEY, 'local');
    setIsTerminalOpen(false);
    if (setUseObServer) setUseObServer(false);
    onDismiss();
  }

  const handleAcceptCertClick = (e: React.MouseEvent) => {
    e.preventDefault();
    const url = localStorage.getItem(LOCAL_STORAGE_KEY) || DEFAULT_SERVER_ADDRESS;
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  return (
    <>
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[100] backdrop-blur-sm p-4">
        <div className="bg-white rounded-xl shadow-xl p-4 sm:p-6 md:p-8 max-w-3xl w-full transition-all duration-300">
          
          {view === 'initial' && (
            <>
              <div className="flex items-center gap-3 mb-4 sm:mb-6">
                <Terminal className="h-7 w-7 sm:h-8 sm:w-8 text-blue-500" />
                <h2 className="text-xl sm:text-2xl font-semibold">Welcome to Observer</h2>
              </div>
              <p className="text-gray-600 mb-6 text-sm sm:text-base">Choose how you want to get started:</p>
              
              <div className="grid md:grid-cols-2 gap-4 md:gap-6 mb-6">
                {/* Ob-Server Cloud Card */}
                <div className="border rounded-lg p-4 sm:p-5 shadow-sm hover:shadow-md transition-shadow bg-blue-50 border-blue-100 flex flex-col justify-between h-full">
                  <div>
                      <div className="flex justify-between items-start mb-4">
                          <h3 className="text-lg font-medium text-blue-700">Ob-Server Cloud</h3>
                          <Cloud className="h-5 w-5 sm:h-6 sm:w-6 text-blue-500" />
                      </div>
                      <ul className="space-y-2 text-sm hidden sm:block">
                          <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0"></div><span className="text-gray-700">No installation needed</span></li>
                          <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0"></div><span className="text-gray-700">Easy to use</span></li>
                          <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0"></div><span className="text-gray-700">Privacy respecting</span></li>
                      </ul>
                      <p className="text-sm text-blue-800/80 block sm:hidden">Easy · No Install · Privacy Respecting</p>
                  </div>
                  <div className="mt-6">
                      <button onClick={handleObServerStart} className="w-full px-4 py-2.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium text-sm">
                          {isAuthenticated ? 'Start with Ob-Server' : 'Log In to Get Started'}
                      </button>
                  </div>
                </div>
                
                {/* Local Server Card */}
                <div className="border rounded-lg p-4 sm:p-5 shadow-sm hover:shadow-md transition-shadow bg-gray-50 border-gray-200 flex flex-col justify-between h-full">
                  <div>
                      <div className="flex justify-between items-start mb-4">
                          <h3 className="text-lg font-medium text-slate-800">Local Server</h3>
                          <Server className="h-5 w-5 sm:h-6 sm:w-6 text-slate-500" />
                      </div>
                      <ul className="space-y-2 text-sm hidden sm:block">
                          <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-slate-500 flex-shrink-0"></div><span className="text-gray-700">Full Control</span></li>
                          <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-slate-500 flex-shrink-0"></div><span className="text-gray-700">Use your own hardware</span></li>
                          <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-slate-500 flex-shrink-0"></div><span className="text-gray-700">Complete privacy</span></li>
                      </ul>
                      <p className="text-sm text-slate-600 block sm:hidden">Full Control · Complete Privacy</p>
                  </div>
                  <div className="mt-6">
                      <p className="text-center text-xs text-gray-600 mb-2 leading-relaxed">
                        Run <a href="https://github.com/Roy3838/Observer" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">observer-ollama</a> and <a href="#" onClick={handleAcceptCertClick} className="text-blue-600 hover:underline">check server</a>.
                      </p>
                      <button onClick={handleSetupLocalClick} disabled={isCheckingModels} className="w-full px-4 py-2.5 bg-slate-700 text-white rounded-lg hover:bg-slate-800 transition-colors font-medium text-sm disabled:bg-slate-400 disabled:cursor-wait">
                          {isCheckingModels ? 'Checking...' : 'Use Local Server'}
                      </button>
                  </div>
                </div>
              </div>
              <div className="text-center text-xs sm:text-sm text-gray-500 mt-6">
                You can switch between options anytime from the app header.
              </div>
            </>
          )}

          {view === 'local-warning' && (
            <div>
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="h-7 w-7 text-orange-500 flex-shrink-0" />
                <h2 className="text-xl sm:text-2xl font-semibold">Heads-Up: Connecting Locally</h2>
              </div>
              
              <p className="text-gray-700 mb-4">
                To connect your local models to our secure website (<span className="font-mono bg-gray-100 px-1 rounded text-sm">app.observer-ai.com</span>), your browser requires that your local server also use a secure (HTTPS) connection.
              </p>

              <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
                  <h3 className="font-semibold text-green-800 mb-2">Recommended Solution: Self-Host</h3>
                  <p className="text-sm text-green-700 mb-3">
                      Our Docker and upcoming Desktop App setups run everything on your machine, which avoids this browser security issue entirely and is the easiest path for local inference.
                  </p>
                  <div className="flex flex-wrap gap-3">
                      <a href="https://github.com/Roy3838/Observer?tab=readme-ov-file#option-1-full-docker-setup-recommended" target="_blank" rel="noopener noreferrer" className="px-4 py-2 bg-slate-800 text-white rounded-md text-sm font-medium hover:bg-slate-900">
                          View Docker Setup
                      </a>
                  </div>
              </div>

              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-800 mb-2">Advanced Option</h3>
                  <p className="text-sm text-gray-600">
                      If you've already configured your local server with a valid HTTPS certificate (e.g., using a reverse proxy), you can proceed.
                  </p>
              </div>

              <div className="mt-8 flex flex-col-reverse sm:flex-row justify-between items-center gap-4">
                  <button onClick={() => setView('initial')} className="text-sm text-gray-600 hover:underline">
                      ← Go Back
                  </button>
                  <button onClick={handleProceedWithLocal} className="w-full sm:w-auto px-5 py-2.5 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors font-medium text-sm shadow-sm hover:shadow-md">
                      Proceed, I know what I'm doing
                  </button>
              </div>
            </div>
          )}
          
          {view === 'no-models' && (
            <div>
              <div className="flex items-center gap-3 mb-4">
                <Download className="h-7 w-7 text-green-500 flex-shrink-0" />
                <h2 className="text-xl sm:text-2xl font-semibold">Let's Get Your First Model</h2>
              </div>
              
              <p className="text-gray-700 mb-6">
                Your local server is running, but it looks like you don't have any AI models installed yet. Models are the "brains" that power your agents. Let's download the recommended one to get you started.
              </p>

              <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
                  <h3 className="font-semibold text-green-800 mb-2 text-lg">Recommended Model: Gemma3 4B</h3>
                  <button 
                    onClick={handlePullModelClick} 
                    className="w-full sm:w-auto px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium text-base shadow-sm hover:shadow-md"
                  >
                    Pull Your First Model!
                  </button>
              </div>

              <div className="mt-8 flex flex-col-reverse sm:flex-row justify-between items-center gap-4">
                  <button onClick={() => setView('initial')} className="text-sm text-gray-600 hover:underline">
                      ← Go Back
                  </button>
                   <button onClick={handleTerminalClose} className="text-sm text-gray-600 hover:underline">
                      I'll do this later
                  </button>
              </div>
            </div>
          )}

        </div>
      </div>
      
      <TerminalModal isOpen={isTerminalOpen} onClose={handleTerminalClose} />
    </>
  );
};

export default StartupDialog;
