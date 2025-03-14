import React, { useState, useEffect, lazy, Suspense, useRef } from 'react';
import { Terminal, Play, Copy } from 'lucide-react';
import { Logger } from '@utils/logging';

// Lazy load CodeMirror component
const LazyCodeMirror = lazy(() => import('@uiw/react-codemirror'));

// Import extensions normally - adding JavaScript language
import { javascript } from '@codemirror/lang-javascript';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import { executeTestIteration } from '@utils/main_loop';

// Code snippets collection
const codeSnippets = [
  {
    name: "Write to Memory",
    description: "Store the entire response in agent memory",
    code: 'setMemory(agentId, response);'
  },
  {
    name: "Read/Write Agent Memory",
    description: "Read memory, append timestamped content",
    code: 'setMemory(`${await getMemory()} \\n[${time()}] ${response}`)'
  },
  {
    name: "Remove Thought Tags",
    description: "Clean response by removing <think>...</think> sections",
    code: 'const cleanedResponse = response.replace(/<think>[\\s\\S]*?<\\/think>/g, \'\').trim();'
  }
];

interface CodeTabProps {
  agentId: string;
  code: string;
  systemPrompt: string;
  model: string;
  onCodeChange: (code: string) => void;
}

const CodeTab: React.FC<CodeTabProps> = ({ 
  agentId,
  code,
  systemPrompt,
  model,
  onCodeChange
}) => {
  const [selectedCodeSnippet, setSelectedCodeSnippet] = useState<string | null>(null);
  const [editorIsLoaded, setEditorIsLoaded] = useState(false);
  
  // State for the Run functionality
  const [testResponse, setTestResponse] = useState<string>('');
  const [testResponseVisible, setTestResponseVisible] = useState<boolean>(false);
  const [agentIdVisible, setAgentIdVisible] = useState<boolean>(false);
  const [isRunningModel, setIsRunningModel] = useState<boolean>(false);
  const [isRunningCode, setIsRunningCode] = useState<boolean>(false);
  const [testOutput, setTestOutput] = useState<string>('');
  const testResponseRef = useRef<HTMLTextAreaElement>(null);

  // Preload editor when component mounts
  useEffect(() => {
    if (!editorIsLoaded) {
      // This will trigger the lazy loading
      import('@uiw/react-codemirror').then(() => {
        setEditorIsLoaded(true);
      });
    }
  }, [editorIsLoaded]);

  const toggleResponseEdit = () => {
    setTestResponseVisible(!testResponseVisible);
    // Focus the textarea when it becomes visible
    if (!testResponseVisible && testResponseRef.current) {
      setTimeout(() => {
        testResponseRef.current?.focus();
      }, 0);
    }
  };
  
  const toggleAgentId = () => {
    setAgentIdVisible(!agentIdVisible);
  };

  const insertCodeSnippet = (snippetCode: string) => {
    const newCode = code + (code.endsWith('\n') ? '' : '\n') + snippetCode;
    onCodeChange(newCode);
    setSelectedCodeSnippet(null);
  };
  
  // Run the model to get a response
  const handleRunModel = async () => {
    if (isRunningModel) return;
    
    setIsRunningModel(true);
    setTestOutput('');
    
    try {
      // Log that we're calling the model
      setTestOutput(current => current + `Running ${model || 'deepseek-r1:8b'}...\n`);
      
      // Use the executeTestIteration function
      const response = await executeTestIteration(
        agentId || 'test-agent', 
        systemPrompt,
        model || 'deepseek-r1:8b'
      );
      
      // Update state with the response
      setTestResponse(response);
      setTestResponseVisible(true);
      
      // Also show in output
      setTestOutput(current => current + `Response received (${response.length} chars)\n`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      setTestOutput(current => current + `Error: ${errorMessage}\n`);
      Logger.error(agentId || 'test', `Error executing agent iteration: ${errorMessage}`, error);
    } finally {
      setIsRunningModel(false);
    }
  };

  const handleRunCode = async () => {
    if (isRunningCode || !testResponse) return;
    
    setIsRunningCode(true);
    setTestOutput('');
    
    try {
      // Import the post-processor
      const { postProcess } = await import('@utils/post-processor');
      
      // Create an array to capture logs
      const logs: string[] = [];
      
      // Create a proxy console object to capture logs
      const originalConsole = { ...console };
      const consoleProxy = {
        log: (...args: any[]) => {
          originalConsole.log(...args);
          logs.push(args.map(arg => String(arg)).join(' '));
        },
        error: (...args: any[]) => {
          originalConsole.error(...args);
          logs.push(`ERROR: ${args.map(arg => String(arg)).join(' ')}`);
        },
        warn: (...args: any[]) => {
          originalConsole.warn(...args);
          logs.push(`WARNING: ${args.map(arg => String(arg)).join(' ')}`);
        },
        info: (...args: any[]) => {
          originalConsole.info(...args);
          logs.push(args.map(arg => String(arg)).join(' '));
        }
      };
      
      // Replace console methods with our proxy
      console.log = consoleProxy.log;
      console.error = consoleProxy.error;
      console.warn = consoleProxy.warn;
      console.info = consoleProxy.info;
      
      // Run the post-processor with the current code and response
      const result = await postProcess(agentId || 'test-agent', testResponse, code);
      
      // Add a small delay to ensure all async console logs complete
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Restore original console
      console.log = originalConsole.log;
      console.error = originalConsole.error;
      console.warn = originalConsole.warn;
      console.info = originalConsole.info;
      
      // Update UI with result and captured logs
      setTestOutput(
        `Post-processing ${result ? 'succeeded' : 'completed without specific action'}\n\n` +
        logs.join('\n')
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      setTestOutput(`Error executing code: ${errorMessage}`);
    } finally {
      setIsRunningCode(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Variable buttons at the top with Run Model button to the right */}
      <div className="flex mb-6 items-center justify-between">
        <div className="flex items-center">
          <div 
            className="border border-blue-200 bg-white rounded-md p-3 cursor-pointer hover:bg-blue-50 mr-3"
            onClick={toggleResponseEdit}
          >
            <div className="text-blue-600 font-medium mb-2">Model's Response</div>
            <span className="bg-blue-100 text-blue-600 px-3 py-1 rounded font-mono">response</span>
          </div>
          <div className="flex items-center">
            <div 
              className="border border-gray-200 bg-white rounded-md p-3 cursor-pointer hover:bg-gray-50"
              onClick={toggleAgentId}
            >
              <div className="text-gray-600 font-medium mb-2">This Agent's ID</div>
              <span className="bg-green-100 text-green-700 px-3 py-1 rounded font-mono">agentId</span>
            </div>
            {agentIdVisible && (
              <div className="ml-3 bg-gray-100 px-4 py-2 rounded-md border">
                <span className="text-gray-800 font-medium">agentId = {agentId}</span>
              </div>
            )}
          </div>
        </div>
        
        <button
          onClick={handleRunModel}
          disabled={isRunningModel}
          className={`flex items-center space-x-2 px-4 py-2 rounded font-medium text-white ${
            isRunningModel ? 'bg-green-400' : 'bg-green-500 hover:bg-green-600'
          }`}
        >
          <Play className="h-4 w-4" />
          <span>{isRunningModel ? 'Running...' : 'Run Model'}</span>
        </button>
      </div>
      
      {/* Response Editor shown when clicking Model's Response button or after running model */}
      {testResponseVisible && (
        <div className="border rounded-md overflow-hidden mb-6">
          <div className="bg-gray-100 px-3 py-2 border-b flex justify-between items-center">
            <span className="font-medium text-sm">Model Response</span>
            <span className="text-xs">{testResponse ? testResponse.length : 0} characters</span>
          </div>
          <textarea
            ref={testResponseRef}
            value={testResponse}
            onChange={(e) => setTestResponse(e.target.value)}
            className="w-full p-3 font-mono text-sm"
            rows={5}
            placeholder="The model response will appear here after running the model..."
          />
        </div>
      )}
      
      {/* We now have the separate model response section that can be toggled */}
      
      {/* Code Editor */}
      <div className="border border-gray-300 rounded-md overflow-hidden shadow-sm">
        <div className="bg-gray-100 border-b px-3 py-2 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Terminal className="h-4 w-4 text-gray-700" />
            <span className="text-sm font-medium text-gray-700">Agent Code</span>
            <span className="bg-blue-100 text-blue-700 text-xs px-2 py-0.5 rounded">JavaScript</span>
          </div>
          <div className="flex items-center space-x-3">
            {testResponse && (
              <button
                onClick={handleRunCode}
                disabled={isRunningCode}
                className={`flex items-center space-x-1 px-3 py-1 rounded text-sm font-medium text-white ${
                  isRunningCode ? 'bg-blue-400' : 'bg-blue-500 hover:bg-blue-600'
                }`}
              >
                <Terminal className="h-3 w-3" />
                <span>{isRunningCode ? 'Executing...' : 'Run Code'}</span>
              </button>
            )}
            <div className="text-xs text-gray-500">
              {code.length} characters
            </div>
          </div>
        </div>

        <Suspense fallback={
          <div className="flex flex-col items-center justify-center bg-gray-50 h-72 space-y-4">
            <div className="w-8 h-8 border-4 border-t-blue-500 border-blue-200 rounded-full animate-spin"></div>
            <div>
              <p className="text-gray-600 font-medium">Loading code editor...</p>
              <p className="text-xs text-gray-500 text-center">This might take a moment</p>
            </div>
          </div>
        }>
          <LazyCodeMirror
            value={code}
            height="360px"
            theme={vscodeDark}
            extensions={[javascript()]}
            onChange={(value) => onCodeChange(value)}
            basicSetup={{
              lineNumbers: true,
              highlightActiveLineGutter: true,
              highlightSpecialChars: true,
              foldGutter: true,
              dropCursor: true,
              allowMultipleSelections: true,
              indentOnInput: true,
              bracketMatching: true,
              closeBrackets: true,
              autocompletion: true,
              rectangularSelection: true,
              crosshairCursor: true,
              highlightActiveLine: true,
              highlightSelectionMatches: true,
              closeBracketsKeymap: true,
              defaultKeymap: true,
              searchKeymap: true,
              historyKeymap: true,
              foldKeymap: true,
              completionKeymap: true,
              lintKeymap: true,
            }}
          />
        </Suspense>
      </div>
      
      {/* Output Display (below the code editor) */}
      {testOutput && (
        <div className="border rounded-md overflow-hidden mt-4">
          <div className="bg-gray-100 px-3 py-2 border-b">
            <span className="font-medium text-sm">Execution Output</span>
          </div>
          <pre className="p-3 font-mono text-sm bg-gray-50 overflow-auto max-h-40">
            {testOutput}
          </pre>
        </div>
      )}

      {/* Code Snippets Section */}
      <div className="bg-white border rounded-md shadow-sm">
        <div className="bg-gray-50 px-4 py-3 border-b flex justify-between items-center">
          <h3 className="font-medium">Quick Code Snippets</h3>
          {selectedCodeSnippet ? (
            <div className="flex space-x-2">
              <button 
                onClick={() => setSelectedCodeSnippet(null)}
                className="px-2 py-1 text-xs text-gray-600 bg-gray-200 rounded"
              >
                Cancel
              </button>
              <button 
                onClick={() => {
                  const snippet = codeSnippets.find(s => s.name === selectedCodeSnippet);
                  if (snippet) insertCodeSnippet(snippet.code);
                }}
                className="px-2 py-1 text-xs text-white bg-blue-500 rounded flex items-center"
              >
                <Copy className="h-3 w-3 mr-1" />
                Insert
              </button>
            </div>
          ) : (
            <button 
              onClick={() => {}}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              Show All
            </button>
          )}
        </div>
        
        <div className="p-3 grid grid-cols-1 md:grid-cols-3 gap-3">
          {codeSnippets.map((snippet) => (
            <div 
              key={snippet.name}
              className={`border rounded-md p-3 cursor-pointer transition-all duration-200 hover:shadow-md ${
                selectedCodeSnippet === snippet.name 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-200 hover:border-blue-300'
              }`}
              onClick={() => setSelectedCodeSnippet(snippet.name)}
            >
              <div className="flex justify-between items-start mb-1">
                <h4 className="font-medium text-gray-800">{snippet.name}</h4>
              </div>
              <p className="text-xs text-gray-600 mb-1">{snippet.description}</p>
              <div className="text-xs text-gray-500 bg-gray-50 p-1 rounded truncate">
                {snippet.code.split('\n')[0]}...
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CodeTab;
