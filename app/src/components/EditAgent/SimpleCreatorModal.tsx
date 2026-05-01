import React, { useState, useRef, useEffect, useCallback } from 'react';
import Modal from '@components/EditAgent/Modal';
import SensorInputText from '@components/EditAgent/SensorInputText';
import { SimpleTool, ToolData } from '@utils/agentTemplateManager';
import { Model, listModels, SKIP_MODEL_SENTINEL } from '@utils/inferenceServer';
import { isIOS } from '@utils/platform';
import { openUrl } from '@tauri-apps/plugin-opener';
import { listAgents, CompleteAgent } from '@utils/agent_database';
import {
  Bell, Save, Monitor, ScanText, Eye, Camera, Clipboard, Mic, ArrowRight, ArrowLeft, ChevronDown, AlertTriangle, Info, Loader2, CheckCircle2, MessageSquare, Smartphone, Mail, Volume2, Blend, Clapperboard, Tag, HelpCircle, MessageCircle, Images, Server, MousePointer, Phone, PartyPopper, MinusCircle
} from 'lucide-react';


// --- NEW: Actual SVG components for Discord and WhatsApp ---
const DiscordIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="24" height="24" viewBox="0 0 24 24" fill="currentColor" {...props}>
    <path d="M 8.3164062 4.0039062 C 7.4484062 4.0039062 6.1811406 4.3226094 5.2441406 4.5996094 C 4.3771406 4.8556094 3.6552344 5.4479531 3.2402344 6.2519531 C 2.4062344 7.8649531 1.166625 11.064 1.015625 16 C 0.995625 16.67 1.2953594 17.314375 1.8183594 17.734375 C 2.8833594 18.589375 4.5907656 19.659375 7.0097656 19.984375 C 7.0847656 19.995375 7.1603281 20 7.2363281 20 C 7.7603281 20 8.2630781 19.758031 8.5800781 19.332031 L 9 18.65625 C 9 18.65625 10.653 19.019531 12 19.019531 L 12.013672 19.019531 C 13.360672 19.019531 15.013672 18.65625 15.013672 18.65625 L 15.433594 19.330078 C 15.751594 19.757078 16.253344 20 16.777344 20 C 16.853344 20 16.929859 19.994375 17.005859 19.984375 C 19.423859 19.659375 21.130313 18.589375 22.195312 17.734375 C 22.718312 17.314375 23.02 16.671953 23 16.001953 C 22.85 11.065953 21.607437 7.8659531 20.773438 6.2519531 C 20.358438 5.4489531 19.638484 4.8556094 18.771484 4.5996094 C 17.832484 4.3216094 16.565266 4.0039062 15.697266 4.0039062 C 15.443266 4.0039062 15.223641 4.0317031 15.056641 4.0957031 C 14.316641 4.3757031 14.001953 5.1445313 14.001953 5.1445312 C 14.001953 5.1445312 12.686625 4.9882813 12.015625 4.9882812 L 12 4.9882812 C 11.329 4.9882812 10.013672 5.1445312 10.013672 5.1445312 C 10.013672 5.1445312 9.6970313 4.37475 8.9570312 4.09375 C 8.7890312 4.03075 8.5704063 4.0039062 8.3164062 4.0039062 z M 8.3164062 5.5039062 C 8.3804063 5.5039062 8.4242188 5.5067656 8.4492188 5.5097656 C 8.5122188 5.5507656 8.5970469 5.6608906 8.6230469 5.7128906 C 8.8560469 6.2808906 9.4097187 6.6445312 10.011719 6.6445312 C 10.069719 6.6445312 10.1275 6.6417656 10.1875 6.6347656 C 10.6625 6.5787656 11.574672 6.4882812 12.013672 6.4882812 C 12.490672 6.4882812 13.484172 6.5947656 13.826172 6.6347656 C 13.889172 6.6417656 13.951672 6.6445312 14.013672 6.6445312 C 14.609672 6.6445312 15.145953 6.3081406 15.376953 5.7441406 C 15.414953 5.6631406 15.501453 5.5527188 15.564453 5.5117188 C 15.589453 5.5087187 15.633266 5.5058594 15.697266 5.5058594 C 16.231266 5.5058594 17.196703 5.7000625 18.345703 6.0390625 C 18.824703 6.1810625 19.213406 6.5004062 19.441406 6.9414062 C 20.148406 8.3094063 21.356 11.312828 21.5 16.048828 C 21.506 16.246828 21.415812 16.439406 21.257812 16.566406 C 19.933812 17.630406 18.435297 18.280953 16.779297 18.501953 C 16.732297 18.501953 16.68725 18.485984 16.65625 18.458984 L 16.390625 18.029297 C 16.833564 17.838865 17.256029 17.625199 17.640625 17.390625 C 17.994625 17.174625 18.105625 16.713375 17.890625 16.359375 C 17.674625 16.005375 17.212375 15.895375 16.859375 16.109375 C 15.733375 16.796375 13.795906 17.488281 12.003906 17.488281 C 10.216906 17.488281 8.2754844 16.796375 7.1464844 16.109375 C 6.7934844 15.894375 6.3321875 16.005375 6.1171875 16.359375 C 5.9021875 16.712375 6.0131875 17.175625 6.3671875 17.390625 C 6.7539143 17.625908 7.1782112 17.838346 7.6230469 18.029297 L 7.3574219 18.457031 C 7.3274219 18.483031 7.2817969 18.498047 7.2167969 18.498047 L 7.2070312 18.498047 C 5.5780312 18.279047 4.0818125 17.629406 2.7578125 16.566406 C 2.5998125 16.439406 2.509625 16.244875 2.515625 16.046875 C 2.659625 11.311875 3.8672187 8.3104063 4.5742188 6.9414062 C 4.8012188 6.5004062 5.1899219 6.1791094 5.6699219 6.0371094 C 6.8189219 5.6971094 7.7834062 5.5039062 8.3164062 5.5039062 z M 8.5 10 A 1.5 2 0 0 0 8.5 14 A 1.5 2 0 0 0 8.5 10 z M 15.5 10 A 1.5 2 0 0 0 15.5 14 A 1.5 2 0 0 0 15.5 10 z"></path>
  </svg>
);

const WhatsAppIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
<svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="50" height="50" viewBox="0 0 50 50" fill="currentColor" {...props}>
<path d="M 25 2 C 12.309534 2 2 12.309534 2 25 C 2 29.079097 3.1186875 32.88588 4.984375 36.208984 L 2.0371094 46.730469 A 1.0001 1.0001 0 0 0 3.2402344 47.970703 L 14.210938 45.251953 C 17.434629 46.972929 21.092591 48 25 48 C 37.690466 48 48 37.690466 48 25 C 48 12.309534 37.690466 2 25 2 z M 25 4 C 36.609534 4 46 13.390466 46 25 C 46 36.609534 36.609534 46 25 46 C 21.278025 46 17.792121 45.029635 14.761719 43.333984 A 1.0001 1.0001 0 0 0 14.033203 43.236328 L 4.4257812 45.617188 L 7.0019531 36.425781 A 1.0001 1.0001 0 0 0 6.9023438 35.646484 C 5.0606869 32.523592 4 28.890107 4 25 C 4 13.390466 13.390466 4 25 4 z M 16.642578 13 C 16.001539 13 15.086045 13.23849 14.333984 14.048828 C 13.882268 14.535548 12 16.369511 12 19.59375 C 12 22.955271 14.331391 25.855848 14.613281 26.228516 L 14.615234 26.228516 L 14.615234 26.230469 C 14.588494 26.195329 14.973031 26.752191 15.486328 27.419922 C 15.999626 28.087653 16.717405 28.96464 17.619141 29.914062 C 19.422612 31.812909 21.958282 34.007419 25.105469 35.349609 C 26.554789 35.966779 27.698179 36.339417 28.564453 36.611328 C 30.169845 37.115426 31.632073 37.038799 32.730469 36.876953 C 33.55263 36.755876 34.456878 36.361114 35.351562 35.794922 C 36.246248 35.22873 37.12309 34.524722 37.509766 33.455078 C 37.786772 32.688244 37.927591 31.979598 37.978516 31.396484 C 38.003976 31.104927 38.007211 30.847602 37.988281 30.609375 C 37.969311 30.371148 37.989581 30.188664 37.767578 29.824219 C 37.302009 29.059804 36.774753 29.039853 36.224609 28.767578 C 35.918939 28.616297 35.048661 28.191329 34.175781 27.775391 C 33.303883 27.35992 32.54892 26.991953 32.083984 26.826172 C 31.790239 26.720488 31.431556 26.568352 30.914062 26.626953 C 30.396569 26.685553 29.88546 27.058933 29.587891 27.5 C 29.305837 27.918069 28.170387 29.258349 27.824219 29.652344 C 27.819619 29.649544 27.849659 29.663383 27.712891 29.595703 C 27.284761 29.383815 26.761157 29.203652 25.986328 28.794922 C 25.2115 28.386192 24.242255 27.782635 23.181641 26.847656 L 23.181641 26.845703 C 21.603029 25.455949 20.497272 23.711106 20.148438 23.125 C 20.171937 23.09704 20.145643 23.130901 20.195312 23.082031 L 20.197266 23.080078 C 20.553781 22.728924 20.869739 22.309521 21.136719 22.001953 C 21.515257 21.565866 21.68231 21.181437 21.863281 20.822266 C 22.223954 20.10644 22.02313 19.318742 21.814453 18.904297 L 21.814453 18.902344 C 21.828863 18.931014 21.701572 18.650157 21.564453 18.326172 C 21.426943 18.001263 21.251663 17.580039 21.064453 17.130859 C 20.690033 16.232501 20.272027 15.224912 20.023438 14.634766 L 20.023438 14.632812 C 19.730591 13.937684 19.334395 13.436908 18.816406 13.195312 C 18.298417 12.953717 17.840778 13.022402 17.822266 13.021484 L 17.820312 13.021484 C 17.450668 13.004432 17.045038 13 16.642578 13 z M 16.642578 15 C 17.028118 15 17.408214 15.004701 17.726562 15.019531 C 18.054056 15.035851 18.033687 15.037192 17.970703 15.007812 C 17.906713 14.977972 17.993533 14.968282 18.179688 15.410156 C 18.423098 15.98801 18.84317 16.999249 19.21875 17.900391 C 19.40654 18.350961 19.582292 18.773816 19.722656 19.105469 C 19.863021 19.437122 19.939077 19.622295 20.027344 19.798828 L 20.027344 19.800781 L 20.029297 19.802734 C 20.115837 19.973483 20.108185 19.864164 20.078125 19.923828 C 19.867096 20.342656 19.838461 20.445493 19.625 20.691406 C 19.29998 21.065838 18.968453 21.483404 18.792969 21.65625 C 18.639439 21.80707 18.36242 22.042032 18.189453 22.501953 C 18.016221 22.962578 18.097073 23.59457 18.375 24.066406 C 18.745032 24.6946 19.964406 26.679307 21.859375 28.347656 C 23.05276 29.399678 24.164563 30.095933 25.052734 30.564453 C 25.940906 31.032973 26.664301 31.306607 26.826172 31.386719 C 27.210549 31.576953 27.630655 31.72467 28.119141 31.666016 C 28.607627 31.607366 29.02878 31.310979 29.296875 31.007812 L 29.298828 31.005859 C 29.655629 30.601347 30.715848 29.390728 31.224609 28.644531 C 31.246169 28.652131 31.239109 28.646231 31.408203 28.707031 L 31.408203 28.708984 L 31.410156 28.708984 C 31.487356 28.736474 32.454286 29.169267 33.316406 29.580078 C 34.178526 29.990889 35.053561 30.417875 35.337891 30.558594 C 35.748225 30.761674 35.942113 30.893881 35.992188 30.894531 C 35.995572 30.982516 35.998992 31.07786 35.986328 31.222656 C 35.951258 31.624292 35.8439 32.180225 35.628906 32.775391 C 35.523582 33.066746 34.975018 33.667661 34.283203 34.105469 C 33.591388 34.543277 32.749338 34.852514 32.4375 34.898438 C 31.499896 35.036591 30.386672 35.087027 29.164062 34.703125 C 28.316336 34.437036 27.259305 34.092596 25.890625 33.509766 C 23.114812 32.325956 20.755591 30.311513 19.070312 28.537109 C 18.227674 27.649908 17.552562 26.824019 17.072266 26.199219 C 16.592866 25.575584 16.383528 25.251054 16.208984 25.021484 L 16.207031 25.019531 C 15.897202 24.609805 14 21.970851 14 19.59375 C 14 17.077989 15.168497 16.091436 15.800781 15.410156 C 16.132721 15.052495 16.495617 15 16.642578 15 z"></path>
</svg>
);

// --- Reusable Model Selector (for Step 2) ---
interface ModelSelectorProps {
  availableModels: Model[];
  selectedModel: string;
  onSelectModel: (modelName: string) => void;
  loading: boolean;
}
const ModelSelector: React.FC<ModelSelectorProps> = ({ availableModels, selectedModel, onSelectModel, loading }) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="relative">
      <button onClick={() => setIsOpen(!isOpen)} disabled={loading} className="w-full p-2 bg-white border-gray-300 rounded-md flex justify-between items-center text-left">
        <span className="truncate">{selectedModel || (loading ? 'Loading…' : 'Select model')}</span>
        <ChevronDown className="h-4 w-4 text-gray-400" />
      </button>
      {isOpen && (
        <div className="absolute z-20 mt-1 w-full max-h-48 bg-white border border-gray-300 rounded-md shadow-lg overflow-y-auto">
          {availableModels.map((m) => (
            <button key={m.name} onClick={() => { onSelectModel(m.name); setIsOpen(false); }} className={`w-full text-left px-3 py-2 text-xs flex justify-between items-center ${selectedModel === m.name ? 'bg-blue-500 text-white' : 'hover:bg-gray-100'}`}>
              <span className="truncate pr-2">{m.name}</span>
              <div className="flex items-center space-x-1">
                {m.multimodal && <span title="Supports Vision" className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded ${selectedModel === m.name ? 'bg-blue-400 text-white' : 'text-purple-600 bg-purple-100'}`}><Eye className="h-3.5 w-3.5 mr-1" />Vision</span>}
                {m.server === SKIP_MODEL_SENTINEL && <span title="No model call" className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded ${selectedModel === m.name ? 'bg-blue-400 text-white' : 'text-gray-500 bg-gray-100'}`}><MinusCircle className="h-3.5 w-3.5 mr-1" />Skip</span>}
                {!m.server.includes('api.observer-ai.com') && m.server !== SKIP_MODEL_SENTINEL && <span title="Running Locally" className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded ${selectedModel === m.name ? 'bg-blue-400 text-white' : 'text-gray-600 bg-gray-100'}`}><Server className="h-3.5 w-3.5 mr-1" />Local</span>}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};


// --- Sensor Button Helper ---
const SensorButton = ({ icon: Icon, label, colorClass, onClick }: { icon: React.ElementType, label: string, colorClass?: string, onClick: () => void }) => (
  <button onClick={onClick} className={`flex items-center space-x-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors ${colorClass || 'text-gray-700'}`}>
    <Icon className="h-5 w-5" />
    <span className="text-sm font-medium">{label}</span>
  </button>
);

// --- NEW: Warning message component for Step 3 ---
const ToolWarning = ({ icon: Icon, message, colorClass }: { icon: React.ElementType, message: React.ReactNode, colorClass: string }) => (
  <div className={`p-3 rounded-lg flex items-start space-x-3 text-sm ${colorClass}`}>
    <Icon className="h-5 w-5 mt-0.5 flex-shrink-0" />
    <span>{message}</span>
  </div>
);
// --- END NEW ---

// --- Cloud Privacy Notice ---
interface SensorDataType {
  sensor: string;
  label: string;
}

const SENSOR_DATA_MAP: SensorDataType[] = [
  { sensor: '$SCREEN', label: 'Screen captures (images)' },
  { sensor: '$SCREEN_OCR', label: 'Text extracted from your screen' },
  { sensor: '$CAMERA', label: 'Camera images' },
  { sensor: '$CAMERA_OCR', label: 'Text extracted from camera' },
  { sensor: '$CLIPBOARD', label: 'Clipboard content' },
  { sensor: '$MICROPHONE', label: 'Microphone audio transcription' },
  { sensor: '$SCREEN_AUDIO', label: 'Screen audio transcription' },
  { sensor: '$ALL_AUDIO', label: 'Microphone and screen audio transcription' },
];

function getDetectedSensorData(prompt: string): SensorDataType[] {
  return SENSOR_DATA_MAP.filter(item => {
    const key = item.sensor.slice(1); // strip leading $
    return new RegExp(`\\$${key}(?![A-Z_])`).test(prompt);
  });
}

function isCloudModel(model: Model | undefined): boolean {
  if (!model) return false;
  return model.server.includes('api.observer-ai.com');
}

const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  'gemini': 'Google AI Studio',
  'gemini-pro': 'Google AI Studio',
  'fireworks': 'Fireworks.ai',
  'openrouter': 'OpenRouter',
  'openai': 'OpenAI',
  'anthropic': 'Anthropic',
};

function getProviderName(model: Model | undefined): string {
  if (!model?.ownedBy) return 'a third-party AI service';
  return PROVIDER_DISPLAY_NAMES[model.ownedBy.toLowerCase()] || model.ownedBy;
}

interface CloudPrivacyNoticeProps {
  detectedData: SensorDataType[];
  providerName: string;
}

const PrivacyLink: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const privacyUrl = 'https://observer-ai.com/#/Privacy';

  if (isIOS()) {
    return (
      <button onClick={() => openUrl(privacyUrl)} className="inline-flex items-center text-blue-500 hover:text-blue-600">
        {children}
      </button>
    );
  }
  return (
    <a href={privacyUrl} target="_blank" rel="noopener noreferrer" className="inline-flex items-center text-blue-500 hover:text-blue-600">
      {children}
    </a>
  );
};

const CloudPrivacyNotice: React.FC<CloudPrivacyNoticeProps> = ({ detectedData, providerName }) => {
  return (
    <div className="flex-1 flex items-center space-x-2 text-sm text-gray-600 px-4">
      <PrivacyLink><Info className="h-4 w-4 flex-shrink-0" /></PrivacyLink>
      <div>
        <span>This agent uses a cloud model hosted by <strong>{providerName}</strong>.</span>
        {detectedData.length > 0 && (
          <span> When running, the following will be sent: <strong>{detectedData.map(d => d.label.toLowerCase()).join(', ')}</strong>.</span>
        )}
      </div>
    </div>
  );
};
// --- END Cloud Privacy Notice ---


// --- MAIN WIZARD COMPONENT ---
interface SimpleCreatorModalProps {
  isOpen: boolean; onClose: () => void; onNext: (config: any) => void; isAuthenticated: boolean; hostingContext?: 'official-web' | 'self-hosted';
}
const SimpleCreatorModal: React.FC<SimpleCreatorModalProps> = ({ isOpen, onClose, onNext, isAuthenticated, hostingContext }) => {
  const [step, setStep] = useState(1);
  const [name, setName] = useState('');
  const [agentId, setAgentId] = useState('');
  const [model, setModel] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  
  const [selectedTools, setSelectedTools] = useState<Map<SimpleTool, ToolData>>(new Map());
  const [smsPhoneNumber, setSmsPhoneNumber] = useState('');
  const [whatsappPhoneNumber, setWhatsappPhoneNumber] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
  const [emailAddress, setEmailAddress] = useState('');
  const [pushoverUserKey, setPushoverUserKey] = useState('');
  const [discordWebhookUrl, setDiscordWebhookUrl] = useState('');
  const [telegramChatId, setTelegramChatId] = useState('');

  const [conditionEnabled, setConditionEnabled] = useState(false);
  const [conditionKeyword, setConditionKeyword] = useState('');
  
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [existingAgents, setExistingAgents] = useState<CompleteAgent[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [visionValidationError, setVisionValidationError] = useState<string | null>(null);
  const [showWhisperWarning, setShowWhisperWarning] = useState(false); 

  // --- NEW: State for contextual warnings ---
  const [showDesktopNotifWarning, setShowDesktopNotifWarning] = useState(false);
  const [showSmsWarning, setShowSmsWarning] = useState(false);
  const [showWhatsappWarning, setShowWhatsappWarning] = useState(false);
  // --- END NEW ---
  
  const promptRef = useRef<HTMLTextAreaElement>(null);

  const resetState = useCallback(() => {
    setStep(1); setName(''); setAgentId(''); setModel(''); setSystemPrompt('');
    setSelectedTools(new Map());
    setSmsPhoneNumber('');
    setWhatsappPhoneNumber('');
    setEmailAddress('');
    setPushoverUserKey('');
    setDiscordWebhookUrl('');
    setTelegramChatId('');
    setConditionEnabled(false); setConditionKeyword('');
    // --- NEW: Reset warnings on close ---
    setShowDesktopNotifWarning(false);
    setShowSmsWarning(false);
    setShowWhatsappWarning(false);
    // --- END NEW ---
  }, []);

  const fetchInitialData = useCallback(async () => {
    setLoadingModels(true);
    try {
      const modelsResponse = listModels();
      const agentsResponse = await listAgents();
      if (modelsResponse.models) {
        setAvailableModels(modelsResponse.models);
        if (modelsResponse.models.length > 0 && !model) {
          setModel(modelsResponse.models[0].name);
        }
      }
      setExistingAgents(agentsResponse);
    } catch (error) { console.error("Failed to fetch initial data", error); }
    finally { setLoadingModels(false); }
  }, [isOpen, model]);

  useEffect(() => { fetchInitialData(); }, [fetchInitialData]);

  useEffect(() => {
    if (!name) { setAgentId(''); return; }
    let baseId = name.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '');
    let finalId = baseId;
    let counter = 2;
    const existingIds = new Set(existingAgents.map(a => a.id));
    while (existingIds.has(finalId)) {
      finalId = `${baseId}_${counter++}`;
    }
    setAgentId(finalId);
  }, [name, existingAgents]);

  useEffect(() => {
    // Vision validation
    const hasVisionSensor = /\$SCREEN(?![A-Z_])|\$CAMERA(?![A-Z_])/.test(systemPrompt);
    const selectedModelInfo = availableModels.find(m => m.name === model);
    const isLocalModel = selectedModelInfo && !selectedModelInfo.server.includes('api.observer-ai.com');

    if (hasVisionSensor && selectedModelInfo && !selectedModelInfo.multimodal && !isLocalModel) {
      setVisionValidationError("This model does not support images. Please select a 'Vision' model.");
    } else {
      setVisionValidationError(null);
    }
    // Whisper warning
    const hasAudioSensor = /\$MICROPHONE|\$SCREEN_AUDIO|\$ALL_AUDIO/.test(systemPrompt);
    setShowWhisperWarning(hasAudioSensor);
  }, [systemPrompt, model, availableModels]);

  const insertSensor = (tag: string) => {
    let finalTag = tag;
    if (tag === '$MEMORY@agent_id') {
      finalTag = `$MEMORY@${agentId || 'your_agent_id'}`;
    } else if (tag === '$IMEMORY@agent_id') {
      finalTag = `$IMEMORY@${agentId || 'your_agent_id'}`;
    }
    if (!promptRef.current) return;
    const { selectionStart, selectionEnd, value } = promptRef.current;
    const newPrompt = `${value.substring(0, selectionStart)} ${finalTag} ${value.substring(selectionEnd)}`;
    setSystemPrompt(newPrompt);
    setTimeout(() => {
      promptRef.current?.focus();
      const newPos = selectionStart + finalTag.length + 2;
      promptRef.current?.setSelectionRange(newPos, newPos);
    }, 0);
  };
  
  // --- MODIFIED: toggleTool now handles showing/hiding warnings ---
  const toggleTool = (tool: SimpleTool) => {
    // Show warnings on selection
    if (tool === 'notification') setShowDesktopNotifWarning(true);
    if (tool === 'sms') setShowSmsWarning(true);
    if (tool === 'whatsapp') setShowWhatsappWarning(true);

    // Hide warnings if the tool is being deselected
    if (tool === 'notification' && selectedTools.has(tool)) setShowDesktopNotifWarning(false);
    if (tool === 'sms' && selectedTools.has(tool)) setShowSmsWarning(false);
    if (tool === 'whatsapp' && selectedTools.has(tool)) setShowWhatsappWarning(false);


    setSelectedTools(prev => {
      const newMap = new Map(prev);
      if (newMap.has(tool)) {
        newMap.delete(tool);
        // If "Start Recording" is turned off, also turn off "Label Recording"
        if (tool === 'start_clip') {
          newMap.delete('mark_clip');
        }
      } else {
        const initialData: ToolData =
            tool === 'sms' ? { smsPhoneNumber } :
            tool === 'whatsapp' ? { whatsappPhoneNumber } :
            tool === 'call' ? { phoneNumber } :
            tool === 'email' ? { emailAddress } :
            tool === 'pushover' ? { pushoverUserKey } :
            tool === 'discord' ? { discordWebhookUrl } :
            tool === 'telegram' ? { telegramChatId } :
            {};
        newMap.set(tool, initialData);
      }
      return newMap;
    });
  };
  // --- END MODIFICATION ---
  
  const handleNext = () => {
    if (step === 3) {
      const config = {
        agentData: { name, id: agentId, model_name: model, system_prompt: systemPrompt },
        selectedTools: selectedTools,
        condition: { enabled: conditionEnabled, keyword: conditionKeyword }
      };
      onNext(config);
      resetState();
      onClose();
    } else {
      setStep(s => s + 1);
    }
  };
  
  const handleBack = () => setStep(s => s - 1);
  
  const handleCloseAndReset = () => {
    resetState();
    onClose();
  };

  const isStep1Valid = name && agentId && model;
  const isStep2Valid = systemPrompt.trim() && model;

  return (
    <Modal open={isOpen} onClose={handleCloseAndReset} className="w-full max-w-4xl h-[700px] flex flex-col">
      <div className="p-6 border-b flex-shrink-0">
        <h2 className="text-2xl font-bold text-gray-900">Create a New Agent</h2>
        <p className="text-gray-500 mt-1">Step {step} of 3: {step === 1 ? 'Setup' : step === 2 ? 'Prompt' : 'Actions'}</p>
      </div>
      
      <div className="flex-grow relative overflow-hidden">
        <div className="absolute inset-0 flex transition-transform duration-300 ease-in-out" style={{ transform: `translateX(-${(step - 1) * 100}%)` }}>
          {/* --- Step 1: Setup --- */}
          <div className="w-full flex-shrink-0 p-8 grid grid-cols-1 md:grid-cols-2 gap-8 overflow-y-auto">
            <div className="space-y-6">
              <h3 className="text-xl font-semibold text-gray-800">1. Name Your Agent</h3>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Agent Name</label>
                <input value={name} onChange={(e) => setName(e.target.value)} className="w-full p-3 bg-white border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500" placeholder="My Screen Watcher" autoFocus/>
                {agentId && <p className="text-xs text-gray-500 mt-2">ID: <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded-md">{agentId}</span></p>}
              </div>
            </div>
            <div className="space-y-6 flex flex-col">
              <h3 className="text-xl font-semibold text-gray-800">2. Choose a Model</h3>
              <div className="flex-grow border border-gray-200 rounded-lg bg-gray-50 p-3 overflow-y-auto">
                {loadingModels && <div className="p-4 text-center text-gray-500 flex items-center justify-center h-full"><Loader2 className="h-5 w-5 mr-2 animate-spin"/>Loading...</div>}
                <div className="flex flex-wrap gap-2">
                    {!loadingModels && availableModels.map((m) => (
                        <button key={m.name} onClick={() => setModel(m.name)} className={`relative group px-3 py-2 rounded-md border text-sm font-medium flex items-center space-x-2 transition-all duration-150 ${model === m.name ? 'bg-blue-600 text-white border-blue-600 shadow-md' : 'bg-white text-gray-700 border-gray-300 hover:border-blue-400'}`}>
                            {model === m.name && <CheckCircle2 className="h-4 w-4" />}
                            <span className="truncate">{m.name}</span>
                            {m.multimodal && (<span title="Vision Model"><Eye className="h-4 w-4 text-purple-400 group-hover:text-purple-500" /></span>)}
                            {m.server === SKIP_MODEL_SENTINEL && (<span title="No model call"><MinusCircle className="h-4 w-4 text-gray-400 group-hover:text-gray-500" /></span>)}
                            {!m.server.includes('api.observer-ai.com') && m.server !== SKIP_MODEL_SENTINEL && (<span title="Running Locally"><Server className="h-4 w-4 text-gray-400 group-hover:text-gray-500" /></span>)}
                        </button>
                    ))}
                </div>
              </div>
            </div>
          </div>

          {/* --- Step 2: Prompt & Sensors --- */}
          <div className="w-full flex-shrink-0 p-8 flex flex-col overflow-y-auto">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Prompt & Sensors</h3>
            <SensorInputText
              value={systemPrompt}
              onChange={setSystemPrompt}
              textareaRef={promptRef}
              className="flex-grow min-h-[250px]"
              placeholder="e.g., Look at the screen for 'ERROR'..."
            />
            <div className="flex flex-wrap gap-2 mt-4">
              <SensorButton icon={ScanText} label="Screen Text" onClick={() => insertSensor('$SCREEN_OCR')} colorClass="text-blue-600" />
              <SensorButton icon={Monitor} label="Screen Image" onClick={() => insertSensor('$SCREEN')} colorClass="text-purple-600" />
              <SensorButton icon={Camera} label="Camera" onClick={() => insertSensor('$CAMERA')} colorClass="text-purple-600" />
              <SensorButton icon={ScanText} label="Camera Text" onClick={() => insertSensor('$CAMERA_OCR')} colorClass="text-blue-600" />
              <SensorButton icon={Clipboard} label="Clipboard" onClick={() => insertSensor('$CLIPBOARD')} colorClass="text-sky-600" />
              <SensorButton icon={Mic} label="Microphone" onClick={() => insertSensor('$MICROPHONE')} colorClass="text-amber-600" />
              <SensorButton icon={Volume2} label="Screen Audio" onClick={() => insertSensor('$SCREEN_AUDIO')} colorClass="text-amber-600" />
              <SensorButton icon={Blend} label="All Audio" onClick={() => insertSensor('$ALL_AUDIO')} colorClass="text-orange-600" />
              <SensorButton icon={Save} label="Memory" onClick={() => insertSensor('$MEMORY@agent_id')} colorClass="text-emerald-600" />
              <SensorButton icon={Images} label="Image Memory" onClick={() => insertSensor('$IMEMORY@agent_id')} colorClass="text-purple-600" />
              <SensorButton icon={ScanText} label="Image Memory Text" onClick={() => insertSensor('$IMEMORY_OCR@agent_id')} colorClass="text-blue-600" />
            </div>
            {visionValidationError && <div className="mt-2 p-2 bg-yellow-50 rounded-md flex items-center text-xs text-yellow-800"><AlertTriangle className="h-4 w-4 mr-2 flex-shrink-0" />{visionValidationError}</div>}
            {/\$SCREEN_OCR|\$CAMERA_OCR|\$IMEMORY_OCR/.test(systemPrompt) && <div className="mt-2 p-2 bg-blue-50 rounded-md flex items-center text-xs text-blue-800"><Info className="h-4 w-4 mr-2 flex-shrink-0" />OCR adds ~15s to each agent loop.</div>}
            {showWhisperWarning && <div className="mt-2 p-2 bg-slate-100 rounded-md flex items-center text-xs text-slate-800"><AlertTriangle className="h-4 w-4 mr-2 flex-shrink-0" />Uses whisper model, may increase CPU/memory usage.</div>}
            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Change Model</label>
              <ModelSelector availableModels={availableModels} selectedModel={model} onSelectModel={setModel} loading={loadingModels} />
            </div>
          </div>

          {/* --- MODIFIED: Step 3: Tools --- Reorganized for clarity --- */}
          <div className="w-full flex-shrink-0 p-8 space-y-6 overflow-y-auto">
            <div>
              <h3 className="text-xl font-semibold text-gray-800">Choose Agent Actions</h3>
              <p className="text-gray-600 mt-1">What should the agent do with the model's response?</p>
            </div>
            
            <div className="p-1 bg-gray-200 rounded-lg flex"><button onClick={() => setConditionEnabled(false)} className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${!conditionEnabled ? 'bg-white text-blue-600 shadow' : 'text-gray-600 hover:bg-gray-300'}`}>Always Trigger</button><button onClick={() => setConditionEnabled(true)} className={`flex-1 py-2 text-sm font-medium rounded-md transition-colors ${conditionEnabled ? 'bg-white text-blue-600 shadow' : 'text-gray-600 hover:bg-gray-300'}`}>On Keyword</button></div>

            {conditionEnabled && (
            <div className="transition-all">
              <label className="block text-sm font-medium text-gray-700 mb-1">Trigger Condition</label>
              <div className="flex items-center bg-white border border-gray-300 rounded-lg p-3 transition-all focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500">
                <span className="font-semibold text-gray-800 pr-3">If</span>
                <input value={conditionKeyword} onChange={(e) => setConditionKeyword(e.target.value)} className="flex-grow bg-gray-50 border border-gray-300 rounded-md p-2 text-center text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500" placeholder="your keyword"/>
                <span className="text-gray-500 px-3">in</span>
                <code className="bg-gray-100 text-blue-600 font-mono text-sm px-3 py-2 rounded-md">response</code>
              </div>
            </div>
            )}
            
            {/* --- NEW: Contextual Warnings Area --- */}
            <div className="space-y-2">
                {showDesktopNotifWarning && <ToolWarning icon={AlertTriangle} colorClass="bg-yellow-50 text-yellow-800" message="Some browsers may block desktop notifications if permissions are not granted." />}
                {showSmsWarning && <ToolWarning icon={AlertTriangle} colorClass="bg-yellow-50 text-yellow-800" message="Due to A2P policy, SMS delivery to US/Canada numbers is unreliable." />}
                {showWhatsappWarning && <ToolWarning icon={AlertTriangle} colorClass="bg-yellow-50 text-yellow-800" message={<>To receive alerts, you must first message <strong>+1 (555) 783 4727</strong>. This opens a 24-hour window due to Meta's anti-spam policy.</>} />}
            </div>
            {/* --- END NEW --- */}

            {/* --- NEW: Sectioned Tool Layout --- */}
            <div className="space-y-6">
                
                {/* Section 1: Reliable Notifications */}
                <div className="space-y-4">
                  <h4 className="text-lg font-semibold text-gray-700">Reliable Notifications</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {/* Discord Tool */}
                    <button type="button" onClick={() => toggleTool('discord')} className={`group flex flex-col space-y-3 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('discord') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}><div className="flex items-center space-x-4"><DiscordIcon className={`h-8 w-8 transition-colors ${selectedTools.has('discord') ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Send to Discord</h3><p className="text-sm text-gray-500">Sends a Discord message.</p></div></div>{selectedTools.has('discord') && (<div className="relative pl-12 pt-2"><input type="text" value={discordWebhookUrl} onClick={(e) => e.stopPropagation()} onChange={(e) => { const newUrl = e.target.value; setDiscordWebhookUrl(newUrl); setSelectedTools(prev => { const newMap = new Map(prev); newMap.set('discord', { discordWebhookUrl: newUrl }); return newMap; }); }} className="w-full p-2 pl-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500" placeholder="Discord Webhook URL"/></div>)}</button>
                    {/* Email Tool */}
                    <button type="button" title={!isAuthenticated ? 'Please sign in to use this tool.' : ''} onClick={() => toggleTool('email')} disabled={!isAuthenticated} className={`group flex flex-col space-y-3 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('email') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'} disabled:opacity-50 disabled:cursor-not-allowed`}><div className="flex items-center space-x-4"><Mail className={`h-8 w-8 transition-colors ${selectedTools.has('email') ? 'text-blue-500' : 'text-gray-400 group-enabled:group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Send an Email</h3><p className="text-sm text-gray-500">Sends response as an email.</p></div></div>{selectedTools.has('email') && (<div className="relative pl-12 pt-2"><Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" /><input type="email" value={emailAddress} onClick={(e) => e.stopPropagation()} onChange={(e) => { const newAddress = e.target.value; setEmailAddress(newAddress); setSelectedTools(prev => { const newMap = new Map(prev); newMap.set('email', { emailAddress: newAddress }); return newMap; }); }} className="w-full p-2 pl-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500" placeholder="recipient@example.com"/></div>)}</button>
                    {/* Pushover Tool */}
                    <button type="button" title={!isAuthenticated ? 'Please sign in to use this tool.' : ''} onClick={() => toggleTool('pushover')} disabled={!isAuthenticated} className={`group flex flex-col space-y-3 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('pushover') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'} disabled:opacity-50 disabled:cursor-not-allowed`}><div className="flex items-center space-x-4"><Smartphone className={`h-8 w-8 transition-colors ${selectedTools.has('pushover') ? 'text-blue-500' : 'text-gray-400 group-enabled:group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Send to Pushover</h3><p className="text-sm text-gray-500">Sends a push notification.</p></div></div>{selectedTools.has('pushover') && (<div className="relative pl-12 pt-2"><input type="text" value={pushoverUserKey} onClick={(e) => e.stopPropagation()} onChange={(e) => { const newKey = e.target.value; setPushoverUserKey(newKey); setSelectedTools(prev => { const newMap = new Map(prev); newMap.set('pushover', { pushoverUserKey: newKey }); return newMap; }); }} className="w-full p-2 pl-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500" placeholder="Pushover User Key"/></div>)}</button>
                    {/* Telegram Tool */}
                    <button type="button" title={!isAuthenticated ? 'Please sign in to use this tool.' : ''} onClick={() => toggleTool('telegram')} disabled={!isAuthenticated} className={`group flex flex-col space-y-3 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('telegram') ? 'border-sky-500 bg-sky-50' : 'border-gray-300 hover:border-gray-400'} disabled:opacity-50 disabled:cursor-not-allowed`}><div className="flex items-center space-x-4"><MessageCircle className={`h-8 w-8 transition-colors ${selectedTools.has('telegram') ? 'text-sky-500' : 'text-gray-400 group-enabled:group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Send to Telegram</h3><p className="text-sm text-gray-500">Sends a Telegram message.</p></div></div>{selectedTools.has('telegram') && (<div className="relative pl-12 pt-2"><input type="text" value={telegramChatId} onClick={(e) => e.stopPropagation()} onChange={(e) => { const newChatId = e.target.value; setTelegramChatId(newChatId); setSelectedTools(prev => { const newMap = new Map(prev); newMap.set('telegram', { telegramChatId: newChatId }); return newMap; }); }} className="w-full p-2 pl-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-sky-500" placeholder="Telegram Chat ID"/></div>)}</button>
                    {/* Call Tool */}
                    <button type="button" title={!isAuthenticated ? 'Please sign in to use this tool.' : ''} onClick={() => toggleTool('call')} disabled={!isAuthenticated} className={`group flex flex-col space-y-3 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('call') ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-gray-400'} disabled:opacity-50 disabled:cursor-not-allowed`}><div className="flex items-center space-x-4"><Phone className={`h-8 w-8 transition-colors ${selectedTools.has('call') ? 'text-green-500' : 'text-gray-400 group-enabled:group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Make a Phone Call</h3><p className="text-sm text-gray-500">Makes an automated phone call.</p></div></div>{selectedTools.has('call') && (<div className="relative pl-12 pt-2"><Phone className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" /><input type="tel" value={phoneNumber} onClick={(e) => e.stopPropagation()} onChange={(e) => { const newNumber = e.target.value; setPhoneNumber(newNumber); setSelectedTools(prev => { const newMap = new Map(prev); newMap.set('call', { phoneNumber: newNumber }); return newMap; }); }} className="w-full p-2 pl-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500" placeholder="+1 555 123-4567"/></div>)}</button>
                  </div>
                </div>

                {/* Section 2: Logging & Recording */}
                <div className="space-y-4">
                  <h4 className="text-lg font-semibold text-gray-700">Logging & Recording</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <button type="button" onClick={() => toggleTool('memory')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('memory') ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-gray-400'}`}><Save className={`h-8 w-8 transition-colors ${selectedTools.has('memory') ? 'text-green-500' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Save to Memory</h3><p className="text-sm text-gray-500">Appends response to memory log.</p></div></button>
                    <button type="button" onClick={() => toggleTool('start_clip')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('start_clip') ? 'border-red-500 bg-red-50' : 'border-gray-300 hover:border-gray-400'}`}><Clapperboard className={`h-8 w-8 transition-colors ${selectedTools.has('start_clip') ? 'text-red-500' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Start Recording</h3><p className="text-sm text-gray-500">Starts a new video recording.</p></div></button>
                    <button type="button" onClick={() => toggleTool('mark_clip')} disabled={!selectedTools.has('start_clip')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('mark_clip') ? 'border-orange-500 bg-orange-50' : 'border-gray-300'} disabled:opacity-50 disabled:cursor-not-allowed enabled:hover:border-gray-400`}><Tag className={`h-8 w-8 transition-colors ${selectedTools.has('mark_clip') ? 'text-orange-500' : 'text-gray-400'} group-enabled:group-hover:text-gray-600`} /><div><h3 className="font-semibold text-gray-900">Label Recording</h3><p className="text-sm text-gray-500">Adds a label to the recording.</p></div></button>
                  </div>
                </div>

                {/* Section 3: App Specific Utils (Only for Self-Hosted) */}
                {hostingContext === 'self-hosted' && (
                  <div className="space-y-4">
                    <h4 className="text-lg font-semibold text-gray-700">App Specific Utils</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {/* Ask Tool */}
                      <button type="button" onClick={() => toggleTool('ask')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('ask') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}><HelpCircle className={`h-8 w-8 transition-colors ${selectedTools.has('ask') ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Ask User</h3><p className="text-sm text-gray-500">Shows a dialog asking for user confirmation.</p></div></button>
                      {/* System Notify Tool */}
                      <button type="button" onClick={() => toggleTool('system_notify')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('system_notify') ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-gray-400'}`}><Bell className={`h-8 w-8 transition-colors ${selectedTools.has('system_notify') ? 'text-green-500' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">System Notification</h3><p className="text-sm text-gray-500">Shows a native system notification.</p></div></button>
                      {/* Message Tool */}
                      <button type="button" onClick={() => toggleTool('message')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('message') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}><MessageCircle className={`h-8 w-8 transition-colors ${selectedTools.has('message') ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Show Message</h3><p className="text-sm text-gray-500">Shows a dialog message to the user.</p></div></button>
                      {/* Overlay Tool */}
                      <button type="button" onClick={() => toggleTool('overlay')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('overlay') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}><Monitor className={`h-8 w-8 transition-colors ${selectedTools.has('overlay') ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Show Overlay</h3><p className="text-sm text-gray-500">Displays message in translucent overlay.</p></div></button>
                      {/* Click Tool */}
                      <button type="button" onClick={() => toggleTool('click')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('click') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}><MousePointer className={`h-8 w-8 transition-colors ${selectedTools.has('click') ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Mouse Click</h3><p className="text-sm text-gray-500">Clicks at current cursor position.</p></div></button>
                      <button type="button" onClick={() => toggleTool('celebrate')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('celebrate') ? 'border-yellow-500 bg-yellow-50' : 'border-gray-300 hover:border-gray-400'}`}><PartyPopper className={`h-8 w-8 transition-colors ${selectedTools.has('celebrate') ? 'text-yellow-500' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Celebrate</h3><p className="text-sm text-gray-500">Triggers a celebration animation.</p></div></button>
                    </div>
                  </div>
                )}

                {/* Section 4: Other Notifications */}
                <div className="space-y-4">
                  <h4 className="text-lg font-semibold text-gray-700">Other Notifications</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {/* Desktop Notification Tool */}
                    <button type="button" onClick={() => toggleTool('notification')} className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('notification') ? 'border-yellow-500 bg-yellow-50' : 'border-gray-300 hover:border-gray-400'}`}><Bell className={`h-8 w-8 transition-colors ${selectedTools.has('notification') ? 'text-yellow-600' : 'text-gray-400 group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Desktop Notification</h3><p className="text-sm text-gray-500">Sends a local desktop alert.</p></div></button>
                    {/* WhatsApp Tool */}
                    <button type="button" title={!isAuthenticated ? 'Please sign in to use this tool.' : ''} onClick={() => toggleTool('whatsapp')} disabled={!isAuthenticated} className={`group flex flex-col space-y-3 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('whatsapp') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'} disabled:opacity-50 disabled:cursor-not-allowed`}><div className="flex items-center space-x-4"><WhatsAppIcon className={`h-8 w-8 transition-colors ${selectedTools.has('whatsapp') ? 'text-blue-500' : 'text-gray-400 group-enabled:group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Send a WhatsApp</h3><p className="text-sm text-gray-500">Sends a WhatsApp alert.</p></div></div>{selectedTools.has('whatsapp') && (<div className="relative pl-12 pt-2"><Smartphone className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" /><input type="tel" value={whatsappPhoneNumber} onClick={(e) => e.stopPropagation()} onChange={(e) => { const newNumber = e.target.value; setWhatsappPhoneNumber(newNumber); setSelectedTools(prev => { const newMap = new Map(prev); newMap.set('whatsapp', { whatsappPhoneNumber: newNumber }); return newMap; }); }} className="w-full p-2 pl-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500" placeholder="+1 555 123-4567"/></div>)}</button>
                    {/* SMS Tool */}
                    <button type="button" title={!isAuthenticated ? 'Please sign in to use this tool.' : ''} onClick={() => toggleTool('sms')} disabled={!isAuthenticated} className={`group flex flex-col space-y-3 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('sms') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'} disabled:opacity-50 disabled:cursor-not-allowed`}><div className="flex items-center space-x-4"><MessageSquare className={`h-8 w-8 transition-colors ${selectedTools.has('sms') ? 'text-blue-500' : 'text-gray-400 group-enabled:group-hover:text-gray-600'}`} /><div><h3 className="font-semibold text-gray-900">Send an SMS</h3><p className="text-sm text-gray-500">Sends a text message alert.</p></div></div>{selectedTools.has('sms') && (<div className="relative pl-12 pt-2"><Smartphone className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" /><input type="tel" value={smsPhoneNumber} onClick={(e) => e.stopPropagation()} onChange={(e) => { const newNumber = e.target.value; setSmsPhoneNumber(newNumber); setSelectedTools(prev => { const newMap = new Map(prev); newMap.set('sms', { smsPhoneNumber: newNumber }); return newMap; }); }} className="w-full p-2 pl-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500" placeholder="+1 555 123-4567"/></div>)}</button>
                  </div>
                </div>
            </div>
            {/* --- END SECTIONED LAYOUT --- */}
          </div>
          {/* --- END MODIFICATION --- */}
        </div>
      </div>
      
      {/* Footer */}
      <div className="p-4 border-t bg-gray-50 flex-shrink-0">
        {(() => {
          const selectedModelInfo = availableModels.find(m => m.name === model);
          const isCloud = step === 2 && isCloudModel(selectedModelInfo);
          const detectedSensors = getDetectedSensorData(systemPrompt);
          const showPrivacyConsent = isCloud && detectedSensors.length > 0;
          const buttonText = step === 3 ? 'Finish & Create' : showPrivacyConsent ? 'I Agree, Next' : 'Next';

          return (
            <>
              {/* Privacy notice - full width on mobile, above buttons */}
              {isCloud && (
                <div className="mb-3 md:hidden">
                  <CloudPrivacyNotice
                    detectedData={detectedSensors}
                    providerName={getProviderName(selectedModelInfo)}
                  />
                </div>
              )}

              {/* Button row */}
              <div className="flex items-center justify-between">
                {step > 1 ? (<button onClick={handleBack} className="inline-flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-200 rounded-md"><ArrowLeft className="h-5 w-5 mr-2" />Back</button>) : (<div></div>)}
                {/* Desktop-only inline notice */}
                {isCloud ? (
                  <div className="hidden md:block flex-1">
                    <CloudPrivacyNotice
                      detectedData={detectedSensors}
                      providerName={getProviderName(selectedModelInfo)}
                    />
                  </div>
                ) : <div className="flex-1" />}
                <button onClick={handleNext} disabled={(step === 1 && !isStep1Valid) || (step === 2 && !isStep2Valid)} className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:opacity-50">
                  {buttonText}
                  {step < 3 && <ArrowRight className="h-5 w-5 ml-2" />}
                </button>
              </div>
            </>
          );
        })()}
      </div>
    </Modal>
  );
};

export default SimpleCreatorModal;
