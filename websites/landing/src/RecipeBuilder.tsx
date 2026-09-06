// Landing-page fork of the app's RecipeSplash "When [wheel] then [wheel]" builder.
// Unlike the in-app version, there's no live MCP session or auth here — "Build it" copies
// the composed system prompt to the clipboard so the user can paste it into the app's
// agent builder themselves.

import { useEffect, useMemo, useState } from 'react';
import { Clipboard, Share } from 'lucide-react';
import EditableWheel from './wheel/EditableWheel';
import type { WheelOption } from './wheel/OptionWheel';

interface TriggerOption extends WheelOption {
  sensor: '$SCREEN' | '$CAMERA';
  promptFragment: string;
}

interface ActionOption extends WheelOption {
  actionFragment: string;
}

const TRIGGERS: TriggerOption[] = [
  { id: 'download_done',     label: 'my download is finished',     sensor: '$SCREEN', promptFragment: 'my download finishes' },
  { id: 'person_camera',     label: 'a person is on camera',       sensor: '$CAMERA', promptFragment: 'a person appears on my camera' },
  { id: 'render_fails',      label: 'my render fails',             sensor: '$SCREEN', promptFragment: 'my render fails or errors out' },
  { id: 'github_workflow',   label: 'my GitHub workflow finishes', sensor: '$SCREEN', promptFragment: 'my GitHub Actions workflow finishes running' },
  { id: 'docker_built',      label: 'my Docker image is built',    sensor: '$SCREEN', promptFragment: 'my Docker image finishes building' },
  { id: 'cowork_done',       label: 'Claude Cowork finishes',      sensor: '$SCREEN', promptFragment: 'Claude Cowork finishes its task' },
  { id: 'uber_arrives',      label: 'my Uber arrives',             sensor: '$SCREEN', promptFragment: 'my Uber arrives (the app shows the driver has arrived)' },
  { id: 'meeting_starts',    label: 'my meeting starts',           sensor: '$SCREEN', promptFragment: 'my meeting starts (a calendar notification or the meeting window appears)' },
  { id: 'minecraft_death',   label: 'my Minecraft character dies', sensor: '$SCREEN', promptFragment: 'my Minecraft character dies' },
  { id: 'steam_ready',       label: 'my Steam game is ready',      sensor: '$SCREEN', promptFragment: 'my Steam game finishes downloading and is ready to play' },
  { id: 'distracted',        label: "I'm distracted",              sensor: '$SCREEN', promptFragment: 'I get distracted (e.g. social media or YouTube on screen)' },
  { id: 'printer_fails',     label: 'my 3D printer fails',         sensor: '$CAMERA', promptFragment: 'my 3D print fails (spaghetti, detached print, or a clog)' },
  { id: 'battery_low',       label: 'my battery is low',           sensor: '$SCREEN', promptFragment: 'my battery indicator shows low battery' },
  { id: 'hour_passes',       label: 'an hour passes',              sensor: '$SCREEN', promptFragment: 'an hour passes (log what is on my screen)' },
  { id: 'tickets_available', label: 'tickets are available',       sensor: '$SCREEN', promptFragment: 'tickets become available on the page I have open' },
];

const ACTIONS: ActionOption[] = [
  { id: 'email',    label: 'send me an email',   actionFragment: 'send me an email' },
  { id: 'whatsapp', label: 'send me a WhatsApp', actionFragment: 'send me a WhatsApp message' },
  { id: 'sms',      label: 'send me an SMS',     actionFragment: 'send me an SMS' },
  { id: 'call',     label: 'call me',            actionFragment: 'call me with a phone call' },
  { id: 'telegram', label: 'send me a Telegram', actionFragment: 'send me a Telegram message' },
  { id: 'discord',  label: 'ping my Discord',    actionFragment: 'send a message to my Discord' },
  { id: 'log',      label: 'log it',             actionFragment: 'log it to memory' },
];

function composePrompt(trigger: TriggerOption | undefined, action: ActionOption | undefined): string {
  const sensor = trigger?.sensor ?? '$SCREEN';
  const watchWhat = sensor === '$CAMERA' ? 'my camera' : 'my screen';
  const triggerFrag = trigger?.promptFragment ?? '';
  const actionFrag = action?.actionFragment ?? '';
  return `Watch ${watchWhat}. When ${triggerFrag}, ${actionFrag}. Use a cloud model.`;
}

const RecipeBuilder = () => {
  const [triggerId, setTriggerId] = useState(TRIGGERS[0].id);
  const [actionId, setActionId] = useState(ACTIONS[0].id);
  const [triggerOverrides, setTriggerOverrides] = useState<Record<string, string>>({});
  const [actionOverrides, setActionOverrides] = useState<Record<string, string>>({});
  const [copied, setCopied] = useState(false);

  const triggerOptions = useMemo(
    () => TRIGGERS.map(t => triggerOverrides[t.id]
      ? { ...t, label: triggerOverrides[t.id], promptFragment: triggerOverrides[t.id] }
      : t),
    [triggerOverrides],
  );
  const actionOptions = useMemo(
    () => ACTIONS.map(a => actionOverrides[a.id]
      ? { ...a, label: actionOverrides[a.id], actionFragment: actionOverrides[a.id] }
      : a),
    [actionOverrides],
  );

  const trigger = useMemo(() => triggerOptions.find(t => t.id === triggerId), [triggerOptions, triggerId]);
  const action = useMemo(() => actionOptions.find(a => a.id === actionId), [actionOptions, actionId]);

  // Any change to the recipe invalidates whatever's on the clipboard — back to "Build it".
  useEffect(() => { setCopied(false); }, [triggerId, actionId, triggerOverrides, actionOverrides]);

  const [toast, setToast] = useState(false);
  useEffect(() => { if (toast) { const t = setTimeout(() => setToast(false), 1800); return () => clearTimeout(t); } }, [toast]);

  const handleBuild = async () => {
    const prompt = composePrompt(trigger, action);
    try {
      await navigator.clipboard.writeText(prompt);
    } catch {
      window.prompt('Copy this system prompt into the Observer agent builder:', prompt);
    }
    setCopied(true);
    setToast(true);
  };

  const handleOpenObserver = () => {
    window.open('https://app.observer-ai.com', '_blank', 'noopener,noreferrer');
  };

  return (
    <section className="pt-4 md:pt-8 pb-24 md:pb-32 bg-[#0D1321] relative" id="how-it-works">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-3">
            Build a monitoring agent
          </h2>
        </div>

        <div className="w-full flex items-center justify-center font-golos">
          <div className="flex flex-col md:flex-row md:flex-nowrap items-center justify-center gap-x-3 gap-y-4">
            <span className="text-3xl md:text-6xl font-bold text-white tracking-tight select-none">When</span>
            <EditableWheel
              options={triggerOptions}
              value={triggerId}
              onChange={setTriggerId}
              onCustom={text => setTriggerOverrides(prev => ({ ...prev, [triggerId]: text }))}
              stopped={copied}
              ariaLabel="Choose a trigger"
              widthClass="w-[15rem] md:w-[19rem]"
            />
            <span className="text-3xl md:text-6xl font-bold text-white tracking-tight select-none">then</span>
            <EditableWheel
              options={actionOptions}
              value={actionId}
              onChange={setActionId}
              onCustom={text => setActionOverrides(prev => ({ ...prev, [actionId]: text }))}
              stopped={copied}
              ariaLabel="Choose an action"
              widthClass="w-[15rem] md:w-[13rem]"
            />
          </div>
        </div>

        <div className="relative flex flex-col items-center mt-12">
          {/* "Copied!" toast — pops up above the button, then fades. */}
          <div
            className={`absolute -top-11 left-1/2 -translate-x-1/2 pointer-events-none transition-all duration-300 ${
              toast ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-1'
            }`}
          >
            <span className="whitespace-nowrap text-sm font-semibold text-slate-900 bg-white rounded-full px-4 py-1.5 shadow-[0_0_20px_-4px_rgba(255,255,255,0.7)]">
              Copied!
            </span>
            <div className="w-2.5 h-2.5 bg-white rotate-45 mx-auto -mt-1.5" />
          </div>

          {!copied ? (
            <button
              onClick={handleBuild}
              className="inline-flex items-center gap-3 px-10 py-4 rounded-full bg-white text-slate-900 font-bold text-xl shadow-[0_0_40px_-8px_rgba(255,255,255,0.4)] hover:shadow-[0_0_60px_-6px_rgba(255,255,255,0.6)] hover:scale-[1.02] transition-all"
            >
              Build it
              <Clipboard className="h-5 w-5" />
            </button>
          ) : (
            <button
              onClick={handleOpenObserver}
              className="inline-flex items-center gap-3 px-10 py-4 rounded-full bg-emerald-400 text-slate-900 font-bold text-xl shadow-[0_0_40px_-8px_rgba(52,211,153,0.5)] hover:shadow-[0_0_60px_-6px_rgba(52,211,153,0.7)] hover:scale-[1.02] transition-all"
            >
              Paste it into Observer
              <Share className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>
    </section>
  );
};

export default RecipeBuilder;
