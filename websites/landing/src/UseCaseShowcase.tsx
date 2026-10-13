// "What do you need monitored?": a prompt bar plus a live stage.
//
// Everything shown here is generated from the MCP e2e suite (app/e2e, via
// scripts/sync-showcase.mjs): the prompts are the ones the suite tests, and the stage runs the
// same demo pages in an iframe. The demos are same-origin, so we hook `__e2eReport` on the
// iframe to learn when each one ends, exactly like the suite does.

import { useEffect, useMemo, useRef, useState } from 'react';
import { Phone, MessageCircle, MessageSquare, Mail, Hash, Send, ExternalLink, type LucideIcon } from 'lucide-react';
import data from './generated/showcase.json';

type Outcome = 'success' | 'fail';
// notifyOn: the demo's `?outcome=` values (success, fail, or a demo's own like `late`).
interface Prompt { prompt: string; kind: 'alert' | 'pattern'; query: string; notifyOn: string[] }
interface UseCase { channel?: { id: string; label: string }; id: string; file: string; title: string; category: string; pitch: string; prompts: Prompt[] }

const CATEGORIES = data.categories as { id: string; label: string }[];
const ITEMS = data.items as UseCase[];

// Demos are laid out as full browser windows; we render them at this size and scale to fit.
const STAGE_W = 1280;
const STAGE_H = 800;
const RUN_MS = 9000;
// The "open in new tab" link runs the demo on its own at the suite's shortest timing, for live demos.
const LIVE_MS = 120000;
const TYPE_STEP = 2;

type Phase = 'idle' | 'watching' | 'done';

const CHANNEL_ICONS: Record<string, LucideIcon> = { call: Phone, whatsapp: MessageCircle, sms: MessageSquare, email: Mail, discord: Hash, telegram: Send };

const prettyMark = (name: string, d: unknown) => {
  const detail = d && typeof d === 'object'
    ? Object.values(d as Record<string, unknown>).find(v => typeof v === 'string' || typeof v === 'number')
    : undefined;
  return `Saved to memory · ${name.replace(/_/g, ' ')}${detail !== undefined ? ` · ${detail}` : ''}`;
};

const UseCaseShowcase = () => {
  const [category, setCategory] = useState(CATEGORIES[0].id);
  const [itemId, setItemId] = useState(ITEMS.find(i => i.category === CATEGORIES[0].id)!.id);
  const [phase, setPhase] = useState<Phase>('idle');
  const [runId, setRunId] = useState(0);
  const [toast, setToast] = useState<string | null>(null);
  const [typed, setTyped] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [scale, setScale] = useState(0.5);

  const item = ITEMS.find(i => i.id === itemId)!;
  // The first prompt is the plain "tell me when it's done" one; the rest are variants.
  const prompt = item.prompts[0];
  const outcome: Outcome = 'success';
  const isAlert = prompt.kind === 'alert';
  const typingDone = typed.length >= prompt.prompt.length;

  const frameRef = useRef<HTMLIFrameElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);

  // Scale the fixed-size demo window to the stage's width.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setScale(e.contentRect.width / STAGE_W));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const src = useMemo(() => {
    const q = new URLSearchParams(prompt.query);
    if (!q.has('outcome')) q.set('outcome', outcome);
    return `/demos/${item.file}?${q}`;
  }, [item.file, prompt.query]);
  const liveSrc = `${src}&autostart=${LIVE_MS}`;

  // Reset whenever the use case, prompt, outcome or run changes, and type the prompt into the bar.
  useEffect(() => {
    setPhase('idle'); setToast(null); setTyped(''); setLoaded(false);
    let n = 0;
    const id = window.setInterval(() => {
      n += TYPE_STEP;
      setTyped(prompt.prompt.slice(0, n));
      if (n >= prompt.prompt.length) clearInterval(id);
    }, 16);
    return () => clearInterval(id);
  }, [itemId, runId, prompt.prompt]);

  // Once the prompt is typed and the demo is loaded, the agent starts watching it.
  useEffect(() => {
    if (phase !== 'idle' || !typingDone || !loaded) return;
    const win = frameRef.current?.contentWindow as (Window & { demo?: any; __e2eReport?: (e: any) => void }) | null;
    if (!win?.demo) return;
    win.__e2eReport = e => {
      if (e.type === 'mark') setToast(prettyMark(e.name, e.data));
      if (e.type === 'trigger') {
        setPhase('done');
        if (!isAlert) return;
        setToast(`${item.title} finished.`);
      }
    };
    setPhase('watching');
    win.demo.start(RUN_MS);
  }, [phase, typingDone, loaded, isAlert, prompt, item.title]);

  const pick = (id: string) => {
    if (id === itemId) { setRunId(r => r + 1); return; } // clicking the active one replays it
    setItemId(id);
  };
  const pickCategory = (id: string) => {
    setCategory(id);
    pick(ITEMS.find(i => i.category === id)!.id);
  };

  const items = ITEMS.filter(i => i.category === category);

  return (
    <section id="use-cases" className="py-24 md:py-32 bg-[#0D1321]">
      <div className="container mx-auto px-6 max-w-6xl">
        <h2 className="font-golos font-bold text-4xl md:text-6xl text-white tracking-tight text-center mb-12">
          What do you need monitored?
        </h2>

        <div role="tablist" className="flex flex-wrap justify-center gap-x-8 gap-y-2 mb-10">
          {CATEGORIES.map(c => (
            <button
              key={c.id}
              role="tab"
              aria-selected={c.id === category}
              onClick={() => pickCategory(c.id)}
              className={`pb-2 border-b transition ${c.id === category ? 'text-white border-white' : 'text-gray-500 border-transparent hover:text-gray-300'}`}
            >
              {c.label}
            </button>
          ))}
        </div>

        <div className="max-w-3xl mx-auto text-center min-h-[4.5rem] mb-8">
          <p className="text-xl md:text-2xl text-gray-200 leading-snug">&ldquo;{typed}&rdquo;</p>
        </div>

        <div className="grid lg:grid-cols-[19rem_1fr] gap-8">
          <ul className="grid grid-cols-2 lg:grid-cols-1 gap-1 content-start">
            {items.map(i => (
              <li key={i.id}>
                <button
                  onClick={() => pick(i.id)}
                  className={`w-full text-left rounded-xl px-3.5 py-3 transition ${i.id === itemId ? 'bg-white/[0.07]' : 'hover:bg-white/[0.03]'}`}
                >
                  <span className={`block text-[15px] ${i.id === itemId ? 'text-white' : 'text-gray-300'}`}>{i.title}</span>
                </button>
              </li>
            ))}
          </ul>

        <div className="self-start min-w-0 rounded-2xl border border-white/10 overflow-hidden shadow-[0_24px_80px_-24px_rgba(0,0,0,0.8)]">
          <div ref={stageRef} className="relative w-full overflow-hidden bg-white" style={{ height: STAGE_H * scale }}>
            <iframe
              key={`${src}-${runId}`}
              ref={frameRef}
              src={src}
              title={`${item.title} demo`}
              allow="autoplay"
              tabIndex={-1}
              onLoad={() => setLoaded(true)}
              style={{ width: STAGE_W, height: STAGE_H, transform: `scale(${scale})`, transformOrigin: 'top left', border: 0 }}
            />
            {/* Blocks interaction with the demo: it's a picture of a job, not a playground. */}
            <div className="absolute inset-0" />
            <a
              href={liveSrc}
              target="_blank"
              rel="noopener"
              title="Open this demo in a new tab. It ends after 2 minutes."
              className="absolute bottom-3 right-3 flex items-center gap-1.5 rounded-full bg-[#0D1321]/80 backdrop-blur border border-white/10 px-3 py-1.5 text-xs text-gray-300 hover:text-white transition"
            >
              Open in new tab <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <div
              className={`absolute top-3 right-3 w-[min(22rem,80%)] rounded-2xl bg-[#0D1321]/95 backdrop-blur border border-white/10 shadow-2xl p-3.5 flex gap-3 items-center transition-all duration-500 ${
                toast ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-3 pointer-events-none'
              }`}
            >
              {(() => { const Icon = CHANNEL_ICONS[item.channel?.id ?? ''];
                return Icon ? <Icon className="w-5 h-5 text-white shrink-0" /> : <img src="/eye-logo-white.svg" alt="" className="w-6 h-6" />; })()}
              <div className="min-w-0">
                {item.channel && <div className="text-xs text-gray-400">{item.channel.label} · Observer</div>}
                <div className="text-sm text-white">{toast}</div>
              </div>
            </div>
          </div>
        </div>
        </div>
      </div>
    </section>
  );
};

export default UseCaseShowcase;
