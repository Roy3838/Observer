// "What do you need monitored?": two big wheels, When … then …, spinning through everything
// Observer is for, with each pair's live demo underneath.
//
// Everything here comes from the ground truth (app/src/utils/groundTruth.ts, via
// scripts/sync-showcase.mjs): the pairs are the ones the MCP e2e suite tests, and the stage runs the
// same demo pages in an iframe. The demos are same-origin, so we hook `__e2eReport` on the iframe to
// learn when each one reaches its moment, exactly like the suite does, and fire the Then there.

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Phone, MessageCircle, MessageSquare, Mail, Hash, Send, NotebookPen, Clapperboard, ChevronLeft, ChevronRight, ExternalLink, type LucideIcon } from 'lucide-react';
import data from './generated/showcase.json';

interface Pair { id: string; file: string; query: string; title: string; when: string; then: string; channel: { id: string; label: string } | null; alert: string }

const PAIRS = data.items as Pair[];

// Demos are laid out as full browser windows; we render them at this size and scale to fit.
const STAGE_W = 1280;
const STAGE_H = 800;
// How long each demo runs before its moment, how long the Then stays up, and the longest we wait
// for a demo that plays its own media (a meeting, a video) before showing the Then anyway.
const RUN_MS = 8000;
const SHOW_MS = 3800;
const MAX_WAIT_MS = RUN_MS + 6000;
// The "open in new tab" link runs the demo on its own at the suite's shortest timing.
const LIVE_MS = 120000;

const ICONS: Record<string, LucideIcon> = { call: Phone, whatsapp: MessageCircle, sms: MessageSquare, email: Mail, discord: Hash, telegram: Send };
const iconFor = (p: Pair): LucideIcon => p.channel ? ICONS[p.channel.id] : /clip/.test(p.then) ? Clapperboard : NotebookPen;

const wrap = (i: number) => ((i % PAIRS.length) + PAIRS.length) % PAIRS.length;

/** A reel of words, three rows tall: the current one in the middle, on the sentence's line, with
 *  its neighbours blurred above and below. Items enter and leave through the faded outer edges, so
 *  nothing gets clipped in the space between When and then. */
const Wheel = ({ items, index, delay = 0, className = '' }: { items: string[]; index: number; delay?: number; className?: string }) => {
  const n = items.length;
  // How much each item has to shrink to fit the reel's width (on a phone, the long ones do).
  const ref = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState<number[]>(() => items.map(() => 1));
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setFit([...el.children].map(c => Math.min(1, el.clientWidth / Math.max(1, (c as HTMLElement).scrollWidth))));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [items]);
  return (
    <div
      ref={ref}
      className="relative h-[3.45em] overflow-hidden [mask-image:linear-gradient(transparent,black_30%,black_70%,transparent)]"
      aria-live="polite"
    >
      {items.map((text, i) => {
        // Shortest way round the reel from the current item.
        let k = i - index;
        if (k > n / 2) k -= n;
        if (k < -n / 2) k += n;
        return (
          <span
            key={i}
            aria-hidden={k !== 0}
            className={`absolute left-0 top-[1.15em] whitespace-nowrap leading-[1.15] will-change-transform ${className}`}
            style={{
              transform: `translateY(${k * 1.15}em) scale(${(k === 0 ? 1 : 0.9) * fit[i]})`,
              transformOrigin: 'left center',
              opacity: k === 0 ? 1 : Math.abs(k) === 1 ? 0.35 : 0,
              filter: k === 0 ? 'none' : 'blur(1.5px)',
              transition: `transform 850ms cubic-bezier(.2,.8,.2,1) ${delay}ms, opacity 600ms ease ${delay}ms, filter 600ms ease ${delay}ms`,
            }}
          >
            {text}
          </span>
        );
      })}
    </div>
  );
};

type Phase = 'running' | 'fired';

const UseCaseShowcase = () => {
  const [index, setIndex] = useState(0);
  const [runId, setRunId] = useState(0);
  const [phase, setPhase] = useState<Phase>('running');
  const [loaded, setLoaded] = useState(false);
  const [hover, setHover] = useState(false);
  const [visible, setVisible] = useState(false);
  const [scale, setScale] = useState(0.5);
  const [firedAt, setFiredAt] = useState(0);

  const pair = PAIRS[index];
  const Icon = iconFor(pair);
  const whens = useMemo(() => PAIRS.map(p => p.when), []);
  const thens = useMemo(() => PAIRS.map(p => p.then), []);

  const sectionRef = useRef<HTMLElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLIFrameElement>(null);

  // Scale the fixed-size demo window to the stage's width.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setScale(e.contentRect.width / STAGE_W));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Only run demos while the section is on screen.
  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0.25 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const go = (i: number) => { setIndex(wrap(i)); setRunId(r => r + 1); };

  useEffect(() => { setPhase('running'); setLoaded(false); }, [index, runId]);

  const src = useMemo(() => {
    const q = new URLSearchParams(pair.query);
    q.set('muted', '1'); // the meeting and the video are here to be seen, not heard
    return `/demos/${pair.file}?${q}`;
  }, [pair]);

  // Once the demo is loaded and in view, start it; its moment fires the Then.
  useEffect(() => {
    if (!loaded || !visible || phase !== 'running') return;
    const win = frameRef.current?.contentWindow as (Window & { demo?: any; __e2eReport?: (e: any) => void }) | null;
    if (!win?.demo) return;
    const fire = () => { setPhase('fired'); setFiredAt(Date.now()); };
    win.__e2eReport = e => { if (e.type === 'trigger') fire(); };
    if (!win.demo.startedAt) win.demo.start(RUN_MS);
    const fallback = window.setTimeout(fire, MAX_WAIT_MS);
    return () => window.clearTimeout(fallback);
  }, [loaded, visible, phase]);

  // After the Then has been up a while, spin to the next pair (not while someone's hovering).
  useEffect(() => {
    if (phase !== 'fired' || hover || !visible) return;
    const t = window.setTimeout(() => go(index + 1), Math.max(600, SHOW_MS - (Date.now() - firedAt)));
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, hover, visible, firedAt, index]);

  const liveSrc = `${src}&autostart=${LIVE_MS}`;

  return (
    <section
      id="use-cases"
      ref={sectionRef}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="relative py-24 md:py-36 bg-[#0D1321] overflow-hidden"
    >
      {/* A soft light behind the sentence. */}
      <div className="pointer-events-none absolute left-1/2 top-24 -translate-x-1/2 w-[70rem] h-[34rem] rounded-full bg-[radial-gradient(closest-side,rgba(129,140,248,0.13),transparent)]" />

      <div className="relative container mx-auto px-6 max-w-6xl">
        <h2 className="font-golos font-bold text-4xl md:text-6xl text-white tracking-tight text-center">
          What do you need monitored?
        </h2>

        {/* When … then … */}
        <div className="mt-12 md:mt-20 mx-auto max-w-5xl font-golos font-semibold tracking-tight text-[clamp(1.05rem,3.6vw,3rem)] leading-none">
          <div className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-[0.4em] gap-y-[0.1em]">
            <span className="self-center leading-[1.15] text-gray-500 text-right">When</span>
            <div className="min-w-0 text-white"><Wheel items={whens} index={index} /></div>
            <span className="self-center leading-[1.15] text-gray-500 text-right">then</span>
            <div className="min-w-0">
              <Wheel items={thens} index={index} delay={140} className="bg-gradient-to-r from-indigo-200 via-white to-sky-200 bg-clip-text text-transparent" />
            </div>
          </div>
        </div>

        {/* Controls: back, where we are, forward, and how long until the next one. */}
        <div className="mt-8 md:mt-12 flex items-center justify-center gap-5 text-gray-500">
          <button onClick={() => go(index - 1)} aria-label="Previous" className="p-2 rounded-full hover:text-white hover:bg-white/5 transition"><ChevronLeft className="w-5 h-5" /></button>
          <span className="font-mono text-xs tracking-widest tabular-nums">{String(index + 1).padStart(2, '0')} / {String(PAIRS.length).padStart(2, '0')}</span>
          <button onClick={() => go(index + 1)} aria-label="Next" className="p-2 rounded-full hover:text-white hover:bg-white/5 transition"><ChevronRight className="w-5 h-5" /></button>
        </div>

        {/* The demo, live. */}
        <div className="mt-8 mx-auto max-w-5xl">
          <div className="rounded-[22px] p-[1px] bg-gradient-to-b from-white/20 via-white/5 to-white/5 shadow-[0_40px_120px_-30px_rgba(99,102,241,0.35)]">
            <div ref={stageRef} className="relative w-full overflow-hidden rounded-[21px] bg-[#0b0f1a]" style={{ height: STAGE_H * scale }}>
              <iframe
                key={`${src}-${runId}`}
                ref={frameRef}
                src={src}
                title={`${pair.title} demo`}
                allow="autoplay"
                tabIndex={-1}
                onLoad={() => setLoaded(true)}
                style={{ width: STAGE_W, height: STAGE_H, transform: `scale(${scale})`, transformOrigin: 'top left', border: 0 }}
                className={`transition-opacity duration-500 ${loaded ? 'opacity-100' : 'opacity-0'}`}
              />
              {/* It's a picture of a job, not a playground. */}
              <div className="absolute inset-0" />

              {/* The Then, firing. */}
              <div
                className={`absolute top-4 right-4 w-[min(23rem,calc(100%-2rem))] rounded-2xl border border-white/10 bg-[#0D1321]/85 backdrop-blur-xl shadow-2xl p-3.5 flex gap-3 items-center transition-all duration-500 ${
                  phase === 'fired' ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4 pointer-events-none'
                }`}
              >
                <span className="w-9 h-9 shrink-0 rounded-full bg-white/10 grid place-items-center"><Icon className="w-[18px] h-[18px] text-white" /></span>
                <div className="min-w-0">
                  <div className="text-[11px] text-gray-400">{pair.channel ? `${pair.channel.label} · Observer` : 'Observer'} · now</div>
                  <div className="text-sm text-white leading-snug">{pair.alert}</div>
                </div>
              </div>

              <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-full bg-[#0D1321]/80 backdrop-blur border border-white/10 px-3 py-1.5 text-xs text-gray-300">
                <span className={`w-1.5 h-1.5 rounded-full ${phase === 'fired' ? 'bg-emerald-400' : 'bg-red-400 animate-pulse'}`} />
                {pair.title}
              </div>
              <a
                href={liveSrc}
                target="_blank"
                rel="noopener"
                title="Open this demo in a new tab. It ends after 2 minutes."
                className="absolute bottom-3 right-3 flex items-center gap-1.5 rounded-full bg-[#0D1321]/80 backdrop-blur border border-white/10 px-3 py-1.5 text-xs text-gray-300 hover:text-white transition"
              >
                <span className="hidden sm:inline">Open in new tab</span> <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default UseCaseShowcase;
