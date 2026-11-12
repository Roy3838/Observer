// "What do you need monitored?": two big wheels, When … then …, spinning through everything
// Observer is for, with each pair's live demo underneath.
//
// Everything here comes from the ground truth (app/src/utils/groundTruth.ts, via
// scripts/sync-showcase.mjs): the pairs are the ones the MCP e2e suite tests, and the stage runs the
// same demo pages in an iframe. The demos are same-origin, so we hook `__e2eReport` on the iframe to
// learn when each one reaches its moment, exactly like the suite does, and fire the Then there.

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Phone, MessageCircle, MessageSquare, Mail, Hash, Send, NotebookPen, Clapperboard, ChevronLeft, ChevronRight, ExternalLink, Check, type LucideIcon } from 'lucide-react';
import data from './generated/showcase.json';

interface Pair { id: string; file: string; query: string; when: string; then: string; channel: { id: string; label: string } | null }

/** What the Then says when it fires: the moment for an alert ("My battery is low"), what got done
 *  for a log ("Log all the topics"). The demo itself shows the rest. */
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const fired = (p: Pair) => (p.channel ? `${p.channel.id === "call" ? "Call" : p.channel.label} from Observer` : cap(p.then));

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

const EASE_OUT = 'cubic-bezier(.16,1,.3,1)';
const reducedMotion = () => typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** One line of text that swaps phrase by phrase. Each word sits in its own clipped slot and rises
 *  into place a beat after the one before it; the outgoing phrase leaves the same way, upward
 *  (or downward when going back), so there is only ever one crisp phrase on screen. */
const Wheel = ({ items, index, delay = 0, className = '' }: { items: string[]; index: number; delay?: number; className?: string }) => {
  const n = items.length;
  // How much each item has to shrink to fit the line's width (on a phone, the long ones do).
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
  const still = reducedMotion();
  return (
    <div ref={ref} className="relative h-[1.3em]" aria-live="polite">
      {items.map((text, i) => {
        // Which side of the current phrase this one is on: it waits below, or has left above.
        let k = i - index;
        if (k > n / 2) k -= n;
        if (k < -n / 2) k += n;
        const side = Math.sign(k);
        const words = text.split(' ');
        return (
          <span
            key={i}
            aria-hidden={k !== 0}
            className={`absolute left-0 top-0 whitespace-nowrap leading-[1.3] ${className}`}
            style={{ transform: `scale(${fit[i]})`, transformOrigin: 'left center', pointerEvents: k === 0 ? 'auto' : 'none' }}
          >
            {words.map((w, j) => {
              // Entering words stagger left to right; leaving words go together, a touch faster.
              const d = k === 0 ? delay + 120 + j * 55 : delay + j * 25;
              return (
                <span key={j} className="inline-block overflow-hidden align-top pb-[0.08em] -mb-[0.08em]">
                  <span
                    className="inline-block will-change-transform"
                    style={{
                      transform: `translateY(${side * 105}%)`,
                      opacity: k === 0 ? 1 : 0,
                      transition: still ? 'none' : `transform ${k === 0 ? 900 : 550}ms ${EASE_OUT} ${d}ms, opacity ${k === 0 ? 500 : 350}ms ease ${d}ms`,
                    }}
                  >
                    {w}
                  </span>
                  {j < words.length - 1 && ' '}
                </span>
              );
            })}
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

  // The heading lives at the bottom of the hero, which is taller than its content; pull this
  // section up so the wheels sit right under the heading instead of after the hero's empty tail.
  const [pullUp, setPullUp] = useState(0);
  useLayoutEffect(() => {
    const measure = () => {
      const h = document.getElementById('use-cases-heading');
      const sec = sectionRef.current;
      if (!h || !sec) return;
      const vh = window.innerHeight;
      const y = window.scrollY;
      const eye = document.getElementById('eye-container')?.offsetHeight ?? 80;
      // The hero sticks to the screen while the eye shrinks (280px -> 80px), then rests centred on
      // 100vh. Work out where the heading ends up once it rests, in page coordinates.
      const fixed = getComputedStyle(h.offsetParent ?? h).position === 'fixed'
        || getComputedStyle(h.closest('.pointer-events-none') ?? h).position === 'fixed';
      const r = h.getBoundingClientRect();
      const headingBottom = fixed ? vh + (r.bottom - vh / 2) - (eye - 80) / 2 : r.bottom + y;
      const prev = parseFloat(sec.style.marginTop) || 0;
      const sectionTop = sec.getBoundingClientRect().top + y - prev;
      setPullUp(Math.max(0, sectionTop - headingBottom - 48));
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
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
      style={{ marginTop: -pullUp }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="relative pt-4 pb-24 md:pb-36 bg-[#0D1321] overflow-hidden"
    >
      <div className="relative container mx-auto px-6 max-w-6xl">
        {/* The heading lives in the hero (ObserverLanding) so it shows as soon as scrolling starts. */}
        {/* When … then … */}
        <div className="mx-auto max-w-5xl font-golos font-semibold tracking-[-0.035em] text-[clamp(1.15rem,4vw,3.4rem)] leading-none">
          <div className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-[0.35em] gap-y-[0.15em]">
            <span className="leading-[1.3] text-[#5C6975] text-right">When</span>
            <div className="min-w-0"><Wheel items={whens} index={index} className="text-white" /></div>
            <span className="leading-[1.3] text-[#5C6975] text-right">then</span>
            <div className="min-w-0">
              <Wheel items={thens} index={index} delay={180} className="text-[#78b8cf]" />
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
          <div className="rounded-[22px] p-[1px] bg-white/10">
            <div ref={stageRef} className="relative w-full overflow-hidden rounded-[21px] bg-[#0b0f1a]" style={{ height: STAGE_H * scale }}>
              <iframe
                key={`${src}-${runId}`}
                ref={frameRef}
                src={src}
                title={`Demo: when ${pair.when}`}
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
                className="absolute top-3 right-3 sm:top-4 sm:right-4 w-[min(22rem,calc(100%-1.5rem))] rounded-[18px] bg-white text-[#0D1321] p-2.5 pr-4 flex gap-3 items-center font-golos shadow-[0_0_0_1px_rgba(13,19,33,0.06),0_2px_4px_rgba(13,19,33,0.06),0_24px_48px_-12px_rgba(13,19,33,0.45)]"
                style={{
                  opacity: phase === 'fired' ? 1 : 0,
                  transform: phase === 'fired' ? 'none' : 'translateY(-14px) scale(0.94)',
                  transformOrigin: 'top center',
                  transition: phase === 'fired'
                    ? 'transform 650ms cubic-bezier(.34,1.45,.64,1), opacity 220ms ease'
                    : 'transform 300ms ease, opacity 200ms ease',
                  pointerEvents: phase === 'fired' ? 'auto' : 'none',
                }}
              >
                <span className="w-10 h-10 shrink-0 rounded-[12px] bg-[#0D1321] grid place-items-center">
                  <Icon className="w-5 h-5 text-white" strokeWidth={2.25} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2 text-[11px] leading-4 text-slate-500">
                    <span className="font-semibold uppercase tracking-[0.08em]">Observer</span>
                    <span>now</span>
                  </div>
                  <div className="text-[15px] font-semibold leading-5 tracking-[-0.01em] truncate">{fired(pair)}</div>
                </div>
              </div>

              <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-full bg-[#0D1321] pl-2.5 pr-3.5 py-1.5 text-xs font-medium font-golos text-white shadow-[0_8px_20px_-6px_rgba(0,0,0,0.5)]">
                {phase === 'fired' ? (
                  <span className="w-4 h-4 rounded-full bg-emerald-400 grid place-items-center"><Check className="w-2.5 h-2.5 text-[#0D1321]" strokeWidth={3.5} /></span>
                ) : (
                  <span className="relative w-4 h-4 grid place-items-center">
                    <span className="absolute inset-[3px] rounded-full bg-red-500 animate-ping opacity-60" />
                    <span className="relative w-2 h-2 rounded-full bg-red-500" />
                  </span>
                )}
                {phase === 'fired' ? 'Done' : 'Watching'}
              </div>
              <a
                href={liveSrc}
                target="_blank"
                rel="noopener"
                title="Open this demo in a new tab. It ends after 2 minutes."
                className="absolute bottom-3 right-3 flex items-center gap-1.5 rounded-full bg-[#0D1321] px-3.5 py-1.5 text-xs font-medium font-golos text-white/80 hover:text-white shadow-[0_8px_20px_-6px_rgba(0,0,0,0.5)] transition-colors"
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
