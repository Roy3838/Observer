// The player behind every camera demo (camera_*.html): a recorded clip, full window, under a
// security-camera overlay (camera name, a running timestamp, REC). The page loads its timings first,
// media/camera/<name>.js, which sets window.CAMERA:
//
//   { label, night, clock, clips: { <?outcome= value>: { file, hold, event, end, marks } } }
//
//   hold    [from, to] seconds: a quiet stretch that loops while the MCP builds its agent
//   event   when the thing the user asked about happens (the person walks in, the print fails).
//           The demo triggers there, then plays on to `end` and stays on the last frame. A clip with
//           no event (a log of what happens) triggers at `end`.
//   marks   [{ at, name, data }]: ground truth along the way (a decoy, someone leaving)
//
// With an event, the suite's durationMs is honored: the quiet stretch keeps looping until the event
// would land durationMs after the start, so the wait before it is as long as for any other demo.
//
// The suite feeds this tab to Observer as the camera ($CAMERA), not as a screen: see fixtures.ts.

(() => {
  const cam = window.CAMERA;
  const clip = cam.clips[demo.outcome] ?? Object.values(cam.clips)[0];
  demo.truth = { clip: clip.file, event: clip.event ?? null, end: clip.end, marks: clip.marks ?? [] };

  const style = document.createElement('style');
  style.textContent = `
    html, body { margin: 0; height: 100%; background: #000; overflow: hidden; }
    video { width: 100%; height: 100%; object-fit: cover; display: block; ${cam.night ? 'filter: grayscale(1) contrast(1.1);' : ''} }
    .osd { position: fixed; left: 0; right: 0; top: 0; display: flex; justify-content: space-between; padding: 18px 22px;
           font: 600 20px/1 Menlo, "DejaVu Sans Mono", monospace; color: #fff; text-shadow: 0 0 3px #000, 0 0 1px #000; letter-spacing: .5px; }
    .rec { color: #ff3b30; margin-left: 18px; }
    .nosig { position: fixed; inset: 0; display: grid; place-items: center; color: #888; font: 600 24px Menlo, monospace; }`;
  document.head.append(style);
  document.body.innerHTML = `<video id="v" muted playsinline></video>
    <div class="osd"><span>${cam.label}</span><span><span id="ts"></span><span class="rec">● REC</span></span></div>`;

  const v = document.getElementById('v');
  const [h, m, s] = (cam.clock ?? '14:02:11').split(':').map(Number);
  const loadedAt = Date.now();
  const tick = () => {
    const t = new Date(Date.UTC(2026, 9, 12, h, m, s) + (Date.now() - loadedAt));
    document.getElementById('ts').textContent = t.toISOString().slice(0, 19).replace('T', ' ');
  };
  tick();
  setInterval(tick, 1000);

  const [hold0, hold1] = clip.hold;
  const marks = (clip.marks ?? []).slice().sort((a, b) => a.at - b.at);
  let phase = 'hold', next = 0, holdUntil = 0, triggered = false;

  v.addEventListener('error', () => document.body.insertAdjacentHTML('beforeend', `<div class="nosig">NO SIGNAL · ${clip.file}</div>`));
  v.addEventListener('loadedmetadata', () => { v.currentTime = hold0; v.play().catch(() => {}); });
  v.addEventListener('timeupdate', () => {
    if (phase === 'hold' || (phase === 'extra' && Date.now() < holdUntil)) {
      if (v.currentTime >= hold1) v.currentTime = hold0;
      return;
    }
    if (phase === 'extra') { phase = 'run'; v.currentTime = hold1; }
    while (next < marks.length && v.currentTime >= marks[next].at) { const mk = marks[next++]; demo.mark(mk.name, { at: mk.at, ...mk.data }); }
    if (!triggered && clip.event != null && v.currentTime >= clip.event) { triggered = true; demo.trigger(); }
    if (v.currentTime >= clip.end) {
      v.pause();
      phase = 'done';
      if (!triggered) { triggered = true; demo.trigger(); }
    }
  });
  v.src = clip.file;

  demo.onStart = durationMs => {
    // Wait out the rest of durationMs on the quiet stretch, so the event lands when the suite expects.
    const lead = clip.event != null ? clip.event - hold1 : 0;
    holdUntil = Date.now() + Math.max(0, durationMs / 1000 - lead) * 1000 * (clip.event != null ? 1 : 0);
    phase = 'extra';
    v.play().catch(() => {});
  };
})();
