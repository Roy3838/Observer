// A macOS desktop for demos that run in a native app (see macos.css). A demo calls
//
//   Mac.mount({ app: 'Terminal', menus: ['Shell', 'Edit', …], battery: 82, dock: [...] })
//
// before its own windows render, then drives it with Mac.setApp(), Mac.setBattery() and
// Mac.notify(). The clock starts at 9:41 AM on Monday Oct 12 and runs in real time, like the
// other desktop demos (workday.html).

(() => {
  const APPLE = '<svg class="apple" viewBox="0 0 24 24"><path d="M12.152 6.896c-.948 0-2.415-1.078-3.96-1.04-2.04.027-3.91 1.183-4.961 3.014-2.117 3.675-.546 9.103 1.519 12.09 1.013 1.454 2.208 3.09 3.792 3.039 1.52-.065 2.09-.987 3.935-.987 1.831 0 2.35.987 3.96.948 1.637-.026 2.676-1.48 3.676-2.948 1.156-1.688 1.636-3.325 1.662-3.415-.039-.013-3.182-1.221-3.22-4.857-.026-3.04 2.48-4.494 2.597-4.559-1.429-2.09-3.623-2.324-4.39-2.376-2-.156-3.675 1.09-4.61 1.09zM15.53 3.83c.843-1.012 1.4-2.427 1.245-3.83-1.207.052-2.662.805-3.532 1.818-.78.896-1.454 2.338-1.273 3.714 1.338.104 2.715-.688 3.559-1.701"/></svg>';
  const WIFI = '<svg width="17" height="13" viewBox="0 0 17 13" fill="none" stroke="#fff" stroke-width="1.7" stroke-linecap="round"><path d="M1.2 4.6a10.4 10.4 0 0 1 14.6 0"/><path d="M3.8 7.3a6.6 6.6 0 0 1 9.4 0"/><path d="M6.4 9.9a2.9 2.9 0 0 1 4.2 0"/><circle cx="8.5" cy="11.8" r=".6" fill="#fff"/></svg>';
  const SEARCH = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="#fff" stroke-width="1.6" stroke-linecap="round"><circle cx="5.8" cy="5.8" r="4.3"/><path d="M9 9l3.8 3.8"/></svg>';
  const CONTROL = '<svg width="16" height="14" viewBox="0 0 16 14"><rect x=".7" y="1" width="14.6" height="5" rx="2.5" fill="none" stroke="#fff" stroke-width="1.3"/><circle cx="12.5" cy="3.5" r="1.6"/><rect x=".7" y="8" width="14.6" height="5" rx="2.5" fill="none" stroke="#fff" stroke-width="1.3"/><circle cx="3.5" cy="10.5" r="1.6"/></svg>';
  const BOLT = '<svg viewBox="0 0 9 11"><path d="M5.4 0 0 6.3h3.3L2.7 11 9 4.2H5.4z"/></svg>';

  const START = Date.UTC(2026, 9, 12, 9, 41);
  const loadedAt = Date.now();
  const fmtTime = d => `${d.getUTCHours() % 12 || 12}:${String(d.getUTCMinutes()).padStart(2, '0')} ${d.getUTCHours() < 12 ? 'AM' : 'PM'}`;
  const now = () => new Date(START + (Date.now() - loadedAt));

  let bar, battery, notes;

  window.Mac = {
    /** The desktop's clock time, as the menu bar shows it: "9:41 AM". */
    time: () => fmtTime(now()),

    mount({ app = 'Finder', menus = ['File', 'Edit', 'View', 'Go', 'Window', 'Help'], battery: pct = 82, dock = null } = {}) {
      document.body.classList.add('mac');
      bar = document.createElement('div');
      bar.className = 'mac-menubar';
      bar.innerHTML = `${APPLE}<span class="app"></span><span class="menus"></span>
        <div class="status">
          <span class="mac-battery"><span class="pct"></span><span class="body"><span class="fill"></span><span class="bolt">${BOLT}</span></span></span>
          ${WIFI}${SEARCH}${CONTROL}<span class="mac-clock"></span>
        </div>`;
      document.body.prepend(bar);
      battery = bar.querySelector('.mac-battery');
      notes = document.createElement('div');
      notes.className = 'mac-notes';
      document.body.append(notes);
      if (dock) {
        const el = document.createElement('div');
        el.className = 'mac-dock';
        el.innerHTML = dock.map(d => d === '|' ? '<span class="sep"></span>'
          : `<span class="icon${d.open ? ' open' : ''}" style="background:${d.bg}">${d.glyph ?? ''}</span>`).join('');
        document.body.append(el);
      }
      this.setApp(app, menus);
      this.setBattery(pct);
      const tick = () => {
        const d = now();
        bar.querySelector('.mac-clock').textContent = `${d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', timeZone: 'UTC' }).replace(',', '')}  ${fmtTime(d)}`;
      };
      tick();
      setInterval(tick, 1000);
    },

    setApp(name, menus) {
      bar.querySelector('.app').textContent = name;
      if (menus) bar.querySelector('.menus').innerHTML = menus.map(m => `<span>${m}</span>`).join('');
    },

    /** The menu bar battery: percentage shown, red under 11%, a bolt while charging. */
    setBattery(pct, { charging = false } = {}) {
      const p = Math.max(0, Math.min(100, Math.round(pct)));
      battery.querySelector('.pct').textContent = `${p}%`;
      battery.querySelector('.fill').style.width = `${Math.max(p, 4)}%`;
      battery.classList.toggle('low', p <= 10 && !charging);
      battery.classList.toggle('charging', charging);
    },

    /** A notification banner. `icon` is HTML for the app icon; `bg` its background. */
    notify({ app, title, body, icon = '', bg = '#888' }) {
      const el = document.createElement('div');
      el.className = 'mac-note';
      el.innerHTML = `<span class="ico" style="background:${bg}">${icon}</span>
        <div class="txt"><div class="hd"><b></b><span>now</span></div><div class="ttl" style="font-weight:600"></div><div class="bd"></div></div>`;
      el.querySelector('.hd b').textContent = app;
      el.querySelector('.ttl').textContent = title;
      el.querySelector('.bd').textContent = body;
      notes.prepend(el);
      return el;
    },
  };
})();
