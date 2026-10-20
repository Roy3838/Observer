// An Ubuntu desktop for demos that run on Linux (see ubuntu.css). A demo calls
//
//   Ubuntu.mount({ dock: [{ bg, glyph, open }] })
//
// before its windows render. The clock starts at 09:41 on Oct 12 and runs in real time.

(() => {
  const NET = '<svg width="16" height="14" viewBox="0 0 16 14"><path d="M8 13 0 4.2a11.8 11.8 0 0 1 16 0z" fill="#fff"/></svg>';
  const VOL = '<svg width="16" height="14" viewBox="0 0 16 14" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"><path d="M1.5 5h3l4-3.5v11L4.5 9h-3z" fill="#fff" stroke="none"/><path d="M11 4.5a3.5 3.5 0 0 1 0 5"/><path d="M13 2.5a6.5 6.5 0 0 1 0 9"/></svg>';
  const POWER = '<svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="#fff" stroke-width="1.6" stroke-linecap="round"><path d="M7.5 1.2v6"/><path d="M4.3 3.2a5.3 5.3 0 1 0 6.4 0"/></svg>';
  const START = Date.UTC(2026, 9, 12, 9, 41);
  const loadedAt = Date.now();

  window.Ubuntu = {
    mount({ dock = [] } = {}) {
      document.body.classList.add('ubuntu');
      const bar = document.createElement('div');
      bar.className = 'ub-topbar';
      bar.innerHTML = `<span class="ws"><i class="on"></i><i></i></span><span class="clock"></span><span class="sys">${NET}${VOL}${POWER}</span>`;
      document.body.prepend(bar);
      const el = document.createElement('div');
      el.className = 'ub-dock';
      el.innerHTML = dock.map(d => `<span class="icon${d.open ? ' open' : ''}" style="background:${d.bg}">${d.glyph ?? ''}</span>`).join('')
        + `<span class="apps">${'<i></i>'.repeat(9)}</span>`;
      document.body.append(el);
      const tick = () => {
        const d = new Date(START + (Date.now() - loadedAt));
        const date = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
        bar.querySelector('.clock').textContent = `${date}  ${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')}`;
      };
      tick();
      setInterval(tick, 1000);
    },
  };
})();
