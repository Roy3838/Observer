// reload.js: a page that refreshes itself, for the "when X is available" demos (DMV, restock,
// passport status, eBay, a site going down). A real location.reload() would throw away the demo's
// state and its run, so the refresh is faked in place: the page blanks, a loading bar runs across
// the top, and it comes back re-rendered with a new "Last checked" time. That's what an agent sees
// when, in real use, it click()s the browser's refresh button.
//
//   const page = selfReload({ everyMs: 15000, render });   // render() draws the current state
//   page.now(() => demo.trigger());                        // refresh right away, then call back
//
// It keeps refreshing after the end too, as the real page would.
//
// Elements with [data-checked] show the time of the last refresh.

window.selfReload = ({ everyMs = 15000, render }) => {
  const style = document.createElement('style');
  style.textContent = `
    #reload-veil { position: fixed; inset: 0; background: #fff; z-index: 2147483000; opacity: 0;
      pointer-events: none; transition: opacity .08s; }
    #reload-veil.on { opacity: 1; }
    #reload-bar { position: fixed; top: 0; left: 0; height: 3px; width: 0; background: #1a73e8;
      z-index: 2147483001; }
    #reload-bar.on { width: 85%; transition: width .45s ease-out; }`;
  document.head.appendChild(style);
  const veil = Object.assign(document.createElement('div'), { id: 'reload-veil' });
  const bar = Object.assign(document.createElement('div'), { id: 'reload-bar' });
  document.body.append(veil, bar);

  const stamp = () => {
    const t = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', second: '2-digit' });
    document.querySelectorAll('[data-checked]').forEach(el => { el.textContent = t; });
  };

  let timer = null;
  const reload = then => {
    clearTimeout(timer);
    bar.className = 'on'; veil.className = 'on';
    setTimeout(() => {
      render(); stamp();
      veil.className = ''; bar.className = '';
      schedule();
      if (then) then();
    }, 550);
  };
  // A little jitter, like someone (or something) hitting refresh.
  const schedule = () => { timer = setTimeout(() => reload(), everyMs * (0.8 + Math.random() * 0.4)); };

  render(); stamp(); schedule();
  return { now: then => reload(then) };
};
