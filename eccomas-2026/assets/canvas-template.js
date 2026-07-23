/* =====================================================================
 * canvas-template.js  —  a tiny framework for visualizing step-based
 * algorithms with three.js.  Loads as a classic <script> so it works
 * from file:// with no server.
 *
 * Your algorithm object implements this contract:
 *   init(ctx)    build the 3D objects ONCE. ctx = {THREE, scene, group, camera, controls, renderer}
 *   substep()    OPTIONAL. advance one sub-step. return true when THIS full step is complete
 *   step()       advance one full step. (auto-built from substep() if you defined it)
 *   isDone()     return true once the OVERALL algorithm is finished (converged / terminated)
 *   draw()       sync the 3D objects to current state. called once per frame, on change
 *   reset()      restore the initial state (reuse meshes — just reposition)
 *   clear()      drop accumulated traces, so draw() shows only the current state
 *   snapshot()   OPTIONAL (needed for backward stepping). return a copy of current state
 *   restore(s)   OPTIONAL. restore a state returned by snapshot()
 *   + any getters/setters your logger & menu callbacks use.
 *
 * Usage (in a module script, after importing three):
 *   ThreeTemplate.useThree({ THREE, OrbitControls });
 *   const viz = new ThreeTemplate.Visualizer(algo, { background: 0x111111 });
 *   viz.addPlayback({ substeps: true });        // {autohideLevel: 1} (or autohide:true)
 *   const log = viz.addLogger();                // addLogger(level)
 *   log.addStatus('iter', () => algo.getIter(), '%4d');
 *   const tab = viz.addMenu().addTab('Parameters');   // addMenu(level)
 *
 *   autohide level:  0 = always visible,  1 = show while pointer is on the
 *   canvas/page,  2 = show only when hovering that element.  true -> level 1.
 *   tab.addSlider('Step', (x) => algo.setStep(x), { min:0, max:1, value:0.25, step:0.01 });
 *   viz.start();
 * ===================================================================== */
(function(global) {
  'use strict';

  let THREE = null, OrbitControls = null;
  function useThree(deps) {
    THREE = deps.THREE;
    OrbitControls = deps.OrbitControls || null;
    global.THREE = THREE;                 // so algorithm files can use `THREE` directly
  }

  let _uid = 0;
  const uid = () => 'tt' + (++_uid);

  /* -------- C-style number formatter: %4d, %4.2f, %12.3e, %s ... -------- */
  function cFormat(spec, value) {
    const m = /^%([-+0 ]*)(\d+)?(?:\.(\d+))?([diouxXeEfgGs%])$/.exec(spec);
    if (!m) return String(value);
    let [, flags, width, prec, conv] = m;
    if (conv === '%') return '%';
    width = width ? +width : 0;
    prec = prec !== undefined ? +prec : undefined;
    const left = flags.includes('-');
    const zero = flags.includes('0') && !left;
    const plus = flags.includes('+');
    const space = flags.includes(' ');
    let num = Number(value), neg = num < 0, s;
    switch (conv) {
      case 'd': case 'i': case 'u': s = Math.round(Math.abs(num)).toString(); break;
      case 'x': s = Math.round(Math.abs(num)).toString(16); break;
      case 'X': s = Math.round(Math.abs(num)).toString(16).toUpperCase(); break;
      case 'o': s = Math.round(Math.abs(num)).toString(8); break;
      case 'f': s = Math.abs(num).toFixed(prec === undefined ? 6 : prec); break;
      case 'e': case 'E':
        s = Math.abs(num).toExponential(prec === undefined ? 6 : prec).replace(/e([+-])(\d)$/, 'e$10$2');
        if (conv === 'E') s = s.toUpperCase();
        break;
      case 'g': case 'G': {
        const p = prec === undefined ? 6 : (prec || 1);
        s = Number(Math.abs(num).toPrecision(p)).toString();
        if (conv === 'G') s = s.toUpperCase();
        break;
      }
      default: s = String(value); neg = false; break;
    }
    const sign = neg ? '-' : (plus ? '+' : (space ? ' ' : ''));
    if (zero && conv !== 's') {
      const pad = width - sign.length - s.length;
      if (pad > 0) s = '0'.repeat(pad) + s;
      return sign + s;
    }
    s = sign + s;
    if (s.length < width) { const pad = ' '.repeat(width - s.length); s = left ? s + pad : pad + s; }
    return s;
  }

  /* -------- inline SVG icons -------- */
  const _svg = body =>
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
  const ICONS = {
    reset: _svg('<path d="M21.5 2v6h-6"/><path d="M21.34 15.57a10 10 0 1 1-.57-8.38"/>'),
    erase: _svg('<path d="m7 21-4.3-4.3a1.7 1.7 0 0 1 0-2.4l9.3-9.3a1.7 1.7 0 0 1 2.4 0l5.6 5.6a1.7 1.7 0 0 1 0 2.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/>'),
    sub: _svg('<path d="M4 13a8 6 0 0 1 15-2.5"/><polyline points="20 4 20 9 15 9"/><circle cx="12" cy="19" r="1.7" fill="currentColor" stroke="none"/>'),
    full: _svg('<polygon points="6 4 15 12 6 20" fill="currentColor" stroke="none"/><line x1="18" y1="5" x2="18" y2="19"/>'),
    back: _svg('<polygon points="18 4 9 12 18 20" fill="currentColor" stroke="none"/><line x1="6" y1="5" x2="6" y2="19"/>'),
    play: _svg('<polygon points="7 4 20 12 7 20" fill="currentColor" stroke="none"/>'),
    pause: _svg('<rect x="6" y="5" width="4" height="14" fill="currentColor" stroke="none"/><rect x="14" y="5" width="4" height="14" fill="currentColor" stroke="none"/>'),
  };

  /* -------- DOM helper -------- */
  function el(tag, attrs, children) {
    const e = document.createElement(tag);
    attrs = attrs || {};
    for (const k in attrs) {
      const v = attrs[k];
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k === 'text') e.textContent = v;
      else e.setAttribute(k, v);
    }
    if (children != null) (Array.isArray(children) ? children : [children]).forEach(c => {
      if (c != null) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return e;
  }

  /* -------- CSS (injected once) -------- */
  const CSS = `
  html,body{margin:0;padding:0;height:100%;}
  body{overflow:hidden;background:#111;}
  .tt-canvas{position:fixed;inset:0;width:100%;height:100%;display:block;background:#111;touch-action:none;}
  .tt-controls{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);display:flex;gap:4px;background:rgba(20,20,22,.92);border:1px solid #333;padding:5px;border-radius:7px;z-index:20;box-shadow:0 2px 12px rgba(0,0,0,.5);}
  .tt-controls button{background:#2b2b2b;color:#ddd;border:1px solid #444;padding:6px 11px;border-radius:4px;cursor:pointer;min-width:36px;font-family:inherit;}
  .tt-controls button:hover:not(:disabled){background:#3a3a3a;}
  .tt-controls button:disabled{opacity:.4;cursor:default;}
  .tt-controls button.active{background:#0056b3;border-color:#4a90e2;color:#fff;}
  .tt-controls button svg{display:block;width:17px;height:17px;}
  .tt-sep{width:1px;background:#444;margin:2px 3px;}
  .tt-status{position:fixed;bottom:12px;right:12px;background:rgba(20,20,22,.92);border:1px solid #333;border-radius:7px;padding:8px 10px;z-index:20;min-width:170px;box-shadow:0 2px 12px rgba(0,0,0,.5);color:#ddd;font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;}
  .tt-status-title{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#999;margin-bottom:5px;}
  .tt-status table{border-collapse:collapse;width:100%;}
  .tt-status td{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;padding:1px 0;}
  .tt-status td.k{color:#999;padding-right:14px;}
  .tt-status td.v{text-align:right;white-space:pre;color:#ddd;}
  .tt-hamburger{position:fixed;top:12px;left:12px;z-index:40;width:38px;height:38px;background:rgba(20,20,22,.92);border:1px solid #333;border-radius:6px;color:#ddd;font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 12px rgba(0,0,0,.5);}
  .tt-hamburger:hover{background:#3a3a3a;}
  .tt-menu{position:fixed;top:0;left:0;bottom:0;width:300px;background:rgba(20,20,22,.96);border-right:1px solid #333;z-index:35;transform:translateX(-110%);transition:transform .22s ease;display:flex;flex-direction:column;box-shadow:2px 0 16px rgba(0,0,0,.5);color:#ddd;font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;}
  .tt-menu.open{transform:translateX(0);}
  .tt-menu-header{position:sticky;top:0;background:rgba(20,20,22,.96);border-bottom:1px solid #333;padding:48px 8px 0 8px;flex:0 0 auto;}
  .tt-menu-title{position:absolute;top:16px;left:60px;font-size:14px;font-weight:600;}
  .tt-menu-tabs{display:flex;gap:4px;flex-wrap:wrap;}
  .tt-tab{background:transparent;border:none;border-bottom:2px solid transparent;color:#999;padding:7px 10px;cursor:pointer;font-size:12px;font-family:inherit;}
  .tt-tab:hover{color:#ddd;}
  .tt-tab.active{color:#ddd;border-bottom-color:#4a90e2;}
  .tt-menu-body{flex:1 1 auto;overflow-y:auto;padding:14px;}
  .tt-tab-panel{display:none;}
  .tt-tab-panel.active{display:block;}
  .tt-ctrl{margin-bottom:16px;}
  .tt-ctrl .lbl{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;}
  .tt-ctrl .lbl .val{font-family:ui-monospace,Menlo,monospace;color:#4a90e2;}
  .tt-ctrl input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:4px;border-radius:2px;background:#444;outline:none;}
  .tt-ctrl input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:50%;background:#4a90e2;cursor:pointer;}
  .tt-ctrl input[type=range]::-moz-range-thumb{width:14px;height:14px;border:none;border-radius:50%;background:#4a90e2;cursor:pointer;}
  .tt-ctrl input[type=number],.tt-ctrl select,.tt-ctrl input[type=color]{width:100%;background:#2b2b2b;color:#ddd;border:1px solid #444;border-radius:4px;padding:7px;font-size:13px;font-family:inherit;box-sizing:border-box;}
  .tt-ctrl input[type=color]{height:34px;padding:2px;}
  .tt-ctrl button.wbtn{width:100%;background:#2b2b2b;color:#ddd;border:1px solid #444;border-radius:4px;padding:8px;cursor:pointer;font-size:13px;font-family:inherit;}
  .tt-ctrl button.wbtn:hover{background:#3a3a3a;}
  .tt-ctrl .row{display:flex;align-items:center;gap:8px;font-size:13px;margin-bottom:4px;}
  .tt-ctrl .row input{width:15px;height:15px;accent-color:#4a90e2;}
  .tt-text{font-size:11px;color:#999;line-height:1.4;}
  .tt-divider{height:1px;background:#333;margin:4px 0 16px;}
  .tt-autohide1{opacity:0;transition:opacity .25s ease;}
  body.tt-active .tt-autohide1{opacity:1;}
  .tt-autohide2{opacity:0;transition:opacity .25s ease;}
  .tt-autohide2:hover{opacity:1;}
  .tt-menu.open{opacity:1;}
  `;
  let _cssDone = false;
  function injectCSS() {
    if (_cssDone) return;
    document.head.appendChild(el('style', { text: CSS }));
    _cssDone = true;
  }

  /* -------- Logger (bottom-right status panel) -------- */
  class Logger {
    constructor() {
      this.items = [];
      this.tbody = el('tbody');
      this.panel = el('div', { class: 'tt-status' }, [
        el('div', { class: 'tt-status-title', text: 'Status' }),
        el('table', {}, [this.tbody]),
      ]);
      document.body.appendChild(this.panel);
    }
    addStatus(label, getter, format) {
      format = format || '%s';
      const cell = el('td', { class: 'v', text: '—' });
      this.tbody.appendChild(el('tr', {}, [el('td', { class: 'k', text: label }), cell]));
      this.items.push({ getter, format, cell });
      return this;
    }
    refresh() {
      for (const it of this.items) {
        let v; try { v = it.getter(); } catch (e) { v = null; }
        it.cell.textContent = (v == null) ? '—' : cFormat(it.format, v);
      }
    }
  }

  /* -------- Tab (a page of controls inside the menu) -------- */
  class Tab {
    constructor(panel, viz) { this.panel = panel; this.viz = viz; }
    _add(node) { this.panel.appendChild(node); return this; }
    _wrap(children) { return el('div', { class: 'tt-ctrl' }, children); }
    _label(text, valEl) {
      return el('label', { class: 'lbl' }, valEl ? [el('span', { text }), valEl] : [el('span', { text })]);
    }
    addSlider(label, onChange, opts) {
      opts = opts || {};
      const min = opts.min != null ? opts.min : 0, max = opts.max != null ? opts.max : 1;
      const value = opts.value != null ? opts.value : min;
      const step = opts.step != null ? opts.step : 'any', fmt = opts.format || '%g';
      const valEl = el('span', { class: 'val', text: cFormat(fmt, value) });
      const input = el('input', { type: 'range', min, max, step, value });
      input.addEventListener('input', () => {
        const x = +input.value; valEl.textContent = cFormat(fmt, x);
        onChange(x); this.viz._markDirty();
      });
      this._add(this._wrap([this._label(label, valEl), input]));
      onChange(value);
      return this;
    }
    addCheckbox(label, onChange, value) {
      const cb = el('input', { type: 'checkbox' }); cb.checked = !!value;
      cb.addEventListener('change', () => { onChange(cb.checked); this.viz._markDirty(); });
      this._add(this._wrap([el('label', { class: 'row' }, [cb, el('span', { text: label })])]));
      onChange(!!value);
      return this;
    }
    addDropdown(label, onChange, options, value) {
      const opts = options.map(o => typeof o === 'string' ? { value: o, label: o } : o);
      if (value === undefined) value = opts[0] && opts[0].value;
      const sel = el('select');
      opts.forEach(o => {
        const op = el('option', { value: o.value, text: o.label }); if (o.value === value) op.selected = true;
        sel.appendChild(op);
      });
      sel.addEventListener('change', () => { onChange(sel.value); this.viz._markDirty(); });
      this._add(this._wrap([this._label(label), sel]));
      onChange(value);
      return this;
    }
    addButton(label, onClick) {
      const b = el('button', { class: 'wbtn', text: label });
      b.addEventListener('click', () => { onClick(); this.viz._markDirty(); });
      this._add(this._wrap([b]));
      return this;
    }
    addNumber(label, onChange, opts) {
      opts = opts || {};
      const input = el('input', { type: 'number', step: opts.step != null ? opts.step : 'any' });
      if (opts.min != null) input.min = opts.min;
      if (opts.max != null) input.max = opts.max;
      input.value = opts.value != null ? opts.value : 0;
      input.addEventListener('change', () => { onChange(+input.value); this.viz._markDirty(); });
      this._add(this._wrap([this._label(label), input]));
      onChange(+input.value);
      return this;
    }
    addRadio(label, onChange, options, value) {
      const opts = options.map(o => typeof o === 'string' ? { value: o, label: o } : o);
      if (value === undefined) value = opts[0] && opts[0].value;
      const name = uid(), group = el('div');
      opts.forEach(o => {
        const r = el('input', { type: 'radio', name }); r.value = o.value; if (o.value === value) r.checked = true;
        r.addEventListener('change', () => { if (r.checked) { onChange(o.value); this.viz._markDirty(); } });
        group.appendChild(el('label', { class: 'row' }, [r, el('span', { text: o.label })]));
      });
      this._add(this._wrap([this._label(label), group]));
      onChange(value);
      return this;
    }
    addColor(label, onChange, value) {
      const input = el('input', { type: 'color', value: value || '#ffffff' });
      input.addEventListener('input', () => { onChange(input.value); this.viz._markDirty(); });
      this._add(this._wrap([this._label(label), input]));
      onChange(input.value);
      return this;
    }
    addReadout(label, getter, format) {
      format = format || '%s';
      const v = el('span', { class: 'val', text: '—' });
      this._add(this._wrap([this._label(label, v)]));
      this.viz._readouts.push(() => {
        let val; try { val = getter(); } catch (e) { val = null; }
        v.textContent = (val == null) ? '—' : cFormat(format, val);
      });
      return this;
    }
    addText(text) { this._add(el('div', { class: 'tt-text', text })); return this; }
    addDivider() { this._add(el('div', { class: 'tt-divider' })); return this; }
  }

  /* -------- Menu (hamburger drawer with sticky tabs) -------- */
  class Menu {
    constructor(viz) {
      this.viz = viz; this.tabs = [];
      this.burger = el('button', { class: 'tt-hamburger', title: 'Menu', html: '☰' });
      this.titleEl = el('div', { class: 'tt-menu-title', text: viz.title || 'Controls' });
      this.tabsEl = el('div', { class: 'tt-menu-tabs' });
      this.bodyEl = el('div', { class: 'tt-menu-body' });
      this.nav = el('nav', { class: 'tt-menu' }, [
        el('div', { class: 'tt-menu-header' }, [this.titleEl, this.tabsEl]),
        this.bodyEl,
      ]);
      document.body.appendChild(this.burger);
      document.body.appendChild(this.nav);
      this.burger.addEventListener('click', () => this.nav.classList.toggle('open'));
    }
    addTab(name) {
      const first = this.tabs.length === 0;
      const btn = el('button', { class: 'tt-tab' + (first ? ' active' : ''), text: name });
      const panel = el('div', { class: 'tt-tab-panel' + (first ? ' active' : '') });
      btn.addEventListener('click', () => this._show(name));
      this.tabsEl.appendChild(btn); this.bodyEl.appendChild(panel);
      const tab = new Tab(panel, this.viz);
      this.tabs.push({ name, btn, panel, tab });
      return tab;
    }
    _show(name) {
      this.tabs.forEach(t => {
        const on = t.name === name;
        t.btn.classList.toggle('active', on);
        t.panel.classList.toggle('active', on);
      });
    }
  }

  /* -------- Visualizer (owns three.js + the run loop) -------- */
  class Visualizer {
    constructor(algo, options) {
      if (!THREE) throw new Error('ThreeTemplate: call useThree({THREE, OrbitControls}) before new Visualizer().');
      this.algo = algo;
      this.options = options || {};
      this.title = this.options.title || document.title;
      this.playing = false; this.done = false;
      this._dirty = true; this._acc = 0; this._last = 0;
      this.speed = 4; this.mode = 'full'; this.loop = false; this.backward = false;
      this._hasSub = false;
      this._undo = []; this.loggers = []; this._readouts = [];
      this.menu = null; this._btns = {}; this._pbTab = false;
      injectCSS();
      this._initThree();
      this.algo.init({ THREE, scene: this.scene, group: this.group, camera: this.camera, controls: this.controls, renderer: this.renderer });
      this._softReset();
    }

    _initThree() {
      const o = this.options;
      this.canvas = el('canvas', { class: 'tt-canvas' });
      document.body.appendChild(this.canvas);
      this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
      this.renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(o.background != null ? o.background : 0x111111);
      this.camera = new THREE.PerspectiveCamera(o.fov || 50, 1, 0.01, 1000);
      const cp = o.cameraPosition || [5, 5, 7];
      this.camera.position.set(cp[0], cp[1], cp[2]);
      const ct = o.cameraTarget || [0, 0, 0];
      if (OrbitControls) {
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.target.set(ct[0], ct[1], ct[2]);
        this.controls.update();
      } else {
        this.controls = null;
        this.camera.lookAt(ct[0], ct[1], ct[2]);
      }
      this.scene.add(new THREE.AmbientLight(0xffffff, o.ambient != null ? o.ambient : 0.6));
      const dl = new THREE.DirectionalLight(0xffffff, 0.9); dl.position.set(5, 10, 7); this.scene.add(dl);
      this.group = new THREE.Group(); this.scene.add(this.group);
    }

    /* ---- builders ---- */
    addPlayback(opts) {
      opts = opts || {};
      injectCSS();
      this.mode = opts.mode || 'full';
      this.speed = opts.speed != null ? opts.speed : 4;
      this.loop = !!opts.loop;
      this.backward = !!opts.backward && typeof this.algo.snapshot === 'function' && typeof this.algo.restore === 'function';
      this._hasSub = !!opts.substeps && typeof this.algo.substep === 'function';
      this.keyboard = opts.keyboard !== false;
      const bar = el('div', { class: 'tt-controls' });
      const mk = (id, icon, title, fn) => {
        const b = el('button', { title, html: icon });
        b.addEventListener('click', fn); bar.appendChild(b); this._btns[id] = b;
      };
      const sep = () => bar.appendChild(el('div', { class: 'tt-sep' }));
      mk('reset', ICONS.reset, 'Reset (rebuild & erase)', () => this.reset());
      mk('erase', ICONS.erase, 'Erase trajectory', () => this.erase());
      sep();
      if (this.backward) mk('back', ICONS.back, 'Step backward (←)', () => this.backStep());
      if (this._hasSub) mk('sub', ICONS.sub, 'Step over — sub-step (↑)', () => this.subStep());
      mk('step', ICONS.full, 'Next full step (→)', () => this.fullStep());
      sep();
      mk('play', ICONS.play, 'Play / pause (space)', () => this.togglePlay());
      document.body.appendChild(bar);
      this.playbackBar = bar;
      this._applyAutohide(bar, opts.autohideLevel != null ? opts.autohideLevel : opts.autohide);
      if (this.keyboard) this._installKeys();
      this._updateButtons();
      return this;
    }
    addLogger(autohide) { injectCSS(); const lg = new Logger(); this._applyAutohide(lg.panel, autohide); this.loggers.push(lg); lg.refresh(); return lg; }
    addMenu(autohide) {
      if (!this.menu) { this.menu = new Menu(this); this._applyAutohide(this.menu.burger, autohide); }
      return this.menu;
    }

    /* autohide: 0 = always visible, 1 = hover on canvas, 2 = hover on element.
       `true` -> level 1, `false`/undefined -> level 0. */
    _normLevel(v) {
      if (v === true) return 1;
      if (v === false || v == null) return 0;
      const n = Math.round(+v);
      return (n >= 0 && n <= 2) ? n : 0;
    }
    _applyAutohide(elem, autohide) {
      const level = this._normLevel(autohide);
      if (level === 1) { elem.classList.add('tt-autohide1'); this._initActive(); }
      else if (level === 2) { elem.classList.add('tt-autohide2'); }
    }
    _initActive() {
      if (this._activeInit) return; this._activeInit = true;
      const show = () => document.body.classList.add('tt-active');
      const hide = () => document.body.classList.remove('tt-active');
      global.addEventListener('pointermove', show);
      global.addEventListener('pointerdown', show);
      document.documentElement.addEventListener('mouseleave', hide);
      global.addEventListener('blur', hide);
    }

    _ensurePlaybackTab() {
      if (this._pbTab || !this.menu || !this.playbackBar) return;
      this._pbTab = true;
      const t = this.menu.addTab('Playback');
      t.addSlider('Speed (steps/s)', v => { this.speed = v; }, { min: 0.5, max: 60, value: this.speed, step: 0.5, format: '%.1f' });
      if (this._hasSub)
        t.addDropdown('Play advances by', v => { this.mode = v; }, [{ value: 'full', label: 'Full step' }, { value: 'sub', label: 'Sub-step' }], this.mode);
      t.addCheckbox('Loop on finish', v => { this.loop = v; }, this.loop);
    }

    /* ---- run loop ---- */
    start() {
      this._ensurePlaybackTab();
      this._updateButtons();
      this._running = true;
      const tick = (now) => {
        if (!this._running) return;
        const dt = Math.min(0.1, (now - (this._last || now)) / 1000); this._last = now;
        this._resize();
        if (this.controls) this.controls.update();
        if (this.playing && !this.done) {
          this._acc += dt;
          const interval = 1 / Math.max(0.1, this.speed);
          let advanced = false, guard = 0;
          while (this._acc >= interval && this.playing && !this.done && guard < 10000) {
            this._acc -= interval; this._advance(); advanced = true; guard++;
          }
          if (advanced) this._markDirty();
          if (this.done) {
            if (this.loop) { this._softReset(); this.playing = true; this._markDirty(); this._updateButtons(); }
            else this.pause();
          }
        }
        if (this._dirty) { this._safeDraw(); this._refresh(); this._dirty = false; }
        this.renderer.render(this.scene, this.camera);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
    stop() { this._running = false; }

    _resize() {
      const c = this.canvas, w = c.clientWidth, h = c.clientHeight;
      if (w && h && (c.width !== w || c.height !== h)) {
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
      }
    }

    /* ---- stepping ---- */
    _completeFull() {
      if (typeof this.algo.substep === 'function') { let g = 0; while (!this.algo.substep() && g < 100000) g++; }
      else this.algo.step();
    }
    _advance() {
      if (this.backward) this._pushSnap();
      if (this.mode === 'sub' && this._hasSub) this.algo.substep();
      else this._completeFull();
      this.done = !!this.algo.isDone();
    }
    _pushSnap() { try { this._undo.push(this.algo.snapshot()); if (this._undo.length > 2000) this._undo.shift(); } catch (e) { } }

    subStep() {
      if (this.done || !this._hasSub) return;
      if (this.backward) this._pushSnap();
      this.algo.substep(); this.done = !!this.algo.isDone();
      this._markDirty(); this._updateButtons();
    }
    fullStep() {
      if (this.done) return;
      if (this.backward) this._pushSnap();
      this._completeFull(); this.done = !!this.algo.isDone();
      this._markDirty(); this._updateButtons();
    }
    backStep() {
      if (!this.backward || !this._undo.length) return;
      this.algo.restore(this._undo.pop());
      this.done = !!this.algo.isDone(); this.playing = false;
      this._markDirty(); this._updateButtons();
    }
    play() { if (this.done) return; this.playing = true; this._acc = 0; this._updateButtons(); }
    pause() { this.playing = false; this._updateButtons(); }
    togglePlay() { this.playing ? this.pause() : this.play(); }

    reset() { this._softReset(); this._markDirty(); this._updateButtons(); }
    _softReset() {
      this.playing = false; this.done = false; this._undo = [];
      try { this.algo.reset(); } catch (e) { console.error(e); }
      try { this.algo.clear(); } catch (e) { console.error(e); }
    }
    erase() { try { this.algo.clear(); } catch (e) { console.error(e); } this._markDirty(); }

    _markDirty() { this._dirty = true; }
    _safeDraw() { try { this.algo.draw(); } catch (e) { console.error(e); } }
    _refresh() { this.loggers.forEach(l => l.refresh()); this._readouts.forEach(f => f()); }

    _updateButtons() {
      const b = this._btns;
      if (b.play) { b.play.innerHTML = this.playing ? ICONS.pause : ICONS.play; b.play.classList.toggle('active', this.playing); b.play.disabled = this.done; }
      if (b.sub) b.sub.disabled = this.done;
      if (b.step) b.step.disabled = this.done;
      if (b.back) b.back.disabled = !this._undo.length;
    }

    _installKeys() {
      if (this._keys) return; this._keys = true;
      global.addEventListener('keydown', (e) => {
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
        switch (e.key) {
          case ' ': e.preventDefault(); this.togglePlay(); break;
          case 'ArrowRight': this.fullStep(); break;
          case 'ArrowUp': this.subStep(); break;
          case 'ArrowLeft': this.backStep(); break;
          case 'r': case 'R': this.reset(); break;
          case 'e': case 'E': this.erase(); break;
        }
      });
    }
  }

  global.ThreeTemplate = { Visualizer, useThree, cFormat, ICONS, version: '0.1.0' };
})(window);
