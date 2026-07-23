/* =====================================================================
 * gradient-descent.js  —  example algorithm for canvas-template.js
 *
 * Minimizes a bumpy surface  z = f(x,y)  by (momentum) gradient descent.
 * One full step = one iteration; two sub-steps = (1) compute gradient,
 * (2) take the descent step.  Loaded as a classic <script>, so it uses
 * the global `THREE` that ThreeTemplate.useThree() installs.
 * ===================================================================== */
(function(global) {
  'use strict';

  // the objective and its gradient (no THREE needed here)
  const f = (x, y) => 0.2 * (x * x + y * y) + 0.6 * Math.sin(1.5 * x) * Math.cos(1.5 * y);
  const fx = (x, y) => 0.4 * x + 0.9 * Math.cos(1.5 * x) * Math.cos(1.5 * y);
  const fy = (x, y) => 0.4 * y - 0.9 * Math.sin(1.5 * x) * Math.sin(1.5 * y);

  // surface mesh coloured by height (built once, in init). Stashes the
  // geometry + per-vertex height fraction on `store` so it can be recoloured.
  function buildSurface(store) {
    const N = 80, R = 4;
    const geo = new THREE.PlaneGeometry(2 * R, 2 * R, N, N);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position, hs = [];
    let zmin = Infinity, zmax = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getZ(i), h = f(x, y);
      hs.push(h); zmin = Math.min(zmin, h); zmax = Math.max(zmax, h); pos.setY(i, h);
    }
    const t = new Float32Array(pos.count);                  // 0 at valleys, 1 at peaks
    for (let i = 0; i < pos.count; i++) t[i] = (hs[i] - zmin) / (zmax - zmin || 1);
    geo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(pos.count * 3), 3));
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: THREE.DoubleSide }));
    const wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.08 }));
    store.surfaceGeo = geo; store.surfaceT = t;
    const g = new THREE.Group(); g.add(mesh, wire); return g;
  }

  class GradientDescent {
    constructor() {
      this.alpha = 0.25;        // step size            (menu-controlled)
      this.momentum = 0.0;      // momentum             (menu-controlled)
      this.showArrow = true;    // gradient arrow       (menu-controlled)
      this.x0 = -3.0; this.y0 = 2.6;   // start point   (menu-controlled)
      this.surfaceColor = '#4fc3f7';   // peak colour    (menu-controlled)
      this._initState();
    }
    _initState() {
      this.x = this.x0; this.y = this.y0;
      this.vx = 0; this.vy = 0;
      this.iter = 0; this.phase = 'init'; this.finished = false; this._sub = 0;
      this.trajectory = [this.world(this.x, this.y)];   // history-array: the visited points
    }
    world(x, y) { return new THREE.Vector3(x, f(x, y), y); }

    // build the 3D objects ONCE
    init(ctx) {
      this.group = ctx.group;
      this.group.add(buildSurface(this));
      this.recolorSurface(this.surfaceColor);
      this.marker = new THREE.Mesh(new THREE.SphereGeometry(0.12, 24, 24),
        new THREE.MeshStandardMaterial({ color: 0xffeb3b, emissive: 0x554400 }));
      this.trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff5252 }));
      this.arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0xff5252, 0.25, 0.15);
      this.group.add(this.marker, this.trail, this.arrow);
    }

    // one sub-step; returns true when THIS full step is complete
    substep() {
      if (this.finished) return true;
      if (this._sub === 0) {                       // sub-step 1: compute gradient
        this._g = { gx: fx(this.x, this.y), gy: fy(this.x, this.y) };
        this.phase = 'grad'; this._sub = 1;
        return false;                              // full step NOT complete yet
      }
      // sub-step 2: take the (momentum) descent step
      this.vx = this.momentum * this.vx - this.alpha * this._g.gx;
      this.vy = this.momentum * this.vy - this.alpha * this._g.gy;
      this.x = Math.max(-4, Math.min(4, this.x + this.vx));
      this.y = Math.max(-4, Math.min(4, this.y + this.vy));
      this.iter++; this.phase = 'update'; this._sub = 0;
      this.trajectory.push(this.world(this.x, this.y));
      if (Math.hypot(this._g.gx, this._g.gy) < 1e-3 || this.iter >= 500) this.finished = true;
      return true;                                 // full step complete
    }
    // step() is auto-derived by the framework from substep()

    isDone() { return this.finished; }

    // sync the 3D objects to current state
    draw() {
      this.marker.position.copy(this.world(this.x, this.y));
      this.trail.geometry.setFromPoints(this.trajectory);
      this.arrow.visible = this.showArrow;
      if (this.showArrow) {
        const gx = fx(this.x, this.y), gy = fy(this.x, this.y), gn = Math.hypot(gx, gy);
        this.arrow.position.copy(this.marker.position);
        if (gn > 1e-6) {
          this.arrow.setDirection(new THREE.Vector3(-gx, 0, -gy).normalize());
          this.arrow.setLength(Math.min(1.5, 0.5 + gn), 0.25, 0.15);
        }
      }
    }

    reset() { this._initState(); }                 // framework calls clear() right after
    clear() { this.trajectory = [this.world(this.x, this.y)]; }

    // OPTIONAL — enables the "step backward" button
    snapshot() {
      return {
        x: this.x, y: this.y, vx: this.vx, vy: this.vy, iter: this.iter,
        phase: this.phase, finished: this.finished, sub: this._sub,
        g: this._g ? { gx: this._g.gx, gy: this._g.gy } : null,
        traj: this.trajectory.map(v => v.clone())
      };
    }
    restore(s) {
      this.x = s.x; this.y = s.y; this.vx = s.vx; this.vy = s.vy; this.iter = s.iter;
      this.phase = s.phase; this.finished = s.finished; this._sub = s.sub;
      this._g = s.g ? { gx: s.g.gx, gy: s.g.gy } : undefined;
      this.trajectory = s.traj.map(v => v.clone());
    }

    // getters for the logger
    getIteration() { return this.iter; }
    getPhase() { return this.phase; }
    getValue() { return f(this.x, this.y); }
    getGradNorm() { return Math.hypot(fx(this.x, this.y), fy(this.x, this.y)); }

    // setters for the menu
    setStepSize(a) { this.alpha = a; }
    setMomentum(m) { this.momentum = m; }
    setArrowVisible(v) { this.showArrow = v; }
    setStart(axis, val) { if (axis === 'x') this.x0 = val; else this.y0 = val; }
    setSurfaceColor(hex) { this.surfaceColor = hex; this.recolorSurface(hex); }

    // re-shade the surface: dark valleys -> picked colour at the peaks
    recolorSurface(hex) {
      const geo = this.surfaceGeo;
      if (!geo) return;                          // surface not built yet
      const low = new THREE.Color(0x14315c), high = new THREE.Color(hex), c = new THREE.Color();
      const t = this.surfaceT, col = geo.attributes.color;
      for (let i = 0; i < t.length; i++) { c.copy(low).lerp(high, t[i]); col.setXYZ(i, c.r, c.g, c.b); }
      col.needsUpdate = true;
    }
  }

  global.GradientDescent = GradientDescent;
})(window);
