/* =====================================================================
 * atlas-app.js  —  "Manifold & Atlas" visualization as a canvas-template
 * algorithm.  Reuses the Manifold math (window.Manifold) for geodesics.
 *
 * Mapping onto the framework contract:
 *   full step  -> advance to the next stage (phase)
 *   sub-step   -> one frame of the exp_p warp animation (stage 4 -> 5)
 *
 * Stages: 1 manifold M · 2 point p · 3 tangent space · 4 polar grid ·
 *         5 exp_p warp · 6 atlas of charts.
 * ===================================================================== */
(function(global) {
  'use strict';

  const PSI_RES = 180, R_STEPS = 180, DISK_R_SEGS = 18, NUM_CIRCLES = 4, NUM_RAYS = 12;
  const LIFT = 0.015, FLAT_COLOR = 0x00e5ff;
  const easeInOutCubic = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

  const STAGES = {
    1: 'M — the manifold',
    2: 'p ∈ M — a point',
    3: 'T_pM — tangent space',
    4: '(r, ψ) — polar coords',
    5: 'exp_p — warp onto M',
    6: 'atlas of charts',
  };
  const DEFAULT_CENTERS = [
    [Math.PI / 3, Math.PI / 4], [Math.PI / 3, 3 * Math.PI / 4 + 0.2],
    [Math.PI / 3, 5 * Math.PI / 4], [Math.PI / 3, 7 * Math.PI / 4 - 0.2],
    [4 * Math.PI / 3, 0.0], [4 * Math.PI / 3, Math.PI / 2 + 0.1],
    [4 * Math.PI / 3, Math.PI], [4 * Math.PI / 3, 3 * Math.PI / 2],
  ];
  const PALETTE = [0x2196f3, 0xf44336, 0x4caf50, 0xff9800, 0x9c27b0, 0x009688, 0xe91e63, 0xffc107];

  class AtlasApp {
    constructor(M, opts = {}) {
      this.M = M;
      this.polarRho = opts.polarRho != null ? opts.polarRho : 1.2;
      this.chartRadius = opts.chartRadius != null ? opts.chartRadius : 0.78;
      this.theta0 = opts.theta0 != null ? opts.theta0 : Math.PI / 3;
      this.phi0 = opts.phi0 != null ? opts.phi0 : 0.0;
      this.centers = opts.centers || DEFAULT_CENTERS;
      this.morphSteps = opts.morphSteps != null ? opts.morphSteps : 60;
      this._initState();
    }
    _initState() { this.stage = 1; this.morphT = 0; this._morphing = false; this._frame = 0; }

    // ── build the 3D objects once ──────────────────────────────────────
    init(ctx) {
      const T = this.THREE = ctx.THREE;
      this.scene = ctx.scene; this.group = ctx.group;
      this.camera = ctx.camera; this.controls = ctx.controls; this.renderer = ctx.renderer;

      // torus sits in a z-up world
      this.camera.up.set(0, 0, 1);
      this.camera.position.set(5, -5, 3.5);
      if (this.controls) { this.controls.target.set(0, 0, 0); this.controls.update(); }

      this.torus = this._paramSurface((u, v) => this.M.pt(u, v),
        this.M.uRange[0], this.M.uRange[1], this.M.vRange[0], this.M.vRange[1], 60, 90,
        new T.MeshPhongMaterial({ color: 0xc0c0c0, transparent: true, opacity: 0.92, side: T.DoubleSide, shininess: 25 }));

      this.dot = new T.Mesh(new T.SphereGeometry(0.085, 24, 24),
        new T.MeshPhongMaterial({ color: 0xffeb3b, emissive: 0x554400 }));

      this.tangent = new T.Mesh(this._buildDiskGeometry(),
        new T.MeshPhongMaterial({ color: 0xffeb3b, transparent: true, opacity: 0.40, side: T.DoubleSide, depthWrite: false, shininess: 18 }));
      this.tangent.renderOrder = 1;

      this.polarGrid = new T.LineSegments(new T.BufferGeometry(),
        new T.LineBasicMaterial({ color: FLAT_COLOR, transparent: true, opacity: 0.95 }));
      this.polarGrid.renderOrder = 2;

      this.atlasGroup = new T.Group();

      this.group.add(this.torus, this.dot, this.tangent, this.polarGrid, this.atlasGroup);

      this.refreshPoint();
      this.rebuildAtlas();
      this._installPointer();
    }

    // ── mesh helpers ───────────────────────────────────────────────────
    _paramSurface(fn, uMin, uMax, vMin, vMax, uSegs, vSegs, material) {
      const T = this.THREE, verts = [], idx = [];
      for (let i = 0; i <= uSegs; i++) {
        const u = uMin + (uMax - uMin) * i / uSegs;
        for (let j = 0; j <= vSegs; j++) { const v = vMin + (vMax - vMin) * j / vSegs; const p = fn(u, v); verts.push(p[0], p[1], p[2]); }
      }
      for (let i = 0; i < uSegs; i++) for (let j = 0; j < vSegs; j++) {
        const a = i * (vSegs + 1) + j, b = a + 1, c = a + (vSegs + 1), d = c + 1; idx.push(a, b, c, b, d, c);
      }
      const geo = new T.BufferGeometry();
      geo.setAttribute('position', new T.Float32BufferAttribute(verts, 3));
      geo.setIndex(idx); geo.computeVertexNormals();
      return new T.Mesh(geo, material);
    }
    _buildDiskGeometry() {
      const T = this.THREE;
      const positions = new Float32Array((DISK_R_SEGS + 1) * PSI_RES * 3), idx = [];
      for (let i = 0; i < DISK_R_SEGS; i++) for (let j = 0; j < PSI_RES; j++) {
        const a = i * PSI_RES + j, b = i * PSI_RES + ((j + 1) % PSI_RES), c = (i + 1) * PSI_RES + j, d = (i + 1) * PSI_RES + ((j + 1) % PSI_RES);
        idx.push(a, b, c, b, d, c);
      }
      const geo = new T.BufferGeometry();
      geo.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
      geo.setIndex(idx);
      return geo;
    }
    _buildGridVerts(positionFn) {
      const verts = [];
      for (let c = 1; c < NUM_CIRCLES; c++) {
        const rad = this.polarRho * c / NUM_CIRCLES;
        for (let i = 0; i < PSI_RES; i++) {
          const p1 = 2 * Math.PI * i / PSI_RES, p2 = 2 * Math.PI * ((i + 1) % PSI_RES) / PSI_RES;
          verts.push(...positionFn(rad, p1), ...positionFn(rad, p2));
        }
      }
      for (let k = 0; k < NUM_RAYS; k++) {
        const psi = 2 * Math.PI * k / NUM_RAYS;
        for (let s = 0; s < R_STEPS; s++) {
          const r1 = this.polarRho * s / R_STEPS, r2 = this.polarRho * (s + 1) / R_STEPS;
          verts.push(...positionFn(r1, psi), ...positionFn(r2, psi));
        }
      }
      for (let i = 0; i < PSI_RES; i++) {
        const p1 = 2 * Math.PI * i / PSI_RES, p2 = 2 * Math.PI * ((i + 1) % PSI_RES) / PSI_RES;
        verts.push(...positionFn(this.polarRho, p1), ...positionFn(this.polarRho, p2));
      }
      return new Float32Array(verts);
    }
    _buildDiskVerts(positionFn) {
      const out = new Float32Array((DISK_R_SEGS + 1) * PSI_RES * 3); let k = 0;
      for (let i = 0; i <= DISK_R_SEGS; i++) {
        const rad = this.polarRho * i / DISK_R_SEGS;
        for (let j = 0; j < PSI_RES; j++) {
          const psi = 2 * Math.PI * j / PSI_RES; const p = positionFn(rad, psi);
          out[k++] = p[0]; out[k++] = p[1]; out[k++] = p[2];
        }
      }
      return out;
    }

    // ── recompute geodesic data for the current p / ρ ──────────────────
    refreshPoint() {
      const M = this.M, T = this.THREE;
      const p = M.pt(this.theta0, this.phi0);
      const { e1, e2, n: nrm } = M.frame(this.theta0, this.phi0);
      const trajs = [];
      for (let i = 0; i < PSI_RES; i++)
        trajs.push(M.geodesicTrajectory(this.theta0, this.phi0, 2 * Math.PI * i / PSI_RES, this.polarRho, R_STEPS));
      const psiToIdx = psi => ((Math.round(psi / (2 * Math.PI) * PSI_RES) % PSI_RES) + PSI_RES) % PSI_RES;
      const radToStep = rad => Math.max(0, Math.min(R_STEPS, Math.round(rad * R_STEPS / this.polarRho)));

      const flatDisk = (rad, psi) => [
        p[0] + rad * Math.cos(psi) * e1[0] + rad * Math.sin(psi) * e2[0],
        p[1] + rad * Math.cos(psi) * e1[1] + rad * Math.sin(psi) * e2[1],
        p[2] + rad * Math.cos(psi) * e1[2] + rad * Math.sin(psi) * e2[2]];
      const flatGrid = (rad, psi) => { const q = flatDisk(rad, psi); return [q[0] + LIFT * nrm[0], q[1] + LIFT * nrm[1], q[2] + LIFT * nrm[2]]; };
      const warpedAt = (rad, psi, margin) => {
        if (rad < 1e-9) return M.pt(this.theta0, this.phi0, margin);
        const [th, ph] = trajs[psiToIdx(psi)][radToStep(rad)]; return M.pt(th, ph, margin);
      };
      const warpedDisk = (rad, psi) => warpedAt(rad, psi, 0.003);
      const warpedGrid = (rad, psi) => warpedAt(rad, psi, 0.003 + LIFT);

      this._flatDisk = this._buildDiskVerts(flatDisk);
      this._warpedDisk = this._buildDiskVerts(warpedDisk);
      this._flatGrid = this._buildGridVerts(flatGrid);
      this._warpedGrid = this._buildGridVerts(warpedGrid);

      const cur = this.polarGrid.geometry.attributes.position;
      if (!cur || cur.array.length !== this._flatGrid.length)
        this.polarGrid.geometry.setAttribute('position', new T.Float32BufferAttribute(new Float32Array(this._flatGrid.length), 3));
      this._applyMorph();
    }

    _applyMorph() {
      const t = this.morphT;
      if (this._flatDisk && this._warpedDisk) {
        const arr = this.tangent.geometry.attributes.position.array, a = this._flatDisk, b = this._warpedDisk;
        for (let i = 0; i < arr.length; i++) arr[i] = (1 - t) * a[i] + t * b[i];
        this.tangent.geometry.attributes.position.needsUpdate = true;
        this.tangent.geometry.computeVertexNormals();
        this.tangent.geometry.computeBoundingSphere();
      }
      if (this._flatGrid && this._warpedGrid) {
        const arr = this.polarGrid.geometry.attributes.position.array, a = this._flatGrid, b = this._warpedGrid;
        for (let i = 0; i < arr.length; i++) arr[i] = (1 - t) * a[i] + t * b[i];
        this.polarGrid.geometry.attributes.position.needsUpdate = true;
        this.polarGrid.geometry.computeBoundingSphere();
      }
    }

    // ── atlas of charts ────────────────────────────────────────────────
    _makeAtlasDisk(th0, ph0, rho, color) {
      const M = this.M, T = this.THREE;
      const trajs = [];
      for (let i = 0; i < PSI_RES; i++) trajs.push(M.geodesicTrajectory(th0, ph0, 2 * Math.PI * i / PSI_RES, rho, R_STEPS));
      const psiIdx = psi => ((Math.round(psi / (2 * Math.PI) * PSI_RES) % PSI_RES) + PSI_RES) % PSI_RES;
      const radStep = rad => Math.max(0, Math.min(R_STEPS, Math.round(rad * R_STEPS / rho)));
      const positions = new Float32Array((DISK_R_SEGS + 1) * PSI_RES * 3), indices = [];
      for (let i = 0; i < DISK_R_SEGS; i++) for (let j = 0; j < PSI_RES; j++) {
        const a = i * PSI_RES + j, b = i * PSI_RES + ((j + 1) % PSI_RES), c = (i + 1) * PSI_RES + j, d = (i + 1) * PSI_RES + ((j + 1) % PSI_RES);
        indices.push(a, b, c, b, d, c);
      }
      let k = 0;
      for (let i = 0; i <= DISK_R_SEGS; i++) {
        const rad = rho * i / DISK_R_SEGS;
        for (let j = 0; j < PSI_RES; j++) {
          const psi = 2 * Math.PI * j / PSI_RES;
          const p = (rad < 1e-9) ? M.pt(th0, ph0, LIFT) : M.pt(...trajs[psiIdx(psi)][radStep(rad)], LIFT);
          positions[k++] = p[0]; positions[k++] = p[1]; positions[k++] = p[2];
        }
      }
      const geo = new T.BufferGeometry();
      geo.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
      geo.setIndex(indices); geo.computeVertexNormals();
      const mesh = new T.Mesh(geo, new T.MeshPhongMaterial({ color, transparent: true, opacity: 0.72, side: T.DoubleSide, shininess: 30, depthWrite: false }));
      mesh.renderOrder = 3;
      return mesh;
    }
    rebuildAtlas() {
      const g = this.atlasGroup;
      while (g.children.length) { g.children[0].geometry.dispose(); g.remove(g.children[0]); }
      this.centers.forEach(([t0, ph0], i) => g.add(this._makeAtlasDisk(t0, ph0, this.chartRadius, PALETTE[i % PALETTE.length])));
    }

    // ── the canvas-template contract ────────────────────────────────────
    substep() {
      if (this._morphing) {                       // animating exp_p warp
        this._frame++;
        this.morphT = easeInOutCubic(Math.min(1, this._frame / this.morphSteps));
        if (this._frame >= this.morphSteps) { this.morphT = 1; this._morphing = false; return true; }
        return false;
      }
      if (this.stage >= 6) return true;
      this.stage += 1;                            // advance to next stage
      if (this.stage === 5) { this._morphing = true; this._frame = 0; this.morphT = 0; return false; }
      return true;
    }
    isDone() { return this.stage >= 6 && !this._morphing; }
    draw() { this._sync(); }
    reset() { this._initState(); this._applyMorph(); }
    clear() { this._sync(); }

    _sync() {
      this.dot.visible = this.stage >= 2 && this.stage <= 5;
      this.tangent.visible = this.stage >= 3 && this.stage <= 5;
      this.polarGrid.visible = this.stage >= 4 && this.stage <= 5;
      this.atlasGroup.visible = this.stage === 6;
      this.dot.position.set(...this.M.pt(this.theta0, this.phi0, LIFT));
      this._applyMorph();
    }

    // OPTIONAL — enables the backward button (steps the animation back too)
    snapshot() { return { stage: this.stage, morphT: this.morphT, morphing: this._morphing, frame: this._frame }; }
    restore(s) { this.stage = s.stage; this.morphT = s.morphT; this._morphing = s.morphing; this._frame = s.frame; }

    // ── getters / setters for logger + menu ────────────────────────────
    getStage() { return this.stage; }
    getPhase() { return STAGES[this.stage]; }
    getRho() { return this.polarRho; }
    getMorph() { return this.morphT; }
    setRho(v) { if (v === this.polarRho) return; this.polarRho = v; this.refreshPoint(); this._sync(); }
    setChartRadius(v) { if (v === this.chartRadius) return; this.chartRadius = v; this.rebuildAtlas(); }
    setPoint(axis, val) {
      const cur = axis === 'theta' ? this.theta0 : this.phi0;
      if (val === cur) return;
      if (axis === 'theta') this.theta0 = val; else this.phi0 = val;
      this.refreshPoint(); this._sync();
    }

    // ── click the torus to move p; double-click resets the camera ──────
    _installPointer() {
      const T = this.THREE, canvas = this.renderer.domElement;
      const ray = new T.Raycaster(), ndc = new T.Vector2();
      let down = null;
      canvas.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; });
      canvas.addEventListener('pointerup', e => {
        if (!down) return;
        const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y); down = null;
        if (moved > 5) return;
        const rect = canvas.getBoundingClientRect();
        ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        ray.setFromCamera(ndc, this.camera);
        const hit = ray.intersectObject(this.torus, false)[0];
        if (!hit) return;
        [this.theta0, this.phi0] = this.M.nearestParam(hit.point.x, hit.point.y, hit.point.z);
        this.refreshPoint(); this._sync();
      });
      canvas.addEventListener('dblclick', () => {
        this.camera.position.set(5, -5, 3.5);
        if (this.controls) { this.controls.target.set(0, 0, 0); this.controls.update(); }
      });
    }
  }

  global.AtlasApp = AtlasApp;
})(window);
