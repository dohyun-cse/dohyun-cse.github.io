/**
 * manifold.js — Generic 2-manifold in R³
 *
 * Defines the Manifold class. Given a smooth parametric map φ: U ⊂ R² → R³,
 * this library auto-computes the Riemannian metric, Christoffel symbols, and
 * geodesics via finite differences + RK4, providing the exponential map
 * needed for tangent-space animations.
 *
 * Usage (classic script — works from file:// with no server):
 *   <script src="manifold.js"></script>   // defines window.Manifold
 *   const M = new Manifold({ param: (u,v) => [x,y,z], uRange, vRange });
 *   M.geodesicTrajectory(u0, v0, psi, arcLen, steps) → [[u,v], …]
 */
(function(global) {
  'use strict';

  // ── internal vector helpers ───────────────────────────────────────────────────
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm3 = v => { const m = Math.hypot(v[0], v[1], v[2]); return [v[0] / m, v[1] / m, v[2] / m]; };
  const scale3 = (v, s) => [v[0] * s, v[1] * s, v[2] * s];
  const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

  class Manifold {
    /**
     * @param {object}   opts
     * @param {function} opts.param     (u,v) → [x,y,z]
     * @param {number[]} opts.uRange    [uMin, uMax]  default [0, 2π]
     * @param {number[]} opts.vRange    [vMin, vMax]  default [0, 2π]
     * @param {number}   [opts.eps=1e-5]   step for 1st-order finite differences
     * @param {number}   [opts.eps2=5e-4]  step for 2nd-order finite differences
     *
     * Optional analytic overrides (skip finite differences for speed / accuracy):
     * @param {function} [opts.duFn]   exact (u,v) → ∂φ/∂u
     * @param {function} [opts.dvFn]   exact (u,v) → ∂φ/∂v
     */
    constructor({ param, uRange = [0, 2 * Math.PI], vRange = [0, 2 * Math.PI],
      eps = 1e-5, eps2 = 5e-4, duFn = null, dvFn = null, normalFn = null }) {
      this._p = param;
      this.uRange = uRange;
      this.vRange = vRange;
      this._eps = eps;
      this._eps2 = eps2;
      this._duFn = duFn;
      this._dvFn = dvFn;
      // Optional analytic normal override — use when cross(∂u,∂v) points inward.
      this._normalFn = normalFn;
    }

    // ── parameterization ───────────────────────────────────────────────────────

    /** Raw point in R³. */
    param(u, v) { return this._p(u, v); }

    /** Point in R³, lifted by `margin` along the outward unit normal. */
    pt(u, v, margin = 0) {
      const p = this._p(u, v);
      if (!margin) return p;
      const n = this.normal(u, v);
      return add3(p, scale3(n, margin));
    }

    // ── first partial derivatives ─────────────────────────────────────────────

    du(u, v) {
      if (this._duFn) return this._duFn(u, v);
      const e = this._eps;
      const a = this._p(u + e, v), b = this._p(u - e, v);
      return [(a[0] - b[0]) / (2 * e), (a[1] - b[1]) / (2 * e), (a[2] - b[2]) / (2 * e)];
    }

    dv(u, v) {
      if (this._dvFn) return this._dvFn(u, v);
      const e = this._eps;
      const a = this._p(u, v + e), b = this._p(u, v - e);
      return [(a[0] - b[0]) / (2 * e), (a[1] - b[1]) / (2 * e), (a[2] - b[2]) / (2 * e)];
    }

    // ── differential geometry ──────────────────────────────────────────────────

    /** Outward unit normal at (u,v). */
    normal(u, v) {
      if (this._normalFn) return norm3(this._normalFn(u, v));
      return norm3(cross(this.du(u, v), this.dv(u, v)));
    }

    /**
     * Orthonormal frame {e1, e2, n} in T_pM via Gram-Schmidt.
     * e1 ∥ ∂φ/∂u,  e2 ⊥ e1 in the tangent plane,  n = outward normal.
     */
    frame(u, v) {
      const fu = this.du(u, v), fv = this.dv(u, v);
      const e1 = norm3(fu);
      const e2 = norm3(sub3(fv, scale3(e1, dot3(fv, e1))));
      const n = this.normal(u, v);   // uses normalFn override if provided
      return { e1, e2, n };
    }

    /** First fundamental form g_ij = ∂φ/∂u^i · ∂φ/∂u^j. */
    metric(u, v) {
      const fu = this.du(u, v), fv = this.dv(u, v);
      return { g11: dot3(fu, fu), g12: dot3(fu, fv), g22: dot3(fv, fv) };
    }

    // ── Christoffel symbols Γ^k_ij (2nd kind) ────────────────────────────────

    _dmetric(u, v) {
      const e = this._eps2;
      const mu = this.metric(u + e, v), mu_ = this.metric(u - e, v);
      const mv = this.metric(u, v + e), mv_ = this.metric(u, v - e);
      const D = 1 / (2 * e);
      return {
        dg11du: (mu.g11 - mu_.g11) * D, dg12du: (mu.g12 - mu_.g12) * D, dg22du: (mu.g22 - mu_.g22) * D,
        dg11dv: (mv.g11 - mv_.g11) * D, dg12dv: (mv.g12 - mv_.g12) * D, dg22dv: (mv.g22 - mv_.g22) * D,
      };
    }

    /**
     * Returns Γ^k_ij at (u,v) using standard index layout:
     *   { G1_11, G1_12, G1_22, G2_11, G2_12, G2_22 }
     * where 1 ↔ u-direction, 2 ↔ v-direction.
     */
    christoffel(u, v) {
      const { g11, g12, g22 } = this.metric(u, v);
      const det = g11 * g22 - g12 * g12;
      if (Math.abs(det) < 1e-14)
        return { G1_11: 0, G1_12: 0, G1_22: 0, G2_11: 0, G2_12: 0, G2_22: 0 };

      // Inverse metric
      const gi11 = g22 / det, gi12 = -g12 / det, gi22 = g11 / det;

      const { dg11du, dg12du, dg22du, dg11dv, dg12dv, dg22dv } = this._dmetric(u, v);

      // 1st-kind Christoffel [ij,k] = ½(∂_i g_jk + ∂_j g_ik − ∂_k g_ij)
      // (indices: 1=u, 2=v)
      const c111 = 0.5 * dg11du;                  // [11,1]
      const c112 = dg12du - 0.5 * dg11dv;         // [11,2]
      const c121 = 0.5 * dg11dv;                  // [12,1]
      const c122 = 0.5 * dg22du;                  // [12,2]
      const c221 = dg12dv - 0.5 * dg22du;         // [22,1]
      const c222 = 0.5 * dg22dv;                  // [22,2]

      return {
        G1_11: gi11 * c111 + gi12 * c112,
        G1_12: gi11 * c121 + gi12 * c122,
        G1_22: gi11 * c221 + gi12 * c222,
        G2_11: gi12 * c111 + gi22 * c112,
        G2_12: gi12 * c121 + gi22 * c122,
        G2_22: gi12 * c221 + gi22 * c222,
      };
    }

    // ── geodesic ODE + RK4 ───────────────────────────────────────────────────

    _rhs([u, v, ud, vd]) {
      const { G1_11, G1_12, G1_22, G2_11, G2_12, G2_22 } = this.christoffel(u, v);
      return [
        ud, vd,
        -(G1_11 * ud * ud + 2 * G1_12 * ud * vd + G1_22 * vd * vd),
        -(G2_11 * ud * ud + 2 * G2_12 * ud * vd + G2_22 * vd * vd),
      ];
    }

    _rk4(s, dt) {
      const k1 = this._rhs(s);
      const k2 = this._rhs(s.map((x, i) => x + dt / 2 * k1[i]));
      const k3 = this._rhs(s.map((x, i) => x + dt / 2 * k2[i]));
      const k4 = this._rhs(s.map((x, i) => x + dt * k3[i]));
      return s.map((x, i) => x + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    }

    // ── exponential map ──────────────────────────────────────────────────────

    /**
     * Initial parameter-space velocity [u̇, v̇] for a unit-speed geodesic
     * leaving (u0,v0) at tangent-plane angle `psi`
     * (psi=0 → along e₁ = φ_u/|φ_u|,  psi=π/2 → along e₂).
     */
    initialVelocity(u0, v0, psi) {
      const fu = this.du(u0, v0), fv = this.dv(u0, v0);
      const e1 = norm3(fu);
      const e2 = norm3(sub3(fv, scale3(e1, dot3(fv, e1))));
      // Unit tangent in R³
      const w = [Math.cos(psi) * e1[0] + Math.sin(psi) * e2[0],
      Math.cos(psi) * e1[1] + Math.sin(psi) * e2[1],
      Math.cos(psi) * e1[2] + Math.sin(psi) * e2[2]];
      // Solve (first fundamental form) · [u̇;v̇] = [w·φ_u; w·φ_v]
      const { g11, g12, g22 } = this.metric(u0, v0);
      const r1 = dot3(w, fu), r2 = dot3(w, fv);
      const det = g11 * g22 - g12 * g12;
      return [(g22 * r1 - g12 * r2) / det, (g11 * r2 - g12 * r1) / det];
    }

    /**
     * Integrate the geodesic starting at (u0,v0) in direction `psi`,
     * for arc length `totalArc`, returning `steps+1` parameter pairs.
     *
     * @returns {number[][]}  [[u,v], …]
     */
    geodesicTrajectory(u0, v0, psi, totalArc, steps) {
      const [ud, vd] = this.initialVelocity(u0, v0, psi);
      let s = [u0, v0, ud, vd];
      const traj = [[u0, v0]];
      const dt = totalArc / steps;
      for (let i = 0; i < steps; i++) {
        s = this._rk4(s, dt);
        traj.push([s[0], s[1]]);
      }
      return traj;
    }

    // ── inverse map ──────────────────────────────────────────────────────────

    /**
     * Find (u,v) whose image under param is closest to the 3D point [px,py,pz].
     * Uses a coarse grid search followed by gradient-descent refinement.
     */
    nearestParam(px, py, pz, nGrid = 30) {
      const [u0, u1] = this.uRange, [v0, v1] = this.vRange;
      let bu = u0, bv = v0, bd = Infinity;
      for (let i = 0; i <= nGrid; i++) {
        for (let j = 0; j <= nGrid; j++) {
          const u = u0 + (u1 - u0) * i / nGrid, v = v0 + (v1 - v0) * j / nGrid;
          const q = this._p(u, v);
          const d = (q[0] - px) ** 2 + (q[1] - py) ** 2 + (q[2] - pz) ** 2;
          if (d < bd) { bd = d; bu = u; bv = v; }
        }
      }
      // Gradient descent refinement
      let u = bu, v = bv, step = (u1 - u0) / nGrid;
      for (let k = 0; k < 60; k++) {
        const fu = this.du(u, v), fv = this.dv(u, v), q = this._p(u, v);
        const diff = [q[0] - px, q[1] - py, q[2] - pz];
        u -= step * dot3(diff, fu);
        v -= step * dot3(diff, fv);
        step *= 0.95;
      }
      return [u, v];
    }
  }

  global.Manifold = Manifold;
})(typeof window !== 'undefined' ? window : globalThis);
