/**
 * lie_group.js — Lie algebra & Lie group blueprints
 *
 * A Lie group G is a smooth manifold equipped with smooth group operations.
 * Its Lie algebra 𝔤 = T_e G is a vector space equipped with the Lie bracket.
 * The two are bridged by the exponential map exp : 𝔤 → G (with local inverse
 * log : G → 𝔤 near the identity).
 *
 * Structure split:
 *
 *   LieAlgebra (𝔤): vector space + bracket
 *     • add(ξ, η)       ξ + η
 *     • scale(ξ, s)     s · ξ
 *     • bracket(ξ, η)   [ξ, η]
 *
 *   LieGroup (G): smooth group, holds a reference to its algebra
 *     • mul(g, h)       g · h
 *     • inv(g)          g⁻¹
 *     • identity()      e ∈ G
 *     • exp(ξ)          algebra → group
 *     • log(g)          group → algebra
 *
 * Usage:
 *   import { LieAlgebra, LieGroup } from './lie_group.js';
 *   const alg = new LieAlgebra({ add, scale, bracket });
 *   const G   = new LieGroup({ alg, mul, id, inv, exp, log });
 */

export class LieAlgebra {
  /**
   * @param {object}   opts
   * @param {function} opts.add     (a, b) → a + b in 𝔤
   * @param {function} opts.scale   (a, s) → s · a in 𝔤
   * @param {function} opts.bracket (a, b) → [a, b] in 𝔤
   */
  constructor({ add, scale, bracket }) {
    this._add = add;
    this._scale = scale;
    this._bracket = bracket;
  }

  add(xi, zeta) { return this._add(xi, zeta); }
  scale(xi, s) { return this._scale(xi, s); }
  bracket(xi, zeta) { return this._bracket(xi, zeta); }
}

export class LieGroup {
  constructor({ alg, mul, id, inv, exp, log, dexp, dlog }) {
    this._alg = alg;
    this._mul = mul;
    this._id = id;
    this._inv = inv;
    this._exp = exp;
    this._log = log;

    this._dexp = dexp;
    this._dlog = dlog;
  }

  mul(g, h) { return this._mul(g, h); }
  inv(g) { return this._inv(g); }
  identity() { return this._id; }

  exp(xi) { return this._exp(xi); }
  dexp(xi) { return this._dexp(xi); }

  log(g) { return this._log(g); }
  dlog(g) { return this._dlog(g); }

  /** Get the location of tangent vector xi in the ambient space of G, i.e. the tangent space at the identity. */
  tangentAmbient(g, xi) { return this.mul(g, this.dexp(xi)); }

  // TODO: parallel transport of ξ ∈ T_pG to T_qG.
  transport(p, xi, q) { throw new Error('LieGroup.transport: not implemented'); }
}
