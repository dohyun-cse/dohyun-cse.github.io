# Numerical Experiments

We test the proposed scheme on the MBB beam and the cantilever
benchmark in 2D, using a SIMP penalization $r(t)=t^p$ with $p=3$, and
a PDE filter with length scale $\epsilon$.

<figure id="fig-mbb">
  <div style="background:#f4f0ec;border:1px dashed #c8bfb6;height:240px;
              display:grid;place-items:center;color:#9c8f82;
              max-width:560px;margin:0 auto;">
    figure placeholder — MBB beam result
  </div>
  <figcaption>
    Optimized density field on the MBB beam, $V_f = 0.5$,
    $\epsilon = 2h$. The convergence bound
    [[03-mirror-descent^thm-conv]] tightens with smaller $L$.
  </figcaption>
</figure>

## Settings

- Mesh: $300\times 100$ Q1 elements
- Step size: $\tau_k = 1/L$ with $L$ estimated by backtracking
- Stopping: $\norm{\rho^{k+1}-\rho^k}_\infty < 10^{-4}$

## Results

Compared to OC and MMA, the mirror-descent iterates remain smoothly in
the interior of $[0,1]$ for longer, then converge to a similar
compliance value with fewer "checkerboard" artefacts near the
boundary.
