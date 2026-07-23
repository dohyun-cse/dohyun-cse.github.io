# Introduction

Topology optimization seeks the best material layout in a design
domain $\Omega \subset \mathbb{R}^d$ subject to physics and
manufacturing constraints. In the density-based formulation, the
design variable is a function $\rho:\Omega\to[0,1]$ with
$\rho=1$ denoting **material** and $\rho=0$ denoting **void**.

A standard compliance-minimization problem reads

$$
\begin{aligned}
  \min_{\rho}\ & F(\rho) = \int_\Omega u\cdot f\,\dd x \\
  \text{s.t.}\ & -\nabla\!\cdot\!\bigl(\mathsf{C}(r(\tilde\rho)):\varepsilon(u)\bigr) = f, \\
               & -\nabla\!\cdot\!(\epsilon^2\nabla\tilde\rho)+\tilde\rho = \rho, \\
               & \int_\Omega \rho\,\dd x = V_f|\Omega|, \quad \RED{0\le\rho\le 1.}
\end{aligned}
$$

## Motivation

Projected-gradient methods enforce $0\le\rho\le 1$ by clipping at
every step, which makes the iterates kink against the boundary and
slows convergence. We argue that a different geometry — induced by a
Bregman divergence — handles these constraints more gracefully.

## Outline

Chapter 2 reviews the Bregman divergence (see
[[02-bregman^def-bregman]]) and its projection
([[02-bregman^thm-three-point]]). Chapter 3 introduces the proposed
mirror-descent scheme with a convergence bound
([[03-mirror-descent^thm-conv]]). Chapter 4 reports numerical
experiments — see [[04-experiments^fig-mbb]] — on benchmarks from
[[@bib-sigmund]] and [[@bib-bendsoe]].
