# A Mirror-Descent Approach

## The update rule

With $\phi$ chosen as a constraint-aware entropy on $[0,1]$, one step
of mirror descent reads

$$
  \rho^{k+1} = \argmin_{\rho\in C}\, \Bigl\{ \langle g^k, \rho\rangle
    + \frac{1}{\tau_k} D_\phi(\rho, \rho^k) \Bigr\},
$$

where $g^k = \nabla F(\rho^k)$ and $\tau_k$ is the step size. The key
property is that the box constraints $0\le\rho\le 1$ are
<span class="green">automatically satisfied</span> by the iterates,
with no projection step.

> [!Example] Idea
> Replace the Euclidean geometry of projected gradient with a geometry
> that "knows about" the box constraints. The iterates stay in the
> relative interior of $[0,1]$ throughout.

## Convergence

> [!Lemma^lem-descent] descent inequality
> Let $\tau_k = 1/L$ and assume $F$ is $L$-smooth relative to $\phi$.
> Then for every $k\ge 0$,
> \begin{equation}\label{eq:descent}
>   F(\rho^{k+1}) \le F(\rho^k) - \frac{1}{L}\,D_\phi(\rho^{k+1},\rho^k).
> \end{equation}

> [!Proof]
> Apply relative smoothness to the update rule and use
> [[02-bregman^thm-three-point]].

> [!Theorem^thm-conv]
> Assume $F$ is $L$-smooth relative to $\phi$. Then mirror descent with
> $\tau_k = 1/L$ satisfies
> \begin{equation}\label{eq:rate}
>   F(\rho^K) - F^\star \le \frac{L\, D_\phi(\rho^\star, \rho^0)}{K}
> \end{equation}
> for all $K\ge 1$.

> [!Proof]
> Apply [[02-bregman^thm-three-point]] and telescope; cf.
> [[@bib-beck]].
