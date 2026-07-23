# The Bregman Divergence

> [!Definition^def-bregman] Bregman divergence
> Given a strictly convex, differentiable function
> $\phi:\mathcal{X}\to\mathbb{R}$, the **Bregman divergence**
> associated with $\phi$ is
> \begin{equation}\label{eq:bregman}
>   D_\phi(x,y) = \phi(x) - \phi(y) - \langle \nabla\phi(y),\, x - y\rangle.
> \end{equation}

Intuitively, the quantity in \eqref{eq:bregman} is the vertical gap
between $\phi(x)$ and the tangent line to $\phi$ at $y$. It is
nonnegative, vanishes iff $x=y$, but is
<span class="red">not symmetric</span> in general; for a survey see
[[@bib-bauschke]].

## Examples

<div class="cols c2">
<div>

**Squared Euclidean.** Take $\phi(x)=\tfrac12\norm{x}^2$. Then
$D_\phi(x,y)=\tfrac12\norm{x-y}^2$ and mirror descent reduces to
vanilla gradient descent.

</div>
<div>

**Negative entropy.** On the simplex,
$\phi(x)=\sum_i x_i\log x_i$. Then mirror descent becomes the
multiplicative-weights / exponentiated-gradient update.

</div>
</div>

## Bregman projection

> [!Block] Bregman projection
> For a closed convex set $C\subset\mathcal{X}$, the
> $D_\phi$-projection of $y$ onto $C$ is
> $$\Pi^\phi_C(y) = \argmin_{x\in C}\, D_\phi(x,y).$$

> [!Theorem^thm-three-point] three-point identity
> Let $\phi$ be Legendre on the interior of $C$, and let
> $C \subset \operatorname{int}(\operatorname{dom}\phi)$. Then
> $\Pi^\phi_C(y)$ exists, is unique, and satisfies
> \begin{equation}\label{eq:three-point}
>   D_\phi(x,y) = D_\phi(x,\Pi^\phi_C(y)) + D_\phi(\Pi^\phi_C(y),y)
> \end{equation}
> for all $x\in C$.

> [!Proof]
> Standard; see [[@bib-bauschke]].
