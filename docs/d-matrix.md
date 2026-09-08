# D-matrix formulation

This is the source of truth for the D-matrix fast path: its update equation,
derivation procedure, and the exactness status of every built-in strategy.

## Contract

Let `X` be a fixed population with `N` organisms and `M` genes, and let
`gamma` be the normalised gene-fitness vector. The regular path first sums all
per-organism strategy contributions into `Delta(gamma)`, then applies:

\[
\gamma_j^+ = \frac{\gamma_j[1 + \Delta_j(\gamma)]}
{\sum_r \gamma_r[1 + \Delta_r(\gamma)]}.
\]

The D-matrix path is an exact acceleration only when the *already summed*
delta has the following population-static form:

\[
\Delta_j(\gamma) = d_j + \gamma_j\sum_k D_{jk}\gamma_k.
\]

It precomputes `(D, d)` once and performs:

\[
\gamma_j^+ =
\frac{\gamma_j[1 + d_j + \gamma_j(D\gamma)_j]}
{\sum_r \gamma_r[1 + d_r + \gamma_r(D\gamma)_r]}.
\]

In code this is `gamma * (1 + d + gamma * (D @ gamma))`, then
normalisation. The elementwise second `gamma` is essential: `D @ gamma + d`
by itself is not the strategy delta used by this implementation.

## Derivation procedure

For each candidate strategy:

1. Write the iterative delta for organism `i` and gene `j`.
2. Sum it over every organism index.
3. Substitute `o_i = sum_k X_ik gamma_k` wherever organism fitness appears.
4. Factor the result into `d_j + gamma_j * sum_k D_jk * gamma_k`.
5. Test the equality for multiple non-uniform `gamma` vectors and compare one
   full replicator step in both paths.

The common exact patterns are:

| Summed iterative delta | Kernel |
|---|---|
| `d_j` | `(None, d)` |
| `gamma_j^2 * a_j` | `(diag(a), None)` |
| `gamma_j * sum_k A_jk * gamma_k` | `(A, None)` |

A state-dependent denominator, for example `1 / o_i`, normally prevents a
static kernel. Freezing or dropping it is an approximation, not an equivalent
acceleration.

## Gene-strategy derivations

### Original dominant: exact

The iterative contribution is:

\[
\Delta_{ij}=\frac4N\gamma_j^2(X_{ij}-\tfrac12).
\]

After summing organisms,
`Delta_j = 4 * gamma_j^2 * (X_bar_j - 0.5)`. Therefore:

\[
D=\operatorname{diag}(4(\bar X-\tfrac12)), \qquad d=0.
\]

### Math-paper dominant: iterative-only

The revised formula is
`Delta_j = gamma_j * (X_bar_j - 0.5)`. It has one power of `gamma_j`, whereas
the static contract provides either a fixed `d` or two powers through
`gamma_j * (D gamma)_j`. It cannot be represented for arbitrary evolving
`gamma`; `DominantGeneStrategy(MATH_PAPER)` intentionally returns no kernel.
Selecting it with `use_d_matrix=True` raises an error rather than silently
omitting it from a mixed D-matrix run.

### Altruistic: exact in both formulations

Define the fixed cross-gene statistic:

\[
K_{jk}=s^g_{jk}\frac1N\sum_i
(X_{ij}-\tfrac12)(X_{ik}-X_{ij}), \qquad K_{jj}=0.
\]

The original total is
`Delta_j = gamma_j * sum_k (16 / M) * K_jk * gamma_k`, hence
`D = (16 / M) * K`.

The math-paper total is
`Delta_j = gamma_j * sum_k (K_jk / g_mpd) * gamma_k`, hence `D = K / g_mpd`.
Here `g_mpd` is `gene_mean_pairwise_difference`, the mean absolute pairwise
difference among column means of `X`. It is fixed for a population, calculated
once, Pydantic-validated, and required to be positive.

### Selfish and kin-altruistic genes

`SelfishGeneStrategy` is the negative of the original altruistic kernel, so
its mapping is exact. `KinAltruisticGeneStrategy` is exact with the full kin
range when its masked interactions agree between `__call__` and `kernel`.
Do not claim bounded-kin equivalence without a direct multi-`gamma` test.

### Static-score and sell genes

Variance, entropy, orthogonality, partial-correlation, and redundancy
strategies use static per-gene scores during one fit. Their summed deltas have
the diagonal quadratic form and therefore exact diagonal `D` kernels. The
three sell strategies sum to population-static delta vectors and use exact
`d` kernels.

## Organism-strategy derivations

### Balanced: reduced dynamics, not an exact kernel

Summing its iterative expression gives:

\[
\Delta_j=-2\bar X_j\gamma_j+
\frac2M\sum_k\bar X_k\gamma_k.
\]

The current row-constant kernel, `D[j,k] = -2 * X_bar_j`, represents the first
term because `sum_k gamma_k = 1`. It drops the second term: that term is the
same for every gene but changes with `gamma`. This preserves the interior
fixed-point equality condition, but changes the trajectory and can change
finite-iteration results.

### Original selfish, altruistic, and kin-selfish: approximations

Each contains the state-dependent factor:

\[
\frac{X_{ij}\gamma_j}{o_i}-\frac1M,
\qquad o_i=\sum_kX_{ik}\gamma_k.
\]

The `1 / o_i` term changes as `gamma` changes, so it cannot be encoded in a
population-static `(D, d)`. Their current organism kernels retain a useful
outer-product component but omit this rational dependence and its baseline.
They are accelerations of reduced dynamics, not exact iterative equivalents.

### Math-paper selfish organism: exact

The math-paper formula uses `X_ij * gamma_j` rather than the expression above.
Substituting `o_i - o_l = sum_k (X_ik - X_lk) * gamma_k` leaves a bilinear,
population-static kernel:

\[
D_{jk}=-\frac{2}{N h_{mpd}}\sum_i\frac{X_{ij}}{R_i}
\sum_{l\in R_i}s^o_{il}(X_{ik}-X_{lk}).
\]

`R_i` is the selected kin range using the historical denominator convention.
`h_mpd` is `harmonic_fitness_mean_pairwise_difference`, the mean absolute
pairwise difference among row means of `X`. The math-paper kernel is exact and
is regression-tested against the iterative path.

## Compatibility ledger

| Strategy | Status |
|---|---|
| Original dominant, altruistic, selfish gene | Exact |
| Math-paper altruistic gene | Exact |
| Math-paper dominant gene | Iterative-only |
| Full-range kin-altruistic gene | Exact if directly equivalence-tested |
| Variance, entropy, orthogonality, partial-correlation, redundancy, sell genes | Exact while their scores are fixed |
| Balanced organism | Reduced dynamics |
| Original selfish, altruistic, kin-selfish organism | Approximation |
| Math-paper selfish organism | Exact |
| Buy strategies and strategies without `kernel()` | Unsupported |

## Mixing and safety

Fixed mixing combines kernels linearly. Self-consistent mixing recombines the
precomputed per-strategy kernels as coefficients change. Both are exact only
when every active strategy has an exact kernel.

The D path runs when at least one selected strategy exposes a kernel. Any
selected strategy without one contributes zero in that path. Therefore a mixed
run containing a non-kernel strategy is not equivalent to the iterative run.

For every new kernel, test the identity
`sum_i Delta_ij == d_j + gamma_j * (D @ gamma)_j` for several non-uniform
fitness vectors, then test a complete update step. Matching only a final
convergence result is not evidence of an algebraically correct kernel.
