# 1. D-matrix formulation

The D-matrix path is an exact, faster way to execute a limited set of Pikaia strategy updates. It is not an approximation mode. When a selected strategy cannot be represented exactly by the D-matrix update, it remains available through the ordinary iterative path and is rejected if `use_d_matrix=True` is requested.

This document explains the ordinary update first, derives the reduced update from it, then records the configurations for which the two paths have been tested to give the same results.

## 1.1. Scope

There are two independent formulation choices:

| Formulation | Purpose | D-matrix scope |
|---|---|---|
| `ORIGINAL` | The established Python-package equations and similarity scaling. | Eight individual gene strategies have verified D-matrix kernels. |
| `MATH_PAPER` | Compatibility with the revised iterative equations in the historical Pikaia branch. | Two supported model configurations: dominant gene plus no-op organism, and altruistic gene plus selfish organism (Alt-Sel). |

`PikaiaModel` owns the formulation for a complete run. It applies that formulation to every selected strategy and rejects a model whose strategies do not implement it. A user therefore cannot accidentally combine an original strategy equation with math-paper similarities or normalisations.

## 1.2. Population, fitness, and symbols

### 1.2.1. Fixed population

Let $X \in \mathbb{R}^{N\times M}$ be the fixed population matrix. Each row is an organism and each column is a gene. The entry $X_{ij}$ is the value of gene $j$ in organism $i$.

| Symbol | Meaning | Shape or range |
|---|---|---|
| $N$ | Number of organisms. | Positive integer. |
| $M$ | Number of genes. | Positive integer. |
| $X$ | Fixed population matrix. | $N\times M$. |
| $X_{ij}$ | Value of gene $j$ in organism $i$. | Usually in $[0,1]$. |
| $\gamma$ | Current gene-fitness vector. | Length $M$, normalised. |
| $\gamma_j$ | Fitness assigned to gene $j$. | $\gamma_j\geq0$. |
| $o_i$ | Current fitness of organism $i$. | Scalar. |

The gene-fitness vector is always normalised:

$$
\gamma_j \geq 0, \qquad \sum_{j=1}^{M}\gamma_j=1.
$$

The organism fitness is the population row weighted by the current gene fitness:

$$
o_i=\sum_{k=1}^{M}X_{ik}\gamma_k.
$$

In this equation, $k$ is only a summation index over genes. The value $o_i$ changes when $\gamma$ changes, while $X$ does not change during a fit.

### 1.2.2. Similarity matrices

Strategies that use kinship receive one gene-similarity matrix $s^g$ and one organism-similarity matrix $s^o$. The formulation selects how the Euclidean distance is normalised.

| Formulation | Gene similarity $s^g_{jk}$ | Organism similarity $s^o_{il}$ |
|---|---|---|
| `ORIGINAL` | One minus distance divided by the largest observed gene distance. | One minus distance divided by the largest observed organism distance. |
| `MATH_PAPER` | $1-\lVert X_{:j}-X_{:k}\rVert_2/N$ | $1-\lVert X_{i:}-X_{l:}\rVert_2/M$ |

$X_{:j}$ denotes column $j$ of $X$, and $X_{i:}$ denotes row $i$. The math-paper definitions deliberately reproduce the historical $N$ and $M$ divisors; they do not modify original-formulation results.

### 1.2.3. Math-paper normalisations

The historical formulations also use fixed quantities derived from $X$:

$$
g_{\mathrm{mpd}}=\operatorname{mean}_{j<k}|\bar X_j-\bar X_k|, \qquad h_{\mathrm{mpd}}=\operatorname{mean}_{i<l}|\bar X_{i:}-\bar X_{l:}|.
$$

Here $\bar X_j$ is the mean of column $j$, $\bar X_{i:}$ is the mean of row $i$, and `mpd` means mean pairwise absolute difference. The implementation names these values `gene_mean_pairwise_difference` and `harmonic_fitness_mean_pairwise_difference`. A math-paper run raises an error if a required normalisation is zero, because division by zero would make the equation undefined.

## 1.3. Ordinary iterative update

### 1.3.1. Per-strategy contributions

For a strategy $q$, let $\delta^{(q)}_{ij}(\gamma)$ denote its contribution for organism $i$ and gene $j$. A gene strategy returns a scalar contribution for each $(i,j)$ pair; an organism strategy returns one contribution per gene for organism $i$.

If fixed mixing is active, each strategy has a fixed non-negative coefficient $w_q$ and the coefficients in each strategy family sum to one. The model first combines the strategy contributions using those coefficients and then sums over organisms:

$$
\Delta_j(\gamma)=\sum_{i=1}^{N}\sum_q w_q\,\delta^{(q)}_{ij}(\gamma).
$$

$\Delta_j(\gamma)$ is therefore the total update signal for gene $j$ in one iteration. This is the quantity a D-matrix kernel must reproduce exactly.

### 1.3.2. Replicator step

The ordinary iterative path applies the total signal and normalises the result:

$$
\gamma_j^+=\frac{\gamma_j[1+\Delta_j(\gamma)]}{\sum_{r=1}^{M}\gamma_r[1+\Delta_r(\gamma)]}.
$$

$\gamma_j^+$ is the next gene-fitness value. The denominator is the same expression summed over every gene $r$; it keeps the next vector normalised. This calculation evaluates each selected strategy directly for every organism and gene.

## 1.4. D-matrix reduction

### 1.4.1. Required form

A strategy is eligible for the reduced path only when its total iterative signal can be written as a population-static linear term plus a bilinear term:

$$
\Delta_j(\gamma)=d_j+\gamma_j\sum_{k=1}^{M}D_{jk}\gamma_k.
$$

| Component | Meaning |
|---|---|
| $d_j$ | A fixed, population-derived linear contribution for gene $j$. |
| $D$ | An $M\times M$ matrix whose entries are fixed for the current population. |
| $D_{jk}$ | Contribution coupling output gene $j$ to input gene $k$. |
| $\gamma_j\gamma_k$ | The two current-fitness factors represented by the bilinear term. |

Substituting this expression for $\Delta_j(\gamma)$ into the ordinary replicator step gives the D-matrix update:

$$
\gamma^+=\frac{\gamma\odot[1+d+\gamma\odot(D\gamma)]}{\sum_{r=1}^{M}\gamma_r[1+d_r+\gamma_r(D\gamma)_r]}.
$$

$\odot$ means elementwise multiplication. $D\gamma$ is ordinary matrix-vector multiplication. The expression is fast because $D$ and $d$ are computed once before the iteration loop instead of re-evaluating every strategy for every organism at every step.

### 1.4.2. Combination rules

The model runs one path per fit. It cannot calculate one strategy through $D$ while calculating another strategy iteratively. Consequently, every selected non-no-op strategy must provide an exact D contribution. `NoneGeneStrategy` and `NoneOrgStrategy` support both formulations because their contribution is exactly zero under either set of equations. A no-op can therefore isolate a supported strategy without changing its result.

The reduction also requires fixed mixing coefficients. `SelfConsistentMixStrategy` updates coefficients from per-organism deltas, which are unavailable after reduction to $D$ and $d$, so self-consistent mixing is iterative-only.

## 1.5. Math-paper dominant derivation

### 1.5.1. Direct signal

For math-paper dominant gene, organism $i$ contributes the following signal to gene $j$:

$$
\delta^{\mathrm{dom}}_{ij}=\frac{1}{N}\gamma_j\left(X_{ij}-\frac12\right).
$$

Here $1/N$ averages the contribution over the $N$ organisms, $\gamma_j$ is the current fitness of gene $j$, and $X_{ij}-1/2$ is organism $i$'s centred expression of that gene. Summing over all organisms gives

$$
\Delta^{\mathrm{dom}}_j(\gamma)=\sum_{i=1}^{N}\delta^{\mathrm{dom}}_{ij}=\gamma_j\left(\bar X_j-\frac12\right),
$$

where $\bar X_j=N^{-1}\sum_i X_{ij}$ is the mean expression of gene $j$ in the fixed population.

### 1.5.2. Row-constant D matrix

Define $a_j=\bar X_j-1/2$ and assign that value to every entry in row $j$:

$$
D^{\mathrm{dom}}_{jk}=a_j \qquad \text{for every } k.
$$

Although the required D-matrix form contains two appearances of $\gamma$, this row-constant construction uses the normalisation $\sum_k\gamma_k=1$:

$$
\gamma_j(D^{\mathrm{dom}}\gamma)_j
=\gamma_j\sum_{k=1}^{M}a_j\gamma_k
=\gamma_j a_j\sum_{k=1}^{M}\gamma_k
=\gamma_j a_j
=\Delta^{\mathrm{dom}}_j(\gamma).
$$

The reduction is therefore exact. The earlier claim that a signal linear in $\gamma_j$ could not satisfy the D-matrix contract was incorrect: the gene-fitness simplex supplies the constant factor through $\sum_k\gamma_k=1$.

The supported model configuration pairs `DominantGeneStrategy` with `NoneOrgStrategy`, both with fixed unit coefficients. The no-op organism contribution is zero, so the comparison isolates the dominant-gene equation. Because the proof uses $\sum_k\gamma_k=1$, the model rejects a math-paper Dominant D-matrix run whose user-supplied `initial_gene_fitness` does not sum to one; every subsequent replicator step preserves this invariant.

## 1.6. Math-paper Alt-Sel derivation

The old Pikaia branch exposed its reduced solver only when the model selected
one altruistic gene strategy and one selfish organism strategy. That restriction
identifies the configuration for which the reduced formulation was designed.
The old reduced-solver code itself was experimental: it contained a hard-coded
matrix adjustment and an interactive debugger breakpoint. The package therefore
does not copy that routine literally. It derives both matrix components from the
old branch's iterative equations and verifies the resulting update against the
iterative path.

### 1.6.1. Altruistic gene contribution

For math-paper altruistic gene, the direct contribution is

$$
\delta^G_{ij}=\frac{1}{N g_{\mathrm{mpd}}}\sum_{k\ne j}s^g_{jk}\gamma_j\left(X_{ij}-\frac12\right)\gamma_k(X_{ik}-X_{ij}).
$$

The outer factor $1/(N g_{\mathrm{mpd}})$ normalises over organisms and the historical gene-difference scale. The sum visits every other gene $k$. $s^g_{jk}$ weights that gene by similarity to $j$; $X_{ij}-1/2$ centres the focal gene; and $X_{ik}-X_{ij}$ measures the other gene relative to the focal gene in the same organism.

After summing the direct contribution over $i$, the factors $\gamma_j\gamma_k$ have exactly the required bilinear form. The corresponding matrix entries are

$$
D^G_{jk}=\frac{s^g_{jk}}{g_{\mathrm{mpd}}}\frac{1}{N}\sum_{i=1}^{N}\left(X_{ij}-\frac12\right)(X_{ik}-X_{ij}), \qquad D^G_{jj}=0.
$$

The diagonal is zero because the direct equation excludes $k=j$.

### 1.6.2. Selfish organism contribution

For each organism $i$, let $\mathcal{R}_i$ be the first $K$ most similar organisms, where $K=\min(\text{requested kin range},N)$. The organism itself is removed from the sum but remains in the denominator $K$, matching the historical implementation. Its direct contribution is

$$
\delta^O_{ij}=-\frac{2}{N K h_{\mathrm{mpd}}}X_{ij}\gamma_j\sum_{l\in\mathcal{R}_i\setminus\{i\}}s^o_{il}(o_i-o_l).
$$

$l$ indexes relatives of organism $i$. $s^o_{il}$ weights a relative by organism similarity. The difference $o_i-o_l$ is not fixed, but substituting the organism-fitness definition gives

$$
o_i-o_l=\sum_{k=1}^{M}(X_{ik}-X_{lk})\gamma_k.
$$

The result is again bilinear in $\gamma_j$ and $\gamma_k$:

$$
D^O_{jk}=-\frac{2}{N K h_{\mathrm{mpd}}}\sum_{i=1}^{N}X_{ij}\sum_{l\in\mathcal{R}_i\setminus\{i\}}s^o_{il}(X_{ik}-X_{lk}).
$$

Each component is independently representable in D-matrix form: $D^G$
reproduces the altruistic-gene contribution and $D^O$ reproduces the
selfish-organism contribution. Neither component needs the other for its
algebra to be valid. For compatibility with the scope of the old reduced
solver, however, the package exposes only their sum, $D=D^G+D^O$, as a
`MATH_PAPER` model configuration. That supported configuration contains one
strategy of each type with a fixed coefficient of one.

| Math-paper component | Iterative contribution reproduced | Role in the supported model configuration |
|---|---|---|
| Altruistic gene, $D^G$ | The sum of $\delta^G_{ij}$ over all organisms $i$. | Gene component of Alt-Sel. |
| Selfish organism, $D^O$ | The sum of $\delta^O_{ij}$ over all organisms $i$. | Organism component of Alt-Sel. |

This distinction also determines how to read the end-to-end comparison below.
The table has one row per strategy, while the partner column identifies the
complete public model configuration accepted by `PikaiaModel`.

## 1.7. Compatibility comparison

### 1.7.1. What is being compared

Every row below compares two independently constructed models with the same fixed population, initial gene fitness, strategy configuration, fixed coefficients, and number of iterations:

1. The iterative model uses `use_d_matrix=False` and evaluates the direct strategy equations.
2. The reduced model uses `use_d_matrix=True` and evaluates the precomputed $D$ matrix and $d$ vector.
3. After exactly 1, 50, or 100 iterations, the comparison records the largest absolute difference between corresponding final gene-fitness entries: $\max_j|\gamma^{\mathrm{iterative}}_j-\gamma^{\mathrm{D}}_j|$.

The acceptance criterion is `rtol=1e-12` and `atol=1e-12`. A displayed zero means the two floating-point vectors were bitwise equal on that fixture; a small value such as `1.11e-16` is ordinary floating-point roundoff and is far below the acceptance threshold. Each assessment also compares the median runtime of seven complete 100-iteration fits. These local timings include D-matrix construction, are rounded to two decimal places, and are illustrative rather than a portable performance guarantee.

For each `ORIGINAL` row, the named gene strategy is paired with `NoneOrgStrategy`. That no-op partner contributes zero, so the comparison isolates the named strategy. The math-paper dominant row uses the same arrangement. The other two `MATH_PAPER` rows refer to the same Alt-Sel run: one row assesses its altruistic-gene component and the other its selfish-organism component. Their numerical results are therefore intentionally identical. This does not mean the matrix components are algebraically inseparable; their isolated D-matrix models are outside the package's public compatibility contract.

### 1.7.2. Iterative vs. D-matrix path differences

| Formulation | Strategy | Partner | Max. absolute difference (1 iter.) | Max. absolute difference (50 iter.) | Max. absolute difference (100 iter.) | Result and timing (100 iter.) |
|---|---|---|---:|---:|---:|---|
| ORIGINAL | Dominant gene | None organism | 0 | 1.11e-16 | 6.51e-18 | Matches. 2.33 ms iterative vs 0.69 ms D matrix; D matrix was 3.4 times faster. |
| ORIGINAL | Selfish gene | None organism | 0 | 5.55e-17 | 2.78e-17 | Matches. 8.32 ms iterative vs 0.67 ms D matrix; D matrix was 12.5 times faster. |
| ORIGINAL | Kin-altruistic gene, default full neighbourhood | None organism | 0 | 1.11e-16 | 5.55e-17 | Matches. 9.41 ms iterative vs 0.69 ms D matrix; D matrix was 13.6 times faster. |
| ORIGINAL | Altruistic gene | None organism | 0 | 0 | 0 | Matches. 11.65 ms iterative vs 0.68 ms D matrix; D matrix was 17.1 times faster. |
| ORIGINAL | Sell hard gene | None organism | 1.39e-17 | 2.84e-16 | 5.72e-17 | Matches. 4.04 ms iterative vs 0.61 ms D matrix; D matrix was 6.6 times faster. |
| ORIGINAL | Sell uniform gene | None organism | 0 | 0 | 0 | Matches. 3.90 ms iterative vs 0.61 ms D matrix; D matrix was 6.4 times faster. |
| ORIGINAL | Sell easy gene | None organism | 0 | 0 | 0 | Matches. 4.02 ms iterative vs 0.61 ms D matrix; D matrix was 6.6 times faster. |
| ORIGINAL | Variance gene | None organism | 0 | 3.33e-16 | 6.94e-17 | Matches. 8.58 ms iterative vs 0.72 ms D matrix; D matrix was 12.0 times faster. |
| MATH_PAPER | Dominant gene | None organism | 0 | 2.22e-16 | 5.20e-18 | Matches. 2.59 ms iterative vs 0.69 ms D matrix; D matrix was 3.8 times faster. |
| MATH_PAPER | Altruistic gene | Selfish organism, unit fixed coefficient | 5.55e-17 | 5.55e-17 | 1.11e-16 | Matches as part of Alt-Sel. 13.43 ms iterative vs 0.74 ms D matrix; D matrix was 18.1 times faster. |
| MATH_PAPER | Selfish organism | Altruistic gene, unit fixed coefficient | 5.55e-17 | 5.55e-17 | 1.11e-16 | Matches as part of Alt-Sel. 13.43 ms iterative vs 0.74 ms D matrix; D matrix was 18.1 times faster. |

The original rows use a fixed four-organism, three-gene fixture with initial $\gamma=(0.6,0.3,0.1)$. The math-paper dominant row and the two rows representing the same Alt-Sel run use a fixed five-organism, three-gene fixture with initial $\gamma=(0.4,0.35,0.25)$ and math-paper similarity scaling. The original kernels and the math-paper dominant kernel also run against three additional deterministic seven-organism, four-gene fixtures at all three iteration counts. All satisfy the same `rtol=1e-12`, `atol=1e-12` criterion; the original kernels' largest observed discrepancy is $3.4\times10^{-16}$.

The assertions are implemented in `tests/unit/test_d_matrix_equivalence.py` and `tests/unit/test_math_paper_formulation.py`.

## 1.8. Safe use

### 1.8.1. Selecting a formulation

Use `ORIGINAL` for the Python-package equations. Use `MATH_PAPER` to reproduce the revised historical equations, similarities, normalisations, and exact math-paper D-matrix configurations. Do not mix formulations in one model. A math-paper fit requires an explicit `max_iter`; the direct analytical Dominant+Balanced fixed point selected by `max_iter=None` implements only `ORIGINAL` and is therefore rejected under `MATH_PAPER`.

### 1.8.2. Selecting the execution path

Use `use_d_matrix=True` only for a configuration represented in the comparison table. In `ORIGINAL`, supported strategies may be combined with fixed coefficients because their direct deltas and D contributions are both added linearly. In `MATH_PAPER`, use dominant gene with a no-op organism strategy or the unmixed Alt-Sel pair. All other supported math-paper strategy arrangements remain usable with `use_d_matrix=False`; strategies that implement only `ORIGINAL` are rejected when the model selects `MATH_PAPER`.
