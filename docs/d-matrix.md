# 1. D-matrix formulation

The D-matrix path is an exact, faster way to execute a limited set of Pikaia strategy updates. It is not an approximation mode. When a selected strategy cannot be represented exactly by the D-matrix update, it remains available through the ordinary iterative path and is rejected if `use_d_matrix=True` is requested.

This document explains the ordinary update first, derives the reduced update from it, then records the configurations for which the two paths have been tested to give the same results.

## 1.1. Scope

There are two independent formulation choices:

| Formulation | Purpose | D-matrix scope |
|---|---|---|
| `LEGACY` | The established Python-package equations and similarity scaling. | Eight individual gene strategies have verified D-matrix kernels. |
| `STANDARD` | The revised iterative equations, similarities, and normalisations. | Two supported model configurations: dominant gene plus no-op organism, and altruistic gene plus selfish organism (Alt-Sel). |

`PikaiaModel` owns the formulation for a complete run. It applies that formulation to every selected strategy and rejects a model whose strategies do not implement it. A user therefore cannot accidentally combine a LEGACY strategy equation with STANDARD similarities or normalisations.

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

The mathematical state is a normalised gene-fitness vector:

$$
\gamma_j \geq 0, \qquad \sum_{j=1}^{M}\gamma_j=1.
$$

The model uses a uniform vector when `initial_gene_fitness` is omitted, and every completed replicator step normalises the next vector. The derivations therefore assume that a user-supplied initial vector also lies on this simplex. The STANDARD dominant D-matrix path explicitly requires finite, non-negative values that sum to one because its row-constant construction depends on that invariant.

The organism fitness is the population row weighted by the current gene fitness:

$$
o_i=\sum_{k=1}^{M}X_{ik}\gamma_k.
$$

In this equation, $k$ is only a summation index over genes. The value $o_i$ changes when $\gamma$ changes, while $X$ does not change during a fit.

### 1.2.2. Similarity matrices

Strategies that use kinship receive one gene-similarity matrix $s^g$ and one organism-similarity matrix $s^o$. The formulation selects how the Euclidean distance is normalised.

| Formulation | Gene similarity $s^g_{jk}$ | Organism similarity $s^o_{il}$ |
|---|---|---|
| `LEGACY` | One minus distance divided by the largest observed gene distance. | One minus distance divided by the largest observed organism distance. |
| `STANDARD` | $1-\lVert X_{:j}-X_{:k}\rVert_2/N$ | $1-\lVert X_{i:}-X_{l:}\rVert_2/M$ |

$X_{:j}$ denotes column $j$ of $X$, and $X_{i:}$ denotes row $i$. The STANDARD definitions deliberately use the $N$ and $M$ divisors; they do not modify LEGACY-formulation results.

### 1.2.3. STANDARD normalisations

The historical formulations also use fixed quantities derived from $X$:

$$
g_{\mathrm{mpd}}=\operatorname{mean}_{j<k}|\bar X_j-\bar X_k|, \qquad h_{\mathrm{mpd}}=\operatorname{mean}_{i<l}|\bar X_{i:}-\bar X_{l:}|.
$$

Here $\bar X_j$ is the mean of column $j$, $\bar X_{i:}$ is the mean of row $i$, and `mpd` means mean pairwise absolute difference. The implementation names these values `gene_mean_pairwise_difference` and `harmonic_fitness_mean_pairwise_difference`. A STANDARD run raises an error if a required normalisation is zero, because division by zero would make the equation undefined.

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

The reduction requires the built-in `FixedMixStrategy` for both strategy families. Adaptive mixers such as `SelfConsistentMixStrategy`, custom mixers, and subclasses that override fixed-mixer behaviour may update coefficients from per-organism deltas that are unavailable after reduction to $D$ and $d$; the model therefore rejects them in D-matrix mode.

## 1.5. STANDARD dominant derivation

### 1.5.1. Direct signal

For STANDARD dominant gene, organism $i$ contributes the following signal to gene $j$:

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

The supported model configuration pairs `DominantGeneStrategy` with `NoneOrgStrategy`, both with fixed unit coefficients. The no-op organism contribution is zero, so the comparison isolates the dominant-gene equation. Because the proof uses $\sum_k\gamma_k=1$, the model rejects a STANDARD Dominant D-matrix run whose user-supplied `initial_gene_fitness` is non-finite, negative, or does not sum to one; every subsequent valid replicator step preserves the simplex invariant.

## 1.6. STANDARD Alt-Sel derivation

The old Pikaia branch exposed its reduced solver only when the model selected
one altruistic gene strategy and one selfish organism strategy. That restriction
identifies the configuration for which the reduced formulation was designed.
The old reduced-solver code itself was experimental: it contained a hard-coded
matrix adjustment and an interactive debugger breakpoint. The package therefore
does not copy that routine literally. It derives both matrix components from the
old branch's iterative equations and verifies the resulting update against the
iterative path.

### 1.6.1. Altruistic gene contribution

For STANDARD altruistic gene, the direct contribution is

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
`STANDARD` model configuration. That supported configuration contains one
strategy of each type with a fixed coefficient of one.

| STANDARD component | Iterative contribution reproduced | Role in the supported model configuration |
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
3. After exactly 1, 50, or 100 iterations, the comparison records the largest absolute differences between corresponding entries in both final fitness outputs.

For gene fitness, the reported difference at iteration $t$ is:

$$
E_{\gamma}(t)
=
\max_j
\left|
\gamma^{\mathrm{iterative}}_j(t)
-
\gamma^{\mathrm{D}}_j(t)
\right|.
$$

Here, $\gamma^{\mathrm{iterative}}_j(t)$ and $\gamma^{\mathrm{D}}_j(t)$ are the fitness values of gene $j$ after iteration $t$ in the iterative and D-matrix models, respectively. The maximum runs over every gene $j$.

For organism fitness, the reported difference is:

$$
E_o(t)
=
\max_i
\left|
o^{\mathrm{iterative}}_i(t)
-
o^{\mathrm{D}}_i(t)
\right|.
$$

Here, $o^{\mathrm{iterative}}_i(t)$ and $o^{\mathrm{D}}_i(t)$ are the fitness values of organism $i$ after iteration $t$. The maximum runs over every organism $i$.

Organism fitness is derived from gene fitness in both execution paths using the same population matrix:

$$
o_i(t)
=
\sum_j X_{ij}\gamma_j(t).
$$

Consequently, equal gene-fitness vectors imply equal organism-fitness vectors for the same $X$. The tests still compare organism fitness explicitly. This protects the public output against implementation errors such as storing the wrong iteration or using inconsistent state even though there is no separate organism-fitness evolution equation.

Each table cell is written as $E_{\gamma}(t) / E_o(t)$, or gene difference followed by organism difference. The acceptance criterion for both outputs is `rtol=1e-12` and `atol=1e-12`. A displayed zero means the corresponding floating-point vectors were bitwise equal on that fixture; a small value such as `1.11e-16` is ordinary floating-point roundoff and is far below the acceptance threshold. Each assessment also compares the median runtime of seven complete 100-iteration fits. These local timings include D-matrix construction, are rounded to two decimal places, and are illustrative rather than a portable performance guarantee.

For each `LEGACY` row, the named gene strategy is paired with `NoneOrgStrategy`. That no-op partner contributes zero, so the comparison isolates the named strategy. The STANDARD dominant row uses the same arrangement. The other two `STANDARD` rows refer to the same Alt-Sel run: one row assesses its altruistic-gene component and the other its selfish-organism component. Their numerical results are therefore intentionally identical. This does not mean the matrix components are algebraically inseparable; their isolated D-matrix models are outside the package's public compatibility contract.

### 1.7.2. Iterative vs. D-matrix path differences

| Formulation | Strategy | Partner | 1 iter. (G/O) | 50 iter. (G/O) | 100 iter. (G/O) | Result and timing (100 iter.) |
|---|---|---|---:|---:|---:|---|
| LEGACY | Dominant gene | None organism | 0 / 0 | 1.11e-16 / 1.11e-16 | 6.51e-18 / 0 | Matches. 3.16 ms iterative vs 1.11 ms D matrix; D matrix was 2.9 times faster. |
| LEGACY | Selfish gene | None organism | 0 / 0 | 5.55e-17 / 1.11e-16 | 2.78e-17 / 0 | Matches. 11.36 ms iterative vs 1.10 ms D matrix; D matrix was 10.3 times faster. |
| LEGACY | Kin-altruistic gene, default full neighbourhood | None organism | 0 / 0 | 1.11e-16 / 1.11e-16 | 5.55e-17 / 1.11e-16 | Matches. 12.77 ms iterative vs 1.12 ms D matrix; D matrix was 11.4 times faster. |
| LEGACY | Altruistic gene | None organism | 0 / 0 | 0 / 0 | 0 / 0 | Matches. 15.66 ms iterative vs 1.10 ms D matrix; D matrix was 14.2 times faster. |
| LEGACY | Sell hard gene | None organism | 1.39e-17 / 0 | 2.84e-16 / 2.22e-16 | 5.72e-17 / 5.55e-17 | Matches. 5.23 ms iterative vs 1.29 ms D matrix; D matrix was 4.0 times faster. |
| LEGACY | Sell uniform gene | None organism | 0 / 0 | 0 / 0 | 0 / 0 | Matches. 5.16 ms iterative vs 1.03 ms D matrix; D matrix was 5.0 times faster. |
| LEGACY | Sell easy gene | None organism | 0 / 0 | 0 / 0 | 0 / 0 | Matches. 5.29 ms iterative vs 1.04 ms D matrix; D matrix was 5.1 times faster. |
| LEGACY | Variance gene | None organism | 0 / 0 | 3.33e-16 / 2.22e-16 | 6.94e-17 / 1.11e-16 | Matches. 11.50 ms iterative vs 1.21 ms D matrix; D matrix was 9.5 times faster. |
| STANDARD | Dominant gene | None organism | 0 / 0 | 2.22e-16 / 1.11e-16 | 5.20e-18 / 0 | Matches. 3.53 ms iterative vs 1.14 ms D matrix; D matrix was 3.1 times faster. |
| STANDARD | Altruistic gene | Selfish organism, unit fixed coefficient | 5.55e-17 / 1.11e-16 | 5.55e-17 / 0 | 1.11e-16 / 2.22e-16 | Matches as part of Alt-Sel. 17.89 ms iterative vs 2.11 ms D matrix; D matrix was 8.5 times faster. |
| STANDARD | Selfish organism | Altruistic gene, unit fixed coefficient | 5.55e-17 / 1.11e-16 | 5.55e-17 / 0 | 1.11e-16 / 2.22e-16 | Matches as part of Alt-Sel. 17.89 ms iterative vs 2.11 ms D matrix; D matrix was 8.5 times faster. |

The LEGACY rows use a fixed four-organism, three-gene fixture with initial $\gamma=(0.6,0.3,0.1)$. The STANDARD dominant row and the two rows representing the same Alt-Sel run use a fixed five-organism, three-gene fixture with initial $\gamma=(0.4,0.35,0.25)$ and STANDARD similarity scaling. The LEGACY kernels, STANDARD dominant kernel, and complete STANDARD Alt-Sel reduction also run against three additional deterministic seven-organism, four-gene fixtures at all three iteration counts. Alt-Sel is checked with kin ranges of 1, 2, and 10, so the tests cover an empty relative set, a bounded neighbourhood, the full population, and clamping a requested range larger than $N$. The automated tests compare every gene- and organism-fitness history entry from the initial state through the requested final iteration. All comparisons satisfy the same `rtol=1e-12`, `atol=1e-12` criterion.

The assertions are implemented in `tests/unit/test_d_matrix_equivalence.py` and `tests/unit/test_standard_formulation.py`.

### 1.7.3. Historical iterative implementation cross-check

The STANDARD iterative path was also compared numerically with the actual `pikaia-gitlab-old/public/src/pikaia/alg.py` implementation rather than only with equations transcribed into tests. Three deterministic populations of shapes $5\times3$, $7\times4$, and $9\times5$ were checked at 1, 50, and 100 iterations. Dominant gene was paired with a no-op organism strategy; Alt-Sel was checked with $K=1$, $K=2$, $K=N$, and $K>N$.

The gene and organism similarity matrices and both mean-pairwise-difference normalisations matched the historical implementation within an absolute tolerance of $10^{-15}$. Dominant trajectories were bitwise equal. The largest Alt-Sel trajectory difference was $3.33\times10^{-16}$, well below the `rtol=1e-12`, `atol=1e-12` acceptance criterion. This cross-check concerns the historical iterative equations; the old experimental reduced solver is not used because, as explained in Section 1.6, it contains a hard-coded matrix adjustment and an interactive debugger breakpoint.

## 1.8. Safe use

### 1.8.1. Selecting a formulation

Use `LEGACY` for the original Python-package equations. Use `STANDARD` for the revised equations, similarities, normalisations, and exact STANDARD D-matrix configurations. Do not mix formulations in one model. A STANDARD fit requires an explicit `max_iter`; the direct analytical Dominant+Balanced fixed point selected by `max_iter=None` implements only `LEGACY` and is therefore rejected under `STANDARD`. The old enum aliases and serialized values, `ORIGINAL` and `MATH_PAPER`, remain accepted for migration but are deprecated.

### 1.8.2. Selecting the execution path

Use `use_d_matrix=True` only for a configuration represented in the comparison table. In `LEGACY`, supported strategies may be combined with fixed coefficients because their direct deltas and D contributions are both added linearly. In `STANDARD`, use dominant gene with a no-op organism strategy or the unmixed Alt-Sel pair. All other supported STANDARD strategy arrangements remain usable with `use_d_matrix=False`; strategies that implement only `LEGACY` are rejected when the model selects `STANDARD`.
