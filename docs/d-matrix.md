# 1. D-matrix formulation

The D-matrix path is an exact, faster way to execute a limited set of Pikaia strategy updates. It is not an approximation mode. When a selected strategy cannot be represented exactly by the D-matrix update, it remains available through the ordinary iterative path and is rejected if `use_d_matrix=True` is requested.

This document explains the ordinary update first, derives the reduced update from it, then records the configurations for which the two paths have been tested to give the same results.

## 1.1. Scope

There are two independent formulation choices:

| Formulation | Purpose | D-matrix scope |
|---|---|---|
| `ORIGINAL` | The established Python-package equations and similarity scaling. | Eight individual gene strategies have verified D-matrix kernels. |
| `MATH_PAPER` | Compatibility with the revised iterative equations in the historical Pikaia branch. | One supported model configuration: altruistic gene plus selfish organism (Alt-Sel). |

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

The model runs one path per fit. It cannot calculate one strategy through $D$
while calculating another strategy iteratively. Consequently, every selected
non-no-op strategy must provide an exact D contribution. In `ORIGINAL`,
`NoneGeneStrategy` and `NoneOrgStrategy` can accompany a supported strategy
because each contributes exactly zero. They do not declare `MATH_PAPER`
support, so they cannot be used to construct an isolated math-paper model; the
public math-paper D-matrix configuration is Alt-Sel.

The reduction also requires fixed mixing coefficients. `SelfConsistentMixStrategy` updates coefficients from per-organism deltas, which are unavailable after reduction to $D$ and $d$, so self-consistent mixing is iterative-only.

## 1.5. Math-paper Alt-Sel derivation

The old Pikaia branch exposed its reduced solver only when the model selected
one altruistic gene strategy and one selfish organism strategy. That restriction
identifies the configuration for which the reduced formulation was designed.
The old reduced-solver code itself was experimental: it contained a hard-coded
matrix adjustment and an interactive debugger breakpoint. The package therefore
does not copy that routine literally. It derives both matrix components from the
old branch's iterative equations and verifies the resulting update against the
iterative path.

### 1.5.1. Altruistic gene contribution

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

### 1.5.2. Selfish organism contribution

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

This distinction explains why the end-to-end comparison below has one Alt-Sel
row rather than two isolated strategy rows. The row tests a public model
configuration accepted by `PikaiaModel`, while the equations above establish
what each strategy contributes to that configuration.

## 1.6. Compatibility comparison

### 1.6.1. What is being compared

Every row below compares two independently constructed models with the same fixed population, initial gene fitness, strategy configuration, fixed coefficients, and number of iterations:

1. The iterative model uses `use_d_matrix=False` and evaluates the direct strategy equations.
2. The reduced model uses `use_d_matrix=True` and evaluates the precomputed $D$ matrix and $d$ vector.
3. After exactly 1, 50, or 100 iterations, the comparison records the largest absolute difference between corresponding final gene-fitness entries: $\max_j|\gamma^{\mathrm{iterative}}_j-\gamma^{\mathrm{D}}_j|$.

The acceptance criterion is `rtol=1e-12` and `atol=1e-12`. A displayed zero means the two floating-point vectors were bitwise equal on that fixture; a small value such as `1.11e-16` is ordinary floating-point roundoff and is far below the acceptance threshold.

For each `ORIGINAL` row, the named gene strategy is paired with
`NoneOrgStrategy`. That no-op partner contributes zero, so the comparison
isolates the named strategy. The `MATH_PAPER` row names both strategies because
the supported public configuration is the complete Alt-Sel pair. It does not
mean the two matrix components are algebraically inseparable; it means isolated
math-paper D-matrix models are intentionally outside the package's compatibility
contract.

### 1.6.2. Results

| Formulation | D-matrix configuration compared | 1 iteration | 50 iterations | 100 iterations | D-matrix vs. iterative-path assessment |
|---|---|---:|---:|---:|---|
| ORIGINAL | Dominant gene | 0 | 1.11e-16 | 6.51e-18 | Matches iterative path. |
| ORIGINAL | Selfish gene | 0 | 5.55e-17 | 2.78e-17 | Matches iterative path. |
| ORIGINAL | Kin-altruistic gene, default full neighbourhood | 0 | 1.11e-16 | 5.55e-17 | Matches iterative path. |
| ORIGINAL | Altruistic gene | 0 | 0 | 0 | Matches iterative path. |
| ORIGINAL | Sell hard gene | 1.39e-17 | 2.84e-16 | 5.72e-17 | Matches iterative path. |
| ORIGINAL | Sell uniform gene | 0 | 0 | 0 | Matches iterative path. |
| ORIGINAL | Sell easy gene | 0 | 0 | 0 | Matches iterative path. |
| ORIGINAL | Variance gene | 0 | 3.33e-16 | 6.94e-17 | Matches iterative path. |
| MATH_PAPER | Altruistic gene + Selfish organism, unit fixed coefficients | 5.55e-17 | 5.55e-17 | 1.11e-16 | Matches iterative path. |

The original rows use a fixed four-organism, three-gene fixture with initial $\gamma=(0.6,0.3,0.1)$. The math-paper row uses a five-organism, three-gene fixture with initial $\gamma=(0.4,0.35,0.25)$ and historical similarity scaling. The original kernels also run against three additional deterministic seven-organism, four-gene fixtures at all three iteration counts. Their largest observed discrepancy is $3.4\times10^{-16}$.

The assertions are implemented in `tests/unit/test_d_matrix_equivalence.py` and `tests/unit/test_math_paper_formulation.py`.

## 1.7. Safe use

### 1.7.1. Selecting a formulation

Use `ORIGINAL` for the Python-package equations. Use `MATH_PAPER` to reproduce the revised historical equations, similarities, normalisations, and Alt-Sel D-matrix formulation. Do not mix formulations in one model.

### 1.7.2. Selecting the execution path

Use `use_d_matrix=True` only for a configuration represented in the comparison table. In `ORIGINAL`, supported strategies may be combined with fixed coefficients because their direct deltas and D contributions are both added linearly. In `MATH_PAPER`, use only the unmixed Alt-Sel pair. All other strategies remain usable with `use_d_matrix=False`.
