# Detection nulls: is barcode co-occurrence real?

Efficacy and completeness histograms tell you *how much* is missing from a chromatin trace. `codelab_pipeline.analysis.detection` answers a sharper question: does the missingness have **structure**? If two genomic bins (barcode rounds) tend to be detected or dropped *together* more than chance allows, that co-occurrence points at something real — a shared fluidics failure across adjacent hybes, coupled decoding errors, or genuine biology — and none of it is visible in per-bin or per-allele marginals.

The module is part of the headless analysis toolbox: importing it loads no Qt and no app code. Everything below runs from a plain script or notebook against a store.

```python
import numpy as np
from codelab_pipeline.analysis import detection, population
```

The module offers **two independent null models** for the same binary matrix, and the honest workflow is to run both.

## The detection matrix

Everything starts from `detection_matrix(pos_um)`, which reduces an allele-position array to a binary detection matrix:

```python
def detection_matrix(pos_um):
    """(n_alleles, n_bins) uint8: bin observed = finite y position."""
```

The input is the `(n_alleles, n_bins, 3)` micrometre-scaled position array that `Population` carries in `pop.alleles['pos_um']` (coordinates are `(y, x, z)`; positions were scaled by `voxel_um = (0.208, 0.208, 0.2)` once, at extraction). A bin counts as detected when its **y** coordinate is finite — by package convention a bin's position is either fully finite or fully NaN, so the first axis stands proxy for all three. NaN is honest missingness here, exactly as everywhere else in the toolbox: `X[i, j] == 0` means "no finite position", never a fabricated value.

Rows of `X` align one-to-one with allele rows, so anything you compute per allele (the posterior efficiencies below, for example) joins back to cells through the pair `(pop.alleles['fov'][i], pop.alleles['cell'][i])` — cells are always keyed by the `(fov, cell)` pair, never a bare cell id. Columns align with `pop.alleles['bin_hybes']` / `bin_ids`.

```python
if __name__ == '__main__':          # required on Windows when jobs != 1
    pop = population.Population.build(r'E:/Students/2026-08-07-SG-test/DNA',
                                      fovs=range(34), records=records, jobs=8)
    X = detection.detection_matrix(pop.alleles['pos_um'])   # (n_alleles, n_bins)
```

## Null 1: the quality model

The first null says: every allele has a single latent detection efficiency, and given that scalar, bins are independent. Concretely, `u_i ~ N(0, tau^2)` per allele, `b_j` per bin, and

```
P(X_ij = 1 | u_i) = sigmoid(b_j + u_i)
```

`fit_quality_model(X, n_nodes=25, l2_sigma=10.0, tau_min=1e-4, maxiter=500)` fits `(b, tau)` by maximum likelihood, integrating the latent `u` out with Gauss–Hermite quadrature (probabilists' nodes scaled by `tau`, `n_nodes` points). The optimizer is L-BFGS-B over `(b, log tau)`; `b` carries an L2 (Gaussian, sd `l2_sigma`) prior, and `tau` is floored at `tau_min`. It returns a dict:

```python
fit = detection.fit_quality_model(X)
fit['b']       # (n_bins,) per-bin log-odds baseline
fit['tau']     # float, sd of the per-allele efficiency
fit['loglik']  # float, penalized log-likelihood at the optimum
```

`posterior_u(X, b, tau, n_nodes=25)` then gives each allele's posterior over its own efficiency, discretized on the same quadrature grid:

```python
u_mean, u_sd, q = detection.posterior_u(X, fit['b'], fit['tau'])
# u_mean, u_sd: (n_alleles,) posterior mean and sd of u_i
# q:            (n_nodes, n_alleles) posterior weight of each node per allele
```

`u_mean` is the per-allele detection-quality scalar the QC views plot per FOV; the full weight matrix `q` is what the exact co-occurrence algebra below consumes.

`predicted_P(q, b, tau, n_nodes=25)` returns the `(n_alleles, n_bins)` allele-specific null detection probabilities, `sum_n q_ni * sigmoid(b_j + u_n)`. Note that the node grid is reconstructed from `q.shape[0]` — the `n_nodes` argument in its signature is ignored by construction, so `q` must simply come from `posterior_u`.

## Co-occurrence against the quality null: `cooccurrence_z`

```python
Z, p_one, ratio, obs, exp = detection.cooccurrence_z(X, q, fit['b'], fit['tau'])
```

The observed statistic is the pairwise co-detection count `C = X.T @ X`. The null expectation is where the module is deliberately careful, and the reason deserves restating.

**Why the naive product of posterior means is biased.** Under the model, `X_ij` and `X_ik` are conditionally independent given `u_i`, so the correct per-allele pair probability is the posterior expectation *of the product*:

```
p_i(j,k) = E[ s_j(u) s_k(u) | X_i ] = sum_n q_ni * s_j(u_n) * s_k(u_n)
```

where `s_j(u) = sigmoid(b_j + u)`. The tempting shortcut `E[s_j | X_i] * E[s_k | X_i]` is wrong: both sigmoids are increasing functions of the *shared* `u`, so their posterior covariance is non-negative, and the product of means understates every pair expectation. An understated expectation inflates `C - E`, and review of the first version confirmed a **positive z bias on every single pair** — the null flagged everything. The shipped version computes the expectation exactly, in closed form over the quadrature nodes:

```
E_jk        = sum_n W_n s_nj s_nk              W = q.sum(over alleles)
sum_i p_i^2 = sum_{n,l} M_nl (s_nj s_lj)(s_nk s_lk),   M = q @ q.T
V_jk        = E_jk - sum_i p_i^2               (sum of Bernoulli variances)
Z           = (C - E) / sqrt(max(V, 1e-12))
```

Returns, in order: `Z` (m, m; NaN diagonal), `p_one = stats.norm.sf(Z)` (one-sided, testing **excess** co-detection), `ratio = (C + 1e-9) / (E + 1e-9)` (obs/exp, NaN diagonal), `obs` (the raw `C`), `exp` (the exact `E`). A pair that co-occurs *less* than the null predicts shows up as negative `Z` with `p_one` near 1 — use `stats.norm.cdf(Z)` yourself if deficits are the question.

## Null 2: the count-stratified (curveball) null

The quality model is parametric. `count_stratified_null_z` asks the same question with no model at all: is the observed matrix more structured than random matrices with **exactly the same margins** — every allele keeping its detected count `k`, every bin keeping its efficacy?

```python
Zc, pc, info = detection.count_stratified_null_z(
    X, n_samples=200, seed=0, max_alleles=4000,
    burn_factor=5, thin_factor=1)
```

The sampler is the curveball trade: pick two random rows, keep their shared bins, and randomly redistribute the bins in their symmetric difference between them. Each row keeps its count and every traded bin stays present in exactly one of the two rows, so **both row sums and column sums are preserved exactly** by every trade. After a burn-in of `burn_factor * n` trades, it takes `n_samples` snapshots of `X.T @ X` separated by `thin_factor * n` trades each, and z-scores the observed co-detection against the empirical mean and variance of those snapshots.

**Why margin preservation is not optional.** The first version of this null resampled each row as a uniform k-subset of the bins — preserving row counts but flattening every column to the same expected sum. Real stores have large per-bin efficacy spread (per-bin efficacy is a first-class QC output of this very package, `polymer.efficacy`). Against a flattened null, any pair of high-efficacy bins co-occurs "astronomically" more than expected and every such pair lit up as significant, while real co-detection among low-efficacy bins was buried below the uniform expectation. Review caught this before any caller shipped. The curveball null makes bin efficacy part of the conditioning, so only structure *beyond both margins* scores.

Alleles are subsampled without replacement to `max_alleles` when the population is larger (the trade chain is O(rows) per sweep, and QC power does not need 10^5 rows). Returns `Zc` (m, m; NaN diagonal), the one-sided `pc = stats.norm.sf(Zc)`, and `info = {'n_used', 'n_total', 'n_samples'}` recording the subsample.

## Which null, when

The two nulls encode different completeness models, and disagreement between them is informative:

- **Quality model** (`fit_quality_model` + `cooccurrence_z`): use when you believe missingness is driven by a per-allele quality scalar (out-of-focus cell, poor hybridization for that allele) times per-bin efficacy. Co-detection surviving this null means bins fail or succeed *together* for reasons one scalar per allele cannot carry — the model is explicit, so the excess is interpretable, and you also get `u_mean` per allele and `predicted_P` for free.
- **Count-stratified null** (`count_stratified_null_z`): use when you do not want to trust a parametric form at all. It conditions on both margins exactly and asks only whether the fill pattern is more structured than any matrix with the same margins. It is the more conservative check and it cannot be fooled by a mis-specified quality model.

A pair significant under both is the strongest evidence that barcode co-occurrence is real. A pair significant only under the quality model may just mean the logistic form fits the margins poorly. Neither null assumes the barcodes are independent — that independence is precisely the hypothesis under test.

## Multiple testing and overdispersion

`bh_fdr(pvals)` converts p-values of any shape to Benjamini–Hochberg q-values, order-preserving, with NaN passthrough (NaNs stay NaN and do not count toward the number of tests — the NaN diagonal is handled for free). Test each unordered pair once by taking the upper triangle:

```python
m = X.shape[1]
iu = np.triu_indices(m, k=1)
qvals = detection.bh_fdr(p_one[iu])
hits = [(int(j), int(k)) for j, k, ok in zip(iu[0], iu[1], qvals < 0.05) if ok]
```

`count_overdispersion(X, b, tau, n_boot=2000, n_nodes=25, seed=0)` is a global one-number check on the quality model itself: is the variance of per-allele detected counts larger than the fitted model allows? It compares `Var(K)` observed to the model's analytic `Var(K)`, then calibrates the ratio by parametric bootstrap under the fitted null, with `p = (1 + #(sim >= obs)) / (n_boot + 1)`. Returns `{'ratio', 'p', 'var_obs', 'var_model'}`. A ratio well above 1 with small `p` says allele completeness varies more than one Gaussian latent explains — expect structure in the pairwise maps.

## Calibration: synthetic independence must come back null

The module's history is two calibration bugs (the biased product expectation; the margin-destroying subset null), and the guard against a third is cheap: generate data that is independent *by construction* under the fitted parameters, rescore it, and demand that the median off-diagonal z is near 0.

```python
rng = np.random.default_rng(1)
n, m = X.shape
u_true = rng.normal(0.0, fit['tau'], n)
P = 1.0 / (1.0 + np.exp(-(fit['b'][None, :] + u_true[:, None])))
X_null = (rng.random((n, m)) < P).astype(np.uint8)

f0 = detection.fit_quality_model(X_null)
_, _, q0 = detection.posterior_u(X_null, f0['b'], f0['tau'])
Z0, *_ = detection.cooccurrence_z(X_null, q0, f0['b'], f0['tau'])
print(np.nanmedian(Z0))    # should sit near 0; a systematic offset means bias
```

Run the same check on `count_stratified_null_z(X_null)`. If either null shows `|median z|` visibly away from 0 on data with no structure, stop trusting its p-values on real data.

## Pitfalls

- **Detection is finiteness of y only.** `detection_matrix` inspects `pos_um[:, :, 0]`. That is correct for arrays produced by this package (a bin is all-finite or all-NaN) but not for hand-built arrays with partially finite coordinates.
- **`q` couples everything.** `cooccurrence_z` and `predicted_P` rebuild the quadrature grid from `q.shape[0]`; `predicted_P`'s own `n_nodes` argument is ignored. Always pass the `q` from `posterior_u` together with the *same* `b` and `tau` it was computed from.
- **One-sided p-values.** Both nulls return `stats.norm.sf(Z)` — excess co-detection only. Deficits require the CDF side.
- **NaN placement.** `Z` and `ratio` carry NaN diagonals; `obs` and `exp` do not. `bh_fdr` passes NaN through, but slice the upper triangle so each pair is one test.
- **Monte-Carlo variance in the count null.** `E` and `V` come from `n_samples` snapshots (default 200); extreme z tails wobble at that budget. Raise `n_samples` (and `thin_factor` for less snapshot autocorrelation) before believing a single spectacular pair. The `max_alleles=4000` subsample is seed-dependent and recorded in `info`.
- **Empty and degenerate input.** `count_overdispersion` reports `var_obs = 0.0` for fewer than two alleles; a fit on a near-empty `X` is driven by the L2 prior on `b`, not data. Zero alleles is a legitimate gate outcome elsewhere in the toolbox, but there is no co-occurrence question to ask of it.
- **Windows parallelism.** The detection functions themselves are single-process, but the `Population.build(..., jobs=8)` call that feeds them spawns workers — any script doing that must live under an `if __name__ == '__main__':` guard on Windows.
- **Fixture data proves nothing here.** Calibrate and measure against the real store; the checked-in `data/` extract is small, unrepresentative, and often absent on Windows machines.
