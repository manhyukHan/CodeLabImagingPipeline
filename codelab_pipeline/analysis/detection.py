"""
Completeness modeled, not just counted: are any bins NON-INDEPENDENT?

Per explicit request (2026-08-30): efficacy/completeness histograms say
how much is missing; this module says whether the missingness has
STRUCTURE. Detection is X (n_alleles, n_bins) binary = bin has a finite
position. Two nulls, ported from BarcodeModels/functions.py:

  QUALITY   every allele has one latent detection efficiency
            u_i ~ N(0, tau^2); P(X_ij = 1) = sigmoid(b_j + u_i). Fitted
            by Gauss-Hermite-integrated maximum likelihood. Co-detection
            beyond this null means bins fail or succeed TOGETHER for
            reasons one per-allele scalar cannot carry.

  COUNT     condition on each allele's detected count k and sample
            column patterns within k-strata (the stratify-by-k idea of
            the chain model, WITHOUT its Ising adjacency fit -- the full
            DP machinery in the source carries duplicated definitions
            and half-finished branches; the stratified permutation null
            tests the same question with none of that surface).

Ported fixes (bugs recorded by the survey, not repeated here): one
sigmoid, overflow-clipped; no unused parameters; n_nodes is a single
argument threaded everywhere, never two inconsistent defaults.
"""
import numpy as np
from scipy import optimize, stats


def detection_matrix(pos_um):
    """(n_alleles, n_bins) uint8: bin observed = finite y position."""
    return np.isfinite(np.asarray(pos_um)[:, :, 0]).astype(np.uint8)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def fit_quality_model(X, n_nodes=25, l2_sigma=10.0, tau_min=1e-4,
                      maxiter=500):
    """MLE of (b (n_bins,), tau) for the per-allele quality null.

    GHQ over u ~ N(0, tau^2): loglik_i = logsumexp_k [ log w_k
    + sum_j X_ij (b_j + u_k) - softplus(b_j + u_k) ]. L2(sigma) prior on
    b. Returns {'b': (n_bins,), 'tau': float, 'loglik': float}.
    """
    X = np.asarray(X, float)
    n, m = X.shape
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    logw = np.log(weights / weights.sum())
    K = X.sum(1)

    def nll(theta):
        b, log_tau = theta[:m], theta[m]
        tau = max(np.exp(log_tau), tau_min)
        u = nodes * tau                       # (k,)
        eta = b[None, :] + u[:, None]         # (k, m)
        C = np.logaddexp(0.0, eta).sum(1)     # (k,) sum_j softplus
        ll_kn = logw[:, None] + (X @ b)[None, :] + u[:, None] * K[None, :] \
            - C[:, None]                      # (k, n)
        ll = np.logaddexp.reduce(ll_kn, axis=0).sum()
        return -(ll - (b ** 2).sum() / (2 * l2_sigma ** 2))

    theta0 = np.concatenate([np.zeros(m), [np.log(1.0)]])
    res = optimize.minimize(nll, theta0, method='L-BFGS-B',
                            options={'maxiter': int(maxiter)})
    b = res.x[:m]
    tau = max(float(np.exp(res.x[m])), tau_min)
    return {'b': b, 'tau': tau, 'loglik': float(-res.fun)}


def posterior_u(X, b, tau, n_nodes=25):
    """(u_mean (n,), u_sd (n,), q (k, n)) posterior of each allele's
    latent efficiency -- the per-allele detection-quality scalar the QC
    views plot per FOV."""
    X = np.asarray(X, float)
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    u = nodes * tau
    logw = np.log(weights / weights.sum())
    C = np.logaddexp(0.0, b[None, :] + u[:, None]).sum(1)
    ll = logw[:, None] + (X @ b)[None, :] + u[:, None] * X.sum(1)[None, :] \
        - C[:, None]
    ll -= ll.max(0, keepdims=True)
    q = np.exp(ll)
    q /= q.sum(0, keepdims=True)
    u_mean = (q * u[:, None]).sum(0)
    u_sd = np.sqrt(np.maximum((q * (u[:, None] - u_mean) ** 2).sum(0), 0))
    return u_mean, u_sd, q


def predicted_P(q, b, tau, n_nodes=25):
    """(n, m) allele-specific null detection probabilities. q from
    posterior_u WITH THE SAME n_nodes (shape-coupled by construction:
    the node count is derived from q itself)."""
    nodes, _ = np.polynomial.hermite_e.hermegauss(q.shape[0])
    u = nodes * tau
    return (q[:, :, None] * _sigmoid(b[None, None, :] + u[:, None, None])).sum(0)


def cooccurrence_z(X, P):
    """Pairwise co-detection beyond the null: (Z, p_one, ratio, obs, exp).

    C = X^T X observed; E = sum_i P_ij P_ik; V = sum_i q(1-q) with
    q = P_ij P_ik. Z (m, m) with NaN diagonal; p_one = one-sided
    enrichment; ratio = (C+eps)/(E+eps).
    """
    X = np.asarray(X, float)
    C = X.T @ X
    E = P.T @ P
    q = P[:, :, None] * P[:, None, :]
    V = (q * (1 - q)).sum(0)
    with np.errstate(all='ignore'):
        Z = (C - E) / np.sqrt(np.maximum(V, 1e-12))
        ratio = (C + 1e-9) / (E + 1e-9)
    np.fill_diagonal(Z, np.nan)
    np.fill_diagonal(ratio, np.nan)
    return Z, stats.norm.sf(Z), ratio, C, E


def count_stratified_null_z(X, n_samples=2000, seed=0):
    """The COUNT null: co-detection z against within-k-stratum shuffles.

    Every allele keeps its detected count k; within each k-stratum the
    column patterns are resampled by permuting each row independently
    (a random k-subset of bins, uniform). Answers: is the observed
    co-detection explained by count heterogeneity alone? Returns
    (Z (m, m) NaN diagonal, p_one).
    """
    X = np.asarray(X, np.uint8)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    C_obs = (X.astype(float).T @ X.astype(float))
    ks = X.sum(1)
    sims = np.zeros((int(n_samples), m, m))
    for s in range(int(n_samples)):
        Xs = np.zeros_like(X)
        for i in range(n):
            k = ks[i]
            if k:
                Xs[i, rng.choice(m, int(k), replace=False)] = 1
        Xf = Xs.astype(float)
        sims[s] = Xf.T @ Xf
    E = sims.mean(0)
    V = sims.var(0)
    with np.errstate(all='ignore'):
        Z = (C_obs - E) / np.sqrt(np.maximum(V, 1e-12))
    np.fill_diagonal(Z, np.nan)
    return Z, stats.norm.sf(Z)


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values, order-preserving, NaN passthrough."""
    p = np.asarray(pvals, float)
    flat = p.ravel()
    out = np.full_like(flat, np.nan)
    ok = np.isfinite(flat)
    if ok.sum():
        ps = flat[ok]
        order = np.argsort(ps)
        ranked = ps[order] * len(ps) / (np.arange(len(ps)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        q = np.empty_like(ranked)
        q[order] = np.clip(ranked, 0, 1)
        out[ok] = q
    return out.reshape(p.shape)


def count_overdispersion(X, b, tau, n_boot=2000, n_nodes=25, seed=0):
    """Is per-allele detected-count variance beyond the quality model?

    Parametric bootstrap of Var(K)_observed / Var(K)_model under the
    fitted null; p = (1 + #(sim >= obs)) / (B + 1). Returns
    {'ratio': float, 'p': float, 'var_obs': float, 'var_model': float}.
    """
    X = np.asarray(X, float)
    n, m = X.shape
    rng = np.random.default_rng(seed)
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    w = weights / weights.sum()
    u = nodes * tau
    p_ku = _sigmoid(b[None, :] + u[:, None])          # (k, m)
    EK_u = p_ku.sum(1)
    var_model = float((w * (p_ku * (1 - p_ku)).sum(1)).sum()
                      + (w * EK_u ** 2).sum() - ((w * EK_u).sum()) ** 2)
    var_obs = float(X.sum(1).var(ddof=1)) if n > 1 else 0.0
    ratio = var_obs / max(var_model, 1e-12)
    exceed = 0
    for _ in range(int(n_boot)):
        uk = rng.choice(len(u), size=n, p=w)
        sim = rng.random((n, m)) < p_ku[uk]
        r = sim.sum(1).var(ddof=1) / max(var_model, 1e-12)
        exceed += (r >= ratio)
    return {'ratio': ratio, 'p': (1 + exceed) / (n_boot + 1),
            'var_obs': var_obs, 'var_model': var_model}
