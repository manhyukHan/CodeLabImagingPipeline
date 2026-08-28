"""
Ensemble distance maps, and the QC views that decide whether to trust one.

Conventions carried from the reference work (ORCA_modelling + the SG
whole-chromosome scripts):
  - the ensemble reducer is NANMEDIAN by default (mean available);
  - a pixel is shown only when observed in >= min_n alleles -- pixels
    below the count become NaN, drawn dark by the display layer, never
    silently averaged from two alleles;
  - subtraction maps mask a pixel when EITHER group is under-observed;
  - per-cell/allele quantities collapse by MEDIAN, never min (a
    zero-bounded noise floor contaminates min).

Flags are groupings, not filters: the FOV / celltype / allele-difference
views take the SAME gated stack and split it, so "show me the FOV
differences" can never change which cells are in the analysis.
"""
import numpy as np


def ensemble_map(dmaps, mask=None, reducer='median', min_n=1):
    """(map (n_bins, n_bins), counts (n_bins, n_bins) int).

    dmaps: (n_alleles, n_bins, n_bins); mask: boolean (n_alleles,) gate
    (None = all). Pixels observed in fewer than min_n alleles are NaN in
    the map; counts always report the truth.
    """
    d = dmaps if mask is None else dmaps[np.asarray(mask, bool)]
    counts = np.isfinite(d).sum(0) if len(d) else np.zeros(dmaps.shape[1:], int)
    if len(d) == 0:
        return np.full(dmaps.shape[1:], np.nan), counts
    with np.errstate(all='ignore'):
        m = np.nanmedian(d, 0) if reducer == 'median' else np.nanmean(d, 0)
    m[counts < int(min_n)] = np.nan
    return m, counts


def subtraction_map(dmaps, mask_a, mask_b, reducer='median', min_n=1):
    """map(A) - map(B), NaN wherever EITHER group is under-observed."""
    ma, ca = ensemble_map(dmaps, mask_a, reducer, min_n)
    mb, cb = ensemble_map(dmaps, mask_b, reducer, min_n)
    out = ma - mb
    out[(ca < int(min_n)) | (cb < int(min_n))] = np.nan
    return out, ca, cb


def scc(map_a, map_b, h=1, max_stratum=None):
    """Stratum-adjusted correlation between two maps (HiCRep, ORCA port).

    Strata are the off-diagonals |i-j| = d; per stratum the Pearson
    correlation over entries finite in BOTH maps; combined with weights
    n_d * std_a * std_b. Smoothing h applies an NaN-aware (2h+1)^2 box
    mean first (h=0 disables). Returns float in [-1, 1], NaN when no
    stratum has enough data.
    """
    a, b = np.asarray(map_a, float), np.asarray(map_b, float)
    if h and h > 0:
        a, b = _nanbox(a, h), _nanbox(b, h)
    n = a.shape[0]
    hi = min(n - 1, max_stratum) if max_stratum else n - 1
    num = den = 0.0
    for d in range(1, hi + 1):
        x, y = np.diagonal(a, d), np.diagonal(b, d)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        x, y = x[ok], y[ok]
        sx, sy = x.std(), y.std()
        if sx <= 0 or sy <= 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        w = ok.sum() * sx * sy
        num += w * r
        den += w
    return num / den if den > 0 else float('nan')


def _nanbox(m, h):
    """NaN-aware (2h+1)^2 box mean via two separable passes; edges shrink."""
    out = np.full_like(m, np.nan)
    n = m.shape[0]
    with np.errstate(all='ignore'):
        for i in range(n):
            out[i] = np.nanmean(m[max(0, i - h):min(n, i + h + 1)], 0)
        out2 = np.full_like(m, np.nan)
        for j in range(n):
            out2[:, j] = np.nanmean(out[:, max(0, j - h):min(n, j + h + 1)], 1)
    return out2


def fov_consistency(dmaps, fovs, mask=None, reducer='median', min_n=1, h=1):
    """Per-FOV ensembles and their pairwise SCC matrix.

    Returns {'fovs': [int], 'maps': {fov: map}, 'counts': {fov: counts},
    'scc': (n_fov, n_fov) float with 1s on the diagonal}. The FOV split
    happens AFTER the gate: a flag, not a filter.
    """
    fovs = np.asarray(fovs)
    base = np.ones(len(fovs), bool) if mask is None else np.asarray(mask, bool)
    uniq = sorted(int(f) for f in np.unique(fovs[base]))
    maps, counts = {}, {}
    for f in uniq:
        m, c = ensemble_map(dmaps, base & (fovs == f), reducer, min_n)
        maps[f], counts[f] = m, c
    k = len(uniq)
    S = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            S[i, j] = S[j, i] = scc(maps[uniq[i]], maps[uniq[j]], h=h)
    return {'fovs': uniq, 'maps': maps, 'counts': counts, 'scc': S}


def fov_msd_test(dmaps, fovs, mask=None, max_pairs_per_class=20000,
                 min_shared=10, seed=0):
    """FOV deviation as a TESTED distribution, not a 2-map correlation.

    SCC compares two ensemble maps -- one number per FOV pair, and at
    ~10^2 alleles per FOV those maps are grainy, so the matrix is
    weak evidence either way. This instead builds the pairwise
    allele-map MSD distribution -- mean squared difference over the two
    maps' SHARED finite upper-triangle entries -- for IN-FOV pairs and
    CROSS-FOV pairs, and Welch-tests the two. Consistent FOVs give
    in ~ cross (large p); a batch effect separates them.

    Per-FOV verdicts too: FOV f's own in-pairs against the pairs
    linking f to the rest, one -log10 p per FOV, so the deviant FOV is
    NAMED rather than implied.

    Pairs are subsampled per class (max_pairs_per_class) and share
    alleles, so observations are not independent: p-values are honest
    RANKING scores, anticonservative in absolute terms -- recorded here
    once instead of rediscovered per reader. A label-permutation
    upgrade slots in behind the same return shape when a decision needs
    calibrated error rates.

    Returns {'msd_in', 'msd_cross', 't', 'p', 'neglog10p',
    'per_fov': [{'fov', 'n_in', 'n_cross', 't', 'p', 'neglog10p'}]}.
    """
    from scipy import stats as _stats
    n = len(dmaps)
    base = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    idx = np.flatnonzero(base)
    rng = np.random.default_rng(seed)
    if len(idx) < 3:
        return {'msd_in': np.array([]), 'msd_cross': np.array([]),
                't': np.nan, 'p': np.nan, 'neglog10p': np.nan,
                'per_fov': []}
    fov_of = np.asarray(fovs)
    iu = np.triu_indices(dmaps.shape[1], k=1)
    flat = dmaps[idx][:, iu[0], iu[1]]                 # (n_kept, n_bin_pairs)
    kept_fov = fov_of[idx]

    def sample_pairs(want_in, k):
        """k random (i, j) index pairs into `idx`, in- or cross-FOV."""
        out = np.empty((0, 2), int)
        tries = 0
        while len(out) < k and tries < 12:
            cand = rng.integers(0, len(idx), (k * 2, 2))
            cand = cand[cand[:, 0] < cand[:, 1]]
            same = kept_fov[cand[:, 0]] == kept_fov[cand[:, 1]]
            cand = cand[same if want_in else ~same]
            out = np.unique(np.concatenate([out, cand]), axis=0)
            tries += 1
        return out[:k]

    def msd(pairs):
        vals = np.full(len(pairs), np.nan)
        step = 2000
        for s in range(0, len(pairs), step):
            blk = pairs[s:s + step]
            a = flat[blk[:, 0]]
            b = flat[blk[:, 1]]
            ok = np.isfinite(a) & np.isfinite(b)
            diff2 = np.where(ok, (a - b) ** 2, 0.0)
            cnt = ok.sum(1)
            with np.errstate(invalid='ignore'):
                v = diff2.sum(1) / cnt
            v[cnt < int(min_shared)] = np.nan
            vals[s:s + step] = v
        return vals[np.isfinite(vals)], pairs

    m_in, p_in = msd(sample_pairs(True, int(max_pairs_per_class)))
    m_cx, p_cx = msd(sample_pairs(False, int(max_pairs_per_class)))

    def welch(a, b):
        if len(a) < 3 or len(b) < 3:
            return np.nan, np.nan
        t, p = _stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)

    t, p = welch(m_in, m_cx)
    per_fov = []
    for f in sorted(set(int(v) for v in kept_fov)):
        # NAMING the deviant FOV needs the right comparison. In-vs-cross
        # per FOV cannot do it: one shifted FOV inflates EVERY other
        # FOV's cross-class, so all FOVs saturate together (measured:
        # three FOVs, one planted deviant, all three at -log10 p = 300).
        # The discriminating split is cross-pairs INVOLVING f against
        # cross-pairs NOT involving f, SIGNED: only the deviant's
        # own cross-pairs are systematically larger than the rest, so it
        # alone scores strongly positive; its neighbours score negative
        # (their with-f mixture sits below the without-f pool).
        touches = (kept_fov[p_cx[:, 0]] == f) | (kept_fov[p_cx[:, 1]] == f)
        a, _ = msd(p_cx[touches]) if touches.sum() else (np.array([]), None)
        b, _ = msd(p_cx[~touches]) if (~touches).sum() else (np.array([]), None)
        tf, pf = welch(a, b)
        score = float(np.sign(tf) * -np.log10(max(pf, 1e-300))) \
            if np.isfinite(pf) else np.nan
        per_fov.append({'fov': f, 'n_with': int(len(a)),
                        'n_without': int(len(b)), 't': tf, 'p': pf,
                        'signed_neglog10p': score})
    return {'msd_in': m_in, 'msd_cross': m_cx, 't': t, 'p': p,
            'neglog10p': float(-np.log10(max(p, 1e-300)))
            if np.isfinite(p) else np.nan,
            'per_fov': per_fov}


def allele_difference(dmaps, fovs, cells, mask=None, rng=None, n_null=1000):
    """Within-cell allele-pair dissimilarity vs an across-cell null.

    There is deliberately NO stable allele indexing per cell -- only the
    DIFFERENCE between a cell's alleles is meaningful. For every gated
    cell holding >= 2 alleles, all its allele pairs contribute
    nanmean|d1 - d2| over the entries finite in both maps; the null is
    the same statistic over randomly drawn cross-cell pairs.

    Returns {'within': (n_pairs,), 'within_cells': [(fov, cell)],
    'null': (n_null,), 'n_multi_allelic': int}.
    """
    n = len(dmaps)
    base = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    idx = np.flatnonzero(base)
    key = {}
    for i in idx:
        if cells[i] < 0:
            continue           # homeless alleles have no within-cell pair
        key.setdefault((int(fovs[i]), int(cells[i])), []).append(i)

    def diff(i, j):
        a, b = dmaps[i], dmaps[j]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            return np.nan
        return float(np.abs(a[ok] - b[ok]).mean())

    within, wcells = [], []
    for (f, c), members in key.items():
        if len(members) < 2:
            continue
        for u in range(len(members)):
            for v in range(u + 1, len(members)):
                within.append(diff(members[u], members[v]))
                wcells.append((f, c))
    rng = np.random.default_rng(0) if rng is None else rng
    null = []
    if len(idx) >= 2:
        for _ in range(int(n_null)):
            i, j = rng.choice(idx, 2, replace=False)
            if cells[i] >= 0 and (fovs[i], cells[i]) == (fovs[j], cells[j]):
                continue       # that would be a within pair
            null.append(diff(i, j))
    return {'within': np.array(within), 'within_cells': wcells,
            'null': np.array(null),
            'n_multi_allelic': len([1 for m in key.values() if len(m) >= 2])}
