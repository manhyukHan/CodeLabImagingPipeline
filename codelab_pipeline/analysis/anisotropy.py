"""
Is Z as trustworthy as X and Y?

A nucleus has no preferred orientation, so the vector between two loci
should be isotropic: every direction equally likely. X and Y are, in
practice, very nearly so. Z frequently is not -- axial localization is
the weakest axis of a light microscope, and focus drift, refractive-index
mismatch and residual Z alignment error all inflate it. Inflated Z does
not announce itself: every distance simply comes out a little too large,
the distance map looks normal, and a real biological difference is
diluted by a measurement artefact that is largest exactly where the
biology is most interesting -- at SHORT separations, where a fixed axial
error is the biggest fraction of the distance.

THE TEST. For an isotropic displacement (dx, dy, dz),

    E[dx^2] = E[dy^2] = E[dz^2]
    d_xyz^2 = dx^2 + dy^2 + dz^2      d_xy^2 = dx^2 + dy^2

so  E[d_xy^2] / E[d_xyz^2] = 2/3, i.e. d_xy ~ sqrt(2/3) * d_xyz. Plotting
d_xyz against d_xy, the isotropic line has slope sqrt(3/2) = 1.2247, and
Z inflation pushes points ABOVE it.

That relation is exact in the MEAN SQUARE, not point by point: a single
pair with dz = 0 sits below the line while being perfectly isotropic.
Every statistic here is therefore built from mean squares over many
pairs, never from a per-pair ratio -- a per-pair d_xyz/d_xy is a
heavy-tailed quantity whose mean is not the thing being asked about.

THE NUMBER TO READ is `factor`:

    factor = sqrt( E[dz^2] / (E[d_xy^2] / 2) )

the axial spread divided by the in-plane spread per axis. 1.0 is
isotropic; 1.3 means Z is 30% wider than X and Y are; below 1.0 means Z
is COMPRESSED, which is its own problem (over-correction). It is a ratio
of two measured quantities from the SAME pairs, so it needs no
calibration and no reference dataset.

dz is recovered as sqrt(d_xyz^2 - d_xy^2) rather than from the positions,
so this works on anything that can produce both distances -- and it means
`factor` and the scatter cannot disagree about what dz is.
"""
import numpy as np

# the isotropic expectation, named once
ISOTROPIC_XY_OVER_XYZ = np.sqrt(2.0 / 3.0)          # 0.8165
ISOTROPIC_XYZ_OVER_XY = np.sqrt(3.0 / 2.0)          # 1.2247


def _finite_pairs(d_xy, d_xyz):
    a = np.asarray(d_xy, float).ravel()
    b = np.asarray(d_xyz, float).ravel()
    keep = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return a[keep], b[keep]


def anisotropy_factor(d_xy, d_xyz):
    """(factor, n_pairs). 1.0 = isotropic, >1 = Z inflated.

    Uses mean squares, and clips a negative dz^2 to zero: d_xyz < d_xy is
    arithmetically impossible and only appears through rounding, so it is
    a zero-size axial component, not a negative one.
    """
    a, b = _finite_pairs(d_xy, d_xyz)
    if len(a) < 2:
        return float('nan'), len(a)
    dz2 = np.maximum(b ** 2 - a ** 2, 0.0)
    in_plane_per_axis = np.mean(a ** 2) / 2.0
    if in_plane_per_axis <= 0:
        return float('nan'), len(a)
    return float(np.sqrt(np.mean(dz2) / in_plane_per_axis)), len(a)


def anisotropy_by_range(d_xy, d_xyz, edges=None, min_pairs=30):
    """The RAW factor per separation bin. Read excess_by_range instead.

    THIS CURVE RISES TOWARD SHORT SEPARATION EVEN WITH NO AXIAL ERROR AT
    ALL, and the rise is an artefact of the binning, not a finding.
    Conditioning on d_xy pins the denominator to about (bin centre)^2 / 2
    while leaving the numerator at the population's axial scale, so the
    factor goes like 1/centre by construction.

    Measured on MP58 geometry with z replaced by an unrelated in-plane
    coordinate -- zero axial error by construction -- this still ran
    6.87 -> 0.52 across the bins, a 13.3x rise, STEEPER than the real
    data's 5.4x. A pure z*1.5 scale error rose 5.4x and additive axial
    noise rose 5.8x, so the shape does not separate those two either.
    An earlier version of this module claimed it did; it does not, and
    the claim is withdrawn.

    Kept because excess_by_range needs it for both the data and its
    surrogate, and because the raw numbers are still what the ratio is
    built from.
    """
    a, b = _finite_pairs(d_xy, d_xyz)
    if len(a) < min_pairs:
        return {'centers': np.array([]), 'factor': np.array([]),
                'n': np.array([], int)}
    if edges is None:
        hi = float(np.quantile(a, 0.98))
        edges = np.linspace(0.0, hi if hi > 0 else 1.0, 13)
    edges = np.asarray(edges, float)
    idx = np.digitize(a, edges) - 1
    centers, factors, counts = [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        n = int(sel.sum())
        if n < min_pairs:
            continue
        f, _ = anisotropy_factor(a[sel], b[sel])
        centers.append(0.5 * (edges[k] + edges[k + 1]))
        factors.append(f)
        counts.append(n)
    return {'centers': np.asarray(centers), 'factor': np.asarray(factors),
            'n': np.asarray(counts, int)}


def isotropic_surrogate(pos_um, seed=0):
    """The same chains with a KNOWN-GOOD z: each allele's axial column
    replaced by an unrelated allele's in-plane column.

    That gives data with the real spatial structure and, by
    construction, no axial error -- an isotropy null this pipeline can
    actually measure against instead of assuming.
    """
    pos = np.asarray(pos_um)
    if len(pos) < 2:
        return pos.copy()
    rng = np.random.default_rng(seed)
    out = pos.copy()
    out[:, :, 2] = pos[rng.permutation(len(pos)), :, 1]
    return out


def excess_by_range(pos_um, edges=None, seed=0, min_pairs=30):
    """Axial excess per separation bin, RELATIVE TO an isotropic
    surrogate: 1.0 means this bin's axial spread is what isotropy would
    give, above 1 means excess.

    The raw per-bin factor cannot be read directly (see
    anisotropy_by_range), so it is divided by the same statistic computed
    on data whose z is known-good. Verified on MP58 geometry: the
    surrogate scores 0.93-1.02 across bins, while the real data runs
    0.93 -> 2.32.

    WHAT THIS DOES NOT DO: separate a scale error from an additive one.
    Both rise here (z*1.5 gave 1.31 -> 3.86, additive 0.15 um gave
    1.03 -> 2.70), so the SHAPE carries no verdict about which kind of
    axial error is present -- only the height carries "there is one".
    """
    pos = np.asarray(pos_um)
    if edges is None:
        d_xy, _ = maps_xy_xyz(pos[:min(len(pos), 200)])
        hi = float(np.quantile(d_xy[np.isfinite(d_xy)], 0.98)) if len(pos) else 1.0
        edges = np.linspace(0.0, hi if hi > 0 else 1.0, 13)

    def factors(p):
        d_xy, d_xyz = maps_xy_xyz(p)
        n_b = d_xy.shape[1]
        iu = np.triu_indices(n_b, k=1)
        a, b = _finite_pairs(d_xy[:, iu[0], iu[1]], d_xyz[:, iu[0], iu[1]])
        return anisotropy_by_range(a, b, edges=edges, min_pairs=min_pairs)

    real = factors(pos)
    null = factors(isotropic_surrogate(pos, seed))
    k = min(len(real['factor']), len(null['factor']))
    if k == 0:
        return {'centers': np.array([]), 'excess': np.array([]),
                'n': np.array([], int)}
    with np.errstate(all='ignore'):
        ratio = real['factor'][:k] / null['factor'][:k]
    return {'centers': real['centers'][:k], 'excess': ratio,
            'n': real['n'][:k]}


def rms_slope(d_xy, d_xyz):
    """The observed d_xyz/d_xy slope in the RMS sense, and what isotropy
    predicts (sqrt(3/2)).

    An RMS ratio, NOT a least-squares fit through the origin: the cloud is
    heteroscedastic (spread grows with separation) and a plain regression
    would be dominated by the long distances, which are the ones an axial
    error affects least.
    """
    a, b = _finite_pairs(d_xy, d_xyz)
    if len(a) < 2:
        return float('nan'), ISOTROPIC_XYZ_OVER_XY, len(a)
    obs = float(np.sqrt(np.mean(b ** 2) / np.mean(a ** 2)))
    return obs, ISOTROPIC_XYZ_OVER_XY, len(a)


def z_scale_correction(d_xy, d_xyz):
    """The factor z would have to be MULTIPLIED by to make the data
    isotropic, IF the error were a pure scale.

    Nothing here establishes that it is. The by-range curve was once
    cited as the test for it and cannot serve: it rises for a pure scale
    error, for additive noise and for no axial error at all. So this is a
    what-if, reported so the size of the implied correction is visible,
    and applied nowhere -- silently rescaling an axis to satisfy an
    unverified assumption is how an artefact becomes a finding.
    """
    f, n = anisotropy_factor(d_xy, d_xyz)
    if not np.isfinite(f) or f <= 0:
        return float('nan'), n
    return float(1.0 / f), n


def maps_xy_xyz(pos_um):
    """(d_xy, d_xyz) pairwise map stacks from (n_alleles, n_bins, 3)
    positions -- the same pairs, measured both ways, which is what makes
    them comparable at all.
    """
    from . import polymer
    return (polymer.polymer_distmaps(pos_um, dims='xy'),
            polymer.polymer_distmaps(pos_um, dims='xyz'))


MAX_ALLELES = 4000


def summary(pos_um, mask=None, max_alleles=MAX_ALLELES, seed=0):
    """Everything the QC figure needs, from one position array.

    mask: boolean (n_alleles,) -- the gated stack, or None for all.

    SOURCED FROM POSITIONS, never from Population.dmaps(): this needs
    d_xy and d_xyz for the SAME pairs, and z is unrecoverable from an
    already-computed in-plane map. It is also the one distance site in
    the toolbox that must NOT follow a global XY/XYZ choice -- fed an
    in-plane map on both sides it would compare XY against XY, return a
    perfect diagonal, and certify as isotropic exactly the data whose
    Z it exists to distrust.

    SUBSAMPLED above max_alleles alleles. Two float32 stacks at the real
    store's scale are ~1.5 GB (polymer.py measures 738 MB for one at
    24k alleles x 62 bins), and this statistic converges on a few
    thousand alleles -- 4000 alleles x 62 bins is already ~7.6M pairs.
    The draw is seeded, so the number does not move between runs on the
    same data.
    """
    pos = np.asarray(pos_um)
    if mask is not None:
        pos = pos[np.asarray(mask, bool)]
    n_sub = 0
    if max_alleles and len(pos) > int(max_alleles):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(pos), int(max_alleles), replace=False)
        pos = pos[np.sort(pick)]
        n_sub = int(max_alleles)
    d_xy, d_xyz = maps_xy_xyz(pos)
    # the upper triangle only: a distance map is symmetric with a zero
    # diagonal, so using all of it would double-count every pair and
    # dilute the statistic with n_bins exact zeros
    n_b = d_xy.shape[1]
    iu = np.triu_indices(n_b, k=1)
    # RAVELLED and finite-filtered here, once: every statistic below and
    # every plot above then works on the SAME pair list, so a figure can
    # never be drawn from a different set of pairs than the number
    # printed beside it.
    a, b = _finite_pairs(d_xy[:, iu[0], iu[1]], d_xyz[:, iu[0], iu[1]])
    obs, expected, n = rms_slope(a, b)
    factor, _ = anisotropy_factor(a, b)
    corr, _ = z_scale_correction(a, b)
    return {'d_xy': a, 'd_xyz': b, 'factor': factor,
            'rms_slope': obs, 'isotropic_slope': expected,
            'z_scale_correction': corr, 'n_pairs': n,
            'n_alleles': len(pos), 'subsampled_to': n_sub,
            'excess_by_range': excess_by_range(pos, seed=seed)}
