"""
An EMPIRICAL PSF, measured from spots humans confirmed, and what to do
with it.

NOT A REPLACEMENT FOR psf.py. That module fits an analytic family
(gaussian / moffat / lorentzian / gaussian_halo) jointly across many
reference crops and stores four scalars -- and because it stores a
FORMULA, it is grid-independent: evaluate it at any voxel size, any
window, any sub-voxel offset, with no resampling. This module stores a
3D ARRAY, which buys the shapes no analytic family can express
(aberration, axial asymmetry, a halo that is not radially symmetric) and
pays for them with that independence. A template is tied to the grid it
was measured on, and nothing raises when the grid changes -- it simply
misregisters. So `load` REFUSES a voxel-size mismatch rather than
interpolating, and every field the registration depends on is stored
next to the array.

WHAT IS STORED, and why each field is not optional:

  mean           the template. Zero-mean, unit-L2, which is what
                 normalised cross-correlation needs. A peak-normalised
                 template used as if it were zero-mean scores wrongly and
                 silently.
  components     optional PCA basis, and often empty. On the first set
                 measured here PC1 held 79% of the residual and PC2 held
                 1.8% -- after the mean there was one axis and then
                 noise, so a basis nobody uses is a liability rather than
                 a feature. `explained_var` is stored so a reader can see
                 whether it earned its place.
  voxel_um       the grid. Refused on mismatch.
  centre         where the emitter sits IN THE ARRAY, in voxels, possibly
                 fractional. Half a voxel of error here is inherited by
                 every position the matched filter ever reports.
  normalisation  named, not assumed.
  labels         WHICH VERDICTS MADE THIS. The template is now a derived
                 artifact of a labelled set -- reviewers, session files,
                 how positives were chosen, how many were contested -- so
                 "which labels produced this PSF" has an answer.
  analytic_ref   the psf.py entry this was compared against, and the
                 residual between them. If the measured template looks
                 like the calibrated analytic one, that is confidence; if
                 it does not, the difference IS the result.

THE LOOP THIS CLOSES:

    spotcheck app -> verdicts -> labels() positives
                                     |
                      bundle / recut (same voxels)
                                     |
                 sub-voxel align + average -> psf/<label>.h5
                                     |
              +----------------------+----------------------+
              v                                             v
        matched filter                                drift monitor
   (finds several spots in one box)         median NCC per FOV/hybe; a
                                            drop is REPORTED, never
                                            silently re-fitted -- see
                                            `drift`.

Re-deriving the PSF per image is not an option, and not for cost: it
needs to know which spots are real, which is what the detector is for.
"""
import hashlib
import json
import os
import time

import numpy as np
from scipy import ndimage

# The MATCHING template's half-widths: 7x7x11. Not the size the spots are
# measured in -- that is dataset.DEFAULT_R/RZ and stays 15x15x25 -- but
# the size the filter is rendered at, which is a different question with
# a different answer.
#
# THREE THINGS DECIDE IT, and the first two contradicted each other until
# the third was measured on real pixels.
#
#   RECALL. In WHITE noise a bigger filter wins at low SNR: at 5 false
#   positives per field, 7x7x11 gives up 21.3 pp against 15x15x25 at
#   SNR 2.0 and 8.3 pp at 2.5. In SPATIALLY CORRELATED background the
#   ordering inverts completely -- 69.8% against 1.3% at SNR 2.5. Real
#   cells are neither. MEASURED by planting the calibrated PSF into 15
#   real MP58/RNA crops that carry no gate-passing candidate, recall at
#   5 FP/field:
#
#       SNR    5x5x9  7x7x11  9x9x13  11x11x17  15x15x25
#       2.0     42.2    43.1    42.6      52.2      54.4
#       3.0     98.5    98.4    96.7      89.8      87.0
#       4.0    100.0    98.6   100.0     100.0      95.8
#       6.0    100.0   100.0   100.0     100.0     100.0
#
#   The big filter leads only at SNR 2, where every size fails anyway,
#   and trails once detection starts working at all.
#
#   WHAT IS NOT SETTLED: 5x5x9 ties 7x7x11 on real-background recall and
#   beats it on close pairs (91% against 73% at 8 planes), so the case
#   for going smaller still is open. 7x7x11 is taken as the more
#   conservative of the two because 5x5x9 falls furthest behind in the
#   white-noise arm at SNR 2-2.5, and 60 trials on one bundle is not
#   enough to spend that on. Revisit when a review has produced labels
#   across many hybes.
#
#   RESOLVING TWO SPOTS. This is what the whole architecture is for: the
#   pillar was centred on one thing by peak_local_max and argmax, and a
#   second locus a few planes away sits inside the same box the
#   classifier judged. Two spots at one (y, x), SNR 6, both to be found
#   within 2 planes, 60 trials, each template at its own 4.5 sigma:
#
#       separation   4pl  5pl  6pl  8pl  10pl  14pl
#       5x5x9         0%   1%  33%  91%  100%  100%
#       7x7x11        0%   1%   0%  73%  100%  100%
#       9x9x13        0%   0%   0%  43%   98%  100%
#       11x11x17      0%   0%   0%  18%   81%  100%
#
#   Smaller resolves closer, monotonically, and nothing resolves below
#   about 6 planes. Two earlier attempts at this table -- one saying 43%
#   at 8 planes for 7x7x11, one saying 100% -- were both run at a fixed
#   raw threshold and are withdrawn; the number is 73%.
#
#   WHAT CAN BE SCORED AT ALL. ncc scores only where the template fully
#   fits. On a 120-plane stack 7x7x11 scores 110 planes and 15x15x25
#   scores 96 -- a fifth of every stack with no score, on the axis where
#   inter-round drift runs to 11 planes.
#
# NOT SPEED. ncc is FFT-based, so its cost is set by the VOLUME: 0.54 to
# 0.61 s for every size from 5x5x9 to 15x15x25 on a 3.2 Mvoxel field, and
# the total is actually lowest for the biggest template because there are
# fewer noise maxima to extract. An earlier claim here that the small
# template was faster came from timing a box rather than a field, and is
# withdrawn.
#
# AND NOT A FIXED THRESHOLD. NCC noise sd goes as 1/sqrt(n_voxels), so one
# threshold means a different false-alarm rate for every size -- which is
# what made an earlier sweep report 229 false positives for 5x5x9 and
# call it a property of the template. Every comparison above reads recall
# at a fixed FALSE-POSITIVE BUDGET; the thresholds that budget picks are
# a constant 4.3-4.5 sigma across all sizes.
DEFAULT_R = 3          # matching template half-width in y, x -> 7
DEFAULT_RZ = 5         # matching template half-depth -> 11
NORMALISATION = 'zero-mean, unit-L2'


# -- building -------------------------------------------------------------

def recentre(patch, order=3):
    """Shift a patch so its emitter sits on the array centre, sub-voxel.

    A three-point parabola per axis about the brightest voxel. Without
    this the average of many spots is the average of many OFFSETS, which
    is a blurred PSF wider than any spot in it -- and the leading
    principal components come out as the derivatives of the mean, i.e.
    the mis-centring itself rather than any property of the optics.
    """
    p = np.asarray(patch, float)
    i = np.unravel_index(int(np.nanargmax(p)), p.shape)
    sub = []
    for ax in range(p.ndim):
        c = i[ax]
        if c <= 0 or c >= p.shape[ax] - 1:
            sub.append(0.0)
            continue
        rest = tuple(v for j, v in enumerate(i) if j != ax)
        a = [float(np.take(p, c + k, axis=ax)[rest]) for k in (-1, 0, 1)]
        den = a[0] - 2 * a[1] + a[2]
        sub.append(0.0 if den == 0 else 0.5 * (a[0] - a[2]) / den)
    mid = np.array(p.shape) // 2
    shift = [(mid[k] - i[k]) - sub[k] for k in range(p.ndim)]
    return ndimage.shift(p, shift, order=order, mode='nearest')


def normalise(patch):
    """Zero-mean, unit-L2 -- the form NCC scores against."""
    p = np.asarray(patch, float)
    p = p - p.mean()
    n = float(np.linalg.norm(p))
    return p / n if n > 0 else p


def fit_from_spots(patches, voxel_um, families=None, verbose=False):
    """Fit an analytic PSF to boxes humans confirmed. THE DENOISING STEP.

    The average of a few hundred boxes is a picture of the PSF plus
    1/sqrt(N) of noise everywhere, and a matched filter pays for that
    noise directly: detection SNR is the cosine between the filter and
    the true shape, so any noise in the filter comes straight off it. A
    four-parameter surface fitted through those boxes has no noise at
    all, and -- because it is a FORMULA -- it can be rendered at any
    voxel size and any window, which a stored array cannot.

    psf.calibrate already does the hard part: one shape shared across
    every box, with amplitude, centre and background profiled out per
    box. What is new here is the INPUT -- spots a person confirmed,
    rather than crops from the reference hybe -- which is what closes the
    loop from the review app back to the calibration.

    EACH BOX IS NORMALISED FIRST. calibrate scores by residual sum of
    squares, which scales with amplitude, so without this the brightest
    few cells would set the shape for all of them.

    A REFIT FROM A SMALL SET IS NOT AN IMPROVEMENT, and the first one run
    here was slightly worse than what it would have replaced. Fitted to
    120 human-confirmed MP58/RNA readout spots it converged on
    gaussian_halo -- the same family as the installed universal default
    -- with sigma_z 11% larger and sigma_xy 3% smaller. Those parameters
    trade off against each other: the two shapes agree at COSINE 0.9938,
    so the 11% is not a physical difference and reading gaussian_halo's
    parameters one at a time is misleading. Scored on the spots
    themselves the refit LOST:

        NCC to the 207 confirmed spots   universal 0.3380, refit 0.3355
        refit higher on 43 of 207 (21%)
        against held-out averages, 20 splits: -0.0108 +- 0.0011

    The universal default is a mean over three experiments and thousands
    of crops and carries almost no noise; a fit to 120 boxes follows
    theirs. Same lesson as the matched filter: for a filter, cleaner
    beats better-fitted.

    So this is a CONFIRMATION path before it is a replacement one --
    running it and getting the installed shape back is the good outcome.
    Re-run it when a review has produced labels across many hybes, and
    install only if it wins on held-out spots.

    Returns (family, params, scores). `scores` keeps every candidate
    family's result, so what LOST is visible too.
    """
    from . import psf as P
    crops = []
    for p in patches:
        a = np.asarray(p, float)
        n = float(np.linalg.norm(a - np.median(a)))
        crops.append(a / n if n > 0 else a)
    res = P.calibrate(crops, voxel_um=voxel_um, families=families,
                      verbose=verbose)
    best = res.get('best')
    if not best:
        raise ValueError('no PSF family could be fitted to these spots')
    return best, res[best]['params'], res


def render(family, params, r=DEFAULT_R, rz=DEFAULT_RZ,
           voxel_um=(0.208, 0.208, 0.2), dy=0.0, dx=0.0, dz=0.0):
    """A clean template of any size from a fitted shape.

    This is what the grid-independence is FOR: the matching template does
    not have to be the size the spots were measured in, and a sub-voxel
    offset costs a re-render rather than an interpolation.
    """
    from . import psf as P
    from . import psf_library as PL
    if isinstance(params, dict):
        st = PL.shape_tuple({'family': family, 'params': params})
        if st is None:
            raise ValueError(f'unknown PSF family {family!r}')
        family, shape_params = st
    else:
        shape_params = tuple(params)
    yy, xx, zz = np.mgrid[0:2 * r + 1, 0:2 * r + 1, 0:2 * rz + 1].astype(float)
    return P.evaluate(family, shape_params,
                      (yy - r - dy) * voxel_um[0],
                      (xx - r - dx) * voxel_um[1],
                      (zz - rz - dz) * voxel_um[2])


def build(patches, voxel_um, n_components=0, align=True):
    """Patches of confirmed spots -> (mean, components, explained_var).

    `patches` are background-subtracted boxes of one fixed shape.
    """
    P = np.asarray(patches, float)
    if P.ndim != 4 or len(P) == 0:
        raise ValueError('expected (n, ny, nx, nz) patches')
    rows = []
    for p in P:
        q = recentre(p) if align else p
        rows.append(normalise(q).ravel())
    X = np.asarray(rows)
    mean = normalise(X.mean(0).reshape(P.shape[1:]))
    k = int(n_components)
    if k <= 0:
        return mean, np.zeros((0,) + P.shape[1:]), np.zeros(0)
    Xc = X - X.mean(0)
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = (s ** 2) / max(float((s ** 2).sum()), 1e-30)
    k = min(k, vt.shape[0])
    comps = vt[:k].reshape((k,) + P.shape[1:])
    return mean, comps, var[:k]


# -- the file -------------------------------------------------------------

def _digest(mean, comps):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mean, np.float32).tobytes())
    h.update(np.ascontiguousarray(comps, np.float32).tobytes())
    return h.hexdigest()


def save(path, mean, voxel_um, components=None, explained_var=None,
         centre=None, n_spots=0, source=None, labels=None,
         analytic_ref=None):
    """Write a bank entry atomically (.part + os.replace)."""
    import h5py
    mean = np.asarray(mean, np.float32)
    comps = np.zeros((0,) + mean.shape, np.float32) if components is None \
        else np.asarray(components, np.float32)
    if centre is None:
        centre = tuple(float(s // 2) for s in mean.shape)
    doc = dict(
        voxel_um=[float(v) for v in voxel_um],
        shape=[int(s) for s in mean.shape],
        centre=[float(c) for c in centre],
        normalisation=NORMALISATION,
        n_spots=int(n_spots),
        explained_var=([] if explained_var is None
                       else [float(v) for v in explained_var]),
        source=source or {}, labels=labels or {},
        analytic_ref=analytic_ref or {},
        built_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
        sha256=_digest(mean, comps))
    tmp = str(path) + '.part'
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with h5py.File(tmp, 'w') as f:
        f.create_dataset('mean', data=mean, compression='gzip')
        f.create_dataset('components', data=comps, compression='gzip')
        for k, v in doc.items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    os.replace(tmp, str(path))
    return str(path)


def load(path, voxel_um=None, tol=1e-6):
    """Read a bank entry. REFUSES a grid it was not measured on.

    Silently resampling would be the worst option available: the template
    would still correlate, still return peaks, and every position would
    carry a bias nobody could see.
    """
    import h5py
    with h5py.File(str(path), 'r') as f:
        mean = np.asarray(f['mean'][...], float)
        comps = np.asarray(f['components'][...], float)
        meta = {}
        for k, v in f.attrs.items():
            if isinstance(v, (bytes, str)):
                s = v.decode() if isinstance(v, bytes) else v
                try:
                    meta[k] = json.loads(s)
                except (ValueError, TypeError):
                    meta[k] = s
            else:
                meta[k] = v.tolist() if hasattr(v, 'tolist') else v
    if voxel_um is not None:
        got = np.asarray(meta.get('voxel_um', []), float)
        want = np.asarray(voxel_um, float)
        if got.shape != want.shape or np.any(np.abs(got - want) > tol):
            raise ValueError(
                f'this PSF was measured on voxel {got.tolist()} um and you '
                f'asked for {want.tolist()}. Resampling it would keep '
                'correlating and quietly bias every position; measure a new '
                'one for this grid instead.')
    return mean, comps, meta


def centre_crop(patch, r=DEFAULT_R, rz=DEFAULT_RZ):
    """The central (2r+1, 2r+1, 2rz+1) of a stored template.

    A BANK IS STORED AT THE SIZE IT WAS MEASURED AT, WHICH IS NOT THE
    SIZE A SEARCH WANTS, and until this existed the two were silently
    the same number. tools/train_spotmodel.py averages the classifier's
    own boxes, so psf_bank.h5 holds a 15 x 15 x 25 mean; ncc() scores
    only where the template fits, so matching a 15 x 15 x 105 pillar
    with it leaves a valid region of 1 x 1 x 81 -- ONE lateral position.

    MEASURED on 200 confirmed-spot pillars of MP58/RNA, every "match"
    that came back was a z peak in the pillar's own centre column, and a
    second emitter three pixels to the side could not be found at any
    threshold. It did not look like a failure: the pillars each returned
    a plausible hit at a plausible depth.

    Cropping is exact, not resampling: these are the same voxels on the
    same grid, so load()'s refusal to resample is not being routed
    around. Growing one is refused -- the voxels are not there to invent.
    """
    p = np.asarray(patch, float)
    cy, cx, cz = (n // 2 for n in p.shape)
    r, rz = int(r), int(rz)
    if p.shape[0] < 2 * r + 1 or p.shape[1] < 2 * r + 1 \
            or p.shape[2] < 2 * rz + 1:
        raise ValueError(
            f'this template is {p.shape} and a {2*r+1} x {2*r+1} x '
            f'{2*rz+1} one was asked for. Cropping can only make a '
            'template smaller; measure or render a larger one.')
    return p[cy - r:cy + r + 1, cx - r:cx + r + 1, cz - rz:cz + rz + 1]


# -- matching -------------------------------------------------------------

# TWO MATCHES CLOSER THAN THE OPTICS CAN RESOLVE ARE ONE MATCH.
#
# A matched filter will happily place two peaks a pixel apart and there
# is no evidence in the image that they are two things. 2 px laterally
# is the floor -- MEASURED from the MP58 store's own calibration,
# sigma_xy is 0.66 px, so 2 px is 3.0 sigma_xy.
#
# AND THE SAME COUNT OF SIGMA AXIALLY, WHICH IS NOT THE SAME COUNT OF
# VOXELS. A lateral-only bound looks right and is not: MEASURED over
# 2,639 match pairs from 300 pillars, 135 sit within 2 px laterally and
# 130 OF THOSE ARE MORE THAN 5 PLANES APART -- median 17.3, which is
# 7 sigma_z. Merging those would not be declining to over-resolve, it
# would be discarding pairs the microscope separates perfectly well
# along its coarser axis. sigma_z is 2.35 planes, so 3.0 sigma is 7.1.
#
# THIS LIVED IN THE REVIEW APP and does not belong there: the same bound
# governs what a LOCALIZE ENGINE may report, and a second copy of it is
# the divergence this codebase keeps having to hunt down.
MERGE_LATERAL_PX = 2.0
MERGE_AXIAL_PLANES = 7.1
DEFAULT_SIGMA_XY_PX = 0.66
DEFAULT_SIGMA_Z_PLANES = 2.35


# HOW CLOSE THE MEASURED TEMPLATE MUST SIT TO THE CALIBRATED OPTICS.
#
# A LOW COSINE IS USUALLY A SHORTAGE OF LABELS, NOT A BROKEN MICROSCOPE,
# and that is what makes it a useful thing to warn on: the template is a
# MEAN, so with few spots it is mostly the noise of whichever few were
# confirmed. MEASURED by titrating the MP58/RNA confirmed set, 20 random
# draws at each size, against the store's own gaussian_halo calibration:
#
#     N spots    cosine     sd        range over 20 draws
#         5      0.736    0.053      [0.582, 0.813]
#        10      0.791    0.037      [0.721, 0.857]
#        20      0.858    0.020      [0.813, 0.899]
#        40      0.897    0.010      [0.875, 0.911]
#        80      0.913    0.005      [0.901, 0.924]
#       160      0.923    0.004      [0.914, 0.928]
#       640      0.930    0.002      [0.927, 0.935]
#      1157      0.931
#
# It climbs monotonically and PLATEAUS AT 0.931, not at 1.0 -- that
# residual is real optics-versus-model difference and no amount of
# labelling closes it, so the bound has to sit below it.
#
# 0.90 falls between N=40 and N=80, and at 80 not one of 20 draws came in
# under it. So "below 0.90" reads as: fewer than about eighty confirmed
# spots went into this template -- go and verify more -- or, if there are
# plenty, the calibrated PSF does not describe this data and that is the
# more interesting problem.
PSF_COSINE_MIN = 0.90


# THE FAMILY IS TESTED, NOT ASSUMED. The installed calibration is a
# READOUT PSF -- for MP58 the universal default, gaussian_halo, a mean
# over three experiments' readout channels -- and every bank was
# compared against it, whatever the bank was measured on. A bank of 905
# confirmed DNA ch555 spots (the fiducial channel) came out at cosine
# 0.776 to it, and the warning said "check the store calibration and
# the voxel size", which was the wrong diagnosis: neither had changed.
# Fiducial emitters are simply not readout emitters, and a plain
# Gaussian usually describes them better than a core-plus-halo. So each
# family is fitted to the template ITSELF: if one of them reaches the
# floor, the template is a clean, simple shape and the low cosine to the
# installed reference is a difference of EMITTER, not of labels or
# optics -- and the bank's own fit, not the readout calibration, is what
# should set its anisotropy.
FIT_FAMILIES = ('gaussian', 'gaussian_halo')


def fit_families(mean, voxel_um, families=FIT_FAMILIES):
    """Fit each analytic family's shape to a measured template.

    Returns {'fits': {family: {'params', 'cosine', 'plausible',
    'warnings'}}, 'best': family}. The objective is the cosine between
    the normalised rendered shape and the normalised template on the
    template's own grid -- the same number cosine_warning judges -- so
    the best fit is, by construction, the best any shape of that family
    can do here. Each family starts from its declared initial guess and
    from the template's own second moments, and the better of the two
    is kept; the best family is the best-scoring PLAUSIBLE one, as
    psf.select_best does for a calibration.
    """
    from scipy.optimize import minimize
    from . import psf as P
    m = np.asarray(mean, float)
    if m.ndim != 3:
        raise ValueError('expected a (ny, nx, nz) template')
    ny, nx, nz = m.shape
    r, rz = ny // 2, nz // 2
    target = normalise(m).ravel()
    vox = tuple(float(v) for v in voxel_um)

    # a moment-based second start: the template's own widths
    w = m - float(np.percentile(m, 10))
    w = np.clip(w, 0, None)
    w = w / max(float(w.sum()), 1e-12)
    yy, xx, zz = np.mgrid[0:ny, 0:nx, 0:nz].astype(float)
    s_lat = float(np.sqrt(0.5 * ((w * (yy - r) ** 2).sum()
                                 + (w * (xx - nx // 2) ** 2).sum()))) * vox[0]
    s_ax = float(np.sqrt((w * (zz - rz) ** 2).sum())) * vox[2]

    def cosine(family, theta):
        try:
            vol = render(family, tuple(float(t) for t in theta), r=r, rz=rz,
                         voxel_um=vox)
        except Exception:                                   # noqa: BLE001
            return -1.0
        v = normalise(np.asarray(vol, float)).ravel()
        return float(v @ target)

    fits = {}
    for family in families:
        _fn, names, init, bounds = P.FAMILIES[family]
        starts = [list(init)]
        alt = list(init)
        alt[0] = float(np.clip(s_lat, bounds[0][0], bounds[0][1]))
        alt[1] = float(np.clip(s_ax, bounds[1][0], bounds[1][1]))
        starts.append(alt)
        best = None
        for x0 in starts:
            res = minimize(lambda t: 1.0 - cosine(family, t),
                           np.array(x0, float), method='Nelder-Mead',
                           bounds=bounds,
                           options={'xatol': 1e-4, 'fatol': 1e-6,
                                    'maxiter': 400})
            if best is None or res.fun < best.fun:
                best = res
        params = {k: float(v) for k, v in zip(names, best.x)}
        ok, why = P.plausible(family, params)
        fits[family] = {'params': params, 'cosine': float(1.0 - best.fun),
                        'plausible': bool(ok), 'warnings': list(why or [])}
    usable = [f for f in fits if fits[f]['plausible']] or list(fits)
    # THE SIMPLEST FAMILY WITHIN A HAIR OF THE BEST WINS. A Gaussian is
    # also a gaussian_halo with halo_frac 0, so on a Gaussian template
    # the two tie to the last decimal and floating point decides -- and
    # it decided for the four-parameter shape. The tolerance is far
    # below any real difference (the RNA readout bank separates them by
    # 0.05, the DNA ch555 bank by 0.03).
    top = max(fits[f]['cosine'] for f in usable)
    close = [f for f in usable if fits[f]['cosine'] >= top - 1e-3]
    best = min(close, key=lambda f: (len(P.FAMILIES[f][1]),
                                     -fits[f]['cosine']))
    return {'fits': fits, 'best': best}


def best_fit(analytic_ref):
    """The bank's own best analytic fit from its reference, or {}."""
    return (analytic_ref or {}).get('best_fit') or {}


def cosine_warning(analytic_ref, n_spots=None, floor=PSF_COSINE_MIN):
    """A sentence when the template does not match the optics, else None.

    Says WHICH of the causes to look at, because the answer is different
    work: more review, a re-calibration, or -- when a fitted family
    describes the template well -- nothing at all, because the bank was
    measured on another kind of emitter than the installed calibration.
    """
    ref = analytic_ref or {}
    c = ref.get('cosine_to_measured')
    bf = best_fit(ref)
    if c is None:
        return ('no analytic PSF to compare against -- train_spotmodel was '
                'run without --storage-path, so nothing checked this '
                'template against the calibrated optics, and '
                'resolution_bound falls back to its default anisotropy '
                "rather than this experiment's.")
    if float(c) >= float(floor):
        return None
    n = int(n_spots) if n_spots else None
    if bf.get('cosine') is not None and float(bf['cosine']) >= float(floor):
        pr = bf.get('params') or {}
        why = (f"but a fitted {bf.get('family')} describes the template at "
               f"cosine {float(bf['cosine']):.3f} (sigma_xy "
               f"{1000 * float(pr.get('sigma_xy_um', float('nan'))):.0f} nm, "
               f"sigma_z {1000 * float(pr.get('sigma_z_um', float('nan'))):.0f}"
               f" nm), so this is neither a label shortage nor a microscope "
               f"problem: the emitters this bank was measured on are not the "
               f"ones the installed {ref.get('family')} was calibrated on -- "
               f"a fiducial-channel bank, most likely. The bank itself is the "
               f"PSF here, and its own fit sets its anisotropy."
               + (f' ({n} confirmed spots went into it.)' if n else ''))
        return (f'PSF template cosine to the installed {ref.get("family")} '
                f'is {float(c):.3f}, below {float(floor):.2f}, {why}')
    if n is not None and n < 80:
        why = (f'{n} confirmed spots went into it, and MEASURED on MP58/RNA '
               f'a template needs about 80 before this settles (at 40 the '
               f'cosine is 0.897 +- 0.010, at 80 it is 0.913 +- 0.005). '
               f'VERIFY MORE SPOTS -- this is very likely a label shortage '
               f'rather than a microscope problem.')
    else:
        why = (f'{n} confirmed spots went into it, which is enough for this '
               f'to have settled, so the calibrated PSF may not describe '
               f'this data. Check the store calibration and the voxel size.'
               if n is not None else
               'check both the number of confirmed spots and the store '
               'calibration.')
    fitted = (f" No fitted family reaches {float(floor):.2f} either (best "
              f"{bf.get('family')} at {float(bf['cosine']):.3f}), so the "
              f"template is not a clean simple shape."
              if bf.get('cosine') is not None else '')
    return (f'PSF template cosine to the calibrated optics is {float(c):.3f}, '
            f'below {float(floor):.2f}. {why}{fitted}')


def resolution_bound(meta=None, lateral_px=MERGE_LATERAL_PX):
    """(lateral px, axial planes) below which two matches are one.

    Takes the anisotropy from the BANK'S OWN analytic reference when it
    carries one, so a bank measured on another objective or z step
    brings its own ratio rather than inheriting MP58's. The lateral
    bound is the number that was chosen; the axial one is the same
    count of sigma.

    THE BANK'S OWN FIT WINS over the installed calibration whenever it
    describes the template better (see fit_families): a fiducial-channel
    bank at cosine 0.776 to the readout gaussian_halo would otherwise
    take the readout's sigma_z/sigma_xy for emitters that are not
    readouts.
    """
    lat = float(lateral_px)
    ref = (meta or {}).get('analytic_ref') or {}
    pr = ref.get('params') or {}
    bf = best_fit(ref)
    try:
        if bf.get('params') and (
                ref.get('cosine_to_measured') is None
                or float(bf.get('cosine', -1.0))
                > float(ref.get('cosine_to_measured'))):
            pr = bf['params']
    except (TypeError, ValueError):
        pass
    vx = (meta or {}).get('voxel_um')
    try:
        sxy = float(pr['sigma_xy_um']) / float(vx[0])
        sz = float(pr['sigma_z_um']) / float(vx[2])
        if sxy > 0 and sz > 0:
            return lat, lat * (sz / sxy)
    except (KeyError, TypeError, IndexError, ZeroDivisionError, ValueError):
        pass
    return lat, float(MERGE_AXIAL_PLANES)


def merge_unresolvable(hits, brightness, lateral_px=MERGE_LATERAL_PX,
                       axial_planes=MERGE_AXIAL_PLANES):
    """Collapse matches the optics cannot separate. BRIGHTEST SURVIVES.

    `brightness` returns the image value at a hit -- not its NCC score.
    Those rank a pair differently 42% of the time, and "which of these
    is the real spot" is a question about the picture rather than about
    which position the filter liked best.

    Greedy from the brightest down, so a survivor is never suppressed by
    something dimmer that it in turn suppresses.

    Works on anything with .y .x .z -- a LocalizedSpot or a plain
    namedtuple -- because the review app and the engine both need it.
    """
    out = []
    for h in sorted(hits, key=lambda k: -float(brightness(k))):
        if any(np.hypot(h.y - k.y, h.x - k.x) < lateral_px
               and abs(h.z - k.z) < axial_planes for k in out):
            continue
        out.append(h)
    return out


def ncc(volume, template, eps=1e-9):
    """Normalised cross-correlation, same shape as `volume`.

    Normalised so the score is a SHAPE match and not a brightness one: a
    threshold on it means the same thing in a dim cell and a bright one,
    which a plain convolution does not.

    SCORED ONLY WHERE THE TEMPLATE FULLY FITS, and zero elsewhere. The
    first version of this let the two halves of the ratio disagree about
    the border -- fftconvolve zero-pads, uniform_filter replicates -- so
    the numerator and the denominator were computed over different data
    within half a template of every face. The result was not a small edge
    artefact: on a synthetic box with three planted spots it returned 345
    peaks, every one of them scoring exactly 1.000, all of them against
    the walls and none of them on a spot. A partial window is not a
    weaker match, it is a different question, and the honest answer to it
    is silence.
    """
    from scipy import signal
    v = np.asarray(volume, float)
    t = np.asarray(template, float)
    if any(a < b for a, b in zip(v.shape, t.shape)):
        return np.zeros_like(v)
    t = t - t.mean()
    tn = float(np.linalg.norm(t))
    if tn == 0:
        return np.zeros_like(v)
    t = t / tn
    n = float(t.size)
    # A zero-mean template makes the mean term vanish exactly, so the
    # numerator is the plain correlation.
    num = signal.fftconvolve(v, t[::-1, ::-1, ::-1], mode='same')
    s1 = ndimage.uniform_filter(v, size=t.shape, mode='constant')
    s2 = ndimage.uniform_filter(v * v, size=t.shape, mode='constant')
    var = np.clip(s2 - s1 * s1, 0.0, None)
    out = num / (np.sqrt(var * n) + eps)
    valid = np.zeros(v.shape, bool)
    sl = tuple(slice(s // 2, dim - (s - 1) // 2)
               for dim, s in zip(v.shape, t.shape))
    valid[sl] = True
    return np.where(valid, np.clip(out, -1.0, 1.0), 0.0)


def _parabolic(v, i):
    """Sub-voxel offset of a peak from three samples."""
    if i <= 0 or i >= len(v) - 1:
        return 0.0
    a, b, c = float(v[i - 1]), float(v[i]), float(v[i + 1])
    den = a - 2 * b + c
    return 0.0 if den == 0 else float(np.clip(0.5 * (a - c) / den, -1, 1))


# How many sigma above the noise a score has to reach to be a spot.
#
# THE THRESHOLD IS IN SIGMA, NOT IN NCC. NCC noise has sd 1/sqrt(n) for a
# template of n voxels, so one raw cut-off means a different false-alarm
# rate for every template size: 0.3 is 7 sigma for a 7x7x11 filter and 22
# sigma for a 15x15x25 one. That is not a stricter setting, it is a
# different experiment -- it made an earlier sweep report 229 false
# positives for 5x5x9 and call it a property of the template, and it made
# the engine's own default miss one of two spots it had just been shown.
#
# AND THE SIGMA IS MEASURED, NOT DERIVED. 1/sqrt(n) is the NCC noise of
# WHITE noise, and a cell is not white: measured over 623,295 scored
# voxels of real MP58/RNA background, the NCC sd is 0.0578 against a
# theoretical 0.0431 -- 1.34x wider. A cut-off of 4.5 theoretical sigma
# is therefore 3.35 REAL sigma, and it let about 26 voxels per pillar
# over the line. That is what put four to seven "matches" in pillars
# holding one spot, on flat background, in the first panel drawn from
# this.
#
# So the threshold comes from the SCORE VOLUME'S OWN noise, by MAD,
# which is immune to the handful of peaks being looked for and follows
# whatever the background actually does. 5.0 of those puts roughly one
# spurious peak in a 15 x 15 x 105 pillar.
K_SIGMA = 5.0


def noise_sd(score):
    """Robust sd of an NCC volume's background, ignoring its peaks.

    MAD rather than std: the spots are in there, and a handful of 0.9s
    would drag a plain standard deviation up and raise the threshold
    that is meant to find them.
    """
    v = np.asarray(score, float)
    v = v[v != 0]
    if v.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def sigma_threshold(template, k=K_SIGMA, score=None):
    """The NCC cut-off at `k` sigma.

    With a score volume, `k` sigma of ITS noise. Without one, the
    white-noise fallback 1/sqrt(n) -- which is optimistic on real cells
    by about a third, and is here only so a caller with no volume in
    hand still gets a number.
    """
    if score is not None:
        sd = noise_sd(score)
        if sd > 0:
            return float(np.median(np.asarray(score)[np.asarray(score) != 0])
                         + k * sd)
    n = int(np.asarray(template).size)
    return float(k) / np.sqrt(max(n, 1))


def match(volume, templates, min_distance=3, threshold=None, n_max=None,
          k_sigma=K_SIGMA):
    """Every place in `volume` that looks like one of `templates`.

    Returns [(y, x, z, score), ...] sub-voxel, strongest first.

    THIS IS WHAT HANDLES A BOX WITH SEVERAL SPOTS. The score volume has
    one maximum per emitter, so the answer is however many are there --
    no mixture model, no decision about how many components to fit.
    """
    from skimage.feature import peak_local_max
    v = np.asarray(volume, float)
    ts = [templates] if np.asarray(templates).ndim == 3 else list(templates)
    score = None
    for t in ts:
        s = ncc(v, t)
        score = s if score is None else np.maximum(score, s)
    if threshold is None:
        # The noise of THIS volume, not of an idealised one.
        threshold = sigma_threshold(ts[0], k_sigma, score=score)
    peaks = peak_local_max(score, min_distance=int(min_distance),
                           threshold_abs=float(threshold))
    out = []
    for (y, x, z) in peaks:
        dy = _parabolic(score[:, x, z], y)
        dx = _parabolic(score[y, :, z], x)
        dz = _parabolic(score[y, x, :], z)
        out.append((float(y + dy), float(x + dx), float(z + dz),
                    float(score[y, x, z])))
    out.sort(key=lambda r: -r[3])
    return out if n_max is None else out[:int(n_max)]


def drift(patches, mean, align=True):
    """How well a set of spots still matches the stored template.

    Returns the per-spot cosine similarities. A falling median is the
    signal that the optics moved -- report it; do NOT quietly re-fit,
    because a PSF that silently follows the data explains everything and
    detects nothing.
    """
    m = normalise(np.asarray(mean, float)).ravel()
    out = []
    for p in np.asarray(patches, float):
        q = recentre(p) if align else p
        out.append(float(normalise(q).ravel() @ m))
    return np.asarray(out)
