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

DEFAULT_R = 7          # template half-width in y, x -> 15
DEFAULT_RZ = 12        # template half-depth -> 25
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


def render(family, params, r, rz, voxel_um, dy=0.0, dx=0.0, dz=0.0):
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


# -- matching -------------------------------------------------------------

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


def match(volume, templates, min_distance=3, threshold=0.3, n_max=None):
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
