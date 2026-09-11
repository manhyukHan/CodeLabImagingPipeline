"""
The fiducial by 3D image correlation against the reference hybe.

WHY THIS EXISTS
---------------
ChrTracer3 (MATLAB, ChrTracer3_FitSpots -> Register3D) does not fit a
shape to each hybe's fiducial and subtract. It cuts a small cube around
the reference hybe's fiducial peak, cuts the SAME cube from the other
hybe, and takes the translation at which the two images match best as
that hybe's drift -- template matching, translation only (maxXYdrift 4 px,
maxZdrift 6 planes there). The readout is then fitted in the translated
crop.

This pipeline's v1/v2 fiducial is the other approach: a Gaussian fit per
hybe, delta = fit(reference) - fit(hybe). shapefree.py makes the case for
registration -- the fiducial is an extended object whose shape cancels
between two rounds -- and shapefree.shift_yxz was measured 14.8% WORSE
than the fit on the median of 305 replicate pairs. That estimator
correlated the whole 17x17x31 slab, so every neighbouring structure in
the crop voted. This module is the MATLAB variant for a fair A/B: a tight
template around the reference fiducial, matched over a search region no
larger than the drift gate allows, scored by the same replicate distance
on the same tracer (tools/fiducial_ab.py, arm 'v2+xc').

WHAT IT COMPUTES
----------------
    NCC -- Pearson correlation over the template window -- at every
    full-overlap translation within the allowed shift (skimage's
    match_template, FFT-based, the same quantity MATLAB's
    template_matching returns as I_NCC). The integer peak is the shift;
    the sub-voxel part comes from plain cross-correlation upsampled x20
    between the template and the window at the peak (shapefree measured
    plain CC at 0.006 vox mean error on synthetic blobs, phase
    correlation 40x worse), with a 3-point parabola as the fallback.

    NCC is invariant to an additive offset and a gain, so MATLAB's
    edge-quantile background subtraction is not needed for the map; it
    is applied (as a median) only where the sub-voxel step needs a
    zero background.

SIGN CONVENTION
---------------
`pos` is where the reference's fiducial sits in the MOVING cube's own
coordinates. Nothing here computes a delta: the tracer takes positions
from every fiducial method and forms baseline - fid itself, so the
correction is applied exactly the way it is for the Gaussian.

Rotation is not searched. Register3D does not either; the FOV-level
rotation is the global alignment's, and across a 9 px template even
0.5 degrees moves an edge voxel 0.04 px.
"""
from collections import namedtuple

import numpy as np

# (y, x, z) half-sizes in voxels. MATLAB: bXY = max(4, maxShiftXY) = 4,
# bZ = max(4, maxShiftZ) = 6 at a coarser z step; here the fiducial's
# measured sigma_z is ~3.8 planes (760 nm at 0.2 um), so +/-8 planes
# holds +/-2 sigma and +/-4 px holds +/-3 sigma laterally (279 nm).
DEFAULT_TEMPLATE_HALF = (4, 4, 8)
# The tracer's drift gate: 7 px, 15 planes. Searching further than the
# gate accepts would only find shifts it then refuses.
DEFAULT_MAX_SHIFT = (7, 7, 15)
# How small the template may shrink to keep the whole search range inside
# the moving crop before the range is clipped instead. MEASURED: with no
# shrinking, every learned-candidate refinement within 4 px of the display
# crop's edge was refused as 'search region smaller than the template'.
MIN_TEMPLATE_HALF = (2, 2, 4)

XcorrFit = namedtuple('XcorrFit', [
    'y', 'x', 'z',          # the fiducial in the moving cube's coordinates
    'amplitude',            # the moving cube at the nearest voxel
    'ncc',                  # peak normalized cross-correlation, [-1, 1]
    'shift',                # (dy, dx, dz) = pos - centre, exactly
    'at_bound',             # axes whose peak sat on the search boundary
    'refined',              # 'cc' (upsampled plain CC) or 'parabola'
])


def _finite_fill(a, min_finite):
    """A finite copy, holes filled with the median; None when too little
    of it was observed for a correlation to mean anything."""
    a = np.asarray(a, dtype=float)
    fin = np.isfinite(a)
    if a.size == 0 or fin.mean() < float(min_finite):
        return None
    if fin.all():
        return a
    return np.where(fin, a, float(np.median(a[fin])))


def _parabola(c_minus, c0, c_plus):
    """Sub-voxel peak offset from three samples; 0 when they do not
    bracket a maximum."""
    denom = c_minus - 2.0 * c0 + c_plus
    if not np.isfinite(denom) or denom >= 0:
        return 0.0
    return float(np.clip(0.5 * (c_minus - c_plus) / denom, -0.5, 0.5))


def _subvoxel_cc(T, W, upsample):
    """Residual translation of W relative to T, in voxels, by upsampled
    plain cross-correlation; None when it does not trust itself (more
    than 1.5 voxels, which the integer peak already excluded)."""
    from skimage.registration import phase_cross_correlation
    a = np.clip(T - np.median(T), 0.0, None)
    b = np.clip(W - np.median(W), 0.0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return None
    try:
        s, _err, _ph = phase_cross_correlation(
            a, b, upsample_factor=int(upsample), normalization=None)
    except Exception:                                    # noqa: BLE001
        return None
    s = np.asarray(s, dtype=float)
    # up to 1.5 voxels: the integer peak is off by one under noise, and
    # the upsampled CC on the window places that confidently; beyond it
    # the integer search itself excluded the answer
    if not np.all(np.isfinite(s)) or np.any(np.abs(s) > 1.5):
        return None
    # skimage returns the shift that registers W onto T, i.e.
    # reference - moving; the moving image's content sits at MINUS that.
    return -s


def register(ref_cube, ref_pos, mov_cube, centre,
             max_shift=DEFAULT_MAX_SHIFT, half=DEFAULT_TEMPLATE_HALF,
             upsample=20, min_finite=0.9):
    """
    Where the reference's fiducial sits in `mov_cube`.

    ref_cube, ref_pos : the reference hybe's crop and its fiducial (y, x, z)
                        in that crop's coordinates (sub-voxel allowed)
    mov_cube, centre  : the other hybe's crop and where the fiducial is
                        EXPECTED there, in that crop's coordinates -- the
                        reference position carried over by the crops'
                        origins, or a learned candidate to refine
    max_shift         : (y, x, z) how far from `centre` to search
    half              : (y, x, z) template half-sizes; shrunk per axis
                        where the reference position is closer to its
                        crop's edge than that

    Returns (XcorrFit, None) or (None, why). The fit's at_bound names
    the axes whose peak sat on the edge of the searched range -- a
    peak that is not bracketed is not a measurement, and the caller
    gates on it exactly as it does on a Gaussian parameter on its bound.
    """
    from skimage.feature import match_template
    ref = np.asarray(ref_cube, dtype=float)
    mov = np.asarray(mov_cube, dtype=float)
    if ref.ndim != 3 or mov.ndim != 3:
        return None, 'crops are not 3D'
    if len(half) != 3 or len(max_shift) != 3 or len(ref_pos) != 3 or len(centre) != 3:
        return None, 'ref_pos, centre, max_shift and half must each have 3 entries'
    if not all(np.isfinite(float(v)) for v in (*ref_pos, *centre, *max_shift, *half)):
        return None, 'non-finite reference position, expected placement or range'
    q = [int(round(float(v))) for v in ref_pos]
    for a in range(3):
        if not (0 <= q[a] < ref.shape[a]):
            return None, 'reference fiducial outside its crop'
    c = [int(round(float(v))) for v in centre]
    ms = [max(0, int(max_shift[a])) for a in range(3)]
    # THE TEMPLATE IS CENTRED ON THE REFERENCE POSITION. It needs at
    # least MIN_TEMPLATE_HALF of crop on every side of it: a template
    # clamped to a 1-voxel half at a crop edge is a sliver that
    # correlates with anything (measured: q on the edge gave an empty
    # slice; q one in gave a 2-row template and a confident, wrong
    # position). And it shrinks, down to the same floor, so that the
    # whole search range around the expected placement fits inside the
    # moving crop; below the floor the range is clipped instead and a
    # peak on the clipped edge is reported at_bound, like one on the
    # drift limit.
    h = [min(int(half[a]), q[a], ref.shape[a] - 1 - q[a]) for a in range(3)]
    if any(h[a] < int(MIN_TEMPLATE_HALF[a]) for a in range(3)):
        return None, 'reference fiducial too close to its crop edge for a template'
    for a in range(3):
        room = min(c[a] - ms[a], mov.shape[a] - 1 - c[a] - ms[a])
        if room < h[a]:
            h[a] = max(min(h[a], int(MIN_TEMPLATE_HALF[a])), room, 1)
    lo = [q[a] - h[a] for a in range(3)]
    T = _finite_fill(ref[lo[0]:lo[0] + 2 * h[0] + 1,
                         lo[1]:lo[1] + 2 * h[1] + 1,
                         lo[2]:lo[2] + 2 * h[2] + 1], min_finite)
    if T is None:
        return None, 'template mostly unobserved'
    if not (T.std() > 0):
        return None, 'template has no contrast'
    o_exp = [c[a] - h[a] for a in range(3)]
    r_lo = [max(0, o_exp[a] - ms[a]) for a in range(3)]
    r_hi = [min(mov.shape[a], o_exp[a] + ms[a] + T.shape[a]) for a in range(3)]
    if any(r_hi[a] - r_lo[a] < T.shape[a] for a in range(3)):
        return None, 'search region smaller than the template'
    S = _finite_fill(mov[r_lo[0]:r_hi[0], r_lo[1]:r_hi[1], r_lo[2]:r_hi[2]],
                     min_finite)
    if S is None:
        return None, 'search region mostly unobserved'
    ncc = np.asarray(match_template(S, T, pad_input=False), dtype=float)
    if ncc.size == 0 or not np.isfinite(ncc).any():
        return None, 'correlation undefined'
    ncc = np.where(np.isfinite(ncc), ncc, -1.0)
    k = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    peak = float(ncc[k])
    if not (peak > 0):
        # a flat or peak-less moving crop: match_template returns 0
        # wherever the window has no variance, and an argmax over zeros
        # is a corner, not a measurement
        return None, 'no correlation peak'
    # A PEAK ON THE EDGE OF THE MAP IS ON THE EDGE OF WHAT WAS SEARCHED,
    # whether the range ran into the drift limit or into the crop --
    # and a map one placement wide on an axis that asked for a range is
    # the same case, not a bracketed peak.
    at_bound = tuple(name for a, name in enumerate(('y', 'x', 'z'))
                     if ms[a] > 0 and (k[a] == 0 or k[a] == ncc.shape[a] - 1))
    o = [r_lo[a] + int(k[a]) for a in range(3)]
    t_int = [o[a] - o_exp[a] for a in range(3)]
    # the window is taken from S -- the SAME filled array the peak was
    # found in -- so the sub-voxel step sees the image the peak came from
    W = S[o[0] - r_lo[0]:o[0] - r_lo[0] + T.shape[0],
          o[1] - r_lo[1]:o[1] - r_lo[1] + T.shape[1],
          o[2] - r_lo[2]:o[2] - r_lo[2] + T.shape[2]]
    t_sub = _subvoxel_cc(T, W, upsample)
    refined = 'cc'
    if t_sub is None:
        refined = 'parabola'
        t_sub = np.zeros(3)
        for a in range(3):
            if 0 < k[a] < ncc.shape[a] - 1:
                idx_m = list(k)
                idx_p = list(k)
                idx_m[a] -= 1
                idx_p[a] += 1
                t_sub[a] = _parabola(ncc[tuple(idx_m)], peak, ncc[tuple(idx_p)])
    # pos = window origin + the feature's offset inside the template
    #     = (o_exp + t_int) + (ref_pos - lo) + t_sub
    #     = c + (ref_pos - q) + t_int + t_sub
    pos = [c[a] + (float(ref_pos[a]) - q[a]) + t_int[a] + float(t_sub[a])
           for a in range(3)]
    # THE SHIFT IS FROM `centre` EXACTLY, not from the rounded placement:
    # the tile prints it beside the drift gate, and a fractional
    # inter-hybe translation would otherwise leak into it.
    shift = tuple(float(pos[a] - float(centre[a])) for a in range(3))
    iy, ix, iz = (int(np.clip(round(pos[a]), 0, mov.shape[a] - 1))
                  for a in range(3))
    amp = float(mov[iy, ix, iz]) if np.isfinite(mov[iy, ix, iz]) else 0.0
    return XcorrFit(float(pos[0]), float(pos[1]), float(pos[2]), amp, peak,
                    shift, at_bound, refined), None
