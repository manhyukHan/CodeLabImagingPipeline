"""
Measuring a fiducial without pretending it is a point source.

WHY
---
The fiducial is not an emitter, it is the whole genomic region the
readouts collectively trace. Fitting it with a PSF family asks "how wide
is the point-spread function" of something that is not a point, and the
data says plainly that the question has no answer: fit a Gaussian to a
fiducial crop and the recovered sigma follows the FIT WINDOW as
sigma ~ r^0.5, with no plateau, on all four experiments measured
(0.6/0.8/1.2/1.6 um lateral radius, 110 crops each). A real Gaussian
gives sigma ~ r^0.0. The width being reported is the window's, not the
object's.

WHAT THE FIDUCIAL IS ACTUALLY FOR
---------------------------------
Exactly one number, per hybe:

    delta = baseline_fiducial_position - this_hybe_fiducial_position

which then corrects every readout in that round. So the pipeline fits two
ABSOLUTE positions with an ill-conditioned model and subtracts them, when
the quantity it wants is the DIFFERENCE. Estimating the difference
directly is better on three counts:

  * no shape model at all, so nothing to be wrong about
  * whatever the object's shape is, it is the SAME shape in both rounds,
    so it cancels instead of biasing
  * an extended object HELPS -- structure sharpens a correlation peak,
    where it only broadened a PSF fit

This module therefore offers registration (`shift_yxz`) for the job the
fiducial actually does, and shape-free moments (`moments_yxz`) for
describing its size honestly when a size is wanted.

WHAT A SHAPE-FREE SIZE MEANS
----------------------------
`moments_yxz` returns an intensity-weighted second moment -- a radius of
gyration. It assumes no profile, but it is NOT window-independent either:
any second moment of a heavy-tailed distribution grows with the aperture
it is computed over. The difference from the Gaussian fit is that the
aperture is stated and the number is a definition rather than an estimate
of a parameter that may not exist. Report it WITH its aperture, always.
"""
import numpy as np

DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)


def _clean(cube, background='median'):
    """Finite, background-subtracted, non-negative copy.

    NaN is not zero. Bench crops are NaN-PADDED where the box ran off the
    slab, and zeros there would read as real dark signal that pulls a
    centroid and anchors a correlation. They are filled with the
    background level instead, which is the value that carries no
    information either way.
    """
    c = np.asarray(cube, dtype=float)
    finite = np.isfinite(c)
    if not finite.any():
        return None, None
    if background == 'median':
        bg = float(np.median(c[finite]))
    elif background is None:
        bg = 0.0
    else:
        bg = float(np.quantile(c[finite], float(background)))
    out = np.where(finite, c - bg, 0.0)
    return out, finite


def moments_yxz(cube, voxel_um=DEFAULT_VOXEL_UM, aperture_um=None,
                background=0.5, centre_yxz=None):
    """
    Intensity-weighted centroid and radius of gyration, no shape assumed.

    Returns {'centroid_yxz', 'rg_y_um', 'rg_x_um', 'rg_z_um', 'rg_xy_um',
             'flux', 'n_voxels', 'aperture_um'} or None.

    Only voxels ABOVE the background quantile contribute. Without that,
    the background -- which is most of the crop -- dominates the second
    moment and every object measures the same size, namely the size of
    the box.
    """
    c, finite = _clean(cube, background)
    if c is None:
        return None
    dy, dx, dz = voxel_um
    w = np.clip(c, 0.0, None)
    if w.sum() <= 0:
        return None

    iy, ix, iz = np.indices(c.shape)
    Y, X, Z = iy * dy, ix * dx, iz * dz
    if centre_yxz is None:
        cy = float((Y * w).sum() / w.sum())
        cx = float((X * w).sum() / w.sum())
        cz = float((Z * w).sum() / w.sum())
    else:
        cy, cx, cz = (float(v) for v in centre_yxz)

    if aperture_um is not None:
        ry, rx, rz = aperture_um
        inside = ((np.abs(Y - cy) <= ry) & (np.abs(X - cx) <= rx)
                  & (np.abs(Z - cz) <= rz))
        w = np.where(inside, w, 0.0)
        if w.sum() <= 0:
            return None
        cy = float((Y * w).sum() / w.sum())
        cx = float((X * w).sum() / w.sum())
        cz = float((Z * w).sum() / w.sum())

    tot = w.sum()
    vy = float((w * (Y - cy) ** 2).sum() / tot)
    vx = float((w * (X - cx) ** 2).sum() / tot)
    vz = float((w * (Z - cz) ** 2).sum() / tot)
    return {
        'centroid_yxz': (cy / dy, cx / dx, cz / dz),   # in VOXELS, as callers expect
        'centroid_um': (cy, cx, cz),
        'rg_y_um': float(np.sqrt(vy)), 'rg_x_um': float(np.sqrt(vx)),
        'rg_z_um': float(np.sqrt(vz)),
        'rg_xy_um': float(np.sqrt(0.5 * (vy + vx))),
        'flux': float(tot), 'n_voxels': int((w > 0).sum()),
        'aperture_um': aperture_um,
    }


def signal_coverage(reference, moving):
    """Fraction of the REFERENCE's signal that `moving` actually observes.

    Missing voxels cannot be repaired by any fill. Filling with the
    background is identical to filling with zero once negatives are
    clipped -- measured, not assumed: a hole cut through a blob biased the
    axial shift by -1.3 voxels under BOTH treatments, to the same third
    decimal. So the honest move is not to choose a better fill, it is to
    notice that the estimate is compromised and say so.

    Weighted by reference intensity, because a hole in the background
    costs nothing and a hole through the object costs everything. Crops
    are NaN-padded at slab edges routinely, and those holes are almost
    always background -- which is why this must not be a plain voxel
    count.
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(moving, dtype=float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not fa.any() or a.shape != b.shape:
        return 0.0
    # Threshold at 3 robust sigma above the background, not at the
    # background itself. Half the background sits ABOVE its own median by
    # definition, and clipping at zero keeps every one of those noise
    # excursions as positive weight -- enough that a hole through pure
    # background scored 0.982 instead of ~1.0 and looked like real loss.
    # MAD rather than std: the object is in this crop and would inflate
    # a plain standard deviation.
    med = np.median(a[fa])
    mad = np.median(np.abs(a[fa] - med))
    noise = 1.4826 * mad if mad > 0 else 0.0
    w = np.clip(np.where(fa, a - (med + 3.0 * noise), 0.0), 0.0, None)
    tot = w.sum()
    if tot <= 0:
        return 0.0
    return float(w[fb & fa].sum() / tot)


def shift_yxz(reference, moving, upsample=20, background='median',
              max_shift_vox=None, min_coverage=0.9):
    """
    Sub-voxel (dy, dx, dz) that best maps `moving` onto `reference`.

    Sign convention matches the pipeline's own delta: the value returned
    is what must be ADDED to a position measured in `moving` to express
    it in `reference`'s frame -- i.e. reference_position - moving_position
    for the same physical feature.

    Returns (shift, quality) where quality is the normalised correlation
    peak height in [0, 1], or (None, 0.0) if either crop is unusable.
    """
    from skimage.registration import phase_cross_correlation
    a, _ = _clean(reference, background)
    b, _ = _clean(moving, background)
    if a is None or b is None or a.shape != b.shape:
        return None, 0.0
    # Reject before fitting, not after. A crop missing part of the object
    # yields a confident-looking shift that is simply wrong, and no
    # downstream gate can tell it from a good one.
    if min_coverage is not None:
        cov = signal_coverage(reference, moving)
        if cov < float(min_coverage):
            return None, 0.0
    # Negative lobes are background structure, not signal, and they drag
    # the correlation peak. Phase correlation is scale-free, so clipping
    # costs nothing it needs.
    a = np.clip(a, 0.0, None)
    b = np.clip(b, 0.0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return None, 0.0

    # normalization=None is PLAIN cross-correlation, and it is not the
    # library default -- skimage defaults to 'phase'. Measured on
    # synthetic blobs with known sub-voxel shifts (4 shifts x point-like
    # and extended, 24 components):
    #
    #     phase, no window, up=20   mean |err| 0.260 vox   max 0.750
    #     phase + Hann window       mean |err| 0.608 vox   max 1.800
    #     PLAIN, no window, up=20   mean |err| 0.006 vox   max 0.050
    #     PLAIN + Hann window       mean |err| 0.344 vox   max 1.150
    #
    # Phase normalisation whitens the spectrum, which is what makes it
    # robust to illumination differences -- and here it just amplifies
    # noise at the high frequencies where these smooth blobs have no
    # signal. A Hann window hurts both, because it multiplies an
    # off-centre object by a position-dependent taper and so moves the
    # very centroid being measured. Neither is a tuning knob to revisit
    # casually: 40x is not a close call.
    shift, _err, _phase = phase_cross_correlation(
        a, b, upsample_factor=int(upsample), normalization=None)
    shift = np.asarray(shift, dtype=float)

    if max_shift_vox is not None:
        lim = np.asarray(max_shift_vox, dtype=float)
        if np.any(np.abs(shift) > lim):
            return None, 0.0

    # Quality: correlation of the two after applying the shift, which is
    # a statement about THIS pair rather than about the transform. A high
    # phase-correlation peak on two crops that share only background
    # would otherwise pass as a good registration.
    q = _corr_after_shift(a, b, shift)
    return (float(shift[0]), float(shift[1]), float(shift[2])), q


def _corr_after_shift(a, b, shift):
    from scipy.ndimage import shift as ndshift
    try:
        bb = ndshift(b, shift, order=1, mode='nearest')
    except Exception:
        return 0.0
    av, bv = a.ravel(), bb.ravel()
    if av.std() <= 0 or bv.std() <= 0:
        return 0.0
    return float(np.clip(np.corrcoef(av, bv)[0, 1], 0.0, 1.0))
