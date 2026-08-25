"""
Ground-truth direction tests for the alignment engine.

A real MIP is warped by a KNOWN transform and fed back through the same
fit the app runs (align_readout_to_reference / _find_z_shift). Each case
asserts three things, because each has failed independently in this
codebase's history:

  1. VALUE: the recovered matrix inverts the known one (H_fit @ M ~ I).
  2. IMAGE DIRECTION: warping the moving image by H_fit (the same
     forward cv2.warpAffine convention every overlay/preview uses)
     actually reproduces the reference -- a sign flip would double the
     error instead of cancelling it, so this can never pass "by
     magnitude" alone.
  3. POINT DIRECTION: a coordinate localized in the moving frame lands
     on its reference-frame position under H_fit[:2] @ [x, y, 1] -- the
     exact math spot mapping / matrix_to_shared applies to spots.

Base image: the real store's own MIP (CODELAB_GT_STORE overrides the
storage path -- point it at a clone when the live app holds the lock).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from codelab_pipeline.alignment import chain as alignment
from codelab_pipeline.alignment.convention import as_cv2, to_yx
from codelab_pipeline.io import analysis_store as vlinks_store

STORE = os.environ.get('CODELAB_GT_STORE', 'data/chr19_downstream_new/RNA_queue')
FOV, HYBE, CHANNEL = 1, 'Hyb_101', 555

_ref = None


def ref_image():
    global _ref
    if _ref is None:
        mip = vlinks_store.read_hybe_mip(STORE, FOV, HYBE, CHANNEL)
        assert mip is not None, f'no MIP at {STORE} FOV{FOV} {HYBE}/ch{CHANNEL}'
        _ref = mip.astype(np.float32)
    return _ref


def warp(img, M3):
    """Forward warp: content at p moves to M3 @ p (same convention the
    app uses to draw a moving image into the reference frame)."""
    h, w = img.shape
    return cv2.warpAffine(img, np.asarray(M3, dtype=np.float64)[:2], (w, h))


def interior(img, margin=60):
    return img[margin:-margin, margin:-margin]


def closure_error(moving, H_fit, ref):
    """Mean abs interior error of moving warped ONTO ref by H_fit,
    vs. the unaligned error -- direction-sensitive: a sign-flipped H
    doubles the offset instead of cancelling it. H_fit is y-major,
    converted at the cv2 boundary."""
    realigned = warp(moving, as_cv2(H_fit))
    e_after = float(np.mean(np.abs(interior(realigned) - interior(ref))))
    e_before = float(np.mean(np.abs(interior(moving) - interior(ref))))
    return e_after, e_before


def M_translation(dx, dy):
    M = np.eye(3)
    M[0, 2], M[1, 2] = dx, dy
    return M


def M_rotation(angle_deg, shape, dx=0.0, dy=0.0):
    h, w = shape
    M = np.eye(3)
    M[:2] = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def test_identity_finds_no_spurious_correction():
    ref = ref_image()
    H = alignment.align_readout_to_reference(ref.copy(), ref)
    dy, dx = H[0, 2], H[1, 2]        # y-major: ty at [0, 2]
    ang = alignment.h_angle_degrees(H)
    assert abs(dx) < 0.3 and abs(dy) < 0.3 and abs(ang) < 0.1, \
        f'identity pair produced correction dx={dx:.2f} dy={dy:.2f} angle={ang:.2f}'


def test_known_translation_recovered_and_inverts():
    ref = ref_image()
    M = M_translation(9.5, -6.25)
    moving = warp(ref, M)
    H = alignment.align_readout_to_reference(moving, ref)
    prod = H @ to_yx(M)          # H is y-major; M was built cv2-style
    err = np.abs(prod - np.eye(3))
    assert err[:2, 2].max() < 0.5 and err[:2, :2].max() < 0.01, \
        f'H_fit @ M != I: recovered t=(tx {H[1,2]:.2f}, ty {H[0,2]:.2f}) for M t=(9.5,-6.25)'
    # 0.35: two bilinear warps blur high-frequency fiducial content, so
    # the floor is interpolation noise, not misalignment (a direction
    # flip would DOUBLE the error -- asserted separately below).
    e_after, e_before = closure_error(moving, H, ref)
    assert e_after < 0.35 * e_before, \
        f'image closure failed: after={e_after:.3f} before={e_before:.3f}'
    e_wrong, _ = closure_error(moving, np.linalg.inv(H), ref)
    assert e_wrong > 1.5 * e_after, \
        f'direction ambiguous: wrong-direction warp scored {e_wrong:.3f} vs {e_after:.3f}'


def test_known_rotation_recovered_and_inverts():
    ref = ref_image()
    M = M_rotation(3.0, ref.shape, dx=5.0, dy=-4.0)
    moving = warp(ref, M)
    H = alignment.align_readout_to_reference(moving, ref)
    # Both angles through the ONE public y-major reader -- mixing it with
    # the engine-internal x-major reader flips a sign and asserts nothing.
    ang_M = alignment.h_angle_degrees(to_yx(M))
    ang_H = alignment.h_angle_degrees(H)
    assert abs(ang_H + ang_M) < 0.3, \
        f'rotation not inverted: M angle={ang_M:.3f}, fit angle={ang_H:.3f}'
    prod = np.abs(H @ to_yx(M) - np.eye(3))
    assert prod[:2, 2].max() < 1.0 and prod[:2, :2].max() < 0.01, \
        f'H_fit @ M != I for rotation case (max t err {prod[:2,2].max():.2f}px)'
    e_after, e_before = closure_error(moving, H, ref)
    assert e_after < 0.35 * e_before, \
        f'image closure failed for rotation: after={e_after:.3f} before={e_before:.3f}'
    e_wrong, _ = closure_error(moving, np.linalg.inv(H), ref)
    assert e_wrong > 1.5 * e_after, \
        f'direction ambiguous: wrong-direction warp scored {e_wrong:.3f} vs {e_after:.3f}'


def test_point_maps_moving_to_reference():
    """The spot-mapping direction: a feature localized in the MOVING
    frame maps to its reference position via H_fit -- never the other
    way around. This is matrix_to_shared's contract for spots."""
    ref = ref_image()
    M = M_translation(11.0, 7.5)
    moving = warp(ref, M)
    H = alignment.align_readout_to_reference(moving, ref)
    p_ref_xy = np.array([412.0, 633.0, 1.0])
    p_moving_xy = M @ p_ref_xy                 # where the feature shows up in moving (cv2 space)
    back_yx = H[:2] @ np.array([p_moving_xy[1], p_moving_xy[0], 1.0])
    err = np.hypot(back_yx[0] - p_ref_xy[1], back_yx[1] - p_ref_xy[0])
    assert err < 0.6, f'point closure failed: |H_yx @ (M @ p) - p| = {err:.2f}px'


def test_z_shift_sign_moves_target_onto_reference():
    """_find_z_shift's contract: the returned dz, ADDED to the target
    profile's position, lands on the reference. A target delayed by +k
    planes must yield dz = -k."""
    z = np.arange(60, dtype=np.float64)
    ref_profile = np.exp(-0.5 * ((z - 28.0) / 3.0) ** 2)
    k = 4
    target_profile = np.roll(ref_profile, k)   # bump now at 28 + k
    dz = alignment._find_z_shift(target_profile, ref_profile)
    assert dz == -k, f'expected dz={-k} for a +{k}-plane delay, got {dz}'
    dz0 = alignment._find_z_shift(ref_profile, ref_profile)
    assert dz0 == 0, f'identical profiles must give dz=0, got {dz0}'


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f'  FAIL  {t.__name__}: {e}')
        except Exception as e:
            print(f'  ERROR {t.__name__}: {type(e).__name__}: {e}')
    print(f'\n{passed}/{len(tests)} passed')
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    sys.exit(main())
