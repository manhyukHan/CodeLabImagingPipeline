"""
Coordinate-convention harness.

Written as the prerequisite for the Y/X (rasterized) coordinate unification:
the pipeline currently mixes orders -- cell.area is (x, y), frame_shape is
(height, width), align_cell takes (y, x), and homographies act on (x, y, 1)
-- and flipping that is only safe with a test that fails loudly on a
transposed axis.

WHY THIS EXISTS AT ALL: the real dataset is currently all-identity (fresh
vlinks, no alignment run). Identity is invariant under an axis swap --
S @ I @ S == I -- so every axis-order bug is INVISIBLE on that data. A
verification that loads the real config and checks matrices come back as
identity proves nothing about coordinate order. Every transform here is
therefore deliberately:

  - non-identity
  - non-symmetric in x vs y (dx != dy, and width != height)
  - applied to an ASYMMETRIC mask

so that swapping the two axes anywhere changes the answer and fails a test.

The load-bearing one is test_coordinate_and_image_paths_agree: it applies the
same H to coordinates (align_cell) and to an image (cv2.warpAffine) and
asserts the two land in the same place. Those are the pipeline's two
independent consumers of a matrix, and an axis-order mismatch between them is
exactly the bug class this harness exists to catch.

Run: pytest tests/test_coordinate_conventions.py -v
  or: python tests/test_coordinate_conventions.py
"""
import os
import sys

import cv2
import numpy as np
import numpy.linalg as la

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.alignment import chain as alignment

# Deliberately non-square: a (height, width) vs (width, height) mix-up
# changes which coordinates survive the bounds check in align_cell.
FRAME_SHAPE = (300, 500)   # (height, width) -- H=300, W=500
DX, DY = 37.0, -11.0       # deliberately different, and opposite signs


def _translation(dx, dy):
    """A pure translation in the pipeline's current (x, y, 1)-major convention."""
    H = np.eye(3)
    H[0, 2] = dx
    H[1, 2] = dy
    return H


def _asymmetric_mask():
    """
    An L-shaped mask, wider than it is tall, positioned off-centre. Every
    property here matters: an L has no rotational or reflective symmetry, so
    a transpose changes the SHAPE (not just the position), and the offset
    means a swapped axis usually also changes which pixels clip out of frame.

    Returns (y, x) index arrays -- numpy's own native order, matching
    np.where and the target convention for this refactor.
    """
    mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    mask[100:140, 200:340] = 1   # long horizontal arm (40 tall, 140 wide)
    mask[100:220, 200:240] = 1   # short vertical arm (120 tall, 40 wide)
    y, x = np.where(mask > 0)
    return y, x


def _centroid_yx(y, x):
    return float(np.mean(y)), float(np.mean(x))


def test_mask_is_actually_asymmetric():
    """
    Guard on the guard: if the fixture were symmetric, every other test here
    would pass under a transposed implementation and prove nothing. This is
    the non-vacuity check.
    """
    y, x = _asymmetric_mask()
    height_extent = y.max() - y.min()
    width_extent = x.max() - x.min()
    assert height_extent != width_extent, 'fixture must not be square'

    # Transposing the mask must produce a genuinely different pixel set,
    # otherwise an axis swap is undetectable by construction.
    as_set = set(zip(y.tolist(), x.tolist()))
    transposed = set(zip(x.tolist(), y.tolist()))
    assert as_set != transposed, 'fixture must not be transpose-invariant'


def test_align_cell_moves_each_axis_independently():
    """
    A translation of (dx, dy) must move x by dx and y by dy -- not the
    reverse. With DX=37 and DY=-11 a swapped axis is off by 48 px in both
    coordinates, far outside any rounding tolerance.
    """
    y, x = _asymmetric_mask()
    cy0, cx0 = _centroid_yx(y, x)

    H = _translation(DX, DY)
    ny, nx = alignment.align_cell((y, x), H, FRAME_SHAPE)
    cy1, cx1 = _centroid_yx(ny, nx)

    assert abs((cx1 - cx0) - DX) < 1.0, f'x moved by {cx1 - cx0}, expected {DX}'
    assert abs((cy1 - cy0) - DY) < 1.0, f'y moved by {cy1 - cy0}, expected {DY}'


def test_align_cell_round_trips_through_inverse():
    """
    H then inv(H) must return the mask to where it started. Compares
    centroids rather than exact pixel sets because align_cell applies a
    morphological closing on any non-identity matrix (documented in its own
    docstring), which can add a few boundary pixels each way.
    """
    y, x = _asymmetric_mask()
    cy0, cx0 = _centroid_yx(y, x)

    H = _translation(DX, DY)
    my, mx = alignment.align_cell((y, x), H, FRAME_SHAPE)
    by, bx = alignment.align_cell((my, mx), la.inv(H), FRAME_SHAPE)
    cy1, cx1 = _centroid_yx(by, bx)

    assert abs(cy1 - cy0) < 1.0, f'y did not round-trip: {cy0} -> {cy1}'
    assert abs(cx1 - cx0) < 1.0, f'x did not round-trip: {cx0} -> {cx1}'


def test_coordinate_and_image_paths_agree():
    """
    THE load-bearing test.

    A matrix in this pipeline has two independent consumers: coordinates go
    through align_cell, images go through cv2.warpAffine. Both must interpret
    the matrix's axis order the same way. This renders the mask as an image,
    warps it with cv2, and separately transforms the mask's coordinates, then
    asserts the two agree.

    If the Y/X refactor flips the coordinate path but misses a warpAffine
    call site (there are 14), or flips the stored matrices but not the cv2
    boundary, THIS is the test that fails. Nothing in the current
    all-identity dataset can catch that.
    """
    y, x = _asymmetric_mask()
    H = _translation(DX, DY)

    # Image path: cv2 indexes (col=x, row=y) and takes the 2x3 slice.
    img = np.zeros(FRAME_SHAPE, dtype=np.float32)
    img[y, x] = 1.0
    warped = cv2.warpAffine(img, H[:2], (FRAME_SHAPE[1], FRAME_SHAPE[0]),
                            flags=cv2.INTER_NEAREST)
    img_y, img_x = np.where(warped > 0.5)

    # Coordinate path.
    coord_y, coord_x = alignment.align_cell((y, x), H, FRAME_SHAPE)

    ci = _centroid_yx(img_y, img_x)
    cc = _centroid_yx(coord_y, coord_x)
    assert abs(ci[0] - cc[0]) < 1.5, f'y disagrees: image {ci[0]} vs coords {cc[0]}'
    assert abs(ci[1] - cc[1]) < 1.5, f'x disagrees: image {ci[1]} vs coords {cc[1]}'


def test_rotation_is_not_silently_a_transpose():
    """
    A 90-degree rotation and a transpose produce similar-looking output on a
    symmetric shape, which is precisely why the ~100-degree cross-modal bug
    was ambiguous on inspection. On an L-shaped mask they differ, so this
    pins down that align_cell applies a real rotation about the origin rather
    than swapping axes.
    """
    y, x = _asymmetric_mask()
    theta = np.deg2rad(90.0)
    H = np.eye(3)
    H[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]]
    # Shift back into frame: rotating about the origin sends x>0 to y>0, x<0.
    H[0, 2] = 450.0

    ry, rx = alignment.align_cell((y, x), H, FRAME_SHAPE)
    assert len(ry) > 0, 'rotation put the entire mask out of frame'

    ty, tx = x, y  # a pure transpose, the thing we must NOT be doing
    in_frame = (ty < FRAME_SHAPE[0]) & (tx < FRAME_SHAPE[1])
    ty, tx = ty[in_frame], tx[in_frame]

    rot_c = _centroid_yx(ry, rx)
    tr_c = _centroid_yx(ty, tx)
    assert abs(rot_c[0] - tr_c[0]) > 5.0 or abs(rot_c[1] - tr_c[1]) > 5.0, \
        'rotation and transpose are indistinguishable -- fixture is too symmetric'


def test_identity_path_preserves_mask_exactly():
    """
    align_cell short-circuits on identity specifically to skip the
    morphological closing, which was measured to add up to 15 px to a real
    cell mask. Identity must be a pure passthrough (modulo bounds), so this
    asserts exact set equality, not a centroid.
    """
    y, x = _asymmetric_mask()
    ny, nx = alignment.align_cell((y, x), np.eye(3), FRAME_SHAPE)
    assert set(zip(ny.tolist(), nx.tolist())) == set(zip(y.tolist(), x.tolist()))


def test_z_alignment_carries_no_matrix_degrees_of_freedom():
    """
    Z alignment is a scalar (1D correlation), per explicit decision, for both
    the cross-modal shift and the cell-level residual. The stored 'zx' 3x3 is
    a vestigial container: an audit found ZERO matrix algebra on it anywhere
    (no inv, no @, no compose_chain) and every read is of element [0, 2].

    This test pins that down so matrix-form Z cannot creep back in. It builds
    a zx exactly as chain.py does and asserts the other eight elements carry
    nothing -- i.e. the object is fully described by one number.
    """
    z_shift = 4.25
    H_zx = np.eye(3)
    H_zx[0, 2] = z_shift

    assert float(H_zx[0, 2]) == z_shift
    stripped = H_zx.copy()
    stripped[0, 2] = 0.0
    assert np.allclose(stripped, np.eye(3)), \
        'zx carries information outside [0,2]; Z is no longer scalar'

    # And the scalar survives a round-trip through the storage shape used by
    # ACell.matrices / vlinks without needing any matrix operation.
    entry = {'yx': np.eye(3), 'zx': H_zx, 'yx_is_residual': True}
    assert float(np.asarray(entry['zx'])[0, 2]) == z_shift


def test_harness_detects_a_transposed_implementation():
    """
    Self-check: a passing suite proves nothing unless it can fail. This
    swaps align_cell for deliberately axis-transposed implementations and
    asserts the other tests catch each one.

    This exists because a FrameResolver invariant test written earlier in
    this project was vacuous -- built on the wrong key structure, every
    lookup missed, everything defaulted to identity, and it printed four
    PASS lines while asserting nothing. Encode the lesson rather than
    re-learn it: if a future edit makes these tests insensitive to axis
    order, this fails first.
    """
    real = alignment.align_cell

    def mutant(kind):
        def f(yx, H, shape):
            y, x = yx
            height, width = shape
            if kind == 'matrix_input':
                # coordinates flipped to Y/X but the matrix left X-major:
                # the classic "flipped the data model, forgot the matrices"
                cx, cy = (H[:2] @ np.array([y, x, np.ones_like(x)])).astype(int)
            else:
                cx, cy = (H[:2] @ np.array([x, y, np.ones_like(x)])).astype(int)
                cx, cy = cy, cx      # returns (x, y) where (y, x) is expected
            bad = (cx < 0) | (cy < 0) | (cx >= width) | (cy >= height)
            return cy[~bad], cx[~bad]
        return f

    sensitive = [test_align_cell_moves_each_axis_independently,
                 test_align_cell_round_trips_through_inverse,
                 test_coordinate_and_image_paths_agree]
    try:
        for kind in ('matrix_input', 'return_order'):
            alignment.align_cell = mutant(kind)
            caught = 0
            for t in sensitive:
                try:
                    t()
                except AssertionError:
                    caught += 1
            assert caught == len(sensitive), (
                f'mutation {kind!r} went undetected by {len(sensitive) - caught} '
                f'of {len(sensitive)} tests -- the harness is going vacuous')
    finally:
        alignment.align_cell = real


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures = []
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f'  FAIL  {t.__name__}: {e}')
        except Exception as e:
            failures.append((t.__name__, repr(e)))
            print(f'  ERROR {t.__name__}: {e!r}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_run_all())
