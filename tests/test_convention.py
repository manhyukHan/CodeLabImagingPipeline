"""
Proofs for the y-major/x-major conjugation (codelab_pipeline/alignment/
convention.py) -- the one adapter the Y/X unification rests on. Each
property here is exactly the mistake a hand-rolled swap makes:
transposing instead of conjugating, double-swapping a translation,
or flipping a rotation's sign.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from codelab_pipeline.alignment.convention import P, swap_axes, as_cv2, to_yx


def test_p_is_self_inverse():
    assert np.array_equal(P @ P, np.eye(3))


def test_round_trip_is_identity():
    rng = np.random.default_rng(5)
    H = np.eye(3)
    H[:2, :2] = rng.normal(size=(2, 2))
    H[:2, 2] = rng.normal(size=2)
    assert np.allclose(to_yx(as_cv2(H)), H)


def test_point_equivalence_translation():
    """H_yx @ [y, x] and its cv2 twin @ [x, y] name the same physical
    point -- checked on a pure translation, where an index mix-up is the
    classic silent bug."""
    ty, tx = -7.25, 3.5
    H_yx = np.eye(3)
    H_yx[0, 2], H_yx[1, 2] = ty, tx
    y, x = 100.0, 40.0
    ny, nx, _ = H_yx @ np.array([y, x, 1.0])
    cx, cy, _ = as_cv2(H_yx) @ np.array([x, y, 1.0])
    assert (ny, nx) == (cy, cx) == (y + ty, x + tx)


def test_point_equivalence_general_affine():
    rng = np.random.default_rng(11)
    H_yx = np.eye(3)
    H_yx[:2, :2] = rng.normal(size=(2, 2))
    H_yx[:2, 2] = rng.normal(size=2)
    for _ in range(20):
        y, x = rng.uniform(0, 500, 2)
        ny, nx, _ = H_yx @ np.array([y, x, 1.0])
        cx, cy, _ = as_cv2(H_yx) @ np.array([x, y, 1.0])
        assert np.allclose((ny, nx), (cy, cx))


def test_rotation_angle_is_continuous_across_conventions():
    """The angle read [0,1]/[0,0] on a y-major H equals the x-major read
    [1,0]/[0,0] on its twin -- reported angles must not flip sign at the
    convention boundary."""
    for theta in (-8.0, -0.5, 0.0, 3.0, 9.9):
        t = np.radians(theta)
        H_xy = np.eye(3)
        H_xy[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
        H_yx = to_yx(H_xy)
        angle_xy = np.degrees(np.arctan2(H_xy[1, 0], H_xy[0, 0]))
        angle_yx = np.degrees(np.arctan2(H_yx[0, 1], H_yx[0, 0]))
        assert np.isclose(angle_xy, angle_yx), (theta, angle_xy, angle_yx)


def test_composition_commutes_with_swap():
    """swap(A @ B) == swap(A) @ swap(B) -- composition can happen in
    either layout without re-permuting intermediates."""
    rng = np.random.default_rng(23)
    A, B = np.eye(3), np.eye(3)
    A[:2, :2], A[:2, 2] = rng.normal(size=(2, 2)), rng.normal(size=2)
    B[:2, :2], B[:2, 2] = rng.normal(size=(2, 2)), rng.normal(size=2)
    assert np.allclose(swap_axes(A @ B), swap_axes(A) @ swap_axes(B))


def test_inverse_commutes_with_swap():
    rng = np.random.default_rng(31)
    H = np.eye(3)
    H[:2, :2], H[:2, 2] = rng.normal(size=(2, 2)) + np.eye(2), rng.normal(size=2)
    assert np.allclose(swap_axes(np.linalg.inv(H)), np.linalg.inv(swap_axes(H)))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f'  FAIL  {t.__name__}: {e}')
    print(f'{passed}/{len(tests)} passed')
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    sys.exit(main())
