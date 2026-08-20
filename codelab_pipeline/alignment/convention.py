"""
THE coordinate convention, and the one adapter to everything x-major.

This pipeline is rasterized (y, x) -- per explicit decision ("I prefer
rasterized version, Y/X"; "the cleanness of the program itself is the
true matter"):

  points        [y, x, 1] column vectors; 3D tuples are (y, x, z),
                matching ndarray indexing img[y, x] and stack[y, x, z].
  matrices      H @ [y, x, 1]: row 0 is y, row 1 is x. Translation is
                ty = H[0, 2], tx = H[1, 2].
  angle         the IMAGE-plane rotation angle is read as
                degrees(arctan2(H[0, 1], H[0, 0])) for a y-major H --
                numerically identical to the x-major arctan2(H[1, 0],
                H[0, 0]) read of the same physical rotation, so reported
                angles are continuous across the convention flip.
  shapes        (height, width) everywhere, as numpy already gives them.
  storage       every vlinks.h5 carries coordinate_order='yx' (see
                vlinks_store); readers refuse an un-stamped/x-major
                store loudly and name tools/migrate_store_to_yx.py.

OpenCV is x-major and stays that way -- cv2.warpAffine/phaseCorrelate/
getRotationMatrix2D all speak (x, y). Every cv2 call site converts AT
THE BOUNDARY with as_cv2 below, and converts results back with to_yx.
No other module may re-derive this permutation.

The math: with P the (self-inverse) permutation that swaps the first
two coordinates, a y-major matrix and its x-major twin are conjugates:

    H_xy = P @ H_yx @ P        (and identically in the other direction)

because  H_xy @ [x, y, 1] = P @ H_yx @ P @ [x, y, 1]
                          = P @ (H_yx @ [y, x, 1]) -- compute in yx,
then P swaps the result back to (x, y). Conjugation is exact for any
affine matrix (rotation, anisotropic scale, shear included), not just
translations.
"""
import numpy as np

# Swap the first two coordinates; P @ P = I.
P = np.array([[0.0, 1.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]])


def swap_axes(H):
    """The conjugation P @ H @ P -- converts a 3x3 affine between y-major
    and x-major layouts (self-inverse, so one function serves both
    directions)."""
    return P @ np.asarray(H, dtype=float) @ P


def as_cv2(H_yx):
    """This pipeline's y-major H, as the x-major 3x3 OpenCV expects.
    Slice [:2] at the call site for APIs taking a 2x3."""
    return swap_axes(H_yx)


def to_yx(H_xy):
    """An x-major matrix (from cv2, or a legacy store), as y-major."""
    return swap_axes(H_xy)
