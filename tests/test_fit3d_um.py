"""
v2 of the 3D Gaussian fit (localization/fit3d_um.py), against SYNTHETIC
ground truth -- the one place a localizer can be checked exactly,
because the true position and width are known by construction.

The real-data bench (tools/fit_testbox.py) answers "is v2 better than v1
on real spots"; this answers "does v2 compute what it claims", which no
amount of real data can, since real spots have no known answer.

Four properties:

1. It recovers a known centre in MICROMETRES, and the returned voxel
   coordinates agree with the physical ones through the voxel size.
2. It recovers a known ANISOTROPIC width -- the whole point of fitting
   in real space is that a lateral and an axial sigma are separate
   physical quantities, not one number reused along two axes.
3. Anisotropic sampling is handled: the SAME physical spot, sampled at a
   different z step, must give the same micrometre answer.
4. at_bound names exactly the parameters that finished on a constraint,
   since every claim made from this engine's output depends on being
   able to tell a measurement from a bound.

Run: python tests/test_fit3d_um.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

from codelab_pipeline.localization import fit3d_um as U     # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name + (f' -- {detail}' if detail and not cond else ''))


def synth(shape=(21, 21, 41), voxel=(0.208, 0.208, 0.2),
          centre_um=None, sigma_um=(0.12, 0.12, 0.35),
          amp=4000.0, offset=100.0, noise=0.0, seed=0):
    """A Gaussian placed at a known PHYSICAL position, sampled on a grid."""
    dy, dx, dz = voxel
    ny, nx, nz = shape
    if centre_um is None:
        centre_um = ((ny - 1) / 2 * dy, (nx - 1) / 2 * dx, (nz - 1) / 2 * dz)
    iy, ix, iz = np.indices(shape)
    Y, X, Z = iy * dy, ix * dx, iz * dz
    cy, cx, cz = centre_um
    sy, sx, sz = sigma_um
    img = offset + amp * np.exp(-(((Y - cy) ** 2) / (2 * sy ** 2)
                                  + ((X - cx) ** 2) / (2 * sx ** 2)
                                  + ((Z - cz) ** 2) / (2 * sz ** 2)))
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, shape)
    return img, centre_um


def main():
    voxel = (0.208, 0.208, 0.2)
    dy, dx, dz = voxel

    # -- 1. recovers a known centre, offset from the grid so a fit that
    #       merely returned its seed would fail --
    true_um = (2.08 + 0.07, 2.08 - 0.05, 4.00 + 0.06)
    img, _ = synth(centre_um=true_um, noise=8.0)
    seed = (10.0, 10.0, 20.0)          # voxel indices, deliberately off-centre
    f = U.fit_gaussian_3d_um(img, *seed, voxel_um=voxel,
                             max_sigma_xy_um=1.0, max_sigma_z_um=2.0)
    check('a clean synthetic spot is accepted', f is not None)
    if f is not None:
        err = np.array([f.y_um - true_um[0], f.x_um - true_um[1], f.z_um - true_um[2]])
        check('centre recovered to better than 20 nm', np.abs(err).max() < 0.02,
              f'error {np.round(err * 1000, 1)} nm')
        check('voxel coords agree with micrometre coords through the voxel size',
              abs(f.y * dy - f.y_um) < 1e-9 and abs(f.x * dx - f.x_um) < 1e-9
              and abs(f.z * dz - f.z_um) < 1e-9)
        check('nothing sits on a bound for an easy spot', f.at_bound == (),
              str(f.at_bound))

    # -- 2. recovers an ANISOTROPIC width in physical units --
    true_sig = (0.13, 0.13, 0.42)
    img, c = synth(sigma_um=true_sig, noise=8.0)
    f = U.fit_gaussian_3d_um(img, 10.0, 10.0, 20.0, voxel_um=voxel,
                             max_sigma_xy_um=1.0, max_sigma_z_um=2.0)
    check('anisotropic sigma is accepted', f is not None)
    if f is not None:
        check('lateral sigma recovered within 15 nm',
              abs(f.sigma_x_um - true_sig[1]) < 0.015
              and abs(f.sigma_y_um - true_sig[0]) < 0.015,
              f'{f.sigma_y_um:.4f}/{f.sigma_x_um:.4f} vs {true_sig[:2]}')
        check('axial sigma recovered within 30 nm',
              abs(f.sigma_z_um - true_sig[2]) < 0.03,
              f'{f.sigma_z_um:.4f} vs {true_sig[2]}')
        check('axial sigma is resolved as LARGER than lateral, not tied to it',
              f.sigma_z_um > 2 * f.sigma_x_um)

    # -- 3. the same physical spot, sampled differently, gives the same
    #       physical answer. This is what index-unit fitting cannot do. --
    got = []
    for zstep in (0.1, 0.2, 0.4):
        nz = int(round(8.0 / zstep)) + 1
        img, c = synth(shape=(21, 21, nz), voxel=(dy, dx, zstep),
                       centre_um=(2.08, 2.08, 4.0), sigma_um=(0.12, 0.12, 0.35),
                       noise=8.0)
        f = U.fit_gaussian_3d_um(img, 10.0, 10.0, (nz - 1) / 2,
                                 voxel_um=(dy, dx, zstep),
                                 max_sigma_xy_um=1.0, max_sigma_z_um=2.0)
        got.append(None if f is None else (f.z_um, f.sigma_z_um))
    check('every z sampling produced a fit', all(g is not None for g in got), str(got))
    if all(g is not None for g in got):
        zs = np.array([g[0] for g in got])
        ss = np.array([g[1] for g in got])
        check('centre agrees across z steps of 0.1/0.2/0.4 um (within 30 nm)',
              zs.ptp() < 0.03, f'{np.round(zs, 4)}')
        check('axial sigma agrees across z steps (within 40 nm)',
              ss.ptp() < 0.04, f'{np.round(ss, 4)}')

    # -- 4. at_bound tells the truth, in both directions --
    img, c = synth(sigma_um=(0.30, 0.30, 0.80), noise=8.0)
    f = U.fit_gaussian_3d_um(img, 10.0, 10.0, 20.0, voxel_um=voxel,
                             max_sigma_xy_um=0.15, max_sigma_z_um=0.30)
    check('a sigma cap below the true width is reported as at-bound',
          f is not None and 'sigma_x' in f.at_bound and 'sigma_z' in f.at_bound,
          'None' if f is None else str(f.at_bound))
    if f is not None:
        check('the at-bound value IS the cap, not a measurement',
              abs(f.sigma_x_um - 0.15) < 1e-9 and abs(f.sigma_z_um - 0.30) < 1e-9,
              f'{f.sigma_x_um} {f.sigma_z_um}')

    # a seed far from the spot, with a tight position bound, rails position
    f = U.fit_gaussian_3d_um(img, 10.0, 10.0, 8.0, voxel_um=voxel,
                             peak_bound_um=0.05,
                             max_sigma_xy_um=1.0, max_sigma_z_um=2.0)
    check('a position bound that blocks the true centre is reported as at-bound',
          f is None or 'z' in f.at_bound, 'None' if f is None else str(f.at_bound))

    # -- 5. contract: never raises, returns None instead --
    for bad in (np.full((5, 5, 5), np.nan), np.zeros((2, 2, 2)),
                np.zeros((21, 21, 41))):
        try:
            U.fit_gaussian_3d_um(bad, 1.0, 1.0, 1.0)
            ok = True
        except Exception as e:
            ok = False
            detail = repr(e)
        check(f'degenerate input {bad.shape} returns rather than raises', ok,
              locals().get('detail', ''))

    # -- 6. fit_radius_um actually restricts the voxels used --
    img, c = synth(noise=8.0)
    full = U.fit_gaussian_3d_um(img, 10.0, 10.0, 20.0, voxel_um=voxel,
                                max_sigma_xy_um=1.0, max_sigma_z_um=2.0)
    small = U.fit_gaussian_3d_um(img, 10.0, 10.0, 20.0, voxel_um=voxel,
                                 max_sigma_xy_um=1.0, max_sigma_z_um=2.0,
                                 fit_radius_um=(0.5, 0.5, 1.0))
    check('a fit radius reduces the voxel count',
          full is not None and small is not None
          and small.n_voxels < full.n_voxels,
          f'{None if small is None else small.n_voxels} vs '
          f'{None if full is None else full.n_voxels}')
    if full is not None and small is not None:
        check('restricting the volume does not move a clean spot',
              abs(small.z_um - full.z_um) < 0.02,
              f'{small.z_um:.4f} vs {full.z_um:.4f}')

    print()
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        raise SystemExit('FAILURES: ' + ', '.join(FAIL))
    print('ALL GOOD')


if __name__ == '__main__':
    main()
