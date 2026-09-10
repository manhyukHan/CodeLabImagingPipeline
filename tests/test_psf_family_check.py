"""
The analytic family is tested against the template, not assumed.

WHY. Every PSF bank was compared to the INSTALLED calibration -- for
MP58 the universal readout gaussian_halo -- and a bank of 905 confirmed
DNA ch555 (fiducial-channel) spots came out at cosine 0.776 to it. The
warning blamed the store calibration and the voxel size; neither had
changed. Fiducial emitters are not readout emitters, and a plain
Gaussian usually describes them better. So each family is now fitted to
the template itself, the best fit is reported, the warning names the
right cause, and the bank's own fit sets its anisotropy.

Synthetic templates check the fit recovers the family and the widths;
the two real banks on this machine (MP58 RNA readout, MP58 DNA ch555)
are measured when present, and the numbers are printed.

Run:  python tests/test_psf_family_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

CHECKS = [0, 0]
VOX = (0.208, 0.208, 0.2)
INSTALLED = {'family': 'gaussian_halo',
             'params': {'sigma_xy_um': 0.13745, 'sigma_z_um': 0.46949,
                        'halo_frac': 0.18019, 'halo_scale': 2.5597}}
BANKS = {'RNA readout (mp58_rna)': 'D:/models/mp58_rna/psf_bank.h5',
         'DNA ch555 (Manhyuk_DNA555_default)':
             os.path.join(os.path.dirname(os.path.dirname(
                 os.path.abspath(__file__))), 'models',
                 'Manhyuk_DNA555_default', 'psf_bank.h5')}


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def test_fit_recovers_the_family():
    print('fit_families on synthetic templates')
    from codelab_pipeline.localization import psf_bank as PB
    rng = np.random.RandomState(0)
    clean = PB.render('gaussian', (0.16, 0.50), r=7, rz=12, voxel_um=VOX)
    g = clean + rng.normal(0, 0.01, clean.shape)
    # THE NOISE SETS THE CEILING: on 15x15x25 voxels a unit-peak Gaussian
    # carries its energy in a few dozen of them, so N(0, 0.01) everywhere
    # caps the cosine near 0.97 even for the true shape. The fit is
    # judged against that ceiling, not against 1.
    ceiling = float(PB.normalise(clean).ravel() @ PB.normalise(g).ravel())
    ff = PB.fit_families(g, VOX)
    fg = ff['fits']['gaussian']
    check('a Gaussian template: gaussian fits to within 0.001 of the '
          'noise ceiling, and is the pick',
          fg['cosine'] > ceiling - 1e-3 and ff['best'] == 'gaussian',
          f"{fg['cosine']:.4f} vs ceiling {ceiling:.4f}, best {ff['best']}")
    check('and recovers its widths within 10%',
          abs(fg['params']['sigma_xy_um'] - 0.16) < 0.016
          and abs(fg['params']['sigma_z_um'] - 0.50) < 0.05,
          str(fg['params']))
    check('the halo family cannot beat it by more than noise',
          ff['fits']['gaussian_halo']['cosine'] <= fg['cosine'] + 0.002,
          f"{ff['fits']['gaussian_halo']['cosine']:.4f}")
    h = PB.render('gaussian_halo', (0.137, 0.469, 0.18, 2.56), r=7, rz=12,
                  voxel_um=VOX)
    ff2 = PB.fit_families(h, VOX)
    check('a core+halo template: gaussian_halo wins and fits at > 0.999',
          ff2['best'] == 'gaussian_halo'
          and ff2['fits']['gaussian_halo']['cosine'] > 0.999,
          str({f: round(v['cosine'], 4) for f, v in ff2['fits'].items()}))
    check('and a plain gaussian is measurably worse on it',
          ff2['fits']['gaussian']['cosine'] < ff2['fits']['gaussian_halo']['cosine'] - 0.005)
    check('every fit is marked plausible or not, with its reasons',
          all('plausible' in v and 'warnings' in v for v in ff['fits'].values()))


def test_warning_and_bound():
    print('the warning names the cause; the bound takes the better fit')
    from codelab_pipeline.localization import psf_bank as PB
    ref = dict(INSTALLED, cosine_to_measured=0.776,
               best_fit={'family': 'gaussian', 'cosine': 0.951,
                         'params': {'sigma_xy_um': 0.160, 'sigma_z_um': 0.480}},
               fits={'gaussian': {'cosine': 0.951},
                     'gaussian_halo': {'cosine': 0.93}})
    w = PB.cosine_warning(ref, n_spots=905)
    check('a template a fitted family describes: another kind of emitter, '
          'not labels or optics',
          w is not None and 'fitted gaussian' in w and '0.951' in w
          and 'fiducial-channel' in w and 'neither a label shortage' in w
          and 'voxel size' not in w, w)
    ref2 = dict(INSTALLED, cosine_to_measured=0.776,
                best_fit={'family': 'gaussian', 'cosine': 0.85,
                          'params': {'sigma_xy_um': 0.2, 'sigma_z_um': 0.5}})
    w2 = PB.cosine_warning(ref2, n_spots=905)
    check('no family reaches the floor: the old diagnosis, and it says so',
          w2 is not None and 'store calibration' in w2
          and 'No fitted family reaches 0.90' in w2, w2)
    w3 = PB.cosine_warning(dict(INSTALLED, cosine_to_measured=0.93,
                                best_fit={'family': 'gaussian', 'cosine': 0.9}),
                           n_spots=1157)
    check('agreement with the installed calibration: no warning', w3 is None)
    meta = {'voxel_um': list(VOX), 'analytic_ref': ref}
    lat, ax = PB.resolution_bound(meta)
    check("the bank's own fit sets the anisotropy when it fits better",
          abs(ax - lat * (0.480 / 0.2) / (0.160 / 0.208)) < 1e-6,
          f'{lat}, {ax:.3f}')
    meta2 = {'voxel_um': list(VOX),
             'analytic_ref': dict(INSTALLED, cosine_to_measured=0.93,
                                  best_fit={'family': 'gaussian', 'cosine': 0.90,
                                            'params': {'sigma_xy_um': 0.3,
                                                       'sigma_z_um': 0.3}})}
    lat2, ax2 = PB.resolution_bound(meta2)
    check('and the installed calibration keeps it when IT fits better',
          abs(ax2 - lat2 * (0.46949 / 0.2) / (0.13745 / 0.208)) < 1e-6)
    lat3, ax3 = PB.resolution_bound({'voxel_um': list(VOX),
                                     'analytic_ref': {'family': 'x'}})
    check('no reference at all: the default axial bound',
          ax3 == float(PB.MERGE_AXIAL_PLANES))


def test_training_reference():
    print("train_spotmodel's reference carries the fits")
    import inspect
    import tools.train_spotmodel as T
    src = inspect.getsource(T.analytic_reference)
    check('the fits are made whether or not the store is calibrated',
          'PB.fit_families(mean, voxel_um)' in src
          and "out['best_fit']" in src)
    from codelab_pipeline.localization import psf_bank as PB
    g = PB.render('gaussian', (0.16, 0.50), r=7, rz=12, voxel_um=VOX)
    ref = T.analytic_reference(g, VOX, storage_path=None)
    check('without a store: fits and best_fit, no installed keys',
          ref.get('best_fit', {}).get('family') == 'gaussian'
          and 'cosine_to_measured' not in ref, str(ref.get('best_fit')))
    from ui.model_build_dialog import ModelBuildDialog
    rsrc = inspect.getsource(ModelBuildDialog.render_report)
    check('the report shows the best analytic fit beside the installed one',
          'best analytic fit' in rsrc and 'vs installed' in rsrc)


def test_real_banks():
    print('the two real banks on this machine')
    from codelab_pipeline.localization import psf_bank as PB
    import h5py
    for name, path in BANKS.items():
        if not os.path.exists(path):
            check(f'{name}: present', False, 'skipped: ' + path + ' absent')
            continue
        with h5py.File(path, 'r') as f:
            mean = np.asarray(f['mean'], float)
            import json
            vox = json.loads(f.attrs['voxel_um']) if isinstance(
                f.attrs['voxel_um'], str) else list(f.attrs['voxel_um'])
            ref = f.attrs.get('analytic_ref')
            ref = json.loads(ref) if isinstance(ref, (str, bytes)) else {}
        ff = PB.fit_families(mean, vox)
        inst = (ref or {}).get('cosine_to_measured')
        fits = {k: round(v['cosine'], 3) for k, v in ff['fits'].items()}
        b = ff['best']
        pr = ff['fits'][b]['params']
        print(f'    {name}: installed cosine {inst}, fitted {fits}, best {b} '
              f"sigma_xy {1000 * pr['sigma_xy_um']:.0f} nm sigma_z "
              f"{1000 * pr['sigma_z_um']:.0f} nm")
        check(f'{name}: the best fitted family describes the template at '
              f'least as well as the installed calibration',
              inst is None or ff['fits'][b]['cosine'] >= float(inst) - 1e-6,
              f"fitted {ff['fits'][b]['cosine']:.3f} vs installed {inst}")


def main():
    test_fit_recovers_the_family()
    test_warning_and_bound()
    test_training_reference()
    test_real_banks()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
