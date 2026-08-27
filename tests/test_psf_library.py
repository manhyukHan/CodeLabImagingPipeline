"""
Tests for the PSF library.

The dangerous failures here are quiet ones: a label that escapes the
library folder, a shape tuple assembled in the wrong ORDER (still a valid
tuple, describes a different PSF), a half-written entry that parses, and
an installed copy that does not say where it came from.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_psf_library.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.localization import psf as P              # noqa: E402
from codelab_pipeline.localization import psf_library as LIB    # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def main():
    tmp = tempfile.mkdtemp(prefix='psflib_')
    real_dir = LIB.library_dir
    LIB.library_dir = lambda: tmp                      # redirect the library

    try:
        print('naming')
        lab = LIB.default_label('MP58', 'Hyb_016', 555, when=0)
        check('the default label follows <exp>-<hybe>-ch<channel>-<date>',
              lab.startswith('MP58-Hyb_016-ch555-') and lab[-8:].isdigit(), lab)
        check('empty parts are dropped, not left as holes',
              '--' not in LIB.default_label('', 'Hyb_001', '', when=0),
              LIB.default_label('', 'Hyb_001', '', when=0))

        # A label becomes a FILENAME. These must not escape the folder.
        for evil in ('../../etc/passwd', 'a/b', 'a\\b', 'C:\\windows\\x'):
            s = LIB.sanitize_label(evil)
            p = os.path.abspath(LIB.entry_path(evil))
            check(f'label {evil!r} cannot escape the library folder',
                  os.path.dirname(p) == os.path.abspath(tmp)
                  and os.sep not in s and '..' not in s, f'-> {s}')

        print('\nwrite / read')
        params = {'sigma_xy_um': 0.1375, 'sigma_z_um': 0.4695,
                  'halo_frac': 0.1802, 'halo_scale': 2.5597}
        LIB.write('unit-test-entry', 'gaussian_halo', params,
                  (0.208, 0.208, 0.2),
                  source={'experiment': 'UNIT', 'n_crops': 160},
                  converged={'converged': True, 'delta_nm': 1.0},
                  notes='written by the test')
        got = LIB.read('unit-test-entry')
        check('an entry round-trips its parameters exactly',
              got and got['params'] == params)
        check('the entry declares kind=readout', got.get('kind') == 'readout')
        check('provenance survives the round trip',
              got['source']['experiment'] == 'UNIT'
              and got['converged']['converged'] is True)
        check('the write left no .part file behind',
              not any(f.endswith('.part') for f in os.listdir(tmp)),
              str(os.listdir(tmp)))

        LIB.write('unit-test-entry', 'gaussian', {'sigma_xy_um': 0.2,
                                                  'sigma_z_um': 0.5},
                  (0.208, 0.208, 0.2), source={})
        check('re-writing a label replaces it rather than duplicating',
              len([e for e in LIB.list_entries()
                   if e['label'] == 'unit-test-entry']) == 1)

        print('\nlisting and damage')
        with open(os.path.join(tmp, 'broken.json'), 'w', encoding='utf-8') as f:
            f.write('{ this is not json')
        labels = [e['label'] for e in LIB.list_entries()]
        check('a malformed entry does not break the listing',
              'unit-test-entry' in labels and 'broken' not in labels, str(labels))
        check('but it IS reported as a problem', 'broken' in LIB.problems())
        check('reading a missing label returns None, not an exception',
              LIB.read('no-such-entry') is None)

        print('\nshape_tuple')
        doc = {'family': 'gaussian_halo', 'params': params}
        fam, shape = LIB.shape_tuple(doc)
        names = P.FAMILIES['gaussian_halo'][1]
        check('the shape tuple follows the FAMILY parameter order, not dict order',
              fam == 'gaussian_halo'
              and list(shape) == [params[n] for n in names],
              f'{names} -> {shape}')
        check('a shape missing a parameter returns None rather than a short tuple',
              LIB.shape_tuple({'family': 'gaussian_halo',
                               'params': {'sigma_xy_um': 0.1}}) is None)
        check('an unknown family returns None',
              LIB.shape_tuple({'family': 'nonesuch', 'params': {}}) is None)

        print('\ninstall into an experiment')
        store = tempfile.mkdtemp(prefix='store_')
        LIB.write('install-me', 'gaussian_halo', params, (0.208, 0.208, 0.2),
                  source={'experiment': 'UNIT'})
        target = LIB.install('install-me', store)
        check('install writes <project>/analysis/psf.json',
              target and os.path.basename(target) == 'psf.json'
              and os.path.exists(target), str(target))
        with open(target, encoding='utf-8') as f:
            inst = json.load(f)
        check('the installed copy records WHICH label it came from',
              inst.get('installed_from') == 'install-me', str(inst.get('installed_from')))
        check('and when it was installed', bool(inst.get('installed_at')))
        check('installed() reads it back',
              (LIB.installed(store) or {}).get('installed_from') == 'install-me')
        check('the install left no .part behind',
              not any(f.endswith('.part')
                      for f in os.listdir(os.path.dirname(target))))
        check('installing a missing label returns None and writes nothing',
              LIB.install('no-such-entry', tempfile.mkdtemp()) is None)

        print('\nthe shipped library')
        LIB.library_dir = real_dir
        entries = LIB.list_entries()
        check('the repo library is populated', len(entries) >= 2,
              f'{len(entries)} entries')
        check('it contains a universal default',
              any(e['label'] == 'universal-default' for e in entries))
        check('every shipped entry is a READOUT psf, never a fiducial one',
              all(e.get('kind') == 'readout' for e in entries),
              str({e.get('kind') for e in entries}))
        check('every shipped entry yields a usable shape tuple',
              all(LIB.shape_tuple(e) is not None for e in entries))
        check('no shipped entry is broken', LIB.problems() == [],
              str(LIB.problems()))
        univ = [e for e in entries if e['label'] == 'universal-default']
        if univ:
            src = univ[0].get('source', {})
            check('the universal default names the experiments it averaged',
                  len(src.get('experiments') or []) >= 2, str(src.get('experiments')))

    finally:
        LIB.library_dir = real_dir

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
