"""
A bank carries one template per experiment, and an engine that knows the
design's resolution matches with the right one.

WHY. The fiducial's shape tracks the design axially: on the three DNA
ch555 bundles pooled first, sigma_z was 641 / 760 / 1068 nm at 5 / 50 /
200 kb per readout step while sigma_xy stayed put, and the pooled mean
served chr19 worst (cosine 0.946 to its own template). So the bank keeps
each experiment's own template as a candidate tagged with its
resolution; select_template takes the exact one, else the nearest with
its axial extent moved to what the candidates' own line predicts, else
the pooled mean; the multispot calibration keeps a Platt per resolution
beside the pooled pair; and V2Params carries a template mode so the
choice can be measured against the pooled mean and against 'all'.

Run:  python tests/test_psf_candidates.py
"""
import inspect
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402

CHECKS = [0, 0]
VOX = (0.208, 0.208, 0.2)


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def halo(sz, sxy=0.27):
    from codelab_pipeline.localization import psf_bank as PB
    return PB.render('gaussian_halo', (sxy, sz, 0.3, 3.0), r=7, rz=12,
                     voxel_um=VOX)


def test_bank_and_selection():
    print('candidates in the bank, and the selection rule')
    from codelab_pipeline.localization import psf_bank as PB
    t5, t50, t200 = halo(0.64), halo(0.76), halo(1.07)
    mean = PB.normalise((t5 + t50 + t200) / 3.0)
    cands = [{'template': t5, 'genomic_resolution_kb': 5.0, 'n_spots': 566,
              'source': 'HoxA', 'sigma_xy_um': 0.22, 'sigma_z_um': 0.64},
             {'template': t50, 'genomic_resolution_kb': 50.0, 'n_spots': 905,
              'source': 'MP58', 'sigma_xy_um': 0.28, 'sigma_z_um': 0.76},
             {'template': t200, 'genomic_resolution_kb': 200.0, 'n_spots': 573,
              'source': 'chr19', 'sigma_xy_um': 0.27, 'sigma_z_um': 1.07}]
    with tempfile.TemporaryDirectory(dir='D:/claude-tmp') as d:
        path = PB.save(os.path.join(d, 'psf_bank.h5'), mean, VOX, n_spots=2044,
                       candidates=cands)
        back = PB.candidates(path)
        check('three candidates round-trip with their tags',
              len(back) == 3 and [m['source'] for _t, m in back] == ['HoxA', 'MP58', 'chr19']
              and np.allclose(back[1][0], t50, atol=1e-6)
              and back[2][1]['genomic_resolution_kb'] == 200.0)
        m2, _c, meta = PB.load(path)
        check('the pooled mean is still the entry, and the file lists them',
              np.allclose(m2, mean, atol=1e-6)
              and [c['source'] for c in meta['candidates']] == ['HoxA', 'MP58', 'chr19'])
        check('a bank without candidates says []',
              PB.candidates(PB.save(os.path.join(d, 'plain.h5'), mean, VOX)) == [])
        raised = ''
        try:
            PB.save(os.path.join(d, 'bad.h5'), mean, VOX,
                    candidates=[{'template': t5[:, :, :11], 'genomic_resolution_kb': 5}])
        except ValueError as exc:
            raised = str(exc)
        check('a candidate of another shape is refused', 'shape' in raised)

    tpl, how = PB.select_template(mean, back, 50.0)
    check('exact resolution -> that template', how['how'] == 'exact'
          and how['source'] == 'MP58' and np.allclose(tpl, t50, atol=1e-6))
    tpl, how = PB.select_template(mean, back, None)
    check('no resolution -> the pooled mean', how['how'] == 'pooled'
          and np.allclose(tpl, mean, atol=1e-6))
    tpl, how = PB.select_template(mean, [], 50.0)
    check('no candidates -> the pooled mean', how['how'] == 'pooled')
    tpl, how = PB.select_template(mean, back, 55.0)
    check('within 20% of a candidate -> that one, unchanged',
          how['how'] == 'nearest' and how['source'] == 'MP58'
          and how['axial_factor'] == 1.0 and np.allclose(tpl, t50, atol=1e-6))
    tpl, how = PB.select_template(mean, back, 20.0)
    want = PB._sigma_z_at(back, 20.0)
    check('further away -> the nearest, widened or narrowed along z to the '
          'sigma_z the candidates predict',
          how['how'] == 'nearest' and how['source'] in ('MP58', 'HoxA')
          and abs(how['axial_factor'] - want / (0.76 if how['source'] == 'MP58' else 0.64)) < 1e-6
          and how['axial_factor'] != 1.0,
          str(how))
    ff = PB.fit_families(tpl, VOX)
    got_sz = ff['fits'][ff['best']]['params']['sigma_z_um']
    check('and the rescaled template really has that sigma_z (within 8%)',
          abs(got_sz - want) / want < 0.08, f'{got_sz:.3f} vs predicted {want:.3f}')
    check('the axial rescale leaves the lateral shape alone',
          abs(ff['fits'][ff['best']]['params']['sigma_xy_um'] - 0.27) < 0.02)
    # THE LINE, NOT THE POINTS: the three measured sigma_z are not
    # collinear in log10(kb) (0.64, 0.76, 1.07 at 5, 50, 200), so the
    # least-squares line passes above the middle one. What matters is
    # that it rises with the resolution and stays inside the measured
    # range, and that one point is no line at all.
    v5, v50, v200 = (PB._sigma_z_at(back, kb) for kb in (5.0, 50.0, 200.0))
    check('sigma_z on log10(kb): a rising line inside the measured range',
          v5 < v50 < v200 and 0.6 <= v5 <= 0.8 and 0.9 <= v200 <= 1.15
          and PB._sigma_z_at(back[:1], 50.0) is None,
          f'{v5:.3f} {v50:.3f} {v200:.3f}')


def test_engine_and_calibration():
    print('the engine matches with the chosen template; the calibration '
          'has a Platt per resolution')
    from codelab_pipeline.localization import psf_bank as PB, engine as E
    from codelab_pipeline.training import classify as C
    t5, t50, t200 = halo(0.64), halo(0.76), halo(1.07)
    mean = PB.normalise((t5 + t50 + t200) / 3.0)
    cands = [{'template': t5, 'genomic_resolution_kb': 5.0, 'source': 'HoxA',
              'sigma_xy_um': 0.22, 'sigma_z_um': 0.64},
             {'template': t200, 'genomic_resolution_kb': 200.0, 'source': 'chr19',
              'sigma_xy_um': 0.27, 'sigma_z_um': 1.07}]
    with tempfile.TemporaryDirectory(dir='D:/claude-tmp') as d:
        path = PB.save(os.path.join(d, 'psf_bank.h5'), mean, VOX, candidates=cands)
        e = E.make_engine('psf-match', bank=path, voxel_um=VOX, resolution_kb=200.0)
        check("'select' with a known resolution takes that candidate",
              e.meta['template_choice']['how'] == 'exact'
              and e.meta['template_choice']['source'] == 'chr19'
              and len(e.templates) == 1 and e.meta['n_candidates'] == 2)
        e2 = E.make_engine('psf-match', bank=path, voxel_um=VOX)
        check('and the pooled mean when none is known',
              e2.meta['template_choice']['how'] == 'pooled')
        e3 = E.make_engine('psf-match', bank=path, voxel_um=VOX, resolution_kb=200.0,
                           template_mode='all')
        check("'all' hands the matcher every candidate and the mean",
              len(e3.templates) == 3 and e3.meta['template_mode'] == 'all')
        e4 = E.make_engine('psf-match', bank=path, voxel_um=VOX, resolution_kb=200.0,
                           template_mode='pooled')
        check("'pooled' forces the mean", e4.meta['template_choice']['how'] == 'pooled')
        check('templates are cut to the search size',
              all(tuple(np.asarray(t).shape) == (7, 7, 11) for t in e3.templates))

        cal = C.MultispotCalibration((1.0, 0.0), (7, 7, 11),
                                     per_resolution={'5': {'platt': [2.0, 0.5], 'n': 500},
                                                     '200': {'platt': [0.5, -0.2], 'n': 108}})
        check('an exact resolution gets its own Platt',
              cal.platt_for(5.0) == (2.0, 0.5) and cal.platt_for(200) == (0.5, -0.2))
        check('a resolution near one (within 20%) uses it; far from all, the '
              'pooled pair', cal.platt_for(5.5) == (2.0, 0.5)
              and cal.platt_for(50.0) == (1.0, 0.0) and cal.platt_for(None) == (1.0, 0.0))
        p3 = np.array([0.3, 0.6, 0.9])
        check('score() takes the resolution',
              not np.allclose(cal.score(p3, resolution_kb=5.0), cal.score(p3))
              and np.allclose(cal.score(p3, resolution_kb=50.0), cal.score(p3)))
        cp = cal.save(os.path.join(d, 'psf_multispot.json'))
        back = C.MultispotCalibration.load(cp)
        check('the per-resolution pairs round-trip',
              back.per_resolution == cal.per_resolution and back.platt == cal.platt)
    from codelab_pipeline.localization import psfmatcher as PM, tracing_v2 as T2
    src = inspect.getsource(PM.PsfMatcherV3Engine)
    check("the v3 engine hands its resolution and template mode to the matcher",
          "resolution_kb=(self.context or {}).get('genomic_resolution_kb')" in src
          and 'template_mode=self.template_mode' in src)
    check('and the joint p_exist asks the calibration with the resolution',
          'resolution_kb=resolution_kb' in inspect.getsource(PM._joint))
    p = T2.V2Params(genomic_resolution_kb=50.0, template_mode='all')

    class Eng:
        context = {}
        template_mode = 'select'
    eng = Eng()
    p._give_context(eng)
    check('V2Params gives an engine both the resolution and the mode',
          eng.context.get('genomic_resolution_kb') == 50.0 and eng.template_mode == 'all')


def test_training_and_ab_plumbing():
    print('training writes candidates and a Platt per resolution; the A/B '
          'has the template arms')
    import tools.train_spotmodel as T
    import tools.fiducial_ab as AB
    src = inspect.getsource(T.main)
    check('each experiment template becomes a candidate, tagged',
          'bank_cands.append(' in src and 'candidates=bank_cands' in src)
    check('a Platt per resolution from bundles with 150+ judged matches',
          'cal.per_resolution = per' in src and "int(rb.get('n', 0)) < 150" in src)
    asrc = inspect.getsource(AB._init)
    check("the A/B runs 'select', 'pooled' and 'all' arms at the registry's "
          "step_kb", "'v2+lf/pooled'" in asrc and "'v2+lf/all'" in asrc
          and 'kb = exp.step_kb' in asrc)
    from tools.experiments import EXPERIMENTS
    check('the registry knows the HoxA 2026-08-22 store at 5 kb',
          EXPERIMENTS['HoxA_0822'].step_kb == 5
          and EXPERIMENTS['HoxA_0822'].config.endswith('2026-09-01-DI_HoxA.xml'))


def main():
    test_bank_and_selection()
    test_engine_and_calibration()
    test_training_and_ab_plumbing()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
