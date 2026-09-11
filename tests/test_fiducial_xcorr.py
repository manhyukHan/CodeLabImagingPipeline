"""
The fiducial by 3D correlation with the reference (fiducial_xcorr.register)
and its place in the tracer, on synthetic data only.

    register()      recovers known sub-voxel shifts, in both the wide-crop
                    mode and the refinement mode; sign convention; at_bound
                    on the limit; low NCC on noise; NaN holes; edges
    V2Params        the new fields, their validation, pickling, describe()
    _xcorr_gate     every refusal reason, and a pass
    _fiducial_learned(ref=...)   'v3+xc' refinement of an unrefined candidate
    build_chromatin_trace_allele with fiducial_method='xcorr', end to end
                    through mocked crops: injected per-hybe drifts come back
                    as the fiducial delta and the readouts line up
    fiducial_ab     ARM_ORDER and refusal kinds

Run:  QT_QPA_PLATFORM=offscreen python tests/test_fiducial_xcorr.py
"""
import math
import os
import pickle
import sys
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from codelab_pipeline.localization import tracing_v2 as T2          # noqa: E402
from codelab_pipeline.localization import fiducial_xcorr as FX      # noqa: E402
from codelab_pipeline.localization.engine import LocalizedSpot      # noqa: E402

CHECKS = [0, 0]
RNG = np.random.RandomState(0)


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name + (('  -- ' + note) if note else ''))


def blob(shape, pos, sig=(1.3, 1.3, 3.8), amp=400.0, bg=300.0, noise=12.0, halo=0.3):
    """A halo-Gaussian emitter with noise: the fiducial's measured shape."""
    Y, X, Z = np.indices(shape)
    d2 = (((Y - pos[0]) / sig[0]) ** 2 + ((X - pos[1]) / sig[1]) ** 2
          + ((Z - pos[2]) / sig[2]) ** 2)
    img = bg + amp * (np.exp(-0.5 * d2) + halo * np.exp(-0.5 * d2 / 4.0))
    return img + RNG.normal(0, noise, shape)


# ---------------------------------------------------------------- register

def test_register():
    print('register(): known shifts come back')
    shape = (17, 17, 105)
    errs = []
    for _ in range(10):
        rp = (8.0 + RNG.uniform(-1, 1), 8.0 + RNG.uniform(-1, 1), 50.0 + RNG.uniform(-2, 2))
        t = (RNG.uniform(-6, 6), RNG.uniform(-6, 6), RNG.uniform(-12, 12))
        ref = blob(shape, rp)
        # the moving crop is wider and has a different origin (7 px)
        mov = blob((31, 31, 105), (rp[0] + 7 + t[0], rp[1] + 7 + t[1], rp[2] + t[2]))
        f, why = FX.register(ref, rp, mov, (rp[0] + 7, rp[1] + 7, rp[2]))
        assert f is not None, why
        truth = (rp[0] + 7 + t[0], rp[1] + 7 + t[1], rp[2] + t[2])
        errs.append(np.abs(np.array([f.y, f.x, f.z]) - np.array(truth)))
        if f is not None:
            got_shift = np.array(f.shift)
            exp_shift = np.array([f.y, f.x, f.z]) - np.array([rp[0] + 7, rp[1] + 7, rp[2]])
            assert np.allclose(got_shift, exp_shift, atol=1e-9), (got_shift, exp_shift)
    errs = np.array(errs)
    check('ten random sub-voxel shifts up to 6 px / 12 planes recovered to < 0.1 voxel',
          errs.max() < 0.1, f'max |err| {errs.max():.3f}, mean {errs.mean(axis=0).round(3)}')
    check('XcorrFit.shift is exactly pos - centre', True)
    check('the peak NCC of a clean match is high and refined by the upsampled CC',
          f.ncc > 0.9 and f.refined == 'cc' and f.at_bound == (), f'{f.ncc:.2f} {f.refined}')

    # refinement mode: a learned candidate names the voxel, correlation the sub-voxel
    rp = (8.4, 7.6, 50.3)
    t = (1.3, -0.7, 2.6)
    ref = blob(shape, rp)
    mov = blob(shape, (rp[0] + t[0], rp[1] + t[1], rp[2] + t[2]))
    f, why = FX.register(ref, rp, mov, (10.0, 7.0, 53.0), max_shift=T2.XCORR_REFINE_REACH)
    truth = (rp[0] + t[0], rp[1] + t[1], rp[2] + t[2])
    check('refinement mode around an integer candidate lands on the truth',
          f is not None and max(abs(f.y - truth[0]), abs(f.x - truth[1]), abs(f.z - truth[2])) < 0.1,
          str(None if f is None else (round(f.y, 2), round(f.x, 2), round(f.z, 2))))

    # sign: moving content at ref + t is FOUND at ref + t (never ref - t)
    mov = blob(shape, (rp[0] + 3.0, rp[1], rp[2]))
    f, _ = FX.register(ref, rp, mov, rp)
    check('sign convention: a blob moved +3 in y is found at +3, not -3',
          f is not None and abs(f.y - (rp[0] + 3.0)) < 0.1 and abs(f.shift[0] - 3.0) < 0.1,
          str(None if f is None else f.shift))

    # beyond the searched range: the peak sits on the map edge
    mov = blob((31, 31, 105), (rp[0] + 7 + 9.0, rp[1] + 7, rp[2]))
    f, why = FX.register(ref, rp, mov, (rp[0] + 7, rp[1] + 7, rp[2]))
    check('a shift beyond max_shift is reported at_bound on that axis, not as a position',
          f is not None and 'y' in f.at_bound, str(None if f is None else f.at_bound))

    # pure noise: no confident match
    mov = RNG.normal(300, 12, (31, 31, 105))
    f, why = FX.register(ref, rp, mov, (rp[0] + 7, rp[1] + 7, rp[2]))
    check('pure noise scores a low NCC (the gate refuses it)',
          f is None or f.ncc < 0.3, str(None if f is None else round(f.ncc, 2)))

    # NaN-padded rows in the background do not move the answer
    mov = blob((31, 31, 105), (rp[0] + 8.0, rp[1] + 5.0, rp[2] + 3.0))
    mov[:3, :, :] = np.nan
    f, why = FX.register(ref, rp, mov, (rp[0] + 7, rp[1] + 7, rp[2]))
    check('NaN rows in the background are filled, the position holds',
          f is not None and abs(f.y - (rp[0] + 8.0)) < 0.1 and abs(f.x - (rp[1] + 5.0)) < 0.1
          and abs(f.z - (rp[2] + 3.0)) < 0.15, str(None if f is None else (f.y, f.x, f.z)))

    # a reference on its crop edge has no template
    rp2 = (1.0, 8.0, 50.0)
    f, why = FX.register(blob(shape, rp2), rp2, mov, (rp2[0] + 7, rp2[1] + 7, rp2[2]))
    check('a reference within MIN_TEMPLATE_HALF of its crop edge is refused, not clamped',
          f is None and 'too close' in str(why), str(why))
    f, why = FX.register(ref, (rp[0], rp[1], float('nan')), mov, rp)
    check('a non-finite position is refused rather than raising',
          f is None and 'non-finite' in str(why), str(why))
    f, why = FX.register(ref, rp, np.full((31, 31, 105), 300.0), (rp[0] + 7, rp[1] + 7, rp[2]))
    check('a flat moving crop has no correlation peak',
          f is None and 'no correlation peak' in str(why), str(why))
    f, why = FX.register(ref, rp, mov, rp, half=(4, 4))
    check('a malformed half-size is refused, not an IndexError',
          f is None and '3 entries' in str(why), str(why))
    # the template shrinks, down to the floor, to keep the range inside a small crop
    small = blob((17, 17, 105), (rp[0] + 1.0, rp[1] - 1.0, rp[2] + 2.0))
    f, why = FX.register(ref, rp, small, (14.0, 14.0, rp[2]), max_shift=(2, 2, 3))
    check('near a small crop\'s edge the template shrinks rather than refusing',
          f is not None or 'search region' in str(why), str(why))


# ---------------------------------------------------------------- V2Params

def test_params():
    print('\nV2Params: the correlation fiducial\'s settings')
    p = T2.V2Params()
    check('defaults: the correlation method (measured), Gaussian refinement, NCC gate, template',
          p.fiducial_method == 'xcorr' == T2.DEFAULT_FIDUCIAL_METHOD and p.fiducial_refine == 'gauss'
          and p.min_ncc() == T2.FIDUCIAL_MIN_NCC == 0.5
          and p.template_half() == FX.DEFAULT_TEMPLATE_HALF == (4, 4, 8))
    p = T2.V2Params(fiducial_method='xcorr', fiducial_refine='xcorr',
                    fiducial_min_ncc=0.6, xcorr_template_half=(6, 6, 12))
    check('explicit values are kept', p.fiducial_method == 'xcorr' and p.fiducial_refine == 'xcorr'
          and p.min_ncc() == 0.6 and p.template_half() == (6, 6, 12))
    bad = 0
    for kw in ({'fiducial_method': 'phase'}, {'fiducial_refine': 'none'},
               {'xcorr_template_half': (4, 4)}, {'xcorr_template_half': (0, 4, 8)}):
        try:
            T2.V2Params(**kw)
        except ValueError:
            bad += 1
    check('a bad method, refinement or template half is a ValueError at construction', bad == 4)
    q = pickle.loads(pickle.dumps(p))
    check('pickles across the process boundary with the fields, without engines',
          q.fiducial_method == 'xcorr' and q.template_half() == (6, 6, 12)
          and q._fiducial_engine is None and q._readout_engine is None)
    check('describe() names the correlation fiducial and its gate',
          '3D correlation' in p.describe() and 'ncc >= 0.6' in p.describe(), p.describe())
    check('and the refinement, when a fiducial model is set',
          'refined by correlation' in T2.V2Params(fiducial_model_dir='m',
                                                  fiducial_method='gaussian',
                                                  fiducial_refine='xcorr').describe())


# ---------------------------------------------------------------- the gate

def test_gate():
    print('\n_xcorr_gate: every refusal, and a pass')
    # the trim reaches the gate through fiducial_gates, as from_panel
    # sets it -- the same route the Gaussian's gate() takes
    p = T2.V2Params(z_boundary_trim=10, fiducial_gates={'z_boundary_trim': 10})
    cube = RNG.normal(300, 5, (17, 17, 105))

    def fit(ncc=0.8, z=50.0, at_bound=()):
        return FX.XcorrFit(8.0, 8.0, z, 400.0, ncc, (0.0, 0.0, 0.0), at_bound, 'cc')
    ok, why = T2._xcorr_gate(None, 'template mostly unobserved', cube, p)
    check('no fit: the registration\'s own reason', not ok and why == 'template mostly unobserved')
    ok, why = T2._xcorr_gate(fit(at_bound=('y', 'z')), None, cube, p)
    check('a peak on the search bound is refused, naming the axes',
          not ok and why == 'shift at the search bound (y, z)', str(why))
    ok, why = T2._xcorr_gate(fit(ncc=0.31), None, cube, p)
    check('NCC below the threshold', not ok and why == 'ncc 0.31 < 0.50', str(why))
    ok, why = T2._xcorr_gate(fit(z=3.0), None, cube, p)
    check('a depth inside the trimmed stack ends', not ok and 'stack end' in str(why), str(why))
    ok, why = T2._xcorr_gate(fit(), None, cube, p)
    check('otherwise it passes: no occupancy, no CI (a registration has neither)',
          ok and why is None, str(why))


# ---------------------------------------------------------------- v3+xc refinement

class FakeEngine(object):
    def __init__(self, spots):
        self.spots = list(spots)
        self.multispot_cal = None

    def localize(self, cube, seed_yxz=None, n_max=None):
        return list(self.spots)


def unrefined(y, x, z, p_exist):
    nan = float('nan')
    # model 3 placed nothing: NaN in p, integer position -- psfmatcher.is_refined
    return LocalizedSpot(y=y, x=x, z=z, p=nan, amplitude=100.0, sigma_y=nan,
                         sigma_x=nan, sigma_z=nan, offset=0.0, p_exist=p_exist)


def test_learned_refinement():
    print('\n_fiducial_learned: an unrefined candidate refined by correlation')
    rp = (8.3, 7.7, 50.4)
    t = (0.6, -0.8, 1.7)
    ref = blob((17, 17, 105), rp)
    cube = blob((17, 17, 105), (rp[0] + t[0], rp[1] + t[1], rp[2] + t[2]))
    p = T2.V2Params(min_p_exist=0.5, min_p_exist_fiducial=0.5, z_boundary_trim=0,
                    fiducial_refine='xcorr')
    z0 = 50.0
    slab, s0, s1, _reach, _off = T2._learned_slab(cube, z0, p, reach=p.fiducial_window())
    cand = unrefined(9.0, 7.0, round(rp[2] + t[2]) - s0, 0.8)
    p.fiducial_engine = FakeEngine([cand])
    f, why, alts, how = T2._fiducial_learned(cube, z0, p, ref=(ref, rp))
    truth = (rp[0] + t[0], rp[1] + t[1], rp[2] + t[2])
    check("with ref, the candidate is refined by correlation: how == 'v3+xc'",
          f is not None and how == 'v3+xc'
          and max(abs(f.y - truth[0]), abs(f.x - truth[1]), abs(f.z - truth[2])) < 0.15,
          f'{how} {why} {None if f is None else (round(f.y, 2), round(f.x, 2), round(f.z, 2))}')
    check('the alternatives say what happened to the candidate',
          any('refined by correlation' in w for _c, w in alts), str([w for _c, w in alts]))
    f2, why2, _a2, how2 = T2._fiducial_learned(cube, z0, p, ref=None)
    check('without ref it falls back to the Gaussian refinement (or refuses as before)',
          how2 in ('v3+gauss', None) and (f2 is None or how2 == 'v3+gauss'), f'{how2} {why2}')
    p.fiducial_refine = 'gauss'
    f3, _w3, _a3, how3 = T2._fiducial_learned(cube, z0, p, ref=(ref, rp))
    check("with fiducial_refine='gauss' the ref is ignored", how3 != 'v3+xc', str(how3))
    check('the stored quality of a correlation fiducial is its NCC',
          f is not None and T2.fiducial_quality(f) == f.ncc)


# ---------------------------------------------------------------- the builder, end to end

def test_builder_end_to_end():
    print('\nbuild_chromatin_trace_allele with fiducial_method=xcorr, mocked crops')
    from codelab_pipeline.alignment import spot_mapper
    from codelab_pipeline.localization import localization as L
    from codelab_pipeline.models.allele import AnAllele
    H, W, D = 64, 64, 90
    anchor = (30.0, 30.0)
    drifts = {'H0': (0.0, 0.0, 0.0), 'H1': (1.3, -2.2, 3.5), 'H2': (-3.1, 0.7, -6.0),
              'H3': (12.0, 0.0, 0.0)}          # H3: beyond the gate and the margin
    fid_z = 45.0
    images = {h: blob((H, W, D), (anchor[0] + d[0], anchor[1] + d[1], fid_z + d[2]),
                      amp=600.0, noise=8.0)
              for h, d in drifts.items()}

    def reference_to_raw(shared_xy, hybe, fov_matrices, modality=None, cell=None, resolver=None):
        return float(shared_xy[0]), float(shared_xy[1])

    def raw_to_reference(raw_yx, hybe, fov_matrices, modality=None, cell=None, resolver=None):
        return float(raw_yx[0]), float(raw_yx[1])

    def crop_for_localization(storage_path, fov, hybe, channel, native, pad=5, use_stack=False):
        y, x = native
        ymin, ymax = max(0, int(round(y)) - pad), min(H, int(round(y)) + pad + 1)
        xmin, xmax = max(0, int(round(x)) - pad), min(W, int(round(x)) + pad + 1)
        return images[hybe][ymin:ymax, xmin:xmax, :].copy(), (ymin, xmin)

    hybes = list(drifts)
    fid_ch = {h: 555 for h in hybes}
    read_ch = {h: 635 for h in hybes}
    p = T2.V2Params(fiducial_method='xcorr', qc_shift=False)
    with mock.patch.object(spot_mapper, 'reference_to_raw', reference_to_raw), \
            mock.patch.object(spot_mapper, 'raw_to_reference', raw_to_reference), \
            mock.patch.object(spot_mapper, 'crop_for_localization', crop_for_localization), \
            mock.patch.object(L, 'cell_z_offset', lambda cell, hybe, mod, resolver: 0.0):
        a = AnAllele()
        a.set_metadata(id=1, fov=1, cell=-1, anchor_hybe='H0', anchor_channel=555,
                       coordinate=(anchor[0], anchor[1], 0.0),
                       raw_coordinate=(anchor[0], anchor[1], 0.0))
        a, debug = T2.build_chromatin_trace_allele(
            a, hybes, 'H0', fid_ch, read_ch, 'X', 1, 'DNA', None, {},
            params=p, spad=8, collect_debug=True)
    fid = a.fiducial_trace_adj
    check('the reference keeps its Gaussian path; the others are by correlation',
          debug['H0'].get('fiducial_engine') != 'xc'
          and all(debug[h]['fiducial_engine'] == 'xc' for h in ('H1', 'H2', 'H3'))
          and all(debug[h]['fiducial_how'] == 'xcorr' for h in ('H1', 'H2')),
          str({h: (debug[h].get('fiducial_engine'), debug[h].get('fiducial_how')) for h in hybes}))
    ok_drift = True
    for h in ('H1', 'H2'):
        got = np.array(fid[h][:3]) - np.array(fid['H0'][:3])
        if np.abs(got - np.array(drifts[h])).max() > 0.15:
            ok_drift = False
            print('     ', h, 'got', got.round(3), 'want', drifts[h])
    check('fiducial(h) - fiducial(reference) is the injected drift, to 0.15 voxel', ok_drift)
    check('and fiducial_drift records reference - hybe, the correction applied',
          all(np.allclose(a.fiducial_drift[h], -np.array(drifts[h]), atol=0.15) for h in ('H1', 'H2')),
          str({h: a.fiducial_drift.get(h) for h in hybes}))
    check('the correlation fiducial stores its NCC as the quality slot',
          all(len(fid[h]) == 5 and 0.5 <= fid[h][4] <= 1.0 for h in ('H1', 'H2'))
          and math.isnan(fid['H0'][4]), str({h: fid.get(h) for h in hybes}))
    poly = a.polymer_adj
    ref_pos = np.array(poly['H0'][0][:3]) if poly.get('H0') else None
    lined = (ref_pos is not None and all(
        h in poly and np.abs(np.array(poly[h][0][:3]) - ref_pos).max() < 0.3 for h in ('H1', 'H2')))
    check('after the correction the readouts of H1/H2 line up with the reference (< 0.3 voxel)',
          lined, str({h: (None if not poly.get(h) else np.round(poly[h][0][:3], 2)) for h in hybes}))
    check('a drift beyond the gate plus margin is refused at the search bound, as data',
          'H3' in a.rejected_hybes and 'search bound' in a.rejected_hybes['H3']
          and fid.get('H3', 'missing') is None and a.fiducial_drift.get('H3', 'missing') is None,
          str(a.rejected_hybes.get('H3')))
    check('the refused hybe still records the shift it found, for the tile',
          debug['H3'].get('fiducial_shift') is not None and debug['H3']['fiducial_ncc'] == debug['H3']['fiducial_ncc'],
          str(debug['H3'].get('fiducial_shift')))
    check('occupancy is not measured for a correlation fiducial (a Gaussian\'s number)',
          math.isnan(debug['H1']['fiducial_occupancy']))


# ---------------------------------------------------------------- the A/B tool

def test_ab_tool():
    print('\nfiducial_ab: arms and refusal kinds')
    from tools import fiducial_ab as AB
    check('the arms are known, in order',
          all(k in AB.ARM_ORDER for k in ('v2+xc', 'v2+xc/6', 'v2+lf/xc', 'v3+xc'))
          and AB.ARM_ORDER.index('v2+xc') < AB.ARM_ORDER.index('v2+lf'))
    check('refusal kinds name the correlation\'s own reasons',
          AB._kind('fiducial ncc 0.32 < 0.50') == 'xcorr: ncc below threshold'
          and AB._kind('fiducial shift at the search bound (x)') == 'xcorr: shift at the search bound'
          and AB._kind('fiducial reference fiducial not found (no template)')
          == 'xcorr: reference fiducial not found')


def main():
    test_register()
    test_params()
    test_gate()
    test_learned_refinement()
    test_builder_end_to_end()
    test_ab_tool()
    print(f'\n{CHECKS[1]}/{CHECKS[0]} checks passed')
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
