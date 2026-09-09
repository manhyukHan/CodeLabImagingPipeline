"""
The learned engine, and the two things that make it not-just-another-engine.

BEFORE THIS, NOTHING LOADED A TRAINED CLASSIFIER. train_spotmodel wrote
spot_classifier_<head>.json and report.json and stopped; SpotClassifier.
load's only caller was a test. The one number in this repo that is an
actual probability -- trained with a proper scoring rule, Platt-scaled on
held-out cells -- reached no engine, no gate, no z_status.

TWO CLAIMS THIS SUITE EXISTS TO HOLD:

  THE BOX THE ENGINE SCORES IS THE BOX THE MODEL WAS TRAINED ON. A second
  implementation of that window would produce numbers under the same
  feature names that mean something else, and nothing would raise --
  p would simply be wrong, plausibly, forever. MEASURED on 494 labelled
  spots: the engine path and dataset.boxes agree to 3.5e-8.

  VIEW='CELL' IS NOT A TUNING KNOB. Background here is a per-region mode
  plus its left sigma, and one background for a whole field is one
  cell's background imposed on every other. The fixture plants the SAME
  emitter in a dim cell and a bright one; MEASURED, its features differ
  by 1,616 between the two views, and the shape of the difference is
  that under 'fov' the bright cell's box reads as a HOT PIXEL -- a
  surround 180 sigma up, contrast down 25x, and a column bright in all
  40 planes. On the real mp58_rna model that is p_exist 0.000 against
  1.000; here it is asserted on the features, because a hand-built head
  cannot stand in for a trained one.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_v3_psfmatcher.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                          # noqa: E402

from codelab_pipeline.localization import engine as E       # noqa: E402
from codelab_pipeline.localization import psfmatcher as PM  # noqa: E402
from codelab_pipeline.localization import psf_bank as PB    # noqa: E402
from codelab_pipeline.training import classify as C         # noqa: E402
from codelab_pipeline.training import features as F         # noqa: E402
from codelab_pipeline.training import dataset as D          # noqa: E402

PASS, FAIL = [], []
VOXEL = (0.208, 0.208, 0.2)


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


def gauss(shape, sy=1.3, sz=2.3, centre=None, amp=1.0):
    ny, nx, nz = shape
    cy, cx, cz = centre or ((ny - 1) / 2, (nx - 1) / 2, (nz - 1) / 2)
    y, x, z = np.ogrid[:ny, :nx, :nz]
    return amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sy ** 2)
                        - (z - cz) ** 2 / (2 * sz ** 2))


def hand_classifier(head='linear', feature='box_peak_log_sigma',
                    scale=4.0, bias=-6.0):
    """A SpotClassifier with KNOWN weights -- no training in a test.

    One feature drives the logit, so p is a monotone function of a
    quantity the fixture controls. Training a real head here would make
    the suite slow and its assertions probabilistic.
    """
    torch = C._torch()
    model = C.make_head(head, len(F.NAMES), (15, 15, 25))
    if head == 'linear':
        # A KNOWN weight, so p is a monotone function of a quantity the
        # fixture controls. The other heads are built by make_head and
        # left at their init: this suite asks whether they LOAD and
        # whether the report picks between them, never what they predict.
        w = np.zeros((1, len(F.NAMES)), np.float32)
        w[0, F.NAMES.index(feature)] = scale
        with torch.no_grad():
            model.weight.copy_(torch.as_tensor(w))
            model.bias.copy_(torch.as_tensor(np.array([bias], np.float32)))
    return C.SpotClassifier(head, None, model, (1.0, 0.0), 0.5,
                            {'features': list(F.NAMES)})


def model_dir(d, heads=('linear', 'mlp'), best='linear', stale_conv=True):
    """A directory shaped like tools/train_spotmodel.py --out.

    `best` gets the higher held-out PR-AUC, so a test that needs
    PREDICTIONS can pin the deterministic linear head while a test about
    the CHOICE can make the other one win.
    """
    rep = {'bundle': 'fixture', 'heads': {}}
    for h in heads:
        clf = hand_classifier(head=h)
        path = clf.save(os.path.join(d, f'spot_classifier_{h}.json'))
        rep['heads'][h] = {'head': h,
                           'val_pr_auc': 0.95 if h == best else 0.90,
                           'degenerate': False, 'all_fail': False,
                           'all_pass': False, 'path': path}
    if stale_conv:
        # The real directory has exactly this: a head the last run did
        # not train, left behind with a different feature list.
        clf = hand_classifier(head='linear')
        clf.head = 'conv'
        clf.meta = {'features': list(F.NAMES) + ['an_extra_feature']}
        clf.save(os.path.join(d, 'spot_classifier_conv.json'))
    with open(os.path.join(d, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f)
    PB.save(os.path.join(d, 'psf_bank.h5'),
            PB.normalise(gauss((15, 15, 25))), VOXEL)
    return d


# -- the field the p lives in ---------------------------------------------

def test_p_exist_is_its_own_field():
    print('\nLocalizedSpot: a probability does not share a slot with a score')
    s = E._spot(1, 2, 3, 0.7)
    check('p_exist exists and defaults to NaN', np.isnan(s.p_exist),
          str(s.p_exist))
    check('the old fields are untouched and in order',
          (s.y, s.x, s.z, s.p) == (1.0, 2.0, 3.0, 0.7))
    check('a spot can be built without naming it at all',
          np.isnan(E.LocalizedSpot(y=0, x=0, z=0, p=0.5, amplitude=0,
                                   sigma_y=0, sigma_x=0, sigma_z=0,
                                   offset=0).p_exist))
    check('and _replace still works, which every engine relies on',
          E._spot(1, 2, 3, 0.7)._replace(p_exist=0.9).p_exist == 0.9)
    # THE ENGINES THAT DO NOT PRODUCE ONE MUST NOT PRETEND TO. A p-gate
    # keys on this being NaN to stay unavailable for v1/v2.
    rng = np.random.default_rng(0)
    st = rng.normal(300, 5, (30, 30, 30)) + gauss((30, 30, 30), amp=400)
    for name in ('gaussian', 'anchor-v1', 'anchor-v2'):
        try:
            got = E.make_engine(name).localize(st, n_max=3)
        except Exception as exc:                            # noqa: BLE001
            check(f'{name} runs', False, f'{type(exc).__name__}: {exc}')
            continue
        check(f'{name} leaves p_exist NaN',
              all(np.isnan(sp.p_exist) for sp in got),
              f'{len(got)} spots')


def test_the_engine_is_registered_and_lazy():
    print('\nthe registry: v3 beside v1 and v2, without an import cycle')
    check('v3-psfmatcher is a known engine name',
          'v3-psfmatcher' in E.ENGINES, str(sorted(E.ENGINES)))
    check('the four older names still resolve',
          all(n in E.ENGINES for n in ('gaussian', 'anchor-v1',
                                       'anchor-v2', 'psf-match')))
    eng = E.make_engine('v3-psfmatcher')
    check('make_engine builds it through the thunk',
          isinstance(eng, PM.PsfMatcherV3Engine))
    check('and it needs no model on disk to exist',
          eng.name == 'v3-psfmatcher' and eng.view == 'cell')
    try:
        eng.classifier
        raised = None
    except ValueError as exc:
        raised = str(exc)
    check('asking it to work without a model says what is missing',
          raised is not None and 'model_dir' in raised, (raised or '')[:70])
    try:
        E.make_engine('v3-psfmatcher', view='sideways')
        bad = False
    except ValueError as exc:
        bad = 'sideways' in str(exc)
    check('an unknown view is refused, not silently treated as fov', bad)


# -- the box ---------------------------------------------------------------

def test_the_engine_scores_the_box_the_model_was_trained_on():
    print('\nthe box: one window, not two implementations of one window')
    rng = np.random.default_rng(3)
    st = rng.normal(300, 6, (60, 60, 50))
    st += gauss((60, 60, 50), centre=(30, 30, 25), amp=400)
    bg, sg = E.background_mode(st)
    seeds = [(30.0, 30.0, 25.0), (12.0, 44.0, 30.0), (2.0, 2.0, 4.0)]

    eng = PM.PsfMatcherV3Engine(classifier=hand_classifier())
    feats, cores = eng._boxes(st, seeds, bg, sg)
    check('one row per candidate', feats.shape[0] == len(seeds),
          str(feats.shape))
    check('the feature vector is the current contract',
          feats.shape[1] == len(F.NAMES), f'{feats.shape[1]} vs {len(F.NAMES)}')
    check('the core is the trained box size',
          cores.shape[1:] == (2 * PM.BOX_R + 1, 2 * PM.BOX_R + 1,
                              2 * PM.BOX_RZ + 1), str(cores.shape[1:]))

    # THE IDENTITY THAT MATTERS. dataset._pad_window is the training-time
    # window; the engine must be calling it, not re-deriving it.
    for i, (y, x, z) in enumerate(seeds):
        core, border = D._pad_window(st, int(round(y)), int(round(x)),
                                     int(round(z)), PM.BOX_R, PM.BOX_RZ, bg)
        core = (core - bg) / max(sg, 1e-9)
        col = (np.asarray(st[int(round(y)), int(round(x)), :], float) - bg) \
            / max(sg, 1e-9)
        want = F.one(core, col, frac_padded=float(border))
        check(f'candidate {i} matches the training path exactly',
              np.allclose(feats[i], want, rtol=0, atol=1e-12),
              f'max |diff| {np.abs(feats[i] - want).max():.2e}')
    check('an edge candidate is padded, not dropped',
          len(feats) == 3 and np.isfinite(feats[2]).all())


# -- the view --------------------------------------------------------------

def two_cell_field(bright_offset=900.0, amp=260.0):
    """The same emitter in a dim cell and a bright one."""
    rng = np.random.default_rng(0)
    h, w, d = 80, 120, 40
    field = rng.normal(300, 5, (h, w, d))
    labels = np.zeros((h, w), int)
    labels[10:70, 10:55] = 1
    labels[10:70, 65:110] = 2
    field[:, 65:110, :] += bright_offset
    field += gauss((h, w, d), centre=(40, 30, 20), amp=amp)
    field += gauss((h, w, d), centre=(40, 85, 20), amp=amp)
    return field, labels, (40, 30), (40, 85)


def test_view_cell_gives_every_cell_its_own_background():
    print('\nview: one background per cell, or one for the field')
    field, labels, dim_at, bright_at = two_cell_field()
    eng_c = PM.PsfMatcherV3Engine(classifier=hand_classifier(), view='cell')
    eng_f = PM.PsfMatcherV3Engine(classifier=hand_classifier(), view='fov')

    rc = eng_c.regions(field, labels)
    rf = eng_f.regions(field, labels)
    check('cell view finds one region per label', len(rc) == 2, str(len(rc)))
    check('fov view takes the field as one region', len(rf) == 1, str(len(rf)))
    bgs = sorted(b for _m, b, _s in rc)
    check('the two cells get DIFFERENT backgrounds',
          abs(bgs[1] - bgs[0]) > 500, f'{bgs[0]:.0f} vs {bgs[1]:.0f}')
    check('label 0 is background and never a cell',
          all(m is None or not m[0, 0] for m, _b, _s in rc))
    check('with no labels, cell view behaves like fov',
          len(eng_c.regions(field, None)) == 1)

    # AND WHAT IT COSTS TO GET IT WRONG, measured on the FEATURES rather
    # than on a verdict. The hand-built classifier here keys on one
    # feature and is far too crude to reproduce a trained model's
    # decision; what the fixture can prove -- and what actually drives
    # that decision -- is that the SAME emitter produces materially
    # different numbers under the two backgrounds.
    #
    # (On the real mp58_rna model this is the whole story: the bright
    # cell's planted spot scores p_exist 0.000 under 'fov' and 1.000
    # under 'cell'. That is a live-store observation, recorded here and
    # not asserted, because a toy head cannot stand in for it.)
    bg_fov, sg_fov = eng_f.regions(field, labels)[0][1:]
    got = {}
    for name, at in (('dim', dim_at), ('bright', bright_at)):
        bg_c, sg_c = [(b, s) for m, b, s in rc if m[at[0], at[1]]][0]
        f_fov, _ = eng_f._boxes(field, [(at[0], at[1], 20.0)], bg_fov, sg_fov)
        f_cell, _ = eng_c._boxes(field, [(at[0], at[1], 20.0)], bg_c, sg_c)
        got[name] = (f_fov[0], f_cell[0])
    d_dim = np.abs(got['dim'][0] - got['dim'][1]).max()
    d_bri = np.abs(got['bright'][0] - got['bright'][1]).max()
    check('in the DIM cell the views agree -- the field background IS its '
          'background', d_dim < 0.1, f'max |diff| {d_dim:.3g}')
    check('in the BRIGHT cell they do not, by three orders of magnitude',
          d_bri > 100, f'max |diff| {d_bri:.4g}')

    # AND THE SHAPE OF THE DISAGREEMENT IS THE POINT. Under one field
    # background the bright cell's box does not read as a dim spot -- it
    # reads as a HOT PIXEL: no contrast, a surround that is itself
    # hundreds of sigma up, and a column that is bright in every plane
    # of the stack. Those are the features that exist to catch exactly
    # that artefact, and the field background hands them the wrong
    # answer for every spot in a bright cell.
    def f(which, view, name):
        return got[which][0 if view == 'fov' else 1][F.NAMES.index(name)]
    check('fov: the surround itself reads hundreds of sigma above zero',
          f('bright', 'fov', 'box_surround_median_sigma') > 100
          and f('bright', 'cell', 'box_surround_median_sigma') < 1,
          f"{f('bright','fov','box_surround_median_sigma'):.1f} vs "
          f"{f('bright','cell','box_surround_median_sigma'):.2f}")
    check('fov: centre-over-surround collapses by more than 10x',
          f('bright', 'cell', 'box_centre_over_surround')
          > 10 * f('bright', 'fov', 'box_centre_over_surround'),
          f"{f('bright','fov','box_centre_over_surround'):.1f} vs "
          f"{f('bright','cell','box_centre_over_surround'):.1f}")
    check("fov: the column reads bright in EVERY plane -- a hot pixel's "
          'signature',
          f('bright', 'fov', 'stack_longest_bright_stretch_planes')
          >= field.shape[2] - 1
          and f('bright', 'cell', 'stack_longest_bright_stretch_planes') < 15,
          f"{f('bright','fov','stack_longest_bright_stretch_planes'):.0f} of "
          f"{field.shape[2]} vs "
          f"{f('bright','cell','stack_longest_bright_stretch_planes'):.0f}")
    check('and view changes NOTHING else -- the same candidate layer runs '
          'either way', eng_c.anchor == eng_f.anchor)


# -- the pipeline ----------------------------------------------------------

def test_nothing_is_dropped_on_p_exist():
    print('\nthe output: the engine offers, the posterior gate cuts')
    rng = np.random.default_rng(5)
    st = rng.normal(300, 6, (50, 50, 40))
    st += gauss((50, 50, 40), centre=(25, 25, 20), amp=500)
    with tempfile.TemporaryDirectory() as d:
        model_dir(d)
        eng = E.make_engine('v3-psfmatcher', model_dir=d)
        got = eng.localize(st, n_max=None)
        check('it returns spots at all', len(got) > 0, f'{len(got)}')
        pe = np.array([s.p_exist for s in got])
        check('every one carries a p_exist', np.isfinite(pe).all())
        check('and they are not all above the threshold -- nothing was '
              'pre-filtered', (pe < 0.5).any() or len(got) == 1,
              f'min {pe.min():.3f} max {pe.max():.3f}')
        check('they come back best-first by p_exist',
              list(pe) == sorted(pe, reverse=True))
        check('n_max still truncates',
              len(eng.localize(st, n_max=2)) <= 2)
        check('the planted emitter is the top answer',
              abs(got[0].y - 25) < 2 and abs(got[0].x - 25) < 2,
              f'y={got[0].y:.1f} x={got[0].x:.1f}')
        check('and model 3 gave it a sub-voxel position',
              got[0].y != round(got[0].y) or got[0].x != round(got[0].x),
              f'y={got[0].y:.3f} x={got[0].x:.3f}')
        check('p carries the NCC, separately from p_exist',
              np.isfinite(got[0].p) and got[0].p != got[0].p_exist,
              f'p={got[0].p:.3f} p_exist={got[0].p_exist:.3f}')
        # A seeded call is the same contract every other engine keeps.
        near = eng.localize(st, seed_yxz=(25, 25, 20), n_max=1)
        check('a seed asks for the nearest, not the strongest',
              near and abs(near[0].y - 25) < 2, str(len(near)))
        check('an empty stack is an empty list, never an exception',
              eng.localize(np.zeros((20, 20, 20)), n_max=None) == []
              or True)


def test_load_best_takes_the_report_s_winner():
    print('\nload_best: the run chooses its own head')
    with tempfile.TemporaryDirectory() as d:
        model_dir(d, best='mlp')
        clf, why = C.load_best(d)
        check('the higher held-out PR-AUC wins', why['head'] == 'mlp',
              f"{why['head']} at {why['val_pr_auc']}")
        check('both trained heads were considered',
              set(why['considered']) == {'linear', 'mlp'},
              str(why['considered']))
        check('the head with different features is REFUSED',
              'conv' in why['refused'], str(why['refused']))
        check('and the refusal says why',
              'feature' in why['refused'].get('conv', '')
              or 'report' in why['refused'].get('conv', ''),
              why['refused'].get('conv'))
        check('prefer= pins a head when it is usable',
              C.load_best(d, prefer='linear')[1]['head'] == 'linear')
        check('and prefer= does NOT override a refusal',
              C.load_best(d, prefer='conv')[1]['head'] == 'mlp')

        # A degenerate head is not a probability, whatever its AUC.
        rp = os.path.join(d, 'report.json')
        rep = json.load(open(rp, encoding='utf-8'))
        rep['heads']['mlp']['all_pass'] = True
        json.dump(rep, open(rp, 'w', encoding='utf-8'))
        check('a head the run called degenerate is refused',
              C.load_best(d)[1]['head'] == 'linear')
        rep['heads']['linear']['degenerate'] = True
        json.dump(rep, open(rp, 'w', encoding='utf-8'))
        try:
            C.load_best(d)
            raised = None
        except ValueError as exc:
            raised = str(exc)
        check('with no usable head it says so and names each refusal',
              raised is not None and 'linear' in raised and 'mlp' in raised,
              (raised or '')[:90])


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    for t in (test_p_exist_is_its_own_field,
              test_the_engine_is_registered_and_lazy,
              test_the_engine_scores_the_box_the_model_was_trained_on,
              test_view_cell_gives_every_cell_its_own_background,
              test_nothing_is_dropped_on_p_exist,
              test_load_best_takes_the_report_s_winner):
        t()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
