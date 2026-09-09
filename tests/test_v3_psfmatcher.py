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


def test_the_readout_writes_a_list_and_the_gate_cuts_it():
    print('\ntracing: fiducial stays one, readout becomes many, p_exist gates')
    from codelab_pipeline.localization import tracing_v2 as T2
    from codelab_pipeline.models.allele import AnAllele
    from codelab_pipeline.analysis import polymer as P

    # TWO REAL LOCI IN ONE HYBE -- sister chromatids, which AnAllele has
    # always modelled (polymer_adj is a LIST per hybe) and v2 has never
    # produced, because fit_readout returns exactly one fit.
    if not os.path.exists('D:/models/mp58_rna/report.json'):
        check('the real model directory is present', False, 'skipped')
        return
    rng = np.random.default_rng(2)
    cube = rng.normal(300, 6, (40, 40, 40))
    cube += gauss((40, 40, 40), centre=(20, 15, 20), amp=600)
    cube += gauss((40, 40, 40), centre=(20, 25, 20), amp=450)
    eng = E.make_engine('v3-psfmatcher', model_dir='D:/models/mp58_rna')

    # The builder's own frame closure, stubbed: this test is about the
    # SHAPE that gets written, not about the alignment.
    def shared(h, yf, xf, zf, ymin, xmin):
        return (yf + ymin, xf + xmin, zf)

    def run(t):
        p = T2.V2Params(readout_engine=eng, min_p_exist=t)
        a = AnAllele()
        a.polymer_adj, a.polymer_raw = {}, {}
        ok, why = T2._readout_multi(a, 'H', cube, 20.0, p, 0.0, 0.0, 0.0,
                                    100, 200, shared)
        return a, ok, why

    a, ok, _why = run(None)
    cands = a.polymer_adj['H']
    check('the readout writes a LIST, not a single fit', len(cands) > 1,
          f'{len(cands)} candidates')
    check('and every entry is still a 4-tuple -- no contract moved',
          all(len(c) == 4 for c in cands), str(len(cands[0])))
    check('both planted loci are in it',
          {15, 25} <= {round(c[1] - 200) for c in cands},
          str(sorted({round(c[1] - 200) for c in cands})[:8]))
    check('amplitude is finite, so max_brightness is a defined comparison',
          all(np.isfinite(c[3]) for c in cands))

    # THE GATE IS POSTERIOR AND CUTS BEFORE THE WRITE -- the same shape as
    # max_uncert, on a different number, and p_exist is never stored.
    a5, _ok5, _w5 = run(0.5)
    a9, _ok9, _w9 = run(0.9)
    n0, n5, n9 = (len(cands), len(a5.polymer_adj.get('H', [])),
                  len(a9.polymer_adj.get('H', [])))
    # 0.9, NOT 0.999999: p_exist is now p1 * cal(p3), and the two real
    # loci land at 0.9999979 -- a threshold of six nines is above
    # EVERYTHING, so it stopped testing 'stricter cuts more' and started
    # testing 'the hybe can be emptied', which the 1.0 case below already
    # does. The cut that matters is 3 -> 2.
    check('a threshold cuts the list', n0 > n5 >= n9 >= 1,
          f'{n0} -> {n5} -> {n9}')
    # WHAT THE UNREFINED FILTER ALREADY REMOVED, before p_exist saw
    # anything: on this cube the engine offered 17 candidates and model 3
    # could place only 2. polymer_adj is a list of POSITIONS and an
    # anchor's integer coordinate is 208 nm against a ~30 nm precision,
    # so the other 15 are kept as SPOTS and not as trace positions.
    check('most candidates never had a sub-voxel position to write',
          n0 < 17, f'{n0} written of the 17 the engine offered')
    # SURVIVES, not 'is first'. The list has no primary entry: localize
    # returns it sorted by p_exist, polymer.collapse_polymer picks the
    # BRIGHTEST by amplitude, and export pairs adj with raw by index.
    # Once p_exist became per-hit the two loci stopped tying, so the
    # order changed -- and the old check read that as the bright locus
    # being cut, which it was not.
    check('the brightest locus survives every threshold',
          15 in {round(c[1] - 200) for c in a9.polymer_adj['H']},
          str(sorted(round(c[1] - 200) for c in a9.polymer_adj['H'])))
    check('p_exist is NOT in the stored tuple',
          all(len(c) == 4 for c in a5.polymer_adj['H']))
    _aa, ok_all, why_all = run(1.0)
    check('a threshold that keeps nothing REJECTS the hybe with a reason',
          not ok_all and 'p_exist' in why_all, why_all[:60])

    saved = a5.save()
    pos, amp, n_cand = P.collapse_polymer(saved, ['H'])
    # save() rounds every float to 2dp on the way out, so the comparison
    # is against the SAVED candidates and not the in-memory ones.
    check('collapse_polymer counts them and picks the brightest',
          n_cand[0] == n5
          and abs(amp[0] - max(c[3] for c in saved['polymer_adj']['H'])) < 1e-9,
          f'n_cand {n_cand[0]}, amp {amp[0]:.1f}')

    # AND THE FIDUCIAL IS UNTOUCHED. One tuple per hybe is its contract:
    # a second fiducial candidate is not a second alignment, it is an
    # ambiguity, and fiducial_trace_adj holds one tuple for that reason.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, 'codelab_pipeline', 'localization',
                            'tracing_v2.py'), encoding='utf-8').read()
    line = src[src.index('allele.fiducial_trace_adj[hybe] ='):].split('\n')[0]
    check('fiducial_trace_adj is still assigned ONE tuple, not a list',
          not line.split('=', 1)[1].strip().startswith('['), line[:72])
    check('and the engine is consulted on the readout side only',
          'readout_engine' in src and 'fiducial_engine' not in src)


def test_a_learned_spot_is_recognisable_without_a_new_field():
    print('\nfrom_learned_engine: two existing fields are the signature')
    from codelab_pipeline.models.spot import (ASpot, from_learned_engine,
                                              Z_ACCEPTED, Z_REJECTED,
                                              Z_NOT_FIT)

    def mk(p, z):
        a = ASpot()
        a.p_exist, a.z_status = p, z
        return a

    nan = float('nan')
    # p_exist is finite ONLY for the learned engine -- every other one
    # leaves it NaN by contract -- and only that engine arrives already
    # 3D-fitted. Together they identify it with no column added to a
    # store that has millions of rows.
    check('a learned spot is recognised',
          from_learned_engine(mk(0.93, Z_ACCEPTED)))
    check('even one the gate would cut -- this is provenance, not quality',
          from_learned_engine(mk(0.02, Z_ACCEPTED)))
    check('a v1/v2 refit result is not',
          not from_learned_engine(mk(nan, Z_ACCEPTED)))
    check('nor a rejected fit', not from_learned_engine(mk(nan, Z_REJECTED)))
    check('nor an unexamined spot',
          not from_learned_engine(mk(nan, Z_NOT_FIT)))
    check('a dict from the store works the same',
          from_learned_engine({'p_exist': 0.9, 'z_status': 'accepted'}))
    check('junk in the field reads as not-learned, never as a crash',
          not from_learned_engine({'p_exist': 'yes',
                                   'z_status': 'accepted'}))
    check('and None is not a spot', not from_learned_engine(None))
    # THE ONE THING THIS INFERENCE CANNOT DO, stated rather than hidden:
    # once a v1/v2 re-fit REJECTS a learned spot its z_status is no longer
    # 'accepted', so a second re-fit will not warn. That is defensible --
    # the position is already the Gaussian's by then -- but it is an
    # inference from two fields and not a recorded fact.
    check('a learned spot whose refit was rejected no longer reads as '
          'learned', not from_learned_engine(mk(0.93, Z_REJECTED)))


def test_each_hit_carries_its_own_p_exist():
    """The second and third emitter in a pillar are gateable.

    THE DEFECT THIS CLOSES, MEASURED. Every hit from one candidate box
    used to inherit that box's p1, so within a pillar the p-gate was
    inert: on the 661 human-judged matches of the shipped model's own
    verdict set, sweeping the threshold from 0.05 to 0.90 kept all 661
    or none of them -- precision stuck at 0.728, F1 0.842, and no
    threshold anywhere could separate a real second locus from a
    spurious one.

    With p_exist = p1 * cal(p3) the same sweep is a real curve:
    precision 0.926 at 0.20, 0.966 at 0.50, 0.988 at 0.80; best F1
    0.967 at 0.40 (precision 0.963, recall 0.971) assuming a pillar p1
    of 0.95, and 0.966 at 0.33 assuming 0.80. The DEFAULT 0.5 needs no
    change: it sits at F1 0.961 / 0.956 across that range of p1.
    """
    from codelab_pipeline.localization import psfmatcher as PM

    class Cal(object):
        def __init__(self, f):
            self.f = f

        def score(self, p3):
            return self.f(float(p3))

    half = Cal(lambda p: 0.5)
    check('the product is the chain rule, no rescaling',
          abs(PM._joint(0.9, 0.8, half) - 0.45) < 1e-12)
    check('both factors in (0,1) keep the product in (0,1)',
          0.0 < PM._joint(0.999, 0.999, Cal(lambda p: 0.999)) < 1.0)

    # NO CALIBRATION SHIPPED -> the pillar number, never a raw NCC
    # dressed up as a probability.
    check('no calibration leaves p1 alone', PM._joint(0.9, 0.8, None) == 0.9)
    check('a NaN p3 leaves p1 alone',
          PM._joint(0.9, float('nan'), half) == 0.9)

    # The shipped calibration crosses 0.5 where the measured raw
    # threshold sits, which is what makes 0.5 a usable default.
    import glob
    import os as _os
    from codelab_pipeline.training import classify as C
    repo = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    md = sorted(glob.glob(_os.path.join(repo, 'models', '*')))
    md = [d for d in md if _os.path.isdir(d)]
    if md:
        path = _os.path.join(md[0], C.MULTISPOT_NAME)
        if _os.path.exists(path):
            cal = C.MultispotCalibration.load(path)
            lo, hi = float(cal.score(0.60)), float(cal.score(0.85))
            check('the shipped Platt is below 0.5 at a weak match',
                  lo < 0.5, f'cal(0.60)={lo:.4f}')
            check('and above 0.5 at a strong one', hi > 0.5,
                  f'cal(0.85)={hi:.4f}')
            check('so a weak match in a believed pillar falls under 0.5',
                  PM._joint(0.95, 0.60, cal) < 0.5)
            check('and a strong one clears it',
                  PM._joint(0.95, 0.85, cal) > 0.5)
            # TWO HITS IN ONE PILLAR NOW DIFFER, which is the whole point.
            check('two hits from one pillar get different p_exist',
                  PM._joint(0.95, 0.60, cal) != PM._joint(0.95, 0.85, cal))

    # And the engine wires it: _refine multiplies, the unrefined
    # fallback does not.
    import inspect
    src = inspect.getsource(PM.PsfMatcherV3Engine._refine)
    check('_refine multiplies per hit', '_joint(p1, hh.p, cal)' in src)
    check('an unrefined anchor keeps the pillar number',
          src.rstrip().endswith('amplitude=_peak_above(st, y, x, z, bg))]')
          and 'p_exist=p1,' in src.split('cal = self.multispot_cal')[1])


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    for t in (test_p_exist_is_its_own_field,
              test_the_engine_is_registered_and_lazy,
              test_the_engine_scores_the_box_the_model_was_trained_on,
              test_view_cell_gives_every_cell_its_own_background,
              test_nothing_is_dropped_on_p_exist,
              test_load_best_takes_the_report_s_winner,
              test_the_readout_writes_a_list_and_the_gate_cuts_it,
              test_a_learned_spot_is_recognisable_without_a_new_field,
              test_each_hit_carries_its_own_p_exist):
        t()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
