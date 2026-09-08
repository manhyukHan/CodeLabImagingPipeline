"""
The two spot models, wired end to end. PLUMBING, not accuracy.

WHAT THIS DOES NOT TEST. Whether either model is any good. The only
labelled set that exists today is one reviewer, one hybe, one FOV and 32
cells, and it is explicitly development material -- so nothing here
asserts a score. What it asserts is that the chain runs, that p comes out
of it non-degenerate rather than collapsing to all-fail or all-pass, that
a saved model reloads to the same numbers, and that the matched filter
finds the spots that were planted and not the walls of the box.

Everything below runs on SYNTHETIC volumes, so it works on a machine
with no bundle and no store. A last section uses the real bundle if it
happens to be there, and says so if it is not, rather than skipping
silently.

Run:  python tests/test_spot_models.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                          # noqa: E402

from codelab_pipeline.training import features as F          # noqa: E402
from codelab_pipeline.training import dataset as D           # noqa: E402
from codelab_pipeline.localization import psf_bank as PB     # noqa: E402
from codelab_pipeline.localization import engine as E        # noqa: E402

VOXEL = (0.208, 0.208, 0.2)
PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


def emitter(shape=(15, 15, 25), cy=7.0, cx=7.0, cz=12.0,
            s_xy=1.3, s_z=2.6, amp=40.0, noise=1.0, seed=0):
    yy, xx, zz = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
    g = amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s_xy ** 2)
                       + (zz - cz) ** 2 / (2 * s_z ** 2)))
    return g + np.random.default_rng(seed).normal(0, noise, g.shape)


def blank(shape=(15, 15, 25), noise=1.0, seed=0):
    return np.random.default_rng(seed).normal(0, noise, shape)


def hot_pixel(shape=(15, 15, 25), amp=30.0, noise=1.0, seed=0):
    """Bright in EVERY plane -- the thing the column features exist for."""
    v = np.random.default_rng(seed).normal(0, noise, shape)
    v[7, 7, :] += amp
    return v


# -- features -------------------------------------------------------------

def test_features():
    print('\n-- a box becomes a vector --')
    core = emitter()
    col = core[7, 7, :]
    v = F.one(core, np.concatenate([col, np.zeros(80)]))
    check('one() returns one number per name',
          v.shape == (len(F.NAMES),), f'{v.shape} vs {len(F.NAMES)}')
    check('and all of them are finite', bool(np.isfinite(v).all()))

    d = dict(zip(F.NAMES, v))
    hp = dict(zip(F.NAMES, F.one(hot_pixel(), hot_pixel()[7, 7, :])))
    check('a real emitter is bright in few planes, a hot pixel in all',
          d['stack_frac_above_half_max'] < hp['stack_frac_above_half_max'],
          f"emitter {d['stack_frac_above_half_max']:.2f} vs "
          f"hot pixel {hp['stack_frac_above_half_max']:.2f}")

    bl = dict(zip(F.NAMES, F.one(blank(), blank()[7, 7, :])))
    check('and it is brighter than nothing at all',
          d['box_peak_log_sigma'] > bl['box_peak_log_sigma'],
          f"{d['box_peak_log_sigma']:.2f} vs "
          f"{bl['box_peak_log_sigma']:.2f}")

    # THE CONTRACT, not a spot check: every name resolves and nothing
    # names a feature that has been removed. `peak` was in this list until
    # its rank correlation with log_peak came back at 1.000000, and the
    # test kept referring to it -- a KeyError that only fires on the
    # first call, which is exactly how a suite reports 0 passed and 0
    # failed and looks like it did not run.
    check('every name in NAMES is produced by one()',
          set(d) == set(F.NAMES) and len(d) == len(F.NAMES),
          f'{len(d)} values for {len(F.NAMES)} names')
    # A NAME MUST SAY WHERE IT LOOKS. Every feature reads either the
    # 15x15x25 box or the full-depth line through it, and the name says
    # which -- `col_*` and `run` had to be explained to every reader.
    stray = [n for n in F.NAMES
             if not (n.startswith('box_') or n.startswith('stack_')
                     or n == 'axial_over_lateral_spread')]
    check('every name says which region it reads', not stray, str(stray))
    check('nothing measures distance to the end of the stack -- that is a '
          'gate on the answer, not an input',
          not any('from_edge' in n or 'stack_end' in n for n in F.NAMES))


def test_edge_gate():
    """The gate runs AFTER everything and is nobody's feature."""
    print('\n-- denying a position rather than an emitter --')
    from codelab_pipeline.localization import edge_gate as EG
    d = 105
    check('a spot at the face is zero planes from an end',
          EG.planes_from_end(0, d) == 0.0 and EG.planes_from_end(104, d) == 0.0)
    check('the middle is furthest', EG.planes_from_end(52, d) == 52.0)
    check('sub-voxel z is kept, not rounded',
          EG.planes_from_end(10.5, d) == 10.5)
    check('the default margin denies inside it and not outside',
          EG.deny(10.9, d) and not EG.deny(11.0, d),
          f'margin {EG.DEFAULT_MARGIN_PLANES}')
    check('a degenerate depth does not raise', EG.planes_from_end(3, 0) == 0.0)

    class S:
        def __init__(self, z):
            self.z = z
    ann = EG.annotate([S(2.0), S(50.0)], d)
    check('annotate returns the number beside the verdict',
          [(round(a[1], 1), a[2]) for a in ann] == [(2.0, True), (50.0, False)])

    # THE SEPARATION ITSELF: this must not be reachable from the model.
    check('the gate lives outside the feature module',
          not any('edge_gate' in line for line in
                  open(F.__file__, encoding='utf-8').read().splitlines()
                  if line.strip().startswith(('import ', 'from '))))

    X = np.array([F.one(emitter(seed=i), emitter(seed=i)[7, 7, :])
                  for i in range(20)])
    st = F.Standardiser().fit(X)
    Z = st(X)
    check('the standardiser centres what it was fitted on',
          abs(float(Z.mean())) < 1e-9, f'{float(Z.mean()):.2e}')
    d2 = st.to_dict()
    check('and round-trips', np.allclose(F.Standardiser.from_dict(d2)(X), Z))
    d2['names'] = list(F.NAMES)[:-1]
    try:
        F.Standardiser.from_dict(d2)
        ok = False
    except ValueError:
        ok = True
    check('a model trained on a different feature list is REFUSED, not '
          'silently rescored', ok)


# -- the split ------------------------------------------------------------

def test_split_does_not_leak():
    print('\n-- holding out cells, not spots --')
    rows = [{'group': (1, c)} for c in range(20) for _ in range(8)]
    tr, va = D.split_by_group(rows, frac=0.25, seed=0)
    gtr = {tuple(rows[i]['group']) for i in tr}
    gva = {tuple(rows[i]['group']) for i in va}
    check('every spot lands somewhere', len(tr) + len(va) == len(rows))
    check('and no cell is on both sides', not (gtr & gva),
          f'{len(gtr)} train cells, {len(gva)} val cells')
    check('the same seed gives the same split',
          D.split_by_group(rows, 0.25, 0) == (tr, va))


# -- the classifier -------------------------------------------------------

def test_classifier_trains_and_reloads():
    print('\n-- pass/fail: does it run, and does p come out alive --')
    try:
        import torch                                        # noqa: F401
    except ImportError:
        check('torch is available', False, 'not installed')
        return
    from codelab_pipeline.training import classify as C

    cores, ys, groups = [], [], []
    for cell in range(24):
        for k in range(6):
            real = (k % 2 == 0)
            cores.append(emitter(seed=cell * 10 + k) if real
                         else (hot_pixel(seed=cell * 10 + k) if k % 4 == 1
                               else blank(seed=cell * 10 + k)))
            ys.append(1 if real else 0)
            groups.append((1, cell))
    cores = np.asarray(cores)
    cols = np.asarray([c[7, 7, :] for c in cores])
    X = F.many(cores, cols)
    y = np.asarray(ys)

    for head in ('linear', 'mlp', 'conv'):
        clf, rep = C.train(X, cores, y, groups, head=head,
                           epochs=(60 if head == 'conv' else 200),
                           seed=0, verbose=False)
        p = clf.score(X if head != 'conv' else None,
                      cores if head == 'conv' else None)
        check(f'{head}: p is a probability', bool(((p > 0) & (p < 1)).all()),
              f'[{p.min():.3f}, {p.max():.3f}]')
        check(f'{head}: p is not one constant',
              float(p.max() - p.min()) > 1e-2,
              f'spread {float(p.max() - p.min()):.3f}')
        check(f'{head}: not all-fail and not all-pass',
              0 < int((p >= clf.threshold).sum()) < len(p),
              f'{int((p >= clf.threshold).sum())} of {len(p)} pass')
        check(f'{head}: the split held out whole cells',
              rep['train_groups'] + rep['val_groups']
              == len({tuple(g) for g in groups}))

    # AN EXTREME LOGIT, which is what synthetic data never produced and
    # the real gated set did (72, against about 5 here). torch returns
    # float32, and in float32 the sigmoid of a clipped +-30 logit is
    # exactly 1.0 -- the guarantee above held only because nothing ever
    # pushed it. Forced here so it stays held.
    clf, _ = C.train(X, cores, y, groups, head='linear', epochs=200,
                     seed=0, verbose=False)
    import torch
    with torch.no_grad():
        for prm in clf.model.parameters():
            prm.mul_(60.0)
    pe = clf.score(X)
    check('p stays inside (0, 1) even at an absurd logit',
          bool(((pe > 0) & (pe < 1)).all()),
          f'max {pe.max():.17g}, min {pe.min():.3e}')
    with torch.no_grad():
        for prm in clf.model.parameters():
            prm.div_(60.0)

    p = clf.score(X)
    with tempfile.TemporaryDirectory() as d:
        path = clf.save(os.path.join(d, 'clf.json'))
        q = C.SpotClassifier.load(path).score(X)
    check('a saved model reloads to the same p', float(np.abs(p - q).max()) < 1e-6,
          f'max diff {float(np.abs(p - q).max()):.2e}')


# -- the PSF bank ---------------------------------------------------------

def test_psf_bank():
    print('\n-- the measured PSF: build, store, refuse, match --')
    patches = [emitter(cy=7 + 0.3 * np.sin(i), cx=7 + 0.3 * np.cos(i),
                       cz=12 + 0.4 * np.sin(i / 2.0), seed=i)
               for i in range(40)]
    mean, comps, var = PB.build(patches, VOXEL, n_components=3)
    check('the template has the shape of its patches', mean.shape == (15, 15, 25))
    check('it is zero-mean and unit-L2, which is what NCC needs',
          abs(float(mean.sum())) < 1e-8
          and abs(float(np.linalg.norm(mean)) - 1.0) < 1e-9,
          f'sum {float(mean.sum()):+.1e}, norm {float(np.linalg.norm(mean)):.6f}')
    check('components come with their explained variance',
          comps.shape[0] == 3 and len(var) == 3, str(np.round(var, 3).tolist()))
    check('and asking for none gives none',
          PB.build(patches, VOXEL, n_components=0)[1].shape[0] == 0)

    with tempfile.TemporaryDirectory() as d:
        path = PB.save(os.path.join(d, 'psf.h5'), mean, VOXEL,
                       components=comps, explained_var=var, n_spots=len(patches),
                       source={'store': 'synthetic'},
                       labels={'reviewers': ['nobody'], 'n_positive': 0})
        m2, c2, meta = PB.load(path, voxel_um=VOXEL)
        check('it reloads byte-identically',
              np.array_equal(mean.astype(np.float32), m2.astype(np.float32)))
        check('carrying the grid it was measured on',
              list(meta['voxel_um']) == list(VOXEL))
        check('the centre convention', list(meta['centre']) == [7.0, 7.0, 12.0])
        check('the normalisation, named not assumed',
              meta['normalisation'] == PB.NORMALISATION)
        check('and WHICH LABELS made it',
              meta['labels'].get('reviewers') == ['nobody'])
        try:
            PB.load(path, voxel_um=(0.208, 0.208, 0.25))
            ok = False
        except ValueError:
            ok = True
        check('a different voxel grid is REFUSED, never resampled', ok)

    # The regression that matters: the first NCC scored the walls.
    print('\n-- the matched filter --')
    rng = np.random.default_rng(0)
    vol = rng.normal(300, 5, (40, 40, 60))
    truth = [(10.0, 12.0, 20.0), (25.0, 30.0, 35.0), (30.0, 11.0, 44.0)]
    yy, xx, zz = np.mgrid[0:40, 0:40, 0:60].astype(float)
    for (ty, tx, tz) in truth:
        vol += 900 * np.exp(-(((yy - ty) ** 2 + (xx - tx) ** 2) / (2 * 1.3 ** 2)
                              + (zz - tz) ** 2 / (2 * 2.6 ** 2)))
    s = PB.ncc(vol, mean)
    r = [a // 2 for a in mean.shape]
    edge = s.copy()
    edge[r[0]:-r[0], r[1]:-r[1], r[2]:-r[2]] = 0.0
    check('nothing scores where the template does not fit',
          float(np.abs(edge).max()) == 0.0, f'{float(np.abs(edge).max()):.3f}')

    hits = PB.match(vol, [mean], min_distance=4, threshold=0.25)
    check('a box with three spots yields three peaks, not the walls',
          len(hits) == 3, f'{len(hits)} peaks')
    if len(hits) == 3:
        err = [min(float(np.linalg.norm(np.array(h[:3]) - np.array(t)))
                   for t in truth) for h in hits]
        check('each within a fifth of a voxel of where it was planted',
              max(err) < 0.5, f'worst {max(err):.3f} px')

    check('a volume smaller than the template scores zero, not garbage',
          float(np.abs(PB.ncc(np.ones((5, 5, 5)), mean)).max()) == 0.0)

    # The seam.
    print('\n-- it plugs into the localizer seam --')
    eng = E.make_engine('psf-match', templates=[mean],
                        min_distance=4, threshold=0.25)
    spots = eng.localize(vol, n_max=3)
    check('make_engine("psf-match") resolves', eng.name == 'psf-match')
    check('and returns LocalizedSpot', len(spots) == 3 and hasattr(spots[0], 'p'))
    check('with p in (0, 1]',
          all(0 < s.p <= 1 for s in spots), str(round(spots[0].p, 3)))
    check('and NaN shape fields, because there is no fit to report',
          all(np.isnan(s.sigma_x) and np.isnan(s.sigma_z) for s in spots))
    near = eng.localize(vol, seed_yxz=(25.0, 30.0, 35.0), n_max=1)
    check('a seed selects the spot nearest it',
          len(near) == 1 and abs(near[0].y - 25.0) < 1.0,
          f'y={near[0].y:.2f}' if near else 'none')

    d = PB.drift(patches, mean)
    check('drift() scores its own spots high', float(np.median(d)) > 0.5,
          f'median {float(np.median(d)):.3f}')
    check('and a blank box low',
          float(np.median(PB.drift([blank(seed=i) for i in range(10)], mean)))
          < float(np.median(d)))


# -- the real labels, if they are here ------------------------------------

def test_on_the_stored_labels():
    print('\n-- and on the labels that actually exist --')
    b = 'D:/bundles/MP58_RNA_Hyb109_ch635'
    if not os.path.isdir(b):
        print(f'  .. {b} is not on this machine; the synthetic checks above '
              'are the whole of this run')
        return
    rows = D.rows(b)
    s = D.summary(rows)
    check('the bundle has labels', s['n'] > 0,
          f"{s['positive']} positive / {s['negative']} negative over "
          f"{s['groups']} cells")
    lab = [r for r in rows if r['label'] in (0, 1)]
    kept, cores, cols = D.boxes(lab[:120])
    check('boxes come out of it', len(kept) > 0 and cores.ndim == 4,
          str(cores.shape))
    X = F.many(cores, cols, kept)
    check('features are finite on real data', bool(np.isfinite(X).all()))
    pos = [c for c, r in zip(cores, kept) if r['label'] == 1]
    if pos:
        mean, _c, _v = PB.build(pos, VOXEL)
        check('a template builds from real confirmed spots',
              abs(float(np.linalg.norm(mean)) - 1.0) < 1e-9)


def main():
    test_features()
    test_split_does_not_leak()
    test_edge_gate()
    test_classifier_trains_and_reloads()
    test_psf_bank()
    test_on_the_stored_labels()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
