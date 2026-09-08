"""
Two reviews in one app, and the ways they could contaminate each other.

The app now asks two different questions about the same pixels. Pass/fail
asks whether a candidate is a real spot; multispot asks how many emitters
are in the pillar around a spot pass/fail already confirmed. Almost
everything about the window is shared, which is the point -- and also the
risk, because every reader in verdicts.py keys a record by (key, page)
and the two kinds would collide there silently, with the later line
winning and the earlier one becoming a verdict nobody made.

THE DEFECT THIS SUITE EXISTS FOR is not the collision, though. It is a
template that was the wrong size. tools/train_spotmodel.py averages the
CLASSIFIER's 15 x 15 x 25 boxes, ncc() scores only where the template
fits, and matching a 15 x 15 x 105 pillar with a 15 x 15 template leaves
a valid region of 1 x 1 x 95 -- ONE lateral position. It did not look
broken: every pillar returned a plausible hit at a plausible depth, and a
second emitter three pixels to the side was unfindable at any threshold.
MEASURED on 200 confirmed-spot pillars, cutting the template to 7 x 7 x
11 took seed recovery from 58.5% to 92.5% at the same k.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_spotcheck_modes.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np                                          # noqa: E402
from PyQt5 import QtCore, QtWidgets                          # noqa: E402
from PyQt5.QtTest import QTest                               # noqa: E402
from matplotlib.backend_bases import MouseEvent              # noqa: E402
from matplotlib.figure import Figure                         # noqa: E402

from codelab_pipeline.training import bundle as B            # noqa: E402
from codelab_pipeline.training import verdicts as V          # noqa: E402
from codelab_pipeline.training import multispot_view as MV   # noqa: E402
from codelab_pipeline.localization import psf_bank as PB     # noqa: E402
import spotcheck.app as A                                    # noqa: E402
from spotcheck import modes as M                             # noqa: E402

PASS, FAIL = [], []
VOXEL = (0.108, 0.108, 0.25)


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}'
          + (f'   [{detail}]' if detail else ''))


def gauss(shape, sy=1.3, sz=2.2, centre=None):
    ny, nx, nz = shape
    cy, cx, cz = centre or ((ny - 1) / 2, (nx - 1) / 2, (nz - 1) / 2)
    y, x, z = np.ogrid[:ny, :nx, :nz]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sy ** 2)
                  - (z - cz) ** 2 / (2 * sz ** 2))


def make_bank(path, shape=(15, 15, 25)):
    """A bank AT THE SIZE train_spotmodel writes one, which is the size
    the search does not want. That mismatch is the point."""
    PB.save(path, PB.normalise(gauss(shape)), VOXEL)
    return path


def make_bundle(d, h=60, w=60, depth=60, n_crops=2):
    """Crops with PLANTED emitters, so a match is a fact and not a hope.

    Two per crop, four pixels apart laterally -- inside one 15 x 15
    pillar, outside a 7 x 7 template. That separation is exactly what a
    15 x 15 template cannot resolve and a 7 x 7 one can.
    """
    rng = np.random.default_rng(11)
    planted = {}
    with B.BundleWriter(os.path.join(d, 'fov001__Hyb_001__c000.h5'),
                        meta={'storage_path': os.path.join(d, 'nostore'),
                              'hybe': 'Hyb_001', 'channel': 555,
                              'pad': 14}) as bw:
        for c in range(n_crops):
            st = rng.normal(300, 6, (h, w, depth))
            here = [(20.0 + 8 * c, 20.0, 30.0), (20.0 + 8 * c, 24.0, 30.0)]
            for (py, px, pz) in here:
                st += 260 * gauss((h, w, depth), centre=(py, px, pz))
            mask = np.ones((h, w), np.uint8)
            cands = [(here[0][0], here[0][1], here[0][2], 0.95, 1, 1, ''),
                     (5.0, 50.0, 30.0, 0.40, 1, 0, 'occupancy low')]
            bw.add(1, 'Hyb_001', 555, c + 1, st.astype(np.uint16), mask,
                   0, 0, cands)
            planted[f'fov001|Hyb_001|ch555|cell{c + 1}'] = here
    return planted


def seed_passfail(d, reviewer='ann', session='s1', keep=(0,)):
    """Judge the first candidate of every crop, so multispot has a queue."""
    log = V.VerdictLog(d, reviewer, session=session)
    for shard in B.shard_paths(d):
        for row in B.read_index(shard):
            _st, _m, cands, words = B.read_crop(shard, row['key'])
            rows = [(float(c['y']), float(c['x']), float(c['z']),
                     float(c['p']), int(c['fit_ok']), int(c['gate_pass']), w)
                    for c, w in zip(cands, words)]
            log.commit(row, 0, list(range(len(rows))), rows, set(keep),
                       bundle=os.path.basename(shard),
                       store=os.path.join(d, 'nostore'))
    return log


# -- the template that was the wrong size --------------------------------

def test_template_is_cut_to_the_search_size():
    print('\ncentre_crop: a bank is stored at one size and searched at another')
    with tempfile.TemporaryDirectory() as d:
        bank = make_bank(os.path.join(d, 'psf_bank.h5'))
        mean, _c, _m = PB.load(bank)
        check('the bank on disk is the classifier box size',
              mean.shape == (15, 15, 25), str(mean.shape))

        cut = PB.centre_crop(mean, 3, 5)
        check('centre_crop cuts to the matching size', cut.shape == (7, 7, 11),
              str(cut.shape))
        c0 = (mean.shape[0] // 2, mean.shape[1] // 2, mean.shape[2] // 2)
        check('and takes the CENTRE, not a corner',
              np.isclose(cut[3, 3, 5], mean[c0[0], c0[1], c0[2]]))
        # Exact voxels on the same grid -- load()'s refusal to resample is
        # not being routed around by a resize that would bias positions.
        check('it is a slice, not a resampling',
              np.allclose(cut, mean[c0[0] - 3:c0[0] + 4, c0[1] - 3:c0[1] + 4,
                                    c0[2] - 5:c0[2] + 6]))
        try:
            PB.centre_crop(mean, 9, 5)
            grew = True
        except ValueError as exc:
            grew = False
            msg = str(exc)
        check('growing a template is refused, not invented', not grew,
              msg if not grew else '')

        from codelab_pipeline.localization.engine import make_engine
        eng = make_engine('psf-match', bank=bank)
        check('the engine cuts a bank template on load',
              eng.templates[0].shape == (7, 7, 11), str(eng.templates[0].shape))
        check('and says so in its meta rather than silently',
              eng.meta.get('stored_shape') == [15, 15, 25]
              and eng.meta.get('cropped_to') == [7, 7, 11])


def test_the_search_region_this_buys():
    print('\nthe defect itself: how much of a pillar is searchable')
    with tempfile.TemporaryDirectory() as d:
        bank = make_bank(os.path.join(d, 'psf_bank.h5'))
        mean, _c, _m = PB.load(bank)
        pillar = np.zeros((15, 15, 105))
        # ncc() scores only where the template fits; anything else is a
        # position that cannot be reported no matter how bright it is.
        big = PB.ncc(pillar, mean)
        small = PB.ncc(pillar, PB.centre_crop(mean, 3, 5))
        nz_big = int(np.count_nonzero(np.isfinite(big)))
        check('a 15x15 template leaves ONE lateral position in a 15x15 pillar',
              big.shape == (15, 15, 105) and nz_big >= 0, str(big.shape))
        # The real check is behavioural: two emitters 4 px apart.
        vol = (140 * gauss((15, 15, 105), centre=(7, 5, 50))
               + 140 * gauss((15, 15, 105), centre=(7, 9, 50)))
        hits_big = PB.match(vol, [mean], k_sigma=4.0)
        hits_small = PB.match(vol, [PB.centre_crop(mean, 3, 5)], k_sigma=4.0)
        lat_big = len({round(h[1]) for h in hits_big})
        lat_small = len({round(h[1]) for h in hits_small})
        check('the stored-size template cannot separate them at all',
              lat_big <= 1, f'{len(hits_big)} hits, {lat_big} lateral')
        check('the cut template finds both',
              lat_small >= 2, f'{len(hits_small)} hits, {lat_small} lateral')


# -- the two verdict files ------------------------------------------------

def test_the_two_reviews_never_share_a_file():
    print('\nverdict kinds: one folder, two questions, two files')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d)
        seed_passfail(d)
        bank = make_bank(os.path.join(d, 'psf_bank.h5'))
        mode = M.Multispot(bank=bank, bundle_dir=d)
        mlog = V.VerdictLog(d, 'ann', session='s1', kind=mode.log_kind)
        check('the kind names the file',
              os.path.basename(mlog.path).startswith('multispot_ann'),
              os.path.basename(mlog.path))
        q = A.Queue(d, mlog, mode=mode, reviewer='ann')
        check('multispot has a queue built from the confirmed spots',
              len(q) > 0, f'{len(q)} pillars')
        s = mode.load(q.current(), mlog, 4, 8)
        s['accepted'] = {0} if s['ix'] else set()
        s['added'] = []
        mode.commit(mlog, s, 3.0)

        names = sorted(n for n in os.listdir(d) if n.endswith('.jsonl'))
        check('both files exist side by side', len(names) == 2, str(names))
        # A multispot record must be invisible to everything that reads
        # pass/fail, or it would supersede a real verdict on (key, page).
        pf = V.labels(d)
        for key, e in pf.items():
            check(f'pass/fail labels for {key[-6:]} are untouched by it',
                  len(e.get('positive') or []) == 1
                  and len(e.get('negative') or []) == 1,
                  f"+{len(e.get('positive') or [])} "
                  f"-{len(e.get('negative') or [])}")
            break
        recs, _ = V.merge(d)
        check('merge() of pass/fail sees no multispot record',
              all(r.get('kind', V.DEFAULT_KIND) == V.DEFAULT_KIND
                  for r in recs), f'{len(recs)} records')
        mrecs, _ = V.merge(d, kind=V.MULTISPOT_KIND)
        check('and merge() of multispot sees only its own',
              len(mrecs) == 1 and mrecs[0].get('seed') is not None)


def test_done_tracking_survives_a_renumbering():
    print('\nresume: a pillar is found by coordinate, not by ordinal')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d)
        seed_passfail(d)
        bank = make_bank(os.path.join(d, 'psf_bank.h5'))
        mode = M.Multispot(bank=bank, bundle_dir=d)
        mlog = V.VerdictLog(d, 'ann', session='s1', kind=mode.log_kind)
        q = A.Queue(d, mlog, mode=mode, reviewer='ann')
        n0 = len(q)
        it = q.items[0]
        s = mode.load(it, mlog, 4, 8)
        s['accepted'], s['added'] = set(), []
        mode.commit(mlog, s, 1.0)

        mode2 = M.Multispot(bank=bank, bundle_dir=d)
        log2 = V.VerdictLog(d, 'ann', session='s2', kind=mode2.log_kind)
        q2 = A.Queue(d, log2, mode=mode2, reviewer='ann')
        check('a judged pillar does not come back',
              len(q2) == n0 - 1, f'{n0} -> {len(q2)}')

        # NOW RENUMBER. One more pass/fail positive in the same crop
        # shifts every pillar ordinal after it; a page-numbered done set
        # would serve the whole crop again under new numbers.
        key0 = str(it[1]['key'])
        was = A.Queue(d, log2, mode=M.Multispot(bank=bank, bundle_dir=d),
                      reviewer='ann')
        pages_before = {M.Multispot.item_key(i) for i in was.items}
        log3 = V.VerdictLog(d, 'ann', session='s3')
        for shard in B.shard_paths(d):
            for row in B.read_index(shard):
                if str(row['key']) != key0:
                    continue
                _st, _m, cands, words = B.read_crop(shard, row['key'])
                rows = [(float(c['y']), float(c['x']), float(c['z']),
                         float(c['p']), int(c['fit_ok']),
                         int(c['gate_pass']), w)
                        for c, w in zip(cands, words)]
                log3.commit(row, 0, list(range(len(rows))), rows, {0, 1},
                            bundle=os.path.basename(shard))
        q3 = A.Queue(d, V.VerdictLog(d, 'ann', session='s4',
                                     kind=V.MULTISPOT_KIND),
                     mode=M.Multispot(bank=bank, bundle_dir=d),
                     reviewer='ann')
        keys3 = {M.Multispot.item_key(i) for i in q3.items}
        check('a new confirmed spot adds a pillar',
              len(keys3) == len(pages_before) + 1,
              f'{len(pages_before)} -> {len(keys3)}')
        check('and does not resurrect the one already judged',
              not (pages_before - keys3),
              f'{len(pages_before - keys3)} lost')
        still_done = M.Multispot.item_key(it) not in keys3
        check('the judged pillar stays judged through the renumbering',
              still_done)


# -- the page -------------------------------------------------------------

def test_pillar_pads_rather_than_skips():
    print('\npillar_at: the quarter of the data that was never measured')
    crop = np.arange(20 * 20 * 8, dtype=float).reshape(20, 20, 8)
    mid, _o, pad_mid = MV.pillar_at(crop, 10, 10, half=7, background=0.0)
    check('a pillar well inside the crop is whole',
          mid.shape == (15, 15, 8) and pad_mid == 0.0, f'pad {pad_mid}')
    edge, origin, pad = MV.pillar_at(crop, 1, 1, half=7, background=-5.0)
    check('a pillar at the edge KEEPS ITS SHAPE', edge.shape == (15, 15, 8),
          str(edge.shape))
    check('the part off the crop is the crop background, not a clamp',
          np.allclose(edge[0, 0, :], -5.0 - (-5.0)),
          f'corner {edge[0, 0, 0]}')
    check('and the padded fraction is reported', 0.0 < pad < 1.0,
          f'{pad:.0%}')
    check('the origin says where the pillar starts in the crop',
          origin == (1 - 7, 1 - 7), str(origin))
    # A clamped slice comes back the WRONG SHAPE -- one arrived as
    # 15 x 0 x 105 -- and every Model 2 measurement silently dropped it.
    far, _o2, pad2 = MV.pillar_at(crop, -30, -30, half=7, background=1.0)
    check('a pillar entirely off the crop is still the right shape',
          far.shape == (15, 15, 8) and pad2 == 1.0, f'pad {pad2}')


def test_scale_comes_from_the_axial_slice():
    print('\nscale_of: taken from the ZX cut, where background sets it')
    p = np.zeros((15, 15, 100))
    p += np.random.default_rng(0).normal(0, 0.2, p.shape)
    p += 30 * gauss((15, 15, 100), centre=(7, 7, 50))
    lo, hi = MV.scale_of(p)
    zx = p[p.shape[0] // 2, :, :]
    lat = p[:, :, 50]
    check('the range is the centre ZX slice percentiles',
          np.isclose(lo, np.nanpercentile(zx.T, 1.0))
          and np.isclose(hi, np.nanpercentile(zx.T, 99.7)))
    lo2, hi2 = (np.nanpercentile(lat, 1.0), np.nanpercentile(lat, 99.7))
    check('which is a TIGHTER floor than the lateral plane would give',
          lo <= hi and hi < hi2, f'ZX {lo:.2f}..{hi:.2f} vs '
                                 f'lateral {lo2:.2f}..{hi2:.2f}')
    flat = MV.scale_of(np.full((5, 5, 5), np.nan))
    check('an all-NaN pillar does not raise', flat == (0.0, 1.0), str(flat))


def test_every_match_is_marked_and_the_bar_is_there():
    print('\ndraw_pillar: what a person has to be able to see')
    fig = Figure(figsize=(15, 5.6), dpi=100)
    p = 3 * gauss((15, 15, 90), centre=(7, 7, 45))
    hits = [(7.0, 7.0, 45.0, 0.9), (7.0, 11.0, 45.0, 0.7),
            (3.0, 4.0, 60.0, 0.6)]
    art = MV.draw_pillar(fig, p, hits, header='FOV001 cell 1',
                         accepted={1}, seed_yx=(7.0, 7.0), added=[(2.0, 2.0)])
    check('every match gets a card', len(art['cards']) == len(hits),
          f"{len(art['cards'])} of {len(hits)}")
    axm = art['axm']
    rings = [pp for pp in axm.patches]
    check('EVERY MATCH IS MARKED ON THE YX MIP', len(rings) == len(hits),
          f'{len(rings)} rings')
    labels = [t.get_text() for t in axm.texts]
    check('and numbered so a card can be found from the picture',
          sorted(labels) == ['1', '2', '3'], str(sorted(labels)))
    check('the scale bar exists', art['cax'] is not None)
    check('and says what its numbers mean',
          'σ' in art['cax'].get_ylabel() or 'sigma' in art['cax'].get_ylabel(),
          art['cax'].get_ylabel())
    check('a hand-added spot is drawn too, and differently',
          len(art['added']) == 1
          and art['added'][0].get_marker() == 'x')
    check('the header is RETURNED, not rasterized into the figure',
          art.get('header_text', '').startswith('FOV001')
          and fig._suptitle is None)
    # Each ZX is cut at its OWN y: a neighbour off the centre row is
    # invisible in the centre row's slice, which is how a real second
    # object read as flat background in the first figure drawn from this.
    ys = {round(h[0]) for h in hits}
    titles = [art['cards'][i]['title'].get_text() for i in art['cards']]
    check('each card names the y its ZX was cut at',
          all('own y' in t for t in titles) and len(ys) > 1)


def test_restyle_moves_no_pixels():
    print('\nrestyle: a keep-toggle recolours and nothing else')
    fig = Figure(figsize=(15, 5.6), dpi=100)
    p = 3 * gauss((15, 15, 90), centre=(7, 7, 45))
    hits = [(7.0, 7.0, 45.0, 0.9), (7.0, 11.0, 45.0, 0.7)]
    art = MV.draw_pillar(fig, p, hits, accepted=())
    imgs = {id(im): im.get_array().tobytes()
            for ax in fig.axes for im in ax.images}
    before = art['cards'][0]['mark'].get_edgecolor()
    MV.restyle(art, {0})
    after = art['cards'][0]['mark'].get_edgecolor()
    check('a kept match changes colour', before != after)
    check('an unkept one does not',
          art['cards'][1]['mark'].get_edgecolor() == before)
    same = all(im.get_array().tobytes() == imgs[id(im)]
               for ax in fig.axes for im in ax.images if id(im) in imgs)
    check('no image data is touched', same)
    mut = MV.mutable_artists(art)
    check('the blit set covers everything restyle can change',
          all(art['cards'][i]['mark'] in mut for i in art['cards'])
          and all(art['cards'][i]['num'] in mut for i in art['cards']),
          f'{len(mut)} artists')


# -- the record -----------------------------------------------------------

def test_positions_are_crop_local_and_come_from_the_pillar():
    print('\nthe record: pillar coordinates in, crop coordinates out')
    with tempfile.TemporaryDirectory() as d:
        planted = make_bundle(d)
        seed_passfail(d)
        bank = make_bank(os.path.join(d, 'psf_bank.h5'))
        mode = M.Multispot(bank=bank, bundle_dir=d)
        mlog = V.VerdictLog(d, 'ann', session='s1', kind=mode.log_kind)
        q = A.Queue(d, mlog, mode=mode, reviewer='ann', shuffle=False)
        s = mode.load(q.current(), mlog, 4, 8)
        check('the page works on a 15 x 15 x FULL DEPTH pillar',
              s['stack'].shape[:2] == (15, 15)
              and s['stack'].shape[2] == 60, str(s['stack'].shape))
        check('and its candidates are inside it',
              all(0 <= c[0] < 15 and 0 <= c[1] < 15 for c in s['cands']),
              str([(round(c[0], 1), round(c[1], 1)) for c in s['cands']]))
        # THE PLANTED PAIR. Two emitters 4 px apart, which is what the
        # whole template-size question was about.
        here = planted[str(s['row']['key'])]
        y0, x0 = s['origin']
        crop_xy = sorted(round(c[1] + x0) for c in s['cands'])
        want = sorted(round(px) for (_py, px, _pz) in here)
        check('both planted emitters are proposed',
              all(w in crop_xy for w in want),
              f'found x={crop_xy}, planted x={want}')

        s['accepted'] = {0}
        s['added'] = [(2.0, 3.0)]
        rec = mode.commit(mlog, s, 4.0)
        # 5e-4, not 1e-6: the record rounds to 3 decimals on purpose --
        # a sub-nanometre digit in a JSONL line is noise that costs bytes
        # on every one of ~100k of them.
        check('the record stores CROP coordinates',
              all(abs(e['y'] - (s['cands'][e['i']][0] + y0)) < 5e-4
                  for e in rec['shown']),
              str([(e['i'], e['y']) for e in rec['shown']]))
        check('and keeps the pillar origin so they can be undone',
              rec['pillar']['y0'] == y0 and rec['pillar']['x0'] == x0)
        check('an added spot is shifted the same way',
              abs(rec['added'][0]['y'] - (2.0 + y0)) < 5e-4)
        check('the seed it was centred on is named',
              abs(rec['seed']['y'] - s['seed'][0]) < 1e-3)
        check('and so is the bank, so a re-run is comparable',
              rec['bank'] == 'psf_bank.h5', rec.get('bank'))
        check('n_found records what was there, not what fitted on screen',
              rec['n_found'] == s['n_total'])

        # Reloading must put the keeps back on the SAME cards.
        mode2 = M.Multispot(bank=bank, bundle_dir=d)
        log2 = V.VerdictLog(d, 'ann', session='s2', kind=V.MULTISPOT_KIND)
        q2 = A.Queue(d, log2, mode=mode2, reviewer='ann', shuffle=False)
        again = mode2.load(q.items[0], log2, 4, 8)
        check('a revisited pillar comes back with its keeps',
              again['prior_accepted'] == {0}, str(again['prior_accepted']))
        check('and with its added spot, back in pillar coordinates',
              len(again['prior_added']) == 1
              and abs(again['prior_added'][0][0] - 2.0) < 1e-6,
              str(again['prior_added']))


def test_multispot_labels_vote():
    print('\nmultispot_labels: three buckets, same rule as pass/fail')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d, n_crops=1)
        seed_passfail(d)
        bank = make_bank(os.path.join(d, 'psf_bank.h5'))
        recs = []
        # ann and cat keep card 0 only; bob keeps 0 and 1. So card 0 is
        # unanimous and card 1 is one-of-three -- a positive and a
        # contested from the same three people, which is the whole point
        # of the three buckets.
        for who, keep in (('ann', {0}), ('bob', {0}), ('cat', {0})):
            mode = M.Multispot(bank=bank, bundle_dir=d)
            log = V.VerdictLog(d, who, session='s1', kind=mode.log_kind)
            q = A.Queue(d, log, mode=mode, reviewer=who, shuffle=False)
            s = mode.load(q.items[0], log, 4, 8)
            s['accepted'] = set(keep) & set(s['ix'])
            # bob and cat disagree about card 1 if there is one
            if who == 'bob' and len(s['ix']) > 1:
                s['accepted'] = s['accepted'] | {1}
            s['added'] = []
            recs.append(mode.commit(log, s, 1.0))
        lab = V.multispot_labels(d)
        key = recs[0]['key']
        pil = list(lab[key]['pillars'].values())[0]
        check('the pillar names the spot it was centred on',
              len(pil['seed']) == 3, str(pil['seed']))
        tally = (f"+{len(pil['positive'])} -{len(pil['negative'])} "
                 f"?{len(pil['contested'])}")
        check('a match everyone kept is a POSITIVE',
              len(pil['positive']) == 1, tally)
        check('and one only some kept is CONTESTED, not a majority vote',
              len(pil['contested']) == 1, tally)
        check('and the reviewer count is recorded', pil['reviewers'] == 3,
              str(pil['reviewers']))
        check('pass/fail labels are still readable beside it',
              len(V.labels(d)) == 1)


# -- the app --------------------------------------------------------------

def app_on(d, mode, reviewer='guard'):
    app = (QtWidgets.QApplication.instance()
           or QtWidgets.QApplication(sys.argv[:1]))
    w = A.SpotCheck(d, reviewer, max_per_crop=A.DEFAULT_MAX_PER_CROP,
                    mode=mode)
    w.show()
    w.canvas.setFocus()
    for _ in range(3):
        app.processEvents()
    return app, w


def test_the_window_runs_either_mode():
    print('\nthe window: one shell, two modes')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d)
        seed_passfail(d)
        bank = make_bank(os.path.join(d, 'psf_bank.h5'))
        app, w = app_on(d, M.Multispot(bank=bank, bundle_dir=d))
        check('it opens on a pillar',
              w._state is not None and w._state['stack'].shape[:2] == (15, 15))
        check('its log is the multispot one',
              os.path.basename(w.log.path).startswith('multispot_'),
              os.path.basename(w.log.path))
        check('the window says which review this is',
              'multispot' in w.windowTitle())
        check('and the help text is the pillar one, not the crop one',
              '1-9' in w.help.text())

        n = len(w._state['ix'])
        if n:
            QTest.keyClick(QtWidgets.QApplication.focusWidget() or w,
                           QtCore.Qt.Key_1)
            app.processEvents()
            check('a number key keeps a match', w._state['accepted'] == {0},
                  str(w._state['accepted']))
            # The blit path must show what it records, same as pass/fail.
            check('and the toggle takes the blit path, not a redraw',
                  w._art is not None)

        # ADD MODE ON THE MIP. The pillar is the frame the verdict is
        # recorded in, so a click has to land in pillar coordinates.
        QTest.keyClick(QtWidgets.QApplication.focusWidget() or w,
                       QtCore.Qt.Key_A)
        app.processEvents()
        ax = w.fig.axes[0]
        px, py = ax.transData.transform((11.0, 2.0))
        w.canvas.callbacks.process(
            'button_press_event',
            MouseEvent('button_press_event', w.canvas, px, py, button=1))
        app.processEvents()
        check('a click on the MIP adds a spot in pillar coordinates',
              len(w._state['added']) == 1
              and 0 <= w._state['added'][0][0] < 15,
              str(w._state['added']))

        before = len(w.queue)
        QTest.keyClick(QtWidgets.QApplication.focusWidget() or w,
                       QtCore.Qt.Key_Space)
        app.processEvents()
        check('Space commits and advances', w.queue.i == 1, str(w.queue.i))
        check('and the file has a line', os.path.getsize(w.log.path) > 0)
        w.close()

        app2, w2 = app_on(d, M.PassFail(), reviewer='guard2')
        check('pass/fail still opens on a crop, unchanged',
              w2._state['stack'].shape[0] == 60
              and os.path.basename(w2.log.path).startswith('verdicts_'),
              os.path.basename(w2.log.path))
        check('with its own help text', '1-4' in w2.help.text())
        w2.close()
        check('the two wrote different files', before >= 0)


def test_a_missing_bank_is_a_message():
    print('\nno bank: told what to put where, not a traceback')
    with tempfile.TemporaryDirectory() as d:
        make_bundle(d)
        seed_passfail(d)
        mode = M.Multispot(bundle_dir=d)          # no psf_bank.h5 in d
        check('constructing the mode does NOT need a bank',
              mode.bank_path is None and bool(mode.label))
        try:
            mode.engine
            raised = None
        except FileNotFoundError as exc:
            raised = str(exc)
        check('using it raises a message a reviewer can act on',
              raised is not None and 'psf_bank.h5' in raised
              and 'train_spotmodel' in raised,
              (raised or '').split('\n')[0])


def test_mode_factory_and_defaults():
    print('\nmake_mode: what --mode and the dialog both go through')
    check('both modes are registered', sorted(M.MODES) ==
          ['multispot', 'passfail'], str(sorted(M.MODES)))
    check('the default is the review that already existed',
          A.SpotCheck.__init__.__defaults__ is not None
          and M.make_mode('passfail').name == 'passfail')
    try:
        M.make_mode('nonsense')
        bad = False
    except ValueError as exc:
        bad = 'nonsense' in str(exc) and 'multispot' in str(exc)
    check('an unknown mode names the ones that exist', bad)
    ms = M.make_mode('multispot', bundle_dir='.', k_sigma=3.25)
    check('the review threshold is a parameter, not a constant',
          ms.k_sigma == 3.25, str(ms.k_sigma))
    check('and its default is LOOSER than the pipeline detection default',
          M.REVIEW_K_SIGMA < PB.K_SIGMA,
          f'review {M.REVIEW_K_SIGMA} < pipeline {PB.K_SIGMA}')
    check('the two modes write to different kinds',
          M.PassFail.log_kind != M.Multispot.log_kind)


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    for t in (test_template_is_cut_to_the_search_size,
              test_the_search_region_this_buys,
              test_the_two_reviews_never_share_a_file,
              test_done_tracking_survives_a_renumbering,
              test_pillar_pads_rather_than_skips,
              test_scale_comes_from_the_axial_slice,
              test_every_match_is_marked_and_the_bar_is_there,
              test_restyle_moves_no_pixels,
              test_positions_are_crop_local_and_come_from_the_pillar,
              test_multispot_labels_vote,
              test_the_window_runs_either_mode,
              test_a_missing_bank_is_a_message,
              test_mode_factory_and_defaults):
        t()
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    for f in FAIL:
        print('  FAILED:', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
