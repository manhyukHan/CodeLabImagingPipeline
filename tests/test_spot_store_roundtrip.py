"""
Acceptance test for the unified spot store.

Detects real spots in four (hybe, modality, channel) slices, writes them
through vlinks_store, and reads them back. Two of the four are different
hybes in the SAME modality and channel, one is a different channel in that
modality, and one is a different modality entirely:

    (Hyb_101, RNA, 635)   (Hyb_105, RNA, 635)
    (Hyb_500, RNA, 555)   (Hyb_002, DNA, 555)

That spread is the point. The store this replaces kept assigned spots inside
the /FOV##/cells blob and unassigned ones in an unscoped
/FOV##/unassigned_spots, full-replaced on every write -- so once both
modalities shared one vlinks file, saving DNA erased RNA. These four slices
fail that store and pass the new one, which is what makes this an acceptance
test for the migration rather than a smoke test.

It also pins the properties the save semantics depend on: a write REPLACES
within its slice (so deletions propagate) but CANNOT touch another slice (so
saving one hybe never destroys spots in a hybe the user never opened), and
assigned and unassigned spots live together, differing only in ASpot.cell.

Threshold is 40% OF THE SCOPE MAX -- SpotLocalizationPanel.threshold_abs's
own meaning ("% of scope max"), mirroring _run_spot_auto_detect_body exactly.
Not a 40th percentile of intensity, which would put ~60% of all pixels over
the line and detect nothing meaningful.

Runs entirely inside a scratch project built from copies of the real MIPs, so
it can never write to real data even if it fails partway.

Run: python tests/test_spot_store_roundtrip.py
"""
import os
import shutil
import sys

import h5py
import numpy as np
from skimage.feature import peak_local_max

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.io import vlinks_store as V
from codelab_pipeline.models.spot import ASpot

REAL = 'data/chr19_downstream_new'
SCRATCH = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'spot_store_acceptance')
FOV = 1
THRESHOLD_PCT = 40.0
MIN_DISTANCE = 5

# (hybe, modality, channel, queue-dir name)
SLICES = [
    ('Hyb_101', 'RNA', 635, 'RNA_queue'),
    ('Hyb_105', 'RNA', 635, 'RNA_queue'),
    ('Hyb_500', 'RNA', 555, 'RNA_queue'),
    ('Hyb_002', 'DNA', 555, 'DNA_queue'),
]


def build_scratch():
    """
    A minimal project holding only the four MIPs under test. Copying the
    whole 80MB vlinks would work but is slow per run; more importantly a
    scratch project means a failure part-way cannot leave spots behind in
    real data.
    """
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    for _, modality, _, queue in SLICES:
        os.makedirs(os.path.join(SCRATCH, queue, f'FOV{FOV:02d}'), exist_ok=True)
        stub = os.path.join(SCRATCH, queue, f'FOV{FOV:02d}', 'Hyb_stub_stack.h5')
        with h5py.File(stub, 'w') as f:          # _modality_of reads this attr
            f.attrs['modality'] = modality
    with h5py.File(os.path.join(REAL, 'vlinks.h5'), 'r') as src, \
            h5py.File(os.path.join(SCRATCH, 'vlinks.h5'), 'w') as dst:
        for hybe, modality, channel, _ in SLICES:
            path = f'FOV{FOV:02d}/mip/{modality}/{hybe}/ch{channel}'
            if path not in src:
                raise AssertionError(f'{path} missing from the real vlinks')
            g = dst.require_group(os.path.dirname(path))
            g.create_dataset(os.path.basename(path), data=src[path][()])


def queue_path(queue):
    return os.path.join(SCRATCH, queue)


def detect(hybe, modality, channel, queue):
    """Mirror of MainWindow._run_spot_auto_detect_body's FOV-view branch."""
    mip = V.read_hybe_mip(queue_path(queue), FOV, hybe, channel)
    assert mip is not None, f'no MIP for {modality}/{hybe}/ch{channel}'
    threshold_abs = (THRESHOLD_PCT / 100.0) * float(mip.max())
    coords = peak_local_max(mip, min_distance=MIN_DISTANCE, exclude_border=1,
                            threshold_abs=threshold_abs)
    spots = []
    for y, x in coords:                      # peak_local_max returns (row, col)
        s = ASpot()
        s.set_metadata(fov=FOV, hybe=hybe, channel=int(channel), cell=-1,
                       raw_coordinate=(float(x), float(y), 0.0),
                       coordinate=(float(x), float(y), 0.0),
                       brightness=float(mip[y, x]))
        spots.append(s)
    return spots


_detected = {}


def populate():
    """
    Fresh scratch project with all four slices detected and written.

    Every test calls this rather than depending on a previous test having
    run: order-dependent tests silently pass or fail on execution order
    rather than on behaviour, which is how the first version of this file
    reported two failures that were entirely its own doing.
    """
    build_scratch()
    _detected.clear()
    for hybe, modality, channel, queue in SLICES:
        spots = detect(hybe, modality, channel, queue)
        _detected[(modality, hybe, channel)] = spots
        V.write_spots(queue_path(queue), FOV, modality, hybe, channel, spots)


def test_detection_finds_spots_in_every_slice():
    """Non-vacuity: an empty slice would make every later assertion trivial."""
    populate()
    for key, spots in _detected.items():
        assert len(spots) > 0, f'no spots detected in {key}'


def test_every_slice_round_trips():
    populate()
    for hybe, modality, channel, queue in SLICES:
        spots = _detected[(modality, hybe, channel)]
        back = V.read_spots(queue_path(queue), FOV, modality, hybe, channel)
        assert len(back) == len(spots), f'{modality}/{hybe}: wrote {len(spots)}, read {len(back)}'
        wrote = {(round(s.raw_coordinate[0], 2), round(s.raw_coordinate[1], 2)) for s in spots}
        read = {(d['raw_coordinate'][0], d['raw_coordinate'][1]) for d in back}
        assert wrote == read, f'{modality}/{hybe}: coordinates did not survive the round trip'


def test_all_four_slices_coexist():
    """
    THE regression this store exists to prevent. Under the old unscoped
    full-replace store, writing the DNA slice erased the RNA ones.
    """
    populate()
    total = 0
    for hybe, modality, channel, queue in SLICES:
        got = V.read_spots(queue_path(queue), FOV, modality, hybe, channel)
        assert len(got) == len(_detected[(modality, hybe, channel)]), \
            f'{modality}/{hybe}/ch{channel} lost spots to another slice write'
        total += len(got)
    every = V.read_spots(queue_path('RNA_queue'), FOV)
    assert len(every) == total, f'FOV-wide read got {len(every)}, expected {total}'


def test_uids_are_unique_across_the_whole_fov():
    populate()
    every = V.read_spots(queue_path('RNA_queue'), FOV)
    uids = [d['uid'] for d in every]
    assert 0 not in uids, 'a spot reached storage without an allocated uid'
    assert len(set(uids)) == len(uids), 'uid collision -- uid no longer identifies one spot'


def test_writing_one_slice_leaves_the_others_untouched():
    populate()
    target = ('Hyb_101', 'RNA', 635, 'RNA_queue')
    others = [s for s in SLICES if s != target]
    before = {(m, h, c): len(V.read_spots(queue_path(q), FOV, m, h, c)) for h, m, c, q in others}
    hybe, modality, channel, queue = target
    V.write_spots(queue_path(queue), FOV, modality, hybe, channel,
                  _detected[(modality, hybe, channel)][:1])
    assert len(V.read_spots(queue_path(queue), FOV, modality, hybe, channel)) == 1
    after = {(m, h, c): len(V.read_spots(queue_path(q), FOV, m, h, c)) for h, m, c, q in others}
    assert before == after, f'a scoped write leaked into other slices: {before} -> {after}'


def test_deletions_propagate_within_a_slice():
    populate()
    hybe, modality, channel, queue = 'Hyb_500', 'RNA', 555, 'RNA_queue'
    V.write_spots(queue_path(queue), FOV, modality, hybe, channel, [])
    assert V.read_spots(queue_path(queue), FOV, modality, hybe, channel) == []
    assert len(V.read_spots(queue_path('DNA_queue'), FOV, 'DNA', 'Hyb_002', 555)) > 0, \
        'clearing one slice cleared another'


def test_assigned_and_unassigned_share_a_slice():
    """They differ only in ASpot.cell -- no separate store, no separate flag."""
    populate()
    hybe, modality, channel, queue = 'Hyb_105', 'RNA', 635, 'RNA_queue'
    spots = _detected[(modality, hybe, channel)][:6]
    for i, s in enumerate(spots):
        s.cell = 7 if i % 2 else -1
    V.write_spots(queue_path(queue), FOV, modality, hybe, channel, spots)
    back = V.read_spots(queue_path(queue), FOV, modality, hybe, channel)
    assigned = [d for d in back if d['cell'] != -1]
    unassigned = [d for d in back if d['cell'] == -1]
    assert len(back) == len(spots)
    assert assigned and unassigned, 'expected a mix of assigned and unassigned in one slice'


def test_real_data_untouched():
    """The whole run must have stayed inside the scratch project."""
    with h5py.File(os.path.join(REAL, 'vlinks.h5'), 'r') as f:
        for fov in (1, 2):
            assert f'FOV{fov:02d}/spots' not in f, \
                f'the test wrote spots into REAL data at FOV{fov:02d}/spots'


def test_harness_detects_the_old_unscoped_store():
    """
    Self-check. Replaces write_spots with the OLD store's behaviour -- one
    unscoped blob per FOV, full-replaced on every write, exactly what
    write_fov_spots did -- and asserts the coexistence test catches it.

    Without this the suite could pass while asserting nothing about scoping.
    A FrameResolver test earlier in this project printed four PASS lines
    while every lookup silently missed and defaulted to identity; the cost of
    proving a test can fail is far lower than the cost of trusting one that
    cannot.
    """
    real_write = V.write_spots
    unscoped = {}

    def old_style(storage_path, fov, modality, hybe, channel, spots):
        # ignores (modality, hybe, channel) -- one pool per FOV, replaced whole
        unscoped[fov] = list(spots)
        real_write(storage_path, fov, modality, hybe, channel, spots)
        for other_h, other_m, other_c, other_q in SLICES:
            if (other_m, other_h, other_c) != (modality, hybe, channel):
                real_write(queue_path(other_q), fov, other_m, other_h, other_c, [])

    V.write_spots = old_style
    try:
        try:
            populate()
            test_all_four_slices_coexist()
        except AssertionError:
            return                      # caught it, as required
        raise AssertionError(
            'the old unscoped full-replace store passed the coexistence test '
            '-- this suite would not catch the regression it exists for')
    finally:
        V.write_spots = real_write


def _run_all():
    by_name = {k: v for k, v in globals().items() if k.startswith('test_')}
    order = sorted(by_name)   # each test calls populate(), so order cannot matter
    failures = []
    for name in order:
        try:
            by_name[name]()
            print(f'  PASS  {name}')
        except AssertionError as e:
            failures.append(name); print(f'  FAIL  {name}: {e}')
        except Exception as e:
            failures.append(name); print(f'  ERROR {name}: {e!r}')
    counts = {f'{m}/{h}/ch{c}': len(v) for (m, h, c), v in _detected.items()}
    print(f'\ndetected at {THRESHOLD_PCT:.0f}% of scope max, min_distance={MIN_DISTANCE}:')
    for k, n in counts.items():
        print(f'    {k:24s} {n:6d} spots')
    print(f'\n{len(order) - len(failures)}/{len(order)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_run_all())
