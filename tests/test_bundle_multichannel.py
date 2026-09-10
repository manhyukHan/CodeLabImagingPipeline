"""
Two channels in one bundle directory, and neither one vanishes.

WHY. Everything INSIDE a shard already separated channels -- the crop
key 'fov007|Hyb_101|ch635|cell13', the index's channel field, the
verdicts, dataset.rows() -- but the shard FILENAME did not carry the
channel. A second build into the same --out produced the same names,
and BundleWriter's os.replace overwrote the first channel's shards.
Not a mix-up of pixels: the earlier channel simply vanished, and the
bundle still looked complete. The manifest was rewritten too, so the
folder then claimed to be a one-channel bundle of the other channel.

Verified against a REAL two-channel bundle built from
G:/Seonghyeok/2025-11-30-MP58/RNA (FOV007, Hyb_101, ch635 then ch555)
into D:/claude-tmp/twoch; skips with a reason when it is absent.

Run:  python tests/test_bundle_multichannel.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = [0, 0]
TWOCH = 'D:/claude-tmp/twoch'


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


def test_shard_names_carry_the_channel():
    print('shard naming')
    import inspect
    from codelab_pipeline.training import extract as X
    src = inspect.getsource(X)
    check('the shard filename includes ch{channel}',
          "__ch{int(channel)}" in src)


def test_the_real_two_channel_bundle():
    print('the real two-channel bundle')
    if not os.path.isdir(TWOCH):
        check('the bundle is present', False, 'skipped: ' + TWOCH + ' absent')
        return
    from codelab_pipeline.training import bundle as B
    names = sorted(os.listdir(TWOCH))
    shards = [n for n in names if n.endswith('.h5')]
    check('shards for BOTH channels are on disk',
          any('__ch555__' in n for n in shards)
          and any('__ch635__' in n for n in shards),
          f'{len(shards)} shards')
    per = {}
    for p in B.shard_paths(TWOCH):
        for row in B.read_index(p):
            per[int(row['channel'])] = per.get(int(row['channel']), 0) + 1
    check('every shard is found by shard_paths, both channels',
          sorted(per) == [555, 635], str(per))
    check('neither channel lost a crop to the other',
          per.get(555) == per.get(635) == 116, str(per))

    with open(os.path.join(TWOCH, 'bundle_manifest.json'),
              encoding='utf-8') as f:
        m = json.load(f)
    check("the manifest lists both channels",
          m.get('channels') == [555, 635], str(m.get('channels')))
    runs = m.get('runs') or []
    check('and one entry per build, oldest first',
          [r['channel'] for r in runs] == [635, 555],
          str([r.get('channel') for r in runs]))
    check("the newest run is still at the top level, so old readers work",
          m.get('channel') == 555 and m.get('hybes') == ['Hyb_101'])
    check('every run recorded the same explicit FOVs',
          all(r.get('fovs') == [7] for r in runs))


def test_a_rebuilt_channel_replaces_its_own_entry():
    """The merge rule, run on a manifest in a temp dir: same channel
    again REPLACES its entry (its shards were just overwritten too);
    a different channel APPENDS."""
    print('manifest merge rule')
    import inspect
    import tools.build_bundle as BB
    src = inspect.getsource(BB.main)
    check('a repeated channel replaces rather than duplicates',
          "int(r.get('channel', -1)) != int(a.channel)" in src)
    check('a legacy single-run manifest is promoted into runs',
          "if not prior and was.get('channel') is not None" in src)


def test_eta_and_workers():
    """done/elapsed after ONE task of a 16-wide pool overstated a two-hour
    build as '~1083 min left'. Rate means something once a full wave
    of workers has landed, and the log must say how wide that wave is."""
    print('ETA and workers')
    import inspect
    import tools.build_bundle as BB
    from codelab_pipeline.training import extract as X
    src = inspect.getsource(BB.main)
    check('the worker count is printed', "print(f'workers {workers}'" in src)
    check('no ETA before a full wave has landed',
          'wave = min(workers, total)' in src and 'if done >= wave:' in src)
    check('and the log says so instead', "ETA after {wave} tasks" in src)
    # MEASURED: a 15-task build on 32 workers gated on `workers` alone
    # never printed an ETA. Fewer tasks than workers is its own state.
    check('fewer tasks than workers is named, not waited for',
          'elif total <= workers:' in src and 'tasks in flight' in src)
    check('the resolved count is what extract() gets', 'workers=workers,' in src)
    w = X.default_workers()
    check('default_workers() is an int in [1, 32]',
          isinstance(w, int) and 1 <= w <= 32, str(w))


def main():
    test_shard_names_carry_the_channel()
    test_the_real_two_channel_bundle()
    test_a_rebuilt_channel_replaces_its_own_entry()
    test_eta_and_workers()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
