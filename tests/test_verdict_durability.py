"""
A committed verdict survives a crash and a second window.

WHY THIS EXISTS: both failures here lose a reviewer's work SILENTLY, and
neither shows up in a single-process happy-path test. An adversarial pass
found them on the real app; this pins them.

  A TORN TAIL SWALLOWED THE NEXT VERDICT. A process killed mid-write
  leaves a fragment with no newline. The next commit appended straight
  onto it, fusing fragment and record into one unparseable line --
  read_log discarded BOTH. So a page that was judged and committed
  vanished, was missing from done_pages, and got silently re-served. The
  fragment alone was the only real loss; the good record was collateral.

  APPEND IS NOT ATOMIC ON WINDOWS. open(path, 'a') seeks to end and then
  writes, so two processes can resolve the same offset and overwrite each
  other. Two windows under one reviewer name is not exotic -- it is what
  happens when someone forgets the first one is open.

The fix for the second is structural rather than a lock: one file per
SESSION, so there is no shared file to race on. A lock would work and
would bring its own failure -- a crashed process leaving one behind, in a
program whose whole purpose is surviving crashes.

These tests use real subprocesses. A test that simulates a crash by
calling a function cannot show that the bytes on disk are recoverable,
which is the only thing being claimed.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_verdict_durability.py
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# THE CONSOLE HERE IS cp949, AND THIS FILE PRINTS THE APP'S OWN WORDS.
# Without this the suite dies on the first em dash it echoes back -- at
# check 18 of 41, on a PASSING check, with no summary line -- so a green
# run and a broken run look the same at a glance and the later test
# groups never execute at all. It was invisible while every run happened
# to carry PYTHONIOENCODING=utf-8.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


from codelab_pipeline.training import verdicts as V      # noqa: E402

PASS, FAIL = [], []
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


ROW = {'key': 'fov001|Hyb_1|ch635|cell1', 'fov': 1, 'hybe': 'Hyb_1',
       'channel': 635, 'cell': 1, 'y0': 0, 'x0': 0, 'h': 8, 'w': 8, 'depth': 4}
CANDS = [(1., 2., 3., 0.9, 1, 1, ''), (4., 5., 6., 0.4, 1, 0, 'occ')]


def all_records(d):
    out = []
    for n in sorted(os.listdir(d)):
        if n.startswith('verdicts_') and n.endswith('.jsonl'):
            out.extend(V.read_log(os.path.join(d, n)))
    return out


def run_child(code):
    return subprocess.run([sys.executable, '-c', textwrap.dedent(code)],
                          cwd=REPO, capture_output=True, text=True, timeout=180)


def main():
    print('\n-- a torn tail must not swallow the next verdict --')
    d = tempfile.mkdtemp()
    log = V.VerdictLog(d, 'tester', session='s1')
    log.commit(ROW, 0, [0, 1], CANDS, accepted=[0])
    with open(log.path, 'a', encoding='utf-8', newline='\n') as f:
        f.write('{"key": "torn", "pag')            # killed mid-write
    later = V.VerdictLog(d, 'tester', session='s1')
    later.commit(ROW, 1, [0, 1], CANDS, accepted=[1])
    recs = V.read_log(log.path)
    check('both real verdicts survive a torn line between them',
          len(recs) == 2, f'{len(recs)} readable of 2 committed')
    check('and they are the right two',
          sorted(r['page'] for r in recs) == [0, 1],
          str(sorted(r.get('page') for r in recs)))
    check('the page after the tear is in done_pages, so it is not re-served',
          (ROW['key'], 1) in V.VerdictLog(d, 'tester', session='s2').done_pages())

    print('\n-- a torn tail mid-file, not just at the end --')
    d2 = tempfile.mkdtemp()
    lg = V.VerdictLog(d2, 'tester', session='s1')
    lg.commit(ROW, 0, [0], CANDS, accepted=[])
    with open(lg.path, 'a', encoding='utf-8', newline='\n') as f:
        f.write('{"half": ')
    for pg in (1, 2, 3):
        V.VerdictLog(d2, 'tester', session='s1').commit(
            ROW, pg, [0], CANDS, accepted=[0])
    recs = V.read_log(lg.path)
    check('every verdict after the tear is still readable',
          len(recs) == 4, f'{len(recs)} of 4')

    print('\n-- a real killed process leaves the file recoverable --')
    d3 = tempfile.mkdtemp()
    child = run_child(f'''
        import os, sys
        sys.path.insert(0, {REPO!r})
        from codelab_pipeline.training import verdicts as V
        log = V.VerdictLog({d3!r}, 'tester', session='crash')
        log.commit({ROW!r}, 0, [0], {CANDS!r}, accepted=[0])
        # write a partial line and die hard, no flush of a full record
        f = open(log.path, 'a', encoding='utf-8', newline='\\n')
        f.write('{{"key": "dying"')
        f.flush()
        os._exit(1)
    ''')
    check('the child died as intended', child.returncode != 0,
          f'rc={child.returncode} {child.stderr.strip()[:60]}')
    after = V.VerdictLog(d3, 'tester', session='resumed')
    after.commit(ROW, 1, [0], CANDS, accepted=[])
    recs = all_records(d3)
    check('the pre-crash verdict is still there', any(r['page'] == 0 for r in recs))
    check('and the post-crash one too', any(r['page'] == 1 for r in recs))

    print('\n-- two windows, one reviewer, no lost verdicts --')
    # THE case: someone forgets the first window is open. Two processes
    # commit many pages each, interleaved, under one reviewer name.
    d4 = tempfile.mkdtemp()
    N = 40
    procs = [run_child(f'''
        import sys
        sys.path.insert(0, {REPO!r})
        from codelab_pipeline.training import verdicts as V
        log = V.VerdictLog({d4!r}, 'tester')
        row = dict({ROW!r})
        for i in range({N}):
            row['key'] = 'w{w}|page%d' % i
            log.commit(row, i, [0], {CANDS!r}, accepted=[0])
        print(log.path)
    ''') for w in (0, 1)]
    for k, p in enumerate(procs):
        check(f'writer {k} finished cleanly', p.returncode == 0,
              p.stderr.strip()[:70])
    recs = all_records(d4)
    check(f'all {2 * N} verdicts survived two concurrent writers',
          len(recs) == 2 * N, f'{len(recs)} of {2 * N}')
    check('every line is valid JSON (nothing was interleaved mid-record)',
          all(isinstance(r, dict) and 'key' in r for r in recs))
    keys = {(r['key'], r['page']) for r in recs}
    check('and none collided or overwrote another', len(keys) == 2 * N,
          f'{len(keys)} distinct')

    print('\n-- sessions are separate files, resume sees them all --')
    files = [n for n in os.listdir(d4) if n.endswith('.jsonl')]
    check('the two writers used different files', len(files) == 2, str(files))
    check('done_pages spans every session of this reviewer',
          len(V.VerdictLog(d4, 'tester').done_pages()) == 2 * N,
          str(len(V.VerdictLog(d4, 'tester').done_pages())))
    check('another reviewer still starts empty',
          len(V.VerdictLog(d4, 'someone-else').done_pages()) == 0)

    print('\n-- merge and labels still work across sessions --')
    merged, _agree = V.merge(d4)
    check('merge sees every record', len(merged) == 2 * N, str(len(merged)))
    lab = V.labels(d4)
    check('labels are produced per crop', len(lab) == 2 * N, str(len(lab)))

    print('\n-- a name that is not a filename cannot escape the folder --')
    d5 = tempfile.mkdtemp()
    for nasty in ('a b/c', '../../etc', 'Ünïcode', ''):
        lg = V.VerdictLog(d5, nasty)
        lg.commit(ROW, 0, [0], CANDS, accepted=[])
        check(f'{nasty!r} writes inside the bundle folder',
              os.path.dirname(os.path.abspath(lg.path)) ==
              os.path.abspath(d5), lg.path)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
