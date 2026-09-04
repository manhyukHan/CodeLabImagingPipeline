"""
Tests for codelab_pipeline/tuning.py + windows/run_probe.py.

These two exist to make an A/B possible, so what has to be pinned is
exactly what a broken A/B would look like from the outside -- which is:
like a working one. Every property below is one whose absence still
produces a plausible-looking run and a plausible-looking number.

  * A tuning edit that does not take effect leaves both arms on one
    setting and they tie. Three real attempts to measure the disk
    contention already died this way (see windows/run_probe.py); the
    third tied at 1.04x because the load came from a build predating the
    change under test.
  * A typo'd level that silently means "off" turns an arm into the other
    arm, and the log still says what the user typed.
  * A mid-run edit that reaches half the children makes an arm that is
    neither arm.
  * A lag figure of zero is indistinguishable from a lag figure that was
    never taken, unless "not measured" is a distinct answer.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_tuning_switches.py
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline import tuning                             # noqa: E402
from codelab_pipeline import process_guard                      # noqa: E402
from codelab_pipeline.alignment import chain                    # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


_STAMP = [1]


def write(path, data):
    """Write the tuning file and force its mtime forward.

    The cache is keyed on (path, mtime, size), and two writes inside one
    filesystem timestamp tick would otherwise be indistinguishable -- the
    test would pass by reading a stale cache entry that happened to hold
    the right answer.
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data if isinstance(data, str) else json.dumps(data))
    t = time.time() + _STAMP[0]
    _STAMP[0] += 1
    os.utime(path, (t, t))


def clear_env():
    for var in (tuning.WORKERS_ENV, tuning.IO_PRIORITY_ENV,
                tuning.IO_PRIORITY_PIN_ENV):
        os.environ.pop(var, None)


def test_resolution(path):
    print('\n-- resolving the two knobs --')
    clear_env()

    os.remove(path) if os.path.exists(path) else None
    n, n_src = tuning.cell_alignment_workers()
    io, io_src = tuning.child_io_priority()
    check('no file: workers falls through to the measured default',
          n is None and n_src == 'default')
    check('no file: I/O priority defaults to verylow',
          io == 'verylow' and io_src == 'default')

    write(path, {'cell_alignment_workers': 4, 'child_io_priority': 'normal'})
    n, n_src = tuning.cell_alignment_workers()
    check('file sets the worker count', n == 4, str((n, n_src)))
    check('file sets the I/O priority', tuning.child_io_priority()[0] == 'normal')

    # The one that matters most: four combinations in ONE session means
    # the file is re-read, not captured at import.
    write(path, {'cell_alignment_workers': 12, 'child_io_priority': 'verylow'})
    check('an edit takes effect without restarting the process',
          tuning.cell_alignment_workers()[0] == 12
          and tuning.child_io_priority()[0] == 'verylow')

    os.environ[tuning.WORKERS_ENV] = '99'
    os.environ[tuning.IO_PRIORITY_ENV] = 'low'
    check('the file outranks a stale exported variable',
          tuning.cell_alignment_workers()[0] == 12
          and tuning.child_io_priority()[0] == 'verylow')

    write(path, {})
    n, n_src = tuning.cell_alignment_workers()
    check('with the key absent, the environment is still honoured',
          n == 99 and n_src == tuning.WORKERS_ENV and tuning.child_io_priority()[0] == 'low')
    clear_env()


def test_bad_input_is_loud(path):
    print('\n-- a typo must not quietly become an arm of its own --')
    write(path, {'cell_alignment_workers': 'eight', 'child_io_priority': 'turbo'})
    n, n_src = tuning.cell_alignment_workers()
    io, io_src = tuning.child_io_priority()
    check('unparseable worker count falls back AND says so',
          n is None and 'unparseable' in n_src, n_src)
    check('unknown I/O level falls back AND says so',
          io == 'verylow' and 'ignored' in io_src, io_src)

    write(path, {'cell_alignment_workers': 0})
    n, n_src = tuning.cell_alignment_workers()
    check('0 means "default", never "a pool with no workers"',
          n is None and 'ignored 0' in n_src, n_src)

    write(path, '{ this is not json')
    check('a malformed file cannot stop a run from starting',
          tuning.cell_alignment_workers()[0] is None
          and tuning.child_io_priority()[0] == 'verylow')


def test_chain_honours_the_cap(path):
    print('\n-- the cap actually reaches the pool --')
    clear_env()
    measured_default = max(1, min((os.cpu_count() or 4) - 2, 16))
    write(path, {})
    check('unset leaves the measured default exactly as it was',
          chain.max_cell_alignment_workers() == measured_default,
          str(measured_default))
    for n in (1, 4, 8, 12):
        write(path, {'cell_alignment_workers': n})
        check(f'cap of {n} is honoured', chain.max_cell_alignment_workers() == n)
    # The interesting direction is downward, but the curve has to be
    # walkable on both sides of the current default or the knee cannot be
    # found -- so the override is deliberately not clamped by hard_ceiling.
    write(path, {'cell_alignment_workers': 24})
    check('the override can exceed the hard ceiling',
          chain.max_cell_alignment_workers() == 24)
    write(path, {'cell_alignment_workers': 'many'})
    check('garbage leaves the measured default',
          chain.max_cell_alignment_workers() == measured_default)

    # The cap throttles a BATCH run that is starving the GUI of disk. The
    # per-hybe pool is what the GUI uses to draw a single-cell preview, so
    # a cap reaching it would throttle the thing the cap is protecting.
    # This is the shape of a real incident: a batch-path change that
    # touched a helper the preview shared made the preview slow enough
    # that the app had to be reverted mid-session.
    write(path, {'cell_alignment_workers': 2})
    check('the batch pool honours the cap', chain.max_cell_alignment_workers() == 2)
    check('the single-cell preview pool is NOT capped with it',
          chain.measured_cell_alignment_workers() == measured_default,
          str(chain.measured_cell_alignment_workers()))


def test_child_side(path):
    print('\n-- what a spawned child ends up doing --')
    clear_env()

    write(path, {'child_io_priority': 'normal'})
    check('unpinned child reads the file: normal switches it off',
          process_guard.lower_io_priority() is False)

    write(path, {'child_io_priority': 'verylow'})
    applied = process_guard.lower_io_priority()
    check('unpinned child reads the file: verylow is applied', applied is True)
    if applied:
        try:
            import psutil
            check('the process really is at VERYLOW on the disk queue',
                  psutil.Process().ionice() == psutil.IOPRIO_VERYLOW,
                  str(psutil.Process().ionice()))
        except Exception as e:
            check('the process really is at VERYLOW on the disk queue', False, str(e))

    # The pin is what makes one run one arm.
    write(path, {'child_io_priority': 'normal'})
    level, _cache_gb, _note = tuning.apply_child_env()
    check('apply_child_env pins the resolved level',
          level == 'normal' and os.environ[tuning.IO_PRIORITY_PIN_ENV] == 'normal')
    write(path, {'child_io_priority': 'verylow'})
    check('a mid-run edit cannot split one run across two settings',
          process_guard.lower_io_priority() is False)

    # A pin is trusted only because apply_child_env validated it. Anything
    # else in that variable is not a pin, and must not bypass validation.
    os.environ[tuning.IO_PRIORITY_PIN_ENV] = 'turbo'
    check('a corrupt pin is ignored rather than silently meaning "off"',
          process_guard.lower_io_priority() is True)
    clear_env()


def test_cache_budget_is_total(path):
    print('\n-- the MIP cache budget is a TOTAL, not a per-child number --')
    from codelab_pipeline import process_guard as pg

    # Measured, 6 children reading MIPs: the free+zero page list fell by
    # 5.89 GB at 1.0 GB each, 3.35 GB at 0.5 GB each, and 0.20 GB with the
    # cache off -- the drain tracks children x budget. A per-child number
    # with no total therefore scales retention with the worker count, and
    # at 16 workers that is 16 GB against a 7.3 GB list.
    for n in (1, 4, 8):
        gb, _note = pg.child_mip_cache_gb(n)
        check(f'{n} worker(s) share the whole budget, not multiply it',
              abs(gb * n - pg.TOTAL_MIP_CACHE_GB) < 1e-6, f'{gb:.2f} x {n}')

    gb, note = pg.child_mip_cache_gb(16)
    check('past the floor, the floor wins over the total',
          gb == pg.MIN_CHILD_MIP_CACHE_GB, str(gb))
    check('and the caller is TOLD the total is exceeded',
          'OVER the' in note and 'worker(s) would fit it' in note, note)
    check('the floor is where the measured cliff is (0.25 GB -> 4061 ms, '
          '0.5 GB -> 3.6 ms)', pg.MIN_CHILD_MIP_CACHE_GB == 0.5,
          str(pg.MIN_CHILD_MIP_CACHE_GB))

    gb, note = pg.child_mip_cache_gb(None)
    check('an unknown pool size falls back to the old flat number',
          gb == pg.CHILD_MIP_CACHE_GB, str(gb))

    # The pin must not resize the PARENT's own cache: the GUI reads
    # CODELAB_MIP_CACHE_GB too and is not part of the pool's budget.
    clear_env()
    os.environ.pop('CODELAB_MIP_CACHE_GB', None)
    write(path, {})
    _io, cache_gb, _n = tuning.apply_child_env(8)
    check('the child share is pinned under its own variable',
          os.environ.get(tuning.MIP_CACHE_PIN_ENV) == f'{cache_gb:g}',
          str(os.environ.get(tuning.MIP_CACHE_PIN_ENV)))
    check("and the parent's own CODELAB_MIP_CACHE_GB is left alone",
          'CODELAB_MIP_CACHE_GB' not in os.environ,
          str(os.environ.get('CODELAB_MIP_CACHE_GB')))

    # A child started with that pin adopts it.
    os.environ['CODELAB_MIP_CACHE_GB'] = '9'      # a stale inherited value
    process_guard.child_initializer()
    check('a pinned child overrides an inherited cache size',
          os.environ['CODELAB_MIP_CACHE_GB'] == f'{cache_gb:g}',
          os.environ['CODELAB_MIP_CACHE_GB'])
    os.environ.pop(tuning.MIP_CACHE_PIN_ENV, None)
    os.environ.pop('CODELAB_MIP_CACHE_GB', None)
    clear_env()


def test_label_reports_reality(path):
    print('\n-- the stamp that makes a number attributable --')
    clear_env()
    write(path, {'cell_alignment_workers': 16, 'child_io_priority': 'verylow'})
    label = tuning.settings_label(workers_in_effect=3)
    check('the label reports the pool that was BUILT, not the one requested',
          'workers=3' in label and 'workers=16' not in label, label)
    check('the label names where each value came from',
          'tuning.json' in label, label)
    write(path, {})
    check('an unset worker count shows as auto, not as a number',
          'workers=auto' in tuning.settings_label(), tuning.settings_label())


def test_lag_probe():
    print('\n-- measuring the stall from inside the event loop --')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtCore, QtWidgets
    from windows.run_probe import EventLoopLag, summary_line

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    probe = EventLoopLag()

    def spin(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)

    tok = probe.mark()
    spin(1.0)
    idle = probe.since(tok)
    check('an idle loop is sampled at all', idle and idle['ticks'] >= 5, str(idle))

    tok = probe.mark()
    spin(0.3)
    time.sleep(0.8)                    # the GUI thread, blocked, as in a stall
    spin(0.3)
    stalled = probe.since(tok)
    check('an 800 ms block is reported at close to its real size',
          stalled['max_ms'] >= 700, str(round(stalled['max_ms'])))
    # Compared against the idle window rather than against a fixed
    # percentage. An absolute bound here tests the MACHINE, not the probe:
    # run inside a full-suite sweep on a loaded box the "idle" loop is
    # genuinely late, and this assertion failed for that reason while
    # passing standalone. The property that matters is that a real block
    # stands out from quiet, and that survives a busy machine.
    check('a block stands out from an idle window',
          stalled['blocked_pct'] > idle['blocked_pct'] + 20.0,
          f'idle {idle["blocked_pct"]:.1f}% vs stalled {stalled["blocked_pct"]:.1f}%')
    check('a rare stall does not move the median far',
          stalled['median_ms'] < 100, str(stalled['median_ms']))

    check('a window too short to sample says so instead of reporting zero',
          probe.since(probe.mark()) is None)

    # The blind spot that a 5-minute profile of the live app caught: the
    # main thread sat in a QMessageBox for 46.7% of it against 0.6% of
    # real work, and lag alone called that window healthy -- a modal runs
    # its own event loop, so the timer stays on time while the window is
    # unusable. Modal time is therefore counted, and counted SEPARATELY.
    idle_modal = probe.since(tok)
    check('no modal, no modal time', idle_modal['modal_pct'] == 0.0)

    # A bare QDialog rather than the QMessageBox this models: the message
    # box segfaults under the offscreen platform, and what is being tested
    # is the detection of modality, which any modal widget exercises.
    box = QtWidgets.QDialog()
    box.setModal(True)
    box.show()
    QtWidgets.QApplication.setActiveWindow(box)
    tok = probe.mark()
    spin(0.8)
    with_modal = probe.since(tok)
    box.hide()
    check('a modal dialog is reported as holding the window',
          with_modal['modal_pct'] > 50.0, str(round(with_modal['modal_pct'], 1)))
    check('a modal does NOT masquerade as event-loop lag',
          with_modal['blocked_pct'] < 20.0, str(round(with_modal['blocked_pct'], 1)))
    check('the summary says so in words',
          'modal dialog held the window' in
          summary_line('r', 's', 5, 'passes', 1.0, with_modal))
    check('the summary distinguishes "no lag" from "not measured"',
          'not sampled' in summary_line('r', 's', 1, 'passes', 0.05, None)
          and 'not sampled' not in summary_line('r', 's', 9, 'passes', 9.0, stalled))


def main():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, 'tuning.json')
    os.environ[tuning.TUNING_FILE_ENV] = path
    try:
        test_resolution(path)
        test_bad_input_is_loud(path)
        test_chain_honours_the_cap(path)
        test_cache_budget_is_total(path)
        test_child_side(path)
        test_label_reports_reality(path)
        test_lag_probe()
    finally:
        os.environ.pop(tuning.TUNING_FILE_ENV, None)
        clear_env()

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
