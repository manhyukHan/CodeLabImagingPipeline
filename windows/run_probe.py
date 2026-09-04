"""
Measure "the app is Not Responding" instead of arguing about it.

This exists because three separate attempts to A/B the disk-contention fix
from OUTSIDE the app all failed, each for a different reason:

  1. A cold baseline was compared against warm re-reads of the SAME files.
     The no-load baseline came out 17x SLOWER than the loaded case (512 ms
     vs 29 ms), which is backwards.
  2. The GUI's files were separated but the WORKERS' files were not. 30
     stacks is 7.4 GB and this box had 157 GB free, so the second condition
     read them from RAM: its workers did 235 passes to the first's 62 and
     the "improvement" was the OS cache, not the change under test.
  3. Files were fully separated -- and the dominant load turned out to be a
     real alignment running the PREVIOUS build, outside the experiment's
     control. Both arms measured the same unpriorit-ised workers, so of
     course they tied (1.04x).

The common failure is that an external prober is not the GUI and does not
share its fate. What the user actually experiences is the event loop
running late: a click is dispatched by that loop, so when the loop is
blocked the window greys out. So measure THAT, from inside, and attribute
it to the run that caused it.

A QTimer asking for 100 ms that arrives at 3100 ms reports 3000 ms of lag.
Qt coalesces missed timeouts rather than queueing 30 of them, so the sum of
the lags is the wall time the loop was unavailable -- reported here as
`blocked_pct`, which is the honest headline number: "during this run the
GUI was unresponsive 18% of the time".

MODAL DIALOGS ARE THE BLIND SPOT, and it is measured, not hypothetical: a
5-minute profile of the running app found the main thread inside a
QMessageBox for 46.7% of it -- 140 seconds during which clicking the main
window did nothing -- against 0.6% spent on all actual work combined, disk
and HDF5 included. Lag alone would have called that window healthy, because
a modal runs its own event loop and the timer keeps firing on time. The
loop was not late; the window was simply unusable. So modal time is sampled
alongside lag and reported next to it, and neither number is allowed to
stand in for the other.

The cost of the monitor is one no-op slot ten times a second, which is far
below the resolution of anything it is used to measure -- confirmed by
sampling the live app: `_tick` accounted for 0.06% of the main thread.
"""
import time

from PyQt5 import QtCore, QtWidgets

# 10 Hz. Fast enough that a 300 ms stall -- around where a click starts to
# feel dropped -- is caught by at least one tick, slow enough to be free.
TICK_MS = 100

# ~5.5 hours at 10 Hz. A bound, not a budget: the deque is 24 bytes an
# entry, and an alignment left running overnight should not grow it without
# limit.
MAX_TICKS = 200_000


class EventLoopLag(QtCore.QObject):
    """How late the GUI event loop is running, sampled continuously.

    Lives on the GUI thread and must be constructed there -- the whole
    point is to share that thread's fate, so a monitor that ran anywhere
    else would measure nothing.

    Usage is a stopwatch with named laps:

        probe = EventLoopLag(self)      # starts sampling immediately
        tok = probe.mark()              # at the start of a run
        ...
        stats = probe.since(tok)        # at the end of it
    """

    def __init__(self, parent=None, tick_ms=TICK_MS):
        super().__init__(parent)
        self._tick_ms = int(tick_ms)
        self._samples = []              # (monotonic_seconds, lag_ms, modal)
        self._last = time.perf_counter()
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.setInterval(self._tick_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        now = time.perf_counter()
        # Only the EXCESS over the interval we asked for is lag; a timer
        # that arrives on time contributes zero, not 100 ms.
        lag = (now - self._last) * 1000.0 - self._tick_ms
        self._last = now
        if lag < 0:
            lag = 0.0
        self._samples.append((now, lag, _modal_is_up()))
        if len(self._samples) > MAX_TICKS:
            del self._samples[:len(self._samples) - MAX_TICKS]

    def mark(self):
        """A token for 'now', to be handed back to since()."""
        return time.perf_counter()

    def since(self, token):
        """Lag statistics for the window from `token` until now.

        Returns None when the window caught no ticks at all -- a run that
        finished inside 100 ms, or a monitor that was never started. A
        caller must not report zeros in that case, because zero lag and no
        measurement are very different claims.
        """
        rows = [(lag, modal) for t, lag, modal in self._samples if t >= token]
        if not rows:
            return None
        window = [lag for lag, _m in rows]
        elapsed = max(time.perf_counter() - token, 1e-9)
        ordered = sorted(window)
        lost_s = sum(window) / 1000.0
        return {
            'ticks': len(window),
            # Counted separately from lag and never folded into it: a modal
            # keeps the loop on time while making the window unusable, so
            # adding them would double-count nothing and hide everything.
            'modal_pct': 100.0 * sum(1 for _l, m in rows if m) / len(rows),
            'median_ms': _quantile(ordered, 0.5),
            'p95_ms': _quantile(ordered, 0.95),
            'max_ms': ordered[-1],
            'lost_s': lost_s,
            # The headline: what fraction of the run the loop could not
            # answer a click. Clamped because a stall straddling the mark
            # can otherwise attribute pre-run blockage to the run.
            'blocked_pct': min(100.0, 100.0 * lost_s / elapsed),
            'elapsed_s': elapsed,
        }


def _process_note():
    """The cheap suspects for a paint that gets slower as a session ages.

    Reported next to every slow draw because the same figure, with the same
    artist counts, was measured at 468 ms early in a session and 44373 ms
    six minutes later. Whatever explains that is accumulating somewhere,
    and these are the accumulations that cost nothing to count: resident
    memory, live matplotlib figures (a leak shows here), the FT2Font cache
    (text rendering's own), and Python's GC backlog.
    """
    bits = []
    try:
        import psutil
        p = psutil.Process()
        bits.append(f'rss {p.memory_info().rss / 2 ** 30:.2f} GB')
        bits.append(f'threads {p.num_threads()}')
        try:
            bits.append(f'handles {p.num_handles()}')
        except Exception:
            pass
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        bits.append(f'figures {len(plt.get_fignums())}')
    except Exception:
        pass
    try:
        # font_manager._get_font is the lru_cache that hands out FT2Font
        # objects; get_font is a thin wrapper with no cache of its own
        # (checked against matplotlib 3.11.1). A miss rate that climbs with
        # session age would put the growth squarely in text rendering.
        from matplotlib import font_manager
        info = font_manager._get_font.cache_info()
        bits.append(f'fontcache {info.currsize}/{info.maxsize} '
                    f'hits {info.hits} misses {info.misses}')
    except Exception:
        pass
    try:
        import gc
        bits.append('gc ' + '/'.join(str(c) for c in gc.get_count()))
        bits.append(f'tracked {len(gc.get_objects()):d}'
                    if False else f'gen2 {gc.get_stats()[2]["collections"]}')
    except Exception:
        pass
    return ', '.join(bits)


def install_slow_draw_logger(log, threshold_ms=300):
    """Name the figure that just blocked the window, and count what is in it.

    Sampling proved WHAT the GUI thread does when it freezes -- one
    uninterrupted matplotlib paint, 120 seconds, 99.4% of it inside
    Text.draw -- but not WHICH figure. The stack above the paint is Qt's
    event loop, so no application frame identifies the widget, and a
    benchmark of the shape I assumed it was renders in about a second.
    Two orders of magnitude of guessing is worse than one measurement.

    So: wrap the canvas draw, and when one takes longer than a person
    would tolerate, log the owning widget and its artist counts. The
    threshold means quiet figures cost nothing; the wrapper is one
    perf_counter pair around work that already took a third of a second.
    """
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except Exception:
        return False
    if getattr(FigureCanvasQTAgg, '_codelab_draw_logged', False):
        return False
    original = FigureCanvasQTAgg.draw

    def draw(self, *args, **kwargs):
        t0 = time.perf_counter()
        # THIS thread's CPU, not the process's: the question a slow paint
        # raises is whether the GUI thread was computing or waiting, and
        # process-wide CPU would be dominated by whatever else is running.
        # wall ~= cpu means the work is real; cpu << wall means it is
        # blocked on something and the artist count is not the story.
        c0 = time.thread_time()
        try:
            return original(self, *args, **kwargs)
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            cpu_ms = (time.thread_time() - c0) * 1000.0
            if ms >= threshold_ms:
                try:
                    fig = self.figure
                    axes = fig.axes
                    texts = sum(len(ax.texts) for ax in axes) + len(fig.texts)
                    colls = sum(len(ax.collections) for ax in axes)
                    lines = sum(len(ax.lines) for ax in axes)
                    imgs = sum(len(ax.images) for ax in axes)
                    ticks = sum(len(ax.get_xticklabels()) +
                                len(ax.get_yticklabels()) for ax in axes)
                    owner = self.parent()
                    while owner is not None and not owner.isWindow():
                        owner = owner.parent()
                    name = type(owner).__name__ if owner is not None else '?'
                    # The same figure was measured drawing in 468 ms early in
                    # a session and 44373 ms an hour later with an identical
                    # artist count, so the counts alone explain nothing and
                    # the growth has to be attributable to something. RSS and
                    # the live figure/font counts are the cheap suspects.
                    log(f'slow draw {ms:.0f} ms (cpu {cpu_ms:.0f} ms) in {name}: '
                        f'{len(axes)} axes, {texts} text, {ticks} tick labels, '
                        f'{colls} collections, {lines} lines, {imgs} images; '
                        f'canvas {self.width()}x{self.height()} px; '
                        f'{_process_note()}')
                except Exception:
                    pass          # instrumentation must never break a paint

    FigureCanvasQTAgg.draw = draw
    FigureCanvasQTAgg._codelab_draw_logged = True
    return True


def _modal_is_up():
    """Is a modal dialog holding the main window hostage right now?

    Qt answers this directly and cheaply. Wrapped because the accessor
    needs a live QApplication and this is called from a timer that must
    never be the thing that raises.
    """
    try:
        app = QtWidgets.QApplication.instance()
        return app is not None and app.activeModalWidget() is not None
    except Exception:
        return False


def _quantile(ordered, q):
    """Nearest-rank quantile of an already-sorted list."""
    if not ordered:
        return 0.0
    i = int(round(q * (len(ordered) - 1)))
    return ordered[max(0, min(i, len(ordered) - 1))]


def summary_line(label, settings, n_items, item_word, elapsed_s, lag):
    """One comparable line per run.

    Deliberately ONE line with every dimension the A/B varies plus both
    outcomes, because the four combinations are compared by reading the log
    -- a number whose settings are recorded somewhere else is a number that
    cannot be attributed.

    `lag` is an EventLoopLag.since() result, or None when the run was too
    short to catch a tick; the difference is stated rather than hidden.
    """
    rate = (n_items / elapsed_s) if elapsed_s > 0 else 0.0
    head = (f'{label}: {settings} | {n_items} {item_word} in {elapsed_s:.1f} s '
            f'({rate:.2f}/s)')
    if not lag:
        return head + ' | GUI lag: not sampled (run shorter than one tick)'
    modal = ''
    if lag.get('modal_pct'):
        # Said plainly rather than folded into the lag figure: a run that
        # spent half its time behind a dialog is not a run that measured
        # contention, and reading it as one is how a good number gets
        # attributed to the wrong cause.
        modal = (f'; a modal dialog held the window for '
                 f'{lag["modal_pct"]:.1f}% of it')
    return (head +
            f' | GUI blocked {lag["blocked_pct"]:.1f}% of the run '
            f'({lag["lost_s"]:.1f} s); lag median {lag["median_ms"]:.0f} ms, '
            f'p95 {lag["p95_ms"]:.0f} ms, worst {lag["max_ms"]:.0f} ms{modal}')
