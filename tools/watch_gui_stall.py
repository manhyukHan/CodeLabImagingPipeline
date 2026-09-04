"""
Why is the app "Not Responding"? Sample the answer while it is happening.

Run this in a terminal WHILE a cell alignment is going and the window has
gone unresponsive. It samples the GUI process and the machine, and after
~30 s prints the one number that decides the diagnosis:

    starvation = wall time / CPU time consumed by the GUI process

  ~1      the GUI is BUSY -- it is computing, and the fix is to move that
          work off the GUI thread.
  >> 1    the GUI is WAITING. Then the page-fault rate says on what:
            faults high  -> memory pressure; it is waiting on the pager.
            faults low   -> waiting on a lock or on disk I/O, NOT paging,
                            and the memory diagnosis was wrong.

This exists because the last diagnosis (438 ms of CPU against 15144 ms of
wall, with the machine at ~5000 pages/s) was never re-checked after the
per-child cache cap that was supposed to fix it. A fix nobody measured is
a hypothesis.

  python tools/watch_gui_stall.py               # auto-pick the GUI process
  python tools/watch_gui_stall.py --pid 12345   # or name it
  python tools/watch_gui_stall.py --seconds 60

Deliberately cheap: no cmdline lookups (opening every process for its
command line took over 3 minutes on this machine under load, which would
make the tool part of the problem it is measuring).
"""
import argparse
import sys
import time

import psutil


def find_gui(pid=None):
    """The app's main process: the biggest python by resident memory.

    The GUI holds the containers, the figures and the session caches, so
    it outweighs its own pool children -- which are the only other python
    processes in play during an alignment. Passing --pid removes the
    guesswork when that is not true.
    """
    if pid:
        return psutil.Process(int(pid))
    best, best_rss = None, -1
    for p in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'python' not in (p.info['name'] or '').lower():
                continue
            rss = p.info['memory_info'].rss
            if rss > best_rss:
                best, best_rss = p, rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if best is None:
        raise SystemExit('no python process found -- is the app running?')
    return psutil.Process(best.pid)


def python_fleet():
    """(count, total resident bytes) over every python process."""
    n, rss = 0, 0
    for p in psutil.process_iter(['name', 'memory_info']):
        try:
            if 'python' in (p.info['name'] or '').lower():
                n += 1
                rss += p.info['memory_info'].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return n, rss


def faults_of(proc):
    """Cumulative page faults, or None where the platform does not say."""
    try:
        return getattr(proc.memory_info(), 'num_page_faults', None)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, default=None)
    ap.add_argument('--seconds', type=float, default=30.0)
    ap.add_argument('--interval', type=float, default=2.0)
    args = ap.parse_args()

    gui = find_gui(args.pid)
    # the fleet walk touches EVERY process on the machine and on this
    # shared server that alone takes seconds (other users' processes each
    # cost an access-denied round trip) -- so it runs once here and once
    # at the end, never inside the sampling loop. A monitor whose own
    # tick is slower than its interval measures itself.
    n_py0, rss_py0 = python_fleet()
    print(f'watching pid {gui.pid} ({gui.name()}) for {args.seconds:.0f} s; '
          f'{n_py0} python process(es), {rss_py0 / 2**30:.1f} GB resident\n')
    print(f'{"t":>5}  {"GUI cpu%":>8}  {"faults/s":>9}  {"threads":>7}  '
          f'{"avail GB":>8}')

    t_start = time.perf_counter()
    cpu0 = sum(gui.cpu_times()[:2])
    f_prev = faults_of(gui)
    t_prev = t_start
    samples = []
    while time.perf_counter() - t_start < args.seconds:
        time.sleep(args.interval)
        now = time.perf_counter()
        dt = now - t_prev
        try:
            cpu_now = sum(gui.cpu_times()[:2])
            threads = gui.num_threads()
        except psutil.NoSuchProcess:
            print('the process exited')
            break
        f_now = faults_of(gui)
        d_cpu = cpu_now - (samples[-1]['cpu'] if samples else cpu0)
        d_f = (f_now - f_prev) if (f_now is not None and f_prev is not None) else 0
        vm = psutil.virtual_memory()
        print(f'{now - t_start:5.0f}  {100 * d_cpu / dt:8.1f}  {d_f / dt:9.0f}  '
              f'{threads:7d}  {vm.available / 2**30:8.1f}')
        samples.append({'cpu': cpu_now, 'faults': f_now, 't': now})
        f_prev, t_prev = f_now, now

    if not samples:
        raise SystemExit('no samples taken')
    wall = samples[-1]['t'] - t_start
    cpu = samples[-1]['cpu'] - cpu0
    print()
    print(f'over {wall:.0f} s of wall time the GUI process used '
          f'{cpu:.2f} s of CPU')
    starved = cpu <= 0.001 or (wall / cpu) > 3.0
    if cpu <= 0.001:
        print('  starvation: TOTAL -- the GUI consumed no CPU at all; it is '
              'waiting, not working')
    else:
        print(f'  starvation ratio: {wall / cpu:.1f}x  '
              f'(1x = busy computing, >>1 = waiting)')
    rates = [(samples[i]['faults'] - samples[i - 1]['faults']) /
             (samples[i]['t'] - samples[i - 1]['t'])
             for i in range(1, len(samples))
             if samples[i]['faults'] is not None
             and samples[i - 1]['faults'] is not None]
    med = sorted(rates)[len(rates) // 2] if rates else None
    if med is not None:
        print(f'  median page faults/s for this process: {med:.0f}')
    # the fault rate only DIAGNOSES anything once the process is shown to
    # be waiting -- a busy process legitimately faults at ~0, and reading
    # that as "not paging, so it must be a lock" is a verdict about a
    # question nobody asked
    if not starved:
        print('  -> the GUI is BUSY, not starved: it is computing. Move that '
              'work off the GUI thread; contention is not the story here.')
    elif med is None:
        print('  -> starved, but this platform reports no page-fault count, '
              'so the cause is undetermined')
    elif med > 2000:
        print('  -> starved AND faulting hard: MEMORY PRESSURE, it is waiting '
              'on the pager')
    else:
        print('  -> starved but NOT faulting: it is waiting on a lock or on '
              'disk I/O, so the memory explanation does not hold here')
    print('\nRun this again with the app IDLE for a baseline to compare against.')


if __name__ == '__main__':
    main()
