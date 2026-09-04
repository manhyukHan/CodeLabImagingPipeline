"""
Catch the freeze in the act, unattended, and say WHICH figure caused it.

Sampling proved what the GUI thread does when the window goes Not
Responding -- one uninterrupted matplotlib paint, 120 seconds, 99.4% of it
inside Text.draw -- but not which figure. Above the paint the stack is
Qt's event loop, so no application frame names the widget, and every
attempt to catch it by hand arrived after it was over.

So this watches instead of asking. It polls py-spy for the GUI's main
thread, and the moment that thread stops being parked in the event loop
it takes a FULL dump WITH LOCALS -- which is what actually identifies the
figure, because the Text.draw frame carries the Text object being drawn
and its string. "146 | 4" is a spot label; a bare cell number is the
context panel; a hybe name is something else entirely.

Deliberately cheap between hits: one py-spy dump every poll, no process
walks (walking every process on this shared server has measured over
three minutes under load, which would make the watcher part of the
problem it is watching).

  python tools/catch_stall.py                     # find the GUI itself
  python tools/catch_stall.py --pid 104288
  python tools/catch_stall.py --seconds 900 --out stalls.txt
"""
import argparse
import os
import subprocess
import sys
import time

# The frame a parked Qt event loop sits in: app.exec_() in main.py. A main
# thread showing only this is idle; anything else is work worth catching.
IDLE_MARKER = 'main.py:122'


def find_pyspy():
    """py-spy, wherever it was installed."""
    for cand in (os.environ.get('PY_SPY'),
                 'py-spy', 'py-spy.exe'):
        if not cand:
            continue
        try:
            subprocess.run([cand, '--version'], capture_output=True, timeout=30)
            return cand
        except Exception:
            continue
    raise SystemExit('py-spy not found -- set PY_SPY to its path')


def find_gui():
    """The python process that owns a visible main window."""
    try:
        import psutil
    except Exception:
        raise SystemExit('pass --pid (psutil unavailable)')
    best = None
    for p in psutil.process_iter(['pid', 'name', 'num_threads']):
        try:
            if 'python' not in (p.info['name'] or '').lower():
                continue
            # the GUI is the one with a Qt thread pool, not a pool child
            if (p.info['num_threads'] or 0) >= 6:
                best = p.pid
        except Exception:
            continue
    if best is None:
        raise SystemExit('no candidate GUI process -- pass --pid')
    return best


def dump(pyspy, pid, locals_=False):
    cmd = [pyspy, 'dump', '--pid', str(pid)]
    if locals_:
        cmd.append('--locals')
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return out.stdout or ''
    except Exception as e:
        return f'(dump failed: {e})'


def kernel_view(pid, samples=8, interval=0.05):
    """The main thread's scheduler state over a short burst, as one line.

    Eight snapshots 50 ms apart, so a single unlucky instant does not get
    reported as a verdict. The main thread is the busiest one, which in a
    Qt app is the GUI thread that has been drawing since launch.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import thread_state
    except Exception as e:
        return f'(thread_state unavailable: {e})'
    try:
        spi, ths = thread_state.threads_of(pid)
        if not ths:
            return '(process not found)'
        tid = thread_state.busiest(ths)['tid']
        seen = []
        for _ in range(samples):
            _spi, ths = thread_state.threads_of(pid)
            me = next((t for t in ths if t['tid'] == tid), None)
            if me:
                seen.append(me)
            time.sleep(interval)
        if not seen:
            return '(thread gone)'
        states = {}
        for m in seen:
            k = thread_state.STATE.get(m['state'], m['state'])
            if m['state'] == 5:
                k += '/' + thread_state.REASON.get(m['reason'], str(m['reason']))
            states[k] = states.get(k, 0) + 1
        dist = ', '.join(f'{k} x{v}' for k, v in sorted(states.items(), key=lambda kv: -kv[1]))
        return (f'tid {tid}: {dist}; prio {seen[-1]["prio"]}/{seen[-1]["base"]}; '
                f'ctxsw +{seen[-1]["ctxsw"] - seen[0]["ctxsw"]} in {samples*interval:.1f}s; '
                f'hard faults {spi["hard_faults"]}')
    except Exception as e:
        return f'(kernel view failed: {e})'


def main_thread_block(text):
    """Just the MainThread section of a dump."""
    lines = text.splitlines()
    out, on = [], False
    for line in lines:
        if line.startswith('Thread ') and 'MainThread' in line:
            on = True
        elif line.startswith('Thread '):
            on = False
        if on:
            out.append(line)
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, default=None)
    ap.add_argument('--seconds', type=float, default=900.0)
    ap.add_argument('--interval', type=float, default=0.4)
    ap.add_argument('--out', default=None)
    ap.add_argument('--min-hits', type=int, default=2,
                    help='consecutive busy polls before taking the full dump, '
                         'so a single fast repaint is not reported as a stall')
    args = ap.parse_args()

    pyspy = find_pyspy()
    pid = args.pid or find_gui()
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'log', 'stall_catch.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f'watching pid {pid} for {args.seconds:.0f} s; writing to {out_path}')
    print('a hit needs the main thread busy on '
          f'{args.min_hits} consecutive polls ({args.interval:.1f} s apart)\n')

    start = time.perf_counter()
    busy = 0
    caught = 0
    with open(out_path, 'a', encoding='utf-8') as fh:
        fh.write(f'\n===== watch started {time.strftime("%Y-%m-%d %H:%M:%S")} '
                 f'pid {pid} =====\n')
        while time.perf_counter() - start < args.seconds:
            text = dump(pyspy, pid)
            mt = main_thread_block(text)
            if not mt:
                time.sleep(args.interval)
                continue
            working = IDLE_MARKER not in mt or mt.count('\n') > 2
            if not working:
                busy = 0
                time.sleep(args.interval)
                continue
            busy += 1
            if busy < args.min_hits:
                time.sleep(args.interval)
                continue
            # Still busy after min_hits polls: this is a real block. Take
            # the expensive dump, the one that names the figure -- and,
            # from the kernel, whether the thread is starved or blocked.
            # py-spy shows WHAT it is doing; only the kernel's ThreadState
            # says whether it is being allowed to do it. A paint that
            # costs 300 ms of CPU was measured taking 2-44 s of wall
            # here, and those two facts together are the whole question.
            t_hit = time.strftime('%H:%M:%S')
            full = dump(pyspy, pid, locals_=True)
            kernel = kernel_view(pid)
            caught += 1
            head = [l for l in mt.splitlines()[1:6]]
            print(f'[{t_hit}] STALL #{caught} caught -- top frames:')
            for h in head:
                print('   ' + h.strip())
            print('   kernel: ' + kernel)
            fh.write(f'\n----- stall #{caught} at {t_hit} -----\n')
            fh.write('kernel view of main thread: ' + kernel + '\n')
            fh.write(full)
            fh.flush()
            # wait for it to end, so one long stall is one record
            while time.perf_counter() - start < args.seconds:
                time.sleep(args.interval)
                mt2 = main_thread_block(dump(pyspy, pid))
                if mt2 and IDLE_MARKER in mt2 and mt2.count('\n') <= 2:
                    break
            busy = 0
    print(f'\ndone: {caught} stall(s) recorded in {out_path}')


if __name__ == '__main__':
    sys.exit(main())
