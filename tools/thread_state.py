"""
Is a thread starved (runnable, denied CPU) or blocked (waiting on something)?

Only the kernel knows, and on Windows it says so through
NtQuerySystemInformation(SystemProcessInformation): one call returns a
snapshot of every process and thread, and each thread carries a ThreadState
and, when waiting, a WaitReason. No handles, no privileges, no per-process
enumeration cost -- which matters here, because the perf-counter route
(\\Thread(python*)) enumerates every python thread on this shared server and
did not answer within two minutes.

Why this exists: a matplotlib paint that costs ~300 ms of CPU was measured
taking 2-44 s of wall time inside the app during cell alignment, and 43 s in
a bare process outside it, while ~30 logical cores sat under 50%. wall >> cpu
with idle cores means the thread is not running, and the two ways that
happens look identical from Python:

  ThreadState 1 (READY)            runnable, the scheduler just is not
                                   giving it a core -- starvation
  ThreadState 5 (Waiting) + reason blocked on something; the reason names it:
      PageIn / WrPageIn            paging -- memory pressure after all
      UserRequest / WrUserRequest  a handle or event (a lock, a filter
                                   driver, a synchronous I/O)
      WrQueue                      an I/O completion / thread-pool queue
      WrCpuRateControl             a Job Object CPU cap (!)
      WrQuantumEnd / WrPreempted   pushed off the core by the scheduler

Struct layouts are the documented x64 ones and are size-asserted at import:
a wrong size here does not raise, it reads garbage, and garbage that looks
like a wait reason is worse than an exception.
"""
import ctypes
import sys
import time
from collections import Counter
from ctypes import wintypes

STATE = {0: 'Initialized', 1: 'READY', 2: 'Running', 3: 'Standby',
         4: 'Terminated', 5: 'Waiting', 6: 'Transition', 7: 'DeferredReady'}
REASON = {0: 'Executive', 1: 'FreePage', 2: 'PageIn', 3: 'PoolAlloc',
          4: 'DelayExec', 5: 'Suspended', 6: 'UserRequest', 7: 'WrExecutive',
          8: 'WrFreePage', 9: 'WrPageIn', 10: 'WrPoolAlloc', 11: 'WrDelayExec',
          12: 'WrSuspended', 13: 'WrUserRequest', 14: 'WrEventPair',
          15: 'WrQueue', 16: 'WrLpcReceive', 17: 'WrLpcReply',
          18: 'WrVirtualMemory', 19: 'WrPageOut', 20: 'WrRendezvous',
          21: 'WrKeyedEvent', 22: 'WrTerminated', 23: 'WrProcessInSwap',
          24: 'WrCpuRateControl', 25: 'WrCalloutStack', 26: 'WrKernel',
          27: 'WrResource', 28: 'WrPushLock', 29: 'WrMutex',
          30: 'WrQuantumEnd', 31: 'WrDispatchInt', 32: 'WrPreempted',
          33: 'WrYieldExecution', 34: 'WrFastMutex', 35: 'WrGuardedMutex',
          36: 'WrRundown', 37: 'WrAlertByThreadId', 38: 'WrDeferredPreempt'}

_SystemProcessInformation = 5
_STATUS_INFO_LENGTH_MISMATCH = 0xC0000004


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [('Length', wintypes.USHORT), ('MaximumLength', wintypes.USHORT),
                ('Buffer', ctypes.c_void_p)]


class _CLIENT_ID(ctypes.Structure):
    _fields_ = [('UniqueProcess', ctypes.c_void_p), ('UniqueThread', ctypes.c_void_p)]


class _THREAD(ctypes.Structure):
    _fields_ = [('KernelTime', ctypes.c_longlong), ('UserTime', ctypes.c_longlong),
                ('CreateTime', ctypes.c_longlong), ('WaitTime', wintypes.ULONG),
                ('StartAddress', ctypes.c_void_p), ('ClientId', _CLIENT_ID),
                ('Priority', wintypes.LONG), ('BasePriority', wintypes.LONG),
                ('ContextSwitches', wintypes.ULONG), ('ThreadState', wintypes.ULONG),
                ('WaitReason', wintypes.ULONG)]


class _PROCESS(ctypes.Structure):
    _fields_ = [('NextEntryOffset', wintypes.ULONG), ('NumberOfThreads', wintypes.ULONG),
                ('WorkingSetPrivateSize', ctypes.c_longlong),
                ('HardFaultCount', wintypes.ULONG), ('NumberOfThreadsHighWatermark', wintypes.ULONG),
                ('CycleTime', ctypes.c_ulonglong), ('CreateTime', ctypes.c_longlong),
                ('UserTime', ctypes.c_longlong), ('KernelTime', ctypes.c_longlong),
                ('ImageName', _UNICODE_STRING), ('BasePriority', wintypes.LONG),
                ('UniqueProcessId', ctypes.c_void_p), ('InheritedFromUniqueProcessId', ctypes.c_void_p),
                ('HandleCount', wintypes.ULONG), ('SessionId', wintypes.ULONG),
                ('UniqueProcessKey', ctypes.c_void_p),
                ('PeakVirtualSize', ctypes.c_size_t), ('VirtualSize', ctypes.c_size_t),
                ('PageFaultCount', wintypes.ULONG),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivatePageCount', ctypes.c_size_t),
                ('ReadOperationCount', ctypes.c_longlong), ('WriteOperationCount', ctypes.c_longlong),
                ('OtherOperationCount', ctypes.c_longlong),
                ('ReadTransferCount', ctypes.c_longlong), ('WriteTransferCount', ctypes.c_longlong),
                ('OtherTransferCount', ctypes.c_longlong)]


if sys.platform == 'win32':
    assert ctypes.sizeof(_PROCESS) == 256, ctypes.sizeof(_PROCESS)
    assert ctypes.sizeof(_THREAD) == 80, ctypes.sizeof(_THREAD)
    _ntdll = ctypes.WinDLL('ntdll')
else:
    _ntdll = None


def _snapshot():
    size = 1 << 22
    while True:
        buf = ctypes.create_string_buffer(size)
        needed = wintypes.ULONG(0)
        st = _ntdll.NtQuerySystemInformation(_SystemProcessInformation, buf, size,
                                             ctypes.byref(needed))
        if st == 0:
            return buf
        if (st & 0xFFFFFFFF) == _STATUS_INFO_LENGTH_MISMATCH:
            size = max(size * 2, needed.value + (1 << 20))
            continue
        raise OSError('NtQuerySystemInformation failed: 0x%08X' % (st & 0xFFFFFFFF))


def threads_of(pid):
    """(process summary dict, [thread dict, ...]) for `pid`, or (None, [])."""
    if _ntdll is None:
        return None, []
    buf = _snapshot()
    base = ctypes.addressof(buf)
    off = 0
    while True:
        p = _PROCESS.from_address(base + off)
        if (p.UniqueProcessId or 0) == pid:
            tb = base + off + ctypes.sizeof(_PROCESS)
            ths = []
            for k in range(p.NumberOfThreads):
                t = _THREAD.from_address(tb + k * ctypes.sizeof(_THREAD))
                ths.append(dict(tid=t.ClientId.UniqueThread or 0,
                                state=t.ThreadState, reason=t.WaitReason,
                                ctxsw=t.ContextSwitches, prio=t.Priority,
                                base=t.BasePriority,
                                cpu_s=(t.KernelTime + t.UserTime) / 1e7))
            return dict(pid=pid, threads=p.NumberOfThreads, base_prio=p.BasePriority,
                        session=p.SessionId, hard_faults=p.HardFaultCount,
                        page_faults=p.PageFaultCount), ths
        if p.NextEntryOffset == 0:
            return None, []
        off += p.NextEntryOffset


def describe(t):
    """One thread, one line."""
    return ('state=%-13s reason=%-16s prio=%d/%d ctxsw=%d cpu=%.2fs'
            % (STATE.get(t['state'], t['state']), REASON.get(t['reason'], t['reason']),
               t['prio'], t['base'], t['ctxsw'], t['cpu_s']))


def busiest(ths):
    """The thread with the most accumulated CPU -- the GUI's main thread, in
    a Qt app, since it has been drawing since launch."""
    return max(ths, key=lambda t: t['cpu_s']) if ths else None


def sample(pid, seconds=30.0, interval=0.25, tid=None, out=print):
    """Watch one thread and report how it spent the window."""
    spi, ths = threads_of(pid)
    if spi is None:
        raise SystemExit(f'pid {pid} not found')
    target = tid or busiest(ths)['tid']
    out(f'pid {pid}: {spi["threads"]} threads, base prio {spi["base_prio"]}, '
        f'hard faults {spi["hard_faults"]}; watching tid {target} for {seconds:.0f}s')
    states, reasons = Counter(), Counter()
    first = last = None
    hf0 = spi['hard_faults']
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        spi, ths = threads_of(pid)
        me = next((t for t in ths if t['tid'] == target), None) if spi else None
        if me is None:
            out('thread gone')
            break
        first = first or me
        last = me
        n += 1
        states[STATE.get(me['state'], me['state'])] += 1
        if me['state'] == 5:
            reasons[REASON.get(me['reason'], me['reason'])] += 1
        if n <= 4 or n % 40 == 0:
            out(f'  t={time.perf_counter()-t0:5.1f}s  {describe(me)}')
        time.sleep(interval)
    wall = time.perf_counter() - t0
    # The process may already be gone -- a watched run that finishes is the
    # normal ending, not an error, and the samples taken before it are the
    # whole point. Reporting must not raise on the way out.
    hf_now = spi['hard_faults'] if spi else hf0
    out(f'\n{n} samples over {wall:.1f}s -- state distribution:')
    for k, v in states.most_common():
        out(f'  {k:<14} {v:4d}  ({100.0*v/max(n,1):5.1f}%)')
    if reasons:
        out('when Waiting, the reason:')
        for k, v in reasons.most_common():
            out(f'  {k:<18} {v:4d}')
    if first and last:
        cpu = last['cpu_s'] - first['cpu_s']
        out(f'thread CPU in window: {cpu:.2f}s ({100.0*cpu/max(wall,1e-9):.1f}% of wall); '
            f'context switches {last["ctxsw"]-first["ctxsw"]} ({(last["ctxsw"]-first["ctxsw"])/max(wall,1e-9):.0f}/s); '
            f'process hard faults +{hf_now-hf0}')
    ready = states.get('READY', 0) / max(n, 1)
    waiting = states.get('Waiting', 0) / max(n, 1)
    if ready > 0.3:
        out('-> READY dominates: runnable but denied a core. STARVATION.')
    elif waiting > 0.5:
        top = reasons.most_common(1)[0][0] if reasons else '?'
        out(f'-> Waiting dominates ({top}): BLOCKED, not computing.')
    else:
        out('-> Running dominates: it is actually computing.')
    return states, reasons


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: thread_state.py <pid> [seconds] [interval] [tid]')
    sample(int(sys.argv[1]),
           float(sys.argv[2]) if len(sys.argv) > 2 else 30.0,
           float(sys.argv[3]) if len(sys.argv) > 3 else 0.25,
           int(sys.argv[4]) if len(sys.argv) > 4 else None)
