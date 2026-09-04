"""
Make worker processes die with the app, however the app dies.

The failure this exists for is confirmed and was measured: after a
force-kill (Task Manager, a crash, anything that is not the app's own
Quit), 97 ProcessPoolExecutor workers from three separate runs were
still alive 2.7 DAYS later, holding 5.0 GB of RAM and file locks on the
conda environment -- locks that made `conda env remove` impossible until
they were hunted down by hand.

The app's own `_kill_running_work` handles the orderly case. It cannot
help with the disorderly one: a terminated parent runs no code. The
guard therefore has to come from the OS.

Two mechanisms, because no single one is portable:

  Windows -- a JOB OBJECT with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. The
    APP process is assigned to the job at startup, and children inherit
    job membership automatically, so every descendant is covered without
    the pools knowing anything about it. When the last handle to the job
    closes -- which the kernel does when the app process dies, by any
    means including TerminateProcess -- the kernel kills everything in
    the job. This is the only Windows mechanism that survives a kill of
    the parent, which is exactly the case that bit us.

  POSIX -- per-child, installed by the pool initializer:
      Linux: prctl(PR_SET_PDEATHSIG, SIGKILL), the kernel's own
        "signal me when my parent dies".
      macOS: no such call, so a tiny daemon thread watches getppid();
        when the parent is gone the child is reparented (to 1 or to
        launchd) and the thread os._exit()s it.

Both POSIX paths are set up in the CHILD, so they work with the 'spawn'
start method this app uses everywhere.

Nothing here changes behaviour when the app exits normally -- the pools
are already shut down by then. It only closes the window where a
killed parent leaves children behind.
"""
import os
import signal
import sys
import threading

_WINDOWS = sys.platform == 'win32'
_LINUX = sys.platform.startswith('linux')

# Kept at module scope for the life of the process ON PURPOSE: the job is
# destroyed (and its members killed) when its last handle closes, so the
# handle must outlive everything it protects.
_JOB_HANDLE = None
_JOB_STATE = 'not installed'


def install_parent_guard():
    """
    Put THIS process (and therefore every process it later spawns) under
    an OS-level guarantee that its children cannot outlive it.

    Windows only -- on POSIX the guarantee is installed per child (see
    child_guard). Safe to call more than once; safe to call when the
    mechanism is unavailable, in which case it reports why and the app
    runs exactly as before.

    Returns a short human-readable status string, which the caller is
    expected to log: a guard that silently failed to install is worse
    than no guard, because nobody goes looking for orphans.
    """
    global _JOB_HANDLE, _JOB_STATE
    if not _WINDOWS:
        _JOB_STATE = 'not needed (POSIX: children guard themselves, see child_guard)'
        return _JOB_STATE
    if _JOB_HANDLE is not None:
        return _JOB_STATE
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [('ReadOperationCount', ctypes.c_ulonglong),
                        ('WriteOperationCount', ctypes.c_ulonglong),
                        ('OtherOperationCount', ctypes.c_ulonglong),
                        ('ReadTransferCount', ctypes.c_ulonglong),
                        ('WriteTransferCount', ctypes.c_ulonglong),
                        ('OtherTransferCount', ctypes.c_ulonglong)]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', wintypes.LARGE_INTEGER),
                        ('PerJobUserTimeLimit', wintypes.LARGE_INTEGER),
                        ('LimitFlags', wintypes.DWORD),
                        ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t),
                        ('ActiveProcessLimit', wintypes.DWORD),
                        # ULONG_PTR, i.e. pointer-sized -- c_size_t. Getting
                        # this wrong silently shifts every field after it.
                        ('Affinity', ctypes.c_size_t),
                        ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ('IoInfo', IO_COUNTERS),
                        ('ProcessMemoryLimit', ctypes.c_size_t),
                        ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t),
                        ('PeakJobMemoryUsed', ctypes.c_size_t)]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        # argtypes/restype are REQUIRED, not decoration: GetCurrentProcess
        # returns the pseudo-handle (HANDLE)-1, and without an explicit
        # HANDLE signature ctypes tries to marshal it as a C int and
        # raises "int too long to convert" -- which is exactly how this
        # guard silently failed to install the first time.
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())

        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            # Already in a job that forbids nesting (some CI/session
            # managers do this). Not fatal -- say so and carry on.
            raise ctypes.WinError(ctypes.get_last_error())

        _JOB_HANDLE = job
        _JOB_STATE = 'installed (Windows job object: children die with this process)'
    except Exception as exc:                      # never block startup over this
        _JOB_STATE = f'UNAVAILABLE ({type(exc).__name__}: {exc}) -- workers may outlive a force-kill'
    return _JOB_STATE


def guard_state():
    """What install_parent_guard last concluded (for logs/tests)."""
    return _JOB_STATE


def _watch_parent(original_ppid, poll_seconds=2.0):
    """macOS/other POSIX: exit as soon as we are reparented away from the
    process that started us. os._exit, not sys.exit -- this runs on a
    daemon thread and must not be swallowed by the interpreter's normal
    shutdown machinery."""
    while True:
        try:
            if os.getppid() != original_ppid:
                os._exit(1)
        except Exception:
            os._exit(1)
        threading.Event().wait(poll_seconds)


def child_guard():
    """
    Called IN a worker process: arrange for it to die when its parent
    does. No-op on Windows, where the job object already covers it.
    """
    if _WINDOWS:
        return
    if _LINUX:
        try:
            import ctypes
            PR_SET_PDEATHSIG = 1
            libc = ctypes.CDLL('libc.so.6', use_errno=True)
            if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) == 0:
                # A parent that died between fork and here means the
                # signal was already missed -- check once, explicitly.
                if os.getppid() == 1:
                    os._exit(1)
                return
        except Exception:
            pass                                   # fall through to the poller
    threading.Thread(target=_watch_parent, args=(os.getppid(),),
                     daemon=True, name='parent-death-watch').start()


# One pool child's MIP cache budget, in GB.
#
# analysis_store sizes that cache as a fraction of available RAM, clamped
# to at most 4 GB. That is right for the ONE GUI process it was written
# against and wrong by a factor of the pool size in here: 42 children x
# 4 GB is 168 GB of nominal budget on a 352 GB machine, and the paging it
# invites hurts far more than the cache helps. Measured while an
# alignment run had the machine at 5000 pages/s: a GUI redraw that used
# 438 ms of CPU took 15144 ms of wall time -- 34.6x starved, waiting on
# page faults, not computing.
#
# A child needs exactly ONE FOV's MIPs resident: 319 MB for 76 hybes,
# ~461 MB for 110. The benefit is a CLIFF at that working set, not a
# gradient -- measured per-cell sweep over a real FOV: 4343 ms at
# 0.125 GB, 4061 ms at 0.25 GB, then 3.6 ms at 0.5 GB, 3.0 ms at 1 GB and
# 3.3 ms at 4 GB. So 1 GB keeps the entire win, with margin for a bigger
# FOV, at a quarter of the memory.
#
# The stack-slab cache is deliberately NOT capped here: one slab is a
# ~134 MB inflation and a worker sweeps every hybe of its FOV, so its
# large budget is earned, and shrinking it blindly would cost real work.
# The lever there is the worker COUNT, not the per-worker budget.
CHILD_MIP_CACHE_GB = 1.0

# ...but per-child is only half a budget, and the missing half was measured
# to hurt the WHOLE MACHINE.
#
# Windows serves a new allocation from its free+zero page list -- pages
# already reclaimed and zeroed, ready to hand out. On this box that list
# holds ~7.3 GB while "available memory" reads 168 GB, because the other
# 167 GB is standby file cache that must be reclaimed and zeroed first,
# under a lock. Drain the list and every process on the machine queues on
# that refill path, in kernel state Waiting/Executive, consuming no CPU
# and no disk.
#
# A cache is the worst shape of demand there is: it takes pages and does
# not give them back. Measured, 6 children reading MIPs:
#
#     per-child cache   free+zero list      reads done
#         1.0 GB          -5.89 GB             2512
#         0.5 GB          -3.35 GB             2933
#         0   (off)       -0.20 GB             3089
#
# The drain tracks children x budget almost exactly. At the default 16
# workers, 1.0 GB each is 16 GB of retention against a 7.3 GB list -- and
# the observed consequence was a GUI paint costing 469 ms of CPU taking
# 23 s of wall, while an unrelated benchmark process on the same machine
# dropped from 100% to 21% CPU.
#
# Note what is NOT the problem: churn. The same probe allocating and
# freeing 945,959 times a second left the list at 7 GB, because freed
# pages are recycled inside the process. Only retention reaches the
# kernel.
#
# So the budget is TOTAL, and the per-child share is derived from it.
TOTAL_MIP_CACHE_GB = 4.0

# ...but not below the cliff. Measured per-cell sweep over a real FOV:
# 4343 ms at 0.125 GB, 4061 ms at 0.25 GB, then 3.6 ms at 0.5 GB. A child
# needs ONE FOV's MIPs resident (319 MB for 76 hybes, ~461 MB for 110) and
# falling under that costs three orders of magnitude. When the floor and
# the total disagree the floor wins and the caller is TOLD, because the
# honest answer at that point is "use fewer workers", not "make every
# worker slow".
MIN_CHILD_MIP_CACHE_GB = 0.5


def child_mip_cache_gb(n_children=None):
    """(GB per child, one-line explanation) for a pool of `n_children`.

    n_children=None means the caller does not know its own pool size; it
    gets the old flat per-child number, which is what the code did before
    a total existed.
    """
    if not n_children or n_children < 1:
        return CHILD_MIP_CACHE_GB, f'{CHILD_MIP_CACHE_GB:.2f} GB (pool size unknown)'
    share = TOTAL_MIP_CACHE_GB / float(n_children)
    if share >= MIN_CHILD_MIP_CACHE_GB:
        return share, (f'{share:.2f} GB each = {TOTAL_MIP_CACHE_GB:.1f} GB '
                       f'total / {n_children} worker(s)')
    over = MIN_CHILD_MIP_CACHE_GB * n_children
    return MIN_CHILD_MIP_CACHE_GB, (
        f'{MIN_CHILD_MIP_CACHE_GB:.2f} GB each (the floor) x {n_children} '
        f'worker(s) = {over:.1f} GB, OVER the {TOTAL_MIP_CACHE_GB:.1f} GB '
        f'budget -- {int(TOTAL_MIP_CACHE_GB // MIN_CHILD_MIP_CACHE_GB)} '
        f'worker(s) would fit it; expect the machine to slow while this runs')


def child_initializer(user_initializer=None, *user_args):
    """
    THE initializer for every ProcessPoolExecutor in this app: installs
    the child's own parent-death guard, sizes this child's caches for
    being one of many, then runs whatever initializer the pool wanted.

    A module-level function taking the user's initializer as an argument,
    so `functools.partial(child_initializer, real_init)` stays picklable
    under the 'spawn' start method -- a closure or a lambda would not.
    """
    child_guard()
    # The parent pins this child's share of the TOTAL cache budget before
    # the pool exists (tuning.apply_child_env), because only the parent
    # knows how many children there will be. A separate variable from
    # CODELAB_MIP_CACHE_GB on purpose: setting that one in the parent
    # would also shrink the GUI's OWN cache, which is not a child and not
    # part of the pool's budget.
    from . import tuning
    pinned = os.environ.get(tuning.MIP_CACHE_PIN_ENV)
    if pinned:
        os.environ['CODELAB_MIP_CACHE_GB'] = pinned
    else:
        # setdefault, so an explicit CODELAB_MIP_CACHE_GB in the environment
        # (inherited from the parent) still wins -- including the 0 that
        # turns the cache off.
        os.environ.setdefault('CODELAB_MIP_CACHE_GB', str(CHILD_MIP_CACHE_GB))
    lower_io_priority()
    if user_initializer is not None:
        user_initializer(*user_args)


def lower_io_priority():
    """Put this pool child at the back of the DISK queue.

    MEASURED, on the live app while a cell alignment ran and the window
    was Not Responding: the GUI process spent 0.89 s of CPU in 31 s
    (35x starved) with 157 GB of RAM free and 123 page faults/s -- so it
    was neither computing nor short of memory. It was, however, issuing
    ~67 reads/s of ~9 KB and getting 0.6 MB/s: roughly 15 ms per read,
    where an idle local disk answers a 9 KB read in well under 1 ms. The
    GUI's small HDF5 chunk reads were queued behind the workers' 245 MB
    streams.

    An earlier diagnosis blamed memory pressure and the per-child MIP
    cache was capped for it. That cap did remove the paging -- and the
    stall survived it. Disk QUEUE POSITION is what is left.

    I/O priority ONLY, deliberately not PROCESS_MODE_BACKGROUND_BEGIN:
    that would also drop the child to IDLE cpu priority, and CPU is not
    the contended resource here (64 cores, the GUI could not even use
    2%). Lowering what is not contended would slow the fits for nothing.

    WHETHER THIS ACTUALLY HELPS IS STILL UNMEASURED. Three attempts to A/B
    it from outside the app each answered a different question than the one
    asked (windows/run_probe.py records how each broke), and the last one
    tied at 1.04x only because the load came from a build that predated this
    function. So it is a switch, defaulting on, to be judged on numbers from
    a real run -- not a fix to be assumed.

    Best effort and silent: a platform without I/O priority, or a psutil
    that cannot set it, simply keeps the default.

    The level comes from the PIN that tuning.apply_child_env() writes just
    before a pool is built, which fixes one value for the whole run. Absent
    a pin the child resolves the setting itself, so pools that were never
    taught about any of this still honour the tuning file. Either way the
    value is validated by tuning, never taken raw from the environment: a
    typo must fall back to the default loudly enough to be logged, not
    quietly mean "off". 'verylow' (default), 'low', or 'normal' to disable.
    """
    from . import tuning
    pinned = (os.environ.get(tuning.IO_PRIORITY_PIN_ENV) or '').strip().lower()
    if pinned in tuning.IO_PRIORITY_CHOICES:
        level = pinned
    else:
        level, _source = tuning.child_io_priority()
    if level == 'normal':
        return False
    try:
        import psutil
        wanted = {'verylow': getattr(psutil, 'IOPRIO_VERYLOW', None),
                  'low': getattr(psutil, 'IOPRIO_LOW', None)}.get(level)
        if wanted is None:
            return False
        psutil.Process().ionice(wanted)
        return True
    except Exception:
        return False
