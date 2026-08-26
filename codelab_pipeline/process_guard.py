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


def child_initializer(user_initializer=None, *user_args):
    """
    THE initializer for every ProcessPoolExecutor in this app: installs
    the child's own parent-death guard, then runs whatever initializer
    the pool actually wanted.

    A module-level function taking the user's initializer as an argument,
    so `functools.partial(child_initializer, real_init)` stays picklable
    under the 'spawn' start method -- a closure or a lambda would not.
    """
    child_guard()
    if user_initializer is not None:
        user_initializer(*user_args)
