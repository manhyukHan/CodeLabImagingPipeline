"""
In-memory FIFO cache of DECOMPRESSED z-slabs from raw stack files.

The problem it solves, measured on the real NAS store
(G:/JP/2026-01-26-cross_modal, 1024x1024x177 stacks):

    read_projection, ONE z-plane   4.6 s cold / 2.3 s warm-ish
    focus_profile, 177 planes     24.7 s

Both are dominated by decompression, not by the network. Stacks are
written with chunks=(32, 32, 64) + gzip, so pulling a single full-frame
z-plane touches 32x32 = 1024 chunks and inflates ~134 MB to hand back a
2 MB plane. Scrolling the Z-plane spinbox therefore froze the GUI for
seconds PER TICK, and focus_profile re-inflated the same chunks once per
plane.

The fix is to keep what was already paid for: a read of one plane
inflates that plane's whole CHUNK SLAB anyway, so this module reads the
full slab, caches it, and serves every other plane in it for free. On
the real store that turns

    plane scroll : 2.3 s per tick        -> 2.3 s once per 64-plane slab
    focus_profile: 177 slab inflations   -> ceil(177/64) = 3

Eviction is FIFO (first in, first out) under a byte budget, not LRU:
the access pattern here is a sweep (scroll a plane range, profile a
stack, then move to another hybe), so the oldest entry really is the
one least likely to be wanted again, and FIFO needs no per-hit
bookkeeping under the lock.

The budget is MEASURED, never hardcoded to one machine: a fraction of
the physical RAM actually available when first asked, clamped to a
sane floor and ceiling (see _budget_bytes). A workstation with plenty
of RAM caches many slabs; a small laptop caches a couple; neither
needs configuring. CODELAB_STACK_CACHE_GB overrides it explicitly
(0 disables caching entirely).

Thread-safe: the GUI thread and every worker read through here at once.
The lock guards only the dict; the HDF5 read itself runs outside it, so
a slow NAS read never blocks other threads' cache hits (two threads
racing the same cold slab simply both read it, and the second store
wins harmlessly).
"""
import collections
import os
import threading

import h5py
import numpy as np

# Fraction of AVAILABLE physical memory to spend on cached slabs, and the
# range that fraction is clamped into. The floor keeps the cache useful
# on a small machine (2 GB still holds ~15 slabs of a 1024x1024x64
# stack); the ceiling stops a huge-RAM box from letting one cache grow
# without bound when nothing else is competing for it.
_BUDGET_FRACTION = 0.25
_BUDGET_MIN_GB = 2.0
_BUDGET_MAX_GB = 32.0


def _available_ram_bytes():
    """Physical memory available right now, or None if it cannot be
    determined on this platform."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if os.name == 'nt':
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong),
                            ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass
    else:
        try:
            return int(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        except (ValueError, OSError, AttributeError):
            pass
    return None


_BUDGET = None


def enabled():
    """
    False when the cache is switched off (CODELAB_STACK_CACHE_GB=0).

    Callers whose natural read is SMALLER than a slab must branch on
    this and read directly instead: slab() always inflates the full
    frame's chunks, which is a win only because the slab is then reused.
    With retention off it would be pure overhead -- measured 4x slower
    than a direct central-crop read in focus_profile.
    """
    return _budget_bytes() > 0


def _budget_bytes():
    """
    Bytes this cache may hold. An explicit CODELAB_STACK_CACHE_GB wins
    (and is re-read every call, so it can be changed live); otherwise a
    clamped fraction of available RAM, measured ONCE and reused -- the
    budget must not shrink mid-session just because this cache itself
    has consumed the memory it is being measured against.
    """
    global _BUDGET
    env = os.environ.get('CODELAB_STACK_CACHE_GB')
    if env is not None:
        try:
            return int(max(float(env), 0.0) * (1 << 30))
        except ValueError:
            pass
    if _BUDGET is None:
        available = _available_ram_bytes()
        if available is None:
            gb = _BUDGET_MIN_GB
        else:
            gb = (available * _BUDGET_FRACTION) / (1 << 30)
        _BUDGET = int(min(max(gb, _BUDGET_MIN_GB), _BUDGET_MAX_GB) * (1 << 30))
    return _BUDGET


_LOCK = threading.Lock()
_CACHE = collections.OrderedDict()   # {(path, channel, slab_index): ndarray}
_BYTES = 0
_SHAPES = {}                         # {(path, channel): (h, w, depth, slab)}

# counters, for measurement/reporting (tools and tests read these)
STATS = {'hits': 0, 'misses': 0, 'evictions': 0, 'bytes': 0}


def _key(h5path, channel, slab_index):
    return (os.path.abspath(h5path), str(channel), int(slab_index))


def _mtime_ns(h5path):
    """The backing file's mtime, or None if it cannot be stat'd."""
    try:
        return os.stat(h5path).st_mtime_ns
    except OSError:
        return None


def clear():
    """Drop everything (a store swap, or a memory-pressure escape hatch)."""
    global _BYTES, _BUDGET
    with _LOCK:
        _CACHE.clear()
        _SHAPES.clear()
        _BYTES = 0
        _BUDGET = None      # re-measure available RAM on the next read
        STATS.update(hits=0, misses=0, evictions=0, bytes=0)


def stack_shape(h5path, channel):
    """
    (height, width, depth, slab_depth) for one stack's channel, or None
    if unreadable. slab_depth is the dataset's OWN chunk z-extent, so a
    cached slab is exactly the unit HDF5 already inflates -- caching any
    other span would either waste work or re-inflate the same chunks.
    """
    k = (os.path.abspath(h5path), str(channel))
    mtime = _mtime_ns(h5path)
    with _LOCK:
        hit = _SHAPES.get(k)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        with h5py.File(h5path, 'r') as f:
            name = f'/stack/ch{channel}'
            if name not in f:
                return None
            ds = f[name]
            h, w, depth = ds.shape
            slab = int(ds.chunks[2]) if ds.chunks else min(64, depth)
            shape = (int(h), int(w), int(depth), max(slab, 1))
    except (OSError, KeyError):
        return None
    with _LOCK:
        _SHAPES[k] = (mtime, shape)
    return shape


def _store(key, mtime, arr):
    """
    Admit one slab, then evict oldest-first until the total fits the
    budget. FIFO by INSERTION order (OrderedDict.popitem(last=False)),
    so the entry dropped is the one read longest ago.

    Budget 0 means the cache is off: nothing is retained at all (the
    read still returned its data -- correctness never depends on this
    cache). Otherwise the slab just read is always kept, even if it
    alone exceeds a very small budget: evicting what the caller is
    about to use would make the cache pure overhead.
    """
    global _BYTES
    budget = _budget_bytes()
    if budget <= 0:
        return
    with _LOCK:
        old = _CACHE.pop(key, None)
        if old is not None:
            _BYTES -= old[1].nbytes
        _CACHE[key] = (mtime, arr)
        _BYTES += arr.nbytes
        while _BYTES > budget and len(_CACHE) > 1:
            _, evicted = _CACHE.popitem(last=False)   # FIFO
            _BYTES -= evicted[1].nbytes
            STATS['evictions'] += 1
        STATS['bytes'] = _BYTES


def slab(h5path, channel, z):
    """
    (slab_array, z0) covering plane `z` -- slab_array is
    (height, width, slab_depth_actual) and z0 is its first plane index,
    so the caller indexes `slab_array[..., z - z0]`. None if unreadable.

    The array is returned as-is (not copied): callers must treat it as
    READ-ONLY, since it is the cached object itself. Every consumer here
    either slices a fresh array out of it or reduces it.
    """
    shape = stack_shape(h5path, channel)
    if shape is None:
        return None
    _h, _w, depth, slab_depth = shape
    z = int(np.clip(z, 0, depth - 1))
    idx = z // slab_depth
    key = _key(h5path, channel, idx)
    # mtime-validated, exactly like the analysis store's read cache: a
    # hybe RE-INGESTED with overwrite replaces this file, and a cache
    # keyed on path alone would then serve the old pixels forever. Any
    # write -- this process or another machine -- bumps mtime and
    # invalidates naturally. One stat per read, sub-millisecond against
    # the seconds this cache saves.
    mtime = _mtime_ns(h5path)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == mtime:
            STATS['hits'] += 1
            return hit[1], idx * slab_depth
        STATS['misses'] += 1
    z0 = idx * slab_depth
    z1 = min(z0 + slab_depth, depth)
    try:
        with h5py.File(h5path, 'r') as f:
            arr = f[f'/stack/ch{channel}'][:, :, z0:z1]
    except (OSError, KeyError):
        return None
    arr.flags.writeable = False
    _store(key, mtime, arr)
    return arr, z0


def plane(h5path, channel, z, dtype=np.float32):
    """One full-frame z-plane as a fresh array, or None."""
    got = slab(h5path, channel, z)
    if got is None:
        return None
    arr, z0 = got
    local = int(np.clip(z - z0, 0, arr.shape[2] - 1))
    return arr[:, :, local].astype(dtype)


def planes(h5path, channel, z0, z1, dtype=np.float32):
    """
    The inclusive [z0, z1] span as one (h, w, n) array, assembled from
    however many cached slabs it crosses. None if unreadable.
    """
    shape = stack_shape(h5path, channel)
    if shape is None:
        return None
    _h, _w, depth, slab_depth = shape
    z0 = int(np.clip(min(z0, z1), 0, depth - 1))
    z1 = int(np.clip(max(z0, z1), 0, depth - 1))
    parts = []
    z = z0
    while z <= z1:
        got = slab(h5path, channel, z)
        if got is None:
            return None
        arr, s0 = got
        lo = z - s0
        hi = min(z1 - s0, arr.shape[2] - 1)
        parts.append(arr[:, :, lo:hi + 1])
        z = s0 + hi + 1
    return np.concatenate(parts, axis=2).astype(dtype) if parts else None
