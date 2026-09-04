"""
The two knobs the disk-contention question turns on, in one place.

WHAT IS ACTUALLY KNOWN, measured on the live app during a real alignment:

  * The GUI was not computing and was not short of memory. In 31 s of wall
    time it used 0.89 s of CPU (35x starved) with 157 GB free and 123 page
    faults/s.
  * The device was saturated: 386 MB/s and 8274 ops/s at ~48 KB an op.
  * A 2.5 MB MIP the GUI wanted took 2.2-3.4 s to read, and re-reading the
    same file was just as slow -- so the cost is queue position, not bytes,
    and the OS file cache cannot buy it back.

WHAT IS NOT KNOWN: whether lowering the workers' I/O priority fixes it.
Three external A/Bs failed to answer it (see windows/run_probe.py for how
each one broke). The remaining honest test is to vary the setting inside a
real run and read the numbers off the app's own log.

So there are two knobs, not one, because the evidence points two ways:

  cell_alignment_workers -- queue LENGTH. This repo has already measured
    that more readers means less throughput on this storage: ingestion did
    117.6 MB/s at 12 workers and 66 MB/s at 36. If that curve holds for the
    cell pool, fewer workers helps the GUI *and* the alignment, with no
    trade-off to make.
  child_io_priority -- queue ORDER. Lets the GUI's small reads overtake the
    workers' 245 MB streams. If it works it costs the alignment something,
    so it has to beat the worker cap on the numbers to be worth keeping.

Both are read from a JSON file, NOT baked in at launch, so all four
combinations can be run in one session without restarting the app:

    {"cell_alignment_workers": 8, "child_io_priority": "verylow"}

written to <repo>/tuning.json (or wherever CODELAB_TUNING_FILE points).
The file is re-read whenever it changes, so editing it between runs is
enough. An absent file, an absent key, or an unparseable value all mean
"use the measured default" -- a typo must not silently reconfigure a run,
which is why every resolved value carries where it came from and the app
logs that with the result.
"""
import json
import os

# <repo>/codelab_pipeline/tuning.py -> <repo>
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TUNING_FILE_ENV = 'CODELAB_TUNING_FILE'
WORKERS_ENV = 'CODELAB_CELL_ALIGN_WORKERS'
IO_PRIORITY_ENV = 'CODELAB_CHILD_IO_PRIORITY'

# The PIN is a different variable from the plain setting on purpose, and the
# distinction is load-bearing rather than cosmetic.
#
# A child cannot tell "the parent resolved this for the run I belong to"
# apart from "someone's launcher exported this months ago" if both arrive
# under one name. Collapsing them puts the two halves of this module in
# direct contradiction: _resolve() deliberately reads the FILE before the
# environment so that a stale exported variable cannot silently kill a
# mid-session edit -- and a child that read that same variable first would
# reinstate exactly the failure the ordering exists to prevent.
#
# So: the pin below is written ONLY by apply_child_env(), immediately before
# a pool is built, and carries an already-resolved and already-validated
# value. Anything else in the environment is just another opinion, ranked
# under the file.
IO_PRIORITY_PIN_ENV = 'CODELAB_CHILD_IO_PRIORITY_PINNED'

# One pool child's share of the TOTAL MIP cache budget, pinned by the
# parent before the pool is built. Separate from CODELAB_MIP_CACHE_GB,
# which every process (the GUI included) reads for its own cache: the
# parent must be able to size its CHILDREN without resizing itself.
MIP_CACHE_PIN_ENV = 'CODELAB_CHILD_MIP_CACHE_GB_PINNED'

IO_PRIORITY_CHOICES = ('verylow', 'low', 'normal')
DEFAULT_IO_PRIORITY = 'verylow'

# (path, mtime, size) -> parsed dict. Keyed by the stat rather than the path
# so an edit is picked up mid-session, and cached rather than re-read so
# that consulting this per pool creation stays free.
_CACHE = {}


def tuning_path():
    """Where the knobs live. CODELAB_TUNING_FILE wins if set."""
    return os.environ.get(TUNING_FILE_ENV) or os.path.join(_REPO, 'tuning.json')


def _load():
    """The file's contents, or {} if it is absent or unreadable.

    Never raises. A malformed tuning file must not stop an alignment from
    starting -- it falls back to the measured defaults, and the caller
    reports the source so the fallback is visible rather than silent.
    """
    path = tuning_path()
    try:
        st = os.stat(path)
    except OSError:
        return {}
    key = (path, st.st_mtime, st.st_size)
    if key in _CACHE:
        return _CACHE[key]
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    _CACHE.clear()               # one entry: only the current file matters
    _CACHE[key] = data
    return data


def _resolve(key, env_var):
    """(raw value, source) for one knob.

    The FILE wins over the environment on purpose. The file is the live
    knob -- the one edited between runs -- and an environment variable
    inherited from a launcher would otherwise pin the value and make those
    edits do nothing, silently, which is the exact failure mode this module
    exists to avoid.
    """
    data = _load()
    if key in data and data[key] is not None:
        return data[key], os.path.basename(tuning_path())
    if os.environ.get(env_var):
        return os.environ[env_var], env_var
    return None, 'default'


def cell_alignment_workers():
    """(workers, source): how many cells to fit at once, or None for the
    measured default in alignment.max_cell_alignment_workers().

    0 and negative values mean "default" rather than "no workers", so a
    zeroed-out file cannot produce a run that never starts.
    """
    raw, source = _resolve('cell_alignment_workers', WORKERS_ENV)
    if raw is None:
        return None, 'default'
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None, f'default (ignored unparseable {raw!r} from {source})'
    if n < 1:
        return None, f'default (ignored {n} from {source})'
    return n, source


def child_io_priority():
    """(level, source): the disk-queue priority every pool child adopts."""
    raw, source = _resolve('child_io_priority', IO_PRIORITY_ENV)
    if raw is None:
        return DEFAULT_IO_PRIORITY, 'default'
    level = str(raw).strip().lower()
    if level not in IO_PRIORITY_CHOICES:
        return DEFAULT_IO_PRIORITY, f'default (ignored {raw!r} from {source})'
    return level, source


def settings_label(workers_in_effect=None):
    """The one-line stamp that makes a run's numbers attributable.

    `workers_in_effect` is what the pool was ACTUALLY built with -- which
    can be lower than the setting, since a pool is capped by the number of
    tasks. Reporting the request instead of the reality would make a
    4-cell run look like a 16-worker condition.
    """
    n, n_src = cell_alignment_workers()
    io, io_src = child_io_priority()
    shown = workers_in_effect if workers_in_effect is not None else n
    if shown is None:
        shown = 'auto'
    return f'workers={shown} ({n_src}) io={io} ({io_src})'


def apply_child_env(n_children=None):
    """Freeze the settings for the children about to be spawned.

    `n_children` sizes each child's MIP cache out of one TOTAL budget --
    see process_guard.child_mip_cache_gb. Only the parent knows the pool
    size, and the per-child cache is retained memory, so a pool that sets
    it per child with no total is what drained this machine's free page
    list from 7.3 GB to 0.03 GB and slowed every process on it. Returns
    (io level, cache GB, human explanation of the cache share).

    Freeze the resolved I/O priority for the children about to be spawned.

    Under 'spawn' a child inherits the environment AS IT IS AT SPAWN TIME --
    the same reason parallel.py sets its BLAS variables in the parent before
    building a pool rather than inside the child's initializer. Writing the
    pin here, once, before the pool exists means one value governs the whole
    run: an edit to the tuning file halfway through cannot leave some
    children on one setting and the rest on the other, which would produce
    an A/B arm that is neither arm.

    Returns the level applied. Nothing is restored afterwards -- the next
    run re-derives it from the file regardless, and leaving the pin in place
    keeps a late-spawned child consistent with the run it belongs to.
    """
    level, _ = child_io_priority()
    os.environ[IO_PRIORITY_PIN_ENV] = level
    from . import process_guard
    cache_gb, note = process_guard.child_mip_cache_gb(n_children)
    os.environ[MIP_CACHE_PIN_ENV] = f'{cache_gb:g}'
    return level, cache_gb, note
