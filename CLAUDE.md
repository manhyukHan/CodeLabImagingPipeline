# Working on this repo

## Validate against the real dataset, not the repo's `data/`

**The checked-in `data/` fixtures are not ground truth.** They are a partial
extract kept for local validation on a Mac. They are small, they are not
representative, and on a Windows machine they are frequently absent entirely --
`tests/test_alignment_ground_truth.py` cannot run here at all for exactly that
reason.

The real case is the one that matters:

| | repo `data/` | the real case |
|---|---|---|
| platform | macOS, local | **Windows** |
| size | partial extract | **huge** |
| storage | local disk | **NAS** |

So:

- **A conclusion measured on the real store outranks one measured on `data/`.**
  If they disagree, the real store is right and the fixture is unrepresentative.
- **A green test suite is not evidence that a change is correct.** Several
  suites here fail or skip purely because their fixture is missing on this
  machine. Read *why* a test failed before treating the count as a signal.
- **Never tune a default, a threshold, or a bound to what `data/` shows.**
  Thresholds derived from the extract have been wrong in practice -- see the
  cell-alignment Z bound, which sat at 5 planes while real inter-round drift
  runs to 11.
- **Performance work must be measured on NAS-and-Windows-shaped inputs.**
  Local-SSD timings understate I/O and overstate CPU share. A 1024x1024 MIP is
  ~2 MB; a real stack file is ~278 MB; a real FOV carries ~111 hybes.
- When the real store cannot be reached, say so explicitly rather than
  silently substituting the fixture, and label the result as unvalidated.

Real store used for most measurement in this work:
`E:/Students/2026-08-07-SG-test/DNA` (v2 layout, 34 FOVs, 111 hybes/FOV,
1024x1024 MIPs, 120 z-planes).

## Measure before optimizing, and report what you measured

Plausible-sounding optimizations here have a track record of being wrong:

- Dropping Powell's free angle parameter looked like pure waste (Powell cannot
  recover rotation, and the 0.5 deg gate discards its answer). Measured, it was
  *slower* and produced *worse* fits -- the parameter acts as a search
  direction, not an estimate.
- More ingestion workers looked faster. Measured: 36 workers gave 66 MB/s,
  12 workers gave 117.6 MB/s.

State the number, where it came from, and what it does not cover.

## Interruption safety

Ingestion writes stacks via a `.part` file plus `os.replace`, and append mode
tests **completeness**, not `os.path.exists` -- a truncated HDF5 file opens
happily and reports the declared shape from its header, so only reading its
first and last element distinguishes complete from truncated. Both exist
because an interrupted overwrite silently destroyed two stacks that then stayed
invisible to every subsequent re-ingestion. Do not reintroduce
delete-then-rewrite, and do not weaken the completeness check to an existence
check.

`tools/verify_store.py <storage_path>` read-probes an entire store and reports
broken files, orphaned MIPs, and leftover `.part` files.
