# Principles

The pipeline's design decisions repeat a small number of ideas. Knowing
them makes the rest of the documentation predictable.

## The real case outranks the fixture

Development fixtures are small, local, and unrepresentative. The real
case is **huge, on network storage, driven from Windows**. Every
conclusion that matters — defaults, thresholds, worker counts,
performance claims — is measured on real-store-shaped inputs, and a
measurement on the real store outranks one on a fixture whenever they
disagree. Corollary: a green test suite is evidence the conventions
hold, not that a change is correct at scale.

## Measure before optimizing

Plausible optimizations here have a track record of being wrong:
dropping a "useless" fit parameter made fits slower *and* worse (it was
acting as a search direction); tripling ingestion workers halved
throughput (12 workers: 117 MB/s; 36 workers: 66 MB/s). Numbers are
stated with where they came from and what they do not cover.

## Four rules for everything that touches the store

1. **Minimize NAS I/O.** Read once, cache what was computed, and never
   re-read what a capsule already answers. Worker counts are tuned to
   the storage, not the CPU count.
2. **FOV-level capsules, with append.** Every artifact lives in per-FOV
   files (cells, alleles, spot slices, matrices, computed attributes).
   Append modes skip what is complete and fill exactly the holes.
3. **Lazy, flag-driven loading.** Nothing heavy loads at startup;
   manifests and per-FOV flag files answer "what exists" without opening
   HDF5.
4. **Background everything.** Long work runs in workers; the GUI thread
   only orchestrates and displays.

## Interruption safety is not optional

Every store write goes through an atomic door: build a `.part` file,
then one rename makes it visible whole. Append modes test
**completeness, not existence** — a truncated HDF5 file opens happily
and reports its declared shape, so only reading first and last elements
distinguishes complete from cut short. Both rules exist because an
interrupted overwrite once silently destroyed data that then stayed
invisible to every later re-run. Delete-then-rewrite is never
reintroduced, and completeness checks are never weakened to existence
checks.

## Descriptions vs. derived data

Two kinds of state, two fates:

- A **description** — a config, a gate, a checked-hybe set — is small,
  declarative, human-readable, and safe to save and restore at any time
  because it binds to nothing until evaluated.
- **Derived data** — which cells pass a gate, an expression table, a
  distance map — is a function of a description × the current store. It
  is either recomputed on demand or cached with an explicit overwrite
  path; it is never frozen into a description, because that is how
  staleness is born. Provenance snapshots (the Save Result sidecar)
  record derived results at the moment they mattered.

## Conditions narrow, flags multiply, everything is additive

A **condition** removes cells from the population (celltype, expression
range, distance range, presence, completeness — ANDed within a clause,
clauses ORed together). A **flag** splits the surviving population into
groups that multiply the figures (celltype decompose, allele-gate
modes, FOV grouping). Conditions and flags compose freely; adding one
never breaks another.

## Cell-level and allele-level are different questions

Cells are keyed by the pair `(fov, cell)`, always. But genetic
modification can be heterogeneous *within* a cell, so predicates about
alleles (barcode presence/absence, completeness) evaluate per allele and
only project onto cells afterwards. There is no allele *indexing* —
only allele *differences* are meaningful.

## The analysis suite is independent by design

`codelab_pipeline.analysis` imports no Qt and no app module — pinned by
a test that fails the build otherwise. The app *uses* the toolbox; it
does not contain it. Anything the Analysis tab can do, a script can do
headlessly against the store, which is also how every feature is
validated.

## NaN is honest

A missing measurement is NaN — in polymer positions, expression
metrics, normalized values with zero denominators, distances. Nothing
downstream ever fabricates a value to fill one, and figures draw
missingness visibly (dark map pixels, stated `n`s) rather than averaging
it away.
