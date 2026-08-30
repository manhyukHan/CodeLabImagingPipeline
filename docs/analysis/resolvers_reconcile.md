# Frames: resolvers and reconcile

Every coordinate question in this pipeline is the same question: take a point expressed in one
`(hybe, modality)` frame and express it in another. Two headless modules answer it against a store
directly, with no Qt and no running app:

- `codelab_pipeline/analysis/resolvers.py` assembles a `FrameResolver` — the object that composes
  the alignment chain — **from the store alone**.
- `codelab_pipeline/analysis/reconcile.py` uses that resolver to re-derive every stored
  `adj_` field from its `raw_` counterpart under the *currently* stored matrices.

Everything here works in the store's native units: y/x in **pixels**, z in **plane index**.
The micrometre scaling (`voxel_um = (0.208, 0.208, 0.2)`) happens once, later, at extraction into
analysis tables — never inside these modules.

## The frame model

A `FrameResolver` holds three XY layers and two Z layers for one FOV:

- **within** — `{modality: {hybe: 3x3}}`: each modality's own FOV-level matrices, each mapping a
  hybe into *its own modality's* reference frame.
- **bridges** — `{modality: (H_yx, dz)}` in a star topology: each non-hub modality gets an
  independent bridge into the **shared** (hub) frame. The hub itself is never a key — its leg to
  itself is identity by definition and is never persisted.
- **anchors** — `{modality: 3x3}`: the cell-alignment anchor of each modality into the shared
  frame, needed to consume per-cell residual matrices.

Z is separable: with no XZ/YZ rotation the transform is block-diagonal, so z travels as an
*additive scalar in planes* alongside the 2D affine, never inside a 4x4.

A layer that has not been computed yet contributes **identity**, never an error — "no correction
known" is the honest default, and mapping works from first segmentation onward. Missing layers are
*named* (see `missing` below) so you can report a result as provisional instead of pretending it
is aligned.

## Building a resolver from the store

```python
def resolver_for(storage_path, fov, shared=None):
    """A FrameResolver for one FOV, assembled purely from the store."""

def resolvers_for(storage_path, fovs, shared=None):
    """{fov: FrameResolver} -- the shape Population.build takes."""
```

`storage_path` is one modality's store directory (e.g. `E:/Students/2026-08-07-SG-test/DNA`); the
project root is its parent, and the root manifest names every modality. Assembly reads only the
store:

- **within** comes from `read_same_modality_matrices` per modality, with hybe lists taken from
  each modality's manifest-recorded experiment layout.
- **bridges** come from `read_cross_modal_matrix` / `read_cross_modal_z` per modality — and their
  *presence* names the moving modalities. The **hub is inferred as the unique modality without a
  stored bridge**: because the hub's own bridge is identity by design and never persisted, absence
  is an identifying fact, not a gap. When the inference is ambiguous (zero or several candidates),
  `resolver_for` raises a `ValueError` listing the modalities and bridges it found — guessing a hub
  silently would be a ~13 px cross-modal claim. Pass `shared='DNA'` (or whichever) to override.
- **anchors** are harvested from the FOV's cells: `matrix_anchors` is a modality-level fact
  snapshotted onto every cell at alignment time, so any carrying cell testifies for the whole FOV.

The result is plain data — numpy arrays and dicts, no Qt, no session objects — so it is
**picklable** and safe to hand to worker processes or build inside them.

```python
from codelab_pipeline.analysis import resolvers

store = 'E:/Students/2026-08-07-SG-test/DNA'
r = resolvers.resolver_for(store, 3)          # one FOV
rs = resolvers.resolvers_for(store, range(34))  # {fov: resolver}, for Population.build
```

## The transform contract

```python
H, dz, missing = resolver.transform(src, dst, cell=None)
```

`src` and `dst` are `(hybe, modality)` pairs — a bare hybe name is never a frame in this pipeline,
because the cross-modal bridge hybe exists in both modalities with a different matrix in each.

- `H` is a 3x3 affine that maps a point **from src's native frame into dst's frame**. Apply it as
  `y_dst, x_dst, _ = H @ np.array([y_src, x_src, 1.0])` — the homogeneous vector is `(y, x, 1)`.
  **Do not invert it yourself**: if you want the other direction, ask for
  `transform(dst, src, cell)`. Inversion symmetry is guaranteed by construction
  (`transform(a, b)` is exactly the inverse of `transform(b, a)`), as are
  `src == dst -> identity` and `a->b->c == a->c`.
- `dz` is **added** to src's z (in planes) to get dst's z.
- `missing` is a `set` of layer names that were defaulted to identity because they have not been
  computed yet — e.g. `'same-modality:RNA/Hyb_103'`, `'cross-modal:RNA->DNA'`,
  `'cell-anchor:RNA'`. Empty means the chain was complete. Check it when you need to tell a user
  the mapping is provisional; ignore it when you just want the best available answer.

Passing a `cell` (an `ACell`-like object with a `.matrices` dict keyed `(hybe, modality)`) engages
that cell's fitted residual route; with `cell=None` you get the FOV route. The cross-modal term is
FOV-bounded, so it applies even without a cell — an unassigned spot still sits in a modality.

The one-sided primitives are also public: `to_shared(hybe, modality, cell=None, missing=None)`
returns the 3x3 into the shared frame, and `z_to_shared(hybe, modality, cell=None)` the additive
plane offset.

```python
import numpy as np
from codelab_pipeline.analysis import resolvers

r = resolvers.resolver_for('E:/Students/2026-08-07-SG-test/DNA', 3)
H, dz, missing = r.transform(('Hyb_103', 'RNA'), ('Hyb_101', 'DNA'))
y, x, z = 512.0, 480.0, 40.0                 # pixels, planes, in the RNA/Hyb_103 frame
yd, xd, _ = H @ np.array([y, x, 1.0])
zd = z + dz
if missing:
    print('provisional -- identity-defaulted layers:', sorted(missing))
```

## Reconcile: re-deriving adj from raw

Every stored `adj_` field is a **save-time snapshot**, computed with whatever matrices existed at
its save. Re-run any alignment layer — an FOV refit, a cross-modal re-bridge, cell residuals — and
every stored adj is stale, while raw stays ground truth. `reconcile` re-derives them:

| stored field | re-derived from |
|---|---|
| `spot.adj_coordinate` | `raw_coordinate` through the resolver |
| `allele.coordinate` | `raw_coordinate` (in the allele's anchor hybe frame) |
| `fiducial_trace_adj[h]` | `fiducial_trace_raw[h]` |
| `polymer_adj[h]` | `polymer_raw[h]` projected, plus the fiducial drift correction re-derived from the *freshly projected* fiducials (`delta = fid_adj[ref] - fid_adj[h]`) |

```python
def reconcile_fov(storage_path, fov, resolver=None, modality=None,
                  reference_hybe=None, write=False): ...

def reconcile(storage_path, fovs, modality=None, reference_hybe=None,
              write=False, jobs=None, on_done=None): ...
```

Order matters and is fixed: **cells first**. Each cell's `matrix_anchors` are refreshed under the
current matrices (`bridge @ within[anchor_hybe]`), with the anchor hybe per modality parsed from
the cell's own `matrix_provenance` (`'Hyb_103(cell 3)->Hyb_101 [...]'` yields `Hyb_101`). The
resolver's harvested anchors are then replaced by the fresh ones, because every cell-route
projection below composes through them. Then **spots**, slice by slice over
`spot_slices(storage_path, fov)` — each slice is one source `(modality, hybe, channel)`. Then
**alleles**, including the polymer re-projection.

`write=False` (the default) is a **dry run**: the report says how far each category has drifted
(`shift_px` stats: `{'n', 'median', 'p90', 'max'}`, or `{'n': 0}` when nothing moved) and the
store is untouched. `write=True` persists through the store's own atomic doors
(`write_cell_dicts`, `write_spots` per slice, `write_allele_dicts`) — `.part` file plus
`os.replace`, so an interruption never leaves a half-written capsule.

Anything that cannot be re-derived is **counted and named, never guessed** — these counts are the
skip ledgers:

- cells: `{'anchor_hybe_unknown': n, 'no_within': n}` — no provenance testifies for the anchor
  hybe, or the current within layer lacks it.
- alleles: `{'no_raw': n, 'no_reference': n, 'ref_not_traced': n}` — saved before raw fields
  existed; neither provenance nor the `modality=`/`reference_hybe=` arguments name a reference
  (recent saves stamp both in provenance); or the reference hybe has no fiducial, in which case
  the polymer is left untouched rather than corrected against a guess.
- spots: `z_placeholder` counts 2D detections (see Pitfalls).

**When to run it:** after any re-alignment — whenever the matrices moved and the stored adj
coordinates therefore disagree with them. Reconcile makes the store agree with itself again, from
raw, which never lies.

## Worked example: one FOV, headless

```python
from codelab_pipeline.analysis import reconcile

store = 'E:/Students/2026-08-07-SG-test/DNA'

# 1. Dry run: how stale is FOV 3 after the re-alignment?
report = reconcile.reconcile_fov(store, 3)            # write=False by default
print(report['cells']['shift_px'])    # e.g. {'n': 41, 'median': 1.8, 'p90': 2.4, 'max': 3.1}
print(report['spots']['updated'], 'spots across', report['spots']['slices'], 'slices')
print(report['alleles']['shift_px'])  # {'anchor': {...}, 'fiducial': {...}, 'polymer': {...}}

# 2. Read the skip ledgers: what could NOT be re-derived, and why.
print(report['cells']['skipped'])     # {'anchor_hybe_unknown': 0, 'no_within': 0}
print(report['alleles']['skipped'])   # {'no_raw': 0, 'no_reference': 12, 'ref_not_traced': 0}
print(report['spots']['z_placeholder'])

# 3. If old alleles lack provenance, name the reference yourself, then persist.
report = reconcile.reconcile_fov(store, 3, modality='DNA',
                                 reference_hybe='Hyb_101', write=True)
assert report['written']
```

For the whole store, `reconcile` runs FOV-major through one pool (`kind='io'`, default 12
workers) and returns per-FOV reports in input order; a failed FOV comes back as
`{'fov': f, 'failed': '<traceback text>'}` without losing the others:

```python
from codelab_pipeline.analysis import reconcile

if __name__ == '__main__':            # REQUIRED on Windows for jobs != 1
    reports = reconcile.reconcile('E:/Students/2026-08-07-SG-test/DNA',
                                  range(34), write=True)
    for r in reports:
        if 'failed' in r:
            print('FOV', r['fov'], 'FAILED:', r['failed'])
```

## Pitfalls

- **Do not invert `H` to go backwards.** `transform(src, dst)` already maps src into dst; the
  reverse direction is `transform(dst, src)`. Hand-inverting invites applying a matrix in the
  wrong source frame — historically the cause of every alignment bug here.
- **Frames are pairs.** `('Hyb_130', 'RNA')` and `('Hyb_130', 'DNA')` are different frames with
  different matrices; a bare hybe name never identifies a frame, just as a bare cell id never
  identifies a cell (cells are keyed by `(fov, cell)`).
- **Identity is the default, silently.** An incomplete chain still returns a usable transform.
  For the cross-modal bridge that default asserts the modalities coincide — on real data a
  ~13 px claim — so inspect `missing` before presenting anything as aligned.
- **Hub inference can fail loudly.** Two modalities without bridges (nothing bridged yet), or all
  modalities bridged, makes the hub ambiguous; `resolver_for` raises with the facts. Pass
  `shared=` explicitly rather than catching and guessing.
- **The 2D-placeholder z rule.** A spot with raw z == 0 *and* adj z == 0 is a 2D detection whose
  depth was never measured. Reconcile keeps z at 0 and counts it under `z_placeholder`; pushing 0
  through the z-chain would mint a depth out of a placeholder. NaN and 0-placeholder alike mean
  "not measured" — never fabricate.
- **Skip ledgers are not errors.** A nonzero `no_reference` means those alleles predate provenance
  stamping; re-run with `modality=` and `reference_hybe=` if you know them. A nonzero count you
  cannot explain is a data question, not something to paper over.
- **`reconcile_fov` mutates a resolver you pass in**: its `anchors` are replaced with the freshly
  refreshed ones. Build a throwaway with `resolver_for` (or pass `resolver=None` and let it) if
  you need pristine harvested anchors elsewhere.
- **Units.** Everything in these two modules is pixels and planes. Micrometres exist only in
  extracted analysis tables, scaled once at extraction.
- **Windows multiprocessing.** `reconcile(..., jobs != 1)` uses a spawn-context process pool, so
  the call must sit under `if __name__ == '__main__':` in any script — the same rule as
  `Population.build`. `jobs=1` is a true serial path (no pool, no pickling) and needs no guard.
  More workers is not faster on NAS I/O; the measured default (12) usually wins.
