# CodeLabImagingPipeline — protocol source material

Facts assembled from the repository for drafting a Nature Protocols
manuscript. **This is source material, not a manuscript**: it is organised
into the sections Nature Protocols expects, but it states facts rather than
writing prose, so the argument stays yours.

Assembled 2026-09-04 from the tree at `e98b284`, branch
`alignment-correctness-and-memory` — **15 commits ahead of `main`
(`a36388c`)**. Which of the two the paper describes is an open decision; see
*Before submission*, A6.

## How this was built

Nine agents read the repository independently — all source, `docs/`, and the
full history of **251 commits carrying roughly 75,000 words of design
rationale**, which in this project is where the reasoning lives. Two further
agents audited the result against what a protocol paper must be able to
state, and their findings are the *Before submission* section. Six more
compressed the harvest into the sections below.

Numbers are marked **MEASURED** or **ASSERTED**, and every fact carries its
source as `file.py:123` or a commit hash. Where the repository contradicts
itself, the seam is left visible rather than resolved by guess.

## How to use it

Each section stands alone and can be pasted on its own; the whole document
is about 19,000 words. Suggested order of work:

1. Read **Before submission** first. It is what the repository cannot
   support, and some of it needs a measurement or a decision before any
   drafting is worth doing.
2. Draft **Experimental design** and **Procedure** from here directly — they
   are the best-evidenced parts.
3. Treat **Timing** as a worklist, not a table to copy: several stages have
   never been timed, and several recorded figures do not say whether they
   were warm or cold.

## Two corrections that affect wording throughout

**The analysis store is not a NAS.** Verified with `fsutil fsinfo drivetype`
on 2026-09-04: `C:`, `D:`, `E:` and `G:` are Fixed Drives; only `Q:`, `S:`,
`P:`, `R:` and `T:` are Remote/Network. Raw DAX lives on the network drives;
every analysis store measured in this project is on the local `G:`. The
repository, `CLAUDE.md` and many commit messages call it a NAS. The
measurements are sound — the storage class in them is not. Where throughput
matters, give the measured rate (`420e9bd`: ~51 MB/s on cold stack reads).

**One headline number could not be traced.** "117.6 MB/s at 12 workers
against 66 MB/s at 36" appears in six places including `CLAUDE.md`, and no
run behind it survives in the history. See *Before submission*, A2.

---

## Experimental design

### 1. The imaging experiment the software consumes

The pipeline consumes sequential DNA-FISH (chromatin tracing) acquisitions, optionally paired with an RNA/immunofluorescence acquisition of the same field. The two acquisitions are **separate imaging sessions of the same sample**, called *modalities*, and each carries its own `ExperimentLayout.xlsx` (`preprocess.parse_experiment_layout`, preprocess.py:134–195). Columns consumed: `FolderName, Readouts, DataType, HybNum, channels, fiducialChannel, channelLayout, totalFrames`, optional `rnaNames`. `datatype` and `readout_id` are **informational only — alignment logic must never branch on them** (preprocess.py:141–145).

**Round types.** `DataType` partitions the rounds and this partition is load-bearing downstream (`tiff_ingestion.py:55–130`; `analysis/polymer.py:60–90`):

| code | folder | role | used as a polymer bin? |
|---|---|---|---|
| `H` | `Hyb_nnn` | genomic bin (a locus in the trace) | yes — the only one |
| `R` | `Rep_nnn` | replicate of an H locus | no; becomes ground truth (§7) |
| `T` | `Toe_nnn` | toehold **displacement control** | no; positive control |
| `B` | `Hyb_nnn` | barcode round (cell-type identity) | no; feeds celltype assignment |

Bin identity comes from the layout, **sorted by `readout_id`, not acquisition order** (`polymer.genomic_bins`, polymer.py:60–90).

**Real dimensions** (MEASURED, walked from the stores on the project volume `G:` on 2026-09-04; harvest §8, storage-ingestion):

| store | modality | FOVs | hybes/FOV | frame | depth | channels (fiducial) | stack file | MIP |
|---|---|---|---|---|---|---|---|---|
| MP58 `G:/Seonghyeok/2025-11-30-MP58` | DNA | 41 | 76 (H62/R7/T4/B3) | 1024×1024 | 110 | 555, 635 (555) | 233.5 MB | 2.49 MB |
| " | RNA | 41 | 12 | 1024×1024 | 105 | 555, 635 (555) | 220.2 MB | — |
| MAZ `G:/Seeun/2026-06-17_MAZ_IF_15ug` | DNA | 50 | 135 (H108/R17/T10) | 1024×1024 | 129 | up to 475,488,390,555,635 (555) | 278.8–821.0 MB | 2.5–8.2 MB |
| cross-modal `G:/JP/2026-01-26-cross_modal` | DNA / RNA | 2 (40 in `76e8c6b`; trimmed since) | 100 / 16 | 1024×1024 | 177 / 130 | 555, 635 (555) | 446.1 MB | — |
| HoxA `G:/Dariya/2026-08-22-RNA_DNA_HoxA` | RNA / DNA | 43 | 35 (fov001) | 1024×1024 | 86–129 | 555, 635 (555) | 187.0 MB | — |

Depth is **derived per hybe**, `total_frames // len(channels)` (preprocess.py:530), and is explicitly not uniform within a store: RNA Hyb_101/105/500 = 130 planes while Hyb_130/131 = 177 in the same experiment (MEASURED, commit `8f778b0`). A raw DAX file is 461,373,440 B = 440 MiB (MEASURED, MP58 `ConvZscan_00.dax`). Stacks compress to **50.6–60.7 % of raw uint16** (MEASURED, six real files).

**Voxel size is a config input with a default, not a measured constant of record**: `DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)` µm (y, x, z) — `fit3d_um.py:54`, `tracing_v2.py:66`, `polymer.py:32` — while real configs carry `0.145` (MAZ) and `0.202` (HoxA) axial. **No objective, NA, immersion, camera or dye-to-channel mapping is documented anywhere in the repo**; NA 1.4 / ~600 nm appears only as rationale for a plausibility floor (`343dc62`). A lateral pixel and an axial plane differ by 4 %, not 2× — which is why v1's `2×max_uncert` axial gate was wrong (`tracing_v2.py:103–107`).

Every hybe is written once as a chunked HDF5 z-stack `(height, width, depth)`, z-LAST, plus a standalone per-hybe MIP whose **existence is the ingestion flag** (`paths.py:36–42`).

### 2. The three-level alignment model

Drift is decomposed into three strictly ordered layers, each correcting a different physical cause, each fitted on different image data. The chain is **FOV → cross-experiment → cell → spot** and composition is strictly ordered — commutativity is deliberately not relied upon, because the ORB/homography path can produce rotation even though all 22 stored matrices in one observed session were pure translation (`frames.py:46–50`).

| level | corrects | image data | algorithm | output |
|---|---|---|---|---|
| **1. FOV, same modality** | stage drift between hybridisation rounds of one modality | each hybe's **fiducial-channel MIP** vs the reference hybe's fiducial MIP, read from the store's own MIP copy, never the stack (chain.py:1027–1030, 1059–1063) | ORB+RANSAC seed → Powell/MSD refine | 3×3 y-major affine per (hybe, modality) |
| **2. cross-modal bridge** | the two imaging sessions' frames not coinciding | each modality's **bridge-hybe MIP**, channel selectable, UI default `readout`; **each side pre-warped by its own level-1 matrix first** (chain.py:1210–1211) | same fit as level 1 (XY) + 1D correlation of depth profiles (Z) | one `(H_across, dz)` per non-hub modality |
| **3. per-cell residual** | local, cell-scale residual the whole-frame fit cannot absorb | a **windowed crop** around the cell mask, projected into each hybe's own native frame; fiducial channel by default | `cv2.phaseCorrelate` (translation only) + 1D ZX correlation | translation-only `H2` + `dz` per (cell, hybe, modality) |

**Level 1 — "ORB detects, Powell refines"** (docstring chain.py:455–538). ORB + `BFMatcher(NORM_HAMMING, crossCheck=True)` + `estimateAffinePartial2D(RANSAC)`, the 2×2 block re-orthogonalised by SVD so the result is rigid (preprocess.py:807–845). The seed is admitted **one-out**: failing dx *or* dy *or* angle discards the whole seed and Powell starts cold at (0,0,0), because ORB's three outputs come from one correspondence set and trusting the translation of a fit with an implausible angle trusts the same bad correspondence twice (chain.py:565–571). The angle is quantised to `ANGLE_QUANTUM_DEG = 0.5` (chain.py:43) so "no rotation" is exactly 0.0. Powell then optimises `[dx, dy, angle]` with the angle **free, seeded not fixed**, at `fixed_scale=1.0` within native bounds ±30 px, ±10°.

The justification for keeping the free angle is measured and counter-intuitive (MEASURED, 24 real pairs, chain.py:588–592): seed+free is **1.28× faster with mean residual −0.217 (22/24 better)**, cold+fixed is **0.93× slower, +0.569 (3/24)**. ORB alone is not the answer either: over **77 real hybe pairs** its centre displacement sat a median **0.165 px** from Powell's final answer at 1/127th the cost, yet Powell's refined fit won on reconstruction residual **76/77 times** (chain.py:479–486; commit `a968c3e`). Net effect of the seed: **1.25× faster (351.8 → 281.0 s; 15,973 → 12,631 objective evaluations)**, matrices moving a median 0.0135 px.

Selection metric is `_reconstruction_residual` (chain.py:234–282): mean squared pixel error after warping, evaluated **only where the reference carries real signal** (`reference_norm > 10` on 0–255), with a geometric guard returning `inf` below `min_overlap_frac = 0.5` coverage — motivated by a real cross-modal pair where a bad transform at **8.6 % overlap** outscored the correct one at **62.2 %** (MEASURED, chain.py:263–270).

**Level 2 is a star, not a mesh.** The first activated modality is the hub; its own bridge is identity **by definition and is never stored**, and that absence is the identifying fact letting a reader infer the hub from a store alone — ambiguity raises rather than guesses (`docs/analysis/resolvers_reconcile.md:53–57`). A non-hub pair composes through the hub: `inv(to_shared(dst)) @ to_shared(src)` (frames.py:145–164). The XY leg defaults to the **readout** channel because DNA and RNA sessions have no generally shared fiducial signal (chain.py:1133–1146). The Z leg reads the **raw fiducial stacks** of both bridge hybes over a hardcoded 640×640 window (`y0=192…x1=832`, no recorded rationale), correlates whole-FOV depth profiles from **both the ZX and the ZY projection with the two correlation curves summed before the peak is taken**, so a weak axis can be outvoted (chain.py:1235–1322). A rejected alternative is on record: focal-plane matching, whose variance-of-Laplacian peak frequently lands on a stack **endpoint** and produced drift estimates of **−51 and −171 planes against a per-cell truth near −12** (MEASURED, chain.py:1249–1255).

**Level 3 fits per modality only.** The genuinely cross-modal fit — a DNA crop phase-correlated against an RNA anchor — was implemented and **dropped**: it "consistently gave worse results than same-modality cell alignment" (ASSERTED, `CellAlignmentWorker` docstring main_window.py:496–517; commit `0a2c769`). Only the cell **mask** crosses the modality boundary, to define the other modality's crop windows. The residual is stored **bare** (`yx_is_residual: True`) and recomposed at read time from the *current* FOV matrices, so re-running level 1 updates every fitted cell without a re-fit (chain.py:2295–2309). Only crops are ever resampled; whole images never are (chain.py:1523–1528).

### 3. Every reject gate

Hard engine bounds are `MAX_ALIGNMENT_TRANSLATION_PX = 30.0` and `MAX_ALIGNMENT_ROTATION_DEG = 10.0` (verified chain.py:32–33), deliberately **not exposed as UI settings**. They are enforced on **centre displacement**, not the raw translation column: an 8° rotation of a 1024×1024 frame written about the origin carries `t = (70.8, −64.1)` while moving the centre not at all, and before the fix correct rotated fits were discarded and angles collapsed to 0.000 (MEASURED, commit `9d9e96a`, chain.py:68–109).

| # | level | gate | threshold | on failure |
|---|---|---|---|---|
| G1 | 1 & 2 | ORB seed admission, one-out | 30 px centre, 10° | seed discarded, cold start |
| G2 | 1 & 2 | Powell native bounds | ±30 px, ±10° | search clipped |
| G3 | 1 & 2 | post-hoc candidate filter | as G1, with shape | candidate removed; none survive → **identity returned** |
| G4 | 1 & 2 | selection | argmin `_reconstruction_residual` | — |
| G4a | " | geometric overlap | `min_overlap_frac = 0.5` | residual `inf` |
| G4b | " | reference signal | `reference_norm > 10`; none → `inf` | — |
| G5 | 1 & 2 | operator `max_shift` (default off) | user px | clamps centre displacement, rotation untouched |
| G6 | 3 (YX) | magnitude vs crop padding | `hypot(dx,dy) > pad` (default 10 px) | `H2 := I`, keep levels 1–2 |
| G7 | 3 (YX) | hard bound, pad-independent | 30 px | `H2 := I` |
| G8 | 3 (YX) | quality | strict improvement required | `H2 := I` |
| G9 | 3 (Z) | magnitude | `abs(z) > MAX_CELL_Z_SHIFT_PLANES = 15` (verified chain.py:65) | `z := 0` |
| G10 | 3 (Z) | quality on the ZX crops | strict improvement required | `z := 0` |
| G11/G12 | multi-peak (unshipped) | per-peak magnitude / quality | `hypot > 12` | peak **dropped**, never zeroed |

Two rationales a referee will want. **G8 exists because `cv2.phaseCorrelate` does not return (0,0) on an already-aligned crop** — it returns a small pad-passing shift that makes the overlap worse; observed noise locks reached **several thousand px on crops only ~80 px wide** (MEASURED, chain.py:2236–2238). **G9's threshold was measured, not chosen**: at the old bound of `pad/2 = 5` (a depth limit derived from an XY search radius) the applied |z| distribution topped out at exactly 5.0 — truncated by its limit, not by physics — with **16.4 % of 110 entries rejected on magnitude alone and real |z| out to 11**; the quality-rejection count is identical (9) at every bound from 5 to 30, and the result plateaus at 15 (MEASURED, real FOV01 cell 55, commit `8b22c09`). The Z gates exist at all because an ungated Z leg once applied **z = −35 planes** on a crop whose XY fit had already been quality-rejected (MEASURED, cell 16, Hyb_130 vs Hyb_101, commit `7a4e823`).

Standing principle: **reject never means "no alignment data."** An earlier `continue` left the hybe out of `cell.matrices` entirely, indistinguishable from a hybe that never overlapped the frame, discarding the perfectly good level-1/2 result too (chain.py:2240–2250). Likewise `CellOffFrameError` — a mask projecting wholly off-frame, caused by up to **~28 px of measured bridge shift** — skips that pass and writes explicit identity residuals rather than aborting: one real store held exactly **2 such slivers among 1091 cells** (MEASURED, chain.py:915–925).

The Z leg's own defence: the target ZX is pre-aligned in X only, and the width axis is then **collapsed entirely before correlating, so no axis is left for an x-shift to exist on**. An earlier design fitted a 2D (x,z) shift and gated x afterwards; a real x-shift slipped through the 3 px bound (MEASURED, cell 3, Hyb_105 vs Hyb_101 — chain.py:2348–2360).

*Seam:* `estimate_cross_modal_z` has **no automatic quality gate**. Its own docstring says to gate on `quality` and keep a manual override; the production worker stores the number and applies no threshold (main_window.py:926–937). On equal-depth stacks it returns ~0 where the truth was +12/+4 planes — a graceful but real failure (chain.py:1257–1261).

### 4. Coordinate and matrix conventions

The pipeline is rasterized **(y, x)** by explicit decision (convention.py:1–37, quoted: *"I prefer rasterized version, Y/X"*). Points are `[y, x, 1]` column vectors; 3-tuples are `(y, x, z)` matching `img[y,x]` and `stack[y,x,z]`; matrices act as `H @ [y,x,1]` with `ty = H[0,2]`, `tx = H[1,2]`; shapes are `(height, width)`; crops are `(Y, X, Z)`. Every store is stamped `coordinate_order='yx'` and readers **refuse** an unstamped store loudly, naming `tools/migrate_store_to_yx.py`.

OpenCV stays x-major. Conversion happens **at the boundary only**, through the self-inverse conjugation `P @ H @ P` (`as_cv2` / `to_yx`, convention.py:41–61), which is exact for any affine — rotation, anisotropic scale and shear included, not just translation. "No other module may re-derive this permutation." The engine boundary is literally one line: `align_readout_to_reference` returns `to_yx(H_final)` (chain.py:653); everything inside is x-major cv2-native, everything outside is y-major. Z is **separable by construction** (no XZ/YZ rotation) so it travels as an additive scalar in **planes**, never inside a 4×4 (frames.py:36–39).

**What breaks without the adapter, all four confirmed real and all four silent:**

| failure | mechanism | source |
|---|---|---|
| every traced position mirrored | a reimplementation trusted a stale `(x,y,z,amp)` docstring over `(y,x,z,amp)` code | `22e7417`, `d49ebb3` |
| readouts "failing", occupancy under threshold, fiducial drift over gate — three unrelated-looking symptoms | v2 passed `shared_xy` transposed: a spot at y=300, x=700 was fitted at y=700, x=300. "The failure is silent, because a crop taken anywhere still contains pixels and still fits something" | `6c7ba9c` |
| cells built transposed in both branches, invisible on square frames | pack-direction error in `load_new_cells` | `a645a26` |
| a ZX pre-alignment that puts X on the depth axis | cv2's (col,row) convention; hence the deliberate read of `H2[0,2]` **before** `to_yx` (verified by a synthetic single-pixel test) | chain.py:2348–2360 |

Because `voxel_um` is `(xy, xy, z)`, `dy == dx`, so **a Y/X inversion produces identical `_um` numbers and plausible plots** — the export tests therefore assert the swap with y and x far apart (`510b7c0`). The convention is not self-checking; that is the argument for the single adapter.

One further convention carries the same weight: matrices are keyed by the **PAIR `(hybe, modality)`, never a bare hybe**, because the bridge hybe (e.g. Hyb_130) is a real distinct acquisition in *both* modalities with a different matrix in each. Bare-string lookups **raise** rather than returning `None`, since a silent miss is indistinguishable from "no alignment", which downstream treats as identity (frames.py:70–131). The whole composition lives in one place — before `frames.py` it was written longhand at **~21 call sites**, and *"Every alignment bug this codebase has had was two of those sites disagreeing… None of them were arithmetic errors; they were all 'wrong set of matrices'"* (frames.py:1–16). `FrameResolver.transform` guarantees by construction that `src == dst` is identity, that `transform(a,b)` inverts `transform(b,a)`, and that `a→b→c == a→c` (frames.py:331–335). Identity-defaulting a missing cross-modal bridge is reported as `missing`, because on this data it **asserts the two modalities coincide, a real ~13 px claim** (MEASURED, frames.py:258–260).

### 5. Segmentation routes

Segmentation is **single-FOV and interactive only** — there is no batch entry point anywhere in the tree (verified by grep; `docs/happy_path.md:43–47` reads as if there were one, and that sentence is about cell *alignment*'s append). All four routes funnel into one path: integer label mask → `CellContainer.load_new_cells` → `ACell` objects.

| route | method | key parameters (defaults) |
|---|---|---|
| Cellpose | `Cellpose(gpu=True, model_type='cyto3').eval(..., channels=[0,0], do_3D=False)` (segment.py:25–71) | diameter 40 px, min 1000 px, max 10000 px |
| Classical | Otsu/Yen/Li/Triangle/absolute → opening → EDT → `peak_local_max` → watershed (segment.py:93–143) | min_distance 7 px, min 500 px (UI) / 1000 px (library — a real discrepancy) |
| Manual | polygon draw, `skimage.draw.polygon` scanline fill; additive on top of either automatic result (cell_displayer.py:284–309) | — |
| Cytoplasm | nucleus-seeded Cellpose in a **separate pop-up**, because the cytoplasm hybe is generally not the nucleus hybe and can belong to the other modality (cytoplasm_panel.py:26–30) | diameter 60, seed style `rim emphasis`, dilation 0 |

Cellpose has no API taking a label mask as a seed, so nuclei are **rendered into a synthetic nuclear image** — never by encoding ids as intensity, since cellpose normalises the channel and reads it as a stain (cell 67 would be sixty-seven times brighter than cell 1). Instance separation comes from cellpose's own predicted flows, i.e. from shape and gaps, which is what all three seed modes manipulate (segment.py:269–322). Merge rules are explicit: a cytoplasm **inherits the id of the nucleus inside it**; a cytoplasm with no nucleus is discarded ("cellpose readily invents these"); several nuclei in one cytoplasm → largest-overlap wins, never a split, because a split would have to invent an id; **nuclei always win overlaps** (segment.py:398–460). Overlap authority is global; claiming authority is not — conflating them let unselected cells claim cytoplasms, so **66 of 67 cells came back with a cytoplasm from a 50-cell selection** (MEASURED, segment.py:409–418).

**The projection is the decision that matters most.** A max projection over the *full* depth accumulates the brightest halo from every plane and visibly destroys the brightfield phase contrast that defines a boundary (segment.py:41–44). Focal plane is found by variance-of-Laplacian per plane, because "the middle plane is the focal plane" measurably fails: a 130-plane stack has its sharpest plane at z=76, off by 11 (MEASURED, segment.py:174–179).

| projection | cells found (MEASURED, repo, same FOV/hybe/channel/params) | cells found (MEASURED, independent replication, HoxA RNA FOV001 `BF` ch635, 129 planes) |
|---|---|---|
| MIP (stored) | 33 | 77 |
| single plane | 91 (z=76) | 103 (z=71) |
| range MIP | — | 98 (z=46–78) |

Sources: segment.py:155–158 / main_window.py:4402–4406; harvest §3 replication. A confirmation dialog fires **only** in the exact combination "a focal plane was detected this session AND the mode is still MIP (stored)", so it cannot become click-through noise.

Cell identity is **segmentation-bound** (cell_container.py:16–21): a fresh run renumbers freely, append continues the numbering, a removed id never returns until the next run, ids are never compacted or reused. `preserve_existing=False` therefore refuses to match by id — Cellpose and watershed both number a fresh run from 1, so id-matching would attach an old cell's `reference_hybe`, matrices and history to a **different physical cell that happens to get the same number**.

Real yields (MEASURED, read with `columnar.unpack_cells`, 2026-09-04): HoxA 43 FOVs / **3547 cells**, median **86 cells/FOV**, median nucleus area **3918 px ≈ 63 px equivalent diameter** against a Cellpose `diameter` setting of 40; the `min_size = 1000` floor is visibly binding (smallest surviving cells 1022–1077 px in all three stores). **`has_cytoplasm()` is false for every one of the 3953 cells in all three stores** — the cytoplasm route works but has not been used on persisted production data; say so rather than implying it is routine. **No accuracy validation of segmentation against manual annotation exists anywhere in the tree** — all evidence is cell counts and self-consistency.

### 6. Spot detection and 3D fitting

**2D detection is a plain `skimage.feature.peak_local_max` over a MIP.** There is no Gaussian sub-pixel refinement in the shipping 2D path — the only code where `fit_gaussian_2d` is reachable, `localize_cell_2d_worker` (localization.py:769), has **no live caller**. Parameters: threshold as % of scope max (default 50) or absolute ADU, `min_distance` 3 px, cell-crop padding 10 px (`ui/spot_localization_panel.py:118–139`). Detected spots carry `z = 0.0` as an explicit **placeholder, not a measurement**; downstream, `analysis/reconcile.py:26–30` counts them as `z_placeholder` and leaves them alone, because "projecting 0 through the z-chain would MINT a depth out of a placeholder." MEASURED, MP58 FOVs 1–3: most detections are fiducial-channel — **372 fiducial vs 49 readout** (`510b7c0`), which is why `channel_role` exists in the export.

**3D refinement adds Z to an already-placed spot; it never detects** (localization.py:483–490). It is the only place the raw z-stack is opened for a spot (`spot_mapper.crop_for_localization`). Two engines:

- **v1** (`fit_gaussian_3d`, a ChrTracer3 `FitPsf3D` port): bounded `least_squares`, 8 params, rejection by CI from the residual Jacobian at `student_t.ppf(0.975, dof)` plus height/background and amplitude/height ratios. Never raises — "absence is not an error."
- **v2** (`fit3d_um.fit_gaussian_3d_um`): the same least-squares objective recast as an intensity **field over real space in µm**. Defaults were *converted, not retuned* (`peak_bound 2.0 px → 0.416 µm`, `max_sigma_xy 2.5 px → 0.520 µm`, etc., fit3d_um.py:36–42).

The v2 gains are a measured ladder, not a redesign (MEASURED, `tools/v2_variants.py` on **48 real MP58 alleles / 7008 crops**, scored by *occupancy* = intensity at the fitted centroid ÷ intensity at the argmax, both above a local background — a score needing no ground truth; reproduced at tracing_v2.py:15–26):

| change | effect |
|---|---|
| fit a **box** with a linear background, not a full-depth pillar | occupancy **0.373 → 0.806**; blank-region fits **31 % → 4 %** |
| place the box at a **consensus depth** | fiducial z spread **1.00 → 0.56 planes** |
| intensity-centroid seed | v1 **0.354 → 0.597**; v2 barely moves |
| loose separate bounds (5 px / 10 planes) | at-bound **75–100 % → 2–13 %** |
| calibrated PSF, σ fixed | fiducial **0.803 → 0.838** and **37 % faster**; readout **0.561 → 0.677** |
| Poisson MLE | **no measurable gain on real crops, 20–40 % slower — not used as the estimator** |

Net: occupancy **0.354 → 0.838**, blank-region fits **34 % → 0 %**. The failure the box fixes is diagnostic: on a 17×17×110 pillar the PSF is a few hundred voxels against ~34k background voxels, so the optimiser buys more by inflating σ than by centring — 60/60 fiducial crops railed on every bound at once (`578842b`, `892d5a5`).

**No bead stack exists**; the PSF is calibrated from reference-hybe spots (`psf.py:1–20`). Across four experiments spanning 0.29–18.5 Mb the readout PSF is **universal**: `gaussian_halo` wins in all four at σ_xy **127–147 nm**, a spread *tighter* than the ~40 nm one experiment moves when its own crops are reselected (MEASURED, `28f4f54`). The shipped default is `universal-default`, the mean of the three that converged. Two honest limits: the **fiducial is not a PSF** — recovered σ tracks the fit window as σ ~ r^0.5 (p = 0.51–0.59 across radii on all four experiments, where a true Gaussian gives p = 0) — and **Chr19's readout PSF never converged** (σ moved 34.7 nm from n=110 to n=160) yet a per-experiment file exists for it. A rejected PSF falls back to free σ and `describe()` names the refusal, "a run that silently became a different run has to say so."

### 7. Spots → alleles → chromatin traces

**Spot → cell** is one integer lookup, not an optimisation: group spots by their own `(hybe, modality)`, build ONE label image **in the spot's own frame** by projecting each cell from *its own* reference hybe, then `mask[int(raw_y), int(raw_x)]` (assignment.py:27–77). Zero → `cell = -1`, unassigned. A `None` matrix means "cannot place this spot", **never identity** — "guessing a frame is how spots land in the wrong cell." Building one mask in `cells[0]`'s frame was a real bug: real data has **101 cells on (Hyb_400, DNA) and 1 on (Hyb_101, RNA)**, and the flattened mask placed the odd cell wrong (`9a3d5e8`). MEASURED cost, 102 real cells, 1024²: **138.7 ms for 100,000 spots (1.39 µs/spot)**, 5.6× the per-spot route and independent of cell count — the stated design consequence is that assignment is simply **re-derived whenever cells or matrices change**, so no staleness-marking scheme is needed. There is **no Hungarian or optimal-transport assignment cost anywhere**; containment is a mask index.

**Spot → allele.** An allele is seeded from one `ASpot` and keeps its `anchor_uid`, so the session can refresh the anchor when a later 3D refinement moves that spot (confirmed real divergence). `anchor_channel` records where the seed came from and **does not determine the traced readout channel** — deriving it from `anchor_channel` meant an allele seeded on a fiducial-channel spot silently traced the *fiducial* through every hybe, producing visually identical fiducial and readout grids (localization.py:1135–1147). Channels come per hybe from the ExperimentLayout record.

**Allele → trace (v2, shipping, `tracing_v2.build_chromatin_trace_allele`:788).** All fiducial crops are cut first, the depth is derived from them *together*, and only then is anything fitted:

1. **Cut** every fiducial crop; record each hybe's `cell_z_offset`.
2. **Depth placement**, fit-free on purpose: `shared_z(h) = argmax_z(h) + offset(h)`; `baseline = median over h`; `native_z(h) = baseline − offset(h)`. MEASURED, placement error **1.05 planes at 0.09 s** against **4.65 planes at 106 s** for a per-hybe pillar fit; a single shared depth for every hybe gives **1.78 µm** median pair distance where correct placement gives **0.41 µm** (consensus_native_z docstring, 449–458).
3. **Fit the fiducial** with free σ, linear background, generous bounds; on gate failure **retry once from the argmax**. The reference hybe is exempted — accepted whenever it fitted at all — because gating it out rejects the *allele*, not one round: 2 of 4 HoxA alleles lost all 45 rounds apiece behind that one symptom (`tracing_v2.py:952–1006`). The verdict is kept as `reference_warning` in provenance.
4. **Drift gate**, evaluated **in the shared frame** (a raw-frame distance would compare two different pixel grids): `delta = fiducial_adj[ref] − fiducial_adj[hybe]`, XY rejected past 7 px, **Z gated separately in planes** past 15. The Z gate was added for a real case: a barcode round's weak fiducial sat 20 planes from the reference at only 1.4 px lateral drift (`ced2957`).
5. **Fit the readout** with σ **fixed from the calibrated PSF**, in a box anchored at *this hybe's own fitted fiducial z*, then add `delta`.

Raising the fiducial peak bound and its gate **together** from 5 px/10 planes to 7 px/15 planes is instructive about how the parameters couple: traced readouts **24 → 35 (+46 %)**, fiducials kept 40 → 60, fiducial at-bound rejections 33 → 1, while occupancy rejections rose 0 → 12 (MEASURED, FOV1 allele 15, `cc08f4b`). *"The gate and the fit bound are the same number and must move together"* — a gate at 7 px against a 5 px bound gates on a quantity the fit cannot produce.

**Ground truth is internal to the experiment.** Ten loci are imaged more than once (H/R rounds carrying the same readout), giving 12 same-locus pairs per allele, 7 valid; two rounds of one locus in one allele should land on the same point, so their 3D distance **is** the localization error. **Only H and R count as replicates** — T (toehold) rounds are designed displacement controls, and scoring them as replicates would book designed displacement as error. Used as a positive control they separate cleanly: **replicates 0.090 µm vs toeholds 0.457 µm, 5.1×** (MEASURED, `892d5a5`).

| experiment | scope | v1 traced | v2 traced | v1 median | v2 median | change |
|---|---|---|---|---|---|---|
| MP58 | 4.0 Mb | 306 | 964 | 0.2291 µm | 0.0741 µm | −67.6 % |
| Chr19 | 18.5 Mb | 727 | 2086 | 0.1679 µm | 0.0955 µm | −43.1 % |
| CrossMod | 18.5 Mb | 218 | 590 | 0.3681 µm | 0.1612 µm | −56.2 % |
| HoxA | 0.29 Mb | 427 | 1423 | 0.1843 µm | 0.0594 µm | −67.7 % |

MEASURED end-to-end through the app, both arms differing by exactly one widget (`b0bf3f7`). Scorable pairs rose 12→70, 27→98, 8→24, 18→60 — "both axes at once, which is the part that cannot be gamed: an engine buying precision by discarding hard spots shows FEWER pairs." The shipping axial-uncertainty gate of 150 nm is an explicit precision-over-coverage trade: it keeps **44 % of readouts** for a **2.4× better p90** than ungated, with a knee near 300 nm if coverage matters more (MEASURED, 127 MP58 FOV1 alleles, `d7dbde8`).

**The trace object.** `polymer_raw/_adj` holds *every* accepted readout detection per hybe, not just the brightest, so multiple candidates (e.g. sister chromatids) sit side by side and are never pruned. `rejected_hybes` keeps the verbatim reason — "a missing bin is a result", and the long-format export gives rejected bins a row with blank coordinates. One asymmetry is stated verbatim in the model and must survive into the paper (allele.py:38–58):

```
fiducial_trace_adj − fiducial_trace_raw = alignment only (FOV + cross-modal + cell residual)
polymer_adj        − polymer_raw        = alignment PLUS the fiducial drift correction
```

*"A fiducial is not corrected by itself; it IS the correction."*

Distance maps then scale to µm **exactly once**, in `fov_polymer_table` (polymer.py:170), z at 0.2 µm/plane rather than the 0.208 µm lateral pitch. QC thresholds are **quantile-derived from the data at hand**, never inherited constants (`brightness_q=(0.05,0.95)`, `jump_q=0.75`, `max_dist_q=0.95`), and the ported ORCA amplitude gate was corrected — the original compared all four tuple components elementwise, so any *coordinate* below the 5th-percentile brightness was falsely flagged (polymer.py:260). An isotropy QC reports `factor = sqrt(E[dz²] / (E[d_xy²]/2))`: MEASURED **1.516 on MP58** (12 FOVs, 1572 alleles, 664,654 pairs) — axial spread 52 % wider than in-plane — validated against a zero-axial-error surrogate scoring 0.985 and a z×1.5 scale error scoring 2.296. A stronger claim was **made and withdrawn**: the per-bin factor curve cannot distinguish a scale error from an additive one, and `z_scale_correction`'s endorsement was withdrawn with it (`b09e0e0`).

**Seams a referee will find, stated here so the manuscript does not have to be caught by them.** (i) **All v2 gate thresholds are single-dataset** — MP58 FOV1, one experiment; the module says so repeatedly and the GUI tiles print the very number each gate tests, which is the re-derivation route. (ii) **Sister-chromatid handling is engine-dependent**: v1 keeps every accepted component per hybe, v2 emits exactly one. (iii) **`final_polymer` is empty on both real stores**, so every export and analysis path recomputes the brightest-candidate collapse at read time — describe the committed polymer as derived, not stored. (iv) **The rotation branch of level-1 alignment has no real-data coverage**: real inter-hybe rotation is `|angle| ≤ 0.1007°` over 77 pairs, the only coverage is synthetic ground truth, and `tests/test_alignment_ground_truth.py` cannot run on the Windows machine at all because its fixture is absent (`a968c3e`). (v) **`background_clip`, `fit_method='multi_peak'` and `integer_shift` have zero callers** — documented and measured on named real cells, but unreachable from the app; they are unshipped experiments, not pipeline behaviour. (vi) The cell-alignment `channel_type` default **disagrees between engine (`readout`, chain.py:1513) and GUI (`fiducial`-first, alignment_panel.py:221)**; only the GUI value is used in practice, but a headless caller gets the other. (vii) `channelLayout` is parsed and round-tripped but **never branched on** — the DAX de-interleave always assumes alternate, so a non-alternate layout would be silently mis-de-interleaved.

---

## Materials

### Software

The conda environment `code_lab_imaging_pipeline` is built from `requirements.txt` at the repo root, which is the single source of truth — "add a dependency here, not into an env by hand" (`requirements.txt:1-4`). Python 3.11 (`launch_codelab.bat:32`, `docs/installation.md:55`); the live env resolves 3.11.15.

Versions below are MEASURED — `importlib.metadata` against the live interpreter `D:\conda-envs\code_lab_imaging_pipeline\python.exe`, run today.

| Package | Resolved | Pin | Role, and why the pin exists |
|---|---|---|---|
| **cellpose** | 3.1.1.3 | `>=3.1,<4` | Nucleus and cytoplasm segmentation. cellpose 4 replaced `models.Cellpose` with `CellposeModel`/cpsam and dropped `model_type='cyto3'`; there are four `cellpose.models.Cellpose(...)` call sites in `segmentation/segment.py` (lines 22, 65, 338, 378 — two GPU, two CPU-fallback), so 4.x imports fine and fails only at segmentation time with `AttributeError: module 'cellpose.models' has no attribute 'Cellpose'`. Verified absent in 4.1.1 (`requirements.txt:13-20`) |
| **torch** | 2.7.1+cu118, `cuda.is_available() == True` | `==2.7.1+cu118` | Cellpose's backend. `--extra-index-url https://download.pytorch.org/whl/cu118` only *adds* an index — plain `torch` resolved to PyPI 2.13.0, which on Windows is CPU-only (verified `2.13.0+cpu`, `cuda.is_available()` False). Naming the `+cu118` local version forces the CUDA wheel because no other index carries it (`requirements.txt:27-34`) |
| **numpy** | 1.26.4 | `<2` | Arrays throughout (~98 import sites). cellpose 3.x and the cu118 torch wheels are both numpy-1.x ABI (`requirements.txt:22-25`) |
| h5py | 3.16.0 (HDF5 2.0.0) | — | Every store file: stacks, MIPs, per-FOV analysis capsules |
| opencv-python | 4.11.0.86 | — | `phaseCorrelate`, `warpAffine`, morphology in the alignment chain (`alignment/chain.py:165,273-276,371`) |
| scipy | 1.17.1 | — | Powell/MLE fits, `ndimage` |
| scikit-image | 0.26.0 | — | Filters, morphology, `peak_local_max` (`segmentation/segment.py:4-5`) |
| pandas / openpyxl | 3.0.5 / 3.1.5 | — | `pd.read_excel` on ExperimentLayout; `.xlsx` export (`analysis/export.py:516`) |
| tifffile | 2026.3.3 | — | TIFF ingestion path only |
| matplotlib | 3.11.1 | — | All figures; Qt5Agg canvas embedded in the GUI |
| PyQt5 | 5.15.11 (Qt 5.15.2) | — | The GUI |
| psutil | 7.2.2 | optional | RAM/core measurement for worker ceilings and cache budgets; falls back to Win32 `GlobalMemoryStatusEx` / POSIX `sysconf` (`io/preprocess.py:37-74`) |
| threadpoolctl | 3.6.0 | optional | Pins a worker's BLAS to one thread *after* numpy is imported, which env vars cannot do; also what lets tests assert the pinning (`requirements.txt:49-55`). *Correction to the harvest: it is absent from the archived conda lockfile but present in the live env — verified today.* |
| ipywidgets | 8.1.9 | — | Declared but **dead**: imported only by `legacy/chain_widget.py`, `legacy/localization_widget.py`, `legacy/segment_widgets.py`; grep finds zero live imports. Consider dropping |

Documentation builds in a separate, light env (`docs/requirements.txt`: `sphinx>=7`, `myst-parser>=2`, `furo`) because the docs import nothing from the package. Install cost is ASSERTED only — "a few GB (the CUDA torch wheel), several minutes, once" (`docs/installation.md:26-27`); never timed.

### Hardware

Reference workstation, all values MEASURED on the machine every performance number in this manuscript was taken on:

| Property | Value | Source |
|---|---|---|
| CPU | 64 logical / 32 physical, `Intel64 Family 6 Model 106` (Ice Lake-SP) | measured; `parallel.py:17-18` |
| RAM | 351.7 GB physical | measured; commit `136d9b5` |
| GPU | NVIDIA RTX 3070, CUDA available, `Cellpose(gpu=True, model_type='cyto3')` constructed and `eval()` verified on synthetic blobs | commit `5f6af71` |
| OS | Windows Server 2019 Standard 10.0.17763 | measured |
| `max_ingestion_workers()` on this box | **42** (memory-bound: `min(64−2, ⌊0.6 × 351.7 GB / 5 GiB⌋)`) | `io/preprocess.py:77-112`; commit `136d9b5` |
| Ingestion spinbox default | **4**, regardless of ceiling — "the ceiling describes what the box can host, which is a different question from what the storage wants" | `ui/ingestion_panel.py:234` |

Derived floors: `MIN_WORKER_CEILING = 4`, `UNKNOWN_RAM_WORKER_CEILING = 16` (`preprocess.py:28,34`). GPU is used only through cellpose; nothing else imports torch, and CPU fallback is automatic per call (`segment.py:55-66`). No minimum spec, minimum VRAM, or CUDA driver floor is stated anywhere — a gap the manuscript must fill. The claim that CPU-only torch "costs an order of magnitude on every segmentation" (`requirements.txt:9`) is ASSERTED; no GPU-vs-CPU timing exists in the repo.

One seam worth stating: `DAX_WORKER_PEAK_BYTES = 5 GiB` is sized for "this project's real 2048×2048×354 uint16 stacks" (`preprocess.py:15-21`, ASSERTED), but every real store measured is 1024×1024, and a real MP58 DAX is 461,373,440 B = 440.0 MiB (MEASURED). The memory limb of the ceiling is therefore sized ~6× above today's actual per-worker peak.

### Storage volumes

Verified today with `fsutil fsinfo drivetype`. **The repo and many commit messages call the project volume "NAS"; on this machine it is not.**

| Drive | Type | Holds |
|---|---|---|
| C:, D:, E:, G: | **Fixed Drive** | D: conda envs; **G: every analysis store** referenced by the shipped configs |
| Q:, S:, P:, R:, T: | Remote/Network Drive | Q: `\\143.248.14.9\Shared`, S: `\\143.248.14.6\Shared` — the **raw acquisition shares** (`layout_path` and `dax_directory` in real `manifest.json` files point at Q:) |

So the raw side genuinely crosses the network; the analysis store does not. Any timing quoted against a `G:` path is local-device contention, not network bandwidth. `E:/Students/2026-08-07-SG-test/DNA`, the store `CLAUDE.md:36-38` names as the primary measurement store, **does not exist on this machine today** (`E:/Students/` holds only `2025-06-09_JP_C7C9`); every figure attributed to it is currently unverifiable.

### Storage layout (v2, `LAYOUT_VERSION = 2`, `io/paths.py:53`)

```
<project_root>/
  manifest.json                            modality registry: layout_version + {modality: {layout_path, dax_directory}}; readable with no HDF5
  {modality}/                              storage_path == <project_root>/{modality} by contract (paths.py:8)
    stacks/fov###/{hybe}.h5                /stack/ch{c} uint16 (h,w,depth) z-LAST, gzip-1 + shuffle, chunked; write-once, atomic
    mips/fov###/{hybe}.h5                  per-hybe MIP, one dataset per channel; EXISTENCE == INGESTED (the flag file)
  analysis/
    params.json                            experiment params: {"shared":{...},"modalities":{name:{...}}}
    psf.json                               the PSF values actually used by a tracing run
    celltype_config.pkl                    pickled: tuple keys + per-FOV float maps are not JSON-shaped
    fov###/
      cells.h5                             columnar cell table + mask geometry; full atomic replace
      alleles.h5                           columnar allele table; full atomic replace
      spots/{mod}__{hybe}__ch{c}.h5        one file per (modality, hybe, channel) — `__` is reserved in names
      matrices__{modality}.h5              one 3x3 float32 dataset per hybe; the only read-merge-write file
      crossmodal.json                      {bridging_modality: {matrix, z, quality}}
      expression.json                      cached computed cell attributes
      manifest.json                        per-FOV flag file: counts, aligned hybes, spot-uid counter; written AFTER its data
  figures/{modality}/{category}/fov###/*.png   categories: alignment, cells, alleles, analysis; 'shared' if modality-less
```

Three-digit `fov###` is enforced — `_require_fov3` (`paths.py:168-190`) refuses a store still carrying `FOV##`, because Windows' case-insensitivity hides the case change but not the digit count and every FOV would silently read as un-ingested. `paths.py`'s own docstring omits `celltype_config.pkl` and `psf.json`, both present in real stores.

### Operator inputs

**1. `ExperimentLayout.xlsx`, one per modality**, supplied by the operator (or generated by the TIFF path) and living outside the store. Parsed by `preprocess.parse_experiment_layout` (`preprocess.py:134-195`), one record per hybe:

| Column | Type | Meaning |
|---|---|---|
| `FolderName` | str | Hybe/round folder (`Hyb_002`, `Rep_010`, `Toe_080`, `DAPI`); also the sub-directory under the DAX directory |
| `Readouts` | int | Readout index — informational only |
| `DataType` | str | `H` hybe / `R` repeat / `T` toe / `B` barcode — informational only; **alignment must never branch on it** (`preprocess.py:139-142`) |
| `HybNum` | int | Ingestion order within the modality |
| `channels` | `"[555, 635]"` | The authoritative per-hybe channel set — "never a fixed 405/488/555/635 set" |
| `fiducialChannel` | int | Which of those is the fiducial |
| `channelLayout` | str | Parsed, round-tripped, **never branched on** — de-interleave always assumes alternating (`dax[:,:,cid::len(channels)]`). A non-`alternate` layout would be silently mis-de-interleaved |
| `totalFrames` | int | Authoritative for depth: `expected_depth = total_frames // len(channels)` (`preprocess.py:530`). Read per hybe, never assumed uniform — barcode hybes at 354 frames vs 260 for regular ones in one real store. A DAX whose real depth disagrees raises |
| `rnaNames` | str, optional | Target name; DNA layouts omit the column, so target rounds are anonymous |

`bufferFrames` and `FirstFrameActive` are written by the TIFF generator (`tiff_ingestion.py:312-338`) and never read back. Parses are mtime-keyed, max 8 entries (`preprocess.py:164-194`); a miss costs ~46 ms at 16 hybes, ~300 ms at 111 hybes over the acquisition share (MEASURED, commit note at `preprocess.py:157`).

Real layouts, MEASURED: MP58 DAX layout 76 rows (H=62, R=7, T=4, B=3), `channels` uniformly `[555,635]`, `totalFrames` 220; MAZ generated layout 135 rows (H=108, R=17, T=10) with per-round channel sets `[555,635]`/`[475,555,635]`/`[475,488,390,555,635]` at 258/387/645 frames.

**2a. DAX source** — a repository root with one sub-directory per hybe folder. The filename is **constructed, not discovered**: `<dax_directory>/<FolderName>/ConvZscan_{fov-1:02d}.dax` (`preprocess.py:549`). Note the trap: **FOVs are 1-based in the app, 0-based on disk**, and there is no configuration hook for the pattern. A `.inf` sidecar of the same stem supplies `frame dimensions`, `number of frames`, `data type` — the only three fields read (`preprocess.py:290-330`).

**2b. TIFF source** — `<experiment dir>/<trial dir>/{opener}_Pos{fov:0Nd}__{job}_{index}_RAW_ch{cc:02d}.tif`. `index` is the microscope's own acquisition counter and varies per FOV and session, so **filenames are discovered by regex, never constructed** (`tiff_ingestion.py:148-153`). `cc` is the 00-based acquisition-order slot; the operator names channels in that order. Pages are round-major (`page = round*Z + z`), and the operator supplies an ordered readout-code list per trial (`n`/`H7` → `Hyb_{n:03d}`; `-n`/`R5` → `Rep_`; `1000+n`/`T10` → `Toe_`; `2000+n`/`B130` → barcode; `0` skips; `a-b` ranges; a bare name such as `DAPI` → `Hyb_DAPI`). Both paths write z-LAST `(h, w, depth)` — `preprocess.py:517-519` still claims the TIFF path writes `(depth, h, w)` and is **wrong**; `tiff_ingestion.py:497-501` transposes explicitly.

**3. Config XML** (`configs/*.xml`, written by `preprocess.make_xml_file:197`). Exactly three global attributes on `<settings>`: `project_root` (required; load fails without it), `fov_list` (comma/whitespace, ranges accepted in the UI), `celltype_names`. Then one `<modality name="...">` per modality — **name only**, order semantic, the first being the cross-modal hub; `layout_path`/`dax_directory`/`storage_path` live in `manifest.json` instead. Then one element per stage: `<acquisition>`, `<cell_segmentation>`, `<fov_alignment>`, `<cross_modal_alignment>`, `<cell_alignment>`, `<spot_localization>`, `<celltype>`, `<chromatin_tracing>`, `<analysis_population>`, `<analysis_polymer_qc>`, `<analysis_gate>`. The authoritative field list is `_CONFIG_PARAM_MAP` (`windows/main_window.py:12865-12990`); a round-trip test on the real MP58 store covers **90 parameters** (`tests/test_config_roundtrip.py`). Design rule: "biological/analysis parameters only — navigation and view state is not configuration" (`main_window.py:12855-12864`).

**4. Voxel size**, on `<acquisition>`: `voxel_xy_um` (lateral µm/pixel) and `voxel_z_um` (axial µm/plane), independent. Defaults `0.208 / 0.2` (`ui/ingestion_panel.py:11`) with the comment "Defaults are this lab's own; every store should set its own." Committed configs use z of 0.2 (MP58), 0.145 (MAZ), 0.202 (HoxA) — verified. It is deliberately a `QLineEdit` plus an explicit Apply, not a spinbox, because a scroll wheel over a spinbox in a scroll area would change it silently and "a wrong voxel size cannot be spotted in the results" (`ingestion_panel.py:141-160`).

### Not portable

| Thing | Where | Note |
|---|---|---|
| `G:\<person>\<date>-<experiment>` project roots | all 12 committed `configs/*.xml` — `G:\Seonghyeok\2025-11-30-MP58`, `G:\JP\2026-01-26-cross_modal`, `G:\Seeun\2026-06-17_MAZ_IF_15ug`, `G:\Dariya\...`, `G:\Junyoung\...` (one is a Mac path, `/Users/hanmanhyuk/...`) | Every shipped config is unusable elsewhere |
| `Q:`/`S:` acquisition shares baked into every store's `manifest.json` and into each stack's `path` provenance attr | e.g. `Q:\Seonghyeok\ORCA3\20251130_MP58_RNAandDNA_\TiftoDaxOutput` | Moving a store to another site breaks re-ingestion provenance |
| `E:/Students/2026-08-07-SG-test/DNA` in every analysis doc | `docs/analysis/*.md` (9 files) | Every documented example points at a store that is gone from this machine |
| DAX filename `ConvZscan_{fov-1:02d}.dax` and `<dax_dir>/<folder>/` nesting | `preprocess.py:549` | No configuration hook; a differently-named acquisition cannot use the DAX path at all |
| TIFF filename pattern and round-major paging | `tiff_ingestion.py:10-21` | One microscope's saving convention; the older interleaved convention was removed by explicit decision |
| Voxel defaults 0.208 / 0.2 µm | `ui/ingestion_panel.py:11` | The code says so itself |
| `channelLayout` assumed `alternate` | `preprocess.py` de-interleave | Portability hazard: any other layout mis-de-interleaves silently |
| Conda discovery probes `D:` and `E:` roots | `launch_codelab.bat:99-105` | Added because this machine's C: filled at 98%; env lives at `D:\conda-envs\code_lab_imaging_pipeline` |
| `.codelab_python` and `CODE Lab Imaging Pipeline.lnk` present in the working tree | repo root | Both gitignored, both naming absolute paths on one machine |
| PSF library, 5 tracked JSONs | `psf/` — `MP58-`, `Chr19-`, `HoxA-`, `CrossMod-`, `universal-default` | Tracked **on purpose** so calibrations accumulate (`localization/psf_library.py:14-18`). `universal-default`: `sigma_xy_um 0.13745`, `sigma_z_um 0.46949`, `halo_frac 0.18019`, `halo_scale 2.5597`, `voxel_um [0.208, 0.208, 0.2]`, mean of 3 converged experiments, between-experiment spread 28.2 nm vs ~40 nm within-experiment reselection noise (MEASURED, file contents) |
| `celltype_names` such as `WT,4A3,8B1` | `configs/*.xml` | Per-experiment, and correctly config-driven |
| **Portable, worth saying so** | — | Channel numbers. No hardcoded 405/488/555/635 list exists in live code; channels come from the layout per hybe (`preprocess.py:138-139`) |

---

## Procedure

Verified against `windows/main_window.py`, `ui/*.py`, `codelab_pipeline/**` at HEAD `e98b284`/`a36388c` (2026-09-04). Tab order is fixed: Ingestion → Cell Segmentation → Alignment → Spot Localization → Celltype Determination → Chromatin Tracing → Analysis (`ui/main_window_ui.py:37–67`); default window 820×620, panels in scroll areas (`:25`, `:96–102`).

**Storage note for the whole Procedure.** Every store used for measurement here sits on a *local fixed disk*: `fsutil fsinfo drivetype` today reports C:, D:, E:, G: as Fixed Drives and only Q:, S:, P:, R:, T: as Remote/Network. Numerous code comments and commit messages call these stores "the NAS" and are wrong about it; where throughput matters, the measured rate is given instead of a label. Note also that `E:/Students/2026-08-07-SG-test/DNA`, the store named in `CLAUDE.md` and behind most alignment measurements, **no longer exists on this machine** (verified `Test-Path` → False, 2026-09-04), so those figures are not currently reproducible.

### Stage 1 — Launch and session setup ● TIMING ~0.4 s after first run

1. Double-click `launch_codelab.vbs` (Windows, no console), `.bat`, `.command` (macOS), or the shortcut. Interpreter resolution: `CODELAB_PYTHON` → cached `.codelab_python` → conda env `code_lab_imaging_pipeline` (legacy name `cellclassifier` accepted) → offer to build from `requirements.txt` → conda base → `python` (`docs/installation.md:29–60`). MEASURED (233e68e): cold resolve+validate+cache **7.4 s once**; every later launch **240–360 ms**.

2. Answer the modal **"Load configuration file (Cancel to start fresh)"**. MEASURED (233e68e): this dialog appears in **99 ms**; the ~1555-module, ~2.5 s import is deliberately paid after the answer (`main.py:100–112`). Loading a config performs Steps 4–9 automatically — it requires `project_root`, reads `manifest.json`, derives every `storage_path` as `<project_root>/<modality>`, rebuilds and locks Modality Setup, parses every layout, applies params last (`main_window.py:13184–13299`).

3. The log window opens first, main window on top. First line: `Worker-process guard: <state>`. Lines are timestamped `2026-08-24-11:14:25-<message>`, bounded at 20k (`ui/log_window.py:5–9`); re-open with **Show Log** (`main_window_ui.py:77`).

### Stage 2 — Declare the experiment (Ingestion tab)

4. ▲ **CRITICAL STEP.** Type N into **# modalities** and press **Set** — which permanently disables itself and the count field (`main_window.py:2499–2500`). Defaults: n=1 → `RNA`; n=2 → `DNA,RNA`; n>2 → blank (`ingestion_panel.py:345–367`).

5. ▲ **CRITICAL STEP — irreversible.** Name each modality (non-empty, unique) and press **Activate Modalities**. Log: `Activated modalities: DNA, RNA` (`main_window.py:2513`). **The order is semantic: the first activated modality is the cross-modal hub frame** (`main_window.py:10589–10604`), and its own `H_across` is identity by definition. `lock_modality_setup()` disables the whole group and **nothing in the tree re-enables it** — verified today: the only `setEnabled` calls on those four widgets are the four `False` calls at `ingestion_panel.py:409–413` and `main_window.py:2499–2500`. Changing the count or names requires restarting the app.

6. Set **Project root (storage)**. `storage_path` is derived, never typed: `<root>/<modality>` (`main_window.py:1935–1953`). A blank root yields `''` for every storage path and every downstream stage silently skips.

7. Fill **`<name> ExperimentLayout.xlsx`** and **`<name> DAX directory`** per modality. Confirming a DAX field re-probes only that modality's checklist rows (`ingestion_panel.py:468–487`); Browse emits `editingFinished` so it behaves like typing (`:333–338`).

8. Enter the **FOV list** (`1,2,3`, `1-10,15,20-25`, or space-separated; order preserved, duplicates dropped — `main_window.py:13353–13379`). ▲ This single field is the authority on which FOVs every batch button in every tab may touch.

9. ▲ **CRITICAL STEP.** Set **Voxel size** lateral/axial and press **Apply**. Defaults 0.208 µm/px and 0.2 µm/plane (`ingestion_panel.py:11`). Deliberately a text field, not a spinbox — a wheel over a spinbox inside a scroll area would silently change it, and a wrong voxel size is invisible in every downstream result (`:141–160`). Apply *refuses* out-of-range values rather than clamping; the label reads `in force: 0.208 x 0.208 x 0.2 um` or `NOT applied -- <why>` (`:579–606`). Nothing reads the boxes until Apply. **`docs/happy_path.md` never mentions voxel size at all** (doc gap J10).

10. Press **Parse Layouts (all modalities)**. Creates `<root>/<name>`, declares each modality, writes one `manifest.json`, records `layout_path`. Log: `Parsed 111 hybe(s) for DNA from <path>` (`main_window.py:2601`). A modality with a bad workbook is reported and the others still parse (`:2547–2556`). This enables **Run Ingestion** (`:2645`). MEASURED: active-hybe sweep 2.77 s cold for 34 FOVs, 17 ms warm (`:2647–2651`).

### Stage 3 — Ingestion ● TIMING: no per-FOV or per-hybe figure is recorded anywhere (gap)

11. In **Hybes to ingest**, review the checklist. Rows are keyed `(folder, modality)` because folder names legitimately collide across modalities — the cross-modal bridge hybe appears in both layouts (`ingestion_panel.py:420–433`). Rows default-checked iff that modality's own DAX directory holds real `.dax` files. To ingest one modality only, uncheck the others' rows.

12. Set **Parallel workers**. Default is `min(4, ceiling)` (`ingestion_panel.py:234`); ceiling = `min(cores−2, 60 % RAM ÷ ~5 GB)` capped at 64 (`preprocess.py:77–112`). MEASURED on a real store: **12 workers 117.6 MB/s, 36 workers 66 MB/s** (`main_window.py:3040–3042`). ? **TROUBLESHOOTING / doc discrepancy J7:** `happy_path.md:§1` reads as if the app uses 12; it does not — an operator following the doc gets 4 unless they type 12. The code comment justifying the 4 default (`ingestion_panel.py:226–231`) argues the ceiling describes the machine, not the storage.

13. Press **Run Ingestion**. One worker job per checked modality, one shared spawn `ProcessPoolExecutor`, task list **FOV-major across jobs**, so FOV *n* completes in every modality before FOV *n+1* begins (`main_window.py:143–168`). The run refuses if another run or queued job targets the same store; different stores may ingest concurrently (`:2998–3009`). Log: `Starting ingestion (append) across 2 modalit(ies) (DNA, RNA): 34 FOV(s), 7548 task(s), demultiplexed FOV-first across every checked modality.` (`:3021–3023`), then `FOV003 Hyb_004 (DNA): OK` per task.

14. ▲ **CRITICAL STEP — the Overwrite/Append dialog.** `_confirm_overwrite` (`main_window.py:3080–3114`) is **skipped entirely when nothing exists**, so a first run never sees it. Buttons `Overwrite All` / `Append (skip existing)` / `Cancel`, **Append is the default button** (`:3107`).

| Mode | What it computes | Consequence |
|---|---|---|
| Append | targets whose stack fails `stack_is_complete` — channels present *and* first/last element readable (`preprocess.py:532–547`) | a damaged file rebuilds instead of being skipped forever; nothing already-good is re-read |
| Overwrite | re-converts everything: `os.remove(stack)` then a **non-atomic** rebuild | ▲ interactive overwrite first raises `_clear_for_overwrite_ingestion` (`:1658–1686`) and terminates every store-reading worker (12 tracked attributes, `:1598–1602`) plus every pool child; a *queued* overwrite does this silently, the queue having been authorised as a whole (`:3982–3987`) |

■ **PAUSE POINT / stop cost.** Ingestion persists per task: `.part` stack → `os.replace`, then the atomic per-hybe MIP whose existence is the completeness flag. A stop costs **at most the one in-flight (FOV, hybe)**, and an append re-run fills exactly the holes. ? **TROUBLESHOOTING — doc discrepancy J9:** `happy_path.md:§1` claims "every write goes through an atomic door… an interrupted run never leaves a half-written capsule". True for analysis capsules, MIPs and appended stacks; **false for an overwrite-mode stack rebuild**, which is exactly the damage `tools/verify_store.py` exists to find. After any interrupted overwrite, run `python tools/verify_store.py <storage_path> [--workers 8] [--list-bad]` (`tools/verify_store.py:1–31`). The app's own status check only lists the MIP directory.

15. (Optional) **Add to Queue / Run Queued Jobs.** Add snapshots the whole checked set, so later form edits cannot change what a queued entry means (`:3880–3889`); the queue asks Overwrite/Append **once for all entries** (`:3926–3956`). A failed job **aborts the remaining queue** (`:4023–4040`).

16. On completion `_on_ingestion_finished` (`:3133–3173`) rescans status, refreshes active-hybe lists, activates the current FOV, refreshes both alignment result lists and celltype config. ? **TROUBLESHOOTING:** the modal reads `Ingestion complete / Ingestion finished successfully (N task(s))` even when **every** task failed — `finished_ok` fires regardless; read `n_errors`, not the dialog title (`:104–110`, `:3158–3173`). Verify with **Check Ingestion Status** (`FOV001: 108/111 ready | MISSING: Hyb_109, …`, `:3217–3256`), the **MIP Viewer**, and **Show Cell/Spot Status Detail…**, which reads what is *persisted*, never session memory, and carries the **Export** button (`:3332–3448`).

**Running analysis while ingestion continues.** ▲ Supported by design, and gated. `_incomplete_fovs` (`:3484–3527`) takes a description of what a run will actually READ and excludes any FOV it can *positively show* is unready; a check that cannot be made never excludes. `_report_excluded_fovs` (`:3529–3542`) logs e.g. `Run FOV Alignment (all FOVs): EXCLUDING 6 FOV(s) whose ingestion is unfinished -- FOV029 (DNA: 3 of 111 hybe(s) not ingested yet); …` Gate coverage: FOV alignment and cross-modal **overwrite only**; cell alignment **both modes**; tracing **append only**. MEASURED: the readiness question costs **7 ms** for a full 2-modality × 40-FOV sweep, and zero disk access while an ingestion is live (`tests/test_analysis_during_ingestion.py:14–16`). Also live during ingestion: every hybe combo (fed incrementally by `task_done`, `:2339–2375`), and Cell/Spot Status Detail (`:3478–3482`). **Parse Layouts mid-run deliberately skips only its status scan** — that scan is a GUI-thread `listdir`/HDF5-open per FOV per modality competing with the workers, a confirmed freeze on 2026-08-24 (`:2604–2641`).

### Stage 4 — Cell segmentation ● TIMING ~1 s of compute per FOV, plus projection read

▲ **This stage is one FOV per click. There is no batch segmentation anywhere in the tree** (grep for `Segment All`/`segment_all`: zero hits). ? **Doc discrepancies J2/J3/J4:** `happy_path.md:43–47` says "then segment **the FOVs**… Append mode skips FOVs that already carry cells" — both clauses are wrong. That sentence describes cell *alignment*'s append; segmentation's Append Mode **merges the new mask into this FOV's current mask** (`main_window.py:4109–4145`). The doc also omits the **Manual** polygon method entirely (`cell_segment_panel.py:41`).

17. Set **FOV**. Selecting a FOV *activates* it — persisted cells, their spots and matrices hydrate automatically, no button (`:5051–5060`).

18. Pick **Reference hybe / Channel**. The combo carries every modality's hybes tagged `(folder, modality)`; channel defaults to the first **readout** channel, never the fiducial (`cell_segment_panel.py:273–294`).

19. Choose **Method** — Cellpose (Diameter 40 px, Min 1000, Max 10000), Classical (Otsu/Yen/Li/Triangle/Absolute; min distance 7 px, min 500 / max 10000 px), or Manual (empty mask, polygon draw pre-armed: left-click vertices, `A` commit, `D` undo). ? Note the UI's classical min-size default (**500**, `cell_segment_panel.py:181`) differs from the library signature default (**1000**, `segment.py:94`); the GUI always passes its own.

20. ▲ **CRITICAL STEP — choose the projection.** Options: `MIP (stored)` (default; the only mode that touches no raw stack), `single plane`, `range MIP`, `range mean` (`segment.py:146`). Press **Detect Focal Plane** to plot variance-of-Laplacian per z; on completion the GUI sets the range to the ≥0.9×peak plateau and **switches the mode to `single plane`** (`main_window.py:4340–4359`). MEASURED, same FOV/hybe/channel/params: stored MIP **33 cells**, single plane z=76 **91 cells** (`segment.py:155–158`); independently replicated on `G:/Dariya/2026-08-22-RNA_DNA_HoxA` FOV001: MIP **77**, single plane z=71 **103**, range MIP z=46–78 **98**. "The middle plane is the focal plane" is measurably false — 130-plane stack, middle z=65, sharpest z=**76** (`segment.py:174–179`).

21. Press **Run Segmentation [\<projection\>]** — the button restates the live projection so the label, the confirmation and the log line cannot disagree (`cell_segment_panel.py:241–243`). A dialog blocks if a focal plane was detected this session *and* the mode is still `MIP (stored)`; that is the only combination that fires it, so it cannot become click-through noise (`main_window.py:4395–4419`). MEASURED on G: (local fixed disk, RTX 3070, cellpose 3.1.1.3 + torch 2.7.1+cu118): model load **9.2–10.7 s once per process**; first eval **3.0 s**; steady state **1.01–1.08 s**; classical otsu/yen/li/triangle **1.03 / 0.65 / 0.69 / 0.33 s**. Projection read: warm stored MIP **0.33 s**, cold single plane z=71 **18.4 s**, cold range MIP (33 planes) **17.2 s**. No CPU-only Cellpose timing exists anywhere (gap).

22. Correct the mask in **Show Cell Displayer**: remove ids (`1-10`, `1,2,3`, `1 2 4 5`), **Remove Edge Cells** (a frame-clipped cell has a partial mask, inward-biased centroid and arbitrary area, inherited by alignment and analysis — `cell_displayer.py:237–268`, 420e9bd), Undo/Redo. Optionally run **Cytoplasmic Segmentation…** after nuclei exist, against a different and often other-modality hybe: **Preview Nuclei → Run → Incorporate** (`main_window.py:4545`, `:4571`, `:4641`). Note: **`has_cytoplasm()` is false for all 3953 persisted cells across three real stores** — the route works but has never been used on production data.

23. ■ **PAUSE POINT.** Press **Save** (Transient Cell Container). It writes the FOV **on screen** (the spinbox), not the last-displayed one — 16d2e0a fixed a case where the panel showed FOV004 and the dialog named FOV002 (`:4976–4995`). Save **also re-runs spot reassignment for the whole FOV and persists every slice** (`:5013`). Log: `Saved 73 cell(s) for FOV004 to permanent container and the analysis store (<paths>).` (`:5020–5021`). ▲ Cells are a whole-FOV atomic **replace**: everything since the last Save is lost on a stop, and a FOV whose existing content was not loaded this session must never be saved over. Repeat Steps 17–23 per FOV.

### Stage 5 — Alignment (three independent layers)

▲ Panel rule (`ui/alignment_panel.py:14–21`): **current-FOV / current-cell buttons stage a result with Accept/Reject; all-FOV / all-cell buttons always compute AND save immediately, no staging.** ? **Doc discrepancy J11:** `happy_path.md:§3` mentions none of the staging model, the Overwrite/Append dialog, or the readiness gate — the three things that decide what a click does. ▲ No layer blocks any other: an absent layer is **identity**, never a blocker (`:11262–11268`, `:12655–12661`); the status bar says `No FOV alignment for this FOV yet -- proceeding with identity.` (`:11268`).

24. **Same-modality (FOV).** Set **Reference hybe** *per modality* (one combo each — alignment is per-modality maths), Border trim (0 px), Max shift (0 = unbounded), **Overlay channel** (`fiducial`). ▲ The Overlay channel is **display only**; matrices are always fitted fiducial-to-fiducial from `record['fiducial_channel']` regardless (`alignment_panel.py:62–77`; `chain.py:828–830`).

25. **Run Current FOV Alignment** → the all-readouts overlay pops immediately, **Accept/Reject** enable (`:9631–9732`). **Accept** writes the per-modality reference params, the matrices per (FOV, modality), queues overlay PNGs, then **re-casts the accepted FOVs' persisted spot coordinates**; dialog `Matrices written to H5; overlay image(s) saved.` (`:9963–10019`, `:10009`). **Reject** discards; nothing was written.

26. **Run All FOV Alignment (Auto-Save).** Readiness gate (overwrite mode only), then Overwrite/Append, then per-FOV compute+save with each overlay drawn as its FOV lands (`:9739–9848`). Append grain is per **(FOV, hybe, modality)**: hybes ready on disk with no persisted matrix (`:9764–9802`). ■ Persistence is **per FOV** as `fov_done` fires; a stop costs the in-flight FOV only. MEASURED: ~**3.5 s per hybe** serial on 1024×1024 MIPs, ~93 % of it inside Powell (`chain.py:708–711`) ⇒ ~4.5 min for a 78-hybe FOV; the per-FOV pool gives **48.6 s → 10.2 s = 4.8×** on 16 real hybes with **bit-identical** matrices (54260f1).

27. **Cross-modal.** The **Shared frame (hub)** label is read-only (the first activated modality). Pick a **Bridge hybe** per non-hub modality; Channel type defaults to `readout` here, because the two imaging sessions have no generally shared fiducial (`chain.py:1133–1146`). ▲ **Z drift is deliberately not a control** — it is measured by the run, shown in Results, persisted by Accept (`alignment_panel.py:134–139`). Then **Run … Current FOV** → Accept/Reject, or **Run … All FOVs (Auto-Save)** (append grain: per (moving modality, FOV), requiring both bridge hybes ingested — `:10188–10269`). MEASURED: whole per-FOV cross-modal **~57.7 s → ~2.3 s (~25×)** after 5357ba5, of which the discarded focus-profile pass was 75 % of the old runtime. ? The Z leg has **no automatic quality gate**; its own docstring says to gate on `quality` and no threshold exists in code (gap).

28. **Cell-level, tier 1 — the single-cell preview/accept loop.** ▲ Set **FOV**, **Cell ID**, **Reference hybe** per modality, **Channel type** (`fiducial` default — here it selects what is actually *fitted*, both the YX phase correlation and the Z leg), **Max Z shift** (default 15 planes), **Pad** (10 px, which is also reject gate G6's threshold). There is no modality picker: the resolved cell tells you its modality.

29. **Preview This Cell's Alignment** runs a background `CellAlignmentWorker` over exactly that cell, hydrating the FOV first, staging into **its own slot separate from any batch**, writing **nothing**. Results rows are tagged `[pending]`; the ref-vs-all overlay pops; status bar: `Cell 12 (FOV004) alignment previewed -- Accept to save.` (`:12583–12743`, `:12736`).

30. **Accept** applies the staged result to the real cell, persists matrices plus overlay PNG and refreshes the tier-3 lists (`:12745`); **Reject** discards (`:12837`). Iterate Steps 28–30 on a few representative cells to settle Pad and Max Z shift before committing a batch. MEASURED, one cell, 111 hybes, one pass, on a loaded store with ingestion running: **84 s, 99.5 % of it storage I/O** (`:12602–12611`).

31. **Tier 2 — commit.** **Align All Cells in FOV** or **Align All Cells in ALL FOVs (Auto-Save)** reuse the tier-1 settings; both refuse FOVs outside the Ingestion list, apply the readiness gate (**both modes** here), then ask Overwrite/Append (`:11212–11296`, `:11508`, `:11541–11560`). Set **Auto-save overlay if shift >** (default **0 px = every cell**) before starting: the all-FOVs confirmation prints the overlay arithmetic, `hours = n_cells × 35 s / 3 workers / 3600` (`:11541–11560`; `CELL_OVERLAY_RENDER_WORKERS = 3` at `:12083`). ■ Persistence is **per FOV** through `.part` + `os.replace` when that FOV's last cell lands, plus a sweep at finish (`:11657–11662`, 5b27578); a stop costs the in-flight FOV only. ▲ Before 5b27578 persistence happened only after the *last* FOV — a run stopped during FOV006 discarded FOV002–005 entirely (confirmed on a live store). MEASURED (3b25b8e, real 134-hybe store, cold FOVs): 6 workers 4.87/4.00/4.37 s per hybe, 3 workers 3.60/2.53 → median **4.37 → 3.07 s/hybe, 1.42×**; `tuning.json` sets `cell_alignment_workers: 3`, re-read at the start of every run. Phase accounting (bbc0076): **~93 % of the cost is per-hybe storage access, ~7 % per-cell** — doubling the cells adds 7 %.

32. ? **TROUBLESHOOTING — append reports work already done that is not on disk.** Two recorded causes, both fixed: a run that mutated `ACell` objects then died before writing left matrices only in-process, and append believed them — **restarting the app clears it, and that is the tell** (9b54fda; `_persisted_cell_matrix_keys` now reads the store); and `read_cells` returning `(None,'')` for both "never written" and "unreadable" routed a never-written FOV to the memory-trusting branch (e98b284). The skip line now names its evidence: `append: 769 cell(s) with nothing to fit -- skipped` (`:746`) plus one worked example per FOV.

33. **Tier 3 (read-only).** Overlay FOV, Results (per cell, overlay), Preview reference hybe, **Save All Cell Overlays**. Clicking a saved overlay **shows the PNG** rather than recomputing a ~35 s composite (110 ms per ZX crop × 3 per hybe at ~100 hybes). ■ Interrupting overlay rendering costs **only pictures** — every matrix is already on disk; quitting warns explicitly (`:1541–1592`).

### Stage 6 — Spot detection and fitting ● TIMING: no wall-time for auto-detect is recorded anywhere (gap)

34. Pick **Hybe / Channel** (one combo over every modality, tagged `(record, modality)`; the channel survives a hybe change — `spot_localization_panel.py:328–354`), then **FOV**, then a row in the **View** list. ▲ Row 0 is always the FOV pseudo-row (the unassigned pool); rows 1+ are cells. **This list is the scope selector; there is no Scope combo** (`:10–17`, `:356–396`).

35. Set **Threshold (% of scope max)** = 50, kept live-synced with **Absolute value** (absolute wins when non-empty), **Min distance** 3 px, **Cell crop padding** 10 px (Cell view only) (`:398–420`).

36. Press **Run Auto-Detect** (+ **Append Mode** to add rather than replace). The shipping 2D path is a plain `skimage.feature.peak_local_max` over a MIP — there is **no Gaussian sub-pixel refinement in 2D**; `localize_cell_2d_worker` (`localization.py:769`), the only place `fit_gaussian_2d` is reachable, has no caller (dead code, gap). ▲ In Cell view a spot gets `cell=<id>` immediately; **in FOV view every spot stays `cell = -1` — ownership is decided at save time, not detect time.** Log: `FOV004 Hyb_012 ch635: 118 peak(s) detected (unassigned -- run Save Current Spots to identify cell ownership).` (`main_window.py:7563–7564`). ? **Doc discrepancy J12:** `happy_path.md:§4` ("assigned to cells where masks say so") hides this; an operator who detects and moves on loses the association. Detected z is `0.0` — a placeholder, not a measurement (`spot.py`; `reconcile.py:26–30`).

37. (Optional) **Show Spot Crop Displayer** to add/remove by click in either view; **Revert / Clear This Hybe/Channel**, **Undo/Redo** over view snapshots.

38. (Optional) **3D Localization…** adds Z to already-placed spots; it never detects (`localization.py:483–490`). Engine defaults `v2`, `spad=5 px`, `v2_min_occupancy=0.40`, `v2_max_uncert_z_nm=150` (`canvas/localize_3d_displayer.py:13–21`). ▲ **Not a prerequisite for tracing** (`chromatin_tracing_panel.py:96–100`). It must run in a child process, not a thread: MEASURED, h5py's per-process lock turned a **16.5 ms MIP open into 2043 ms** in a GUI thread (`localization.py:1457–1460`).

39. ■ **PAUSE POINT — spots persist on explicit Save only.** **Save Current Spots** writes every cell's in-memory spots for this FOV plus the identified unassigned pool, **spots only, never cells** (`:6682–6695`). **Save ALL FOV Spots (every hybe/channel)** reassigns every spot against current cells and persists every `(modality, hybe, channel)` slice, clearing stale ones; the log prints the real elapsed — `FOV004: ALL slices saved -- 4211 spot(s) across every hybe/channel (118 unassigned spot(s) newly identified into 61 cell(s); 3.4s). Cells are never written here.` (`:6673–6676`; the "3.4s" is illustrative — no representative value is recorded, gap). MEASURED: assignment itself is **138.7 ms for 100 000 spots** over 102 cells at 1024² (9a3d5e8), which is why it is simply re-derived whenever cells or matrices change. There is **no across-FOV batch spot detection in the GUI**.

### Stage 7 — Celltype determination

40. Build the **Celltypes** list, choose **Mode** — `FOV` or `Barcode` (**Barcode is the default**). ? **Doc discrepancy J5:** `happy_path.md:§5` says "barcode readouts **and/or** FOV ranges"; they are **exclusive radio modes** and one run uses exactly one (`main_window.py:9310`). In Barcode mode assign a `datatype == 'B'` hybe/channel per celltype, pick `Vote (200-sample)` or `Median`, and press **Apply Calibration…** (Quantile lower 0.3, upper 0.999, scale 1.0). Assigning a channel now calibrates it immediately.

41. Leave **Also classify unsaved (transient) cells** checked and press **Run Celltype Determination**. ▲ The run **hydrates every FOV in the Ingestion FOV list first** — a FOV never viewed this session has nothing resident, and the earlier version silently classified only the FOVs the operator had looked at (`:9276–9282`). Execution is FOV-parallel in processes, `max_workers=min(6, n_jobs)`. Results persist onto **permanent cells and spots** immediately. Celltype `''` is a first-class group named **Unassigned** downstream, never dropped (`analysis_wiring.py:544–547`). ● TIMING: nothing measured (gap).

### Stage 8 — Chromatin tracing ● TIMING 113.9 s per FOV at 127 alleles × 73 hybes, 32 workers

42. **§1 Hybes Involved** — default-checked is `modality == 'DNA'` and `datatype in ('H','R','T')` (`main_window.py:7580–7593`). Set the **Reference hybe (drift baseline)** and the **Readout channel to trace**, then press **Activate** — the channel combo is a staging field and nothing changes until Activate; the label reads `active: auto` (`chromatin_tracing_panel.py:162–175`).

43. **§2** uses its own FOV/Hybe/Channel pickers, deliberately independent of Spot Localization's: inheriting them was a confirmed bug that let an allele's readout channel collide with its own fiducial (`:113–122`). Select spots → **Add Alleles from Selected Spots**, or **Add Alleles from ALL Spots in This Hybe/Channel (every FOV, auto-save)**, which confirms, shows a cancellable progress dialog, and **saves each FOV as it completes**; already-anchored spots are skipped so re-running adds nothing (`:7733–7838`, `:7826`). ■ Cancel keeps every FOV already saved.

44. ■ **PAUSE POINT.** **§2 Save** is a whole-FOV atomic **REPLACE**. It refuses a FOV never loaded this session ("saving now would replace it with an empty list") and asks before deleting when staged < committed (`:7936–7983`).

45. **§3 Fit engine** — `v1 (ChrTracer3 port)` or **`v2` (default)**. Selecting an engine swaps the whole parameter page and each page keeps its own values while hidden, because a pixel bound and a µm bound are not the same number (`:939–950`). MEASURED end-to-end through the app across four experiments: v2 gives **43.1–67.7 % better same-locus repeat distance while tracing 2.7–3.3× more rounds** (b0bf3f7). ? Historical: in an earlier build the engine combo was **connected to nothing** and all 14 v1 fit fields stayed enabled and labelled in pixels (fixed in b0bf3f7).

46. **§3 Readout PSF (v2)** — default `universal-default`, justified because the between-experiment spread (**28 nm across 4 experiments spanning 64× in genomic scope**) is smaller than the **~40 nm** a single experiment moves when its calibration crops are reselected (`:47–52`). The chosen entry is copied into `<project>/analysis/psf.json` **when tracing runs** — not when previewing (b0bf3f7). A label missing from this machine's `<repo>/psf` library is shown as `MISSING FROM LIBRARY` and v2 falls back to free sigma and says so (`:992–1008`). ● TIMING for **Fit Readout PSF…**: MEASURED 40 crops → fiducial 4.6 min, readout **31 min**, once per experiment.

47. Set the cross-mode parameters: **Crop half-width** 8 px; **Max fiducial drift vs. reference** 7.0 px; **Max fiducial Z drift** 15.0 planes — gated separately because a fit can pass the XY bound while landing 20 planes off at only 1.4 px XY (confirmed real, ced2957) (`:58`, `:361–366`). ▲ The gate and the fit bound are the same number and must move together: a 7 px gate against a 5 px bound gates on a quantity the fit cannot produce (cc08f4b).

48. **§4** — select exactly ONE allele, then **Fit + View Crop (re-fits this allele)** or **View Stored (no re-fit)**. View Crop mutates the transient allele, so §2's **Save** persists it; there is deliberately no save button here (`:442–468`).

49. **§5 Fit All FOVs.** Stages each FOV's alleles, refreshes each allele's anchor from its source spot's current position, hydrates the FOV, then asks Overwrite/Append. Append grain is **membership**: alleles staged but not committed *with a trace* (`has_traced`, not `has` — an Add→Save allele has an empty `polymer_adj` and must stay reachable); a FOV whose checked hybes are not all on disk is **skipped whole**, logged as `Fit All FOVs (append): skipped FOV 012, 013 -- not every checked hybe has landed on disk yet.` (`:8960–9006`). Status on completion: `Fit All FOVs done -- 3120 allele(s) across 34 FOV(s), saved to the analysis store.` (`:9072`). ■ Persistence is **per FOV**: promote per allele, then write the whole permanent tier (`:9037–9063`). ▲ Writing `alleles` directly would be a whole-FOV replace fed a subset — in append mode that list holds only the newly traced alleles, so **every previously committed allele in the FOV would be deleted silently, with no error and no undo** (`:9043–9049`). MEASURED (e42d92a, MP58 FOV1, 127 alleles × 73 hybes, 64-CPU machine, 32-worker pool): serial 36.4 min projected; pooled with the shipping dual-fit fallback **113.9 s**, ≈**19×** (conservative — the serial baseline ran warm); the fallback retry costs **+16.8 % wall ≈ 16 s per FOV**.

50. ? **DEAD CONTROL — verified today.** `ui/chromatin_tracing_panel.py:474–479` creates **`FitThisFovPushButton` ("Fit This FOV")** with a tooltip, and `main_window.py:1890` connects it to `_run_chromatin_tracing_fit_this_fov` (implemented at `:8869–8877`) — but `fitAllLayout` adds only `FitAllFovsPushButton`, `ProgressBar` and `StatusLabel` (`:481–486`). A scan of every `QPushButton`/`QComboBox`/spin/check/line/list widget across `ui/*.py` found this as **the only widget created and never added to any layout**. Introduced in 44eb948. **The per-FOV tracing workflow is therefore unreachable from the GUI**, and its tooltip advertises exactly the mid-ingestion use case the readiness gate was built for.

### Stage 9 — Analysis and export

51. **§1** — leave **FOVs** blank to use the Ingestion list. Press **Check All With Spots**, or leave nothing checked and press Build, which does it for you (`analysis_wiring.py:210–216`). Leave **mask-based intensity** checked (labelled "slower", with no number recorded — gap); leave **overwrite cached cell attributes** unchecked unless spots were re-detected or matrices re-fitted (`analysis_panel.py:165–172`).

52. **Build / Refresh Population** runs in an `FnWorker` with the HDF5 extraction fanned out to child processes; **rebuilding clears QC** (`analysis_wiring.py:244–251`). MEASURED (6155883): `Population.build` **2.0 s**, `dmaps` **0.10 s** — but at 4 FOVs / 386 cells / 426 alleles, and predating mask-intensity-by-default. **No build time at 34 FOVs exists** (gap).

53. **§1b QC** — **Derive Thresholds (quantiles of this population)** fills min/max amplitude, `max jump (um)`, `max median dist (um)`; `min traced bins` defaults to 2. Edit any field then **Preview QC** — an unparseable edit **refuses**, it does not silently revert (`analysis_panel.py:440–453`). ▲ Changing the **distance selector** (`XYZ (3D)` / `XY (in-plane)`) **clears QC**, because the thresholds are µm lengths specific to one metric (`analysis_wiring.py:345–361`). MEASURED on MP58: `max_jump` **0.885 µm (xyz)** vs **0.614 µm (xy)**; ensemble median **0.617** vs **0.438 µm**. **Isotropy QC** deliberately ignores the selector and always uses both axes, or it would compare XY against XY and certify isotropy (`:778–789`); MEASURED on MP58 (12 FOVs, 1572 alleles, 664 654 pairs) factor **1.516** — Z spread 52 % wider than in-plane. ? **Doc discrepancy J8:** the distance selector and Isotropy QC postdate the docs (b09e0e0, 09a3219) and appear nowhere in `happy_path.md`.

54. **§2 Gate** — pick a **Kind** (`ExpressionRange`, `PairDistanceRange`, `BarcodePresence`, `CompletenessRange`, `CelltypeIn`, `FovIn`, `AlleleCount`), **Preview Histogram**, **Add Condition**; **New OR Group** starts a new AND clause. The persistent Gate summary shows `gated 270/386 cells` plus sequential survivor counts and per-celltype gated/total (`analysis_wiring.py:472–489`). **§3 flags** (decompose by celltype, min alleles per map pixel = 5, per-FOV maps) multiply figures and **never re-gate** (`analysis_panel.py:327–330`).

55. Open the views — **Ensemble Distance Map**, **FOV Consistency (SCC + MSD test)**, **Allele Differences**, **Expression Histogram**, **Brightness vs Count**, **Distance Histogram**. Each opens its **own additive window**; earlier windows stay open (`analysis_wiring.py:32–33`).

56. In each result window press **Save Result… (PNG + CSV + provenance JSON)** — PNG, one CSV per underlying table, and a JSON sidecar carrying the exact gate, the sequential survivor counts and the per-celltype counts (`canvas/analysis_figure_displayer.py:84–88`). ? **Doc discrepancy J6:** `happy_path.md:§7` lists this as a step "inside the tab"; it lives on each pop-up result window. For tabular export of spots and alleles use the **Export** button in **Show Cell/Spot Status Detail…** → `spots_and_alleles.xlsx` or two CSVs (`main_window.py:3332–3423`, 69a0b92). ▲ `collect()` reads **once per FOV, not once per modality** — `analysis/` is shared between `<proj>/DNA` and `<proj>/RNA`, and reading per modality returned every spot twice (**842 reported against a truth of 421**, 69a0b92).

57. ■ **Ending the session.** **File → Save Config…** writes `project_root`, modality **order**, FOV list, celltype names and every analysis parameter; it refuses if the modalities do not share one v2 project root beside a `manifest.json` (`:13301–13351`). Note the cytoplasm panel's parameters are **not** in `_CONFIG_PARAM_MAP` — a cytoplasm run is not reproducible from a saved config (gap). Closing asks `Really quit…?` and warns separately if cell overlays are still rendering ("Quitting discards the unrendered ones — the alignment matrices themselves are already saved"); Yes terminates every worker and pool child, then `os._exit(0)` (`:1541–1592`).

**Completion notices.** Most runs now report through `_notify_complete` (`:1509–1534`): full text to the log, a 20 s headline to the status bar. MEASURED: sampling the live app at 100 Hz for ten minutes during a cell alignment put **15.6 % of the GUI thread (~93 s) inside one `QMessageBox.information`**, against 0.5 % for everything else; an earlier three-minute profile put a completion dialog at 46.7 %. Errors and decisions stay modal, and **ingestion completion is still modal** (`:3167–3173`).

**Doc-vs-code seam, overall.** `docs/happy_path.md` was last touched at f1cbe56 (2026-08-30); **21 commits touching `ui/` or `windows/` landed after it**, including the whole append/readiness rework (d5d6e63, ffab9fe, 9b54fda, 3b25b8e, e98b284) and the Export button (69a0b92). None are reflected (J13). Treat the code as authoritative throughout.

---

## Timing

**Tags on every figure:** `[M]` measured and the commit states the input; `[P]` projection — a measured per-unit rate multiplied out, labelled as an extrapolation by its own commit; `[A]` asserted, no measurement found. **Cache state** is `cold` / `warm` / **`not stated`** — most recorded numbers do not say, and on this hardware the difference reaches 50x (see the `stack_is_complete` row).

**Storage naming.** All measurement stores are on local fixed disks (`fsutil fsinfo drivetype`, verified 2026-09-04: C:, D:, E:, G: Fixed; only Q:, S:, P:, R:, T: Remote). Commits and `tuning.json` call these "NAS"; they are wrong. Rows below say *the analysis store on G:*, *the E: store*, or *the project volume*. The single genuine network-share profile is `f2e0236`. **Seam:** `E:/Students/2026-08-07-SG-test/DNA`, which `CLAUDE.md:36-38` calls the primary measurement store, no longer exists (verified: `E:/Students/` holds only `2025-06-09_JP_C7C9`), so every row attributed to it is currently unreproducible.

**Tree state:** all figures are from branch `alignment-correctness-and-memory` @ `e98b284` (251 commits); `main` @ `a36388c` (236) lacks hybe-major dispatch, the 3-worker setting, phase accounting and even z-slabs.

### Stage 1 — Launch and environment

| Stage | Figure | Input | Source | Cache |
|---|---|---|---|---|
| Launcher, first run | 7.4 s, once `[M]` | workstation, idle | `233e68e` | cold |
| Launcher, every later run | 240–360 ms `[M]` | idem | `233e68e` | warm |
| App import | 2.5 s, 1555 modules `[M]` | idem | `233e68e` | not stated |
| Startup phase 2 | 7.68 s → 0.88 s `[M]` | real store | `5b27578` | not stated |
| First conda install | "a few GB, several minutes" `[A]` | — | `docs/installation.md:26` | n/a |

### Stage 2 — Ingestion (DAX/TIFF → HDF5)

| Stage | Figure | Input | Source | Cache |
|---|---|---|---|---|
| Throughput vs workers | 117.6 MB/s @12; 66 MB/s @36 `[M, contested]` | unstated store/machine | `1ca292c` | n/a |
| Completeness probe per existing file | 3.7 ms/file; ≈1 s over 34 FOVs @12 workers `[M]` | E: store | `145db0c`, `preprocess.py:349-393` | **not stated** |
| — same probe, re-measured | 179.8 ms median (95.4–272.5) `[M]` | 8 real 233 MB MP58 stacks, G: | in-session, 2026-09-04, not in repo | **cold** |
| Transient disk during publish | ~278 MB × n_workers (~3.3 GB @12) `[M]` | real store | `145db0c` | n/a |
| TIFF stride validation | 1.0 s per 50-FOV trial (replaced ~50 min) `[M]` | TIFF trial | `c48a531` | not stated |

> **Caveat the author must carry.** 117.6 MB/s is the paper's headline parallelism datum and is not a worker sweep. Tracing it back: `ea05374` records it as "1 GbE link saturated" observed incidentally during a live run *over a share*. The 66 MB/s arm appears exactly once in 251 commits with no store, machine or protocol. The transferable finding is "at 12 workers a 1 GbE link is already saturated" — it does not transfer to local disk or 10 GbE. The shipped spinbox default is nonetheless 4 (`ui/ingestion_panel.py:232-234`), the TIFF dialog 12 (`tiff_ingest_dialog.py:171`), the tooltip "3–4".

### Stage 3 — Storage primitives (the floor under every later stage)

| Stage | Figure | Input | Source | Cache |
|---|---|---|---|---|
| 3D crop, contiguous → chunked (32,32,z-slab) | 3.6 s → 0.1 s, 36x; 289 reads → 3 `[M]` | network-share profile | `65a562f`, `f2e0236` | cold |
| Z-plane read, uncached → slab cache | 2.3–4.6 s → 9.8 ms `[M]` | JP 1024×1024×177 | `76e8c6b`, `stack_cache.py:4-9` | cold → warm |
| `read_hybe_mip` whole-file slurp | 1449 ms → 388 ms `[M]` | real store, 76-hybe FOV | `8bfdb6d` | **cold (stated)** |
| Windowed z-stack read, decomposed | 1.04 ms total `[M]` | real store | `9d15b35` | **warm (stated)** |
| h5py global-lock discriminator | GUI MIP open 16.5 ms; behind a background *thread* doing h5py 2043.6 ms (124x); in separate *processes* 16.8 ms `[M]` | Windows real store | `f300868` | warm |
| Device throughput under 16 concurrent readers | ~51 MB/s `[M]` | analysis store on G: | `420e9bd` | cold |
| Genuine network share (2024 dataset) | open 1.3 ms; 56 MB/s sequential; 12.6 ms per scattered 4 KB `[M]` | SMB share | `f2e0236` | cold |
| Readiness scan, v2 directory listing | 7 ms, 2 modalities × 40 FOVs `[M]` | real store | `5a5e1f2`, `9946a3b` | **warm or cold, explicitly both** |

### Stage 4 — Cell segmentation

**No row exists.** See *What has never been timed*.

### Stage 5 — Alignment

| Stage | Figure | Input | Source | Cache |
|---|---|---|---|---|
| FOV align, per hybe (93% Powell) | ~3.5 s `[M]` | E: store FOV01, 1024×1024 MIPs | `2df2c62` | not stated |
| FOV align, one FOV serial | ~267 s (4.5 min), 78 hybes `[M]` | idem | `2df2c62` | not stated |
| FOV align, spawn pool | 3.04 → 0.63 s/hybe, 4.8x; matrices bit-identical `[M]` | E: FOV01, 16 real hybes | `54260f1` | not stated |
| → 78-hybe FOV | ~3.9 min → ~0.8 min `[P]` | — | `54260f1` | n/a |
| Cross-modal Z per FOV | 57.1 s → 1.71 s (whole step ~57.7 → ~2.3 s, 25x) `[M]` | local SSD clone | `5357ba5` | not stated |
| Cell alignment, baseline per cell | 8.01 s cold vs 7.99 s warm — **the ~28 GB working set does not cache** `[M]` | FOV01, 155 cells, 111 hybes | `bae5abf` | **both, stated** |
| — decomposition | 88% I/O, 12% arithmetic `[M]` | idem | `bae5abf` | idem |
| Cell alignment, shipping config (3 workers, hybe-major) | 3.07 s/hybe median (2.53–3.60) vs 4.37 @6 workers `[M]` | MAZ, 134-hybe shape, 5 FOVs | `3b25b8e`, `tuning.json:4` | **cold (stated)** |
| — phase split | fit 42.97–49.93 s per FOV, **flat in workers and in cells** (96 cells 46.10 s vs 49 cells 42.97 s) ⇒ ≈93% per-hybe device access | MAZ FOVs 40–43 | `bbc0076` | cold |
| Preparation before first fit | 84.8 s for 40 FOVs → 0.1 s `[M]` | real store | `c48a531` | not stated |
| Historic whole-run projections | FOV01 ~12 min → ~0.6 min; 34-FOV ~6.9 h → ~0.4 h `[P]` | — | `2daf319` | n/a |

> `[P]` rows above **predate hybe-major dispatch (`76183bd`) and the 3-worker default (`3b25b8e`)** and describe a configuration that no longer ships. Do not quote them as current.

### Stage 6 — Spot localization, assignment, tracing, PSF

| Stage | Figure | Input | Source | Cache |
|---|---|---|---|---|
| `assign_spots` | 138.7 ms @100k spots (was 782.2 ms, 5.6x) `[M]` | 102 real cells FOV01, 1024×1024 mask | `9a3d5e8`, `2513f03` | in-memory |
| Full spot save path | 100k spots in 0.47 s `[M]` | real store | `7421958` | not stated |
| Tracing v1, per-hybe pool | 84.7 → 6.0 s/allele (14.04x, 32 workers) `[M]` | 2 real alleles × 111 hybes | `f92506b` | not stated |
| Tracing v2, allele-parallel | 127 alleles × 73 hybes in **113.9 s** shipping (97.5 s fallback off); per-allele 10.3–28.2 s `[M]` | MP58 FOV1, 64-CPU box, 32 workers | `e42d92a` | **serial arm warm, parallel arm cold — mixed, stated** |
| v2 vs v1 speed | v2 2.3–4.1x **slower** `[M]` | 4 experiments | `b0bf3f7` | not stated |
| PSF calibration | 80.6 s / 8 crops; 99% inside 4024 `least_squares` fits `[M]` → ~8 min / 48 crops `[P]` | 48-allele bench | `e14f783` | not stated |
| PSF per channel, 40 crops | fiducial 4.6 min; readout 31 min `[M]` | bench | `notes/chromatin_tracing_optimization.md:165` | not stated |
| Bench re-harvest | ~52 min → 330 MB `.npz` `[M]` | MP58 | `notes/…:23-25` | not stated |

### Stage 7 — Analysis and rendering

| Stage | Figure | Input | Source | Cache |
|---|---|---|---|---|
| `Population.build` | 2.0 s; `dmaps` 0.10 s `[M]` | MP58 4 FOVs, 386 cells, 426 alleles | `6155883` | not stated |
| Interactive scale ceiling | `ensemble_map` 3.1 s over 24k alleles `[M]` | **synthetic** scale | `7829d91` | in-memory |
| Store-wide expression drive | 41 FOVs × 9 sources: 333 computed cold, 369/369 cached warm `[M]` | MP58 | `39c2b06` | **both, stated** |
| One cell overlay PNG | ~35 s `[M]` | real store | `76e8c6b`, `64ef83f` | not stated |
| Whole-project overlay pass | 699 renders ≈ 7 min *(observed, ÷3 workers assumed)* `[M/A]` | real store | `d5d6e63`, `main_window.py:11541-11560` | n/a |

### What has never been timed

Complete list; each verified absent from commits, `docs/`, `notes/` and code comments unless noted.

| Missing | Verification |
|---|---|
| **Cell segmentation, any variant** — Cellpose GPU, Cellpose CPU, classical | No figure in `codelab_pipeline/segmentation/` (only `focus_profile` and MIP-read numbers) and no commit. `requirements.txt:6-10` asserts CPU-only torch "costs an order of magnitude" `[A]`, never measured. |
| **cyto3 weight download** (size, time, offline route) | Not mentioned anywhere. |
| **Cytoplasm segmentation** | Route has never run on persisted production data (0 of 3953 real cells carry a cytoplasm). |
| **2D spot auto-detection** | Counts recorded (`2202584`), time never. |
| **Celltype determination** | FOV-parallel and backgrounded (`76e8c6b`); unmeasured. `codelab_pipeline/celltype.py` does not exist at this path — locate before citing. |
| **Ingestion wall time per FOV or per hybe** | Only MB/s vs worker count exists. |
| **Any end-to-end wall clock** — per stage or whole experiment | No CLI to time (`main.py` is GUI-only); `preprocess.py:6` sets `logging.basicConfig(level=INFO)` **with no timestamp format**, so `log/*.log` cannot supply one retroactively. Worse, per-stage rows cannot be summed: they come from four stores of four shapes — MP58 41 FOV × 76 hybes, MAZ 50 × 135, HoxA 43 × 35, JP 2 × 100 (verified on disk). |
| **`reconcile` / export runtime** | `dcf0ac4` gives ~42,000 coordinates and no seconds. |
| **`tools/verify_store.py` over a whole store** | No timing in the file. |
| **`Population.build` at real scale, and with mask-based intensity** | Only 4 FOVs / 386 cells; the checkbox is labelled "slower" with no number. |
| **GUI process memory footprint** | Only pool children's cache budgets are stated. |
| **`cell_alignment_workers: 2`** | `tuning.json` verbatim: "Nobody has tried 2." Sweep is 1/3/6 only, and the 1-vs-3 gap (691 vs 615 ms/cell) sits inside the reported 10–24% within-condition spread. |
| **Whether `child_io_priority: 'verylow'` helps** | Ships as a **global** default; `tuning.py:14-19` states three external A/Bs failed to answer it. |
| **Any pipeline stage on true network storage** | Only `f2e0236`'s 2024 share profile. The project's own rule (`CLAUDE.md`) requires it. |
| **A declared cache regime for most I/O rows** | The `stack_is_complete` 3.7 ms / 179.8 ms spread is the sharpest instance; the MIP-read, crop and z-plane rows carry the same ambiguity. |

---

## Troubleshooting

*Drive-letter note for every row below: `fsutil fsinfo drivetype` on the work machine (verified 2026-09-04) reports **C:, D:, E:, G: as Fixed Drives**; only **Q:, S:, P:, R:, T:** are Remote/Network. The analysis stores live on G:/E: — one bounded local device, not a share. The **raw DAX repositories do sit on shares** (`Q:\\143.248.14.9\Shared`, `S:\\143.248.14.6\Shared`, recorded in stack attrs `path=`), so ingestion reads across the network and writes locally. Commit messages that say "NAS" for the store are wrong (`420e9bd` states the correction itself).*

| Step / stage | Problem as the operator sees it | Cause | Solution |
|---|---|---|---|
| 13 Run Ingestion | Every task for one FOV returns `FileNotFoundError`; log reads `DAX file not found: …\ConvZscan_02.dax` | The DAX name is **constructed, not discovered**: `f'ConvZscan_{fov-1:02d}.dax'` (`io/preprocess.py:549`) — the app is 1-based, the disk 0-based. The checklist's default-check probe only asks whether the hybe *folder* holds any `.dax` (`ui/ingestion_panel.py:490-494`), so a numbering mismatch passes the probe and fails per task | Verify on-disk indices against the FOV list before running. The TIFF door is immune — names are matched by regex `_Pos(\d+)__…_RAW_ch(\d+)` (`io/tiff_ingestion.py:148-153`). ASSERTED (code-derived; no commit records this failure) |
| 12 Workers | Throughput falls as workers are added | One bounded device serving all readers. MEASURED **117.6 MB/s at 12 workers, 66 MB/s at 36** (`1ca292c`; `tuning.py:22-23`) | The ceiling (**42** on this 64-core/352 GB box, MEASURED, `preprocess.py:77-112`) describes RAM, not the storage. Spinbox default is 4 (`ui/ingestion_panel.py:232-243`); `docs/happy_path.md` reads as though the app uses 12 — it does not |
| 13, interrupted | Cell alignment later dies with `object 'ch635' doesn't exist`, in a stage unrelated to ingestion | An interrupted **overwrite** left a stack with one channel written. Append tested `os.path.exists`, so the damaged file was skipped forever — surfacing only at cell alignment, the first stage that reads stacks, not MIPs | `stack_is_complete` now tests structure **and readability**, touching first and last element of every dataset (`preprocess.py:349-393`); such files rebuild on the next append. Cost MEASURED **3.7 ms/file, ≈1 s across 34 FOVs at 12 workers** (`145db0c`) — but re-measured today on 8 real 233 MB stacks it was **median 179.8 ms** (cache regime differs; say which you quote) |
| 13, interrupted | A hybe is complete on disk yet every readiness check says un-ingested; append skips it forever | Killed between the stack publish and the MIP publish. The MIP file *is* the ingested flag | `_restore_missing_mip` rebuilds the flag from the stack's own `/mip/ch*` — ~2 MB read, no DAX (`preprocess.py:396-424`, `76e8c6b`) |
| 13 vs any running job | An alignment run across a concurrent overwrite fits some hybes against old pixels, some against new | Overwrite rewrites files other work is mid-read of; on Windows `os.replace` over a held file raises `PermissionError [WinError 5]` (MEASURED, `8be3224`) | Interactive overwrite asks, then stops every store reader; a queued one stops them and logs it. **Append is exempt by design** — it only adds files nobody reads (`ffab9fe`) |
| 10 Parse Layouts | Mid-ingestion, Parse Layouts appears to hang and the status box goes stale | The GUI-thread status scan competes with the workers for the same link; the layout parse itself cost **~300 ms at 111 hybes per call** over the share and a single-slot cache made every DNA/RNA alternation a fresh read (MEASURED, `677ec62`) | The scan (job 2 of 3) is deliberately skipped mid-run and re-syncs at end or on **Check Ingestion Status**; layout parse is mtime-keyed, 8-entry cached (**265 ms → 0.1 ms**, MEASURED) |
| 43 Align All Cells in ALL FOVs | Stopping the run throws away every FOV it finished; Results shows residuals the store never received | Persistence ran only in `finished_ok`, after the **last** FOV. Confirmed live: FOV001 73/73 with matrices, FOV002-005 zero | Per-FOV `fov_done` write through `.part` + `os.replace` (`5b27578`). With append, a stopped run now genuinely resumes |
| 43, append | `append: 3600 cell(s) already fully aligned` three passes running, while FOVs 12-50 have an empty matrices dict on disk. **Restarting the app clears it — that is the tell** | Append decided from **in-memory** `cell.matrices`; a run killed after mutating cells but before writing left matrices in the process and nowhere else | `_persisted_cell_matrix_keys` reads the store (`main_window.py:10927`); `_resolve_passes` consults that (`9b54fda`) |
| 43, append | `append: 3600 cell(s) with nothing to fit -- skipped` on a fresh app, though the log said "loaded FOV-level matrices for 49/49 FOV(s)" | Pass dicts carried an **object snapshot** of `fov_matrices` built *before* the store backfill ran | Backfill moved above pass construction; call order pinned by `tests/test_cell_alignment_all_fovs.py` (`b1cafe8`) |
| 43, append | Segment a FOV, don't save, append dies, append again → that cell is skipped forever | `read_cells` returned `(None,'')` for both "never written" and "cells.h5 won't open" | No file ⇒ FOV enters the map empty ⇒ fit everything; file present but unreadable ⇒ defer to memory (`e98b284`) |
| 32 / 43, during ingestion | A hybe is permanently marked aligned with a perfect **0.0 px** residual — the one entry guaranteed never to be flagged for review | A hybe whose MIP is absent reaches `chain._cell_native_crop`, which returns the same bare `None` as a genuinely off-frame cell; an identity residual is persisted with off-frame provenance, and append sees the key and calls it done | The readiness gate excludes FOVs with unfinished ingestion in every alignment type and **names which and why** (`ffab9fe`, `d5d6e63`). Cost of asking: MEASURED **7 ms** for a 2-modality × 40-FOV sweep |
| 43 | "Align All Cells in All FOVs only calculated a single FOV" (reported twice) | Three indistinguishable causes: session-state matrices filled one FOV at a time and the bulk backfill skipped the store's own modality; the completion report did not separate "fitted" from "considered"; unfinished FOVs produced identity residuals | Preparation reads missing matrices from the store (49/49 on the first real run); the report names the FOVs actually fitted; fitting nothing says so and names **Overwrite** (`d5d6e63`) |
| 38 Cell alignment | Z corrections pile up at exactly 5.0 planes; 16.4% of entries rejected on magnitude alone | The Z gate was `abs(z_shift) > pad/2` — a depth bound derived from an **XY** crop radius | `MAX_CELL_Z_SHIFT_PLANES = 15.0` (`alignment/chain.py:65`), exposed as "Max Z shift". Sweep MEASURED: z_max 5 → 83 applied/18 mag-rejected (max 5.0, truncated); 15 → 101/0 (max 11.0); 20 and 30 identical (`8b22c09`). **The canonical case of a threshold tuned to the `data/` extract being wrong on the real store** |
| Any long run | Window goes "Not Responding" during cell alignment, for FOVs the run never touched | Preparation called the GUI's `_activate_fov` per FOV (MEASURED **84.8 s for 40 FOVs before a single fit**); the finish handler recast every FOV's spots inline; spot views read MIPs synchronously (0.12 s idle, "seconds to minutes" under load) | Preparation is a plain `read_cells` (**0.1 s**); recast moved to child processes, verified bit-identical; MIP reads via `_mip_then` (`c48a531`, `420e9bd`) |
| Any long run | Moving the work to a background QThread changes nothing | **h5py takes one process-wide lock per call, held through gzip inflation.** Discriminator MEASURED: 16.5 ms baseline; **2043.6 ms behind a background thread doing h5py (124×)**; 16.8 ms with the same reads in separate processes (`f300868`) | Every HDF5-reading worker runs in a child process (`ProcWorker`); `test_overlay_specs` asserts the four handlers use it |
| 43 | Every process on the machine slows — an unrelated matplotlib benchmark falls to 21% CPU — while "Available memory" reads 168 GB and the disk is idle | Not churn, **retention**: Windows serves allocations from a free+zero page list of ~7.3 GB here; 6 children × 1.0 GB MIP cache drained **5.89 GB** from it (MEASURED, `5e073ca`) | MIP cache is a **4 GB total ÷ pool size, floored at 0.5 GB** (the MEASURED cliff: 4343 ms @0.125 GB, 4061 @0.25, **3.6 ms @0.5**, `8bfdb6d`). When floor and total disagree the floor wins and the log says to use fewer workers (`process_guard.py:255-279`) |
| 43 | A 20-FOV run ends with the window hard-frozen for minutes *after* the progress bar said done | Every overlay PNG (~10 s of matplotlib per FOV, MEASURED) was drawn inside `finished_ok`, on the GUI thread | Per-FOV `fov_done(fov, matrices)`; the freeze is one FOV long. Signal must be `pyqtSignal(int, object)` — a `dict`-typed signal silently drops an ndarray (`bf2fd05`, `c18639b`) |
| Overnight runs | Nothing rendered until someone clicked OK on a dialog claiming rendering was underway | Completion notices were modal; one held the window **93 s of a 10-minute profile** (MEASURED; an earlier profile put one at 46.7%) | Notices non-modal; the overlay pass moved ahead of the dialog (`d5d6e63`). Ingestion completion is still modal |
| Any freeze | "The GUI is blocked and I don't know why" | Three different questions | `tools/watch_gui_stall.py` prints **starvation = wall/CPU**: ≈1 ⇒ busy; ≫1 ⇒ waiting, and the fault rate says memory (high) vs lock/IO (low). `tools/thread_state.py` asks the kernel READY vs Waiting + reason; `tools/catch_stall.py` dumps with locals when the main thread leaves the event loop; `windows/run_probe.py` measures from inside |
| Force-kill | Worker processes outlive the app; `conda env remove` fails on locked files | A terminated parent runs no code. MEASURED: **97 pool workers from three runs alive 2.7 days later, holding 5.0 GB** | Windows Job Object with `KILL_ON_JOB_CLOSE` (POSIX: `PR_SET_PDEATHSIG`). Verified guard off → 4 alive after `TerminateProcess`; on → 0 (`b80c9bb`). First log line is `Worker-process guard: <state>` |
| 27 Save cells | The confirm dialog names FOV002 while the panel shows FOV004; the emptied FOV is never written and an unrelated one is overwritten | Save took the FOV from `_last_segment_context`, updated only when a FOV successfully *displays* cells; "has cells" was treated as "is staged" | Save starts from the panel's own spinbox; a deliberate edit counts as staged at zero cells (`16d2e0a`, pinned by `tests/test_save_cells_target.py`) |
| 52 Save Current Spots | A spot save silently wipes every cell's alignment matrices | The save bulk-wrote the *transient* cell container while alignment mutates the *permanent* tier | Spot saves write spots only; verified 101 disk cells keep matrices (`d31d779`, `5ff45b4`) |
| 65 / 70 Alleles | Saving alleles for one FOV deletes every other allele in it | `write_fov_alleles` is a whole-FOV atomic **replace**, handed a subset in append mode | Per-allele promotion into the permanent tier, then write that. **Never Save a FOV whose existing content was not loaded this session** (`44eb948`, `main_window.py:9043-9049`) |
| 1-3 Launch | Segmentation imports fine and fails only when you segment | conda base carried **cellpose 4.1.1**, which removed `models.Cellpose` and `model_type='cyto3'` | `cellpose>=3.1,<4`, `numpy<2` (`requirements.txt:13-25`) |
| 1-3 Launch | Cellpose runs on CPU on a GPU machine | `--extra-index-url` only *adds* an index; pip took PyPI's Windows torch, CPU-only (verified `2.13.0+cpu`, `cuda.is_available()` False on an RTX 3070) | Pin `torch==2.7.1+cu118` — naming the local version forces the CUDA wheel (`5f6af71`) |
| 74 Build Population | App aborts with `0xC0000409` and no traceback | **PyQt escalates an unhandled slot exception to `qFatal`/abort** | Every analysis slot runs under a guard catching all exceptions; `main.py` installs a `sys.excepthook` (`f57c9b5`, `7d08495`) |

### Store integrity

`python tools/verify_store.py <storage_path> [--workers N] [--list-bad]` is read-only, process-pooled (default `min(8, cpu_count//2)`), enumerates every `(fov, hybe)` from `stacks/` and `mips/`, and probes each dataset by **touching its first and last element** — HDF5 opens a truncated file happily and reports the declared shape from its header, so only reading the ends distinguishes complete from truncated (`tools/verify_store.py:23-25,43-49`). It prints five counters and exits 1 if any is non-zero:

| Counter | What it means |
|---|---|
| `checked` | Files opened and read-probed (both kinds). |
| `UNREADABLE/BROKEN` | Zero bytes, missing `/stack` group, a declared channel absent, wrong `ndim`, `shape[-1] != expected_depth`, an unreadable end element, or a missing in-stack `/mip/ch*`. Killed mid-rebuild, or a bad volume. |
| `MIP but NO STACK` | Labelled in the output `<- interrupted overwrite`: the MIP (written atomically, and *after* the stack) survived from the previous run while the stack did not. |
| `STACK but no MIP` | Killed between the two publishes. The one class the app self-heals — `_restore_missing_mip` rebuilds the flag from the stack's own `/mip` on the next append. |
| `.part leftovers` | An `os.walk` for `*.part`. Inert (nothing reads that suffix) but each is up to ~278 MB of an interrupted atomic write. |

Real damage found by this tool in a live store (MEASURED, `2550405`/`145db0c`): **13 affected (FOV, hybe) entries, all contiguous in FOV01** — 11 MIPs with no stack (Hyb_018…028), plus Hyb_016 missing `/stack/ch635` and Hyb_017 missing `/stack/ch555`. Today's cheap directory-level equivalent across seven modality trees (MP58 DNA+RNA, MAZ, JP DNA+RNA, HoxA DNA+RNA) returned **0 in every class** (MEASURED; a full read-probe of MAZ would be 13,500 opens and was not run).

**The repo documents no recovery procedure for any of these classes.** `verify_store.py` reports and exits; it never repairs, and no other tool repairs stacks. In practice the recovery is implicit and unwritten: the two file-level classes are fixed by re-running ingestion in **Append** mode, which now rebuilds anything failing `stack_is_complete` rather than skipping it (`preprocess.py:532-547`); a stray `.part` must be deleted by hand. `tools/repair_vlinks_mips.py` (~1 s/hybe, no DAX) repairs only the retired v1 `vlinks.h5` MIPs and does not apply to a v2 store (`c659f32`).

Two documentation seams the manuscript must not inherit: `tools/verify_store.py:6-10` and `windows/main_window.py:3057` still assert **in the present tense** that overwrite does `os.remove(stack)` then a non-atomic rebuild. Grep confirms the only `os.remove` in `codelab_pipeline/` are on `.part` files (`io/preprocess.py:344`, `io/paths.py:135`) — overwrite goes through `publish_stack`'s `.part` + `os.replace` like everything else. The text is history written as present fact; the damage classes it describes are real and still findable in stores written before the fix.

---

## Anticipated results

Two reference stores carry the numbers below. **MAZ** — `G:\Seeun\2026-06-17_MAZ_IF_15ug`, 50 FOVs, single DNA modality, 135 hybes/FOV, 129 z-planes — completed a full ingestion → segmentation → three-layer alignment run today (2026-09-04); it holds **zero alleles**, so every tracing number comes from **MP58** (`G:\Seeun\2025-11-30-MP58`, 111 hybes/FOV, 73 traced DNA hybes) or the E: SG-test store. All MAZ figures below were re-derived from the store this session by reading the per-FOV manifests and `cells.h5` capsules directly (MEASURED, 2026-09-04).

### 1. After ingestion — store shape and footprint

| quantity | value | source |
|---|---|---|
| stack files per FOV (MAZ) | **135**, 277–282 MB each | MEASURED, `DNA/stacks/fov001` listing |
| stacks per FOV / whole store | **43 GB** / ≈2.1 TB (×50, extrapolated) | MEASURED / ASSERTED extrapolation |
| MIPs per FOV | **397 MB** (one 1024×1024 MIP ≈2 MB) | MEASURED; `CLAUDE.md` |
| derived analysis capsules, whole store | **710 MB** — `cells.h5` 15–26 MB/FOV, `matrices__DNA.h5` 73 KB/FOV | MEASURED |
| E: SG-test comparison | 34 FOVs × 111 hybes, 120 planes, stack ≈278 MB | `CLAUDE.md`, `54260f1` |

A store that read-probes clean under `tools/verify_store.py` reports no BROKEN files, no MIP-without-stack, no `.part` leftovers (`verify_store.py:203-227`).

### 2. After segmentation — cells

| quantity | value | source |
|---|---|---|
| cells per FOV, MAZ | **44–127, median 71, mean 72; 3,600 total over 50 FOVs** | MEASURED, manifests |
| other observed | 155 (E: FOV01), 101–102 (E: FOV01, later pass), 1,091 project-wide elsewhere | `bae5abf`, `9a3d5e8`, `chain.py:919` |
| cell bbox | median 58×58 px, p90 66×67, max 76×118 (227 cells, 2 FOVs) | MEASURED, `chain.py:1346-1351` |

### 3. After alignment

FOV level (77 real hybe pairs, one FOV): rotational drift **0.000–0.100°** (max |angle| 0.1007°), translation well inside the 30 px cap, reconstruction residual **median ≈60**, mean 84.4 (`chain.py:38-40`; `a968c3e`). Cross-modal: bridge magnitude **≈13 px**, up to **28 px** observed; per-cell z truth near **−12 planes** (`frames.py:260`, `chain.py:1254`).

Cell-level residual, MAZ full run (MEASURED today, 486,000 entries read back):

| quantity | count | % of entries |
|---|---|---|
| entries (3,600 cells × 135 hybes) | **486,000** | — |
| entries per cell — histogram is `{135: 3600}` | **zero incomplete cells** | — |
| fully identity (`yx = I`, `dz = 0`) | 24,717 | **5.09 %** |
| `yx` identity (residual not applied) | 41,972 | 8.64 % |
| **any** reject verdict in provenance | 213,214 | **43.87 %** |
| — of which XY residual rejected → FOV-level fallback | 38,372 | 7.90 % |
| — of which Z leg rejected → `z = 0` | 195,959 | 40.32 % |
| `dz` exactly zero | 199,566 | **41.06 %** |
| median \|dz\| | **2.00 planes** | max 54.0 |

The 43.9 % figure is **entries carrying any reject verdict**, not XY fallback alone — the handoff phrasing ("cell-level residuals rejected … fallen back to FOV-level") describes the 7.9 % column; flagging the seam because both numbers will be quoted. The zero-`dz` count closes arithmetically: 195,959 z-rejections + 3,600 reference-hybe entries (identity by definition, no provenance) + 7 genuine zero fits = 199,566. **Inconsistency to resolve:** 16,635 entries (3.4 %) carry an *applied* \|dz\| above 15 planes (sample: `Rep_104(cell 1)->Hyb_DAPI [z-alignment applied: z=-20.0px]`), while `MAX_CELL_Z_SHIFT_PLANES = 15.0` (`chain.py:65`) and `configs/2026-06-17-SG_MAZ.xml` declares `max_z_shift="15"` — so the run used a larger operator bound than the config now on disk, or those entries predate it. Run envelope, inferred from capsule mtimes (write order not strictly monotone): **15:50 → 22:39, ≈6.8 h for 50 FOVs**.

### 4. Spots

MAZ carries **311 spots across 4 (hybe, channel) slices** — detection barely exercised there (MEASURED, manifests). MP58 FOVs 1–3: **421 spots, of which 372 are fiducial-channel and 49 readout** (`510b7c0`) — an export handed to a collaborator is mostly beads unless filtered on `channel_role`. Homeless spots (`cell = -1`) are exported, never dropped. Assignment of 100k spots over 102 cells costs 138.7 ms (`9a3d5e8`).

### 5. Tracing (MP58 FOV1, 127 alleles × 73 hybes = 9,271 possible per channel)

| quantity | value | source |
|---|---|---|
| fiducials fitted | **7,694** (no seed fallback) → **8,661** (shipping) = 83–93 % | `966d3e6` |
| readouts traced | **3,395 → 3,626** = 37–39 % | `966d3e6` |
| traced hybes per allele | ~27–35 of 73 | `cc08f4b`, `8b1cd87` |
| occupancy, fiducial / readout | **0.838 / 0.677**; blank-region fits **0 %** (from 34 %) | `tracing_v2.py:19-26` |
| same-locus repeat distance | median **129–137 nm**, p90 **320–387 nm**; 59–161 nm across four experiments | `9c2a128`, `966d3e6`, `b0bf3f7` |
| readouts kept at z-uncert ≤150 nm | **44 %** | `d7dbde8` |
| v1 for contrast | rejects **92 %** of readout crops; 26 of 336 possible pairs at 183 nm | `gate_sweep_v2`, §9 |
| `final_polymer` | **empty on both real stores** — the bin collapse is recomputed at read time (`selection_rule='computed'`) | `510b7c0` |

Analysis outputs: ensemble map range **0.23–0.99 µm** (12 FOVs, 1,572 alleles, 62 bins); per-FOV panels span vmin 0.274–0.351 / vmax 0.967–1.184 on one shared scale; QC `max_jump_um` **0.885 (xyz) / 0.614 (xy)**; isotropy factor **1.516** (`1821b2c`, `09a3219`, `b09e0e0`). R + T rounds are ~15 % of exported traced positions at `bin_index = -1` (`510b7c0`).

### Validation status

**Validated.** (i) *Synthetic ground truth for the fitter*: `tests/test_fit3d_um.py`, 29 checks — 20 nm centre recovery, anisotropic σ recovered separately, the same physical spot at 0.1/0.2/0.4 µm z-steps giving the same µm answer (`5e06468`, `d0ac7e8`). (ii) *Synthetic ground truth for alignment*: known translations/rotations recovered to ≤0.5 px and ≤0.3°, `tests/test_alignment_ground_truth.py:87-150` — **but this suite cannot run on the Windows machine at all** (its `data/` fixture is absent), and no real pair in either store carries enough rotation to exercise the rotation branch (`a968c3e`). (iii) *Internal precision*: the same-locus replicate score (H/R rounds, 7 valid pairs per allele, 336 possible over 48 alleles) with toehold rounds as a **displacement positive control** separating 5.1× (0.457 µm vs 0.090 µm, `892d5a5`). (iv) *Equivalence, not accuracy*: pooled vs serial matrices bit-identical (`54260f1`), hybe-major vs cell-major byte-identical over 696 entries (`76183bd`), `reconcile` dry-run over ~42,000 coordinates at the 2-decimal on-disk rounding floor (`dcf0ac4`).

**Not validated — verified by grep across code, `docs/`, `notes/` and 251 commits today, no hits:** there is **no biological validation** (no independent confirmation that a trace reflects the locus it claims), **no orthogonal measurement** (no Hi-C, sequencing, or second imaging modality compared against), and **no comparison with published traces or with another tracing pipeline on the same data**. The only external lineage is code provenance — ORCA's `QualityControlORCA.combineFOV` collapse and the HiCRep `scc` statistic were ported and are cited as such (`polymer.py:81`, `ensemble.md:83`) — which is inheritance, not validation. Further caveats the manuscript must carry: every v2 gate threshold was tuned on **one dataset** (MP58 FOV1) and the module says so repeatedly; the v2 **standalone-spot** engine was never A/B'd against v1 (only tracing was); 6 of 43 test suites fail at fixture load on this machine, so a pass count is not evidence; and MAZ's config declares `voxel_z_um="0.145"` while the pipeline default and every µm figure quoted above use 0.200 µm/plane (`fit3d_um.py:54`, `polymer.py:170`) — any MAZ length in µm must be recomputed before publication.

---

## Before submission

Nine agents audited the repository against what a Nature Protocols paper
must be able to state. What follows is what they could **not** find, or
found to be inconsistent. It is ordered by what it costs to leave alone.

Every row was verified against the code or the store at the time of writing;
where a claim could not be checked, it says so.

### A. Would be wrong in print

| # | Finding | Where it leaks into the paper |
|---|---|---|
| A1 | **The analysis store is not a NAS.** `fsutil fsinfo drivetype` today: `C:`, `D:`, `E:`, `G:` are **Fixed Drives**; only `Q:`, `S:`, `P:`, `R:`, `T:` are Remote/Network. Raw DAX lives on `Q:`/`S:` (network); every analysis store measured — MP58, MAZ, JP, HoxA — is on `G:` (local). | `CLAUDE.md:36-38` and many commits call `G:`/`E:` "the real NAS store". Any Methods sentence saying performance was measured over a network share is false. The measurements stand; the label does not. Note the store is still *slow* for a local volume — `420e9bd` measures ~51 MB/s on cold stack reads — so describe it by its measured rate, not by a storage class. |
| A2 | **The headline parallelism number has no traceable measurement.** "117.6 MB/s at 12 workers against 66 MB/s at 36" is restated in `tuning.py:22-25`, `parallel.py:8-12`, `docs/principles.md:19-22`, `docs/happy_path.md:26-28` and `main_window.py`, and in `CLAUDE.md`. The auditors read the full log and could not find the run that produced it: no dataset, no FOV count, no date, no method. | It is the paper's central claim about I/O-bound parallelism. Either re-measure it and record the conditions, or do not make the claim. |
| A3 | **A docstring at HEAD states the opposite of the code.** `preprocess.py:517-519` says the TIFF path writes `(depth, height, width)` "unlike the DAX path"; `tiff_ingestion.py:497-501` explicitly transposes to z-last, so TIFF and DAX stacks have the **same** axis order. | Anyone reproducing the format from the documented axis order gets a transposed volume. |
| A4 | **GUI and library defaults disagree, silently.** Classical segmentation `min_size`: UI ships 500 (`ui/cell_segment_panel.py:181`), library signature 1000 (`segment.py:94`). Cell alignment `channel_type`: engine defaults `'readout'` (`chain.py:1513`), the panel puts `'fiducial'` first and that is what is used in practice. | A headless reproduction of a GUI run uses different thresholds and a different channel, and nothing warns. State the values actually used, not the signatures. |
| A5 | **`DAX_WORKER_PEAK_BYTES = 5 GiB` is justified by a stack shape no store has.** `preprocess.py:15-21` cites "~4.5 GB for this project's real 2048×2048×354 uint16 stacks"; every store verified on this machine (MP58, MAZ, JP, HoxA) is 1024×1024. | The memory budget, and the worker count derived from it, are sized for a 4× larger image than the one in the paper. |
| A6 | **No software-availability record.** `git tag` returns nothing; there is no LICENSE, no `CITATION.cff`, no `pyproject.toml`, no DOI. The working tree is on branch `alignment-correctness-and-memory` at `e98b284`, **15 commits ahead of `main`** (`a36388c`). | Nature Protocols requires a citable, versioned artefact and a license. Also decide which commit the protocol describes — several audit reports disagreed about HEAD because of the branch. |

### B. Must be measured before submission

None of these exist anywhere in the repository. They were searched for in
code, docs and all 251 commit messages.

| # | Missing measurement | Why the paper needs it |
|---|---|---|
| B1 | **Any end-to-end wall clock**, for one stage or a whole experiment. | Nature Protocols requires a Timing section. Note there is no CLI to time and `preprocess.py:6` configures logging without timestamps, so this must be timed deliberately. |
| B2 | **Cell segmentation runtime**, on any platform. All reports confirm this independently. | Segmentation is a numbered step with no time against it. |
| B3 | **Cellpose CPU vs GPU cost**, and the `cyto3` weight download (size, URL, time, offline route). | `requirements.txt:6-10` asserts CPU-only torch "silently costs an order of magnitude" — this is the sole justification for the `+cu118` pin a reader must reproduce, and it has never been measured. |
| B4 | **Cache state for every I/O figure.** The sharpest case: `stack_is_complete` is recorded at 3.7 ms/file (`145db0c`) and re-measured today at **179.8 ms median** on 8 real 233 MB stacks — a 50× spread with nothing distinguishing warm from cold. | This machine's file cache holds ~8 FOVs of working set and changes per-FOV timings by 5–30×. Any timing table without a cache column is not reproducible. |
| B5 | **Timing for celltype determination, 2D spot auto-detection, and `Population.build` at real scale.** | Three numbered steps with no time. |
| B6 | **Store size per raw byte, and minimum free disk.** Only the transient bound is documented (one `.part` per in-flight worker). | Readers must plan storage. |
| B7 | **Whether `cell_alignment_workers = 2` beats 3.** `tuning.json` says verbatim: "Nobody has tried 2." The measured curve is 1 / 3 / 6 only. | Minor, but the shipped default is presented as optimal. |

### C. Only the author can answer

Everything here is outside the software and absent from the repository.

- **Acquisition optics**: objective, NA, immersion, coverslip, camera model and pixel pitch, laser lines, emission filters, exposure. Nothing names any of them, and the default voxel size (`0.208` µm lateral, `0.2` µm axial, `ui/ingestion_panel.py:11`) is asserted as "this lab's own" with no derivation.
- **Camera characterisation**: `localization/fit3d_mle.py:39-42` runs with `camera_gain=1.0, camera_offset=0.0` and says so explicitly. Real gain and offset are needed for the MLE to be what it claims.
- **Fluorophore assignment** for the channel integers (390, 475, 488, 555, 635, 647), and which is fiducial by chemistry rather than by config.
- **Sample preparation and hybridisation chemistry** — probe design, buffers, wash and imaging rounds, per-round timing. No wet-lab step exists in the repo.
- **Recommended operator values with reasoning** for every parameter that changes results: ingestion workers (measured 12, shipped default 4 — see D3), Cellpose diameter, spot thresholds, alignment pad and z bound. The values actually used on the published datasets are recorded in the config XMLs and would be the honest source.
- **Statistical treatment**: n, the replicate unit (allele / cell / FOV / experiment), how medians and p90s were computed.
- **Validation.** A grep across every `.py` and `.md` found **zero DOIs and zero literature citations**, and no comparison against an orthogonal method. A protocol paper needs at least one anchor showing the output is right, not merely self-consistent.

### D. Repository hygiene — cheap, and visible to a referee

| # | Issue | Verified at |
|---|---|---|
| D1 | `FitThisFovPushButton` is created with a tooltip and **connected to nothing** — a per-FOV chromatin tracing control that does not work. | `ui/chromatin_tracing_panel.py:474-481` |
| D2 | `docs/happy_path.md` describes an all-FOV segmentation step. **No batch segmentation entry point exists anywhere in the tree.** The doc was last touched 2026-08-30 with 21 commits landing after it. | `docs/happy_path.md:44` |
| D3 | Ingestion worker guidance gives **four different numbers** from one datum: measured 12, spinbox default `min(4, ceiling)`, TIFF dialog default `min(12, ceiling)`, `parallel.py` DEFAULT another. | `ui/ingestion_panel.py:232`, `ui/tiff_ingest_dialog.py:171`, `parallel.py:53` |
| D4 | `focus_profile` post-cache cost is given as **~3 s, 4.6 s and 6.3 s** for the same operation in three places. | `segment.py:196-209`, `76e8c6b`, `main_window.py:4302-4305` |
| D5 | **Stale `vlinks.h5` references** survive in live docstrings although the file was retired: `chain.py` (9 sites), `spot_mapper.py:114`, `convention.py:18`, `columnar.py:18`, `paths.py:19`, and more. | grep |
| D6 | `paths.py`'s own layout docstring omits `analysis/celltype_config.pkl`, `analysis/psf.json` and the per-FOV `manifest.json`, all of which exist and are load-bearing. | `paths.py:10-52` |
| D7 | `notes/chromatin_tracing_optimization.md` is cited for its measured ladder but its conclusions are stale — it states the optimisation is "not wired into the app", which is no longer true. | `954e078` |
| D8 | `tuning.json` documents a four-arm A/B whose arms use `cell_alignment_workers: 8`; the shipped value is 3 and the measured sweep was 1 / 3 / 6. | `tuning.json` |

### E. Parameters that are asserted, not derived

These are defensible as engineering choices but should not be presented as
calibrated, because nothing in the repository calibrates them:

`MAX_CROSS_MODAL_Z_PLANES = 80.0` (`chain.py:1232`); the cross-modal Z window
hardcoded to a 640×640 crop at `y0=192,y1=832,x0=192,x1=832` with no
rationale and no operator control; `ANGLE_QUANTUM_DEG = 0.5` (`chain.py:43`),
which is both the quantisation step and the rejection threshold;
`_ZX_MAX_SIDE = 160` (`chain.py:1329`), sized from 227 cells in two FOVs and
never checked against MAZ, MP58 or HoxA; the occupancy gates 0.25 (fiducial)
and 0.40 (readout), which the repo itself says are calibrated on a single
dataset (MP58 FOV1); `min_overlap_frac = 0.5` and `signal_threshold = 10`
(`chain.py:234-282`), each carrying a motivating anecdote rather than a sweep.

By contrast `MAX_CELL_Z_SHIFT_PLANES = 15.0` **does** carry a full sweep
table, and is the model for how the others should read if they are to appear
in the paper as chosen values.

### F. One claim contradicted by its own data

`spots-tracing` states that allele provenance records the quality gates each
trace was fitted under. Reading 654 real provenance entries from the MP58
store: all 654 carry the six base keys, 230 carry `reference_hybe`, and
**1 carries the gates**. If the paper claims per-trace gate provenance, it is
claiming something the data does not support.
