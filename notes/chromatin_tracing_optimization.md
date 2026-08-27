# Chromatin tracing: fit and gate optimization

State of the investigation, so it survives outside anyone's memory.
Every number here was measured on the real store, and the tool that
produced it is named so it can be re-run.

Dataset: `G:\Seonghyeok\2025-11-30-MP58`, DNA, FOV 1/2/4/5.
Allele candidates: Hyb_016 ch555 spots (439 available).
Traced set: 73 DNA hybes of datatype H/T/R. Drift baseline: Hyb_016.
Voxel: 0.208 x 0.208 x 0.2 um.

## The apparatus

| tool | what it does |
|---|---|
| `tools/fit_testbox.py` | harvests real crops through the production path into a frozen `.npz`; `--figures` renders View Crop's four pop-ups per allele |
| `tools/fit_quality.py` | occupancy / drift-from-argmax: is the fitted centroid ON the emitter |
| `tools/gate_sweep_v2.py` | sweeps gate thresholds against the replicate score, renders the curves |
| `tools/v2_variants.py` | the v1 -> v2 ladder, one change at a time |
| `tools/calibrate_psf.py` | PSF calibration + data-vs-candidate profile figures |
| `tools/fit_gates.py` | the earlier sweep, on v1-era fits (superseded by `gate_sweep_v2.py`) |

Benches live in the session scratchpad, not the repo (330 MB):
`mp58_48.npz` (48 alleles, 7008 crops), `mp58_16.npz`, `mp58_probe.npz`.
Re-harvest with `fit_testbox.py harvest --alleles 48` (~52 min) if lost.

## The two ground truths

**Replicate score** (the SCORE, never a gate input -- a gate judged by it
would be circular). 10 loci are imaged more than once; a pair is two
rounds of the same locus in the same allele, so their distance is
localization error directly. Measured in 3D with the pipeline's own
fiducial correction applied.

Only **H and R** rounds count as replicates. **Toehold (T) rounds are
displacement controls** -- scoring them as replicates books designed
displacement as error. Used as a POSITIVE control instead they separate
cleanly: replicates 0.090 um vs toeholds 0.457 um, 5.1x.

7 valid pairs per allele => 336 possible at 48 alleles.

**Occupancy** (fit quality, needs no ground truth): intensity at the
fitted centroid over intensity at the argmax, both above local
background. 1.0 = on the emitter, <=0 = in background.

## What was wrong, and what fixed it

### The fit (not the gate)

The tracing crop is a PILLAR: bounded in XY, full slab in Z (17x17x110).
The PSF is a few hundred voxels; out-of-focus content is ~34k. Least
squares is dominated by the latter, so the fit spent its position and
sigma parameters describing background.

Symptoms, all one cause: sigma at its bound in ~100% of fits; z position
at its bound in 75-100%, which is why every `dz` in a fiducial overlay
printed as a whole number (`fitted z = integer argmax +/- exactly
peak_bound`); the centroid landing in visibly empty space.

Fixes, each measured separately (`v2_variants.py`):

| change | effect |
|---|---|
| BOX not pillar, + linear background | occupancy 0.373 -> 0.806, blank-region fits 31% -> 4% |
| box placed at consensus depth | fiducial z spread 1.00 -> 0.56 planes |
| intensity-weighted centroid seed | v1 0.354 -> 0.597; v2 barely moves (0.799 -> 0.818) |
| loose separate bounds (5 px / 10 planes) | at-bound 75-100% -> 2-13% |
| calibrated PSF (sigma fixed) | fiducial occ 0.803 -> 0.838 AND 37% faster; readout 0.561 -> 0.677 |
| Poisson MLE | **no gain on real data**, 20-40% slower |

Final: 48 alleles, occupancy **0.354 -> 0.838** (fiducial), blank-region
fits **34% -> 0%**.

**Poisson MLE does not earn its place.** Synthetic Poisson data predicted
15-27% better axial precision; on real crops it delivered nothing
measurable and cost 20-40% more time.

### Box placement in Z, without an allele depth

The alleles here have **z = 0** -- detected on MIPs, never 3D-refined --
so there is no anchor depth to place a box with. Derived instead:

    expected_native_z(hybe) = baseline_shared_z - cell_z_offset(hybe)

`baseline` = each hybe's argmax z mapped to the shared frame, then the
MEDIAN across hybes. Fit-free and correct whether or not the tracing
reference differs from the shared-frame reference (in MP58 it does:
tracing references Hyb_016/DNA, the cell references Hyb_101/RNA).

Do NOT build the baseline from full fits: measured 1.05 planes of
placement error at 0.09 s for argmax-median, against 4.65 planes at 106 s
for a pillar fit per hybe -- the expensive route is WORSE, because a
pillar fit is the degenerate fit being fixed.

Boxes are NaN-PADDED where they run off the slab (0.1% of crops), never
clipped: clipping changes a box's shape and centre, so two hybes' boxes
stop being comparable.

### The gates

Every v1 gate quantity changed MEANING when the background became local,
so the inherited constants do not transfer. Re-derived on the fixed fit
(`gate_sweep_v2.py`), 48 alleles:

At v1's own coverage (26 of 336 pairs -- it rejects 92% of readout crops):

| gate | threshold | pairs | median |
|---|---|---|---|
| **v1, all it can do** | -- | 26 | **0.1832 um** |
| min_hb_ratio | >= 1.3 | 30 | 0.0614 um |
| max_uncert_z_nm | <= 60 | 30 | 0.0634 um |
| max_uncert_xy_nm | <= 20 | 40 | 0.0687 um |
| occupancy | >= 0.95 | 27 | 0.0720 um |
| min_ah_ratio | >= 1.0 | 31 | 0.0790 um |

**v2 reaches 61-79 nm against v1's 183 nm at equal coverage** -- 2.3-3x.

Free, no threshold to choose:

    ungated               311 pairs @ 0.2944 um
    at_bound filter only  295 pairs @ 0.2180 um

Dropping fits that stopped on a constraint keeps 95% of pairs and
improves the median 26%. An at-bound fit reports the bound, not a
measurement, and its Jacobian CI does not describe it either.

**Recommendation (one dataset -- verify elsewhere):** `at_bound` as an
unconditional filter, `occupancy` as the tunable gate, `max_uncert_z_nm`
if independent depth control is wanted. Drop both v1 heritage ratios:
`min_hb_ratio` is untunable (311 pairs at 1.0, 40 at 1.2, ~10 by 1.6 --
a 0.1 change swings coverage by an order of magnitude) and
`min_ah_ratio` is dominated by occupancy, which measures the same intent
properly.

Gate thresholds are in NANOMETRES with lateral and axial INDEPENDENT. v1
wrote the axial gate as `2 * max_uncert` in pixels, assuming a plane is
twice a pixel; here a plane is 0.2 um and a pixel 0.208 um.

## PSF calibration

No bead stack exists, so the PSF is recovered from reference-hybe spots
(many point-like emitters, one optical configuration, no alignment
needed). Candidates: gaussian / moffat / lorentzian / gaussian_halo.

    fiducial  gaussian_halo  sigma_xy 246 nm  sigma_z 628 nm
    readout   gaussian_halo  core 146 nm, halo ~2.1x, 17% of peak

The fiducial is an EXTENDED object (it is the whole genomic region the
readouts collectively trace); the readout is a point source with a
scattering halo.

**The readout calibration is poorly constrained** and this is a property
of the data, not the code. The selected FAMILY flips with crop count and
with crop SELECTION: 40 crops brightest-first give gaussian 188 nm,
first-40 give gaussian_halo 146 nm; at 2-8 crops it flips to lorentzian
with sigma at or near its lower bound.

`psf.plausible()` therefore rejects any parameter on a declared bound and
any sigma below the optical limit (70 nm lateral / 150 nm axial), and
`calibrate()` prefers the best-scoring PLAUSIBLE candidate. Score alone
cannot arbitrate: a 39 nm core scored rss/vox 2986 against 2931 for a
312 nm one on the same data. **Known limit:** it does not catch merely
BIASED answers, e.g. the 2-crop readout result of 87 nm.

Cost, 40 crops: fiducial 4.6 min, readout 31 min. Once per experiment.

**Tiering** (2 crops for a fast preview, 40 for the real thing) works for
the fiducial -- within 27 nm of the 40-crop answer in 16 s vs 227 s -- and
is UNSAFE for the readout for the reasons above. A preview tier there
needs the plausibility gate plus a fallback to the last validated PSF.

**Projection mode** (`mode='projections'`) is correct -- sigma agrees to
1-5 nm with the 3D fit -- but 0.2-0.7x the speed. Diagnosis: 13
parameters vs 8, and `least_squares` with a numerical Jacobian costs
O(n_params) per iteration, which swamps the point-count saving. Kept as
an independent cross-check, not the default.

## Open

- **Universal PSF.** One microscope means one optical configuration, so a
  single default shape (overridable) may serve every experiment. Test:
  calibrate other experiments independently and see whether shapes agree
  better than the ~40 nm scatter seen WITHIN MP58 across crop selections.
- **Do the gate curve shapes hold elsewhere?** The recommendation above
  rests on one dataset.
- **Nothing is wired into the app.** v1 runs unchanged in the pipeline;
  `engine.py` does not expose v2; no default threshold was altered; and
  `analysis/psf.json` has never been written -- deliberately, since the
  fiducial and readout shapes differ enough that one experiment-wide
  entry would be the wrong schema.
- The readout replicate score has only ~7 valid pairs per allele, so
  tight gates leave few pairs. 48 alleles gives 336 possible; more
  alleles would help the tightest thresholds.
