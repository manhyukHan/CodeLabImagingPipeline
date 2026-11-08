# The physics term: how it was measured

Measured 2026-11-06 on every chromatin trace this project holds. This
file is the record: what was fitted, on what, what the numbers are, and
which attempts failed first. `chain.py` carries the values; nothing here
is tuned to make a downstream answer come out.

## What it replaces

Bottom-up path optimization scored a spot by its distance to a cluster
centroid. That loss **knew nothing of genomic separation** — readout 1
and readout 90 were treated identically — and the only thing holding an
allele together was `bottom_up.DEFAULT_REACH_UM`, a single 1.0 µm radius
that served four roles at once:

| role | what it did |
|---|---|
| support radius | how many distinct bins count as "near" a seed |
| seed spacing | two seeds within it are one allele |
| **hard cutoff** | `d = np.where(d <= reach, d, np.inf)` — beyond it, *not this path's at any price* |
| **unassigned price** | the cost of giving a spot to nobody |

The third role is the damaging one. An allele larger than the radius had
its outer half structurally excluded: measured on chr19 (allele Rg 1.50 µm
against reach 1.00 µm), 52% of the bins bottom-up dropped had a spot
present but unassigned, and **100% of those spots lay beyond reach**,
at a median 2.33 µm from the path centre. The value 1.0 µm came from
MP58, whose domain is half chr19's.

The replacement is a likelihood, so the scale is set by the pair's own
genomic separation rather than by a constant:

```
s (kb)    sigma_poly    P(same chain) = 0.5 at
    25         0.135                 0.50 um
   100         0.196                 0.66 um
  1000         0.362                 1.09 um
  2000         0.435                 1.31 um
```

A fixed 1.0 µm was too generous at short s — two adjacent loci a micron
apart are essentially never one fibre — and too tight at long s, where
1.3 µm is ordinary.

## The model

For a pair of loci with separation vector `d` (µm) and genomic
separation `s` (kb):

```
P(d | s) = (1 - e) N(d; 0, C_chain(s)) + e N(d; 0, C_other)

C_chain(s) = diag(sp^2 + 2 Lxy^2, sp^2 + 2 Lxy^2, sp^2 + 2 Lz^2)
C_other    = diag(Txy^2, Txy^2, Tz^2)
sp         = sigma_poly(s) = A * s^alpha
```

The second component is not a nuisance term. It is what bounds the cost:
when `d` is large for its `s` the chain term collapses and the cost
saturates at `-log(e N(d; 0, C_other))` instead of growing as
`|d|^2 / 2 sigma^2`. One wrongly assigned locus therefore cannot dominate
a path, and when enough loci pay the fixed price, **two paths explain the
cell more cheaply than one** — which is the mechanism that separates
sister chromatids, and the job `reach` was doing badly.

Use **consecutive observed loci** (a Markov factorisation, O(n)); all
pairs double-counts. Because loci go missing, "consecutive" has a
varying `s`, which is exactly why `sigma` must be a function of `s`.

## The data

Every stored trace, any route — the curve is a property of the polymer,
not of how we found it. Positions are `polymer_adj` (fiducial-corrected);
up to 8 FOVs per store.

| panel | store | kb/step | alleles used | pairs |
|---|---|---|---|---|
| chr19 downstream | `G:\JP\2026-01-26-JP-chr19_downstream` | 200 | 2034 | 2.35 M |
| RSF1 chr20 | `Q:\...\RSF1Analysis_withCODELabAnalyzer` | 30 | 3346 | 3.84 M |
| MP58 (Q) | `Q:\...\20251130_MP58_RNAandDNA_\Project_Root` | 50 | 1033 | 0.48 M |
| MP58 (G) | `G:\Seonghyeok\2025-11-30-MP58` | 50 | 894 | 0.46 M |
| HoxA | `G:\Dariya\2026-08-22-RNA_DNA_HoxA` | 5 | 53 | 2.6 k |

Three genomes (hg38, hg19, CriGri-PICRH), two species, 5 kb to 200 kb
per step. `s` comes from each panel's own probe design on `E:\Oligos`,
midpoint to midpoint — **never `|delta readout| * resolution`**, because
chr19's panel covers two regions with a 148 kb gap and a design may skip
a readout that failed specificity.

`RSF1 images its readouts at 647, not 635`; asking for 635 returned zero
bins and reported the panel as having no traces at all.

## The fit

Two stages, because `sigma_poly` and `sigma_loc` are perfectly confounded
inside one `s` bin — only their sum is observed.

1. **Repeat pairs (s = 0)**, where a `Rep_` round re-images the same
   readout. The polymer contributes nothing, so the per-axis variance is
   exactly `2 sigma_loc^2`.
2. **s > 0 pairs**, with `sigma_loc` fixed, fitting `sigma_xy,b` and
   `sigma_z,b` per `s` bin plus a global `(Txy, Tz)` and `e_b`.
3. **Isotropy is measured, not imposed.** Leaving the two axes free keeps
   every M step closed form *and* turns the assumption into a test:
   `sigma_poly,xy` and `sigma_poly,z` must agree if the polymer is
   isotropic and the instrument model is right.

Both components are diagonal Gaussians, so every M step is a weighted
second moment — **no optimizer anywhere**:

```
chain   sigma_xy,b^2 = sum w (dy^2 + dx^2) / (2 sum w)
        sigma_z,b^2  = sum w dz^2 / sum w
other   Txy^2 = sum (1-w)(dy^2 + dx^2) / (2 sum (1-w))      pooled over bins
        Tz^2  = sum (1-w) dz^2 / sum (1-w)                   pooled over bins
mix     e_b = 1 - mean(w)_b
```

Separation **vectors**, not distances: reducing to `|d|` discards the
axis information, and the axes were the question.

## The numbers

Isolated alleles (nearest other allele in the same cell > 3 µm):

| panel | alpha | A | Lxy | Lz | Txy | Tz | poly z/xy | e |
|---|---|---|---|---|---|---|---|---|
| RSF1 30kb | 0.222 | 0.0669 | 0.018 | 0.043 | 0.638 | 1.076 | 1.11 | 0.270 |
| MP58 Q 50kb | 0.280 | 0.0539 | 0.031 | 0.076 | 0.737 | 1.257 | 1.03 | 0.347 |
| MP58 G 50kb | 0.267 | 0.0584 | 0.033 | 0.087 | 0.685 | 1.177 | 1.00 | 0.320 |
| *chr19 200kb* | *0.269* | *0.0550* | *0.035* | *0.162* | *0.667* | *1.559* | ***1.56*** | *0.281* |

R² 0.97–0.99 on every fit.

**`alpha ~ 0.27`** — confined chromatin, not an ideal chain's 0.5.

### Why chr19 sets no default

Its `sigma_poly,z / sigma_poly,xy` is **1.56** where the others read
1.00–1.12, and it is also the panel whose z precision is worst by far
(`Lz/Lxy` = 4.6 against 2.3–2.7). The two move together, so the likely
reading is an **incomplete z instrument model on that store**, not an
anisotropic polymer — but that is not settled. On chr19, carry the axes
separately or use lateral only. It does **not** support applying
`z_scale_correction`: the large ratio sits in localization *precision*,
which the loss already handles by carrying `Lz` separately.

HoxA is excluded for having 53 usable alleles — its fit returned a
negative exponent at R² 0.45.

### How the defaults were combined

`A` and `alpha` trade off, so the default pair is **not** the median of
each column. The three panels are anchored at s = 100 kb, where they
give `sigma_poly` = 0.186, 0.196 and 0.200 µm; the median there with the
median alpha fixes `A`. Over 30–1000 kb the default stays within ~10% of
all three.

```
SIGMA_POLY_A = 0.0572      SIGMA_POLY_ALPHA = 0.267
TAU_XY = 0.685             TAU_Z = 1.177             E_MIX = 0.32
SIGMA_LOC_XY = 0.031       SIGMA_LOC_Z = 0.076       <- per store, measure it
```

`sigma_loc` spans 0.018–0.033 µm laterally across three panels, a factor
of 1.8, because it is a property of the imaging. Measure it per store
from the repeats (`calibration_from_repeats`); the module default is a
starting point, and a result that used it should say so.

## Two things the measurement itself found

**The contaminant is stretched in z, in every panel**: `Tz/Txy` =
1.69–2.47, no exception. That is the "one MIP anchor, two alleles stacked
in z" case as a number, on four experiments rather than one picture.

**Existing traces mix alleles at ~15–19% per locus.** The pair-level `e`
falls on the isolated subset in all four panels (0.318→0.281,
0.356→0.270, 0.385→0.347, 0.338→0.320) but does not collapse, and `Tau`
falls with it — the surviving contaminant is *closer*. Isolation removes
non-sister alleles; sister chromatids are adjacent by nature and cannot
be excluded that way, so the residual is read as sister mixing. That is
precisely what path optimization is for and what the old loss could not
see.

## What failed first

**A plain rms gave alpha 0.08–0.16** — far flatter than confined
chromatin. The tables said why: chr19's 100–250 kb bin had median 0.522 µm
against rms 1.097, a ratio of 2.1 where a clean 3D Gaussian gives 1.126.
Allele-mixed pairs sit ~4 µm apart whatever their `s`, and because that
contamination is `s`-independent it flattens the exponent. Fitting the
loss to it would have taught the loss that a large separation is normal.
Every scale is therefore robust (median under the Maxwell law) or comes
from the mixture fit.

**A plain moment initialised the EM wrong.** Repeat pairs carry the same
heavy tail (rms/median 2.9–8.1), so `mean(d^2)` starts the *chain*
component at the contaminant's width, and the only fixed point EM then
finds puts 87–92% of the repeats in the contaminant and reports
`sigma_loc` as 0.35–0.41 µm — an order of magnitude past the panel that
converged properly. Initialise from `1.4826 * MAD` and start the
contaminant at the 99th percentile.

**`e` is not identifiable at large `s`.** As `sigma_chain` approaches
`Tau` the two components stop being distinguishable and EM splits them
arbitrarily; chr19's `e` rising 0.24 → 0.55 with `s` is that degeneracy,
not more contamination. Read `e` from the short-`s` bins.
`saturation_um` returns infinity in the same regime, which is the honest
statement that a long-range pair carries little information about which
path it belongs to.

## Where `s` comes from in the pipeline

`s` requires a readout's genomic position, and **nothing in the pipeline
carried one** before `panel.py`. The experiment layout has FolderName,
Readouts, DataType, HybNum and channels; `genomic_resolution_kb` is one
per-experiment number, often blank, used as classifier context. The
designs live on `E:\Oligos`.

A store now names its own design, per modality, in the same places
`layout_path` lives:

```
Ingestion panel    "<modality> probe design (optional):"  lineedit + Browse
configs/*.xml      <modality name="DNA" probe_design_path="E:\Oligos\..." />
manifest.json      modalities.DNA.probe_design_path
params.json        modalities.DNA.probe_design_path       <- written by Parse Layouts
```

Read it back with `analysis_store.probe_design_path(storage_path)` (the
store's own params win over the manifest, which every startup rewrites
from whatever config was loaded) and hand the result to
`panel.read_probe_design`. Parse Layouts logs what the design contains
and says so when its own median step disagrees with the configured
`genomic_resolution_kb` by more than 10% — a design pointed at another
panel of the same library is otherwise invisible, because it yields a
plausible wrong separation rather than an error.

The five configs whose panel is known are filled in (MP58 ×2, chr19 ×2,
RSF1); each one's design steps exactly what its config claims
(`tests/test_probe_design_slot.py` checks that against the real files
when `E:` is reachable). HoxA and MAZ are not filled: their library
folders hold several candidate designs and which one was imaged is not
recorded anywhere I can read.

## Where the loss ended up, and one correction

`localization/pathfind.py` is the loss in use. Two things about it are
corrections to what this file said when the constants were measured.

**The step term is the chain component alone, not `pair_cost`.** The
mixture's second component *is* "a second allele, or a wrong pick" —
which a multi-path model names explicitly as another path or as clutter.
Scoring a step with the mixture double-counts that alternative and makes
a step onto the other allele cheap exactly where the two have to be told
apart. Measured, it costs purity only where the alleles overlap (0.4 µm
apart: 0.885 with the mixture, 0.934 with the chain term; identical at
0.6 µm and beyond). The mixture is not retracted — it is right where it
was measured, as the description of a pair in a *stored* trace, and
`p_same_chain` scores one.

**The alternative to a step is a Poisson clutter rate, not a volume.**
`-log rho` with `rho` the cell's own unclaimed-spot density, re-estimated
from what the previous pass left over. A flat `log V` built whole paths
out of noise (one allele plus 120 clutter spots returned k = 3 in six
seeds of six); assuming *every* spot is clutter went the other way and
declined 11.5% of the real loci on a 91×200 kb panel. The fixed point
gives 0.1%, with k and the clutter cells unchanged.

`DEFAULT_REACH_UM` is gone. What survives is `bottom_up.SEED_RADIUS_UM`,
for seeding and for the count proposals only; nothing gates a
measurement any more. The distance the loss actually works at is
`pathfind.acceptance_um(s, n_spots)`, which moves with `s` (0.43 µm at
25 kb to 0.70 µm at 1 Mb for a 100-spot cell) and turns over past about
a megabase, where a step stops carrying information at any distance.

## Reproducing

Scripts are in the 2026-11-06 session scratchpad: `trace_inventory.py`
(which stores hold traces), `rs_curve2.py` (robust R(s)),
`mixture_fit.py` (scalar two-component), `clean_and_lateral.py` (isolated
subset and lateral-only), `aniso_em.py` (the two-stage anisotropic fit
these numbers come from).
