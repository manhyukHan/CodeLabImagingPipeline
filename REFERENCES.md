# References

Every published method this pipeline actually calls, tied to the code that
calls it. Assembled by reading the source, not the dependency list — a
package appearing in `requirements.txt` is not evidence that a method from
it is used.

Entries marked **⚠ verify** are ones the author should check against the
publisher's record before submission; the rest are stable citations.

---

## Segmentation

**Cellpose** — the default nuclear/cell segmentation route. `cellpose.models`
is imported lazily and the `cyto3` model is loaded on first use
(`codelab_pipeline/segmentation/segment.py:18-21`). The environment pins
`cellpose>=3.1,<4` because Cellpose 4 removed `models.Cellpose`
(`requirements.txt:13-20`).

**Both papers belong in the manuscript, not just the first.** The 2021 paper
is the method; `cyto3` is a model introduced by the Cellpose 3 paper, so
naming the model without that citation attributes it to the wrong work.

> Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a
> generalist algorithm for cellular segmentation. *Nat. Methods* **18**,
> 100–106 (2021). doi:10.1038/s41592-020-01018-x

> Stringer, C. & Pachitariu, M. Cellpose3: one-click image restoration for
> improved cellular segmentation. *Nat. Methods* **22**, 592–599 (2025).
> doi:10.1038/s41592-025-02595-5

### The model weights are not in this repository

Cellpose downloads them to `~/.cellpose/models/` on first use. They are
neither versioned nor pinned, so "install `cellpose>=3.1,<4`" does **not**
by itself specify the segmentation. Record the weights, not just the
package. Verified on the machine this protocol was developed on
(2026-09-06), SHA-256:

| file | bytes | sha256 |
|---|---|---|
| `cyto3` | 26,566,255 | `2dc3087a8abd7da46d1ab0ddd5824639933cc3ff63b382af3fa1939a392db93c` |
| `size_cyto3.npy` | 3,627 | `87f91feb48019b4dfb0400e7bbb89aca5873368686815aa7b4431c0faee97fc5` |
| `nucleitorch_0` | 26,563,614 | `89ca45e4a45048d5010d29621466b1274b91b3dcb9714dce9f9e90a9a8671303` |
| `size_nucleitorch_0.npy` | 3,627 | `a79107c9f284b569dd65f3a7363014d613a3403241c062822f07b6f026b4d3ef` |

`cyto3` is what `segment.py` loads; `nucleitorch_0` is the size/nuclei
companion Cellpose fetches alongside it. Re-check these before submission
— if they differ, the machine has a different `cyto3` than the one every
number in this protocol was produced with, and that is worth knowing.

### Two caveats that belong in the manuscript

- **Version.** The protocol was developed and validated against Cellpose
  **3.1.1.3**. Cellpose 4 (Cellpose-SAM) is the actively developed line and
  replaces `models.Cellpose`/`model_type='cyto3'` with `CellposeModel`/`cpsam`;
  this pipeline has not been ported to it, so results here should not be
  assumed to reproduce under 4.x.
- **Device.** Segmentation runs on GPU when one is present and falls back to
  a fresh CPU model if the GPU path raises
  (`segmentation/segment.py:57-65`). The fallback exists for PyTorch MPS on
  macOS, which is less mature than CUDA. Masks are therefore not guaranteed
  bit-identical across machines.

**Watershed** — the classical segmentation route, via
`skimage.segmentation.watershed` (`segmentation/segment.py`, 3 call sites),
seeded by `skimage.feature.peak_local_max`.

> Beucher, S. & Lantuéjoul, C. Use of watersheds in contour detection. In
> *Int. Workshop on Image Processing: Real-time Edge and Motion
> Detection/Estimation* (1979).

⚠ verify — scikit-image's own implementation notes also credit Beucher &
Meyer (1993); cite whichever matches the description you write.

---

## Alignment

**ORB feature detection** — seeds the FOV-level fit.
`cv2.ORB_create` with `cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)`
(`codelab_pipeline/io/preprocess.py`, feature-matrix path).

> Rublee, E., Rabaud, V., Konolige, K. & Bradski, G. ORB: an efficient
> alternative to SIFT or SURF. In *Proc. IEEE Int. Conf. Computer Vision*
> 2564–2571 (2011).

**RANSAC** — robust estimation of the seed affine, via
`cv2.estimateAffinePartial2D(..., cv2.RANSAC)`.

> Fischler, M. A. & Bolles, R. C. Random sample consensus: a paradigm for
> model fitting with applications to image analysis and automated
> cartography. *Commun. ACM* **24**, 381–395 (1981).

**Powell's method** — refines the seed over `[dx, dy, angle]`, the pipeline's
FOV-level optimiser (`preprocess.py:721`, `preprocess.py:802`,
`method='Powell'`). The decision to leave the angle *free but seeded* is
measured, not assumed (`alignment/chain.py:588-592`).

> Powell, M. J. D. An efficient method for finding the minimum of a function
> of several variables without calculating derivatives. *Comput. J.* **7**,
> 155–162 (1964).

**Phase correlation** — the per-cell residual fit and the Z leg.
`cv2.phaseCorrelate` at 10 call sites is the primary path; the
sub-pixel-refined variant `skimage.registration.phase_cross_correlation`
appears at 2.

> Kuglin, C. D. & Hines, D. C. The phase correlation image alignment method.
> In *Proc. IEEE Int. Conf. Cybernetics and Society* 163–165 (1975).

> Guizar-Sicairos, M., Thurman, S. T. & Fienup, J. R. Efficient subpixel
> image registration algorithms. *Opt. Lett.* **33**, 156–158 (2008).
> — the algorithm behind `phase_cross_correlation`; cite only if the
> manuscript describes the sub-pixel path.

---

## Spot detection and fitting

**Local maxima detection** — `skimage.feature.peak_local_max`, 10 call
sites across detection, multi-peak alignment and tracing.

**L-BFGS-B** — the bounded optimiser for the intensity model
(`codelab_pipeline/analysis/detection.py:65`).

> Byrd, R. H., Lu, P., Nocedal, J. & Zhu, C. A limited memory algorithm for
> bound constrained optimization. *SIAM J. Sci. Comput.* **16**, 1190–1208
> (1995).

**Levenberg–Marquardt / trust-region least squares** —
`scipy.optimize.least_squares` for the 3D Gaussian fits.

> More, J. J. The Levenberg-Marquardt algorithm: implementation and theory.
> In *Numerical Analysis* (ed. Watson, G. A.) 105–116 (Springer, 1978).

⚠ verify — `least_squares` defaults to Trust Region Reflective, not LM.
Check which method the call actually requests before citing LM.

---

## Thresholding

Four automatic thresholds are reachable; which one runs is a parameter, so
cite only those the manuscript reports using.

| Method | Call sites | Citation |
|---|---|---|
| Otsu | `threshold_otsu` ×2 | Otsu, N. A threshold selection method from gray-level histograms. *IEEE Trans. Syst. Man Cybern.* **9**, 62–66 (1979). |
| Yen | `threshold_yen` ×2 — the default for `_clip_background` (`chain.py:284`) | Yen, J.-C., Chang, F.-J. & Chang, S. A new criterion for automatic multilevel thresholding. *IEEE Trans. Image Process.* **4**, 370–378 (1995). |
| Li | `threshold_li` ×1 | Li, C. H. & Lee, C. K. Minimum cross entropy thresholding. *Pattern Recognit.* **26**, 617–625 (1993). |
| Triangle | `threshold_triangle` ×1 | Zack, G. W., Rogers, W. E. & Latt, S. A. Automatic measurement of sister chromatid exchange frequency. *J. Histochem. Cytochem.* **25**, 741–753 (1977). |

---

## Software

Cite these for the implementations the pipeline is built on.

> Harris, C. R. et al. Array programming with NumPy. *Nature* **585**,
> 357–362 (2020).

> Virtanen, P. et al. SciPy 1.0: fundamental algorithms for scientific
> computing in Python. *Nat. Methods* **17**, 261–272 (2020).

> van der Walt, S. et al. scikit-image: image processing in Python. *PeerJ*
> **2**, e453 (2014).

> Bradski, G. The OpenCV Library. *Dr. Dobb's Journal of Software Tools*
> (2000).

> Hunter, J. D. Matplotlib: a 2D graphics environment. *Comput. Sci. Eng.*
> **9**, 90–95 (2007).

> McKinney, W. Data structures for statistical computing in Python. In
> *Proc. 9th Python in Science Conf.* 56–61 (2010).

> Collette, A. *Python and HDF5* (O'Reilly, 2013). — for h5py; the file
> format itself is The HDF Group's HDF5, https://www.hdfgroup.org/HDF5/

`tifffile` (C. Gohlke) has no canonical paper; cite the software and version
if the TIFF ingestion path is described.

---

## Not cited, and why

- **PyQt5 / Qt** — user interface only; no method depends on it.
- **psutil, threadpoolctl** — process and thread-pool control; engineering,
  not method.
- **torch** — present only as Cellpose's backend. Cite Cellpose, not torch,
  unless the manuscript reports GPU timings, in which case name the CUDA
  build (`torch 2.7.1+cu118`, `requirements.txt:6-34`).

---

## Still missing

There is no citation anywhere in the source for the **biological** method —
sequential DNA-FISH / chromatin tracing itself. A protocol paper needs the
lineage it builds on. Nothing in the repository records which prior protocol
this implements, and that is the author's to supply. See
`docs/protocol_source.md`, *Before submission*, section C.
