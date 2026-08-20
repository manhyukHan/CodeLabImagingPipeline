"""
Dead code -- see legacy/README.md. A ChrTracer3 (MATLAB) CSV-import bridge, never
imported by anything in this project: the live chromatin-tracing path builds
AnAllele objects from this pipeline's own 3D localization instead
(codelab_pipeline/localization/localization.py's localize_chromatin_trace_hybe,
which ui/chromatin_tracing_panel.py drives). Moved here on 2026-08-17, verbatim
apart from making the AnAllele import absolute so it still resolves from legacy/.

Parses ChrTracer3's AllFits.csv / selectSpots.csv output and reconstructs
true image-referenced (full-FOV, top-left-origin) coordinates from them, then
builds AnAllele objects ready to link to ACell/ASpot.

Confirmed by tracing ChrTracer3_FitSpots.m and ChrTracer3p3_CropAllSpots.m
(local copy: /Users/hanmanhyuk/Remote Server/ChrTracingLib/) and verified
against real fov001_AllFits.csv / fov001_selectSpots.csv (197 cells):
 - x, y, z, fid_x, fid_y, fid_z in AllFits.csv are crop-local (relative to a
   small box cropped around the selected spot), in nm. ChrTracer3_FitSpots.m
   only adds back the *inner* fit-window trim, never the *outer* crop
   offset (ChrTracer3p3_CropAllSpots.m). locusX/locusY is the one column
   holding the true full-image position of the crop center, in nm:
   locusX_nm = selectSpots_locusX_px * nmXYpix.
 - s = cell/allele index within a FOV (matches selectSpots.csv row order,
   1-indexed; occasionally short a few rows where a spot's fit failed and
   was silently dropped by MATLAB's try/catch).
 - fs = CantorPair(fov, s) (globally unique, invertible). fsr =
   CantorPair(fs, readout).
 - dataType: H = main hybe readouts (genomic loci), R = repeat (QC),
   T = toehold (QC), B = barcode (cell/allele identity markers).
"""
import os
import warnings

import numpy as np
import pandas as pd

from codelab_pipeline.models.allele import AnAllele


def read_box_width(analysis_folder):
    """boxWidth (pixels) from that experiment's 'Pars Crop Spots.csv' -- a
    single scalar reused across the whole FOV batch loop in ChrTracer3.m,
    not a per-row value."""
    df = pd.read_csv(os.path.join(analysis_folder, 'Pars Crop Spots.csv'))
    return float(df['boxWidth'].iloc[0])


def read_pixel_size(analysis_folder):
    """(nmXYpix, nmZpix) from that experiment's 'Pars Fit Spots.csv'."""
    df = pd.read_csv(os.path.join(analysis_folder, 'Pars Fit Spots.csv'))
    return float(df['nmXYpix'].iloc[0]), float(df['nmZpix'].iloc[0])


def infer_pixel_size(all_fits_df, select_spots_df):
    """
    Fallback for when 'Pars Fit Spots.csv' isn't available: locusX_nm =
    selectSpots_locusX_px * nmXYpix is an exact per-cell relationship, so
    nmXYpix can be recovered directly from the first cell (s == min(s),
    matching selectSpots.csv's 1-indexed row order).
    """
    first_s = all_fits_df['s'].min()
    locus_x_nm = all_fits_df.loc[all_fits_df['s'] == first_s, 'locusX'].iloc[0]
    locus_x_px = select_spots_df['locusX'].iloc[0]
    return float(locus_x_nm) / float(locus_x_px)


def parse_all_fits(analysis_folder, fov):
    path = os.path.join(analysis_folder, f'fov{fov:03d}_AllFits.csv')
    return pd.read_csv(path)


def parse_select_spots(analysis_folder, fov):
    path = os.path.join(analysis_folder, f'fov{fov:03d}_selectSpots.csv')
    return pd.read_csv(path)


def validate_spot_count(all_fits_df, select_spots_df, tolerance=5):
    """
    selectSpots.csv (every spot the user selected) is usually a few rows
    longer than AllFits.csv's unique 's' count (every spot that actually got
    a valid fit) -- ChrTracer3_FitAllSpots.m silently drops a spot on any
    fitting error via try/catch. A mismatch within `tolerance` is expected,
    not a bug; only warn if it's larger than that.
    """
    n_selected = len(select_spots_df)
    n_fit = all_fits_df['s'].nunique()
    if abs(n_selected - n_fit) > tolerance:
        warnings.warn(
            f'selectSpots.csv has {n_selected} rows but AllFits.csv only has '
            f'{n_fit} unique s values -- larger gap than the usual few '
            f'silently-dropped-on-fit-error spots, worth checking.')
    return n_selected, n_fit


def _matlab_round(x):
    """
    MATLAB's round() rounds half away from zero; numpy/Python round to even.
    Every value passed through here is a non-negative pixel index, so
    floor(x + 0.5) reproduces MATLAB's convention exactly.
    """
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def add_image_coordinates(all_fits_df, box_width, nm_xy_pix):
    """
    Reconstructs true image-referenced coordinates by re-deriving the crop
    origin exactly as ChrTracer3p3_CropAllSpots.m computes it -- MATLAB,
    1-indexed: crop_x0 = round(max(1, locusX_px - boxWidth/2)) -- then
    EXPLICITLY converts that 1-indexed MATLAB pixel index to a 0-indexed
    Python/numpy pixel index (crop_x0_python = crop_x0_matlab - 1) before
    adding back the crop-local offset. codelab_pipeline is 0-indexed
    throughout (matches ACell/ASpot pixel conventions already established) --
    a missed -1 here is a full pixel (~nm_xy_pix nm) of systematic drift
    against every other image-space coordinate in the pipeline.

    Adds both pixel-space columns (abs_x_px/abs_y_px/abs_fid_x_px/abs_fid_y_px
    -- 0-indexed, top-left-referenced, for compatibility with the
    alignment-chain/ACell/ASpot pixel coordinate system) and nm-space columns
    (abs_x_nm/abs_y_nm/abs_fid_x_nm/abs_fid_y_nm -- physical distance, for
    polymer/distance-map analysis). z passes through unchanged (the crop
    step does not trim Z -- documented simplification, not re-derived here).
    """
    df = all_fits_df.copy()
    locus_x_px = df['locusX'] / nm_xy_pix
    locus_y_px = df['locusY'] / nm_xy_pix

    crop_x0_matlab = _matlab_round(np.maximum(1, locus_x_px - box_width / 2))
    crop_y0_matlab = _matlab_round(np.maximum(1, locus_y_px - box_width / 2))
    crop_x0_px = crop_x0_matlab - 1  # MATLAB 1-indexed -> Python 0-indexed
    crop_y0_px = crop_y0_matlab - 1

    df['abs_x_nm'] = crop_x0_px * nm_xy_pix + df['x']
    df['abs_y_nm'] = crop_y0_px * nm_xy_pix + df['y']
    df['abs_fid_x_nm'] = crop_x0_px * nm_xy_pix + df['fid_x']
    df['abs_fid_y_nm'] = crop_y0_px * nm_xy_pix + df['fid_y']

    df['abs_x_px'] = df['abs_x_nm'] / nm_xy_pix
    df['abs_y_px'] = df['abs_y_nm'] / nm_xy_pix
    df['abs_fid_x_px'] = df['abs_fid_x_nm'] / nm_xy_pix
    df['abs_fid_y_px'] = df['abs_fid_y_nm'] / nm_xy_pix

    return df


def _cantor_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 0.5 * (a + b) * (a + b + 1) + b


def _cantor_unpair(p):
    p = np.asarray(p, dtype=float)
    w = np.floor((np.sqrt(8 * p + 1) - 1) / 2)
    t = (w * w + w) / 2
    b = p - t
    a = w - b
    return a, b


def decode_fs(fs):
    """Inverse of MATLAB's fs = CantorPair(fov, s). Returns (fov, s)."""
    fov, s = _cantor_unpair(fs)
    return int(round(float(fov))), int(round(float(s)))


def decode_fsr(fsr):
    """Inverse of MATLAB's fsr = CantorPair(fs, readout). Returns (fs, readout)."""
    fs, readout = _cantor_unpair(fsr)
    return int(round(float(fs))), int(round(float(readout)))


def select_brightest_polymer(allele, n_bins):
    """
    Collapses allele.polymer (every candidate per readout) down to one
    committed (x, y, z) position per genomic-locus bin, picking the
    brightest candidate (max h) at each bin -- the same 'maxBrightness'
    selection QualityControlORCA.combineFOV uses (see
    /Users/hanmanhyuk/Remote Server/utils.py:
        t = candidates[:,-1].argmax()
        total_raw_polymers[i,j] = candidates[t]
    ). Only dataType == 'H' readouts (the main hybe/genomic-locus readouts)
    are binned, matching that module's own H-only 'hybes' convention -- R/T/B
    readouts are QC/identity markers, not polymer positions. Bins are
    1-indexed by readout number (readout r -> final_polymer[r-1]); readouts
    with no candidate at all stay NaN. n_bins must be passed in (not
    inferred per-allele) so every allele from the same FOV shares one
    consistent bin count -- distance-map computation downstream needs a
    uniform shape across cells.

    Sets and returns allele.final_polymer, an (n_bins, 3) np.array.
    """
    final_polymer = np.full((n_bins, 3), np.nan)
    for readout, candidates in allele.polymer.items():
        if not candidates or candidates[0][4] != 'H':
            continue
        if readout < 1 or readout > n_bins:
            continue
        brightest = max(candidates, key=lambda c: c[3])  # c = (x, y, z, h, dataType)
        final_polymer[readout - 1] = brightest[:3]
    allele.final_polymer = final_polymer
    return final_polymer


def build_alleles_from_fov(all_fits_df, fov, box_width, nm_xy_pix):
    """
    Groups AllFits rows by 's' (cell/allele index within this FOV) and
    builds one AnAllele per group. coordinate/raw_coordinate come from the
    group's (constant-per-s) reconstructed fiducial position; polymer keeps
    EVERY candidate detection per readout, not just the brightest one --
    multi-candidate loci are real (~4.8% of (s, readout) pairs have 2+
    candidates in real data, confirmed), and codelab_pipeline's other
    localization workers already carry a max_num_alleles-style
    keep-every-candidate parameter this mirrors. Each candidate keeps its
    dataType (H/R/T/B) for downstream QC. final_polymer is then filled in via
    select_brightest_polymer (see above) so every allele comes out ready for
    distance-map computation without a separate collapse step.
    """
    df = add_image_coordinates(all_fits_df, box_width, nm_xy_pix)
    h_readouts = df.loc[df['dataType'] == 'H', 'readout']
    n_bins = int(h_readouts.max()) if len(h_readouts) else 0

    alleles = []
    for s, group in df.groupby('s'):
        first = group.iloc[0]
        fiducial = (float(first['abs_fid_x_px']), float(first['abs_fid_y_px']), float(first['fid_z']))
        polymer = {}
        for readout, rows in group.groupby('readout'):
            polymer[int(readout)] = [
                (float(r.abs_x_px), float(r.abs_y_px), float(r.z), float(r.h), str(r.dataType))
                for r in rows.itertuples()
            ]
        allele = AnAllele()
        allele.set_metadata(id=int(s), fov=int(fov), cell=-1,
                             coordinate=fiducial, raw_coordinate=fiducial,
                             polymer=polymer)
        select_brightest_polymer(allele, n_bins)
        alleles.append(allele)
    return alleles
