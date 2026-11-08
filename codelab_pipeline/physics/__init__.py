"""Measured physics for the chromatin path loss.

Two things live here and nothing else: WHERE a readout sits on the
genome (`panel`), and WHAT a separation between two loci is worth given
that (`chain`). Neither reads a store; both are given their inputs.

Which design file a given store was imaged with is the store's own
fact, so it is recorded there and read back with
`io.analysis_store.probe_design_path(storage_path)` -- hand that to
`panel.read_probe_design`.

They are kept out of `analysis` because analysis modules read a store
and produce a result, and out of `localization` because the same law
belongs to tracing as much as to bottom-up. `MEASUREMENT.md` beside this
file carries the derivation, the data it was measured on, and the fits
that failed first.
"""
from codelab_pipeline.physics.chain import (
    Calibration, calibration_from_repeats, log_p_chain, log_p_other,
    p_same_chain, pair_cost, saturation_um,
    SIGMA_POLY_A, SIGMA_POLY_ALPHA, SIGMA_LOC_XY, SIGMA_LOC_Z,
    TAU_XY, TAU_Z, E_MIX,
)
from codelab_pipeline.physics.panel import ProbeDesign, read_probe_design

__all__ = [
    'Calibration', 'calibration_from_repeats', 'log_p_chain', 'log_p_other',
    'p_same_chain', 'pair_cost', 'saturation_um', 'ProbeDesign',
    'read_probe_design', 'SIGMA_POLY_A', 'SIGMA_POLY_ALPHA',
    'SIGMA_LOC_XY', 'SIGMA_LOC_Z', 'TAU_XY', 'TAU_Z', 'E_MIX',
]
