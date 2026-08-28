"""
The analysis toolbox: cell-based analysis over the v2 store, WITHOUT the app.

DESIGN CONTRACT (explicit decision, 2026-08-30): this package is an
independent utility suite. It imports nothing from windows/, ui/ or
canvas/, touches no Qt, and every entry point is callable from a plain
script or notebook against a bare storage_path:

    from codelab_pipeline.analysis import population, gate, ensemble
    pop = population.Population.build(r'G:/.../DNA', fovs=[1, 2],
                                      records=records)
    mask = gate.Condition([gate.CelltypeIn(['WT'])]).mask(pop)
    m, n = ensemble.ensemble_map(pop.dmaps(), mask)

The app USES these functions -- panels gather widget values, call here,
and display the returned arrays -- it never re-implements them, and
nothing here knows the app exists.

The two composition axes, per the founding spec:
  - CONDITIONS (gate.py) narrow the cell set: each predicate ANDs into a
    boolean cell mask over Population.cells.
  - FLAGS multiply the figures shown: celltype demultiplex, allele
    differences, abnormal alleles, FOV-level differences are GROUPINGS
    applied after gating, never filters.

Everything downstream of the store is in MICROMETRES: extractors scale
(y, x, z) by voxel_um per axis exactly once, at extraction. The one
pre-existing distmap in the app pdists raw (px, px, planes) -- display-
only there, wrong for science, and the trap this rule exists to bury.
"""
from codelab_pipeline.analysis import polymer          # noqa: F401
from codelab_pipeline.analysis import detection        # noqa: F401
from codelab_pipeline.analysis import distances        # noqa: F401
from codelab_pipeline.analysis import ensemble         # noqa: F401
from codelab_pipeline.analysis import expression       # noqa: F401
from codelab_pipeline.analysis import gate             # noqa: F401
from codelab_pipeline.analysis import population       # noqa: F401

DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)
