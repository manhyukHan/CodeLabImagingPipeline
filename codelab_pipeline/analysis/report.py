"""
Every figure leaves with its evidence: PNG + CSVs + a provenance JSON.

A map without its population is not a result. save_result writes, next
to the PNG, one CSV per underlying table/array and a JSON sidecar
recording: the exact gate (Condition.to_dict), the sequential survivor
counts per predicate, how many cells survived PER CELLTYPE, the
population's shape, and the caller's parameters. A figure found on disk
months later then answers "what was gated, and how many" by itself.
"""
import datetime
import json
import os

import numpy as np
import pandas as pd


def gate_summary(pop, condition):
    """The provenance block the GUI shows and the sidecar records.

    {'predicates': [...to_dict...], 'sequential': [(repr, n_after)],
     'n_cells_total', 'n_cells_gated', 'by_celltype': {name: [gated,
     total]}} -- celltype '' reported as 'Unassigned', never dropped.
    """
    if condition is None or not getattr(condition, 'predicates', None):
        mask = np.ones(len(pop.cells), dtype=bool)
        seq = []
        gate_dict = {'clauses': []}
    else:
        mask = condition.mask(pop)
        seq = condition.report(pop)
        # the WHOLE serialized gate, not a hand-picked key: conditions
        # serialize as OR-of-AND {'clauses': ...}, and reaching for
        # ['predicates'] here was a KeyError that PyQt escalated to a
        # no-traceback abort when it escaped the gate-summary slot
        gate_dict = condition.to_dict()
    ct = pop.cells['celltype'].replace('', 'Unassigned')
    by = {}
    for name in sorted(ct.unique()):
        sel = (ct == name).to_numpy()
        by[name] = [int((sel & mask).sum()), int(sel.sum())]
    return {'gate': gate_dict, 'sequential': [[r, n] for r, n in seq],
            'n_cells_total': int(len(pop.cells)),
            'n_cells_gated': int(mask.sum()),
            'by_celltype': by}


def _table_to_csv(path, obj):
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    elif isinstance(obj, pd.Series):
        obj.to_frame().to_csv(path)
    else:
        arr = np.asarray(obj)
        if arr.ndim <= 2:
            np.savetxt(path, np.atleast_2d(arr), fmt='%.6g', delimiter=',')
        else:
            # stacks (e.g. per-group maps) flatten one slab per block row
            with open(path, 'w') as f:
                for i, slab in enumerate(arr):
                    f.write(f'# slab {i}\n')
                    np.savetxt(f, np.atleast_2d(slab), fmt='%.6g',
                               delimiter=',')


def save_result(out_dir, name, fig=None, tables=None, pop=None,
                condition=None, params=None, dpi=150):
    """Write <name>.png, <name>__<table>.csv..., <name>.json.

    Returns the list of written paths. Atomicity is not needed here --
    these are exports, not the store -- but names never collide silently:
    an existing <name>.png gets a numeric suffix rather than an
    overwrite.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = name
    k = 1
    while os.path.exists(os.path.join(out_dir, f'{base}.png')) or \
            os.path.exists(os.path.join(out_dir, f'{base}.json')):
        base = f'{name}_{k}'
        k += 1
    written = []
    if fig is not None:
        p = os.path.join(out_dir, f'{base}.png')
        fig.savefig(p, dpi=dpi, facecolor='white', bbox_inches='tight')
        written.append(p)
    for tname, obj in (tables or {}).items():
        p = os.path.join(out_dir, f'{base}__{tname}.csv')
        _table_to_csv(p, obj)
        written.append(p)
    side = {'name': base,
            'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'params': params or {}}
    if pop is not None:
        side['population'] = {
            'storage_path': str(getattr(pop, 'storage_path', '')),
            'fovs': list(getattr(pop, 'fovs', [])),
            'voxel_um': list(getattr(pop, 'voxel_um', [])),
            'summary': pop.summary() if hasattr(pop, 'summary') else '',
        }
        side['gate'] = gate_summary(pop, condition)
    p = os.path.join(out_dir, f'{base}.json')
    with open(p, 'w') as f:
        json.dump(side, f, indent=1)
    written.append(p)
    return written
