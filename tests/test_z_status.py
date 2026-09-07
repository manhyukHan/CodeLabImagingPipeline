"""
z_status has THREE states, and every consumer respects the third one.

WHY THIS EXISTS: the verdict of 3D localization used to live in
`spot._z_status`, a session-transient Python attribute that its own
docstring said was deliberately not persisted. Reopening the app erased
it. Three features now depend on it surviving -- Remove Z-rejected,
allele building, and the Analysis distance gate -- and all three turn on
the SAME distinction:

    accepted   the fit ran and succeeded
    rejected   the fit ran and said no          <- a measured negative
    not_fit    nobody has run the fit           <- unknown, not a negative

Collapsing the last two is the failure mode this guards. A sweep aimed at
"the fit rejected this" that also removed "nobody looked yet" would
delete data on the strength of a question never asked, and it would do so
silently, because the two are indistinguishable from the coordinate
alone: manual anchoring writes z exactly 0.0, and so does a real emitter
on plane 0.

The compatibility direction matters as much. Every store written before
2026-09-07 has no z_status column, and every spot in one is genuinely
unfitted -- so absence must read as not_fit, never as an error and never
as accepted.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_z_status.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import h5py                                                 # noqa: E402

from codelab_pipeline.models.spot import (                   # noqa: E402
    ASpot, Z_ACCEPTED, Z_REJECTED, Z_NOT_FIT, Z_STATUSES, z_status_of)
from codelab_pipeline.io import columnar                     # noqa: E402
from codelab_pipeline.analysis import distances              # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def spot(uid, status, cell=1, y=0.0, x=0.0, z=0.0):
    s = ASpot()
    s.set_metadata(uid=uid, fov=1, modality='DNA', hybe='Hyb_001', channel=635,
                   cell=cell, adj_coordinate=(y, x, z), raw_coordinate=(y, x, z),
                   z_status=status)
    return s


class FakePop:
    """The smallest thing distances._spots_of will accept."""

    def __init__(self, frame):
        self.spots = frame
        self.alleles = None


def frame(rows, with_status=True):
    cols = ['fov', 'cell', 'celltype', 'modality', 'hybe', 'channel',
            'y_um', 'x_um', 'z_um', 'brightness']
    if with_status:
        cols.append('z_status')
    return pd.DataFrame([{k: r[k] for k in cols} for r in rows])


def row(cell, y, x, z, status):
    return dict(fov=1, cell=cell, celltype='', modality='DNA', hybe='Hyb_001',
                channel=635, y_um=y, x_um=x, z_um=z, brightness=1.0,
                z_status=status)


def main():
    print('\n-- the model carries three states and defaults to the third --')
    check('a fresh spot is not_fit', ASpot().z_status == Z_NOT_FIT)
    check('there are exactly three', len(Z_STATUSES) == 3, str(Z_STATUSES))
    for st in Z_STATUSES:
        check(f'{st} survives set_metadata', spot(1, st).z_status == st)
    s = ASpot()
    s.set_metadata(z_status='something a newer build wrote')
    check('an unknown value degrades to not_fit, never raises',
          s.z_status == Z_NOT_FIT)

    class Bare:
        pass
    check('z_status_of tolerates an object with no attribute at all',
          z_status_of(Bare()) == Z_NOT_FIT)

    print('\n-- it survives a store round trip --')
    tmp = os.path.join(tempfile.mkdtemp(), 'spots.h5')
    dicts = [spot(i + 1, st).save() for i, st in enumerate(Z_STATUSES)]
    check('save() carries it',
          [d['z_status'] for d in dicts] == list(Z_STATUSES))
    with h5py.File(tmp, 'w') as f:
        columnar.pack_spots(f.create_group('spots'), dicts)
    with h5py.File(tmp, 'r') as f:
        back = columnar.unpack_spots(f['spots'])
    check('unpack returns what pack was given',
          [d['z_status'] for d in back] == list(Z_STATUSES))

    print('\n-- a store older than the field reads as not_fit --')
    with h5py.File(tmp, 'a') as f:
        del f['spots/z_status']            # exactly what an old store looks like
    with h5py.File(tmp, 'r') as f:
        old = columnar.unpack_spots(f['spots'])
    check('every spot in a pre-field store is not_fit',
          [d['z_status'] for d in old] == [Z_NOT_FIT] * 3)
    check('and it is not an error to read one', len(old) == 3)

    print('\n-- the distance gate: which statuses each mode admits --')
    check("3D admits accepted only",
          set(distances.usable_z_statuses('xyz')) == {Z_ACCEPTED},
          str(distances.usable_z_statuses('xyz')))
    check("2D admits accepted and not_fit",
          set(distances.usable_z_statuses('xy')) == {Z_ACCEPTED, Z_NOT_FIT},
          str(distances.usable_z_statuses('xy')))
    check('rejected is admitted by NEITHER -- the fit rejected the emitter, '
          'not just its depth',
          Z_REJECTED not in distances.usable_z_statuses('xyz')
          and Z_REJECTED not in distances.usable_z_statuses('xy'))

    print('\n-- and the gate really filters the table --')
    rows = [row(1, 0.0, 0.0, 0.0, Z_ACCEPTED),
            row(1, 1.0, 0.0, 0.0, Z_NOT_FIT),
            row(1, 2.0, 0.0, 0.0, Z_REJECTED),
            row(1, 3.0, 0.0, 0.0, Z_ACCEPTED)]
    pop = FakePop(frame(rows))
    got3 = distances._spots_of(pop, ('DNA', 'Hyb_001', 635), dims='xyz')
    got2 = distances._spots_of(pop, ('DNA', 'Hyb_001', 635), dims='xy')
    check('3D keeps only the two accepted', len(got3) == 2, str(len(got3)))
    check('2D keeps accepted + not_fit, three of four', len(got2) == 3,
          str(len(got2)))
    check('no rejected spot survives either mode',
          Z_REJECTED not in set(got3['z_status']) | set(got2['z_status']))

    print('\n-- pair_distances honours it end to end --')
    # Two accepted spots 3 um apart in y, plus a rejected and an unfitted
    # one. 3D must see exactly one pair; 2D must see three (3 choose 2).
    d3 = distances.pair_distances(pop, ('DNA', 'Hyb_001', 635),
                                  ('DNA', 'Hyb_001', 635), dims='xyz')
    d2 = distances.pair_distances(pop, ('DNA', 'Hyb_001', 635),
                                  ('DNA', 'Hyb_001', 635), dims='xy')
    check('3D yields one pair from the two accepted spots', len(d3) == 1,
          str(len(d3)))
    check('and its length is the real 3 um', abs(d3['d_um'].iloc[0] - 3.0) < 1e-9,
          f'{d3["d_um"].iloc[0]:.3f}' if len(d3) else 'none')
    check('2D yields three pairs from the three admitted spots', len(d2) == 3,
          str(len(d2)))

    print('\n-- a hand-built population without the column is NOT filtered --')
    # Tools and tests assemble these directly. Filtering an absent column
    # would drop every row and turn a working analysis into an empty one
    # for a reason nothing on screen explains.
    bare = FakePop(frame(rows, with_status=False))
    got = distances._spots_of(bare, ('DNA', 'Hyb_001', 635), dims='xyz')
    check('all four rows survive when there is no column to judge them by',
          len(got) == 4, str(len(got)))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
