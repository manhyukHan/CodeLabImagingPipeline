"""
The per-cell cross-modal layer, and the gate that decides it may run.

This layer sits between the per-FOV cross-modal bridge and the per-cell
same-modality residual. Its correctness is almost entirely about ORDER
and about what happens when it is absent, so that is what is checked:

1. Absent, it changes nothing. A gated-out store supplies no entries and
   every mapping must be byte-identical to what it was before the layer
   existed -- that IS the hibernation, and it has no other mechanism.
2. Present, it lands between the bridge and the cell residual, and the
   resolver's three standing guarantees still hold.
3. Within one modality it cancels, because both legs carry it.
4. The gate refuses on the two things that produced confident wrong
   answers while this layer was being built: an FFT search bound that
   wraps, and an unverified sign convention.

Run: python tests/test_cell_bridge_layer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np                                            # noqa: E402

from codelab_pipeline.alignment.frames import FrameResolver   # noqa: E402
from codelab_pipeline.alignment import cell_bridge_gate as G  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + ((' -- %s' % detail) if detail and not cond else ''))


def shift(dy, dx):
    H = np.eye(3)
    H[0, 2], H[1, 2] = dy, dx
    return H


class Cell(object):
    def __init__(self, cid, matrices=None):
        self.id = cid
        self.matrices = matrices or {}


def base_kwargs(**extra):
    kw = dict(within={'DNA': {'H1': shift(2, 0)}, 'RNA': {'H1': shift(0, 3)}},
              shared='RNA',
              bridges={'DNA': (shift(10, 20), 5.0)})
    kw.update(extra)
    return kw


def main():
    print('\n-- absent, the layer changes nothing --')
    plain = FrameResolver(**base_kwargs())
    withed = FrameResolver(**base_kwargs(cell_bridges={}))
    c = Cell(7)
    for lab, cell in (('no cell', None), ('a cell', c)):
        a = plain.to_shared('H1', 'DNA', cell)
        b = withed.to_shared('H1', 'DNA', cell)
        check('to_shared unchanged with an empty layer (%s)' % lab,
              np.allclose(a, b), '%s vs %s' % (a[:2, 2], b[:2, 2]))
        check('z_to_shared unchanged with an empty layer (%s)' % lab,
              plain.z_to_shared('H1', 'DNA', cell)
              == withed.z_to_shared('H1', 'DNA', cell))
    # a gated-out store is SILENT, not 'missing': the layer is off by
    # design, and naming it on every cell would bury the real gaps
    miss = set()
    withed.to_shared('H1', 'DNA', c, miss)
    check('an off layer reports nothing missing',
          not any(m.startswith('cell-bridge') for m in miss), str(miss))

    print('\n-- present, it composes after the bridge --')
    cb = {'DNA': {7: (shift(1, 2), 0.5)}}
    r = FrameResolver(**base_kwargs(cell_bridges=cb))
    p = np.array([0.0, 0.0, 1.0])
    got = (r.to_shared('H1', 'DNA', c) @ p)[:2]
    want = (shift(1, 2) @ shift(10, 20) @ shift(2, 0) @ p)[:2]
    check('to_shared = cell_bridge @ bridge @ within', np.allclose(got, want),
          '%s vs %s' % (got, want))
    check('z picks up the per-cell term',
          r.z_to_shared('H1', 'DNA', c) == plain.z_to_shared('H1', 'DNA', c) + 0.5)

    print('\n-- it applies only to the cell it was fitted for --')
    other = Cell(8)
    check('another cell is untouched',
          np.allclose(r.to_shared('H1', 'DNA', other),
                      plain.to_shared('H1', 'DNA', other)))
    miss = set()
    r.to_shared('H1', 'DNA', other, miss)
    check('an ACTIVE layer reports the cell it has no entry for',
          'cell-bridge:DNA/8' in miss, str(miss))
    check('a string id still matches an int key',
          np.allclose(r.to_shared('H1', 'DNA', Cell('7')),
                      r.to_shared('H1', 'DNA', c)))

    print('\n-- the resolver guarantees still hold --')
    src, dst = ('H1', 'DNA'), ('H1', 'RNA')
    H, dz, _m = r.transform(src, dst, c)
    Hb, dzb, _m = r.transform(dst, src, c)
    check('src == dst is identity',
          np.allclose(r.transform(src, src, c)[0], np.eye(3))
          and r.transform(src, src, c)[1] == 0.0)
    check('transform is invertible', np.allclose(H @ Hb, np.eye(3))
          and abs(dz + dzb) < 1e-9)
    check('within one modality the layer cancels',
          np.allclose(r.transform(('H1', 'DNA'), ('H1', 'DNA'), c)[0],
                      np.eye(3)))

    print('\n-- the shared modality may not carry one --')
    try:
        FrameResolver(**base_kwargs(cell_bridges={'RNA': {7: (np.eye(3), 0.0)}}))
        check('shared modality is refused', False, 'it was accepted')
    except ValueError as exc:
        check('shared modality is refused', 'identity by definition' in str(exc))

    print('\n-- the gate refuses what fooled us --')
    check('a wrapping search bound is refused',
          not G.check_search_bound(28, 25).ok)
    check('a clear search bound passes', G.check_search_bound(12, 26).ok)
    rng = np.random.default_rng(0)
    z = np.arange(60)
    vol = rng.random((24, 24, 60)) * np.exp(-((z - 30) ** 2) / 32.0)[None, None, :]
    s, v = G.shift_sign(lambda a, b, p, q: (0, 0, -4.0, 0.9, False), vol, 4, 10, 5)
    check('a reversed sign convention is detected, not assumed',
          v.ok and s == -1, repr(v))
    s, v = G.shift_sign(lambda a, b, p, q: (0, 0, 0.1, 0.9, False), vol, 4, 10, 5)
    check('a registration that loses the shift is refused',
          s is None and not v.ok, repr(v))
    check('too few cells reads as "did not run", not "failed"',
          'did not run' in G.recovery([0.9] * 3, [0.0] * 3).reason)
    check('decide needs both tiers',
          not G.decide(G.admission(vol, 0.25, offset_um=3.0),
                       G.recovery([0.4] * 20, [0.0] * 20)).ok)
    check('decide passes when both do',
          G.decide(G.admission(vol, 0.25, offset_um=3.0),
                   G.recovery([0.9] * 20,
                              list(np.linspace(-0.08, 0.08, 20)))).ok)

    print('\n%d passed, %d failed' % (len(PASS), len(FAIL)))
    if FAIL:
        print('failed: ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
