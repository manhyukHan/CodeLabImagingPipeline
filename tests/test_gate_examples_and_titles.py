"""
The p-gate's example pictures, and p_exist on the 3D grid's titles.

WHY THIS FILE EXISTS. The gate dialog opened with every example box
reading 'no pixels' and a note saying it had been given no way to read
them. It was right: MainWindow has no `storage_path` attribute at all --
each modality carries its own -- so `self.storage_path` answered None,
`_spot_crop_for_gate` returned None, and the dialog said so honestly.
Nothing raised, nothing logged; the feature was simply absent. The same
missing attribute made 'Make new model...' always refuse.

These are GUI methods, so they are exercised the way the rest of this
suite exercises GUI logic: unbound, against a stand-in `self` that
provides exactly the collaborators the method reads. That tests the
method's own reasoning, which is where the bug was, and it runs with no
display and no store.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_gate_examples_and_titles.py
"""
import os
import sys

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = [0, 0]


def check(name, ok):
    CHECKS[0] += 1
    if ok:
        CHECKS[1] += 1
        print('  ok   ' + name)
    else:
        print('  FAIL ' + name)


class Spot(object):
    def __init__(self, raw=(100.0, 200.0, 5.0), p_exist=float('nan')):
        self.raw_coordinate = raw
        self.p_exist = p_exist


class Cell(object):
    def __init__(self, cid):
        self.id = cid


# --------------------------------------------------------------------
# _spot_crop_for_gate
# --------------------------------------------------------------------

class FakeWindow(object):
    """Only what _spot_crop_for_gate reads."""

    def __init__(self, path):
        self._path = path
        self.reads = []

    def _storage_path_for_modality(self, modality):
        return self._path if modality == 'DNA' else ''


def crop_maker():
    from windows.main_window import MainWindow
    return MainWindow._spot_crop_for_gate


def test_crop_for_gate():
    print('_spot_crop_for_gate')
    from codelab_pipeline.io import analysis_store as AS
    make = crop_maker()

    mip = np.zeros((1024, 1024), float)
    mip[100, 200] = 7.0
    calls = []

    def fake_read(storage_path, fov, hybe, channel, window=None):
        calls.append((storage_path, fov, hybe, channel))
        return mip

    real = AS.read_hybe_mip
    AS.read_hybe_mip = fake_read
    try:
        w = FakeWindow('E:/fake/DNA')

        # THE BUG: a window-wide storage_path does not exist. The crop
        # maker must ask the modality, and must still answer None when
        # that modality has no store.
        check('no store for the modality -> None',
              make(w, 3, 'H1', 'RNA', 1) is None)

        crop_of = make(w, 3, 'H1', 'DNA', 1)
        check('a store for the modality -> a callable',
              callable(crop_of))

        img = crop_of(Spot(raw=(100.0, 200.0, 5.0)))
        check('crop is 2*pad+1 square', img is not None
              and img.shape == (15, 15))
        check('crop is centred on the spot', img is not None
              and img[7, 7] == 7.0)

        # ONE READ FOR EVERY EXAMPLE AND EVERY REFRESH. Eight thumbnails
        # that each re-read the frame is eight NAS round-trips for one
        # picture the store already holds.
        for _ in range(8):
            crop_of(Spot(raw=(100.0, 200.0, 5.0)))
        check('the frame is read exactly once', len(calls) == 1)

        # A spot at the edge yields a smaller, still-valid crop.
        edge = crop_of(Spot(raw=(2.0, 3.0, 0.0)))
        check('edge crop is clipped, not empty',
              edge is not None and edge.shape == (10, 11))

        # NO COORDINATE IS NOT THE ORIGIN.
        check('no raw_coordinate -> None (not the image corner)',
              crop_of(Spot(raw=None)) is None)

        class Bare(object):
            pass
        check('a spot with no attribute at all -> None',
              crop_of(Bare()) is None)

        # A store-shaped dict, as read back from a capsule.
        check('dict spot works too',
              crop_of({'raw_coordinate': (100.0, 200.0, 5.0)}) is not None)

        # An unreadable frame is missing pictures, never a failed dialog.
        AS.read_hybe_mip = lambda *a, **k: (_ for _ in ()).throw(
            OSError('no such file'))
        w2 = FakeWindow('E:/fake/DNA')
        c2 = make(w2, 3, 'H1', 'DNA', 1)
        check('unreadable frame -> callable that answers None',
              callable(c2) and c2(Spot()) is None)

        # And a frame that is simply absent.
        AS.read_hybe_mip = lambda *a, **k: None
        c3 = make(FakeWindow('E:/fake/DNA'), 3, 'H1', 'DNA', 1)
        check('absent frame -> None per spot', c3(Spot()) is None)
    finally:
        AS.read_hybe_mip = real


# --------------------------------------------------------------------
# grid titles
# --------------------------------------------------------------------

class FakeTitleWindow(object):
    def __init__(self, index):
        self._index = index

    def _global_spot_index_map(self, storage_path, fov, hybe, channel):
        return self._index

    # the real methods, unbound, so the batch rule is the real one
    from windows.main_window import MainWindow as _MW
    _spot_grid_title = _MW._spot_grid_title
    _spot_grid_titles = _MW._spot_grid_titles
    del _MW


def test_titles():
    print('_spot_grid_title / _spot_grid_titles')
    learned = Spot(p_exist=0.8412)
    gauss = Spot(p_exist=float('nan'))
    cell = Cell(36)
    w = FakeTitleWindow({id(learned): 20, id(gauss): 21})

    one = w._spot_grid_title('sp', 3, 'H1', 1, learned, cell)
    check('one line by default', one == 'Spot 20 | Cell 36')

    two = w._spot_grid_title('sp', 3, 'H1', 1, learned, cell, show_p=True)
    check('p_exist on a second line',
          two == 'Spot 20 | Cell 36' + chr(10) + 'p_exist: 0.84')

    nan = w._spot_grid_title('sp', 3, 'H1', 1, gauss, cell, show_p=True)
    check("no p_exist reads '--', never 'nan'",
          nan.endswith('p_exist: --'))

    unassigned = w._spot_grid_title('sp', 3, 'H1', 1, learned, None,
                                    show_p=True)
    check('unassigned cell still labelled',
          unassigned.startswith('Spot 20 | Cell unassigned'))

    # ALL PANELS OR NONE, per batch.
    ts = w._spot_grid_titles('sp', 3, 'H1', 1, [(learned, cell),
                                                (gauss, cell)])
    check('one learned spot gives every panel a second line',
          all(chr(10) in t for t in ts))
    check("the gaussian one still says '--'", ts[1].endswith('p_exist: --'))

    ts2 = w._spot_grid_titles('sp', 3, 'H1', 1, [(gauss, cell)])
    check('a v1/v2-only batch stays one line',
          all(chr(10) not in t for t in ts2))

    # A p_exist that is not a number must not crash a whole grid.
    junk = Spot(p_exist='nope')
    w2 = FakeTitleWindow({id(junk): 5})
    check('junk p_exist is treated as absent',
          all(chr(10) not in t
              for t in w2._spot_grid_titles('sp', 3, 'H1', 1,
                                            [(junk, cell)])))


# --------------------------------------------------------------------
# the grid makes room for the second line
# --------------------------------------------------------------------

def test_grid_makes_room():
    print('grid layout')
    import inspect
    from canvas import localize_3d_displayer as D
    src = inspect.getsource(D.Localize3DGridDisplayer.show_fit_status_grid)
    check('a two-line title is detected', 'tall =' in src)
    check('the top margin moves for it', "top=0.87 if tall else 0.90" in src)
    check('the between-pair gap moves too',
          "hspace=0.72 if tall else 0.55" in src)


# --------------------------------------------------------------------
# the dialog no longer mislabels an unreadable store
# --------------------------------------------------------------------

def test_note_distinguishes():
    print('p-gate note')
    import inspect
    from ui import p_gate_dialog as P
    src = inspect.getsource(P.PGateDialog._draw_examples)
    check('there is a branch for "read and failed"',
          'NONE of' in src and 'could be read' in src)
    check('it is separate from "no way to read"',
          src.index('no way to read') < src.index('NONE of'))


def main():
    test_crop_for_gate()
    test_titles()
    test_grid_makes_room()
    test_note_distinguishes()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
