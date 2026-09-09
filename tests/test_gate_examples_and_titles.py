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


def check(name, ok, note=''):
    CHECKS[0] += 1
    if ok:
        CHECKS[1] += 1
        print('  ok   ' + name + (('  -- ' + note) if note else ''))
    else:
        print('  FAIL ' + name + (('  -- ' + note) if note else ''))


class Spot(object):
    def __init__(self, raw=(100.0, 200.0, 5.0), p_exist=float('nan'),
                 z_status='accepted'):
        self.raw_coordinate = raw
        self.p_exist = p_exist
        self.z_status = z_status


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

    def _storage_path_for_modality(self, modality):
        return self._path if modality == 'DNA' else ''


def a_store(root, depth=41):
    """A store-shaped tree with one real stack file in it.

    A REAL HDF5 FILE, not a stub: the provider opens the store itself
    now (it has to -- a ZX panel needs the z axis, and the stored MIP
    has none), so a fake that patches a reader would test nothing about
    the path, the dataset name or the windowing that actually run.
    """
    import h5py
    import numpy as _np
    from codelab_pipeline.io import paths
    path = paths.stack_path(root, 3, 'H1')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cube = _np.zeros((64, 64, depth), _np.uint16)
    cube[40, 50, 7] = 900          # the spot, at (y=40, x=50, z=7)
    cube[2, 3, 9] = 700            # a second one, near the corner
    with h5py.File(path, 'w') as f:
        f.create_dataset('/stack/ch1', data=cube)
    return path


def test_crop_for_gate():
    print('_spot_crop_for_gate')
    import shutil
    import tempfile
    from windows.main_window import MainWindow
    from ui.p_gate_dialog import GateCrop
    make = MainWindow._spot_crop_for_gate

    root = tempfile.mkdtemp(prefix='gatecrop_')
    try:
        a_store(root)
        w = FakeWindow(root)

        # THE BUG THIS FILE OPENED WITH: a window-wide storage_path does
        # not exist. The provider must ask the modality, and must still
        # answer None when that modality has no store.
        check('no store for the modality -> None',
              make(w, 3, 'H1', 'RNA', 1) is None)

        crop_of = make(w, 3, 'H1', 'DNA', 1)
        check('a store for the modality -> a callable', callable(crop_of))

        got = crop_of(Spot(raw=(40.0, 50.0, 7.0)))
        check('it hands back a GateCrop, not bare pixels',
              isinstance(got, GateCrop))
        check('the cube carries the FULL z depth -- a ZX panel needs it',
              got is not None and got.cube.shape == (15, 15, 41),
              str(None if got is None else got.cube.shape))
        check('the spot is where the crop says it is',
              got is not None and (got.x, got.y) == (7.0, 7.0)
              and got.cube[7, 7, 7] == 900)

        # THE EDGE CASE THE OLD CONTRACT COULD NOT EXPRESS. A window
        # clipped at the frame edge puts the spot OFF centre, so a ring
        # drawn at (pad, pad) would sit on the wrong pixel -- which is
        # exactly the picture a reviewer would then judge.
        edge = crop_of(Spot(raw=(2.0, 3.0, 9.0)))
        check('an edge crop is clipped, not empty',
              edge is not None and edge.cube.shape == (10, 11, 41),
              str(None if edge is None else edge.cube.shape))
        check('and the spot is off-centre in it, correctly',
              edge is not None and (edge.x, edge.y) == (3.0, 2.0)
              and edge.cube[2, 3, 9] == 700)

        # A FITTED Z OR NONE, never a placeholder. spot.py is explicit
        # that z == 0.0 cannot be read as 'unfitted'.
        acc = crop_of(Spot(raw=(40.0, 50.0, 7.0), z_status='accepted'))
        rej = crop_of(Spot(raw=(40.0, 50.0, 7.0), z_status='rejected'))
        nof = crop_of(Spot(raw=(40.0, 50.0, 0.0), z_status='not_fit'))
        check('an accepted fit reports its z', acc.z == 7.0
              and acc.rejected is False)
        check('a rejected fit reports its z AND that it was rejected',
              rej.z == 7.0 and rej.rejected is True)
        check('an unfitted spot reports NO z, not 0.0',
              nof.z is None and nof.rejected is False)

        # NO COORDINATE IS NOT THE ORIGIN.
        check('no raw_coordinate -> None (not the frame corner)',
              crop_of(Spot(raw=None)) is None)

        class Bare(object):
            pass
        check('a spot with no attribute at all -> None',
              crop_of(Bare()) is None)

        # A store-shaped dict, as read back from a capsule.
        d = crop_of({'raw_coordinate': (40.0, 50.0, 7.0),
                     'z_status': 'accepted'})
        check('a dict spot works, z and all',
              d is not None and d.z == 7.0 and d.cube[7, 7, 7] == 900)

        # ONE OPEN PER BATCH, and a handle that is actually released.
        check('the file handle is kept across the batch',
              crop_of.__closure__ is not None and callable(crop_of.close))
        crop_of.close()
        check('close() releases it and the next call reopens',
              crop_of(Spot(raw=(40.0, 50.0, 7.0))) is not None)
        crop_of.close()

        # An unreadable store is missing pictures, never a failed dialog.
        c2 = make(FakeWindow(os.path.join(root, 'nope')), 3, 'H1', 'DNA', 1)
        check('a store with no such file -> callable that answers None',
              callable(c2) and c2(Spot()) is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------
# the dialog draws the pair and rings the spot
# --------------------------------------------------------------------

def test_the_dialog_draws_a_pair_and_rings_it():
    print('the example tile is a YX/ZX pair with the spot ringed')
    import numpy as _np
    from PyQt5 import QtWidgets
    from ui.p_gate_dialog import PGateDialog, GateCrop, N_EXAMPLES

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rng = _np.random.default_rng(0)
    spots = []
    for _ in range(40):
        sp = Spot(raw=(100.0, 200.0, 20.0),
                  p_exist=float(rng.uniform(0.36, 0.64)))
        spots.append(sp)
    cube = rng.normal(300.0, 6.0, (15, 15, 41))

    def crop_of(spot):
        return GateCrop(cube, 7.0, 7.0, 20.0, False)

    d = PGateDialog(spots, crop_of=crop_of, threshold=0.5, seed=1)
    axes = d.efig.axes
    check('two panels per example, both rows',
          len(axes) == 2 * 2 * N_EXAMPLES, str(len(axes)))
    ringed = [ax for ax in axes if ax.collections]
    check('every drawn panel is ringed -- YX and ZX alike',
          len(ringed) == len(axes), f'{len(ringed)} of {len(axes)}')
    ec = ringed[0].collections[0].get_edgecolor()[0]
    check('an accepted spot is ringed yellow',
          tuple(_np.round(ec, 2)) == (1.0, 1.0, 0.0, 1.0))

    # A BARE ARRAY IS STILL A LEGAL ANSWER -- a caller with pixels and no
    # spot position gets a drawn tile and NO ring, rather than a ring at
    # a guessed centre.
    d2 = PGateDialog(spots, crop_of=lambda s: cube, threshold=0.5, seed=1)
    drawn = [ax for ax in d2.efig.axes if ax.images]
    check('a bare cube still draws both panels', len(drawn) == len(d2.efig.axes),
          f'{len(drawn)} of {len(d2.efig.axes)}')
    check('and rings nothing, because nothing said where the spot is',
          not any(ax.collections for ax in d2.efig.axes))

    # The label names the quantity being gated, not a hard-coded 'p'.
    titles = [ax.get_title() for ax in d.efig.axes if ax.get_title()]
    check('the tile label names the gated quantity',
          titles and all(t.startswith('p_exist=') for t in titles),
          titles[0] if titles else '(none)')

    # ONE OPEN PER REFRESH: the dialog closes the provider's handle at
    # the end of a batch rather than leaving it to garbage collection.
    closed = {'n': 0}

    def closing_crop(spot):
        return GateCrop(cube, 7.0, 7.0, 20.0, False)
    closing_crop.close = lambda: closed.__setitem__('n', closed['n'] + 1)
    d3 = PGateDialog(spots, crop_of=closing_crop, threshold=0.5, seed=1)
    n_after_open = closed['n']
    d3._draw_examples()
    check('the batch closes the store handle', n_after_open >= 1,
          f'{n_after_open} close(s) on open')
    check('and again on every Refresh', closed['n'] == n_after_open + 1,
          str(closed['n']))


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


# --------------------------------------------------------------------
# what an adversarial review found
# --------------------------------------------------------------------

def test_a_store_it_cannot_address_degrades():
    """A legacy fov## store must lose its PICTURES, not the whole gate.

    paths.stack_path runs _require_fov3, which RAISES on a store still
    holding 'fov07'-style directories. Called eagerly the raise escaped
    both callers and replaced the gate with an 'Unexpected error' box --
    while the MIP reader it replaced hit the same check inside its own
    try and simply drew the histogram with no pictures.
    """
    import shutil
    import tempfile
    from windows.main_window import MainWindow
    root = tempfile.mkdtemp(prefix='gatelegacy_')
    try:
        os.makedirs(os.path.join(root, 'stacks', 'fov07'))
        os.makedirs(os.path.join(root, 'mips', 'fov07'))
        from codelab_pipeline.io import paths
        raised = False
        try:
            paths.stack_path(root, 7, 'H1')
        except Exception:                                   # noqa: BLE001
            raised = True
        check('the store really is one stack_path refuses', raised)
        got = MainWindow._spot_crop_for_gate(FakeWindow(root), 7, 'H1',
                                             'DNA', 1)
        check('and the provider answers None instead of raising',
              got is None)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_z_outside_the_crop_still_draws():
    """A stored z outlives a re-ingestion with fewer planes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as _np
    from canvas import spot_fit_status as F

    cube = _np.random.RandomState(0).normal(300.0, 5.0, (15, 15, 20))
    for name, z in (('past the end', 500.0), ('negative', -40.0),
                    ('infinite', float('inf')), ('NaN', float('nan'))):
        fig = plt.figure()
        a1 = fig.add_subplot(211)
        a2 = fig.add_subplot(212, sharex=a1)
        try:
            F.draw_spot_fit_status(a1, a2, cube, centroid=(7.0, 7.0, z))
            drew = a2.images and _np.asarray(
                a2.images[-1].get_array()).size > 0
        except Exception as exc:                            # noqa: BLE001
            drew = f'{type(exc).__name__}: {exc}'
        check(f'a z {name} still draws a ZX panel', drew is True, str(drew))
        plt.close(fig)

    # A ONE-PLANE CUBE is a legal crop: the ZX panel is one row.
    fig = plt.figure()
    a1 = fig.add_subplot(211)
    a2 = fig.add_subplot(212, sharex=a1)
    F.draw_spot_fit_status(a1, a2, cube[:, :, :1], centroid=(7.0, 7.0, 0.0))
    check('a one-plane cube draws a one-row ZX panel',
          _np.asarray(a2.images[-1].get_array()).shape[0] == 1)
    plt.close(fig)


def test_infinity_is_not_a_fitted_z():
    """`zv == zv` lets inf through; int(round(inf)) then raises."""
    import shutil
    import tempfile
    from windows.main_window import MainWindow
    root = tempfile.mkdtemp(prefix='gateinf_')
    try:
        a_store(root)
        crop_of = MainWindow._spot_crop_for_gate(FakeWindow(root), 3, 'H1',
                                                 'DNA', 1)
        for name, z in (('inf', float('inf')), ('-inf', float('-inf')),
                        ('nan', float('nan'))):
            got = crop_of(Spot(raw=(40.0, 50.0, z), z_status='accepted'))
            check(f'a {name} z is reported as NO z, not as a fit',
                  got is not None and got.z is None)
        crop_of.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_handle_is_released_even_when_a_tile_raises():
    """Outside a finally this leaked an HDF5 handle per Refresh."""
    import numpy as _np
    from PyQt5 import QtWidgets
    from ui.p_gate_dialog import PGateDialog, GateCrop

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rng = _np.random.default_rng(0)
    spots = [Spot(raw=(100.0, 200.0, 20.0),
                  p_exist=float(rng.uniform(0.36, 0.64))) for _ in range(40)]
    closed = {'n': 0}

    def exploding(spot):
        raise RuntimeError('the store went away mid-draw')
    exploding.close = lambda: closed.__setitem__('n', closed['n'] + 1)

    d = PGateDialog(spots, crop_of=exploding, threshold=0.5, seed=1)
    check('a provider that raises on every spot still closes its handle',
          closed['n'] >= 1, f"{closed['n']} close(s)")
    check('and the dialog still exists, with the distribution drawn',
          len(d.fig.axes) >= 1)

    # A close() that itself fails must not take down a finished dialog.
    def ok_crop(spot):
        return GateCrop(rng.normal(300.0, 5.0, (15, 15, 41)),
                        7.0, 7.0, 20.0, False)
    ok_crop.close = lambda: (_ for _ in ()).throw(OSError('handle gone'))
    raised = False
    try:
        PGateDialog(spots, crop_of=ok_crop, threshold=0.5, seed=1)
    except Exception:                                       # noqa: BLE001
        raised = True
    check('a close() that fails does not take the dialog with it',
          not raised)


def test_the_words_match_what_the_code_reads():
    """Notes and docstrings that name the wrong file mislead a reader."""
    import inspect
    from ui import p_gate_dialog as P
    from canvas import spot_fit_status as F

    src = inspect.getsource(P.PGateDialog._draw_examples)
    check('the unreadable-pixels note names the Z-STACK, not the MIP',
          'Z-STACK' in src and 'the MIP for this hybe' not in src)
    check('and it says why an intact MIP does not save it',
          'a ZX panel needs the z axis' in src)

    doc = F.draw_spot_fit_status.__doc__ or ''
    check("the 'no circle = no fit' rule is corrected, not left standing",
          'no circle = no fit at all' not in
          inspect.getsource(F.draw_spot_fit_status))
    check('and the four states are named', 'FOUR STATES NOW' in doc)

    mod = inspect.getsource(P)
    check('the module docstring no longer promises four examples',
          'one fixed sample of four' not in mod)

    # THUMB_R was dead: two numbers that had to agree, with nothing
    # making them.
    from windows.main_window import MainWindow
    sig = inspect.signature(MainWindow._spot_crop_for_gate)
    check('THUMB_R is what actually cuts the crop',
          sig.parameters['pad'].default is None
          and 'THUMB_R' in inspect.getsource(MainWindow._spot_crop_for_gate))


def test_the_dialog_is_released():
    """A parented dialog outlives exec_() unless someone says otherwise."""
    import inspect
    from windows.main_window import MainWindow
    from ui import p_gate_dialog as P
    for name, src in (('preview',
                       inspect.getsource(MainWindow._preview_p_gate)),
                      ('gate',
                       inspect.getsource(MainWindow._remove_z_rejected_spots)),
                      ('choose_threshold',
                       inspect.getsource(P.choose_threshold))):
        check(f'{name} releases its dialog', 'deleteLater()' in src)
    src = inspect.getsource(MainWindow._remove_z_rejected_spots)
    check('the gate reads its threshold BEFORE releasing the dialog',
          src.index('dlg.threshold()') < src.index('dlg.deleteLater()'))



def main():
    test_crop_for_gate()
    test_the_dialog_draws_a_pair_and_rings_it()
    test_a_store_it_cannot_address_degrades()
    test_a_z_outside_the_crop_still_draws()
    test_infinity_is_not_a_fitted_z()
    test_the_handle_is_released_even_when_a_tile_raises()
    test_the_words_match_what_the_code_reads()
    test_the_dialog_is_released()
    test_titles()
    test_grid_makes_room()
    test_note_distinguishes()
    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
