"""
The View list counts THIS hybe and channel, not the cell's lifetime.

WHY. The middle list read 'Cell 25: 6 spot(s)' where 6 was every spot
that cell holds across every hybe. This panel localizes, saves, gates
and clears ONE (hybe, channel, modality) at a time, so that number could
not be acted on: it never said whether the hybe on screen had been
localized in that cell, which is the question the list is read to
answer.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_cell_list_scope.py
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = [0, 0]


def check(name, ok, note=''):
    CHECKS[0] += 1
    CHECKS[1] += 1 if ok else 0
    print(('  ok   ' if ok else '  FAIL ') + name
          + (('  -- ' + note) if note else ''))


class Spot(object):
    def __init__(self, cell, hybe, channel, modality='DNA'):
        self.cell = cell
        self.hybe = hybe
        self.channel = channel
        self.modality = modality


class Cell(object):
    def __init__(self, cid):
        self.id = cid


class Panel(object):
    def __init__(self, hybe, channel, modality='DNA'):
        self._hybe, self._ch, self._mod = hybe, channel, modality
        self.got = None

        class Combo(object):
            def __init__(self, text):
                self._t = text

            def currentText(self):
                return self._t
        self.ChannelComboBox = Combo(channel)

    def current_hybe_folder(self):
        return self._hybe

    def current_hybe_modality(self):
        return self._mod

    def populate_cell_choices(self, cells, n_spots_by_cell=None):
        self.got = dict(n_spots_by_cell or {})


class Container(object):
    def __init__(self, items):
        self._items = items

    def all(self, fov):
        return self._items

    def get_cells(self, fov):
        return self._items


class Win(object):
    from windows.main_window import MainWindow as _MW
    _refresh_spot_cell_list = _MW._refresh_spot_cell_list
    del _MW

    def __init__(self, panel, cells, spots):
        self.cell_container = Container(cells)
        self.spot_container = Container(spots)
        self._panel = panel
        self.ui = type('UI', (), {'SpotLocalizationPanel': panel})()

    def _current_spot_fov(self):
        return 3

    def _activate_fov(self, fov):
        pass

    def log(self, _msg):
        pass

    def __getattr__(self, name):
        # Everything else _refresh_spot_cell_list touches is a redraw of
        # some other widget; this test is about the counting only.
        if name.startswith('_refresh') or name.startswith('_load'):
            return lambda *a, **k: None
        raise AttributeError(name)


SPOTS = [
    Spot(25, 'Hyb_105', '635'), Spot(25, 'Hyb_105', '635'),
    Spot(25, 'Hyb_106', '635'), Spot(25, 'Hyb_105', '561'),
    Spot(25, 'Hyb_105', '635', modality='RNA'),
    Spot(26, 'Hyb_106', '635'),
    Spot(-1, 'Hyb_105', '635'),          # unassigned: never counted here
]
CELLS = [Cell(25), Cell(26)]


def counts(hybe, channel, modality='DNA'):
    p = Panel(hybe, channel, modality)
    Win(p, CELLS, SPOTS)._refresh_spot_cell_list()
    return p.got


def main():
    print('the count is the slice on screen')
    got = counts('Hyb_105', '635')
    check('cell 25 counts only this hybe+channel+modality',
          got.get(25) == 2, 'got %r (5 spots in that cell overall)'
          % got.get(25))
    check('a cell with nothing in this slice reads 0, not absent',
          got.get(26, 0) == 0)
    check('unassigned spots are still excluded', -1 not in got)

    other = counts('Hyb_106', '635')
    check('switching hybe changes the numbers',
          other.get(25) == 1 and other.get(26) == 1,
          'Hyb_106: %r' % other)

    ch = counts('Hyb_105', '561')
    check('switching channel changes them too', ch.get(25) == 1)

    rna = counts('Hyb_105', '635', modality='RNA')
    check('modality separates two hybes of the same name',
          rna.get(25) == 1)

    print('degenerate states')
    p = Panel('', '')
    Win(p, CELLS, SPOTS)._refresh_spot_cell_list()
    check('no hybe/channel yet -> the old unscoped total, not zero',
          p.got.get(25) == 5 and p.got.get(26) == 1, '%r' % p.got)

    print('the slice change is wired to the recount')
    import inspect
    from windows.main_window import MainWindow as MW
    src = inspect.getsource(MW._on_spot_slice_changed)
    check('a hybe/channel change recounts before redrawing',
          '_refresh_spot_cell_list' in src and '_show_spot_displayer' in src
          and src.index('_refresh_spot_cell_list')
          < src.index('_show_spot_displayer'))

    from ui import spot_localization_panel as P
    header = inspect.getsource(P.SpotLocalizationPanelUI.setupUi)
    check('the header names the scope it counts',
          'this hybe+channel' in header)

    print()
    print('%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    return 0 if CHECKS[1] == CHECKS[0] else 1


if __name__ == '__main__':
    sys.exit(main())
