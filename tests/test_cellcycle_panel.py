"""
The Cell Cycle panel's widget contract, headless: sources -> gene rows
with default roles, the config string round trip, training celltypes,
arcs and gates. Then the wiring module imports and a MainWindow-free
smoke of its pure helpers.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_cellcycle_panel.py
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtWidgets, QtCore                            # noqa: E402

from ui.cellcycle_panel import CellCyclePanelUI                # noqa: E402
from codelab_pipeline.analysis import cellcycle as CC          # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''), flush=True)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])   # noqa: F841
    w = QtWidgets.QWidget()
    p = CellCyclePanelUI()
    p.setupUi(w)
    records = {'RNA': [
        {'folder': 'Hyb_101', 'readout_name': 'MCM2_mRNA', 'channels': [635, 555], 'fiducial_channel': 555},
        {'folder': 'Hyb_102', 'readout_name': 'MCM2_nascent', 'channels': [635, 555], 'fiducial_channel': 555},
        {'folder': 'Hyb_103', 'readout_name': 'CCNB1_exon', 'channels': [635, 555], 'fiducial_channel': 555},
        {'folder': 'Hyb_104', 'readout_name': 'GAPDH_exon', 'channels': [635, 555], 'fiducial_channel': 555},
        {'folder': 'Rep_101', 'readout_name': 'Rep_MCM2_mRNA', 'channels': [635, 555], 'fiducial_channel': 555},
        {'folder': 'Hyb_500', 'readout_name': 'DAPI', 'channels': [405], 'fiducial_channel': None},
    ]}
    print('sources -> genes')
    p.populate_sources(records)
    rows = p.gene_rows()
    check('one row per (hybe, channel)', len(rows) == 11, str(len(rows)))
    got = {(r['source'], r['gene'], r['role'], r['use']) for r in rows}
    check('an mRNA round on the readout channel is a checked S gene',
          (('RNA', 'Hyb_101', 635), 'MCM2', 'S', True) in got)
    check('the same round on the fiducial channel is unchecked',
          (('RNA', 'Hyb_101', 555), 'MCM2', 'S', False) in got)
    check('exon rounds get their gene and role', (('RNA', 'Hyb_103', 635), 'CCNB1', 'G2/M', True) in got)
    check('GAPDH is housekeeping', (('RNA', 'Hyb_104', 635), 'GAPDH', 'housekeeping', True) in got)
    check('a nascent round keeps its name and starts unchecked',
          (('RNA', 'Hyb_102', 635), 'MCM2_nascent', '-', False) in got)
    check('a repeat round starts unchecked', (('RNA', 'Rep_101', 635), 'Rep_MCM2_mRNA', '-', False) in got)
    check('DAPI starts unchecked', any(s == ('RNA', 'Hyb_500', 405) and not u for s, _g, _r, u in got))
    cg = p.checked_genes()
    check('checked_genes lists the three in use', sorted(g for _s, g, _r in cg) == ['CCNB1', 'GAPDH', 'MCM2'])

    print('config string round trip')
    text = p.gene_config()
    p._set_all_use(False)
    check('unchecked everything', p.checked_genes() == [])
    p.set_gene_config(text)
    check('set_gene_config restores the checked rows, genes and roles',
          sorted(p.checked_genes()) == sorted(cg), f'{p.checked_genes()}')
    p.set_gene_config('RNA|Hyb_103|635|CyclinB1|G2/M,RNA|Hyb_999|635|GHOST|S')
    cg2 = p.checked_genes()
    check('a config renames a gene and skips sources the layout lacks',
          cg2 == [(('RNA', 'Hyb_103', 635), 'CyclinB1', 'G2/M')], str(cg2))
    p._check_mrna_rounds()
    check('Check mRNA/exon rounds re-checks the countable readout-channel rows only, keeping edited names',
          sorted(g for _s, g, _r in p.checked_genes()) == ['CyclinB1', 'GAPDH', 'MCM2'],
          str(sorted(g for _s, g, _r in p.checked_genes())))

    print('training celltypes, arcs, gates')
    p.set_celltype_names(['Unsynchronized', 'Hydroxyurea', 'Nocodazole'])
    check('no celltype checked by default', p.training_celltypes() == [])
    p.set_training_celltypes(['Unsynchronized'])
    check('training celltypes set', p.training_celltypes() == ['Unsynchronized'])
    p.set_celltype_names(['Unsynchronized', 'Hydroxyurea', 'Nocodazole', 'WT'])
    check('a refresh keeps what was checked', p.training_celltypes() == ['Unsynchronized'])
    p.set_arcs([{'name': 'S', 'start_deg': 330, 'end_deg': 60}, {'name': 'G2/M', 'start_deg': 90, 'end_deg': 210}])
    arcs = p.arcs()
    check('arcs round-trip through the table', arcs == [{'name': 'S', 'start_deg': 330.0, 'end_deg': 60.0},
                                                        {'name': 'G2/M', 'start_deg': 90.0, 'end_deg': 210.0}], str(arcs))
    p._add_arc_row('', 0, 0)
    check('a row without a name is skipped', len(p.arcs()) == 2)
    g = p.gates()
    check('default gates: only min log BF_ring on, at 0', g['min_bf_ring'] == 0.0 and g['min_R'] is None and g['min_fit_z'] is None)
    p.set_gates({'min_R': 0.6, 'min_bf_ring': None, 'max_radius': 1.5})
    g = p.gates()
    check('gates round-trip, None = off', g['min_R'] == 0.6 and g['min_bf_ring'] is None and g['max_radius'] == 1.5
          and g['min_fit_z'] is None and g['min_total'] is None, str(g))
    cat = CC.assign(__import__('pandas').DataFrame({'theta_deg': [10.0, 100.0, 250.0], 'R': [0.9, 0.9, 0.9],
                                                     'radius': [1.0, 1.0, 1.0]}), p.arcs(), g)
    check('the panel\'s arcs and gates drive cellcycle.assign', list(cat) == ['S', 'G2/M', ''], str(list(cat)))
    check('proxy metric', p.proxy_metric() == 'n' and (p.ProxyComboBox.setCurrentIndex(1) or p.proxy_metric() == 'soft'))

    print('the wiring module')
    from windows import cellcycle_wiring as W
    check('imports without a MainWindow', hasattr(W, 'CellCycleWiring') and W.TAB_TITLE == 'Cell Cycle')
    import json
    import tempfile
    import numpy as np
    d = tempfile.mkdtemp(prefix='ccbridge_')
    path = os.path.join(d, 'b.json')
    json.dump({'name': 'B', 'genes': ['a', 'b', 'c'], 'X': np.random.default_rng(0).integers(1, 20, (60, 3)).tolist(),
               'groups': ['u'] * 30 + ['h'] * 30, 'alpha': 100, 'train_celltypes': ['u']}, open(path, 'w'))
    ds, raw = W.CellCycleWiring.load_bridge(path)
    check('an exported counts file loads as a Dataset of its training cells', ds.name == 'B' and len(ds.X) == 30 and ds.genes == ['a', 'b', 'c'])
    print('the main window shell')
    from ui.main_window_ui import MainWindowUI
    mw = QtWidgets.QMainWindow()
    ui = MainWindowUI()
    ui.setupUi(mw)
    titles = [ui.tabWidget.tabText(i) for i in range(ui.tabWidget.count())]
    check('the Cell Cycle tab sits between Celltype Determination and Chromatin Tracing',
          titles.index('Cell Cycle') == titles.index('Celltype Determination') + 1
          and titles.index('Chromatin Tracing') == titles.index('Cell Cycle') + 1, str(titles))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('ALL GOOD')


if __name__ == '__main__':
    main()
