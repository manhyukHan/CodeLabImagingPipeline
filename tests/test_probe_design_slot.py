"""A store can say which probe design it was imaged with.

WHY THERE IS A SLOT AT ALL. The experiment layout says which readout was
imaged in which hybe; it never says where on the genome that readout
sits. `genomic_resolution_kb` is one number per experiment, often blank,
and multiplying it by a readout index is wrong on any panel that covers
two regions or skips a readout. So the genomic separation the physics
term needs (codelab_pipeline/physics/MEASUREMENT.md) is unavailable from
a store unless the store records its design file -- which is what this
pins, end to end: the Ingestion panel row, the config attribute, the
manifest field, and the params fact a fresh session reads back.

THE ONE ORDERING THAT MATTERS: params.json wins over the manifest. The
manifest is rewritten at every startup from whatever config is loaded,
while params.json is written by the parse that actually established this
modality's inputs.

Run: QT_QPA_PLATFORM=offscreen python tests/test_probe_design_slot.py
"""
import os
import shutil
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.io import analysis_store, paths, preprocess
from codelab_pipeline.physics import panel as P

_n = {'pass': 0, 'fail': 0}

#: a two-region design with a jump, so a separation read from the file
#: cannot be confused with index * resolution
BED = ('track name="toy"\n'
       'chr20\t100000\t130000\tTOY_001\t1\t+\n'
       'chr20\t130001\t160001\tTOY_002\t1\t+\n'
       'chr20\t160002\t190002\tTOY_003\t1\t+\n'
       'chr20\t690003\t720003\tTOY_004\t1\t+\n')


def check(name, ok, extra=''):
    _n['pass' if ok else 'fail'] += 1
    print('  %-4s %s%s' % ('ok' if ok else 'FAIL', name,
                           ('   [%s]' % extra) if extra else ''))


def test_the_manifest_carries_it():
    print('\nthe manifest records it per modality')
    root = tempfile.mkdtemp()
    try:
        dp = os.path.join(root, 'proj')
        design = os.path.join(root, 'toy_bed.txt')
        with open(design, 'w') as f:
            f.write(BED)
        paths.write_manifest(dp, ['DNA', 'RNA'],
                             layout_paths={'DNA': 'L_DNA'},
                             probe_design_paths={'DNA': design})
        m = paths.read_manifest(dp)
        check('DNA carries the design',
              m['modalities']['DNA']['probe_design_path'] == design)
        check('RNA carries an empty one, not a missing key',
              m['modalities']['RNA']['probe_design_path'] == '')
        check('the layout slot is untouched',
              m['modalities']['DNA']['layout_path'] == 'L_DNA')
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_store_answers_and_params_win():
    print("\nthe store answers, and its own params win over the manifest")
    root = tempfile.mkdtemp()
    try:
        dp = os.path.join(root, 'proj')
        sp_dna, sp_rna = os.path.join(dp, 'DNA'), os.path.join(dp, 'RNA')
        from_manifest = os.path.join(root, 'from_manifest_bed.txt')
        from_params = os.path.join(root, 'from_params_bed.txt')
        for p in (from_manifest, from_params):
            with open(p, 'w') as f:
                f.write(BED)
        os.makedirs(sp_dna)
        os.makedirs(sp_rna)
        paths.write_manifest(dp, ['DNA', 'RNA'],
                             probe_design_paths={'DNA': from_manifest})
        check('with nothing in params, the manifest answers',
              analysis_store.probe_design_path(sp_dna) == from_manifest)
        check('a modality with no design answers empty',
              analysis_store.probe_design_path(sp_rna) == '')

        analysis_store.write_global_params(sp_dna, probe_design_path=from_params)
        check('the parse-written fact wins',
              analysis_store.probe_design_path(sp_dna) == from_params,
              analysis_store.probe_design_path(sp_dna))
        # modality-scoped, like layout_path: one modality's design must
        # not become the other's
        check("and does not leak to the other modality",
              analysis_store.probe_design_path(sp_rna) == '',
              analysis_store.probe_design_path(sp_rna))
        check('probe_design_path is modality-scoped',
              'probe_design_path' in analysis_store.MODALITY_SCOPED_PARAMS)

        design = P.read_probe_design(analysis_store.probe_design_path(sp_dna))
        check('and what comes back reads as a design',
              len(design) == 4 and abs(design.separation_kb(3, 4) - 530.0) < 0.1,
              design.summary())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_config_carries_it():
    print('\nthe config attribute survives a save/load round trip')
    root = tempfile.mkdtemp()
    try:
        out = os.path.join(root, 'cfg.xml')
        cfg = {'global': {'project_root': root},
               'modalities': {'DNA': {'probe_design_path': 'E:/Oligos/x_bed.txt'},
                              'RNA': {}},
               'params': {}}
        preprocess.make_xml_file(cfg, out)
        back = preprocess.load_xml_file(out)
        check('DNA keeps its design',
              back['modalities']['DNA']['probe_design_path'] == 'E:/Oligos/x_bed.txt')
        check('RNA stays a bare modality element',
              back['modalities']['RNA'] == {})
        with open(out) as f:
            text = f.read()
        check('and it is written as a <modality> attribute',
              'probe_design_path' in text.split('<acquisition')[0])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_panel_has_a_row():
    print('\nthe Ingestion panel has one row per modality, and it round-trips')
    from PyQt5 import QtWidgets
    from ui.ingestion_panel import IngestionPanelUI
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ui = IngestionPanelUI()
    # the widget must outlive setupUi -- a temporary one is collected and
    # takes the panel's layouts with it
    host = QtWidgets.QWidget()
    ui.setupUi(host)
    ui.build_modality_input_rows(['DNA', 'RNA'])
    check('every modality has a design edit',
          all('design' in r for r in ui.modality_input_rows.values()))
    # the three must be DISTINCT widgets: a shared one would silently
    # copy the layout path into the design slot
    ids = {id(w) for r in ui.modality_input_rows.values() for w in r.values()}
    check('layout / DAX / design are six distinct widgets', len(ids) == 6,
          str(len(ids)))
    ui.set_modality_input_values({'DNA': {'layout_path': 'L', 'dax_directory': 'D',
                                          'probe_design_path': 'E:/Oligos/x_bed.txt'},
                                  'RNA': {'layout_path': 'LR'}})
    vals = ui.modality_input_values()
    check('the design comes back out', vals['DNA']['probe_design_path'] == 'E:/Oligos/x_bed.txt',
          repr(vals['DNA']))
    check('without disturbing layout or DAX',
          vals['DNA']['layout_path'] == 'L' and vals['DNA']['dax_directory'] == 'D')
    check('a modality given none reads blank',
          vals['RNA']['probe_design_path'] == '')
    # a fields dict with no design key at all is what an older config
    # produces; it must not raise
    ui.set_modality_input_values({'DNA': {'layout_path': 'L2'}})
    check('an older field set without the key clears it, not crashes',
          ui.modality_input_values()['DNA']['probe_design_path'] == '')
    app.processEvents()


def test_the_real_designs_the_configs_now_name():
    print('\nthe designs the shipped configs point at, if this machine has them')
    import re
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    named, seen = 0, 0
    for fn in sorted(os.listdir(os.path.join(here, 'configs'))):
        if not fn.endswith('.xml'):
            continue
        with open(os.path.join(here, 'configs', fn), encoding='utf-8') as f:
            text = f.read()
        for pth in re.findall(r'probe_design_path="([^"]+)"', text):
            named += 1
            if not os.path.exists(pth):
                print('  --   %s: not reachable here (%s)' % (fn, pth))
                continue
            seen += 1
            d = P.read_probe_design(pth)
            res = re.search(r'genomic_resolution_kb="([0-9.]+)"', text)
            step = d.step_kb()
            ok = len(d) > 1 and step
            if res and step:
                # the design's own step must agree with what the config
                # claims -- a design pointed at another panel of the same
                # library is exactly what this catches
                ok = ok and abs(step - float(res.group(1))) < 1.5
            check('%-44s %s' % (fn, d.summary()), bool(ok),
                  'config says %s kb' % (res.group(1) if res else '?'))
    check('at least one config names a design', named > 0, '%d named' % named)
    if not seen:
        print('  --   none reachable on this machine, contents unchecked')


def main():
    for t in (test_the_manifest_carries_it,
              test_the_store_answers_and_params_win,
              test_the_config_carries_it,
              test_the_panel_has_a_row,
              test_the_real_designs_the_configs_now_name):
        t()
    print('\n%d passed, %d failed' % (_n['pass'], _n['fail']))
    return 1 if _n['fail'] else 0


if __name__ == '__main__':
    sys.exit(main())
