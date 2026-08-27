"""
The experiments the fit/gate investigation runs against, in one place.

WHY A REGISTRY
--------------
fit_testbox.py was pinned to MP58 by module-level constants, which was
right while there was one dataset and wrong the moment there were four.
Every conclusion so far -- the PSF shape, the gate thresholds, the
box-not-pillar fix -- rests on ONE experiment, and a claim that survives
only where it was derived is not a finding. This is the list it has to
survive on.

Each entry names the allele-candidate hybe and channel explicitly rather
than deriving them, because they were chosen by a person who knows which
round is clean in that experiment, and that is not recoverable from the
store.

`survey` reports what is actually THERE -- hybes by datatype, replicate
pairs, candidate spots per FOV -- so a sample size is chosen against real
counts instead of a guess.

Usage:
    python tools/experiments.py survey [--name MP58]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Experiment(object):
    """One experiment, plus the one property that predicts its PSF.

    GENOMIC SCOPE IS NOT METADATA -- IT IS A CONFOUNDER
    ---------------------------------------------------
    The fiducial is not a point source. It is the whole genomic region
    the readouts collectively trace, so its apparent size is physical,
    not optical, and it scales with how much DNA is in scope. MP58's
    fiducial already measured 246 nm laterally against its readout's
    146 nm -- a 1.7x difference that was read as "extended object" but
    never tied to a number.

    These four span 2 Mb to ~20 Mb, an order of magnitude, which makes
    the universal-PSF question two separate questions with opposite
    expected answers:

        readout   point source, one microscope   -> SHOULD be universal
        fiducial  extended, scope-dependent      -> should NOT be, and
                                                    should grow with Mb

    That is a prediction with a direction, so it can be wrong. If the
    fiducial shape does NOT track scope, the "extended object" reading
    is incomplete; if the readout does NOT hold across experiments, a
    single default PSF cannot be shipped no matter how tidy it would be.
    """

    def __init__(self, name, config, modality, anchor_hybe, anchor_channel,
                 reference_hybe, fovs=None, datatypes=('H', 'T', 'R'),
                 seed=20260827, scope_mb=None, step_kb=None, barcodes=None,
                 locus='', note=''):
        self.name = name
        self.config = config
        self.modality = modality
        self.anchor_hybe = anchor_hybe
        self.anchor_channel = anchor_channel
        # The drift baseline. Kept SEPARATE from anchor_hybe even where
        # they coincide: they are different jobs (which spots become
        # alleles, versus which round every other round is aligned to),
        # and collapsing them would hide the difference in experiments
        # where the config picks another baseline.
        self.reference_hybe = reference_hybe
        self.fovs = fovs                    # None = whatever the config lists
        self.datatypes = datatypes
        self.seed = seed
        # Approximate size of the traced region, in megabases, as stated
        # by the person who designed the probe set. Approximate is fine:
        # the four span an order of magnitude, so a trend either shows at
        # that scale or there is no trend to find.
        self.scope_mb = scope_mb
        # Probe spacing and barcode count, from the probe design. Kept
        # because they are NOT redundant with scope_mb: barcode counts are
        # similar across these four (58-91) while the span differs 64-fold,
        # so the labelled FRACTION of the region differs enormously and any
        # size comparison has to know that.
        self.step_kb = step_kb
        self.barcodes = barcodes
        self.locus = locus
        self.note = note

    def __repr__(self):
        return f'<Experiment {self.name}>'


# Anchors are the user's explicit choices, one per experiment.
EXPERIMENTS = {
    'MP58': Experiment(
        'MP58', 'configs/2025-11-30-MP58-testbox.xml', 'DNA',
        anchor_hybe='Hyb_016', anchor_channel=555, reference_hybe='Hyb_016',
        fovs=(1, 2, 4, 5), seed=20260826,
        scope_mb=4.0, step_kb=50, barcodes=80,
        locus='NC_048598.1:26887070-28887070 + 28889070-30889070',
        note='two 2 Mb segments with a 17 kb gap at the integration site'),
    'Chr19': Experiment(
        'Chr19', 'configs/2025-12-08-JY-Chr19_Downstream.xml', 'DNA',
        anchor_hybe='Hyb_001', anchor_channel=555, reference_hybe='Hyb_001',
        scope_mb=18.5, step_kb=200, barcodes=91,
        locus='chr19:40000000-54716295 + 54864547-58617616',
        note='tiles chr19 after the centromere; 6mA spreading study'),
    'CrossMod': Experiment(
        'CrossMod', 'configs/2026-01-26-JP-C7cSP8.xml', 'DNA',
        anchor_hybe='Hyb_002', anchor_channel=555, reference_hybe='Hyb_002',
        scope_mb=18.5, step_kb=200, barcodes=91,
        locus='chr19:40000000-54716295 + 54864547-58617616',
        note='SAME probe design as Chr19 -- so the two are a reproducibility '
             'check on the calibration, not two independent scope points'),
    'HoxA': Experiment(
        'HoxA', 'configs/2026-07-22-DI-DNA-HoxA.xml', 'DNA',
        anchor_hybe='Hyb_006', anchor_channel=555, reference_hybe='Hyb_006',
        scope_mb=0.29, step_kb=5, barcodes=58,
        locus='hg19 chr7:27067612-27357711',
        note='290 kb, the smallest scope by 14x. Was initially estimated at '
             '2 Mb; the layout filename (HoxA_5kb) contradicted that and the '
             'real design confirmed 290 kb. It anchors the low end of the '
             'range, so the error mattered: the fitted exponent moved from '
             '+0.16 to +0.09 when it was corrected.'),
}


def config_fovs(config_path):
    """The FOV list the config declares, as ints.

    Read from the XML rather than from the running MainWindow: the widget
    that holds it lives inside an input panel, and reaching through the UI
    for a value that is sitting in a file is how a survey ends up
    reporting zero FOVs (which is exactly what the first version did).
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(config_path).getroot()
    raw = root.get('fov_list', '') or ''
    out = []
    for part in raw.replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(out)


_APP = None      # keeps the QApplication alive; see open_session


def open_session(exp):
    """A real MainWindow on this experiment's config.

    The production path, not a reimplementation of it: hybe records,
    datatypes and channel roles all come from the layout the app reads,
    and a survey that guessed them could report a bench that cannot be
    harvested.
    """
    global _APP
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from unittest import mock
    from PyQt5 import QtWidgets
    # The reference is NOT decoration. Written as a bare expression the
    # QApplication is created and immediately garbage collected, and every
    # Qt call after that runs against a destroyed application -- which
    # exits 0xC0000409 with no Python traceback and no output at all, so
    # it looks like the process died before it started.
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for m in ('critical', 'warning', 'information', 'question'):
        mock.patch.object(QtWidgets.QMessageBox, m,
                          return_value=QtWidgets.QMessageBox.Yes).start()
    from windows.main_window import MainWindow
    mw = MainWindow(exp.config)
    sp = mw._storage_path_for_modality(exp.modality)
    return mw, sp


def survey_one(exp):
    """What this experiment actually offers. Read-only."""
    from codelab_pipeline.io import analysis_store as V
    from tools.fit_testbox import replicate_pairs

    mw, sp = open_session(exp)
    records = mw._hybe_records_for_storage_path(sp)
    by_type = {}
    for r in records:
        by_type.setdefault(str(r['datatype']).upper(), []).append(r['folder'])
    traced = [r['folder'] for r in records
              if str(r['datatype']).upper() in exp.datatypes]
    pairs = replicate_pairs(records)

    fovs = list(exp.fovs) if exp.fovs else list(config_fovs(exp.config))
    spots = {}
    for fov in fovs:
        try:
            s = V.read_spots(sp, fov, exp.modality, exp.anchor_hybe,
                             exp.anchor_channel)
            spots[fov] = len(s)
        except Exception as e:
            spots[fov] = f'ERR {type(e).__name__}'
    return {
        'exp': exp, 'storage_path': sp, 'records': records,
        'by_type': by_type, 'traced': traced, 'pairs': pairs,
        'fovs': fovs, 'spots': spots,
    }


def print_survey(s):
    e = s['exp']
    print(f'=== {e.name} ===')
    print(f'  config      : {e.config}')
    print(f'  store       : {s["storage_path"]}')
    if e.note:
        print(f'  note        : {e.note}')
    print(f'  rounds      : {len(s["records"])} total  ' +
          '  '.join(f'{k}:{len(v)}' for k, v in sorted(s['by_type'].items())))
    print(f'  traced set  : {len(s["traced"])} of datatype {"/".join(e.datatypes)}')
    print(f'  anchor      : {e.anchor_hybe} ch{e.anchor_channel}   '
          f'reference: {e.reference_hybe}')
    print(f'  replicates  : {len(s["pairs"])} same-locus H/R pairs')
    if s['pairs']:
        show = ', '.join(f'{a}~{b}' for a, b, _rid in s['pairs'][:6])
        print(f'                {show}{" ..." if len(s["pairs"]) > 6 else ""}')
    tot = sum(v for v in s['spots'].values() if isinstance(v, int))
    per = '  '.join(f'FOV{f:03d}:{v}' for f, v in sorted(s['spots'].items()))
    print(f'  candidates  : {tot} spots over {len(s["fovs"])} FOV   {per}')
    print()
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['survey', 'list'])
    ap.add_argument('--name', action='append', default=None)
    a = ap.parse_args()

    names = a.name or list(EXPERIMENTS)
    if a.command == 'list':
        for n in names:
            print(n, EXPERIMENTS[n].config)
        return

    grand = 0
    for n in names:
        try:
            grand += print_survey(survey_one(EXPERIMENTS[n]))
        except Exception as e:
            import traceback
            print(f'=== {n} ===\n  FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            print()
    print(f'total candidate spots across surveyed experiments: {grand}')


if __name__ == '__main__':
    main()
