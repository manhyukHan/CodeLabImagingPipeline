"""One hybe may carry more than one readout.

The probe design allows a second dye on the same adapter -- 635 reading
R1 and 747 reading R51 in one round -- which doubles the bins an
experiment can address. No layout in hand uses it; these pin the schema
and the parser so one can.

Run: python tests/test_multichannel_readout.py
"""
import os
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.io import preprocess as PRE          # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


BASE = {'FolderName': 'Hyb_001', 'Readouts': 1, 'DataType': 'H',
        'HybNum': 1, 'channels': '[475, 555, 635, 747]',
        'fiducialChannel': 555, 'channelLayout': 'alternate',
        'totalFrames': 440}


def layout(root, rows):
    p = os.path.join(root, 'L_%d.xlsx' % abs(hash(str(rows))))
    pd.DataFrame(rows).to_excel(p, index=False)
    return p


def main():
    root = tempfile.mkdtemp(prefix='mcro_')
    try:
        print('without the column: exactly as before')
        r = PRE.parse_experiment_layout(layout(root, [dict(BASE)]))[0]
        check('readout_id is still the scalar', r['readout_id'] == 1)
        check('readouts says the channel is UNKNOWN, not a guess',
              r['readouts'] == [(None, 1)], str(r['readouts']))

        print('\nwith the column')
        rows = [dict(BASE, readoutChannels='635:1, 747:51')]
        r = PRE.parse_experiment_layout(layout(root, rows))[0]
        check('both readouts are named', r['readouts'] == [(635, 1), (747, 51)],
              str(r['readouts']))
        check('readout_id keeps its old meaning', r['readout_id'] == 1)
        rows = [dict(BASE, readoutChannels='747:51;635:1')]
        r = PRE.parse_experiment_layout(layout(root, rows))[0]
        check('semicolons and order are accepted',
              set(r['readouts']) == {(635, 1), (747, 51)}, str(r['readouts']))
        rows = [dict(BASE, readoutChannels='   ')]
        r = PRE.parse_experiment_layout(layout(root, rows))[0]
        check('a blank cell falls back, it does not raise',
              r['readouts'] == [(None, 1)], str(r['readouts']))

        print('\nit refuses a layout error rather than repairing it')
        for bad, why in (('999:1', 'a channel the hybe does not have'),
                         ('555:1', 'the FIDUCIAL channel'),
                         ('635:1, 747:1', 'a duplicate readout id'),
                         ('635', 'not channel:readout'),
                         ('a:b', 'not integers')):
            try:
                PRE.parse_experiment_layout(
                    layout(root, [dict(BASE, readoutChannels=bad)]))
                check('refuses %s' % why, False, 'accepted %r' % bad)
            except ValueError as e:
                check('refuses %s' % why, 'Hyb_001' in str(e), str(e))

        print('\nthe real layouts still parse')
        real = (r'Q:\Seonghyeok\ORCA3\20251130_MP58_RNAandDNA_\TiftoDaxOutput'
                r'\251208_YSH_MP58_RNADNA-ExperimentLayout.xlsx')
        if os.path.exists(real):
            recs = PRE.parse_experiment_layout(real)
            check('MP58: every record has one unnamed readout',
                  all(x['readouts'] == [(None, x['readout_id'])] for x in recs),
                  '%d records' % len(recs))
        else:
            print('  (skipped: %s not reachable)' % real)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
