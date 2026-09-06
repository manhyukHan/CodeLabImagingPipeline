"""
'readout' is a collective noun, not a channel, and the code now says so.

WHY THIS EXISTS. Hybes in one experiment do not carry the same channels.
On the real MAZ layout, with fiducial 555:

    Hyb_071  [555, 635]                 'readout' -> 635
    Hyb_104  [475, 555, 635]            'readout' -> 475
    Hyb_007  [475, 555, 635, 640]       'readout' -> 475   (635, 640 lost)
    Hyb_BF   [999, 475, 390, 555, 635]  'readout' -> 999   (a brightfield slot)

The rule behind that is "the first non-fiducial in layout order", so a
single run resolving 'readout' per hybe compared 635 against 475 across
hybes of one experiment, stored the result as if one channel had been used,
and recorded nothing about it. Asking for a CONCRETE channel was no safer:
a hybe lacking it silently got the same first-non-fiducial substitute.

What this file pins:

  - resolve_channel reports the substitution instead of hiding it, and
    reports it only on POSITIVE evidence -- a channel list that exists and
    lacks the channel. Absent metadata is unknown, not absent, because the
    ingestion-readiness gate owns that decision and a channel check that
    also rejected incomplete records would take it away.
  - channel_coverage names the hybes that cannot supply a channel, which
    is what lets a run skip them and say which.
  - every label an operator sees names the wavelength: "555 (fiducial)".
  - the chromatin-tracing default check runs all five specified steps,
    including the channel step that was missing: gate on modality and
    datatype, count the non-fiducial channels, take the most shared one,
    drop the hybes without it.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_channel_semantics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets                                # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from codelab_pipeline.alignment import chain               # noqa: E402
from windows.main_window import MainWindow                 # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def rec(folder, channels, fiducial=555, datatype='H'):
    return {'folder': folder, 'channels': channels,
            'fiducial_channel': fiducial, 'datatype': datatype}


# The real MAZ shapes, kept as the fixture because they are what broke.
H071 = rec('Hyb_071', [555, 635])
H104 = rec('Hyb_104', [475, 555, 635])
H007 = rec('Hyb_007', [475, 555, 635, 640])
HBF = rec('Hyb_BF', [999, 475, 390, 555, 635])


def main():
    print('\n-- fiducial resolves from the layout, and always could --')
    for r in (H071, H104, H007, HBF):
        ch, missing = chain.resolve_channel(r, 'fiducial')
        check(f'{r["folder"]}: fiducial -> 555', ch == 555 and not missing, str(ch))

    print("\n-- 'readout' is NOT one channel, and that is now visible --")
    got = {r['folder']: chain.resolve_channel(r, 'readout')[0]
           for r in (H071, H104, H007, HBF)}
    check('the same label gives different wavelengths per hybe',
          got == {'Hyb_071': 635, 'Hyb_104': 475, 'Hyb_007': 475, 'Hyb_BF': 999},
          str(got))
    check('including a brightfield slot when one sorts first',
          got['Hyb_BF'] == 999)
    check('and it never reports itself as missing, because it always resolves',
          not any(chain.resolve_channel(r, 'readout')[1]
                  for r in (H071, H104, H007, HBF)))

    print('\n-- a concrete channel is consistent, or says it is not there --')
    for r in (H071, H104, H007, HBF):
        ch, missing = chain.resolve_channel(r, '635')
        check(f'{r["folder"]}: 635 -> 635, present', ch == 635 and not missing)
    ch, missing = chain.resolve_channel(H071, '640')
    check('a hybe without 640 REPORTS it rather than substituting silently',
          missing, f'returned {ch}, missing={missing}')
    check('and still returns something, so a renderer has pixels to draw',
          ch is not None)
    check('pick_channel_by_type keeps the substituting behaviour for renderers',
          chain.pick_channel_by_type(H071, '640') == 635)

    print('\n-- missing evidence is not evidence of missing --')
    # The ingestion-readiness gate owns "is this hybe usable". A record with
    # no channel list is unknown, and must not be skipped by the CHANNEL
    # check -- doing so took FOVs away from that gate and broke its report.
    bare = {'folder': 'Hyb_X'}
    _ch, missing = chain.resolve_channel(bare, '635')
    check('a record with no channel list is NOT reported as missing', not missing)
    _resolved, gaps = chain.channel_coverage([bare], '635')
    check('so coverage leaves it alone', gaps == [], str(gaps))

    print('\n-- coverage names exactly the hybes that cannot supply it --')
    resolved, missing = chain.channel_coverage([H071, H104, H007, HBF], '640')
    check('640 is carried by one hybe only',
          resolved == {'Hyb_007': 640}, str(resolved))
    check('and the other three are named',
          missing == ['Hyb_071', 'Hyb_104', 'Hyb_BF'], str(missing))
    resolved, missing = chain.channel_coverage([H071, H104, H007, HBF], '635')
    check('635 is carried by all four, so nothing is skipped',
          not missing and len(resolved) == 4, str(missing))

    print('\n-- every label names the wavelength --')
    check('fiducial reads as "555 (fiducial)"',
          chain.channel_label(H104, 'fiducial') == '555 (fiducial)',
          chain.channel_label(H104, 'fiducial'))
    check('the auto rule says so, and which one it picked',
          chain.channel_label(H104, 'readout') == '475 (readout, auto)',
          chain.channel_label(H104, 'readout'))
    check('a concrete channel is just the number',
          chain.channel_label(H104, '635') == '635')
    check('and a channel this hybe lacks says that instead of a number',
          'not in this hybe' in chain.channel_label(H071, '640'),
          chain.channel_label(H071, '640'))

    print('\n-- the chromatin-tracing default runs all five steps --')
    listing = [
        (rec('Hyb_001', [555, 635]), 'DNA'),
        (rec('Hyb_002', [475, 555, 635]), 'DNA'),
        (rec('Hyb_003', [475, 555, 635, 640]), 'DNA'),
        (rec('Hyb_004', [475, 555]), 'DNA'),                    # no 635
        (rec('Rep_001', [555, 635], datatype='R'), 'DNA'),
        (rec('Toe_001', [555, 635], datatype='T'), 'DNA'),
        (rec('Hyb_500', [555, 635], datatype='B'), 'DNA'),      # barcode
        (rec('Hyb_101', [555, 635]), 'RNA'),                    # wrong modality
    ]
    target, allowed, n_gated = MainWindow._default_chromatin_tracing_hybes(listing)
    check('step 1: B rounds and RNA are gated out',
          n_gated == 6, str(n_gated))
    check('step 3: the most-shared non-fiducial channel wins (635 in 5, 475 in 3)',
          target == 635, str(target))
    check('step 4-5: only the hybes carrying it are checked',
          sorted(f for f, _m in allowed) ==
          ['Hyb_001', 'Hyb_002', 'Hyb_003', 'Rep_001', 'Toe_001'],
          str(sorted(f for f, _m in allowed)))
    check('so the hybe without the target channel is left unchecked',
          ('Hyb_004', 'DNA') not in allowed)

    print('\n-- and it is stable, not dict-order dependent --')
    tie = [(rec('A', [555, 635]), 'DNA'), (rec('B', [555, 475]), 'DNA')]
    first, _a, _n = MainWindow._default_chromatin_tracing_hybes(tie)
    second, _a2, _n2 = MainWindow._default_chromatin_tracing_hybes(list(reversed(tie)))
    check('a tie breaks the same way whichever order the hybes arrive in',
          first == second, f'{first} vs {second}')

    print('\n-- nothing to gate on is not a crash --')
    t, a, n = MainWindow._default_chromatin_tracing_hybes([])
    check('an empty listing yields no target and no selection',
          t is None and a == set() and n == 0)
    t, a, n = MainWindow._default_chromatin_tracing_hybes(
        [(rec('Hyb_9', [555]), 'DNA')])
    check('a hybe with only a fiducial yields no target',
          t is None and a == set(), f'{t} {a}')

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
