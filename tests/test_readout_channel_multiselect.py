"""The readout channel selector takes more than one channel.

One hybe can carry two readouts -- 635 reading R1 and 747 reading R51 on
the same adapter -- so the panel's single-select combo became a checkable
list. These pin the selector's behaviour without building the whole app,
which cannot load here (its fixture config names a Mac path).

Run: QT_QPA_PLATFORM=offscreen python tests/test_readout_channel_multiselect.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets, QtCore                         # noqa: E402

from ui.chromatin_tracing_panel import ChromatinTracingPanelUI  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(('  ok ' if cond else 'FAIL ') + name
          + (f'   [{detail}]' if detail and not cond else ''))


def states(p):
    lw = p.TraceReadoutChannelListWidget
    return {lw.item(i).data(QtCore.Qt.UserRole):
            lw.item(i).checkState() == QtCore.Qt.Checked
            for i in range(lw.count())}


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host = QtWidgets.QWidget()
    p = ChromatinTracingPanelUI()
    p.setupUi(host)
    if not hasattr(p, 'TraceReadoutChannelListWidget'):
        print('panel did not build its selector; API changed?')
        return 1

    print('defaults')
    check('starts on auto', p.active_readout_channels() == ('auto',),
          str(p.active_readout_channels()))

    print('\npopulate keeps the active choice')
    p.populate_readout_channel_choices([555, 635, 747], dominant=635)
    check('auto plus every channel is offered',
          set(states(p)) == {'auto', '555', '635', '747'}, str(states(p)))
    check('auto is still the active one', states(p)['auto'] is True)
    check('the label names the resolved channel',
          '635' in p.ActiveReadoutChannelLabel.text(),
          p.ActiveReadoutChannelLabel.text())

    print('\ntwo channels at once')
    p.set_active_readout_channels('635,747')
    check('both are active', p.active_readout_channels() == ('635', '747'),
          str(p.active_readout_channels()))
    check('and both are checked',
          states(p)['635'] and states(p)['747'] and not states(p)['auto'])
    check('the label shows both', '635' in p.ActiveReadoutChannelLabel.text()
          and '747' in p.ActiveReadoutChannelLabel.text(),
          p.ActiveReadoutChannelLabel.text())

    print('\nauto is exclusive')
    lw = p.TraceReadoutChannelListWidget
    for i in range(lw.count()):
        if lw.item(i).data(QtCore.Qt.UserRole) == 'auto':
            lw.item(i).setCheckState(QtCore.Qt.Checked)
    s = states(p)
    check('checking auto clears the named channels',
          s['auto'] and not s['635'] and not s['747'], str(s))
    for i in range(lw.count()):
        if lw.item(i).data(QtCore.Qt.UserRole) == '747':
            lw.item(i).setCheckState(QtCore.Qt.Checked)
    s = states(p)
    check('naming a channel clears auto', s['747'] and not s['auto'], str(s))

    print('\nActivate is still the gate')
    before = p.active_readout_channels()
    for i in range(lw.count()):
        if lw.item(i).data(QtCore.Qt.UserRole) == '555':
            lw.item(i).setCheckState(QtCore.Qt.Checked)
    check('ticking a box does not retarget the run',
          p.active_readout_channels() == before, str(p.active_readout_channels()))
    p._activate_readout_channel()
    check('Activate commits it', set(p.active_readout_channels()) == {'555', '747'},
          str(p.active_readout_channels()))

    print('\nan old config holding one channel still restores')
    p.set_active_readout_channels('635')
    check('single string works', p.active_readout_channels() == ('635',),
          str(p.active_readout_channels()))

    print('\na channel that left the layout is dropped, not carried')
    p.set_active_readout_channels('635,747')
    p.populate_readout_channel_choices([635], dominant=635)
    check('747 is gone, 635 stays', p.active_readout_channels() == ('635',),
          str(p.active_readout_channels()))

    print('\nauto is exclusive on the config door too')
    p.populate_readout_channel_choices([555, 635, 747], dominant=635)
    try:
        p.set_active_readout_channels('auto,635')
        check('auto beside a named channel is refused', False, 'accepted')
    except ValueError as e:
        check('auto beside a named channel is refused',
              'exclusive' in str(e), str(e))
    p.set_active_readout_channels('auto')
    check('auto alone restores', p.active_readout_channels() == ('auto',),
          str(p.active_readout_channels()))
    # and ticking auto beside one in the WIDGET clears the other, so the
    # pair can never reach the activated tuple in the first place
    lw = p.TraceReadoutChannelListWidget
    for i in range(lw.count()):
        if lw.item(i).data(QtCore.Qt.UserRole) == '635':
            lw.item(i).setCheckState(QtCore.Qt.Checked)
    for i in range(lw.count()):
        if lw.item(i).data(QtCore.Qt.UserRole) == 'auto':
            lw.item(i).setCheckState(QtCore.Qt.Checked)
    p._activate_readout_channel()
    check('ticking auto unticks the named channel',
          p.active_readout_channels() == ('auto',),
          str(p.active_readout_channels()))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
