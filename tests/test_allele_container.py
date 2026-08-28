"""
Tests for the two-tier allele container.

The failures worth pinning are the silent ones: tiers sharing objects (so
the tracing worker's in-place mutation reaches the permanent tier and
`has_traced` answers about data nobody saved), ids reused across a Build
(so the listview and the fit key onto the wrong allele), and Remove
appearing to work while leaving the store untouched.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_allele_container.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline.models.allele import AnAllele                 # noqa: E402
from codelab_pipeline.models.allele_container import AlleleContainer  # noqa: E402

PASS, FAIL = [], []
KEY = ('G:/store/DNA', 3)


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


def make(aid, uid=0, traced=False):
    a = AnAllele()
    a.set_metadata(id=aid, fov=KEY[1], cell=7, anchor_uid=uid,
                   anchor_hybe='Hyb_016', anchor_channel=555,
                   coordinate=(100.0, 200.0, 0.0), raw_coordinate=(100.0, 200.0, 0.0))
    if traced:
        a.polymer_adj = {'Hyb_020': [(1.0, 2.0, 3.0, 500.0)]}
    return a


def main():
    print('identity')
    t = AlleleContainer()
    check('next_id starts at 1 on an empty FOV', t.next_id(KEY) == 1)
    t.add(KEY, make(1))
    t.add(KEY, make(2))
    check('next_id is one past the highest', t.next_id(KEY) == 3, str(t.next_id(KEY)))
    try:
        t.add(KEY, make(0))
        check('id=0 is refused', False)
    except ValueError as e:
        check('id=0 is refused with an explanation', 'retrofitted' in str(e))
    try:
        t.add(KEY, make(2))
        check('a duplicate id is refused', False)
    except ValueError:
        check('a duplicate id is refused', True)
    check('the key is normalised, so str/int mixes still find the FOV',
          t.count(('G:/store/DNA', 3)) == 2 and t.by_id(('G:/store/DNA', '3'), 1) is not None)
    check('anchor_uids ignores legacy uid 0',
          t.anchor_uids(KEY) == set(), str(t.anchor_uids(KEY)))
    t.add(KEY, make(3, uid=99))
    check('and reports a real anchor uid', t.anchor_uids(KEY) == {99})

    print('\ntiers never share objects')
    perm = AlleleContainer()
    perm.sync_from(t, KEY)
    check('sync_from copies every allele', perm.count(KEY) == 3)
    # the tracing worker mutates AnAllele IN PLACE; shared objects would
    # make every fit silently write into the permanent tier
    t.by_id(KEY, 1).polymer_adj = {'Hyb_030': [(9.0, 9.0, 9.0, 1.0)]}
    check('mutating transient does NOT reach permanent',
          not perm.by_id(KEY, 1).polymer_adj, str(perm.by_id(KEY, 1).polymer_adj))
    perm.by_id(KEY, 2).rejected_hybes = {'Hyb_040': 'nope'}
    check('and mutating permanent does not reach transient',
          not t.by_id(KEY, 2).rejected_hybes)

    print('\nmembership: what append asks')
    p2 = AlleleContainer()
    p2.add(KEY, make(1))                    # saved, never traced
    p2.add(KEY, make(2, traced=True))       # saved and traced
    check('has() is true for a saved-but-untraced allele', p2.has(KEY, 1))
    check('has_traced() is FALSE for it -- append must still fit it',
          not p2.has_traced(KEY, 1))
    check('has_traced() is true once it carries a polymer_adj', p2.has_traced(KEY, 2))
    check('both are false for an allele that was never saved',
          not p2.has(KEY, 99) and not p2.has_traced(KEY, 99))

    print('\nremove is transient-only until Save')
    t2, p3 = AlleleContainer(), AlleleContainer()
    for i in (1, 2, 3):
        t2.add(KEY, make(i))
    p3.sync_from(t2, KEY)
    dropped = t2.remove(KEY, [2])
    check('remove returns what it dropped', len(dropped) == 1 and dropped[0].id == 2)
    check('and drops it from the transient tier', t2.count(KEY) == 2)
    check('but the permanent tier is UNTOUCHED until Save',
          p3.count(KEY) == 3, f'{p3.count(KEY)} still committed')
    kept, removed = p3.sync_from(t2, KEY)
    check('Save then really deletes it', p3.count(KEY) == 2 and removed == 1,
          f'kept {kept}, removed {removed}')
    check('removing an id that is not there is a no-op',
          t2.remove(KEY, [999]) == [])

    print('\npromote_one: what a partial batch needs')
    t3, p4 = AlleleContainer(), AlleleContainer()
    for i in (1, 2, 3):
        t3.add(KEY, make(i, traced=(i == 2)))
    p4.add(KEY, make(1, traced=True))       # already committed, NOT in this batch
    p4.promote_one(t3, KEY, 2)
    check('promoting one allele leaves the others committed',
          p4.count(KEY) == 2 and p4.has_traced(KEY, 1) and p4.has_traced(KEY, 2),
          f'{sorted(p4.data[(str(KEY[0]), KEY[1])])}')
    check('promote_one also copies rather than sharing',
          p4.by_id(KEY, 2) is not t3.by_id(KEY, 2))
    check('promoting a missing id returns None',
          p4.promote_one(t3, KEY, 999) is None)

    print('\nordering')
    t4 = AlleleContainer()
    for i in (5, 1, 3):
        t4.add(KEY, make(i))
    check('of_fov is ordered by id, not by insertion',
          [a.id for a in t4.of_fov(KEY)] == [1, 3, 5],
          str([a.id for a in t4.of_fov(KEY)]))

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


if __name__ == '__main__':
    sys.exit(main())
