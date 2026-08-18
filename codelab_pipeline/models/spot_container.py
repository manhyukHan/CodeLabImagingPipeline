"""
The two-tier spot container.

One class, two instances: MainWindow.spot_container (transient -- what every
displayer and edit touches) and spot_container_permanent (in-memory mirror of
vlinks). Without saving, nothing changes data: edits mutate transient only;
Save copies the current slice transient -> permanent -> disk; Revert copies
permanent -> transient.

One collection per FOV, keyed by uid, holding EVERY spot -- assigned and
unassigned together. `cell` is a field (-1 = unassigned), never a location:
the assigned/unassigned split across two structures is exactly what this
replaces (fov_unassigned_spots + cell.spots), and per explicit decision the
old structures are removed outright rather than kept in sync -- preserving
them would hand back false green signs.

uid is the identity everything keys on. Spots must arrive with a real uid
(allocate via vlinks_store.allocate_spot_uids at CREATION, not at save):
diff/undo cannot tell "moved" from "removed + added" without it.

Undo/redo is DIFF-based, two streaks deep, native to the container: a diff
spans the whole FOV (no further narrowing by hybe/channel -- per explicit
decision, the slice view exists for persistence scope, not for diffing).
"""
from copy import deepcopy

from .spot import ASpot


class SpotContainer:
    def __init__(self):
        self.data = {}   # {fov: {uid: ASpot}}

    # -- membership ------------------------------------------------------
    def add(self, fov, spot):
        uid = int(getattr(spot, 'uid', 0))
        if uid == 0:
            raise ValueError('spot has uid=0 -- allocate a real uid at creation '
                             '(vlinks_store.allocate_spot_uids); identity cannot '
                             'be retrofitted at save time')
        fov_spots = self.data.setdefault(int(fov), {})
        if uid in fov_spots and fov_spots[uid] is not spot:
            raise ValueError(f'uid {uid} already present in FOV {fov} -- uids are never reused')
        fov_spots[uid] = spot
        return spot

    def add_many(self, fov, spots):
        for s in spots:
            self.add(fov, s)

    def remove(self, fov, uids):
        fov_spots = self.data.get(int(fov), {})
        removed = [fov_spots.pop(int(u)) for u in uids if int(u) in fov_spots]
        return removed

    # -- views (filters, never separate storage) -------------------------
    def all(self, fov):
        return list(self.data.get(int(fov), {}).values())

    def slice(self, fov, modality, hybe, channel):
        """The persistence/edit unit: one (modality, hybe, channel)."""
        return [s for s in self.all(fov)
                if s.modality == modality and s.hybe == hybe and int(s.channel) == int(channel)]

    def of_cell(self, fov, cell_id):
        """Replaces ACell.spots reads -- a cell's spots are a QUERY."""
        return [s for s in self.all(fov) if int(s.cell) == int(cell_id)]

    def unassigned(self, fov):
        return [s for s in self.all(fov) if int(s.cell) == -1]

    def replace_slice(self, fov, modality, hybe, channel, spots):
        """Slice-scoped replace -- the edit primitive Clear/auto-detect use."""
        for s in self.slice(fov, modality, hybe, channel):
            del self.data[int(fov)][int(s.uid)]
        self.add_many(fov, spots)

    # -- tier transfer ---------------------------------------------------
    def copy_slice_from(self, other, fov, modality, hybe, channel):
        """Pull one slice from the other tier (Save into permanent; Revert
        into transient). Deep-copies so the tiers never share objects."""
        self.replace_slice(fov, modality, hybe, channel,
                           [deepcopy(s) for s in other.slice(fov, modality, hybe, channel)])

    # -- diff / undo -----------------------------------------------------
    def fingerprint(self, fov):
        """{uid: save()-dict} -- the comparison form. Plain dicts, so a
        fingerprint costs ~200 bytes/spot, not a deep ACell/ASpot graph."""
        return {uid: s.save() for uid, s in self.data.get(int(fov), {}).items()}

    @staticmethod
    def diff(before, after):
        """
        Native whole-FOV diff between two fingerprints:
        {'added': {uid: dict}, 'removed': {uid: dict},
         'changed': {uid: (before_dict, after_dict)}}
        Invertible by construction -- see apply_inverse.
        """
        added = {u: d for u, d in after.items() if u not in before}
        removed = {u: d for u, d in before.items() if u not in after}
        changed = {u: (before[u], after[u])
                   for u in before.keys() & after.keys() if before[u] != after[u]}
        return {'added': added, 'removed': removed, 'changed': changed}

    @staticmethod
    def is_empty_diff(d):
        return not (d['added'] or d['removed'] or d['changed'])

    def apply_inverse(self, fov, d):
        """Undo one diff in place: drop what was added, restore what was
        removed, roll changed spots back to their before-state."""
        fov_spots = self.data.setdefault(int(fov), {})
        for uid in d['added']:
            fov_spots.pop(int(uid), None)
        for uid, saved in d['removed'].items():
            s = ASpot(); s.set_metadata(**saved)
            fov_spots[int(uid)] = s
        for uid, (before_d, _after) in d['changed'].items():
            s = ASpot(); s.set_metadata(**before_d)
            fov_spots[int(uid)] = s

    def apply_forward(self, fov, d):
        """Redo one diff in place -- the mirror of apply_inverse."""
        fov_spots = self.data.setdefault(int(fov), {})
        for uid, saved in d['added'].items():
            s = ASpot(); s.set_metadata(**saved)
            fov_spots[int(uid)] = s
        for uid in d['removed']:
            fov_spots.pop(int(uid), None)
        for uid, (_before, after_d) in d['changed'].items():
            s = ASpot(); s.set_metadata(**after_d)
            fov_spots[int(uid)] = s


class DiffUndo:
    """
    Two-deep undo/redo over one SpotContainer, storing DIFFS, not
    snapshots. Usage: fp = container.fingerprint(fov) before an edit;
    push(fov, fp) after it. Empty diffs are dropped so a no-op edit never
    consumes one of the two slots.
    """
    DEPTH = 2

    def __init__(self, container):
        self.container = container
        self._undo = []   # [(fov, diff)]
        self._redo = []

    def push(self, fov, before_fingerprint):
        d = SpotContainer.diff(before_fingerprint, self.container.fingerprint(fov))
        if SpotContainer.is_empty_diff(d):
            return False
        self._undo.append((int(fov), d))
        del self._undo[:-self.DEPTH]
        self._redo.clear()
        return True

    def undo(self):
        if not self._undo:
            return None
        fov, d = self._undo.pop()
        self.container.apply_inverse(fov, d)
        self._redo.append((fov, d))
        del self._redo[:-self.DEPTH]
        return fov

    def redo(self):
        if not self._redo:
            return None
        fov, d = self._redo.pop()
        self.container.apply_forward(fov, d)
        self._undo.append((fov, d))
        return fov

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)
