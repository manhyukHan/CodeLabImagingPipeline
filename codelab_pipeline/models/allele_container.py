"""
The two-tier allele container.

One class, two instances -- MainWindow.chromatin_alleles (transient: what
the listview shows and every edit touches) and chromatin_alleles_permanent
(the in-memory mirror of what is on disk). Same contract SpotContainer and
CellContainer already state: without saving, nothing changes data. Build
and Remove mutate transient only; Save copies transient -> permanent ->
disk.

WHY ALLELES NEEDED THIS
-----------------------
They were the odd one out. Cells and spots are staged, edited and then
saved; alleles lived in a plain dict that Build REPLACED wholesale and
that only the Fit All run ever wrote. Two consequences, both real:

  * saved alleles were never read back into the session, so restarting the
    app lost every allele that had not just been fitted;
  * APPEND meant "hybes not yet traced on this allele", which is a
    per-hybe rule. An allele half-traced by one engine could be finished
    by another, leaving one polymer_adj built from two estimators with nothing
    on disk saying so.

With two tiers, append becomes a MEMBERSHIP question -- trace the alleles
that are staged but not committed -- and the per-hybe rule disappears
along with the mixing it allowed.

THIS IS ENGINE-AGNOSTIC. Nothing here knows about v1 or v2. It is about
allele identity and staging, exactly as SpotContainer is about spot
identity; the engine question rides on top and is not its business.

IDENTITY
--------
`id`, unique per (storage_path, fov), allocated from `next_id` at the
moment an allele enters the transient tier -- never renumbered later.
AnAllele.set_metadata mints ids positionally from `enumerate`, restarting
at 1 on every Build, which was safe only while Build replaced the whole
list. Under ADD semantics that collides with the id-keyed lookups the
listview and the fit both use.

`anchor_uid` is NOT the identity: it is the dedup guard at the Build door
(do not stage two alleles on one spot). Legacy alleles carry anchor_uid=0,
so uid 0 never dedups.
"""
from .allele import AnAllele


class AlleleContainer:
    def __init__(self):
        self.data = {}   # {(storage_path, fov): {id: AnAllele}}

    # -- membership ------------------------------------------------------

    @staticmethod
    def _key(key):
        storage_path, fov = key
        return (str(storage_path), int(fov))

    def add(self, key, allele):
        """Stage one allele. Its id must already be real."""
        aid = int(getattr(allele, 'id', 0) or 0)
        if aid <= 0:
            raise ValueError(
                'allele has id=0 -- allocate a real id at creation via '
                'AlleleContainer.next_id; identity cannot be retrofitted at '
                'save time, and the listview, the preview fit and the append '
                'membership test all key on it')
        bucket = self.data.setdefault(self._key(key), {})
        if aid in bucket and bucket[aid] is not allele:
            raise ValueError(f'allele id {aid} already staged for {key}')
        bucket[aid] = allele
        return allele

    def of_fov(self, key):
        """Every staged allele for this (storage_path, fov), in id order."""
        return [a for _i, a in sorted(self.data.get(self._key(key), {}).items())]

    def by_id(self, key, allele_id):
        return self.data.get(self._key(key), {}).get(int(allele_id))

    def has(self, key, allele_id):
        return int(allele_id) in self.data.get(self._key(key), {})

    def has_traced(self, key, allele_id):
        """Present AND carrying at least one traced hybe.

        The membership test append uses. Presence alone would be wrong:
        Build -> Save stages an allele with an empty polymer_adj, and if that
        counted as committed the allele could never be fitted by append --
        only a full Overwrite would ever reach it. Requiring a non-empty
        polymer_adj makes "saved but not yet traced" a recoverable state
        instead of a trap.
        """
        a = self.by_id(key, allele_id)
        return bool(a is not None and getattr(a, 'polymer_adj', None))

    def remove(self, key, allele_ids):
        """Drop alleles from THIS tier only.

        Removing from transient does not touch disk -- exactly as removing
        a cell or a spot does not. The deletion reaches the store when the
        next Save replaces the FOV from the permanent tier.
        """
        bucket = self.data.get(self._key(key), {})
        return [bucket.pop(int(i)) for i in allele_ids if int(i) in bucket]

    def next_id(self, key):
        """One past the highest id this tier holds for the FOV.

        Callers staging into transient should pass the max over BOTH tiers
        so a Build cannot reuse an id that only exists on disk.
        """
        bucket = self.data.get(self._key(key), {})
        return (max(bucket) + 1) if bucket else 1

    def anchor_uids(self, key):
        """Non-zero anchor uids already staged, for the Build-door dedup."""
        return {int(getattr(a, 'anchor_uid', 0) or 0)
                for a in self.of_fov(key)} - {0}

    def count(self, key):
        return len(self.data.get(self._key(key), {}))

    # -- tier transfer ---------------------------------------------------

    def sync_from(self, other, key):
        """REPLACE this tier's FOV with the other's. Returns (n_kept, n_removed).

        Replace, not merge, so a removed allele really is removed -- the
        cells behaviour. Every transferred allele is a fresh object built
        through save()/set_metadata: the tracing worker mutates AnAllele
        objects IN PLACE, so if the tiers shared objects every fit would
        silently mutate permanent and `has_traced` would answer about data
        that was never saved.

        AnAllele has no light_copy and its arrays are not write-locked, so
        CellContainer's fingerprint shortcut does not apply. The
        save()/set_metadata round trip is what MainWindow already uses to
        rehydrate an allele, and it rounds on the way through, which makes
        the permanent tier byte-equal to what write_fov_alleles would emit.
        """
        k = self._key(key)
        src = other.data.get(k, {})
        bucket = self.data.setdefault(k, {})
        removed = [i for i in bucket if i not in src]
        for i in removed:
            del bucket[i]
        for aid, allele in src.items():
            copy = AnAllele()
            copy.set_metadata(**allele.save())
            bucket[aid] = copy
        return len(bucket), len(removed)

    def promote_one(self, other, key, allele_id):
        """Copy ONE allele across, leaving the rest of the FOV alone.

        What a per-FOV fit result needs: a batch in append mode traces a
        SUBSET, and a whole-FOV sync_from would delete the committed
        alleles it deliberately skipped.
        """
        allele = other.by_id(key, allele_id)
        if allele is None:
            return None
        copy = AnAllele()
        copy.set_metadata(**allele.save())
        self.data.setdefault(self._key(key), {})[int(allele_id)] = copy
        return copy

    def clear_fov(self, key):
        self.data.pop(self._key(key), None)
