"""Which (hybe, channel) is which readout -- the last layer before a trace.

Everything BELOW a trace works in (hybe, channel): ingestion, alignment,
spot localization. Everything ABOVE it works in (rindex, dtype, pindex):
AnAllele, export, every polymer analysis. This module is the one place
the two meet, so that generalising the acquisition side never reaches
into the analysis side and the other way round.

    (hybe, channel)  ->  ReadoutBin(rindex, dtype, pindex)

  rindex  the readout id -- the genomic-locus identity the layout gives,
          shared by an H bin and the R round that re-images it
  dtype   H / R / T / B, the layout's own DataType. Only H is a polymer
          bin; R and T are QC rounds and B is the cell barcode
  pindex  position along the polymer, 0-based, H only. -1 for everything
          else -- the convention export._allele_axis already uses for
          "a real measurement, not a genomic bin"

WHY A MAP AND NOT A RULE. One hybe can carry two readouts: a second dye
on the same adapter reads a different barcode in the same round
(635/adapter1 -> R1 and 747/adapter1 -> R51), which doubles the bins an
experiment can address. A rule that derives the readout from the hybe
alone cannot express that; a table can. The layout SEEDS the table
(preprocess._parse_readouts fills `record['readouts']`), and the
chromatin tracing panel is where it is finally settled, so an operator
can correct a mapping without editing the ExperimentLayout.

The settled map is stamped into the trace, and the per-coordinate
rindex/dtype/pindex are stored beside each position. That is deliberate
duplication: a trace then says what it is without needing its map
re-resolved, and editing the map later cannot silently re-label
measurements already taken.
"""
import collections

ReadoutBin = collections.namedtuple(
    'ReadoutBin', 'hybe channel rindex dtype pindex')

NOT_A_BIN = -1          # pindex for anything that is not a polymer bin


class ReadoutMap(object):
    """An ordered table of ReadoutBin, keyed by (hybe, channel)."""

    def __init__(self, bins=()):
        self._by_key = {}
        for b in bins:
            key = (str(b.hybe), int(b.channel))
            if key in self._by_key:
                raise ValueError(
                    'readout map: (%s, %d) appears twice' % key)
            self._by_key[key] = ReadoutBin(str(b.hybe), int(b.channel),
                                           int(b.rindex), str(b.dtype).upper(),
                                           int(b.pindex))
        seen = {}
        for b in self._by_key.values():
            if b.pindex == NOT_A_BIN:
                continue
            if b.pindex in seen:
                raise ValueError(
                    'readout map: polymer index %d is claimed by both %s/%d '
                    'and %s/%d' % (b.pindex, seen[b.pindex].hybe,
                                   seen[b.pindex].channel, b.hybe, b.channel))
            seen[b.pindex] = b

    # -- construction ---------------------------------------------------

    @classmethod
    def from_layout(cls, records, channels=None, datatypes=('H', 'R', 'T'),
                    pindex_from='rindex'):
        """Seed from layout records.

        pindex_from: 'rindex' (default) places each bin at
        `rindex - min(rindex)`, so a readout the design skips leaves a
        GAP -- readouts 1-51 and 54-80 give an 80-long polymer whose
        slots 51 and 52 are empty, and every distance map carries two
        NaN stripes where that unimaged stretch is. 'dense' instead packs
        the acquired bins 0..N-1.

        The default is 'rindex' because closing a gap is invisible and
        wrong: two loci three bins apart on the genome would read as
        neighbours, and every distance-against-genomic-separation curve
        would be bent by it. Where the numbering is contiguous the two
        policies agree exactly, so the honest one costs nothing.

        'dense' is for a design whose readout numbers simply are not
        contiguous and were never meant to be -- there the gap is an
        artefact of the numbering, not of the experiment, and only the
        operator knows which it is. That is why pindex is a stored,
        editable field rather than a rule.

        channels: the readout channels the operator activated. Used only
        where the layout did NOT name one -- `record['readouts']` is
        [(None, readout_id)] for every layout written before the
        readoutChannels column existed, and the channel then comes from
        the panel, exactly as it always has.

        A hybe that does not carry a named channel is skipped rather than
        given one it does not have.
        """
        want = [int(c) for c in (channels or [])]
        rows = []
        for r in records or []:
            dtype = str(r.get('datatype') or '').upper()
            if datatypes and dtype not in datatypes:
                continue
            hybe = str(r['folder'])
            have = [int(c) for c in (r.get('channels') or [])]
            pairs = r.get('readouts') or [(None, r.get('readout_id'))]
            for channel, rindex in pairs:
                if rindex is None:
                    continue
                if channel is None:
                    # the layout did not say; the panel's activated
                    # channels stand in. More than one of them is
                    # ambiguous -- both would claim this hybe's single
                    # readout id -- so it is refused rather than guessed.
                    usable = [c for c in want if c in have]
                    if len(usable) > 1:
                        raise ValueError(
                            '%s: the layout names no channel for readout %s, '
                            'so %d activated channels (%s) would all claim '
                            'it. Name them in the layout\'s readoutChannels '
                            'column, or activate one channel.'
                            % (hybe, rindex, len(usable),
                               ', '.join(str(c) for c in usable)))
                    if not usable:
                        continue
                    channel = usable[0]
                elif channel not in have:
                    continue
                rows.append((hybe, int(channel), int(rindex), dtype))

        # pindex: the genomic axis over the H rows, ordered by readout id
        # and then by channel so two readouts of one hybe order stably
        h_rows = sorted((r for r in rows if r[3] == 'H'),
                        key=lambda r: (r[2], r[1], r[0]))
        if pindex_from == 'dense':
            pindex = {(r[0], r[1]): i for i, r in enumerate(h_rows)}
        elif pindex_from == 'rindex':
            base = min((r[2] for r in h_rows), default=0)
            pindex = {(r[0], r[1]): r[2] - base for r in h_rows}
        else:
            raise ValueError("pindex_from must be 'rindex' or 'dense', "
                             "not %r" % (pindex_from,))
        return cls(ReadoutBin(h, c, rid, dt, pindex.get((h, c), NOT_A_BIN))
                   for h, c, rid, dt in rows)

    # -- reading --------------------------------------------------------

    def of(self, hybe, channel):
        """The bin for one (hybe, channel), or None."""
        return self._by_key.get((str(hybe), int(channel)))

    def bins(self):
        """The polymer bins, in pindex order. dtype 'H' only."""
        return sorted((b for b in self._by_key.values()
                       if b.pindex != NOT_A_BIN), key=lambda b: b.pindex)

    def extras(self):
        """Everything that is measured but is not a polymer bin -- the
        R and T rounds -- in (rindex, channel) order."""
        return sorted((b for b in self._by_key.values()
                       if b.pindex == NOT_A_BIN),
                      key=lambda b: (b.rindex, b.channel, b.hybe))

    def n_slots(self):
        """How long the polymer axis is -- max pindex + 1, NOT the number
        of bins. The two differ by exactly the unimaged stretches, and
        allocating `len(bins())` instead is how a gap gets closed up."""
        return max((b.pindex for b in self._by_key.values()
                    if b.pindex != NOT_A_BIN), default=NOT_A_BIN) + 1

    def axis(self):
        """[ReadoutBin or None] of length n_slots -- the polymer axis with
        its holes in place. A None is a slot the design skipped, and the
        position that carries NaN in every trace and every distance map."""
        out = [None] * self.n_slots()
        for b in self._by_key.values():
            if b.pindex != NOT_A_BIN:
                out[b.pindex] = b
        return out

    def gaps(self):
        """The pindex values nothing images -- the NaN stripes."""
        return [i for i, b in enumerate(self.axis()) if b is None]

    def channels(self):
        return sorted({b.channel for b in self._by_key.values()})

    def hybe_channels(self):
        """[(hybe, channel), ...] -- what a trace has to read, pindex
        order first, then the QC rounds."""
        return [(b.hybe, b.channel) for b in self.bins() + self.extras()]

    def __len__(self):
        return len(self._by_key)

    def __iter__(self):
        return iter(self.bins() + self.extras())

    def __eq__(self, other):
        return (isinstance(other, ReadoutMap)
                and self._by_key == other._by_key)

    # -- serialisation ---------------------------------------------------
    #
    # One config attribute, same shape as the panel's checked-hybe list:
    # `hybe|channel|rindex|dtype|pindex`, comma separated.

    def to_text(self):
        return ','.join('%s|%d|%d|%s|%d'
                        % (b.hybe, b.channel, b.rindex, b.dtype, b.pindex)
                        for b in self)

    @classmethod
    def from_text(cls, text):
        bins = []
        for piece in str(text or '').split(','):
            piece = piece.strip()
            if not piece:
                continue
            parts = piece.split('|')
            if len(parts) != 5:
                raise ValueError(
                    'readout map entry %r is not hybe|channel|rindex|dtype|'
                    'pindex' % piece)
            h, c, rid, dt, pi = parts
            bins.append(ReadoutBin(h.strip(), int(c), int(rid),
                                   dt.strip().upper(), int(pi)))
        return cls(bins)

    def rows(self):
        """Editable rows for the panel's table, in display order."""
        return [dict(hybe=b.hybe, channel=b.channel, rindex=b.rindex,
                     dtype=b.dtype, pindex=b.pindex) for b in self]

    @classmethod
    def from_rows(cls, rows):
        return cls(ReadoutBin(r['hybe'], int(r['channel']), int(r['rindex']),
                              str(r['dtype']).upper(), int(r['pindex']))
                   for r in rows or [])
