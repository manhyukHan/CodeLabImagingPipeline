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

#: the channel of a bin whose channel nobody named -- neither the layout
#: nor the caller. Every layout written before the readoutChannels column
#: is in this state, and so is every analysis that reads a polymer axis
#: without caring which wavelength made it (polymer.genomic_bins with no
#: channels). It is NOT a wavelength and never matches one: a lookup for
#: 635 on such a map misses, which is the honest answer, because the map
#: does not know that the bin was 635.
ANY_CHANNEL = 0


def parse_range(text):
    """'1-80' -> (1, 80). None / '' -> None.

    The DESIGNED readout range: which loci the experiment set out to
    read, as against which it got. A single number means a range of one.
    """
    s = str(text or '').strip()
    if not s:
        return None
    parts = s.replace('–', '-').split('-')
    try:
        if len(parts) == 1:
            lo = hi = int(parts[0])
        elif len(parts) == 2:
            lo, hi = int(parts[0]), int(parts[1])
        else:
            raise ValueError
    except ValueError:
        raise ValueError('readout range %r is not N or N-M' % (text,))
    if hi < lo:
        raise ValueError('readout range %r runs backwards' % (text,))
    return (lo, hi)


class ReadoutMap(object):
    """An ordered table of ReadoutBin, keyed by (hybe, channel).

    `slots` is how long the polymer axis is -- the DESIGNED length, which
    is not the same as the number of bins acquired. Left out, the axis is
    exactly as long as the bins reach.
    """

    def __init__(self, bins=(), slots=None):
        self._slots = None if slots is None else int(slots)
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
        reach = max(seen) + 1 if seen else 0
        if self._slots is not None and self._slots < reach:
            raise ValueError(
                'readout map: %d declared slots cannot hold a bin at '
                'polymer index %d' % (self._slots, reach - 1))

    # -- construction ---------------------------------------------------

    @classmethod
    def from_layout(cls, records, channels=None, datatypes=('H', 'R', 'T'),
                    pindex_from='rindex', declared=None):
        """Seed from layout records.

        declared: the DESIGNED readout range, '1-80' or (1, 80) -- which
        loci the experiment set out to read, as against which it got. It
        is settled in the chromatin tracing panel, not in the layout,
        because only the operator knows it: an acquired set of readouts
        cannot say whether the numbers it is missing were skipped by
        design, not acquired yet, or never meant to be contiguous. Given
        one, pindex is `rindex - lo` and the axis is hi - lo + 1 long, so
        a declared readout nothing imaged is an empty slot and the NaN
        stripe of every distance map. A bin outside the range is refused.

        pindex_from, when no range is declared: 'rindex' places each bin
        at `rindex - min(rindex)`, which is the same thing with the range
        taken to be exactly the acquired span -- gaps inside it show,
        nothing is added at either end. 'dense' instead packs the
        acquired bins 0..N-1 and reports no gaps at all.

        Neither is a safe blanket default, which is why polymer.py
        chooses per modality rather than here: measured over the 16
        layouts this project can reach, the two agree exactly on every
        DNA layout (MP58 15-76, chr19 1-91, HEK 5-73, HoxA 6-42 and
        10-37 -- no gaps either way), and differ only where the readout
        number is not a genomic coordinate at all. An RNA panel numbers
        its genes 101-400, so 'rindex' turns 15 acquired readouts into a
        300-slot axis with 285 holes; a partially ingested DNA store
        spans 2-115 with 10 rounds in, where the holes are rounds not
        yet acquired rather than loci skipped by design.

        channels: the readout channels the operator activated. Used only
        where the layout did NOT name one -- `record['readouts']` is
        [(None, readout_id)] for every layout written before the
        readoutChannels column existed, and the channel then comes from
        the panel, exactly as it always has.

        channels=None means the caller does not care which wavelength
        made a bin, and an unnamed readout then gets ANY_CHANNEL rather
        than being dropped. That is the channel-agnostic polymer axis
        every analysis has read so far, and it is a different thing from
        channels=[]: an empty activation names no usable channel, so a
        layout that does not name one either yields no bins at all.

        A hybe that does not carry a named channel is skipped rather than
        given one it does not have.
        """
        any_channel = channels is None
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
                if channel is None and any_channel:
                    channel = ANY_CHANNEL
                elif channel is None:
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
                elif not any_channel and int(channel) not in want:
                    # THE ACTIVATION IS THE FILTER, both ways round. A
                    # channel the layout names is still only traced if
                    # the operator activated it -- otherwise activating
                    # 635 alone would quietly put every 747 bin on the
                    # axis as well, and the panel's staged choice would
                    # describe the run without governing it. channels
                    # =None is the separate case of a caller that wants
                    # whatever the layout names.
                    continue
                rows.append((hybe, int(channel), int(rindex), dtype))

        # pindex: the genomic axis over the H rows, ordered by readout id
        # and then by channel so two readouts of one hybe order stably
        h_rows = sorted((r for r in rows if r[3] == 'H'),
                        key=lambda r: (r[2], r[1], r[0]))
        want_range = parse_range(declared) if not isinstance(
            declared, (tuple, list)) else (int(declared[0]), int(declared[1]))
        slots = None
        if want_range is not None:
            lo, hi = want_range
            out = sorted({r[2] for r in h_rows if not lo <= r[2] <= hi})
            if out:
                raise ValueError(
                    'declared readout range %d-%d does not contain acquired '
                    'readout(s) %s -- widen the range or mark those rounds '
                    'R/T' % (lo, hi, out[:8]))
            pindex = {(r[0], r[1]): r[2] - lo for r in h_rows}
            slots = hi - lo + 1
        elif pindex_from == 'dense':
            pindex = {(r[0], r[1]): i for i, r in enumerate(h_rows)}
        elif pindex_from == 'rindex':
            # A readout id must identify ONE locus for this policy to
            # mean anything. Several real layouts leave the DataType
            # column at 'H' for the Rep_/Toe_ rounds too, so the same
            # readout arrives two or three times under different folder
            # names -- not two loci, one locus re-imaged. Name them
            # rather than letting it surface as a pindex collision.
            seen = {}
            for r in h_rows:
                seen.setdefault(r[2], []).append(r[0])
            dups = {rid: fs for rid, fs in seen.items() if len(fs) > 1}
            if dups:
                some = sorted(dups.items())[:3]
                raise ValueError(
                    'readout-id indexed polymer: readout %s imaged by %s '
                    'under datatype H -- one readout is one locus, so a '
                    'repeat or toe round must carry DataType R or T, or '
                    'this axis must be built with pindex_from=\'dense\''
                    % (', '.join(str(r) for r, _ in some),
                       ', '.join('/'.join(f) for _, f in some)))
            base = min((r[2] for r in h_rows), default=0)
            pindex = {(r[0], r[1]): r[2] - base for r in h_rows}
        else:
            raise ValueError("pindex_from must be 'rindex' or 'dense', "
                             "not %r" % (pindex_from,))
        return cls((ReadoutBin(h, c, rid, dt, pindex.get((h, c), NOT_A_BIN))
                    for h, c, rid, dt in rows), slots=slots)

    @classmethod
    def seed_rows(cls, records, channels=None, datatypes=('H', 'R', 'T')):
        """Editable rows for the panel's table -- rindex may be None.

        from_layout REFUSES the case this exists for: the layout names
        one readout id per round and the operator activated two
        channels, so both would claim it. That refusal is right -- the
        second channel's readout id is a design fact, and neither
        continuing the numbering nor calling it a repeat recovers it --
        but refusing leaves nothing to EDIT, and the operator is the one
        who knows the answer.

        So this builds the rows instead of the map. A bin the layout
        names outright gets its readout id. Where the layout names no
        channel at all, the LOWEST activated channel the hybe carries
        takes the round's id -- it is the one the single-channel default
        would have used -- and every other channel gets rindex None,
        meaning "the layout does not say": the blank the operator fills.

        pindex is left None too; it follows from rindex and the declared
        range, which the table derives rather than stores by hand.
        """
        want = None if channels is None else [int(c) for c in channels]
        rows = []
        for r in records or []:
            dtype = str(r.get('datatype') or '').upper()
            if datatypes and dtype not in datatypes:
                continue
            hybe = str(r['folder'])
            have = [int(c) for c in (r.get('channels') or [])]
            pairs = r.get('readouts') or [(None, r.get('readout_id'))]
            named = {int(c): int(rid) for c, rid in pairs
                     if c is not None and rid is not None}
            unnamed = next((rid for c, rid in pairs
                            if c is None and rid is not None), None)
            usable = ([c for c in have if c in want] if want is not None
                      else sorted(named) or have)
            # the lowest activated channel inherits the round's own id;
            # see the docstring for why only one of them can
            first = min(usable) if usable else None
            for channel in sorted(usable):
                rindex = named.get(channel)
                if rindex is None and channel == first:
                    rindex = unnamed
                rows.append({'hybe': hybe, 'channel': int(channel),
                             'rindex': None if rindex is None else int(rindex),
                             'dtype': dtype, 'pindex': None})
        rows.sort(key=lambda d: (d['rindex'] is None,
                                 d['rindex'] if d['rindex'] is not None else 0,
                                 d['channel'], d['hybe']))
        return rows

    @staticmethod
    def place_rows(rows, declared=None):
        """Fill each row's pindex from its rindex, and say how long the
        axis is: (rows, n_slots, [complaints]).

        H rows only -- R/T are measured but are not polymer bins, so
        they take NOT_A_BIN. A row with no rindex cannot be placed and
        is reported rather than dropped: it is the blank the operator
        still has to fill, and a map built without it would quietly be
        one bin short.
        """
        out = [dict(r) for r in rows or []]
        rng = (parse_range(declared)
               if not isinstance(declared, (tuple, list))
               else (int(declared[0]), int(declared[1])))
        h = [r for r in out if str(r.get('dtype', '')).upper() == 'H']
        blanks = [r for r in h if r.get('rindex') is None]
        ids = [int(r['rindex']) for r in h if r.get('rindex') is not None]
        lo = rng[0] if rng else (min(ids) if ids else 0)
        hi = rng[1] if rng else (max(ids) if ids else -1)
        complaints = []
        for r in out:
            if str(r.get('dtype', '')).upper() != 'H' or r.get('rindex') is None:
                r['pindex'] = NOT_A_BIN
                continue
            rid = int(r['rindex'])
            if rng and not lo <= rid <= hi:
                complaints.append('%s/%s: readout %d is outside the declared '
                                  'range %d-%d' % (r['hybe'], r['channel'],
                                                   rid, lo, hi))
                r['pindex'] = NOT_A_BIN
                continue
            r['pindex'] = rid - lo
        for r in blanks:
            complaints.append('%s/%s: no readout number -- the layout does '
                              'not name one for this channel'
                              % (r['hybe'], r['channel']))
        seen = {}
        for r in out:
            if r['pindex'] == NOT_A_BIN:
                continue
            if r['pindex'] in seen:
                complaints.append(
                    '%s/%s and %s/%s both sit at polymer index %d'
                    % (seen[r['pindex']]['hybe'], seen[r['pindex']]['channel'],
                       r['hybe'], r['channel'], r['pindex']))
            seen[r['pindex']] = r
        n_slots = (hi - lo + 1) if hi >= lo else 0
        return out, n_slots, complaints

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
        """How long the polymer axis is -- the DECLARED length where a
        range was declared, else max pindex + 1. NOT the number of bins:
        the two differ by exactly the unimaged stretches, and allocating
        `len(bins())` instead is how a gap gets closed up.

        A declared range can run past the last bin acquired, which is the
        case a derived length cannot express -- a design of 80 readouts
        with the last four not yet in is an 80-long axis, not a 76-long
        one that will silently grow by four later.
        """
        if self._slots is not None:
            return self._slots
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

    def rindex_base(self):
        """The readout id sitting at pindex 0, or None.

        On a readout-id indexed axis every bin satisfies
        rindex - pindex == base, so even a SLOT NOTHING IMAGED has an
        identity: base + pindex. That is the readout the design skipped,
        and naming it is how an empty slot stays legible instead of
        reading as a bin that failed.

        None where the offsets disagree -- a densely packed or
        hand-edited map. There an empty slot really has no identity to
        report, and inventing one would be a guess.
        """
        offs = {b.rindex - b.pindex for b in self._by_key.values()
                if b.pindex != NOT_A_BIN}
        return offs.pop() if len(offs) == 1 else None

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
        # the declared length is part of the map: two maps with the same
        # bins but different trailing slots describe different polymers
        return (isinstance(other, ReadoutMap)
                and self._by_key == other._by_key
                and self.n_slots() == other.n_slots())

    # -- serialisation ---------------------------------------------------
    #
    # One config attribute, same shape as the panel's checked-hybe list:
    # `hybe|channel|rindex|dtype|pindex`, comma separated.

    def to_text(self):
        """One config attribute.

        A declared axis length rides in front as `slots=<n>`, because it
        is the one thing the bins cannot say for themselves: trailing
        slots nothing has imaged yet leave no bin to derive them from.
        Text written before the prefix existed reads back unchanged, with
        the length derived as it was.
        """
        head = ('slots=%d,' % self._slots) if self._slots is not None else ''
        return head + ','.join(
            '%s|%d|%d|%s|%d'
            % (b.hybe, b.channel, b.rindex, b.dtype, b.pindex)
            for b in self)

    @classmethod
    def from_text(cls, text):
        bins, slots = [], None
        for piece in str(text or '').split(','):
            piece = piece.strip()
            if not piece:
                continue
            if piece.lower().startswith('slots='):
                slots = int(piece.split('=', 1)[1])
                continue
            parts = piece.split('|')
            if len(parts) != 5:
                raise ValueError(
                    'readout map entry %r is not hybe|channel|rindex|dtype|'
                    'pindex' % piece)
            h, c, rid, dt, pi = parts
            bins.append(ReadoutBin(h.strip(), int(c), int(rid),
                                   dt.strip().upper(), int(pi)))
        return cls(bins, slots=slots)

    def rows(self):
        """Editable rows for the panel's table, in display order."""
        return [dict(hybe=b.hybe, channel=b.channel, rindex=b.rindex,
                     dtype=b.dtype, pindex=b.pindex) for b in self]

    @classmethod
    def from_rows(cls, rows, slots=None):
        return cls((ReadoutBin(r['hybe'], int(r['channel']), int(r['rindex']),
                               str(r['dtype']).upper(), int(r['pindex']))
                    for r in rows or []), slots=slots)

    def declared_slots(self):
        """The declared axis length, or None where it is derived."""
        return self._slots

    def with_range(self, declared):
        """The same bins on a DECLARED axis -- what the panel settles.

        pindex is re-placed as `rindex - lo`, so the range decides both
        where bin 0 sits and how many empty slots trail the last one. The
        bins themselves are untouched; only their position on the axis
        is, which is the one thing the operator is being asked.
        """
        rng = (parse_range(declared)
               if not isinstance(declared, (tuple, list))
               else (int(declared[0]), int(declared[1])))
        if rng is None:
            return ReadoutMap(self, slots=None)
        lo, hi = rng
        out = sorted({b.rindex for b in self.bins() if not lo <= b.rindex <= hi})
        if out:
            raise ValueError(
                'declared readout range %d-%d does not contain acquired '
                'readout(s) %s' % (lo, hi, out[:8]))
        moved = [b._replace(pindex=b.rindex - lo) if b.pindex != NOT_A_BIN
                 else b for b in self]
        return ReadoutMap(moved, slots=hi - lo + 1)
