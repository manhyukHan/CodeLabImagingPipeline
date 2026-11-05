"""
Polymer construction from stored alleles, and the ORCA-style QC on it.

The collapse recipe is the one AnAllele.final_polymer has cited since the
field existed (QualityControlORCA.combineFOV, 'maxBrightness'): per
genomic bin keep the BRIGHTEST candidate among polymer_adj[hybe] -- the
amplitude is index 3 of every candidate tuple -- one (y, x, z) per bin,
NaN where the hybe was rejected or absent. Bin order comes from the
layout's readout_id over datatype 'H' records only; R/T/B rounds are QC
and identity markers, never polymer bins.

Units: positions leave this module in MICROMETRES, scaled per axis by
voxel_um exactly once. z is stored as plane index and pixels are not
planes (0.208 vs 0.2 um) -- the axes scale independently or the maps are
wrong anisotropically.

QC follows QualityControlORCA with its measured bugs fixed:
  - brightness bounds test the AMPLITUDE column only (the original
    compared all four components elementwise, so any coordinate below
    the 5th-percentile brightness was falsely flagged);
  - thresholds are quantile-derived from the data at hand
    (polymeric_qc), never inherited constants -- the repo rule.
"""
import collections
import os

import numpy as np

from codelab_pipeline.analysis import readout_map
from codelab_pipeline.io import analysis_store
from codelab_pipeline.io import paths as store_paths
from codelab_pipeline.io import preprocess

DEFAULT_VOXEL_UM = (0.208, 0.208, 0.2)


def records_for(storage_path, modality=None):
    """Parse the experiment layout the STORE's own manifest names.

    The headless entry point: give the storage_path (the modality
    directory, e.g. <root>/DNA), get the parsed layout records -- no app,
    no config file. modality defaults to the storage_path's basename.
    Raises with the manifest content when the layout is not recorded or
    the file has moved (the layout lives on the acquisition share, which
    is not always mounted -- say so rather than guessing).
    """
    root = os.path.dirname(os.path.abspath(storage_path))
    manifest = store_paths.read_manifest(root)
    name = modality or os.path.basename(os.path.normpath(storage_path))
    entry = (manifest.get('modalities') or {}).get(name) or {}
    layout = entry.get('layout_path')
    if not layout:
        raise ValueError(f'manifest at {root} records no layout_path for '
                         f'modality {name!r}: {manifest!r}')
    if not os.path.exists(layout):
        raise FileNotFoundError(
            f'layout {layout} (from the manifest) is not reachable -- the '
            f'acquisition share may not be mounted on this machine')
    return preprocess.parse_experiment_layout(layout)


#: the modalities whose readout number IS a genomic coordinate, so that
#: the layout's numbering can seed the polymer axis directly. Measured
#: over the 16 layouts this project can reach: every DNA layout is
#: contiguous (MP58 15-76, chr19 1-91, HEK 5-73, HoxA 6-42 and 10-37), so
#: readout-id and dense indexing agree exactly there. An RNA layout
#: numbers its GENES -- 101-400 for 15 acquired readouts -- and three of
#: them leave Rep_/Toe_ rounds at datatype H, so the same readout id
#: arrives twice. Neither is a polymer, and neither should be forced onto
#: a genomic axis by a rule meant for one.
POLYMER_MODALITIES = ('DNA',)


def readout_map_for(records, channels=None, modality=None, declared=None,
                    pindex_from=None):
    """The ReadoutMap these layout records seed.

    declared: the DESIGNED readout range ('1-80'), settled in the
    chromatin tracing panel. It decides the axis outright -- where bin 0
    sits and how many slots trail the last acquired bin -- because the
    acquired readouts cannot say whether a number they are missing was
    skipped by design, not yet acquired, or never contiguous.

    Without one, the LAYOUT seeds it: pindex = rindex for datatype H of a
    polymer modality (POLYMER_MODALITIES), which is what the numbering
    means there, and dense packing anywhere else. modality=None is read
    as "not stated", and takes the dense reading -- the conservative one,
    since it is exactly what every analysis here did before the axis had
    a policy at all.

    channels=None keeps the channel-agnostic axis every analysis here has
    always read: an unnamed readout becomes (hybe, ANY_CHANNEL). Pass the
    activated channels to get the real wavelengths, and pass a settled
    map directly wherever the operator has corrected one.
    """
    if pindex_from is None:
        pindex_from = ('rindex' if str(modality or '') in POLYMER_MODALITIES
                       else 'dense')
    return readout_map.ReadoutMap.from_layout(
        records, channels=channels, pindex_from=pindex_from,
        declared=declared)


def _axis_of(records=None, channels=None, rmap=None, modality=None,
             declared=None):
    """(ReadoutMap, [ReadoutBin or None] per polymer slot)."""
    if rmap is None:
        if records is None:
            raise ValueError('give either layout records or a readout map')
        rmap = readout_map_for(records, channels=channels, modality=modality,
                               declared=declared)
    return rmap, rmap.axis()


def genomic_bins(records=None, channels=None, rmap=None, modality=None,
                 declared=None):
    """[(readout_id, hybe_folder)] -- ONE ENTRY PER POLYMER SLOT.

    The layout is authoritative for bin identity and order; readout_id is
    the genomic-locus axis every distance map shares.

    A slot the design never imaged -- readouts 1-51 and 54-80, say --
    comes back as (readout_id, '') and is where every polymer carries
    NaN. The readout id is still named there, taken from the axis base
    (see ReadoutMap.rindex_base), because the skipped locus has an
    identity; where the base is unknowable it is -1. The empty hybe is
    what makes the slot unmatchable: no allele has data under '', so the
    collapse leaves it NaN without a special case.

    Use bin_keys() to INDEX an allele's stored data and bin_hybes() for
    labels. Under a multi-channel design one hybe occupies two slots, so
    the hybe alone stops being a unique key -- the pair is.
    """
    rmap, axis = _axis_of(records, channels, rmap, modality, declared)
    base = rmap.rindex_base()
    return [(b.rindex, b.hybe) if b is not None
            else ((base + j) if base is not None else -1, '')
            for j, b in enumerate(axis)]


def bin_keys(records=None, channels=None, rmap=None, modality=None,
             declared=None):
    """[(hybe, channel) or None] -- one per polymer slot, the key an
    allele's polymer_adj / fiducial_trace_adj is read under.

    None at a slot nothing imaged. collapse_polymer takes this list.
    """
    _rmap, axis = _axis_of(records, channels, rmap, modality, declared)
    return [None if b is None else (b.hybe, b.channel) for b in axis]


def bin_hybes(records=None, channels=None, rmap=None, modality=None,
              declared=None):
    """[hybe] -- one per polymer slot, '' at a slot nothing imaged.

    LABELS, not keys: a multi-channel round puts the same hybe in two
    slots, so this list can repeat. Index stored data with bin_keys().
    """
    return [h for _rid, h in
            genomic_bins(records, channels, rmap, modality, declared)]


def slot_index(bin_hybes, hybe, bin_channels=None, channel=None):
    """Which polymer slot a hybe names. Raises KeyError when none does.

    Refuses rather than taking the first when the hybe names SEVERAL
    slots. A multi-channel round reads two loci in one hybe, so the
    folder name alone stops identifying a locus there, and answering
    with the lower-numbered readout would answer about a different
    stretch of the genome without saying so. Pass `channel` to pick.
    """
    hits = [j for j, h in enumerate(bin_hybes) if str(h) == str(hybe)]
    if channel is not None and bin_channels is not None:
        hits = [j for j in hits if int(bin_channels[j]) == int(channel)]
    if not hits:
        raise KeyError(hybe)
    if len(hits) > 1:
        chans = [(bin_channels[j] if bin_channels is not None else '?')
                 for j in hits]
        raise ValueError(
            f'{hybe!r} names {len(hits)} polymer bins (channels {chans}) '
            f'-- name the channel, the hybe alone does not say which '
            f'locus is meant')
    return hits[0]


def qc_rounds(records):
    """{'repeats': [(readout_id, hybe)], 'toes': [...]} -- the R and T
    rounds, readout_id-ordered. R re-images an H bin (same readout_id);
    T carries identity markers. Neither is ever a polymer bin."""
    out = {'repeats': [], 'toes': []}
    for r in records:
        dt = str(r.get('datatype', '')).upper()
        if dt == 'R':
            out['repeats'].append((int(r['readout_id']), str(r['folder'])))
        elif dt == 'T':
            out['toes'].append((int(r['readout_id']), str(r['folder'])))
    out['repeats'].sort()
    out['toes'].sort()
    return out


def max_brightness(candidates):
    """The default Selector: the brightest candidate, whole.

    A Selector is any callable
        selector([(y, x, z, amplitude), ...]) -> (y, x, z, amplitude)
    called only on NON-EMPTY candidate lists. ORCA gated this choice
    behind a 'barcode_selection' string and never implemented a second
    rule; here the rule is an ordinary function argument, so swapping it
    (nearest-to-anchor, amplitude-weighted centroid, ...) is a one-line
    call-site change and never a code fork. A composite selector may
    return a synthetic amplitude (e.g. the sum); the QC brightness gates
    then act on whatever the selector reported.
    """
    return max(candidates, key=lambda c: c[3])


def bin_lookup_keys(keys):
    """[(key, ...) per slot] -- which keys to try, in order, when reading
    one polymer slot out of a stored {key: value} dict.

    A slot arrives as (hybe, channel) from bin_keys(), as a bare hybe
    from every caller written before channels existed, or as None for a
    slot nothing imaged (which tries nothing and stays NaN).

    A (hybe, channel) slot also tries the BARE HYBE, because every store
    written before the channel joined the key holds {hybe: [...]} and
    that data is this slot's data. The fallback is offered only where the
    hybe occupies ONE slot: when a multi-channel design puts it in two,
    the bare key cannot say which it is, and giving the same candidates
    to both would copy one measurement into a locus never read there.
    """
    once = collections.Counter(k[0] if isinstance(k, tuple) else k
                               for k in keys if k is not None)
    out = []
    for k in keys:
        if k is None:
            out.append(())
        elif isinstance(k, tuple):
            h, c = str(k[0]), int(k[1])
            out.append(((h, c), h) if once[h] == 1 else ((h, c),))
        else:
            out.append((str(k),))
    return out


def split_key(key, channel=None):
    """(hybe, channel) from a STORAGE key.

    A stored key is the bare hybe while its round has one readout -- that
    is what keeps every store written so far readable -- and
    (hybe, channel) once the round has two. So a key is not a hybe, and
    anything BELOW the trace that needs one (a crop, a stack read, a
    FrameResolver) cannot take it as it stands: handed a tuple it would
    look up a frame under the text "('Hyb_001', 635)" and answer wrongly
    or not at all.

    This is the one place that turns a key back into the pair. `channel`
    stands in where the key names none, so a caller that knows which
    wavelength it configured can pass it and get a complete answer
    rather than a half one.
    """
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return str(key[0]), int(key[1])
    return str(key), (None if channel is None else int(channel))


def key_hybe(key):
    """Just the hybe of a storage key -- for a frame, which is per ROUND.

    A FrameResolver maps a whole round, not one of its readouts: both
    bins of a two-readout hybe share its matrices. So the channel is not
    dropped here, it was never part of the question.
    """
    return split_key(key)[0]


def lookup_bin(d, tries):
    """(key_that_hit, value) for one slot, or (None, None).

    PRESENCE decides, not truth: a fiducial key that is present and None
    records a fit that failed, which a missing key does not.
    """
    d = d or {}
    for t in tries:
        if t in d:
            return t, d[t]
    return None, None


def _refuse_channelled_store(d, keys, what):
    """Raise when the stored dict is keyed by (hybe, channel) and the
    caller's axis is not.

    Otherwise every lookup misses and the allele comes back as an empty
    polymer -- a multi-channel store read through a single-channel axis
    would report 'nothing traced' rather than 'you need bin_keys()'.
    """
    bare = {k for k in keys if k is not None and not isinstance(k, tuple)}
    if not bare:
        return
    clash = sorted({k[0] for k in (d or {}) if isinstance(k, tuple)
                    and k[0] in bare})
    if clash:
        raise ValueError(
            f'{what} is keyed by (hybe, channel) for {clash[:4]} but the '
            f'polymer axis names bare hybes -- pass polymer.bin_keys(...) '
            f'so each slot reads its own channel')


def collapse_polymer(allele_dict, keys, selector=max_brightness):
    """One allele's polymer_adj -> (pos (n_bins, 3) yxz, amp (n_bins,),
    n_cand (n_bins,) int). NaN / 0 where the bin has no candidate.

    `keys` is one entry per polymer slot: (hybe, channel) from
    bin_keys(), a bare hybe, or None for a slot nothing imaged. A None
    slot is NaN by construction -- that is the NaN stripe of a readout
    the design skipped, and it needs no branch of its own.

    `selector` picks ONE position per multi-candidate bin (sister
    chromatids); see max_brightness for the contract. Positions stay in
    px/planes here; the caller scales to um once, so this stays
    unit-agnostic and testable.
    """
    n = len(keys)
    pos = np.full((n, 3), np.nan)
    amp = np.full(n, np.nan)
    n_cand = np.zeros(n, dtype=np.int32)
    poly = allele_dict.get('polymer_adj') or {}
    _refuse_channelled_store(poly, keys, 'polymer_adj')
    for j, tries in enumerate(bin_lookup_keys(keys)):
        _k, cands = lookup_bin(poly, tries)
        if not cands:
            continue
        n_cand[j] = len(cands)
        y, x, z, a = selector(cands)[:4]
        pos[j] = (y, x, z)
        amp[j] = a
    return pos, amp, n_cand


def fov_polymer_table(storage_path, fov, keys, voxel_um=DEFAULT_VOXEL_UM,
                      selector=max_brightness):
    """Every stored allele of one FOV as stacked arrays, um-scaled.

    keys: one entry per polymer slot, from bin_keys() -- (hybe, channel),
    a bare hybe, or None for a slot nothing imaged. A None slot is an
    all-NaN column across every allele of the FOV.

    Returns a plain dict (picklable, pmap-friendly):
      pos_um    (n_alleles, n_bins, 3) float  y, x, z in MICROMETRES
      amp       (n_alleles, n_bins)           brightest-candidate amplitude
      n_cand    (n_alleles, n_bins) int       candidates per bin (sister
                                              chromatids show up here)
      allele_id, cell, fov (n_alleles,) int
      celltype  list[str]                     joined via this FOV's cells;
                                              '' where the cell is unknown
                                              or the allele is homeless
      n_traced  (n_alleles,) int              finite bins per allele

    Alleles with zero traced bins are KEPT as all-NaN rows -- dropping is
    a QC decision (apply_qc), not an extraction side effect.
    """
    dicts = analysis_store.read_fov_alleles(storage_path, fov)
    cells, _ = analysis_store.read_cells(storage_path, fov)
    ct_by_id = {int(c['id']): str(c.get('celltype') or '')
                for c in (cells or [])}
    n_a, n_b = len(dicts), len(keys)
    pos = np.full((n_a, n_b, 3), np.nan)
    amp = np.full((n_a, n_b), np.nan)
    n_cand = np.zeros((n_a, n_b), dtype=np.int32)
    aid = np.zeros(n_a, dtype=np.int64)
    cid = np.full(n_a, -1, dtype=np.int64)
    celltype = []
    for i, d in enumerate(dicts):
        p, a, k = collapse_polymer(d, keys, selector=selector)
        pos[i], amp[i], n_cand[i] = p, a, k
        aid[i] = int(d['id'])
        cid[i] = int(d.get('cell', -1))
        celltype.append(ct_by_id.get(cid[i], ''))
    dy, dx, dz = (float(v) for v in voxel_um)
    pos *= np.array([dy, dx, dz])[None, None, :]
    return {'pos_um': pos, 'amp': amp, 'n_cand': n_cand,
            'allele_id': aid, 'cell': cid,
            'fov': np.full(n_a, int(fov), dtype=np.int64),
            'celltype': celltype,
            'n_traced': np.isfinite(pos[:, :, 0]).sum(1).astype(np.int32)}


DIMS_CHOICES = ('xyz', 'xy')


def _dim_axes(dims):
    """Which columns of a (..., 3) y/x/z position array to use.

    'xy' drops z -- an IN-PLANE distance, for data whose axial component
    is not trustworthy (see analysis/anisotropy.py). Positions are stored
    (y, x, z), so the in-plane pair is columns 0 and 1.
    """
    d = str(dims or 'xyz').lower()
    if d not in DIMS_CHOICES:
        raise ValueError(f'dims must be one of {DIMS_CHOICES}, got {dims!r}')
    return [0, 1] if d == 'xy' else [0, 1, 2]


def polymer_distmaps(pos_um, dims='xyz'):
    """(n_alleles, n_bins, n_bins) pairwise Euclidean um distances.

    dims='xy' measures IN-PLANE distance only, dropping z. That is a
    different quantity, not a cleaner version of the same one: it is
    smaller by sqrt(2/3) even when the data are perfectly isotropic, so a
    map, a gate bound or a threshold from one dimensionality must never
    be compared against the other. Callers that persist or cache a result
    are responsible for recording which one it is.

    NaN positions propagate to NaN rows/columns, exactly the ORCA kappa
    behaviour. Vectorized over alleles; ~25 MB for 127 alleles x 90 bins.
    """
    axes = _dim_axes(dims)
    pos = np.asarray(pos_um, np.float32)[..., axes]
    n_a, n_b = pos.shape[0], pos.shape[1]
    out = np.empty((n_a, n_b, n_b), np.float32)
    # float32 and CHUNKED: cells run to tens of thousands, so alleles do
    # too. At 24k alleles x 62 bins the float64 all-at-once broadcast
    # peaked at ~1.1 GB of scratch for a 738 MB result; float32 halves
    # both and um distances do not need 15 significant digits. Measured:
    # 5.3 s / 738 MB -> chunked float32 keeps peak scratch ~50 MB.
    step = max(1, int(2e7 // max(n_b * n_b * 3, 1)))
    for i0 in range(0, n_a, step):
        blk = pos[i0:i0 + step]
        diff = blk[:, :, None, :] - blk[:, None, :, :]
        out[i0:i0 + step] = np.sqrt((diff ** 2).sum(-1))
    return out


# -- ORCA QC ---------------------------------------------------------------

def qc_thresholds(table, dmaps, dims='xyz',
                  brightness_q=(0.05, 0.95), jump_q=0.75, max_dist_q=0.95):
    """Quantile-derived thresholds, polymeric_qc-style, from THIS data.

    Returns {'min_brightness', 'max_brightness', 'max_jump_um',
    'max_dist_um'}. The jump quantile runs over finite genomically-
    adjacent distances; max_dist over per-bin median distances. Derived,
    not inherited: the repo has been burned by every constant it ever
    ported across datasets.
    """
    amp = table['amp']
    finite_amp = amp[np.isfinite(amp)]
    neighbors = np.concatenate([np.diagonal(d, 1) for d in dmaps]) \
        if len(dmaps) else np.array([])
    neighbors = neighbors[np.isfinite(neighbors)]
    med_per_bin = np.nanmedian(dmaps, axis=1) if len(dmaps) else np.empty((0, 0))
    med = med_per_bin[np.isfinite(med_per_bin)]
    q = lambda v, p, fallback: float(np.quantile(v, p)) if len(v) else fallback
    return {
        'min_brightness': q(finite_amp, brightness_q[0], 0.0),
        'max_brightness': q(finite_amp, brightness_q[1], np.inf),
        'max_jump_um': q(neighbors, jump_q, np.inf),
        'max_dist_um': q(med, max_dist_q, np.inf),
        # STAMPED with the dimensionality of the maps they were fitted
        # to. max_jump_um and max_dist_um are lengths, and an in-plane
        # length is ~0.82x its 3D counterpart under perfect isotropy --
        # so a threshold reused in the other dimensionality is not merely
        # imprecise, it is measuring a different quantity. The stamp lets
        # every consumer (the GUI fields, the config, apply_qc) check
        # rather than assume.
        'dims': str(dims or 'xyz').lower(),
    }


def apply_qc(table, dmaps, thresholds, min_traced=2, dims='xyz'):
    """ORCA filtering with the elementwise-brightness bug fixed.

    Marks a bin bad when: amplitude outside [min, max] brightness (the
    AMPLITUDE only); both genomic neighbors further than max_jump_um
    (interior bins); its median distance to all bins exceeds max_dist_um.
    Bad bins become NaN; alleles keep membership only with >= min_traced
    finite bins (ORCA used > 1).

    Returns dict: pos_um/amp (filtered COPIES), bads (n_a, n_bins) bool,
    kept (n_a,) bool over the INPUT rows, plus the recomputed dmaps of
    the kept alleles and their index into the input.
    """
    pos = table['pos_um'].copy()
    amp = table['amp'].copy()
    n_a, n_b = amp.shape
    if n_a == 0:
        # zero alleles is a legitimate gate outcome, not an error: the
        # jump gate's diagonal stack collapses to 1-D on empty input and
        # crashed here (IndexError) before this guard.
        return {'pos_um': pos, 'amp': amp,
                'bads': np.zeros((0, n_b), bool), 'kept': np.zeros(0, bool),
                'index': np.zeros(0, np.int64),
                'dmaps': np.empty((0, n_b, n_b), np.float32)}
    bads = np.zeros((n_a, n_b), dtype=bool)
    bads |= np.isfinite(amp) & (amp > thresholds['max_brightness'])
    bads |= np.isfinite(amp) & (amp < thresholds['min_brightness'])
    if n_b >= 3:
        d1 = np.array([np.diagonal(d, 1) for d in dmaps])   # (n_a, n_b-1)
        far = d1 > thresholds['max_jump_um']
        bads[:, 1:n_b - 1] |= far[:, 1:] & far[:, :-1]
    med = np.nanmedian(dmaps, axis=1)                       # (n_a, n_b)
    with np.errstate(invalid='ignore'):
        bads |= np.isfinite(med) & (med > thresholds['max_dist_um'])
    pos[bads] = np.nan
    amp[bads] = np.nan
    kept = np.isfinite(pos[:, :, 0]).sum(1) >= int(min_traced)
    out_pos = pos[kept]
    return {'pos_um': out_pos, 'amp': amp[kept], 'bads': bads,
            'kept': kept, 'index': np.flatnonzero(kept),
            # THE SAME dimensionality the caller gated in. This recompute
            # feeds every downstream map view, so a hard-coded 3D here
            # would gate bins in one metric and display another -- and
            # Apply QC is on by default, so that would be the DEFAULT
            # path, silently, with nothing on any figure saying so.
            'dims': str(dims or 'xyz').lower(),
            'dmaps': polymer_distmaps(out_pos, dims=dims)}


def efficacy(pos_um):
    """(n_bins,) fraction of alleles with a finite position per bin."""
    if len(pos_um) == 0:
        return np.zeros(pos_um.shape[1])
    return np.isfinite(pos_um[:, :, 0]).mean(0)


def completeness(pos_um):
    """(n_alleles,) count of finite bins per allele."""
    return np.isfinite(pos_um[:, :, 0]).sum(1)
