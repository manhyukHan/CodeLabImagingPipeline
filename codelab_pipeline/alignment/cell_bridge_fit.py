# -*- coding: utf-8 -*-
"""Fitting the per-cell cross-modal term, for one FOV.

The layer this produces sits between the per-FOV cross-modal bridge and
the per-cell same-modality residual. What it stores, per cell, is a
DEVIATION FROM ITS FOV'S MEDIAN -- not an absolute displacement.

That choice is the conclusion of a comparison against an independent
implementation (half-maximum masks, 3D IoU) on the Lamin store, 233
cells shared:

    the per-cell PATTERN reproduces      every FOV positive, r +0.45 to
                                         +0.87, pooled +0.69
    the per-cell VALUE is coarse         difference IQR 0.20 um against
                                         a term whose own spread is
                                         0.25-0.28 um IQR
    the FOV-level ZERO does not agree    -0.22 to +0.07 um between the
                                         two, and up to 0.29 um between
                                         either and the stored bridge

The FOV-level disagreement is estimator-specific bias: a 14 um cell box
read by cross-correlation is a far worse instrument for a FOV-wide
offset than the fiducial-to-fiducial fit that produced the stored
bridge. So the per-cell layer is not allowed to carry it. Median-centred
it cannot move a whole FOV at all -- when it is wrong it is wrong one
cell at a time, which is the failure this layer can afford.

The median is still RECORDED, as a per-FOV diagnostic that is never
applied. It is a second, independent reading of the bridge, and its
disagreement with the stored value says something about the bridge. A
version that subtracted it silently would hide exactly the quantity
somebody would later have to rediscover.

Four things are checked rather than assumed, because each of them
produced a confident wrong answer while this layer was being built:

    the sign       register3d's direction is not the store's. Read per
                   call by probe_sign, from a MAJORITY of cells -- one
                   cell is not enough, and reading it from one cost two
                   whole FOVs before this was found.
    the wrap       FFT correlation is circular, so a search bound near
                   half the depth makes +s and -(n-s) one peak. Refused,
                   not warned.
    the frame      a cell mask lives in the frame it was segmented on.
                   Crop centres go through the resolver; using the raw
                   centroid cut boxes ~100 px off the cell.
    the edges      a clipped box registers against its own padding.
"""
import numpy as np

from . import cell_bridge_gate as gate

#: bounds for the per-cell search, microns. The bound must stay well
#: inside the stack: see gate.check_search_bound.
DZ_MAX_UM = 3.0
DXY_MAX_UM = 1.5
#: a correlation peak below this is not an alignment
MIN_PEAK = 0.15
#: px, how far a cell's box must stay from the field edge
EDGE_MARGIN_PX = 4


def register3d(a, b, dz_max, dxy_max):
    """(dy, dx, dz, peak, at_edge) aligning b onto a, sub-voxel.

    Normalised cross-correlation by FFT, searched only within the given
    bounds, then refined by a parabola through the peak's neighbours.
    The returned sign is NOT the store's -- see shift_sign.
    """
    A = (a - a.mean()) / (a.std() + 1e-12)
    B = (b - b.mean()) / (b.std() + 1e-12)
    C = np.fft.fftshift(np.real(np.fft.ifftn(np.fft.fftn(A) * np.conj(np.fft.fftn(B)))))
    C /= A.size
    c0 = np.array(C.shape) // 2
    sl = tuple(slice(max(c0[i] - m, 0), min(c0[i] + m + 1, C.shape[i]))
               for i, m in enumerate((dxy_max, dxy_max, dz_max)))
    W = C[sl]
    k = np.unravel_index(int(np.argmax(W)), W.shape)
    peak = float(W[k])
    at_edge = any(k[i] in (0, W.shape[i] - 1) for i in range(3))
    off = np.array([sl[i].start + k[i] - c0[i] for i in range(3)], float)
    for i in range(3):
        if 0 < k[i] < W.shape[i] - 1:
            idx = list(k)
            idx[i] = k[i] - 1
            y0 = W[tuple(idx)]
            idx[i] = k[i] + 1
            y2 = W[tuple(idx)]
            den = y0 - 2 * peak + y2
            if den != 0:
                off[i] += 0.5 * (y0 - y2) / den
    return off[0], off[1], off[2], peak, at_edge


def translation(dy, dx):
    H = np.eye(3)
    H[0, 2], H[1, 2] = dy, dx
    return H


class FovFit(object):
    """What one FOV produced, and everything needed to doubt it.

    `entries` is what the resolver consumes. Everything else exists so a
    reader can tell a fit that worked from one that merely ran.
    """

    __slots__ = ('entries', 'median_dz_planes', 'stored_bridge_z_planes',
                 'n_fitted', 'n_cells', 'refused', 'verdict', 'sign',
                 'peak_median', 'self_recovery', 'transit_cut')

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    @property
    def disagreement_planes(self):
        """this estimator's FOV reading minus the stored bridge.

        Not applied anywhere. It is the per-FOV diagnostic the
        median-centring would otherwise throw away.
        """
        if self.median_dz_planes is None or self.stored_bridge_z_planes is None:
            return None
        return self.median_dz_planes - self.stored_bridge_z_planes

    def as_record(self):
        d = self.disagreement_planes
        return {
            'n_fitted': self.n_fitted, 'n_cells': self.n_cells,
            'refused': dict(self.refused or {}),
            'median_dz_planes': self.median_dz_planes,
            'stored_bridge_z_planes': self.stored_bridge_z_planes,
            'disagreement_planes': d,
            'sign': self.sign, 'peak_median': self.peak_median,
            'transit_cut': self.transit_cut,
            # how much of a KNOWN shift the probe cells recovered. Spread
            # here is a property of the data, not of the fit: on Lamin
            # IF 408 the quartiles run 0.56 / 0.93 / 0.97, so a cell
            # whose dz comes back compressed is not an anomaly and a
            # caller comparing against another method should know it.
            'self_recovery_median': (float(np.median(self.self_recovery))
                                     if self.self_recovery else None),
            'self_recovery_p25': (float(np.percentile(self.self_recovery, 25))
                                  if self.self_recovery else None),
            'admitted': bool(self.verdict) if self.verdict is not None else None,
            'verdict': None if self.verdict is None else self.verdict.reason,
        }

    def __repr__(self):
        return ('<FovFit %s: %s of %s cells, median %s, stored %s>'
                % ('admitted' if self.verdict else 'REFUSED',
                   self.n_fitted, self.n_cells,
                   _f(self.median_dz_planes), _f(self.stored_bridge_z_planes)))


def _f(v):
    return 'n/a' if v is None else '%+.2f' % v


#: how many cells to read the sign convention from before deciding
SIGN_PROBE_CELLS = 9


def probe_sign(volumes, dz_max, dxy_max, want=SIGN_PROBE_CELLS):
    """(sign, verdict, recoveries) -- the convention, by majority.

    An earlier version read the sign from ONE volume and refused the
    whole FOV when that one cell failed. Measured on Lamin IF 408, a
    cell's recovery of its own known 3-plane shift has 10th/25th/50th
    percentiles of 0.37/0.56/0.93, so roughly a quarter of cells fail a
    tolerance that the median passes comfortably -- and which cell
    happens to be first is arbitrary. Two FOVs were lost that way.

    The sign is a property of the CODE, not of a cell, so a majority
    over several cells is both more robust and more honest than one
    draw. The recoveries come back too, because their spread is a real
    property of the data and the caller should be able to see it.
    """
    signs, rec = [], []
    s = max(2, dz_max // 4)
    for v in volumes[:want]:
        n = v.shape[2] - s
        if n < 5:
            continue
        got = register3d(v[:, :, :n], v[:, :, s:s + n], dz_max, dxy_max)[2]
        rec.append(abs(got) / float(s))
        if abs(got) > 0.2 * s:          # a direction at all
            signs.append(1 if got * s > 0 else -1)
    if not signs:
        return None, gate.Verdict(
            False, 'no probe cell recovered a known %d-plane shift '
                   '(best %.2f of it)' % (s, max(rec) if rec else 0.0),
            probes=len(rec)), rec
    pos = sum(1 for x in signs if x > 0)
    if pos != len(signs) and pos != 0:
        return None, gate.Verdict(
            False, 'the sign convention is not consistent across probe '
                   'cells (%d of %d positive)' % (pos, len(signs))), rec
    return (1 if pos else -1), gate.Verdict(
        True, '%d of %d probe cells agree; median recovery %.2f'
              % (len(signs), len(rec), float(np.median(rec)) if rec else 0.0)), rec


def fit_cells(pairs, voxel_z_um, voxel_xy_um, stored_bridge_z_planes=None,
              dz_max_um=DZ_MAX_UM, dxy_max_um=DXY_MAX_UM,
              min_peak=MIN_PEAK, sign_probe=None):
    """Fit every cell of one FOV and median-centre the result.

    `pairs` is [(cell_id, from_volume, to_volume), ...], already cut at
    frame-correct, edge-excluded centres -- this function does no I/O,
    so it can be tested without a store.

    DIRECTION, which the names used to get wrong. The stored dz is
    planes to ADD to a coordinate in FROM's frame to reach TO's. For the
    cell-bridge layer that means FROM is the modality the entry is keyed
    under and TO is `shared`: on the Lamin store, from = IF 408,
    to = DNA 545, matching bridge_z_between('IF', 'DNA').
    These used to be called (hub, moving), which invited a caller to put
    the resolver's hub -- `shared`, i.e. DNA -- first and get the
    negative of what z_to_shared wants.

    `sign_probe` overrides the probe volumes; by default the first
    SIGN_PROBE_CELLS `from` volumes are used and the majority decides.
    """
    pairs = list(pairs)
    if not pairs:
        return FovFit(entries={}, n_cells=0, n_fitted=0, refused={},
                      verdict=gate.Verdict(False, 'no cells'))

    nz = min(min(h.shape[2], m.shape[2]) for _c, h, m in pairs)
    dz_max = int(round(dz_max_um / voxel_z_um))
    dxy_max = int(round(dxy_max_um / voxel_xy_um))
    ok = gate.check_search_bound(dz_max, nz)
    if not ok:
        return FovFit(entries={}, n_cells=len(pairs), n_fitted=0,
                      refused={'search-bound': len(pairs)}, verdict=ok)

    # the probe volumes are trimmed too, or the sign is read through the
    # same artefact the fit is being protected from
    probes = ([sign_probe] if sign_probe is not None
              else [gate.trim_transit(p[1])[0][0] for p in pairs[:SIGN_PROBE_CELLS]])
    sign, sv, recoveries = probe_sign(probes, dz_max, dxy_max)
    if sign is None:
        return FovFit(entries={}, n_cells=len(pairs), n_fitted=0,
                      refused={'sign-check': len(pairs)}, verdict=sv,
                      self_recovery=recoveries)

    raw, refused, peaks, cuts = {}, {}, [], []
    for cid, hub, mov in pairs:
        n = min(hub.shape[2], mov.shape[2])
        # TRANSIT PLANES, removed before anything is measured. A z-scan
        # returns the stage home and the frame that catches it is written
        # at one end of the stack, holding an image from the MIDDLE. It
        # sits at the same index in both stacks, so it correlates hardest
        # at exactly zero lag and drags the fit there. Measured over the
        # full Lamin store by another session, leaving it in read the
        # cross-modal z at about a third of its value; on FOV 1 here the
        # median goes +0.27 -> +1.22 planes against a stored +1.44 once
        # it is cut. Detected, not assumed -- chr19 and MP58 have none.
        (h, m), cut = gate.trim_transit(hub[:, :, :n], mov[:, :, :n])
        cuts.append(cut)
        dy, dx, dz, peak, at_edge = register3d(h, m, dz_max, dxy_max)
        why = ('search-edge' if at_edge else
               'weak-peak' if peak < min_peak else '')
        if why:
            refused[why] = refused.get(why, 0) + 1
            continue
        # into the STORE direction: planes to ADD to the hub's z to reach
        # the moving frame. `sign` is what the probe measured, never a
        # constant written in by hand.
        raw[cid] = (-sign * dy, -sign * dx, -sign * dz)
        peaks.append(peak)

    if not raw:
        return FovFit(entries={}, n_cells=len(pairs), n_fitted=0,
                      refused=refused, sign=sign, self_recovery=recoveries,
                      verdict=gate.Verdict(False, 'every cell refused'))

    arr = np.array(list(raw.values()), float)
    med = np.median(arr, axis=0)
    # THE CENTRING. What is stored is the deviation; the median itself
    # goes to the diagnostic and is never applied.
    # UNITS: register3d works in voxels, so dy/dx are pixels and dz is
    # already PLANES. The resolver's z leg is in planes too -- see
    # bridge_z_between and z_to_shared -- so nothing is converted here.
    # An earlier version divided dz by voxel_z_um, which multiplied
    # every axial value by four on this store and was caught only
    # because the self-test checks the diagnostic against a known
    # imposed offset.
    entries = {cid: (translation(v[0] - med[0], v[1] - med[1]),
                     float(v[2] - med[2]))
               for cid, v in raw.items()}
    return FovFit(entries=entries, n_cells=len(pairs), n_fitted=len(raw),
                  refused=refused, sign=sign, self_recovery=recoveries,
                  transit_cut=(max(c[0] for c in cuts),
                               max(c[1] for c in cuts)) if cuts else None,
                  median_dz_planes=float(med[2]),
                  stored_bridge_z_planes=stored_bridge_z_planes,
                  peak_median=float(np.median(peaks)),
                  verdict=gate.Verdict(True, 'fitted %d of %d cells'
                                       % (len(raw), len(pairs))))


def _selftest():
    from scipy import ndimage as ndi
    rng = np.random.default_rng(7)
    vz, vxy = 0.25, 0.108

    def cell(dz_planes, dy_px=0):
        """a hub/moving pair offset by an EXACT integer number of voxels.

        Built by slicing, not by ndi.shift. An earlier version shifted
        with order=1 interpolation, which blurs the moving copy and pulls
        the correlation toward zero: a known 2.0 plane offset came back
        as 1.74, and the test was measuring its own interpolation rather
        than the fitter.
        """
        s, dy = int(dz_planes), int(dy_px)
        pad = 8
        big = ndi.gaussian_filter(rng.random((40 + 2 * pad, 40, 60 + 2 * pad)),
                                  2.0)
        hub = big[pad:pad + 40, :, pad:pad + 60]
        mov = big[pad + dy:pad + dy + 40, :, pad + s:pad + s + 60]
        return hub, mov

    # a FOV where every cell shares one offset: after centring the layer
    # must be EMPTY of signal, because a common offset is the bridge's
    # job and not this layer's
    pairs = [(i,) + cell(2) for i in range(6)]
    f = fit_cells(pairs, vz, vxy, stored_bridge_z_planes=1.5)
    assert f.verdict, f
    dz = np.array([f.entries[i][1] for i in range(6)])
    # What is left is the ESTIMATOR'S OWN noise, since these six cells
    # share one offset and differ only in their noise realisation. It
    # comes out well under a tenth of a plane on exact integer offsets,
    # which is the fitter's floor; on real Lamin data the same quantity
    # measured 0.037 um. The bound admits that rather than pretending
    # the fit is exact.
    assert np.abs(dz).max() < 0.15, dz
    assert abs(np.median(dz)) < 1e-9
    # and the common offset is not lost -- it is in the diagnostic
    assert abs(abs(f.median_dz_planes) - 2.0) < 0.15, f.median_dz_planes
    # The diagnostic is an arithmetic identity, checked as one. Its SIGN
    # cannot be asserted here: on synthetic data there is no store, so
    # shift_sign has nothing to anchor to and either orientation is a
    # correct answer to "which way does this registration report". An
    # earlier version asserted a signed value and was testing its own
    # guess about that.
    assert abs(f.disagreement_planes
               - (f.median_dz_planes - 1.5)) < 1e-9, f.disagreement_planes

    # a FOV where cells differ: the deviations must track, and still
    # have zero median
    want = [-2, -1, 0, 1, 2, 3]
    pairs = [(i,) + cell(4 + w) for i, w in enumerate(want)]
    f = fit_cells(pairs, vz, vxy)
    got = np.array([f.entries[i][1] for i in range(6)])
    got = got - np.median(got)
    ref = np.array(want) - np.median(want)
    r = float(np.corrcoef(np.abs(got), np.abs(ref))[0, 1])
    assert abs(r) > 0.95, (r, got, ref)
    assert abs(np.median(got)) < 1e-9

    # the xy leg rides along
    pairs = [(i,) + cell(0, dy_px=(i - 3)) for i in range(7)]
    f = fit_cells(pairs, vz, vxy)
    dy = np.array([f.entries[i][0][0, 2] for i in range(7)])
    dy = dy - np.median(dy)
    ref = np.arange(7) - 3.0
    assert abs(float(np.corrcoef(dy, ref)[0, 1])) > 0.95, (dy, ref)

    # a wrapping search bound is refused, not fitted
    f = fit_cells([(0,) + cell(0)], vz, vxy, dz_max_um=40.0)
    assert not f.verdict and 'wrap' in f.verdict.reason, f.verdict
    assert f.entries == {}

    # no cells is a refusal with a reason, not a crash
    assert not fit_cells([], vz, vxy).verdict

    rec = fit_cells([(i,) + cell(2) for i in range(6)], vz, vxy,
                    stored_bridge_z_planes=1.5).as_record()
    for k in ('median_dz_planes', 'stored_bridge_z_planes',
              'disagreement_planes', 'sign', 'n_fitted'):
        assert k in rec, k
    print('cell_bridge_fit: all checks pass')


if __name__ == '__main__':
    _selftest()
