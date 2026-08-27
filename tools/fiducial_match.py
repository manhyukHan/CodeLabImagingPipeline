"""
Centroid-match versus image-match, scored on real replicate pairs.

THE ONE THING THE FIDUCIAL IS FOR
---------------------------------
    delta(hybe) = reference_fiducial_position - this_hybe_fiducial_position

and that delta then corrects every readout in the round. Today it is
obtained by fitting a PSF to each fiducial and subtracting two ABSOLUTE
positions. But the fiducial is not a point source -- fit a Gaussian to it
and the recovered width follows the fit window as sigma ~ r^0.5 with no
plateau, measured on all four experiments. A model that cannot describe
the object is being asked for a position.

Image-match estimates the DIFFERENCE directly, by registering this hybe's
fiducial crop against the reference's:

  * no shape model, so nothing to be wrong about
  * the object's shape is the SAME in both rounds, so it cancels rather
    than biasing
  * structure HELPS -- it sharpens a correlation peak, where it only
    broadened the PSF fit

WHY THE REPLICATE SCORE CAN JUDGE THIS
--------------------------------------
H and R rounds probe the SAME locus, so after correction they must land
on the same physical point and their separation IS localization error.
The fiducial correction enters a pair difference as

    delta(a) - delta(b) = fid(b) - fid(a)

so the reference position cancels and the comparison is purely between
the two ways of measuring one fiducial displacement.

WHAT IS HELD FIXED
------------------
Both arms use the SAME v2 readout fits, from the same crops, with the
same settings. The only thing that varies is where delta comes from.
Toehold rounds are excluded -- they are displacement controls, not
replicates, and scoring them books designed displacement as error.

Usage:
    python tools/fiducial_match.py bench/mp58_48.npz --exp MP58
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codelab_pipeline import parallel as PL                      # noqa: E402
from codelab_pipeline.localization import fit3d_um as U          # noqa: E402
from codelab_pipeline.localization import shapefree as SF        # noqa: E402
from tools.experiments import EXPERIMENTS                        # noqa: E402

VOXEL = (0.208, 0.208, 0.2)
# The v2 settings established earlier: NaN-padded box, centroid seed,
# linear background, loose separate bounds. Not re-tuned here -- this
# comparison is about delta, and moving two things at once would make
# the result unattributable.
FIT = dict(voxel_um=VOXEL, peak_bound_um=1.04, peak_bound_z_um=2.0,
           max_sigma_xy_um=3.0, max_sigma_z_um=6.0,
           fit_radius_um=(1.0, 1.0, 3.0), background='linear',
           apply_gates=False)


def _expected_native_z(arrays, hybes, offsets):
    """Where this allele should sit, in depth, in EACH hybe's own stack.

    A single consensus depth shared by every hybe is WRONG, and wrongly in
    a way that still produces numbers: the crops are XY-transformed but
    take the full Z slab, so their z is NATIVE, and two hybes' fiducials
    differ by the cell-level z offset -- up to 21 planes in this dataset.
    Seeding every fit at one depth put the first version of this tool at a
    1.78 um median pair distance against the 0.090 um this bench is known
    to reach.

    So: map each hybe's own argmax into the SHARED frame, take the median
    there, and map it back per hybe.

        shared_z(h) = argmax_z(h) + offset(h)
        baseline    = median over h
        native_z(h) = baseline - offset(h)

    Fit-free by design -- measured 1.05 planes of placement error at
    0.09 s, against 4.65 planes at 106 s for a per-hybe pillar fit. The
    expensive route is worse, because a pillar fit is the degenerate fit
    this whole exercise exists to get away from.
    """
    shared = []
    for h in hybes:
        c = arrays.get(f'{h}|fiducial')
        off = offsets.get(h)
        if c is None or off is None or not np.isfinite(c).any():
            continue
        z = float(np.unravel_index(int(np.nanargmax(c)), c.shape)[2])
        shared.append(z + float(off))
    if not shared:
        return {}
    baseline = float(np.median(shared))
    return {h: baseline - float(offsets[h]) for h in hybes if h in offsets}


def _slab(cube, zc, half):
    """Planes [zc-half, zc+half] of a pillar, NaN-padded where it runs off.

    Padded rather than clipped: clipping changes the slab's extent and so
    its centre, and two slabs of different extent are not comparable --
    which is the whole point of correlating them.
    """
    if cube is None:
        return None
    nz = cube.shape[2]
    z0 = int(round(zc - half))
    z1 = z0 + 2 * int(half) + 1
    out = np.full((cube.shape[0], cube.shape[1], z1 - z0), np.nan,
                  dtype=float)
    a0, b0 = max(z0, 0), max(-z0, 0)
    a1 = min(z1, nz)
    if a1 <= a0:
        return None
    out[:, :, b0:b0 + (a1 - a0)] = cube[:, :, a0:a1]
    return out if np.isfinite(out).any() else None


def _fit_at(cube, zc):
    """v2 fit seeded at the crop centre laterally, `zc` axially."""
    if cube is None or not np.isfinite(cube).any():
        return None
    cy = (cube.shape[0] - 1) / 2.0
    cx = (cube.shape[1] - 1) / 2.0
    zc = float(np.clip(zc, 0, cube.shape[2] - 1))
    seed = U.intensity_centroid(cube, (cy, cx, zc), (5, 5, 10), VOXEL)
    if seed is None:
        seed = (cy, cx, zc)
    return U.fit_gaussian_3d_um(cube, seed[0], seed[1], seed[2], **FIT)


def _one_allele(item):
    """Every delta, both ways, for one allele. Returns picklable rows."""
    aid, hybes, ref_hybe, arrays, offsets, half = item
    ref_fid = arrays.get(f'{ref_hybe}|fiducial')
    if ref_fid is None:
        return {'aid': aid, 'readout': {}, 'delta_fit': {}, 'delta_xcorr': {},
                'delta_slab': {}, 'cov': {}, 'q': {}, 'qs': {}}

    zexp = _expected_native_z(arrays, hybes, offsets)
    ref_z = zexp.get(ref_hybe, ref_fid.shape[2] / 2)
    ref_fit = _fit_at(ref_fid, ref_z)

    readout, delta_fit, delta_xcorr, cov, q = {}, {}, {}, {}, {}
    delta_slab, qs = {}, {}
    for h in hybes:
        fid = arrays.get(f'{h}|fiducial')
        rd = arrays.get(f'{h}|readout')
        zh = zexp.get(h)
        if rd is not None:
            f = _fit_at(rd, zh if zh is not None else rd.shape[2] / 2)
            if f is not None:
                readout[h] = (f.y, f.x, f.z)
        if fid is None:
            continue
        # -- arm A: centroid-match (two absolute PSF fits, subtracted)
        if ref_fit is not None:
            ff = _fit_at(fid, zh if zh is not None else fid.shape[2] / 2)
            if ff is not None:
                delta_fit[h] = (ref_fit.y - ff.y, ref_fit.x - ff.x,
                                ref_fit.z - ff.z)
        # -- arm B: image-match on the whole PILLAR
        s, qq = SF.shift_yxz(ref_fid, fid, upsample=20, min_coverage=0.9)
        if s is not None:
            delta_xcorr[h] = s
            q[h] = qq
        cov[h] = SF.signal_coverage(ref_fid, fid)

        # -- arm C: image-match on a SLAB around each crop's expected depth
        #
        # The pillar is 110 planes and the fiducial occupies a few of them,
        # so a pillar correlation is mostly out-of-focus content correlating
        # with out-of-focus content. That is the same failure that made the
        # PILLAR fit degenerate, and box-not-pillar was the largest single
        # improvement in the fit. Registration should not be exempt from it.
        #
        # Each crop's slab is cut around its OWN expected depth, so the
        # bulk z difference is carried by the two origins and the
        # correlation only has to find the residual:
        #     delta_z = (z_ref - z_h) + residual_z
        if zh is not None and ref_z is not None:
            sa = _slab(ref_fid, ref_z, half)
            sb = _slab(fid, zh, half)
            if sa is not None and sb is not None:
                s2, q2 = SF.shift_yxz(sa, sb, upsample=20, min_coverage=0.9)
                if s2 is not None:
                    delta_slab[h] = (s2[0], s2[1], s2[2] + (ref_z - zh))
                    qs[h] = q2
    return {'aid': aid, 'readout': readout, 'delta_fit': delta_fit,
            'delta_xcorr': delta_xcorr, 'delta_slab': delta_slab,
            'cov': cov, 'q': q, 'qs': qs}


def _pair_distances(results, pairs, deltas_key):
    """{(allele, a, b): (d3_um, dxy_um)} after applying that arm's delta.

    Keyed rather than listed so the two arms can be compared on the pairs
    they BOTH scored. One arm rejecting the hard rounds and then showing a
    better median would otherwise look like an improvement, when it is
    just a different, easier sample.
    """
    dy, dx, dz = VOXEL
    out, either = {}, 0
    for r in results:
        rd, dl = r['readout'], r[deltas_key]
        for a, b, _rid in pairs:
            if a not in rd or b not in rd:
                continue
            either += 1
            if a not in dl or b not in dl:
                continue
            pa = np.array(rd[a], dtype=float) + np.array(dl[a], dtype=float)
            pb = np.array(rd[b], dtype=float) + np.array(dl[b], dtype=float)
            d = (pa - pb) * np.array([dy, dx, dz])
            out[(r['aid'], a, b)] = (float(np.linalg.norm(d)),
                                     float(np.linalg.norm(d[:2])))
    return out, either


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bench')
    ap.add_argument('--exp', required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--alleles', type=int, default=None)
    ap.add_argument('--slab-half', type=int, default=15,
                    help='half-depth in planes of the slab arm')
    a = ap.parse_args()

    exp = EXPERIMENTS[a.exp]
    b = np.load(a.bench, allow_pickle=False)
    meta = b['__meta__']
    hybes = [str(h) for h in b['__hybes__']]
    pairs = [tuple(s.split('|')) for s in b['__pairs__']]
    # H/R only. Toehold rounds are displacement controls.
    pairs = [(x, y, r) for x, y, r in pairs
             if not x.startswith('Toe') and not y.startswith('Toe')]

    # __zpred__ rows are 'a{id}|{hybe}|{native_z}|{offset}'. The OFFSET is
    # what matters here: it is the correction that maps this hybe's raw z
    # into the shared frame, and without it every hybe's crop is at a
    # different depth with nothing to say so.
    offsets_by = {}
    for s in b['__zpred__']:
        key, hy, _nz, off = str(s).split('|')
        offsets_by.setdefault(key, {})[hy] = float(off)

    aids = [int(m[0]) for m in meta][:a.alleles] if a.alleles else [int(m[0]) for m in meta]
    items = []
    for aid in aids:
        arrays = {}
        for h in hybes:
            for kind in ('fiducial', 'readout'):
                k = f'a{aid}|{h}|{kind}'
                if k in b.files:
                    arrays[f'{h}|{kind}'] = b[k].astype(np.float32)
        items.append((aid, hybes, exp.reference_hybe, arrays,
                      offsets_by.get(f'a{aid}', {}), a.slab_half))

    print(f'bench      : {a.bench}')
    print(f'experiment : {exp.name}   reference {exp.reference_hybe}')
    print(f'alleles    : {len(items)}   hybes: {len(hybes)}')
    print(f'replicates : {len(pairs)} H/R pairs per allele '
          f'-> {len(pairs) * len(items)} possible\n', flush=True)

    t0 = time.perf_counter()
    done = [0]

    def prog(n, total, _i, r):
        done[0] = n
        if n % 5 == 0 or n == total:
            el = time.perf_counter() - t0
            print(f'  [{n:>3}/{total}] {el:6.0f}s  '
                  f'eta {(total - n) * el / max(n, 1) / 60:5.1f} min', flush=True)

    res = PL.pmap(_one_allele, items, kind='cpu', jobs=a.jobs, on_done=prog)
    good = PL.ok(res)
    print(f'\n{len(good)}/{len(items)} alleles processed in '
          f'{time.perf_counter() - t0:.0f}s')
    for f in PL.failures(res)[:3]:
        print('  FAILED:', f.error)

    covs = np.array([c for r in good for c in r['cov'].values()])
    qs = np.array([q for r in good for q in r['q'].values()])
    print(f'\nfiducial signal coverage : median {np.median(covs):.3f}   '
          f'{100 * (covs < 0.9).mean():.1f}% below 0.90 (rejected)')
    if qs.size:
        print(f'registration correlation : median {np.median(qs):.3f}   '
              f'10th pct {np.percentile(qs, 10):.3f}')

    A, either = _pair_distances(good, pairs, 'delta_fit')
    B, _ = _pair_distances(good, pairs, 'delta_xcorr')
    C, _ = _pair_distances(good, pairs, 'delta_slab')
    common = sorted(set(A) & set(B) & set(C))

    print('\n' + '=' * 78)
    print(f'{"delta from":<24}{"pairs":>12}{"median 3D":>13}{"median XY":>13}{"p90 3D":>13}')
    print('=' * 78)

    def line(label, d, n_either):
        if not d:
            print(f'{label:<24}   no pairs scored')
            return
        a3 = np.array([v[0] for v in d.values()])
        axy = np.array([v[1] for v in d.values()])
        print(f'{label:<24}{len(d):>6}/{n_either:<5}{np.median(a3):>12.4f}u'
              f'{np.median(axy):>12.4f}u{np.percentile(a3, 90):>12.4f}u')

    line('centroid-match (fit)', A, either)
    line('image-match, pillar', B, either)
    line(f'image-match, slab +/-{a.slab_half}', C, either)

    if common:
        print(f'\n-- on the {len(common)} pairs BOTH arms scored '
              f'(the only like-for-like comparison) --')
        line('  centroid-match', {k: A[k] for k in common}, len(common))
        line('  image-match, pillar', {k: B[k] for k in common}, len(common))
        line('  image-match, slab', {k: C[k] for k in common}, len(common))
        ma = np.median([A[k][0] for k in common])
        print()
        for label, D in (('image-match, pillar', B), ('image-match, slab', C)):
            m = np.median([D[k][0] for k in common])
            # A median can move on a handful of pairs. The per-pair win
            # rate says whether a change is systematic or a few large ones,
            # and the two disagree often enough to be worth printing both.
            wins = sum(1 for k in common if D[k][0] < A[k][0])
            print(f'  {label:<22} vs centroid-match: '
                  f'{100 * (m - ma) / ma:+6.1f}% on the median   '
                  f'closer on {wins:>3}/{len(common)} pairs '
                  f'({100 * wins / len(common):>2.0f}%)   '
                  f'{"better" if m < ma else "WORSE"}')


if __name__ == '__main__':
    main()
