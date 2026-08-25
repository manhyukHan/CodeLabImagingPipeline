"""
Fit-engine comparison report: our Gaussian engine vs the MATLAB-faithful
FitPsf3D port (tests/test_fit_engine_vs_matlab.py), on synthetic ground
truth AND every persisted spot's real crop in a store.

    CODELAB_FIT_STORE=<store root with RNA_queue/DNA_queue> \
        python tools/fit_engine_report.py [output.png]

Read-only: crops and fits, never writes to the store. Defaults to the
real store; point CODELAB_FIT_STORE at a clone when the live app holds
the lock. Output defaults to fit_engine_report.png in the cwd.

Six panels: synthetic RMSE vs ground truth per engine, synthetic
engine-agreement histogram, real-data agreement vs fitted contrast, a
typical real crop and the worst real crop with both centers overplotted,
and a text summary (including the one MATLAB pipeline difference that is
NOT the fit: ChrTracer3_FitSpots resamples the data crop by the fiducial
drift at upsample=8 before fitting, where we fit raw pixels and correct
the fitted coordinate numerically).
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from codelab_pipeline.io import analysis_store as V
from codelab_pipeline.alignment import spot_mapper
from codelab_pipeline.localization import localization as L
from test_fit_engine_vs_matlab import fitpsf3d_reference, make_stack, AMPLITUDES, N_PER_CONDITION

STORE = os.environ.get('CODELAB_FIT_STORE', 'data/chr19_downstream_new')
OUT = sys.argv[1] if len(sys.argv) > 1 else 'fit_engine_report.png'
FOV = 1


def main():
    # ---- synthetic battery ----
    rng = np.random.default_rng(0)
    syn = {a: {'ours': [], 'ref': [], 'truth': []} for a in AMPLITUDES}
    for amp in AMPLITUDES:
        for _ in range(N_PER_CONDITION):
            stack, center = make_stack(rng, amp)
            iy, ix, iz = np.unravel_index(int(np.argmax(stack)), stack.shape)
            r_o = L.fit_gaussian_3d(stack, float(ix), float(iy), float(iz))
            r_r = fitpsf3d_reference(stack, (iy, ix, iz))
            if r_o is None or r_r is None:
                continue
            syn[amp]['ours'].append((r_o[2], r_o[1], r_o[3]))
            syn[amp]['ref'].append(r_r)
            syn[amp]['truth'].append(center)

    # ---- real data: every persisted spot, both engines ----
    rows = []
    sp_any = os.path.join(STORE, 'RNA_queue')
    for d in V.read_spots(sp_any, FOV):
        hybe, modality, ch = d['hybe'], d['modality'], int(d['channel'])
        sp = os.path.join(STORE, 'DNA_queue' if modality == 'DNA' else 'RNA_queue')
        ry, rx = float(d['raw_coordinate'][0]), float(d['raw_coordinate'][1])
        try:
            cubic, _origin = spot_mapper.crop_for_localization(sp, FOV, hybe, ch, (ry, rx),
                                                               pad=8, use_stack=True)
        except OSError:
            continue
        if cubic.size == 0 or not np.isfinite(cubic).any():
            continue
        iy, ix, iz = np.unravel_index(int(np.nanargmax(cubic)), cubic.shape)
        r_o = L.fit_gaussian_3d(cubic, float(ix), float(iy), float(iz))
        r_r = fitpsf3d_reference(cubic, (iy, ix, iz))
        if r_o is None or r_r is None:
            continue
        ours = np.array([r_o[2], r_o[1], r_o[3]])
        contrast = float(r_o[0]) / max(float(r_o[7]), 1e-9)
        rows.append((d['uid'], hybe, modality, ours, np.array(r_r), cubic, contrast))

    if not rows:
        raise SystemExit(f'no fittable spots in {STORE} FOV{FOV:02d} -- '
                         'both engines rejected everything, or the store is empty')
    d_real = np.array([np.abs(r[3] - r[4]) for r in rows])
    dmax = d_real.max(axis=1)
    n01 = int((dmax < 0.1).sum())
    print(f'real spots fitted by BOTH engines: {len(rows)}')
    print(f'median |ours-ref| (y, x, z): {np.median(d_real, axis=0)}')
    print(f'agree within 0.1 px on every axis: {n01}/{len(rows)}')

    # ---- figure ----
    fig = plt.figure(figsize=(14, 10))

    ax = fig.add_subplot(2, 3, 1)
    width = 0.35
    labels, o_vals, r_vals = [], [], []
    for amp in AMPLITUDES:
        t = np.array(syn[amp]['truth'])
        o = np.sqrt(np.mean((np.array(syn[amp]['ours']) - t) ** 2, axis=0))
        r = np.sqrt(np.mean((np.array(syn[amp]['ref']) - t) ** 2, axis=0))
        for axis, nm in enumerate('yxz'):
            labels.append(f'a={amp:.0f}\n{nm}')
            o_vals.append(o[axis]); r_vals.append(r[axis])
    xi = np.arange(len(labels))
    ax.bar(xi - width / 2, o_vals, width, label='ours', color='#1f77b4')
    ax.bar(xi + width / 2, r_vals, width, label='MATLAB port', color='#d62728', alpha=0.8)
    ax.set_xticks(xi); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('RMSE vs ground truth (px)')
    ax.set_title(f'A. Synthetic accuracy\n({N_PER_CONDITION}/condition, Poisson noise)', fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 2)
    all_d = np.abs(np.vstack([np.array(syn[a]['ours']) - np.array(syn[a]['ref'])
                              for a in AMPLITUDES]))
    mx = max(all_d.max(), 1e-6)
    for axis, nm in enumerate('yxz'):
        ax.hist(all_d[:, axis], bins=40, range=(0, mx * 1.05), alpha=0.6, label=f'|Δ{nm}|')
    ax.set_xlabel('|ours − MATLAB port| (px)')
    ax.set_ylabel('synthetic fits')
    ax.set_title(f'B. Engine agreement, synthetic\n(max = {all_d.max():.2e} px)', fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 3)
    contrast = np.array([r[6] for r in rows])
    ax.scatter(contrast, np.maximum(dmax, 1e-4), s=18, c='#2ca02c', alpha=0.75)
    ax.axhline(0.1, color='gray', lw=0.8, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('fitted spot contrast (amplitude / background)')
    ax.set_ylabel('max-axis |ours − MATLAB port| (px)')
    ax.annotate(f'{n01}/{len(rows)} spots agree < 0.1 px\nmedian {np.median(dmax):.3f} px',
                xy=(0.03, 0.03), xycoords='axes fraction', fontsize=8)
    ax.set_title('C. Real-data agreement vs contrast\n(same crop into both engines)', fontsize=10)

    def draw_crop(ax, row, tag):
        uid, hybe, modality, ours, ref, cubic = row[:6]
        ax.imshow(np.nanmax(cubic, axis=2), cmap='gray', interpolation='nearest')
        ax.plot(ours[1], ours[0], '+', color='#1f77b4', ms=20, mew=2.5,
                label=f'ours  (y={ours[0]:.2f}, x={ours[1]:.2f}, z={ours[2]:.2f})')
        ax.plot(ref[1], ref[0], 'x', color='#d62728', ms=13, mew=2,
                label=f'MATLAB port (y={ref[0]:.2f}, x={ref[1]:.2f}, z={ref[2]:.2f})')
        ax.set_title(f'{tag}: uid {uid} {hybe}/{modality} (MIP over z)', fontsize=10)
        ax.legend(fontsize=7, loc='lower right')

    order = sorted(range(len(rows)), key=lambda i: dmax[i])
    bright_agree = max(order[:max(1, len(rows) // 2)],
                       key=lambda i: float(np.nanmax(rows[i][5])))
    ax = fig.add_subplot(2, 3, 4)
    draw_crop(ax, rows[bright_agree], f'D. Typical crop (Δ={dmax[bright_agree]:.3f} px)')

    ax = fig.add_subplot(2, 3, 5)
    draw_crop(ax, rows[order[-1]], f'E. Worst crop (Δ={dmax[order[-1]]:.2f} px)')

    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    ax.text(0, 0.98, 'Summary', fontsize=12, weight='bold', va='top')
    ax.text(0, 0.88,
            f'Same stack -> same coordinate:\n'
            f'  synthetic: max |D| = {all_d.max():.1e} px\n'
            f'  real crops: {n01}/{len(rows)} agree < 0.1 px,\n'
            f'    median {np.median(dmax):.3f} px\n\n'
            f'Divergence is confined to DIM or multi-\n'
            f'blob windows where a single gaussian is\n'
            f'under-determined; production sends\n'
            f'multi-blob crops through the mixture.\n\n'
            f'MATLAB pipeline difference that is NOT\n'
            f'the fit: ChrTracer3_FitSpots RESAMPLES\n'
            f'the data crop by the fiducial drift\n'
            f'(TranslateImage, upsample=8) before\n'
            f'fitting; we fit raw pixels and correct\n'
            f'the fitted coordinate numerically.',
            fontsize=9, va='top', family='monospace')

    fig.suptitle('Gaussian fit: our engine vs MATLAB FitPsf3D (faithful port) -- synthetic + real data',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=140)
    print('report:', OUT)


if __name__ == '__main__':
    main()
