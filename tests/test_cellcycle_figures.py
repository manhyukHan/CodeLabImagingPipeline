"""
The Cell Cycle stage's figures on a synthetic fit: every figure function
returns a Figure whose axes follow the conventions (no top/right spine,
degree ticks on phase axes, titles inside the figure), the cycle-time
clock inverts, the arcs proposed from time cover the circle, and the
FOV overlay paints the cells.

Run:  python tests/test_cellcycle_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402

from codelab_pipeline.analysis import cellcycle as CC          # noqa: E402
from codelab_pipeline.analysis import figures_cellcycle as FC  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''), flush=True)


def conventions(fig, name):
    axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    spines = all(not ax.spines['top'].get_visible() and not ax.spines['right'].get_visible() for ax in axes)
    check(f'{name}: no top/right spines on any axes', spines)
    inside = all(0.0 <= t.get_position()[1] <= 1.0 for t in fig.texts)
    check(f'{name}: every figure-level text sits inside the figure', inside,
          str([round(t.get_position()[1], 3) for t in fig.texts]))


def main():
    import matplotlib.pyplot as plt
    X, theta, coef, a, genes = CC.simulate(n=900, seed=7, total=(80, 400))
    m = CC.CycleModel().fit([CC.Dataset('A', X, genes, alpha=100.0)])
    pk, _ = m.peak_phase()
    order = np.argsort(pk)
    roles = {genes[i]: 'S' for i in order[:3]}
    roles.update({genes[i]: 'G2/M' for i in order[4:7]})
    m.orient([g for g, r in roles.items() if r == 'S'], [g for g, r in roles.items() if r == 'G2/M'])
    placed = m.place('A', X)
    th = np.degrees(placed['theta']) % 360
    q = placed['posterior']
    groups = np.array(['u'] * 450 + ['h'] * 450, dtype=object)
    marks = FC.role_marks(m, roles)
    check('role marks name S and G2/M with the S mark at 0', set(marks) == {'S', 'G2/M'} and min(marks['S'], 360 - marks['S']) < 5,
          str(marks))

    print('figures')
    fig, notes = FC.fig_embeddings(m, 'A', X, th, tsne=False, marks=marks, title='x' * 200)
    conventions(fig, 'embeddings')
    ax = fig.axes[0]
    check('the ring plane has no ticks and labels the variance', ax.get_xticks().size == 0 and '%' in ax.get_xlabel(), ax.get_xlabel())
    check('a long title is wrapped inside the figure', fig._suptitle is not None and '\n' in fig._suptitle.get_text())
    plt.close(fig)
    fig = FC.fig_profiles_by_role(m, {'S': [g for g, r in roles.items() if r == 'S'],
                                      'G2/M': [g for g, r in roles.items() if r == 'G2/M'],
                                      'unassigned': [g for g in genes if g not in roles]}, title='profiles', marks=marks)
    conventions(fig, 'profiles')
    ax = fig.axes[0]
    check('phase axes carry degree ticks', [t for t in ax.get_xticks()] == list(FC.PHASE_TICKS), str(ax.get_xticks()))
    check('and the role marks as texts', any(t.get_text() == 'G2/M' for t in ax.texts))
    plt.close(fig)
    fig = FC.fig_spectrum_and_totals(m, groups, q, X.sum(1), title='spectrum', marks=marks)
    conventions(fig, 'spectrum')
    plt.close(fig)
    fig = FC.fig_anchor_spectra(m, [('A', 'u', q[:450]), ('A', 'h', q[450:])], ['u', 'h'], title='anchors', marks=marks)
    conventions(fig, 'anchors')
    plt.close(fig)
    thetas, st = FC.subpanel_agreement(m, 'A', X, {'A1': genes[0::2], 'A2': genes[1::2]})
    fig = FC.fig_subpanel(thetas, 'A1', 'A2', title='half ' + ', '.join(genes) * 3, marks=marks)
    conventions(fig, 'subpanel')
    check('the difference panel is a line, not bars', len(fig.axes[1].lines) >= 1 and not fig.axes[1].patches)
    plt.close(fig)

    print('cycle time')
    w = m.spectrum(q)
    tau, tg, theta_of = FC.cycle_time_map(w, m.grid, 215.0, 'uniform')
    xs = np.linspace(0, 359.9, 200)
    back = theta_of(tau(xs))
    err = np.abs((back - xs + 180) % 360 - 180)
    check('the clock inverts (theta -> tau -> theta)', np.median(err) < 3, f'median {np.median(err):.2f} deg')
    check('tau is 0 at birth and grows forward', abs(tau(215.0)) < 1e-6 and tau(216.0) > 0 and tau(214.0) > 0.9)
    arcs = FC.propose_arcs_from_time(w, m.grid, 215.0)
    check('three arcs from the default shares, starting at birth', [a['name'] for a in arcs] == ['G1', 'S', 'G2/M']
          and abs(arcs[0]['start_deg'] - 215.0) < 1e-6 and abs(arcs[-1]['end_deg'] - 215.0) < 1e-6)
    cover = CC.categorize(np.linspace(0, 359.9, 720), arcs)
    check('the arcs cover the whole circle without a gap', all(c != '' for c in cover) and set(cover) == {'G1', 'S', 'G2/M'})
    arcs_r = FC.propose_arcs_from_profiles(m, roles, birth_deg=215.0)
    check('arcs from the role profiles: G1 from birth, then S, then G2/M, covering the circle',
          [a['name'] for a in arcs_r] == ['G1', 'S', 'G2/M'] and abs(arcs_r[0]['start_deg'] - 215.0) < 1e-6
          and set(CC.categorize(np.linspace(0, 359.9, 720), arcs_r)) == {'G1', 'S', 'G2/M'}, str(arcs_r))
    s_arc = arcs_r[1]
    check('the S arc holds the S genes\' mean peak (0)', ((0.0 - s_arc['start_deg']) % 360) <= ((s_arc['end_deg'] - s_arc['start_deg']) % 360),
          str(s_arc))
    fig = FC.fig_cycle_time(m, {'u': m.spectrum(q[:450]), 'h': m.spectrum(q[450:]), 'train': w}, birth_deg=215.0,
                            training='train', groups_theta={'u': th[:450], 'h': th[450:]}, title='cycle time', marks=marks)
    conventions(fig, 'cycle time')
    check('four panels with the conditions over tau', len([ax for ax in fig.axes if ax.get_label() != '<colorbar>']) == 4)
    plt.close(fig)

    print('overlay and the gated views')
    mip = np.random.default_rng(0).normal(100, 10, (64, 64))
    cells = [{'id': 1, 'area': (np.arange(5, 15), np.arange(5, 15))},
             {'id': 2, 'area': (np.arange(30, 40), np.arange(30, 40))},
             {'id': 3, 'area': (np.arange(50, 60), np.arange(50, 60))}]
    fig = FC.fig_fov_overlay(mip, cells, {1: 10.0, 2: 200.0}, mode='phase', marks=marks)
    conventions(fig, 'overlay phase')
    imgs = fig.axes[0].get_images()
    check('the overlay paints the two placed cells and leaves the third faint',
          len(imgs) == 2 and imgs[1].get_array()[7, 7, 3] > 0.5 and imgs[1].get_array()[55, 55, 3] < 0.3)
    plt.close(fig)
    fig = FC.fig_fov_overlay(mip, cells, {1: 'S', 2: '', 3: 'G1'}, mode='category', categories=['G1', 'S'])
    conventions(fig, 'overlay category')
    plt.close(fig)
    rng_d = np.random.default_rng(3)
    fov = np.repeat([1, 2, 3], 300)
    dna = np.where(((th - 60) % 360) < 120, 2.0, 1.0) * rng_d.lognormal(0, 0.1, 900) * np.where(fov == 2, 3.0, 1.0)
    area = np.where(((th - 60) % 360) < 120, 3000.0, 2000.0) * rng_d.lognormal(0, 0.1, 900)
    fig = FC.fig_dapi_vs_phase(th, dna, groups=groups, fov=fov, training=np.ones(900, bool), marks=marks,
                               area=area, area_unit='px')
    conventions(fig, 'dapi')
    ttl = fig._suptitle.get_text()
    check('the DAPI figure carries the mask-area panel and its drop', len(fig.axes) == 2 and 'mask area falls' in ttl, ttl)
    check('the DAPI figure finds the halving angle near 180 despite a 3x FOV', 'halves at' in ttl
          and abs(float(ttl.split('halves at ')[1].split(' ')[0]) - 180) <= 15, ttl)
    plt.close(fig)
    tiles = [(np.random.default_rng(i).normal(100, 10, (16, 16)), np.zeros((16, 16), bool), 80.0, 130.0) for i in range(5)]
    tiles[0][1][4:10, 4:10] = True
    fig = FC.fig_gallery([('0-45', tiles[:3]), ('45-90', tiles[3:])], size=16, title='gallery')
    check('the gallery has one axes per tile slot and hides the empty ones',
          len(fig.axes) == 6 and sum(ax.get_visible() for ax in fig.axes) == 5)
    plt.close(fig)
    # arcs from DAPI: flat 2N from birth (215), rising from 300 to a 2x plateau at 60, back at birth
    th_d = np.linspace(0, 359.9, 1800)
    rel = (th_d - 215.0) % 360.0
    dna_d = np.where(rel < 85, 1.0, np.where(rel < 205, 1.0 + (rel - 85) / 120.0, 2.0)) * np.random.default_rng(5).lognormal(0, 0.05, 1800)
    arcs_d, info = FC.propose_arcs_from_dapi(th_d, dna_d, birth_deg=215.0)
    check('arcs from DAPI: S starts where the content rises, G2/M near the plateau',
          [a['name'] for a in arcs_d] == ['G1', 'S', 'G2/M'] and abs(((arcs_d[1]['start_deg'] - 318.0) + 180) % 360 - 180) <= 12
          and abs(((arcs_d[2]['start_deg'] - 50.0) + 180) % 360 - 180) <= 25 and 1.8 < info['ratio'] < 2.3, str((arcs_d, info)))
    try:
        FC.propose_arcs_from_dapi(th_d, np.ones(1800), birth_deg=215.0)
        check('a flat DNA curve refuses to propose', False)
    except ValueError:
        check('a flat DNA curve refuses to propose', True)
    # post-M: a density with a hole right after birth (215) that recovers by ~250
    w_h = np.ones(72)
    for i in range(72):
        dd = (i * 5 + 2.5 - 215.0) % 360.0
        if dd < 35:
            w_h[i] = 0.1
    w_h /= w_h.sum()
    end = FC.post_division_end(w_h, m.grid, 215.0)
    check('post_division_end finds where the density recovers after birth', end is not None and 240 <= end <= 262, str(end))
    check('and reports None when nothing is sparse after birth', FC.post_division_end(np.ones(72) / 72, m.grid, 215.0) is None)
    split = FC.with_post_m([{'name': 'G1', 'start_deg': 215.0, 'end_deg': 320.0}, {'name': 'S', 'start_deg': 320.0, 'end_deg': 215.0}],
                           w_h, m.grid, 215.0)
    check('with_post_m splits the first arc and keeps the rest', [a['name'] for a in split] == ['post-M', 'G1', 'S']
          and split[0]['start_deg'] == 215.0 and split[1]['start_deg'] == split[0]['end_deg'] and split[1]['end_deg'] == 320.0)
    fig = FC.fig_phase_hist(th, groups=groups, marks=marks)
    conventions(fig, 'phase hist')
    check('phase histogram draws a line per group plus all', len(fig.axes[0].lines) == 3)
    plt.close(fig)
    fig = FC.fig_category_hist(CC.categorize(th, arcs), groups=groups, order=['G1', 'S', 'G2/M'])
    conventions(fig, 'category hist')
    check('category histogram has one bar group per category', len(fig.axes[0].get_xticks()) == 3)
    plt.close(fig)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('ALL GOOD')


if __name__ == '__main__':
    main()
