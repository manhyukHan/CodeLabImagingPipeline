"""
Panel selection on synthetic cells whose truth is known: three genes
carry the cycle (one S, one G2/M, one unassigned), three are flat
nuisance, and the DNA content and the mask area follow the true phase.
The forward search must find the carriers, the backward curve must stop
before they are dropped, and both must refuse a panel that cannot be
oriented.

Run:  python tests/test_panel_select.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                             # noqa: E402

from codelab_pipeline.analysis import panel_select as PS       # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''), flush=True)


def make(n=1200, seed=0):
    """Cells on a true phase: three cycling genes with peaks 120 deg
    apart, three flat ones, DNA doubling through the cycle and halving
    at 0 deg, area following DNA."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    genes = ['S_true', 'G2M_true', 'FREE_true', 'FLAT_A', 'FLAT_B', 'FLAT_C']
    peaks = {'S_true': 0.0, 'G2M_true': 2 * np.pi / 3, 'FREE_true': 4 * np.pi / 3}
    pi = np.zeros((n, len(genes)))
    for j, g in enumerate(genes):
        pi[:, j] = np.exp(1.2 * np.cos(theta - peaks[g])) if g in peaks else 1.0
    pi /= pi.sum(1, keepdims=True)
    total = rng.integers(120, 400, n)
    X = np.array([rng.multinomial(t, p) for t, p in zip(total, pi)], float)
    # DNA: 1 just after division (theta 0), 2 just before it
    # a clean halving at 0 deg: 2N through the second half of the cycle,
    # 1N after division, so the step detector has a real step to find
    frac = theta / (2 * np.pi)
    dna = np.where(frac >= 0.5, 2.0, 1.0) * rng.lognormal(0, 0.08, n)
    area = np.where(frac >= 0.5, 1.6, 1.0) * rng.lognormal(0, 0.10, n)
    roles = {'S_true': 'S', 'G2M_true': 'G2/M', 'FREE_true': '-', 'FLAT_A': 'S', 'FLAT_B': 'G2/M', 'FLAT_C': '-'}
    groups = np.array(['cycling'] * n, dtype=object)
    return PS.PanelData(X, genes, groups, dna, area, roles, train='cycling', alpha=100.0, name='synth'), theta


def main():
    data, theta = make()
    print('one panel, scored')
    full = PS.evaluate(data, data.genes, seed=0, with_half_panel=True)
    check('the full panel fits, is oriented and finds its origin', full['ok'] and full['origin_found'], str(full.get('note', '')))
    check('the DNA content is explained by the angle', full['dna_r2'] > 0.4, f"{full['dna_r2']:.2f}")
    check('the division step is near 2', abs(full['origin_drop'] - 2) < 0.5, f"{full['origin_drop']:.2f}")
    check('the mask area follows too', full['area_r2'] > 0.2, f"{full['area_r2']:.2f}")
    check('effective genes is below the count when the shares are uneven',
          full['effective_genes'] < len(data.genes) and full['inv_simpson'] <= full['effective_genes'] + 1e-9,
          f"{full['effective_genes']:.2f} / {full['inv_simpson']:.2f}")
    flat_only = PS.evaluate(data, ['FLAT_A', 'FLAT_B', 'FLAT_C'], seed=0)
    check('a flat panel explains nothing of the DNA', (not flat_only['fitted']) or flat_only['dna_r2'] < 0.15,
          f"fitted={flat_only['fitted']} r2={flat_only.get('dna_r2')}")
    no_role = PS.evaluate(data, ['S_true', 'FLAT_A'], seed=0)
    check('a panel with no G2/M gene is refused', not no_role['fitted'] and 'oriented' in no_role['note'], str(no_role.get('note')))
    check('objective is -inf for a refused panel', PS.objective(no_role) == -np.inf)

    print('forward (bottom-up)')
    pairs = PS.seeds_for_forward(data)
    check('the seeds are every (S, G2/M) pair', sorted(map(tuple, pairs)) == sorted([('S_true', 'G2M_true'), ('S_true', 'FLAT_B'),
                                                                                     ('FLAT_A', 'G2M_true'), ('FLAT_A', 'FLAT_B')]), str(pairs))
    path = PS.forward(data, seeds=(0, 1), max_genes=5, with_half_panel=False)
    carriers = {'S_true', 'G2M_true', 'FREE_true'}
    # a pair is a RATIO, so a flat gene works as the denominator of a
    # cyclic one: the winning pair need not be two carriers, but it must
    # contain one, and the path must keep adding carriers
    check('the winning pair carries the cycle', bool(carriers & set(path[0]['genes'])), str(path[0]['genes']))
    kept = path[-1]['genes']
    check('the path ends on at least two carriers', len(carriers & set(kept)) >= 2, str(kept))
    check('no panel on the path is all-flat', all(carriers & set(e['genes']) for e in path))
    check('the path grows to at least the minimum size', path[-1]['size'] >= 4, str(path[-1]['genes']))
    ref = PS.evaluate(data, data.genes, seed=0)
    moved = dict(ref)
    for k in [k for k in ref if k.endswith('_deg') and k != 'origin_deg']:
        moved[k] = (ref[k] + 90) % 360
    ok, worst = PS.arrested_shift(ref, moved)
    check('arrested_shift flags a population that moved', (not ok) or worst == 0.0, f'ok={ok} worst={worst:.0f}')
    r_bad = dict(ref, origin_found=False)
    check('a panel without an origin ranks below one with it', PS.rank(r_bad, ref) < PS.rank(ref, ref))
    # the objective may dip while the search is still below min_size or
    # still looking for an origin: the rank, not the objective, is what
    # the search climbs
    ref0 = PS.evaluate(data, data.genes, seed=0)
    ranks = [PS.rank(e['rows'][0], ref0) for e in path]
    grown = [r for e, r in zip(path, ranks) if e['size'] >= 4]
    check('the rank never falls once the panel is big enough',
          all(a <= b for a, b in zip(grown, grown[1:])), str([(round(r[2], 3), r[0], r[1]) for r in ranks]))
    check('every kept panel is scored on every seed', all(len(e['rows']) == 2 for e in path))

    print('backward (top-down) and the stopping rule')
    order = PS.drop_order(data)
    check('the drop order starts with a flat gene', order[0].startswith('FLAT'), str(order))
    curve = PS.backward(data, seeds=(0, 1), min_genes=2, with_half_panel=False)
    check('the curve runs from the full panel down', [e['size'] for e in curve][:2] == [6, 5], str([e['size'] for e in curve]))
    stop = PS.stop_size(curve)
    check('the stop keeps the three carriers', stop is not None and {'S_true', 'G2M_true', 'FREE_true'} <= set(stop['genes']),
          str(stop['genes']) if stop else 'none')
    small = [e for e in curve if e['size'] == 2]
    if small:
        check('a two-gene panel that lost a role is marked not ok', not any(r.get('ok') for r in small[0]['rows'])
              or {'S_true', 'G2M_true'} <= set(small[0]['genes']), str(small[0]['genes']))
    v, lo, hi = PS.band(curve[0], 'dna_r2')
    check('band returns the seed-0 value inside its own range', lo - 1e-9 <= v <= hi + 1e-9, f'{v:.3f} [{lo:.3f}-{hi:.3f}]')

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', FAIL)
        sys.exit(1)
    print('ALL GOOD')


if __name__ == '__main__':
    main()
