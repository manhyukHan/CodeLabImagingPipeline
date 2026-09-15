"""Which rounds a cell-cycle panel actually needs, chosen on what the
model never saw.

The circle is fitted from the panel's counts, so any criterion the fit
itself reports (the posterior concentration, the agreement of two halves
of the panel, the held-out evidence) can be improved by adding a gene
that carries a non-cyclic covariate -- mask intensity scores R 0.99 and
6 deg between halves while telling nothing about DNA. So a panel is
judged here by quantities the model never sees:

    dna_drop      the DAPI content's step at division, against its
                  physical value of 2 (scored as |drop - 2|)
    dna_g2_g1     the DNA content of the 90 deg before division over
                  the 90-270 deg after it (also 2)
    dna_r2        the DNA content explained by the angle (24-bin medians)
    area_r2       the same for the cell-mask area
    arrested      where an arrested population lands and how tight it is

A gene can also be refused outright, before any panel is scored, and
for one reason only: the cycle its accepted counts show has no
counterpart in the candidates they were gated from (gate_made_cycle).
Flatness and weakness are not validity failures -- see that function.

and three hard requirements: the panel must keep at least one S and one
G2/M gene (nothing orients the circle otherwise), the DAPI step must be
findable at all (no step, no division origin), and the DNA must actually
double over the cycle it draws (before/after ratio >= MIN_DNA_RATIO --
without this a search happily reached the best DNA R2 on a cycle whose
DNA content was flat).

Two searches, both reporting the same metrics:

    backward()  drop in a pre-registered order (orders() offers three:
                Fisher share, cycle amplitude, counts per cell), fixed
                BEFORE the external scoring so the curve is not tuned on
                the criterion that judges it -- but one order is one
                path, and a path can miss the best panel
    backward_greedy()  drop whichever gene leaves the highest rank; the
                mirror of forward(), to be read beside a fixed-order
                curve rather than instead of one
    forward()   start from every (S, G2/M) pair, keep the best, then add
                whichever remaining gene ranks highest: origin found
                first, arrested populations still in place second, the
                objective third (a single-metric greedy search traded
                both away -- see forward's docstring)

A pair is a ratio, so the winner of the seeding round is often a cyclic
gene against a FLAT one rather than two cyclic genes: in a composition
the flat gene is the denominator, and the ratio tracks the cyclic one
just as well. Seeds are therefore a starting point, not a verdict on a
gene's worth -- read the path, not its first step.

`stop_size` reads a backward curve: the smallest panel whose external
numbers still sit inside the bootstrap band of the BEST panel on the
curve (not of the full panel -- a poor full panel would pass anything).
"""
import numpy as np

from codelab_pipeline.analysis import cellcycle as CC

DEFAULT_SEEDS = (0, 1, 2, 3)
MIN_CELLS_FOR_STEP = 200
EXTERNAL = ('abs_drop_minus_2', 'dna_g2_over_g1', 'dna_r2', 'area_r2')
LOWER_IS_BETTER = ('abs_drop_minus_2',)


class PanelData:
    """Everything a panel score needs, once per experiment.

    X: cells x genes counts (every candidate gene), genes: their names,
    groups: the per-cell celltype, dna/area: the per-cell DNA content
    (per-FOV normalised) and mask area, roles: {gene: 'S'|'G2/M'|...},
    train: the cycling celltype (None = every cell), alpha: the DM
    concentration, gates: the on-ring rule.
    """

    def __init__(self, X, genes, groups, dna, area, roles, train=None, alpha=CC.DEFAULT_ALPHA,
                 min_total=10, gates=None, name='panel'):
        self.X = np.asarray(X, float)
        self.genes = list(genes)
        self.groups = np.asarray(groups, dtype=object)
        self.dna = np.asarray(dna, float)
        self.area = np.asarray(area, float)
        self.roles = dict(roles or {})
        self.train = train
        self.alpha = float(alpha)
        self.min_total = int(min_total)
        self.gates = dict(gates or {'min_bf_ring': 0.0, 'min_radius': 0.5})
        self.name = name
        self.gi = {g: i for i, g in enumerate(self.genes)}

    def columns(self, genes):
        return self.X[:, [self.gi[g] for g in genes]]

    def roles_of(self, genes):
        early = [g for g in genes if self.roles.get(g) == 'S']
        late = [g for g in genes if self.roles.get(g) == 'G2/M']
        return early, late


def _binned_r2(theta_deg, values, n_bins=24, min_per_bin=3):
    v = np.asarray(values, float)
    t = np.asarray(theta_deg, float)
    ok = np.isfinite(v) & np.isfinite(t)
    v, t = v[ok], t[ok]
    if len(v) < 50 or np.var(v) <= 0:
        return np.nan
    b = np.clip(np.digitize(t % 360.0, np.linspace(0, 360, n_bins + 1)) - 1, 0, n_bins - 1)
    pred = np.full(len(v), np.nan)
    for k in range(n_bins):
        sel = b == k
        if sel.sum() >= min_per_bin:
            pred[sel] = np.median(v[sel])
    ok = np.isfinite(pred)
    if ok.sum() <= 20 or np.var(v[ok]) <= 0:
        return np.nan
    return float(1.0 - np.var(v[ok] - pred[ok]) / np.var(v[ok]))


def _on_ring(placed, gates):
    on = np.ones(len(placed['theta']), bool)
    if gates.get('min_bf_ring') is not None:
        on &= placed['bf_ring'] >= float(gates['min_bf_ring'])
    if gates.get('min_radius') is not None:
        on &= placed['radius'] >= float(gates['min_radius'])
    return on


def evaluate(data, genes, seed=0, with_half_panel=False):
    """Fit `genes` (bootstrap-resampling the training cells when seed >
    0) and score the panel. Returns a dict; 'ok' is False when the panel
    cannot be oriented or its origin cannot be found."""
    genes = list(genes)
    early, late = data.roles_of(genes)
    row = {'genes': genes, 'n_genes': len(genes), 'seed': int(seed), 'n_S': len(early), 'n_G2M': len(late),
           'fitted': False, 'ok': False}
    if not early or not late:
        row['note'] = 'no S or no G2/M gene: the circle cannot be oriented'
        return row
    X = data.columns(genes)
    keep = X.sum(1) >= data.min_total
    X = X[keep]
    groups, dna, area = data.groups[keep], data.dna[keep], data.area[keep]
    k = (groups == data.train) if data.train else np.ones(len(X), bool)
    if k.sum() < 100:
        row['note'] = f'only {int(k.sum())} training cells reach the minimum panel total'
        return row
    tr = np.flatnonzero(k)
    if seed:
        tr = np.random.default_rng(int(seed)).choice(tr, len(tr), replace=True)
    hk = [g for g in genes if data.roles.get(g) == 'housekeeping'] + list(CC.HOUSEKEEPING)
    m = CC.CycleModel().fit([CC.Dataset(data.name, X[tr], genes, groups[tr], alpha=data.alpha)],
                            bridge_exclude=hk, orient_by=(early, late))
    m.orient(early, late)
    placed = m.place(data.name, X)
    th = np.degrees(placed['theta']) % 360.0
    on = _on_ring(placed, data.gates)
    cyc = k & on
    if cyc.sum() < MIN_CELLS_FOR_STEP:
        cyc = k
        row['origin_on'] = 'every training cell (too few on the ring)'
    else:
        row['origin_on'] = 'on-ring training cells'
    angle, factor = CC.total_drop_angle(th[cyc], dna[cyc])
    found = angle is not None and factor is not None and factor >= 1.3
    if found:
        th = (th - angle) % 360.0
    before = cyc & (th >= 270.0)
    after = cyc & (th >= 90.0) & (th < 270.0)
    g2g1 = (float(np.nanmedian(dna[before]) / np.nanmedian(dna[after]))
            if found and before.sum() >= 20 and after.sum() >= 20 else np.nan)
    share = m.fisher_share(data.name)
    _pk, f_pk = m.peak_phase()
    s = np.array([share[g] for g in genes], float)
    s = s / max(s.sum(), 1e-12)
    row.update({'fitted': True, 'ok': bool(found), 'cells': int(keep.sum()), 'origin_found': bool(found),
                'origin_deg': float(angle) if found else np.nan,
                'origin_drop': float(factor) if factor else np.nan,
                'abs_drop_minus_2': abs(float(factor) - 2.0) if found else np.nan,
                'dna_g2_over_g1': g2g1,
                'dna_r2': _binned_r2(th[cyc], dna[cyc]), 'area_r2': _binned_r2(th[cyc], area[cyc]),
                'on_ring': float(on[k].mean()), 'R_med': float(np.median(placed['R'][k])),
                'effective_genes': float(np.exp(-(s * np.log(s + 1e-12)).sum())),
                'inv_simpson': float(1.0 / (s ** 2).sum()), 'top_share': float(s.max()),
                'fisher_share': {g: float(share[g]) for g in genes},
                'amplitude': {g: float(np.exp(f_pk[:, m.gi[g]].max() - f_pk[:, m.gi[g]].min())) for g in genes}})
    for cond in sorted({str(c) for c in data.groups if c and c != data.train}):
        sel = on & (groups == cond)
        if sel.sum() >= 20:
            z = np.mean(np.exp(1j * np.radians(th[sel])))
            row[f'{cond}_deg'] = float(np.degrees(np.angle(z)) % 360.0)
            row[f'{cond}_conc'] = float(abs(z))
    if with_half_panel and len(genes) >= 4:
        from codelab_pipeline.analysis import figures_cellcycle as FC
        order = sorted(share, key=lambda g: -share[g])
        _t, st = FC.subpanel_agreement(m, data.name, X[k], {'A': order[0::2], 'B': order[1::2]})
        ab = st[(st['a'] == 'A') & (st['b'] == 'B')].iloc[0]
        row['half_panel_deg'] = float(ab['median_abs_diff_deg'])
    return row


def objective(row, key='dna_r2'):
    """The single number a forward search climbs. Declared up front (the
    DNA content explained by the angle) so the search is not tuned to
    whichever metric happens to move.

    A panel that cannot be fitted or oriented scores -inf. A panel whose
    DAPI step is not findable still scores: at two or three genes the
    step is often too shallow to detect, and refusing those would leave
    the search with nothing to start from -- the origin is a requirement
    of the FINAL panel (stop_size enforces it), not of every step."""
    if not row.get('fitted'):
        return -np.inf
    v = row.get(key, np.nan)
    if not np.isfinite(v):
        return -np.inf
    return -v if key in LOWER_IS_BETTER else v


def drop_order(data, genes=None, seeds=(0,)):
    """The pre-registered elimination order: the weakest Fisher share
    first, taken from one fit of the full panel."""
    genes = list(genes or data.genes)
    row = evaluate(data, genes, seed=seeds[0])
    share = row.get('fisher_share') or {g: 0.0 for g in genes}
    return sorted(genes, key=lambda g: share.get(g, 0.0))


def backward(data, genes=None, seeds=DEFAULT_SEEDS, min_genes=3, order=None, runner=None, with_half_panel=True):
    """Elimination curve: [{'size', 'dropped', 'rows': [per-seed rows]}]
    from the full panel down to min_genes, removing `order` in turn."""
    genes = list(genes or data.genes)
    order = list(order or drop_order(data, genes))
    steps, current = [], list(genes)
    while len(current) >= min_genes:
        steps.append(list(current))
        nxt = next((g for g in order if g in current), None)
        if nxt is None:
            break
        current = [g for g in current if g != nxt]
    jobs = [(list(step), int(seed)) for step in steps for seed in seeds]
    rows = _run(data, jobs, runner, with_half_panel)
    out = []
    for step in steps:
        got = [r for r in rows if r['genes'] == step]
        out.append({'size': len(step), 'genes': list(step), 'dropped': [g for g in genes if g not in step], 'rows': got})
    return out


def seeds_for_forward(data, genes=None):
    """Every (S gene, G2/M gene) pair: the smallest panel that can be
    oriented at all. Unassigned genes cannot start a panel on their own,
    so they enter through the additions."""
    genes = list(genes or data.genes)
    early, late = data.roles_of(genes)
    return [[a, b] for a in early for b in late]


def arrested_shift(row, reference, tol_deg=45.0):
    """(ok, worst shift in deg) of the arrested populations against a
    reference panel: an addition that moves an arrested population by
    more than tol_deg has changed what the circle means, whatever it did
    to the objective."""
    if not reference:
        return True, 0.0
    worst = 0.0
    for key in [k for k in reference if k.endswith('_deg') and k != 'origin_deg']:
        a, b = row.get(key), reference.get(key)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        worst = max(worst, abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))
    return worst <= float(tol_deg), worst


MIN_DNA_RATIO = 1.2
"""The smallest before/after DNA ratio a panel may report and still be
considered. The physical value is 2; the three stores read 1.45-1.83
with their full panels, and a panel that reads near 1 is saying the DNA
does not double over its own cycle -- its origin is a fitted angle with
no biology behind it. Measured: chr19's greedy elimination reached four
genes with the best DNA R2 on the board (0.263) and a ratio of 0.93."""


def rank(row, reference=None, key='dna_r2', tol_deg=45.0, min_ratio=MIN_DNA_RATIO):
    """The lexicographic score the searches maximise: a panel that finds
    its division origin, then one whose DNA actually doubles over the
    cycle (before/after ratio >= min_ratio), then one whose arrested
    populations have not moved, and only then the objective.

    Ranking this way (instead of the objective alone) is what keeps a
    search from trading the division step, the DNA doubling and the
    arrested anchors for a better DNA R2 -- each of those trades was
    observed on the real stores before the condition was added."""
    if not row.get('fitted'):
        return (-1, -1, -1, -np.inf)
    ok, _worst = arrested_shift(row, reference, tol_deg)
    ratio = row.get('dna_g2_over_g1', np.nan)
    doubles = 1 if (np.isfinite(ratio) and ratio >= float(min_ratio)) else 0
    return (1 if row.get('origin_found') else 0, doubles, 1 if ok else 0, objective(row, key))


def forward(data, genes=None, max_genes=None, seeds=DEFAULT_SEEDS, key='dna_r2', runner=None,
            tol=0.0, with_half_panel=True, min_size=4, reference=None, tol_deg=45.0):
    """Greedy addition under hard conditions: score every (S, G2/M)
    pair, keep the best, then add whichever remaining gene ranks highest
    -- origin found first, arrested populations still in place second,
    the objective third.

    The search keeps going while the panel is smaller than `min_size` or
    has no division origin, whatever the objective does; after that it
    stops when no addition improves the rank by more than `tol`.
    `reference` is a row (usually the full panel) the arrested check is
    made against; without it that check passes.

    Returns [{'size', 'genes', 'added', 'rows', 'candidates'}], the first
    entry being the winning pair and each later one a single addition;
    the search runs on seeds[0] and every kept panel is re-scored on all
    seeds."""
    genes = list(genes or data.genes)
    max_genes = int(max_genes or len(genes))
    pairs = seeds_for_forward(data, genes)
    if not pairs:
        raise ValueError('the panel has no (S, G2/M) pair: nothing can orient the circle')
    search_seed = int(seeds[0])
    if reference is None:
        reference = _run(data, [(list(genes), search_seed)], runner, False)[0]

    def _rank(r):
        return rank(r, reference, key, tol_deg)
    rows = _run(data, [(p, search_seed) for p in pairs], runner, False)
    best = max(rows, key=_rank)
    if not np.isfinite(_rank(best)[2]):
        raise ValueError('no (S, G2/M) pair could be fitted')
    current = list(best['genes'])
    score = _rank(best)
    path = [{'size': len(current), 'genes': list(current), 'added': list(current),
             'rows': _run(data, [(list(current), int(s)) for s in seeds], runner, with_half_panel),
             'candidates': [{'genes': r['genes'], 'rank': _rank(r)} for r in rows]}]
    while len(current) < max_genes:
        rest = [g for g in genes if g not in current]
        if not rest:
            break
        tried = _run(data, [(current + [g], search_seed) for g in rest], runner, False)
        cand = max(tried, key=_rank)
        cand_score = _rank(cand)
        must_grow = len(current) < int(min_size) or score[0] < 1
        if not must_grow and cand_score <= score[:-1] + (score[-1] + tol,):
            break
        if not np.isfinite(cand_score[2]):
            break
        added = [g for g in cand['genes'] if g not in current]
        current = list(cand['genes'])
        score = cand_score
        path.append({'size': len(current), 'genes': list(current), 'added': added,
                     'rows': _run(data, [(list(current), int(s)) for s in seeds], runner, with_half_panel),
                     'candidates': [{'genes': r['genes'], 'rank': _rank(r)} for r in tried]})
    return path


def _run(data, jobs, runner, with_half_panel):
    if runner is None:
        return [evaluate(data, g, seed=s, with_half_panel=with_half_panel) for g, s in jobs]
    return runner(data, jobs, with_half_panel)


def band(entry, key):
    """(seed-0 value, min, max) of a metric over an entry's seeds."""
    vals = [r.get(key, np.nan) for r in entry['rows'] if r.get('ok')]
    vals = [v for v in vals if np.isfinite(v)]
    first = next((r.get(key, np.nan) for r in entry['rows'] if r.get('seed') == 0), np.nan)
    if not vals:
        return first, np.nan, np.nan
    return first, float(min(vals)), float(max(vals))


def entry_rank(entry, reference=None, key='dna_r2', tol_deg=45.0, min_ratio=MIN_DNA_RATIO):
    """rank() of an entry's seed-0 row."""
    row = next((r for r in entry['rows'] if r.get('seed') == 0), (entry['rows'] or [{}])[0])
    return rank(row, reference, key, tol_deg, min_ratio)


def best_entry(curve, key='dna_r2', reference=None, tol_deg=45.0, min_ratio=MIN_DNA_RATIO):
    """The entry a curve should be judged against: the highest-ranking
    one -- origin found, DNA doubling, arrested populations in place,
    then the objective.

    NOT the full panel: anchoring the stop rule there asks only 'is this
    no worse than everything together', and when the full panel is poor
    the rule passes anything (chr19's eleven genes read a before/after
    DNA ratio of 1.09, so every smaller panel cleared that bar while
    better ones existed). And not the objective alone: on the same store
    a four-gene panel had the best DNA R2 with a ratio of 0.93."""
    if not curve:
        return None
    ref = reference if reference is not None else next((r for r in curve[0]['rows'] if r.get('seed') == 0), None)
    ok = [e for e in curve if any(r.get('origin_found') for r in e['rows'])]
    if not ok:
        return None
    return max(ok, key=lambda e: entry_rank(e, ref, key, tol_deg, min_ratio))


def stop_size(curve, keys=EXTERNAL, reference=None, key='dna_r2'):
    """The smallest panel of a backward curve whose external metrics all
    stay inside the reference panel's bootstrap band. The reference is
    the best panel on the curve (best_entry) unless one is given: a panel
    is kept because it matches the best result, not because it matches
    the biggest panel. A panel that cannot be oriented, or whose origin
    is not found, never qualifies."""
    if not curve:
        return None
    ref = reference if reference is not None else (best_entry(curve, key) or curve[0])
    bands = {k: band(ref, k)[1:] for k in keys}
    best = None
    for entry in curve:
        if not any(r.get('ok') for r in entry['rows']):
            continue          # not fitted, or no division origin: never a final panel
        ok = True
        for k in keys:
            v, lo, hi = band(entry, k)
            rlo, rhi = bands[k]
            if not np.isfinite(v) or not np.isfinite(rlo):
                continue
            if k in LOWER_IS_BETTER:
                if v > rhi:
                    ok = False
            elif v < rlo:
                ok = False
        if ok:
            best = entry
    return best

def orders(data, genes=None, seeds=(0,), runner=None):
    """Pre-registered elimination orders, weakest first, each computed
    from ONE fit of the full panel and fixed before any external
    scoring:

        fisher     the share of the phase information a gene carries
        amplitude  the gene's own cycle signal, log(max pi / min pi)
        counts     the median counts per cell (a weak round first)

    A single order explores a single path; running all three and taking
    the best entry across them costs three curves and covers the panels
    one order alone would miss."""
    genes = list(genes or data.genes)
    row = _run(data, [(genes, int(seeds[0]))], runner, False)[0]
    share = row.get('fisher_share') or {g: 0.0 for g in genes}
    X = data.columns(genes)
    counts = {g: float(np.median(X[:, i])) for i, g in enumerate(genes)}
    amp = row.get('amplitude') or {}
    out = {'fisher': sorted(genes, key=lambda g: share.get(g, 0.0)),
           'counts': sorted(genes, key=lambda g: counts.get(g, 0.0))}
    if amp:
        out['amplitude'] = sorted(genes, key=lambda g: amp.get(g, 0.0))
    return out


def backward_greedy(data, genes=None, seeds=DEFAULT_SEEDS, min_genes=3, runner=None, key='dna_r2',
                    reference=None, tol_deg=45.0, with_half_panel=True):
    """Elimination that chooses what to drop at every step: the gene
    whose removal leaves the highest rank (origin found, arrested
    populations within tol_deg of the reference, then the objective).

    This is the mirror of forward() and carries the same warning: it
    optimises the criterion it is scored by, so read it next to a
    fixed-order curve rather than instead of one.

    Returns the same shape as backward()."""
    genes = list(genes or data.genes)
    if reference is None:
        reference = _run(data, [(list(genes), int(seeds[0]))], runner, False)[0]

    def _rank(r):
        return rank(r, reference, key, tol_deg)
    current = list(genes)
    out = [{'size': len(current), 'genes': list(current), 'dropped': [],
            'rows': _run(data, [(list(current), int(s)) for s in seeds], runner, with_half_panel)}]
    while len(current) > min_genes:
        tried = _run(data, [([g for g in current if g != drop], int(seeds[0])) for drop in current], runner, False)
        tried = [r for r in tried if r.get('fitted')]
        if not tried:
            break
        best = max(tried, key=_rank)
        current = list(best['genes'])
        out.append({'size': len(current), 'genes': list(current), 'dropped': [g for g in genes if g not in current],
                    'rows': _run(data, [(list(current), int(s)) for s in seeds], runner, with_half_panel)})
    return out


def best_of(curves, key='dna_r2', reference=None, tol_deg=45.0, min_ratio=MIN_DNA_RATIO):
    """The highest-ranking entry across several curves (a dict of name ->
    curve, or a list of curves), by the same rank the searches use --
    not by the objective alone. Returns (name, entry)."""
    items = list(curves.items()) if isinstance(curves, dict) else list(enumerate(curves))
    ref = reference
    if ref is None and items:
        first = items[0][1]
        ref = next((r for r in first[0]['rows'] if r.get('seed') == 0), None) if first else None
    best = (None, None, None)
    for name, curve in items:
        e = best_entry(curve, key, ref, tol_deg, min_ratio)
        if e is None:
            continue
        sc = entry_rank(e, ref, key, tol_deg, min_ratio)
        if best[2] is None or sc > best[2]:
            best = (name, e, sc)
    return best[0], best[1]

# -- the gene-level validity screen ------------------------------------------

MIN_ACC_SWING = 1.5
"""An accepted profile flatter than this has no cycle to explain, so the
screen has nothing to judge and passes the gene."""

MIN_PROFILE_RHO = 0.3
"""How much the accepted profile must follow the candidate profile. A
real gene's counts rise because there are more transcripts to find:
its candidates and its accepted counts move together (measured: 0.46 to
0.99 over 33 of 34 rounds in three stores). A cycle the gate invented
does not (chr19's GMNN: -0.02)."""


def profile_agreement(cand_profile, acc_profile):
    """(rho, candidate swing, accepted swing) of a gene's two profiles
    along the cycle -- the medians per angle bin of the candidates the
    detector proposed and of the spots the gate accepted."""
    from scipy.stats import spearmanr
    c = np.asarray(cand_profile, float)
    a = np.asarray(acc_profile, float)
    ok = np.isfinite(c) & np.isfinite(a)
    if ok.sum() >= 6 and np.std(c[ok]) > 0 and np.std(a[ok]) > 0:
        rho = float(spearmanr(c[ok], a[ok])[0])
    elif ok.sum() >= 6:
        # a constant profile predicts nothing: a candidate pool that does
        # not move cannot account for accepted counts that do
        rho = 0.0
    else:
        rho = np.nan
    cs = float(np.nanmax(c) / max(np.nanmin(c), 1.0)) if ok.any() else np.nan
    as_ = float(np.nanmax(a) / max(np.nanmin(a), 0.5)) if ok.any() else np.nan
    return rho, cs, as_


def gate_made_cycle(cand_profile, acc_profile, min_acc_swing=MIN_ACC_SWING, min_rho=MIN_PROFILE_RHO):
    """True when a gene's accepted counts cycle (swing >= min_acc_swing)
    while the candidates they were gated from do not follow that cycle
    (rho < min_rho): the profile the model would read from this gene was
    made by the classifier, not by the transcripts.

    This is the ONLY validity reason to refuse a gene. Flatness is not
    one (a flat gene is a composition's denominator) and neither is a
    low contribution (the panel search decides that, on what the model
    never saw)."""
    rho, _cs, acc_swing = profile_agreement(cand_profile, acc_profile)
    if not np.isfinite(rho) or not np.isfinite(acc_swing):
        return False
    return bool(acc_swing >= float(min_acc_swing) and rho < float(min_rho))


def screen_genes(profiles, min_acc_swing=MIN_ACC_SWING, min_rho=MIN_PROFILE_RHO):
    """profiles: {gene: (cand_profile, acc_profile)} -> {gene: row} with
    rho, the two swings and 'refused'."""
    out = {}
    for gene, (c, a) in profiles.items():
        rho, cs, as_ = profile_agreement(c, a)
        out[gene] = {'gene': gene, 'rho': rho, 'cand_swing': cs, 'acc_swing': as_,
                     'refused': gate_made_cycle(c, a, min_acc_swing, min_rho)}
    return out
