"""
Cell-cycle phase from the composition of an RNA panel: one circle, many
experiments, three per-cell verdicts.

THE MODEL (settled 2026-09-13 after validation on chr19_downstream and
the two synchronised experiments JP_001 / JP_002; see
notes/cellcycle and the design document)

Every cell i sits at a latent angle theta_i on a circle. Every gene g has
a periodic log-profile shared by every experiment,

    f_g(theta) = sum_{k<=K} b_gk cos(k theta) + c_gk sin(k theta),

an experiment d has an intercept a_dg per gene it measured (probe,
round, detection -- the only per-experiment parameter besides the
dispersion), and the expected composition of d's panel at theta is

    pi_dg(theta) = softmax_g( a_dg + f_g(theta) )    over S_d, d's panel.

Counts are Dirichlet-multinomial given the cell's panel total s_i:
conditioning on the total is what makes cell size, mask depth and
detection efficiency cancel, and alpha_d is the between-cell dispersion
of composition at one phase (chosen per experiment by held-out
evidence: chr19 100, JP_001 300, JP_002 100). Each (experiment,
condition) group carries its own free, smoothed prior over the grid --
the SPECTRUM, which is a result, not an input.

WHAT IS FITTED ON WHAT. Profiles and intercepts are fitted on CYCLING
populations only (unsynchronised cells; every chr19 condition). Arrested
or otherwise perturbed populations are PLACED afterwards under a uniform
prior: fitting them would let the profiles absorb the condition
difference and split the conditions into clusters (measured:
unsynchronised concentration 0.27 -> 0.91).

THREE VERDICTS PER CELL, because posterior concentration alone cannot
tell them apart (locally the posterior is von Mises with concentration
kappa = s_eff * r^2 * rho -- effective count x ring radius^2 x the
cell's own radial position, s_eff = s(1+alpha)/(s+alpha)):
    R        posterior concentration: how sharply the angle is known
    fit_z    KL distance to the ring against cycling cells of the same
             total: OUTSIDE the ring (drug response, doublets, debris)
    bf_ring  log Bayes factor ring-vs-centre (centre = softmax(a_d),
             exactly the CLR centre of the ring): INSIDE the ring, the
             composition of a cell with no phase (G0-like, transcription
             shut down). Measured on JP_001 hydroxyurea: radius 0.45 of
             the ring with fit_z +0.5 and, at alpha 300, R 0.84 -- a
             confidently random angle that only this verdict flags.

FITTING. EM on a T-point grid. E-step closed form (DM). M-step by
default on the multinomial sufficient statistics S_tg = sum_i q_it x_ig
and N_t = sum_i q_it s_i -- a T x G problem, seconds not minutes -- with
the DM evidence tracked every iteration, the best parameters kept, a
patience stop, and a final DM polish from the best point
(mstep='dm' runs the exact DM M-step throughout). The prior update is
closed form (group mean posterior, circularly smoothed). Initialisation:
CLR-PCA angle per experiment, then the relative rotation/reflection of
experiments from the peaks of shared genes by weighted circular least
squares (weight = the smaller of the two single-fit Fisher
informations), housekeeping genes excluded from the bridge FIRST, then
a residual outlier drop -- measured: weighting cut JP_001's residual
24 -> 19 deg, but with GAPDH allowed the outlier rule removed the true
bridge (GAPDH and CCNB1 agreed on the wrong reflection), so exclusion is
not optional. Orientation: the S-list genes' mean peak is 0 deg and the
G2/M-list mean peak lies within the forward half turn.

Nothing here imports Qt or the store; the app builds the count tables
(gene_table) from a Population and calls in.
"""
import numpy as np
from scipy import optimize
from scipy.special import logsumexp, gammaln, digamma

TWO_PI = 2.0 * np.pi
DEFAULT_K = 2
DEFAULT_T = 72
DEFAULT_ALPHA = 100.0
HOUSEKEEPING = ('GAPDH', 'ACTB', 'TUBB', 'RPLP0', 'HPRT1')
# Default gene roles for ORIENTATION only (0 = the S genes' mean peak,
# the G2/M genes' mean peak in the forward half turn): the Seurat /
# Tirosh 2016 cell-cycle lists, plus the G1/S and mitotic cyclins the
# synchronised panels carry (CCNE1, CDT1; CCNB1). A panel gene absent
# from both lists takes no part in orientation; the stage lets the user
# change every role.
S_GENES = ('MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6',
           'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'CENPU', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP',
           'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2',
           'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2',
           'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8', 'CCNE1', 'CDT1')
G2M_GENES = ('HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2',
             'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'PIMREG', 'FAM64A', 'SMC4',
             'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1',
             'KIF20B', 'HJURP', 'CDCA3', 'JPT1', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C',
             'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', 'AURKA',
             'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5',
             'CENPA', 'CCNB1')
ROLES = ('-', 'S', 'G2/M', 'housekeeping')


def default_role(gene):
    """The role a gene name gets before the user touches it."""
    g = str(gene).upper()
    if g in HOUSEKEEPING:
        return 'housekeeping'
    if g in S_GENES:
        return 'S'
    if g in G2M_GENES:
        return 'G2/M'
    return '-'


def gene_from_readout(name):
    """'MCM2_mRNA' / 'CCNB1_exon' -> 'MCM2' / 'CCNB1'; a nascent, intron,
    repeat or toe round keeps its full name so it never silently counts
    as the gene (those rounds are not usable as counts -- design 3b)."""
    name = str(name or '')
    if name.startswith(('Rep_', 'Toe_')):
        return name
    for suffix in ('_mRNA', '_exon'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def countable_round(name):
    """Whether a readout name looks like an mRNA/exon round the stage
    should count by default."""
    name = str(name or '')
    return name.endswith(('_mRNA', '_exon')) and not name.startswith(('Rep_', 'Toe_'))


# -- small pieces -----------------------------------------------------------

def _basis(theta, K):
    """(T, 2K): [cos t, sin t, cos 2t, sin 2t, ...] -- harmonics only."""
    cols = []
    for k in range(1, K + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    return np.stack(cols, axis=1)


def _dbasis(theta, K):
    cols = []
    for k in range(1, K + 1):
        cols.append(-k * np.sin(k * theta))
        cols.append(k * np.cos(k * theta))
    return np.stack(cols, axis=1)


def _circ_smooth(w, width):
    """Circular moving average; clipped at 0 because the FFT rings by
    ~1e-17 around a sharp peak and log(negative) is NaN."""
    if width <= 1:
        return w
    T = len(w)
    k = np.zeros(T)
    h = int(width) // 2
    for d in range(-h, h + 1):
        k[d % T] = 1.0
    k /= k.sum()
    return np.maximum(np.real(np.fft.ifft(np.fft.fft(w) * np.fft.fft(k))), 0.0)


def clr(F):
    L = np.log(F)
    return L - L.mean(1, keepdims=True)


def pca_angle(X, pseudo=0.5):
    """Initial angle per cell: atan2 of the first two PCs of the per-gene
    z-scored CLR fractions (the tricycle idea)."""
    Z = clr(np.asarray(X, float) + pseudo)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    return np.arctan2(pcs[:, 1], pcs[:, 0]) % TWO_PI, pcs, (S ** 2) / (S ** 2).sum()


def circular_agreement(a, b):
    """(median |error| as a fraction of a turn, sign, shift): the best
    rigid alignment of two phase estimates of the same cells."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    best = None
    for sgn in (1.0, -1.0):
        d = a - sgn * b
        shift = np.angle(np.mean(np.exp(1j * d)))
        err = np.abs(np.angle(np.exp(1j * (d - shift))))
        med = float(np.median(err)) / TWO_PI
        if best is None or med < best[0]:
            best = (med, sgn, shift)
    return best


def _dm_loglik(X, api):
    """(n, T): sum_g lgamma(x + a pi) - lgamma(a pi) for api (T, G)."""
    return (gammaln(X[:, None, :] + api[None]) - gammaln(api)[None]).sum(2)


def _dm_const(X, alpha):
    s = X.sum(1)
    c = gammaln(s + 1) - gammaln(X + 1).sum(1)
    if alpha is None:
        return c
    return c + gammaln(alpha) - gammaln(s + alpha)


class Dataset:
    """One experiment's counts: X (n, |genes|), gene names in column order,
    a condition label per cell, and the dispersion alpha (None =
    multinomial)."""

    def __init__(self, name, X, genes, groups=None, alpha=DEFAULT_ALPHA):
        self.name = str(name)
        self.X = np.asarray(X, float)
        self.genes = [str(g) for g in genes]
        if self.X.ndim != 2 or self.X.shape[1] != len(self.genes):
            raise ValueError(f'{name}: X is {self.X.shape}, genes are {len(self.genes)}')
        self.groups = (np.asarray(groups, dtype=object) if groups is not None
                       else np.array(['all'] * len(self.X), dtype=object))
        self.alpha = None if alpha is None else float(alpha)


# -- the model ----------------------------------------------------------------

class CycleModel:
    def __init__(self, K=DEFAULT_K, T=DEFAULT_T, prior_smooth=5, l2=1e-2,
                 mstep='multinomial', patience=5, polish=2, m_iter=60):
        if mstep not in ('multinomial', 'dm'):
            raise ValueError("mstep must be 'multinomial' or 'dm'")
        self.K, self.T = int(K), int(T)
        self.prior_smooth = int(prior_smooth)
        self.l2 = float(l2)
        self.mstep = mstep
        self.patience = int(patience)
        self.polish = int(polish)
        self.m_iter = int(m_iter)
        self.grid = np.arange(self.T) * TWO_PI / self.T
        self.B = _basis(self.grid, self.K)
        self.genes = []
        self.coef = None            # (G, 2K) shared harmonics
        self.a = {}                 # name -> (|S_d|,) intercepts
        self.w = {}                 # (name, group) -> (T,) spectrum
        self.alphas = {}            # name -> alpha
        self.panels = {}            # name -> [genes]
        self.cols = {}              # name -> gene indices into self.genes
        self.ref = {}               # name -> fit-quality reference (edges, mean, sd)
        self.history = []
        self.align_report = {}
        self.q_ = {}

    # -- bookkeeping --

    def _index(self, datasets):
        genes = []
        for d in datasets:
            for g in d.genes:
                if g not in genes:
                    genes.append(g)
        self.genes = genes
        self.gi = {g: i for i, g in enumerate(genes)}
        for d in datasets:
            self.panels[d.name] = list(d.genes)
            self.cols[d.name] = np.array([self.gi[g] for g in d.genes])
            self.alphas[d.name] = d.alpha

    def bridge_report(self):
        names = list(self.panels)
        return {(a, b): sorted(set(self.panels[a]) & set(self.panels[b]))
                for i, a in enumerate(names) for b in names[i + 1:]}

    # -- composition and likelihood --

    def log_pi(self, name, coef=None, a=None, subset=None):
        """(T, |subset|) log composition of `name`'s panel (or a subset
        of it) on the grid."""
        coef = self.coef if coef is None else coef
        a = self.a[name] if a is None else a
        if subset is None:
            gidx, aidx = self.cols[name], np.arange(len(self.panels[name]))
        else:
            aidx = np.array([self.panels[name].index(g) for g in subset])
            gidx = np.array([self.gi[g] for g in subset])
        eta = self.B @ coef[gidx].T + a[aidx][None, :]
        return eta - logsumexp(eta, axis=1, keepdims=True)

    def loglik_grid(self, name, X, coef=None, a=None, subset=None, alpha='own'):
        alpha = self.alphas[name] if alpha == 'own' else alpha
        lpi = self.log_pi(name, coef, a, subset)
        if alpha is None:
            return X @ lpi.T
        return _dm_loglik(X, alpha * np.exp(lpi))

    def posterior(self, name, X, groups=None, prior='spectrum', subset=None):
        """(n, T). prior='spectrum' uses the fitted group spectra (groups
        required); prior='uniform' asks the counts alone -- the PLACEMENT
        prior for populations that were not fitted."""
        ll = self.loglik_grid(name, X, subset=subset)
        if prior == 'uniform' or groups is None:
            lp = ll
        else:
            groups = np.asarray(groups, dtype=object)
            lp = np.empty_like(ll)
            for c in np.unique(groups):
                k = groups == c
                w = self.w.get((name, c), np.full(self.T, 1.0 / self.T))
                lp[k] = ll[k] + np.log(w + 1e-300)[None, :]
        lp = lp - lp.max(1, keepdims=True)
        q = np.exp(lp)
        return q / q.sum(1, keepdims=True)

    def evidence(self, name, X, groups=None, prior='spectrum'):
        """Per-cell log marginal likelihood (proper, alpha-comparable)."""
        ll = self.loglik_grid(name, X)
        if prior == 'uniform' or groups is None:
            out = logsumexp(ll, axis=1) - np.log(self.T)
        else:
            groups = np.asarray(groups, dtype=object)
            out = np.empty(len(X))
            for c in np.unique(groups):
                k = groups == c
                w = self.w.get((name, c), np.full(self.T, 1.0 / self.T))
                out[k] = logsumexp(ll[k] + np.log(w + 1e-300)[None, :], axis=1)
        return out + _dm_const(X, self.alphas[name])

    # -- fitting --

    def _unpack(self, v):
        G, P = len(self.genes), self.B.shape[1]
        coef = v[:G * P].reshape(G, P)
        a, off = {}, G * P
        for name in self.panels:
            n = len(self.cols[name])
            a[name] = v[off:off + n]
            off += n
        return coef, a

    def _mstep(self, datasets, qs, exact):
        """One M-step over shared harmonics and every experiment's
        intercepts. exact=False folds each dataset to its multinomial
        sufficient statistics; exact=True uses the DM objective."""
        G, P = len(self.genes), self.B.shape[1]
        x0 = np.concatenate([self.coef.ravel()] + [self.a[d.name] for d in datasets])
        pre = {}
        for d in datasets:
            q = qs[d.name]
            s = d.X.sum(1)
            # THE DM'S OWN WEIGHTING, kept in the fast path. A Dirichlet-
            # multinomial cell carries s(1+alpha)/(s+alpha) effective counts,
            # never more than ~alpha; the plain multinomial statistics would
            # weight a 1000-count JP_001 cell ten times a 100-count chr19
            # cell and let the deepest experiment own every shared profile
            # (measured on the real joint fit: chr19 collapsed to one lobe,
            # the two nocodazole anchors drifted 90 deg apart). Scaling each
            # cell's counts to its effective count restores the balance the
            # exact M-step has; the polish then refines from there.
            w = np.ones_like(s) if d.alpha is None else (1.0 + d.alpha) / (s + d.alpha)
            pre[d.name] = (q, q.T @ (d.X * w[:, None]), q.T @ (s * w))

        def f(v):
            coef, a = self._unpack(v)
            val = 0.5 * self.l2 * (coef ** 2).sum()
            gcoef = self.l2 * coef
            ga = {}
            for d in datasets:
                q, Sx, Ss = pre[d.name]
                lpi = self.log_pi(d.name, coef, a[d.name])
                pi = np.exp(lpi)
                if not exact or d.alpha is None:
                    val -= (Sx * lpi).sum()
                    g_eta = -(Sx - Ss[:, None] * pi)
                else:
                    api = d.alpha * pi
                    Xb = d.X[:, None, :]
                    val -= (q[:, :, None] * (gammaln(Xb + api[None]) - gammaln(api)[None])).sum()
                    dpi = d.alpha * (q[:, :, None] * (digamma(Xb + api[None]) - digamma(api)[None])).sum(0)
                    g_eta = -(pi * (dpi - (pi * dpi).sum(1, keepdims=True)))
                gcoef[self.cols[d.name]] += g_eta.T @ self.B
                ga[d.name] = g_eta.sum(0)
            return val, np.concatenate([gcoef.ravel()] + [ga[d.name] for d in datasets])

        res = optimize.minimize(f, x0, jac=True, method='L-BFGS-B',
                                options={'maxiter': self.m_iter})
        coef, a = self._unpack(res.x)
        for name in a:
            a[name] = a[name] - a[name].mean()          # softmax gauge
        return coef, a

    def _update_w(self, datasets, qs):
        for d in datasets:
            for c in np.unique(d.groups):
                k = d.groups == c
                w = _circ_smooth(qs[d.name][k].mean(0), self.prior_smooth)
                self.w[(d.name, c)] = w / w.sum()

    def _mean_evidence(self, datasets):
        return float(np.mean(np.concatenate(
            [self.evidence(d.name, d.X, d.groups) for d in datasets])))

    def stagewise_init(self, datasets, bridge_exclude=HOUSEKEEPING, weights='fisher_min',
                       robust=True, verbose=False, orient_by=None):
        """Initial angles that agree ACROSS experiments (see the module
        docstring). Returns {name: theta0}.

        orient_by=(early, late): gene-role lists. When given, every
        single fit that holds at least one gene of each list is FIRST
        oriented by them (early mean peak at 0, late in the forward
        half turn), and the bridge between two oriented experiments
        then solves the rotation only -- the reflection is fixed by
        the biology, never by the bridge. Measured need: with three
        shared genes (chr19-JP_002: GMNN, CCNA2, CCNB1) the bridge's
        reflection choice flipped between two fits of the same cells
        whose counts differed at the 1% level (the hydroxyurea anchor
        landed at 45 deg in one and 268 deg in the other).
        """
        singles, peaks, info, oriented = {}, {}, {}, set()
        for d in datasets:
            # the dataset's OWN dispersion in these single fits, not the
            # multinomial: measured on a synthetic bridge experiment with
            # weak genes, the multinomial single fit landed 35 deg off
            # (a local optimum the over-confident E-step cannot leave)
            # while the DM fit of the same cells was within 9 deg -- and
            # every experiment aligned to that anchor inherited the error
            m = CycleModel(K=self.K, T=self.T, prior_smooth=self.prior_smooth, l2=self.l2,
                           mstep=self.mstep, polish=self.polish, patience=self.patience)
            m._fit_single(Dataset(d.name, d.X, d.genes, d.groups, alpha=d.alpha))
            if orient_by is not None:
                early = [g for g in orient_by[0] if g in m.gi]
                late = [g for g in orient_by[1] if g in m.gi]
                if early and late:
                    m.orient(early, late)
                    oriented.add(d.name)
            singles[d.name] = m.phase(d.name, d.X, prior='uniform')[0]
            pk, _ = m.peak_phase()
            peaks[d.name] = dict(zip(m.genes, pk))
            fi = m.fisher_information(d.name)
            ws = [w for (n, _c), w in m.w.items() if n == d.name]
            wmean = np.mean(ws, axis=0) if ws else np.full(self.T, 1.0 / self.T)
            info[d.name] = dict(zip(m.panels[d.name], (fi * wmean[:, None]).sum(0)))
        names = [d.name for d in datasets]
        excl = set(bridge_exclude or ())
        shared = {(a, b): sorted((set(peaks[a]) & set(peaks[b])) - excl)
                  for a in names for b in names if a != b}
        anchor = max(names, key=lambda n: sum(len(shared[(n, b)]) > 0 for b in names if b != n))
        theta0, placed, frontier = {anchor: singles[anchor]}, {anchor}, [anchor]
        self.align_report = {}
        while frontier:
            ref = frontier.pop(0)
            for n in names:
                if n in placed or not shared[(ref, n)]:
                    continue
                genes = list(shared[(ref, n)])
                if weights == 'fisher_min':
                    wg = np.array([min(info[ref][g], info[n][g]) for g in genes])
                elif weights == 'fisher_gmean':
                    wg = np.array([np.sqrt(info[ref][g] * info[n][g]) for g in genes])
                else:
                    wg = np.ones(len(genes))
                wg = wg / max(wg.sum(), 1e-12)

                # both experiments oriented by their role genes: the
                # reflection is settled, the bridge only rotates
                signs = (1.0,) if (ref in oriented and n in oriented) else (1.0, -1.0)

                def solve(gs, w):
                    pr = np.array([peaks[ref][g] for g in gs])
                    pn = np.array([peaks[n][g] for g in gs])
                    best = None
                    for sgn in signs:
                        dd = pr - sgn * pn
                        shift = np.angle(np.sum(w * np.exp(1j * dd)))
                        r = np.abs(np.angle(np.exp(1j * (dd - shift))))
                        err = float((w * r).sum() / w.sum())
                        if best is None or err < best[0]:
                            best = (err, sgn, shift, r)
                    return best

                err, sgn, shift, r = solve(genes, wg)
                dropped = []
                if robust and len(genes) >= 3:
                    med = np.median(r)
                    bad = [g for g, ri in zip(genes, r) if ri > 3 * med and ri > np.radians(20)]
                    # never drop below three genes: with two the reflection
                    # rests on one difference and a single noisy peak flips it
                    if bad and len(genes) - len(bad) >= 3:
                        keep = [i for i, g in enumerate(genes) if g not in bad]
                        err, sgn, shift, r = solve([genes[i] for i in keep], wg[keep] / wg[keep].sum())
                        dropped, genes = bad, [genes[i] for i in keep]
                self.align_report[(ref, n)] = {
                    'genes': genes, 'dropped': dropped, 'flip': bool(sgn < 0),
                    'reflection_fixed_by_roles': bool(len(signs) == 1),
                    'shift_deg': float(np.degrees(shift)), 'err_deg': float(np.degrees(err)),
                    'residual_deg': dict(zip(genes, np.round(np.degrees(r), 1)))}
                theta0[n] = (sgn * singles[n] + shift) % TWO_PI
                peaks[n] = {g: (sgn * v + shift) % TWO_PI for g, v in peaks[n].items()}
                placed.add(n)
                frontier.append(n)
                if verbose:
                    print(f'  init: {n} onto {ref} via {genes}: flip {sgn < 0}, shift '
                          f'{np.degrees(shift):.0f} deg, residual {np.degrees(err):.0f} deg'
                          + (f', dropped {dropped}' if dropped else ''), flush=True)
        for n in names:
            if n not in placed:
                theta0[n] = singles[n]
                self.align_report[(None, n)] = {'genes': [], 'note': 'no bridge: own frame'}
        return theta0

    def _fit_single(self, d, n_iter=40, tol=1e-4):
        """A single dataset from its own CLR-PCA angle -- the piece the
        stagewise initialisation runs per experiment."""
        self._index([d])
        theta0 = {d.name: pca_angle(d.X)[0]}
        return self._em([d], theta0, n_iter, tol, verbose=False)

    def _em(self, datasets, theta0, n_iter, tol, verbose):
        G, P = len(self.genes), self.B.shape[1]
        self.coef = np.zeros((G, P))
        self.a = {d.name: np.zeros(len(d.genes)) for d in datasets}
        qs = {}
        for d in datasets:
            dd = np.angle(np.exp(1j * (self.grid[None, :] - theta0[d.name][:, None])))
            q = np.exp(np.cos(dd) * 4.0)
            qs[d.name] = q / q.sum(1, keepdims=True)
        exact = self.mstep == 'dm'
        best, best_ev, since = None, -np.inf, 0
        self.history = []
        prev = -np.inf
        for it in range(int(n_iter)):
            self.coef, self.a = self._mstep(datasets, qs, exact)
            self._update_w(datasets, qs)
            qs = {d.name: self.posterior(d.name, d.X, d.groups) for d in datasets}
            ev = self._mean_evidence(datasets)
            self.history.append(ev)
            if verbose:
                print(f'  iter {it}: mean evidence {ev:.4f}', flush=True)
            if ev > best_ev + 1e-12:
                best_ev, since = ev, 0
                best = (self.coef.copy(), {k: v.copy() for k, v in self.a.items()},
                        {k: v.copy() for k, v in self.w.items()})
            else:
                since += 1
            if abs(ev - prev) < tol or since >= self.patience:
                break
            prev = ev
        if best is not None:
            self.coef, self.a, self.w = best
        # polish: exact DM M-steps from the best point (no-op if every
        # alpha is None or polish == 0)
        if self.polish and not exact and any(d.alpha is not None for d in datasets):
            for _ in range(self.polish):
                qs = {d.name: self.posterior(d.name, d.X, d.groups) for d in datasets}
                coef, a = self._mstep(datasets, qs, True)
                ev = None
                old = (self.coef, self.a)
                self.coef, self.a = coef, a
                self._update_w(datasets, qs)
                ev = self._mean_evidence(datasets)
                if ev < best_ev:                 # a polish that hurts is undone
                    self.coef, self.a = old
                    break
                best_ev = ev
                self.history.append(ev)
        self.q_ = {d.name: self.posterior(d.name, d.X, d.groups) for d in datasets}
        return self

    def fit(self, datasets, n_iter=40, tol=1e-4, bridge_exclude=HOUSEKEEPING,
            weights='fisher_min', robust=True, verbose=False, theta0=None, orient_by=None):
        """Fit on cycling populations. `datasets`: [Dataset, ...].
        orient_by=(early, late) fixes every bridge's reflection by gene
        roles (see stagewise_init)."""
        datasets = list(datasets)
        self._index(datasets)
        if theta0 is None:
            theta0 = (self.stagewise_init(datasets, bridge_exclude, weights, robust, verbose, orient_by)
                      if len(datasets) > 1 else {datasets[0].name: pca_angle(datasets[0].X)[0]})
        self._em(datasets, theta0, n_iter, tol, verbose)
        # fit-quality reference: the training cells' per-count evidence
        # under a UNIFORM prior, by octile of the panel total
        for d in datasets:
            fq = self.fit_quality(d.name, d.X)
            s = d.X.sum(1)
            edges = np.quantile(s, np.linspace(0, 1, 9))
            b = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, 7)
            mean = np.array([fq[b == j].mean() if (b == j).sum() > 5 else np.nan for j in range(8)])
            sd = np.array([fq[b == j].std() if (b == j).sum() > 5 else np.nan for j in range(8)])
            self.ref[d.name] = {'edges': edges, 'mean': mean, 'sd': sd}
        return self

    # -- orientation --

    def peak_phase(self):
        f = self.B @ self.coef.T
        return self.grid[np.argmax(f, axis=0)], f

    def orient(self, early, late):
        """Rotate/flip so the mean peak of the `early` genes (names) is
        0 and the `late` genes' mean peak lies in the forward half turn.
        Returns (shift, flip); posteriors and spectra are carried along."""
        peaks, _ = self.peak_phase()
        e = [self.gi[g] for g in early if g in self.gi]
        l_ = [self.gi[g] for g in late if g in self.gi]
        if not e or not l_:
            raise ValueError('orient needs at least one early and one late gene in the model')
        pe = np.angle(np.mean(np.exp(1j * peaks[e])))
        pl = np.angle(np.mean(np.exp(1j * peaks[l_])))
        flip = ((pl - pe) % TWO_PI) > np.pi
        shift = pe
        sgn = -1.0 if flip else 1.0
        new = self.coef.copy()
        for k in range(1, self.K + 1):
            b, c = self.coef[:, 2 * k - 2], self.coef[:, 2 * k - 1]
            cs, sn = np.cos(k * shift), np.sin(k * shift)
            new[:, 2 * k - 2] = b * cs + c * sn
            new[:, 2 * k - 1] = (-b * sn + c * cs) * sgn
        self.coef = new
        idx = np.round(((sgn * self.grid + shift) % TWO_PI) / (TWO_PI / self.T)).astype(int) % self.T
        for key in list(self.w):
            w = self.w[key][idx]
            self.w[key] = w / w.sum()
        for name in list(self.q_):
            q = self.q_[name][:, idx]
            self.q_[name] = q / q.sum(1, keepdims=True)
        return shift, flip

    # -- readouts --

    def phase(self, name, X, groups=None, prior='uniform', subset=None):
        q = self.posterior(name, X, groups, prior, subset)
        z = q @ np.exp(1j * self.grid)
        return np.angle(z) % TWO_PI, np.abs(z), q

    def fit_quality(self, name, X):
        """Per-count log evidence under a uniform prior."""
        return self.evidence(name, X, prior='uniform') / np.maximum(X.sum(1), 1)

    def fit_z(self, name, X):
        """fit_quality against the training cells of the same total."""
        fq = self.fit_quality(name, X)
        ref = self.ref.get(name)
        if ref is None:
            return np.full(len(X), np.nan)
        s = X.sum(1)
        b = np.clip(np.searchsorted(ref['edges'], s, side='right') - 1, 0, 7)
        return (fq - ref['mean'][b]) / (ref['sd'][b] + 1e-9)

    def centre(self, name):
        """The composition with no phase: softmax of the intercepts, the
        CLR centre of the ring (harmonics average to zero on the grid)."""
        a = self.a[name]
        return np.exp(a - logsumexp(a))

    def bf_ring(self, name, X):
        """log Bayes factor, ring (uniform over angles) vs centre."""
        alpha = self.alphas[name]
        ring = self.evidence(name, X, prior='uniform')
        pc = self.centre(name)
        if alpha is None:
            cen = X @ np.log(pc) + _dm_const(X, None)
        else:
            api = alpha * pc
            cen = (gammaln(X + api[None]) - gammaln(api)[None]).sum(1) + _dm_const(X, alpha)
        return ring - cen

    def radius(self, name, X, pseudo=0.5):
        """(rho, residual): the cell's distance from the ring's centre in
        the ring's own plane, relative to the ring's radius (1 = on the
        ring, 0 = centre), and the length of what lies outside that
        plane. Geometry only -- alpha-free, count-noise blind; the
        diagnostic picture beside bf_ring.

        The plane is the first two principal directions of the ring's
        CLR points and the radius is their median distance from the
        centre. An earlier form took the largest projection onto any of
        the T ring directions instead; the maximum over 72 noisy
        projections is biased upward by about two noise sigmas
        (measured on JP_001: every group read 1.4-1.6, the hydroxyurea
        cells that sit at 0.45 of the ring included), so it could not
        tell inside from outside. This form is the one that found them.
        """
        ring = clr(np.exp(self.log_pi(name)))
        cen = ring.mean(0)
        U, S, Vt = np.linalg.svd(ring - cen, full_matrices=False)
        plane = Vt[:2]                                       # (2, G)
        ring2 = (ring - cen) @ plane.T
        r_ring = float(np.median(np.hypot(ring2[:, 0], ring2[:, 1]))) + 1e-12
        F = (X + pseudo) / (X + pseudo).sum(1, keepdims=True)
        c = clr(F) - cen                                     # (n, G)
        c2 = c @ plane.T
        rho = np.hypot(c2[:, 0], c2[:, 1]) / r_ring
        resid = np.linalg.norm(c - c2 @ plane, axis=1)
        return rho, resid

    def place(self, name, X, groups=None, prior='uniform'):
        """Every per-cell verdict at once: theta, R, fit_z, bf_ring,
        radius (and the posterior)."""
        th, R, q = self.phase(name, X, groups, prior)
        rho, resid = self.radius(name, X)
        return {'theta': th, 'R': R, 'fit_z': self.fit_z(name, X),
                'bf_ring': self.bf_ring(name, X), 'radius': rho, 'residual': resid,
                'posterior': q}

    def spectrum(self, q, smooth=None):
        """A population's spectrum from its cells' posteriors."""
        w = _circ_smooth(np.asarray(q).mean(0), self.prior_smooth if smooth is None else smooth)
        return w / w.sum()

    def group_summary(self, name, X, groups, prior='uniform'):
        """Per condition: n, circular mean, concentration, spectrum mode,
        median R, median fit_z, fraction fit_z < -2, median log bf_ring,
        median radius."""
        out = self.place(name, X, groups, prior)
        groups = np.asarray(groups, dtype=object)
        rows = {}
        for c in np.unique(groups):
            k = groups == c
            z = np.mean(np.exp(1j * out['theta'][k]))
            w = self.spectrum(out['posterior'][k])
            rows[c] = {'n': int(k.sum()), 'mean_deg': float(np.degrees(np.angle(z)) % 360),
                       'concentration': float(abs(z)),
                       'mode_deg': float(np.degrees(self.grid[int(np.argmax(w))])),
                       'median_R': float(np.median(out['R'][k])),
                       'median_fit_z': float(np.nanmedian(out['fit_z'][k])),
                       'frac_outside': float(np.nanmean(out['fit_z'][k] < -2)),
                       'median_log_bf_ring': float(np.median(out['bf_ring'][k])),
                       'median_radius': float(np.median(out['radius'][k]))}
        return rows

    def profiles(self, name):
        return np.exp(self.log_pi(name))

    def fisher_information(self, name):
        """(T, |S_d|): per-gene Fisher information about the phase per
        count -- the shares sum to the total."""
        lpi = self.log_pi(name)
        pi = np.exp(lpi)
        deta = _dbasis(self.grid, self.K) @ self.coef[self.cols[name]].T
        dlpi = deta - (pi * deta).sum(1, keepdims=True)
        return pi * dlpi ** 2

    def fisher_share(self, name):
        fi = self.fisher_information(name)
        ws = [w for (n, _c), w in self.w.items() if n == name]
        w = np.mean(ws, axis=0) if ws else np.full(self.T, 1.0 / self.T)
        avg = (fi * w[:, None]).sum(0)
        return dict(zip(self.panels[name], avg / avg.sum()))

    # -- persistence --

    def to_dict(self):
        return {'K': self.K, 'T': self.T, 'prior_smooth': self.prior_smooth, 'l2': self.l2,
                'genes': list(self.genes), 'coef': self.coef.tolist(),
                'panels': {n: list(p) for n, p in self.panels.items()},
                'a': {n: v.tolist() for n, v in self.a.items()},
                'alphas': dict(self.alphas),
                'w': [[n, str(c), v.tolist()] for (n, c), v in self.w.items()],
                'ref': {n: {k: np.asarray(v).tolist() for k, v in r.items()} for n, r in self.ref.items()}}

    @classmethod
    def from_dict(cls, d):
        m = cls(K=d['K'], T=d['T'], prior_smooth=d.get('prior_smooth', 5), l2=d.get('l2', 1e-2))
        m.genes = list(d['genes'])
        m.gi = {g: i for i, g in enumerate(m.genes)}
        m.coef = np.asarray(d['coef'], float)
        m.panels = {n: list(p) for n, p in d['panels'].items()}
        m.cols = {n: np.array([m.gi[g] for g in p]) for n, p in m.panels.items()}
        m.a = {n: np.asarray(v, float) for n, v in d['a'].items()}
        m.alphas = {n: (None if v is None else float(v)) for n, v in d['alphas'].items()}
        m.w = {(n, c): np.asarray(v, float) for n, c, v in d['w']}
        m.ref = {n: {k: np.asarray(v, float) for k, v in r.items()} for n, r in d.get('ref', {}).items()}
        return m


# -- helpers around the model ------------------------------------------------

# -- the store: counts in, placements out --------------------------------------

COUNT_GATE = 0.5        # the classifier's own operating point (design doc 4.2)
CAPSULE_VERSION = 1
CAPSULE_COLUMNS = ('cell', 'theta_deg', 'R', 'fit_z', 'bf_ring', 'radius', 'total')


def source_key(src):
    """'MOD|HYBE|CH' -- the same string key population.py uses."""
    return f'{src[0]}|{src[1]}|{int(src[2])}'


def _count_fov(item):
    """One FOV's per-cell counts over the STORED spots of each source --
    module-level for pmap. Three numbers per (cell, source): n (spots at
    p_exist >= gate), soft (sum of p_exist) and candidates (every stored
    spot of that cell). A stored spot with no p_exist -- an older store,
    or a manual keep -- counts as one accepted spot with soft weight 1:
    somebody kept it, and nothing here can second-guess that."""
    storage_path, fov, sources, gate = item
    from codelab_pipeline.io import analysis_store
    cells, _ = analysis_store.read_cells(storage_path, fov)
    ids = [int(c['id']) for c in (cells or [])]
    celltype = {int(c['id']): str(c.get('celltype') or '') for c in (cells or [])}
    rows = []
    for src in sources:
        modality, hybe, channel = src
        spots = analysis_store.read_spots(storage_path, fov, modality=modality,
                                          hybe=hybe, channel=int(channel))
        n = {cid: 0 for cid in ids}
        soft = {cid: 0.0 for cid in ids}
        cand = {cid: 0 for cid in ids}
        p_min = float('inf')
        for sp in spots:
            cid = int(sp.get('cell', -1))
            if cid not in n:
                continue
            cand[cid] += 1
            p = float(sp.get('p_exist', float('nan')))
            if not np.isfinite(p):
                n[cid] += 1
                soft[cid] += 1.0
            else:
                p_min = min(p_min, p)
                soft[cid] += p
                if p >= gate:
                    n[cid] += 1
        for cid in ids:
            rows.append({'fov': int(fov), 'cell': cid, 'celltype': celltype[cid],
                         'modality': modality, 'hybe': hybe, 'channel': int(channel),
                         'n': n[cid], 'soft': soft[cid], 'candidates': cand[cid],
                         'p_min': (p_min if np.isfinite(p_min) else float('nan')),
                         'n_stored': len(spots)})
    return rows


def count_table(storage_path, fovs, sources, gate=COUNT_GATE, jobs=None, on_done=None):
    """Tidy per-(cell, source) counts for many FOVs, read FOV-major
    through one pool: fov, cell, celltype, modality, hybe, channel, n,
    soft, candidates, p_min, n_stored.

    sources: [(modality, hybe, channel)]. The counts are over what the
    spot store HOLDS: if the store was gated above `gate` when the
    spots were saved, n is the store's gate, not this one -- p_min per
    source says which (a p_min at or above `gate` means the store was
    gated there). Pivot with gene_table(table, source_names, metric='n')
    or metric='soft'.

    Returns (table, failures) -- failures [(fov, message)], never
    raised, so one unreadable FOV does not lose the other 33.
    """
    import pandas as pd
    from codelab_pipeline import parallel
    fovs = [int(f) for f in fovs]
    items = [(storage_path, f, [tuple(s) for s in sources], float(gate)) for f in fovs]
    results = parallel.pmap(_count_fov, items, kind='io', jobs=jobs, on_done=on_done)
    rows, fails = [], []
    for f, r in zip(fovs, results):
        if isinstance(r, parallel.Failure):
            fails.append((f, str(r)))
        else:
            rows.extend(r)
    cols = ['fov', 'cell', 'celltype', 'modality', 'hybe', 'channel',
            'n', 'soft', 'candidates', 'p_min', 'n_stored']
    return pd.DataFrame(rows, columns=cols), fails


def capsule_rows(placed, cell_ids, totals):
    """The per-FOV capsule's rows from place()'s dict, in the order the
    cells were placed."""
    out = []
    for i, cid in enumerate(cell_ids):
        out.append({'cell': int(cid),
                    'theta_deg': float(np.degrees(placed['theta'][i]) % 360.0),
                    'R': float(placed['R'][i]),
                    'fit_z': float(placed['fit_z'][i]),
                    'bf_ring': float(placed['bf_ring'][i]),
                    'radius': float(placed['radius'][i]),
                    'total': float(totals[i])})
    return out


def categorize(theta_deg, arcs):
    """Category name per cell from the user's arcs on the circle:
    arcs = [{'name': 'G1', 'start_deg': 300, 'end_deg': 60}, ...], each
    running FORWARD from start to end (so an arc may cross 0). A phase
    inside no arc reads ''; overlapping arcs resolve to the first
    listed. Categories are derived at read time, never stored: the
    boundaries are the user's choice and re-cutting them must not need
    a refit."""
    th = np.asarray(theta_deg, float) % 360.0
    out = np.array([''] * len(th), dtype=object)
    done = np.zeros(len(th), bool)
    for arc in arcs:
        lo, hi = float(arc['start_deg']) % 360.0, float(arc['end_deg']) % 360.0
        width = (hi - lo) % 360.0
        if width == 0:
            width = 360.0
        inside = ((th - lo) % 360.0) <= width
        take = inside & ~done & np.isfinite(th)
        out[take] = str(arc['name'])
        done |= take
    return out


GATE_KEYS = (('min_R', 'R', 'ge'), ('min_bf_ring', 'bf_ring', 'ge'),
             ('min_fit_z', 'fit_z', 'ge'), ('max_fit_z', 'fit_z', 'le'),
             ('min_radius', 'radius', 'ge'), ('max_radius', 'radius', 'le'),
             ('min_total', 'total', 'ge'))


def assign(table, arcs, gates=None):
    """The final call per placed cell: the arc's name when every verdict
    gate passes, '' (unassigned) otherwise. table: rows with theta_deg
    and the verdict columns; arcs as in categorize; gates: {gate key:
    value or None} over GATE_KEYS (min_R, min_bf_ring, min_fit_z,
    max_fit_z, min_radius, max_radius, min_total). A cell with no value
    for a gated verdict fails that gate. The Cell Cycle stage decides
    both the arcs and the gates and stores them with the model; the app
    applies them here when it reads the placements, so changing either
    never needs a refit."""
    cat = categorize(np.asarray(table['theta_deg'], float), arcs)
    ok = np.ones(len(cat), bool)
    for key, col, op in GATE_KEYS:
        v = (gates or {}).get(key)
        if v is None:
            continue
        if col not in table:
            ok &= False
            continue
        x = np.asarray(table[col], float)
        ok &= np.isfinite(x) & ((x >= float(v)) if op == 'ge' else (x <= float(v)))
    cat[~ok] = ''
    return cat


def gene_table(expression, source_names, metric='n_spots'):
    """(table, celltype): a cells x genes count table from a Population's
    tidy expression rows.

    expression: DataFrame with fov, cell, celltype, modality, hybe,
    channel and the metric column. source_names: {(modality, hybe,
    channel): gene_name} -- the app supplies it from the layout's
    readout_name; a source absent from the map is not a panel gene.
    Cells missing any gene are dropped (they cannot be composed).
    """
    import pandas as pd
    t = expression.copy()
    key = list(zip(t['modality'], t['hybe'], t['channel'].astype(int)))
    t['gene'] = [source_names.get(k) for k in key]
    t = t[t['gene'].notna()]
    table = t.pivot_table(index=['fov', 'cell'], columns='gene', values=metric, aggfunc='first')
    table = table[[g for g in source_names.values() if g in table.columns]]
    ok = table.notna().all(axis=1)
    table = table[ok]
    celltype = t.groupby(['fov', 'cell'])['celltype'].first().reindex(table.index).fillna('')
    return table, celltype


def _alpha_fold(item):
    """One (alpha, fold) of select_alpha -- module-level for pmap."""
    X, train, test, a, K, kw = item
    m = CycleModel(K=K, **kw).fit([Dataset('cv', X[train], [str(i) for i in range(X.shape[1])], alpha=a)])
    return float(m.evidence('cv', X[test], prior='uniform').mean())


def select_alpha(X, alphas=(10, 30, 100, 300, None), K=DEFAULT_K, folds=3, seed=0, jobs=None,
                 n_iter=20, **kw):
    """Held-out mean evidence per alpha for one cycling population:
    {alpha: score}, None = multinomial.

    Every (alpha, fold) is an independent single fit, so they run
    through ONE pool (codelab_pipeline.parallel.pmap, kind='cpu'); a
    cross-validation fit stops at n_iter=20 with no DM polish -- the
    ranking of alphas was the same at 20 and 40 iterations on JP_002
    and the full budget made the chooser fifteen times slower than
    the fit it serves. jobs=1 runs serially (tests).
    """
    from codelab_pipeline import parallel
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(X)), folds)
    kw = dict(kw)
    kw.setdefault('polish', 0)
    items, keys = [], []
    for a in alphas:
        for f in range(folds):
            test = parts[f]
            train = np.concatenate([parts[j] for j in range(folds) if j != f])
            items.append((X, train, test, a, K, kw))
            keys.append(a)
    # n_iter rides on the fit call, not the model: pass it through kw
    # by wrapping fit's iteration budget
    res = parallel.pmap(_alpha_fold_iter, [(it, n_iter) for it in items], kind='cpu', jobs=jobs)
    out = {}
    for a, r in zip(keys, res):
        if isinstance(r, parallel.Failure):
            raise RuntimeError(f'select_alpha: alpha {a}: {r}')
        out.setdefault(a, []).append(r)
    return {a: float(np.mean(v)) for a, v in out.items()}


def _alpha_fold_iter(item):
    (X, train, test, a, K, kw), n_iter = item
    m = CycleModel(K=K, **kw)
    m.fit([Dataset('cv', X[train], [str(i) for i in range(X.shape[1])], alpha=a)], n_iter=n_iter)
    return float(m.evidence('cv', X[test], prior='uniform').mean())


def backward_elimination(model, name, X, groups=None, train_group=None, anchors=('Hydroxyurea', 'Nocodazole')):
    """Drop, one at a time, the gene whose removal least degrades the
    placement of the cycling cells (median |dtheta| against the full
    panel); placement only, no refit. Returns a list of dict rows."""
    genes = list(model.panels[name])
    th_full = model.phase(name, X, prior='uniform')[0]
    groups = None if groups is None else np.asarray(groups, dtype=object)
    ktr = np.ones(len(X), bool) if (groups is None or train_group is None) else (groups == train_group)

    def evaluate(subset):
        cols = [genes.index(g) for g in subset]
        th, R, _ = model.phase(name, X[:, cols], prior='uniform', subset=subset)
        row = {'genes': '+'.join(subset), 'n_genes': len(subset),
               'err_deg': float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (th[ktr] - th_full[ktr]))))))),
               'median_R': float(np.median(R[ktr]))}
        if groups is not None:
            for c in anchors:
                k = groups == c
                if k.any():
                    row[f'{c[:3]}_conc'] = float(abs(np.mean(np.exp(1j * th[k]))))
        return row

    rows, subset = [evaluate(genes)], list(genes)
    while len(subset) > 2:
        best = None
        for g in subset:
            r = evaluate([h for h in subset if h != g])
            if best is None or r['err_deg'] < best[0]['err_deg']:
                best = (r, g)
        best[0]['dropped'] = best[1]
        rows.append(best[0])
        subset = [h for h in subset if h != best[1]]
    return rows


def saturation_table(rows, accepted='n_p50', candidates='n_all', min_candidates=20, max_fraction=0.12):
    """Per gene: mean candidates, mean accepted, accepted fraction, and a
    flag. The flag lists every round whose acceptance is low with many
    candidates; it does NOT say why (saturation of an abundant
    transcript, a weak probe, or an empty round look alike here -- the
    Toe rounds give the false-positive floor to read the others
    against)."""
    import pandas as pd
    g = pd.DataFrame(rows).groupby('gene')
    t = pd.DataFrame({'candidates': g[candidates].mean(), 'accepted': g[accepted].mean()})
    t['accepted_fraction'] = t['accepted'] / t['candidates'].clip(lower=1e-9)
    t['flag'] = (t['accepted_fraction'] < max_fraction) & (t['candidates'] > min_candidates)
    return t.sort_values('accepted_fraction')


def simulate(n=1500, genes=None, K=DEFAULT_K, total=(40, 300), seed=0, coef=None, a=None,
             theta=None, alpha=None):
    """Synthetic cells on a circle, for tests and self-checks."""
    rng = np.random.default_rng(seed)
    genes = genes or [f'g{i}' for i in range(9)]
    G = len(genes)
    if coef is None:
        coef = np.zeros((G, 2 * K))
        coef[:, :2] = rng.normal(0, 1.0, (G, 2))
        coef[:, 2:] = rng.normal(0, 0.3, (G, 2 * K - 2))
    a = rng.normal(0, 0.7, G) if a is None else np.asarray(a, float)
    theta = rng.uniform(0, TWO_PI, n) if theta is None else np.asarray(theta, float)
    eta = _basis(theta, K) @ coef.T + a[None, :]
    pi = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
    if alpha is not None:
        pi = np.stack([rng.dirichlet(alpha * p) for p in pi])
    s = rng.integers(total[0], total[1], n)
    X = np.stack([rng.multinomial(int(si), p) for si, p in zip(s, pi)])
    return X, theta, coef, a, genes
