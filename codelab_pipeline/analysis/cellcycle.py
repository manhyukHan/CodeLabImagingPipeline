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
evidence: chr19 100, JP_001 300, JP_002 100).

THE DISPERSION MAY DEPEND ON DEPTH: alpha_i = alpha_d (s_i / s_ref)^beta_d,
beta 0 being the single alpha. A single alpha caps every cell at about
alpha effective counts, however deep it is. Measured with held-out FOVs
per depth quintile, JP_001 wants alpha 100 at s ~150 and 1000 at s
~1000 -- its deep cells are more informative and its shallow ones less
than one alpha allows, and its shallow cells failed the on-ring gate at
0.65-0.76 where 0.95 is expected -- while JP_002 wants 100 at every
depth. The exponent is chosen AFTER the fit, by held-out-FOV evidence
with the profiles held (calibrate_dispersion): beta enters the
likelihood, the per-cell weight and the ring reference, and fitting WITH
it instead hands the profiles to the deep cells. Each (experiment,
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
cell's own radial position, s_eff = s(1+alpha_i)/(s+alpha_i)):
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
    """(n, T): sum_g lgamma(x + a pi) - lgamma(a pi) for api (T, G), or
    (n, T, G) when every cell has its own alpha."""
    if api.ndim == 3:
        return (gammaln(X[:, None, :] + api) - gammaln(api)).sum(2)
    return (gammaln(X[:, None, :] + api[None]) - gammaln(api)[None]).sum(2)


def _alpha_at(alpha, beta, s_ref, s):
    """The Dirichlet concentration a cell of panel total s is read with:
    alpha * (s / s_ref)^beta. None stays None (multinomial); beta 0
    returns the plain float, so a single-alpha model takes exactly the
    path it always took. Totals below 1 count as 1 -- a zero total would
    make the concentration zero and every lgamma infinite."""
    if alpha is None:
        return None
    if not beta:
        return float(alpha)
    s = np.maximum(np.asarray(s, float), 1.0)
    return float(alpha) * (s / float(s_ref)) ** float(beta)


def _api(alpha, pi):
    """alpha * pi for a scalar alpha (T, G) or a per-cell one (n, T, G)."""
    if np.ndim(alpha) == 0:
        return alpha * pi
    return np.asarray(alpha, float)[:, None, None] * pi[None]


RING_LEVELS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


def _simulate_cells(rng, pi, alpha, total, n):
    """n count vectors of `total` reads from composition pi: multinomial
    around a Dirichlet(alpha * pi) draw per cell, or exactly pi when
    alpha is None -- the model's own noise."""
    pi = np.clip(np.asarray(pi, float), 1e-12, None)
    pi = pi / pi.sum()
    if alpha is None:
        return rng.multinomial(int(total), pi, size=n).astype(float)
    p = rng.dirichlet(alpha * pi, size=n)
    p = np.clip(p, 1e-12, None)
    p /= p.sum(1, keepdims=True)
    return rng.multinomial(int(total), p).astype(float)


def _dm_const(X, alpha):
    s = X.sum(1)
    c = gammaln(s + 1) - gammaln(X + 1).sum(1)
    if alpha is None:
        return c
    return c + gammaln(alpha) - gammaln(s + alpha)


class Dataset:
    """One experiment's counts: X (n, |genes|), gene names in column order,
    a condition label per cell, and the dispersion alpha (None =
    multinomial). alpha_beta makes the dispersion depend on depth,
    alpha_i = alpha * (s_i / alpha_sref)^alpha_beta; alpha_sref defaults
    to the median panel total of these cells."""

    def __init__(self, name, X, genes, groups=None, alpha=DEFAULT_ALPHA, dapi=None,
                 alpha_beta=0.0, alpha_sref=None):
        self.name = str(name)
        self.X = np.asarray(X, float)
        # log DAPI per cell, already normalised within its FOV; NaN where
        # a cell has none. Only fit(dapi_weight > 0) looks at it.
        self.dapi = None if dapi is None else np.asarray(dapi, float)
        self.genes = [str(g) for g in genes]
        if self.X.ndim != 2 or self.X.shape[1] != len(self.genes):
            raise ValueError(f'{name}: X is {self.X.shape}, genes are {len(self.genes)}')
        self.groups = (np.asarray(groups, dtype=object) if groups is not None
                       else np.array(['all'] * len(self.X), dtype=object))
        self.alpha = None if alpha is None else float(alpha)
        self.alpha_beta = 0.0 if alpha is None else float(alpha_beta or 0.0)
        s = self.X.sum(1)
        self.alpha_sref = (float(alpha_sref) if alpha_sref is not None
                           else float(np.median(s[s > 0])) if (s > 0).any() else 1.0)

    def alpha_cells(self):
        """The concentration each of these cells is read with (None, one
        float, or one per cell)."""
        return _alpha_at(self.alpha, self.alpha_beta, self.alpha_sref, self.X.sum(1))


# -- the model ----------------------------------------------------------------

class CycleModel:
    def __init__(self, K=DEFAULT_K, T=DEFAULT_T, prior_smooth=5, l2=1e-2,
                 mstep='multinomial', patience=5, polish=2, m_iter=60, gain_l2=1.0):
        if mstep not in ('multinomial', 'dm'):
            raise ValueError("mstep must be 'multinomial' or 'dm'")
        self.K, self.T = int(K), int(T)
        self.prior_smooth = int(prior_smooth)
        self.l2 = float(l2)
        self.gain_l2 = float(gain_l2)   # ridge on log s; 0 frees the gains, large pins them to 1
        self.mstep = mstep
        self.patience = int(patience)
        self.polish = int(polish)
        self.m_iter = int(m_iter)
        self.grid = np.arange(self.T) * TWO_PI / self.T
        self.B = _basis(self.grid, self.K)
        self.genes = []
        self.coef = None            # (G, 2K) shared harmonics
        self.a = {}                 # name -> (|S_d|,) intercepts
        self.s = {}                 # name -> (|S_d|,) gains on the harmonics
        self.free_s = {}            # name -> bool over the panel: gains that are free
        self._gene_gains = False    # fit(gene_gains=True) frees the shared genes' gains
        self._gains_active = True   # _em holds this False through the warm-up
        self.dapi_weight = 0.0      # fit(dapi_weight=...) lets the E-step see DAPI
        self.balance_datasets = False   # fit(balance_datasets=True): equal say per experiment
        self.dataset_scale = {}     # name -> the factor that equalised it (1 when off)
        self.h = None               # (T,) shared log-DAPI curve on the grid, mean 0
        self.c = {}                 # name -> DAPI offset
        self.sigma = {}             # name -> DAPI residual sd
        self.w = {}                 # (name, group) -> (T,) spectrum
        self.alphas = {}            # name -> alpha (at the reference depth when beta != 0)
        self.alpha_beta = {}        # name -> depth exponent of the dispersion (0 = one alpha)
        self.alpha_sref = {}        # name -> the reference panel total of that exponent
        self.panels = {}            # name -> [genes]
        self.cols = {}              # name -> gene indices into self.genes
        self.ref = {}               # name -> fit-quality reference (edges, mean, sd)
        self.ring_ref = {}          # name -> what true on-ring cells score, per angle and depth
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
        seen = {}
        for d in datasets:
            for g in d.genes:
                seen[g] = seen.get(g, 0) + 1
        gains = bool(getattr(self, '_gene_gains', False))
        for d in datasets:
            self.panels[d.name] = list(d.genes)
            self.cols[d.name] = np.array([self.gi[g] for g in d.genes])
            self.alphas[d.name] = d.alpha
            self.alpha_beta[d.name] = d.alpha_beta
            self.alpha_sref[d.name] = d.alpha_sref
            self.s[d.name] = np.ones(len(d.genes))
            # a gain is redundant with the gene's own (b, c) unless the
            # gene is also fitted somewhere else: free it only then
            self.free_s[d.name] = np.array([gains and seen[g] > 1 for g in d.genes], bool)

    def gains(self, name=None):
        """The per-gene gain on the harmonic part: how much deeper (or
        shallower) each gene swings in this experiment than the shared
        profile says. 1.0 everywhere unless a joint fit freed them.

        Only genes shared by two or more datasets carry a free gain --
        for a gene fitted in one place the gain is redundant with its own
        (b, c) -- and each dataset's free log gains average to zero, so
        what a gain reports is the RATIO against the other experiments,
        never an absolute scale."""
        if name is not None:
            return {g: float(v) for g, v in zip(self.panels[name], self.s.get(name, np.ones(len(self.panels[name]))))}
        return {n: self.gains(n) for n in self.panels}

    def bridge_report(self):
        names = list(self.panels)
        return {(a, b): sorted(set(self.panels[a]) & set(self.panels[b]))
                for i, a in enumerate(names) for b in names[i + 1:]}

    # -- composition and likelihood --

    def log_pi(self, name, coef=None, a=None, subset=None, s=None):
        """(T, |subset|) log composition of `name`'s panel (or a subset
        of it) on the grid."""
        coef = self.coef if coef is None else coef
        a = self.a[name] if a is None else a
        sv = self.s.get(name) if s is None else s
        if subset is None:
            gidx, aidx = self.cols[name], np.arange(len(self.panels[name]))
        else:
            aidx = np.array([self.panels[name].index(g) for g in subset])
            gidx = np.array([self.gi[g] for g in subset])
        sv = np.ones(len(self.panels[name])) if sv is None else np.asarray(sv, float)
        eta = (self.B @ coef[gidx].T) * sv[aidx][None, :] + a[aidx][None, :]
        return eta - logsumexp(eta, axis=1, keepdims=True)

    def alpha_of(self, name, X=None, total=None):
        """The concentration rows of X (or cells of panel total `total`)
        are read with: None (multinomial), one float (beta 0), or one per
        row. It follows the total of the counts it is given, so a subset
        placement reads a cell at its subset depth."""
        s = np.asarray(X, float).sum(1) if total is None else total
        return _alpha_at(self.alphas[name], self.alpha_beta.get(name, 0.0),
                         self.alpha_sref.get(name, 1.0), s)

    def loglik_grid(self, name, X, coef=None, a=None, subset=None, alpha='own'):
        alpha = self.alpha_of(name, X) if isinstance(alpha, str) and alpha == 'own' else alpha
        lpi = self.log_pi(name, coef, a, subset)
        if alpha is None:
            return X @ lpi.T
        return _dm_loglik(X, _api(alpha, np.exp(lpi)))

    def posterior(self, name, X, groups=None, prior='spectrum', subset=None, aux=None):
        """(n, T). prior='spectrum' uses the fitted group spectra (groups
        required); prior='uniform' asks the counts alone -- the PLACEMENT
        prior for populations that were not fitted. `aux` (n, T) is an
        extra log-likelihood per cell and angle: the fit passes the DAPI
        term here; placement never does."""
        ll = self.loglik_grid(name, X, subset=subset)
        if aux is not None:
            ll = ll + aux
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
        return out + _dm_const(X, self.alpha_of(name, X))

    # -- fitting --

    def _free_of(self, name):
        """The gains that are free in THIS M-step: none during the
        warm-up, the shared ones after it."""
        free = self.free_s.get(name)
        if free is None or not self._gains_active:
            return None
        return free if free.any() else None

    def _unpack(self, v, names=None):
        """(coef, intercepts, gains) from the flat vector. The gains are
        carried as LOG gains, and only for the entries free_s marks; the
        rest stay at 1."""
        G, P = len(self.genes), self.B.shape[1]
        coef = v[:G * P].reshape(G, P)
        names = list(self.panels) if names is None else list(names)
        a, off = {}, G * P
        for name in names:
            n = len(self.cols[name])
            a[name] = v[off:off + n]
            off += n
        s = {}
        for name in names:
            free = self._free_of(name)
            sv = np.ones(len(self.cols[name]))
            if free is not None:
                k = int(free.sum())
                sv[free] = np.exp(v[off:off + k])
                off += k
            s[name] = sv
        return coef, a, s

    def _mstep(self, datasets, qs, exact):
        """One M-step over shared harmonics and every experiment's
        intercepts. exact=False folds each dataset to its multinomial
        sufficient statistics; exact=True uses the DM objective."""
        G, P = len(self.genes), self.B.shape[1]
        names = [d.name for d in datasets]
        x0 = np.concatenate([self.coef.ravel()] + [self.a[d.name] for d in datasets]
                            + [np.log(self.s[d.name][self._free_of(d.name)])
                               for d in datasets if self._free_of(d.name) is not None])
        pre = {}
        # Per cell the DM effective count already levels a deep cell with
        # a shallow one, but nothing levels the EXPERIMENTS: the sum over
        # cells still hands the shared profile to whichever experiment has
        # more cells and a larger alpha (measured: JP_001 carries 2.7x
        # chr19). balance_datasets scales each dataset's q so their
        # effective masses match -- every experiment gets an equal say.
        self.dataset_scale = {d.name: 1.0 for d in datasets}
        if self.balance_datasets and len(datasets) > 1:
            mass = {}
            for d in datasets:
                s = d.X.sum(1)
                ac = d.alpha_cells()
                w = np.ones_like(s) if ac is None else (1.0 + ac) / (s + ac)
                mass[d.name] = float((s * w).sum())
            target = float(np.mean(list(mass.values())))
            self.dataset_scale = {n: target / max(m, 1e-9) for n, m in mass.items()}
        for d in datasets:
            q = qs[d.name] * self.dataset_scale[d.name]
            s = d.X.sum(1)
            # THE DM'S OWN WEIGHTING, kept in the fast path. A Dirichlet-
            # multinomial cell carries s(1+alpha)/(s+alpha) effective counts,
            # never more than ~alpha; the plain multinomial statistics would
            # weight a 1000-count JP_001 cell ten times a 100-count chr19
            # cell and let the deepest experiment own every shared profile
            # (measured on the real joint fit: chr19 collapsed to one lobe,
            # the two nocodazole anchors drifted 90 deg apart). Scaling each
            # cell's counts to its effective count restores the balance the
            # exact M-step has; the polish then refines from there. With a
            # depth-dependent alpha the weight is each cell's own.
            ac = d.alpha_cells()
            w = np.ones_like(s) if ac is None else (1.0 + ac) / (s + ac)
            pre[d.name] = (q, q.T @ (d.X * w[:, None]), q.T @ (s * w), ac)

        def f(v):
            coef, a, s = self._unpack(v, names)
            val = 0.5 * self.l2 * (coef ** 2).sum()
            gcoef = self.l2 * coef
            ga, gs = {}, {}
            for d in datasets:
                q, Sx, Ss, ac = pre[d.name]
                lpi = self.log_pi(d.name, coef, a[d.name], s=s[d.name])
                pi = np.exp(lpi)
                if not exact or ac is None:
                    val -= (Sx * lpi).sum()
                    g_eta = -(Sx - Ss[:, None] * pi)
                elif np.ndim(ac) == 0:
                    api = ac * pi
                    Xb = d.X[:, None, :]
                    val -= (q[:, :, None] * (gammaln(Xb + api[None]) - gammaln(api)[None])).sum()
                    dpi = ac * (q[:, :, None] * (digamma(Xb + api[None]) - digamma(api)[None])).sum(0)
                    g_eta = -(pi * (dpi - (pi * dpi).sum(1, keepdims=True)))
                else:
                    api = _api(ac, pi)                          # (n, T, G)
                    Xb = d.X[:, None, :]
                    val -= (q[:, :, None] * (gammaln(Xb + api) - gammaln(api))).sum()
                    dpi = (q[:, :, None] * ac[:, None, None] * (digamma(Xb + api) - digamma(api))).sum(0)
                    g_eta = -(pi * (dpi - (pi * dpi).sum(1, keepdims=True)))
                sv = s[d.name]
                gcoef[self.cols[d.name]] += (g_eta * sv[None, :]).T @ self.B
                ga[d.name] = g_eta.sum(0)
                free = self._free_of(d.name)
                if free is not None:
                    # d eta / d log s = (B @ coef) * s
                    h = (self.B @ coef[self.cols[d.name]].T) * sv[None, :]
                    g = (g_eta * h).sum(0)[free]
                    ls = np.log(sv[free])
                    val += 0.5 * self.gain_l2 * (ls ** 2).sum()
                    g = g + self.gain_l2 * ls
                    # the dataset's own scale is redundant with the shared
                    # profile: keep the mean of its log gains at zero by
                    # projecting it out of the gradient
                    gs[d.name] = g - g.mean()
            return val, np.concatenate([gcoef.ravel()] + [ga[d.name] for d in datasets]
                                       + [gs[n] for n in names if n in gs])

        res = optimize.minimize(f, x0, jac=True, method='L-BFGS-B',
                                options={'maxiter': self.m_iter})
        coef, a, s = self._unpack(res.x, names)
        for name in a:
            a[name] = a[name] - a[name].mean()          # softmax gauge
        for name in s:
            free = self._free_of(name)
            if free is not None:                        # the gauge on the gains
                s[name][free] = np.exp(np.log(s[name][free]) - np.log(s[name][free]).mean())
        return coef, a, s

    def _dapi_aux(self, d):
        """(n, T) the weighted DAPI log-likelihood of dataset d on the
        grid, or None when the fit is not using DAPI."""
        if self.dapi_weight <= 0 or d.dapi is None or self.h is None or d.name not in self.c:
            return None
        y = d.dapi
        ok = np.isfinite(y)
        aux = np.zeros((len(y), self.T))
        r = y[ok][:, None] - self.c[d.name] - self.h[None, :]
        aux[ok] = -self.dapi_weight * 0.5 * r ** 2 / (self.sigma[d.name] ** 2)
        return aux

    def _update_dapi(self, datasets, qs, sweeps=2):
        """Closed-form M-step for the DAPI curve: with q fixed, the shared
        h(t) is the q-weighted mean of every cell's offset-corrected log
        DAPI at t (smoothed like the spectrum, mean 0 -- the level lives
        in c_d), then each dataset's offset and residual scale.

        The shape is left free on purpose. A ramp fixed to the textbook
        cycle (0 to log 2, drop at division) was tried on the real pair
        and hurt: the count evidence fell with the weight, chr19's R
        median went 0.80 -> 0.74 and its DAPI drop 2.09 -> 1.5x, because
        the real curve is flat through G1, rises through S and plateaus,
        and forcing a linear rise drags G1 cells early and G2/M cells
        late. The free curve at weight 0.5-1 raised the count evidence
        (-38.143 -> -38.128 per cell) and closed the two experiments'
        DAPI halving to 0 deg while the DAPI-free checks stayed put.

        What neither form can do is rescue a grossly mis-rotated start:
        seeded 120 deg off, both leave the divisions 100-110 deg apart at
        any weight, because h is estimated from the same mis-rotated q's
        and simply grows one hump per dataset. Within about 60 deg the
        counts alone already align; the term keeps and refines that."""
        use = [d for d in datasets if d.dapi is not None and np.isfinite(d.dapi).any()]
        if self.dapi_weight <= 0 or not use:
            return
        # Estimated from every dataset's q, h is a summary of the current
        # placement and can only reinforce it. Shaping it from ONE
        # dataset's q was tried (an 'anchor') and changed nothing: at a
        # mis-rotated fixed point that dataset's own q is smeared too, and
        # in any case a per-cell DAPI term cannot rotate a dataset -- the
        # counts pin each cell by ~28 log-units against a DAPI gain of at
        # most ~6. Rotation is a global move; see the rotation scan.
        shape = use
        if self.h is None:
            self.h = np.zeros(self.T)
        for d in use:
            self.c.setdefault(d.name, float(np.nanmean(d.dapi)))
            self.sigma.setdefault(d.name, max(float(np.nanstd(d.dapi)), 1e-3))
        for _ in range(sweeps):
            num, den = np.zeros(self.T), np.zeros(self.T)
            for d in shape:
                ok = np.isfinite(d.dapi)
                q = qs[d.name][ok]
                num += q.T @ (d.dapi[ok] - self.c[d.name])
                den += q.sum(0)
            h = num / np.maximum(den, 1e-9)
            h = _circ_smooth(h, self.prior_smooth)
            self.h = h - h.mean()
            for d in use:
                ok = np.isfinite(d.dapi)
                q = qs[d.name][ok]
                pred = q @ self.h
                self.c[d.name] = float(np.mean(d.dapi[ok] - pred))
                resid2 = (q * (d.dapi[ok][:, None] - self.c[d.name] - self.h[None, :]) ** 2).sum(1)
                self.sigma[d.name] = max(float(np.sqrt(np.mean(resid2))), 1e-3)

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
            m._fit_single(Dataset(d.name, d.X, d.genes, d.groups, alpha=d.alpha,
                                  alpha_beta=d.alpha_beta, alpha_sref=d.alpha_sref))
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

    def _em(self, datasets, theta0, n_iter, tol, verbose, gain_warmup=0):
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
            self._gains_active = it >= int(gain_warmup)
            self.coef, self.a, self.s = self._mstep(datasets, qs, exact)
            self._update_w(datasets, qs)
            self._update_dapi(datasets, qs)
            qs = {d.name: self.posterior(d.name, d.X, d.groups, aux=self._dapi_aux(d)) for d in datasets}
            ev = self._mean_evidence(datasets)
            self.history.append(ev)
            if verbose:
                print(f'  iter {it}: mean evidence {ev:.4f}', flush=True)
            if ev > best_ev + 1e-12:
                best_ev, since = ev, 0
                best = (self.coef.copy(), {k: v.copy() for k, v in self.a.items()},
                        {k: v.copy() for k, v in self.w.items()},
                        {k: v.copy() for k, v in self.s.items()})
            else:
                since += 1
            if abs(ev - prev) < tol or since >= self.patience:
                break
            prev = ev
        self._gains_active = True
        if best is not None:
            self.coef, self.a, self.w, self.s = best
        # polish: exact DM M-steps from the best point (no-op if every
        # alpha is None or polish == 0)
        if self.polish and not exact and any(d.alpha is not None for d in datasets):
            for _ in range(self.polish):
                qs = {d.name: self.posterior(d.name, d.X, d.groups, aux=self._dapi_aux(d)) for d in datasets}
                coef, a, s = self._mstep(datasets, qs, True)
                ev = None
                old = (self.coef, self.a, self.s)
                self.coef, self.a, self.s = coef, a, s
                self._update_w(datasets, qs)
                ev = self._mean_evidence(datasets)
                if ev < best_ev:                 # a polish that hurts is undone
                    self.coef, self.a, self.s = old
                    break
                best_ev = ev
                self.history.append(ev)
        self.q_ = {d.name: self.posterior(d.name, d.X, d.groups, aux=self._dapi_aux(d)) for d in datasets}
        return self

    def fit(self, datasets, n_iter=40, tol=1e-4, bridge_exclude=HOUSEKEEPING,
            weights='fisher_min', robust=True, verbose=False, theta0=None, orient_by=None,
            gene_gains=False, gain_warmup=0, dapi_weight=0.0, balance_datasets=False):
        """Fit on cycling populations. `datasets`: [Dataset, ...].
        orient_by=(early, late) fixes every bridge's reflection by gene
        roles (see stagewise_init).

        gene_gains=True lets a gene shared by two or more datasets swing
        by a different depth in each of them (see gains()). It is off by
        default because it changes what a joint fit can express, and so
        changes joint results; a single-dataset fit is unaffected either
        way, since there a gain is redundant with the gene's own (b, c).
        gain_warmup=k holds the gains at 1 for the first k EM iterations,
        so the angles settle before the fit is allowed to trade a
        profile's shape against its gain.

        dapi_weight > 0 lets the E-step see each dataset's log DAPI
        (Dataset.dapi) through a shared curve on the angle grid, which is
        what pins the relative rotation between experiments; the counts
        alone do not. Placement stays count-only.

        What the term can and cannot do, measured: from a start within
        about 60 deg it keeps and refines the alignment (weight 0.5-1;
        at 2 the count evidence still rises but the arrested populations
        lose their place -- the 3e rule's 'internal up, external down').
        It cannot pull a dataset out of a wrongly rotated basin at any
        weight, with any shape of curve, anchored or not: a cell's
        counts hold it ~28 log-units against a DAPI gain of ~6 at most.
        Choose the basin with the rotation scan; use this to hold it.

        balance_datasets=True gives every experiment the same total
        weight in the shared M-step (their effective masses are scaled
        to a common value). Off, the shared profile belongs to the
        experiment with more cells and a larger alpha, as a likelihood
        does; on, it is an average of experiments."""
        datasets = list(datasets)
        self._gene_gains = bool(gene_gains)
        self.dapi_weight = float(dapi_weight)
        self.balance_datasets = bool(balance_datasets)
        self.h, self.c, self.sigma = None, {}, {}
        self._index(datasets)
        if theta0 is None:
            theta0 = (self.stagewise_init(datasets, bridge_exclude, weights, robust, verbose, orient_by)
                      if len(datasets) > 1 else {datasets[0].name: pca_angle(datasets[0].X)[0]})
        self._em(datasets, theta0, n_iter, tol, verbose, gain_warmup=gain_warmup)
        # Orientation is a property of the ANSWER, not of the seed. It used
        # to be applied only inside stagewise_init, which a single-dataset
        # fit never calls (theta0 comes straight from the PCA angle there),
        # so fit(orient_by=...) on one experiment silently did nothing and
        # the handedness fell out of whichever way the PCA happened to
        # point -- refits of the same cells came out mirrored from each
        # other, and from the stored models. Re-applying it here makes the
        # constraint hold whatever path produced theta0.
        if orient_by is not None:
            early = [g for g in orient_by[0] if g in self.gi]
            late = [g for g in orient_by[1] if g in self.gi]
            if early and late:
                self.orient(early, late)
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
        self.rotate(shift, flip)
        return shift, flip

    def dna_direction(self, name, X, dna, n_bins=20, min_gap=0.5, min_cells=200):
        """Which way round the cycle runs, read off the DNA.

        The circle has two mirror images and the likelihood cannot tell
        them apart: reverse theta, reverse every profile with it, and
        every cell keeps its likelihood. `orient` breaks the tie with
        gene roles -- the S genes must peak before the G2/M genes within
        half a turn -- but that says nothing when the two mean peaks sit
        a half turn apart, and chr19's are 178 degrees apart, a margin of
        two degrees.

        DNA settles it, because replication and division are not
        symmetric in time: the content climbs through S over hours and
        halves at division in minutes. So the binned DNA is a sawtooth,
        and its steepest FALL must beat its steepest RISE. Mirroring
        swaps the two, so the ratio below is the evidence, and it needs
        no origin -- a rotation moves the bins but not the extremes.

        Two details are not optional. It has to be read in CYCLE TIME,
        not in degrees: the ring can crawl through division and sprint
        through S, and then the fall looks gradual (chr19 reads 1.28 in
        degrees and 1.88 in tau). And cells at angles where the ring
        passes through its own centre have to go, because their angles
        are not determined and they smear the fall (JP_001 reads 0.44
        with them and 1.77 without). Returns (ratio, n) with ratio > 1
        supporting the current handedness.

        It has power only where the content's own step is resolved.
        Measured against each stored model AND its mirror image, so a
        working test must read a number and its reciprocal:

            JP_002   3.37 / 0.25   (DNA halving 2.00, the physical value)
            JP_001   1.93 / 0.50   (halving 1.85)
            chr19    1.42 / 1.45   (halving 1.43 -- no power)

        chr19's DAPI resolves the halving least well of the three and
        the test cannot separate its two mirror images; refits of the
        same cells read 0.63-0.93 whichever way they are oriented. Read
        the halving factor beside the ratio, and do not orient on a
        ratio near 1.
        """
        X = np.asarray(X, float)
        dna = np.asarray(dna, float)
        placed = self.place(name, X)
        th = np.degrees(placed['theta']) % 360.0
        ring = clr(np.exp(self.log_pi(name)))
        d = np.linalg.norm(ring - ring.mean(0), axis=1)
        d = d / max(np.median(d), 1e-12)
        keep = (placed['posterior'] @ d >= float(min_gap)) & np.isfinite(dna) & (dna > 0)
        if keep.sum() < int(min_cells):
            return np.nan, int(keep.sum())
        w = np.histogram(th[keep], bins=self.T, range=(0, 360))[0].astype(float)
        w = _circ_smooth(w / max(w.sum(), 1), 5)
        tau_grid = (np.cumsum(w / w.sum()) - w[0] / w.sum() / 2.0) % 1.0
        tau = np.interp(th[keep], np.degrees(self.grid), tau_grid, period=360)
        b = np.clip((tau * n_bins).astype(int), 0, n_bins - 1)
        med = np.array([np.nanmedian(dna[keep][b == i]) if (b == i).sum() >= 8 else np.nan
                        for i in range(n_bins)])
        good = np.isfinite(med)
        if good.sum() < n_bins - 6:
            return np.nan, int(keep.sum())
        lm = _circ_smooth(np.log(np.interp(np.arange(n_bins), np.where(good)[0], med[good],
                                           period=n_bins)), 1)
        step = np.diff(np.r_[lm, lm[0]])
        return float((-step.min()) / max(step.max(), 1e-9)), int(keep.sum())

    def orient_by_dna(self, name, X, dna, **kw):
        """Flip the frame when dna_direction says the cycle runs the
        other way. Returns (ratio, flipped); a ratio that cannot be
        measured leaves the model alone."""
        ratio, n = self.dna_direction(name, X, dna, **kw)
        if not np.isfinite(ratio) or ratio >= 1.0:
            return ratio, False
        self.rotate(0.0, True)
        return ratio, True

    def rotate_to_zero(self, deg):
        """Rotate (no reflection) so the angle `deg` becomes 0 -- the
        origin at a measured event, e.g. division (total_drop_angle)."""
        self.rotate(np.radians(float(deg)), False)

    def rotate(self, shift, flip=False):
        """The frame change every readout shares: the profile at the new
        angle t is the old profile at sgn*t + shift, so the point that
        was at `shift` moves to 0 and a flip reverses the direction.
        Profiles, spectra and stored posteriors move together."""
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
        alpha = self.alpha_of(name, X)
        ring = self.evidence(name, X, prior='uniform')
        pc = self.centre(name)
        if alpha is None:
            cen = X @ np.log(pc) + _dm_const(X, None)
        else:
            api = (np.asarray(alpha, float)[:, None] if np.ndim(alpha) else alpha) * pc[None]
            cen = (gammaln(X + api) - gammaln(api)).sum(1) + _dm_const(X, alpha)
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

    def ring_level(self, name, X):
        """Per count, how much better the ring explains this cell than a
        composition free to be anything:

            [ log P(X | ring, theta marginalised)
              - log P(X | free composition) ] / total

        A uniform prior over the simplex makes the free hypothesis
        integrate to lgamma(G) - lgamma(N+G) + lgamma(N+1), a function
        of the depth alone, so the statistic costs one evidence call.

        This is what the on-ring gate asks. The older pair (bf_ring >= 0
        and radius >= 0.5) compares the cell with the ring's CENTRE
        instead, which is a different question and the wrong one: a
        random composition lies far from the centre too and sails
        through it (measured: 72-97% of random compositions and 88-100%
        of gene-scrambled ones pass on the three panels), while cells
        NEAR the centre are rejected -- and those are exactly the cells
        a panel whose ring folds through its centre places on its ring.
        Against ring_level a random composition scores -0.13 to -0.17
        per count where a true on-ring cell scores about +0.03.
        """
        X = np.asarray(X, float)
        G = X.shape[1]
        N = X.sum(1)
        free = gammaln(G) - gammaln(N + G) + gammaln(N + 1)
        return (self.evidence(name, X, prior='uniform') - free) / np.maximum(N, 1.0)

    def ring_reference(self, name, totals, n_sim=200, seed=0, n_bins=6, levels=RING_LEVELS):
        """What a cell that truly sits ON the ring scores, at every grid
        angle and a few depths: cells are simulated from the fitted
        composition of each angle (Dirichlet-multinomial with the
        panel's alpha, totals drawn from the depth bin) and the
        quantiles of their ring_level are kept.

        Keyed by where the simulated cells are PLACED, not by the angle
        they came from: on_ring judges a real cell through its
        posterior, so the reference has to be built the same way, or a
        cell whose posterior straddles two angles is held to a standard
        neither of them sets.

        Why calibrate at all: ring_level drifts along theta, because the
        fitted composition is sharper at some angles than others, so one
        flat threshold would cut hardest where the model is least
        certain. Against this reference the gate keeps true on-ring
        cells at the same rate everywhere -- measured 0.96-0.97 at both
        a fold and the far side of the ring, on all three panels."""
        rng = np.random.default_rng(seed)
        totals = np.asarray(totals, float)
        totals = totals[np.isfinite(totals) & (totals > 0)]
        if len(totals) == 0:
            raise ValueError('ring_reference needs the totals of the cells it will judge')
        cuts = np.unique(np.quantile(totals, np.linspace(0, 1, n_bins + 1)[1:-1]))
        b_all = np.clip(np.searchsorted(cuts, totals, side='right'), 0, len(cuts))
        reps = np.array([max(int(round(np.median(totals[b_all == b]))), 1) if (b_all == b).any() else 1
                         for b in range(len(cuts) + 1)])
        levels = np.asarray(levels, float)
        pis = np.exp(self.log_pi(name))
        tab = np.empty((self.T, len(reps), len(levels)))
        for b, N in enumerate(reps):
            # each depth bin is simulated with the dispersion of ITS depth:
            # with one alpha for every bin, a panel whose shallow cells are
            # noisier than its deep ones fails its own shallow cells
            # (JP_001: on-ring 0.76 in the lowest depth quintile)
            alpha = self.alpha_of(name, total=float(N))
            alpha = None if alpha is None else float(alpha)
            Xs = np.vstack([_simulate_cells(rng, pis[t], alpha, int(N), n_sim) for t in range(self.T)])
            lv = self.ring_level(name, Xs)
            Q = self.posterior(name, Xs, prior='uniform')
            o = np.argsort(lv)
            for t in range(self.T):
                w = Q[o, t]
                s = float(w.sum())
                if s <= 0:
                    w, s = np.ones(len(o)), float(len(o))
                tab[t, b, :] = np.interp(levels, np.cumsum(w) / s, lv[o])
        self.ring_ref[name] = {'cuts': cuts, 'totals': reps.astype(float), 'levels': levels, 'lvl': tab}
        return self.ring_ref[name]

    def _level_tail(self, name, X, placed):
        """Per cell, the probability that a cell truly on the ring,
        placed where this one is placed and read as deeply, scores a
        ring_level at or below its own. NaN without a reference."""
        ref = self.ring_ref.get(name)
        X = np.asarray(X, float)
        if ref is None or 'lvl' not in ref:
            return np.full(len(X), np.nan)
        lv = self.ring_level(name, X)
        q = placed['posterior']
        levels = ref['levels']
        b = np.clip(np.searchsorted(ref['cuts'], X.sum(1), side='right'), 0, ref['lvl'].shape[1] - 1)
        out = np.zeros(len(lv))
        for bi in np.unique(b):
            k = b == bi
            for t in range(self.T):
                out[k] += q[k, t] * np.interp(lv[k], ref['lvl'][t, bi], levels, left=0.0, right=1.0)
        return out

    def on_ring(self, name, X, miss=0.05, placed=None):
        """(mask, tail). The cell is on the ring when its tail (see
        _level_tail) is at least `miss`, so true on-ring cells are lost
        at a rate of about `miss` at EVERY angle -- by construction, not
        by threshold, and a fold is no longer a hole.

        What this does NOT ask is whether the cell sits inside the ring
        rather than on it. That question needs the centre as its
        reference, it has no answer wherever the ring passes through its
        own centre, and asking it anyway is what made the old gate empty
        those angles. Cells halfway in pass here (0.93-0.97); R is what
        says whether a cell may carry an angle.

        Without a reference for this panel the old fixed gate
        (bf_ring >= 0 and radius >= 0.5) is applied instead, and the
        tail is reported as 1 (pass) or 0 (fail)."""
        X = np.asarray(X, float)
        if placed is None:
            placed = self.place(name, X)
        tail = np.asarray(placed.get('ring_tail', np.nan), float) * np.ones(len(X))
        if not np.isfinite(tail).any():
            ok = ((np.asarray(placed['bf_ring'], float) >= 0.0)
                  & (np.asarray(placed['radius'], float) >= 0.5))
            return ok, ok.astype(float)
        return tail >= float(miss), tail


    def place(self, name, X, groups=None, prior='uniform'):
        """Every per-cell verdict at once: theta, R, fit_z, bf_ring,
        radius, ring_level and its calibrated tail (and the posterior)."""
        X = np.asarray(X, float)
        th, R, q = self.phase(name, X, groups, prior)
        rho, resid = self.radius(name, X)
        out = {'theta': th, 'R': R, 'fit_z': self.fit_z(name, X),
               'bf_ring': self.bf_ring(name, X), 'radius': rho, 'residual': resid,
               'ring_level': self.ring_level(name, X), 'posterior': q}
        out['ring_tail'] = self._level_tail(name, X, out)
        return out

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
                'gain_l2': self.gain_l2,
                'genes': list(self.genes), 'coef': self.coef.tolist(),
                'panels': {n: list(p) for n, p in self.panels.items()},
                'a': {n: v.tolist() for n, v in self.a.items()},
                's': {n: v.tolist() for n, v in self.s.items()},
                'dapi': {'weight': self.dapi_weight,
                         'h': None if self.h is None else self.h.tolist(),
                         'c': dict(self.c), 'sigma': dict(self.sigma)},
                'alphas': dict(self.alphas),
                'alpha_beta': {n: float(v) for n, v in self.alpha_beta.items()},
                'alpha_sref': {n: float(v) for n, v in self.alpha_sref.items()},
                'w': [[n, str(c), v.tolist()] for (n, c), v in self.w.items()],
                'ref': {n: {k: np.asarray(v).tolist() for k, v in r.items()} for n, r in self.ref.items()},
                'ring_ref': {n: {k: np.asarray(v).tolist() for k, v in r.items()} for n, r in self.ring_ref.items()}}

    @classmethod
    def from_dict(cls, d):
        m = cls(K=d['K'], T=d['T'], prior_smooth=d.get('prior_smooth', 5), l2=d.get('l2', 1e-2),
                gain_l2=d.get('gain_l2', 1.0))
        m.genes = list(d['genes'])
        m.gi = {g: i for i, g in enumerate(m.genes)}
        m.coef = np.asarray(d['coef'], float)
        m.panels = {n: list(p) for n, p in d['panels'].items()}
        m.cols = {n: np.array([m.gi[g] for g in p]) for n, p in m.panels.items()}
        m.a = {n: np.asarray(v, float) for n, v in d['a'].items()}
        # a model written before the gains existed has none: every gain is 1
        m.s = {n: np.asarray(d.get('s', {}).get(n, np.ones(len(p))), float) for n, p in m.panels.items()}
        m.free_s = {n: np.zeros(len(p), bool) for n, p in m.panels.items()}
        dp = d.get('dapi') or {}
        m.dapi_weight = float(dp.get('weight', 0.0))
        m.h = None if dp.get('h') is None else np.asarray(dp['h'], float)
        m.c = {k: float(v) for k, v in (dp.get('c') or {}).items()}
        m.sigma = {k: float(v) for k, v in (dp.get('sigma') or {}).items()}
        m.alphas = {n: (None if v is None else float(v)) for n, v in d['alphas'].items()}
        # a model written before the depth exponent existed has one alpha
        m.alpha_beta = {n: float(v) for n, v in (d.get('alpha_beta') or {}).items()}
        m.alpha_sref = {n: float(v) for n, v in (d.get('alpha_sref') or {}).items()}
        m.w ={(n, c): np.asarray(v, float) for n, c, v in d['w']}
        m.ref = {n: {k: np.asarray(v, float) for k, v in r.items()} for n, r in d.get('ref', {}).items()}
        m.ring_ref = {n: {k: np.asarray(v, float) for k, v in r.items()} for n, r in (d.get('ring_ref') or {}).items()}
        return m


# -- helpers around the model ------------------------------------------------

# -- the store: counts in, placements out --------------------------------------

COUNT_GATE = 0.5        # the classifier's own operating point (design doc 4.2)
CAPSULE_VERSION = 1
CAPSULE_COLUMNS = ('cell', 'theta_deg', 'R', 'fit_z', 'bf_ring', 'radius',
                   'ring_level', 'ring_tail', 'total')


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


def _mask_intensity_fov(item):
    """One FOV's per-cell mask statistics on one source's MIP -- module-
    level for pmap. The mask is projected into the source hybe's own
    frame through the store-built resolver (cell_crop's geometry); the
    background is the MIP's own mode over the FOV."""
    import os
    import numpy.linalg as la
    from codelab_pipeline.alignment import chain as alignment
    from codelab_pipeline.analysis import resolvers as R
    from codelab_pipeline.io import analysis_store, paths
    from codelab_pipeline.models.cell import ACell
    storage_path, fov, (modality, hybe, channel) = item
    dicts, _ = analysis_store.read_cells(storage_path, fov)
    if not dicts:
        return []
    mip_sp = os.path.join(paths.project_root(storage_path), modality)
    mip = np.asarray(analysis_store.read_hybe_mip(mip_sp, fov, hybe, int(channel)), float)
    try:
        resolver = R.resolver_for(storage_path, fov)
    except ValueError:
        # a store with two declared modalities and no cross-modal bridge
        # cannot name its hub; the cells' own modality is the frame here
        resolver = R.resolver_for(storage_path, fov, shared=str(dicts[0].get('reference_modality') or modality))
    H, W = mip.shape
    # THE BACKGROUND IS WHAT LIES OUTSIDE EVERY CELL. The MIP's histogram
    # mode is not it on a confluent field: measured on JP_002's DAPI, the
    # mode (665) sat above a third of the nuclei's own means and zeroed
    # their sums. Project every mask first, then take the median of the
    # uncovered pixels (the 5th percentile when almost nothing is
    # uncovered).
    projected = []
    covered = np.zeros((H, W), bool)
    for d in dicts:
        c = ACell()
        c.set_metadata(**d)
        Hc, _dz, _missing = resolver.transform((hybe, modality), (c.reference_hybe, c.reference_modality), c)
        y_lit, x_lit = c.area
        cy, cx = alignment.align_cell((y_lit, x_lit), la.inv(Hc), c.frame_shape)
        if len(cy) == 0:
            continue
        ys = np.clip(cy.astype(int), 0, H - 1)
        xs = np.clip(cx.astype(int), 0, W - 1)
        covered[ys, xs] = True
        projected.append((c, ys, xs))
    outside = mip[~covered & np.isfinite(mip)]
    finite = mip[np.isfinite(mip)]
    bg = float(np.median(outside)) if outside.size >= 0.01 * mip.size else float(np.percentile(finite, 5))
    rows = []
    for c, ys, xs in projected:
        v = mip[ys, xs]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({'fov': int(fov), 'cell': int(c.id), 'celltype': str(c.celltype or ''),
                     'area': int(v.size), 'mask_mean': float(v.mean()), 'mask_median': float(np.median(v)),
                     'sum_above_bg': float(np.clip(v - bg, 0, None).sum()), 'background': bg})
    return rows


def mask_intensity_table(storage_path, fovs, source, jobs=None, on_done=None):
    """Per cell: area, mask mean and median, and the sum above the FOV
    background of one source's MIP inside the cell mask -- the DNA
    content proxy when the source is DAPI (the routine verification:
    G2/M cells should carry about twice the DAPI of G1 cells). FOV-major
    through one pool. Returns (DataFrame, failures)."""
    import pandas as pd
    from codelab_pipeline import parallel
    fovs = [int(f) for f in fovs]
    items = [(storage_path, f, tuple(source)) for f in fovs]
    results = parallel.pmap(_mask_intensity_fov, items, kind='io', jobs=jobs, on_done=on_done)
    rows, fails = [], []
    for f, r in zip(fovs, results):
        if isinstance(r, parallel.Failure):
            fails.append((f, str(r)))
        else:
            rows.extend(r)
    cols = ['fov', 'cell', 'celltype', 'area', 'mask_mean', 'mask_median', 'sum_above_bg', 'background']
    return pd.DataFrame(rows, columns=cols), fails


def total_drop_angle(theta_deg, totals, n_bins=36, smooth=2, min_cells=200):
    """The angle where the panel total per cell falls most steeply --
    division, where the mRNA is halved and the G2/M transcripts go
    (measured on JP_001: every condition's median total drops several-
    fold within ~30 deg at the same angle). Returns (angle_deg,
    drop_factor) with the factor = the smoothed log-median's peak
    before the drop over its trough after it, or (None, None) when
    there are too few cells. A factor near 1 means no drop was found."""
    th = np.asarray(theta_deg, float) % 360.0
    s = np.asarray(totals, float)
    ok = np.isfinite(th) & np.isfinite(s) & (s > 0)
    if ok.sum() < min_cells:
        return None, None
    th, s = th[ok], s[ok]
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    b = np.clip(np.digitize(th, edges) - 1, 0, n_bins - 1)
    med = np.array([np.median(s[b == k]) if (b == k).sum() >= 5 else np.nan for k in range(n_bins)])
    if np.isnan(med).any():
        good = np.where(np.isfinite(med))[0]
        if len(good) < 6:
            return None, None
        x = np.concatenate([good - n_bins, good, good + n_bins])
        y = np.tile(med[good], 3)
        med = np.interp(np.arange(n_bins), x, y)
    lm = np.log(med)
    # a step detector, not a derivative: for every bin boundary, the mean
    # log-median over the `smooth`+2 bins before it minus the mean over
    # the same number after it; a symmetric window puts its maximum
    # exactly at the edge of a step (a smoothed derivative slides it)
    win = smooth + 2
    contrast = np.empty(n_bins)
    for j in range(n_bins):
        before = lm[[(j - i) % n_bins for i in range(1, win + 1)]].mean()
        after = lm[[(j + i) % n_bins for i in range(0, win)]].mean()
        contrast[j] = before - after
    j = int(np.argmax(contrast))
    angle = float(edges[j] % 360.0)
    before = lm[[(j - i) % n_bins for i in range(1, win + 1)]].max()
    after = lm[[(j + i) % n_bins for i in range(0, win)]].min()
    return angle, float(np.exp(before - after))


def _crops_fov(item):
    """One FOV's crops of one source's MIP around the wanted cells --
    module-level for pmap. Each crop is `size` x `size`, centred on the
    cell mask's centroid in the SOURCE hybe's own frame (the mask
    projected through the store-built resolver), NaN beyond the frame,
    with the mask as a boolean of the same shape. Also the FOV's 1st
    and 99.5th percentiles, so every tile of a FOV shares one scale."""
    import os
    import numpy.linalg as la
    from codelab_pipeline.alignment import chain as alignment
    from codelab_pipeline.analysis import resolvers as R
    from codelab_pipeline.io import analysis_store, paths
    from codelab_pipeline.models.cell import ACell
    storage_path, fov, (modality, hybe, channel), cell_ids, size = item
    dicts, _ = analysis_store.read_cells(storage_path, fov)
    if not dicts:
        return {}
    want = {int(c) for c in cell_ids}
    mip_sp = os.path.join(paths.project_root(storage_path), modality)
    mip = np.asarray(analysis_store.read_hybe_mip(mip_sp, fov, hybe, int(channel)), float)
    finite = mip[np.isfinite(mip)]
    lo, hi = (np.percentile(finite, [1, 99.5]) if finite.size else (0.0, 1.0))
    try:
        resolver = R.resolver_for(storage_path, fov)
    except ValueError:
        resolver = R.resolver_for(storage_path, fov, shared=str(dicts[0].get('reference_modality') or modality))
    H, W = mip.shape
    half = size // 2
    out = {}
    for d in dicts:
        if int(d['id']) not in want:
            continue
        c = ACell()
        c.set_metadata(**d)
        Hc, _dz, _missing = resolver.transform((hybe, modality), (c.reference_hybe, c.reference_modality), c)
        y_lit, x_lit = c.area
        cy, cx = alignment.align_cell((y_lit, x_lit), la.inv(Hc), c.frame_shape)
        if len(cy) == 0:
            continue
        y0, x0 = int(round(float(np.mean(cy)))), int(round(float(np.mean(cx))))
        crop = np.full((size, size), np.nan)
        mask = np.zeros((size, size), bool)
        ya, yb = y0 - half, y0 - half + size
        xa, xb = x0 - half, x0 - half + size
        sy, sx = slice(max(ya, 0), min(yb, H)), slice(max(xa, 0), min(xb, W))
        crop[sy.start - ya:sy.stop - ya, sx.start - xa:sx.stop - xa] = mip[sy, sx]
        my, mx = cy.astype(int) - ya, cx.astype(int) - xa
        k = (my >= 0) & (my < size) & (mx >= 0) & (mx < size)
        mask[my[k], mx[k]] = True
        out[int(c.id)] = (crop, mask, float(lo), float(hi))
    return out


def gallery_crops(storage_path, source, wanted, size=64, jobs=None):
    """{(fov, cell): (crop, mask, lo, hi)} for wanted = {fov: [cell ids]},
    one MIP read per FOV through one pool."""
    from codelab_pipeline import parallel
    items = [(storage_path, int(f), tuple(source), list(ids), int(size)) for f, ids in wanted.items() if ids]
    results = parallel.pmap(_crops_fov, items, kind='io', jobs=jobs)
    out = {}
    for (f, _ids), r in zip(((int(f), ids) for f, ids in wanted.items() if ids), results):
        if isinstance(r, parallel.Failure):
            continue
        for cid, v in r.items():
            out[(f, cid)] = v
    return out


def capsule_rows(placed, cell_ids, totals):
    """The per-FOV capsule's rows from place()'s dict, in the order the
    cells were placed. A verdict the caller's dict does not carry is
    written as NaN, the way fit_z already is when a model has no
    fit-quality reference."""
    def val(key, i):
        v = placed.get(key)
        return float(v[i]) if v is not None else float('nan')

    out = []
    for i, cid in enumerate(cell_ids):
        out.append({'cell': int(cid),
                    'theta_deg': float(np.degrees(placed['theta'][i]) % 360.0),
                    'R': float(placed['R'][i]),
                    'fit_z': val('fit_z', i),
                    'bf_ring': val('bf_ring', i),
                    'radius': val('radius', i),
                    'ring_level': val('ring_level', i),
                    'ring_tail': val('ring_tail', i),
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
             ('min_ring_tail', 'ring_tail', 'ge'),
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


def _calibration_fold(item):
    """One fold of calibrate_dispersion: fit the profiles on the training
    FOVs at the model's own alpha, then score the held-out cells under
    every (alpha, beta) candidate."""
    X, genes, train, test, fit_alpha, cands, s_ref, K, kw, n_iter = item
    m = CycleModel(K=K, **kw)
    m.fit([Dataset('cv', X[train], genes, alpha=fit_alpha)], n_iter=n_iter)
    m.alpha_sref['cv'] = float(s_ref)
    out = {}
    for a0, beta in cands:
        m.alphas['cv'] = None if a0 is None else float(a0)
        m.alpha_beta['cv'] = 0.0 if a0 is None else float(beta)
        out[(a0, beta)] = m.evidence('cv', X[test], prior='uniform')
    return out


DISPERSION_ALPHAS = (30, 50, 100, 200, 300, 500, 1000, 2000, None)
DISPERSION_BETAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
DISPERSION_MIN_GAIN = 0.1       # evidence per cell a beta must earn to be adopted


def calibrate_dispersion(model, name, X, fov, alphas=DISPERSION_ALPHAS, betas=DISPERSION_BETAS,
                         folds=5, seed=0, jobs=None, n_iter=20, n_depth=5,
                         min_gain=DISPERSION_MIN_GAIN, apply=True):
    """Set the dispersion a placed cell is read with, AFTER the profiles
    are fitted: alpha_i = alpha_0 (s_i / s_ref)^beta, chosen by held-out
    evidence in folds of whole FOVs.

    WHY AFTER THE FIT. alpha also weights each cell in the M-step
    ((1+alpha_i)/(s_i+alpha_i) effective counts), so fitting WITH a
    depth-dependent alpha hands the profiles to the deep cells: measured
    on JP_001 (beta 1), the ring came out 10.5 deg away and MIRRORED,
    its DNA ratio across division fell 1.62 -> 1.37, and the deepest
    cells then failed their own gate (0.81 kept). Applied after the fit
    the same calibration moves placements by 0.7 deg, leaves the DNA
    ratio at 1.60, and evens the gate across depth (0.77 -> 0.94 in the
    shallowest fifth).

    WHY FOLDS OF FOVS. Cells of one field share its focus, background and
    stain; random folds put a field's twins on both sides and reward the
    alpha that fits that field's quirks.

    WHEN IT DOES NOTHING. beta is adopted only if it earns `min_gain`
    evidence per cell over the best single alpha -- on JP_002 it earns
    0.036 and the single alpha is kept; on JP_001, 0.24, and alpha(s) is
    taken (its depth quintiles want alpha 100 to 1000).

    `X` and `fov` are the cells the model was fitted on. With apply=True
    and the gain cleared, the model's dispersion is replaced and its ring
    reference dropped, so the caller rebuilds it (ring_reference) before
    gating. Returns the report."""
    from codelab_pipeline import parallel
    X = np.asarray(X, float)
    fov = np.asarray(fov)
    if len(fov) != len(X):
        raise ValueError('calibrate_dispersion: one FOV label per cell')
    genes = list(model.panels[name])
    if X.shape[1] != len(genes):
        raise ValueError(f'calibrate_dispersion: X has {X.shape[1]} columns, the panel has {len(genes)}')
    s = X.sum(1)
    s_ref = float(np.median(s[s > 0])) if (s > 0).any() else 1.0
    uf = np.unique(fov)
    if len(uf) < 2:
        raise ValueError('calibrate_dispersion needs cells from at least two FOVs')
    folds = int(min(folds, len(uf)))
    perm = np.random.default_rng(seed).permutation(uf)
    fold_of = {f: i % folds for i, f in enumerate(perm)}
    fo = np.array([fold_of[f] for f in fov])
    cands = [(a, b) for a in alphas for b in (betas if a is not None else (0.0,))]
    kw = {'T': model.T, 'prior_smooth': model.prior_smooth, 'l2': model.l2,
          'mstep': model.mstep, 'polish': 0}
    fit_alpha = model.alphas.get(name)
    items = [(X, genes, np.flatnonzero(fo != f), np.flatnonzero(fo == f), fit_alpha,
              cands, s_ref, model.K, kw, n_iter) for f in range(folds)]
    res = parallel.pmap(_calibration_fold, items, kind='cpu', jobs=jobs)
    ev = {c: np.full(len(X), np.nan) for c in cands}
    for f, r in enumerate(res):
        if isinstance(r, parallel.Failure):
            raise RuntimeError(f'calibrate_dispersion: fold {f}: {r}')
        test = np.flatnonzero(fo == f)
        for c, v in r.items():
            ev[c][test] = v
    scores = {c: float(np.nanmean(v)) for c, v in ev.items()}
    edges = np.quantile(s, np.linspace(0, 1, n_depth + 1)[1:-1])
    qb = np.searchsorted(edges, s, side='right')
    by_depth = {c: [float(np.nanmean(v[qb == j])) for j in range(n_depth)] for c, v in ev.items()}
    best = max(scores, key=lambda c: scores[c])
    flat = {c: v for c, v in scores.items() if c[1] == 0.0}
    best_scalar = max(flat, key=lambda c: flat[c])
    gain = scores[best] - flat[best_scalar]
    take = bool(apply and best[1] != 0.0 and gain >= float(min_gain))
    if apply:
        a0, beta = (best if take else (best_scalar[0], 0.0))
        model.alphas[name] = None if a0 is None else float(a0)
        model.alpha_beta[name] = float(beta)
        model.alpha_sref[name] = s_ref
        model.ring_ref.pop(name, None)
    return {'scores': scores, 'by_depth': by_depth, 'depth_edges': edges.tolist(), 's_ref': s_ref,
            'best': best, 'best_scalar': best_scalar[0], 'gain': float(gain),
            'adopted': ((best if take else (best_scalar[0], 0.0)) if apply else None),
            'min_gain': float(min_gain), 'folds': folds, 'n_cells': int(len(X))}


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
             theta=None, alpha=None, alpha_beta=0.0):
    """Synthetic cells on a circle, for tests and self-checks. alpha_beta
    makes the dispersion depend on depth, alpha_i = alpha * (s_i /
    median s)^alpha_beta (the totals are then drawn first, so the random
    stream differs from the single-alpha one)."""
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
    if alpha is not None and alpha_beta:
        s = rng.integers(total[0], total[1], n)
        ai = _alpha_at(alpha, alpha_beta, float(np.median(s)), s)
        pi = np.stack([rng.dirichlet(a_ * p) for a_, p in zip(ai, pi)])
        X = np.stack([rng.multinomial(int(si), p) for si, p in zip(s, pi)])
        return X, theta, coef, a, genes
    if alpha is not None:
        pi = np.stack([rng.dirichlet(alpha * p) for p in pi])
    s = rng.integers(total[0], total[1], n)
    X = np.stack([rng.multinomial(int(si), p) for si, p in zip(s, pi)])
    return X, theta, coef, a, genes
