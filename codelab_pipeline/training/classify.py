"""
Pass/fail for one spot box, as a calibrated p.

WHY TORCH FOR A MODEL THIS SMALL. Not for capacity -- a logistic
regression on thirty features has thirty-one parameters and numpy could
solve it. For the SCAFFOLDING: a loss, an optimiser, a train/validation
loop, early stopping, a saved artifact with its own preprocessing baked
in. When there are enough labels to justify a real network, the network
drops into this same loop and everything around it stays. torch is
already a first-class dependency here (requirements.txt pins it for
cellpose), so this adds nothing to install. scikit-learn is NOT
installed and is not worth adding for a linear model.

THREE HEADS, ONE HARNESS, so they are compared rather than argued about:

  linear   logistic regression on features.NAMES. The default. At a
           thousand labels this is usually the honest ceiling.
  mlp      one hidden layer of 16 on the same features.
  conv     a small 3D convnet on the raw box -- ~4k parameters. Here so
           that "the features are throwing information away" is a
           measurement rather than a worry.

WHAT p MEANS. LocalizedSpot's own docstring says p is a per-engine
quality scalar and NOT a probability unless the engine says otherwise.
This one says otherwise: the head is trained with a proper scoring rule
(BCE) and then Platt-scaled on held-out cells, so p is meant to be read
as a probability. It is only as calibrated as the labels are consistent,
and the contested bucket is the honest measure of that.

AUGMENTATION IS PHYSICS, NOT NOISE. Flips and 90-degree rotations in xy
are label-preserving because the lateral PSF is symmetric under them.
A FLIP IN Z IS NOT: the PSF is axially asymmetric -- above focus does
not look like below focus -- so mirroring z would teach the model that
the two are the same thing.

THE SPLIT IS BY CELL (dataset.split_by_group). Anything else scores the
model on cells it has already seen.
"""
import json
import os

import numpy as np

from . import features as F

HEADS = ('linear', 'mlp', 'conv')


def _torch():
    import torch
    return torch


def make_head(name, n_features, box_shape=None):
    torch = _torch()
    nn = torch.nn
    if name == 'linear':
        return nn.Linear(n_features, 1)
    if name == 'mlp':
        return nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(),
                             nn.Dropout(0.2), nn.Linear(16, 1))
    if name == 'conv':
        return _Conv3d(box_shape)
    raise ValueError(f'unknown head {name!r}; expected one of {HEADS}')


def _Conv3d(box_shape):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(8, 16, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool3d(1))
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            if x.dim() == 4:
                x = x.unsqueeze(1)
            return self.head(self.body(x).flatten(1))

    return Net()


def augment(batch, rng):
    """Label-preserving symmetries of a lateral PSF. No z flip."""
    out = batch
    if rng.random() < 0.5:
        out = np.flip(out, axis=1)
    if rng.random() < 0.5:
        out = np.flip(out, axis=2)
    k = int(rng.integers(0, 4))
    if k:
        out = np.rot90(out, k, axes=(1, 2))
    return np.ascontiguousarray(out) * float(rng.uniform(0.8, 1.25))


def pr_auc(y, p):
    """Average precision. Not accuracy: the classes are unbalanced and
    the reviewer's default is reject, so accuracy rewards saying no."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    o = np.argsort(-p)
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def roc_auc(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    from scipy import stats
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class SpotClassifier:
    """A trained head plus everything needed to score a fresh box."""

    def __init__(self, head='linear', std=None, model=None,
                 platt=(1.0, 0.0), threshold=0.5, meta=None):
        self.head = head
        self.std = std
        self.model = model
        self.platt = tuple(platt)
        self.threshold = float(threshold)
        self.meta = dict(meta or {})

    # -- inference -------------------------------------------------------

    def _logits(self, X=None, boxes=None):
        torch = _torch()
        self.model.eval()
        with torch.no_grad():
            if self.head == 'conv':
                t = torch.as_tensor(np.asarray(boxes, np.float32))
            else:
                # NO STANDARDISER MEANS IDENTITY, not a crash. load()
                # already builds one with std=None whenever the saved
                # document carries none -- that is the conv path today,
                # but nothing stops a feature head from being built
                # without one, and `self.std(X)` on None is a
                # TypeError raised from inside scoring rather than an
                # answer or a refusal.
                t = torch.as_tensor(np.asarray(
                    X if self.std is None else self.std(X), np.float32))
            return self.model(t).flatten().numpy()

    @property
    def feature_names(self):
        """What this head was trained on: features.NAMES, plus any context
        names after them."""
        return tuple((self.meta or {}).get('features') or F.NAMES)

    @property
    def context_names(self):
        """The context features this head needs at scoring time."""
        return tuple(self.feature_names[len(F.NAMES):])

    def with_context(self, X, context=None):
        """Append the context columns this head was trained with to a
        (n, len(NAMES)) matrix. A head without context returns X as is;
        one with it raises when the context is missing."""
        names = self.context_names
        X = np.asarray(X, float)
        if not names or X.ndim != 2:
            return X
        if X.shape[1] == len(self.feature_names):
            return X                      # already complete
        v = F.context_vector(context, names)
        return np.hstack([X, np.tile(v, (X.shape[0], 1))])

    def score(self, X=None, boxes=None):
        """Calibrated p, STRICTLY inside (0, 1).

        A p of exactly 0 or 1 claims certainty no finite training set
        supports, and it is an infinity to anything downstream that takes
        a log of it.

        THE FLOAT64 CAST IS THE LOAD-BEARING PART, not the clip. torch
        hands back float32, and in float32 the sigmoid of the clipped
        logit IS exactly 1.0 -- 1 - 9.3e-14 needs more precision than
        float32 has, its epsilon being 1.2e-7. Clipping the logit to
        +-30 looked like enough and was not: MEASURED on the real gated
        set, where logits reach 72, p.max() came back as exactly 1.0
        while p.min() was a healthy 9.4e-14. The asymmetry is the
        giveaway -- small numbers survive float32, numbers a hair below
        one do not. The synthetic tests never pushed a logit far enough
        to see it.
        """
        a, b = self.platt
        z = np.clip(np.asarray(self._logits(X, boxes), dtype=np.float64)
                    * float(a) + float(b), -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.clip(p, 1e-12, 1.0 - 1e-12)

    def decide(self, X=None, boxes=None):
        return self.score(X, boxes) >= self.threshold

    # -- persistence -----------------------------------------------------

    def save(self, path):
        torch = _torch()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.',
                    exist_ok=True)
        doc = {'head': self.head, 'platt': list(self.platt),
               'threshold': self.threshold, 'meta': self.meta,
               'standardiser': (self.std.to_dict() if self.std else None),
               'state': {k: v.tolist()
                         for k, v in self.model.state_dict().items()}}
        tmp = str(path) + '.part'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        os.replace(tmp, str(path))       # the project's one write pattern
        return str(path)

    @staticmethod
    def load(path):
        torch = _torch()
        with open(str(path), encoding='utf-8') as f:
            doc = json.load(f)
        std = (F.Standardiser.from_dict(doc['standardiser'])
               if doc.get('standardiser') else None)
        shape = doc.get('meta', {}).get('box_shape')
        names = (doc.get('meta') or {}).get('features') or F.NAMES
        model = make_head(doc['head'], len(names), shape)
        model.load_state_dict({k: torch.as_tensor(np.asarray(v, np.float32))
                               for k, v in doc['state'].items()})
        return SpotClassifier(doc['head'], std, model, doc['platt'],
                              doc['threshold'], doc.get('meta'))


def _platt(logit, y):
    """One-dimensional logistic fit mapping a logit onto a probability.

    Held-out cells only. Without it the head's raw output is a ranking,
    not a probability, and a threshold chosen on it means nothing on the
    next bundle.

    THE TARGETS ARE SMOOTHED, which is Platt's own correction and not a
    detail. Fitted against hard 0/1 on a validation set the head happens
    to separate cleanly, the slope has no reason to stop growing: it runs
    away, the sigmoid saturates, and p comes back as exactly 0.0 and
    exactly 1.0 -- infinite confidence from a few hundred labels, and a
    number no downstream log-likelihood can use. (Caught by this
    project's own test on synthetic separable data, where every head did
    it.) The smoothed targets bound the slope by the amount of evidence
    there actually is.
    """
    torch = _torch()
    y = np.asarray(y, float)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    hi = (n1 + 1.0) / (n1 + 2.0)
    lo = 1.0 / (n0 + 2.0)
    t = torch.as_tensor(np.where(y > 0.5, hi, lo).astype(np.float32))
    z = torch.as_tensor(np.asarray(logit, np.float32))
    a = torch.ones(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=200)
    lossf = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(a * z + b, t)
        loss.backward()
        return loss
    opt.step(closure)
    av, bv = float(a.item()), float(b.item())
    if not (np.isfinite(av) and np.isfinite(bv)):
        return 1.0, 0.0
    return av, bv


def train(X, boxes, y, groups, head='linear', epochs=400, lr=0.02,
          weight_decay=1e-3, seed=0, val_frac=0.25, verbose=True,
          feature_names=None):
    """Fit one head. Returns (SpotClassifier, report dict).

    `groups` splits by CELL. `y` must be 0/1 -- contested rows belong
    nowhere in here. `feature_names` records what the columns of X are
    (features.NAMES, plus context names after them); the head is sized
    from X and refuses to score anything else.
    """
    names = list(feature_names or F.NAMES)
    if head != 'conv' and np.asarray(X).shape[1] != len(names):
        raise ValueError(f'X has {np.asarray(X).shape[1]} columns but '
                         f'{len(names)} feature names were given')
    torch = _torch()
    rows = [{'group': g} for g in groups]
    from . import dataset as D
    tr, va = D.split_by_group(rows, frac=val_frac, seed=seed)
    y = np.asarray(y, int)
    if not tr or not va or y[tr].sum() == 0 or y[va].sum() == 0:
        raise ValueError('the split left a side with no positives; there are '
                         'too few labelled cells to hold any out')

    std = F.Standardiser().fit(np.asarray(X)[tr]) if head != 'conv' else None
    # THE WEIGHTS ARE SEEDED TOO, or two trainings of the same verdicts
    # are two different models. `seed` already fixed the split and the
    # augmentation; make_head draws its initial weights from torch's own
    # generator, which nothing set. MEASURED: identical inputs, linear
    # PR-AUC 0.9595 vs 0.9604 and different `state` in both heads, while
    # the PSF bank came out numerically identical. A person retraining
    # under the same run name expects the same M1 back.
    _torch().manual_seed(int(seed))
    model = make_head(head, np.asarray(X).shape[1],
                      None if head != 'conv' else np.asarray(boxes).shape[1:])

    if head == 'conv':
        Xtr = np.asarray(boxes, np.float32)[tr]
        Xva = torch.as_tensor(np.asarray(boxes, np.float32)[va])
    else:
        Xtr = np.asarray(std(np.asarray(X)[tr]), np.float32)
        Xva = torch.as_tensor(np.asarray(std(np.asarray(X)[va]), np.float32))
    ytr = torch.as_tensor(y[tr].astype(np.float32))
    yva = y[va]

    # The reviewer's default is reject, so negatives outnumber positives;
    # without this the cheapest loss is to say no to everything.
    pos = float(y[tr].sum())
    neg = float(len(tr) - pos)
    pw = torch.as_tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    best = (-1.0, None, 0)
    hist = []
    for ep in range(int(epochs)):
        model.train()
        xb = (torch.as_tensor(augment(Xtr, rng).astype(np.float32))
              if head == 'conv' else torch.as_tensor(Xtr))
        opt.zero_grad()
        loss = lossf(model(xb).flatten(), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            lv = model(Xva).flatten().numpy()
        ap = pr_auc(yva, lv)
        hist.append((ep, float(loss.item()), ap))
        if np.isfinite(ap) and ap > best[0]:
            best = (ap, {k: v.detach().clone()
                         for k, v in model.state_dict().items()}, ep)
    if best[1] is not None:
        model.load_state_dict(best[1])

    model.eval()
    with torch.no_grad():
        lv = model(Xva).flatten().numpy()
    platt = _platt(lv, yva)
    clf = SpotClassifier(head, std, model, platt, 0.5,
                         meta={'box_shape': (list(np.asarray(boxes).shape[1:])
                                             if head == 'conv' else None),
                               'features': names,
                               'n_train': len(tr), 'n_val': len(va),
                               'seed': seed})
    p = clf.score(np.asarray(X)[va] if head != 'conv' else None,
                  np.asarray(boxes)[va] if head == 'conv' else None)
    report = {
        'head': head, 'n_train': len(tr), 'n_val': len(va),
        'val_positive_frac': float(yva.mean()),
        'val_pr_auc': float(pr_auc(yva, p)),
        'val_roc_auc': float(roc_auc(yva, p)),
        'best_epoch': int(best[2]),
        'p_min': float(p.min()), 'p_max': float(p.max()),
        'p_median': float(np.median(p)),
        'predicted_positive_frac': float((p >= 0.5).mean()),
        'train_groups': len({tuple(groups[i]) for i in tr}),
        'val_groups': len({tuple(groups[i]) for i in va}),
    }
    if verbose:
        print(f"   {head:6s} val PR-AUC {report['val_pr_auc']:.3f}"
              f"  ROC {report['val_roc_auc']:.3f}"
              f"  p in [{report['p_min']:.3f}, {report['p_max']:.3f}]"
              f"  says yes to {100 * report['predicted_positive_frac']:.0f}%")
    return clf, report


REPORT_NAME = 'report.json'


def load_best(model_dir, report=REPORT_NAME, prefer=None):
    """The trained head a run's own numbers chose. (classifier, why).

    NOTHING IN THIS REPO LOADED A TRAINED CLASSIFIER BEFORE THIS. train
    wrote spot_classifier_<head>.json and report.json and stopped; the
    only caller of SpotClassifier.load was a test. So the existence
    probability the whole training stack exists to produce reached no
    engine, no gate and no z_status -- it was a file on disk.

    THE CHOICE IS THE REPORT'S, NOT A CONSTANT. train stores every head
    it was asked for and records no winner, so a hard-coded 'mlp' here
    would be a number that stops tracking the data the moment anyone
    re-trains. This reads held-out PR-AUC out of the run's own report and
    takes the best, which follows a re-train for free.

    THREE WAYS A SAVED HEAD IS REFUSED, all of them seen in the one model
    directory that exists (D:/models/mp58_rna):

      it is not in the report          spot_classifier_conv.json is there
                                       and conv is not, because the last
                                       run was --heads linear,mlp. A file
                                       nobody measured is not a candidate.

      its features are not ours        that same conv file records 21
                                       feature names against features.
                                       NAMES' 16. Scoring it would read
                                       the wrong column for every one.

      the run called it degenerate     all_fail / all_pass / degenerate
                                       are already in the report; a head
                                       that predicts one class is not a
                                       probability, whatever its AUC.

    `prefer` names a head to take if it is usable, for pinning a
    comparison. It never overrides a refusal.
    """
    d = str(model_dir)
    rp = os.path.join(d, report)
    with open(rp, encoding='utf-8') as f:
        rep = json.load(f)
    heads = rep.get('heads') or {}
    if not heads:
        raise ValueError(f'{rp} records no trained head')

    usable, refused = [], []
    for name, h in sorted(heads.items()):
        path = h.get('path') or os.path.join(d, f'spot_classifier_{name}.json')
        if not os.path.exists(path):
            path = os.path.join(d, f'spot_classifier_{name}.json')
        if not os.path.exists(path):
            refused.append((name, 'no saved file'))
            continue
        if h.get('degenerate') or h.get('all_fail') or h.get('all_pass'):
            refused.append((name, 'the run called it degenerate'))
            continue
        try:
            with open(path, encoding='utf-8') as f:
                feats = (json.load(f).get('meta') or {}).get('features')
        except (OSError, ValueError):
            refused.append((name, 'unreadable'))
            continue
        # OURS: features.NAMES first, then only context names we know.
        if feats is not None and not (
                tuple(feats[:len(F.NAMES)]) == tuple(F.NAMES)
                and all(f in F.CONTEXT_NAMES for f in feats[len(F.NAMES):])):
            refused.append((name, f'{len(feats)} features, not our '
                                  f'{len(F.NAMES)} (+ known context)'))
            continue
        auc = h.get('val_pr_auc')
        if auc is None:
            refused.append((name, 'no held-out PR-AUC'))
            continue
        usable.append((float(auc), name, path))

    # And the files sitting in the directory that the report never mentions.
    for n in sorted(os.listdir(d)):
        if n.startswith('spot_classifier_') and n.endswith('.json'):
            head = n[len('spot_classifier_'):-len('.json')]
            if head not in heads:
                refused.append((head, 'not in the report'))
    if not usable:
        raise ValueError(
            f'{rp} has no usable head. Refused: '
            + '; '.join(f'{n} ({why})' for n, why in refused))
    usable.sort(key=lambda t: (-t[0], t[1]))
    if prefer:
        for auc, name, path in usable:
            if name == str(prefer):
                usable = [(auc, name, path)] + [u for u in usable
                                                if u[1] != name]
                break
    auc, name, path = usable[0]
    why = {'head': name, 'val_pr_auc': auc, 'path': path,
           'considered': {n: a for a, n, _p in usable},
           'refused': dict(refused),
           'report': rp, 'bundle': rep.get('bundle')}
    return SpotClassifier.load(path), why


MULTISPOT_NAME = 'psf_multispot.json'


class MultispotCalibration(object):
    """Platt on the matcher's own score. TWO PARAMETERS, SHIPPED.

    THE PROBLEM THIS SOLVES IS A SHIPPING PROBLEM, not an accuracy one.
    A model directory carries a classifier (weights + Platt) and a PSF
    template, and both work on a new experiment with ZERO new labels.
    The matcher's operating point did not: it was a number measured from
    multispot verdicts, so using it meant assuming every experiment comes
    with its own multispot review, and hard-coding it meant shipping one
    experiment's answer as a universal constant. Neither is acceptable.

    A Platt pair fixes it by being the same KIND of artefact as the
    classifier: fitted once against people, written into the model
    directory, applied with no labels, retrained by anyone who has their
    own. And the threshold stops being a magic number -- a calibrated
    probability is thresholded at 0.5.

    IT ALSO REMOVES A TEMPLATE DEPENDENCY. The raw score is an affine
    remap of an NCC, and psf_bank.K_SIGMA's header explains why a raw NCC
    means something different at another template size. The calibration
    is fitted with a template recorded beside it and refuses one it was
    not measured on, so that dependency is checked instead of forgotten.

    MEASURED on 661 human-judged matches from 260 pillars of MP58/RNA:
    the raw score already separates keep from drop at PR-AUC 0.993, and
    the best raw threshold (0.732) is stable across the experiment's own
    hybes (per-hybe optima 0.680-0.739, and the global one holds every
    hybe at precision >= 0.90, recall >= 0.96). So this layer is not
    buying discrimination -- the ranking is p3's and stays p3's. It is
    buying a number that can be SHIPPED.
    """

    def __init__(self, platt, template=None, meta=None):
        self.platt = (float(platt[0]), float(platt[1]))
        self.template = tuple(template) if template else None
        self.meta = dict(meta or {})

    def score(self, p3):
        """Raw matcher score -> calibrated probability, strictly in (0, 1).

        float64 and clipped for the reason SpotClassifier.score gives at
        length: a p of exactly 0 or 1 claims certainty no finite label
        set supports and is an infinity to anything taking its log.
        """
        p = np.clip(np.asarray(p3, dtype=np.float64), 1e-9, 1.0 - 1e-9)
        logit = np.log(p / (1.0 - p))
        a, b = self.platt
        z = np.clip(logit * a + b, -30.0, 30.0)
        return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-12, 1.0 - 1e-12)

    def check(self, template):
        """Refuse a template this was not measured on."""
        if self.template and template is not None \
                and tuple(template) != tuple(self.template):
            raise ValueError(
                f'this multispot calibration was fitted with a '
                f'{tuple(self.template)} template and the engine is using '
                f'{tuple(template)}. The score it calibrates is an affine '
                f'remap of a raw NCC, which means something different at '
                f'another template size -- see psf_bank.K_SIGMA.')
        return True

    def save(self, path):
        doc = {'platt': list(self.platt),
               'template': list(self.template) if self.template else None,
               'meta': self.meta}
        tmp = str(path) + '.part'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=1)
        os.replace(tmp, str(path))
        return str(path)

    @staticmethod
    def load(path):
        with open(str(path), encoding='utf-8') as f:
            doc = json.load(f)
        return MultispotCalibration(doc['platt'], doc.get('template'),
                                    doc.get('meta'))


def fit_multispot(bundle_dir, template=None):
    """(MultispotCalibration, report) from the multispot verdicts of one
    bundle, or of several pooled.

    Returns (None, report) when there is nothing to fit -- no verdicts, or
    every match judged the same way. A calibration invented from one class
    is a number with no evidence under it. With several bundles the
    report names the ones that actually carried verdicts.
    """
    from . import verdicts as V
    dirs = ([str(bundle_dir)] if isinstance(bundle_dir, str)
            else [str(b) for b in bundle_dir])
    recs, used = [], []
    for b in dirs:
        rb, _agree = V.merge(b, kind=V.MULTISPOT_KIND)
        if rb:
            used.append(b)
        recs.extend(rb)
    shown = [e for r in recs for e in (r.get('shown') or [])
             if e.get('p') is not None]
    # WHICH BANK SCORED THEM. Each record carries the bank Spot Check was
    # matching with; pooled bundles may have been judged against
    # different banks, and a calibration fitted across them should say
    # so rather than look like one measurement.
    banks = sorted({str(r.get('bank_path')) for r in recs if r.get('bank_path')})
    rep = {'pillars': len(recs), 'n': len(shown),
           'template': list(template) if template else None,
           'bundles': used, 'banks': banks}
    if not shown:
        rep['skipped'] = ('no multispot verdicts in this bundle'
                          if len(dirs) == 1 else
                          'no multispot verdicts in any of these bundles')
        return None, rep
    y = np.asarray([1 if int(e.get('keep', 0)) == 1 else 0 for e in shown],
                   float)
    p = np.clip(np.asarray([float(e['p']) for e in shown], float),
                1e-9, 1.0 - 1e-9)
    rep['n_kept'] = int(y.sum())
    if not y.any() or y.all():
        rep['skipped'] = 'every judged match fell in one class'
        return None, rep
    logit = np.log(p / (1.0 - p))
    a, b = _platt(logit, y)
    cal = MultispotCalibration((a, b), template,
                              {k: rep[k] for k in ('pillars', 'n', 'n_kept')})
    q = cal.score(p)
    rep.update(platt=[float(a), float(b)],
               raw_pr_auc=float(pr_auc(y, p)),
               cal_pr_auc=float(pr_auc(y, q)),
               kept_at_half=int((q >= 0.5).sum()),
               precision_at_half=float(y[q >= 0.5].mean()) if (q >= 0.5).any()
               else float('nan'),
               recall_at_half=float((q[y == 1] >= 0.5).mean()))
    return cal, rep
