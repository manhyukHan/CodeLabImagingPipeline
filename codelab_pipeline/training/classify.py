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
                t = torch.as_tensor(np.asarray(self.std(X), np.float32))
            return self.model(t).flatten().numpy()

    def score(self, X=None, boxes=None):
        """Calibrated p, STRICTLY inside (0, 1).

        The clip is not cosmetic. A p of exactly 0 or 1 claims certainty
        no finite training set supports, and it is an infinity to
        anything downstream that takes a log of it.
        """
        a, b = self.platt
        z = np.clip(a * self._logits(X, boxes) + b, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

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
        model = make_head(doc['head'], len(F.NAMES), shape)
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
          weight_decay=1e-3, seed=0, val_frac=0.25, verbose=True):
    """Fit one head. Returns (SpotClassifier, report dict).

    `groups` splits by CELL. `y` must be 0/1 -- contested rows belong
    nowhere in here.
    """
    torch = _torch()
    rows = [{'group': g} for g in groups]
    from . import dataset as D
    tr, va = D.split_by_group(rows, frac=val_frac, seed=seed)
    y = np.asarray(y, int)
    if not tr or not va or y[tr].sum() == 0 or y[va].sum() == 0:
        raise ValueError('the split left a side with no positives; there are '
                         'too few labelled cells to hold any out')

    std = F.Standardiser().fit(np.asarray(X)[tr]) if head != 'conv' else None
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
                               'features': list(F.NAMES),
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
