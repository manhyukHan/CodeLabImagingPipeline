"""
Segmentation works on Cellpose 3 and Cellpose 4, and calls each correctly.

WHY THIS EXISTS: Cellpose 4 deleted `models.Cellpose` and the cyto3 weights
with it, so the four construction sites in segment.py failed with
`AttributeError: module 'cellpose.models' has no attribute 'Cellpose'` --
at SEGMENTATION time, not import time, which is the worst place to find
out. The project pinned `cellpose>=3.1,<4` to avoid it. That pin is now
gone and the installed version decides, so something has to guard the
dispatch.

Only ONE version can be installed at a time, so the dispatch cannot be
covered by running it. These tests substitute a fake `cellpose` module and
assert on what segment.py *asks for* -- which class, which arguments. That
is exactly the part that broke.

The two facts the fakes encode were MEASURED against the real packages, not
assumed (scratchpad probe, 2026-09-06, cellpose 4.2.1.1 + torch 2.7.1+cu118,
RTX 3070):

  * models.Cellpose is absent in 4.x; CellposeModel is present.
  * 4.x accepts `channels=` and IGNORES it, warning once per call. Masks
    were bit-identical with and without, so segment.py drops it there
    rather than passing it to be swallowed.

Set CODELAB_TEST_REAL_CELLPOSE=1 to additionally run whatever version is
really installed against a synthetic field -- slow (a model load is 10-16 s)
and needs the weights on disk, so it is off by default.

Run:  QT_QPA_PLATFORM=offscreen python tests/test_cellpose_compat.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402

from codelab_pipeline.segmentation import segment          # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"ok" if cond else "FAIL"} {name}' + (f'   [{detail}]' if detail else ''))


class FakeModel:
    """Records how it was constructed and how eval() was called."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.eval_calls = []

    def eval(self, images, **kwargs):
        self.eval_calls.append((images, kwargs))
        n = 4 if 'channels' in kwargs else 3
        return tuple([np.zeros((4, 4), dtype=np.uint16)] for _ in range(n))


def install_fake_cellpose(version, classes):
    """Put a fake `cellpose` package in sys.modules and reset segment.py's cache."""
    pkg = types.ModuleType('cellpose')
    pkg.version = version
    models = types.ModuleType('cellpose.models')
    for name, factory in classes.items():
        setattr(models, name, factory)
    pkg.models = models
    sys.modules['cellpose'] = pkg
    sys.modules['cellpose.models'] = models
    segment._cellpose_major_cached = None
    segment._model_cyto = None
    segment._model_cyto_nuc = None
    return models


def fake_v3():
    """3.x: models.Cellpose exists, CellposeModel does not."""
    made = []

    def Cellpose(**kw):
        m = FakeModel(**kw)
        made.append(('Cellpose', m))
        return m

    install_fake_cellpose('3.1.1.3', {'Cellpose': Cellpose})
    return made


def fake_v4():
    """4.x: only CellposeModel. Constructing Cellpose must never be attempted."""
    made = []

    def CellposeModel(**kw):
        m = FakeModel(**kw)
        made.append(('CellposeModel', m))
        return m

    install_fake_cellpose('4.2.1.1', {'CellposeModel': CellposeModel})
    return made


def main():
    saved = {k: sys.modules.get(k) for k in ('cellpose', 'cellpose.models')}
    try:
        print('\n-- the installed version is detected, not guessed --')
        for version, want in [('3.1.1.3', 3), ('3.0', 3), ('4.2.1.1', 4),
                              ('4.0.1', 4), ('4.1.1', 4)]:
            install_fake_cellpose(version, {'Cellpose': FakeModel,
                                            'CellposeModel': FakeModel})
            got = segment.cellpose_major()
            check(f'cellpose {version} reads as major {want}', got == want, str(got))

        print('\n-- on 3.x it builds the cyto3 Cellpose --')
        made = fake_v3()
        model = segment.make_cellpose_model(gpu=True)
        check('constructs models.Cellpose', made and made[0][0] == 'Cellpose')
        check('asks for cyto3', model.init_kwargs.get('model_type') == 'cyto3',
              str(model.init_kwargs))
        check('honours gpu=True', model.init_kwargs.get('gpu') is True)
        cpu = segment.make_cellpose_model(gpu=False)
        check('honours gpu=False for the CPU fallback',
              cpu.init_kwargs.get('gpu') is False)

        print('\n-- on 4.x it builds CellposeModel, and never touches Cellpose --')
        made = fake_v4()
        model4 = segment.make_cellpose_model(gpu=True)
        check('constructs models.CellposeModel', made and made[0][0] == 'CellposeModel')
        check('names the checkpoint explicitly rather than taking the default',
              model4.init_kwargs.get('pretrained_model') == segment.CELLPOSE4_MODEL,
              str(model4.init_kwargs))
        check('does not pass model_type -- 4.x has no cyto3',
              'model_type' not in model4.init_kwargs)
        check('the pinned name is one 4.x actually ships',
              segment.CELLPOSE4_MODEL in ('cpsam', 'cpsam_v2'),
              segment.CELLPOSE4_MODEL)

        print('\n-- eval passes channels on 3.x and withholds it on 4.x --')
        fake_v3()
        m3 = segment.make_cellpose_model()
        segment.cellpose_eval(m3, ['img'], 40, [0, 0])
        _, kw3 = m3.eval_calls[0]
        check('3.x is given channels', kw3.get('channels') == [0, 0], str(kw3))
        check('3.x is given the diameter', kw3.get('diameter') == 40)
        check('3.x is given do_3D=False', kw3.get('do_3D') is False)

        fake_v4()
        m4 = segment.make_cellpose_model()
        segment.cellpose_eval(m4, ['img'], 40, [0, 0])
        _, kw4 = m4.eval_calls[0]
        check('4.x is NOT given channels (it would only warn and ignore)',
              'channels' not in kw4, str(kw4))
        check('4.x is still given the diameter', kw4.get('diameter') == 40)
        check('4.x is still given do_3D=False', kw4.get('do_3D') is False)

        print('\n-- masks are read by position, so both tuple lengths work --')
        # 3.x returns (masks, flows, styles, diams); 4.x returns 3 of those.
        fake_v3()
        r3 = segment.cellpose_eval(segment.make_cellpose_model(), ['img'], 40, [0, 0])
        fake_v4()
        r4 = segment.cellpose_eval(segment.make_cellpose_model(), ['img'], 40, [0, 0])
        check('3.x result is a 4-tuple', len(r3) == 4, str(len(r3)))
        check('4.x result is a 3-tuple', len(r4) == 3, str(len(r4)))
        check('result[0][0] is the mask in both',
              r3[0][0].shape == (4, 4) and r4[0][0].shape == (4, 4))

        print('\n-- the singletons rebuild per version, not per process --')
        fake_v3()
        a = segment.get_model_cyto()
        b = segment.get_model_cyto()
        check('get_model_cyto caches', a is b)
        check('get_model_cyto_nuclear is a SEPARATE instance',
              segment.get_model_cyto_nuclear() is not a)

    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        segment._cellpose_major_cached = None
        segment._model_cyto = None
        segment._model_cyto_nuc = None

    if os.environ.get('CODELAB_TEST_REAL_CELLPOSE'):
        print('\n-- against the really installed cellpose --')
        real_run()

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for f in FAIL:
            print('  FAILED:', f)
        return 1
    print('ALL GOOD')
    return 0


def real_run():
    """Segment a synthetic field through the real package, whichever is installed."""
    import time
    rng = np.random.default_rng(0)
    h = w = 512
    img = rng.integers(200, 400, size=(h, w)).astype(np.uint16)
    yy, xx = np.mgrid[0:h, 0:w]
    placed = []
    while len(placed) < 25:
        cy, cx = rng.integers(24, h - 24), rng.integers(24, w - 24)
        if any((cy - py) ** 2 + (cx - px) ** 2 < 53 ** 2 for py, px in placed):
            continue
        placed.append((cy, cx))
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < 22 ** 2] = rng.integers(3000, 5000)

    import cellpose
    print(f'  cellpose {cellpose.version} -> major {segment.cellpose_major()}')
    t = time.time()
    model = segment.make_cellpose_model(gpu=True)
    print(f'  model load {time.time() - t:.1f} s')
    t = time.time()
    result = segment.cellpose_eval(model, [img], 40, [0, 0])
    masks = np.asarray(result[0][0])
    print(f'  eval {time.time() - t:.1f} s -> {int(masks.max())} labels '
          f'(planted {len(placed)})')
    check('the real package segments the planted blobs',
          int(masks.max()) == len(placed), f'{int(masks.max())} vs {len(placed)}')


if __name__ == '__main__':
    sys.exit(main())
