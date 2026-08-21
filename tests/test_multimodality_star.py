"""
Star-topology cross-modal generalization: N modalities, each bridging
independently into ONE shared frame, never colliding with each other.

Two layers, both real code, no mocks:

1. codelab_pipeline.io.vlinks_store's modality-keyed cross-modal
   matrix/z (write_cross_modal_matrix/read_cross_modal_matrix,
   write_cross_modal_z/read_cross_modal_z): confirms a SECOND bridging
   modality's write does not clobber the first's -- the collision this
   generalization exists to fix (both used to share one flat dataset
   name with no modality dimension at all). Also confirms the legacy
   flat key (modality=None) stays fully intact and independently
   readable, and that a modality-keyed read falls back to the legacy
   flat key when its own entry is absent (an upgraded single-bridge
   store still answers for its one real bridge).

2. codelab_pipeline.alignment.frames.FrameResolver's `bridges` dict:
   confirms three modalities (one shared, two bridging) each resolve
   correctly and independently -- to_shared/z_to_shared/bridge/
   bridge_z_between never mix modality B's transform into modality C's
   answer -- and that the legacy bridge_xy/bridge_from kwargs still
   produce byte-identical results to the new bridges= form.

Run: python tests/test_multimodality_star.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.linalg as la

from codelab_pipeline.io import vlinks_store as V
from codelab_pipeline.alignment.frames import FrameResolver, IDENTITY

SCRATCH = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'multimodality_star')
FOV = 1


def setup_store():
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    sp = os.path.join(SCRATCH, 'RNA_queue')
    os.makedirs(sp, exist_ok=True)
    V.declare_modality(sp, 'RNA')
    return sp


def test_modality_keyed_matrix_no_collision():
    sp = setup_store()
    H_dna = np.array([[1, 0, 5.0], [0, 1, -3.0], [0, 0, 1]])
    H_chr19 = np.array([[1, 0, 12.0], [0, 1, 8.5], [0, 0, 1]])
    V.write_cross_modal_matrix(sp, FOV, H_dna, modality='DNA')
    V.write_cross_modal_matrix(sp, FOV, H_chr19, modality='Chr19')
    back_dna = V.read_cross_modal_matrix(sp, FOV, modality='DNA')
    back_chr19 = V.read_cross_modal_matrix(sp, FOV, modality='Chr19')
    assert np.allclose(back_dna, H_dna), 'DNA bridge corrupted'
    assert np.allclose(back_chr19, H_chr19), 'Chr19 bridge corrupted'
    assert not np.allclose(back_dna, back_chr19), 'the two bridges collided onto one value'
    # a modality never written returns None, not someone else's matrix
    assert V.read_cross_modal_matrix(sp, FOV, modality='Nascent') is None


def test_legacy_flat_key_untouched_and_fallback_works():
    sp = setup_store()
    H_legacy = np.array([[1, 0, 1.0], [0, 1, 2.0], [0, 0, 1]])
    V.write_cross_modal_matrix(sp, FOV, H_legacy)          # modality=None, legacy path
    assert np.allclose(V.read_cross_modal_matrix(sp, FOV), H_legacy)
    # a store upgraded to per-modality reads: asking BY NAME for the one
    # modality that predates the keyed writes still finds it (fallback)
    assert np.allclose(V.read_cross_modal_matrix(sp, FOV, modality='DNA'), H_legacy)
    # once that modality gets its OWN keyed entry, the keyed one wins
    H_new = np.array([[1, 0, 99.0], [0, 1, 99.0], [0, 0, 1]])
    V.write_cross_modal_matrix(sp, FOV, H_new, modality='DNA')
    assert np.allclose(V.read_cross_modal_matrix(sp, FOV, modality='DNA'), H_new)
    assert np.allclose(V.read_cross_modal_matrix(sp, FOV), H_legacy), \
        'writing a keyed entry must not disturb the legacy flat one'


def test_z_keyed_no_collision():
    sp = setup_store()
    V.write_cross_modal_z(sp, FOV, -5.0, modality='DNA')
    V.write_cross_modal_z(sp, FOV, 12.0, modality='Chr19')
    assert V.read_cross_modal_z(sp, FOV, modality='DNA') == -5.0
    assert V.read_cross_modal_z(sp, FOV, modality='Chr19') == 12.0
    assert V.read_cross_modal_z(sp, FOV, modality='Nascent') == 0.0  # never written -> 0


def test_resolver_star_three_modalities():
    """RNA is shared; DNA and Chr19 each bridge independently into it."""
    within = {'RNA': {'Hyb_A': IDENTITY}, 'DNA': {'Hyb_B': IDENTITY}, 'Chr19': {'Hyb_C': IDENTITY}}
    H_dna = np.array([[1, 0, 5.0], [0, 1, -3.0], [0, 0, 1]])
    H_chr19 = np.array([[1, 0, 12.0], [0, 1, 8.5], [0, 0, 1]])
    resolver = FrameResolver(within, shared='RNA',
                             bridges={'DNA': (H_dna, -5.0), 'Chr19': (H_chr19, 12.0)})

    assert np.allclose(resolver.bridge('DNA', 'RNA'), H_dna)
    assert np.allclose(resolver.bridge('Chr19', 'RNA'), H_chr19)
    assert np.allclose(resolver.bridge('RNA', 'DNA'), la.inv(H_dna))
    assert np.allclose(resolver.bridge('RNA', 'Chr19'), la.inv(H_chr19))
    # DNA's own answer must never leak into Chr19's
    assert not np.allclose(resolver.bridge('DNA', 'RNA'), resolver.bridge('Chr19', 'RNA'))
    assert resolver.bridge_z_between('DNA', 'RNA') == -5.0
    assert resolver.bridge_z_between('Chr19', 'RNA') == 12.0
    assert resolver.bridge_z_between('RNA', 'DNA') == 5.0
    assert resolver.bridge_z_between('RNA', 'Chr19') == -12.0
    # non-shared-to-non-shared pairs are not directly resolvable (star, not a graph)
    assert resolver.bridge('DNA', 'Chr19') is None

    # a modality with NO bridge entry: missing is flagged, transform is identity
    missing = set()
    H = resolver.to_shared('Hyb_D', 'Nascent', cell=None, missing=missing)
    assert np.allclose(H, IDENTITY)
    assert 'cross-modal:Nascent->RNA' in missing

    # full to_shared for DNA and Chr19 must differ from each other and from identity
    H_dna_shared = resolver.to_shared('Hyb_B', 'DNA', cell=None)
    H_chr19_shared = resolver.to_shared('Hyb_C', 'Chr19', cell=None)
    assert np.allclose(H_dna_shared, H_dna)
    assert np.allclose(H_chr19_shared, H_chr19)
    assert not np.allclose(H_dna_shared, H_chr19_shared)


def test_legacy_kwargs_match_new_bridges_form():
    within = {'RNA': {}, 'DNA': {}}
    H = np.array([[1, 0, 7.0], [0, 1, -2.0], [0, 0, 1]])
    legacy = FrameResolver(within, shared='RNA', bridge_xy=H, bridge_z=3.0, bridge_from='DNA')
    modern = FrameResolver(within, shared='RNA', bridges={'DNA': (H, 3.0)})
    assert np.allclose(legacy.bridge('DNA', 'RNA'), modern.bridge('DNA', 'RNA'))
    assert legacy.bridge_z_between('DNA', 'RNA') == modern.bridge_z_between('DNA', 'RNA')
    assert np.allclose(legacy.bridge('RNA', 'DNA'), modern.bridge('RNA', 'DNA'))


def test_no_bridges_at_all_is_identity_not_missing():
    """Nothing configured system-wide yet -- identity, and the CROSS-MODAL
    leg specifically is NOT flagged as missing (distinct from 'this
    specific modality has none yet' once at least one other bridge
    exists -- see the resolver's own docstring). The same-modality leg
    is populated here so only the cross-modal question is isolated;
    that leg's own missing-ness is a separate, already-covered concern."""
    resolver = FrameResolver({'RNA': {}, 'DNA': {'Hyb_X': IDENTITY}}, shared='RNA', bridges={})
    missing = set()
    H = resolver.to_shared('Hyb_X', 'DNA', cell=None, missing=missing)
    assert np.allclose(H, IDENTITY)
    assert not any(m.startswith('cross-modal:') for m in missing), missing


def _run_all():
    tests = {k: v for k, v in globals().items() if k.startswith('test_')}
    failures = []
    for name in sorted(tests):
        try:
            tests[name]()
            print(f'  PASS  {name}')
        except Exception as e:
            failures.append(name)
            print(f'  FAIL  {name}: {e!r}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_run_all())
