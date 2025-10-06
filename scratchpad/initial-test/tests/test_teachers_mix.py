import numpy as np
import pytest


def test_mix_instantiation_and_route_smoke(monkeypatch):
    # Minimal toy CSS code: small Hx, Hz; fake code_obj exposing detector->fault mapping
    H = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]], dtype=np.uint8)

    class ToyCode:
        num_detectors = 4
        detectors_per_fault = [[0, 2], [1, 3]]  # two faults flip pairs of detectors
        fault_weights = [1.0, 1.0]
        Hx = H
        Hz = H

    code = ToyCode()
    # monkeypatch mwpf/ldpc imports if packages are missing
    try:
        from teachers.mwpf_teacher import MWPFTeacher, MWPFConfig  # noqa: F401
        from teachers.lsd_teacher import LSDTeacher, LSDConfig  # noqa: F401
    except Exception:
        pytest.skip("mwpf/ldpc not installed in CI")

    from teachers.mix import TeacherMix, MixConfig

    mix = TeacherMix(code, H, H, mwpf_cfg=MWPFConfig(cluster_node_limit=10),
                     lsd_cfg=LSDConfig(max_iter=1), mix_cfg=MixConfig(p_mwpf=0.5, p_lsd=0.4, p_mwpm=0.1))
    B = 3
    dets = np.zeros((B, 4), dtype=np.uint8)
    dets[0, [0, 2]] = 1  # activate first fault
    dets[1, [1, 3]] = 1  # second fault
    dets[2, [0, 1]] = 1  # mixed
    sx = (H @ np.array([1, 0, 1, 0], dtype=np.uint8)) % 2  # shape [2]
    sz = (H @ np.array([0, 1, 0, 1], dtype=np.uint8)) % 2
    SX = np.stack([sx, sx, sx], axis=0)
    SZ = np.stack([sz, sz, sz], axis=0)
    out = mix.route_batch(dets, SX, SZ, rng=np.random.default_rng(123))
    assert "which" in out
