import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd.core.core import MGHDv2, MGHDDecoderPublic, rotated_surface_pcm


def test_decoder_public_priors_from_subgraphs_batched(tmp_path):
    # Save a tiny MGHDv2 checkpoint
    model = MGHDv2()
    ckpt = tmp_path / "mghd_v2_ckpt.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    # Code matrices for binding (d=3 rotated surface)
    Hx = rotated_surface_pcm(3, "X")
    Hz = rotated_surface_pcm(3, "Z")

    dec = MGHDDecoderPublic(str(ckpt), device="cpu")
    dec.bind_code(Hx, Hz)

    # Build a minimal subgraph entry: 2 checks by 3 data qubits
    H_sub = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    s_sub = np.array([1, 0], dtype=np.uint8)
    q_l2g = np.array([0, 1, 2], dtype=np.int64)
    c_l2g = np.array([0, 1], dtype=np.int64)
    meta = {
        "xy_qubit": np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32),
        "xy_check": np.array([[0, 1], [1, 2]], dtype=np.int32),
        "k": 3,
        "r": 1,
        "bbox": (0, 0, 2, 3),
        "kappa_stats": {"size": float(H_sub.shape[0] + H_sub.shape[1])},
        "side": "Z",
        "d": 3,
        "p": 0.01,
    }
    probs, report = dec.priors_from_subgraphs_batched([(H_sub, s_sub, q_l2g, c_l2g, meta)])
    assert isinstance(probs, list) and len(probs) == 1
    assert probs[0].shape[0] == H_sub.shape[1]
    # Report should contain basic keys
    assert "device" in report and "batch_sizes" in report

