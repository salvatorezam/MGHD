import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd.core.core import MGHDv2, pack_cluster


def test_mghdv2_forward_handles_feature_resize():
    # Build a tiny crop with erasure channel to trigger node_in resize
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    xy_q = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    xy_c = np.array([[0, 1], [1, 2]], dtype=np.int32)
    s = np.array([1, 0], dtype=np.uint8)
    nQ = H.shape[1]
    erase_local = np.zeros(nQ, dtype=np.uint8)
    erase_local[1] = 1
    pack = pack_cluster(
        H_sub=H,
        xy_qubit=xy_q,
        xy_check=xy_c,
        synd_Z_then_X_bits=s,
        k=nQ,
        r=1,
        bbox_xywh=(0, 0, 2, 3),
        kappa_stats={"size": int(H.shape[0] + H.shape[1])},
        y_bits_local=np.zeros(nQ, dtype=np.uint8),
        side="Z",
        d=3,
        p=0.01,
        seed=0,
        N_max=nQ + H.shape[0],
        E_max=int(H.sum()),
        S_max=H.shape[0],
        add_jump_edges=False,
        erase_local=erase_local,
    )
    # Also trigger edge_in resize by adding a dummy fourth edge feature
    pack.edge_attr = torch.cat([pack.edge_attr, torch.zeros((pack.edge_attr.shape[0], 1))], dim=1)

    model = MGHDv2()
    logits, node_mask = model(pack)
    assert logits.shape[0] == node_mask.shape[0]
    assert logits.shape[1] == 2
