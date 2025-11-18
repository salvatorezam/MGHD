import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd.core.core import pack_cluster


def test_pack_cluster_includes_erasure_feature_when_provided():
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    xy_q = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    xy_c = np.array([[0, 1], [1, 2]], dtype=np.int32)
    s = np.array([1, 0], dtype=np.uint8)
    nQ = H.shape[1]
    erase_local = np.zeros(nQ, dtype=np.uint8)
    erase_local[0] = 1
    crop = pack_cluster(
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
    # Erasure channel should add +1 feature dim (base is 8 → 9)
    assert crop.x_nodes.shape[1] >= 9
