import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd.core.core import MGHDv2, pack_cluster


def _toy_cluster():
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    xy_q = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    xy_c = np.array([[0, 1], [1, 2]], dtype=np.int32)
    s = np.array([1, 0], dtype=np.uint8)
    return H, xy_q, xy_c, s, (0, 0, 2, 3)


def test_mghdv2_forward_on_packed_crop():
    H, xy_q, xy_c, s, bbox = _toy_cluster()
    pack = pack_cluster(
        H_sub=H,
        xy_qubit=xy_q,
        xy_check=xy_c,
        synd_Z_then_X_bits=s,
        k=H.shape[1],
        r=1,
        bbox_xywh=bbox,
        kappa_stats={"size": int(H.shape[0] + H.shape[1])},
        y_bits_local=np.zeros(H.shape[1], dtype=np.uint8),
        side="Z",
        d=3,
        p=0.01,
        seed=0,
        N_max=H.shape[0] + H.shape[1],
        E_max=int(H.sum()),
        S_max=H.shape[0],
        add_jump_edges=False,
    )
    model = MGHDv2()
    logits, node_mask = model(pack)
    assert logits.shape[0] == node_mask.shape[0]
    assert logits.shape[1] == 2
