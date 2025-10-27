import numpy as np
import pytest
from mghd.core.core import pack_cluster

torch = pytest.importorskip("torch")


def build_sample_cluster():
    H_sub = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    xy_qubit = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    xy_check = np.array([[0, 1], [1, 2]], dtype=np.int32)
    synd_bits = np.array([1, 0], dtype=np.uint8)
    bbox = (0, 0, 2, 3)
    return H_sub, xy_qubit, xy_check, synd_bits, bbox


def test_pack_cluster_deterministic():
    H_sub, xy_qubit, xy_check, synd_bits, bbox = build_sample_cluster()
    kwargs = dict(
        H_sub=H_sub,
        xy_qubit=xy_qubit,
        xy_check=xy_check,
        synd_Z_then_X_bits=synd_bits,
        k=H_sub.shape[1],
        r=1,
        bbox_xywh=bbox,
        kappa_stats={"size": int(H_sub.shape[0] + H_sub.shape[1])},
        y_bits_local=np.zeros(H_sub.shape[1], dtype=np.uint8),
        side="Z",
        d=3,
        p=0.01,
        seed=7,
        N_max=H_sub.shape[0] + H_sub.shape[1],
        E_max=int(H_sub.sum()),
        S_max=H_sub.shape[0],
        add_jump_edges=False,
    )
    crop_a = pack_cluster(**kwargs)
    crop_b = pack_cluster(**kwargs)

    assert torch.equal(crop_a.x_nodes, crop_b.x_nodes)
    assert torch.equal(crop_a.edge_index, crop_b.edge_index)
    assert torch.equal(crop_a.y_bits, crop_b.y_bits)


def test_pack_cluster_masks_match_sizes():
    H_sub, xy_qubit, xy_check, synd_bits, bbox = build_sample_cluster()
    crop = pack_cluster(
        H_sub=H_sub,
        xy_qubit=xy_qubit,
        xy_check=xy_check,
        synd_Z_then_X_bits=synd_bits,
        k=H_sub.shape[1],
        r=1,
        bbox_xywh=bbox,
        kappa_stats={"size": int(H_sub.shape[0] + H_sub.shape[1])},
        y_bits_local=np.zeros(H_sub.shape[1], dtype=np.uint8),
        side="X",
        d=3,
        p=0.02,
        seed=11,
        N_max=H_sub.shape[0] + H_sub.shape[1],
        E_max=int(H_sub.sum()),
        S_max=H_sub.shape[0],
        add_jump_edges=False,
    )
    node_count = H_sub.shape[0] + H_sub.shape[1]
    edge_count = int(H_sub.sum())

    assert int(crop.node_mask.sum().item()) == node_count
    assert int(crop.edge_mask.sum().item()) == edge_count
    assert crop.g_token.ndim == 1
