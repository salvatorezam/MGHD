import numpy as np
import pytest

pytest.importorskip("torch")

from mghd_public.features_v2 import pack_cluster


def test_masks_align_with_node_types():
    H_sub = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    xy_qubit = np.array([[0, 0], [1, 0]], dtype=np.int32)
    xy_check = np.array([[0, 1], [1, 2]], dtype=np.int32)
    synd_bits = np.array([1, 1], dtype=np.uint8)
    crop = pack_cluster(
        H_sub=H_sub,
        xy_qubit=xy_qubit,
        xy_check=xy_check,
        synd_Z_then_X_bits=synd_bits,
        k=H_sub.shape[1],
        r=1,
        bbox_xywh=(0, 0, 2, 3),
        kappa_stats={"size": int(H_sub.shape[0] + H_sub.shape[1])},
        y_bits_local=np.zeros(H_sub.shape[1], dtype=np.uint8),
        side="Z",
        d=5,
        p=0.03,
        seed=3,
        N_max=H_sub.shape[0] + H_sub.shape[1],
        E_max=int(H_sub.sum()),
        S_max=H_sub.shape[0],
        add_jump_edges=False,
    )
    node_mask = crop.node_mask
    node_type = crop.node_type

    data_mask = node_mask & (node_type == 0)
    check_mask = node_mask & (node_type == 1)

    assert int(data_mask.sum().item()) == H_sub.shape[1]
    assert int(check_mask.sum().item()) == H_sub.shape[0]

    # Labels exist only on data nodes
    labeled = crop.y_bits[data_mask]
    assert labeled.shape[0] == H_sub.shape[1]
