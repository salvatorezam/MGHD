import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd.cli.train import _validate_packed_contract
from mghd.core.core import pack_cluster


def _small_pack(node_dim: int = 8, edge_dim: int = 3):
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    xy_q = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    xy_c = np.array([[0, 1], [1, 2]], dtype=np.int32)
    pack = pack_cluster(
        H_sub=H,
        xy_qubit=xy_q,
        xy_check=xy_c,
        synd_Z_then_X_bits=np.array([1, 0], dtype=np.uint8),
        k=H.shape[1],
        r=1,
        bbox_xywh=(0, 0, 2, 3),
        kappa_stats={"size": float(H.shape[0] + H.shape[1])},
        y_bits_local=np.zeros(H.shape[1], dtype=np.uint8),
        side="Z",
        d=3,
        p=0.01,
        seed=0,
        N_max=H.shape[1] + H.shape[0],
        E_max=int(H.sum()),
        S_max=H.shape[0],
        add_jump_edges=False,
    )
    if node_dim != pack.x_nodes.shape[-1]:
        pack.x_nodes = pack.x_nodes[:, :node_dim]
    if edge_dim != pack.edge_attr.shape[-1]:
        pack.edge_attr = pack.edge_attr[:, :edge_dim]
    return pack


def test_validate_packed_contract_accepts_matching_dims():
    pack = _small_pack()
    _validate_packed_contract(pack, node_feat_dim=8, edge_feat_dim=3)


def test_validate_packed_contract_rejects_node_mismatch():
    pack = _small_pack()
    pack.x_nodes = pack.x_nodes[:, :-1]
    with pytest.raises(ValueError):
        _validate_packed_contract(pack, node_feat_dim=8, edge_feat_dim=3)


def test_validate_packed_contract_rejects_missing_edge_attr():
    pack = _small_pack()
    pack.edge_attr = None
    with pytest.raises(ValueError):
        _validate_packed_contract(pack, node_feat_dim=8, edge_feat_dim=3)


def test_validate_packed_contract_rejects_edge_mismatch():
    pack = _small_pack()
    pack.edge_attr = pack.edge_attr[:, :2]
    with pytest.raises(ValueError):
        _validate_packed_contract(pack, node_feat_dim=8, edge_feat_dim=3)
