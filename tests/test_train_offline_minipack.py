import os
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd.core.core import pack_cluster
from mghd.cli import train as train_mod


def _to_numpy(t):
    import torch as _torch
    return t.detach().cpu().numpy() if _torch.is_tensor(t) else np.asarray(t)


def test_train_inprocess_offline_minipack(tmp_path, monkeypatch):
    # Build a tiny packed crop and save as a shard (.npz) expected by the loader
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    xy_q = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    xy_c = np.array([[0, 1], [1, 2]], dtype=np.int32)
    s = np.array([1, 0], dtype=np.uint8)
    pack = pack_cluster(
        H_sub=H,
        xy_qubit=xy_q,
        xy_check=xy_c,
        synd_Z_then_X_bits=s,
        k=H.shape[1],
        r=1,
        bbox_xywh=(0, 0, 2, 3),
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
    item = {
        "x_nodes": _to_numpy(pack.x_nodes),
        "node_mask": _to_numpy(pack.node_mask),
        "node_type": _to_numpy(pack.node_type),
        "edge_index": _to_numpy(pack.edge_index),
        "edge_attr": _to_numpy(pack.edge_attr),
        "edge_mask": _to_numpy(pack.edge_mask),
        "seq_idx": _to_numpy(pack.seq_idx),
        "seq_mask": _to_numpy(pack.seq_mask),
        "g_token": _to_numpy(pack.g_token),
        "y_bits": _to_numpy(pack.y_bits),
        "meta": {
            "k": int(pack.meta.k),
            "r": int(pack.meta.r),
            "bbox_xywh": tuple(pack.meta.bbox_xywh),
            "side": str(pack.meta.side),
            "d": int(pack.meta.d),
            "p": float(pack.meta.p),
            "kappa": int(pack.meta.kappa),
            "seed": int(pack.meta.seed),
        },
        "H_sub": H,
        "idx_data_local": np.arange(H.shape[1], dtype=np.int64),
        "idx_check_local": np.arange(H.shape[0], dtype=np.int64),
    }

    data_root = tmp_path / "crops"
    data_root.mkdir()
    shard = data_root / "shard.npz"
    # Save as scalar object array to match loader logic (shape==())
    np.savez(shard, packed=np.array(item, dtype=object))

    save_root = tmp_path / "runs"
    ns = SimpleNamespace()
    ns.online = False
    ns.data_root = str(data_root)
    ns.epochs = 1
    ns.batch = 1
    ns.lr = 1e-4
    ns.wd = 1e-6
    ns.profile = "S"
    ns.ema = 0.999
    ns.parity_lambda = 0.0
    ns.projection_aware = 0
    ns.seed = 7
    ns.label_smoothing = 0.0
    ns.noise_injection = 0.0
    ns.grad_clip = 1.0
    ns.save = None
    ns.save_root = str(save_root)
    ns.save_auto = True

    out_path = train_mod.train_inprocess(ns)
    assert out_path.endswith("best.pt")
    # Ensure logs exist
    found = list(save_root.rglob("train_log.json"))
    assert found and json.loads(found[0].read_text())

