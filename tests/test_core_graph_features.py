import numpy as np
import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from mghd.core.core import (
    GraphDecoderCore,
    GraphDecoderAdapter,
    MGHDDecoderPublic,
    MGHDv2,
    CropMeta,
    pack_cluster,
    rotated_surface_pcm,
)


def _basic_pack():
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    xy_q = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    xy_c = np.array([[0, 1], [1, 2]], dtype=np.int32)
    return pack_cluster(
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


def _make_decoder(tmp_path):
    model = MGHDv2()
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    dec = MGHDDecoderPublic(str(ckpt), device="cpu")
    dec.bind_code(rotated_surface_pcm(3, "X"), rotated_surface_pcm(3, "Z"))
    return dec


class _CaptureNet(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return torch.zeros(x.shape[0], self.out_dim, device=x.device)


def test_graph_decoder_core_uses_edge_features():
    hidden = 4
    edge_dim = 2
    core = GraphDecoderCore(
        n_iters=1,
        n_node_features=hidden,
        n_node_inputs=hidden,
        n_edge_features=edge_dim,
        msg_net_size=8,
    )
    capture = _CaptureNet(edge_dim)
    core.msg_net = capture
    node_inputs = torch.randn(3, hidden)
    src_ids = torch.tensor([0, 1])
    dst_ids = torch.tensor([1, 2])
    edge_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    _ = core(node_inputs, src_ids, dst_ids, edge_attr=edge_attr)
    assert capture.last_input.shape[1] == 2 * hidden + edge_dim
    assert torch.allclose(capture.last_input[:, -edge_dim:], edge_attr)


def test_graph_decoder_core_edge_defaults_zero():
    hidden = 3
    edge_dim = 1
    core = GraphDecoderCore(
        n_iters=1,
        n_node_features=hidden,
        n_node_inputs=hidden,
        n_edge_features=edge_dim,
        msg_net_size=6,
    )
    capture = _CaptureNet(edge_dim)
    core.msg_net = capture
    node_inputs = torch.randn(3, hidden)
    src_ids = torch.tensor([0])
    dst_ids = torch.tensor([1])
    _ = core(node_inputs, src_ids, dst_ids)
    zeros = torch.zeros(src_ids.shape[0], edge_dim)
    assert torch.allclose(capture.last_input[:, -edge_dim:], zeros)


def test_decoder_fast_infer_matches_forward(tmp_path):
    decoder = _make_decoder(tmp_path)
    pack_ref = _basic_pack()
    pack_fast = _basic_pack()
    ref_logits, ref_mask = decoder.model(decoder._move_packed_crop(pack_ref, decoder.device))
    logits_fast, mask_fast = decoder.fast_infer(pack_fast)
    assert torch.allclose(logits_fast, ref_logits)
    assert torch.equal(mask_fast, ref_mask)


def test_decoder_copy_into_static_pack_error(tmp_path):
    decoder = _make_decoder(tmp_path)
    pack_a = _basic_pack()
    pack_b = _basic_pack()
    pack_b.x_nodes = pack_b.x_nodes[:-1]  # shape mismatch
    with pytest.raises(ValueError):
        decoder._copy_into_static_pack(pack_a, pack_b)


def test_decoder_warmup_capture_without_cuda(tmp_path, monkeypatch):
    decoder = _make_decoder(tmp_path)
    pack = _basic_pack()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert decoder.warmup_and_capture(pack) is None


def test_pack_cluster_bucket_and_jump_edges():
    H = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    xy_q = np.array([[0, 0], [1, 0]], dtype=np.int32)
    pack = pack_cluster(
        H_sub=H,
        xy_qubit=xy_q,
        xy_check=np.array([[0, 1], [1, 2]], dtype=np.int32),
        synd_Z_then_X_bits=np.array([1, 0], dtype=np.uint8),
        k=H.shape[1],
        r=1,
        bbox_xywh=(0, 0, 2, 2),
        kappa_stats={"size": float(H.shape[0] + H.shape[1])},
        y_bits_local=np.zeros(H.shape[1], dtype=np.uint8),
        side="Z",
        d=3,
        p=0.02,
        seed=0,
        N_max=H.shape[1] + H.shape[0],
        E_max=int(H.sum()),
        S_max=H.shape[0],
        bucket_spec=[(8, 16, 4)],
        add_jump_edges=True,
        jump_k=2,
    )
    assert pack.meta.bucket_id == 0
    # jump edges add extra attributes beyond base H edges
    assert pack.edge_attr.shape[0] > H.sum()


def test_graph_decoder_adapter_respects_masks():
    adapter = GraphDecoderAdapter(hidden_dim=4, edge_feat_dim=4, n_iters=2)
    x_nodes = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, 4)
    node_mask = torch.ones(4, dtype=torch.bool)
    edge_mask = torch.tensor([True, False, True])
    adapter.set_iteration_override(1)
    out = adapter(x_nodes, edge_index, edge_attr, node_mask, edge_mask)
    assert out.shape == (4, 2)


def test_mghdv2_static_helpers():
    model = MGHDv2()
    static = model.allocate_static_batch(
        batch_size=1,
        nodes_pad=3,
        edges_pad=2,
        seq_pad=2,
        feat_dim=8,
        edge_feat_dim=3,
        g_dim=4,
        device=torch.device("cpu"),
    )
    host = SimpleNamespace(
        x_nodes=torch.randn_like(static.x_nodes),
        node_mask=torch.ones_like(static.node_mask),
        node_type=torch.zeros_like(static.node_type),
        edge_index=torch.zeros_like(static.edge_index),
        edge_attr=torch.randn_like(static.edge_attr),
        edge_mask=torch.ones_like(static.edge_mask),
        seq_idx=torch.zeros_like(static.seq_idx),
        seq_mask=torch.ones_like(static.seq_mask),
        g_token=torch.randn_like(static.g_token),
        y_bits=torch.zeros((1, 3), dtype=torch.int8),
        meta=CropMeta(
            k=3,
            r=1,
            bbox_xywh=(0, 0, 1, 1),
            side="Z",
            d=3,
            p=0.01,
            kappa=4,
            seed=0,
            bucket_id=-1,
        ),
        H_sub=np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8),
        idx_data_local=np.arange(3),
        idx_check_local=np.arange(2),
        bucket_id=-1,
        batch_size=1,
        nodes_pad=3,
    )
    model.copy_into_static(static, host)
    moved = model.move_packed_to_device(host, torch.device("cpu"))
    assert moved.x_nodes.device.type == "cpu"
    logits = torch.randn(2, 3, 2)
    mask = torch.ones(3, dtype=torch.bool)
    gathered = model.gather_from_static((logits, mask))
    assert gathered[0].shape == (2, 3, 2)
    data_idx = torch.tensor([0, 2], dtype=torch.long)
    probs = model.scatter_outputs(logits[-1], [{"data_idx": data_idx}])
    assert probs[0].shape[0] == 2
    model.set_message_iters(1)
    model.ensure_g_proj(6, torch.device("cpu"))
    model.ensure_node_in(model.node_in.in_features + 1, torch.device("cpu"))
    model.ensure_edge_in(model.edge_in.in_features + 1, torch.device("cpu"))
