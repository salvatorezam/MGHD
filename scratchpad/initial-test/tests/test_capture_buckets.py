from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from mghd_public.features_v2 import pack_cluster
from mghd_public.infer import MGHDDecoderPublic


class DummyModel:
    def __init__(self, device: torch.device):
        self.device = device
        self.captures = 0
        self.calls = 0

    def __call__(self, packed) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        x = packed.x_nodes.to(self.device)
        node_mask = packed.node_mask.to(self.device)
        if x.dim() == 3:
            batch, pad, feat = x.shape
            flat = x.view(batch * pad, feat)
            mask = node_mask.view(batch * pad)
        else:
            flat = x
            mask = node_mask
        logits = torch.zeros((flat.shape[0], 2), dtype=torch.float32, device=self.device)
        sum_feat = flat.sum(dim=1)
        logits[:, 1] = torch.tanh(sum_feat) + 1.0
        logits[:, 0] = -logits[:, 1]
        self.last_logits = logits
        self.last_mask = mask.bool()
        return logits, self.last_mask

    def allocate_static_batch(
        self,
        *,
        batch_size: int,
        nodes_pad: int,
        edges_pad: int,
        seq_pad: int,
        feat_dim: int,
        edge_feat_dim: int,
        g_dim: int,
        device: torch.device,
    ) -> SimpleNamespace:
        self.captures += 1

        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        return SimpleNamespace(
            x_nodes=zeros((batch_size, nodes_pad, feat_dim), torch.float32),
            node_mask=zeros((batch_size, nodes_pad), torch.bool),
            node_type=zeros((batch_size, nodes_pad), torch.int8),
            edge_index=zeros((2, batch_size * edges_pad), torch.long),
            edge_attr=zeros((batch_size * edges_pad, edge_feat_dim), torch.float32),
            edge_mask=zeros((batch_size * edges_pad,), torch.bool),
            seq_idx=zeros((batch_size * seq_pad,), torch.long),
            seq_mask=zeros((batch_size * seq_pad,), torch.bool),
            g_token=zeros((batch_size, g_dim), torch.float32),
            batch_size=batch_size,
            nodes_pad=nodes_pad,
        )

    def copy_into_static(self, static_ns: SimpleNamespace, host_ns: SimpleNamespace, *, non_blocking: bool = True) -> None:
        for name in ("x_nodes", "node_mask", "node_type", "edge_index", "edge_attr", "edge_mask", "seq_idx", "seq_mask", "g_token"):
            getattr(static_ns, name).copy_(getattr(host_ns, name), non_blocking=non_blocking)

    def move_packed_to_device(self, host_ns: SimpleNamespace, device: torch.device) -> SimpleNamespace:
        tensors = {}
        for name in ("x_nodes", "node_mask", "node_type", "edge_index", "edge_attr", "edge_mask", "seq_idx", "seq_mask", "g_token"):
            tensors[name] = getattr(host_ns, name).to(device, non_blocking=True)
        tensors["batch_size"] = host_ns.batch_size
        tensors["nodes_pad"] = host_ns.nodes_pad
        return SimpleNamespace(**tensors)

    def gather_from_static(self, static_output):
        return static_output

    def scatter_outputs(self, logits: torch.Tensor, cluster_infos, *, temp: float = 1.0):
        probs_all = torch.sigmoid((logits[:, 1] - logits[:, 0]) / float(temp))
        return [probs_all.index_select(0, info["data_idx"].to(logits.device)) for info in cluster_infos]


def _make_processed(bucket_spec, idx: int):
    H = sp.csr_matrix([[1, 0], [0, 1]], dtype=np.uint8)
    s = np.array([idx & 1, (idx + 1) & 1], dtype=np.uint8)
    q_l2g = np.array([0, 1], dtype=np.int64)
    c_l2g = np.array([0, 1], dtype=np.int64)
    extra = {
        "xy_qubit": np.array([[0, 0], [1, 0]], dtype=np.int32),
        "xy_check": np.array([[0, 1], [1, 1]], dtype=np.int32),
        "k": 2,
        "r": 1,
        "bbox": (0, 0, 2, 2),
        "kappa_stats": {"size": 4},
        "side": "Z",
        "d": 3,
        "p": 0.01,
    }
    return (H, s, q_l2g, c_l2g), extra


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for capture test")
def test_cuda_graph_capture_reused_within_bucket():
    device = torch.device("cuda")
    decoder = MGHDDecoderPublic.__new__(MGHDDecoderPublic)
    decoder.model = DummyModel(device)
    decoder.device = device
    decoder.model_version = "v2"
    decoder._graph_capture_enabled = True
    decoder._last_graph_used = False
    decoder._device_info = lambda: {"device": "cuda"}

    processed = []
    extras = []
    bucket_spec = [(16, 32, 16)]
    for i in range(4):
        (H, s, q, c), extra = _make_processed(bucket_spec, i)
        processed.append((H, s, q, c))
        extras.append(extra)

    probs, report = decoder._priors_from_subgraphs_v2(
        processed,
        extras,
        temp=1.0,
        bucket_spec=bucket_spec,
        microbatch=2,
        flush_ms=0.0,
        use_graphs=True,
    )

    assert all(arr.shape[0] == 2 for arr in probs)
    assert report["graph_used"] is True
    assert report["graph_used_shots"] == len(processed)
    # Capture should have been attempted exactly once for this bucket batch size
    assert decoder.model.captures == 1
