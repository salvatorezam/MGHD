"""INT8/Binary quantization smoke test for MGHDv2."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

try:  # PyTorch renamed quantization to torch.ao.quantization
    from torch.ao.quantization import quantize_dynamic
except Exception:  # pragma: no cover
    from torch.quantization import quantize_dynamic

from mghd.core.core import MGHDv2

DEFAULT_MODEL = (
    Path(__file__).resolve().parents[3]
    / "data/results_validation_d3_5/20251120-175741_surface_d5_iqm_garnet_example/best.pt"
)


def _infer_arch(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infer model hyperparameters from a checkpoint state_dict."""
    d_model = 192
    d_state = 80
    node_feat_dim = 8
    edge_feat_dim = 3
    se_reduction = 4
    gnn_msg_net_size: Optional[int] = None

    node_in = state.get("node_in.weight")
    if node_in is not None:
        d_model = int(node_in.shape[0])
        node_feat_dim = int(node_in.shape[1])
    edge_in = state.get("edge_in.weight")
    if edge_in is not None:
        edge_feat_dim = int(edge_in.shape[1])
    A_log = state.get("seq_encoder.core.A_log")
    if A_log is not None and A_log.ndim == 2:
        d_state = int(A_log.shape[1])
    se_fc1 = state.get("se.fc1.weight")
    if se_fc1 is not None and se_fc1.shape[0] > 0:
        se_reduction = max(1, int(d_model // se_fc1.shape[0]))
    msg0 = state.get("gnn.core.msg_net.0.weight")
    if msg0 is not None:
        gnn_msg_net_size = int(msg0.shape[0])

    return {
        "d_model": d_model,
        "d_state": d_state,
        "node_feat_dim": node_feat_dim,
        "edge_feat_dim": edge_feat_dim,
        "se_reduction": se_reduction,
        "gnn_msg_net_size": gnn_msg_net_size,
    }


def quantize_weights_int8(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to Linear/GRU layers."""
    print(" > Applying INT8 dynamic quantization")
    return quantize_dynamic(model, {nn.Linear, nn.GRU}, dtype=torch.qint8)


def quantize_weights_binary(model: nn.Module) -> nn.Module:
    """Simulated 1-bit weight quantization (sign with mean-abs scaling)."""
    print(" > Applying simulated binary quantization")
    q_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in q_model.named_parameters():
            if "weight" in name and param.dim() > 1:
                scale = param.abs().mean()
                param.copy_(param.sign() * scale)
    return q_model


def _model_size_mb(model: nn.Module) -> float:
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (params + buffers) / (1024**2)


def _build_model(model_path: Path, n_iters: Optional[int]) -> MGHDv2:
    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    state = {k.replace("model.", ""): v for k, v in state.items()}
    g_proj = state.get("g_proj.weight")
    arch = _infer_arch(state)
    model = MGHDv2(
        d_model=arch["d_model"],
        d_state=arch["d_state"],
        n_iters=int(n_iters or 8),
        node_feat_dim=arch["node_feat_dim"],
        edge_feat_dim=arch["edge_feat_dim"],
        se_reduction=arch["se_reduction"],
        gnn_msg_net_size=arch["gnn_msg_net_size"],
    ).to(device)
    if g_proj is not None:
        model.ensure_g_proj(int(g_proj.shape[1]), device)
    missing = model.load_state_dict(state, strict=False)
    if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
        print("Warning: load_state_dict non-strict result", missing)
    model.eval()
    return model


def _allocate_dummy(model: MGHDv2, d: int, device: torch.device) -> object:
    nodes_pad = max(d * d + (d * d - 1) // 2, 32)
    edges_pad = max(12 * d * d, 128)
    seq_pad = max(2 * d * d, 64)
    feat_dim = int(model.node_in.in_features)
    edge_feat_dim = int(model.edge_in.in_features)
    g_dim = feat_dim if model.g_proj is None else int(model.g_proj.in_features)
    return model.allocate_static_batch(
        batch_size=1,
        nodes_pad=nodes_pad,
        edges_pad=edges_pad,
        seq_pad=seq_pad,
        feat_dim=feat_dim,
        edge_feat_dim=edge_feat_dim,
        g_dim=g_dim,
        device=device,
    )


@torch.no_grad()
def benchmark_decoder(
    model_path: Path, d: int, shots: int, n_iters: Optional[int], skip_forward: bool
) -> None:
    device = torch.device("cpu")
    base_model = _build_model(model_path, n_iters)
    static_pack = _allocate_dummy(base_model, d, device=device)

    variants = {
        "FP32 (original)": base_model,
        "INT8 (dynamic)": quantize_weights_int8(base_model),
        "Binary (sim)": quantize_weights_binary(base_model),
    }

    print(f"\n--- Quantization smoke test | d={d} | shots={shots} ---")
    print(f"Checkpoint: {model_path}")
    print(f"{'Variant':18} | {'Size (MB)':9} | Status")
    print("-" * 60)

    for name, model in variants.items():
        size_mb = _model_size_mb(model)
        status = "ready"
        if not skip_forward:
            try:
                logits, mask = model(static_pack)
                status = f"logits {tuple(logits.shape)} | mask {tuple(mask.shape)}"
            except Exception as exc:  # pragma: no cover - runtime guard
                status = f"error: {exc.__class__.__name__}: {exc}"
        print(f"{name:18} | {size_mb:9.2f} | {status}")

    print("-" * 60)
    print("Note: For real LER, plug the quantized models into mghd.cli.eval.")


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8/Binary quantization sanity check.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a trained MGHDv2 checkpoint (default: small d=5 run).",
    )
    parser.add_argument("--distance", type=int, default=5, help="Code distance for dummy batch.")
    parser.add_argument("--shots", type=int, default=5000, help="Only used for logging context.")
    parser.add_argument(
        "--n-iters",
        type=int,
        default=None,
        help="Override GNN iteration count if different from checkpoint default (8).",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Only report model sizes without a dummy forward pass.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")

    benchmark_decoder(args.model, d=args.distance, shots=args.shots, n_iters=args.n_iters, skip_forward=args.skip_forward)


if __name__ == "__main__":
    main()
