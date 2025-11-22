"""FPGA latency estimate helper using hls4ml."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from mghd.benchmarks.quantize_test import DEFAULT_MODEL, _infer_arch
from mghd.core.core import MGHDv2

DEFAULT_PART = "xcvu9p-flgb2104-2-i"


def _load_model(model_path: Path, n_iters: Optional[int]) -> MGHDv2:
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
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _print_estimate() -> None:
    print("Rule-of-thumb latency estimate (no synthesis): ~100-150 ns at ~300 MHz for ~8 layers.")


def generate_hls_project(
    model_path: Path,
    output_dir: Path,
    *,
    part: str = DEFAULT_PART,
    precision: str = "ap_fixed<16,6>",
    reuse_factor: int = 1,
    n_iters: Optional[int] = None,
) -> None:
    try:
        import hls4ml
    except Exception as exc:  # pragma: no cover - dependency guard
        print("hls4ml is not installed. Install with `pip install hls4ml` before running.")
        print(f"Reason: {exc}")
        return

    model = _load_model(model_path, n_iters)

    try:
        config = hls4ml.utils.config_from_pytorch_model(
            model,
            input_shape=(1, model.node_in.in_features),
            granularity="name",
            default_precision=precision,
            default_reuse_factor=reuse_factor,
        )
    except Exception as exc:  # pragma: no cover - config errors
        print("config_from_pytorch_model failed (custom MGHD modules likely unsupported).")
        print(f"Error: {exc}")
        _print_estimate()
        return

    print("--- Generating HLS latency project ---")
    print(f"Checkpoint: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"FPGA part:  {part}")
    print(f"Precision:  {precision} | Reuse factor: {reuse_factor}")

    try:
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model,
            input_shape=(1, model.node_in.in_features),
            hls_config=config,
            output_dir=str(output_dir),
            part=part,
            io_type="io_parallel",
        )
    except Exception as exc:  # pragma: no cover - conversion errors
        print("Conversion failed. hls4ml may need custom layer mappings for Mamba/GNN blocks.")
        print(f"Error: {exc}")
        _print_estimate()
        return

    # hls_model.write() creates project files without running Vivado.
    hls_model.write()
    print("[SUCCESS] HLS project generated. Run Vivado HLS C Synthesis to obtain ns latency.")
    _print_estimate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an hls4ml project for MGHDv2.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a trained MGHDv2 checkpoint (default: small d=5 run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mghd_hls_project"),
        help="Directory to place the generated HLS project.",
    )
    parser.add_argument("--part", type=str, default=DEFAULT_PART, help="FPGA part number.")
    parser.add_argument(
        "--precision",
        type=str,
        default="ap_fixed<16,6>",
        help="Default numeric precision for hls4ml.",
    )
    parser.add_argument(
        "--reuse-factor",
        type=int,
        default=1,
        help="Reuse factor (1 = fully parallel, lower latency).",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=None,
        help="Override GNN iteration count if different from checkpoint default (8).",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")

    generate_hls_project(
        args.model,
        args.output_dir,
        part=args.part,
        precision=args.precision,
        reuse_factor=args.reuse_factor,
        n_iters=args.n_iters,
    )


if __name__ == "__main__":
    main()
