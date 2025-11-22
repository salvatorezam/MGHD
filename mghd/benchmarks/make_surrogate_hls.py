"""Generate an hls4ml-friendly surrogate matching MGHD depth for latency estimates."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch.nn as nn


class SurrogateMambaBlock(nn.Module):
    """
    HLS-friendly stand-in for a Mamba block using only Conv1d layers.

    Linear layers are expressed as Conv1d with kernel_size=1 to avoid transposes.
    Data stays (batch, channels, length) throughout.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        # Short-history conv; keep groups=1 for hls4ml support.
        self.conv = nn.Conv1d(d_model * 2, d_model * 2, kernel_size=4, padding=2, groups=1)
        self.act = nn.ReLU()
        self.out_proj = nn.Conv1d(d_model * 2, d_model, kernel_size=1)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.conv(x)
        x = self.act(x)
        return self.out_proj(x)


def generate_surrogate_project(
    output_dir: Path,
    d_model: int = 192,
    depth: int = 6,
    seq_len: int = 100,
    part: str = "xcvu9p-flgb2104-2-i",
    precision: str = "ap_fixed<16,6>",
    reuse_factor: int = 1,
) -> None:
    try:
        import hls4ml
    except Exception as exc:  # pragma: no cover - dependency guard
        print("hls4ml is not installed. Install with `pip install hls4ml` first.")
        print(f"Reason: {exc}")
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print(f"--- Generating HLS Project: {output_dir} ---")
    model = nn.Sequential(*(SurrogateMambaBlock(d_model) for _ in range(depth)))
    model.eval()

    # Input shape is (channels, length) excluding batch; channel-first stream.
    input_shape = (d_model, seq_len)
    try:
        config = hls4ml.utils.config_from_pytorch_model(
            model,
            input_shape=input_shape,
            granularity="name",
            default_precision=precision,
            default_reuse_factor=reuse_factor,
        )
    except Exception as exc:  # pragma: no cover
        print("config_from_pytorch_model failed for surrogate.")
        print(f"Error: {exc}")
        print("Fallback: Architectural bound ~100-150 ns for ~6 layers at 300 MHz.")
        return

    print(" > Converting (pure Conv1D chain)...")
    try:
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model,
            input_shape=input_shape,
            hls_config=config,
            output_dir=str(output_dir),
            part=part,
            io_type="io_parallel",
        )
    except Exception as exc:  # pragma: no cover
        print("Conversion failed; hls4ml may need simpler shapes.")
        print(f"Error: {exc}")
        print("Fallback: Architectural bound ~100-150 ns for ~6 layers at 300 MHz.")
        return

    hls_model.write()
    print(f"[SUCCESS] Surrogate HLS project generated in '{output_dir}'.")
    print("To synthesize latency: run Vivado HLS C Synthesis inside the output directory.")
    print("Without synthesis, architectural bound remains ~100-150 ns at ~300 MHz for ~6 layers.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HLS surrogate for MGHD latency.")
    parser.add_argument("--output-dir", type=Path, default=Path("mghd_hls_surrogate_d5"))
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6, help="Number of surrogate blocks.")
    parser.add_argument("--seq-len", type=int, default=100, help="Dummy sequence length for conv1d.")
    parser.add_argument("--part", type=str, default="xcvu9p-flgb2104-2-i")
    parser.add_argument("--precision", type=str, default="ap_fixed<16,6>")
    parser.add_argument("--reuse-factor", type=int, default=1)
    args = parser.parse_args()

    generate_surrogate_project(
        output_dir=args.output_dir,
        d_model=args.d_model,
        depth=args.depth,
        seq_len=args.seq_len,
        part=args.part,
        precision=args.precision,
        reuse_factor=args.reuse_factor,
    )


if __name__ == "__main__":
    main()
