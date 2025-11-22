"""Save a binarized (LogicNets-style) copy of an MGHD checkpoint."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict

import torch


def _binarize_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return a copy of the state_dict with +/- mean-abs weights."""
    binary = {}
    for k, v in state.items():
        if "weight" in k and hasattr(v, "dim") and v.dim() > 1:
            scale = v.abs().mean()
            binary[k] = v.sign() * scale
        else:
            binary[k] = v
    return binary


def save_binary_version(input_path: Path) -> Path:
    """Load a checkpoint, binarize weights, and write a new file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")

    print(f"Loading: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt

    print("Binarizing weights (SignSGD style)...")
    binary_state = _binarize_state(state)

    output_path = input_path.with_name(input_path.stem + "_BINARY.pt")
    new_ckpt = copy.deepcopy(ckpt)
    if isinstance(new_ckpt, dict) and "model" in new_ckpt:
        new_ckpt["model"] = binary_state
    elif isinstance(new_ckpt, dict) and "state_dict" in new_ckpt:
        new_ckpt["state_dict"] = binary_state
    else:
        new_ckpt = binary_state

    torch.save(new_ckpt, output_path)
    print(f"Saved binary model to: {output_path}")
    return output_path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m mghd.benchmarks.save_binary_ckpt <path_to_checkpoint>")
        sys.exit(1)
    path = Path(sys.argv[1])
    save_binary_version(path)


if __name__ == "__main__":
    main()

