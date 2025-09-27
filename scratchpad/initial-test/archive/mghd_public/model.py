"""Factory helpers for public MGHD inference."""

from __future__ import annotations

import inspect
import json
import os
import warnings
from typing import Tuple

import torch

from .config import MGHDConfig

# Import the training model definition (binary head enforced for rotated d=3).
from poc_my_models import MGHD as RawMGHD  # type: ignore


def _has_disable_mamba_param(model_class) -> bool:
    """Check if the model constructor supports disable_mamba parameter using inspect.signature."""
    try:
        sig = inspect.signature(model_class.__init__)
        return 'disable_mamba' in sig.parameters
    except Exception:
        return False


def build_model(cfg: MGHDConfig, disable_mamba: bool = False) -> torch.nn.Module:
    """Instantiate the MGHD module using saved configuration."""
    
    # Check if the model supports disable_mamba parameter
    if disable_mamba and _has_disable_mamba_param(RawMGHD):
        model = RawMGHD(gnn_params=cfg.gnn, mamba_params=cfg.mamba, disable_mamba=True)
    elif disable_mamba:
        warnings.warn("disable_mamba=True requested but model does not support it, proceeding with normal initialization")
        model = RawMGHD(gnn_params=cfg.gnn, mamba_params=cfg.mamba)
    else:
        model = RawMGHD(gnn_params=cfg.gnn, mamba_params=cfg.mamba)
    
    model.eval()
    return model


def save_run(outdir: str, cfg: MGHDConfig, ckpt_path: str) -> None:
    """Persist config metadata next to an existing checkpoint."""

    os.makedirs(outdir, exist_ok=True)
    cfg_path = os.path.join(outdir, "mghd_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Record the checkpoint location for convenience (without copying potentially large files).
    with open(os.path.join(outdir, "CKPT_PATH.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.abspath(ckpt_path) + "\n")


def load_run(ckpt_path: str, cfg: MGHDConfig, *, map_location: str | torch.device = "cpu") -> Tuple[torch.nn.Module, dict]:
    """Load a model+state_dict using the public config."""

    model = build_model(cfg).to(map_location)
    try:
        model.set_rotated_layout()
    except AttributeError:
        pass
    state = torch.load(ckpt_path, map_location=map_location)
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, {"missing": list(missing), "unexpected": list(unexpected)}


def load_mghd_checkpoint(ckpt_path: str, cfg: MGHDConfig, *, device: str | torch.device = "cpu", disable_mamba: bool = False) -> Tuple[torch.nn.Module, dict]:
    """Safe checkpoint loading with error handling and optional mamba disabling."""
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    try:
        # Build model with optional mamba disabling
        model = build_model(cfg, disable_mamba=disable_mamba)
        model = model.to(device)
        
        # Set rotated layout if available
        try:
            model.set_rotated_layout()
        except AttributeError:
            warnings.warn("Model does not have set_rotated_layout() method")
        
        # Load checkpoint
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = state.get("state_dict", state)
        
        # Load state dict with error reporting
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        # Report loading issues
        if missing:
            warnings.warn(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys in checkpoint: {unexpected}")
        
        model.eval()
        return model, {
            "missing": list(missing), 
            "unexpected": list(unexpected),
            "disable_mamba": disable_mamba,
            "checkpoint_path": ckpt_path
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}") from e
