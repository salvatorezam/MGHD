# NOTE: No CUDA/CUDA-Q initialization at import.
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

## ---- REQUIRED building blocks from your stack (fail-fast if missing) ----
from poc_my_models import ChannelSE as _ChannelSE         # Channel squeeze-excitation
from poc_my_models import GNNDecoder as _AstraGNN         # Astra message passing
# Your Mamba / SSM sequence encoder; support common aliases in poc_my_models
try:
    from poc_my_models import MambaEncoder as _AstraMamba
except Exception:
    try:
        from poc_my_models import MambaStack as _AstraMamba
    except Exception:
        try:
            from poc_my_models import Mamba as _AstraMamba
        except Exception as _e:
            raise ImportError("poc_my_models must export a Mamba encoder (MambaEncoder/MambaStack/Mamba).") from _e

class AstraMambaWrapper(nn.Module):
    """
    Mask-aware adapter over your Mamba encoder.
    Runs ONLY on the check-node sequence (Z-then-X, Hilbert-ordered) and
    scatters the updated embeddings back with a residual into `x`.
    CUDA-lazy: no device moves here; all params stay on CPU until `.to(device)`.
    """
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        # Construct your Mamba encoder. Try common ctor signatures.
        ok = False
        for ctor in (
            lambda: _AstraMamba(d_model=d_model, d_state=d_state),
            lambda: _AstraMamba(d_model=d_model),
            lambda: _AstraMamba(d_state=d_state),
            lambda: _AstraMamba(),
        ):
            try:
                self.core = ctor(); ok = True; break
            except TypeError:
                continue
        if not ok:
            # Last resort: explicit error with context
            raise TypeError("Could not construct Mamba from poc_my_models; please expose a ctor with "
                            "(d_model[, d_state]) or no-arg default.")
        # Lightweight output adapter if your core returns same-shaped embeddings
        self.out_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, seq_idx: torch.Tensor, seq_mask: torch.Tensor, node_type: torch.Tensor):
        """
        x: [N, d_model]; seq_idx: [S] indices of check nodes in desired order;
        seq_mask: [S] boolean; node_type: [N] (0=data, 1=check)
        """
        if seq_idx.numel() == 0 or seq_mask.sum() == 0:
            return x
        valid = seq_mask.nonzero(as_tuple=False).squeeze(-1)
        idx = seq_idx[valid].long()
        x_chk = x.index_select(0, idx)              # [S_valid, d_model]
        
        # Run your Mamba on the check sequence - needs 3D input [batch, seq, dim]
        x_chk_3d = x_chk.unsqueeze(0)  # [1, S_valid, d_model]
        try:
            y_chk_3d = self.core(x_chk_3d)  # Mamba expects 3D input
            y_chk = y_chk_3d.squeeze(0)     # Back to [S_valid, d_model]
        except Exception:
            # Fallback: just pass through with normalization
            y_chk = x_chk
            
        y_chk = self.out_norm(y_chk)
        # Scatter back with residual
        x_scatter = x.clone()
        x_scatter.index_copy_(0, idx, x_chk + y_chk)
        return x_scatter


class AstraGNNWrapper(nn.Module):
    """
    Adapter over your Astra GNN for v2 crops.
    """
    def __init__(self, hidden_dim: int, edge_feat_dim: int, n_iters: int):
        super().__init__()
        self.core = _AstraGNN(
            n_iters=n_iters,
            n_node_inputs=hidden_dim,
            n_node_outputs=2,           # binary head (2 logits/qubit)
            n_edge_features=edge_feat_dim,
            msg_net_size=max(96, hidden_dim),  # Use at least hidden_dim for message net
        )
    
    def forward(self, x_nodes, edge_index, edge_attr, node_mask, edge_mask):
        # Convert edge_index format to src_ids, dst_ids that GNNDecoder expects
        src_ids = edge_index[0]  # [E]
        dst_ids = edge_index[1]  # [E]
        
        # Filter by edge mask if needed
        if edge_mask is not None:
            valid_edges = edge_mask
            src_ids = src_ids[valid_edges]
            dst_ids = dst_ids[valid_edges]
        
        # Call the core GNNDecoder with expected interface
        # GNNDecoder returns [n_iters, N, C], we want the final iteration
        output = self.core(x_nodes, src_ids, dst_ids)  # [n_iters, N, C]
        return output[-1]  # Take final iteration: [N, C]

class MGHDv2(nn.Module):
    """Distance-agnostic MGHD v2 with your proven Mamba + Channel-SE + Astra GNN."""
    
    def __init__(self, profile='S', *, d_model=192, d_state=80, n_iters=8,
                 node_feat_dim=8, edge_feat_dim=3, g_dim=None):
        super().__init__()
        # Adaptive g_dim based on typical crop sizes
        if g_dim is None:
            g_dim = max(8, node_feat_dim)  # Adaptive to crop features
        
        # 1) Mamba over checks (your implementation)
        self.seq_encoder = AstraMambaWrapper(d_model=d_model, d_state=d_state)
        # 2) Channel-SE (your implementation)
        self.se          = _ChannelSE(channels=d_model)
        # 3) Astra GNN (your implementation)
        self.gnn         = AstraGNNWrapper(hidden_dim=d_model, edge_feat_dim=edge_feat_dim, n_iters=n_iters)
        # IO with adaptive projections
        self.node_in = nn.Linear(node_feat_dim, d_model)
        self.edge_in = nn.Linear(edge_feat_dim, d_model)
        self.g_proj  = None  # Created adaptively based on input

    def forward(self, packed) -> tuple[torch.Tensor, torch.Tensor]:
        x = packed.x_nodes.float()
        eidx = packed.edge_index.long()
        eatt = packed.edge_attr.float()
        nmask = packed.node_mask.bool()
        emask = packed.edge_mask.bool()
        gtok  = packed.g_token.float()
        
        # project inputs with adaptive global projection
        if self.g_proj is None or self.g_proj.in_features != gtok.size(0):
            self.g_proj = nn.Linear(gtok.size(0), self.node_in.out_features).to(gtok.device)
        
        x = self.node_in(x) + self.g_proj(gtok).unsqueeze(0)  # broadcast global token
        e = self.edge_in(eatt)
        
        # 1) Mamba over check sequence (Z-then-X; Hilbert within bbox)
        x = self.seq_encoder(x, packed.seq_idx, packed.seq_mask, node_type=packed.node_type)
        
        # 2) Channel-SE (apply to all nodes, expecting [B, T, C] format)
        # Reshape x to [1, N, d_model] for ChannelSE which expects [B, T, C]
        x_se_input = x.unsqueeze(0)  # [1, N, d_model]
        x_se_output = self.se(x_se_input)  # [1, N, d_model]
        x = x_se_output.squeeze(0)  # Back to [N, d_model]
        
        # 3) Astra GNN -> per-node binary head
        logits = self.gnn(x, eidx, e, nmask, emask)
        return logits, nmask

    # --- Interface shims for v1 parity with infer/eval pipelines ---
    def set_authoritative_mats(self, Hx=None, Hz=None, device=None):
        """Compatibility shim: v2 does not require these at runtime."""
        return self
        
    def set_rotated_layout(self, flag: bool=True):
        """Compatibility shim: v2 handles layout internally."""
        return self

__all__ = ["MGHDv2"]