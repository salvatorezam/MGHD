#!/usr/bin/env python3
"""Generate minimal test crops for sanity training."""

import numpy as np
import torch
import os
from mghd_public.features_v2 import PackedCrop
from mghd_clustered.garnet_adapter import sample_round
from teachers.mwpf_ctx import MWPFContext


def create_mock_crop(d: int = 3, p: float = 0.003, seed: int = 42) -> PackedCrop:
    """Create a single mock crop with real H_sub data."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Use the real Garnet adapter to get syndrome data
    synd_data = sample_round(d, p, seed)
    
    # Create basic graph structure (minimal)
    num_data_qubits = d * d
    num_check_qubits = (d - 1) * d + d * (d - 1)  # Simplified surface code
    num_nodes = num_data_qubits + num_check_qubits
    
    # Basic node features
    x_nodes = torch.randn(num_nodes, 8)  # Model expects 8-dimensional features
    node_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    # Node types: 0=data, 1=check
    node_type = torch.zeros(num_nodes, dtype=torch.int8)
    node_type[num_data_qubits:] = 1  # Check qubits
    
    # Basic edge structure (star graph for simplicity)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).T
    edge_attr = torch.randn(2, 3)  # Model expects 3-dimensional edge features
    edge_mask = torch.ones(2, dtype=torch.bool)
    
    # Sequence data
    seq_idx = torch.arange(num_nodes, dtype=torch.long)
    seq_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    # Global token
    g_token = torch.randn(8)  # Model expects 8-dimensional global token
    
    # Target bits (use syndrome from Garnet)
    # y_bits should match the TOTAL number of nodes, not just data qubits
    y_bits = torch.zeros(num_nodes, dtype=torch.int8)  # Changed from num_data_qubits to num_nodes
    if 'synZ' in synd_data and len(synd_data['synZ']) > 0:
        # Use some syndrome data as target for data qubits only
        synd_bits = synd_data['synZ']
        data_indices = torch.where(torch.arange(num_nodes) < num_data_qubits)[0]
        synd_tensor = torch.from_numpy(synd_bits[:min(len(synd_bits), num_data_qubits)]).to(torch.int8)
        y_bits[data_indices[:min(len(synd_bits), num_data_qubits)]] = synd_tensor
    
    # Meta data
    meta = {
        'distance': d,
        'p_error': p,
        'layout': "surface_code",
        'seed': seed
    }
    
    # H_sub from Garnet data or create a simple one
    if 'Hx' in synd_data:
        # Use a subset of the real H matrix that matches our data qubits
        H_full = synd_data['Hx']
        # Take a submatrix that matches our simplified graph structure
        H_sub = H_full[:min(3, H_full.shape[0]), :min(num_data_qubits, H_full.shape[1])]
    else:
        # Create H_sub that matches the number of data qubits
        H_sub = np.random.randint(0, 2, (3, num_data_qubits))  # 3 checks x num_data_qubits
    
    # Local index maps (should match H_sub dimensions)
    idx_data_local = np.arange(min(num_data_qubits, H_sub.shape[1]))
    idx_check_local = np.arange(min(num_check_qubits, H_sub.shape[0]))
    
    return PackedCrop(
        x_nodes=x_nodes,
        node_mask=node_mask,
        node_type=node_type,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        seq_idx=seq_idx,
        seq_mask=seq_mask,
        g_token=g_token,
        y_bits=y_bits,
        meta=meta,
        H_sub=H_sub,
        idx_data_local=idx_data_local,
        idx_check_local=idx_check_local,
    )


def main():
    """Generate test crops and save to NPZ."""
    print("Generating test crops...")
    
    os.makedirs('test_crops', exist_ok=True)
    
    # Generate 4 crops with different seeds
    crops = []
    for i in range(4):
        crop = create_mock_crop(d=3, p=0.003, seed=42 + i)
        crops.append(crop)
        print(f"Generated crop {i+1}: H_sub shape {crop.H_sub.shape}")
    
    # Convert to dict format
    packed_crops = []
    for crop in crops:
        crop_dict = {
            'x_nodes': crop.x_nodes.cpu().numpy(),
            'node_mask': crop.node_mask.cpu().numpy(),
            'node_type': crop.node_type.cpu().numpy(),
            'edge_index': crop.edge_index.cpu().numpy(),
            'edge_attr': crop.edge_attr.cpu().numpy(),
            'edge_mask': crop.edge_mask.cpu().numpy(),
            'seq_idx': crop.seq_idx.cpu().numpy(),
            'seq_mask': crop.seq_mask.cpu().numpy(),
            'g_token': crop.g_token.cpu().numpy(),
            'y_bits': crop.y_bits.cpu().numpy(),
            'meta': crop.meta,
            'H_sub': crop.H_sub,
            'idx_data_local': crop.idx_data_local,
            'idx_check_local': crop.idx_check_local,
        }
        packed_crops.append(crop_dict)
    
    # Save to NPZ
    crop_file = 'test_crops/sanity_test.npz'
    np.savez(crop_file, packed=np.array(packed_crops))
    print(f"Saved {len(crops)} crops to {crop_file}")
    
    # Verify
    data = np.load(crop_file, allow_pickle=True)
    packed_array = data['packed']
    print(f"Verification: shape {packed_array.shape}, first crop H_sub shape {packed_array[0]['H_sub'].shape}")


if __name__ == "__main__":
    main()