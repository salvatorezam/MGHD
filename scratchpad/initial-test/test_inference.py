#!/usr/bin/env python3
"""Quick inference spot check to verify binary classification works correctly."""

import torch
import numpy as np
from poc_my_models import MGHD

def test_inference():
    """Test inference to verify binary classification outputs"""
    print("🔬 Testing MGHD inference with binary classification...")
    
    # Create model parameters (matching training script)
    n_node_features = 128
    n_edge_features = 384
    n_iters = 7
    n_node_outputs = 2  # Binary classification: 2 logits per data qubit
    
    gnn_params = {
        'dist': 3,
        'n_node_inputs': n_node_features,
        'n_node_outputs': n_node_outputs,
        'n_iters': n_iters,
        'n_node_features': n_node_features,
        'n_edge_features': n_edge_features,
        'msg_net_size': 96,
        'msg_net_dropout_p': 0.05,
        'gru_dropout_p': 0.09
    }
    
    mghd_params = {
        'd_model': 192,
        'd_state': 64,
        'd_conv': 4,
        'expand': 2,
        'n_layers': 4,
        'attention': 'channel_attention',
        'se_reduction': 4
    }
    
    model = MGHD(
        gnn_params=gnn_params,
        mamba_params=mghd_params
    )
    
    # Set up for rotated d=3 (like in pack processing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.set_rotated_layout()
    model._ensure_static_indices(device)
    model.eval()
    
    print(f"✓ Model created with rotated layout: {model._num_check_nodes} checks + {model._num_data_qubits} data = {model._num_check_nodes + model._num_data_qubits} nodes")
    print(f"✓ Head outputs {model.gnn.n_node_outputs} logits per data qubit")
    
    # Create test syndrome (packed format - 1 byte for 8 syndromes)
    test_syndrome = torch.randint(0, 256, size=(1,), dtype=torch.uint8)
    
    # Test inference
    with torch.no_grad():
        decisions = model.decode_one(test_syndrome, device=device.type)
    
    print(f"✓ Input syndrome (packed): {test_syndrome.cpu().numpy()}")
    print(f"✓ Output decisions shape: {decisions.shape}")
    print(f"✓ Output decisions dtype: {decisions.dtype}")
    print(f"✓ Output decisions: {decisions.cpu().numpy().flatten()}")
    
    # Verify output format
    expected_shape = (1, model._num_data_qubits)
    if decisions.shape == expected_shape:
        print(f"✅ Correct shape: {decisions.shape} == {expected_shape}")
    else:
        print(f"❌ Wrong shape: {decisions.shape} != {expected_shape}")
        return False
    
    # Verify binary values
    unique_values = torch.unique(decisions).cpu().numpy()
    if set(unique_values).issubset({0, 1}):
        print(f"✅ Binary output: unique values {unique_values}")
    else:
        print(f"❌ Non-binary output: unique values {unique_values}")
        return False
    
    print("🎉 Binary classification inference test PASSED!")
    return True

if __name__ == "__main__":
    test_inference()
