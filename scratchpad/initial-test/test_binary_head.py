#!/usr/bin/env python3
"""Test the binary head enforcement for rotated layout"""

import torch
from poc_my_models import MGHD

def test_binary_head_enforcement():
    """Test that rotated layout enforces binary head regardless of metadata"""
    print("🧪 Testing binary head enforcement for rotated layout...")
    
    # Create model
    gnn_params = {
        'dist': 3,
        'n_node_inputs': 128,
        'n_node_outputs': 9,  # Start with wrong value
        'n_iters': 7,
        'n_node_features': 128,
        'n_edge_features': 384,
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
    
    model = MGHD(gnn_params=gnn_params, mamba_params=mghd_params)
    
    print(f"Initial head outputs: {model.gnn.n_node_outputs}")
    
    # Test 1: Metadata with rotated layout should enforce binary head
    meta_rotated = {
        'surface_layout': 'rotated',
        'N_bits': 9  # This should be ignored
    }
    
    model.set_output_size_from_metadata(meta_rotated)
    print(f"After rotated metadata: {model.gnn.n_node_outputs}")
    
    # Test 2: Try to set it back to 9 - should be enforced to 2 again
    meta_rotated_again = {
        'surface_layout': 'rotated',
        'N_bits': 13
    }
    
    model.set_output_size_from_metadata(meta_rotated_again)
    print(f"After trying to set to 13: {model.gnn.n_node_outputs}")
    
    # Test 3: Non-rotated metadata should do nothing (preserve current state)
    meta_other = {
        'surface_layout': 'planar',
        'N_bits': 13
    }
    
    model.set_output_size_from_metadata(meta_other)
    print(f"After planar metadata: {model.gnn.n_node_outputs}")
    
    # Verify final state
    if model.gnn.n_node_outputs == 2:
        print("✅ Binary head enforcement working correctly!")
        return True
    else:
        print(f"❌ Expected 2 logits, got {model.gnn.n_node_outputs}")
        return False

if __name__ == "__main__":
    test_binary_head_enforcement()
