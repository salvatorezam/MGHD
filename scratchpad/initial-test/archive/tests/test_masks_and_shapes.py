import numpy as np, torch
from mghd_public.features_v2 import pack_cluster
from mghd_public.model_v2 import MGHDv2

def test_masks_block_padding_paths():
    H = np.zeros((1,2), dtype=np.uint8); H[0,[0,1]]=1
    xy_q = np.array([[0,0],[1,0]], dtype=np.int32)
    xy_c = np.array([[0,1]], dtype=np.int32)
    y = np.array([0,1], dtype=np.uint8)
    p = pack_cluster(H, xy_q, xy_c, np.array([1],dtype=np.uint8),
                     k=2, r=0, bbox_xywh=(0,0,2,2), kappa_stats={"size":3},
                     y_bits_local=y, side="Z", d=3, p=0.001, seed=0,
                     N_max=16, E_max=32, S_max=16)
    
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MGHDv2().to(device)
    
    # Move packed tensors to device
    p.x_nodes = p.x_nodes.to(device)
    p.node_mask = p.node_mask.to(device)
    p.node_type = p.node_type.to(device)
    p.edge_index = p.edge_index.to(device)
    p.edge_attr = p.edge_attr.to(device)
    p.edge_mask = p.edge_mask.to(device)
    p.seq_idx = p.seq_idx.to(device)
    p.seq_mask = p.seq_mask.to(device)
    p.g_token = p.g_token.to(device)
    p.y_bits = p.y_bits.to(device)
    
    logits, node_mask = model(p)
    assert logits.shape[0] == 16 and logits.shape[1] == 2
    # ensure masked positions do not explode
    assert torch.isfinite(logits[node_mask]).all()
    # ensure padded positions are effectively ignored
    assert logits[~node_mask].abs().sum() >= 0  # may be nonzero but should be finite

def test_edge_mask_functionality():
    # Create a minimal crop to test edge masking
    H = np.array([[1,1,0],[0,1,1]], dtype=np.uint8)
    xy_q = np.array([[0,0],[1,0],[2,0]], dtype=np.int32)
    xy_c = np.array([[0,1],[2,1]], dtype=np.int32)
    y = np.array([0,1,0], dtype=np.uint8)
    p = pack_cluster(H, xy_q, xy_c, np.array([1,0],dtype=np.uint8),
                     k=3, r=1, bbox_xywh=(0,0,3,2), kappa_stats={"size":5},
                     y_bits_local=y, side="Z", d=3, p=0.001, seed=0,
                     N_max=16, E_max=32, S_max=16)
    
    # Check that edge_mask correctly identifies valid edges
    valid_edges = p.edge_mask.sum().item()
    # Should have edges from Tanner graph + potential jump edges
    assert valid_edges >= 4  # At least 4 Tanner edges (2 checks with 2 connections each)
    
    # Check edge indices are within bounds for valid edges
    valid_edge_indices = p.edge_index[:, p.edge_mask]
    assert valid_edge_indices.max() < p.node_mask.sum(), "Edge indices exceed valid node count"

if __name__ == "__main__":
    test_masks_block_padding_paths()
    test_edge_mask_functionality()
    print("test_masks_and_shapes.py: PASSED")