import numpy as np, torch
from mghd_public.features_v2 import pack_cluster

def test_translation_distance_invariance():
    # Construct a tiny crop, then translate and change d; packed tensors (excluding meta) should match.
    H = np.zeros((2,3), dtype=np.uint8); H[0,[0,1]]=1; H[1,[1,2]]=1
    xy_q = np.array([[10,10],[11,10],[12,10]], dtype=np.int32)
    xy_c = np.array([[10,9],[12,9]], dtype=np.int32)
    bbox = (9,8,6,4)
    y = np.array([0,1,0], dtype=np.uint8)
    p1 = pack_cluster(H, xy_q, xy_c, np.array([1,0],dtype=np.uint8),
                      k=3, r=1, bbox_xywh=bbox, kappa_stats={"size":5},
                      y_bits_local=y, side="Z", d=3, p=0.005, seed=1,
                      N_max=32, E_max=64, S_max=16)
    # translate + different d
    xy_q2 = xy_q + np.array([100, 50]); xy_c2 = xy_c + np.array([100, 50])
    bbox2 = (bbox[0]+100, bbox[1]+50, bbox[2], bbox[3])
    p2 = pack_cluster(H, xy_q2, xy_c2, np.array([1,0],dtype=np.uint8),
                      k=3, r=1, bbox_xywh=bbox2, kappa_stats={"size":5},
                      y_bits_local=y, side="Z", d=31, p=0.005, seed=2,
                      N_max=32, E_max=64, S_max=16)
    # Compare tensors
    assert torch.allclose(p1.x_nodes, p2.x_nodes)
    assert torch.equal(p1.node_mask, p2.node_mask)
    assert torch.equal(p1.node_type, p2.node_type)
    assert torch.equal(p1.edge_index, p2.edge_index)
    assert torch.allclose(p1.edge_attr, p2.edge_attr)
    assert torch.equal(p1.seq_mask, p2.seq_mask)

if __name__ == "__main__":
    test_translation_distance_invariance()
    print("test_features_v2_invariance.py: PASSED")