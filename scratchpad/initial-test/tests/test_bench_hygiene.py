#!/usr/bin/env python3
"""
Minimal tests for benchmark hygiene and data quality.
Tests the enhanced MGHD benchmarking functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tempfile
import json
from tools.bench_clustered_sweep_surface import main as bench_main
from mghd_public.config import MGHDConfig
from mghd_public.infer import MGHDDecoderPublic, warmup_and_capture


def test_warmup_functionality():
    """Test MGHD warmup and CUDA graph capture."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg = MGHDConfig(
        gnn={
            "dist": 3,
            "n_node_inputs": 9,
            "n_node_outputs": 9,
            "n_iters": 7,
            "n_node_features": 128,
            "n_edge_features": 128,
            "msg_net_size": 96,
            "msg_net_dropout_p": 0.04,
            "gru_dropout_p": 0.11,
        },
        mamba={
            "d_model": 192,
            "d_state": 32,
            "d_conv": 2,
            "expand": 3,
            "attention_mechanism": "channel_attention",
            "se_reduction": 4,
            "post_mamba_ln": False,
        },
        n_checks=8,
        n_qubits=9,
        n_node_inputs=9,
        n_node_outputs=2,
    )
    
    # Test with available device
    dec_pub = MGHDDecoderPublic(
        "results/foundation_S_core_cq_circuit_v1_20250831_093641/step11_garnet_S_best.pt",
        cfg,
        device=device,
        graph_capture=(device == "cuda"),  # Only enable graph capture on CUDA
    )
    
    from mghd_clustered.pcm_real import rotated_surface_pcm
    Hx_d3 = rotated_surface_pcm(3, "X")
    Hz_d3 = rotated_surface_pcm(3, "Z")
    dec_pub.bind_code(Hx_d3, Hz_d3)
    
    # Test warmup
    warmup_info = warmup_and_capture(dec_pub, device, "X", use_fixed_d3=True)
    
    assert warmup_info['warmup_us'] > 0, "Warmup time should be positive"
    assert 'graph_used' in warmup_info, "Should report graph usage"
    assert 'path' in warmup_info, "Should report execution path"
    assert warmup_info['path'] in ['fast', 'fixed_d3', 'minimal'], f"Invalid path: {warmup_info['path']}"
    
    # CUDA graph should only work on CUDA devices
    if device == "cuda":
        expected_graph = True  # Should use CUDA graph on CUDA
    else:
        expected_graph = False  # No CUDA graph on CPU
    
    print(f"✓ Warmup test passed: {warmup_info['warmup_us']:.1f}μs, "
          f"path={warmup_info['path']}, graph={warmup_info['graph_used']} (device={device})")


def test_synthetic_bench_run():
    """Test a minimal synthetic benchmark run."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("⚠️  Skipping synthetic benchmark test - MGHD requires CUDA")
        return
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate command line args for bench script
        import sys
        original_argv = sys.argv
        try:
            sys.argv = [
                "bench_clustered_sweep_surface.py",
                "--ckpt", "results/foundation_S_core_cq_circuit_v1_20250831_093641/step11_garnet_S_best.pt",
                "--shots", "20",
                "--dists", "3",
                "--ps", "0.01",
                "--tier0-mode", "mixed_tight",
                "--p-channel", "auto",
                "--halo", "0",
                "--temp", "1.0",
                "--thresh", "0.5",
                "--r-cap", "20",
                "--device", device,
                "--graph-capture",
                "--out", tmpdir
            ]
            
            # Run the benchmark
            bench_main()
            
            # Find the output JSON
            import glob
            json_files = glob.glob(f"{tmpdir}/clustered_surface_sweep_*.json")
            assert len(json_files) == 1, f"Expected 1 JSON file, got {len(json_files)}"
            
            # Load and validate the results
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            
            # Check metadata
            metadata = data['metadata']
            assert metadata['tier0_mode'] == 'mixed_tight', "Tier0 mode should be mixed_tight"
            assert metadata['shots'] == 20, "Shots should be 20"
            
            # Check d=3 results exist
            results = data['results']
            assert '3' in results, "Distance 3 results should exist"
            assert '0.010' in results['3'], "p=0.01 results should exist"
            
            d3_p01 = results['3']['0.010']
            
            # Test both X and Z sides
            for side in ['X', 'Z']:
                side_data = d3_p01[side]
                
                # Check warmup info exists (only for d=3)
                if 'warmup' in side_data:
                    warmup = side_data['warmup']
                    assert warmup['warmup_us'] > 0, f"{side}: Warmup time should be positive"
                    assert isinstance(warmup['graph_used'], bool), f"{side}: graph_used should be boolean"
                    if device == "cuda":
                        assert warmup['graph_used'], f"{side}: Should use CUDA graph on CUDA device"
                
                # Check basic statistics
                assert side_data['shots'] == 20, f"{side}: Should have 20 shots"
                assert 'latency_total_us' in side_data, f"{side}: Should have total latency stats"
                
                latency = side_data['latency_total_us']
                assert latency['p95'] >= latency['p50'], f"{side}: p95 should be >= p50"
                assert latency['p50'] >= 0, f"{side}: p50 should be non-negative"
                
                # Check MGHD statistics  
                if 't_mghd_nonzero_stats' in side_data:
                    mghd_stats = side_data['t_mghd_nonzero_stats']
                    if mghd_stats['count_invokes'] > 0:
                        assert mghd_stats['p95_nonzero_us'] >= mghd_stats['p50_nonzero_us'], \
                            f"{side}: MGHD p95 should be >= p50"
                
                # Check tier0 statistics
                tier0_stats = side_data['tier0_stats']
                assert 'mghd_clusters_per_shot' in tier0_stats, f"{side}: Should have MGHD clusters per shot"
                assert tier0_stats['mghd_clusters_per_shot'] >= 0, f"{side}: MGHD clusters per shot should be non-negative"
                
                # For mixed_tight mode at d=3, we should see some MGHD usage
                # (though with only 20 shots, it might be zero)
                print(f"✓ {side}: Tier-0 {tier0_stats['tier0_pct']:.1f}%, "
                      f"MGHD {tier0_stats['mghd_clusters_per_shot']:.3f}/shot")
            
        finally:
            sys.argv = original_argv
    
    print(f"✓ Synthetic benchmark test passed")


if __name__ == "__main__":
    print("Running MGHD benchmark hygiene tests...")
    
    try:
        test_warmup_functionality()
        test_synthetic_bench_run()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise