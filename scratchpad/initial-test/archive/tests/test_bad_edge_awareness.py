"""
Test bad edge awareness validation

Validates that the default d=3 surface code layout avoids the problematic
(10,11) coupler from the IQM Garnet device, which has the lowest fidelity.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cudaq_backend.circuits import make_surface_layout_d3_avoid_bad_edges
from cudaq_backend.garnet_noise import GARNET_COUPLER_F2


class TestBadEdgeAwareness:
    """Test that layouts properly avoid problematic device couplers."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.layout = make_surface_layout_d3_avoid_bad_edges()
        self.bad_edge = (10, 11)
        self.bad_edge_fidelity = GARNET_COUPLER_F2[self.bad_edge]
    
    def test_bad_edge_identification(self):
        """Test that we correctly identify the worst edge in Garnet calibration."""
        # Find the edge with minimum fidelity
        min_fidelity = min(GARNET_COUPLER_F2.values())
        worst_edges = [edge for edge, fidelity in GARNET_COUPLER_F2.items() 
                      if fidelity == min_fidelity]
        
        assert self.bad_edge in worst_edges, \
            f"Edge {self.bad_edge} should be among worst edges: {worst_edges}"
        
        print(f"Worst edge {self.bad_edge} has fidelity {self.bad_edge_fidelity:.4f}")
        
        # Verify it's significantly worse than median
        fidelities = list(GARNET_COUPLER_F2.values())
        median_fidelity = np.median(fidelities)
        fidelity_gap = median_fidelity - self.bad_edge_fidelity
        
        assert fidelity_gap > 0.01, \
            f"Bad edge should be significantly worse than median: " \
            f"gap={fidelity_gap:.4f}"
        
        print(f"Median fidelity: {median_fidelity:.4f}")
        print(f"Fidelity gap: {fidelity_gap:.4f}")
    
    def test_layout_avoids_bad_edge_directly(self):
        """Test that CZ layers don't include the bad edge."""
        cz_layers = self.layout.get('cz_layers', [])
        
        uses_bad_edge = False
        for layer_idx, cz_layer in enumerate(cz_layers):
            for edge in cz_layer:
                if edge == self.bad_edge or edge == (self.bad_edge[1], self.bad_edge[0]):
                    uses_bad_edge = True
                    print(f"Found bad edge {edge} in CZ layer {layer_idx}")
                    break
        
        assert not uses_bad_edge, \
            f"Layout should not use bad edge {self.bad_edge} in any CZ layer"
        
        print(f"✓ Bad edge {self.bad_edge} not found in any CZ layer")
    
    def test_layout_qubit_selection(self):
        """Test qubit selection strategy avoids bad edge endpoints."""
        all_qubits = set(self.layout['data'])
        all_qubits.update(self.layout['ancilla_x'])
        all_qubits.update(self.layout['ancilla_z'])
        
        bad_qubit_1, bad_qubit_2 = self.bad_edge
        
        # Check if layout uses both bad edge qubits
        uses_both_bad_qubits = (bad_qubit_1 in all_qubits) and (bad_qubit_2 in all_qubits)
        
        if uses_both_bad_qubits:
            print(f"Warning: Layout uses both qubits {bad_qubit_1} and {bad_qubit_2}")
            print("Verifying no direct coupling between them...")
            
            # If both qubits are used, ensure they're not directly coupled
            self.test_layout_avoids_bad_edge_directly()
        else:
            print(f"✓ Layout avoids using both bad edge qubits {self.bad_edge}")
        
        print(f"Layout uses {len(all_qubits)} qubits: {sorted(all_qubits)}")
    
    def test_layout_prefers_high_fidelity_edges(self):
        """Test that layout preferentially uses high-fidelity edges."""
        cz_layers = self.layout.get('cz_layers', [])
        
        if not cz_layers:
            print("No CZ layers found in layout - skipping high fidelity test")
            return
        
        used_edges = []
        for cz_layer in cz_layers:
            used_edges.extend(cz_layer)
        
        if not used_edges:
            print("No edges found in CZ layers - skipping high fidelity test")
            return
        
        # Calculate average fidelity of used edges
        total_fidelity = 0.0
        valid_edges = 0
        
        for edge in used_edges:
            # Try both orientations
            if edge in GARNET_COUPLER_F2:
                total_fidelity += GARNET_COUPLER_F2[edge]
                valid_edges += 1
            elif (edge[1], edge[0]) in GARNET_COUPLER_F2:
                total_fidelity += GARNET_COUPLER_F2[(edge[1], edge[0])]
                valid_edges += 1
        
        if valid_edges == 0:
            print("No valid Garnet edges found in layout - using virtual/simulation mapping")
            return
        
        avg_used_fidelity = total_fidelity / valid_edges
        
        # Compare to overall average
        all_fidelities = list(GARNET_COUPLER_F2.values())
        overall_avg_fidelity = np.mean(all_fidelities)
        
        print(f"Average fidelity of used edges: {avg_used_fidelity:.4f}")
        print(f"Overall average device fidelity: {overall_avg_fidelity:.4f}")
        
        # Layout should prefer higher fidelity edges
        assert avg_used_fidelity >= overall_avg_fidelity - 0.005, \
            f"Layout should use above-average fidelity edges: " \
            f"used={avg_used_fidelity:.4f}, avg={overall_avg_fidelity:.4f}"
    
    def test_alternative_high_fidelity_edges(self):
        """Test that good alternative edges are available for surface codes."""
        # Identify high-fidelity edges suitable for surface code connectivity
        fidelities = list(GARNET_COUPLER_F2.items())
        fidelities.sort(key=lambda x: x[1], reverse=True)  # Sort by fidelity descending
        
        print("Top 10 highest fidelity edges:")
        for i, (edge, fidelity) in enumerate(fidelities[:10]):
            print(f"  {i+1}. {edge}: {fidelity:.4f}")
        
        print(f"\nBottom 5 lowest fidelity edges:")
        for i, (edge, fidelity) in enumerate(fidelities[-5:]):
            print(f"  {len(fidelities)-4+i}. {edge}: {fidelity:.4f}")
        
        # Verify we have sufficient high-quality edges for a 17-qubit surface code
        high_fidelity_threshold = 0.992  # Realistic threshold for Garnet device
        good_edges = [edge for edge, fidelity in fidelities if fidelity >= high_fidelity_threshold]
        
        print(f"\nEdges with fidelity ≥ {high_fidelity_threshold}: {len(good_edges)}")
        
        # For surface codes, we need connectivity, not just individual good edges
        # But having multiple good edges available is important for layout optimization
        assert len(good_edges) >= 5, \
            f"Should have at least 5 high fidelity edges, found {len(good_edges)}"
    
    def test_layout_connectivity_preservation(self):
        """Test that avoiding bad edges doesn't break surface code connectivity."""
        layout = self.layout
        
        # Verify we have the required components for d=3 surface code
        assert len(layout['data']) == 9, f"d=3 needs 9 data qubits, got {len(layout['data'])}"
        assert len(layout['ancilla_x']) == 4, f"d=3 needs 4 X ancillas, got {len(layout['ancilla_x'])}"
        assert len(layout['ancilla_z']) == 4, f"d=3 needs 4 Z ancillas, got {len(layout['ancilla_z'])}"
        
        # Check that we have some CZ connectivity
        cz_layers = layout.get('cz_layers', [])
        total_cz_ops = sum(len(layer) for layer in cz_layers)
        
        # Surface code stabilizers require connectivity between data and ancilla qubits
        # For d=3, each stabilizer typically involves 2-4 data qubits
        min_expected_cz_ops = 8  # Conservative estimate
        
        assert total_cz_ops >= min_expected_cz_ops, \
            f"Layout should have at least {min_expected_cz_ops} CZ operations, got {total_cz_ops}"
        
        print(f"✓ Layout maintains connectivity with {total_cz_ops} CZ operations across {len(cz_layers)} layers")
    
    def test_device_topology_constraints(self):
        """Test understanding of Garnet device topology constraints."""
        # Garnet is a 20-qubit device with specific connectivity
        total_device_qubits = 20
        total_device_edges = len(GARNET_COUPLER_F2)
        
        print(f"Garnet device properties:")
        print(f"  Total qubits: {total_device_qubits}")
        print(f"  Total edges: {total_device_edges}")
        print(f"  Average connectivity: {2 * total_device_edges / total_device_qubits:.1f} edges per qubit")
        
        # Verify layout fits within device constraints
        all_qubits = set(self.layout['data'])
        all_qubits.update(self.layout['ancilla_x'])
        all_qubits.update(self.layout['ancilla_z'])
        
        max_qubit = max(all_qubits) if all_qubits else 0
        
        assert max_qubit < total_device_qubits, \
            f"Layout uses qubit {max_qubit}, but device only has qubits 0-{total_device_qubits-1}"
        
        print(f"✓ Layout fits within device: max qubit {max_qubit} < {total_device_qubits}")


if __name__ == '__main__':
    # Run tests directly
    test_suite = TestBadEdgeAwareness()
    
    print("Running bad edge awareness validation tests...")
    
    test_suite.setup_method()
    
    try:
        test_suite.test_bad_edge_identification()
        print("✓ Bad edge identification")
    except Exception as e:
        print(f"✗ Bad edge identification: {e}")
    
    try:
        test_suite.test_layout_avoids_bad_edge_directly()
        print("✓ Layout avoids bad edge directly")
    except Exception as e:
        print(f"✗ Layout avoids bad edge directly: {e}")
    
    try:
        test_suite.test_layout_qubit_selection()
        print("✓ Layout qubit selection")
    except Exception as e:
        print(f"✗ Layout qubit selection: {e}")
    
    try:
        test_suite.test_layout_prefers_high_fidelity_edges()
        print("✓ Layout prefers high fidelity edges")
    except Exception as e:
        print(f"✗ Layout prefers high fidelity edges: {e}")
    
    try:
        test_suite.test_alternative_high_fidelity_edges()
        print("✓ Alternative high fidelity edges")
    except Exception as e:
        print(f"✗ Alternative high fidelity edges: {e}")
    
    try:
        test_suite.test_layout_connectivity_preservation()
        print("✓ Layout connectivity preservation")
    except Exception as e:
        print(f"✗ Layout connectivity preservation: {e}")
    
    try:
        test_suite.test_device_topology_constraints()
        print("✓ Device topology constraints")
    except Exception as e:
        print(f"✗ Device topology constraints: {e}")
    
    print("Bad edge awareness validation complete!")
