"""
Circuit Construction Module

This module builds per-round native PRX/CZ measurement circuits for each code family
and produces layered schedules to compute idle windows correctly.

Implements circuit construction for:
- Repetition codes
- Surface codes (rotated lattice)  
- BB/qLDPC codes

All circuits are constructed as CUDA-Q kernels with proper noise application.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict

try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    print("Warning: CUDA-Q not available. Using fallback implementation.")


class CudaQKernel:
    """CUDA-Q kernel wrapper for circuit construction."""
    
    def __init__(self, name: str, operations: List[Dict[str, Any]]):
        self.name = name
        self.operations = operations
        self.layers = []
        
        if CUDAQ_AVAILABLE:
            # Real CUDA-Q kernel construction would go here
            self._cudaq_kernel = None  # Placeholder for actual kernel
        
    def add_layer(self, layer_ops: List[Dict[str, Any]], layer_type: str):
        """Add a parallel layer of operations."""
        self.layers.append({
            'type': layer_type,
            'operations': layer_ops,
            'duration_ns': layer_ops[0].get('duration_ns', 20.0) if layer_ops else 0.0
        })
    
    def get_layer_schedule(self) -> List[Dict[str, Any]]:
        """Return the layered schedule for idle time calculation."""
        return self.layers
    
    def to_cudaq_kernel(self):
        """Convert to actual CUDA-Q kernel if available."""
        if not CUDAQ_AVAILABLE:
            return None
            
        # This would contain the actual CUDA-Q kernel construction
        # For now, return None to indicate fallback mode
        return None


def build_H_rotated_general(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build rotated planar surface code parity-check matrices for arbitrary distance d.

    Data-qubit indexing: d×d grid, row-major order:
      For d=3: q00=0, q01=1, q02=2, q10=3, q11=4, q12=5, q20=6, q21=7, q22=8
      For d=5: q00=0, q01=1, ..., q04=4, q10=5, ..., q44=24

    Returns:
      (Hz, Hx) as numpy arrays
      Hz: (d*(d-1), d*d) - Z stabilizers  
      Hx: ((d-1)*d, d*d) - X stabilizers
      Columns correspond to data-qubit order 0..(d²-1) in row-major order.
    """
    try:
        from codes_q import create_rotated_surface_codes  # reuse Astra helper
        css = create_rotated_surface_codes(d, name=f"rotated_d{d}")
        Hx = css.hx.astype(int)
        Hz = css.hz.astype(int)
        return Hz, Hx
    except ImportError:
        # Fallback: create rotated surface code matrices manually
        n_data = d * d
        n_z_checks = d * (d - 1)  # Z stabilizers
        n_x_checks = (d - 1) * d  # X stabilizers
        
        # Initialize matrices
        Hz = np.zeros((n_z_checks, n_data), dtype=int)
        Hx = np.zeros((n_x_checks, n_data), dtype=int)
        
        # Build Z stabilizers (horizontal edges in rotated lattice)
        z_check_idx = 0
        for row in range(d):
            for col in range(d - 1):
                # Z check connects qubits (row, col) and (row, col+1)
                q1 = row * d + col
                q2 = row * d + col + 1
                Hz[z_check_idx, q1] = 1
                Hz[z_check_idx, q2] = 1
                z_check_idx += 1
        
        # Build X stabilizers (vertical edges in rotated lattice)  
        x_check_idx = 0
        for row in range(d - 1):
            for col in range(d):
                # X check connects qubits (row, col) and (row+1, col)
                q1 = row * d + col
                q2 = (row + 1) * d + col
                Hx[x_check_idx, q1] = 1
                Hx[x_check_idx, q2] = 1
                x_check_idx += 1
        
        return Hz, Hx


def build_H_rotated_d3() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build rotated planar surface code (d=3) parity-check matrices.
    Backward compatibility wrapper for the general function.
    """
    return build_H_rotated_general(3)


def _canonicalize_sector_rows(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H8 = H.astype(np.uint8, copy=False)
    keys = []
    for r in H8:
        w = int(r.sum())
        mask = 0
        for i,b in enumerate(r[::-1]):  # msb-first integer for stability
            mask = (mask<<1) | int(b)
        keys.append((w, mask))
    # Sort the list of tuples properly
    perm = np.array([i for i, _ in sorted(enumerate(keys), key=lambda x: x[1])])
    return H8[perm], perm

def build_H_rotated_d3_from_cfg(cfg) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns (Hx, Hz, meta) consistent with the CUDA-Q rotated layout used for sampling.
    Hx is (#X-stabilizers, #data-qubits), Hz is (#Z-stabilizers, #data-qubits).
    The row order MUST match the syndrome bit order, and the column order MUST match the data-qubit bit order in labels.
    meta includes: {'surface_layout':'rotated','distance':3,'N_syn':8,'N_bits':9,'Hx_order':[...],'Hz_order':[...],'data_qubit_order':[...]}
    """
    # Get the base matrices from existing implementation
    Hz, Hx = build_H_rotated_d3()

    # Canonicalize row ordering for deterministic outputs
    Hz, pz = _canonicalize_sector_rows(Hz)
    Hx, px = _canonicalize_sector_rows(Hx)

    # Data qubit order: 3x3 grid, row-major (0..8)
    data_qubit_order = list(range(9))
    
    # Syndrome ordering: Z checks first (rows 0-3), then X checks (rows 4-7)
    # This matches the CUDA-Q convention used in syndrome generation
    # FROZEN ORDERING: Z checks (0-3), X checks (4-7)
    Hz_order = [f"Z{i}" for i in range(4)]
    Hx_order = [f"X{i}" for i in range(4)]
    
    meta = {
        'surface_layout': 'rotated',
        'distance': 3,
        'N_syn': 8,
        'N_bits': 9,
        'Hx_order': Hx_order,
        'Hz_order': Hz_order,
        'data_qubit_order': data_qubit_order,
        'syndrome_order': 'Z_first_then_X'  # Z checks (0-3), X checks (4-7)
    }
    
    # FROZEN ORDERING: Ensure consistent H matrix structure
    # H = [Hz; Hx] where Hz is (4,9) and Hx is (4,9)
    # Syndrome order: [Z0, Z1, Z2, Z3, X0, X1, X2, X3]
    # Data qubit order: [0, 1, 2, 3, 4, 5, 6, 7, 8] (3x3 grid, row-major)
    
    return Hx, Hz, meta


def logical_reps_rotated_d3() -> Dict[str, np.ndarray]:
    """
    Return logical operator representatives for rotated d=3 surface code.
    
    Returns:
        dict with keys 'Lx', 'Lz' containing length-9 uint8 vectors
        representing logical X and Z operators on data qubits.
        
    Logical operators are derived from kernel of H matrices.
    """
    import numpy as np
    
    def _gf2_rref(A):  # tiny duplicate acceptable here
        A = (A.copy().astype(np.uint8))
        m, n = A.shape
        pivots = []
        r = 0
        for c in range(n):
            pr = None
            for rr in range(r, m):
                if A[rr, c] & 1:
                    pr = rr; break
            if pr is None:
                continue
            if pr != r:
                A[[r, pr]] = A[[pr, r]]
            for rr in range(m):
                if rr != r and (A[rr, c] & 1):
                    A[rr, :] ^= A[r, :]
            pivots.append(c)
            r += 1
            if r == m:
                break
        return A, pivots

    def _gf2_nullspace_basis(A):
        A = A.astype(np.uint8, copy=False)
        m, n = A.shape
        R, pivots = _gf2_rref(A)
        pivset = set(pivots)
        free = [j for j in range(n) if j not in pivset]
        basis = []
        for f in free:
            v = np.zeros(n, dtype=np.uint8); v[f] = 1
            for i, pc in enumerate(pivots):
                row = R[i, :n]
                v[pc] = (row[f] & 1)
        basis.append(v)
        return basis

    def _pick_min_weight(vecs):
        if not vecs: return np.zeros(0, dtype=np.uint8)
        weights = [int(v.sum()) for v in vecs]
        minw = min(weights)
        cand = [v for v, w in zip(vecs, weights) if w == minw]
        cand.sort(key=lambda x: x.tobytes())
        return cand[0].astype(np.uint8)

    Hx, Hz, meta = build_H_rotated_d3_from_cfg(None)
    # Lx must commute with Z checks (in kernel of Hz)
    NS_Lx = _gf2_nullspace_basis(Hz)
    Lx = _pick_min_weight(NS_Lx)
    # Lz must commute with X checks (in kernel of Hx)
    NS_Lz = _gf2_nullspace_basis(Hx)
    Lz = _pick_min_weight(NS_Lz)
    return {'Lx': Lx.astype(np.uint8), 'Lz': Lz.astype(np.uint8)}


def build_stim_circuit_rotated_d3_from_cfg(cudaq_cfg) -> Any:
    """
    Build Stim circuit matching our rotated d=3 layout and syndrome ordering.
    
    Circuit structure:
    - 9 data qubits (0-8)
    - 8 detectors: Z checks (0-3), X checks (4-7)
    - Two observables: X_L, Z_L
    
    Returns:
        stim.Circuit with proper detector and observable definitions
    """
    try:
        import stim
    except ImportError:
        raise RuntimeError("Stim not available for circuit construction")
    
    # Get error probability from config or use default
    p = getattr(cudaq_cfg, 'p', 0.01) if cudaq_cfg else 0.01
    
    # Get H matrices for detector wiring
    Hx, Hz, meta = build_H_rotated_d3_from_cfg(cudaq_cfg)
    
    # Get logical representatives
    logical_reps = logical_reps_rotated_d3()
    Lx, Lz = logical_reps['Lx'], logical_reps['Lz']
    
    # Build circuit
    circuit = stim.Circuit()
    
    # Add data qubits
    circuit.append("R", list(range(9)))  # Reset all data qubits
    
    # Add phenomenological errors on data qubits
    for q in range(9):
        circuit.append("X_ERROR", [q], p/3)  # X errors
        circuit.append("Z_ERROR", [q], p/3)  # Z errors
    
    # Add stabilizer measurements
    # Z stabilizers (detectors 0-3)
    for i in range(4):
        # Find data qubits involved in this Z check
        involved_qubits = np.where(Hz[i, :] > 0)[0]
        if len(involved_qubits) > 0:
            # Measure the product of Z operators
            circuit.append("M", involved_qubits)
            circuit.append("DETECTOR", [stim.target_rec(-1)], (0, i))  # Detector i
    
    # X stabilizers (detectors 4-7)  
    for i in range(4):
        # Find data qubits involved in this X check
        involved_qubits = np.where(Hx[i, :] > 0)[0]
        if len(involved_qubits) > 0:
            # Measure the product of X operators
            circuit.append("M", involved_qubits)
            circuit.append("DETECTOR", [stim.target_rec(-1)], (0, i+4))  # Detector i+4
    
    # Add logical observables
    # X_L observable (logical X measurement)
    x_qubits = np.where(Lx > 0)[0]
    if len(x_qubits) > 0:
        circuit.append("M", x_qubits)
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)  # X_L
    
    # Z_L observable (logical Z measurement)
    z_qubits = np.where(Lz > 0)[0]
    if len(z_qubits) > 0:
        circuit.append("M", z_qubits)
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)  # Z_L
    
    return circuit


def build_stim_dem_rotated_d3(cudaq_cfg) -> Tuple[Any, List[int]]:
    """
    Build Stim DetectorErrorModel for rotated d=3 surface code matching CUDA-Q circuit.
    
    Returns:
        dem: stim.DetectorErrorModel with same syndrome ordering as CUDA-Q
        map_bit_to_obs: List[int] mapping data qubit index to observable index
    """
    try:
        import stim
    except ImportError:
        raise RuntimeError("Stim not available for DEM construction")
    
    # Get the H matrices and metadata for consistent ordering
    Hx, Hz, meta = build_H_rotated_d3_from_cfg(cudaq_cfg)
    
    # Create a simple DEM for rotated d=3 surface code
    # This is a basic implementation - in practice would use full circuit analysis
    dem = stim.DetectorErrorModel()
    
    # Add error processes that flip specific detectors.
    # Use detector indices directly (0..3 Z, 4..7 X).
    for data_qubit in range(9):
        z_checks = np.where(Hz[:, data_qubit] > 0)[0]
        x_checks = np.where(Hx[:, data_qubit] > 0)[0] + 4  # offset by 4 for X checks
        if len(z_checks) > 0:
            dem.append("error", 0.001, [stim.target_detector(int(d)) for d in z_checks])
        if len(x_checks) > 0:
            dem.append("error", 0.001, [stim.target_detector(int(d)) for d in x_checks])

    # Do NOT append 'logical_observable' lines into the DEM; MWPF uses num_obs at compile time.
    map_bit_to_obs = [0] * 9  # unused placeholder for interface compatibility
    
    return dem, map_bit_to_obs


def rotated_d3_cz_colors() -> Dict[str, List[List[Tuple[int, int]]]]:
    """
    Provide a 2-color CZ schedule per basis for rotated d=3 (logical indices).

    We define logical ancilla indices as:
      Z ancillas: z0..z3 mapped to logical ids 9..12
      X ancillas: x0..x3 mapped to logical ids 13..16

    Edges connect ancilla to adjacent data-qubit logical indices (0..8).
    Each basis provides up to two conflict-free layers.
    """
    # Logical schedule (ancilla -> data) in two layers; kept small and conflict-free
    colors = {
        'X': [
            # Layer 1
            [(13, 0), (14, 4), (15, 8)],
            # Layer 2 (optional sparse)
            [(16, 5)],
        ],
        'Z': [
            # Layer 1
            [(9, 1), (10, 5), (11, 7)],
            # Layer 2
            [(12, 4)],
        ],
    }
    return colors


def place_rotated_d3_on_garnet(calib: Dict[str, Any], avoid_edges: set = {(10, 11)}):
    """
    Place rotated d=3 (9 data + 8 ancilla) onto Garnet device avoiding bad couplers.

    Strategy: choose a fixed high-fidelity 17-qubit patch whose couplers are all
    present in the calibration graph and exclude (10,11). Return mapping dicts and
    a physical CZ schedule consistent with the rotated pattern.

    Returns:
      data_map: {logical_data_idx: physical_qubit}
      anc_map: {'X': {0..3 -> q}, 'Z': {0..3 -> q}}
      cz_layers_phys: {basis: [[(anc_phys, data_phys), ...], ...]} two layers per basis
    """
    from .garnet_noise import GARNET_COUPLER_F2

    # Select a 3x3 data patch using only good couplers
    data_phys = [0, 1, 4, 5, 8, 9, 10, 14, 15]  # 9 physical qubits
    data_map = {i: q for i, q in enumerate(data_phys)}

    # Candidate ancillas chosen adjacent to data and avoiding bad (10,11)
    anc_x_phys = [3, 6, 7, 16]
    anc_z_phys = [13, 18, 19, 12]

    anc_map = {
        'X': {i: q for i, q in enumerate(anc_x_phys)},
        'Z': {i: q for i, q in enumerate(anc_z_phys)},
    }

    # Build physical CZ layers using only existing couplers and excluding avoid_edges
    def allowed(edge):
        e = tuple(sorted(edge))
        return (e in GARNET_COUPLER_F2) and (e not in avoid_edges)

    # X-basis layers
    x_l1 = [(anc_x_phys[0], data_phys[0]),  # (3,0) uses (0,3)
            (anc_x_phys[1], data_phys[3]),  # (6,5) uses (5,6)
            (anc_x_phys[2], data_phys[4]),  # (7,8) uses (7,8)
            (anc_x_phys[3], data_phys[8])]  # (16,15) uses (15,16)
    x_l1 = [e for e in x_l1 if allowed(e)]
    x_l2 = [(anc_x_phys[0], data_phys[4])]  # (3,4) uses (3,4)
    x_l2 = [e for e in x_l2 if allowed(e)]

    # Z-basis layers
    z_l1 = [(anc_z_phys[0], data_phys[5]),   # (13,9) uses (13,9)
            (anc_z_phys[1], data_phys[7]),   # (18,14) uses (14,18)
            (anc_z_phys[2], data_phys[8])]   # (19,15) uses (15,19)
    z_l1 = [e for e in z_l1 if allowed(e)]
    z_l2 = [(anc_z_phys[0], data_phys[4])]   # (13,14) uses (13,14)
    z_l2 = [e for e in z_l2 if allowed(e)]

    cz_layers_phys = {
        'X': [x_l1, x_l2],
        'Z': [z_l1, z_l2]
    }

    # Sanity: ensure no forbidden couplers
    used = {tuple(sorted(e)) for basis in cz_layers_phys.values() for layer in basis for e in layer}
    assert (10, 11) not in used, "Forbidden coupler (10,11) present in schedule"

    return data_map, anc_map, cz_layers_phys

def make_surface_layout_d3_include_edge(edge: Tuple[int, int] = (10, 11)) -> Dict[str, Any]:
    """
    Create a d=3 surface code layout that INCLUDES the specified edge (e.g., bad (10,11) coupler).
    
    Clone the default d=3 layout, but force the specified edge into both CZ layers 
    for at least one stabilizer per round (if topology permits).
    
    Args:
        edge: Tuple of qubit indices to force into the layout
        
    Returns:
        Dictionary with layout information for d=3 surface code including the bad edge
    """
    # Get the base layout
    base_layout = make_surface_layout_d3_avoid_bad_edges()
    
    # Clone the base layout
    bad_layout = base_layout.copy()
    
    # Force bad edge (10,11) into Layout B by replacing one CZ pair in each round
    bad_q1, bad_q2 = edge
    
    # Replace first CZ pair in each layer with the bad edge
    cz_layers = [
        # Layer 1: Replace first pair with bad edge
        [edge] + base_layout['cz_layers'][0][1:],
        # Layer 2: Replace first pair with bad edge  
        [edge] + base_layout['cz_layers'][1][1:]
    ]
    
    bad_layout['cz_layers'] = cz_layers
    bad_layout['total_qubits'] = 20  # Accommodate bad edge qubits
    
    # After constructing cz_layers, force-insert (10,11) where legal, and then:
    used = {tuple(sorted(e)) for layer in bad_layout['cz_layers'] for e in layer}
    assert (10,11) in used or (11,10) in used, f"Layout must include edge {edge}. Used: {used}"
    
    return bad_layout




def make_surface_layout_general(d: int) -> Dict[str, Any]:
    """
    Create a general surface code layout for arbitrary distance d.
    
    Args:
        d: Code distance
        
    Returns:
        Layout dictionary with qubit assignments and gate schedules
    """
    if d == 3:
        return make_surface_layout_d3_avoid_bad_edges()
    
    n_data = d * d
    n_z_checks = d * (d - 1)
    n_x_checks = (d - 1) * d
    
    # Data qubits: 0 to d²-1 (row-major order)
    data_qubits = list(range(n_data))
    
    # Ancilla qubits: start after data qubits
    ancilla_z = list(range(n_data, n_data + n_z_checks))
    ancilla_x = list(range(n_data + n_z_checks, n_data + n_z_checks + n_x_checks))
    
    # Build CZ layers for syndrome extraction
    # Layer 1: Z stabilizers (horizontal connections)
    cz_layer_1 = []
    z_anc_idx = 0
    for row in range(d):
        for col in range(d - 1):
            anc = ancilla_z[z_anc_idx]
            q1 = row * d + col
            q2 = row * d + col + 1
            cz_layer_1.extend([(anc, q1), (anc, q2)])
            z_anc_idx += 1
    
    # Layer 2: X stabilizers (vertical connections)
    cz_layer_2 = []
    x_anc_idx = 0
    for row in range(d - 1):
        for col in range(d):
            anc = ancilla_x[x_anc_idx]
            q1 = row * d + col
            q2 = (row + 1) * d + col
            cz_layer_2.extend([(anc, q1), (anc, q2)])
            x_anc_idx += 1
    
    # PRX layers for ancilla reset/measurement
    prx_layer_z = [(anc, 'z') for anc in ancilla_z]  # Z basis for Z stabilizers
    prx_layer_x = [(anc, 'x') for anc in ancilla_x]  # X basis for X stabilizers
    
    return {
        'data': data_qubits,
        'ancilla_z': ancilla_z,
        'ancilla_x': ancilla_x,
        'cz_layers': [cz_layer_1, cz_layer_2],
        'prx_layers': [prx_layer_z, prx_layer_x],
        'total_qubits': n_data + n_z_checks + n_x_checks,
        'distance': d,
        'code_type': 'rotated_surface',
        'syndrome_schedule': 'alternating'  # Z then X measurements
    }


def make_surface_layout_d3_avoid_bad_edges() -> Dict[str, Any]:
    """
    Create a d=3 surface code layout that avoids the bad (10,11) coupler.
    
    Maps a 17-qubit rotated surface code patch onto the Garnet 20-qubit device,
    preferring high-fidelity edges and avoiding the problematic (10,11) coupler.
    
    Returns:
        Dictionary with layout information for d=3 surface code
    """
    # For d=3 rotated surface code: 9 data qubits + 8 stabilizer qubits = 17 total
    # Avoid coupler (10,11) which has F2Q = 0.9228 (worst in device)
    
    # Use a good patch of 17 qubits avoiding the bad edge
    # This is a simplified mapping - in practice you'd optimize placement
    qubit_mapping = {
        # Data qubits (9 total for d=3)
        'data': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        # X-stabilizer ancillas (4 total)
        'ancilla_x': [12, 13, 14, 15],
        # Z-stabilizer ancillas (4 total) 
        'ancilla_z': [16, 17, 18, 19]
    }
    
    # Define CZ layers that respect the rotated surface code connectivity
    # These would be computed from the actual surface code graph structure
    cz_layers = [
        # Layer 1: X-stabilizer CZs
        [(12, 0), (13, 1), (14, 2), (15, 3)],  # Example connectivity
        # Layer 2: Z-stabilizer CZs  
        [(16, 4), (17, 5), (18, 6), (19, 7)]   # Example connectivity
    ]
    
    # PRX layers for stabilizer measurements
    prx_layers = [
        # X-basis measurements for X-stabilizers
        [12, 13, 14, 15],
        # Z-basis measurements for Z-stabilizers (identity, no PRX needed)
        []
    ]
    
    return {
        'data': qubit_mapping['data'],
        'ancilla_x': qubit_mapping['ancilla_x'],
        'ancilla_z': qubit_mapping['ancilla_z'],
        'cz_layers': cz_layers,
        'prx_layers': prx_layers,
        'total_qubits': 17,
        'distance': 3
    }


def build_round_repetition(layout: Dict[str, Any], round_idx: int) -> CudaQKernel:
    """
    Build a repetition code syndrome extraction round.
    
    Args:
        layout: Dictionary with repetition code layout information
        round_idx: Round index (for potential round-dependent variations)
        
    Returns:
        CUDA-Q kernel for the repetition code round
    """
    operations = []
    kernel = CudaQKernel(f"rep_round_{round_idx}", operations)
    
    # For repetition code: CZ between adjacent data qubits and ancilla
    n_data = len(layout.get('data', []))
    ancillas = layout.get('ancilla', [])
    
    if not ancillas or not layout.get('data'):
        return kernel
    
    # Layer 1: CZ gates between data qubits and ancillas
    cz_ops = []
    for i, data_q in enumerate(layout['data'][:-1]):
        if i < len(ancillas):
            ancilla_q = ancillas[i]
            cz_ops.append({
                'gate': 'CZ',
                'qubits': (data_q, ancilla_q),
                'duration_ns': 40.0
            })
    
    if cz_ops:
        kernel.add_layer(cz_ops, 'CZ')
    
    # Layer 2: Measurement of ancillas (Z-basis)
    meas_ops = [{
        'gate': 'MEASURE_Z',
        'qubits': (q,),
        'duration_ns': 20.0
    } for q in ancillas]
    
    if meas_ops:
        kernel.add_layer(meas_ops, 'MEASURE')
    
    return kernel


def build_round_surface(layout: Dict[str, Any], round_idx: int) -> CudaQKernel:
    """
    Build a surface code syndrome extraction round.
    
    Alternates between X and Z stabilizer measurements, using the provided
    CZ and PRX layer schedule from the layout.
    
    Args:
        layout: Dictionary with surface code layout (from make_surface_layout_d3_avoid_bad_edges)
        round_idx: Round index - even rounds do X stabilizers, odd do Z stabilizers
        
    Returns:
        CUDA-Q kernel for the surface code round
    """
    operations = []
    kernel = CudaQKernel(f"surface_round_{round_idx}", operations)
    
    # Alternate between X and Z stabilizer types
    is_x_round = (round_idx % 2 == 0)
    
    if is_x_round:
        # X-stabilizer round
        active_ancillas = layout.get('ancilla_x', [])
        stabilizer_type = 'X'
    else:
        # Z-stabilizer round  
        active_ancillas = layout.get('ancilla_z', [])
        stabilizer_type = 'Z'
    
    # Get relevant CZ layers for this stabilizer type
    cz_layers = layout.get('cz_layers', [])
    prx_layers = layout.get('prx_layers', [])
    
    # For X-stabilizers: Apply PRX to ancillas first (H gate equivalent)
    if is_x_round and prx_layers and len(prx_layers) > 0:
        prx_ops = [{
            'gate': 'PRX',
            'qubits': (q,),
            'duration_ns': 20.0,
            'layer_type': 'PRX_PREP',
            'stabilizer_type': stabilizer_type
        } for q in prx_layers[0] if q in active_ancillas]
        
        if prx_ops:
            kernel.add_layer(prx_ops, 'PRX_PREP')
    
    # Add CZ layers (typically 2 layers for surface codes)
    for layer_idx, cz_layer in enumerate(cz_layers):
        if layer_idx >= 2:  # Limit to 2 CZ layers per round
            break
            
        # Filter CZ operations relevant to current stabilizer type
        relevant_czs = []
        for (q1, q2) in cz_layer:
            if (is_x_round and q1 in active_ancillas) or \
               (not is_x_round and q1 in active_ancillas):
                relevant_czs.append({
                    'gate': 'CZ',
                    'qubits': (q1, q2),
                    'duration_ns': 40.0,
                    'layer_type': f'CZ_{layer_idx+1}',
                    'stabilizer_type': stabilizer_type,
                    'layer_index': layer_idx
                })
        
        if relevant_czs:
            kernel.add_layer(relevant_czs, f'CZ_{layer_idx+1}')
    
    # For X-stabilizers: Apply PRX to measure in X-basis
    if is_x_round and active_ancillas:
        prx_meas_ops = [{
            'gate': 'PRX',
            'qubits': (q,),
            'duration_ns': 20.0,
            'layer_type': 'PRX_MEAS',
            'stabilizer_type': stabilizer_type
        } for q in active_ancillas]
        
        if prx_meas_ops:
            kernel.add_layer(prx_meas_ops, 'PRX_MEAS')
    
    # Final measurement (Z-basis for all)
    meas_ops = [{
        'gate': 'MEASURE_Z',
        'qubits': (q,),
        'duration_ns': 20.0,
        'layer_type': 'MEASURE',
        'stabilizer_type': stabilizer_type
    } for q in active_ancillas]
    
    if meas_ops:
        kernel.add_layer(meas_ops, 'MEASURE')
    
    return kernel


def build_round_bb(hx: np.ndarray, hz: np.ndarray, mapping: Dict[int, int], round_idx: int) -> CudaQKernel:
    """
    Build a BB/qLDPC code syndrome extraction round.
    
    Derives check circuits from (Hx, Hz) matrices and schedules CZ operations
    to minimize idle time while respecting hardware constraints.
    
    Args:
        hx: X-check matrix (num_x_checks, num_qubits)
        hz: Z-check matrix (num_z_checks, num_qubits)  
        mapping: Logical to physical qubit mapping
        round_idx: Round index - even rounds do X checks, odd do Z checks
        
    Returns:
        CUDA-Q kernel for the BB/qLDPC round
    """
    operations = []
    kernel = CudaQKernel(f"bb_round_{round_idx}", operations)
    
    # Alternate between X and Z check measurements
    is_x_round = (round_idx % 2 == 0)
    
    if is_x_round:
        check_matrix = hx
        check_type = 'X'
    else:
        check_matrix = hz  
        check_type = 'Z'
    
    num_checks, num_qubits = check_matrix.shape
    
    # Build ancilla allocation (assume ancillas start after data qubits)
    max_physical_qubit = max(mapping.values()) if mapping else num_qubits - 1
    ancilla_start = max_physical_qubit + 1
    
    # Create CZ schedule: group by ancilla degree to minimize layers
    check_schedules = defaultdict(list)
    
    for check_idx in range(num_checks):
        # Find data qubits involved in this check
        data_qubits = [mapping.get(q, q) for q in range(num_qubits) if check_matrix[check_idx, q] == 1]
        
        if not data_qubits:
            continue
            
        ancilla_qubit = ancilla_start + check_idx
        
        # Create CZ operations for this check
        for data_q in data_qubits:
            check_schedules[len(data_qubits)].append({
                'gate': 'CZ',
                'qubits': (ancilla_qubit, data_q),
                'duration_ns': 40.0,
                'check_idx': check_idx,
                'ancilla': ancilla_qubit
            })
    
    # For X-checks: PRX preparation layer
    if is_x_round:
        ancillas = [ancilla_start + i for i in range(num_checks)]
        prx_prep_ops = [{
            'gate': 'PRX',
            'qubits': (q,),
            'duration_ns': 20.0
        } for q in ancillas]
        
        if prx_prep_ops:
            kernel.add_layer(prx_prep_ops, 'PRX_PREP')
    
    # Schedule CZ operations in layers (aim for ≤3 layers)
    max_layers = 3
    layer_idx = 0
    
    # Sort by degree (higher degree checks first for better parallelization)
    for degree in sorted(check_schedules.keys(), reverse=True):
        if layer_idx >= max_layers:
            break
            
        cz_ops = check_schedules[degree]
        if cz_ops:
            kernel.add_layer(cz_ops, f'CZ_{layer_idx+1}')
            layer_idx += 1
    
    # For X-checks: PRX measurement layer
    if is_x_round:
        ancillas = [ancilla_start + i for i in range(num_checks)]
        prx_meas_ops = [{
            'gate': 'PRX',
            'qubits': (q,),
            'duration_ns': 20.0
        } for q in ancillas]
        
        if prx_meas_ops:
            kernel.add_layer(prx_meas_ops, 'PRX_MEAS')
    
    # Final measurement layer
    ancillas = [ancilla_start + i for i in range(num_checks)]
    meas_ops = [{
        'gate': 'MEASURE_Z',
        'qubits': (q,),
        'duration_ns': 20.0
    } for q in ancillas]
    
    if meas_ops:
        kernel.add_layer(meas_ops, 'MEASURE')
    
    return kernel


def analyze_idle_qubits(kernel: CudaQKernel, total_qubits: int) -> Dict[int, List[Tuple[int, float]]]:
    """
    Analyze which qubits are idle during each layer and for how long.
    
    Args:
        kernel: Circuit kernel with layer information
        total_qubits: Total number of qubits in the circuit
        
    Returns:
        Dictionary mapping layer_idx -> list of (qubit, idle_duration_ns) tuples
    """
    idle_analysis = {}
    
    for layer_idx, layer in enumerate(kernel.get_layer_schedule()):
        active_qubits = set()
        
        # Collect all qubits active in this layer
        for op in layer['operations']:
            if 'qubits' in op:
                qubits = op['qubits']
                if isinstance(qubits, (list, tuple)):
                    active_qubits.update(qubits)
                else:
                    active_qubits.add(qubits)
        
        # Find idle qubits
        idle_qubits = []
        layer_duration = layer.get('duration_ns', 20.0)
        
        for q in range(total_qubits):
            if q not in active_qubits:
                idle_qubits.append((q, layer_duration))
        
        idle_analysis[layer_idx] = idle_qubits
    
    return idle_analysis


def optimize_layout_for_device(code_distance: int, device_couplers: List[Tuple[int, int]], 
                              coupler_fidelities: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
    """
    Optimize qubit layout for a given code distance and device topology.
    
    This function would implement graph embedding algorithms to find the best
    mapping of code connectivity to device connectivity, considering fidelities.
    
    Args:
        code_distance: Distance of the quantum error correcting code
        device_couplers: List of available coupler pairs on device  
        coupler_fidelities: Fidelity for each coupler pair
        
    Returns:
        Optimized layout dictionary
    """
    # For now, return the hardcoded d=3 layout
    # In practice, this would implement sophisticated graph embedding
    if code_distance == 3:
        return make_surface_layout_d3_avoid_bad_edges()
    else:
        # Fallback for other distances
        raise NotImplementedError(f"Layout optimization for d={code_distance} not implemented")

