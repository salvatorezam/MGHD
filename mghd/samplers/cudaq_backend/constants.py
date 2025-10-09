"""
Canonical constants for MGHD quantum error correction.

Defines standard parameters and ordering conventions used across the project.
"""

# =============================================================================
# SYNDROME ORDERING CONVENTIONS
# =============================================================================

# Canonical syndrome order: Z checks first, then X checks
SYNDROME_ORDER = 'Z_first_then_X'

# =============================================================================
# ROTATED D=3 SURFACE CODE PARAMETERS
# =============================================================================

# Rotated d=3 surface code canonical parameters
ROTATED_D3 = {
    'N_bits': 9,           # Number of data qubits (3x3 grid)
    'N_syn': 8,            # Number of syndrome bits (4 Z + 4 X checks)
    'data_qubit_order': list(range(9)),  # Row-major order: 0,1,2,3,4,5,6,7,8
    'surface_layout': 'rotated',
    'distance': 3,
    'syndrome_order': SYNDROME_ORDER
}

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Expected matrix shapes for rotated d=3
ROTATED_D3_MATRIX_SHAPES = {
    'Hx_shape': (4, 9),    # 4 X-checks, 9 data qubits
    'Hz_shape': (4, 9),    # 4 Z-checks, 9 data qubits
    'H_shape': (8, 9)      # Combined H matrix
}

# =============================================================================
# PACKING PARAMETERS
# =============================================================================

# Default parameters for dataset generation
DEFAULT_PACK_PARAMS = {
    'B': 8192,             # Default batch size
    'p': 0.05,             # Default error rate
    'rng_seed': 42,        # Fixed RNG seed for reproducibility
    'timeout_ms': 50       # Default timeout for teachers
}
