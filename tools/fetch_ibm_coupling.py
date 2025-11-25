#!/usr/bin/env python3
"""
Fetch IBM Heron R3 coupling map from Qiskit and update the profile JSON.

Usage:
    python tools/fetch_ibm_coupling.py [--backend ibm_brisbane]
    
Requires Qiskit and IBM Quantum account setup.
"""

import argparse
import json
from pathlib import Path


def fetch_coupling_map(backend_name: str = "ibm_brisbane") -> list[list[int]]:
    """Fetch coupling map from IBM Quantum backend via Qiskit."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("ERROR: qiskit-ibm-runtime not installed.")
        print("Install with: pip install qiskit-ibm-runtime")
        raise
    
    print(f"Fetching coupling map for backend: {backend_name}")
    
    # Initialize service (requires IBM Quantum account)
    service = QiskitRuntimeService()
    
    # Get backend
    backend = service.backend(backend_name)
    
    # Get configuration
    config = backend.configuration()
    
    # Extract coupling map
    coupling_map = config.coupling_map
    
    print(f"✅ Retrieved {len(coupling_map)} edges from {backend_name}")
    print(f"   Qubits: {config.n_qubits}")
    print(f"   Basis gates: {config.basis_gates}")
    
    # Verify it's a Heron processor (156 qubits, heavy-hex topology)
    if config.n_qubits != 156:
        print(f"WARNING: Expected 156 qubits for Heron, got {config.n_qubits}")
    
    return coupling_map


def update_profile_json(coupling_map: list[list[int]], profile_path: str):
    """Update the ibm_heron_r3.json profile with the coupling map."""
    path = Path(profile_path)
    
    if not path.exists():
        print(f"ERROR: Profile not found at {profile_path}")
        return
    
    # Read current profile
    with open(path, 'r') as f:
        content = f.read()
    
    # Parse JSON (allowing comments)
    # Find the coupling array and replace it
    import re
    
    # Pattern to match the coupling array (with or without comments)
    pattern = r'"coupling":\s*\[([^\]]*)\]'
    
    # Format coupling map as JSON
    coupling_str = "[\n    " + ",\n    ".join(
        f"[{edge[0]}, {edge[1]}]" for edge in coupling_map
    ) + "\n  ]"
    
    # Replace
    new_content = re.sub(
        pattern,
        f'"coupling": {coupling_str}',
        content,
        flags=re.DOTALL
    )
    
    # Write back
    with open(path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Updated {profile_path}")
    print(f"   Added {len(coupling_map)} coupling edges")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch IBM Heron coupling map and update profile JSON"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ibm_brisbane",
        help="IBM Quantum backend name (default: ibm_brisbane)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="mghd/qpu/profiles/ibm_heron_r3.json",
        help="Path to profile JSON (default: mghd/qpu/profiles/ibm_heron_r3.json)",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available IBM Quantum backends and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_backends:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            backends = service.backends()
            print("\nAvailable IBM Quantum Backends:")
            for backend in backends:
                config = backend.configuration()
                print(f"  - {backend.name}: {config.n_qubits} qubits, {len(config.coupling_map)} edges")
        except Exception as e:
            print(f"ERROR: {e}")
        return
    
    try:
        # Fetch coupling map
        coupling_map = fetch_coupling_map(args.backend)
        
        # Update profile
        update_profile_json(coupling_map, args.profile)
        
        print("\n✅ Done! Profile updated with heavy-hex coupling map.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have:")
        print("  1. Installed qiskit-ibm-runtime: pip install qiskit-ibm-runtime")
        print("  2. Saved your IBM Quantum token: QiskitRuntimeService.save_account(token='YOUR_TOKEN')")
        print("  3. Access to IBM Quantum backends (free tier or premium)")


if __name__ == "__main__":
    main()
