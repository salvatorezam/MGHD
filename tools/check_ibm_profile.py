#!/usr/bin/env python3
"""
Test script to verify IBM Heron profile structure and coupling map readiness.
"""

import json
from pathlib import Path


def check_profile_status():
    """Check the current state of ibm_heron_r3.json profile."""
    profile_path = Path(__file__).parent.parent / "mghd/qpu/profiles/ibm_heron_r3.json"
    
    if not profile_path.exists():
        print("❌ Profile not found at", profile_path)
        return
    
    print(f"📁 Profile: {profile_path.relative_to(Path.cwd())}\n")
    
    # Load JSON with comments (strip // comments)
    with open(profile_path) as f:
        content = f.read()
    
    # Remove // comments
    lines = []
    for line in content.split('\n'):
        # Find // outside of strings
        in_string = False
        cleaned = []
        i = 0
        while i < len(line):
            if line[i] == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
                cleaned.append(line[i])
            elif line[i:i+2] == '//' and not in_string:
                break  # Rest of line is comment
            else:
                cleaned.append(line[i])
            i += 1
        lines.append(''.join(cleaned))
    
    profile = json.loads('\n'.join(lines))
    
    # Basic structure
    print("✅ Profile Structure:")
    print(f"   Name: {profile.get('name')}")
    print(f"   Qubits: {profile.get('n_qubits')}")
    
    # Coupling map status
    coupling = profile.get("coupling", [])
    if len(coupling) == 0:
        print(f"\n⚠️  Coupling Map: EMPTY (needs to be filled)")
        print(f"   Expected: ~360 edges for heavy-hex topology")
        print(f"\n   To fill, run:")
        print(f"     python tools/fetch_ibm_coupling.py --backend ibm_brisbane")
    else:
        print(f"\n✅ Coupling Map: {len(coupling)} edges")
        print(f"   Sample edges: {coupling[:5]}")
        
        # Verify edge format
        valid = all(isinstance(e, list) and len(e) == 2 for e in coupling[:10])
        if valid:
            print(f"   Format: ✅ Valid [[u,v], ...] format")
        else:
            print(f"   Format: ❌ Invalid edge format")
    
    # Gate errors
    gate_err = profile.get("gate_error", {})
    print(f"\n✅ Gate Errors:")
    print(f"   1Q (sx): {gate_err.get('p_1q', {}).get('sx', 0)*100:.3f}%")
    print(f"   2Q (cz): {gate_err.get('p_2q', {}).get('cz', {}).get('default', 0)*100:.3f}%")
    print(f"   Readout: {gate_err.get('p_meas', 0)*100:.3f}%")
    
    # Coherence
    t1_vals = gate_err.get("t1_us", {})
    t2_vals = gate_err.get("t2_us", {})
    if t1_vals and t2_vals:
        t1 = float(list(t1_vals.values())[0])
        t2 = float(list(t2_vals.values())[0])
        print(f"   T1: {t1} μs")
        print(f"   T2: {t2} μs")
    
    # Meta
    meta = profile.get("meta", {})
    print(f"\n✅ Metadata:")
    print(f"   Basis gates: {meta.get('basis_gates')}")
    print(f"   Source: {meta.get('source')}")
    
    # Readiness check
    print(f"\n{'='*60}")
    if len(coupling) > 0:
        print("✅ READY: Profile has coupling map and can be used for training")
    else:
        print("⚠️  NOT READY: Need to fetch coupling map from IBM Quantum")
        print("\n   Setup steps:")
        print("   1. pip install qiskit-ibm-runtime")
        print("   2. python tools/fetch_ibm_coupling.py --list-backends")
        print("   3. python tools/fetch_ibm_coupling.py --backend ibm_brisbane")


if __name__ == "__main__":
    check_profile_status()
