#!/bin/bash
# Generate offline training crops for Gross BB code
# Usage: bash scripts/generate_bb_crops.sh [num_shots]

set -e
cd "$(dirname "$0")/.."

source /u/home/kulp/miniconda3/etc/profile.d/conda.sh
conda activate mlqec-env

# Configuration
FAMILY="gross"
DISTANCE=12
SAVE_DIR="/u/home/kulp/MGHD-data/crops/gross_heron_$(date +%Y%m%d)"
SHOTS="${1:-100}"  # Default 100 shots for testing
P_VALUES="0.009,0.006,0.003,0.001,0.0007,0.0004"

echo "Generating ${SHOTS} crops for ${FAMILY} d=${DISTANCE}"
echo "p values: ${P_VALUES}"
echo "Save dir: ${SAVE_DIR}"

mkdir -p ${SAVE_DIR}

# Export config for Python
export FAMILY DISTANCE SHOTS P_VALUES SAVE_DIR

# Generate crops using CudaQSampler
python3 << 'PYEOF'
import os
import sys
import json
import numpy as np
from pathlib import Path
from mghd.codes.registry import get_code
from mghd.decoders.lsd_teacher import LSDTeacher
from mghd.core.core import pack_cluster
from mghd.qpu.adapters.garnet_adapter import split_components_for_side

# Config from environment
family = os.getenv('FAMILY', 'gross')
d = int(os.getenv('DISTANCE', '12'))
shots = int(os.getenv('SHOTS', '100'))
p_values = [float(x) for x in os.getenv('P_VALUES', '0.009,0.006,0.003').split(',')]
save_dir = Path(os.getenv('SAVE_DIR', '/tmp/gross_crops'))
save_dir.mkdir(parents=True, exist_ok=True)

# Get code
print(f'Loading {family} code at distance {d}...')
code = get_code(family, distance=d)
print(f'Code: [[{code.n},{code.k},{code.distance}]]')

# Initialize teacher
lsd_teacher = LSDTeacher(code.Hx, code.Hz)

# Pad limits (auto-scaled for d=12)
N_max = 512
E_max = 4096
S_max = 512

# Generate crops for each p value
rng = np.random.default_rng(42)
manifest = []

for p in p_values:
    print(f'\\nGenerating crops for p={p:.6f}...')
    p_crops = []
    
    for shot_idx in range(shots):
        if shot_idx % 100 == 0:
            print(f'  Shot {shot_idx}/{shots}...')
        
        seed = int(rng.integers(0, 2**31 - 1))
        rng_shot = np.random.default_rng(seed)

        # Simple phenomenological noise on the BB code:
        # X errors trigger Z checks (Hz), Z errors trigger X checks (Hx).
        err_x = rng_shot.random(code.n) < p
        err_z = rng_shot.random(code.n) < p
        synZ = (code.Hz @ err_x.astype(np.uint8)) % 2
        synX = (code.Hx @ err_z.astype(np.uint8)) % 2
        
        # Generate dummy coordinates (BB codes don't have geometric layout)
        n_q = code.n
        n_z = code.Hz.shape[0]
        n_x = code.Hx.shape[0]
        coords_q = np.column_stack([np.arange(n_q), np.zeros(n_q)]).astype(np.int32)
        coords_c = np.column_stack([
            np.concatenate([np.arange(n_z), np.arange(n_x)]),
            np.concatenate([np.zeros(n_z), np.ones(n_x)])
        ]).astype(np.float32)
        
        sample = {
            'Hx': code.Hx,
            'Hz': code.Hz,
            'synZ': synZ,
            'synX': synX,
            'coords_q': coords_q,
            'coords_c': coords_c,
        }
        
        # Split into components and generate crops
        for side in ('Z', 'X'):
            comps = split_components_for_side(
                side=side,
                Hx=code.Hx,
                Hz=code.Hz,
                synZ=synZ,
                synX=synX,
                coords_q=coords_q,
                coords_c=coords_c,
            )
            
            for comp in comps:
                H_sub = comp['H_sub']
                synd_bits = comp['synd_bits']
                qubit_indices = comp['qubit_indices']
                
                # Supervision: use ground-truth errors for this side
                if side == 'Z':
                    bits_global = err_x.astype(np.uint8)
                else:
                    bits_global = err_z.astype(np.uint8)
                bits_local = bits_global[qubit_indices].astype(np.uint8)
                
                # Pack crop
                try:
                    packed = pack_cluster(
                        H_sub=H_sub,
                        xy_qubit=comp['xy_qubit'],
                        xy_check=comp['xy_check'],
                        synd_Z_then_X_bits=synd_bits,
                        k=int(comp['k']),
                        r=int(comp['r']),
                        bbox_xywh=tuple(comp['bbox_xywh']),
                        kappa_stats=comp.get('kappa_stats', {}),
                        y_bits_local=bits_local,
                        side=side,
                        d=d,
                        p=p,
                        seed=seed,
                        N_max=N_max,
                        E_max=E_max,
                        S_max=S_max,
                    )
                    
                    p_crops.append(packed)
                except Exception as e:
                    print(f'Warning: Failed to pack crop: {e}')
                    continue
    
    # Save shard
    if len(p_crops) == 0:
        print(f'Warning: No crops generated for p={p}')
        continue
        
    shard_path = save_dir / f'{family}_d{d}_p{p:.6f}.shard'
    print(f'Saving {len(p_crops)} crops to {shard_path}...')
    
    # Save as pickle (compatible with CropShardDataset)
    import pickle
    with open(shard_path, 'wb') as f:
        pickle.dump(p_crops, f)
    
    manifest.append({
        'family': family,
        'distance': d,
        'p': p,
        'num_crops': len(p_crops),
        'path': str(shard_path.relative_to(save_dir)),
    })

# Save manifest
manifest_path = save_dir / 'manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'\nDone! Generated crops saved to {save_dir}')
total = sum(m["num_crops"] for m in manifest)
print(f'Total crops: {total}')
print(f'Manifest saved to {manifest_path}')
PYEOF

echo ""
echo "Crop generation complete!"
echo "To train with these crops:"
echo "  python -m mghd.cli.train --data-root ${SAVE_DIR} --save checkpoints/gross_offline --epochs 30 --batch 512"
