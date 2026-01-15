#!/bin/bash
# Continue MGHD training with extended distance curriculum
# Loads from best.pt and trains on d=3,5,7,9,11

set -e
cd /u/home/kulp/MGHD

OUTPUT_DIR="data/results_surface_mwpm_circuit_extended_$(date +%Y%m%d_%H%M%S)"
RESUME_FROM="data/results_surface_mwpm_circuit_20251205_130649/best.pt"

echo "=============================================="
echo "MGHD Training: Extended Distance Curriculum"
echo "=============================================="
echo "Resuming from: $RESUME_FROM"
echo "Distance curriculum: 3,5,7,9,11"
echo "P curriculum: 0.005 -> 0.0005"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

torchrun --nproc_per_node=2 mghd/cli/train.py \
    --online \
    --sampler stim \
    --family surface \
    --distance-curriculum 3,5,7,9,11 \
    --p-curriculum 0.005,0.003,0.002,0.001,0.0005 \
    --epochs-per-p 30 \
    --epochs 150 \
    --teacher-mix "mwpm=1.0" \
    --erasure-frac 0.0 \
    --shots-per-epoch 32768 \
    --batch 512 \
    --workers 32 \
    --prefetch-factor 16 \
    --progress-prints 50 \
    --amp bf16 \
    --resume "$RESUME_FROM" \
    --save "$OUTPUT_DIR"
