#!/bin/bash
# Fast validation run for distances 3 and 5 with high-fidelity settings.
# Uses Stim circuit-level noise, Erasure (5%), Online RL, and MWPM/LSD teacher mix.
# Batch size 1536 (safe for d=5), 100k shots per epoch for high statistics.

source /u/home/kulp/miniconda3/bin/activate mlqec-env
export MGHD_SAMPLER=stim

# Save directory
SAVE_ROOT="data/results_validation_d3_5"

# Run training
# We use a p-curriculum starting high (0.012) and going lower.
# 100k shots per epoch ensures very low variance in gradients and validation metrics.
CUDA_VISIBLE_DEVICES=0 python -m mghd.cli.train \
    --online \
    --sampler stim \
    --teacher-mix "mwpm=0.8,lsd=0.2" \
    --family surface \
    --distance 5 \
    --distance-curriculum "3,5" \
    --p-curriculum "0.012,0.010,0.008,0.005" \
    --epochs-per-p 5 \
    --epochs 20 \
    --batch 1536 \
    --shots-per-epoch 32000 \
    --workers 64 \
    --online-rl \
    --erasure-frac 0.05 \
    --qpu-profile "/u/home/kulp/MGHD/mghd/qpu/profiles/iqm_garnet_example.json" \
    --save-root "$SAVE_ROOT" \
    --save "{auto}" \
    --progress-prints 10
