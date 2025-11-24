#!/bin/bash
# Run curriculum training on distances 3-31 using Stim circuit-level noise.
# Optimized for H100 with batching and high worker count.

source /u/home/kulp/miniconda3/bin/activate mlqec-env
export MGHD_SAMPLER=stim

# Distances 3 to 31 (odd)
DISTANCES=$(seq -s, 3 2 31)

# Use GPU 0. To use GPU 1, change to CUDA_VISIBLE_DEVICES=1.
# "Nature-Level" Run Configuration
# - Distances: 3 to 31 (Curriculum)
# - p-Curriculum: 0.012 down to 0.004 (Threshold region)
# - Batch: 1024 (Safe H100 utilization, 2048 OOMs at d=31)
# - Shots: 32k/epoch (High statistics)
# - Epochs: 25 per p-value (Convergence)
# - Features: Erasure (5%) + Online RL (using IQM Garnet profile)

# Note: --p-curriculum must be comma-separated.

CUDA_VISIBLE_DEVICES=0 python -m mghd.cli.train \
    --online \
    --sampler stim \
    --teacher-mix "mwpm=1.0" \
    --family surface \
    --distance 31 \
    --distance-curriculum "3,5,7,9,11,13,15,17,19,21,23,25,27,29,31" \
    --p-curriculum "0.012,0.011,0.010,0.009,0.008,0.007,0.006,0.005,0.004" \
    --epochs-per-p 25 \
    --batch 1024 \
    --shots-per-epoch 32768 \
    --workers 64 \
    --online-rl \
    --qpu-profile "/u/home/kulp/MGHD/mghd/qpu/profiles/iqm_garnet_example.json" \
    --erasure-frac 0.05 \
    --save-root "data/results_FINAL_THRESHOLD_NOV20_2025" \
    --save "{auto}" \
    --progress-prints 10 \
    --lr 1e-4 \
    --d_model 192 \
    --d_state 80 \
    --n_iters 8
