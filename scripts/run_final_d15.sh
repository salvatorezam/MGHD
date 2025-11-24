#!/bin/bash
# Final Training Run (d=3..15)
# Configuration:
# - Distances: 3,5,7,9,11,13,15
# - p-Curriculum: 0.012 .. 0.005 (8 steps)
# - Epochs: 30 per p (Total 240 epochs)
# - Shots: 32,768 per epoch
# - Batch: 1536
# - Workers: 64
# - Features: Erasure (5%) + Online RL (Garnet Profile)
# - Teachers: MWPM (80%) + LSD (20%) for better physical error handling

# Estimated Runtime: ~2-3 days (depending on defect density decrease)

source /u/home/kulp/miniconda3/bin/activate mlqec-env
export MGHD_SAMPLER=stim

CUDA_VISIBLE_DEVICES=0 python -m mghd.cli.train \
    --online \
    --sampler stim \
    --teacher-mix "mwpm=0.8,lsd=0.2" \
    --family surface \
    --distance 15 \
    --distance-curriculum "3,5,7,9,11,13,15" \
    --p-curriculum "0.012,0.011,0.010,0.009,0.008,0.007,0.006,0.005" \
    --epochs-per-p 30 \
    --batch 1536 \
    --shots-per-epoch 32768 \
    --workers 64 \
    --online-rl \
    --erasure-frac 0.05 \
    --qpu-profile "/u/home/kulp/MGHD/mghd/qpu/profiles/iqm_garnet_example.json" \
    --save-root "data/results_SMALL_D_NOV20_2025_FINAL" \
    --save "{auto}" \
    --progress-prints 10 \
    --lr 1e-4 \
    --d_model 192 \
    --d_state 80 \
    --n_iters 8
