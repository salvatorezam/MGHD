#!/bin/bash
source /u/home/kulp/miniconda3/etc/profile.d/conda.sh
conda activate mlqec-env
cd /u/home/kulp/MGHD
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=29517
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Tunables for throughput (per-process workers; total workers = this * nproc_per_node)
WORKERS=20
BATCH=768
PREFETCH=12
torchrun --nproc_per_node=2 mghd/cli/train.py \
    --online \
    --sampler stim \
    --online-rl \
    --family surface \
    --distance 31 \
    --p-curriculum "0.013,0.011,0.009,0.007,0.005,0.003,0.001" \
    --epochs-per-p 20 \
    --teacher-mix "mwpm=1.0" \
    --qpu-profile "/u/home/kulp/MGHD/mghd/qpu/profiles/iqm_garnet_example.json" \
    --context-source "none" \
    --erasure-frac 0.05 \
    --shots-per-epoch 32768 \
    --epochs 1000 \
    --batch ${BATCH} \
    --workers ${WORKERS} \
    --prefetch-factor ${PREFETCH} \
    --progress-prints 50 \
    --amp bf16 \
    --save "/u/home/kulp/MGHD/data/results_NATURE_FINAL_NOV20_2025/20251120-223045_surface_d31_iqm_garnet_example" \
    --resume "/u/home/kulp/MGHD/data/results_NATURE_FINAL_NOV20_2025/20251120-223045_surface_d31_iqm_garnet_example/last.pt"
