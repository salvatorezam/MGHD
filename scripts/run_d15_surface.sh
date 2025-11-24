#!/bin/bash
# Launch a fresh online training run for depolarizing noise (Stim sampler).
# Adjust BATCH/WORKERS/PREFETCH below if you hit OOM or see dataloader stalls.

set -euo pipefail

# Optional tmux wrapper: set NO_TMUX=1 to run inline; SESSION_NAME to override.
if [ -z "${TMUX:-}" ] && [ -z "${NO_TMUX:-}" ]; then
  SCRIPT_PATH="$(realpath "$0")"
  SESSION_NAME="${SESSION_NAME:-d15_run}"
  tmux has-session -t "$SESSION_NAME" 2>/dev/null || tmux new-session -d -s "$SESSION_NAME"
  tmux send-keys -t "$SESSION_NAME" "NO_TMUX=1 bash \"$SCRIPT_PATH\"" C-m
  echo "Training started in tmux session '$SESSION_NAME'. Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

source /u/home/kulp/miniconda3/etc/profile.d/conda.sh
conda activate mlqec-env
cd /u/home/kulp/MGHD

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29517}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Tunables
WORKERS=${WORKERS:-40}
BATCH=${BATCH:-1152}
PREFETCH=${PREFETCH:-24}

# Standard depolarizing-noise benchmark:
# - Sampler: Stim (surface_code:rotated_memory_x)
# - Teachers: MWPM (0.7) + LSD (0.3)
# - Distance: 7 (set pads via --distance)
# - P-curriculum: 0.007 -> 0.001, 30 epochs each (total 210 epochs)

torchrun --nproc_per_node=2 mghd/cli/train.py \
  --sampler stim \
  --family surface \
  --distance 7 \
  --p-curriculum "0.007,0.006,0.005,0.004,0.003,0.002,0.001" \
  --epochs-per-p 30 \
  --epochs 210 \
  --teacher-mix "mwpm=0.7,lsd=0.3" \
  --erasure-frac 0.0 \
  --shots-per-epoch 32768 \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --prefetch-factor "${PREFETCH}" \
  --progress-prints 50 \
  --amp bf16 \
  --save "/u/home/kulp/MGHD/data/results_DepNoiseOnly_24112015"
