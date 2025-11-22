#!/bin/bash
# Launch a fresh online training run capped at distance 15.
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

torchrun --nproc_per_node=2 mghd/cli/train.py \
  --online \
  --sampler stim \
  --online-rl \
  --family surface \
  --distance 15 \
  --distance-curriculum "3,5,7,9,11,13,15" \
  --p-curriculum "0.011,0.009,0.007,0.005,0.003,0.001" \
  --epochs-per-p 20 \
  --teacher-mix "mwpm=1.0" \
  --qpu-profile "/u/home/kulp/MGHD/mghd/qpu/profiles/iqm_garnet_example.json" \
  --context-source "none" \
  --erasure-frac 0.05 \
  --shots-per-epoch 32768 \
  --epochs 200 \
  --batch ${BATCH} \
  --workers ${WORKERS} \
  --prefetch-factor ${PREFETCH} \
  --progress-prints 50 \
  --amp bf16 \
  --save "/u/home/kulp/MGHD/data/results_d15_run" \
  --resume "/u/home/kulp/MGHD/data/results_d15_run/last.pt"
