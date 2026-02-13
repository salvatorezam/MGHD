#!/bin/bash
# Train MGHD with distance curriculum and LSD teacher for phenomenological noise.
# Key fixes vs previous runs:
# 1. Use --distance-curriculum to train on multiple distances (model sees d=3,5,7,9,11)
# 2. Use LSD teacher only (consistent with Stim phenomenological noise)
# 3. Start with higher p, curriculum down to lower p

set -euo pipefail

# Optional tmux wrapper
if [ -z "${TMUX:-}" ] && [ -z "${NO_TMUX:-}" ]; then
  SCRIPT_PATH="$(realpath "$0")"
  SESSION_NAME="${SESSION_NAME:-mghd_lsd_curriculum}"
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
export MASTER_PORT=${MASTER_PORT:-29518}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Tunables - reduce batch for multi-distance to avoid OOM
WORKERS=${WORKERS:-32}
BATCH=${BATCH:-768}
PREFETCH=${PREFETCH:-16}

# Output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="/u/home/kulp/MGHD/data/results_surface_lsd_curriculum_${TIMESTAMP}"

echo "=============================================="
echo "MGHD Training: Surface Code + LSD Teacher"
echo "=============================================="
echo "Distance curriculum: 3,5,7,9,11"
echo "P curriculum: 0.008 -> 0.001"
echo "Teacher: LSD only (works with phenomenological noise)"
echo "Output: ${SAVE_DIR}"
echo "=============================================="

torchrun --nproc_per_node=2 mghd/cli/train.py \
  --online \
  --sampler stim \
  --family surface \
  --distance-curriculum "3,5,7,9,11" \
  --p-curriculum "0.008,0.006,0.004,0.002,0.001" \
  --epochs-per-p 25 \
  --epochs 125 \
  --teacher-mix "lsd=1.0" \
  --erasure-frac 0.0 \
  --shots-per-epoch 32768 \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --prefetch-factor "${PREFETCH}" \
  --progress-prints 50 \
  --amp bf16 \
  --save "${SAVE_DIR}"

echo ""
echo "Training complete. Now running evaluation..."
echo ""

# Run evaluation on the trained model
python scripts/evaluate_model.py \
  --checkpoint "${SAVE_DIR}/best.pt" \
  --sampler phenomenological \
  --distances "3,5,7,9,11" \
  --p-values "0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01" \
  --shots 10000 \
  --node-feat-dim 9 \
  --output "${SAVE_DIR}/evaluation_phenom.json"

echo ""
echo "Evaluation saved to ${SAVE_DIR}/evaluation_phenom.json"
echo "Done!"
