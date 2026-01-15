#!/bin/bash
# Train MGHD with MWPM teacher on Stim circuit-level noise.
# This uses REAL circuit-level noise from Stim, not phenomenological approximation.
#
# Key settings:
# 1. --sampler stim: Uses Stim's circuit-level noise model with proper temporal correlations
# 2. --teacher-mix "mwpm=1.0": MWPM teacher (optimal for circuit-level decoding)
# 3. Distance curriculum: 3,5,7 (smaller distances for faster training with circuit-level)

set -euo pipefail

# Optional tmux wrapper
if [ -z "${TMUX:-}" ] && [ -z "${NO_TMUX:-}" ]; then
  SCRIPT_PATH="$(realpath "$0")"
  SESSION_NAME="${SESSION_NAME:-mghd_mwpm_circuit}"
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
export MASTER_PORT=${MASTER_PORT:-29519}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Tunables
WORKERS=${WORKERS:-32}
BATCH=${BATCH:-512}  # Slightly smaller for circuit-level (more complex samples)
PREFETCH=${PREFETCH:-16}

# Output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="/u/home/kulp/MGHD/data/results_surface_mwpm_circuit_${TIMESTAMP}"

echo "=============================================="
echo "MGHD Training: Surface Code + MWPM Teacher"
echo "=============================================="
echo "Noise model: Stim CIRCUIT-LEVEL (not phenomenological!)"
echo "Distance curriculum: 3,5,7"
echo "P curriculum: 0.005 -> 0.001"
echo "Teacher: MWPM (optimal for circuit-level)"
echo "Output: ${SAVE_DIR}"
echo "=============================================="

torchrun --nproc_per_node=2 mghd/cli/train.py \
  --online \
  --sampler stim \
  --family surface \
  --distance-curriculum "3,5,7" \
  --p-curriculum "0.005,0.003,0.002,0.001" \
  --epochs-per-p 30 \
  --epochs 120 \
  --teacher-mix "mwpm=1.0" \
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
  --sampler stim \
  --distances "3,5,7" \
  --p-values "0.001,0.002,0.003,0.004,0.005" \
  --shots 10000 \
  --output "${SAVE_DIR}/evaluation_circuit.json"

echo ""
echo "Evaluation saved to ${SAVE_DIR}/evaluation_circuit.json"
echo "Done!"
