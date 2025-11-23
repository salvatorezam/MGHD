#!/usr/bin/env bash
set -euo pipefail

# Launch the standard depolarizing-noise benchmark training in a tmux session.
SESSION_NAME="depnoise_benchmark"

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

# Create new session in detached mode
tmux new-session -d -s "${SESSION_NAME}"

# Activate environment and run training from repo root
tmux send-keys -t "${SESSION_NAME}" "source /u/home/kulp/miniconda3/bin/activate mlqec-env" C-m
tmux send-keys -t "${SESSION_NAME}" "cd /u/home/kulp/MGHD" C-m
tmux send-keys -t "${SESSION_NAME}" 'export PYTHONPATH="/u/home/kulp/MGHD:${PYTHONPATH:-}"' C-m

# Standard Benchmark Run:
# - Sampler: Stim (standard circuit-level depolarizing noise)
# - Teachers: MWPM (0.7) + LSD (0.3)
# - Distance curriculum: 3,5,7
# - P curriculum: 0.007 -> 0.001
tmux send-keys -t "${SESSION_NAME}" \
  "MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 mghd/cli/train.py \
    --online \
    --sampler stim \
    --family surface \
    --distance-curriculum 3,5,7 \
    --p-curriculum 0.007,0.006,0.005,0.004,0.003,0.002,0.001 \
    --epochs-per-p 30 \
    --epochs 210 \
    --shots-per-epoch 32768 \
    --teacher-mix mwpm=0.7,lsd=0.3 \
    --save data/results_DepNoiseOnly_24112025 \
    --erasure-frac 0.0" C-m

echo "----------------------------------------------------------------"
echo "Depolarizing-noise benchmark training started in tmux session: '${SESSION_NAME}'"
echo "The job is running in the background and will survive disconnects."
echo "----------------------------------------------------------------"
echo "To view progress, run:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "To detach again (leave it running), press: Ctrl+B, then D"
echo "----------------------------------------------------------------"
