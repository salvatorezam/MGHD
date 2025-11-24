#!/usr/bin/env bash
set -euo pipefail

# Launch a Heron-style circuit-level surface-code curriculum run inside tmux.
# Distances: 5,7,9,11; sampler: cudaq; teacher mix: MWPF (0.6) + LSD (0.4).

SESSION_NAME="heron_surface_d11"

# Bail out if the session already exists.
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

# Create new session in detached mode.
tmux new-session -d -s "${SESSION_NAME}"

# Activate environment and move to repo root.
tmux send-keys -t "${SESSION_NAME}" "source /u/home/kulp/miniconda3/bin/activate mlqec-env" C-m
tmux send-keys -t "${SESSION_NAME}" "cd /u/home/kulp/MGHD" C-m
tmux send-keys -t "${SESSION_NAME}" 'export PYTHONPATH="/u/home/kulp/MGHD:${PYTHONPATH:-}"' C-m

# Tunables (override via env before running this script).
# NvQldpcTeacher is GPU-heavy; default to workers=0 to avoid
# replicating decoders across many DataLoader processes.
WORKERS=${WORKERS:-0}
PREFETCH=${PREFETCH:-8}
BATCH=${BATCH:-2048}
SAVE_DIR=${SAVE_DIR:-/u/home/kulp/MGHD/data/results_HeronSurf_d11_$(date +%y%m%d_%H%M%S)}

# Single-GPU launch (no torchrun/DDP).
tmux send-keys -t "${SESSION_NAME}" \
  "CUDA_VISIBLE_DEVICES=0 python -m mghd.cli.train \
    --online \
    --sampler cudaq \
    --family surface \
    --distance-curriculum 5,7,9,11 \
    --p-curriculum 0.008,0.007,0.006,0.005,0.004,0.003,0.002 \
    --epochs-per-p 25 \
    --epochs 175 \
    --shots-per-epoch 50000 \
    --teacher-mix nvqldpc=1.0 \
    --qpu-profile /u/home/kulp/MGHD/mghd/qpu/profiles/ibm_heron_r3.json \
    --context-source cudaq \
    --erasure-frac 0.01 \
    --batch ${BATCH} \
    --workers ${WORKERS} \
    --prefetch-factor ${PREFETCH} \
    --progress-prints 50 \
    --amp bf16 \
    --save ${SAVE_DIR}" C-m

echo "----------------------------------------------------------------"
echo "Heron surface-code training started in tmux session: '${SESSION_NAME}'"
echo "The job is running in the background and will survive disconnects."
echo "----------------------------------------------------------------"
echo "To view progress, run:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "To detach again (leave it running), press: Ctrl+B, then D"
echo "----------------------------------------------------------------"
