#!/usr/bin/env bash
set -euo pipefail

# Launch an online Garnet (CUDA-Q) benchmark training in a tmux session.
# - Sampler: cudaq (Garnet noise model via qpu_profile)
# - Distances: 3,5,7 (via distance curriculum)
# - P-curriculum: 0.007 -> 0.001, 30 epochs each (total 210 epochs)
# - Shots per epoch: 100k
# - Batch size: 2000
# - Teacher mix: all zero on CLI (training code will still need some labels internally)

SESSION_NAME="garnet_cudaq_benchmark"

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

# Tunables for this benchmark
WORKERS=${WORKERS:-40}
BATCH=${BATCH:-2000}
PREFETCH=${PREFETCH:-24}

# For single-GPU training we avoid torchrun/DDP entirely to sidestep rendezvous ports.
tmux send-keys -t "${SESSION_NAME}" \
  "CUDA_VISIBLE_DEVICES=1 python -m mghd.cli.train \
    --online \
    --sampler cudaq \
    --family surface \
    --distance 7 \
    --distance-curriculum 3,5,7 \
    --p-curriculum 0.007,0.006,0.005,0.004,0.003,0.002,0.001 \
    --epochs-per-p 30 \
    --epochs 210 \
    --teacher-mix mwpf=1.0 \
    --qpu-profile /u/home/kulp/MGHD/mghd/qpu/profiles/iqm_garnet_example.json \
    --context-source cudaq \
    --erasure-frac 0.05 \
    --shots-per-epoch 100000 \
    --batch ${BATCH} \
    --workers ${WORKERS} \
    --prefetch-factor ${PREFETCH} \
    --progress-prints 50 \
    --amp bf16 \
    --save /u/home/kulp/MGHD/data/results_GarnetCudaQ_24112025" C-m

echo "----------------------------------------------------------------"
echo "Garnet CUDA-Q benchmark training started in tmux session: '${SESSION_NAME}'"
echo "The job is running in the background and will survive disconnects."
echo "----------------------------------------------------------------"
echo "To view progress, run:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "To detach again (leave it running), press: Ctrl+B, then D"
echo "----------------------------------------------------------------"
