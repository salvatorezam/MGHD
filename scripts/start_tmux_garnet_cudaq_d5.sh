#!/usr/bin/env bash
set -euo pipefail

# Launch Garnet (CUDA-Q) MWPF-trained MGHD for d=5 in a tmux session.
SESSION_NAME="garnet_cudaq_d5"

# If session exists, just tell the user how to attach.
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}"

tmux send-keys -t "${SESSION_NAME}" "source /u/home/kulp/miniconda3/bin/activate mlqec-env" C-m
tmux send-keys -t "${SESSION_NAME}" "cd /u/home/kulp/MGHD" C-m
tmux send-keys -t "${SESSION_NAME}" 'export PYTHONPATH="/u/home/kulp/MGHD:${PYTHONPATH:-}"' C-m

# Run single-GPU training on GPU 1 (d=5, Garnet + MWPF)
tmux send-keys -t "${SESSION_NAME}" \
  "CUDA_VISIBLE_DEVICES=1 python -m mghd.cli.train \
    --online \
    --sampler cudaq \
    --family surface \
    --distance 5 \
    --p-curriculum 0.010,0.008,0.006,0.004,0.002,0.001 \
    --epochs-per-p 30 \
    --epochs 180 \
    --teacher-mix mwpf=1.0,mwpm=0.0,lsd=0.0 \
    --qpu-profile mghd/qpu/profiles/iqm_garnet_example.json \
    --context-source cudaq \
    --erasure-frac 0.05 \
    --shots-per-epoch 50000 \
    --batch 2000 \
    --workers 40 \
    --prefetch-factor 24 \
    --progress-prints 50 \
    --amp bf16 \
    --save data/results_GarnetCudaQ_d5_24112025" C-m

# ---------------------------------------------------------------------------
# Queued follow-up runs (templates)
# ---------------------------------------------------------------------------
# After d=5 finishes, you can immediately start the next distance on the
# same GPU by copy-pasting one of the commands below into the tmux pane.
# This keeps GPU1 busy while you analyse the finished checkpoint.
#
# Example: distance 7 on GPU 1
# CUDA_VISIBLE_DEVICES=1 python -m mghd.cli.train \
#   --online \
#   --sampler cudaq \
#   --family surface \
#   --distance 7 \
#   --p-curriculum 0.010,0.008,0.006,0.004,0.002,0.001 \
#   --epochs-per-p 30 \
#   --epochs 180 \
#   --teacher-mix mwpf=1.0,mwpm=0.0,lsd=0.0 \
#   --qpu-profile mghd/qpu/profiles/iqm_garnet_example.json \
#   --context-source cudaq \
#   --erasure-frac 0.05 \
#   --shots-per-epoch 50000 \
#   --batch 2000 \
#   --workers 40 \
#   --prefetch-factor 24 \
#   --progress-prints 50 \
#   --amp bf16 \
#   --save data/results_GarnetCudaQ_d7_24112025

echo "----------------------------------------------------------------"
echo "Garnet CUDA-Q d=5 training started in tmux session: '${SESSION_NAME}'"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "----------------------------------------------------------------"
