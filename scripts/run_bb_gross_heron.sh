#!/bin/bash
# Train MGHD on BB codes with IBM Heron noise and LSD teacher
# Usage: bash scripts/run_bb_gross_heron.sh [gross|double_gross]

set -e
cd "$(dirname "$0")/.."

# Auto-launch in tmux if not already inside one
FAMILY="${1:-gross}"
SESSION_NAME="bb_${FAMILY}"
if [ -z "$TMUX" ]; then
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo "Session $SESSION_NAME exists. Attaching..."
        tmux attach -t $SESSION_NAME
        exit 0
    fi
    tmux new-session -d -s $SESSION_NAME
    tmux send-keys -t $SESSION_NAME "cd $(pwd) && bash scripts/run_bb_gross_heron.sh $FAMILY inside_tmux" C-m
    echo "Started training in tmux session: $SESSION_NAME"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

# Inside tmux
source /u/home/kulp/miniconda3/etc/profile.d/conda.sh
conda activate mlqec-env

# Configuration
PROFILE="mghd/qpu/profiles/ibm_heron_r3.json"
DATE=$(date +%d%m%Y)
SAVE_DIR="/u/home/kulp/MGHD/data/results_${FAMILY}_heron_lsd_${DATE}"
export CUDA_VISIBLE_DEVICES=0

echo "Training $FAMILY with LSD teacher"
echo "Save dir: $SAVE_DIR"

mkdir -p $SAVE_DIR

torchrun --nproc_per_node=1 mghd/cli/train.py \
    --online \
    --sampler stim \
    --online-rl \
    --family $FAMILY \
    --teacher-mix lsd=1.0 \
    --qpu-profile $PROFILE \
    --context-source none \
    --shots-per-epoch 32768 \
    --epochs 120 \
    --p-curriculum "0.009,0.006,0.003,0.001,0.0007,0.0004" \
    --epochs-per-p 20 \
    --batch 512 \
    --workers 20 \
    --prefetch-factor 4 \
    --progress-prints 50 \
    --amp bf16 \
    --save $SAVE_DIR \
    --erasure-frac 0.05

echo "Training finished"
