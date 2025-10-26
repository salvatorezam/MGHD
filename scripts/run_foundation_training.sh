#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/u/home/kulp/MGHD:${PYTHONPATH:-}"


# Foundation training driver for MGHD v2.
# Configure via environment variables to match your cluster setup.

FAMILIES=${FAMILIES:-surface}
DISTANCES=${DISTANCES:-3}
PS=${PS:-"0.001 0.005 0.01"}
SHOTS_PER_GRID=${SHOTS_PER_GRID:-256}
TEACHER_MIX=${TEACHER_MIX:-"mwpf=0.7,lsd=0.3,mwpm=0.0"}
QPU_PROFILE=${QPU_PROFILE:-mghd/qpu/profiles/iqm_garnet_example.json}
CONTEXT_SOURCE=${CONTEXT_SOURCE:-qiskit}
OUT_CROPS=${OUT_CROPS:-data/crops_fm}
TRAIN_SAVE_DIR=${TRAIN_SAVE_DIR:-checkpoints/foundation_s}
BATCH_SIZE=${BATCH_SIZE:-1024}
EPOCHS=${EPOCHS:-20}
LR=${LR:-5.952925899948483e-05}
WEIGHT_DECAY=${WEIGHT_DECAY:-6.65850238574699e-05}
EMA=${EMA:-0.999}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.13831652882929857}
NOISE_INJECTION=${NOISE_INJECTION:-0.009883059279379016}
GRAD_CLIP=${GRAD_CLIP:-0.8545326095750816}
PARITY_LAMBDA=${PARITY_LAMBDA:-0.1}
SEED=${SEED:-42}

python -m mghd.cli.make_cluster_crops \
  --families "${FAMILIES}" \
  --distances "${DISTANCES}" \
  --ps ${PS} \
  --shots-per-grid "${SHOTS_PER_GRID}" \
  --teacher-mix "${TEACHER_MIX}" \
  --qpu-profile "${QPU_PROFILE}" \
  --context-source "${CONTEXT_SOURCE}" \
  --out "${OUT_CROPS}" \
  --seed "${SEED}"

python -m mghd.cli.cluster_crops_train \
  --data-root "${OUT_CROPS}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH_SIZE}" \
  --lr "${LR}" \
  --wd "${WEIGHT_DECAY}" \
  --ema "${EMA}" \
  --parity-lambda "${PARITY_LAMBDA}" \
  --save "${TRAIN_SAVE_DIR}" \
  --seed "${SEED}" \
  --projection-aware 1 \
  --label-smoothing "${LABEL_SMOOTHING}" \
  --noise-injection "${NOISE_INJECTION}" \
  --grad-clip "${GRAD_CLIP}"