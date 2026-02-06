#!/usr/bin/env bash
set -euo pipefail

# MGHD Execution Plan v3 launcher
# Usage:
#   bash scripts/run_execution_plan_v3.sh capability
#   bash scripts/run_execution_plan_v3.sh audit
#   bash scripts/run_execution_plan_v3.sh phase_a
#   bash scripts/run_execution_plan_v3.sh phase_b
#   bash scripts/run_execution_plan_v3.sh phase_c
#   bash scripts/run_execution_plan_v3.sh eval
#   bash scripts/run_execution_plan_v3.sh memory
#   bash scripts/run_execution_plan_v3.sh ablation

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="${ENV_NAME:-mlqec-env}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OUT_ROOT="${OUT_ROOT:-data/plan_v3_runs}"
mkdir -p "${OUT_ROOT}"

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Missing mode. Choose: capability|audit|phase_a|phase_b|phase_c|eval|memory|ablation"
  exit 1
fi

run_python() {
  conda run -n "${ENV_NAME}" python "$@"
}

run_torchrun() {
  conda run -n "${ENV_NAME}" torchrun --nproc_per_node="${NPROC_PER_NODE}" "$@"
}

case "${MODE}" in
  capability)
    run_python scripts/decoder_capability_gate.py \
      --output "${OUT_ROOT}/decoder_capability_matrix.json"
    ;;

  audit)
    run_python scripts/audit_teacher_contracts.py \
      --distances "${DISTANCES:-3,5,7}" \
      --p-values "${P_VALUES:-0.001,0.003,0.005}" \
      --shots "${SHOTS_AUDIT:-64}" \
      --sampler "${SAMPLER_AUDIT:-synthetic}" \
      --enable-nvqldpc \
      --output "${OUT_ROOT}/teacher_contract_report.json"
    ;;

  phase_a)
    run_torchrun mghd/cli/train.py \
      --online \
      --online-fast \
      --family surface \
      --distance 7 \
      --distance-curriculum 3,5,7 \
      --sampler synthetic \
      --teacher-mix "${TEACHER_MIX_A:-lsd=0.7,mwpm=0.3,mwpf=0.0}" \
      --teacher-contract-report "${OUT_ROOT}/teacher_contract_report.json" \
      --teacher-contract-strict \
      --p-curriculum "${P_CURR_A:-0.01,0.008,0.006,0.004,0.003,0.002,0.001}" \
      --epochs-per-p "${EPOCHS_PER_P_A:-5}" \
      --epochs "${EPOCHS_A:-35}" \
      --shots-per-epoch "${SHOTS_PER_EPOCH_A:-16384}" \
      --batch "${BATCH_A:-1024}" \
      --workers "${WORKERS_A:-8}" \
      --prefetch-factor "${PREFETCH_A:-8}" \
      --amp "${AMP_A:-bf16}" \
      --save "${OUT_ROOT}/phase_a_phenomenological"
    ;;

  phase_b)
    run_torchrun mghd/cli/train.py \
      --online \
      --online-fast \
      --family surface \
      --distance 7 \
      --distance-curriculum 3,5,7 \
      --sampler cudaq \
      --noise-model generic_cl \
      --generic-p1q "${GENERIC_P1Q_B:-0.0015}" \
      --generic-p2q "${GENERIC_P2Q_B:-0.01}" \
      --generic-pidle "${GENERIC_PIDLE_B:-0.0008}" \
      --generic-pmeas0 "${GENERIC_PMEAS0_B:-0.02}" \
      --generic-pmeas1 "${GENERIC_PMEAS1_B:-0.02}" \
      --generic-phook "${GENERIC_PHOOK_B:-0.002}" \
      --generic-pcrosstalk "${GENERIC_PCROSSTALK_B:-0.0005}" \
      --teacher-mix "${TEACHER_MIX_B:-lsd=0.6,mwpm=0.2,nvqldpc=0.2,mwpf=0.0}" \
      --teacher-contract-report "${OUT_ROOT}/teacher_contract_report.json" \
      --p-curriculum "${P_CURR_B:-0.01,0.008,0.006,0.004,0.003,0.002,0.001}" \
      --epochs-per-p "${EPOCHS_PER_P_B:-6}" \
      --epochs "${EPOCHS_B:-42}" \
      --shots-per-epoch "${SHOTS_PER_EPOCH_B:-16384}" \
      --batch "${BATCH_B:-1024}" \
      --workers "${WORKERS_B:-8}" \
      --prefetch-factor "${PREFETCH_B:-8}" \
      --amp "${AMP_B:-bf16}" \
      --save "${OUT_ROOT}/phase_b_mild_correlated"
    ;;

  phase_c)
    run_torchrun mghd/cli/train.py \
      --online \
      --online-fast \
      --family surface \
      --distance 9 \
      --distance-curriculum 3,5,7,9 \
      --sampler cudaq \
      --noise-model generic_cl \
      --generic-p1q "${GENERIC_P1Q_C:-0.0015}" \
      --generic-p2q "${GENERIC_P2Q_C:-0.012}" \
      --generic-pidle "${GENERIC_PIDLE_C:-0.0008}" \
      --generic-pmeas0 "${GENERIC_PMEAS0_C:-0.02}" \
      --generic-pmeas1 "${GENERIC_PMEAS1_C:-0.02}" \
      --generic-phook "${GENERIC_PHOOK_C:-0.01}" \
      --generic-pcrosstalk "${GENERIC_PCROSSTALK_C:-0.002}" \
      --teacher-mix "${TEACHER_MIX_C:-lsd=0.5,mwpm=0.2,nvqldpc=0.3,mwpf=0.0}" \
      --teacher-contract-report "${OUT_ROOT}/teacher_contract_report.json" \
      --p-curriculum "${P_CURR_C:-0.01,0.008,0.006,0.004,0.003,0.002,0.001}" \
      --epochs-per-p "${EPOCHS_PER_P_C:-12}" \
      --epochs "${EPOCHS_C:-84}" \
      --shots-per-epoch "${SHOTS_PER_EPOCH_C:-16384}" \
      --batch "${BATCH_C:-2048}" \
      --workers "${WORKERS_C:-8}" \
      --prefetch-factor "${PREFETCH_C:-8}" \
      --amp "${AMP_C:-bf16}" \
      --save "${OUT_ROOT}/phase_c_full_correlated"
    ;;

  eval)
    CKPT="${CHECKPOINT:-${OUT_ROOT}/phase_c_full_correlated/best.pt}"
    EXTRA_EVAL_FLAGS=()
    if [[ "${ENABLE_NVQLDPC_EVAL:-0}" == "1" ]]; then
      EXTRA_EVAL_FLAGS+=(--enable-nvqldpc)
    fi
    if [[ "${ENABLE_TN_EVAL:-1}" == "1" ]]; then
      EXTRA_EVAL_FLAGS+=(--enable-tn --tn-noise-model "${TN_NOISE_MODEL:-auto}" --tn-device "${TN_DEVICE:-cpu}")
    fi
    run_python scripts/evaluate_model.py \
      --checkpoint "${CKPT}" \
      --family surface \
      --distances "${EVAL_DISTANCES:-3,5,7,9}" \
      --p-values "${EVAL_P_VALUES:-0.001,0.002,0.003,0.005,0.008,0.01}" \
      --shots "${EVAL_SHOTS:-2048}" \
      --batch-size "${EVAL_BATCH:-256}" \
      --sampler "${EVAL_SAMPLER:-cudaq}" \
      --cuda \
      --disable-mwpf \
      --mghd-error-policy raise \
      --output "${OUT_ROOT}/eval_main.json" \
      "${EXTRA_EVAL_FLAGS[@]}"
    ;;

  memory)
    CKPT="${CHECKPOINT:-${OUT_ROOT}/phase_c_full_correlated/best.pt}"
    run_python scripts/memory_experiment.py \
      --checkpoint "${CKPT}" \
      --family surface \
      --distance "${MEMORY_D:-5}" \
      --p-values "${MEMORY_P_VALUES:-0.001,0.003,0.005}" \
      --rounds "${MEMORY_ROUNDS:-5,10,20,40}" \
      --shots "${MEMORY_SHOTS:-2048}" \
      --batch-size "${MEMORY_BATCH:-128}" \
      --sampler "${MEMORY_SAMPLER:-stim}" \
      --cuda \
      --output "${OUT_ROOT}/memory_scaling_report.json"
    ;;

  ablation)
    echo "Ablation runner expects separate checkpoints per architecture variant."
    echo "Run phase_c separately with variant configs (full/no_mamba/no_gnn/no_attention)."
    echo "Then evaluate each checkpoint with scripts/evaluate_model.py and compare JSON outputs."
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    exit 1
    ;;
esac
