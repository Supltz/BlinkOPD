#!/usr/bin/env bash
METHOD=opd_aha
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
parse_args "$@"

fetch_repo OPD-Aha https://github.com/Echochef/OPD-Aha.git 66982fb
restore_untracked verl/trainer/config/data
prepare_pool opd_aha
build "$DATA_OUT/train.parquet" "$DATA_OUT/train.jsonl" scripts/prepare_data.py --data-dir "$DATA_OUT" --skip-download

BATCH=96
check_world $((BATCH * ROLLOUT_N))
STEP=${BASELINE_STEPS:-40}
[[ $STEP =~ ^[1-9][0-9]*$ ]] || die "BASELINE_STEPS must be positive"
setup_runtime

train_and_eval env \
  MODEL_PATH="$MODEL_PATH" \
  TASK_TRAIN_FILE="$DATA_OUT/train.parquet" \
  EXPERIMENT_NAME_OVERRIDE="$RUN_NAME" \
  PROJECT_NAME="$PROJECT_NAME" \
  TRAINER_N_GPUS_PER_NODE="$N_GPUS" \
  TRAINER_NNODES="$NNODES" \
  TRAINER_TOTAL_TRAINING_STEPS="$STEP" \
  TRAINER_SAVE_FREQ="$STEP" \
  TRAINER_DEFAULT_LOCAL_DIR="$CKPT_DIR" \
  TRAINER_ROLLOUT_DATA_DIR="$ROLLOUT_DIR" \
  TRAINER_LOGGER="$(logger_list)" \
  bash scripts/train_opd_aha.sh "$SIZE"
