#!/usr/bin/env bash
METHOD=opd_v
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
parse_args "$@"

fetch_repo OPD-V https://github.com/aniri15/OPD-V.git a780c15
prepare_pool opd_v
build "$DATA_OUT/train.parquet" "$DATA_OUT/train.jsonl" scripts/prepare_data.py --data-dir "$DATA_OUT" --skip-download

BATCH=48
check_world $((BATCH * ROLLOUT_N))
STEP=$((N_ROWS / BATCH))
setup_runtime
CKPT_DIR=$RUN_DIR/checkpoints/$RUN_NAME

train_and_eval env \
  MODEL_PATH="$MODEL_PATH" \
  DATA_DIR="$DATA_OUT" \
  HF_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}" \
  OPDV_WORK_DIR="$RUN_DIR" \
  EXPERIMENT_NAME="$RUN_NAME" \
  PROJECT_NAME="$PROJECT_NAME" \
  bash scripts/run_opdv.sh \
  trainer.n_gpus_per_node="$N_GPUS" \
  "trainer.logger=$(logger_list)"
