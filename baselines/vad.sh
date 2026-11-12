#!/usr/bin/env bash
METHOD=vad
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
parse_args "$@"

fetch_repo VAD https://github.com/DeepExperience/VAD_Multimodal_OPD.git 4f2dd54
prepare_pool vad
build "$DATA_OUT/train_vad.parquet" "$DATA_OUT/train.jsonl" scripts/prepare_data.py --data-dir "$DATA_OUT" --skip-download --overwrite

BATCH=96
check_world $((BATCH * ROLLOUT_N))
STEP=64
setup_runtime
JUDGE_STUB=$RUN_DIR/judge_stub
mkdir -p "$JUDGE_STUB"
printf '{"object":"list","data":[{"id":"openai/gpt-oss-120b","object":"model"}]}\n' > "$JUDGE_STUB/models"

train_and_eval env \
  MODEL_PATH="$MODEL_PATH" \
  TEACHER_MODEL_PATH="$MODEL_PATH" \
  TASK_TRAIN_FILE="$DATA_OUT/train_vad.parquet" \
  BENCHMARK_EVAL_FILE="$DATA_OUT/train_vad.parquet" \
  BENCHMARK_EVAL_JUDGE_API_BASE="file://$JUDGE_STUB" \
  EXPERIMENT_NAME="$RUN_NAME" \
  PROJECT_NAME="$PROJECT_NAME" \
  TRAINER_N_GPUS_PER_NODE="$N_GPUS" \
  TRAINER_NNODES="$NNODES" \
  TRAINER_LOGGER="$(logger_list)" \
  TRAINER_DEFAULT_LOCAL_DIR="$CKPT_DIR" \
  TRAINER_ROLLOUT_DATA_DIR="$ROLLOUT_DIR" \
  TOTAL_TRAINING_STEPS="$STEP" \
  TRAINER_SAVE_FREQ="$STEP" \
  bash scripts/train_vad.sh --profile "$SIZE" -- trainer.test_freq=-1
