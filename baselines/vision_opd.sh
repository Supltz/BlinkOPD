#!/usr/bin/env bash
METHOD=vision_opd
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
parse_args "$@"

fetch_repo Vision-OPD https://github.com/VisionOPD/Vision-OPD.git c8a8fdd
prepare_pool vision_opd
build "$DATA_OUT/train.parquet" "$DATA_OUT/train.jsonl" scripts/prepare_data.py --data-dir "$DATA_OUT" --skip-download

case $SIZE in
  4b) BATCH=${VOPD_TRAIN_BATCH_SIZE:-96} ;;
  9b) BATCH=${VOPD_TRAIN_BATCH_SIZE:-100} ;;
esac
if [[ $SIZE == 9b ]] && (( WORLD < 6 )); then GMU=0.6 MAX_BATCHED_TOKENS=9216; else GMU=0.8 MAX_BATCHED_TOKENS=16384; fi
check_world $((BATCH * ROLLOUT_N))
STEP=$((N_ROWS / BATCH))
setup_runtime

train_and_eval bash scripts/run_vision_opd.sh \
  "data.train_files=[\"$DATA_OUT/train.parquet\"]" \
  data.train_batch_size="$BATCH" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "critic.model.path=$MODEL_PATH" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GMU" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.agent.num_workers="$WORLD" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes="$NNODES" \
  "trainer.default_local_dir=$CKPT_DIR" \
  "trainer.rollout_data_dir=$ROLLOUT_DIR" \
  "trainer.project_name=$PROJECT_NAME" \
  "trainer.group_name=$RUN_NAME" \
  "trainer.experiment_name=$RUN_NAME" \
  "trainer.logger=$(logger_list)"
