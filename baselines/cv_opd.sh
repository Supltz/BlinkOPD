#!/usr/bin/env bash
METHOD=cv_opd
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
parse_args "$@"

fetch_repo CV-OPD https://github.com/HaoLin3419/Contrastive-Vision-OPD.git 8be9a55
restore_untracked verl/models verl/trainer/config/data

prepare_pool cv_opd
build "$DATA_OUT/train.parquet" "$DATA_OUT/train.jsonl" scripts/prepare_data.py --data-dir "$DATA_OUT" --skip-download
build "$DATA_OUT/train_bbox.parquet" "$DATA_OUT/train.parquet" scripts/prepare_crop_full_rank_dataset.py \
  --input-jsonl "$DATA_OUT/train.jsonl" --input-parquet "$DATA_OUT/train.parquet" --output-parquet "$DATA_OUT/train_bbox.parquet"

BATCH=96
check_world $((BATCH * ROLLOUT_N))
STEP=$((N_ROWS / BATCH))
setup_runtime

train_and_eval bash scripts/run_vision_opd.sh \
  "data.train_files=[\"$DATA_OUT/train_bbox.parquet\"]" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "critic.model.path=$MODEL_PATH" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes="$NNODES" \
  "trainer.default_local_dir=$CKPT_DIR" \
  "trainer.rollout_data_dir=$ROLLOUT_DIR" \
  "trainer.project_name=$PROJECT_NAME" \
  "trainer.group_name=$RUN_NAME" \
  "trainer.experiment_name=$RUN_NAME" \
  "trainer.logger=$(logger_list)" \
  actor_rollout_ref.actor.self_distillation.distillation_target_mode=crop_only_v10_2 \
  actor_rollout_ref.actor.self_distillation.ranking_margin=0.1 \
  actor_rollout_ref.actor.self_distillation.ranking_loss_weight=0.1 \
  actor_rollout_ref.actor.self_distillation.negative_crop_max_iou=0.1 \
  actor_rollout_ref.actor.self_distillation.ranking_token_selection=batch_mean_jsd \
  actor_rollout_ref.actor.self_distillation.distillation_temperature=1.0 \
  "custom_reward_function.path=$REPO_DIR/scripts/vstar_mcq_binary_reward.py" \
  custom_reward_function.name=compute_score
