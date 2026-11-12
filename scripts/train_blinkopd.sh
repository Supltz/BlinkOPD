#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIZE="${1:-}"
shift || true
GPUS=""
PRINT_CONFIG=0
EXTRA=()
while (( $# )); do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

case "$SIZE" in
  4b) DEFAULT_MODEL="Qwen/Qwen3.5-4B"; SKIP_FIRST=0; DEFAULT_TOKENS=12288; DEFAULT_OFFLOAD=False ;;
  9b) DEFAULT_MODEL="Qwen/Qwen3.5-9B"; SKIP_FIRST=1; DEFAULT_TOKENS=9216; DEFAULT_OFFLOAD=True ;;
  *) echo "Usage: bash scripts/train_blinkopd.sh {4b|9b} --gpus 0,1,2,3 [hydra overrides]" >&2; exit 2 ;;
esac

if [[ -n "$GPUS" ]]; then
  [[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Invalid --gpus list" >&2; exit 2; }
  export CUDA_VISIBLE_DEVICES="$GPUS"
  IFS=, read -r -a GPU_IDS <<< "$GPUS"
  N_GPUS="${N_GPUS:-${#GPU_IDS[@]}}"
fi
N_GPUS="${N_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
[[ "$N_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "Set N_GPUS or select at least one GPU" >&2; exit 2; }
NNODES="${NNODES:-1}"
[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || { echo "NNODES must be positive" >&2; exit 2; }
if (( NNODES > 1 )) && [[ -z "${RAY_ADDRESS:-}" ]]; then
  echo "Set RAY_ADDRESS to an existing Ray cluster for multi-node training" >&2
  exit 2
fi
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/blinkopd_k500/train_${SIZE}.parquet}"
RUN_NAME="${RUN_NAME:-blinkopd_${SIZE}_k500}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$RUN_NAME}"
TOTAL_STEPS="${TOTAL_STEPS:-38}"

if (( ! PRINT_CONFIG )) && [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Training file not found: $TRAIN_FILE (run python scripts/prepare_data.py first)" >&2
  exit 1
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
unset VLLM_ATTENTION_BACKEND
if (( ! PRINT_CONFIG )); then mkdir -p "$OUTPUT_DIR/rollouts"; fi
if (( PRINT_CONFIG )); then
  COMMAND=(python "$ROOT/scripts/print_config.py")
else
  COMMAND=(python -m verl.trainer.main_ppo)
fi

SD=actor_rollout_ref.actor.self_distillation
"${COMMAND[@]}" --config-name vopd \
  data.train_files="[\"$TRAIN_FILE\"]" \
  data.val_files="[]" \
  data.filter_overlong_prompts=False \
  data.max_prompt_length=8192 \
  data.max_response_length=1024 \
  data.seed=42 \
  data.truncation=error \
  data.shuffle=True \
  data.trust_remote_code=True \
  data.return_multi_modal_inputs=True \
  data.image_key=images \
  data.train_batch_size=96 \
  data.dataloader_num_workers=8 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.custom_chat_template_file="$ROOT/chat_templates/perception_chat_template_qwen35.jinja" \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=96 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-$DEFAULT_TOKENS}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-$DEFAULT_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-$DEFAULT_OFFLOAD}" \
  actor_rollout_ref.actor.clip_ratio_high=0.3 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
  actor_rollout_ref.actor.calculate_entropy=False \
  $SD.distillation_topk=100 \
  $SD.distillation_add_tail=True \
  $SD.max_reprompt_len=10240 \
  $SD.is_clip=2.0 \
  $SD.teacher_always_on=True \
  $SD.teacher_model_source=legacy \
  $SD.teacher_regularization=frozen \
  $SD.teacher_update_rate=0.0 \
  $SD.teacher_image_key=bbox_images \
  $SD.counterfactual_null_mode=cut \
  $SD.counterfactual_extrapolation_beta=3.4 \
  $SD.counterfactual_beta_gate=verifier \
  $SD.counterfactual_gate_on_scale=2.0 \
  $SD.counterfactual_gate_off_scale=1.0 \
  $SD.isolate_packed_sequences=True \
  $SD.skip_first_tokens=$SKIP_FIRST \
  $SD.dont_reprompt_on_self_success=True \
  $SD.alpha=0.5 \
  $SD.include_environment_feedback=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
  actor_rollout_ref.rollout.max_model_len=9216 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
  actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.enable=False \
  reward_model.use_reward_loop=False \
  critic.model.path="$MODEL_PATH" \
  custom_reward_function.path=null \
  trainer.project_name=BlinkOPD \
  trainer.group_name="$RUN_NAME" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes="$NNODES" \
  trainer.save_freq=10 \
  trainer.test_freq=-1 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.total_epochs=2 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.val_before_train=False \
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
  trainer.rollout_data_dir="$OUTPUT_DIR/rollouts" \
  ++ray_kwargs.ray_init.include_dashboard=False \
  ${EXTRA[@]+"${EXTRA[@]}"}

if (( PRINT_CONFIG )); then exit 0; fi
bash "$ROOT/scripts/merge_checkpoint.sh" "$OUTPUT_DIR/checkpoints/global_step_${TOTAL_STEPS}" "$OUTPUT_DIR/model"
