#!/usr/bin/env bash
set -euo pipefail

BASELINES_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$BASELINES_DIR/.." && pwd)
POOL_DIR=${POOL_DIR:-$REPO_ROOT/data/vision_opd_6k_decontaminated}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs}
THIRD_PARTY_DIR=${THIRD_PARTY_DIR:-$REPO_ROOT/third_party}
NNODES=${NNODES:-1}
LOGGER=${LOGGER:-console}
PROJECT_NAME=${PROJECT_NAME:-blinkopd-baselines}
ROLLOUT_N=8

log() { printf '[%s] %s\n' "$METHOD" "$*" >&2; }
die() { log "error: $*"; exit 1; }

usage() {
  echo "Usage: bash baselines/$METHOD.sh --size 4b|9b --gpus 0,1,2,3 [--no-eval] [--print-config] [-- HYDRA_OVERRIDE ...]"
}

parse_args() {
  SIZE= GPUS= NO_EVAL=0 PRINT_CONFIG=0
  EXTRA_ARGS=()
  while (( $# )); do
    case $1 in
      --size) (( $# >= 2 )) || die "--size needs a value"; SIZE=$2; shift 2 ;;
      --gpus) (( $# >= 2 )) || die "--gpus needs a value"; GPUS=$2; shift 2 ;;
      --no-eval) NO_EVAL=1; shift ;;
      --print-config) PRINT_CONFIG=1; shift ;;
      --) shift; EXTRA_ARGS=("$@"); break ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; die "unknown argument: $1" ;;
    esac
  done
  [[ $SIZE == 4b || $SIZE == 9b ]] || { usage >&2; die "--size must be 4b or 9b"; }
  [[ $GPUS =~ ^[0-9]+(,[0-9]+)*$ ]] || { usage >&2; die "--gpus must be a comma-separated list of GPU indices"; }
  [[ $NNODES =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer"
  IFS=, read -r -a GPU_LIST <<< "$GPUS"
  N_GPUS=${#GPU_LIST[@]}
  WORLD=$((N_GPUS * NNODES))
  case $SIZE in
    4b) MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B} ;;
    9b) MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B} ;;
  esac
  if [[ -e $MODEL_PATH ]]; then MODEL_PATH=$(cd "$MODEL_PATH" && pwd); fi
  mkdir -p "$OUTPUT_DIR" "$THIRD_PARTY_DIR"
  OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
  THIRD_PARTY_DIR=$(cd "$THIRD_PARTY_DIR" && pwd)
  RUN_NAME=${RUN_NAME:-$METHOD-$SIZE}
  RUN_DIR=$OUTPUT_DIR/baselines/$RUN_NAME
  CKPT_DIR=$RUN_DIR/checkpoints
  ROLLOUT_DIR=$RUN_DIR/rollouts
}

logger_list() {
  local items item out=
  IFS=, read -r -a items <<< "$LOGGER"
  for item in "${items[@]}"; do out+="\"$item\","; done
  echo "[${out%,}]"
}

fetch_repo() {
  local name=$1 url=$2 commit=$3
  REPO_DIR=$THIRD_PARTY_DIR/$name
  if [[ ! -d $REPO_DIR/.git ]]; then
    log "cloning $url into $REPO_DIR"
    git clone -q "$url" "$REPO_DIR"
  fi
  git -C "$REPO_DIR" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$REPO_DIR" fetch -q origin
  git -C "$REPO_DIR" -c advice.detachedHead=false checkout -q "$commit"
  [[ $(git -C "$REPO_DIR" rev-parse HEAD) == $(git -C "$REPO_DIR" rev-parse "$commit^{commit}") ]] || die "$REPO_DIR is not at $commit"
  if [[ -n $(git -C "$REPO_DIR" status --porcelain --untracked-files=no) ]]; then
    log "warning: $REPO_DIR has local modifications to tracked files"
  fi
}

restore_untracked() {
  local target=$REPO_DIR path
  fetch_repo Vision-OPD https://github.com/VisionOPD/Vision-OPD.git c8a8fdd
  for path in "$@"; do
    [[ -e $target/$path ]] && continue
    log "restoring $path in $target from Vision-OPD c8a8fdd (not tracked at the pinned commit)"
    git -C "$REPO_DIR" archive c8a8fdd "$path" | tar -x -C "$target"
  done
  REPO_DIR=$target
}

prepare_pool() {
  local d
  DATA_OUT=$OUTPUT_DIR/data/$1
  [[ -f $POOL_DIR/train.jsonl && -d $POOL_DIR/images && -d $POOL_DIR/teacher_images ]] \
    || die "training pool not found: $POOL_DIR must contain train.jsonl, images/ and teacher_images/"
  mkdir -p "$DATA_OUT"
  cmp -s "$POOL_DIR/train.jsonl" "$DATA_OUT/train.jsonl" || cp "$POOL_DIR/train.jsonl" "$DATA_OUT/train.jsonl"
  for d in images teacher_images; do ln -sfn "$(cd "$POOL_DIR/$d" && pwd)" "$DATA_OUT/$d"; done
  N_ROWS=$(grep -c '[^[:space:]]' "$DATA_OUT/train.jsonl")
}

build() {
  local target=$1 dep=$2
  shift 2
  if [[ -f $target && ! $dep -nt $target ]]; then return 0; fi
  log "building $target"
  (cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" python "$@")
  [[ -f $target ]] || die "data preparation did not produce $target"
  touch "$target"
}

check_world() {
  local rollouts=$1
  (( rollouts % WORLD == 0 )) \
    || die "$rollouts rollouts per step are not divisible by $WORLD GPUs ($N_GPUS per node x $NNODES nodes)"
}

setup_runtime() {
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS WORLD_SIZE=$NNODES
  if (( NNODES == 1 )); then
    export RAY_ADDRESS=${RAY_ADDRESS:-local}
  else
    [[ -n ${RAY_ADDRESS:-} ]] || die "NNODES=$NNODES needs a running Ray cluster ($N_GPUS GPUs per node) and RAY_ADDRESS set to its head"
  fi
  mkdir -p "$RUN_DIR"
  MERGED_DIR=$RUN_DIR/merged/global_step_$STEP
}

trained() {
  [[ -d $CKPT_DIR/global_step_$STEP/actor && $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null) == "$STEP" ]]
}

merge_checkpoint() {
  log "merging $CKPT_DIR/global_step_$STEP into $MERGED_DIR"
  rm -rf "$MERGED_DIR.partial"
  mkdir -p "$(dirname "$MERGED_DIR")"
  (cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR" CUDA_VISIBLE_DEVICES= python -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT_DIR/global_step_$STEP/actor" --target_dir "$MERGED_DIR.partial")
  mv "$MERGED_DIR.partial" "$MERGED_DIR"
}

run_eval() {
  local model=$1
  [[ -f $REPO_ROOT/eval/run_eval.sh ]] || die "missing $REPO_ROOT/eval/run_eval.sh"
  log "evaluating $model as $RUN_NAME on GPUs $GPUS"
  (cd "$REPO_ROOT" && env -u CUDA_VISIBLE_DEVICES -u WORLD_SIZE -u RAY_ADDRESS \
    bash eval/run_eval.sh --model "$model" --gpus "$GPUS" --name "$RUN_NAME")
}

train_and_eval() {
  if (( PRINT_CONFIG )); then
    (cd "$REPO_DIR" && "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --cfg job --resolve)
    exit 0
  fi
  if [[ -f $MERGED_DIR/config.json ]]; then
    log "found merged model $MERGED_DIR"
  else
    if trained; then
      log "found checkpoint $CKPT_DIR/global_step_$STEP"
    else
      log "training $RUN_NAME: $MODEL_PATH, $STEP steps, GPUs $GPUS x $NNODES node(s), checkpoints in $CKPT_DIR"
      (cd "$REPO_DIR" && "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
      trained || die "training ended without a complete checkpoint at $CKPT_DIR/global_step_$STEP"
    fi
    merge_checkpoint
  fi
  (( NO_EVAL )) || run_eval "$MERGED_DIR"
}
