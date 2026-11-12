#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${1:?Usage: bash scripts/merge_checkpoint.sh <checkpoints/global_step_N> <output_dir>}"
TARGET_DIR="${2:?Usage: bash scripts/merge_checkpoint.sh <checkpoints/global_step_N> <output_dir>}"
CKPT_DIR="${CKPT_DIR%/}"

if [[ ! -d "$CKPT_DIR/actor" ]]; then
  echo "Actor checkpoint not found: $CKPT_DIR/actor" >&2
  exit 1
fi
mkdir -p "$TARGET_DIR"
PYTHONPATH="$ROOT:${PYTHONPATH:-}" python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$CKPT_DIR/actor" \
  --target_dir "$TARGET_DIR"
cp "$ROOT/chat_templates/perception_chat_template_qwen35.jinja" "$TARGET_DIR/chat_template.jinja"
echo "Merged model written to $TARGET_DIR"
