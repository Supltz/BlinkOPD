#!/usr/bin/env bash
METHOD=base_model
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
parse_args "$@"
(( ! PRINT_CONFIG )) || die "--print-config does not apply: the base model is evaluated without training"
(( ! NO_EVAL )) || die "--no-eval leaves nothing to do for the base model"
run_eval "$MODEL_PATH"
