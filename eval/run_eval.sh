set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
VLLM="${VLLM:-vllm}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
MODEL_PORT="${MODEL_PORT:-8100}"
JUDGE_PORT="${JUDGE_PORT:-8200}"
MODEL_TP="${MODEL_TP:-1}"
JUDGE_TP="${JUDGE_TP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"

MODEL=""
GPUS=""
JUDGE_GPUS=""
NAME=""
BENCHMARKS="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn"

usage() {
  echo "usage: bash eval/run_eval.sh --model PATH_OR_HF_ID --gpus 0,1,2,3 [--judge-gpus 4,5] [--name NAME] [--benchmarks b1,b2]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?}"; shift 2 ;;
    --gpus) GPUS="${2:?}"; shift 2 ;;
    --judge-gpus) JUDGE_GPUS="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --benchmarks) BENCHMARKS="${2:?}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[[ -n "${MODEL}" && -n "${GPUS}" ]] || usage
if [[ -n "${JUDGE_GPUS}" ]]; then
  IFS=',' read -r -a MODEL_IDS <<< "${GPUS}"
  IFS=',' read -r -a JUDGE_IDS <<< "${JUDGE_GPUS}"
  for m in "${MODEL_IDS[@]}"; do
    for j in "${JUDGE_IDS[@]}"; do
      [[ "$m" != "$j" ]] || { echo "GPU $m is selected for both model and judge" >&2; exit 2; }
    done
  done
fi
[[ -n "${NAME}" ]] || NAME="$(basename "${MODEL%/}")"

RUN_DIR="${OUTPUT_ROOT}/${NAME}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

ALL_PIDS=()
REPLICA_PIDS=()
REPLICA_URLS=()
REPLICA_LOGS=()

alive() {
  local state
  state="$(ps -o stat= -p "$1" 2>/dev/null || true)"
  [[ -n "${state}" && "${state}" != Z* ]]
}

stop_servers() {
  local pid
  for pid in "$@"; do
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "$@"; do
    for _ in $(seq 1 60); do
      alive "${pid}" || break
      sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || true
  done
}

cleanup() {
  if [[ ${#ALL_PIDS[@]} -gt 0 ]]; then
    stop_servers "${ALL_PIDS[@]}"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start_replicas() {
  local role="$1" model="$2" served="$3" port="$4" gpus="$5" tp="$6"
  shift 6
  local ids devices log url i=0
  IFS=',' read -r -a ids <<< "${gpus}"
  if (( ${#ids[@]} == 0 || ${#ids[@]} % tp != 0 )); then
    echo "${role}: ${#ids[@]} GPU(s) cannot be split into tensor-parallel groups of ${tp}" >&2
    exit 2
  fi
  REPLICA_PIDS=()
  REPLICA_URLS=()
  REPLICA_LOGS=()
  while (( i < ${#ids[@]} )); do
    devices="$(IFS=','; echo "${ids[*]:i:tp}")"
    url="http://${SERVE_HOST}:${port}/v1"
    log="${LOG_DIR}/${role}_${port}.log"
    if curl -s -o /dev/null --max-time 5 "${url}/models"; then
      echo "port ${port} is already in use; set ${role^^}_PORT" >&2
      exit 1
    fi
    echo "[serve] ${role} on GPU ${devices} at ${url} (log: ${log})"
    CUDA_VISIBLE_DEVICES="${devices}" setsid "${VLLM}" serve "${model}" \
      --host "${SERVE_HOST}" --port "${port}" --served-model-name "${served}" \
      --tensor-parallel-size "${tp}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-num-seqs 64 --trust-remote-code "$@" > "${log}" 2>&1 &
    REPLICA_PIDS+=("$!")
    ALL_PIDS+=("$!")
    REPLICA_URLS+=("${url}")
    REPLICA_LOGS+=("${log}")
    i=$((i + tp))
    port=$((port + 1))
  done
}

wait_ready() {
  local n=$(( $# / 3 )) k pid url log deadline=$((SECONDS + SERVER_TIMEOUT))
  local args=("$@")
  for (( k = 0; k < n; k++ )); do
    pid="${args[k]}"
    url="${args[n + k]}"
    log="${args[2 * n + k]}"
    until curl -sf -o /dev/null --max-time 5 "${url}/models"; do
      if ! alive "${pid}"; then
        echo "server exited before becoming ready: ${log}" >&2
        tail -n 40 "${log}" >&2
        exit 1
      fi
      if (( SECONDS >= deadline )); then
        echo "server not ready after ${SERVER_TIMEOUT}s: ${log}" >&2
        exit 1
      fi
      sleep 10
    done
  done
}

join_urls() {
  local IFS=','
  echo "$*"
}

echo "[eval] model=${MODEL} name=${NAME} benchmarks=${BENCHMARKS}"
echo "[eval] outputs: ${RUN_DIR}"
"${PYTHON}" "${EVAL_DIR}/prepare_data.py" --data-dir "${DATA_DIR}" --benchmarks "${BENCHMARKS}"

INFER_ARGS=(--data-dir "${DATA_DIR}" --run-dir "${RUN_DIR}" --benchmarks "${BENCHMARKS}" --model-id eval-model)
SCORE_ARGS=(--data-dir "${DATA_DIR}" --run-dir "${RUN_DIR}" --benchmarks "${BENCHMARKS}"
            --judge-model judge --judge-tokenizer "${JUDGE_MODEL}" --name "${NAME}")
JUDGE_FLAGS=(--max-model-len 131072 --enforce-eager)

JUDGE_PIDS=()
JUDGE_URLS=()
JUDGE_LOGS=()
if [[ -n "${JUDGE_GPUS}" ]]; then
  start_replicas judge "${JUDGE_MODEL}" judge "${JUDGE_PORT}" "${JUDGE_GPUS}" "${JUDGE_TP}" "${JUDGE_FLAGS[@]}"
  JUDGE_PIDS=("${REPLICA_PIDS[@]}")
  JUDGE_URLS=("${REPLICA_URLS[@]}")
  JUDGE_LOGS=("${REPLICA_LOGS[@]}")
fi

PENDING="$("${PYTHON}" "${EVAL_DIR}/infer.py" "${INFER_ARGS[@]}" --count-pending | tail -n 1)"
if (( PENDING > 0 )); then
  echo "[eval] ${PENDING} samples to answer"
  start_replicas model "${MODEL}" eval-model "${MODEL_PORT}" "${GPUS}" "${MODEL_TP}"
  MODEL_PIDS=("${REPLICA_PIDS[@]}")
  MODEL_URLS=("${REPLICA_URLS[@]}")
  wait_ready "${MODEL_PIDS[@]}" "${MODEL_URLS[@]}" "${REPLICA_LOGS[@]}"
  "${PYTHON}" "${EVAL_DIR}/infer.py" "${INFER_ARGS[@]}" --api-bases "$(join_urls "${MODEL_URLS[@]}")"
  stop_servers "${MODEL_PIDS[@]}"
else
  echo "[eval] all answers present; skipping inference"
fi

PENDING="$("${PYTHON}" "${EVAL_DIR}/score.py" "${SCORE_ARGS[@]}" --count-pending | tail -n 1)"
if (( PENDING > 0 )); then
  if [[ -z "${JUDGE_GPUS}" ]]; then
    start_replicas judge "${JUDGE_MODEL}" judge "${JUDGE_PORT}" "${GPUS}" "${JUDGE_TP}" "${JUDGE_FLAGS[@]}"
    JUDGE_PIDS=("${REPLICA_PIDS[@]}")
    JUDGE_URLS=("${REPLICA_URLS[@]}")
    JUDGE_LOGS=("${REPLICA_LOGS[@]}")
  fi
  wait_ready "${JUDGE_PIDS[@]}" "${JUDGE_URLS[@]}" "${JUDGE_LOGS[@]}"
  "${PYTHON}" "${EVAL_DIR}/score.py" "${SCORE_ARGS[@]}" --judge-bases "$(join_urls "${JUDGE_URLS[@]}")"
else
  "${PYTHON}" "${EVAL_DIR}/score.py" "${SCORE_ARGS[@]}"
fi
