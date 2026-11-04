set -euo pipefail

GE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${GE_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
VLLM="${VLLM:-vllm}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/general}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/general_eval}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
MODEL_PORT="${MODEL_PORT:-8300}"
JUDGE_PORT="${JUDGE_PORT:-8400}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.85}"
JUDGE_TP="${JUDGE_TP:-1}"
THREADS_PER_SERVER="${THREADS_PER_SERVER:-288}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"

MODEL=""
GPUS=""
JUDGE_GPUS=""
NAME=""
BENCHMARKS="mmstar,cvbench,mmvp,pope,hallusionbench,mathvista,mathverse,mathvision,wemath,dynamath"

usage() {
  echo "usage: bash general_eval/run.sh --model PATH_OR_HF_ID --gpus 0,1,2,3 [--judge-gpu 4] [--name NAME] [--benchmarks b1,b2]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?}"; shift 2 ;;
    --gpus) GPUS="${2:?}"; shift 2 ;;
    --judge-gpu|--judge-gpus) JUDGE_GPUS="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --benchmarks) BENCHMARKS="${2:?}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[[ -n "${MODEL}" && -n "${GPUS}" ]] || usage
[[ -n "${NAME}" ]] || NAME="$(basename "${MODEL%/}")"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ -n "${JUDGE_GPUS}" ]]; then
  IFS=',' read -r -a JUDGE_LIST <<< "${JUDGE_GPUS}"
  for g in "${JUDGE_LIST[@]}"; do
    for m in "${GPU_LIST[@]}"; do
      if [[ "${g}" == "${m}" ]]; then
        echo "GPU ${g} is in both --gpus and --judge-gpu" >&2
        exit 2
      fi
    done
  done
else
  if (( ${#GPU_LIST[@]} < JUDGE_TP )); then
    echo "the judge needs ${JUDGE_TP} GPU(s); pass --judge-gpu or more --gpus" >&2
    exit 2
  fi
  JUDGE_LIST=("${GPU_LIST[@]:0:JUDGE_TP}")
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

RUN_DIR="${OUTPUT_ROOT}/${NAME}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

ALL_PIDS=()
MODEL_PIDS=()
MODEL_URLS=()
MODEL_LOGS=()
JUDGE_PID=""
JUDGE_URL="http://${SERVE_HOST}:${JUDGE_PORT}/v1"
JUDGE_LOG="${LOG_DIR}/judge.log"

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

check_port() {
  if curl -s -o /dev/null --max-time 5 "$1/models"; then
    echo "$1 is already in use; set MODEL_PORT or JUDGE_PORT" >&2
    exit 1
  fi
}

wait_ready() {
  local pid="$1" url="$2" log="$3" deadline=$((SECONDS + SERVER_TIMEOUT))
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
}

start_judge() {
  local devices
  devices="$(IFS=','; echo "${JUDGE_LIST[*]}")"
  check_port "${JUDGE_URL}"
  echo "[general_eval] judge ${JUDGE_MODEL} on GPU ${devices} at ${JUDGE_URL} (log: ${JUDGE_LOG})"
  CUDA_VISIBLE_DEVICES="${devices}" setsid "${VLLM}" serve "${JUDGE_MODEL}" \
    --host "${SERVE_HOST}" --port "${JUDGE_PORT}" --served-model-name judge \
    --tensor-parallel-size "${#JUDGE_LIST[@]}" --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code --max-num-seqs 64 --max-model-len 131072 --enforce-eager > "${JUDGE_LOG}" 2>&1 &
  JUDGE_PID="$!"
  ALL_PIDS+=("${JUDGE_PID}")
}

start_models() {
  local i=0 g port url log
  for g in "${GPU_LIST[@]}"; do
    port=$((MODEL_PORT + i))
    url="http://${SERVE_HOST}:${port}/v1"
    log="${LOG_DIR}/model_gpu${g}.log"
    check_port "${url}"
    echo "[general_eval] model on GPU ${g} at ${url} (log: ${log})"
    CUDA_VISIBLE_DEVICES="${g}" setsid "${VLLM}" serve "${MODEL}" \
      --host "${SERVE_HOST}" --port "${port}" --served-model-name general-eval \
      --tensor-parallel-size 1 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --trust-remote-code \
      --max-num-seqs 256 --api-server-count 4 --generation-config vllm > "${log}" 2>&1 &
    MODEL_PIDS+=("$!")
    ALL_PIDS+=("$!")
    MODEL_URLS+=("${url}")
    MODEL_LOGS+=("${log}")
    i=$((i + 1))
  done
}

echo "[general_eval] model=${MODEL} name=${NAME} benchmarks=${BENCHMARKS}"
echo "[general_eval] outputs: ${RUN_DIR}"
"${PYTHON}" "${GE_DIR}/prepare_data.py" --data-dir "${DATA_DIR}" --benchmarks "${BENCHMARKS}"

COMMON=(--data-dir "${DATA_DIR}" --run-dir "${RUN_DIR}" --benchmarks "${BENCHMARKS}")
INFER_PENDING="$("${PYTHON}" "${GE_DIR}/evaluate.py" infer "${COMMON[@]}" --count-pending | tail -n 1)"

if [[ -n "${JUDGE_GPUS}" ]]; then
  if (( INFER_PENDING > 0 )); then
    start_judge
  else
    JUDGE_PENDING="$("${PYTHON}" "${GE_DIR}/evaluate.py" score "${COMMON[@]}" --count-pending | tail -n 1)"
    if (( JUDGE_PENDING > 0 )); then
      start_judge
    fi
  fi
fi

if (( INFER_PENDING > 0 )); then
  echo "[general_eval] ${INFER_PENDING} questions to answer"
  start_models
  for k in "${!MODEL_PIDS[@]}"; do
    wait_ready "${MODEL_PIDS[k]}" "${MODEL_URLS[k]}" "${MODEL_LOGS[k]}"
  done
  "${PYTHON}" "${GE_DIR}/evaluate.py" infer "${COMMON[@]}" \
    --api-bases "$(IFS=','; echo "${MODEL_URLS[*]}")" --threads "${THREADS_PER_SERVER}"
  stop_servers "${MODEL_PIDS[@]}"
else
  echo "[general_eval] all answers present; skipping inference"
fi

SCORE_ARGS=("${COMMON[@]}" --name "${NAME}" --model "${MODEL}")
if [[ -z "${JUDGE_PID}" ]]; then
  JUDGE_PENDING="$("${PYTHON}" "${GE_DIR}/evaluate.py" score "${COMMON[@]}" --count-pending | tail -n 1)"
  if (( JUDGE_PENDING > 0 )); then
    start_judge
  fi
fi
if [[ -n "${JUDGE_PID}" ]]; then
  wait_ready "${JUDGE_PID}" "${JUDGE_URL}" "${JUDGE_LOG}"
  SCORE_ARGS+=(--judge-base "${JUDGE_URL}")
fi
"${PYTHON}" "${GE_DIR}/evaluate.py" score "${SCORE_ARGS[@]}"
