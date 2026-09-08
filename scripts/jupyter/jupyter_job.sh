#!/bin/bash
# Runs on an MPI compute node. Starts a Jupyter Lab server for the vLLM
# tutorial notebooks and records how to reach it.
#
# Environment (set by jupyter.sub):
#   REPO_DIR   absolute path to the tutorial checkout on /lustre
#   VENV       python venv to run in           (default: $REPO_DIR/.venv)
#   SIF        Singularity image; if set, runs in the container instead of VENV
#   HF_HOME    model cache (must be pre-populated -- nodes have no internet)
#   JOB_ID     "$(ClusterId).$(ProcId)", set by Condor

set -o errexit

: "${REPO_DIR:?REPO_DIR must be set}"
VENV="${VENV:-${REPO_DIR}/.venv}"

# Condor expands $(ClusterId) in `environment` to the GlobalJobId form
# ("sched#17522597.0#7734"), which is useless as a filename and does not match
# what condor_q reports. Read the real ClusterId.ProcId out of the job ad.
if [[ -r "${_CONDOR_JOB_AD:-}" ]]; then
  CLUSTER=$(awk '/^ClusterId/ {print $3; exit}' "${_CONDOR_JOB_AD}")
  PROC=$(awk '/^ProcId/ {print $3; exit}' "${_CONDOR_JOB_AD}")
  JOB_ID="${CLUSTER}.${PROC}"
fi
JOB_ID="${JOB_ID//[^0-9.]/_}"   # never let stray characters into the path

INFO_FILE="${REPO_DIR}/.jupyter/session-${JOB_ID}.env"

mkdir -p "$(dirname "${INFO_FILE}")"

# A fixed 8888 collides with other users on shared nodes; take a free one.
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
TOKEN=$(python3 -c 'import secrets;print(secrets.token_hex(24))')

cat > "${INFO_FILE}" <<INFO
JOB_ID=${JOB_ID}
NODE=$(hostname -f)
PORT=${PORT}
TOKEN=${TOKEN}
INFO

echo "############## ############## ##############"
echo "Jupyter for vllm-tutorial: job ${JOB_ID} on $(hostname -f):${PORT}"
nvidia-smi || true
echo "############## ############## ##############"

# vLLM compiles kernels on first load and wants a writable cache. /tmp is local
# to the node and fast; /lustre is shared but slow for many small files.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# --- Three settings vLLM needs to work in a Jupyter kernel on these nodes. ---
# All three were found by running the tutorial notebook's own cell order; drop
# any one of them and LLM(...) fails. See README.md "vLLM in a kernel".
#
# 1. The notebook initializes CUDA (the prefill/decode benchmark) before it
#    builds an LLM. vLLM then switches to `spawn`, and spawn cannot re-import
#    __main__ inside a kernel -> "bootstrapping phase" RuntimeError. Running
#    the engine in-process avoids subprocesses altogether.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
#
# 2. Compute nodes have no system CUDA toolkit (/usr/local/cuda is absent), but
#    pip installed nvcc into the venv.
export CUDA_HOME="${CUDA_HOME:-${VENV}/lib/python3.10/site-packages/nvidia/cu13}"
#
# 3. FlashInfer JIT-compiles its sampling kernels on first use, and its vendored
#    CCCL headers reject that nvcc ("CUDA compiler and CUDA toolkit headers are
#    incompatible"). It also compiles into ~/.cache, which is over quota.
#    vLLM's native sampler needs no compiler.
export VLLM_USE_FLASHINFER_SAMPLER=0

JUPYTER_ARGS=(
  jupyter lab
  --no-browser
  --ip=0.0.0.0
  --port="${PORT}"
  --ServerApp.token="${TOKEN}"
  --ServerApp.root_dir="${REPO_DIR}"
  --ServerApp.allow_origin='*'
)

if [[ -n "${SIF:-}" ]]; then
  exec singularity exec --nv \
    --bind "${REPO_DIR}" \
    ${HF_HOME:+--bind "${HF_HOME}"} \
    --env HF_HOME="${HF_HOME:-}" \
    --env VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    --env VLLM_USE_FLASHINFER_SAMPLER=0 \
    "${SIF}" "${JUPYTER_ARGS[@]}"
else
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  cd "${REPO_DIR}"
  exec "${JUPYTER_ARGS[@]}"
fi
