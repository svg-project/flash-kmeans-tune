#!/usr/bin/env bash
# Focused tuning sweep targeted at the gaps the existing fp16 D≤512 grid did
# not cover: fp32 + small-D, and any-dtype + large-D. Parallelizes across the
# CUDA devices listed in SWEEP_GPUS (default: 1..N-1; GPU 0 is reserved for
# interactive correctness/perf tests during the sweep so timing data is not
# polluted by competing kernels).
#
# Output: ${OUT_DIR:-grid_results_h200_v2}/
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/grid_results_h200_v2}"
FLASH_KMEANS_ROOT="${FLASH_KMEANS_ROOT:-${ROOT_DIR}/../flash-kmeans}"

# Determine which GPUs the sweep is allowed to use. By default reserve GPU 0
# for ad-hoc tests and use 1..N-1. Override via SWEEP_GPUS="0 1 2 ...".
ALL_NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
ALL_NGPU=${ALL_NGPU:-1}
if [ -z "${SWEEP_GPUS:-}" ]; then
  if [ "${ALL_NGPU}" -ge 2 ]; then
    SWEEP_GPUS=""
    for ((i=1; i<ALL_NGPU; i++)); do
      SWEEP_GPUS+="${i} "
    done
  else
    SWEEP_GPUS="0"
  fi
fi
read -r -a GPU_ARRAY <<< "${SWEEP_GPUS}"
NGPU=${#GPU_ARRAY[@]}
echo "Sweep using GPUs: ${GPU_ARRAY[*]} (NGPU=${NGPU})"

mkdir -p "${OUT_DIR}"

N_LIST=(65536 262144 1048576)
K_SMALL_D=(256 1024 4096 16384 65536 200000)
K_LARGE_D=(256 1024 4096 16384 65536)
WARMUP="${WARMUP:-1}"
REP="${REP:-3}"

# Build the workload list as (dtype, D) slices. fp16 D≤512 already covered by
# the prior grid_results_h200/, no need to re-tune that. fp32 small-D and any
# large-D are new.
SLICES=()
for d in 64 128 256 512; do
  SLICES+=("fp32 ${d}")
done
for d in 1024 2048 4096; do
  for dt in fp16 fp32; do
    SLICES+=("${dt} ${d}")
  done
done

run_slice() {
  local gpu_id="$1"; shift
  local dtype="$1"; shift
  local d="$1"; shift
  local kchoice
  if [ "${d}" -le 512 ]; then
    kchoice=small_d
    ks=("${K_SMALL_D[@]}")
  else
    kchoice=split_d
    ks=("${K_LARGE_D[@]}")
  fi
  for k in "${ks[@]}"; do
    for n in "${N_LIST[@]}"; do
      echo "[GPU${gpu_id}] === N=${n} K=${k} D=${d} B=1 dtype=${dtype} kernel=${kchoice} ==="
      CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${ROOT_DIR}/benchmark_euclid_configs.py" \
        --n "${n}" --k "${k}" --d "${d}" --batch-size 1 \
        --dtype "${dtype}" --kernel "${kchoice}" \
        --warmup "${WARMUP}" --rep "${REP}" \
        --flash-kmeans-root "${FLASH_KMEANS_ROOT}" \
        --output-dir "${OUT_DIR}"
    done
  done
}

# Round-robin slices to allowed GPUs, run in parallel, wait for all.
i=0
PIDS=()
for slice in "${SLICES[@]}"; do
  gpu="${GPU_ARRAY[$(( i % NGPU ))]}"
  read -r dtype d <<< "${slice}"
  ( run_slice "${gpu}" "${dtype}" "${d}" > "${OUT_DIR}/log_gpu${gpu}_${dtype}_D${d}.txt" 2>&1 ) &
  PIDS+=($!)
  i=$((i + 1))
  # Cap concurrent jobs at NGPU.
  if [ "${#PIDS[@]}" -ge "${NGPU}" ]; then
    wait "${PIDS[0]}"
    PIDS=("${PIDS[@]:1}")
  fi
done
wait
echo "All slices complete. Results under ${OUT_DIR}/"
