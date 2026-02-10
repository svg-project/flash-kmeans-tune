#!/usr/bin/env bash
set -euo pipefail

# Benchmark grid for euclid assign kernel configs.
# Adjust arrays below to trade coverage vs runtime.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/grid_results}"
FLASH_KMEANS_ROOT="${FLASH_KMEANS_ROOT:-${ROOT_DIR}/../flash-kmeans}"

N_LIST=(
  65536
  262144
  1048576
)
K_LIST=(
  256
  1024
  4096
  16384
  65536
  200000
)
D_LIST=(
  64
  128
  256
  512
)
B_LIST=(
  1
  32
)
DTYPE_LIST=(
  fp16
)

WARMUP="${WARMUP:-1}"
REP="${REP:-5}"

mkdir -p "${OUT_DIR}"

for dtype in "${DTYPE_LIST[@]}"; do
  for b in "${B_LIST[@]}"; do
    for d in "${D_LIST[@]}"; do
      for k in "${K_LIST[@]}"; do
        for n in "${N_LIST[@]}"; do
          echo "=== N=${n} K=${k} D=${d} B=${b} dtype=${dtype} ==="
          "${PYTHON_BIN}" "${ROOT_DIR}/benchmark_euclid_configs.py" \
            --n "${n}" \
            --k "${k}" \
            --d "${d}" \
            --batch-size "${b}" \
            --dtype "${dtype}" \
            --warmup "${WARMUP}" \
            --rep "${REP}" \
            --flash-kmeans-root "${FLASH_KMEANS_ROOT}" \
            --output-dir "${OUT_DIR}"
        done
      done
    done
  done
done
