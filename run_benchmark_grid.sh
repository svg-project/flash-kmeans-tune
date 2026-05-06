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
# Large-D K_LIST trims 200000 to keep wall-clock manageable (centroid memory at
# B=1 K=200000 D=4096 fp32 is already 3 GiB).
LARGE_D_K_LIST=(
  256
  1024
  4096
  16384
  65536
)
D_LIST=(
  64
  128
  256
  512
  1024
  2048
  4096
)
B_LIST=(
  1
  32
)
DTYPE_LIST=(
  fp16
  fp32
)

WARMUP="${WARMUP:-1}"
REP="${REP:-5}"

mkdir -p "${OUT_DIR}"

for dtype in "${DTYPE_LIST[@]}"; do
  for b in "${B_LIST[@]}"; do
    for d in "${D_LIST[@]}"; do
      # Skip B=32 + D >= 2048: centroid tensor alone exceeds practical memory.
      if [ "${b}" -ge 32 ] && [ "${d}" -ge 2048 ]; then
        echo "--- skip B=${b} D=${d} (memory) ---"
        continue
      fi
      # Pick K_LIST based on D regime.
      if [ "${d}" -ge 1024 ]; then
        ks=("${LARGE_D_K_LIST[@]}")
      else
        ks=("${K_LIST[@]}")
      fi
      for k in "${ks[@]}"; do
        for n in "${N_LIST[@]}"; do
          echo "=== N=${n} K=${k} D=${d} B=${b} dtype=${dtype} ==="
          "${PYTHON_BIN}" "${ROOT_DIR}/benchmark_euclid_configs.py" \
            --n "${n}" \
            --k "${k}" \
            --d "${d}" \
            --batch-size "${b}" \
            --dtype "${dtype}" \
            --kernel auto \
            --warmup "${WARMUP}" \
            --rep "${REP}" \
            --flash-kmeans-root "${FLASH_KMEANS_ROOT}" \
            --output-dir "${OUT_DIR}"
        done
      done
    done
  done
done
