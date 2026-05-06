# flash-kmeans config tuning

This folder contains standalone scripts to tune Triton configs **without**
modifying the `flash-kmeans` codebase.

## Tune Euclid assign kernel

Example:
```
python tune_euclid_config.py --n 1000000 --k 200000 --d 512 --output best_euclid.json
```

Options:
- `--flash-kmeans-root`: path to the `flash-kmeans` repo root
- `--dtype`: `fp16` or `fp32`
- `--batch-size`: batch size (default 1)
- `--kernel`: `auto` (default) | `small_d` | `split_d`. `auto` picks `split_d`
  when `D > 512`, else `small_d`. The split-D kernel adds a `BLOCK_D` tile
  axis and is required for `D > 512` (or for any (D, dtype) combo whose
  shared-memory footprint exceeds the GPU's per-block budget).
- `--warmup` / `--rep`: timing parameters

## Benchmark grid (all configs)

This writes a JSONL file per workload that contains *all* configs and their
timings, plus a summary line with the best config.

Example:
```
bash run_benchmark_grid.sh
```

Per-workload file naming:
`N{N}_K{K}_D{D}_B{B}_{dtype}.jsonl`

Behavior:
- If the output file already exists, the benchmark is skipped by default.
- Use `--no-skip-existing` to overwrite.
- Output is written to a temporary file and atomically renamed on success.

Output JSON includes the best config and measured time. Split-D entries
additionally carry a `BLOCK_D` field and a `kernel: "split_d"` marker;
small-D entries carry `kernel: "small_d"` (no `BLOCK_D`), so existing
analysis tools that ignore the new fields still parse old grid_results.

## Focused sweep for fp32 / large-D

`run_focused_sweep.sh` runs only the (D, dtype) slices that are *not* covered
by the existing `grid_results_<gpu>` directories: fp32 small-D, and any
dtype + `D ∈ {1024, 2048, 4096}`. It parallelises across multiple GPUs by
default, using GPUs 1..N-1 (GPU 0 is reserved for ad-hoc correctness/perf
tests so timing on the sweeping GPUs is not contaminated). Override with
`SWEEP_GPUS="0 1 2 ..."` if you want a different mapping.

```
bash run_focused_sweep.sh
```

## Grid extensions

`run_benchmark_grid.sh` now sweeps `D ∈ {64, 128, 256, 512, 1024, 2048, 4096}`
and `dtype ∈ {fp16, fp32}`. The script automatically selects the small-D
or split-D kernel via `--kernel auto`. `(B = 32, D ≥ 2048)` combinations
are skipped to avoid host-memory blowups.
