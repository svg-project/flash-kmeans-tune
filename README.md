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

Output JSON includes the best config and measured time.
