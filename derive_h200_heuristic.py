"""Parse the JSONL summaries from a focused sweep and emit a Python
heuristic body for H200.

Outputs decision tables for:
  - small-D fp32 (D ∈ {64, 128, 256, 512})
  - split-D any-dtype (D ∈ {1024, 2048, 4096})

For each cell in the sweep, picks the best config (lowest time_ms) and
prints a Python snippet keyed on (D, K, N) with branches that match what
``_heuristic_euclid_config_h200_smallD`` / ``_heuristic_euclid_config_h200_largeD``
expect to consume.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Tuple


def _load_summaries(grid_dir: str):
    """Yield the summary line of each JSONL workload file."""
    for name in sorted(os.listdir(grid_dir)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(grid_dir, name)
        last = None
        with open(path) as f:
            for line in f:
                last = line
        if not last:
            continue
        try:
            entry = json.loads(last)
        except Exception:
            continue
        if entry.get("summary") and entry.get("best") is not None:
            yield entry


def _bucket_K(K: int) -> str:
    if K <= 256:
        return "K<=256"
    if K <= 1024:
        return "K<=1024"
    if K <= 4096:
        return "K<=4096"
    if K <= 16384:
        return "K<=16384"
    if K <= 65536:
        return "K<=65536"
    return "K>65536"


def _bucket_N(N: int) -> str:
    if N <= 65536:
        return "N<=65536"
    if N <= 262144:
        return "N<=262144"
    return "N>262144"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    args = ap.parse_args()

    by_segment: Dict[Tuple[str, str, int, str], list] = defaultdict(list)
    raw_by_key: Dict[Tuple[str, str, int, int, int, int], dict] = {}

    for entry in _load_summaries(args.grid_dir):
        kernel = entry.get("kernel", "small_d")
        dtype = entry["dtype"]
        D = int(entry["D"])
        K = int(entry["K"])
        N = int(entry["N"])
        B = int(entry["B"])
        best = entry["best"]
        seg = (kernel, dtype, D, _bucket_K(K))
        by_segment[seg].append({
            "N": N, "K": K, "B": B,
            "BN": int(best["BLOCK_N"]),
            "BK": int(best["BLOCK_K"]),
            "BD": int(best.get("BLOCK_D", 0)),
            "warps": int(best["num_warps"]),
            "stages": int(best["num_stages"]),
            "time_ms": float(best["time_ms"]),
        })
        raw_by_key[(kernel, dtype, D, K, N, B)] = best

    print(f"# Loaded {sum(len(v) for v in by_segment.values())} summary entries")
    print()

    # --- Print best config per (kernel, dtype, D) bucket of K ---
    print("=" * 80)
    print("Per-segment majority config (kernel, dtype, D, K-bucket → most-common best)")
    print("=" * 80)
    for seg in sorted(by_segment):
        kernel, dtype, D, kbucket = seg
        rows = by_segment[seg]
        # Vote by config tuple
        from collections import Counter
        if kernel == "split_d":
            votes = Counter((r["BN"], r["BK"], r["BD"], r["warps"], r["stages"]) for r in rows)
        else:
            votes = Counter((r["BN"], r["BK"], r["warps"], r["stages"]) for r in rows)
        top, count = votes.most_common(1)[0]
        avg_ms = sum(r["time_ms"] for r in rows) / len(rows)
        n = len(rows)
        if kernel == "split_d":
            print(f"  {kernel:8s} {dtype:5s} D={D:>4} {kbucket:>11}  → BN={top[0]:3} BK={top[1]:3} BD={top[2]:3} W={top[3]} S={top[4]}  votes={count}/{n}  avg={avg_ms:.3f} ms")
        else:
            print(f"  {kernel:8s} {dtype:5s} D={D:>4} {kbucket:>11}  → BN={top[0]:3} BK={top[1]:3} W={top[2]} S={top[3]}  votes={count}/{n}  avg={avg_ms:.3f} ms")

    # --- Print full per-cell best for spot-checking ---
    print()
    print("=" * 80)
    print("Per-(N, K) best (one row per workload)")
    print("=" * 80)
    for k in sorted(raw_by_key):
        kernel, dtype, D, K, N, B = k
        b = raw_by_key[k]
        if kernel == "split_d":
            print(f"  {kernel:8s} {dtype:5s} D={D:>4} K={K:>6} N={N:>7} B={B} → BN={b['BLOCK_N']:3} BK={b['BLOCK_K']:3} BD={b.get('BLOCK_D', 0):3} W={b['num_warps']} S={b['num_stages']}  {b['time_ms']:.3f} ms")
        else:
            print(f"  {kernel:8s} {dtype:5s} D={D:>4} K={K:>6} N={N:>7} B={B} → BN={b['BLOCK_N']:3} BK={b['BLOCK_K']:3} W={b['num_warps']} S={b['num_stages']}  {b['time_ms']:.3f} ms")


if __name__ == "__main__":
    main()
