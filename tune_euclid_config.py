from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional


def _resolve_repo_root(cli_root: Optional[str]) -> str:
    if cli_root:
        return os.path.abspath(cli_root)
    env_root = os.environ.get("FLASH_KMEANS_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "flash-kmeans"))
    return default_root


def _load_flash_kmeans(repo_root: str):
    repo_root = os.path.abspath(repo_root)
    if not os.path.exists(os.path.join(repo_root, "flash_kmeans")):
        raise FileNotFoundError(f"flash_kmeans package not found under: {repo_root}")
    sys.path.insert(0, repo_root)
    from flash_kmeans.assign_euclid_triton import _TUNE_CONFIGS, _euclid_assign_kernel

    return _TUNE_CONFIGS, _euclid_assign_kernel


def _do_bench(run, warmup: int, rep: int):
    try:
        import triton.testing as tt

        return tt.do_bench(run, warmup=warmup, rep=rep)
    except Exception:
        import torch

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(rep, 1)):
            run()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / max(rep, 1)


def main():
    parser = argparse.ArgumentParser(description="Tune euclid assign Triton config")
    parser.add_argument("--n", type=int, required=True, help="Number of points (N)")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters (K)")
    parser.add_argument("--d", type=int, required=True, help="Dimensionality (D)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument("--flash-kmeans-root", type=str, default=None)
    parser.add_argument("--output", type=str, default="best_euclid.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = _resolve_repo_root(args.flash_kmeans_root)
    tune_configs, kernel = _load_flash_kmeans(repo_root)

    import torch
    import triton

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    torch.manual_seed(args.seed)

    B = args.batch_size
    N, K, D = args.n, args.k, args.d

    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    centroids = torch.randn(B, K, D, device="cuda", dtype=dtype)
    x_sq = (x.to(torch.float32) ** 2).sum(-1)
    c_sq = (centroids.to(torch.float32) ** 2).sum(-1)
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    best_cfg = None
    best_ms = None

    print(f"Tuning configs for N={N} K={K} D={D} B={B} dtype={args.dtype}")
    start_total = time.time()

    skipped = 0
    for cfg in tune_configs:
        BN = cfg.kwargs["BLOCK_N"]
        BK = cfg.kwargs["BLOCK_K"]

        def _run():
            kernel[grid](
                x,
                centroids,
                x_sq,
                c_sq,
                out,
                B,
                N,
                K,
                D,
                stride_x_b,
                stride_x_n,
                stride_x_d,
                stride_c_b,
                stride_c_k,
                stride_c_d,
                stride_xsq_b,
                stride_xsq_n,
                stride_csq_b,
                stride_csq_k,
                stride_out_b,
                stride_out_n,
                BLOCK_N=BN,
                BLOCK_K=BK,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )

        try:
            ms = _do_bench(_run, warmup=args.warmup, rep=args.rep)
        except Exception as e:
            msg = str(e)
            if "out of resource" in msg.lower() or "outofresources" in type(e).__name__.lower():
                print(
                    f"BN={BN:3d} BK={BK:3d} warps={cfg.num_warps} stages={cfg.num_stages} -> skipped (OOR)"
                )
                skipped += 1
                continue
            raise
        print(
            f"BN={BN:3d} BK={BK:3d} warps={cfg.num_warps} stages={cfg.num_stages} -> {ms:.3f} ms"
        )
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    total_s = time.time() - start_total
    if best_cfg is None:
        raise RuntimeError("No valid config found.")

    result = {
        "N": N,
        "K": K,
        "D": D,
        "B": B,
        "dtype": args.dtype,
        "BLOCK_N": int(best_cfg.kwargs["BLOCK_N"]),
        "BLOCK_K": int(best_cfg.kwargs["BLOCK_K"]),
        "num_warps": int(best_cfg.num_warps),
        "num_stages": int(best_cfg.num_stages),
        "time_ms": float(best_ms),
        "tuning_seconds": float(total_s),
    }

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Best config saved to: {out_path}")
    if skipped:
        print(f"Skipped {skipped} configs due to resource limits.")


if __name__ == "__main__":
    main()
