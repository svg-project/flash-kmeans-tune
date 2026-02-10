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


def _default_output_path(output_dir: str, n: int, k: int, d: int, b: int, dtype: str) -> str:
    name = f"N{n}_K{k}_D{d}_B{b}_{dtype}.jsonl"
    return os.path.join(output_dir, name)


def main():
    parser = argparse.ArgumentParser(description="Benchmark all Triton configs for euclid assign kernel")
    parser.add_argument("--n", type=int, required=True, help="Number of points (N)")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters (K)")
    parser.add_argument("--d", type=int, required=True, help="Dimensionality (D)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument("--flash-kmeans-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="grid_results")
    parser.add_argument("--output", type=str, default=None, help="Override output file path")
    parser.add_argument("--append", action="store_true", help="Append to existing output file")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", help="Skip if output file exists")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Do not skip even if output file exists")
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(skip_existing=True)
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

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or _default_output_path(output_dir, N, K, D, B, args.dtype)
    if args.skip_existing and (not args.append) and os.path.exists(output_path):
        print(f"Skip existing output: {output_path}")
        return
    mode = "a" if args.append else "w"
    tmp_path = None
    if not args.append:
        tmp_path = output_path + f".tmp.{os.getpid()}"
        mode = "w"

    best_cfg = None
    best_ms = None
    skipped = 0
    start_total = time.time()

    try:
        target_path = tmp_path or output_path
        with open(target_path, mode, encoding="utf-8") as f:
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

                entry = {
                    "N": N,
                    "K": K,
                    "D": D,
                    "B": B,
                    "dtype": args.dtype,
                    "BLOCK_N": int(BN),
                    "BLOCK_K": int(BK),
                    "num_warps": int(cfg.num_warps),
                    "num_stages": int(cfg.num_stages),
                }

                try:
                    ms = _do_bench(_run, warmup=args.warmup, rep=args.rep)
                    entry["time_ms"] = float(ms)
                    entry["status"] = "ok"
                    if best_ms is None or ms < best_ms:
                        best_ms = ms
                        best_cfg = entry.copy()
                except Exception as e:
                    msg = str(e)
                    if "out of resource" in msg.lower() or "outofresources" in type(e).__name__.lower():
                        entry["status"] = "oor"
                        skipped += 1
                    else:
                        entry["status"] = "error"
                        entry["error"] = msg
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    if entry["status"] == "error":
                        raise
                    continue

                f.write(json.dumps(entry) + "\n")
                f.flush()

            summary = {
                "summary": True,
                "N": N,
                "K": K,
                "D": D,
                "B": B,
                "dtype": args.dtype,
                "best": best_cfg,
                "best_time_ms": float(best_ms) if best_ms is not None else None,
                "tuning_seconds": float(time.time() - start_total),
                "skipped": skipped,
            }
            f.write(json.dumps(summary) + "\n")
            f.flush()

        if tmp_path:
            os.replace(tmp_path, output_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    print(f"Wrote: {output_path}")
    if best_cfg is not None:
        print(
            "Best: "
            f"BN={best_cfg['BLOCK_N']} BK={best_cfg['BLOCK_K']} "
            f"warps={best_cfg['num_warps']} stages={best_cfg['num_stages']} "
            f"({best_ms:.3f} ms)"
        )
    if skipped:
        print(f"Skipped {skipped} configs due to resource limits.")


if __name__ == "__main__":
    main()
