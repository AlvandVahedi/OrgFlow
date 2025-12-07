# import argparse
# import json
# import os
# import shlex
# import subprocess
# import sys
# import time
# from pathlib import Path
# from typing import List, Dict, Any, Optional
#
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# # -----------------------------
# # Helper utilities
# # -----------------------------
#
# def run_cmd(cmd: str, env: Optional[dict] = None) -> int:
#     """Run a shell command, stream output to console. Returns exit code."""
#     print(f"[CMD] {cmd}")
#     process = subprocess.Popen(
#         shlex.split(cmd),
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         text=True,
#         env=env or os.environ.copy(),
#         bufsize=1,
#         universal_newlines=True,
#     )
#     for line in process.stdout:
#         sys.stdout.write(line)
#     process.wait()
#     return process.returncode
#
# def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
#     if path.exists():
#         with open(path, "r") as f:
#             return json.load(f)
#     return None
#
# def infer_num_samples_from_consolidated(consolidated_path: Path) -> int:
#     """
#     consolidated_{task}.pt is a dict saved by _consolidate().
#     It stores lists for keys like 'atom_types', 'frac_coords', etc.
#     For non-trajectory tasks, these were concatenated and then wrapped into a list at end via _list_of_dicts_to_dict_of_lists.
#     In consolidate(), they later do 'consolidated = {k: v[0] for k, v in consolidated.items()}' when saving eval_pt.
#     Here we just load consolidated and count elements of 'num_atoms'[0] or 'atom_types'[0].
#     """
#     r = torch.load(consolidated_path, map_location="cpu")
#     # r is dict[str, list[Tensor]] from consolidate()
#     # Grab first eval's tensor length along dim 0.
#     for key in ["num_atoms", "atom_types", "frac_coords", "lengths"]:
#         if key in r and len(r[key]) > 0:
#             v0 = r[key][0]
#             if isinstance(v0, torch.Tensor):
#                 return int(v0.shape[0])
#     # Fallback: unknown
#     return -1
#
# def pick_quality_from_metrics(metrics: Dict[str, Any], task: str) -> (str, float):
#     """
#     Heuristic: find a single "primary" metric.
#     For 'generate': look for common keys from compute_generation_metrics().
#     For 'reconstruct': prefer the single-eval JSON if present, then choose a low-is-better metric like MAE/RMSE if available.
#     """
#     # Flat dict of metrics (already)
#     # Prefer explicit keys if present:
#     candidates_high_is_better = [
#         "valid_frac", "success_rate", "stability", "match_rate", "unique_frac"
#     ]
#     candidates_low_is_better = [
#         "mae", "rmse", "lattice_mae", "angle_mae", "length_mae"
#     ]
#
#     # Try exact match
#     for k in candidates_high_is_better:
#         if k in metrics:
#             return k, float(metrics[k])
#     for k in candidates_low_is_better:
#         if k in metrics:
#             return k, float(metrics[k])
#
#     # Next, try to detect FID/IS-like names if user provided custom metrics
#     for k in metrics:
#         lk = k.lower()
#         if "fid" in lk or "inception" in lk or "is" == lk:
#             return k, float(metrics[k])
#
#     # Otherwise, pick the first numeric
#     for k, v in metrics.items():
#         try:
#             return k, float(v)
#         except Exception:
#             continue
#     raise RuntimeError("Could not pick a quality metric from metrics dict. "
#                        "Use --metric-key to specify explicitly.")
#
# def flatten_recon_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     reconstruction saves two jsons: *_single.json and *_multi.json.
#     The user likely wants single-eval. If provided a dict loaded from *_single.json,
#     keep top-level numeric keys. Strip 'val/' or 'test/' prefixes if present.
#     """
#     out = {}
#     for k, v in d.items():
#         if isinstance(v, (int, float)) and not (isinstance(v, bool)):
#             key = k.split("/")[-1]
#             out[key] = v
#     return out
#
# # -----------------------------
# # Main sweep
# # -----------------------------
#
# def main():
#     parser = argparse.ArgumentParser(description="Quality vs Speed sweep for sampling.")
#     parser.add_argument("checkpoint", type=str, help="Path to .ckpt")
#     parser.add_argument("--scripts_root", type=str, default="scripts_model",
#                         help="Where evaluate.py lives")
#     parser.add_argument("--task", type=str, choices=["reconstruct", "generate"], default="reconstruct",
#                         help="Which task to benchmark")
#     parser.add_argument("--stage", type=str, choices=["train", "val", "test"], default="test")
#     parser.add_argument("--subdir", type=str, default="sweep_eval", help="Subdir next to checkpoint to store evals")
#     parser.add_argument("--num_steps_list", type=int, nargs="+",
#                         default=[50, 100, 200, 400, 800, 1000],
#                         help="Sampling step counts to compare")
#     parser.add_argument("--single_gpu", action="store_true", default=True,
#                         help="Force single GPU for fair wall-clock")
#     parser.add_argument("--batch_size", type=int, default=None, help="Optional override for eval batch size")
#     parser.add_argument("--gen_num_samples", type=int, default=10000, help="Generation-only: number of samples to draw")
#     parser.add_argument("--metric_key", type=str, default=None, help="Explicit metric key to use on the Y axis")
#     parser.add_argument("--limit_predict_batches", type=str, default="1.", help="Lightning limit_predict_batches for reconstruct task")
#     parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging inside old_eval_metrics")
#     parser.add_argument("--plots_prefix", type=str, default="tradeoff", help="Prefix for saved plot files")
#     args = parser.parse_args()
#
#     checkpoint = Path(args.checkpoint).resolve()
#     ckpt_dir = checkpoint.parent
#     target_dir = (ckpt_dir / args.subdir).resolve()
#     target_dir.mkdir(parents=True, exist_ok=True)
#
#     # Where evaluate.py lives
#     eval_py = Path(args.scripts_root) / "evaluate.py"
#     if not eval_py.exists():
#         print(f"Could not find evaluate.py at {eval_py}. Please set --scripts_root.")
#         sys.exit(1)
#
#     rows = []
#     env = os.environ.copy()
#
#     for ns in tqdm(args.num_steps_list, desc="Sweep num_steps"):
#         # 1) Run predict step for this num_steps, measure wall-clock
#         t0 = time.perf_counter()
#
#         if args.task == "reconstruct":
#             # We let your existing loader pick the dataset for the stage.
#             cmd = f"python {eval_py} reconstruct {shlex.quote(str(checkpoint))} " \
#                   f"--stage {args.stage} --num_evals 1 --limit_predict_batches \"{args.limit_predict_batches}\" " \
#                   f"--num_steps {ns} {'--single_gpu' if args.single_gpu else '--multi_gpu'} --subdir {shlex.quote(args.subdir)}"
#             if args.batch_size is not None:
#                 cmd += f" --batch_size {args.batch_size}"
#         else:
#             # generate
#             cmd = f"python {eval_py} generate {shlex.quote(str(checkpoint))} " \
#                   f"--num_samples {args.gen_num_samples} --num_steps {ns} " \
#                   f"{'--single_gpu' if args.single_gpu else '--multi_gpu'} --subdir {shlex.quote(args.subdir)}"
#
#         rc = run_cmd(cmd, env=env)
#         if rc != 0:
#             print(f"[ERROR] Command failed with code {rc}. Aborting.")
#             sys.exit(rc)
#
#         elapsed = time.perf_counter() - t0
#
#         # 2) Consolidate
#         cmd_cons = f"python {eval_py} consolidate {shlex.quote(str(checkpoint))} --subdir {shlex.quote(args.subdir)}"
#         rc = run_cmd(cmd_cons, env=env)
#         if rc != 0:
#             print(f"[ERROR] Consolidate failed with code {rc}. Aborting.")
#             sys.exit(rc)
#
#         # 3) Compute metrics without wandb
#         no_wandb_flag = "--do_not_log_wandb" if args.no_wandb else ""
#         cmd_metrics = f"python {eval_py} old_eval_metrics {shlex.quote(str(checkpoint))} {no_wandb_flag} " \
#                       f"--subdir {shlex.quote(args.subdir)} --stage {args.stage}"
#         rc = run_cmd(cmd_metrics, env=env)
#         if rc != 0:
#             print(f"[ERROR] Metrics failed with code {rc}. Aborting.")
#             sys.exit(rc)
#
#         # 4) Parse metrics JSONs that evaluate.py writes into target_dir
#         #    and infer the number of samples we processed for fair time/sample.
#
#         # Consolidated path
#         consolidated_path = target_dir / f"consolidated_{'generate' if args.task=='generate' else 'reconstruct'}.pt"
#         num_elems = infer_num_samples_from_consolidated(consolidated_path)
#
#         # Metrics paths
#         if args.task == "generate":
#             metrics_path = target_dir / "old_eval_metrics_generate.json"
#             metrics = read_json_if_exists(metrics_path) or {}
#             # compute_generation_metrics likely writes a single JSON with many keys
#             flat_metrics = {}
#             for k, v in metrics.items():
#                 # flatten stage prefix if any
#                 key = k.split("/")[-1]
#                 flat_metrics[key] = v
#         else:
#             # reconstruct has both multi and single
#             single_path = target_dir / "old_eval_metrics_reconstruct_single.json"
#             multi_path = target_dir / "old_eval_metrics_reconstruct_multi.json"
#             if single_path.exists():
#                 metrics = read_json_if_exists(single_path) or {}
#             elif multi_path.exists():
#                 metrics = read_json_if_exists(multi_path) or {}
#             else:
#                 metrics = {}
#             flat_metrics = flatten_recon_metrics(metrics)
#
#         # Choose quality metric
#         if args.metric_key is not None:
#             metric_key = args.metric_key
#             if metric_key not in flat_metrics:
#                 raise RuntimeError(f"Metric key '{metric_key}' not found in metrics: {list(flat_metrics.keys())}")
#             metric_val = float(flat_metrics[metric_key])
#         else:
#             metric_key, metric_val = pick_quality_from_metrics(flat_metrics, args.task)
#
#         # Prefer speed as time per sample, else total time
#         if num_elems and num_elems > 0:
#             sec_per_sample = elapsed / float(num_elems)
#         else:
#             sec_per_sample = float('nan')
#
#         # Approximate NFE: num_steps for Euler method. For other methods this is a proxy.
#         nfe = ns
#
#         row = dict(
#             num_steps=ns,
#             elapsed_seconds=elapsed,
#             sec_per_sample=sec_per_sample,
#             metric_key=metric_key,
#             quality=float(metric_val),
#             nfe=int(nfe),
#             num_items=int(num_elems if num_elems > 0 else 0),
#         )
#         rows.append(row)
#
#         # Save intermediate CSV to avoid loss on crash
#         df = pd.DataFrame(rows)
#         csv_path = target_dir / f"{args.plots_prefix}_sweep_{args.task}.csv"
#         df.to_csv(csv_path, index=False)
#
#     # Final CSV
#     df = pd.DataFrame(rows).sort_values("num_steps")
#     csv_path = target_dir / f"{args.plots_prefix}_sweep_{args.task}.csv"
#     df.to_csv(csv_path, index=False)
#     print(f"[OK] Wrote CSV to {csv_path}")
#
#     # -----------------------------
#     # Plots
#     # -----------------------------
#     # 1) Quality vs elapsed time
#     plt.figure()
#     plt.plot(df["elapsed_seconds"], df["quality"], marker="o")
#     plt.xlabel("Wall-clock seconds (total)")
#     plt.ylabel(df["metric_key"].iloc[0])
#     plt.title(f"Quality vs Time ({args.task}, stage={args.stage})")
#     plt.grid(True)
#     out1 = target_dir / f"{args.plots_prefix}_{args.task}_quality_vs_time.png"
#     plt.savefig(out1, bbox_inches="tight")
#     plt.close()
#
#     # 2) Quality vs sec/sample (if available)
#     if df["sec_per_sample"].notna().any():
#         plt.figure()
#         plt.plot(df["sec_per_sample"], df["quality"], marker="o")
#         plt.xlabel("Seconds per sample")
#         plt.ylabel(df["metric_key"].iloc[0])
#         plt.title(f"Quality vs Sec/Sample ({args.task}, stage={args.stage})")
#         plt.grid(True)
#         out2 = target_dir / f"{args.plots_prefix}_{args.task}_quality_vs_sec_per_sample.png"
#         plt.savefig(out2, bbox_inches="tight")
#         plt.close()
#     else:
#         out2 = None
#
#     # 3) Quality vs NFE (≈ num_steps for Euler)
#     plt.figure()
#     plt.plot(df["nfe"], df["quality"], marker="o")
#     plt.xlabel("NFE (≈ num_steps)")
#     plt.ylabel(df["metric_key"].iloc[0])
#     plt.title(f"Quality vs NFE ({args.task}, stage={args.stage})")
#     plt.grid(True)
#     out3 = target_dir / f"{args.plots_prefix}_{args.task}_quality_vs_nfe.png"
#     plt.savefig(out3, bbox_inches="tight")
#     plt.close()
#
#     print("[OK] Saved plots:")
#     print(f"  - {out1}")
#     if out2:
#         print(f"  - {out2}")
#     print(f"  - {out3}")
#
#
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm


def run(cmd):
    # Print short, exact command for debugging
    print("[cmd]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def find_match_rate(metrics_json_path: Path) -> float:
    """
    Read old_eval_metrics_reconstruct_single.json and return match_rate.
    This is robust to small key-name differences.
    """
    with open(metrics_json_path, "r") as f:
        data = json.load(f)

    # Try common names first
    for k in ["match_rate", "matchrate", "match-rate"]:
        if k in data:
            return float(data[k])

    # Fallback: search keys that look like match rate
    lower = {k.lower(): k for k in data.keys()}
    for lk, orig in lower.items():
        if "match" in lk and "rate" in lk:
            return float(data[orig])

    # If nothing found, raise with available keys for quick debugging
    raise KeyError(
        f"Could not find match_rate in {metrics_json_path}. Available keys: {list(data.keys())}"
    )


def infer_num_samples_from_consolidated(consolidated_path: Path) -> int:
    """
    consolidated_reconstruct.pt is a dict saved by evaluate.py consolidate.
    Use tensor sizes to infer how many structures were processed.
    """
    if not consolidated_path.exists():
        return -1
    payload = torch.load(consolidated_path, map_location="cpu")
    for key in ["num_atoms", "atom_types", "frac_coords", "lengths"]:
        if key in payload and len(payload[key]) > 0:
            sample = payload[key][0]
            if isinstance(sample, torch.Tensor):
                return int(sample.shape[0])
    return -1


def read_wallclock_meta(target_dir: Path) -> tuple[float, float]:
    meta_path = target_dir / "wallclock.json"
    if not meta_path.exists():
        return float("nan"), float("nan")
    with meta_path.open("r") as f:
        data = json.load(f)
    return float(data.get("elapsed_seconds", float("nan"))), float(
        data.get("sec_per_sample", float("nan"))
    )


def main():
    parser = argparse.ArgumentParser(description="Sweep num_steps for reconstruction, consolidate, evaluate, and log match_rate.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to .ckpt")
    parser.add_argument("--num_steps", nargs="+", required=True, type=int, help="List of num_steps to sweep, e.g. 5 10 20 40")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--stage", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--inference_anneal_slope", type=float, default=1.0)
    parser.add_argument("--limit_predict_batches", type=str, default="1.")
    parser.add_argument("--single_gpu", action="store_true", help="Pass --single_gpu to evaluate.py")
    parser.add_argument("--base_subdir", type=str, default="eval_sweep_steps", help="Base subdir under the checkpoint folder where results are written")
    parser.add_argument("--python_bin", type=str, default="python", help="Python executable to call evaluate.py")
    parser.add_argument("--evaluate_py", type=Path, default=Path("scripts_model/evaluate.py"), help="Path to evaluate.py")
    parser.add_argument("--skip_if_done", action="store_true", help="Skip a step if old_eval_metrics_reconstruct_single.json already exists")

    args = parser.parse_args()

    # Where to collect summary artifacts (CSV + plot)
    sweep_root = args.checkpoint.parent / args.base_subdir
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows = []

    pbar = tqdm(args.num_steps, desc="sweep")
    for steps in pbar:
        pbar.set_description(f"sweep | steps={steps}")
        # Make a unique subdir per num_steps
        subdir = f"{args.base_subdir}/steps_{steps}"
        target_dir = args.checkpoint.parent / subdir
        metrics_json = target_dir / "old_eval_metrics_reconstruct_single.json"
        consolidated_path = target_dir / "consolidated_reconstruct.pt"

        if args.skip_if_done and metrics_json.exists():
            match_rate = find_match_rate(metrics_json)
            elapsed_cached, sec_per_sample_cached = read_wallclock_meta(target_dir)
            rows.append(
                {
                    "num_steps": steps,
                    "match_rate": match_rate,
                    "elapsed_seconds": elapsed_cached,
                    "sec_per_sample": sec_per_sample_cached,
                }
            )
            continue

        # 1) reconstruct
        cmd_reconstruct = [
            args.python_bin, str(args.evaluate_py), "reconstruct", str(args.checkpoint),
            "--num_steps", str(steps),
            "--batch_size", str(args.batch_size),
            "--subdir", subdir,
            "--inference_anneal_slope", str(args.inference_anneal_slope),
            "--stage", args.stage,
            "--limit_predict_batches", args.limit_predict_batches,
        ]
        if args.single_gpu:
            cmd_reconstruct.append("--single_gpu")
        else:
            cmd_reconstruct.append("--multi_gpu")
        pbar.set_description(f"reconstruct | steps={steps}")
        t0 = time.perf_counter()
        run(cmd_reconstruct)
        elapsed = time.perf_counter() - t0

        # 2) consolidate
        cmd_consolidate = [
            args.python_bin, str(args.evaluate_py), "consolidate", str(args.checkpoint),
            "--subdir", subdir,
        ]
        pbar.set_description(f"consolidate | steps={steps}")
        run(cmd_consolidate)

        # 3) old_eval_metrics
        cmd_old_eval = [
            args.python_bin, str(args.evaluate_py), "old_eval_metrics", str(args.checkpoint),
            "--subdir", subdir,
            "--stage", args.stage,
            "--do_not_log_wandb",  # keep the file outputs, avoid wandb logging by default
        ]
        pbar.set_description(f"metrics | steps={steps}")
        run(cmd_old_eval)

        # 4) read match_rate
        if not metrics_json.exists():
            raise FileNotFoundError(f"Expected metrics file not found: {metrics_json}")
        match_rate = find_match_rate(metrics_json)
        num_samples = infer_num_samples_from_consolidated(consolidated_path)
        sec_per_sample = (
            elapsed / float(num_samples) if num_samples and num_samples > 0 else float("nan")
        )

        rows.append(
            {
                "num_steps": steps,
                "match_rate": match_rate,
                "elapsed_seconds": elapsed,
                "sec_per_sample": sec_per_sample,
            }
        )

        wallclock_meta = {
            "num_steps": steps,
            "elapsed_seconds": elapsed,
            "num_samples": num_samples,
            "sec_per_sample": sec_per_sample,
        }
        with open(target_dir / "wallclock.json", "w") as f:
            json.dump(wallclock_meta, f, indent=2)

    # Save CSV
    df = pd.DataFrame(rows).sort_values("num_steps").reset_index(drop=True)
    csv_path = sweep_root / "sweep_num_steps_match_rate.csv"
    df.to_csv(csv_path, index=False)

    # Save plot
    plt.figure()
    plt.plot(df["num_steps"], df["match_rate"], marker="o")
    plt.xlabel("num_steps")
    plt.ylabel("match_rate")
    plt.title("Reconstruction match_rate vs num_steps")
    plt.grid(True, which="both", alpha=0.5)  # add grids
    plt.tight_layout()
    plot_path = sweep_root / "sweep_num_steps_match_rate.png"
    plt.savefig(plot_path, dpi=150)

    # Wall-clock vs num_steps
    plt.figure()
    plt.plot(df["num_steps"], df["elapsed_seconds"], marker="o")
    plt.xlabel("num_steps")
    plt.ylabel("elapsed_seconds")
    plt.title("Wall-clock vs num_steps")
    plt.grid(True, which="both", alpha=0.5)
    plt.tight_layout()
    wallclock_plot = sweep_root / "sweep_num_steps_wallclock.png"
    plt.savefig(wallclock_plot, dpi=150)

    # Match-rate vs wall-clock
    plt.figure()
    plt.plot(df["elapsed_seconds"], df["match_rate"], marker="o")
    plt.xlabel("elapsed_seconds")
    plt.ylabel("match_rate")
    plt.title("Match rate vs wall-clock")
    plt.grid(True, which="both", alpha=0.5)
    plt.tight_layout()
    tradeoff_plot = sweep_root / "match_rate_vs_wallclock.png"
    plt.savefig(tradeoff_plot, dpi=150)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved match-rate plot: {plot_path}")
    print(f"Saved wall-clock plot: {wallclock_plot}")
    print(f"Saved trade-off plot: {tradeoff_plot}")


if __name__ == "__main__":
    main()
