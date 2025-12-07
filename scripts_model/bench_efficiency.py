import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
import pandas as pd


def run_cmd(cmd: str) -> int:
    print(f"[CMD] {cmd}")
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line)
    p.wait()
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description="Compare VRAM, speed, throughput between two models.")
    ap.add_argument("checkpoint_a", type=str, help="Path to model A .ckpt")
    ap.add_argument("checkpoint_b", type=str, help="Path to model B .ckpt")
    ap.add_argument("--scripts_root", type=str, default="scripts_model")
    ap.add_argument("--stage", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--task", type=str, default="reconstruct", choices=["reconstruct", "generate"])
    ap.add_argument("--num_steps", type=int, default=200)
    ap.add_argument("--limit_predict_batches", type=str, default="1.")
    ap.add_argument("--single_gpu", action="store_true", default=True)
    ap.add_argument("--out_dir", type=str, default="eff_compare")
    args = ap.parse_args()

    eval_py = Path(args.scripts_root) / "evaluate.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, ckpt in [("A", args.checkpoint_a), ("B", args.checkpoint_b)]:
        subdir = f"{out_dir.name}_{label}"
        t0 = time.perf_counter()

        if args.task == "reconstruct":
            cmd = f"python {eval_py} reconstruct {ckpt} --stage {args.stage} --num_evals 1 " \
                  f"--limit_predict_batches {args.limit_predict_batches} --num_steps {args.num_steps} " \
                  f"{'--single_gpu' if args.single_gpu else '--multi_gpu'} --subdir {subdir}"
        else:
            cmd = f"python {eval_py} generate {ckpt} --num_samples 10000 --num_steps {args.num_steps} " \
                  f"{'--single_gpu' if args.single_gpu else '--multi_gpu'} --subdir {subdir}"

        rc = run_cmd(cmd)
        if rc != 0:
            sys.exit(rc)
        elapsed = time.perf_counter() - t0

        # Rough VRAM snapshot
        mem_gb = None
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
            )
            used = [int(x.strip()) for x in out.decode().strip().splitlines()]
            mem_gb = max(used) / 1024.0
        except Exception:
            mem_gb = float("nan")

        rows.append(dict(model=label, checkpoint=ckpt, elapsed_seconds=elapsed, peak_gpu_mem_gb=mem_gb))

    df = pd.DataFrame(rows)
    csv = out_dir / "efficiency_compare.csv"
    df.to_csv(csv, index=False)
    print(f"[OK] Saved {csv}")
    print(df)


if __name__ == "__main__":
    main()


    # python bench_efficiency.py path/to/modelA.ckpt path/to/modelB.ckpt --scripts_root scripts_model --task reconstruct --stage test --num_steps 200 --limit_predict_batches "1." --single_gpu --out_dir eff_compare