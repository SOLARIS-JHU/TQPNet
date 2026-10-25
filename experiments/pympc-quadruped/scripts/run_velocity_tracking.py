"""Run isaacgym_a1.py once per solver method and capture vx_mean(t) CSVs.

The output CSVs are consumed by make_paper_plots.py::plot_velocity_tracking.

  conda run -n isaacgym python scripts/run_velocity_tracking.py [--max-iter 1500]
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iter", type=int, default=1500,
                    help="sim steps per method (default 1500 ≈ 30 s at dt=0.02)")
    ap.add_argument("--num-robots", type=int, default=64,
                    help="batch size for closed-loop runs (default 64; smaller "
                         "than 4096 because qpth's PDIPM scales poorly with B)")
    args = ap.parse_args()

    repo  = Path(__file__).resolve().parent.parent
    out   = repo / 'runs' / 'velocity_tracking_logs'
    out.mkdir(parents=True, exist_ok=True)
    script = repo / 'scripts' / 'isaacgym_a1.py'

    methods = [
        ("qpth.csv",       ["--solver", "qpth"]),
        ("cold_K50.csv",   ["--solver", "cold-admm", "--cold-k", "50",  "--cold-rho", "0.1"]),
        ("tqpnet_K0.csv",  ["--solver", "tqpnet", "--tqpnet-k-refine", "0"]),
        ("tqpnet_K10.csv", ["--solver", "tqpnet", "--tqpnet-k-refine", "10"]),
        ("tqpnet_K20.csv", ["--solver", "tqpnet", "--tqpnet-k-refine", "20"]),
    ]

    common = ["--headless", "--max-iter", str(args.max_iter)]
    env = os.environ.copy()
    env["NUM_ROBOTS"] = str(args.num_robots)
    print(f"NUM_ROBOTS = {args.num_robots}, max_iter = {args.max_iter}")

    for csv_name, extras in methods:
        log_path = out / csv_name
        cmd = ["python", str(script), *common, *extras, "--log-vx", str(log_path)]
        print("\n" + "=" * 70)
        print("running:", " ".join(cmd))
        print("=" * 70)
        ret = subprocess.run(cmd, cwd=str(repo), env=env)
        if ret.returncode != 0:
            print(f"  [WARN] {csv_name} returned {ret.returncode}")
        else:
            print(f"  → {log_path}")

    print(f"\nAll CSVs in {out}.  Now run scripts/make_paper_plots.py to plot.")


if __name__ == "__main__":
    main()
