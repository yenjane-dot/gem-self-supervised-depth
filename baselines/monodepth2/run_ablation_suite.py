"""
Run the public GEM ablation experiments sequentially.

Example:
    python baselines/monodepth2/run_ablation_suite.py --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "gem"
EXPERIMENTS = ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GEM ablation suite sequentially")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs for each experiment")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=EXPERIMENTS,
        help="subset of experiments to run (default: e0-e9)",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {
        "epochs": args.epochs,
        "experiments": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for exp in args.experiments:
        cmd = [sys.executable, "train_gem.py", "--exp", exp, "--epochs", str(args.epochs)]
        print("=" * 72)
        print(f"Running {exp}: {' '.join(cmd)}")
        print("=" * 72)
        t0 = time.time()
        completed = subprocess.run(cmd, cwd=SCRIPT_DIR, check=False)
        dt = time.time() - t0

        summary["experiments"].append(
            {
                "exp": exp,
                "return_code": completed.returncode,
                "duration_seconds": round(dt, 2),
            }
        )

        if completed.returncode != 0:
            print(f"[FAIL] {exp} exited with code {completed.returncode}")
        else:
            print(f"[OK] {exp} completed in {dt / 60:.1f} min")

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary_path = OUTPUT_ROOT / "ablation_suite_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    failures = [item for item in summary["experiments"] if item["return_code"] != 0]
    print("=" * 72)
    print(f"Summary written to {summary_path}")
    print(f"Completed {len(summary['experiments'])} experiments; failures: {len(failures)}")
    print("=" * 72)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
