"""
scripts/run_pipeline.py
=========================
Master pipeline runner. Executes each phase in order.
Useful for local testing or end-to-end run on a single node.

Usage:
    python scripts/run_pipeline.py --config configs/config.yaml --phases all
    python scripts/run_pipeline.py --config configs/config.yaml --phases data
    python scripts/run_pipeline.py --config configs/config.yaml --phases eval
"""

import argparse
import subprocess
import sys
import os

PHASES = {
    "data_alpaca":      "python -m src.data_prep.prepare_alpaca",
    "data_teacher":     "python -m src.data_prep.generate_teacher_data",
    "train_stage1":     "python -m src.training.train_stage1",
    "train_stage2":     "python -m src.training.train_stage2",
    "infer_ckpt0":      "python -m src.inference.run_inference --checkpoint 0 --eval_set both",
    "infer_ckpt1":      "python -m src.inference.run_inference --checkpoint 1 --eval_set both",
    "infer_ckpt2":      "python -m src.inference.run_inference --checkpoint 2 --eval_set both",
    "eval_json":        "python -m src.evaluation.eval_json_metrics",
    "eval_judge":       "python -m src.evaluation.eval_judge --mode all",
    "aggregate":        "python -m src.evaluation.aggregate_results",
}

PHASE_GROUPS = {
    "data":     ["data_alpaca", "data_teacher"],
    "train":    ["train_stage1", "train_stage2"],
    "infer":    ["infer_ckpt0", "infer_ckpt1", "infer_ckpt2"],
    "eval":     ["eval_json", "eval_judge", "aggregate"],
    "all":      list(PHASES.keys()),
}


def run_phase(name: str, cmd: str, config: str):
    full_cmd = f"{cmd} --config {config}"
    print(f"\n{'='*60}")
    print(f"[Pipeline] Running: {name}")
    print(f"  CMD: {full_cmd}")
    print(f"{'='*60}")
    result = subprocess.run(full_cmd, shell=True, check=False)
    if result.returncode != 0:
        print(f"[Pipeline] ERROR in phase '{name}' (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"[Pipeline] ✓ Completed: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--phases", default="all",
                        help="Comma-separated list of phases or group name "
                             f"(groups: {list(PHASE_GROUPS.keys())})")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Parse phases
    if args.phases in PHASE_GROUPS:
        phases_to_run = PHASE_GROUPS[args.phases]
    else:
        phases_to_run = [p.strip() for p in args.phases.split(",")]

    print(f"[Pipeline] Phases to run: {phases_to_run}")

    for phase_name in phases_to_run:
        if phase_name not in PHASES:
            print(f"[Pipeline] Unknown phase: {phase_name}")
            sys.exit(1)
        run_phase(phase_name, PHASES[phase_name], args.config)

    print("\n[Pipeline] All phases completed successfully.")


if __name__ == "__main__":
    main()
