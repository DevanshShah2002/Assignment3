"""
src/evaluation/aggregate_results.py
======================================
Aggregate all evaluation results into the final tables for the blog post.
Produces:
  - Three-checkpoint comparison table
  - Forgetting analysis
  - Per-category breakdown
  - Ablation results summary

Usage:
    python -m src.evaluation.aggregate_results --config configs/config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINTS = [
    "checkpoint_0_base",
    "checkpoint_1_alpaca",
    "checkpoint_2_teacher_json",
]

CHECKPOINT_LABELS = {
    "checkpoint_0_base":         "Checkpoint 0: Untuned Base",
    "checkpoint_1_alpaca":       "Checkpoint 1: After Stage 1 (Alpaca)",
    "checkpoint_2_teacher_json": "Checkpoint 2: After Stage 2 (Teacher JSON)",
}


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_pairwise_win_rate(summary_list: list, ckpt_a: str, ckpt_b: str) -> dict:
    for s in summary_list:
        if s.get("checkpoint_a") == ckpt_a and s.get("checkpoint_b") == ckpt_b:
            return s
    return {}


def aggregate(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["eval"]["results_dir"]

    # ── Load all result files ──────────────────────────────────
    judge_summary  = load_json(f"{results_dir}/judge_eval_summary.json")
    json_metrics   = load_json(f"{results_dir}/json_evaluation_metrics.json")
    auto_metrics   = {m["checkpoint"]: m for m in judge_summary.get("auto_metrics", [])}
    pairwise       = judge_summary.get("pairwise_alpaca", [])

    # ── Three-Checkpoint Comparison Table ─────────────────────
    print("\n" + "="*80)
    print("THREE-CHECKPOINT COMPARISON TABLE")
    print("="*80)
    print(f"{'Checkpoint':<45} {'Alpaca Win%':>12} {'ROUGE-L':>8} {'BERTScore':>10} {'JSON Valid%':>12} {'Schema%':>9} {'ExactMatch%':>12}")
    print("-"*80)

    table_rows = {}

    for ckpt in CHECKPOINTS:
        label = CHECKPOINT_LABELS[ckpt]
        am = auto_metrics.get(ckpt, {})
        jm = json_metrics.get(ckpt, {})

        # Alpaca win rate: how often this model beat Checkpoint 0
        # Use ckpt vs ckpt_0 pairwise if available
        alpaca_win_rate = "—"
        if ckpt == "checkpoint_0_base":
            alpaca_win_rate = "baseline"
        else:
            pair = get_pairwise_win_rate(pairwise, "checkpoint_0_base", ckpt)
            if pair:
                alpaca_win_rate = f"{pair.get('win_rate_b', 0):.1%}"

        rouge_l    = f"{am.get('rougeL', 0):.4f}" if am else "—"
        bert       = f"{am.get('bert_score_f1', 0):.4f}" if am else "—"
        validity   = f"{jm.get('json_validity_rate', 0):.1%}" if jm else "—"
        schema     = f"{jm.get('schema_compliance_rate', 0):.1%}" if jm else "—"
        exact      = f"{jm.get('exact_match_rate', 0):.1%}" if jm else "—"

        print(f"{label:<45} {alpaca_win_rate:>12} {rouge_l:>8} {bert:>10} {validity:>12} {schema:>9} {exact:>12}")
        table_rows[ckpt] = {
            "alpaca_win_rate": alpaca_win_rate,
            "rouge_l": rouge_l,
            "bert_score": bert,
            "json_validity": validity,
            "schema_compliance": schema,
            "exact_match": exact,
        }

    # ── Forgetting Analysis ────────────────────────────────────
    print("\n" + "="*80)
    print("FORGETTING ANALYSIS: Checkpoint 1 → Checkpoint 2")
    print("="*80)

    am1 = auto_metrics.get("checkpoint_1_alpaca", {})
    am2 = auto_metrics.get("checkpoint_2_teacher_json", {})

    if am1 and am2:
        delta_rouge = am2.get("rougeL", 0) - am1.get("rougeL", 0)
        delta_bert  = am2.get("bert_score_f1", 0) - am1.get("bert_score_f1", 0)
        print(f"  ROUGE-L change:     {delta_rouge:+.4f} ({'↑ improvement' if delta_rouge >= 0 else '↓ forgetting'})")
        print(f"  BERTScore change:   {delta_bert:+.4f} ({'↑ improvement' if delta_bert >= 0 else '↓ forgetting'})")

    # Win rate comparison
    c1_vs_c2 = get_pairwise_win_rate(pairwise, "checkpoint_1_alpaca", "checkpoint_2_teacher_json")
    if c1_vs_c2:
        c1_wr = c1_vs_c2.get("win_rate_a", 0)
        c2_wr = c1_vs_c2.get("win_rate_b", 0)
        print(f"  Alpaca judge win rate:")
        print(f"    Checkpoint 1 wins: {c1_wr:.1%}")
        print(f"    Checkpoint 2 wins: {c2_wr:.1%}")
        print(f"    Tie rate:          {c1_vs_c2.get('tie_rate', 0):.1%}")
        if c1_wr > c2_wr:
            print(f"  → CATASTROPHIC FORGETTING detected: Stage 2 degraded Alpaca capability")
        elif c2_wr > c1_wr:
            print(f"  → NO FORGETTING: Stage 2 maintained or improved Alpaca capability")
        else:
            print(f"  → NEUTRAL: No significant change in Alpaca capability")

    # JSON improvement
    jm1 = json_metrics.get("checkpoint_1_alpaca", {})
    jm2 = json_metrics.get("checkpoint_2_teacher_json", {})
    if jm1 and jm2:
        delta_val = jm2.get("json_validity_rate", 0) - jm1.get("json_validity_rate", 0)
        delta_sch = jm2.get("schema_compliance_rate", 0) - jm1.get("schema_compliance_rate", 0)
        print(f"\n  JSON Validity improvement (Stage 1→2): {delta_val:+.1%}")
        print(f"  Schema Compliance improvement:          {delta_sch:+.1%}")

    # ── Per-Task JSON Breakdown ────────────────────────────────
    print("\n" + "="*80)
    print("PER-TASK JSON VALIDITY BREAKDOWN")
    print("="*80)
    print(f"{'Task Type':<35} {'Ckpt0':>8} {'Ckpt1':>8} {'Ckpt2':>8}")
    print("-"*60)

    task_types = [
        "json_extraction",
        "schema_constrained_generation",
        "json_classification",
        "json_repair",
        "tool_call_generation",
    ]
    for tt in task_types:
        vals = []
        for ckpt in CHECKPOINTS:
            jm = json_metrics.get(ckpt, {})
            per_task = jm.get("per_task_validity", {})
            rate = per_task.get(tt, {}).get("validity_rate", 0)
            vals.append(f"{rate:.1%}")
        print(f"{tt:<35} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

    # ── Save aggregated results ────────────────────────────────
    aggregated = {
        "three_checkpoint_table": table_rows,
        "forgetting_analysis": {
            "rouge_l_delta": delta_rouge if am1 and am2 else None,
            "bert_score_delta": delta_bert if am1 and am2 else None,
            "alpaca_pairwise_1vs2": c1_vs_c2 if c1_vs_c2 else None,
        },
        "json_metrics_all": json_metrics,
        "auto_metrics_all": dict(auto_metrics),
    }

    out_path = f"{results_dir}/aggregated_results.json"
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n[Aggregate] Results saved → {out_path}")

    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    aggregate(args.config)
