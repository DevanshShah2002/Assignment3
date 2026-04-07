"""
src/evaluation/eval_json_metrics.py
=====================================
Automatic JSON evaluation metrics for structured-output tasks.

Computes per checkpoint:
  - JSON validity rate
  - Schema compliance rate
  - Exact-match accuracy
  - Field-level F1 (for extraction tasks)
  - Common error taxonomy

Usage:
    python -m src.evaluation.eval_json_metrics --config configs/config.yaml
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_NAMES = [
    "checkpoint_0_base",
    "checkpoint_1_alpaca",
    "checkpoint_2_teacher_json",
]

ERROR_CATEGORIES = [
    "missing_bracket",
    "wrong_type",
    "extra_fields",
    "missing_fields",
    "truncated_output",
    "invalid_string_escape",
    "other",
]


def try_parse_json(text: str) -> Optional[Any]:
    """Attempt to parse text as JSON. Returns parsed object or None."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def classify_json_error(text: str) -> str:
    """Classify the type of JSON error in a string."""
    text = text.strip()
    if not text:
        return "truncated_output"
    if len(text) < 5:
        return "truncated_output"
    # Check for common issues
    open_braces = text.count("{") + text.count("[")
    close_braces = text.count("}") + text.count("]")
    if abs(open_braces - close_braces) > 0:
        return "missing_bracket"
    if "True" in text or "False" in text or "None" in text:
        return "wrong_type"
    if "\\" in text and text.count('\\"') == 0:
        return "invalid_string_escape"
    return "other"


def check_schema_compliance(parsed: Any, expected: Any) -> bool:
    """
    Check whether parsed JSON has the same top-level keys and value types
    as expected JSON. Shallow check for robustness.
    """
    if type(parsed) != type(expected):
        return False
    if isinstance(expected, dict):
        for key in expected:
            if key not in parsed:
                return False
            # Check type matches
            if type(parsed[key]) != type(expected[key]):
                return False
        return True
    elif isinstance(expected, list):
        if len(parsed) == 0 and len(expected) > 0:
            return False
        return True
    return parsed == expected


def field_level_f1(parsed: dict, expected: dict) -> dict:
    """Compute precision, recall, F1 at the field level for dict outputs."""
    if not isinstance(parsed, dict) or not isinstance(expected, dict):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    expected_keys = set(expected.keys())
    parsed_keys   = set(parsed.keys())

    # True positives: key present in both
    tp = 0
    for key in expected_keys & parsed_keys:
        # Check value match (string comparison of JSON-serialized values)
        if json.dumps(parsed[key], sort_keys=True) == json.dumps(expected[key], sort_keys=True):
            tp += 1

    precision = tp / len(parsed_keys)  if parsed_keys  else 0.0
    recall    = tp / len(expected_keys) if expected_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def evaluate_json_checkpoint(results: list) -> dict:
    """
    Compute all JSON metrics for a checkpoint's results list.
    Each item in results has: task_type, expected_output, model_response.
    """
    total = len(results)
    valid_count = 0
    schema_compliant = 0
    exact_match = 0
    field_f1_scores = []
    error_counts = Counter()
    task_type_valid = defaultdict(lambda: {"valid": 0, "total": 0})

    for item in results:
        response = item["model_response"]
        expected_str = item.get("expected_output", "")
        task_type = item.get("task_type", "unknown")

        expected_parsed = try_parse_json(expected_str)
        response_parsed = try_parse_json(response)

        task_type_valid[task_type]["total"] += 1

        if response_parsed is not None:
            valid_count += 1
            task_type_valid[task_type]["valid"] += 1

            # Schema compliance
            if expected_parsed is not None:
                if check_schema_compliance(response_parsed, expected_parsed):
                    schema_compliant += 1

                # Exact match (after re-serialization for canonical form)
                if (json.dumps(response_parsed, sort_keys=True) ==
                        json.dumps(expected_parsed, sort_keys=True)):
                    exact_match += 1

                # Field-level F1 (only for dict outputs)
                if isinstance(response_parsed, dict) and isinstance(expected_parsed, dict):
                    f1_result = field_level_f1(response_parsed, expected_parsed)
                    field_f1_scores.append(f1_result["f1"])
        else:
            error_category = classify_json_error(response)
            error_counts[error_category] += 1

    avg_f1 = sum(field_f1_scores) / len(field_f1_scores) if field_f1_scores else 0.0

    # Per-task validity
    per_task = {}
    for tt, counts in task_type_valid.items():
        rate = counts["valid"] / counts["total"] if counts["total"] > 0 else 0.0
        per_task[tt] = {
            "validity_rate": round(rate, 4),
            "valid": counts["valid"],
            "total": counts["total"],
        }

    return {
        "total_samples": total,
        "json_validity_rate":      round(valid_count / total, 4) if total else 0.0,
        "schema_compliance_rate":  round(schema_compliant / total, 4) if total else 0.0,
        "exact_match_rate":        round(exact_match / total, 4) if total else 0.0,
        "field_level_f1":          round(avg_f1, 4),
        "error_taxonomy":          dict(error_counts),
        "per_task_validity":       per_task,
    }


def eval_json_all_checkpoints(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["eval"]["results_dir"]
    all_results = {}

    for ckpt in CHECKPOINT_NAMES:
        path = f"{results_dir}/{ckpt}_json_responses.jsonl"
        if not os.path.exists(path):
            print(f"[JSON Eval] Missing: {path} — skipping")
            continue

        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line.strip()))

        metrics = evaluate_json_checkpoint(data)
        all_results[ckpt] = metrics
        print(f"\n[JSON Eval] {ckpt}")
        print(f"  Validity:         {metrics['json_validity_rate']:.2%}")
        print(f"  Schema Compliance:{metrics['schema_compliance_rate']:.2%}")
        print(f"  Exact Match:      {metrics['exact_match_rate']:.2%}")
        print(f"  Field-level F1:   {metrics['field_level_f1']:.4f}")
        print(f"  Error taxonomy:   {metrics['error_taxonomy']}")

    # Save
    out_path = f"{results_dir}/json_evaluation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[JSON Eval] Metrics saved → {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    eval_json_all_checkpoints(args.config)
