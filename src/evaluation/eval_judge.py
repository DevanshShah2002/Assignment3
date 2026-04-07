"""
src/evaluation/eval_judge.py
===============================
LLM-as-a-Judge evaluation following the Self-Instruct / Alpaca paper protocol.

Runs pairwise comparison at:
  - Checkpoint 0 vs Checkpoint 1  (Alpaca eval set)
  - Checkpoint 1 vs Checkpoint 2  (Alpaca eval set)
  - Checkpoint 0 vs Checkpoint 2  (Alpaca eval set)

Also evaluates JSON quality at each checkpoint.

Usage:
    python -m src.evaluation.eval_judge --config configs/config.yaml --mode alpaca
    python -m src.evaluation.eval_judge --config configs/config.yaml --mode json
    python -m src.evaluation.eval_judge --config configs/config.yaml --mode all
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI  # ← replaced InferenceClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from prompts.templates import JUDGE_JSON_QUALITY_PROMPT, JUDGE_PAIRWISE_ALPACA_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/judge_eval.log")],
)
logger = logging.getLogger(__name__)

COMPARISONS = [
    ("checkpoint_0_base",        "checkpoint_1_alpaca"),
    ("checkpoint_1_alpaca",      "checkpoint_2_teacher_json"),
    ("checkpoint_0_base",        "checkpoint_2_teacher_json"),
]


def load_jsonl(path: str) -> list:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def try_parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except Exception:
        return None


class JudgeClient:
    def __init__(self, model: str, api_base: Optional[str] = None):
        api_key = os.getenv("UTSA_API_KEY", "EMPTY")
        base_url = api_base or "http://10.246.100.230/v1"  # ← reads from config
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        logger.info(f"JudgeClient → base_url={base_url}  model={model}")

    def judge(self, prompt: str, max_new_tokens: int = 1024,
              temperature: float = 0.1) -> str:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Judge attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        return ""


# ── Alpaca pairwise judge ─────────────────────────────────────

def run_pairwise_alpaca_eval(
    judge: JudgeClient,
    ckpt_a_data: list,
    ckpt_b_data: list,
    ckpt_a_name: str,
    ckpt_b_name: str,
    cfg: dict,
) -> list:
    """
    Run pairwise LLM judge between two checkpoints on the Alpaca eval set.
    Randomizes A/B order to mitigate position bias.
    """
    results = []
    win_a = win_b = ties = errors = 0

    # Build prompt_id → response mapping
    response_map_a = {item["prompt_id"]: item for item in ckpt_a_data}
    response_map_b = {item["prompt_id"]: item for item in ckpt_b_data}

    common_ids = list(set(response_map_a.keys()) & set(response_map_b.keys()))
    logger.info(f"Pairwise eval: {ckpt_a_name} vs {ckpt_b_name} | {len(common_ids)} prompts")

    for prompt_id in common_ids:
        item_a = response_map_a[prompt_id]
        item_b = response_map_b[prompt_id]
        instruction = item_a["instruction"]
        input_text = item_a.get("input", "")
        full_instruction = f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction

        resp_a = item_a["model_response"]
        resp_b = item_b["model_response"]

        # Randomize order to reduce position bias
        swap = random.random() < 0.5
        if swap:
            displayed_a, displayed_b = resp_b, resp_a
            displayed_a_name, displayed_b_name = ckpt_b_name, ckpt_a_name
        else:
            displayed_a, displayed_b = resp_a, resp_b
            displayed_a_name, displayed_b_name = ckpt_a_name, ckpt_b_name

        prompt = JUDGE_PAIRWISE_ALPACA_PROMPT.format(
            instruction=full_instruction,
            checkpoint_a=displayed_a_name,
            checkpoint_b=displayed_b_name,
            response_a=displayed_a,
            response_b=displayed_b,
            prompt_id=prompt_id,
        )

        raw = judge.judge(prompt, max_new_tokens=cfg["eval"]["judge_max_new_tokens"])
        parsed = try_parse_json(raw)

        if parsed is None:
            logger.warning(f"  [Judge] Could not parse response for {prompt_id}")
            errors += 1
            results.append({
                "prompt_id": prompt_id,
                "checkpoint_a": ckpt_a_name,
                "checkpoint_b": ckpt_b_name,
                "winner": "error",
                "swapped": swap,
                "raw_judge_response": raw,
            })
            continue

        # De-swap winner if needed
        raw_winner = parsed.get("winner", "tie")
        if swap:
            if raw_winner == "A":
                winner = "B"
            elif raw_winner == "B":
                winner = "A"
            else:
                winner = "tie"
        else:
            winner = raw_winner

        if winner == "A":
            win_a += 1
        elif winner == "B":
            win_b += 1
        else:
            ties += 1

        results.append({
            "prompt_id": prompt_id,
            "checkpoint_a": ckpt_a_name,
            "checkpoint_b": ckpt_b_name,
            "winner": winner,
            "response_a_scores": parsed.get("response_a_scores", {}),
            "response_b_scores": parsed.get("response_b_scores", {}),
            "justification": parsed.get("justification", ""),
            "swapped": swap,
        })

    total_judged = win_a + win_b + ties
    summary = {
        "checkpoint_a": ckpt_a_name,
        "checkpoint_b": ckpt_b_name,
        "total_judged": total_judged,
        "errors": errors,
        "win_a": win_a,
        "win_b": win_b,
        "ties": ties,
        "win_rate_a": round(win_a / total_judged, 4) if total_judged else 0,
        "win_rate_b": round(win_b / total_judged, 4) if total_judged else 0,
        "tie_rate":   round(ties / total_judged, 4) if total_judged else 0,
    }
    logger.info(
        f"  → {ckpt_a_name} win: {summary['win_rate_a']:.2%} | "
        f"{ckpt_b_name} win: {summary['win_rate_b']:.2%} | "
        f"tie: {summary['tie_rate']:.2%}"
    )

    return results, summary


# ── JSON quality judge ────────────────────────────────────────

def run_json_quality_eval(
    judge: JudgeClient,
    ckpt_data: list,
    ckpt_name: str,
    cfg: dict,
) -> list:
    """Score JSON responses at a single checkpoint for quality."""
    results = []

    for item in ckpt_data:
        prompt = JUDGE_JSON_QUALITY_PROMPT.format(
            instruction=item["instruction"],
            response=item["model_response"],
            expected_output=item.get("expected_output", ""),
            prompt_id=item["prompt_id"],
        )
        raw = judge.judge(prompt, max_new_tokens=512)
        parsed = try_parse_json(raw)

        if parsed is None:
            results.append({
                "prompt_id": item["prompt_id"],
                "checkpoint": ckpt_name,
                "error": "parse_failed",
                "raw": raw,
            })
            continue

        results.append({
            "prompt_id": item["prompt_id"],
            "checkpoint": ckpt_name,
            "task_type": item.get("task_type", "unknown"),
            **parsed,
        })

    return results


# ── Also compute ROUGE and BERTScore ─────────────────────────

def compute_auto_metrics(ckpt_data: list, ckpt_name: str) -> dict:
    """Compute ROUGE-1/2/L and BERTScore for Alpaca responses."""
    try:
        from rouge_score import rouge_scorer
        import bert_score as bs
    except ImportError:
        logger.warning("Install rouge-score and bert-score for auto metrics.")
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    predictions, references = [], []
    lengths = []

    for item in ckpt_data:
        pred = item["model_response"]
        ref  = item.get("expected_output", "")
        if not ref:
            continue
        scores = scorer.score(ref, pred)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)
        predictions.append(pred)
        references.append(ref)
        lengths.append(len(pred.split()))

    # BERTScore
    bert_f1 = []
    if predictions:
        _, _, F1 = bs.score(predictions, references, lang="en", verbose=False, model_type="distilbert-base-uncased", device="cpu")
        bert_f1 = F1.tolist()

    return {
        "checkpoint": ckpt_name,
        "n_samples": len(rouge1),
        "rouge1":     round(sum(rouge1) / len(rouge1), 4) if rouge1 else 0,
        "rouge2":     round(sum(rouge2) / len(rouge2), 4) if rouge2 else 0,
        "rougeL":     round(sum(rougeL) / len(rougeL), 4) if rougeL else 0,
        "bert_score_f1": round(sum(bert_f1) / len(bert_f1), 4) if bert_f1 else 0,
        "avg_response_length_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
    }


# ── Main ──────────────────────────────────────────────────────

def run_judge_eval(config_path: str = "configs/config.yaml", mode: str = "all"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs("logs", exist_ok=True)
    results_dir = cfg["eval"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    judge_model = cfg["model"]["judge_model"]

    judge = JudgeClient(model=judge_model, api_base=cfg["data_gen"].get("teacher_api_base"))
    logger.info(f"Judge model: {judge_model}")

    all_pairwise_summaries = []
    all_rouge_metrics = []

    # ── Pairwise Alpaca eval ──────────────────────────────────
    if mode in ("alpaca", "all"):
        for ckpt_a, ckpt_b in COMPARISONS:
            path_a = f"{results_dir}/{ckpt_a}_alpaca_responses.jsonl"
            path_b = f"{results_dir}/{ckpt_b}_alpaca_responses.jsonl"
            if not os.path.exists(path_a) or not os.path.exists(path_b):
                logger.warning(f"Missing response files for {ckpt_a} or {ckpt_b}")
                continue

            data_a = load_jsonl(path_a)
            data_b = load_jsonl(path_b)

            pairwise_results, summary = run_pairwise_alpaca_eval(
                judge, data_a, data_b, ckpt_a, ckpt_b, cfg
            )
            all_pairwise_summaries.append(summary)

            out_path = f"{results_dir}/pairwise_{ckpt_a}_vs_{ckpt_b}.jsonl"
            with open(out_path, "w") as f:
                for r in pairwise_results:
                    f.write(json.dumps(r) + "\n")
            logger.info(f"Pairwise results saved → {out_path}")

        # Auto metrics (ROUGE + BERTScore)
        for ckpt in ["checkpoint_0_base", "checkpoint_1_alpaca", "checkpoint_2_teacher_json"]:
            path = f"{results_dir}/{ckpt}_alpaca_responses.jsonl"
            if os.path.exists(path):
                data = load_jsonl(path)
                metrics = compute_auto_metrics(data, ckpt)
                all_rouge_metrics.append(metrics)

    # ── JSON quality eval ─────────────────────────────────────
    if mode in ("json", "all"):
        all_json_judge_results = {}
        for ckpt in ["checkpoint_0_base", "checkpoint_1_alpaca", "checkpoint_2_teacher_json"]:
            path = f"{results_dir}/{ckpt}_json_responses.jsonl"
            if not os.path.exists(path):
                continue
            data = load_jsonl(path)
            judge_results = run_json_quality_eval(judge, data, ckpt, cfg)
            all_json_judge_results[ckpt] = judge_results

            out_path = f"{results_dir}/json_judge_{ckpt}.jsonl"
            with open(out_path, "w") as f:
                for r in judge_results:
                    f.write(json.dumps(r) + "\n")

        with open(f"{results_dir}/json_judge_all.json", "w") as f:
            json.dump(all_json_judge_results, f, indent=2)

    # ── Save summary ──────────────────────────────────────────
    summary_all = {
        "pairwise_alpoca": all_pairwise_summaries,
        "auto_metrics":    all_rouge_metrics,
    }
    with open(f"{results_dir}/judge_eval_summary.json", "w") as f:
        json.dump(summary_all, f, indent=2)
    logger.info(f"Full summary saved → {results_dir}/judge_eval_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["alpaca", "json", "all"], default="all")
    args = parser.parse_args()
    run_judge_eval(args.config, args.mode)