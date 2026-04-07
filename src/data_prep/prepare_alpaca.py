"""
src/data_prep/prepare_alpaca.py
================================
Download, clean, and split the Alpaca-Cleaned dataset.
Saves train, validation, and held-out evaluation splits as JSONL files.

Usage:
    python -m src.data_prep.prepare_alpaca --config configs/config.yaml
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import yaml
from datasets import load_dataset

# ── Add project root to path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Cleaning helpers ──────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    if not text:
        return ""
    text = text.strip()
    # Remove null bytes and other control chars (keep newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def is_valid_sample(sample: dict) -> bool:
    """Return True if the sample meets quality requirements."""
    instruction = clean_text(sample.get("instruction", ""))
    output = clean_text(sample.get("output", ""))

    # Must have non-empty instruction and output
    if not instruction or not output:
        return False
    # Instruction must be at least 10 chars
    if len(instruction) < 10:
        return False
    # Output must be at least 20 chars
    if len(output) < 20:
        return False
    # Skip samples with placeholder-like content
    bad_phrases = ["n/a", "not applicable", "todo", "fill in", "placeholder"]
    for phrase in bad_phrases:
        if phrase in instruction.lower() or phrase in output.lower():
            return False
    return True


def normalize_sample(sample: dict) -> dict:
    """Convert to unified (instruction, input, output) schema."""
    return {
        "instruction": clean_text(sample.get("instruction", "")),
        "input": clean_text(sample.get("input", "")),
        "output": clean_text(sample.get("output", "")),
    }


# ── Main ──────────────────────────────────────────────────────

def prepare_alpaca(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    s1_cfg = cfg["stage1"]
    random.seed(cfg["stage1"].get("seed", 42))

    print("[Alpaca Prep] Loading dataset...")
    dataset = load_dataset(s1_cfg["dataset"], split="train")
    print(f"[Alpaca Prep] Raw samples: {len(dataset)}")

    # Clean and filter
    cleaned = []
    for sample in dataset:
        norm = normalize_sample(sample)
        if is_valid_sample(norm):
            cleaned.append(norm)

    print(f"[Alpaca Prep] After cleaning: {len(cleaned)} samples")

    # Shuffle
    random.shuffle(cleaned)

    # Limit to max_train_samples if set
    max_samples = s1_cfg.get("max_train_samples")
    eval_holdout_n = s1_cfg.get("eval_holdout_samples", 200)

    # Reserve held-out evaluation set FIRST (never seen during training)
    holdout = cleaned[:eval_holdout_n]
    remaining = cleaned[eval_holdout_n:]

    if max_samples:
        remaining = remaining[:max_samples]

    # Train / validation split
    train_frac = s1_cfg.get("train_split", 0.95)
    split_idx = int(len(remaining) * train_frac)
    train_data = remaining[:split_idx]
    val_data = remaining[split_idx:]

    print(f"[Alpaca Prep] Train: {len(train_data)} | Val: {len(val_data)} | Held-out eval: {len(holdout)}")

    # Save
    os.makedirs("data", exist_ok=True)
    splits = {
        "data/alpaca_train.jsonl":   train_data,
        "data/alpaca_val.jsonl":     val_data,
        "data/alpaca_eval_holdout.jsonl": holdout,
    }
    for path, data in splits.items():
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"[Alpaca Prep] Saved {len(data)} samples → {path}")

    print("[Alpaca Prep] Done.")
    return train_data, val_data, holdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    prepare_alpaca(args.config)
