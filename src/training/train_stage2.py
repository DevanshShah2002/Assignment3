"""
src/training/train_stage2.py
==============================
Stage 2: Continue QLoRA fine-tuning on teacher-generated JSON Instruct data,
starting from the Stage 1 Alpaca checkpoint.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from prompts.templates import phi35_format

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/stage2_train.log")],
)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_json_instruct_for_phi35(sample: dict) -> str:
    return phi35_format(
        instruction=sample["instruction"],
        input_text=sample.get("input", ""),
        output=sample["output"],
    )


def train_stage2(config_path: str = "configs/config.yaml", ablation_overrides: dict = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    s2 = cfg["stage2"]
    if ablation_overrides:
        s2.update(ablation_overrides)
        logger.info(f"[Ablation] Overrides applied: {ablation_overrides}")

    lora_cfg = cfg["lora"]
    bnb_cfg  = cfg["bnb"]
    model_name = cfg["model"]["student_model"]

    output_dir = s2["output_dir"].replace("${USER}", os.environ.get("USER", "user"))
    stage1_dir = s2["stage1_adapter_path"].replace("${USER}", os.environ.get("USER", "user"))
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== Stage 2: Teacher-JSON Fine-Tuning ===")
    logger.info(f"Base model: {model_name}")
    logger.info(f"Stage 1 checkpoint: {stage1_dir}")
    logger.info(f"Output: {output_dir}")

    # ── Load and split data ───────────────────────────────────
    all_data = load_jsonl(s2["dataset_path"])

    holdout_n = s2.get("eval_holdout_samples", 100)
    holdout = all_data[:holdout_n]
    remaining = all_data[holdout_n:]

    frac = s2.get("data_fraction", 1.0)
    if frac < 1.0:
        n_keep = max(1, int(len(remaining) * frac))
        remaining = remaining[:n_keep]
        logger.info(f"[Ablation] Using {frac:.0%} of data: {n_keep} samples")

    split_idx = int(len(remaining) * s2.get("train_split", 0.90))
    train_data = remaining[:split_idx]
    val_data   = remaining[split_idx:]

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Held-out: {len(holdout)}")

    # ── Pre-format datasets ───────────────────────────────────
    train_texts = [{"text": format_json_instruct_for_phi35(s)} for s in train_data]
    val_texts   = [{"text": format_json_instruct_for_phi35(s)} for s in val_data]

    train_dataset = Dataset.from_list(train_texts)
    val_dataset   = Dataset.from_list(val_texts)

    # ── Tokenizer ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        stage1_dir,
        trust_remote_code=cfg["model"]["trust_remote_code"],
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    # ── BitsAndBytes config ───────────────────────────────────
    compute_dtype = torch.bfloat16 if s2["bf16"] else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["load_in_4bit"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
    )

    # ── Load base model ───────────────────────────────────────
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg["model"]["trust_remote_code"],
        attn_implementation="eager",
    )
    base_model.config.use_cache = False

    # ── Load Stage 1 adapter weights ─────────────────────────
    logger.info(f"Loading Stage 1 LoRA adapter from {stage1_dir}...")
    model = PeftModel.from_pretrained(base_model, stage1_dir, is_trainable=True)
    model = prepare_model_for_kbit_training(model)
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    model.print_trainable_parameters()

    # ── LoRA config for Stage 2 ───────────────────────────────
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
    )

    # ── Training args ─────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=s2["num_train_epochs"],
        per_device_train_batch_size=s2["per_device_train_batch_size"],
        per_device_eval_batch_size=s2["per_device_train_batch_size"],
        gradient_accumulation_steps=s2["gradient_accumulation_steps"],
        learning_rate=s2["learning_rate"],
        lr_scheduler_type=s2["lr_scheduler_type"],
        warmup_ratio=s2["warmup_ratio"],
        weight_decay=s2["weight_decay"],
        max_grad_norm=s2["max_grad_norm"],
        fp16=s2["fp16"],
        bf16=s2["bf16"],
        optim=s2["optim"],
        gradient_checkpointing=s2["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=s2["save_steps"],
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=s2["save_steps"],
        logging_steps=s2["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=s2["seed"],
        remove_unused_columns=True,
        dataloader_num_workers=8,
    )

    # ── SFT Trainer ───────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=s2["max_seq_length"],
        peft_config=peft_config,
    )

    # ── Train ─────────────────────────────────────────────────
    logger.info("Starting Stage 2 training...")
    train_result = trainer.train()

    # ── Save ──────────────────────────────────────────────────
    logger.info("Saving Stage 2 adapter checkpoint...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "stage": "stage2_teacher_json",
        "model": model_name,
        "stage1_checkpoint": stage1_dir,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "epochs": s2["num_train_epochs"],
        "learning_rate": s2["learning_rate"],
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
        "train_loss": train_result.metrics.get("train_loss", 0),
        "ablation_overrides": ablation_overrides,
        "output_dir": output_dir,
    }
    with open(f"{output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Stage 2 Complete ===")
    logger.info(f"Checkpoint saved to: {output_dir}")
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_fraction", type=float, default=None)
    parser.add_argument("--output_suffix", type=str, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.data_fraction is not None:
        overrides["data_fraction"] = args.data_fraction
    if args.output_suffix:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        base_dir = cfg["stage2"]["output_dir"].replace("${USER}", os.environ.get("USER", "user"))
        overrides["output_dir"] = f"{base_dir}_{args.output_suffix}"

    train_stage2(args.config, ablation_overrides=overrides if overrides else None)