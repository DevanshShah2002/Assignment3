"""
src/training/train_stage1.py
==============================
Stage 1: QLoRA fine-tuning of Phi-3.5-Mini-Instruct on Alpaca-Cleaned.
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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
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
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/stage1_train.log")],
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


def format_alpaca_for_phi35(sample: dict) -> str:
    return phi35_format(
        instruction=sample["instruction"],
        input_text=sample.get("input", ""),
        output=sample["output"],
    )


def train_stage1(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    s1 = cfg["stage1"]
    lora_cfg = cfg["lora"]
    bnb_cfg = cfg["bnb"]
    model_name = cfg["model"]["student_model"]

    output_dir = s1["output_dir"].replace("${USER}", os.environ.get("USER", "user"))
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== Stage 1: Alpaca Fine-Tuning ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {output_dir}")

    # ── Load data ─────────────────────────────────────────────
    train_data = load_jsonl("data/alpaca_train.jsonl")
    val_data = load_jsonl("data/alpaca_val.jsonl")
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── Pre-format datasets ───────────────────────────────────
    train_texts = [{"text": format_alpaca_for_phi35(s)} for s in train_data]
    val_texts   = [{"text": format_alpaca_for_phi35(s)} for s in val_data]

    train_dataset = Dataset.from_list(train_texts)
    val_dataset   = Dataset.from_list(val_texts)

    # ── Tokenizer ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"]["trust_remote_code"],
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    # ── BitsAndBytes (4-bit QLoRA) ────────────────────────────
    compute_dtype = torch.bfloat16 if s1["bf16"] else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["load_in_4bit"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
    )

    # ── Load model ────────────────────────────────────────────
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg["model"]["trust_remote_code"],
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=s1["num_train_epochs"],
        per_device_train_batch_size=s1["per_device_train_batch_size"],
        per_device_eval_batch_size=s1["per_device_train_batch_size"],
        gradient_accumulation_steps=s1["gradient_accumulation_steps"],
        learning_rate=s1["learning_rate"],
        lr_scheduler_type=s1["lr_scheduler_type"],
        warmup_ratio=s1["warmup_ratio"],
        weight_decay=s1["weight_decay"],
        max_grad_norm=s1["max_grad_norm"],
        fp16=s1["fp16"],
        bf16=s1["bf16"],
        optim=s1["optim"],
        gradient_checkpointing=s1["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=s1["save_steps"],
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=s1["save_steps"],
        logging_steps=s1["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=s1["seed"],
        remove_unused_columns=True,
    )

    # ── SFT Trainer ───────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=s1["max_seq_length"],
        peft_config=peft_config,
    )

    # ── Train ─────────────────────────────────────────────────
    logger.info("Starting Stage 1 training...")
    train_result = trainer.train()

    # ── Save ──────────────────────────────────────────────────
    logger.info("Saving Stage 1 adapter checkpoint...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "stage": "stage1_alpaca",
        "model": model_name,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "epochs": s1["num_train_epochs"],
        "learning_rate": s1["learning_rate"],
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
        "train_loss": train_result.metrics.get("train_loss", 0),
        "output_dir": output_dir,
    }
    with open(f"{output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Stage 1 Complete ===")
    logger.info(f"Checkpoint saved to: {output_dir}")
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_stage1(args.config)