"""
src/inference/run_inference.py
================================
Generate responses at each of the 3 checkpoints:
  - checkpoint_0: untuned base model
  - checkpoint_1: Stage 1 Alpaca-tuned
  - checkpoint_2: Stage 2 Teacher-JSON-tuned

Usage:
    python -m src.inference.run_inference \
        --config configs/config.yaml \
        --checkpoint 0   # or 1 or 2
        --eval_set alpaca  # or json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from prompts.templates import phi35_format

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

CHECKPOINT_NAMES = {
    0: "checkpoint_0_base",
    1: "checkpoint_1_alpaca",
    2: "checkpoint_2_teacher_json",
}


def load_jsonl(path: str) -> list:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_model_and_tokenizer(cfg: dict, checkpoint_id: int):
    """Load the appropriate model/adapter for the given checkpoint."""
    model_name = cfg["model"]["student_model"]
    bnb_cfg = cfg["bnb"]

    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["load_in_4bit"],
        bnb_4bit_quant_type=bnb_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_cfg["bnb_4bit_use_double_quant"],
    )

    if checkpoint_id == 0:
        # Pure base model (no adapters)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    elif checkpoint_id == 1:
        adapter_dir = cfg["stage1"]["output_dir"].replace("${USER}", os.environ.get("USER", "user"))
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir, trust_remote_code=True, padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    elif checkpoint_id == 2:
        adapter_dir = cfg["stage2"]["output_dir"].replace("${USER}", os.environ.get("USER", "user"))
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir, trust_remote_code=True, padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        raise ValueError(f"Invalid checkpoint_id: {checkpoint_id}")

    model.eval()
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    samples: List[dict],
    max_new_tokens: int,
    batch_size: int = 4,
) -> List[str]:
    """Generate responses for a list of samples."""
    responses = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch = samples[i : i + batch_size]
        prompts = []
        for s in batch:
            prompt = phi35_format(
                instruction=s["instruction"],
                input_text=s.get("input", ""),
                output="",  # Leave empty — we generate this
            )
            prompts.append(prompt)

        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),
            )

        for j, output in enumerate(outputs):
            # Decode only the newly generated tokens
            input_len = input_ids.shape[1]
            new_tokens = output[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # Clean any trailing end tokens
            response = response.replace("<|end|>", "").strip()
            responses.append(response)

    return responses


def run_inference(
    config_path: str = "configs/config.yaml",
    checkpoint_id: int = 0,
    eval_set: str = "alpaca",
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg["eval"]
    os.makedirs(eval_cfg["results_dir"], exist_ok=True)

    checkpoint_name = CHECKPOINT_NAMES[checkpoint_id]
    logger.info(f"Running inference: {checkpoint_name} | eval_set: {eval_set}")

    # ── Load eval data ────────────────────────────────────────
    if eval_set == "alpaca":
        data_path = "data/alpaca_eval_holdout.jsonl"
        n_samples = eval_cfg["alpaca_eval_samples"]
    elif eval_set == "json":
        data_path = "data/json_eval_holdout.jsonl"
        n_samples = eval_cfg["json_eval_samples"]
    else:
        raise ValueError(f"Unknown eval_set: {eval_set}")

    all_data = load_jsonl(data_path)
    samples = all_data[:n_samples]
    logger.info(f"Loaded {len(samples)} eval samples from {data_path}")

    # ── Load model ────────────────────────────────────────────
    logger.info(f"Loading {checkpoint_name}...")
    model, tokenizer = load_model_and_tokenizer(cfg, checkpoint_id)

    # ── Generate ──────────────────────────────────────────────
    responses = generate_responses(
        model, tokenizer, samples,
        max_new_tokens=eval_cfg["max_new_tokens"],
        batch_size=eval_cfg["inference_batch_size"],
    )

    # ── Save results ──────────────────────────────────────────
    results = []
    for idx, (sample, response) in enumerate(zip(samples, responses)):
        results.append({
            "prompt_id": f"{eval_set}_eval_{idx:04d}",
            "checkpoint": checkpoint_name,
            "eval_set": eval_set,
            "task_type": sample.get("task_type", "general"),
            "instruction": sample["instruction"],
            "input": sample.get("input", ""),
            "expected_output": sample.get("output", ""),
            "model_response": response,
        })

    out_path = f"{eval_cfg['results_dir']}/{checkpoint_name}_{eval_set}_responses.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Saved {len(results)} responses → {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=int, choices=[0, 1, 2], required=True,
                        help="0=base, 1=stage1, 2=stage2")
    parser.add_argument("--eval_set",  choices=["alpaca", "json", "both"], default="both")
    args = parser.parse_args()

    if args.eval_set == "both":
        for es in ["alpaca", "json"]:
            run_inference(args.config, args.checkpoint, es)
    else:
        run_inference(args.config, args.checkpoint, args.eval_set)
