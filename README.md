# Assignment 3: Sequential Instruction Tuning of a Small LLM

**LLM & Agentic Systems — UTSA Graduate Course**
**Dr. Peyman Najafirad (Paul Rad) | TA: Mohammad Bahrami**

> Two-Stage Post-Training Alignment: Alpaca → Teacher-JSON Imitation Learning on UTSA ARC HPC

---

## Overview

This project implements a two-stage QLoRA fine-tuning pipeline for **Phi-3.5-Mini-Instruct** (3.8B parameters):

1. **Stage 1** — Fine-tune on Alpaca-style general instruction data
2. **Stage 2** — Continue fine-tuning on a teacher-generated JSON instruction dataset (imitation learning from Llama-3.1-70B-Instruct)

The central research question: *Does Stage 2 cause catastrophic forgetting of Stage 1 gains?*

---

## Repository Structure

```
assignment3/
├── configs/
│   └── config.yaml              # All hyperparameters & paths (single source of truth)
├── prompts/
│   └── templates.py             # All prompt templates (teacher gen, judge, formatting)
├── src/
│   ├── data_prep/
│   │   ├── prepare_alpaca.py    # Download, clean, split Alpaca-Cleaned dataset
│   │   └── generate_teacher_data.py  # Imitation learning: query teacher → JSON dataset
│   ├── training/
│   │   ├── train_stage1.py      # Stage 1: QLoRA on Alpaca
│   │   └── train_stage2.py      # Stage 2: QLoRA on Teacher-JSON (from Stage 1 ckpt)
│   ├── inference/
│   │   └── run_inference.py     # Generate responses at all 3 checkpoints
│   └── evaluation/
│       ├── eval_json_metrics.py # JSON validity, schema compliance, exact match, F1
│       ├── eval_judge.py        # LLM-as-a-Judge: pairwise Alpaca + JSON quality
│       └── aggregate_results.py # Three-checkpoint table + forgetting analysis
├── slurm/
│   ├── train_stage1.slurm       # ARC batch job: Stage 1 training
│   ├── train_stage2.slurm       # ARC batch job: Stage 2 training + full eval
│   ├── generate_teacher_data.slurm  # ARC batch job: teacher data generation
│   └── ablation_study.slurm     # ARC batch job: ablation (epoch variants)
├── scripts/
│   └── run_pipeline.py          # Master pipeline runner (for local or single-node)
├── data/                        # Auto-created; stores all JSONL datasets
├── logs/                        # Auto-created; training and eval logs
├── results/                     # Auto-created; model responses and eval scores
├── requirements.txt
└── README.md
```

---

## Model Choice: Phi-3.5-Mini-Instruct

**Justification:** Phi-3.5-Mini (3.8B) is selected because:
- Strong small-model benchmark performance (competitive with 7B+ models on many tasks)
- Fits comfortably in V100 16GB VRAM with 4-bit QLoRA quantization
- Well-supported by HuggingFace Transformers ≥4.44.0 without custom code
- Instruction-tuned baseline provides a meaningful Checkpoint 0

---

## Setup Instructions

### 1. Access UTSA ARC

```bash
ssh your_abc123@arc.utsa.edu
```

### 2. Transfer your code to ARC `/work`

```bash
# From your local machine:
scp -r assignment3/ your_abc123@arc.utsa.edu:/work/your_abc123/assignment3
```

Or clone from GitHub:
```bash
# On ARC login node:
git clone https://github.com/YOUR_USERNAME/assignment3.git /work/$USER/assignment3
```

### 3. Set up conda environment on ARC

```bash
# On ARC login node:
module load anaconda3

# Create env in /work (NOT /home — limited space!)
conda create -p /work/$USER/envs/assignment3 python=3.10 -y
conda activate /work/$USER/envs/assignment3

# Install PyTorch with CUDA 11.8 (compatible with ARC V100s)
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all other requirements
pip install -r requirements.txt
```

### 4. Set HuggingFace cache to `/work`

```bash
export HF_HOME="/work/$USER/.HF_cache"
echo 'export HF_HOME="/work/$USER/.HF_cache"' >> ~/.bashrc
```

### 5. Set your HuggingFace token (needed for gated models like Llama-3.1)

```bash
export HF_TOKEN="hf_your_token_here"
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
```

> **Note on teacher model access:** Llama-3.1-70B-Instruct requires a HuggingFace account with Meta's license accepted. Alternatively, use the HF Inference API (paid) or OpenRouter. The generation script supports any HF-compatible endpoint.

---

## Running the Pipeline on ARC

All jobs use SLURM batch scheduling. **Do NOT run training on the login node.**

### Step 1 — Prepare Alpaca Data

```bash
cd /work/$USER/assignment3

# Run locally on login node (lightweight — just downloads and splits data)
module load anaconda3
conda activate /work/$USER/envs/assignment3

python -m src.data_prep.prepare_alpaca --config configs/config.yaml
```

Expected output files in `data/`:
- `alpaca_train.jsonl` (~9,500 samples)
- `alpaca_val.jsonl` (~500 samples)
- `alpaca_eval_holdout.jsonl` (200 samples — never used in training)

### Step 2 — Generate Teacher JSON Data

```bash
# Option A: Using HF Inference API (set HF_TOKEN)
python -m src.data_prep.generate_teacher_data --config configs/config.yaml

# Option B: Submit as batch job on ARC (if running local vLLM)
sbatch slurm/generate_teacher_data.slurm
```

Expected output:
- `data/teacher_json_dataset.jsonl` (~1000 samples across 5 task types)
- `data/json_eval_holdout.jsonl` (100 samples — never used in training)

### Step 3 — Stage 1 Training (Alpaca Fine-Tuning)

```bash
sbatch slurm/train_stage1.slurm

# Monitor job:
squeue -u $USER
tail -f logs/stage1_train_slurm_<JOBID>.log
```

This script:
- Trains Stage 1 QLoRA adapter on Alpaca data
- Automatically runs Checkpoint 0 (base) and Checkpoint 1 (Stage 1) inference

Expected runtime: ~4–6 hours on V100 with 10,000 training samples.

### Step 4 — Stage 2 Training (Teacher-JSON Fine-Tuning)

```bash
# IMPORTANT: Only submit after Stage 1 completes!
sbatch slurm/train_stage2.slurm

# Monitor:
tail -f logs/stage2_train_slurm_<JOBID>.log
```

This script:
- Loads Stage 1 checkpoint → continues fine-tuning on teacher JSON data
- Runs Checkpoint 2 inference
- Runs all evaluation (JSON metrics + LLM Judge)
- Produces aggregated results

### Step 5 — Ablation Study

```bash
sbatch slurm/ablation_study.slurm
```

Runs Stage 2 with 1, 2, and 3 epochs to quantify how forgetting scales.

---

## Running Locally (VS Code / Development)

For local development, use the master pipeline runner:

```bash
# Full pipeline (not recommended locally for training — use ARC for that)
python scripts/run_pipeline.py --config configs/config.yaml --phases all

# Data preparation only
python scripts/run_pipeline.py --config configs/config.yaml --phases data

# Evaluation only (after loading pre-generated responses)
python scripts/run_pipeline.py --config configs/config.yaml --phases eval

# Individual steps
python -m src.data_prep.prepare_alpaca --config configs/config.yaml
python -m src.evaluation.eval_json_metrics --config configs/config.yaml
python -m src.evaluation.aggregate_results --config configs/config.yaml
```

---

## Configuration

All hyperparameters are in `configs/config.yaml`. Key settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Student model | `microsoft/Phi-3.5-mini-instruct` | 3.8B params |
| Teacher model | `meta-llama/Llama-3.1-70B-Instruct` | Imitation learning only |
| QLoRA rank (r) | 16 | Per assignment recommendation |
| LoRA alpha | 32 | 2× rank |
| LoRA dropout | 0.05 | |
| 4-bit quant type | nf4 | BitsAndBytes |
| Stage 1 epochs | 3 | |
| Stage 2 epochs | 3 | |
| Learning rate | 2e-5 | Both stages |
| Max seq length | 1024 | |
| Batch size | 4 × 4 grad accum = 16 eff | Per device |

---

## Evaluation Protocol

### Checkpoint 0 (Untuned Base)
- Baseline: no adapters loaded

### Checkpoint 1 (After Stage 1 — Alpaca)
- QLoRA adapter from `checkpoints/stage1/`

### Checkpoint 2 (After Stage 2 — Teacher JSON)
- QLoRA adapter from `checkpoints/stage2/`

### Alpaca Evaluation (Self-Instruct Protocol)
- 150 held-out Alpaca prompts
- Pairwise judge comparison (0 vs 1, 1 vs 2, 0 vs 2)
- Metrics: win rate, tie rate, per-dimension scores (1–5)
- Auto metrics: ROUGE-1/2/L, BERTScore

### JSON Structured Output Evaluation
- 100 held-out prompts (20 per task type)
- JSON validity rate, schema compliance, exact match, field-level F1
- Error taxonomy categorization

---

## Output Artifacts

After full pipeline execution:

```
results/
├── checkpoint_0_base_alpaca_responses.jsonl
├── checkpoint_0_base_json_responses.jsonl
├── checkpoint_1_alpaca_alpaca_responses.jsonl
├── checkpoint_1_alpaca_json_responses.jsonl
├── checkpoint_2_teacher_json_alpaca_responses.jsonl
├── checkpoint_2_teacher_json_json_responses.jsonl
├── pairwise_checkpoint_0_base_vs_checkpoint_1_alpaca.jsonl
├── pairwise_checkpoint_1_alpaca_vs_checkpoint_2_teacher_json.jsonl
├── pairwise_checkpoint_0_base_vs_checkpoint_2_teacher_json.jsonl
├── json_evaluation_metrics.json
├── judge_eval_summary.json
└── aggregated_results.json
```

---

## Reproducing Results

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/assignment3.git
cd assignment3

# Set up environment
conda create -p ./env python=3.10 -y
conda activate ./env
pip install -r requirements.txt

# Download pre-generated data (if provided)
# OR re-run data generation:
python -m src.data_prep.prepare_alpaca
python -m src.data_prep.generate_teacher_data

# Run evaluation on pre-existing checkpoints
python -m src.inference.run_inference --checkpoint 0 --eval_set both
python -m src.inference.run_inference --checkpoint 1 --eval_set both
python -m src.inference.run_inference --checkpoint 2 --eval_set both
python -m src.evaluation.eval_json_metrics
python -m src.evaluation.aggregate_results
```

---

## Key References

1. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*
2. Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*
3. Taori et al. (2023). *Alpaca: A Strong, Replicable Instruction-Following Model*
4. Wang et al. (2023). *Self-Instruct: Aligning Language Models with Self-Generated Instructions*
5. Gu et al. (2024). *A Survey on LLM-as-a-Judge*
