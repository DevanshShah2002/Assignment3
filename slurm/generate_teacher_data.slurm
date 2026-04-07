#!/bin/bash
# =============================================================
# SLURM Batch Script — Teacher Data Generation (Imitation Learning)
# UTSA ARC HPC | Assignment 3: Sequential Instruction Tuning
# =============================================================
# This script optionally uses a locally-served vLLM endpoint.
# If you are using HuggingFace API, just run the Python script directly.
# =============================================================

#SBATCH --job-name=a3_teacher_data
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:v100:2       # 2x V100 recommended for 70B teacher
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/teacher_gen_slurm_%j.log
#SBATCH --error=logs/teacher_gen_slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@my.utsa.edu

echo "============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================="

WORK_DIR="/work/${USER}/assignment3"
cd "$WORK_DIR" || exit 1

module purge
module load anaconda3
conda activate "/work/${USER}/envs/assignment3"

export HF_HOME="/work/${USER}/.HF_cache"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"   # Replace with your token
mkdir -p logs data

# ── Option A: Use HuggingFace Inference API (no local vLLM) ──
# Just run the Python script directly — teacher_api_base=null in config
echo ">>> Generating teacher data via HF Inference API..."
python -m src.data_prep.generate_teacher_data --config configs/config.yaml

echo ""
echo "============================================="
echo "Teacher data generation COMPLETE"
echo "Output: data/teacher_json_dataset.jsonl"
echo "Eval holdout: data/json_eval_holdout.jsonl"
echo "End time: $(date)"
echo "============================================="
