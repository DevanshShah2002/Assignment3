#!/bin/bash
# =============================================================
# SLURM Batch Script — Ablation Study (Stage 2 Epoch Variants)
# UTSA ARC HPC | Assignment 3: Sequential Instruction Tuning
# =============================================================
# Runs 3 ablation variants sequentially on a single GPU node.
# Vary epochs: 1, 2, 3 — to measure forgetting vs JSON accuracy tradeoff.
# =============================================================

#SBATCH --job-name=a3_ablation
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10

#SBATCH --time=24:00:00
#SBATCH --output=logs/ablation_slurm_%j.log
#SBATCH --error=logs/ablation_slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@my.utsa.edu

echo "============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Ablation: Stage 2 Epoch Variants [1, 2, 3]"
echo "Start time:    $(date)"
echo "============================================="

WORK_DIR="/work/${USER}/assignment3"
cd "$WORK_DIR" || exit 1

module purge
module load anaconda3
conda activate "/work/${USER}/envs/assignment3"

export HF_HOME="/work/${USER}/.HF_cache"
mkdir -p logs results

# ── Ablation: vary number of Stage 2 epochs ───────────────────
for EPOCHS in 1 2 3; do
    echo ""
    echo ">>> Ablation: Stage 2 with $EPOCHS epoch(s)"
    echo "    Start: $(date)"

    python -m src.training.train_stage2 \
        --config configs/config.yaml \
        --epochs $EPOCHS \
        --output_suffix "ablation_epochs${EPOCHS}"

    # Run inference
    python -m src.inference.run_inference \
        --config configs/config.yaml \
        --checkpoint 2 \
        --eval_set both

    # JSON metrics
    python -m src.evaluation.eval_json_metrics --config configs/config.yaml

    echo "    Done: $(date)"
done

echo ""
echo "============================================="
echo "Ablation study COMPLETE"
echo "End time: $(date)"
echo "============================================="
