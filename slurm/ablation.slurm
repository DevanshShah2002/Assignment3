#!/bin/bash

#SBATCH --job-name=a3_ablation

#SBATCH --partition=gpu1v100

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --time=12:00:00

#SBATCH --output=/work/qgi899/assignment3/logs/ablation_%j.log

#SBATCH --error=/work/qgi899/assignment3/logs/ablation_%j.err



echo "=== Ablation Started: $(date) ==="



module load anaconda3

source activate /work/qgi899/envs/assignment3

source ~/.bashrc



export HF_HOME="/work/qgi899/.HF_cache"



cd /work/qgi899/assignment3



echo ">>> Ablation: 1 epoch..."

python -m src.training.train_stage2 --config configs/config.yaml --epochs 1 --output_suffix ablation_ep1



echo ">>> Ablation: 2 epochs..."

python -m src.training.train_stage2 --config configs/config.yaml --epochs 2 --output_suffix ablation_ep2



echo "=== Ablation Done: $(date) ==="

