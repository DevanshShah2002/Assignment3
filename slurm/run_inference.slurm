#!/bin/bash

#SBATCH --job-name=a3_inference

#SBATCH --partition=gpu1v100

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --time=06:00:00

#SBATCH --output=/work/qgi899/assignment3/logs/inference_%j.log

#SBATCH --error=/work/qgi899/assignment3/logs/inference_%j.err



echo "=== Inference Started: $(date) ==="



module load anaconda3

source activate /work/qgi899/envs/assignment3

source ~/.bashrc



export HF_HOME="/work/qgi899/.HF_cache"



cd /work/qgi899/assignment3

mkdir -p results



echo ">>> Checkpoint 0: Base model..."

python -m src.inference.run_inference --config configs/config.yaml --checkpoint 0 --eval_set both



echo ">>> Checkpoint 1: Stage 1 Alpaca..."

python -m src.inference.run_inference --config configs/config.yaml --checkpoint 1 --eval_set both



echo ">>> Checkpoint 2: Stage 2 Teacher JSON..."

python -m src.inference.run_inference --config configs/config.yaml --checkpoint 2 --eval_set both



echo "=== Inference Done: $(date) ==="

