#!/bin/bash

#SBATCH --job-name=a3_stage2

#SBATCH --partition=gpu1v100

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16

#SBATCH --time=12:00:00

#SBATCH --output=/work/qgi899/assignment3/logs/stage2_%j.log

#SBATCH --error=/work/qgi899/assignment3/logs/stage2_%j.err



echo "=== Stage 2 Started: $(date) ==="



module load anaconda3

source activate /work/qgi899/envs/assignment3

source ~/.bashrc



export HF_HOME="/work/qgi899/.HF_cache"

export OMP_NUM_THREADS=16

export TOKENIZERS_PARALLELISM=true



cd /work/qgi899/assignment3



echo "Python: $(which python)"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo "CPUs: $(nproc)"



python -m src.training.train_stage2 --config configs/config.yaml



echo "=== Stage 2 Done: $(date) ==="

