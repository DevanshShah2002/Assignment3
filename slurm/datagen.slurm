#!/bin/bash

#SBATCH --job-name=a3_datagen

#SBATCH --partition=gpu1v100

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --time=04:00:00

#SBATCH --output=/work/qgi899/assignment3/logs/datagen_%j.log

#SBATCH --error=/work/qgi899/assignment3/logs/datagen_%j.err



echo "=== Started: $(date) ==="



module load anaconda3

source activate /work/qgi899/envs/assignment3

source ~/.bashrc



cd /work/qgi899/assignment3



echo ">>> Step 1: Preparing Alpaca data..."

python -m src.data_prep.prepare_alpaca --config configs/config.yaml



echo ">>> Step 2: Generating teacher JSON data..."

python -m src.data_prep.generate_teacher_data --config configs/config.yaml



echo "=== Done: $(date) ==="

