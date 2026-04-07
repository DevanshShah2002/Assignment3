#!/bin/bash

#SBATCH --job-name=a3_stage1

#SBATCH --partition=gpu1v100

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --time=12:00:00

#SBATCH --output=/work/qgi899/assignment3/logs/stage1_%j.log

#SBATCH --error=/work/qgi899/assignment3/logs/stage1_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --mail-user=qgi899@my.utsa.edu



echo "=== Stage 1 Started: $(date) ==="



module load anaconda3

source activate /work/qgi899/envs/assignment3



export HF_HOME="/work/qgi899/.HF_cache"

export UTSA_API_KEY=$(grep UTSA_API_KEY ~/.bashrc | cut -d"'" -f2)



cd /work/qgi899/assignment3



echo "Python: $(which python)"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"



python -m src.training.train_stage1 --config configs/config.yaml



echo "=== Stage 1 Done: $(date) ==="

