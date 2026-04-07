#!/bin/bash

#SBATCH --job-name=a3_setup

#SBATCH --partition=gpu1v100

#SBATCH --gres=gpu:1



#SBATCH --cpus-per-task=8

#SBATCH --time=04:00:00

#SBATCH --output=/work/qgi899/assignment3/logs/setup_%j.log

#SBATCH --error=/work/qgi899/assignment3/logs/setup_%j.err



echo "=== Setup Started: $(date) ==="



module load anaconda3

source activate /work/qgi899/envs/assignment3



export HF_HOME="/work/qgi899/.HF_cache"

export HF_TOKEN="PUT_YOUR_TOKEN_HERE"



cd /work/qgi899/assignment3



python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"



echo ">>> Preparing Alpaca data..."

python -m src.data_prep.prepare_alpaca --config configs/config.yaml



echo ">>> Generating teacher data..."

python -m src.data_prep.generate_teacher_data --config configs/config.yaml



echo "=== All Done: $(date) ==="

