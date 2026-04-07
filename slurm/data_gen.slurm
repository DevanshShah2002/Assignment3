#!/bin/bash

#SBATCH --job-name=a3_data

#SBATCH --partition=gpu1v100

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --time=04:00:00

#SBATCH --output=/work/qgi899/assignment3/logs/data_%j.log

#SBATCH --error=/work/qgi899/assignment3/logs/data_%j.err



echo "=== Started: $(date) ==="

module load anaconda3

source activate /work/qgi899/envs/assignment3



export HF_HOME="/work/qgi899/.HF_cache"

export UTSA_API_KEY="utsa-jABQlGLaTrae2bqMHyAvPxTvE9KTP0DEWYIXhvtgkDkVcGjp44rN6G56x1aGiyem"



cd /work/qgi899/assignment3

echo "Python: $(which python)"

echo "UTSA_API_KEY set: ${#UTSA_API_KEY} chars"



python -m src.data_prep.generate_teacher_data --config configs/config.yaml



echo "=== Done: $(date) ==="

