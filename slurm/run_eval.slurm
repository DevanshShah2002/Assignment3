#!/bin/bash

#SBATCH --job-name=a3_eval

#SBATCH --partition=gpu2v100

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --time=06:00:00

#SBATCH --output=/work/qgi899/assignment3/logs/eval_%j.log

#SBATCH --error=/work/qgi899/assignment3/logs/eval_%j.err



echo "=== Evaluation Started: $(date) ==="



module load anaconda3

source activate /work/qgi899/envs/assignment3

source ~/.bashrc



export HF_HOME="/work/qgi899/.HF_cache"

export UTSA_API_KEY=$(grep UTSA_API_KEY ~/.bashrc | cut -d"'" -f2)



cd /work/qgi899/assignment3



echo ">>> JSON automatic metrics..."

python -m src.evaluation.eval_json_metrics --config configs/config.yaml



echo ">>> LLM Judge evaluation..."

python -m src.evaluation.eval_judge --config configs/config.yaml --mode all



echo ">>> Aggregating results..."

python -m src.evaluation.aggregate_results --config configs/config.yaml



echo "=== Evaluation Done: $(date) ==="

