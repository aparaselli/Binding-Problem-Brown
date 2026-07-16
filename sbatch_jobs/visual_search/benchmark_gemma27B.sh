#!/bin/bash

#SBATCH -J Gemma27B_search_binary
#SBATCH -A default
#SBATCH -p 3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

cd /oscar/data/epavlick/aparasel/Binding-Problem-Brown/scripts

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate patchscopes-llava

python benchmark.py \
  --model_name google/gemma-3-27b-it \
  --prompt_type binary \
  --out_path /oscar/data/epavlick/aparasel/Binding-Problem-Brown/Results/gemma27b_search_results_counterfactual_binary.csv