#!/bin/bash

#SBATCH -J qwen_vlm_search
#SBATCH -A default
#SBATCH -p 3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH -t 30:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

cd /oscar/data/epavlick/aparasel/Binding-Problem-Brown/Scripts

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate patchscopes-llava

mkdir -p $HF_HUB_CACHE

python /oscar/data/epavlick/aparasel/Binding-Problem-Brown/scripts/benchmark_headpatch.py   --conditions conjunctive   --source_distractors 10   --target_distractors 40  --out_path /oscar/data/epavlick/aparasel/Binding-Problem-Brown/Results/gemma27B_conj_headpatch_all_in_one.csv --model_name google/gemma-3-27b-it