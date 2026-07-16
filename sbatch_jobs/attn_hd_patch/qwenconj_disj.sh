#!/bin/bash

#SBATCH -J qwen_disj40_to_conj40
#SBATCH -A default
#SBATCH -p 3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 5:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

cd /oscar/data/epavlick/aparasel/Binding-Problem-Brown/Scripts

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate patchscopes-llava

mkdir -p $HF_HUB_CACHE

python /oscar/data/epavlick/aparasel/Binding-Problem-Brown/scripts/benchmark_headpatch.py \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --source_condition disjunctive \
    --target_condition conjunctive \
    --source_distractors 40 \
    --target_distractors 40 \
    --require_source_correct \
    --require_margin_drop \
    --out_path /oscar/data/epavlick/aparasel/Binding-Problem-Brown/Results/disj40_to_conj40_headpatch_qwen.csv