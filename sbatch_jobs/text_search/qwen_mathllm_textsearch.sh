#!/bin/bash

#SBATCH -J qwen_vlm_search
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

export HF_HOME=/oscar/scratch/aparasel/hf_cache
export HF_HUB_CACHE=/oscar/scratch/aparasel/hf_cache/hub

mkdir -p $HF_HUB_CACHE

python /oscar/data/epavlick/aparasel/Binding-Problem-Brown/scripts/benchmark_text.py \
  --model_name Qwen/Qwen2.5-Math-7B-Instruct \
  --out_path \
  /oscar/data/epavlick/aparasel/Binding-Problem-Brown/Results/qwen_math_text_search_no_padding.csv \
  --target_tokens 0