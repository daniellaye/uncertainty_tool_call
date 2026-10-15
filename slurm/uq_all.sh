#!/bin/bash
#SBATCH --job-name=utoolcall-uq
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

# Compute the estimators for all 8 models x 10 tasks, then build Table 1 (CPU
# only, ~90 min). Submit from the repo root with the environment where you ran
# `pip install -e .` active:
#
#   sbatch slurm/uq_all.sh
#
# Reads data/combined/ and writes uq/, or $UTOOLCALL_COMBINED and
# $UTOOLCALL_UQ_OUT if set.

set -eo pipefail

COMBINED=${UTOOLCALL_COMBINED:-data/combined}
OUT=${UTOOLCALL_UQ_OUT:-uq}

models="Qwen_Qwen2.5-0.5B-Instruct Qwen_Qwen2.5-3B-Instruct Qwen_Qwen2.5-7B-Instruct \
Qwen_Qwen3-4B-Instruct-2507 google_gemma-2-9b-it google_gemma-3-12b-it \
google_gemma-3-4b-it mistralai_Ministral-8B-Instruct-2410"

tasks="simple multiple parallel parallel_multiple \
simple-multiple simple-parallel multiple-parallel_multiple \
simple-multiple-parallel-parallel_multiple \
simple-irrelevance simple-multiple-parallel-parallel_multiple-irrelevance"

for model in $models; do
    for task in $tasks; do
        echo "===== $model / $task ====="
        utoolcall uq --combined_path "$COMBINED/$model" --test_name "$task" --save_path "$OUT/$model"
    done
done

utoolcall table1 --scores_dir "$OUT" -o "$OUT/table1.tex"
cat "$OUT/table1.tex"
