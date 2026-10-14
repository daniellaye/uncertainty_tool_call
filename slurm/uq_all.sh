#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name="utoolcall-uq"
#SBATCH --output=/users/zihye/logs/slurm-utoolcall-uq-%j.out
#SBATCH --error=/users/zihye/logs/slurm-utoolcall-uq-%j.err

# Recompute the uncertainty estimators for all 8 models x 10 tasks, then build
# Table 1. CPU only -- no GPU is needed for this step.
#
#   sbatch slurm/uq_all.sh
#
# Roughly 1.5 min per model/task, so ~2 hours for the full sweep.

set -eo pipefail

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/
export TMPDIR=/scratch-ssd/${USER}/tmp
mkdir -p "$TMPDIR"

REPO=~/workspace/uncertainty_tool_call
COMBINED=${UTOOLCALL_COMBINED:-$REPO/data/combined}
OUT=${UTOOLCALL_UQ_OUT:-$REPO/uq}

# /scratch-ssd is local to each node, so create (or update) the env on this one.
/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda env update -f "$REPO/environment.yaml"
source /scratch-ssd/oatml/miniconda3/bin/activate utoolcall

cd "$REPO"
pip install -e .

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
        utoolcall uq \
            --combined_path "$COMBINED/$model" \
            --test_name "$task" \
            --save_path "$OUT/$model" \
            --plot_rc_figures false
    done
done

echo "===== Table 1 ====="
utoolcall table1 --scores_dir "$OUT" -o "$OUT/table1.tex"
cat "$OUT/table1.tex"
