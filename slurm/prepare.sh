#!/bin/bash
#SBATCH --job-name=utoolcall-prepare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Build one combined file from scratch, then compute its uncertainty scores:
# 1 greedy output + M samples at temperature T for one model and one task
# (`utoolcall prepare`, then `utoolcall uq`). Needs one GPU; about 10 minutes
# for the defaults, Qwen2.5-0.5B on simple, on an A100.
#
# Submit from the repository root, with the environment where you ran
# `pip install -e '.[generate,vllm]'` activated (sbatch passes it on to the job):
#
#   sbatch slurm/prepare.sh
#   MODEL=google/gemma-2-9b-it TASK=multiple sbatch slurm/prepare.sh
#
# Settings, as environment variables: MODEL, TASK, SAMPLES (M, default 10),
# TEMPERATURE (T, default 1.0). Add the #SBATCH options your cluster needs
# (partition, account, GPU type, ...). The log goes to slurm-<jobid>.out in the
# directory you submitted from. Without SLURM, `bash slurm/prepare.sh` runs the
# same steps on a machine with a GPU.
#
# Writes under generations/ (set UTOOLCALL_OUTPUT_DIR to change it):
#   result/, score/     BFCL's generations and their evaluation
#   combined/<model>/   the combined file, in the format `utoolcall fetch-combined` downloads
#   uq/<model>/         its uncertainty scores
# Sampling is random, so the scores come out close to the released ones, not equal.

set -eo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
TASK=${TASK:-simple}
SAMPLES=${SAMPLES:-10}
TEMPERATURE=${TEMPERATURE:-1.0}
MODEL_DIR=${MODEL//\//_}
export UTOOLCALL_OUTPUT_DIR=${UTOOLCALL_OUTPUT_DIR:-$PWD/generations}

command -v utoolcall >/dev/null \
    || { echo "utoolcall not found: activate the environment where you ran pip install -e '.[generate,vllm]'" >&2; exit 1; }
python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('vllm') is None)" \
    || { echo "vllm not installed: pip install -e '.[generate,vllm]'" >&2; exit 1; }

# A separate vLLM port per job, so jobs sharing a node don't collide.
if [ -n "$SLURM_JOB_ID" ] && [ -z "$VLLM_PORT" ]; then
    export VLLM_PORT=$((10000 + (SLURM_JOB_ID % 500) * 10))
fi

echo "===== prepare: $MODEL on $TASK, $SAMPLES samples at T=$TEMPERATURE ====="
utoolcall prepare --model "$MODEL" --task "$TASK" --samples "$SAMPLES" --temperature "$TEMPERATURE"

echo "===== uq on the combined file ====="
utoolcall uq \
    --combined_path "$UTOOLCALL_OUTPUT_DIR/combined/$MODEL_DIR" \
    --test_name "$TASK" \
    --save_path "$UTOOLCALL_OUTPUT_DIR/uq/$MODEL_DIR"
cat "$UTOOLCALL_OUTPUT_DIR/uq/$MODEL_DIR/combined_uq_$TASK.csv"
