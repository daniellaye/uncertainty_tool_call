#!/bin/bash
#SBATCH --job-name=utoolcall-prepare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Build one combined file from scratch (needs a GPU). Submit from the repo root
# with the environment where you ran `pip install -e '.[generate,vllm]'` active:
#
#   sbatch slurm/prepare.sh
#   MODEL=google/gemma-2-9b-it TASK=multiple sbatch slurm/prepare.sh
#
# Writes to generations/, or to $UTOOLCALL_OUTPUT_DIR if set.

set -eo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
TASK=${TASK:-simple}
export UTOOLCALL_OUTPUT_DIR=${UTOOLCALL_OUTPUT_DIR:-$PWD/generations}

# A separate vLLM port per job, so jobs sharing a node don't collide.
export VLLM_PORT=${VLLM_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 500 * 10))}

utoolcall prepare --model "$MODEL" --task "$TASK" --samples 10 --temperature 1.0
