#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --job-name="utoolcall-prepare"
#SBATCH --output=/users/zihye/logs/slurm-utoolcall-prepare-%j.out
#SBATCH --error=/users/zihye/logs/slurm-utoolcall-prepare-%j.err

# Build one combined file from scratch -- 1 greedy + 10 samples at T=1.5 for
# Qwen2.5-0.5B on `simple` -- then compute its uncertainty scores.
#
#   sbatch slurm/prepare_simple.sh
#
# Writes everything under generations/ in the repo (gitignored). Sampling is
# stochastic, so the scores will be close to, not identical with, the published
# ones; this checks that the from-scratch path runs end to end.

set -eo pipefail

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface
export HF_HOME=/scratch-ssd/$USER/huggingface
mkdir -p "$HF_HOME"
chmod 700 "$HF_HOME"

export XDG_CACHE_HOME=/scratch-ssd/oatml/
export TMPDIR=/scratch-ssd/${USER}/tmp
mkdir -p "$TMPDIR"

# A separate vLLM port per job, so parallel jobs on one node don't collide.
export VLLM_PORT=$((10000 + (SLURM_JOB_ID % 500) * 10))
echo "Using VLLM_PORT=$VLLM_PORT (SLURM_JOB_ID=$SLURM_JOB_ID)"

REPO=~/workspace/uncertainty_tool_call
MODEL=Qwen/Qwen2.5-0.5B-Instruct
MODEL_DIR=${MODEL//\//_}
TASK=simple
export UTOOLCALL_OUTPUT_DIR=$REPO/generations

# /scratch-ssd is local to each node, so create (or update) the env on this one.
# A separate env from uq_all.sh's, so the two jobs never pip-install into the same one.
/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda env update -n utoolcall-gpu -f "$REPO/environment.yaml"
source /scratch-ssd/oatml/miniconda3/bin/activate utoolcall-gpu

cd "$REPO"
pip install -e '.[generate,vllm]'
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"

nvidia-smi

echo "===== prepare: generate -> evaluate -> combine ====="
utoolcall prepare --model $MODEL --task $TASK --samples 10 --temperature 1.5

echo "===== uq on the combined file ====="
utoolcall uq \
    --combined_path "$UTOOLCALL_OUTPUT_DIR/combined/$MODEL_DIR" \
    --test_name $TASK \
    --save_path "$UTOOLCALL_OUTPUT_DIR/uq/$MODEL_DIR"

cat "$UTOOLCALL_OUTPUT_DIR/uq/$MODEL_DIR/combined_uq_$TASK.csv"
cat "$UTOOLCALL_OUTPUT_DIR/uq/$MODEL_DIR/mean_acc_$TASK.txt"

# Greedy decoding is (near-)deterministic, so compare it with the release's
# copy if `utoolcall fetch-combined` has been run. Sampled outputs will differ.
python - "$REPO/data/combined/$MODEL_DIR/combined_$TASK.json" \
         "$UTOOLCALL_OUTPUT_DIR/combined/$MODEL_DIR/combined_$TASK.json" <<'PY'
import json, os, sys
ref_path, new_path = sys.argv[1:3]
if not os.path.isfile(ref_path):
    sys.exit(f"skipping greedy comparison: {ref_path} not found (run utoolcall fetch-combined)")
ref = {r["id"]: r["greedy_output"] for r in json.load(open(ref_path))}
new = {r["id"]: r["greedy_output"] for r in json.load(open(new_path))}
ids = ref.keys() & new.keys()
same = sum(ref[i]["text"] == new[i]["text"] for i in ids)
acc = lambda d: sum(d[i]["accuracy"] for i in ids) / len(ids)
print(f"records: {len(new)} new, {len(ref)} in release, {len(ids)} shared")
print(f"greedy outputs identical to the release: {same}/{len(ids)}")
print(f"greedy accuracy: {acc(new):.4f} new vs {acc(ref):.4f} release")
PY
