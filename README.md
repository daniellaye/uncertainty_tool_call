# Uncertainty Quantification for LLM Function-Calling

Code for the paper *Uncertainty Quantification for LLM Function-Calling*.

Given a request and a set of available functions, how well does an uncertainty
estimate predict whether an LLM's function call is correct? The paper compares
eight uncertainty estimators across ten task settings from the Berkeley
Function Calling Leaderboard (BFCL), averaged over eight open-weight LLMs.
**Table 1** is the main result: mean AUROC ± mean standard error per
(task, estimator) pair. 

## Reproducing Table 1

### 1. Install

```bash
git clone https://github.com/daniellaye/uncertainty_tool_call.git
cd uncertainty_tool_call
pip install -e .
```



### 2. Download the generations

```bash
utoolcall fetch-combined
```

~100 MB files, unpacked to `data/combined/<model>/combined_<task>.json` show pre-generated results across eight models × ten tasks. Each record is one BFCL request plus what the model did with it:

```json
{
  "id": "simple_0",
  "question": [[{"role": "user", "content": "Find the area of a triangle ..."}]],
  "function": [{"name": "calculate_triangle_area", ...}],

  "greedy_output": {
    "text":           "[calculate_triangle_area(base=10, height=5)]",
    "ast_decoded":    [...],
    "accuracy":       1,                      // was the call correct?
    "token_logprobs": [-0.07, -0.0, ...],     // 14 tokens
    "top_logprobs":   [...]
  },

  "high_temp_outputs": [ ... ]                // M = 10 samples, same fields
}
```

There are no uncertainty scores in these files yet, which we compute in the next step.

### 3. Compute the estimators

Compute different UQ scores for models and tasks selected:

```bash
utoolcall uq \
  --combined_path data/combined/Qwen_Qwen2.5-0.5B-Instruct \
  --test_name simple \
  --save_path uq/Qwen_Qwen2.5-0.5B-Instruct
```

It will write `uq/<model>/combined_uq_simple.csv`, one row per estimator:

```
UQ methods,AUROC,AUPRC,1-NAURC (greedy),1-NAURC (avg),standard_error
min,0.8205,0.9902,0.7840,0.7867,0.0363
sum,0.8312,0.9908,0.7976,0.7988,0.0362
...
```

Table 1 needs all eight models × ten tasks:

```bash
sbatch slurm/uq_all.sh          # writes to uq/, the directory step 4 reads
```



### 4. Build Table 1

```bash
utoolcall table1 --scores_dir uq/ -o table1.tex
```

---



## Building a combined file from scratch

Optional, and needs vllm and one GPU. This step shows how the combined files from step 2 are produced. For example, for a single model and task, the below command generates both 1 greedy and 10 high temperature repsonses using a Qwen model, and save it accordingly.

```bash
pip install -e '.[generate,vllm]'

utoolcall prepare --model Qwen/Qwen2.5-0.5B-Instruct \
              --task simple --samples 10 --temperature 1.0
```

On a SLURM cluster, `sbatch slurm/prepare.sh` runs this command. Then to compute UQ follow step 3 above to run `utoolcall uq` on the result.

To provide more context, these are the steps `prepare` runs. 

```bash
M=Qwen/Qwen2.5-0.5B-Instruct
T=simple
MODEL_DIR=Qwen_Qwen2.5-0.5B-Instruct
MULTI=m=10_temperature=1.0_model=$MODEL_DIR

# 1. greedy decode -- the prediction whose correctness is scored
python -m uncertainty_tool_call.bfcl generate \
    --model $M --test-category $T --temperature 0

# 2. M samples at high temperature -- these estimate its uncertainty.
for m in $(seq 1 10); do
  python -m uncertainty_tool_call.bfcl generate \
      --model $M --test-category $T --temperature 1.0 \
      --result-dir result/$MULTI/m=$m
done

# 3. score both passes -- must run BEFORE combine, which attaches these labels
python -m uncertainty_tool_call.bfcl evaluate --model $M --test-category $T
for m in $(seq 1 10); do
  python -m uncertainty_tool_call.bfcl evaluate \
      --model $M --test-category $T \
      --result-dir result/$MULTI/m=$m --score-dir score/$MULTI/m=$m
done

# 4. merge into the format `utoolcall uq` consumes
utoolcall combine \
    --greedy_logprobs_dir result/$MODEL_DIR --greedy_scores_dir score/$MODEL_DIR \
    --multi_logprobs_dir  result/$MULTI     --multi_scores_dir  score/$MULTI \
    --test_names $T --model_name $MODEL_DIR --save_path combined
```

The result lands in `combined/<model>/combined_<task>.json`, the same format as
the step-2 download.

**AST decoding.** BFCL's `evaluate` step parses each output into function calls
(`ast_parse` in `src/uncertainty_tool_call/bfcl/model_handler/utils.py`). We
modified it to save these parsed calls, and `utoolcall combine` stores them as
`ast_decoded` in the combined file, where the AST-based estimators read them.

## License

Apache-2.0 (see `LICENSE`). The Berkeley Function Calling Leaderboard (BFCL,
part of [Gorilla](https://github.com/ShishirPatil/gorilla), Apache-2.0) is
included at `src/uncertainty_tool_call/bfcl/`, modified from upstream commit
`f092d154` to record per-token log-probabilities and add Qwen, Gemma and
Ministral handlers. `src/uncertainty_tool_call/entropy.py` copies four
functions from [semantic_uncertainty](https://github.com/jlko/semantic_uncertainty).