"""Compute the uncertainty estimators for one combined file and score them.

Reads ``<combined_path>/combined_<test_name>.json``: one record per BFCL
request, holding the greedy output, M high-temperature samples, and whether
each is correct. Writes to ``<save_path>``:

    combined_<test_name>_with_uq.json   the records, with per-request scores under "uq_scores"
    combined_uq_<test_name>.csv         one row per estimator: AUROC, AUPRC, 1-NAURC, bootstrap SE
    mean_acc_<test_name>.txt            greedy accuracy and number of requests

Usage:
    utoolcall uq --combined_path data/combined/<model> --test_name simple --save_path uq/<model>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn import metrics

from uncertainty_tool_call.entropy import (
    cluster_assignment_entropy,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)

# Seed for the bootstrap and for the random baseline.
SEED = 28032025

# Prefix of the AST-clustering scores in "uq_scores"; the exact-match ones use
# "string_match". Names kept from the published run so annotated files compare.
AST = "ast_single_multiple_parallel"

# The estimators scored, in CSV row order: (CSV name, key in "uq_scores",
# negate). Negating turns an uncertainty into a confidence, so that for every
# score higher means "more likely correct". Paper names on the right.
ESTIMATORS = [
    ("min", "min", False),  # MAX
    ("sum", "sum", False),  # G-NLL
    ("mean", "mean", False),  # AVG
    ("SMT_min", "SMT_min", False),  # MAX_SMT
    ("SMT_sum", "SMT_sum", False),  # G-NLL_SMT
    ("SMT_mean", "SMT_mean", False),  # AVG_SMT
    ("logprobs SE", "string_match_se", True),  # SE_EXM
    ("discrete SE", "string_match_dse", True),  # DSE_EXM
    ("Predictive E", "string_match_pe", True),  # PE
    ("AST logprobs SE", f"{AST}_se", True),  # SE_AST
    ("AST discrete SE", f"{AST}_dse", True),  # DSE_AST
    ("N_DSNLL", "string_match_dsnll_normalised", False),  # SC_EXM
    ("AST_N_DSNLL", f"{AST}_dsnll_normalised", False),  # SC_AST
    ("num_clusters", "string_match_num_clusters", True),
    ("num_ast_clusters", f"{AST}_num_clusters", True),
    ("length", "length", True),  # LEN
]
# A "random" row, with scores drawn uniformly using SEED, is appended last.

AGGREGATE = {"min": np.min, "sum": np.sum, "mean": np.mean}


# --------------------------------------------------------------------------
# Multi-sample estimators: cluster the M samples, then take entropies
# --------------------------------------------------------------------------


def assign_semantic_ids(responses, are_equivalent):
    """Cluster greedily: each response not yet in a cluster opens a new one,
    which every later response equivalent to it joins."""
    semantic_ids = [-1] * len(responses)
    next_id = 0
    for i, first in enumerate(responses):
        if semantic_ids[i] == -1:
            semantic_ids[i] = next_id
            for j in range(i + 1, len(responses)):
                if are_equivalent(first, responses[j]):
                    semantic_ids[j] = next_id
            next_id += 1
    return semantic_ids


def exact_match_ids(texts):
    """EXM: two samples share a cluster if their output strings are equal."""
    return assign_semantic_ids(texts, lambda a, b: a == b)


def ast_match_ids(decoded_calls):
    """AST: two samples share a cluster if their decoded calls are equal.

    A sample is a list of calls, ``[{func_name: {arg: value}}, ...]``. An empty
    list (decoding failed) becomes ``[{}]``, so all failures share one cluster.
    Calls compare as dicts, so the order of the arguments does not matter.

    The sort was meant to make the order of parallel calls not matter either,
    but it leaves the list as it is: ``dict_items`` compare as sets, where
    ``<`` means "subset". So the same parallel calls in a different order land
    in different clusters. Kept, since the paper's AST numbers used it.
    """
    calls = [c if len(c) > 0 else [{}] for c in decoded_calls]

    def are_equivalent(a, b):
        return sorted(x.items() for x in a) == sorted(x.items() for x in b)

    return assign_semantic_ids(calls, are_equivalent)


def write_multi_sample_scores(data, method):
    """SE, DSE, PE, SC and the cluster count, with clusters formed by `method`."""
    for row in data:
        samples = row["high_temp_outputs"]
        if method == "string_match":
            semantic_ids = exact_match_ids([s["text"] for s in samples])
        else:
            semantic_ids = ast_match_ids([s["ast_decoded"] for s in samples])

        # Length-normalised log-likelihood of each sample.
        log_liks = [np.mean(s["token_logprobs"]) for s in samples]
        log_lik_per_cluster = logsumexp_by_id(semantic_ids, log_liks, agg="sum_normalized")

        scores = row["uq_scores"]
        scores[f"{method}_se"] = predictive_entropy_rao(log_lik_per_cluster)
        scores[f"{method}_dse"] = cluster_assignment_entropy(semantic_ids)
        scores[f"{method}_pe"] = predictive_entropy(log_liks)
        scores[f"{method}_semantic_ids"] = semantic_ids
        scores[f"{method}_num_clusters"] = np.float64(len(set(semantic_ids)))
        # SC (self-consistency): the share of samples in the largest cluster.
        scores[f"{method}_dsnll_normalised"] = np.max(np.bincount(semantic_ids) / len(semantic_ids))


# --------------------------------------------------------------------------
# Single-sample estimators: aggregate the greedy output's token log-probs
# --------------------------------------------------------------------------


def write_single_sample_scores(data, json_calls):
    """MAX, G-NLL and AVG over all tokens and over the SMT only, and LEN."""
    for row in data:
        greedy = row["greedy_output"]
        scores = row["uq_scores"]
        scores["length"] = greedy["token_count"]
        for agg, fn in AGGREGATE.items():
            scores[agg] = fn(greedy["token_logprobs"])
        smt_logprobs = smt_token_logprobs(greedy["top_logprobs"], json_calls)
        for agg, fn in AGGREGATE.items():
            scores[f"SMT_{agg}"] = fn(smt_logprobs)


# --------------------------------------------------------------------------
# Semantically meaningful tokens (SMT)
#
# Each token of the greedy output gets a type, and these types are kept:
#   nfp  number of functions or parameters: separators and closing brackets
#   nf   name of a function
#   np   name of a parameter
#   pv   parameter value
# One rule set per output format, as in the paper's appendix algorithms.
# --------------------------------------------------------------------------


def format_top_logprobs(top_logprobs):
    result = []
    for entry in top_logprobs:
        for token_text, token_logprob in entry.items():
            result.append({
                'token_text': token_text,
                'token_logprob': token_logprob,
                'token_type': '-'
            })
    return result


def extract_relevant_logprobs(tokens, uq_method=('nfp', 'pv', 'nf', 'np')):
    
    target_token_types = uq_method

    result = []
    for entry in tokens:
        if entry['token_type'] in target_token_types:
            result.append(entry['token_logprob'])
    return result


def smt_token_logprobs(top_logprobs, json_calls):
    """Log-probabilities of the semantically meaningful tokens of one output.

    `top_logprobs` holds one ``{token_text: logprob}`` dict per output token.
    """
    tokens = format_top_logprobs(top_logprobs)
    if json_calls:
        tokens_processed = classify_token_types_json(tokens)
    else:
        tokens_processed = classify_token_types(tokens)
    return extract_relevant_logprobs(tokens_processed)


def classify_token_types(list_of_tokens):

    for idx, token in enumerate(list_of_tokens):
        if idx > 0:
            previous_token = list_of_tokens[idx-1]
        preprevious_token = list_of_tokens[idx-2] if idx > 1 else None

        if (idx == 0) or check_is_num_of_func_or_param(token):
            token['token_type'] = 'nfp' # Aggregating Categories 1., 5., and 6. in Section 4.2
        elif idx > 0 and check_is_param_value(token, previous_token):
            token['token_type'] = 'pv' # Category 4. in Section 4.2
        elif idx > 0 and check_is_name_of_param(token, previous_token, preprevious_token):
            token['token_type'] = 'np' # Category 3. in Section 4.2
        elif idx > 0 and check_is_name_of_func(token, previous_token, preprevious_token):
            token['token_type'] = 'nf' # Category 2. in Section 4.2
        else:
            token['token_type'] = '-'

    return list_of_tokens


def check_is_num_of_func_or_param(token):
    if (
        ('(' in token['token_text'])
        or (')' in token['token_text'])
        or (',' in token['token_text'])
    ):
        return True
    else:
        return False


def check_is_param_value(token, previous_token):
    if (
        ('=' in previous_token['token_text']) or (previous_token['token_type'] == 'pv') and (not ('(' in token['token_text']) or (',' in token['token_text']))
    ):
        return True
    else:
        return False


def check_is_name_of_param(token, previous_token, preprevious_token):
    if (
        (previous_token['token_text'].endswith('('))
        or (('(' in token['token_text']) and (not token['token_text'].endswith('(')))
        or ((',' in previous_token['token_text']) and (not '(' in previous_token['token_text']) and (preprevious_token['token_type'] == 'pv'))
    ):
        return True
    else:
        return False


def check_is_name_of_func(token, previous_token, preprevious_token=None):
    if (
        ('[' in previous_token['token_text'])
        or ((',' in previous_token['token_text']) and (preprevious_token['token_type'] == 'nfp'))
        or ('),' in previous_token['token_text'])
        or ((',' in token['token_text']) and (token['token_text'] != ',') and (previous_token['token_type'] == 'nfp'))
    ):
        return True
    else:
        return False


def classify_token_types_json(list_of_tokens):

    max_idx = len(list_of_tokens)

    for idx, token in enumerate(list_of_tokens):

        preprevious_token = list_of_tokens[idx-2] if idx > 1 else None
        previous_token = list_of_tokens[idx-1] if idx > 0 else None
        two_token_later = list_of_tokens[idx+2] if idx < (max_idx - 2) else None

        if (idx == 0) or check_is_num_of_func_or_param_json(token, two_token_later):
            token['token_type'] = 'nfp' # Aggregating Categories 1., 5., and 6. in Section 4.2
        elif idx > 1 and check_is_param_value_or_name_of_func_or_param_json(preprevious_token):
            token['token_type'] = 'pv' # Aggregating Categories 2., 3., and first tokens of Category 4. in Section 4.2
        elif idx > 0 and fill_param_value_json(token, previous_token):
            token['token_type'] = 'pv' # All remaining tokens of Category 4. in Section 4.2
        else:
            token['token_type'] = '-'

    return list_of_tokens


def check_is_num_of_func_or_param_json(token, two_token_later):
    if two_token_later:
        if (
            (']' in token['token_text']) and (token['token_text'] != '[TOOL_CALLS]')
            or ('}}' in token['token_text']) or (',' in token['token_text']) and (two_token_later['token_text'] != 'arguments')
        ):
            return True
    else:
        if ((']' in token['token_text']) and (token['token_text'] != '[TOOL_CALLS]')) or ('}}' in token['token_text']):
            return True
    return False


def check_is_param_value_or_name_of_func_or_param_json(preprevious_token):
    if (
        (':' in preprevious_token['token_text'])
        or ((',' in preprevious_token['token_text']) and (preprevious_token['token_type'] == 'nfp') and
           ('}}' not in preprevious_token['token_text']) and (']}' not in preprevious_token['token_text']))
    ):
        return True
    else:
        return False


def fill_param_value_json(token, previous_token):
    if (not ('"' in token['token_text'])) and (previous_token['token_type'] == 'pv'):
        return True
    else:
        return False


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def compute_auroc(accs, scores):
    """AUROC and AUPRC of `scores` (higher = more confident) at predicting `accs`."""
    assert len(scores) == len(accs)
    scores = np.array(scores, dtype=float)
    accs = np.array(accs, dtype=float)
    return metrics.roc_auc_score(accs, scores), metrics.average_precision_score(accs, scores)


def cummean(arr):
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


def compute_aurc(loss, confidence_scores):
    """Area under the risk-coverage curve, most confident predictions first."""
    ordered_loss = loss[np.argsort(-confidence_scores)]
    return float(np.mean(cummean(ordered_loss)))


def compute_naurc(loss, confidence_scores):
    """Normalised AURC, eq. 10 of https://arxiv.org/abs/2305.15508."""
    aurc = compute_aurc(loss=loss, confidence_scores=confidence_scores)
    oracle_aurc = compute_aurc(loss=loss, confidence_scores=-loss)
    risk = np.mean(loss)  # the expected AURC of random scores
    return float((aurc - oracle_aurc) / (risk - oracle_aurc))


def bootstrap_auroc_se(y_true, y_score, n_resamples=1000):
    """Standard error of the AUROC over paired bootstrap resamples."""

    def auroc(y_true, y_score):
        # AUROC is undefined on a resample with one class only; count it as 0.5.
        if np.all(y_true == y_true[0]):
            return 0.5
        return metrics.roc_auc_score(y_true, y_score)

    bs = scipy.stats.bootstrap(
        [y_true, y_score],
        auroc,
        n_resamples=n_resamples,
        confidence_level=0.95,
        random_state=np.random.default_rng(SEED),
        paired=True,
    )
    return bs.standard_error


# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="utoolcall uq",
        description="Compute the uncertainty estimators for one combined file and score them.",
    )
    parser.add_argument("--combined_path", required=True, help="directory holding combined_<test_name>.json")
    parser.add_argument("--test_name", required=True, help="task name, e.g. simple or simple-irrelevance")
    parser.add_argument("--save_path", required=True, help="output directory (created if missing)")
    args = parser.parse_args(argv)

    with open(Path(args.combined_path) / f"combined_{args.test_name}.json") as f:
        data = json.load(f)
    for row in data:
        row["uq_scores"] = {}

    write_multi_sample_scores(data, "string_match")
    write_multi_sample_scores(data, AST)
    # Ministral writes JSON-style calls; the other models write Python-style ones.
    write_single_sample_scores(data, json_calls="mistral" in args.combined_path.lower())

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / f"combined_{args.test_name}_with_uq.json", "w") as f:
        json.dump(data, f)

    # Scored against the greedy output's correctness. 1-NAURC (avg) uses the
    # accuracy averaged over the M samples instead.
    accs = np.array([row["greedy_output"]["accuracy"] for row in data])
    loss = 1.0 - accs
    avg_loss = 1.0 - np.array([np.mean([s["accuracy"] for s in row["high_temp_outputs"]]) for row in data])

    estimators = [
        (name, np.array([-row["uq_scores"][key] if negate else row["uq_scores"][key] for row in data]))
        for name, key, negate in ESTIMATORS
    ]
    estimators.append(("random", np.random.default_rng(SEED).random(len(data))))

    columns = {k: [] for k in ("UQ methods", "AUROC", "AUPRC", "1-NAURC (greedy)", "1-NAURC (avg)", "standard_error")}
    for name, score in estimators:
        auroc, auprc = compute_auroc(accs, score)
        columns["UQ methods"].append(name)
        columns["AUROC"].append(auroc)
        columns["AUPRC"].append(auprc)
        columns["1-NAURC (greedy)"].append(1 - compute_naurc(loss=loss, confidence_scores=score))
        columns["1-NAURC (avg)"].append(1 - compute_naurc(loss=avg_loss, confidence_scores=score))
        columns["standard_error"].append(bootstrap_auroc_se(accs, score))
    pd.DataFrame(columns).to_csv(save_path / f"combined_uq_{args.test_name}.csv", index=False)

    with open(save_path / f"mean_acc_{args.test_name}.txt", "w") as f:
        f.write(str({"greedy": {"acc": np.mean(accs)}, "test_size": len(accs)}))

    print(f"{args.test_name}: {len(data)} requests, greedy accuracy {np.mean(accs):.4f} -> {save_path}")


if __name__ == "__main__":
    main()
