"""Entropy estimators over semantic clusters.

Vendored from the semantic-entropy reference implementation
(https://github.com/jlko/semantic_uncertainty, Apache 2.0), which accompanies
Farquhar et al., "Detecting hallucinations in large language models using
semantic entropy", Nature 2024.

Only these four functions are used by this project. The upstream module also
provides neural entailment models for clustering free-form text; this work
clusters function calls by exact match and by abstract syntax tree instead
(see `ast_checker.py`), so none of that is needed here.
"""

import numpy as np


def logsumexp_by_id(semantic_ids, log_likelihoods, agg="sum_normalized"):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities are in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == "sum_normalized":
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Monte-Carlo estimate of entropy.

    ``E[-log p(x)] ~= -1/N sum_i log p(x_i)``, i.e. the average token
    likelihood.
    """
    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy


def predictive_entropy_rao(log_probs):
    """Rao's entropy over cluster log-probabilities."""
    # With a single semantic cluster the entropy is 0. Special-cased to avoid
    # underflow.
    if len(log_probs) == 1:
        return 0

    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Entropy of the distribution over cluster assignments.

    The categorical distribution over clusters is estimated from how often each
    semantic id occurs; the uncertainty is the entropy of that distribution.
    This estimate ignores token likelihoods and relies solely on the cluster
    assignments: probability mass spread over many clusters gives high entropy,
    mass concentrated on a few gives low entropy.

    Input:
        semantic_ids: list of semantic ids, e.g. ``[0, 1, 2, 1]``.
    Output:
        entropy, e.g. ``(-p log p).sum()`` for ``p = [1/4, 2/4, 1/4]``.
    """
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = -(probabilities * np.log(probabilities)).sum()
    return entropy
