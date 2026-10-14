import argparse
import json
import csv
import ast
import re
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 

"""
Extract logprobs from evaluation results to compute AUROC 
"""


def get_combined_simple_irrelevance_examples(to_combine_logprobs_data: Dict, to_combine_eval_data: Dict):

    # Remove those where the models refuse on irrelevance correctly, meaning that any refusal is correct,
    # so, we remove examples where accuracies is 1 for the irrelevance task     

    irrelevance_data = to_combine_logprobs_data["irrelevance"] 
    irrelevance_eval =  to_combine_eval_data["irrelevance"]
     

    # Now remove correct refusal in irrelevance 
    # First line is a metric dict, rest are individual questions where the model gets wrong 
    irrelevance_metric, irrelevance_errors = irrelevance_eval[0], irrelevance_eval[1:]
    print(f"Irrelevance Accuracy: {irrelevance_metric['accuracy']}")

    # Extract wrong prediction ids from evaluation results
    irrelevance_error_ids = [i["prompt"]["id"] for i in irrelevance_errors] # [simple_1, simple_4,...]
    
    # Extract ids for all wrong predictions: those the model fails to refuse
    irrelevance_error_predicted_data = [i for i in irrelevance_data if i["id"] in irrelevance_error_ids] 

    # Retain only the wrong ones 
    # Initialise accs as all incorrect predictions  
    irrelevance_accs = np.zeros(len(irrelevance_error_predicted_data))

    # For simple, it follows the standard get_accs()
    simple_data = to_combine_logprobs_data["simple"] 
    simple_eval = to_combine_eval_data["simple"]
     
    # First line is a metric dict, rest are individual questions where the model gets wrong 
    simple_metric, simple_errors = simple_eval[0], simple_eval[1:]
    print(f"Simple Accuracy: {simple_metric['accuracy']}")

    # Extract wrong prediction ids 
    simple_error_qs = [i["prompt"]["id"] for i in simple_errors] # [simple_1, simple_4,...]
    
    # Extract ids for all predictions
    simple_predicted_qs = [i["id"] for i in simple_data]

    # Initialise as all correct predictions and fill in the wrong ones
    simple_accs = np.ones(len(simple_predicted_qs))

    for i, q in enumerate(simple_predicted_qs):
        if q in simple_error_qs:
            simple_accs[i] = 0
    
    assert simple_accs.mean() == simple_metric["accuracy"]
    
    # Combine by concatenating the wrong irrelevance examples to all the simple examples
    combind_logprobs_data = simple_data + irrelevance_error_predicted_data
    combined_accs = np.concatenate([simple_accs, irrelevance_accs])
    assert len(combind_logprobs_data) == len(combined_accs)
    print(f"Number of wrongly non-refused examples:\
           {len(irrelevance_accs)} / {len(combined_accs)} = {len(irrelevance_accs) / len(combined_accs)} ")

    return combind_logprobs_data, combined_accs


def aggregate_toks_to_seq(logprobs: List[float], aggregate="min_logprob") -> float:
    """
    Aggregate token level scores to sequence level scores.

    Higher uncertainty score corresponds to lower odds of the response being correct.
    """
    if aggregate == "min":
        aggregated_score = np.min(logprobs)  
    elif aggregate == "sum":
        aggregated_score = np.sum(logprobs)  
    elif aggregate == "mean":
        aggregated_score = np.mean(logprobs)  
    else:
        raise NotImplementedError
    return aggregated_score


def get_uq_measure(data: List[Dict], aggregate_method="min_logprob") -> np.array:
 
    token_logprobs = [i["logprobs"]["token_logprobs"] for i in data] # list of token logprob lists 
    seq_logprobs = np.array([aggregate_toks_to_seq(i, aggregate_method) for i in token_logprobs])

    return seq_logprobs

def get_length_as_uq_measure(data: List[Dict]) -> np.array:
    # The longer an output is, the less certain it should be

    output_token_counts = np.array([ - i['output_token_count'] for i in data])
    return output_token_counts




def get_accs(predictions: List[Dict], eval_result: List[Dict]) -> List[float]:

    # First line is a metric dict, rest are individual questions where the model gets wrong 
    metric, errors = eval_result[0], eval_result[1:]
    print(f"Accuracy: {metric['accuracy']}")

    # Extract wrong prediction ids 
    error_qs = [i["prompt"]["id"] for i in errors] # [simple_1, simple_4,...]
    
    # Extract ids for all predictions
    predicted_qs = [i["id"] for i in predictions]

    # Initialise as all correct predictions and fill in the wrong ones
    accs = np.ones(len(predicted_qs))

    for i, q in enumerate(predicted_qs):
        if q in error_qs:
            accs[i] = 0
    
    assert accs.mean() == metric["accuracy"]
    return accs

def cummean(arr: np.array) -> np.array:
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)

def compute_aurc(
    loss: np.array,
    confidence_scores: np.array,
) -> float:
    """
    Compute AURC.

    Args:
        confidence_scores: uncertainty/confidence scores of shape :math:`(B,)`. The higher, the more likely correct.
        loss: vector of loss values of shape :math:`(B,)`
    """
    idxs = np.argsort(-confidence_scores)  # negative sign here because np.argsort sorts in ascending order and we need descending.
    ordered_loss = loss[idxs]
    risk_coverage_curve = cummean(ordered_loss)
    area_under_risk_coverage_curve = np.mean(risk_coverage_curve)
    return float(area_under_risk_coverage_curve)

def compute_naurc(
    loss: np.array,
    confidence_scores: np.array,
) -> float:
    """
    Compute NAURC as per eq. 10 in https://arxiv.org/abs/2305.15508.


    Args:
        confidence_scores: uncertainty/confidence scores of shape :math:`(B,)`. The higher, the more likely correct.
        loss: vector of loss values of shape :math:`(B,)`
    """
    # As per eq. 10 in https://arxiv.org/abs/2305.15508.
    aurc = compute_aurc(loss=loss, confidence_scores=confidence_scores)
    oracle_aurc = compute_aurc(loss=loss, confidence_scores=-loss)  # negative loss because it acts as confidence_scores
    risk = np.mean(loss)  # equivalent to expected AURC over random UQ scores
    naurc = (aurc - oracle_aurc)/(risk - oracle_aurc)
    return float(naurc)


def plot_rc_curve(
    loss: np.array,
    confidence_scores: np.array,
    fig_title: str="",
) -> float:
    """
    Plot NAURC graph.

    Args:
        confidence_scores: uncertainty/confidence scores of shape :math:`(B,)`. The higher, the more likely correct.
        loss: vector of loss values of shape :math:`(B,)`
    """
    xs = np.arange(1, len(loss)+1)/len(loss)

    idxs = np.argsort(-confidence_scores)  # negative sign here because np.argsort sorts in ascending order and we need descending.
    ordered_loss = loss[idxs]
    risk_coverage_curve = cummean(ordered_loss)
    plt.plot(xs, risk_coverage_curve, label="empirical")

    ordered_oracle_loss = np.sort(loss)
    plt.plot(xs, cummean(ordered_oracle_loss), label="oracle")

    plt.legend()
    plt.title(fig_title)
    plt.ylim(0, 0.5)
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.show()


def compute_auroc(accs, predictor_scores):
    """Compute auroc """
    assert len(predictor_scores) == len(accs)

    predictor_scores = np.array(predictor_scores, dtype=float)
    # since in the classification setting the higher the value the larger the likelihood of positive class (correct)
 
    accs = np.array(accs, dtype=float)

    # roc
    aucroc = metrics.roc_auc_score(accs, predictor_scores)

    # pr    
    aucprc = metrics.average_precision_score(accs, predictor_scores)

    return aucroc, aucprc

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--logprobs_path", type=str, nargs="+", help="Path to the generated result after running bfcl generate.")
    parser.add_argument("--evaluation_path", type=str, nargs="+", help="Path to the evaluated metric file after running bfcl evaluate.")
    parser.add_argument("--plot_rc_figures", type=str2bool, help="Set to true to plot the AURC figures.")
    parser.add_argument('--combine_simple_irrelevance', default=False, action=argparse.BooleanOptionalAction, \
                        help="Whether to combine simple and irrelevance. If True, then logprobs_path and evaluation_path each has to be of length 2 containing paths to each test.")
    args, unknown = parser.parse_known_args()

    all_logprobs_data = []
    for path in args.logprobs_path:
        with open(path, "r") as f:
            logprobs_data = []
            for line in f:
                logprobs_data.append(json.loads(line))
            all_logprobs_data.append(logprobs_data)

    

    all_eval_data = []
    for path in args.evaluation_path:
        with open(path, "r") as f:
            eval_data = []
            for line in f:
                eval_data.append(json.loads(line))
            all_eval_data.append(eval_data)

    # Get logprobs_data and corresponding accs differently depending on if combine_simple_irrelevance
    if args.combine_simple_irrelevance:
        assert len(all_logprobs_data) == 2 and len(all_eval_data) == 2, \
            f"Combine_simple_irrelevance is set to True, but length of data paths is not 2" 
        assert all_logprobs_data[0][0]["id"].startswith("simple") and \
            all_logprobs_data[1][0]["id"].startswith("irrelevance"), \
            f"Expect first path to be simple and second to be irrelevance"


        to_combine_logprobs_data = {"simple": all_logprobs_data[0], "irrelevance": all_logprobs_data[1]}
        to_combine_eval_data = {"simple": all_eval_data[0], "irrelevance": all_eval_data[1]}
        
        logprobs_data, accs = get_combined_simple_irrelevance_examples(to_combine_logprobs_data, to_combine_eval_data)
        loss = 1 - accs

    # If not combined, then each file path is length one
    else:
        assert len(all_logprobs_data) == 1 and len(all_eval_data) == 1

        logprobs_data = all_logprobs_data[0]
        eval_data = all_eval_data[0]
        accs = get_accs(logprobs_data, eval_data)
        loss = 1 - accs
   

    

    # To compute metrics like AUROC, we need two lists: 
    # (1) accuracy 1/0 for each test set example
    # (2) uncertainty measure scores (which is currently a function of single prediction sequence logprobs)
 
    # (1):
    # First, process the evaluated result file to see where the model makes wrong predictions in test set

    ###### Commenting this part out to be decided in each condition above #####
    # accs = get_accs(logprobs_data, eval_data)
    # loss = 1 - accs  # with binary accuracy: correct -> loss=0, incorrect -> loss=1
    ########

    # (2): 
    # Minimum token logprob of entire sequence (Semantic Parsing: https://arxiv.org/abs/2211.07443)
    print(f"Aggregation by:")
    for aggregate_method in ("min", "sum", "mean"):
        uq_scores = get_uq_measure(logprobs_data, aggregate_method=aggregate_method)
        aucroc, aucprc = compute_auroc(accs, uq_scores)
        naurc = compute_naurc(loss=loss, confidence_scores=uq_scores)
        print(f"{aggregate_method}:\tAUROC={aucroc:.2f}\tAUPRC={aucprc:.2f}\t1-NAURC={1-naurc:.2f}")
        if args.plot_rc_figures:
            plot_rc_curve(loss=loss, confidence_scores=uq_scores, fig_title=f"aggregate_method={aggregate_method}")


    aggregate_method = "Output token count"
    uq_scores = get_length_as_uq_measure(logprobs_data)
    aucroc, aucprc = compute_auroc(accs, uq_scores)
    naurc = compute_naurc(loss=loss, confidence_scores=uq_scores)
    print(f"{aggregate_method}: \tAUROC={aucroc:.2f}\tAUPRC={aucprc:.2f}\t1-NAURC={1-naurc:.2f}")
    print("\n")
    if args.plot_rc_figures:
        plot_rc_curve(loss=loss, confidence_scores=uq_scores, fig_title=f"aggregate_method={aggregate_method}")


    # Random baseline 
    aggregate_method = "random"
    uq_scores = np.random.default_rng(28032025).random(len(uq_scores))
    aucroc, aucprc = compute_auroc(accs, uq_scores)
    naurc = compute_naurc(loss=loss, confidence_scores=uq_scores)
    print(f"{aggregate_method}:\tAUROC={aucroc:.2f}\tAUPRC={aucprc:.2f}\t1-NAURC={1-naurc:.2f}")
    print("\n")
    if args.plot_rc_figures:
        plot_rc_curve(loss=loss, confidence_scores=uq_scores, fig_title=f"aggregate_method={aggregate_method}")