import argparse
import json
import pandas as pd
import ast
import re
import os
from typing import List, Dict, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 

import compute_auc 


from collections import defaultdict
import logging

from semantic_uncertainty.semantic_uncertainty.uncertainty.uncertainty_measures import p_true as p_true_utils
from semantic_uncertainty.semantic_uncertainty.uncertainty.utils import utils
from semantic_uncertainty.semantic_uncertainty.uncertainty.uncertainty_measures.p_ik import get_p_ik
from semantic_uncertainty.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids, \
                                                                                                        logsumexp_by_id, \
                                                                                                        predictive_entropy, \
                                                                                                        predictive_entropy_rao, \
                                                                                                        cluster_assignment_entropy, \
                                                                                                        context_entails_response, \
                                                                                                        EntailmentDeberta, \
                                                                                                        EntailmentGPT4, \
                                                                                                        EntailmentGPT35, \
                                                                                                        EntailmentGPT4Turbo, \
                                                                                                        EntailmentLlama
# from bfcl.eval_checker.ast_eval.ast_checker import ast_checker, simple_function_checker
from ast_checker_for_entropy import ast_checker, simple_function_for_entropy_checker


import scipy
import numpy as np
from sklearn.metrics import roc_auc_score


# TODO: Refactor computing semantic IDs.

def my_bootstrap(y_true, y_score, *args, **kwargs):
    if np.all(y_true == y_true[0]):
        # Only one class present, return AUROC 0.5 (arbitrary value; AUROC is not defined in this case)
        return 0.5
    return roc_auc_score(y_true, y_score, *args, **kwargs)



# bs = scipy.stats.bootstrap(
#     [y_true, y_score],
#     my_bootstrap,
#     n_resamples=1000,
#     confidence_level=0.95,
#     random_state=np.random.default_rng(28032025),
#     paired=True
# )

# print(bs.standard_error)

def bootstrap_auroc(y_true, y_score, n_resamples=1000):
    """
    e.g.
    y_true=np.array([0, 0, 1., 0, 1.])
    y_score=np.array([0.1, 0.7, 0.5, 0.9, 0.8])
    """
    bs = scipy.stats.bootstrap(
        [y_true, y_score],
        my_bootstrap,
        n_resamples=n_resamples,
        confidence_level=0.95,
        random_state=np.random.default_rng(28032025),
        paired=True
    )
    return bs








def normalise_high_temp_single_ast(list_response: list) -> list:
    """
    Single AST match. list_response is a list of list, where the length is 0 if decoding failed. If multiple ASTs are decoded, 
    take the first one for now.
    e.g. 
    [[], [{ast1_dict}], [{ast2_dict}, {ast3_dict}], ...] -> [[{}], [{ast1_dict}], [{ast2_dict}], ...]
    """
    # Single AST match. list_response is a list of list, where the length is 0 if decoding failed, otherwise 1
    # If multiple ASTs are decoded (which shouldn't be correct as we're in single mode but this occasionaly happens), take the first one for now 
    list_response = [ [l[0]] if len(l) > 0 else l for l in list_response]
    assert all(len(i) in [0, 1] for i in list_response) 
    # Assuming decoding failure as one extra cluster: empty dict
    # Replace empty lists with empty dict 
    
    processed_ast_list = [ [dict()] if len(ast) == 0 else ast for ast in list_response]
    return processed_ast_list


def create_possible_answer_item(item, prompt_item) -> Union[None, dict]:
    """
    {k:v} -> {k: [v]}, or {k:[v, ""]} if optional
    """
    # If the func_name in item matches any of the function names in functions provided, and within that func_name the 
    # parameter is optional, then add additional "" to the possible answers  
        
    # If the item is empty, meaning AST decoding failed, we cannot create a possible answer so return None
    if len(item) == 0:
        return None
    
    func_name, func_content = list(item)[0], list(item.values())[0]

    optional_params = []
    for function in prompt_item:
        # If the output function name matches the function name in the prompt 
        if func_name == function["name"]:
            # First get all the parameters both required and optional 
            all_params = list(function["parameters"]["properties"].keys())

            required_params = function["parameters"]["required"]
            # A list of optional params
            optional_params = [param for param in all_params if param not in required_params] 
            print("optional_params: ", optional_params)

    # If there are optional params, add additional "" to the possible answers
    possible_answer_item = {func_name: dict()}
    for k, v in func_content.items():
        if type(v) == dict:
            v = {key: [value] for key, value in v.items()} 

        possible_answer_item[func_name][k] = [v]
        if k in optional_params:
            possible_answer_item[func_name][k].append("")
            
    print("possible_answer_item: ", possible_answer_item)
    return possible_answer_item



def single_ast_equivalence(list1, list2, prompt_item, language) -> bool:
    """
    Is AST1 (result_item) within the range of AST2 (possible_answer_item)? Or, does AST2 entail AST1 uni-directionally? 
    """
    assert len(list1) == 1 and len(list2) == 1, f"Single AST match should have len=1, but list1: {list1}, list2: {list2}"
    ast1 = list1[0]
    ast2 = list2[0]
    print("In single_ast_equivalence...")
    print("ast1: ", ast1)
    print("ast2: ", ast2)

    # Use list2 as possible answer 
    ast2_as_possible_answer = create_possible_answer_item(ast2, prompt_item)
    print("ast2_as_possible_answer: ", ast2_as_possible_answer)
     # Repeat, use list1 as possible answer 
    ast1_as_possible_answer = create_possible_answer_item(ast1, prompt_item)

    # If we cannot create a possible answer, it means that AST2 decoding failed,
    # so we cannot compare AST1 with AST2, meaning they are not equivalent
    if ast2_as_possible_answer is None or ast1_as_possible_answer is None:
  
        return False
        

    print("func_description: ", prompt_item)
    # Use simple function checker for now 
    # TODO: Relax this assumption 
    
    # TODO: Use temporary model name for now  
    result1 = simple_function_for_entropy_checker(func_description=prompt_item[0],
            model_output=ast1, possible_answer=ast2_as_possible_answer, possible_answer_orig_output=ast2, language=language, model_name="TEMP"
        )
    print("result1: ", result1)

   
        # Use list2 as possible answer 
 
    print("ast1_as_possible_answer: ", ast1_as_possible_answer)
    # If we cannot create a possible answer, it means that AST1 decoding failed,
    # so we cannot compare AST1 with AST2, meaning they are not equivalent
 
 
    result2 = simple_function_for_entropy_checker(func_description=prompt_item[0],
    model_output=ast2, possible_answer=ast1_as_possible_answer, possible_answer_orig_output=ast1, language=language, model_name="TEMP"
        )
    print("result2: ", result2)
    # TEMP
    # Non-strict entailment 
    if result1["valid"] or result2["valid"]:
        return True
    else:
        return False 



def get_ids_with_ast_checker(prompt_item, processed_ast_list, language):
  

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(processed_ast_list)
    # Keep track of current id.
    next_id = 0
    for i, list1 in enumerate(processed_ast_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(processed_ast_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if single_ast_equivalence(list1, processed_ast_list[j], prompt_item, language):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def compute_ast_entropy(prompt_item, ast_responses, language, list_seq_logprobs: List[float]) -> Tuple[float]:
   
    semantic_ids = get_ids_with_ast_checker(prompt_item, ast_responses, language)
    
    print("semantic_ids: ", semantic_ids)
    # Compute entropy from frequencies of cluster assignments.
    dse = cluster_assignment_entropy(semantic_ids)
    # print("calculating log lik agg for: ", list_seq_logprobs)

    # Length normalization of generation probabilities.
    log_liks_agg = [np.mean(log_lik) for log_lik in list_seq_logprobs]
    # print("log_liks_agg: ", log_liks_agg)
    # Compute naive entropy.
    regular_entropy = predictive_entropy(log_liks_agg)

     # Compute semantic entropy.
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
    # print("log_likelihood_per_semantic_id: ", log_likelihood_per_semantic_id)
    se = predictive_entropy_rao(log_likelihood_per_semantic_id)
    # print("entailment_method: ", entailment_method)
    # print("se: ", se)
    return (se, dse, regular_entropy, semantic_ids)


def compute_SNLL(list_response: List, list_seq_logprobs: List[float], entailment_method="string_match"):
    
     # If entailment_method is not provided, get semantic_ids by string match
    if entailment_method == "string_match":
        semantic_ids = get_semantic_ids_with_string_match(list_response)
    elif entailment_method == "ast_single":
        # AST: list_response is a list of list, where the length is 0 if decoding failed, otherwise 1 

        # Single AST match. list_response is a list of list, where the length is 0 if decoding failed, otherwise 1
        # If multiple ASTs are decoded (which shouldn't be correct as we're in single mode but this occasionaly happens), take the first one for now 
        list_response = [ [l[0]] if len(l) > 0 else l for l in list_response]

        assert all(len(i) in [0, 1] for i in list_response), f"Using single ast match, but some responses have more than one AST: {list_response}"
        semantic_ids = get_semantic_ids_with_ast_match(list_response, single=True)

    elif entailment_method == "ast_multiple_or_parallel":
        # AST multiple or parallel. list_response is a list of list, where the length is 0 if decoding failed, otherwise >= 1
        semantic_ids = get_semantic_ids_with_ast_match(list_response, single=False)

    elif entailment_method == "ast_single_multiple_parallel":
        # Testing if if evaluating single as multiple would make a difference
        semantic_ids = get_semantic_ids_with_ast_match(list_response, single=False)

    else:
        raise NotImplementedError


    # Length normalization of generation probabilities.
    log_liks_agg = [np.mean(log_lik) for log_lik in list_seq_logprobs]
    
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')


    # Find the maximum cluster
    max_cluster_log_likelihood = np.max(np.array(log_likelihood_per_semantic_id))
    normalised_max_cluster_liklihood = np.exp(max_cluster_log_likelihood - np.log(np.sum(np.exp(log_liks_agg))))

    max_cluster_likelihood = np.exp(max_cluster_log_likelihood)
    # print("log_likelihood_per_semantic_id: ", log_likelihood_per_semantic_id)
    # print("max cluster: ", max_cluster_likelihood)
    # print("max_cluster_likelihood: ", max_cluster_likelihood)
    # print("normalised_max_cluster_liklihood: ", normalised_max_cluster_liklihood)
    # print("sanity check normalised: ", max_cluster_likelihood / np.sum(np.exp(log_liks_agg) ))

    # Compute discrete SNLL
    counts = np.bincount(semantic_ids)
    max_discrete_likelihood = np.float64(np.max(counts))

    categorical_distribution = counts / len(semantic_ids)
    assert np.isclose(categorical_distribution.sum(), 1)
    
    normalised_max_discrete_likelihood = np.max(categorical_distribution)


    return max_cluster_likelihood, normalised_max_cluster_liklihood, \
        max_discrete_likelihood, normalised_max_discrete_likelihood



def write_semantic_cluster_pooled_methods(greedy_high_temp_generation: list, entailment_method="string_match") -> dict:
 
    for row in greedy_high_temp_generation:
        assert "uq_scores" in row
        samples = row["high_temp_outputs"]

        # If entailment is using string match, then use the text field as response predictions
        if entailment_method == "string_match":
            responses, logprobs = zip(*[(sample["text"], sample["token_logprobs"]) for sample in samples])
        # If entailment is using ast match, then use the decoded ast as response predictions 
        else:
            assert entailment_method in ["ast_single", "ast_multiple_or_parallel", "ast_single_multiple_parallel"]
            responses, logprobs = zip(*[(sample["ast_decoded"], sample["token_logprobs"]) for sample in samples])
 
        ### Instead of computing se, dse, pe, we compute S-NLL 
        unnormalised_snll_score, normalised_snll_score, \
        unnormalised_dsnll_score, normalised_dsnll_score = compute_SNLL(responses, logprobs, entailment_method)


        row["uq_scores"][f"{entailment_method}_snll_unnormalised"] = unnormalised_snll_score
        row["uq_scores"][f"{entailment_method}_snll_normalised"] = normalised_snll_score 
        # print("unnormalised_dsnll_score: ", type(unnormalised_dsnll_score))
        # print("normalised_dsnll_score: ", type(normalised_dsnll_score))
   
        row["uq_scores"][f"{entailment_method}_dsnll_unnormalised"] = unnormalised_dsnll_score
        row["uq_scores"][f"{entailment_method}_dsnll_normalised"] = normalised_dsnll_score 

    return greedy_high_temp_generation
    



def write_ast_entropy(greedy_high_temp_generation: list, language) -> dict:
    """
    Use ast_checker logic to compare pair-wise entailment between high_temp generations.  
    """

    for row in greedy_high_temp_generation:
        assert "uq_scores" in row
        samples = row["high_temp_outputs"]
        
        
        function = row["function"]
      
        ast_responses, logprobs = zip(*[(sample["ast_decoded"], sample["token_logprobs"]) for sample in samples])
        # Normalise high_temp generations to have single AST match
        processed_ast_list = normalise_high_temp_single_ast(ast_responses) # [[{}], [{ast1_dict}], [{ast2_dict}], ...]
        
        
        se, dse, pe, semantic_ids = compute_ast_entropy(function, processed_ast_list, language, logprobs)
        
        #print("se: ", se)ailment_meth
        #print("dse: ", dse)
        #print("pe: ", pe)
        
        # for k, v in zip(["neg_se", "neg_dse", "neg_pe"], [se, dse, pe]):
        #     entropy_dict[k].append(-v)
        row["uq_scores"]["simple_ast_se"] = se
        row["uq_scores"]["simple_ast_dse"] = dse
        row["uq_scores"]["simple_ast_pe"] = pe
        row["uq_scores"]["simple_ast_semantic_ids"] = semantic_ids
    #entropy_dict = {k: np.array(v) for k, v in entropy_dict.items()}
    # print("entropy_dict: ", entropy_dict)
    #return entropy_dict
 
    return greedy_high_temp_generation
    
        



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



### Mistral SMT ####

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




 

################ MISTRAL-automated rubric ######################
from typing import List, Dict, Literal, Optional, Any

TokenType = Literal["nf", "np", "pv", "nfp", "-_"]


def classify_token_types_json_new(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: list of dicts, each containing at least:
        {"token_text": <str>, ...}

    Output: same list of dicts, with an added key:
        entry["token_type"] in {"nf","np","pv","nfp","-_"}

    Classification targets JSON-format tool calls like:
      [TOOL_CALLS][{"name":"math_toolkit.gcd","arguments":{"num1":15,"num2":25}}]
    """
    print("classifying token types with GPT's rubric")
    # JSON container stack: "object" or "array"
    stack: List[str] = []

    # Whether we're inside a JSON string right now
    in_string = False
    # When we ENTER a string, we decide whether it is a key or a value
    current_string_kind: Optional[Literal["key", "value"]] = None
    # Buffer for the full string content (for key tracking)
    string_buf: List[str] = []

    # Track the most recent completed key (for mapping values)
    last_key: Optional[str] = None

    # Track whether we are currently inside the object that is the value of "arguments"
    in_arguments = False
    # The object-depth (count of nested { ... }) at which the arguments object lives
    arguments_obj_depth: Optional[int] = None
    obj_depth = 0  # counts '{' minus '}' outside strings

    # Track last non-space char seen OUTSIDE strings (helps decide key vs value when a string opens)
    last_sig_char: Optional[str] = None

    def is_pure_syntax(tok: str) -> bool:
        s = tok.strip()
        if s == "":
            return True
        glue_chars = set('" :{[}]')
        return all(ch in glue_chars for ch in s)

    def has_content_chars(tok: str) -> bool:
        return any(ch.isalnum() or ch in "._-/" for ch in tok)

    def decide_string_kind() -> Literal["key", "value"]:
        # Keys occur only when inside an object and just after '{' or ','.
        # Values occur after ':' or inside arrays after '[' or ','.
        if stack and stack[-1] == "object" and last_sig_char in ("{", ","):
            return "key"
        return "value"

    for entry in tokens:
        tok = entry.get("token_text", "")
        if tok is None:
            tok = ""
        tok = str(tok)

        token_type: Optional[TokenType] = None

        # Mark "continue/stop" punctuation as semantic nfp
        stripped = tok.strip()
        if "," in stripped:
            token_type = "nfp"
        elif any(c in stripped for c in ("}", "]")):
            token_type = "nfp"

        # Character-level scan to track strings/containers robustly
        i = 0
        while i < len(tok):
            ch = tok[i]

            if in_string:
                if ch == '"':
                    # close string
                    in_string = False
                    completed = "".join(string_buf)
                    string_buf.clear()

                    if current_string_kind == "key":
                        last_key = completed

                    current_string_kind = None
                else:
                    string_buf.append(ch)
                i += 1
                continue

            # NOT in_string
            if ch == '"':
                in_string = True
                current_string_kind = decide_string_kind()
                string_buf.clear()
                i += 1
                continue

            # Track containers outside strings
            if ch == "{":
                stack.append("object")
                obj_depth += 1
                last_sig_char = "{"

                if last_key == "arguments":
                    in_arguments = True
                    arguments_obj_depth = obj_depth
                i += 1
                continue

            if ch == "}":
                if stack and stack[-1] == "object":
                    stack.pop()
                obj_depth = max(0, obj_depth - 1)
                last_sig_char = "}"

                if in_arguments and arguments_obj_depth is not None and obj_depth < arguments_obj_depth:
                    in_arguments = False
                    arguments_obj_depth = None
                i += 1
                continue

            if ch == "[":
                stack.append("array")
                last_sig_char = "["
                i += 1
                continue

            if ch == "]":
                if stack and stack[-1] == "array":
                    stack.pop()
                last_sig_char = "]"
                i += 1
                continue

            if ch == ":":
                last_sig_char = ":"
                i += 1
                continue

            if ch == ",":
                last_sig_char = ","
                i += 1
                continue

            if ch.isspace():
                i += 1
                continue

            last_sig_char = ch
            i += 1

        # Decide classification if not already nfp
        if token_type is None:
            if in_string and has_content_chars(tok):
                if current_string_kind == "key":
                    token_type = "np" if in_arguments else "-_"
                else:
                    if last_key == "name":
                        token_type = "nf"
                    elif in_arguments and (last_key is not None) and (last_key != "arguments"):
                        token_type = "pv"
                    else:
                        token_type = "-_"
            else:
                if in_arguments and stripped and all(c.isdigit() or c in ".-+eE" for c in stripped):
                    token_type = "pv"
                else:
                    token_type = "-_" if is_pure_syntax(tok) else "-_"

        entry["token_type"] = token_type
    return tokens








#####################################




#### Non-mistral SMT ###

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


####

def write_SMT_greedy_logprobs_measure(greedy_high_temp_generation: list, is_mistral: bool) -> dict:
    
    for row in greedy_high_temp_generation:
        assert "uq_scores" in row
        sample = row["greedy_output"]

        tokens = format_top_logprobs(sample["top_logprobs"])
                                    
        if is_mistral:
            tokens_processed = classify_token_types_json(tokens)
            # tokens_processed = classify_token_types_json_new(tokens)
            print("is mistral, token_processed: ", tokens_processed)
        else:
            tokens_processed = classify_token_types(tokens)
            print("not mistral, token_processed: ", tokens_processed)
        
        # Newly filtered token_logprobs for SMT
        SMT_token_logprobs = extract_relevant_logprobs(tokens_processed) #[extract_relevant_logprobs(i) for i in tokens_processed]
        # Calculate GNLL, AVG, MAX using SMT_token_logprobs
        print("SMT_token_logprobs: ", SMT_token_logprobs)
        for aggregate_method in ["min", "sum", "mean"]:

            aggregated_logprob = aggregate_toks_to_seq(SMT_token_logprobs, aggregate=aggregate_method)
            row["uq_scores"][f"SMT_{aggregate_method}"] = aggregated_logprob
 

    return greedy_high_temp_generation


def write_greedy_logprobs_measure(greedy_high_temp_generation: list) -> dict:
    for row in greedy_high_temp_generation:
        assert "uq_scores" in row
        sample = row["greedy_output"]
        token_logprobs = sample["token_logprobs"]
        for aggregate_method in ["min", "sum", "mean"]:

            aggregated_logprob = aggregate_toks_to_seq(token_logprobs, aggregate=aggregate_method)
            row["uq_scores"][aggregate_method] = aggregated_logprob

    return greedy_high_temp_generation


def write_length_measure(greedy_high_temp_generation: list) -> dict:
    
    for row in greedy_high_temp_generation:
        assert "uq_scores" in row
        sample = row["greedy_output"]
 
        row["uq_scores"]["length"]  =  sample["token_count"]
    return greedy_high_temp_generation


def write_ptrue(greedy_high_temp_generation: list) -> dict:
    
    for row in greedy_high_temp_generation:
        assert "uq_scores" in row
        sample = row["greedy_output"]
 
        row["uq_scores"]["ptrue"]  =  sample["p_true_log_prob"]
    return greedy_high_temp_generation



def write_semantic_entropy(greedy_high_temp_generation: list, entailment_method="string_match") -> dict:
    

    # TODO: implement condition_on_question by concatenating question to responses 
 
    for row in greedy_high_temp_generation:
        assert "uq_scores" in row
        samples = row["high_temp_outputs"]

        # If entailment is using string match, then use the text field as response predictions
        if entailment_method == "string_match":
            responses, logprobs = zip(*[(sample["text"], sample["token_logprobs"]) for sample in samples])
        # If entailment is using ast match, then use the decoded ast as response predictions 
        else:
            assert entailment_method in ["ast_single", "ast_multiple_or_parallel", "ast_single_multiple_parallel"]
            responses, logprobs = zip(*[(sample["ast_decoded"], sample["token_logprobs"]) for sample in samples])
        #print("responses: ", responses)
        #print("logprobs: ", logprobs)
        se, dse, pe, semantic_ids = compute_semantic_entropy(responses, logprobs, entailment_method)
        #print("se: ", se)
        #print("dse: ", dse)
        #print("pe: ", pe)
        

        row["uq_scores"][f"{entailment_method}_se"] = se
        row["uq_scores"][f"{entailment_method}_dse"] = dse
        row["uq_scores"][f"{entailment_method}_pe"] = pe
        row["uq_scores"][f"{entailment_method}_semantic_ids"] = semantic_ids
        row["uq_scores"][f"{entailment_method}_num_clusters"] = np.float64(len(set(semantic_ids)))
        print("num_clusters type: ", type(np.float64(len(set(semantic_ids)))))
        print("se type: ", type(se))
      
    #entropy_dict = {k: np.array(v) for k, v in entropy_dict.items()}
    # print("entropy_dict: ", entropy_dict)
    #return entropy_dict
    return greedy_high_temp_generation
    

    


def compute_semantic_entropy(list_response: List, list_seq_logprobs: List[float], entailment_method="string_match") -> Tuple[float]:
    """
    Compute SE given one sample.
    
    Args:
        list_response: list of high temperature response texts or ASTs
        list_seq_logprobs: list of sequence log probs for all the responses for aggregation
    
    
    Returns SE for this sample: (SE, discrete_SE, regular_entropy)
    """
    # If entailment_method is not provided, get semantic_ids by string match
    if entailment_method == "string_match":
        semantic_ids = get_semantic_ids_with_string_match(list_response)
    elif entailment_method == "ast_single":
        # AST: list_response is a list of list, where the length is 0 if decoding failed, otherwise 1 

        # Single AST match. list_response is a list of list, where the length is 0 if decoding failed, otherwise 1
        # If multiple ASTs are decoded (which shouldn't be correct as we're in single mode but this occasionaly happens), take the first one for now 
        list_response = [ [l[0]] if len(l) > 0 else l for l in list_response]

        assert all(len(i) in [0, 1] for i in list_response), f"Using single ast match, but some responses have more than one AST: {list_response}"
        semantic_ids = get_semantic_ids_with_ast_match(list_response, single=True)

    elif entailment_method == "ast_multiple_or_parallel":
        # AST multiple or parallel. list_response is a list of list, where the length is 0 if decoding failed, otherwise >= 1
        semantic_ids = get_semantic_ids_with_ast_match(list_response, single=False)

    elif entailment_method == "ast_single_multiple_parallel":
        # Testing if if evaluating single as multiple would make a difference
        semantic_ids = get_semantic_ids_with_ast_match(list_response, single=False)

    else:
        raise NotImplementedError
    # print("semantic_ids: ", semantic_ids)
    # Compute entropy from frequencies of cluster assignments.
    dse = cluster_assignment_entropy(semantic_ids)
    # print("calculating log lik agg for: ", list_seq_logprobs)

    # Length normalization of generation probabilities.
    log_liks_agg = [np.mean(log_lik) for log_lik in list_seq_logprobs]
    # print("log_liks_agg: ", log_liks_agg)
    # Compute naive entropy.
    regular_entropy = predictive_entropy(log_liks_agg)

     # Compute semantic entropy.
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
    # print("log_likelihood_per_semantic_id: ", log_likelihood_per_semantic_id)
    se = predictive_entropy_rao(log_likelihood_per_semantic_id)
    # print("entailment_method: ", entailment_method)
    # print("se: ", se)
    return (se, dse, regular_entropy, semantic_ids)


def get_semantic_ids_with_ast_match(ast_list, single=True):
    def are_equivalent(list1, list2):
        # Check for pair-wise dictionary equivalence 
        if single:
            assert len(list1) == 1 and len(list2) == 1, f"Single AST match should have len=1, but list1: {list1}, list2: {list2}"
            ast1 = list1[0]
            ast2 = list2[0]
            # TODO: add more sophisticated AST comparison. Now it's just dictionary match 
            semantically_equivalent = (ast1 == ast2)

        else:
            assert len(list1) >= 1 and len(list2) >= 1, f"Multiple AST match should have len>=1, but list1: {list1}, list2: {list2}"
            # order invariant 
            sorted_list1 = sorted([i.items() for i in list1])
            sorted_list2 = sorted([i.items() for i in list2])
            semantically_equivalent = (sorted_list1 == sorted_list2)
            # print("simple checking no order preserves")
            # semantically_equivalent = (list1 == list2)

        return semantically_equivalent

    # Assuming decoding failure as one extra cluster: empty dict
    # First replace empty lists with empty dict 
    # [[], [ast1_dict], [ast2_dict, ast3_dict], ...] -> [[{}], [ast1_dict], [ast2_dict, ast3_dict], ...]
    processed_ast_list = [ [dict()] if len(ast) == 0 else ast for ast in ast_list]

    # print("processed_ast_list: ", processed_ast_list)
    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(processed_ast_list)
    # Keep track of current id.
    next_id = 0
    for i, list1 in enumerate(processed_ast_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(processed_ast_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(list1, processed_ast_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids





def get_semantic_ids_with_string_match(strings_list):
    """Create semantic id clusters solely based on string match."""

    def are_equivalent(text1, text2):
        semantically_equivalent = (text1 == text2)         
        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids



def get_sampled_negative_predictive_entropy(greedy_generation, multi_generations) -> np.array:

    predictive_entropy_lst = []
    # match id by comparing with greedy generation
    # then extract logprobs of samples
    for row in greedy_generation:
        id = row["id"] 
        samples = multi_generations[id]
        samples_logprobs = [sample["seq_logprobs"] for sample in samples]
        predictive_entropy = compute_predictive_entropy(samples_logprobs)
        predictive_entropy_lst.append(predictive_entropy)
        # negate since we take confidence score: lower the entropy the higher the confidence
        neg_sampled_predictive_entropy_lst = - np.array(predictive_entropy_lst)

    return neg_sampled_predictive_entropy_lst


def compute_predictive_entropy(list_seq_logprobs: List[List[float]]) -> float:
    """
    Compute MC estimate of predictive entropy 
    Eq. (8) from https://arxiv.org/pdf/2002.07650

    Args:
        list_seq_logprobs: a list of lists of length :math:`(M,)`, where M is the sample size of high temperature generations.
        Each of the M lists within contains the logprobs of the sequence sampled.
    """
    
    # Avg token logprob for each sampled sequence
    # print("list_seq_logprobs: ", list_seq_logprobs)
    mean_logprob = [np.mean(seq_logprobs) for seq_logprobs in list_seq_logprobs]
    
    # Negative mean across samples
    predictive_entropy = - np.mean(mean_logprob)
 
    # print("predictive_entropy: ", predictive_entropy)
    return float(predictive_entropy)




if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--combined_path", type=str, help="Path to the combined path")
    parser.add_argument("--test_name", type=str, help="Test name")
    parser.add_argument("--save_path", type=str, help="Path to save the combined json.")
    parser.add_argument("--plot_rc_figures", type=compute_auc.str2bool, help="Set to true to plot the AURC figures.")
    # parser.add_argument("--model_name", type=str, help="Model to compute UQ for")
    # parser.add_argument("--path_to_multi_generations", type=str, nargs="+", help="Path to multiple high temperature generations")
    # parser.add_argument('--combine_simple_irrelevance', default=False, action=argparse.BooleanOptionalAction, \
    #                     help="Whether to combine simple and irrelevance. If True, then logprobs_path and evaluation_path each has to be of length 2 containing paths to each test.")

    args, unknown = parser.parse_known_args()
    # with open(f'{args.combined_path}/combined_{args.test_name}_with_uq.json') as f:
    with open(f'{args.combined_path}/combined_{args.test_name}.json') as f:
        data = json.load(f)

    
    # First, given a combined file with greedy and high temp outputs, write different UQ scores in-place to create a new UQ-annotated file

    # First write string match semantic entropy
    data = write_semantic_entropy(data, entailment_method="string_match")

    # Then write AST semantic entropy with dictionary match
    entailment_method = "ast_single_multiple_parallel"
    data = write_semantic_entropy(data, entailment_method=entailment_method)
    
    # Write length as UQ 
    data = write_length_measure(data)

    # Write ptrue as UQ
    # data = write_ptrue(data)

    # Write different greedy logprobs-based methods (sum, min, mean)
    data = write_greedy_logprobs_measure(data)

    # Write SMT greedy logprobs-based method
    is_mistral = "mistral" in args.combined_path.lower()

    data = write_SMT_greedy_logprobs_measure(data, is_mistral=is_mistral)

    data = write_semantic_cluster_pooled_methods(data, entailment_method="string_match")
    
    data = write_semantic_cluster_pooled_methods(data, entailment_method="ast_single_multiple_parallel")


    
    # se_scores = get_negative_semantic_entropy(greedy_high_temp_combined, entailment_method=None)
    # Save the annotated file
    with open(f'{args.save_path}/combined_{args.test_name}_with_uq.json', 'w') as f:
        json.dump(data, f)
    print(f"Finished writing uq scores to json: {args.save_path}/combined_{args.test_name}_with_uq.json")




    # Greedy accs
    accs = np.array([row["greedy_output"]["accuracy"] for row in data])
    loss = 1. - accs

    # Average high temp accs
    avg_accs = np.array([np.mean([high_temp["accuracy"] for high_temp in row["high_temp_outputs"]]) for row in data])
    avg_loss = 1. - avg_accs 

    loss_dict = {"greedy": {"acc": accs, "loss": loss}, "avg": {"acc": avg_accs, "loss": avg_loss}}


    mean_acc_dict = {"greedy": {"acc": np.mean(accs)}, "test_size": len(accs)}
    
    print(f"Accuracy for {args.test_name}: ", mean_acc_dict)
    print(f"Size of test set: {len(accs)}")


    # Compute AUROC, NAURC for different UQ methods

    # greedy logprob-based methods
    min_logprob = np.array([row["uq_scores"]["min"] for row in data])
    sum_logprob = np.array([row["uq_scores"]["sum"] for row in data])
    mean_logprob = np.array([row["uq_scores"]["mean"] for row in data])

    # SMT greedy logprob-based methods
    smt_min_logprob = np.array([row["uq_scores"]["SMT_min"] for row in data])
    smt_sum_logprob = np.array([row["uq_scores"]["SMT_sum"] for row in data])
    smt_mean_logprob = np.array([row["uq_scores"]["SMT_mean"] for row in data])

    
    # S-NLL 

    snll = np.array([row["uq_scores"]["string_match_snll_unnormalised"] for row in data])
    n_snll = np.array([row["uq_scores"]["string_match_snll_normalised"] for row in data])
    ast_snll = np.array([row["uq_scores"]["ast_single_multiple_parallel_snll_unnormalised"] for row in data])
    ast_n_snll = np.array([row["uq_scores"]["ast_single_multiple_parallel_snll_normalised"] for row in data])

    # DS-LL
    dsnll = np.array([row["uq_scores"]["string_match_dsnll_unnormalised"] for row in data])
    n_dsnll = np.array([row["uq_scores"]["string_match_dsnll_normalised"] for row in data])
    ast_dsnll = np.array([row["uq_scores"]["ast_single_multiple_parallel_dsnll_unnormalised"] for row in data])
    ast_n_dsnll = np.array([row["uq_scores"]["ast_single_multiple_parallel_dsnll_normalised"] for row in data])

    # Num of clusters
    num_clusters = np.array([- row["uq_scores"]["string_match_num_clusters"] for row in data])
    num_ast_clusters = np.array([- row["uq_scores"]["ast_single_multiple_parallel_num_clusters"] for row in data])
    

    # Length 
    nlen = np.array([- row["uq_scores"]["length"] for row in data])
    
    # PTrue
    # If ptrue is not found, set to an array of None
    if "ptrue" not in data[0]["uq_scores"]:
        ptrue = np.array([None] * len(data))
    else:
        ptrue = np.array([row["uq_scores"]["ptrue"] for row in data])

    # String match methods 
    nse = np.array([- row["uq_scores"]["string_match_se"] for row in data])
    ndse = np.array([- row["uq_scores"]["string_match_dse"] for row in data])
    npe = np.array([- row["uq_scores"]["string_match_pe"] for row in data])

    # AST match methods
    ast_nse = np.array([- row["uq_scores"][f"{entailment_method}_se"] for row in data])
    ast_ndse = np.array([- row["uq_scores"][f"{entailment_method}_dse"] for row in data])
    
  



    # Random baseline
    random_scores = np.random.rand(len(data))

    # Results to write to csv:
    uq_methods = []
    aucrocs = []
    aucprcs = []
    one_minus_greedy_naurcs = []
    one_minus_avg_naurcs = []

    # SE
    auroc_ses = []



    # For all uq_methods, compute AUCROC, AUCPRC, greedy_NAURC, and avg_NAURC on the given task 

 
    # Remove checker-logic ast for now
    # ["Simple AST logprobs SE", "AST logprobs Discrete SE"] 
    # simple_ast_nse, simple_ast_ndse]:

    for name, score in zip(["min", "sum", "mean", \
                            "SMT_min", "SMT_sum", "SMT_mean", \
                            "logprobs SE", "discrete SE", "Predictive E",\
                            "AST logprobs SE", "AST discrete SE", \
                            "SNLL", "N_SNLL", "AST_SNLL", "AST_N_SNLL", \
                            "DSNLL", "N_DSNLL", "AST_DSNLL", "AST_N_DSNLL", \
                            "num_clusters", "num_ast_clusters", \
                            "ptrue", \
                            "length", "random",\
                             ], \
                            [min_logprob, sum_logprob, mean_logprob, \
                            smt_min_logprob, smt_sum_logprob, smt_mean_logprob, \
                            nse, ndse, npe, \
                            ast_nse, ast_ndse, \
                            snll, n_snll, ast_snll, ast_n_snll, \
                            dsnll, n_dsnll, ast_dsnll, ast_n_dsnll, \
                            num_clusters, num_ast_clusters, \
                            ptrue, \
                            nlen, random_scores, \
                            ]):


        if np.all(score == None):
            print(f"Skipping {name} as all scores are None.")
            continue

        # Greedy scores first 
        print("COMPUTING AUROC")
        print("score: ", score)
        print("greedy acc: ", loss_dict["greedy"]["acc"])
        aucroc, aucprc = compute_auc.compute_auroc(loss_dict["greedy"]["acc"], score)
        greedy_naurc = compute_auc.compute_naurc(loss=loss_dict["greedy"]["loss"], confidence_scores=score)

        if args.plot_rc_figures:
            compute_auc.plot_rc_curve(loss=loss_dict["greedy"]["loss"], confidence_scores=score, fig_title=f"aggregate_method={name}")
        # print("\n")

        # naurc with avg score  
        avg_naurc = compute_auc.compute_naurc(loss=loss_dict["avg"]["loss"], confidence_scores=score)

        if args.plot_rc_figures:
            compute_auc.plot_rc_curve(loss=loss_dict["avg"]["loss"], confidence_scores=score, fig_title=f"aggregate_method={name}")
        # print("\n")


        ###########################
        # Add bootstrap SE to compute confidence intervals
        bs = bootstrap_auroc(y_true=loss_dict["greedy"]["acc"], y_score=score, n_resamples=1000)
         
        fig, ax = plt.subplots()
        ax.hist(bs.bootstrap_distribution, bins=25)
        ax.set_title('Bootstrap Distribution')
        ax.set_xlabel('statistic value')
        ax.set_ylabel('frequency')
        plt.savefig(f'{args.combined_path}/bs_{name}_{args.test_name}.png')
        plt.close(fig)
        ############################

        # Append to csv 
        uq_methods.append(name)
        aucrocs.append(aucroc)
        aucprcs.append(aucprc)
        one_minus_greedy_naurcs.append(1 - greedy_naurc)
        one_minus_avg_naurcs.append(1 - avg_naurc)

        # print("uq_methods: ", name, "aucrocs: ", aucroc, "se: ", bs.standard_error)
        auroc_ses.append(bs.standard_error)

    uq_metrics = {"UQ methods": uq_methods, \
                        "AUROC": aucrocs, \
                        "AUPRC": aucprcs, \
                        "1-NAURC (greedy)": one_minus_greedy_naurcs, \
                        "1-NAURC (avg)": one_minus_avg_naurcs, \
                        "standard_error": auroc_ses}
    
    uq_metrics_table = pd.DataFrame(uq_metrics)
    # Save the metrics output to combined file
    metrics_save_path = f'{args.save_path}/combined_uq_{args.test_name}.csv'
    uq_metrics_table.to_csv(metrics_save_path, index=False)  
    print(f"Saved metrics to {metrics_save_path}")

    # Save the mean accuracy to a txt file
    mean_acc_save_path = f'{args.save_path}/mean_acc_{args.test_name}.txt'
    with open(mean_acc_save_path, 'w') as f:
        f.write(str(mean_acc_dict))
    print(f"Saved mean accuracy to {mean_acc_save_path}")


 
    