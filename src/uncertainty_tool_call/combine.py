import argparse
import json
import csv
import ast
import re
import os
from pprint import pprint
from typing import List, Dict, Tuple, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics 

import compute_auc 


from collections import defaultdict
from dataclasses import dataclass, field, asdict 

from bfcl.constant import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    MULTI_TURN_FUNC_DOC_PATH,
    PROJECT_ROOT,
    PROMPT_PATH,
    RESULT_PATH,
    POSSIBLE_ANSWER_PATH,
    TEST_FILE_MAPPING,
    TEST_IDS_TO_GENERATE_PATH,
)
from bfcl.eval_checker.eval_runner_helper import load_file

"""
Once the generated and evaluated files are saved to result/ and score/, given a model, combine all greedy and
high_temp generations to a single json here, for specified categories
Input: 
file_path_to_greedy_result, file_path_to_greedy_score, file_path_to_high_temp_result, file_path_to_high_temp_score

Output:
Combined json files under generated_outputs/model_name

"""



@dataclass
class UQEntity:
    text: str
    token_count: int
    token_logprobs: list[float]
    top_logprobs: list[dict]
    accuracy: int
    error_type: Union[str, None] = None
    ast_decoded: Union[list, None] = None
    p_true_log_prob: Union[float, None] = None
    





def combine_outputs_per_task(task_name, model_name, greedy_generation_path, greedy_score_path, greedy_ast_path, \
                             ptrue_path, \
                             path_to_multi_generations, path_to_multi_scores,\
                             skip_non_irrelevance_decoder_failure:bool=True, skip_irrelevance_success:bool=True, \
                            skip_any_incorrect_simple:bool=False) -> list:
    """
    Processes and aggregates generation outputs from an LLM across both greedy and high-temperature decoding settings,
    preparing data for uncertainty quantification (UQ) analysis.

    This function performs the following steps:
    - Iterates through multiple decoding temperature directories (e.g., m=3, m=5) to load model outputs.
    - For each example ID:
        - Loads and evaluates greedy (temperature=0) output accuracy and error type.
        - Optionally filters out examples based on task-specific conditions and error types.
        - Iterates over high-temperature generations, collecting logprobs, decoded ASTs, and error labels.
        - Put each output into a UQEntity dataclass containing token-level info and metadata.
        - Adds question/function metadata for downstream analysis or evaluation.

    Filtering conditions are controlled via flags such as:
    - `skip_irrelevance_success`
    - `skip_non_irrelevance_decoder_failure`
    - `skip_any_incorrect_simple`

    Returns:
        List[Dict]: A list of dictionaries, each representing a single example with fields:
            - "id": Unique identifier for the example
            - "greedy_output": Dict with logprobs, token count, accuracy, etc.
            - "high_temp_outputs": List of dicts, length is the number of high-temp generations
            - "question": The input question for the example
            - "function": The function in the original prompt
            - "uq_scores": Placeholder dict for future uncertainty quantification metrics

    Notes:
        - Skipped examples (based on flags and task-specific logic) are not included in the final output.
        - The function assumes access to external data loading functions:
            `load_error_data`, `load_ast_data`, and a `UQEntity` dataclass.
    """



    # Create mapping of {id: {question: , function:,}}. Size of dict is length of dataset / greedy_generation.
    def load_question_data(path) -> dict:
        question_data = dict()

        for row in dataset_question_file:
            id = row["id"]
            question_data[id] = {"question": row["question"], "function": row["function"]}#result 

        return question_data 
    


    # Create mapping of {id: logprobs_data generated}. Size of dict is length of all greedy_generation.
    def load_logprobs_data(path) -> dict:
        logprobs_data = dict()
        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)
                id = row["id"]
                logprobs_data[id] = row #result 
        # print("logprobs_data: ", logprobs_data)
        return logprobs_data 
    

    # Create mapping of {id: error_type}. Size of dict is number of errors - contains only the error entries.
    def load_error_data(path) -> dict:
        score_data = []
        score_dict = dict()
        with open(path, "r") as f:
            for line in f:
                score_data.append(json.loads(line))
        metric, errors = score_data[0], score_data[1:]

        for row in errors:
            error_id = row["prompt"]["id"]
            score_dict[error_id] = row["error_type"]
        # print("score_dict: ", score_dict)
        return score_dict
    
    # Create mapping of {id: ast decoded}. Size of dict is length of all greedy_generation.
    def load_ast_data(path) -> dict:
        ast_dict = dict()
        ast_data = []
     
        with open(path, "r") as f:
            for line in f:
                ast_data.append(json.loads(line))
        metric, ast = ast_data[0], ast_data[1:]

        for row in ast:
            ast_id = row["prompt"]["id"]
            ast_dict[ast_id] = row["model_result_decoded"] # a list 
 
        return ast_dict

    def load_ptrue(path) -> dict:
        print("path for ptrue: ", path)
        ptrue_data_by_id = dict()
        with open(path) as f:
            ptrue_logprobs_data = json.load(f)
        
        for row in ptrue_logprobs_data:
            id = row["id"]
            ptrue_val = row["p_true_log_prob"]
            print("row: ", row)
            
            ptrue_data_by_id[id] = ptrue_val
        return ptrue_data_by_id
    

 
    greedy_ids = []
    with open(greedy_generation_path, "r") as f:
        for line in f:
            row = json.loads(line)
            greedy_ids.append(row["id"])
    

    # Load the original dataset, this is to extract the question and function for each id
    # questions = ["simple_0": {"question": , "function": }, "simple_1": {"question": , "function": }, ...]
    # len(questions) == len(greedy_generation)
    test_file_path = TEST_FILE_MAPPING[task_name]
    dataset_question_file = load_file(PROMPT_PATH / test_file_path)
    questions = load_question_data(dataset_question_file) # Create mapping of {id: {question: , function:}, id: {...},}.


    # Load greedy outputs
    # greedy_logprobs_dict = {"simple_0": logprobs_data, "simple_1": logprobs_data}.
    greedy_logprobs_dict = load_logprobs_data(greedy_generation_path)
    # greedy_error_dict is a dictionary with only ids for which the models answer wrong. 
    # greedy_error_dict = {"simple_49": "decoder_error...", "simple_302": "wrong_argument...", ...} 
    # len(greedy_error_dict) == # errors <= len(greedy_generation)
    greedy_error_dict = load_error_data(greedy_score_path)

    greedy_ast_dict = load_ast_data(greedy_ast_path)
    
    ptrue_dict = load_ptrue(ptrue_path) if ptrue_path is not None else None

    # Load high temp outputs
    high_temp_logprobs_data = dict()
    high_temp_error_data = dict()
    high_temp_ast_data = dict()


    # Now open each of the high temp files and load the logprobs  
    # Collect a dict of high_temp logprobs per m: {"m=1": {"simple_0": logprobs, "simple_1": logprobs, ...},\
    #                                              "m=2": {"simple_0": logprobs, "simple_1": logprobs, ...} ...}

    ################################################
    multi_generation_dir = os.listdir(path_to_multi_generations)
    # ### Combine only the first 10 high-temp  ###
    # multi_generation_dir = multi_generation_dir[:10]
    #####
    for dir_m in multi_generation_dir:
        assert "m=" in dir_m
        generated_jsons = os.listdir(os.path.join(path_to_multi_generations, dir_m, model_name))
  
        for task_name_json in generated_jsons:
            # e.g. Open the right file by comparing file name: extract "simple" from file name "BFCL_v3_simple_result.json"
            name = task_name_json.split("_result")[0].split("v3_")[-1]
  
            if name == task_name: 
                high_temp_path_m = os.path.join(path_to_multi_generations, dir_m, model_name, task_name_json)
                # print("high_temp_path_m_logprob: ", high_temp_path_m)
                high_temp_logprobs_m = load_logprobs_data(high_temp_path_m)
                high_temp_logprobs_data[dir_m] = high_temp_logprobs_m
 

    # Now open each of the high temp files and load the scores and ast
    # Collect a dict of high_temp_error_data per m: {"m=1": {"simple_30": "decoding_failed", ...},\
    #                                               "m=2": {"simple_20": "wrong_function_name", "simple_61": "missing_func", ...} ...}
    # Collect a dict of high_temp_ast_data per m: {"m=1": {"simple_0": ast, "simple_1": ast...},\
    #                                               "m=2": {"simple_0": ast, "simple_1": ast...} ...}

    multi_generation_dir = os.listdir(path_to_multi_scores)
    ### Combine only the first 10 high-temp  ###
    # multi_generation_dir = multi_generation_dir[:10]
    print("multi_generation_dir: ", multi_generation_dir)
    ####
    for dir_m in multi_generation_dir:
        # print("dir_m under path_to_multi_scores: ", dir_m)
        assert "m=" in dir_m
        generated_jsons = os.listdir(os.path.join(path_to_multi_scores, dir_m, model_name))

        for task_name_json in generated_jsons:

            # Process both decoded_ast files and score files 
            if "score" in task_name_json:
                # e.g. Open the right file by comparing file name: extract "javascript" from file name "BFCL_v3_javascript_score.json"
                name = task_name_json.split("_score")[0].split("v3_")[-1]
                # print("name: ", name)
                if name == task_name: 
                    high_temp_path_m = os.path.join(path_to_multi_scores, dir_m, model_name, task_name_json)
                    # print("high_temp_path_m_score: ", high_temp_path_m)
                    high_temp_scores_m = load_error_data(high_temp_path_m)
                    high_temp_error_data[dir_m] = high_temp_scores_m
            if "decoded_ast" in task_name_json:
                name = task_name_json.split("_decoded_ast")[0].split("v3_")[-1]
                # print("name: ", name)
                if name == task_name: 
                    high_temp_path_m = os.path.join(path_to_multi_scores, dir_m, model_name, task_name_json)
                    # print("high_temp_path_m_decoded_ast: ", high_temp_path_m)
                    high_temp_ast_m = load_ast_data(high_temp_path_m)
                    high_temp_ast_data[dir_m] = high_temp_ast_m
                    
    ################################################

    # Optionally return info about skipped examples, by type
    skipped_info = dict()

    res = []
    for id in greedy_ids:
        row = {"id": id, "greedy_output": [], "high_temp_outputs": []}

        # Extract the specified example by id 
        greedy_logprob = greedy_logprobs_dict[id]
        # print("greedy_logprob: ", greedy_logprob)
        greedy_ast = greedy_ast_dict[id]

        ptrue_val = ptrue_dict[id] if ptrue_dict is not None else None
        # ######### check if relevant keys e.g. "output_token_count" are in the logprob
        
        # if "output_token_count" not in greedy_logprob or "logprobs" not in greedy_logprob or "output_token_count" not in greedy_logprob:
        #     print("output_token_count/logprobs/output_token_count not in greedy_logprob: ", greedy_logprob)
        #     continue
        # #########


        # Decide accuracy using the greedy outputs
        # Meanwhile, check if we need to skip any example (along with their high-temp generations), based on the error type of the greedy output only.
        if id in greedy_error_dict: 
            # error
            # print("id: ", id)
            # print("in greedy_error_dict: ", greedy_error_dict)
            greedy_error_type = greedy_error_dict[id]

            if skip_any_incorrect_simple and task_name in ["simple"]:
                # Skip any incorrect examples in simple task for confabulation setup
                print(f"Skipping {id} for task {task_name} for confabulation setup.")
                skipped_info["skip_any_incorrect_simple"] = skipped_info.get("skip_any_incorrect_simple", 0) + 1
                continue
            # TODO: for non-irrelevance tasks, check if error type is ast_decoder:decoder_failed
            # if so, skip them and print the id 
            
            if skip_non_irrelevance_decoder_failure: 
                if task_name not in ["irrelevance"]:
                    # If bfcl eval evaluates to decoder failure, it's most probably a decoding failure already, so skip this example
                    if greedy_error_type == "ast_decoder:decoder_failed":
                        print(f"Skipping {id} for task {task_name} as it is ast_decoder:decoder_failed")
                        skipped_info["skip_non_irrelevance_decoder_failure"] = skipped_info.get("skip_non_irrelevance_decoder_failure", 0) + 1
                        continue


                    # If bfcl checker doesn't catch the error, we check manually if the greedy ast is a list of dicts 
                    # If the decoded ast is empty, then certainty a decoding failure
                    if len(greedy_ast) == 0:
                        print(f"Skipping {id} for task {task_name} as it is an empty list")
                        skipped_info["skip_non_irrelevance_decoder_failure"] = skipped_info.get("skip_non_irrelevance_decoder_failure", 0) + 1
                        continue
                        
                    # If any of the asts are not a dict, then it's a decoding failure
                    for ast in greedy_ast:
                        if type(ast) != dict:
                            print(f"Skipping {id} for task {task_name} as it is not a list of dicts")
                            skipped_info["skip_non_irrelevance_decoder_failure"] = skipped_info.get("skip_non_irrelevance_decoder_failure", 0) + 1
                            continue




                if task_name == "irrelevance":
                    assert greedy_error_type == "irrelevance_error:decoder_success", f"Error type is not 'irrelevance_error:decoder_success' for task {task_name}"
                  
            # if irrelevance, assert "irrelevance_error:decoder_success" as its error_type
            # as the model incorrectly produces a decodable output
            greedy_accuracy = 0

        else:

            # TODO: for irrelevance task, we need to skip this example as this greedy output is error-free - meaning the
            # model hasn't produced anything decodable 
            # correct

            if skip_irrelevance_success and task_name == "irrelevance":
                print(f"Skipping {id} for task {task_name} as it correctly refused.")
                skipped_info["skip_irrelevance_success"] = skipped_info.get("skip_irrelevance_success", 0) + 1
                continue
            

            greedy_error_type = None
            greedy_accuracy = 1

     
        
        greedy_item = UQEntity(text=greedy_logprob["result"], \
                                token_count=greedy_logprob["output_token_count"], \
                                token_logprobs=greedy_logprob["logprobs"]["token_logprobs"], \
                                top_logprobs=greedy_logprob["logprobs"]["top_logprobs"], \
                                accuracy=greedy_accuracy, \
                                error_type=greedy_error_type, \
                                ast_decoded=greedy_ast, \
                                p_true_log_prob=ptrue_val)
        
        greedy_item = asdict(greedy_item)
        row["greedy_output"] = greedy_item

        # Repeat for M high_temp outputs
        for m in high_temp_logprobs_data.keys():
            
            # Match logprobs and errors by m 
            # high_temp_errors_m is a dict containing all the errors during the m-th high temp generation 
            high_temp_errors_m = high_temp_error_data[m]
            

            # Decide accuracy 
            if id in high_temp_errors_m: 
                # error
                high_temp_error_type = high_temp_errors_m[id]
                high_temp_accuracy = 0
            else:
                # correct
                high_temp_error_type = None
                high_temp_accuracy = 1

            high_temp_logprob = high_temp_logprobs_data[m][id]

            # # TODO: right now this is hard coded. Fix inference issue, "Error during inference: list index out of range" or argument errors.
            # ######### check if "output_token_count" is in the logprob
            if "output_token_count" not in high_temp_logprob or "logprobs" not in high_temp_logprob or "result" not in high_temp_logprob:
                print(f"output_token_count/logprobs/result not in high_temp_logprob: m={m}, id={id}")
                continue
            #########

            # Get the high temp ast at this ID
            if id not in high_temp_ast_data[m]:
                print("high_temp_path_m: ", high_temp_path_m)
                print("m: ", m)
                print("id: ", id)
                breakpoint()
            high_temp_ast_m = high_temp_ast_data[m][id]
          

            high_temp_item = UQEntity(text=high_temp_logprob["result"], \
                                    token_count=high_temp_logprob["output_token_count"], \
                                    token_logprobs=high_temp_logprob["logprobs"]["token_logprobs"], \
                                    top_logprobs=greedy_logprob["logprobs"]["top_logprobs"], \
                                    accuracy=high_temp_accuracy, \
                                    error_type=high_temp_error_type, \
                                    ast_decoded=high_temp_ast_m)

            high_temp_item = asdict(high_temp_item)
            row["high_temp_outputs"].append(high_temp_item)

        # # TODO: right now this is hard coded. Fix inference issue, "Error during inference: list index out of range"
        # if len(row["high_temp_outputs"]) == 0:
        #     continue
        # #########

        # Add meta data like question and label
        row["question"] = questions[id]["question"]
        row["function"] = questions[id]["function"]
        row["uq_scores"] = dict()
        res.append(row)



    # return info about skipped examples, by type
    skipped_info["total"] = len(greedy_ids)
    return res, skipped_info





if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
 
    parser.add_argument("--greedy_logprobs_dir", type=str, help="Path to the generated greedy result after running bfcl generate.")
    parser.add_argument("--greedy_scores_dir", type=str, help="Path to the generated greedy result after running evaluate.")
    parser.add_argument("--multi_logprobs_dir", type=str, help="Path to the generated multiple high temperature result after running bfcl generate.")
    parser.add_argument("--multi_scores_dir", type=str, help="Path to the generated multiple high temperature scores after running evaluate.")
    parser.add_argument("--ptrue_dir", type=str, help="Path to the processed ptrue logprob path.", default=None)


    parser.add_argument("--model_name", type=str, help="Model to compute UQ for")
    parser.add_argument("--test_names", type=str, nargs="+", help="List of test names to combine.")
    
    parser.add_argument("--save_path", type=str, help="Path to save the combined json.")
    parser.add_argument("--skip_non_irrelevance_decoder_failure", action=argparse.BooleanOptionalAction, default=True, \
                        help="On non-irrelevance tasks, whether to skip decoder failure.")
    parser.add_argument("--skip_irrelevance_success", action=argparse.BooleanOptionalAction, default=True, \
                        help="On irrelevance tasks, turn on flag to skip successfully refused examples.")
    parser.add_argument("--skip_any_incorrect_simple", action=argparse.BooleanOptionalAction, default=False, \
                        help="On task simple (python only), whether to skip all examples that are incorrect, whichever error type.")
    
 
    args, unknown = parser.parse_known_args()
    print("Combination config:")
    print(f"skip_non_irrelevance_decoder_failure: {args.skip_non_irrelevance_decoder_failure}")
    print(f"skip_irrelevance_success: {args.skip_irrelevance_success}")
    print(f"skip_any_incorrect_simple: {args.skip_any_incorrect_simple}")


   
    # For each task in test_names, get a greedy_generation_path, greedy_score_path, multi_logprobs_path, multi_scores_path
    print(f"Combining {len(args.test_names)} tasks: {args.test_names}....")
    res = []
    for task in args.test_names:
        greedy_generation_path = os.path.join(args.greedy_logprobs_dir, f"BFCL_v3_{task}_result.json")
        greedy_score_path = os.path.join(args.greedy_scores_dir, f"BFCL_v3_{task}_score.json")
        greedy_ast_path = os.path.join(args.greedy_scores_dir, f"BFCL_v3_{task}_decoded_ast.json")
        
        ptrue_path = None
        if args.ptrue_dir is not None:
            ptrue_path = os.path.join(args.ptrue_dir, f"{task}.json")
        # print("model_name: ", args.model_name)
        # print("greedy_generation_path: ", greedy_generation_path)
        # print("greedy_score_path: ", greedy_score_path)
        # print("multi_logprobs_dir: ", args.multi_logprobs_dir)
        # print("multi_scores_dir: ", args.multi_scores_dir)
        res_per_task, skipped_info = combine_outputs_per_task(task_name=task, model_name=args.model_name, \
                                                greedy_generation_path=greedy_generation_path, greedy_score_path=greedy_score_path, \
                                                greedy_ast_path=greedy_ast_path, \
                                                ptrue_path=ptrue_path, \
                                                path_to_multi_generations=args.multi_logprobs_dir, path_to_multi_scores=args.multi_scores_dir,\
                                                skip_non_irrelevance_decoder_failure=args.skip_non_irrelevance_decoder_failure, \
                                                skip_irrelevance_success=args.skip_irrelevance_success, \
                                                skip_any_incorrect_simple=args.skip_any_incorrect_simple) 
        res.extend(res_per_task)


        
        print(f"Task {task} done. Skipped info:")
        pprint(skipped_info)


    save_path = f"{args.save_path}/{args.model_name}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    save_suffix = "-".join(args.test_names)
    print(f"Saving combined file {save_suffix}")
    with open(f'{save_path}/combined_{save_suffix}.json', 'w') as f:
        json.dump(res, f)

 