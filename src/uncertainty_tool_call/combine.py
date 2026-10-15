"""Merge greedy and sampled generations into one combined file per model.

Run after `bfcl generate` and `bfcl evaluate` for the greedy pass and for each
of the M high-temperature samples. Joins them per request into
``<save_path>/<model_name>/combined_<tasks>.json``, the input of `utoolcall uq`.
Each record holds the request ("question", "function"), the greedy output and
the M samples, each with its text, token log-probabilities, decoded AST and
whether it is correct.

Usage (see "Building a combined file from scratch" in README.md):
    utoolcall combine \\
        --greedy_logprobs_dir result/<model> --greedy_scores_dir score/<model> \\
        --multi_logprobs_dir result/<multi> --multi_scores_dir score/<multi> \\
        --test_names simple --model_name <model> --save_path combined
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

from uncertainty_tool_call.bfcl.constant import PROMPT_PATH, TEST_FILE_MAPPING
from uncertainty_tool_call.bfcl.utils import load_file


@dataclass
class UQEntity:
    """One generation: the greedy output or one high-temperature sample."""

    text: str
    token_count: int
    token_logprobs: list[float]
    top_logprobs: list[dict]
    accuracy: int
    error_type: Union[str, None] = None
    ast_decoded: Union[list, None] = None
    # Always null: P(true) is not computed. Kept so that files built here have
    # the same fields as the released ones.
    p_true_log_prob: Union[float, None] = None


def read_jsonl(path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_generations(path) -> dict:
    """{id: generation} from a `bfcl generate` result file."""
    return {row["id"]: row for row in read_jsonl(path)}


def load_errors(path) -> dict:
    """{id: error_type} from a `bfcl evaluate` score file.

    The first line is the summary metric; each further line is one request
    answered wrongly, so requests not in the dict were answered correctly.
    """
    return {row["prompt"]["id"]: row["error_type"] for row in read_jsonl(path)[1:]}


def load_decoded_asts(path) -> dict:
    """{id: decoded calls} from a `bfcl evaluate` decoded_ast file (first line: summary)."""
    return {row["prompt"]["id"]: row["model_result_decoded"] for row in read_jsonl(path)[1:]}


def sample_dirs(path) -> list:
    """The m=1 ... m=M subdirectories of `path`, in numeric order."""
    dirs = os.listdir(path)
    assert all("m=" in d for d in dirs), f"expected only m=<n> directories in {path}"
    return sorted(dirs, key=lambda d: int(d.split("=")[1]))


def task_of(filename, suffix) -> str:
    """'BFCL_v3_simple_result.json', '_result' -> 'simple'."""
    return filename.split(suffix)[0].split("v3_")[-1]


def combine_outputs_per_task(task_name, model_name, greedy_generation_path, greedy_score_path, greedy_ast_path,
                             path_to_multi_generations, path_to_multi_scores,
                             skip_non_irrelevance_decoder_failure: bool = True,
                             skip_irrelevance_success: bool = True,
                             skip_any_incorrect_simple: bool = False):
    """Join the greedy output and the M samples of every request of one task.

    The greedy output decides the label (correct or not) and whether a request
    is kept. Requests are dropped when:
      - skip_non_irrelevance_decoder_failure: the greedy output could not be
        decoded into a function call (all tasks but irrelevance);
      - skip_irrelevance_success: on irrelevance, the model correctly refused,
        so there is no call to score;
      - skip_any_incorrect_simple: on simple, the greedy output is wrong
        (off by default).

    Returns the records and a count of the dropped requests per reason.
    """
    greedy_rows = read_jsonl(greedy_generation_path)
    greedy_ids = [row["id"] for row in greedy_rows]
    greedy_generations = {row["id"]: row for row in greedy_rows}
    greedy_errors = load_errors(greedy_score_path)  # wrong answers only
    greedy_asts = load_decoded_asts(greedy_ast_path)

    # The question and the available functions of each request.
    dataset = load_file(PROMPT_PATH / TEST_FILE_MAPPING[task_name])
    questions = {row["id"]: {"question": row["question"], "function": row["function"]} for row in dataset}

    # The same per sample, keyed by sample directory: {"m=1": {id: ...}, ...}
    sample_generations, sample_errors, sample_asts = {}, {}, {}
    for m in sample_dirs(path_to_multi_generations):
        folder = os.path.join(path_to_multi_generations, m, model_name)
        for filename in os.listdir(folder):
            if task_of(filename, "_result") == task_name:
                sample_generations[m] = load_generations(os.path.join(folder, filename))
    for m in sample_dirs(path_to_multi_scores):
        folder = os.path.join(path_to_multi_scores, m, model_name)
        for filename in os.listdir(folder):
            if "score" in filename and task_of(filename, "_score") == task_name:
                sample_errors[m] = load_errors(os.path.join(folder, filename))
            if "decoded_ast" in filename and task_of(filename, "_decoded_ast") == task_name:
                sample_asts[m] = load_decoded_asts(os.path.join(folder, filename))

    skipped_info = dict()

    def skip(reason):
        skipped_info[reason] = skipped_info.get(reason, 0) + 1

    res = []
    for id in greedy_ids:
        greedy = greedy_generations[id]
        greedy_ast = greedy_asts[id]

        if id in greedy_errors:
            greedy_error_type = greedy_errors[id]

            if skip_any_incorrect_simple and task_name in ["simple"]:
                skip("skip_any_incorrect_simple")
                continue

            if skip_non_irrelevance_decoder_failure:
                if task_name not in ["irrelevance"]:
                    if greedy_error_type == "ast_decoder:decoder_failed":
                        skip("skip_non_irrelevance_decoder_failure")
                        continue
                    # Decoded to nothing: also a decoding failure.
                    if len(greedy_ast) == 0:
                        skip("skip_non_irrelevance_decoder_failure")
                        continue
                    for ast in greedy_ast:
                        if type(ast) != dict:
                            # NOTE: this `continue` only moves on to the next
                            # `ast`, so the request is counted as skipped but
                            # kept. Left as is: the released files were built so.
                            skip("skip_non_irrelevance_decoder_failure")
                            continue

                if task_name == "irrelevance":
                    # The model answered with a call where it should have refused.
                    assert greedy_error_type == "irrelevance_error:decoder_success", \
                        f"Error type is not 'irrelevance_error:decoder_success' for task {task_name}"

            greedy_accuracy = 0
        else:
            if skip_irrelevance_success and task_name == "irrelevance":
                skip("skip_irrelevance_success")
                continue
            greedy_error_type = None
            greedy_accuracy = 1

        greedy_output = asdict(UQEntity(
            text=greedy["result"],
            token_count=greedy["output_token_count"],
            token_logprobs=greedy["logprobs"]["token_logprobs"],
            top_logprobs=greedy["logprobs"]["top_logprobs"],
            accuracy=greedy_accuracy,
            error_type=greedy_error_type,
            ast_decoded=greedy_ast,
        ))
        row = {"id": id, "greedy_output": greedy_output, "high_temp_outputs": []}

        for m, generations in sample_generations.items():
            errors = sample_errors[m]
            sample = generations[id]
            # A sample whose inference failed has no output; leave it out.
            if "output_token_count" not in sample or "logprobs" not in sample or "result" not in sample:
                skip("samples_without_output")
                continue
            if id not in sample_asts[m]:
                raise KeyError(f"{id}: no decoded AST in sample {m}")

            row["high_temp_outputs"].append(asdict(UQEntity(
                text=sample["result"],
                token_count=sample["output_token_count"],
                token_logprobs=sample["logprobs"]["token_logprobs"],
                # NOTE: the greedy output's top_logprobs, not this sample's. No
                # estimator reads a sample's top_logprobs (SMT uses the greedy
                # output's only). Left as is: the released files were built so.
                top_logprobs=greedy["logprobs"]["top_logprobs"],
                accuracy=0 if id in errors else 1,
                error_type=errors.get(id),
                ast_decoded=sample_asts[m][id],
            )))

        row["question"] = questions[id]["question"]
        row["function"] = questions[id]["function"]
        res.append(row)

    skipped_info["total"] = len(greedy_ids)
    return res, skipped_info


def main(argv=None):
    parser = argparse.ArgumentParser(prog="utoolcall combine", description=__doc__.split("\n")[0])
    parser.add_argument("--greedy_logprobs_dir", required=True, help="greedy `bfcl generate` results, result/<model>")
    parser.add_argument("--greedy_scores_dir", required=True, help="greedy `bfcl evaluate` scores, score/<model>")
    parser.add_argument("--multi_logprobs_dir", required=True, help="results of the M samples, holding m=1 ... m=M")
    parser.add_argument("--multi_scores_dir", required=True, help="scores of the M samples, holding m=1 ... m=M")
    parser.add_argument("--model_name", required=True, help="model directory name, e.g. Qwen_Qwen2.5-0.5B-Instruct")
    parser.add_argument("--test_names", required=True, nargs="+", help="tasks to combine into one file")
    parser.add_argument("--save_path", required=True, help="writes <save_path>/<model_name>/combined_<tasks>.json")
    parser.add_argument("--skip_non_irrelevance_decoder_failure", action=argparse.BooleanOptionalAction, default=True,
                        help="on tasks other than irrelevance, drop requests whose greedy output fails to decode")
    parser.add_argument("--skip_irrelevance_success", action=argparse.BooleanOptionalAction, default=True,
                        help="on irrelevance, drop requests the model correctly refused")
    parser.add_argument("--skip_any_incorrect_simple", action=argparse.BooleanOptionalAction, default=False,
                        help="on simple, drop every request whose greedy output is wrong")
    args = parser.parse_args(argv)

    res = []
    for task in args.test_names:
        records, skipped_info = combine_outputs_per_task(
            task_name=task,
            model_name=args.model_name,
            greedy_generation_path=os.path.join(args.greedy_logprobs_dir, f"BFCL_v3_{task}_result.json"),
            greedy_score_path=os.path.join(args.greedy_scores_dir, f"BFCL_v3_{task}_score.json"),
            greedy_ast_path=os.path.join(args.greedy_scores_dir, f"BFCL_v3_{task}_decoded_ast.json"),
            path_to_multi_generations=args.multi_logprobs_dir,
            path_to_multi_scores=args.multi_scores_dir,
            skip_non_irrelevance_decoder_failure=args.skip_non_irrelevance_decoder_failure,
            skip_irrelevance_success=args.skip_irrelevance_success,
            skip_any_incorrect_simple=args.skip_any_incorrect_simple,
        )
        res.extend(records)
        print(f"{task}: {len(records)} of {skipped_info.pop('total')} requests kept; skipped: {skipped_info}")

    save_path = Path(args.save_path) / args.model_name
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"combined_{'-'.join(args.test_names)}.json"
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
