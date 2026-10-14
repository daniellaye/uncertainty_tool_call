"""Build Table 1 of the paper: mean AUROC +/- mean standard error.

Reads the per-model score CSVs written by `uq.py` and averages them over the
eight LLMs. Each CSV has one row per uncertainty estimator:

    UQ methods,AUROC,AUPRC,1-NAURC (greedy),1-NAURC (avg),standard_error

so a cell of Table 1 is just the mean of column `AUROC` and the mean of column
`standard_error` across the eight models.

Usage:
    python -m uncertainty_tool_call.table1                 # print
    python -m uncertainty_tool_call.table1 -o table1.tex   # write
"""

import argparse
import csv
import os
from collections import defaultdict

# --------------------------------------------------------------------------
# Configuration: what goes into the table
# --------------------------------------------------------------------------

# The eight LLMs averaged over. The published run also scored
# google_gemma-3-1b-it, which is not part of Table 1.
MODELS = [
    "Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen_Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct",
    "Qwen_Qwen3-4B-Instruct-2507",
    "google_gemma-2-9b-it",
    "google_gemma-3-12b-it",
    "google_gemma-3-4b-it",
    "mistralai_Ministral-8B-Instruct-2410",
]

# Rows: (row group, label in the table, task name in the score filenames)
TEST_CATEGORIES = [
    ("Individual", "Simple", "simple"),
    ("Individual", "Multiple", "multiple"),
    ("Individual", "Parallel", "parallel"),
    ("Individual", "Parallel-Multiple", "parallel_multiple"),
    ("Combinations", "Simple + Multiple", "simple-multiple"),
    ("Combinations", "Simple + Parallel", "simple-parallel"),
    ("Combinations", "Multiple + Parallel-Multiple", "multiple-parallel_multiple"),
    ("Combinations", "All Combined", "simple-multiple-parallel-parallel_multiple"),
    ("Unanswerable", "Simple + Irrelevance", "simple-irrelevance"),
    (
        "Unanswerable",
        "All Combined + Irrelevance",
        "simple-multiple-parallel-parallel_multiple-irrelevance",
    ),
]

# Columns: (column group, header in the table, name in the score CSVs)
#
# P(true) appeared in the published Table 1 but is dropped here: unlike every
# other estimator it cannot be derived from a combined file, needing a separate
# generation pass that asks the model whether its own call is correct.
UQ_METHODS = [
    ("Single-sample", r"\textbf{MAX}", "min"),
    ("Single-sample", r"\textbf{AVG}", "mean"),
    ("Single-sample", r"\textbf{G-NLL}", "sum"),
    ("Multi-sample", r"\textbf{SE}$_{EXM}$", "logprobs SE"),
    ("Multi-sample", r"\textbf{DSE}$_{EXM}$", "discrete SE"),
    ("Multi-sample", r"\textbf{PE}", "Predictive E"),
    ("Other", r"\textbf{LEN}", "length"),
]


def default_scores_dir():
    """Where `utoolcall uq` writes its score CSVs: ./uq, or $UTOOLCALL_SCORES_DIR.

    No scores are shipped with this repository. They are produced by running
    `utoolcall uq` over the downloaded combined files; see README.md.
    """
    if os.environ.get("UTOOLCALL_SCORES_DIR"):
        return os.path.expanduser(os.environ["UTOOLCALL_SCORES_DIR"])
    return os.path.join(os.getcwd(), "uq")


def main():
    parser = argparse.ArgumentParser(description="Build Table 1 of the paper.")
    parser.add_argument("--scores_dir", default=None, help="directory of per-model score CSVs")
    parser.add_argument("-o", "--output", default=None, help="write LaTeX here instead of stdout")
    args = parser.parse_args()

    scores_dir = args.scores_dir or default_scores_dir()
    if not os.path.isdir(scores_dir):
        raise SystemExit(
            f"no scores found at {scores_dir}.\n"
            "\n"
            "Scores are not shipped with this repository; compute them first:\n"
            "    utoolcall fetch-combined                 # download the generations\n"
            "    sbatch slurm/uq_all.sh               # compute the estimators\n"
            "\n"
            "Then point this at the output:\n"
            "    utoolcall table1 --scores_dir <dir>      (or set UTOOLCALL_SCORES_DIR)"
        )

    # ----------------------------------------------------------------------
    # Collect (AUROC, standard_error) per task per method, across models
    # ----------------------------------------------------------------------
    avg_aurocs = defaultdict(lambda: defaultdict(list))

    for model in MODELS:
        for _, _, test_category in TEST_CATEGORIES:
            path = os.path.join(scores_dir, model, f"combined_uq_{test_category}.csv")
            if not os.path.isfile(path):
                raise SystemExit(f"missing score file: {path}")
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)  # header
                for d in reader:
                    # d = [UQ method, AUROC, AUPRC, 1-NAURC greedy, 1-NAURC avg, SE]
                    avg_aurocs[test_category][d[0]].append((float(d[1]), float(d[-1])))

    # ----------------------------------------------------------------------
    # Average over models
    # ----------------------------------------------------------------------
    table_rows = []
    for group, label, test_category in TEST_CATEGORIES:
        row_scores = {}
        for _, _, uq in UQ_METHODS:
            if uq not in avg_aurocs[test_category]:
                raise SystemExit(f"method {uq!r} missing for task {test_category!r}")
            scores, ses = zip(*avg_aurocs[test_category][uq])
            row_scores[uq] = (sum(scores) / len(scores), sum(ses) / len(ses))
        table_rows.append((group, label, row_scores))

    # ----------------------------------------------------------------------
    # Emit LaTeX
    # ----------------------------------------------------------------------
    groups = []
    for group, _, _ in UQ_METHODS:
        if group not in groups:
            groups.append(group)

    lines = []
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{\begin{tabular}{ll" + "c" * len(UQ_METHODS) + "}")
    lines.append(r"\toprule")

    # Group header: two empty cells for the Category and Task columns.
    header = " & "
    for group in groups:
        span = sum(1 for g, _, _ in UQ_METHODS if g == group)
        header += rf" & \multicolumn{{{span}}}{{c}}{{\textbf{{{group}}}}}"
    lines.append(header + r"  \\")

    rules, col = [], 3
    for group in groups:
        span = sum(1 for g, _, _ in UQ_METHODS if g == group)
        rules.append(rf"\cmidrule(lr){{{col}-{col + span - 1}}}")
        col += span
    lines.append(" ".join(rules))

    headers = " & ".join(header_text for _, header_text, _ in UQ_METHODS)
    lines.append(rf"\textbf{{Category}} & \textbf{{Task}} & {headers} \\")
    lines.append(r"\hline")

    row_groups = []
    for group, _, _ in TEST_CATEGORIES:
        if group not in row_groups:
            row_groups.append(group)

    for row_group in row_groups:
        rows = [r for r in table_rows if r[0] == row_group]
        lines.append(rf"\multirow{{{len(rows)}}}{{*}}{{\textbf{{{row_group}}}}}")

        for _, label, row_scores in rows:
            # Best gets \best{}, runner-up gets \underline{}, at 2 decimal places.
            values = [round(row_scores[uq][0], 2) for _, _, uq in UQ_METHODS]
            ranked = sorted(set(values), reverse=True)
            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else None

            cells = []
            for _, _, uq in UQ_METHODS:
                score, se = row_scores[uq]
                point = f"{score:.2f}"
                if round(score, 2) == best:
                    point = rf"\best{{${point}$}}"
                elif second is not None and round(score, 2) == second:
                    point = rf"\underline{{${point}$}}"
                else:
                    point = f"${point}$"
                # Standard errors print without the leading zero, as in the paper.
                cells.append(rf"{point} \tiny{{±${se:.3f}$}}".replace("±$0.", "±$."))

            lines.append(rf"& \textbf{{{label}}} & " + " & ".join(cells) + r" \\")

        lines.append(r"\hline" if row_group != row_groups[-1] else r"\bottomrule")

    lines.append(r"\end{tabular}}")
    latex = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(latex)
        print(f"Wrote {args.output} (over {len(MODELS)} models)")
    else:
        print(latex, end="")


if __name__ == "__main__":
    main()
