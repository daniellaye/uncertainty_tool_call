"""The `utoolcall` command line interface.

    utoolcall fetch-combined     download the combined generation files (~100 MB)
    utoolcall uq                 combined files -> per-model AUROC score CSVs
    utoolcall table1             score CSVs -> Table 1

    utoolcall prepare            demo: build one combined file from scratch
                             (generate -> evaluate -> combine; needs a GPU)

Run `utoolcall <command> --help` for each command's options. `uq` and `combine`
forward unrecognised arguments to the underlying scripts, so every original
flag still works.
"""

import argparse
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

RELEASE_TAG = "combined-v1"
ASSET_NAME = "combined.tar.gz"
REPO = os.environ.get("UTOOLCALL_REPO", "daniellaye/uncertainty_tool_call")
ASSET_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{ASSET_NAME}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_module(module: str, argv: list) -> int:
    """Execute a module as __main__ with the given argv."""
    saved = sys.argv
    sys.argv = [module.rsplit(".", 1)[-1] + ".py", *argv]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    except SystemExit as exc:
        # SystemExit carries either an exit status or a message to print.
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = saved


# --------------------------------------------------------------------------


def cmd_fetch_combined(args, extra):
    """Download and unpack the generation files into data/combined/."""
    dest = Path(args.dest or repo_root() / "data" / "combined").resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()) and not args.force:
        print(f"{dest} is not empty; pass --force to re-download.")
        return 0

    archive = dest / ASSET_NAME
    print(f"downloading {ASSET_URL}")
    try:
        with urllib.request.urlopen(ASSET_URL) as response, archive.open("wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            read = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                read += len(chunk)
                if total:
                    print(f"\r  {read / 2**20:,.0f} / {total / 2**20:,.0f} MB", end="")
            print()
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"download failed ({exc.code}). The release asset may not be published yet.\n"
            f"  expected: {ASSET_URL}\n"
            "Override the repository with UTOOLCALL_REPO=owner/name."
        )

    print(f"unpacking into {dest}")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink()

    models = sorted(p.name for p in dest.iterdir() if p.is_dir())
    print(f"done: {len(models)} models under {dest}")
    return 0


def cmd_uq(args, extra):
    """Compute the uncertainty estimators for one model and one task."""
    if extra or args.combined_path:
        argv = list(extra)
        if args.combined_path:
            argv += ["--combined_path", args.combined_path]
        if args.test_name:
            argv += ["--test_name", args.test_name]
        if args.save_path:
            # uq.py writes into --save_path but does not create it.
            Path(args.save_path).mkdir(parents=True, exist_ok=True)
            argv += ["--save_path", args.save_path]
        argv += ["--plot_rc_figures", str(args.plot_rc_figures).lower()]
        return _run_module("uncertainty_tool_call.uq", argv)
    raise SystemExit("utoolcall uq needs --combined_path and --test_name; see --help")


def cmd_combine(args, extra):
    """Merge greedy and sampled generations into combined records."""
    return _run_module("uncertainty_tool_call.combine", list(extra))


def cmd_table1(args, extra):
    argv = list(extra)
    if args.output:
        argv += ["-o", args.output]
    if args.scores_dir:
        argv += ["--scores_dir", args.scores_dir]
    return _run_module("uncertainty_tool_call.table1", argv)


def cmd_prepare(args, extra):
    """Demo: build one combined file from scratch for a single model and task."""
    from uncertainty_tool_call.bfcl.constant import ensure_output_dirs

    ensure_output_dirs()

    out = Path(os.environ.get("UTOOLCALL_OUTPUT_DIR", Path.cwd())).resolve()
    model_dir = args.model.replace("/", "_")
    multi = f"result/m={args.samples}_temperature={args.temperature}_model={model_dir}"
    multi_score = f"score/m={args.samples}_temperature={args.temperature}_model={model_dir}"

    def bfcl(*argv):
        print(f"\n$ python -m uncertainty_tool_call.bfcl {' '.join(argv)}")
        code = _run_module("uncertainty_tool_call.bfcl.__main__", list(argv))
        if code:
            raise SystemExit(f"step failed: {' '.join(argv)}")

    # 1. greedy decode -- the prediction whose correctness is scored
    bfcl("generate", "--model", args.model, "--test-category", args.task, "--temperature", "0")

    # 2. M samples at high temperature -- these estimate its uncertainty.
    #    One invocation per sample: the harness has no "draw N samples" flag.
    for m in range(1, args.samples + 1):
        bfcl("generate", "--model", args.model, "--test-category", args.task,
             "--temperature", str(args.temperature), "--result-dir", f"{multi}/m={m}")

    # 3. score both passes. Must precede combine, which attaches these labels.
    bfcl("evaluate", "--model", args.model, "--test-category", args.task)
    for m in range(1, args.samples + 1):
        bfcl("evaluate", "--model", args.model, "--test-category", args.task,
             "--result-dir", f"{multi}/m={m}", "--score-dir", f"{multi_score}/m={m}")

    # 4. combine into the format `utoolcall uq` consumes
    save = out / "combined" / model_dir
    print(f"\n$ utoolcall combine -> {save}")
    code = _run_module("uncertainty_tool_call.combine", [
        "--greedy_logprobs_dir", str(out / "result" / model_dir),
        "--greedy_scores_dir", str(out / "score" / model_dir),
        "--multi_logprobs_dir", str(out / multi),
        "--multi_scores_dir", str(out / multi_score),
        "--test_names", args.task,
        "--model_name", model_dir,
        "--save_path", str(out / "combined"),   # combine.py appends <model_name>
    ])
    if code:
        raise SystemExit("combine failed")

    print(f"\nCombined file written under {save}")
    print(f"Next:  utoolcall uq --combined_path {save} --test_name {args.task} --save_path uq/{model_dir}")
    return 0


# --------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(prog="utoolcall", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch-combined", help="download the combined generation files")
    f.add_argument("--dest", default=None, help="where to unpack (default: data/combined)")
    f.add_argument("--force", action="store_true", help="re-download into a non-empty directory")
    f.set_defaults(func=cmd_fetch_combined)

    u = sub.add_parser("uq", help="combined files -> AUROC score CSVs")
    u.add_argument("--combined_path", default=None, help="directory holding combined_<task>.json")
    u.add_argument("--test_name", default=None, help="task name, e.g. simple")
    u.add_argument("--save_path", default=None, help="where to write the scores")
    u.add_argument("--plot_rc_figures", default=False, help="also save risk-coverage plots")
    u.set_defaults(func=cmd_uq)

    # add_help=False so --help reaches combine.py, which defines the real flags.
    c = sub.add_parser("combine", help="merge greedy + sampled generations", add_help=False)
    c.set_defaults(func=cmd_combine)

    t = sub.add_parser("table1", help="score CSVs -> Table 1")
    t.add_argument("-o", "--output", default=None, help="write LaTeX here")
    t.add_argument("--scores_dir", default=None, help="directory of per-model score CSVs")
    t.set_defaults(func=cmd_table1)

    d = sub.add_parser("prepare", help="demo: build one combined file from scratch (needs a GPU)")
    d.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    d.add_argument("--task", default="simple", help="BFCL test category")
    d.add_argument("--samples", type=int, default=10, help="M, high-temperature samples")
    d.add_argument("--temperature", default="1.5", help="T, sampling temperature")
    d.set_defaults(func=cmd_prepare)

    return p


def main():
    parser = build_parser()
    args, extra = parser.parse_known_args()
    sys.exit(args.func(args, extra))


if __name__ == "__main__":
    main()
