"""Build the combined-file archive published as a GitHub Release asset.

Strips the derived uncertainty scores from the project's `*_with_uq.json`
files, leaving only the generation fields that `utoolcall uq` consumes. The result
is what a default-tier user downloads.

    python scripts/build_combined_release.py \
        --source ~/workspace/uncertainty_bfcl/berkeley-function-call-leaderboard/combined-24-11 \
        --out dist/combined.tar.gz
"""

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path

# Only these per-record fields are inputs. Everything else in the source files
# is derived by `utoolcall uq` and is dropped so the artefact carries no answers.
KEEP = {"id", "question", "function", "greedy_output", "high_temp_outputs"}

MODELS = [
    "Qwen_Qwen2.5-0.5B-Instruct", "Qwen_Qwen2.5-3B-Instruct",
    "Qwen_Qwen2.5-7B-Instruct", "Qwen_Qwen3-4B-Instruct-2507",
    "google_gemma-2-9b-it", "google_gemma-3-12b-it", "google_gemma-3-4b-it",
    "mistralai_Ministral-8B-Instruct-2410",
]

TASKS = [
    "simple", "multiple", "parallel", "parallel_multiple",
    "simple-multiple", "simple-parallel", "multiple-parallel_multiple",
    "simple-multiple-parallel-parallel_multiple",
    "simple-irrelevance",
    "simple-multiple-parallel-parallel_multiple-irrelevance",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="directory of <model>/combined_<task>_with_uq.json")
    ap.add_argument("--out", default="dist/combined.tar.gz")
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "combined"
        raw = kept = 0
        for model in args.models:
            (staged / model).mkdir(parents=True, exist_ok=True)
            for task in TASKS:
                src = source / model / f"combined_{task}_with_uq.json"
                if not src.is_file():
                    raise SystemExit(f"missing: {src}")
                raw += src.stat().st_size
                data = json.load(src.open())
                recs = data if isinstance(data, list) else data.get("data", data)
                out_recs = [{k: v for k, v in r.items() if k in KEEP} for r in recs]
                dst = staged / model / f"combined_{task}.json"
                json.dump(out_recs, dst.open("w"))
                kept += dst.stat().st_size
            print(f"  {model}: {len(TASKS)} tasks")

        print(f"\nstaged {kept / 2**30:.2f} GB (from {raw / 2**30:.2f} GB of source)")
        print(f"writing {out}")
        with tarfile.open(out, "w:gz", compresslevel=6) as tar:
            tar.add(staged, arcname=".")

    print(f"done: {out} ({out.stat().st_size / 2**20:.0f} MB)")
    print("\nUpload with:")
    print(f"  gh release create combined-v1 {out} --notes 'Combined generation files'")


if __name__ == "__main__":
    main()
