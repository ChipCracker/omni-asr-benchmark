#!/usr/bin/env python3
"""Deprecated thin wrapper around ``scripts/benchmark.py`` for BAS RVG1.

Kept for backward compatibility with the old ``--model-card`` interface. New
code should call the generic CLI directly:

    python scripts/benchmark.py --model <model> --dataset bas_rvg1 [...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``import benchmark`` (scripts/benchmark.py) resolvable.
sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="(Deprecated) Evaluate an ASR model on BAS RVG1. "
        "Use scripts/benchmark.py --dataset bas_rvg1 instead.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-card", type=str, default="omniASR_LLM_Unlimited_7B_v2")
    parser.add_argument("--language", type=str, default="deu_Latn")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--channel", type=str, default="c", choices=["c", "h", "l"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    print(
        "[deprecated] scripts/evaluate_rvg1.py forwards to scripts/benchmark.py "
        "--dataset bas_rvg1",
        file=sys.stderr,
    )
    args = parse_args()
    argv = [
        "--model", args.model_card,
        "--dataset", "bas_rvg1",
        "--language", args.language,
        "--batch-size", str(args.batch_size),
        "--channel", args.channel,
    ]
    if args.data_dir:
        argv += ["--data-dir", str(args.data_dir)]
    if args.max_samples is not None:
        argv += ["--max-samples", str(args.max_samples)]
    if args.output:
        argv += ["--output", str(args.output)]
    if args.model_dir:
        argv += ["--model-dir", str(args.model_dir)]
    if args.verbose:
        argv += ["--verbose"]

    import benchmark  # scripts/benchmark.py

    sys.argv = ["benchmark.py"] + argv
    return benchmark.main()


if __name__ == "__main__":
    sys.exit(main())
