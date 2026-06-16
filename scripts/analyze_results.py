#!/usr/bin/env python3
"""Analyze ASR evaluation results from JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.result import BenchmarkResult  # noqa: E402


def print_summary(result: BenchmarkResult) -> None:
    """Print a minimal benchmark summary (works for v1 and v2 results)."""
    print(f"Model:    {result.model}")
    print(f"Dataset:  {result.dataset}")
    print(f"Language: {result.language}")
    print(f"Time:     {result.timestamp}")
    print(f"Samples:  {result.num_samples} (skipped {result.num_skipped})")

    for ref, metrics in result.results.items():
        marker = " (primary)" if ref == result.primary_reference else ""
        wer = metrics.get("wer")
        cer = metrics.get("cer")
        wer_s = f"{wer:.2%}" if wer is not None else "n/a"
        cer_s = f"{cer:.2%}" if cer is not None else "n/a"
        print(f"{ref}{marker} WER/CER: {wer_s} / {cer_s}")

    if result.speed.get("rtfx"):
        print(f"RTFx:     {result.speed['rtfx']:.1f} (indicative)")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze ASR evaluation results from JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "result_file",
        type=Path,
        help="Path to the evaluation results JSON file",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if not args.result_file.exists():
        print(f"Error: File not found: {args.result_file}", file=sys.stderr)
        return 1

    try:
        with open(args.result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        return 1

    print_summary(BenchmarkResult.from_dict(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
