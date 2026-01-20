#!/usr/bin/env python3
"""Analyze ASR evaluation results from JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def load_results(file_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary(data: Dict[str, Any]) -> None:
    """Print a minimal evaluation summary."""
    print(f"Model:    {data.get('model', 'N/A')}")
    print(f"Dataset:  {data.get('dataset', 'N/A')}")
    print(f"Language: {data.get('language', 'N/A')}")
    print(f"Time:     {data.get('timestamp', 'N/A')}")
    print(f"Samples:  {data.get('num_samples', 0)} (skipped {data.get('num_skipped', 0)})")

    results = data.get("results", {})
    dialect_res = results.get("dialect_reference", {})
    if dialect_res:
        print(f"Dialect WER/CER: {dialect_res.get('wer', 0):.2%} / {dialect_res.get('cer', 0):.2%}")

    ort_res = results.get("ort_reference", {})
    if ort_res and ort_res.get("wer") is not None:
        print(f"ORT WER/CER:     {ort_res.get('wer', 0):.2%} / {ort_res.get('cer', 0):.2%}")


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
        data = load_results(args.result_file)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        return 1

    print_summary(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
