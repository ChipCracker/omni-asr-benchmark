#!/usr/bin/env python3
"""Evaluate OmniASR model on the BAS RVG1 dataset."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.datasets import BasRvg1Source
from src.evaluation import OmniASREvaluator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate OmniASR model on BAS RVG1 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to BAS RVG1 data directory (overrides BAS_RVG1_DATA_DIR env var)",
    )
    parser.add_argument(
        "--model-card",
        type=str,
        default="omniASR_LLM_7B_v2",
        help="Model card name for the ASR model",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="deu_Latn",
        help="Language code for transcription",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=40.0,
        help="Maximum audio duration in seconds (skip longer files)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="c",
        choices=["c", "h", "l"],
        help="Audio channel to use (c=close, h=headset, l=laryngograph)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation.json"),
        help="Output file path for results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load environment variables
    load_dotenv()

    # Determine data directory
    data_dir = args.data_dir
    if data_dir is None:
        data_dir_env = os.getenv("BAS_RVG1_DATA_DIR")
        if data_dir_env:
            data_dir = Path(data_dir_env)
        else:
            logger.error(
                "No data directory specified. Set BAS_RVG1_DATA_DIR environment "
                "variable or use --data-dir argument."
            )
            return 1

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return 1

    logger.info(f"Using data directory: {data_dir}")
    logger.info(f"Model: {args.model_card}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Channel: {args.channel}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")
    logger.info(f"Max duration: {args.max_duration}s")

    # Initialize dataset source
    dataset = BasRvg1Source(
        data_dir=data_dir,
        channel=args.channel,
    )

    # Initialize evaluator
    evaluator = OmniASREvaluator(
        model_card=args.model_card,
        language=args.language,
        batch_size=args.batch_size,
        max_audio_duration=args.max_duration,
    )

    # Run evaluation
    try:
        result = evaluator.evaluate(
            dataset_source=dataset,
            max_samples=args.max_samples,
        )
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1

    # Save results
    result.save(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model:           {result.model}")
    print(f"Dataset:         {result.dataset}")
    print(f"Language:        {result.language}")
    print(f"Samples:         {result.num_samples}")
    print(f"Skipped:         {result.num_skipped}")
    print("-" * 60)

    dialect_res = result.results.get("dialect_reference", {})
    print("Dialect Reference:")
    print(f"  WER:           {dialect_res.get('wer', 0):.2%}")
    print(f"  CER:           {dialect_res.get('cer', 0):.2%}")
    print(f"  Substitutions: {dialect_res.get('substitutions', 0)}")
    print(f"  Deletions:     {dialect_res.get('deletions', 0)}")
    print(f"  Insertions:    {dialect_res.get('insertions', 0)}")

    ort_res = result.results.get("ort_reference", {})
    if ort_res.get("wer") is not None:
        print("-" * 60)
        print("Standard Orthography Reference:")
        print(f"  WER:           {ort_res.get('wer', 0):.2%}")
        print(f"  CER:           {ort_res.get('cer', 0):.2%}")
        print(f"  Substitutions: {ort_res.get('substitutions', 0)}")
        print(f"  Deletions:     {ort_res.get('deletions', 0)}")
        print(f"  Insertions:    {ort_res.get('insertions', 0)}")

    print("=" * 60)
    print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
