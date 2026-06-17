#!/usr/bin/env python3
"""Generic ASR benchmark: run one model on one dataset.

Any model from the registry (:func:`src.models.get_model`) is benchmarked on any
dataset from the registry (:func:`src.datasets.get_dataset`). The result is
written as a v2 result JSON and, unless ``--no-leaderboard`` is given, the
color-coded leaderboard over all results in the output directory is printed.

Examples:
    python scripts/benchmark.py --model openai/whisper-large-v3 --dataset bas_rvg1
    python scripts/benchmark.py --model nvidia/parakeet-tdt-0.6b-v3 --dataset bas_rvg1 \\
        --data-dir /data/bas_rvg1 --batch-size 8 --max-samples 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

from src.datasets import available_datasets, get_dataset  # noqa: E402
from src.models import get_model  # noqa: E402


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one ASR model on one dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model identifier / card.")
    parser.add_argument(
        "--dataset",
        default="bas_rvg1",
        help=f"Dataset name. Available: {', '.join(available_datasets())}.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the dataset directory (otherwise the dataset class "
        "default / BAS_RVG1_DATA_DIR is used).",
    )
    parser.add_argument("--language", default="deu_Latn", help="Language code.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--channel",
        default="c",
        choices=["c", "h", "l"],
        help="BAS RVG1 audio channel (c=close, h=headset, l=laryngograph).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSONL manifest path (for ksof / generic manifest datasets; "
        "ksof has a default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: results/<model>__<dataset>.json).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local model directory (required for syllabic-asr models).",
    )
    parser.add_argument("--no-leaderboard", action="store_true", help="Skip leaderboard print.")
    parser.add_argument("--no-speed", action="store_true", help="Do not measure RTFx.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _build_dataset_params(args: argparse.Namespace) -> dict:
    """Dataset constructor params. Extend here when adding datasets.

    ``data_dir`` is only passed when explicitly given; otherwise the dataset
    class resolves its own default (and the ``BAS_RVG1_DATA_DIR`` env var).
    """
    params: dict = {}
    ds_key = args.dataset.lower().replace("-", "_")
    if args.data_dir is not None:
        params["data_dir"] = args.data_dir
    if ds_key in ("bas_rvg1", "basrvg1source"):
        params["channel"] = args.channel
    if args.manifest is not None:
        params["manifest_path"] = args.manifest
    return params


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    load_dotenv()

    model_safe = args.model.replace("/", "_").replace("\\", "_")
    output_path = args.output or Path(f"results/{model_safe}__{args.dataset}.json")

    logger.info("Model:   %s", args.model)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Output:  %s", output_path)

    try:
        dataset = get_dataset(args.dataset, **_build_dataset_params(args))
    except KeyError as e:
        logger.error(str(e))
        return 1

    model = get_model(
        model_name=args.model,
        language=args.language,
        batch_size=args.batch_size,
        model_dir=str(args.model_dir) if args.model_dir else None,
    )

    try:
        result = model.benchmark(
            dataset,
            max_samples=args.max_samples,
            split=args.split,
            measure_speed=not args.no_speed,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Benchmark failed: %s", e)
        return 1

    result.save(output_path)
    _print_summary(result)

    if not args.no_leaderboard:
        _print_leaderboard(output_path.parent)

    return 0


def _print_summary(result) -> None:
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model:   {result.model}")
    print(f"Dataset: {result.dataset}")
    print(f"Samples: {result.num_samples}")
    for ref, metrics in result.results.items():
        marker = " (primary)" if ref == result.primary_reference else ""
        wer = metrics.get("wer")
        cer = metrics.get("cer")
        wer_s = f"{wer:.2%}" if wer is not None else "n/a"
        cer_s = f"{cer:.2%}" if cer is not None else "n/a"
        print(f"  [{ref}{marker}] WER={wer_s}  CER={cer_s}")
    if result.speed.get("rtfx"):
        print(f"  RTFx: {result.speed['rtfx']:.1f} (indicative)")
    print("=" * 60)


def _print_leaderboard(results_dir: Path) -> None:
    from src.leaderboard import build_leaderboard, load_results, render_terminal

    results = load_results(results_dir)
    if results:
        print()
        render_terminal(build_leaderboard(results))


if __name__ == "__main__":
    sys.exit(main())
