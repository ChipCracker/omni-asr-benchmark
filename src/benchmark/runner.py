"""Generic benchmark engine.

``run_benchmark(model, dataset, ...)`` is the decoupled successor of the old
``BaseEvaluator.evaluate()``. It iterates a dataset, transcribes in batches,
scores the hypothesis against *every named reference* the dataset exposes, and
returns a :class:`~src.benchmark.result.BenchmarkResult`. It also measures a
(wall-clock, indicative) real-time factor (RTFx).

The model only ever sees :meth:`AsrModel.transcribe_batch` — the exact same
contract the 18 legacy evaluators already implement — so every existing model
runs here unchanged.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import List, Optional, TYPE_CHECKING

from .metrics import compute_asr_metrics, compute_single_sample_metrics
from .result import BenchmarkResult, SampleResult

if TYPE_CHECKING:
    from ..datasets.base import DatasetSource, Sample
    from ..models.base import AsrModel

logger = logging.getLogger(__name__)


def _collect_reference_names(samples: List["Sample"]) -> List[str]:
    """Ordered union of reference names across all samples (first-seen order)."""
    names: List[str] = []
    for sample in samples:
        for name in sample.get_references():
            if name not in names:
                names.append(name)
    return names


def run_benchmark(
    model: "AsrModel",
    dataset: "DatasetSource",
    max_samples: Optional[int] = None,
    split: str = "test",
    measure_speed: bool = True,
) -> BenchmarkResult:
    """Benchmark ``model`` on ``dataset``.

    Args:
        model: An :class:`AsrModel` instance.
        dataset: A :class:`DatasetSource` instance.
        max_samples: Cap on the number of samples (None = all).
        split: Dataset split to use.
        measure_speed: Time inference and report RTFx.

    Returns:
        A :class:`BenchmarkResult` with per-reference aggregate metrics,
        per-sample detail, and (optional) speed stats.
    """
    logger.info(
        "Starting benchmark: model=%s, dataset=%s, max_samples=%s",
        model.model_name,
        dataset.name,
        max_samples,
    )

    samples: List["Sample"] = list(
        dataset.iter_samples(split=split, max_samples=max_samples)
    )

    if not samples:
        logger.warning("No samples found for benchmark")
        return BenchmarkResult(
            model=model.display_name,
            dataset=dataset.name,
            language=model.language,
            num_samples=0,
        )

    ref_names = _collect_reference_names(samples)
    primary_reference = _resolve_primary(samples, ref_names)
    logger.info("Benchmarking %d samples; references=%s", len(samples), ref_names)

    model.load()

    # Per-reference accumulators
    hyps_by_ref = {name: [] for name in ref_names}
    refs_by_ref = {name: [] for name in ref_names}
    per_sample_results: List[SampleResult] = []

    total_infer_s = 0.0
    total_audio_s = 0.0

    audio_paths = [s.audio_path or "" for s in samples]
    n_batches = (len(audio_paths) + model.batch_size - 1) // model.batch_size

    for i in range(0, len(audio_paths), model.batch_size):
        batch_paths = audio_paths[i : i + model.batch_size]
        batch_samples = samples[i : i + model.batch_size]
        logger.info("Processing batch %d/%d", i // model.batch_size + 1, n_batches)

        t0 = time.perf_counter()
        try:
            hypotheses = model.transcribe_batch(batch_paths)
        except Exception as e:  # noqa: BLE001 - mirror legacy behavior
            logger.error("Error transcribing batch: %s: %s", type(e).__name__, e)
            logger.error("Traceback: %s", traceback.format_exc())
            if isinstance(e, ImportError):
                raise
            hypotheses = [""] * len(batch_paths)
        batch_infer_s = time.perf_counter() - t0
        total_infer_s += batch_infer_s

        for hyp, sample in zip(hypotheses, batch_samples):
            sample_refs = sample.get_references()
            total_audio_s += sample.duration or 0.0

            sample_metrics = {}
            for name in ref_names:
                ref_text = sample_refs.get(name)
                if ref_text:
                    hyps_by_ref[name].append(hyp)
                    refs_by_ref[name].append(ref_text)
                    sample_metrics[name] = compute_single_sample_metrics(hyp, ref_text)

            per_sample_results.append(
                SampleResult(
                    index=sample.dataset_info.get("index", i),
                    audio_path=sample.audio_path or "",
                    hypothesis=hyp,
                    duration=sample.duration,
                    references=sample_refs,
                    metrics=sample_metrics,
                    speaker_id=sample.metadata.get("speaker_id"),
                )
            )

    # Aggregate per reference
    results = {}
    for name in ref_names:
        if hyps_by_ref[name]:
            results[name] = compute_asr_metrics(hyps_by_ref[name], refs_by_ref[name])

    speed = {}
    if measure_speed and total_infer_s > 0:
        speed = {
            "rtfx": (total_audio_s / total_infer_s) if total_infer_s else 0.0,
            "total_audio_s": total_audio_s,
            "total_infer_s": total_infer_s,
        }

    result = BenchmarkResult(
        model=model.display_name,
        dataset=dataset.name,
        language=model.language,
        num_samples=len(samples),
        num_skipped=0,
        references=[n for n in ref_names if n in results],
        primary_reference=primary_reference,
        results=results,
        speed=speed,
        per_sample=per_sample_results,
    )

    primary_wer = results.get(primary_reference, {}).get("wer")
    logger.info(
        "Benchmark complete: primary=%s WER=%s",
        primary_reference,
        f"{primary_wer:.2%}" if primary_wer is not None else "n/a",
    )
    return result


def _resolve_primary(samples: List["Sample"], ref_names: List[str]) -> str:
    """Pick the reference used for ranking / the leaderboard average column."""
    for sample in samples:
        if sample.primary_reference and sample.primary_reference in ref_names:
            return sample.primary_reference
    return ref_names[0] if ref_names else ""
