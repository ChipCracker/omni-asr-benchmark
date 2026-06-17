"""Aggregate benchmark result JSONs into a leaderboard matrix.

Rows = models, columns = ``(dataset, reference)`` pairs. The *primary* reference
of each dataset feeds the per-model **Average WER** used for ranking
(HuggingFace Open ASR style); non-primary references appear as extra columns but
do not affect the ranking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..benchmark.result import BenchmarkResult

logger = logging.getLogger(__name__)

Column = Tuple[str, str]  # (dataset, reference)

# Files that live in results/ but are not benchmark results.
_SKIP_NAMES = {"leaderboard.json"}


@dataclass
class Cell:
    wer: Optional[float] = None
    cer: Optional[float] = None


@dataclass
class Row:
    model: str
    cells: Dict[Column, Cell] = field(default_factory=dict)
    rtfx: Optional[float] = None
    average_wer: Optional[float] = None
    average_cer: Optional[float] = None
    rank: int = 0


@dataclass
class Leaderboard:
    datasets: List[str]
    columns: List[Column]            # ordered (dataset, reference)
    primary_by_dataset: Dict[str, str]
    rows: List[Row]                  # sorted by average_wer ascending
    dataset_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Per-utterance drill-down (used by the HTML export):
    #   ground_truth[dataset][index] = {"spk", "dur", "refs": {ref: text}}   (GT deduped across models)
    #   details[model][dataset] = [ {"i", "hyp", "m": {ref: [wer, cer]}} ]
    ground_truth: Dict[str, Dict[int, Dict[str, Any]]] = field(default_factory=dict)
    details: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)


def _display_name(result: BenchmarkResult, filename: str) -> str:
    """Mirror the legacy plot naming: tag Voxtral Realtime online/offline."""
    model = result.model
    if "-online" in filename:
        return f"{model} (online)"
    if "Realtime" in model and "online" not in filename:
        return f"{model} (offline)"
    return model


def load_results(results_dir: Path) -> List[Tuple[BenchmarkResult, str]]:
    """Load every benchmark result JSON in ``results_dir``.

    Returns a list of ``(result, filename)`` tuples. Non-result JSONs are
    skipped silently.
    """
    out: List[Tuple[BenchmarkResult, str]] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name.startswith(".") or path.name in _SKIP_NAMES:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping unreadable %s: %s", path.name, e)
            continue
        if not isinstance(data, dict) or "results" not in data or "model" not in data:
            continue
        try:
            out.append((BenchmarkResult.from_dict(data), path.name))
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping malformed result %s: %s", path.name, e)
    return out


def build_leaderboard(
    results: List[Tuple[BenchmarkResult, str]],
    exclude_failed: bool = False,
    failed_threshold: float = 0.99,
) -> Leaderboard:
    """Build a :class:`Leaderboard` from loaded results.

    Args:
        results: ``(result, filename)`` tuples from :func:`load_results`.
        exclude_failed: Drop a ``(model, dataset)`` entry whose primary WER is
            ``>= failed_threshold`` (broken runs that would distort the scale).
        failed_threshold: WER cutoff for ``exclude_failed``.
    """
    # Keyed by (model_display_name, dataset); later files win on collision.
    by_model_dataset: Dict[Tuple[str, str], BenchmarkResult] = {}
    model_order: List[str] = []
    dataset_order: List[str] = []
    primary_by_dataset: Dict[str, str] = {}

    for result, filename in results:
        name = _display_name(result, filename)
        primary = result.primary_reference
        if exclude_failed and primary in result.results:
            wer = result.results[primary].get("wer")
            if wer is not None and wer >= failed_threshold:
                logger.info("Excluding failed run: %s on %s (WER=%.2f)", name, result.dataset, wer)
                continue
        by_model_dataset[(name, result.dataset)] = result
        if name not in model_order:
            model_order.append(name)
        if result.dataset not in dataset_order:
            dataset_order.append(result.dataset)
        if result.dataset not in primary_by_dataset and primary:
            primary_by_dataset[result.dataset] = primary

    # Column order: per dataset, primary reference first, then the rest sorted.
    columns: List[Column] = []
    for dataset in dataset_order:
        refs_present: List[str] = []
        for (_, ds), result in by_model_dataset.items():
            if ds == dataset:
                for ref in result.results:
                    if ref not in refs_present:
                        refs_present.append(ref)
        primary = primary_by_dataset.get(dataset, refs_present[0] if refs_present else "")
        ordered = ([primary] if primary in refs_present else []) + sorted(
            r for r in refs_present if r != primary
        )
        columns.extend((dataset, ref) for ref in ordered)

    # Build rows.
    rows: List[Row] = []
    for name in model_order:
        row = Row(model=name)
        rtfx_values: List[float] = []
        primary_wers: List[float] = []
        primary_cers: List[float] = []
        for dataset in dataset_order:
            result = by_model_dataset.get((name, dataset))
            if result is None:
                continue
            for ref, metrics in result.results.items():
                row.cells[(dataset, ref)] = Cell(wer=metrics.get("wer"), cer=metrics.get("cer"))
            primary = primary_by_dataset.get(dataset, "")
            pm = result.results.get(primary, {})
            if pm.get("wer") is not None:
                primary_wers.append(pm["wer"])
            if pm.get("cer") is not None:
                primary_cers.append(pm["cer"])
            rtfx = result.speed.get("rtfx")
            if rtfx:
                rtfx_values.append(rtfx)
        row.average_wer = sum(primary_wers) / len(primary_wers) if primary_wers else None
        row.average_cer = sum(primary_cers) / len(primary_cers) if primary_cers else None
        row.rtfx = sum(rtfx_values) / len(rtfx_values) if rtfx_values else None
        rows.append(row)

    rows.sort(key=lambda r: r.average_wer if r.average_wer is not None else float("inf"))
    for i, row in enumerate(rows, start=1):
        row.rank = i

    # Per-dataset metadata for the "corpus used" banner.
    dataset_meta: Dict[str, Dict[str, Any]] = {}
    for dataset in dataset_order:
        ds_results = [r for (_, ds), r in by_model_dataset.items() if ds == dataset]
        dataset_meta[dataset] = {
            "samples": max((r.num_samples for r in ds_results), default=0),
            "references": [ref for (d, ref) in columns if d == dataset],
            "primary": primary_by_dataset.get(dataset, ""),
            "models": len(ds_results),
            "language": next((r.language for r in ds_results if r.language), ""),
        }

    # Per-utterance drill-down: ground truth deduped across models,
    # hypotheses + metrics kept per model.
    ground_truth: Dict[str, Dict[int, Dict[str, Any]]] = {}
    details: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for (name, dataset), result in by_model_dataset.items():
        gt_ds = ground_truth.setdefault(dataset, {})
        rows_out: List[Dict[str, Any]] = []
        for s in result.per_sample:
            idx = s.index
            g = gt_ds.setdefault(idx, {"spk": s.speaker_id, "dur": round(s.duration or 0.0, 1), "refs": {}})
            for ref_name, ref_text in s.references.items():
                if ref_text and ref_name not in g["refs"]:
                    g["refs"][ref_name] = ref_text
            m = {}
            for ref_name, mm in s.metrics.items():
                wer = mm.get("wer")
                cer = mm.get("cer")
                m[ref_name] = [
                    round(wer, 4) if wer is not None else None,
                    round(cer, 4) if cer is not None else None,
                ]
            rows_out.append({"i": idx, "hyp": s.hypothesis, "m": m})
        details.setdefault(name, {})[dataset] = rows_out

    return Leaderboard(
        datasets=dataset_order,
        columns=columns,
        primary_by_dataset=primary_by_dataset,
        rows=rows,
        dataset_meta=dataset_meta,
        ground_truth=ground_truth,
        details=details,
    )
