"""Backward-compatibility shim.

The framework moved to :mod:`src.models` (model interface + registry) and
:mod:`src.benchmark` (engine, metrics, result schema). This module re-exports
the old names so existing imports keep working. Prefer the new modules in new
code; this shim may be removed in a future cleanup.
"""

from __future__ import annotations

from typing import Optional

from ..benchmark.metrics import compute_asr_metrics, compute_single_sample_metrics
from ..benchmark.result import BenchmarkResult as EvaluationResult  # noqa: F401 (deprecated alias)
from ..benchmark.result import SampleResult
from ..models import AsrModel as BaseEvaluator  # noqa: F401 (deprecated alias)
from ..models import get_model


def get_evaluator(
    model_name: str,
    language: str,
    batch_size: int,
    model_dir: Optional[str] = None,
) -> BaseEvaluator:
    """Deprecated alias for :func:`src.models.get_model`."""
    return get_model(model_name, language, batch_size, model_dir)


__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "SampleResult",
    "compute_asr_metrics",
    "compute_single_sample_metrics",
    "get_evaluator",
    "get_model",
]
