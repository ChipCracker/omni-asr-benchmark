"""Generic benchmark engine: result schema, metrics, and runner."""

from .metrics import compute_asr_metrics, compute_single_sample_metrics, normalize_text
from .result import BenchmarkResult, ReferenceMetrics, SampleResult
from .runner import run_benchmark

__all__ = [
    "BenchmarkResult",
    "ReferenceMetrics",
    "SampleResult",
    "compute_asr_metrics",
    "compute_single_sample_metrics",
    "normalize_text",
    "run_benchmark",
]
