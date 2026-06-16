"""ASR models and the model registry.

A "model" implements the :class:`~src.models.base.AsrModel` interface: it knows
how to turn a batch of audio file paths into transcription strings, and nothing
about datasets, references, or metrics. The generic benchmark engine
(:mod:`src.benchmark.runner`) drives it.
"""

from .base import AsrModel
from .registry import get_model

__all__ = ["AsrModel", "get_model"]
