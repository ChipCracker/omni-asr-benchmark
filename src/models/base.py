"""Abstract base class for ASR models.

This is the *model* interface of the framework. A model is responsible for one
thing only: transcribing a batch of audio files. Everything else — iterating a
dataset, computing WER/CER against named references, timing inference,
serializing results — lives in the generic benchmark engine
(:mod:`src.benchmark.runner`).

Concrete models subclass :class:`AsrModel` and implement
:meth:`transcribe_batch`. Heavy backends (transformers, NeMo, ...) are loaded
lazily inside the subclass so importing the registry never pulls in optional
dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark.result import BenchmarkResult
    from ..datasets.base import DatasetSource


class AsrModel(ABC):
    """Interface every benchmarkable ASR model inherits from."""

    def __init__(self, model_name: str, language: str, batch_size: int) -> None:
        """Initialize the model.

        Args:
            model_name: Model identifier / card (e.g. ``openai/whisper-large-v3``).
            language: Language code for transcription (e.g. ``deu_Latn``).
            batch_size: Batch size for inference.
        """
        self.model_name = model_name
        self.language = language
        self.batch_size = batch_size

    @property
    def display_name(self) -> str:
        """Human-readable name used as the leaderboard row label."""
        return self.model_name

    def load(self) -> None:
        """Eagerly load the underlying model (optional).

        The default is a no-op; subclasses that load lazily inside
        :meth:`transcribe_batch` need not override this.
        """

    def unload(self) -> None:
        """Release the underlying model / free GPU memory (optional)."""

    def benchmark(
        self,
        dataset: "DatasetSource",
        max_samples: Optional[int] = None,
        split: str = "test",
        measure_speed: bool = True,
    ) -> "BenchmarkResult":
        """Run the generic benchmark engine on this model.

        The default delegates to :func:`src.benchmark.runner.run_benchmark`,
        which only ever calls :meth:`transcribe_batch`. Models that need a
        custom evaluation loop (e.g. oracle speaker selection) override this.
        """
        from ..benchmark.runner import run_benchmark

        return run_benchmark(
            self, dataset, max_samples=max_samples, split=split, measure_speed=measure_speed
        )

    @abstractmethod
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe a batch of audio files.

        Args:
            audio_paths: Paths to audio files.

        Returns:
            One transcription string per input path, in the same order.
        """
        raise NotImplementedError
