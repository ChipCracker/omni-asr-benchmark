"""Base classes for dataset sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class Sample:
    """Represents a single sample from a dataset.

    Attributes:
        transcript: The reference transcription text.
        duration: Audio duration in seconds.
        dataset_info: Dataset-level information (name, language, split, audio_path, etc.).
        metadata: Additional sample-specific metadata.
    """
    transcript: str
    duration: float
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def audio_path(self) -> Optional[str]:
        """Get the audio file path from dataset_info."""
        return self.dataset_info.get("audio_path")

    @property
    def ort_transcript(self) -> Optional[str]:
        """Get the standard orthography transcript if available."""
        return self.metadata.get("ort_transcript")


class DatasetSource(ABC):
    """Abstract base class for dataset sources.

    Provides a common interface for iterating over dataset samples.
    """

    name: str = "base"

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize the dataset source.

        Args:
            cache_dir: Optional directory for caching processed data.
        """
        self.cache_dir = cache_dir

    @abstractmethod
    def iter_samples(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> Iterable[Sample]:
        """Iterate over samples in the dataset.

        Args:
            split: The dataset split to use (e.g., "train", "test", "dev").
            max_samples: Maximum number of samples to yield. None for all samples.
            start_index: Index to start from (for resumption).

        Yields:
            Sample objects containing transcripts and metadata.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
