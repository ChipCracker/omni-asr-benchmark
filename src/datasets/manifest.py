"""Generic NeMo-style manifest dataset source.

Reads a JSONL manifest where each line is an object with at least
``audio_filepath`` and ``text`` (``duration`` and ``utt_id`` optional) and
yields :class:`Sample` objects. Reusable for any manifest-based corpus; the
``ksof`` dataset is a thin subclass with a default manifest path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import soundfile as sf

from .base import DatasetSource, Sample

logger = logging.getLogger(__name__)


class ManifestSource(DatasetSource):
    """Dataset source backed by a NeMo-style JSONL manifest."""

    name = "manifest"

    def __init__(
        self,
        manifest_path: str | Path,
        name: str = "manifest",
        reference_key: str = "ref",
        audio_root: Optional[str | Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.manifest_path = Path(manifest_path)
        self.name = name  # instance attribute overrides the class default
        self.reference_key = reference_key
        self.audio_root = Path(audio_root) if audio_root else None

    def _resolve_audio(self, audio_filepath: str) -> str:
        """Resolve the audio path, optionally relocating under ``audio_root``."""
        if self.audio_root is not None:
            return str(self.audio_root / Path(audio_filepath).name)
        return audio_filepath

    def iter_samples(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> Iterable[Sample]:
        if not self.manifest_path.is_file():
            logger.error("Manifest not found: %s", self.manifest_path)
            return

        logger.info("Loading %s manifest from %s", self.name, self.manifest_path)
        current_idx = 0
        end_index = float("inf") if max_samples is None else start_index + max_samples

        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if current_idx >= end_index:
                    break
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed manifest line %d", current_idx)
                    continue

                text = (entry.get("text") or "").strip()
                audio_filepath = entry.get("audio_filepath") or entry.get("audio")
                if not text or not audio_filepath:
                    continue

                if current_idx < start_index:
                    current_idx += 1
                    continue

                duration = entry.get("duration")
                if duration is None:
                    try:
                        duration = sf.info(self._resolve_audio(audio_filepath)).duration
                    except Exception:
                        duration = 0.0

                yield Sample(
                    transcript=text,
                    duration=float(duration),
                    references={self.reference_key: text},
                    primary_reference=self.reference_key,
                    dataset_info={
                        "dataset_name": self.name,
                        "split": split,
                        "index": current_idx,
                        "audio_path": self._resolve_audio(audio_filepath),
                    },
                    metadata={"utt_id": entry.get("utt_id")},
                )
                current_idx += 1


class KsofSource(ManifestSource):
    """KSOF corpus (German conversational ASR), shipped as a JSONL manifest."""

    name = "ksof"
    DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "manifests" / "ksof-manifest.jsonl"

    def __init__(self, manifest_path: str | Path | None = None, **kwargs) -> None:
        kwargs.pop("name", None)
        kwargs.pop("reference_key", None)
        super().__init__(
            manifest_path or self.DEFAULT_MANIFEST,
            name="ksof",
            reference_key="ref",
            **kwargs,
        )
