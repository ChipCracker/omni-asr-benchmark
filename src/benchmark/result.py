"""Generic benchmark result schema.

Replaces the RVG1-specific ``EvaluationResult`` (fixed ``dialect_reference`` /
``ort_reference`` keys) with a structure that is generic over *named
references*. A dataset may expose any number of references (e.g. ``ort``,
``dialect``, ``kan``); the benchmark scores the hypothesis against each one and
stores the result under that name.

``BenchmarkResult.from_dict`` understands both the new schema (version 2) and
the legacy schema (version 1, no ``schema_version`` field) so the leaderboard
can consume old and new result JSONs side by side without rewriting them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# Legacy v1 result keys ("<name>_reference") map onto bare reference names.
_LEGACY_SUFFIX = "_reference"


@dataclass
class ReferenceMetrics:
    """Aggregate metrics of a model against a single named reference."""

    wer: Optional[float] = None
    cer: Optional[float] = None
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReferenceMetrics":
        return cls(
            wer=d.get("wer"),
            cer=d.get("cer"),
            substitutions=d.get("substitutions", 0),
            deletions=d.get("deletions", 0),
            insertions=d.get("insertions", 0),
            num_samples=d.get("num_samples", 0),
        )


@dataclass
class SampleResult:
    """Result for a single sample, generic over named references."""

    index: int
    audio_path: str
    hypothesis: str
    duration: float
    references: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    inference_time: Optional[float] = None
    speaker_id: Optional[str] = None
    raw_hypothesis: Optional[str] = None  # full model output (e.g. multi-speaker JSON)
    extra: Optional[Dict[str, Any]] = None  # model-specific data (e.g. selected speaker)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SampleResult":
        return cls(
            index=d.get("index", 0),
            audio_path=d.get("audio_path", ""),
            hypothesis=d.get("hypothesis", ""),
            duration=d.get("duration", 0.0),
            references=dict(d.get("references", {})),
            metrics={k: dict(v) for k, v in d.get("metrics", {}).items()},
            inference_time=d.get("inference_time"),
            speaker_id=d.get("speaker_id"),
            raw_hypothesis=d.get("raw_hypothesis"),
            extra=d.get("extra"),
        )


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one (model, dataset) run."""

    model: str
    dataset: str
    language: str
    num_samples: int
    num_skipped: int = 0
    schema_version: int = SCHEMA_VERSION
    references: List[str] = field(default_factory=list)
    primary_reference: str = ""
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    speed: Dict[str, float] = field(default_factory=dict)
    per_sample: List[SampleResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "language": self.language,
            "num_samples": self.num_samples,
            "num_skipped": self.num_skipped,
            "schema_version": self.schema_version,
            "references": self.references,
            "primary_reference": self.primary_reference,
            "timestamp": self.timestamp,
            "results": self.results,
            "speed": self.speed,
            "per_sample": [s.to_dict() for s in self.per_sample],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", path)

    # ------------------------------------------------------------------ #
    # Deserialization (v1 + v2)
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: Path) -> "BenchmarkResult":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        if d.get("schema_version", 1) >= 2:
            return cls._from_v2(d)
        return cls._from_v1(d)

    @classmethod
    def _from_v2(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        return cls(
            model=d.get("model", "unknown"),
            dataset=d.get("dataset", "unknown"),
            language=d.get("language", ""),
            num_samples=d.get("num_samples", 0),
            num_skipped=d.get("num_skipped", 0),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            references=list(d.get("references", [])),
            primary_reference=d.get("primary_reference", ""),
            results=dict(d.get("results", {})),
            speed=dict(d.get("speed", {})),
            per_sample=[SampleResult.from_dict(s) for s in d.get("per_sample", [])],
            timestamp=d.get("timestamp", ""),
        )

    @classmethod
    def _from_v1(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        """Normalize a legacy ``dialect_reference``/``ort_reference`` result."""

        def strip(name: str) -> str:
            return name[: -len(_LEGACY_SUFFIX)] if name.endswith(_LEGACY_SUFFIX) else name

        # Aggregate metrics: keep only references that were actually scored.
        results: Dict[str, Dict[str, Any]] = {}
        for key, metrics in d.get("results", {}).items():
            if metrics and metrics.get("wer") is not None:
                results[strip(key)] = dict(metrics)
        references = list(results.keys())

        # Pick a primary reference for ranking. RVG1 historically sorted by ORT
        # (standard orthography), which is the fairer cross-model metric.
        if "ort" in results:
            primary = "ort"
        elif references:
            primary = references[0]
        else:
            primary = ""

        # Per-sample: collapse the dialect_*/ort_* columns into nested metrics.
        per_sample: List[SampleResult] = []
        for s in d.get("per_sample", []):
            refs: Dict[str, str] = {}
            metrics: Dict[str, Dict[str, Optional[float]]] = {}
            for ref in ("dialect", "ort", "kan"):
                ref_text = s.get(f"{ref}_reference")
                if ref_text is not None:
                    refs[ref] = ref_text
                wer_v = s.get(f"{ref}_wer")
                cer_v = s.get(f"{ref}_cer")
                if wer_v is not None or cer_v is not None:
                    metrics[ref] = {"wer": wer_v, "cer": cer_v}
            extra = {}
            if s.get("selected_speaker") is not None:
                extra["selected_speaker"] = s.get("selected_speaker")
            if s.get("all_speakers") is not None:
                extra["all_speakers"] = s.get("all_speakers")
            per_sample.append(
                SampleResult(
                    index=s.get("index", 0),
                    audio_path=s.get("audio_path", ""),
                    hypothesis=s.get("hypothesis", ""),
                    duration=s.get("duration", 0.0),
                    references=refs,
                    metrics=metrics,
                    speaker_id=s.get("speaker_id"),
                    raw_hypothesis=s.get("raw_hypothesis"),
                    extra=extra or None,
                )
            )

        return cls(
            model=d.get("model", "unknown"),
            dataset=d.get("dataset", "unknown"),
            language=d.get("language", ""),
            num_samples=d.get("num_samples", 0),
            num_skipped=d.get("num_skipped", 0),
            schema_version=1,
            references=references,
            primary_reference=primary,
            results=results,
            speed={},
            per_sample=per_sample,
            timestamp=d.get("timestamp", ""),
        )
