"""OmniASR Evaluator for ASR model evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..datasets.base import DatasetSource, Sample
from .metrics import compute_asr_metrics, compute_single_sample_metrics

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Result for a single sample evaluation."""
    index: int
    audio_path: str
    hypothesis: str
    dialect_reference: str
    ort_reference: Optional[str]
    dialect_wer: float
    dialect_cer: float
    ort_wer: Optional[float]
    ort_cer: Optional[float]
    duration: float
    speaker_id: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    model: str
    dataset: str
    language: str
    num_samples: int
    num_skipped: int
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_sample: List[SampleResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model": self.model,
            "dataset": self.dataset,
            "language": self.language,
            "num_samples": self.num_samples,
            "num_skipped": self.num_skipped,
            "timestamp": self.timestamp,
            "results": self.results,
            "per_sample": [asdict(s) for s in self.per_sample],
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {path}")


class OmniASREvaluator:
    """Evaluator for OmniASR models.

    Uses the omnilingual-asr ASRInferencePipeline for transcription
    and computes WER/CER metrics against reference transcriptions.
    """

    def __init__(
        self,
        model_card: str = "omniASR_LLM_7B_v2",
        language: str = "deu_Latn",
        batch_size: int = 2,
        max_audio_duration: float = 40.0,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model_card: The model card name for the ASR model.
            language: Language code for transcription (e.g., "deu_Latn" for German).
            batch_size: Batch size for inference (small due to 7B model size).
            max_audio_duration: Maximum audio duration in seconds (skip longer files).
        """
        self.model_card = model_card
        self.language = language
        self.batch_size = batch_size
        self.max_audio_duration = max_audio_duration
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy-load the ASR pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading ASR pipeline with model_card={self.model_card}")
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
            self._pipeline = ASRInferencePipeline(model_card=self.model_card)
        return self._pipeline

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe a batch of audio files.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of transcription strings.
        """
        pipeline = self._get_pipeline()
        lang = [self.language] * len(audio_paths)
        return pipeline.transcribe(audio_paths, lang=lang, batch_size=self.batch_size)

    def evaluate(
        self,
        dataset_source: DatasetSource,
        max_samples: Optional[int] = None,
        split: str = "test",
    ) -> EvaluationResult:
        """Evaluate the ASR model on a dataset.

        Args:
            dataset_source: The dataset source to evaluate on.
            max_samples: Maximum number of samples to evaluate (None for all).
            split: Dataset split to use.

        Returns:
            EvaluationResult containing metrics and per-sample results.
        """
        logger.info(
            f"Starting evaluation: model={self.model_card}, dataset={dataset_source.name}, "
            f"max_samples={max_samples}"
        )

        samples: List[Sample] = []
        skipped_count = 0

        # Collect samples, filtering by duration
        for sample in dataset_source.iter_samples(split=split, max_samples=max_samples):
            if sample.duration > self.max_audio_duration:
                logger.debug(
                    f"Skipping sample (duration {sample.duration:.1f}s > {self.max_audio_duration}s): "
                    f"{sample.audio_path}"
                )
                skipped_count += 1
                continue
            samples.append(sample)

        if not samples:
            logger.warning("No samples found for evaluation")
            return EvaluationResult(
                model=self.model_card,
                dataset=dataset_source.name,
                language=self.language,
                num_samples=0,
                num_skipped=skipped_count,
            )

        logger.info(f"Evaluating {len(samples)} samples (skipped {skipped_count} due to duration)")

        # Process in batches
        all_hypotheses: List[str] = []
        dialect_references: List[str] = []
        ort_references: List[str] = []
        per_sample_results: List[SampleResult] = []

        audio_paths = [s.audio_path for s in samples]

        # Batch transcription
        for i in range(0, len(audio_paths), self.batch_size):
            batch_paths = audio_paths[i:i + self.batch_size]
            batch_samples = samples[i:i + self.batch_size]

            logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(audio_paths) + self.batch_size - 1) // self.batch_size}")

            try:
                hypotheses = self.transcribe_batch(batch_paths)
            except Exception as e:
                logger.error(f"Error transcribing batch: {e}")
                hypotheses = [""] * len(batch_paths)

            for j, (hyp, sample) in enumerate(zip(hypotheses, batch_samples)):
                all_hypotheses.append(hyp)
                dialect_references.append(sample.transcript)

                ort_ref = sample.ort_transcript
                ort_references.append(ort_ref or "")

                # Compute per-sample metrics
                dialect_metrics = compute_single_sample_metrics(hyp, sample.transcript)
                ort_metrics = (
                    compute_single_sample_metrics(hyp, ort_ref)
                    if ort_ref
                    else {"wer": None, "cer": None}
                )

                per_sample_results.append(
                    SampleResult(
                        index=sample.dataset_info.get("index", i + j),
                        audio_path=sample.audio_path or "",
                        hypothesis=hyp,
                        dialect_reference=sample.transcript,
                        ort_reference=ort_ref,
                        dialect_wer=dialect_metrics["wer"],
                        dialect_cer=dialect_metrics["cer"],
                        ort_wer=ort_metrics["wer"],
                        ort_cer=ort_metrics["cer"],
                        duration=sample.duration,
                        speaker_id=sample.metadata.get("speaker_id"),
                    )
                )

        # Compute aggregate metrics
        dialect_metrics = compute_asr_metrics(all_hypotheses, dialect_references)

        # Filter out samples without ORT reference for ORT metrics
        valid_ort_pairs = [
            (hyp, ref)
            for hyp, ref in zip(all_hypotheses, ort_references)
            if ref
        ]
        if valid_ort_pairs:
            ort_hyps, ort_refs = zip(*valid_ort_pairs)
            ort_metrics = compute_asr_metrics(list(ort_hyps), list(ort_refs))
        else:
            ort_metrics = {
                "wer": None,
                "cer": None,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "num_samples": 0,
            }

        result = EvaluationResult(
            model=self.model_card,
            dataset=dataset_source.name,
            language=self.language,
            num_samples=len(samples),
            num_skipped=skipped_count,
            results={
                "dialect_reference": dialect_metrics,
                "ort_reference": ort_metrics,
            },
            per_sample=per_sample_results,
        )

        logger.info(
            f"Evaluation complete: "
            f"Dialect WER={dialect_metrics['wer']:.2%}, "
            f"ORT WER={ort_metrics['wer']:.2%}" if ort_metrics['wer'] else ""
        )

        return result
