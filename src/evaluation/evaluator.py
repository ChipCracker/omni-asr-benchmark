"""OmniASR Evaluator for ASR model evaluation."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator, EvaluationResult, SampleResult

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = ["OmniASREvaluator", "EvaluationResult", "SampleResult", "get_evaluator"]


class OmniASREvaluator(BaseEvaluator):
    """Evaluator for OmniASR models.

    Uses the omnilingual-asr ASRInferencePipeline for transcription
    and computes WER/CER metrics against reference transcriptions.
    """

    def __init__(
        self,
        model_card: str = "omniASR_LLM_Unlimited_7B_v2",
        language: str = "deu_Latn",
        batch_size: int = 2,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model_card: The model card name for the ASR model.
            language: Language code for transcription (e.g., "deu_Latn" for German).
            batch_size: Batch size for inference (small due to 7B model size).
        """
        super().__init__(model_card, language, batch_size)
        self.model_card = model_card  # Keep for backward compatibility
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


def get_evaluator(model_name: str, language: str, batch_size: int) -> BaseEvaluator:
    """Factory function to create the appropriate evaluator based on model name.

    Args:
        model_name: The model identifier (e.g., "openai/whisper-large-v3",
                    "nvidia/parakeet-ctc-1.1b", or "omniASR_LLM_Unlimited_7B_v2").
        language: Language code for transcription (e.g., "deu_Latn").
        batch_size: Batch size for inference.

    Returns:
        An appropriate evaluator instance for the specified model.
    """
    model_lower = model_name.lower()

    # Check CrisperWhisper before generic whisper (since it contains "whisper")
    if "crisperwhisper" in model_lower or "crisper" in model_lower:
        from .crisperwhisper_evaluator import CrisperWhisperEvaluator
        return CrisperWhisperEvaluator(model_name, language, batch_size)
    elif "whisper" in model_lower:
        from .whisper_evaluator import WhisperEvaluator
        return WhisperEvaluator(model_name, language, batch_size)
    elif "parakeet" in model_lower:
        from .parakeet_evaluator import ParakeetEvaluator
        return ParakeetEvaluator(model_name, language, batch_size)
    elif "vibevoice" in model_lower:
        from .vibevoice_evaluator import VibeVoiceEvaluator
        return VibeVoiceEvaluator(model_name, language, batch_size)
    elif "canary" in model_lower:
        from .canary_evaluator import CanaryEvaluator
        return CanaryEvaluator(model_name, language, batch_size)
    elif "voxtral" in model_lower:
        from .voxtral_evaluator import VoxtralEvaluator
        return VoxtralEvaluator(model_name, language, batch_size)
    else:
        # Default to OmniASR evaluator
        return OmniASREvaluator(model_name, language, batch_size)
