"""OmniASR Evaluator for ASR model evaluation."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import List

import torch
import torchaudio

from .base_evaluator import BaseEvaluator, EvaluationResult, SampleResult

logger = logging.getLogger(__name__)

MAX_AUDIO_SEC = 35  # safety margin below pipeline's 40s hard cap
CHUNK_DURATION_SEC = 30
MIN_CHUNK_DURATION_SEC = 2  # minimum chunk length to avoid 0-length feature sequences

# Models with a 40s audio limit that need chunking
MODELS_WITH_AUDIO_LIMIT = {"ctc", "llm_300m"}

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

        For models with a 40s audio limit (CTC models, LLM-300M), long audio
        files (>35s) are split into ~30s chunks, transcribed individually,
        and concatenated back together.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of transcription strings.
        """
        pipeline = self._get_pipeline()
        model_lower = self.model_card.lower()
        needs_chunking = any(tag in model_lower for tag in MODELS_WITH_AUDIO_LIMIT)

        if not needs_chunking:
            lang = [self.language] * len(audio_paths)
            return pipeline.transcribe(audio_paths, lang=lang, batch_size=self.batch_size)

        # Split long files into chunks (40s hard cap)
        all_paths = []       # flat list of paths to transcribe
        file_map = []        # (original_index, num_chunks) to reassemble
        temp_files = []

        for idx, path in enumerate(audio_paths):
            info = torchaudio.info(path)
            duration = info.num_frames / info.sample_rate

            if duration <= MAX_AUDIO_SEC:
                all_paths.append(path)
                file_map.append((idx, 1))
            else:
                waveform, sr = torchaudio.load(path)
                chunk_samples = int(CHUNK_DURATION_SEC * sr)
                min_samples = int(MIN_CHUNK_DURATION_SEC * sr)
                chunks = list(waveform.split(chunk_samples, dim=1))
                # Merge tiny last chunk into previous to avoid 0-length features
                if len(chunks) > 1 and chunks[-1].shape[1] < min_samples:
                    chunks[-2] = torch.cat([chunks[-2], chunks[-1]], dim=1)
                    chunks.pop()
                for chunk in chunks:
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    torchaudio.save(tmp.name, chunk, sr)
                    temp_files.append(tmp.name)
                    all_paths.append(tmp.name)
                file_map.append((idx, len(chunks)))
                logger.info(f"Split {path} ({duration:.1f}s) into {len(chunks)} chunks")

        try:
            lang = [self.language] * len(all_paths)
            all_transcriptions = pipeline.transcribe(all_paths, lang=lang, batch_size=self.batch_size)
        finally:
            for tmp_path in temp_files:
                os.unlink(tmp_path)

        # Reassemble: concatenate chunk transcriptions per original file
        results = []
        pos = 0
        for _, num_chunks in file_map:
            chunk_texts = all_transcriptions[pos:pos + num_chunks]
            results.append(" ".join(chunk_texts))
            pos += num_chunks

        return results


def get_evaluator(
    model_name: str,
    language: str,
    batch_size: int,
    model_dir: str | None = None,
) -> BaseEvaluator:
    """Factory function to create the appropriate evaluator based on model name.

    Args:
        model_name: The model identifier (e.g., "openai/whisper-large-v3",
                    "nvidia/parakeet-ctc-1.1b", or "omniASR_LLM_Unlimited_7B_v2").
        language: Language code for transcription (e.g., "deu_Latn").
        batch_size: Batch size for inference.
        model_dir: Path to model directory (used by syllabic-asr evaluator).

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
    elif "canary-1b" in model_lower:
        from .canary1b_evaluator import Canary1bEvaluator
        return Canary1bEvaluator(model_name, language, batch_size)
    elif "canary" in model_lower:
        # Canary-Qwen (existing)
        from .canary_evaluator import CanaryEvaluator
        return CanaryEvaluator(model_name, language, batch_size)
    elif "voxtral" in model_lower and "realtime" in model_lower and "online" in model_lower:
        from .voxtral_realtime_evaluator import VoxtralRealtimeOnlineEvaluator
        return VoxtralRealtimeOnlineEvaluator(model_name, language, batch_size)
    elif "voxtral" in model_lower and "realtime" in model_lower:
        from .voxtral_realtime_evaluator import VoxtralRealtimeEvaluator
        return VoxtralRealtimeEvaluator(model_name, language, batch_size)
    elif "voxtral" in model_lower:
        from .voxtral_evaluator import VoxtralEvaluator
        return VoxtralEvaluator(model_name, language, batch_size)
    elif "phi-4" in model_lower or "phi4" in model_lower:
        from .phi4_evaluator import Phi4Evaluator
        return Phi4Evaluator(model_name, language, batch_size)
    elif "qwen3-asr" in model_lower:
        from .qwen3asr_evaluator import Qwen3ASREvaluator
        return Qwen3ASREvaluator(model_name, language, batch_size)
    elif "syllabic-asr" in model_lower or "syllabic_asr" in model_lower:
        from .syllabic_asr_evaluator import SyllabicASREvaluator
        decode_mode = "rnnt" if "rnnt" in model_lower else "ctc"
        return SyllabicASREvaluator(
            model_name, language, batch_size,
            decode_mode=decode_mode, model_dir=model_dir,
        )
    elif "granite" in model_lower:
        from .granite_evaluator import GraniteEvaluator
        return GraniteEvaluator(model_name, language, batch_size)
    elif "omniasr" in model_lower or "omni-asr" in model_lower or "omni_asr" in model_lower:
        # OmniASR models (CTC-1B, LLM-7B, etc.)
        return OmniASREvaluator(model_name, language, batch_size)
    else:
        # Default to OmniASR evaluator
        return OmniASREvaluator(model_name, language, batch_size)
