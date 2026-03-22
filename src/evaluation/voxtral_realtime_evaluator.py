"""Voxtral Realtime evaluator using HuggingFace Transformers."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class VoxtralRealtimeEvaluator(BaseEvaluator):
    """Evaluator for Mistral Voxtral Realtime models via HuggingFace Transformers.

    Uses the native Transformers integration for file-based transcription.
    For production-grade low-latency streaming, Mistral recommends vLLM upstream,
    but the Transformers path is sufficient for offline benchmark runs.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Voxtral-Mini-4B-Realtime-2602",
        language: str = "deu_Latn",
        batch_size: int = 1,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._processor = None
        self._audio_cls = None

    def _load_model(self):
        """Lazy-load the Voxtral Realtime model and processor."""
        if self._model is None:
            logger.info(f"Loading Voxtral Realtime model: {self.model_name}")
            try:
                import torch
                from mistral_common.tokens.tokenizers.audio import Audio
                from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

                self._processor = AutoProcessor.from_pretrained(self.model_name)
                self._audio_cls = Audio

                has_cuda = torch.cuda.is_available()
                self._model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
                    device_map="auto" if has_cuda else None,
                )
                if not has_cuda:
                    self._model = self._model.to("cpu")

                self._model.eval()
                self._device = self._model.device
                logger.info(f"Voxtral Realtime model loaded on {self._device}")

            except ImportError as e:
                raise ImportError(
                    "Voxtral Realtime support requires transformers>=5.2.0 and "
                    "mistral-common with audio extras. Install with: "
                    "pip install --upgrade transformers 'mistral-common[audio]'. "
                    f"Original error: {e}"
                ) from e

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Voxtral Realtime."""
        self._load_model()

        results = []
        for audio_path in audio_paths:
            try:
                audio = self._audio_cls.from_file(audio_path, strict=False)
                audio.resample(self._processor.feature_extractor.sampling_rate)

                inputs = self._processor(
                    audio.audio_array,
                    return_tensors="pt",
                )
                inputs = inputs.to(self._device, dtype=self._model.dtype)

                outputs = self._model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=500,
                )

                decoded = self._processor.batch_decode(
                    outputs, skip_special_tokens=True
                )

                transcript = decoded[0].strip() if decoded else ""
                results.append(transcript)

            except Exception as e:
                import traceback

                logger.warning(f"Error transcribing {audio_path}: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                results.append("")

        return results
