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

    def _load_model(self):
        """Lazy-load the Voxtral Realtime model and processor."""
        if self._model is None:
            logger.info(f"Loading Voxtral Realtime model: {self.model_name}")
            try:
                import torch
                from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

                self._processor = AutoProcessor.from_pretrained(self.model_name)

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
                    "Voxtral Realtime support requires transformers>=5.2.0. "
                    "Install with: pip install --upgrade 'transformers>=5.2.0'. "
                    f"Original error: {e}"
                ) from e

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Voxtral Realtime with true batching."""
        import soundfile as sf

        self._load_model()

        # Load all audio files in the batch
        audios = []
        for audio_path in audio_paths:
            audio, _ = sf.read(audio_path, dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            audios.append(audio)

        try:
            # True batching: single processor + generate call for entire batch
            inputs = self._processor(audios, return_tensors="pt")
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
            return [t.strip() for t in decoded]

        except Exception as e:
            logger.warning(
                f"Batch transcription failed, falling back to per-file: {e}"
            )
            # Fallback: process each file individually
            results = []
            for audio_path, audio in zip(audio_paths, audios):
                try:
                    inputs = self._processor(audio, return_tensors="pt")
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
                    results.append(decoded[0].strip() if decoded else "")
                except Exception as inner_e:
                    logger.warning(f"Error transcribing {audio_path}: {inner_e}")
                    results.append("")
            return results
