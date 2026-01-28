"""Canary-Qwen Evaluator using NVIDIA NeMo SALM."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CanaryEvaluator(BaseEvaluator):
    """Evaluator for NVIDIA Canary-Qwen models via NeMo SALM.

    Uses the Speech-Augmented Language Model (SALM) API for transcription.
    English-only model with 5.63% mean WER on HuggingFace OpenASR Leaderboard.
    """

    def __init__(
        self,
        model_name: str = "nvidia/canary-qwen-2.5b",
        language: str = "eng_Latn",
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None

        # Warn if using non-English language
        if not language.startswith("eng"):
            logger.warning(
                f"Canary-Qwen is English-only. Language {language} will be ignored."
            )

    def _get_model(self):
        """Lazy-load the NeMo SALM model."""
        if self._model is None:
            logger.info(f"Loading Canary model: {self.model_name}")
            try:
                from nemo.collections.speechlm2.models import SALM

                self._model = SALM.from_pretrained(self.model_name)
                self._model.eval()

                import torch
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    logger.info("Canary model loaded on CUDA")
                else:
                    logger.info("Canary model loaded on CPU")

            except ImportError as e:
                raise ImportError(
                    "Canary support requires latest NeMo trunk. "
                    "Install with: pip install 'nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git'"
                ) from e
        return self._model

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Canary-Qwen."""
        model = self._get_model()

        results = []
        for audio_path in audio_paths:
            try:
                # Build chat-style prompt per official example
                prompt = [{
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [audio_path]
                }]

                answer_ids = model.generate(
                    prompts=[prompt],
                    max_new_tokens=128,
                )

                transcript = model.tokenizer.ids_to_text(answer_ids[0].cpu())
                results.append(transcript.strip())

            except Exception as e:
                logger.warning(f"Error transcribing {audio_path}: {e}")
                results.append("")

        return results
