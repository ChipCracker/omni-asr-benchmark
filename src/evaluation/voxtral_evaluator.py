"""Voxtral Evaluator using HuggingFace Transformers."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Language code mapping from ISO codes to Voxtral language codes
LANGUAGE_MAP = {
    "deu_Latn": "de",
    "eng_Latn": "en",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "por_Latn": "pt",
    "ita_Latn": "it",
    "nld_Latn": "nl",
    "hin_Deva": "hi",
}


class VoxtralEvaluator(BaseEvaluator):
    """Evaluator for Mistral Voxtral models via HuggingFace Transformers.

    Supports transcription in English, Spanish, French, Portuguese,
    Hindi, German, Dutch, and Italian. Handles up to 30 minutes of audio.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Voxtral-Mini-3B-2507",
        language: str = "deu_Latn",
        batch_size: int = 1,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._processor = None

        # Map language code to Voxtral format
        self._voxtral_lang = LANGUAGE_MAP.get(language, "en")
        if language not in LANGUAGE_MAP:
            logger.warning(
                f"Language {language} not in supported list. "
                f"Using 'en'. Supported: {list(LANGUAGE_MAP.keys())}"
            )

    def _load_model(self):
        """Lazy-load the Voxtral model and processor."""
        if self._model is None:
            logger.info(f"Loading Voxtral model: {self.model_name}")
            try:
                import torch
                from transformers import AutoProcessor, VoxtralForConditionalGeneration

                self._processor = AutoProcessor.from_pretrained(self.model_name)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = VoxtralForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    device_map=device,
                )
                self._device = device
                logger.info(f"Voxtral model loaded on {device}")

            except ImportError as e:
                raise ImportError(
                    "Voxtral support requires transformers and mistral-common. "
                    "Install with: pip install -U transformers && "
                    "pip install --upgrade 'mistral-common[audio]'"
                ) from e

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Voxtral."""
        import torch

        self._load_model()

        results = []
        for audio_path in audio_paths:
            try:
                # Apply transcription request
                inputs = self._processor.apply_transcription_request(
                    language=self._voxtral_lang,
                    audio=audio_path,
                    model_id=self.model_name,
                )

                dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
                inputs = inputs.to(self._device, dtype=dtype)

                # Generate transcription
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.0,
                )

                # Decode output, skipping input tokens
                decoded = self._processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                transcript = decoded[0].strip() if decoded else ""
                results.append(transcript)

            except Exception as e:
                logger.warning(f"Error transcribing {audio_path}: {e}")
                results.append("")

        return results
