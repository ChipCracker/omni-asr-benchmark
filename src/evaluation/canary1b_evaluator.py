"""Canary-1b-v2 Evaluator using NVIDIA NeMo ASRModel."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Language code mapping from Glottolog to Canary-1b format
LANGUAGE_MAP = {
    "deu_Latn": "de",
    "eng_Latn": "en",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    "swe_Latn": "sv",
    "dan_Latn": "da",
    "nor_Latn": "no",
    "fin_Latn": "fi",
    "ron_Latn": "ro",
    "hun_Latn": "hu",
    "ces_Latn": "cs",
    "slk_Latn": "sk",
    "bul_Cyrl": "bg",
    "ell_Grek": "el",
    "tur_Latn": "tr",
    "ukr_Cyrl": "uk",
    "rus_Cyrl": "ru",
    "hrv_Latn": "hr",
    "lit_Latn": "lt",
    "slv_Latn": "sl",
    "cat_Latn": "ca",
}


class Canary1bEvaluator(BaseEvaluator):
    """Evaluator for NVIDIA Canary-1b-v2 model via NeMo ASRModel.

    Uses the ASRModel API with source_lang/target_lang parameters.
    Supports 25 European languages including German.
    """

    def __init__(
        self,
        model_name: str = "nvidia/canary-1b-v2",
        language: str = "deu_Latn",
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None

        # Map language code to Canary format
        self._canary_lang = LANGUAGE_MAP.get(language, language[:2].lower())
        logger.info(f"Using Canary language code: {self._canary_lang}")

    def _get_model(self):
        """Lazy-load the NeMo ASRModel."""
        if self._model is None:
            logger.info(f"Loading Canary-1b model: {self.model_name}")
            try:
                from nemo.collections.asr.models import ASRModel

                self._model = ASRModel.from_pretrained(model_name=self.model_name)
                self._model.eval()

                import torch
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    logger.info("Canary-1b model loaded on CUDA")
                else:
                    logger.info("Canary-1b model loaded on CPU")

            except ImportError as e:
                raise ImportError(
                    "Canary-1b support requires NeMo toolkit. "
                    "Install with: pip install nemo_toolkit[asr]"
                ) from e
        return self._model

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Canary-1b-v2."""
        model = self._get_model()

        results = []
        for audio_path in audio_paths:
            try:
                # Canary-1b uses transcribe() with source_lang and target_lang
                output = model.transcribe(
                    [audio_path],
                    source_lang=self._canary_lang,
                    target_lang=self._canary_lang,
                )

                # Output is a list of objects with .text attribute
                transcript = output[0].text if hasattr(output[0], 'text') else str(output[0])
                results.append(transcript.strip())

            except Exception as e:
                logger.warning(f"Error transcribing {audio_path}: {e}")
                results.append("")

        return results
