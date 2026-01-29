"""Qwen3-ASR Evaluator using qwen-asr library."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Language code mapping from Glottolog to Qwen3-ASR format
LANGUAGE_MAP = {
    "deu_Latn": "German",
    "eng_Latn": "English",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "nld_Latn": "Dutch",
    "pol_Latn": "Polish",
    "swe_Latn": "Swedish",
    "dan_Latn": "Danish",
    "nor_Latn": "Norwegian",
    "fin_Latn": "Finnish",
    "ron_Latn": "Romanian",
    "hun_Latn": "Hungarian",
    "ces_Latn": "Czech",
    "slk_Latn": "Slovak",
    "bul_Cyrl": "Bulgarian",
    "ell_Grek": "Greek",
    "tur_Latn": "Turkish",
    "ukr_Cyrl": "Ukrainian",
    "rus_Cyrl": "Russian",
    "hrv_Latn": "Croatian",
    "lit_Latn": "Lithuanian",
    "slv_Latn": "Slovenian",
    "cat_Latn": "Catalan",
    "zho_Hans": "Chinese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "ara_Arab": "Arabic",
    "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "ind_Latn": "Indonesian",
}


class Qwen3ASREvaluator(BaseEvaluator):
    """Evaluator for Qwen3-ASR models (0.6B and 1.7B) via qwen-asr library.

    Uses the Qwen3ASRModel API for transcription.
    Supports 30+ languages with automatic language detection.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        language: str = "deu_Latn",
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None

        # Map language code to Qwen3-ASR format
        self._qwen_lang = LANGUAGE_MAP.get(language)
        if self._qwen_lang:
            logger.info(f"Using Qwen3-ASR language: {self._qwen_lang}")
        else:
            logger.info("Using Qwen3-ASR auto language detection")

    def _get_model(self):
        """Lazy-load the Qwen3-ASR model."""
        if self._model is None:
            logger.info(f"Loading Qwen3-ASR model: {self.model_name}")
            try:
                import torch
                from qwen_asr import Qwen3ASRModel

                # Determine device and dtype
                if torch.cuda.is_available():
                    device_map = "cuda:0"
                    dtype = torch.bfloat16
                    logger.info("Loading Qwen3-ASR model on CUDA with bfloat16")
                else:
                    device_map = "cpu"
                    dtype = torch.float32
                    logger.info("Loading Qwen3-ASR model on CPU with float32")

                self._model = Qwen3ASRModel.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    device_map=device_map,
                    max_inference_batch_size=self.batch_size,
                    max_new_tokens=256,
                )

            except ImportError as e:
                raise ImportError(
                    "Qwen3-ASR support requires the qwen-asr library. "
                    "Install with: pip install qwen-asr"
                ) from e
        return self._model

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Qwen3-ASR."""
        model = self._get_model()

        try:
            # Qwen3-ASR supports batch transcription natively
            results = model.transcribe(
                audio=audio_paths,
                language=self._qwen_lang,  # None for auto-detection
            )

            # Output is a list of objects with .text attribute
            return [r.text.strip() if hasattr(r, 'text') else str(r).strip() for r in results]

        except Exception as e:
            logger.warning(f"Error transcribing batch: {e}")
            # Fall back to individual transcription
            transcripts = []
            for audio_path in audio_paths:
                try:
                    result = model.transcribe(
                        audio=[audio_path],
                        language=self._qwen_lang,
                    )
                    text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                    transcripts.append(text.strip())
                except Exception as inner_e:
                    logger.warning(f"Error transcribing {audio_path}: {inner_e}")
                    transcripts.append("")
            return transcripts
