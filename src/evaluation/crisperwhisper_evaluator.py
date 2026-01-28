"""CrisperWhisper Evaluator using nyrahealth's custom transformers fork."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CrisperWhisperEvaluator(BaseEvaluator):
    """Evaluator for CrisperWhisper models via nyrahealth's transformers fork.

    CrisperWhisper is a fine-tuned Whisper Large V3 model that provides:
    - Verbatim transcription (including filler words like "um", "uh")
    - Improved word-level timestamps
    - ~1% better WER than base Whisper Large V3
    """

    # Mapping from omnilingual language codes to Whisper language names
    LANGUAGE_MAP = {
        "deu_Latn": "german",
        "eng_Latn": "english",
        "fra_Latn": "french",
        "spa_Latn": "spanish",
        "ita_Latn": "italian",
        "por_Latn": "portuguese",
        "nld_Latn": "dutch",
        "pol_Latn": "polish",
        "rus_Cyrl": "russian",
        "jpn_Jpan": "japanese",
        "zho_Hans": "chinese",
        "kor_Hang": "korean",
        "ara_Arab": "arabic",
        "hin_Deva": "hindi",
        "tur_Latn": "turkish",
        "vie_Latn": "vietnamese",
        "tha_Thai": "thai",
        "heb_Hebr": "hebrew",
        "ukr_Cyrl": "ukrainian",
        "ces_Latn": "czech",
        "swe_Latn": "swedish",
        "dan_Latn": "danish",
        "fin_Latn": "finnish",
        "nor_Latn": "norwegian",
        "hun_Latn": "hungarian",
        "ron_Latn": "romanian",
        "bul_Cyrl": "bulgarian",
        "hrv_Latn": "croatian",
        "slk_Latn": "slovak",
        "slv_Latn": "slovenian",
        "ell_Grek": "greek",
        "cat_Latn": "catalan",
        "eus_Latn": "basque",
        "glg_Latn": "galician",
    }

    def __init__(
        self,
        model_name: str = "nyrahealth/CrisperWhisper",
        language: str = "deu_Latn",
        batch_size: int = 4,
    ) -> None:
        """Initialize the CrisperWhisper evaluator.

        Args:
            model_name: HuggingFace model ID (e.g., "nyrahealth/CrisperWhisper").
            language: Language code for transcription (e.g., "deu_Latn").
            batch_size: Batch size for inference.
        """
        super().__init__(model_name, language, batch_size)
        self._pipeline = None
        self._whisper_language = self.LANGUAGE_MAP.get(language, "german")

    def _get_pipeline(self):
        """Lazy-load the CrisperWhisper ASR pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading CrisperWhisper pipeline: {self.model_name}")
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                # Load model explicitly (required for CrisperWhisper)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                )
                model.to(device)

                # Load processor for tokenizer and feature extractor
                processor = AutoProcessor.from_pretrained(self.model_name)

                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    chunk_length_s=30,
                    batch_size=1,  # Process one file at a time (official recommendation)
                    return_timestamps=True,
                    torch_dtype=torch_dtype,
                    device=device,
                )
                logger.info(f"CrisperWhisper pipeline loaded on {device}")
            except ImportError as e:
                raise ImportError(
                    "CrisperWhisper support requires nyrahealth's transformers fork. "
                    "Install with: pip install git+https://github.com/nyrahealth/transformers.git@crisper_whisper"
                ) from e
        return self._pipeline

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe a batch of audio files using CrisperWhisper.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of transcription strings.
        """
        pipe = self._get_pipeline()

        generate_kwargs = {
            "language": self._whisper_language,
            "task": "transcribe",
        }

        # Process files individually (batch_size=1 to avoid tensor mismatches)
        results = []
        for audio_path in audio_paths:
            output = pipe(audio_path, generate_kwargs=generate_kwargs)
            text = output.get("text", "").strip()
            results.append(text)

        return results
