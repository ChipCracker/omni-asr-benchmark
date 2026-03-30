"""Cohere Transcribe Evaluator using HuggingFace Transformers."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Language code mapping (Cohere Transcribe supports 14 languages)
LANGUAGE_MAP = {
    "deu_Latn": "de",
    "eng_Latn": "en",
    "fra_Latn": "fr",
    "ita_Latn": "it",
    "spa_Latn": "es",
    "por_Latn": "pt",
    "ell_Grek": "el",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    "zho_Hans": "zh",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "vie_Latn": "vi",
    "ara_Arab": "ar",
}


class CohereTranscribeEvaluator(BaseEvaluator):
    """Evaluator for Cohere Transcribe models via HuggingFace Transformers.

    Uses CohereAsrForConditionalGeneration + AutoProcessor for speech-to-text.
    Supports 14 languages with native batching and long-form audio handling.
    """

    def __init__(
        self,
        model_name: str = "CohereLabs/cohere-transcribe-03-2026",
        language: str = "deu_Latn",
        batch_size: int = 4,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._processor = None

        self._cohere_lang = LANGUAGE_MAP.get(language, "de")
        if language not in LANGUAGE_MAP:
            logger.warning(
                f"Language {language} not in supported list. "
                f"Using 'de'. Supported: {list(LANGUAGE_MAP.keys())}"
            )

    def _load_model(self):
        """Lazy-load the Cohere Transcribe model and processor."""
        if self._model is not None:
            return

        logger.info(f"Loading Cohere Transcribe model: {self.model_name}")
        try:
            import torch
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration

            has_cuda = torch.cuda.is_available()
            dtype = torch.bfloat16 if has_cuda else torch.float32

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = CohereAsrForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if has_cuda else None,
            )

            if not has_cuda:
                self._model = self._model.to("cpu")

            self._model.eval()
            logger.info(
                f"Cohere Transcribe loaded on {self._model.device} with {dtype}"
            )

        except ImportError as e:
            raise ImportError(
                "Cohere Transcribe support requires transformers>=5.4.0. "
                "Install with: pip install -U 'transformers>=5.4.0' "
                "sentencepiece protobuf. "
                f"Original error: {e}"
            ) from e

    def _load_audio(self, audio_path: str):
        """Load audio file at 16kHz."""
        try:
            from transformers.audio_utils import load_audio

            return load_audio(audio_path, sampling_rate=16000)
        except ImportError:
            import soundfile as sf

            audio, sr = sf.read(audio_path, dtype="float32")
            if sr != 16000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            return audio

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Cohere Transcribe."""
        import torch

        self._load_model()

        audios = []
        for path in audio_paths:
            try:
                audios.append(self._load_audio(path))
            except Exception as e:
                logger.warning(f"Error loading audio {path}: {e}")
                audios.append(None)

        # Filter out failed loads and track indices
        valid_indices = [i for i, a in enumerate(audios) if a is not None]
        valid_audios = [audios[i] for i in valid_indices]

        if not valid_audios:
            return [""] * len(audio_paths)

        # Try batch processing
        try:
            inputs = self._processor(
                valid_audios,
                sampling_rate=16000,
                return_tensors="pt",
                language=self._cohere_lang,
            )
            audio_chunk_index = inputs.pop("audio_chunk_index", None)
            inputs = inputs.to(self._model.device, dtype=self._model.dtype)

            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_new_tokens=256)

            texts = self._processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=self._cohere_lang,
            )

            if isinstance(texts, str):
                texts = [texts]

            # Map back to original indices
            results = [""] * len(audio_paths)
            for idx, text in zip(valid_indices, texts):
                results[idx] = text.strip()
            return results

        except Exception as e:
            logger.warning(
                f"Batch processing failed ({e}), falling back to per-file processing"
            )

        # Per-file fallback
        results = [""] * len(audio_paths)
        for idx in valid_indices:
            try:
                inputs = self._processor(
                    audios[idx],
                    sampling_rate=16000,
                    return_tensors="pt",
                    language=self._cohere_lang,
                )
                audio_chunk_index = inputs.pop("audio_chunk_index", None)
                inputs = inputs.to(self._model.device, dtype=self._model.dtype)

                with torch.no_grad():
                    outputs = self._model.generate(**inputs, max_new_tokens=256)

                text = self._processor.decode(
                    outputs,
                    skip_special_tokens=True,
                    audio_chunk_index=audio_chunk_index,
                    language=self._cohere_lang,
                )

                if isinstance(text, list):
                    text = text[0]

                results[idx] = text.strip()

            except Exception as e:
                logger.warning(f"Error transcribing {audio_paths[idx]}: {e}")

        return results
