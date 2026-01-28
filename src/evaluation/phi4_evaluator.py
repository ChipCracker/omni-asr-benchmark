"""Phi-4 Multimodal Evaluator using HuggingFace Transformers."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Language code mapping
LANGUAGE_MAP = {
    "deu_Latn": "German",
    "eng_Latn": "English",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "por_Latn": "Portuguese",
    "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese",
    "zho_Hans": "Chinese",
}


class Phi4Evaluator(BaseEvaluator):
    """Evaluator for Microsoft Phi-4 Multimodal models via HuggingFace Transformers.

    Supports transcription in English, Chinese, German, French, Italian,
    Japanese, Spanish, and Portuguese. Recommended max 40 seconds per audio.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-multimodal-instruct",
        language: str = "deu_Latn",
        batch_size: int = 1,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._processor = None
        self._generation_config = None

        # Map language code
        self._lang_name = LANGUAGE_MAP.get(language, "English")
        if language not in LANGUAGE_MAP:
            logger.warning(
                f"Language {language} not in supported list. "
                f"Using 'English'. Supported: {list(LANGUAGE_MAP.keys())}"
            )

    def _load_model(self):
        """Lazy-load the Phi-4 model and processor."""
        if self._model is None:
            logger.info(f"Loading Phi-4 model: {self.model_name}")
            try:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoProcessor,
                    GenerationConfig,
                )

                self._processor = AutoProcessor.from_pretrained(
                    self.model_name, trust_remote_code=True
                )

                # Try flash_attention_2 first, fall back to eager for older GPUs
                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map=device,
                        torch_dtype="auto",
                        trust_remote_code=True,
                        _attn_implementation="flash_attention_2",
                    )
                    logger.info("Phi-4 model loaded with flash_attention_2")
                except Exception as e:
                    logger.warning(f"Flash attention not available: {e}")
                    logger.info("Falling back to eager attention")
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map=device,
                        torch_dtype="auto",
                        trust_remote_code=True,
                        _attn_implementation="eager",
                    )
                    logger.info("Phi-4 model loaded with eager attention")

                if device == "cuda":
                    self._model = self._model.cuda()

                self._generation_config = GenerationConfig.from_pretrained(
                    self.model_name
                )
                self._device = device
                logger.info(f"Phi-4 model loaded on {device}")

            except ImportError as e:
                raise ImportError(
                    f"Phi-4 support requires transformers>=4.48.2 and soundfile. "
                    f"Install with: pip install -U transformers soundfile. "
                    f"Original error: {e}"
                ) from e

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Phi-4 Multimodal."""
        import soundfile as sf

        self._load_model()

        results = []
        for audio_path in audio_paths:
            try:
                # Load audio file
                audio, samplerate = sf.read(audio_path)

                # Build prompt for transcription
                user_prompt = "<|user|>"
                assistant_prompt = "<|assistant|>"
                prompt_suffix = "<|end|>"
                speech_prompt = f"Transcribe the audio to text in {self._lang_name}."
                prompt = f"{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}"

                # Process audio
                inputs = self._processor(
                    text=prompt,
                    audios=[(audio, samplerate)],
                    return_tensors="pt",
                ).to(self._device)

                # Generate transcription
                generate_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    generation_config=self._generation_config,
                )

                # Decode output, skipping input tokens
                generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
                response = self._processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                results.append(response.strip())

            except Exception as e:
                import traceback
                logger.warning(f"Error transcribing {audio_path}: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                results.append("")

        return results
