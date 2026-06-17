"""IBM Granite Speech evaluator using HuggingFace Transformers.

Supports the Granite Speech 4.x family, e.g. ``ibm-granite/granite-4.0-1b-speech``
and ``ibm-granite/granite-speech-4.1-2b``. Both share the same API
(``AutoProcessor`` + ``AutoModelForSpeechSeq2Seq`` with an ``<|audio|>`` chat
prompt) and require ``transformers>=4.52.1`` plus ``peft`` (LoRA adapter).
"""

from __future__ import annotations

import logging
from typing import List

from .base import AsrModel

logger = logging.getLogger(__name__)

# Language code mapping (Granite supports 6 languages)
LANGUAGE_MAP = {
    "deu_Latn": "German",
    "eng_Latn": "English",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "por_Latn": "Portuguese",
    "jpn_Jpan": "Japanese",
}


class GraniteEvaluator(AsrModel):
    """Evaluator for IBM Granite Speech models via HuggingFace Transformers.

    Uses AutoModelForSpeechSeq2Seq + AutoProcessor for speech-to-text.
    Covers the Granite Speech 4.x family (e.g. granite-4.0-1b-speech,
    granite-speech-4.1-2b). Supports English, German, French, Spanish,
    Portuguese, and Japanese.
    """

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-speech-4.1-2b",
        language: str = "deu_Latn",
        batch_size: int = 1,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._processor = None

        self._lang_name = LANGUAGE_MAP.get(language, "English")
        if language not in LANGUAGE_MAP:
            logger.warning(
                f"Language {language} not in supported list. "
                f"Using 'English'. Supported: {list(LANGUAGE_MAP.keys())}"
            )

    def _load_model(self):
        """Lazy-load the Granite model and processor."""
        if self._model is not None:
            return

        logger.info(f"Loading Granite model: {self.model_name}")
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            ).to(device)

            self._device = device
            self._dtype = dtype
            logger.info(f"Granite model loaded on {device} with {dtype}")

        except ImportError as e:
            raise ImportError(
                f"Granite support requires transformers and torchaudio. "
                f"Install with: pip install -U transformers torchaudio. "
                f"Original error: {e}"
            ) from e

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Granite Speech."""
        import torch
        import torchaudio

        self._load_model()

        tokenizer = self._processor.tokenizer

        results = []
        for audio_path in audio_paths:
            try:
                # Load audio and ensure 16kHz mono
                waveform, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    sr = 16000
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Scale generation length to audio duration so long-form
                # utterances (RVG1 monologues run up to ~130s) are not truncated.
                duration_s = waveform.shape[-1] / sr
                max_new = min(1024, max(256, int(duration_s * 8)))

                # Build chat prompt with audio placeholder (official ASR prompt)
                chat = [
                    {
                        "role": "user",
                        "content": "<|audio|>transcribe the speech with proper punctuation and capitalization.",
                    }
                ]
                text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                # Process inputs
                model_inputs = self._processor(
                    text,
                    waveform.squeeze(0),
                    device=self._device,
                    return_tensors="pt",
                ).to(self._device)

                # Generate transcription
                with torch.no_grad():
                    output_ids = self._model.generate(
                        **model_inputs,
                        max_new_tokens=max_new,
                        do_sample=False,
                        num_beams=1,
                    )

                # Strip input tokens and decode only new tokens
                input_len = model_inputs["input_ids"].shape[1]
                new_tokens = output_ids[:, input_len:]
                response = tokenizer.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                )[0]

                results.append(response.strip())

            except Exception as e:
                import traceback
                logger.warning(f"Error transcribing {audio_path}: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                results.append("")

        return results
