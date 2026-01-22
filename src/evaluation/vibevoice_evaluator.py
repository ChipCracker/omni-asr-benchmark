"""VibeVoice Evaluator for Microsoft's VibeVoice-ASR model."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class VibeVoiceEvaluator(BaseEvaluator):
    """Evaluator for Microsoft VibeVoice-ASR model.

    VibeVoice-ASR is a 9B parameter unified speech-to-text model that:
    - Processes up to 60 minutes of continuous audio in a single pass
    - Generates structured transcriptions with speaker diarization and timestamps
    - Supports customized hotwords for domain-specific terms

    Note: Requires vibevoice package. Install from:
        git clone https://github.com/microsoft/VibeVoice.git
        cd VibeVoice
        pip install -e .[asr]
    """

    def __init__(
        self,
        model_name: str = "microsoft/VibeVoice-ASR",
        language: str = "deu_Latn",
        batch_size: int = 1,
    ) -> None:
        """Initialize the VibeVoice evaluator.

        Args:
            model_name: HuggingFace model ID (default: "microsoft/VibeVoice-ASR").
            language: Language code for transcription (e.g., "deu_Latn").
            batch_size: Batch size for inference (default 1 due to model size).
        """
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._processor = None

    def _get_model(self):
        """Lazy-load the VibeVoice model and processor."""
        if self._model is None:
            logger.info(f"Loading VibeVoice-ASR model: {self.model_name}")

            # Check for vibevoice package first
            try:
                from vibevoice.modular.modeling_vibevoice_asr import (
                    VibeVoiceASRForConditionalGeneration,
                )
                from vibevoice.processor.vibevoice_asr_processor import (
                    VibeVoiceASRProcessor,
                )
            except ImportError as e:
                raise ImportError(
                    "VibeVoice support requires the vibevoice package. "
                    "Install with:\n"
                    "  git clone https://github.com/microsoft/VibeVoice.git\n"
                    "  cd VibeVoice\n"
                    "  pip install -e .[asr]"
                ) from e

            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            # Load processor
            logger.info("Loading VibeVoice processor...")
            self._processor = VibeVoiceASRProcessor.from_pretrained(
                self.model_name,
                llm_model="Qwen/Qwen2.5-7B",
            )

            # Load model
            logger.info("Loading VibeVoice model...")
            attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
            try:
                self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load with {attn_impl}, trying eager attention: {e}"
                )
                self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    attn_implementation="eager",
                )
                attn_impl = "eager"

            self._model = self._model.to(device)
            self._model.eval()

            self._device = device
            self._dtype = dtype

            logger.info(
                f"VibeVoice-ASR loaded on {device} with dtype={dtype}, "
                f"attention={attn_impl}"
            )
        return self._model, self._processor

    def _extract_plain_text(self, raw_output: str) -> str:
        """Extract plain transcription text from VibeVoice structured output.

        VibeVoice outputs structured transcriptions with speaker tags and timestamps.
        This extracts just the spoken text for WER/CER evaluation.

        Args:
            raw_output: Raw model output with structure like:
                "[00:00.00 - 00:05.00] Speaker 1: Hello world"

        Returns:
            Plain text without timestamps or speaker tags.
        """
        if not raw_output:
            return ""

        lines = raw_output.strip().split("\n")
        text_parts = []

        for line in lines:
            # Remove timestamp patterns like [00:00.00 - 00:05.00]
            line = re.sub(r"\[\d{2}:\d{2}\.\d{2}\s*-\s*\d{2}:\d{2}\.\d{2}\]", "", line)
            # Remove speaker tags like "Speaker 1:" or "Speaker_1:"
            line = re.sub(r"Speaker[_\s]?\d+:\s*", "", line, flags=re.IGNORECASE)
            # Clean up extra whitespace
            line = line.strip()
            if line:
                text_parts.append(line)

        return " ".join(text_parts)

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe a batch of audio files using VibeVoice-ASR.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of transcription strings (plain text, without timestamps/speakers).
        """
        import torch

        model, processor = self._get_model()
        results = []

        # Process each audio file (batching handled by model internally)
        for audio_path in audio_paths:
            try:
                # Load audio
                import soundfile as sf

                audio_data, sample_rate = sf.read(audio_path)

                # Prepare inputs
                inputs = processor(
                    audio=audio_data,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                    add_generation_prompt=True,
                )

                # Move to device
                inputs = {
                    k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                # Generate transcription
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=32768,
                        temperature=0.0,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )

                # Decode output
                input_length = inputs["input_ids"].shape[1]
                generated_ids = output_ids[0, input_length:]
                raw_text = processor.decode(generated_ids, skip_special_tokens=True)

                # Extract plain text for evaluation
                plain_text = self._extract_plain_text(raw_text)
                results.append(plain_text)

                logger.debug(f"Transcribed {audio_path}: {plain_text[:100]}...")

            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                results.append("")

        return results
