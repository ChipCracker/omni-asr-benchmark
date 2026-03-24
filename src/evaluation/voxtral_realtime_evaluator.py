"""Voxtral Realtime evaluators using HuggingFace Transformers.

Provides two evaluator classes:
- VoxtralRealtimeEvaluator: Offline mode (full audio at once, supports batching)
- VoxtralRealtimeOnlineEvaluator: Online/streaming mode (chunked audio via generator)
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

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
                max_new_tokens=4096,
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
                        max_new_tokens=4096,
                    )

                    decoded = self._processor.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                    results.append(decoded[0].strip() if decoded else "")
                except Exception as inner_e:
                    logger.warning(f"Error transcribing {audio_path}: {inner_e}")
                    results.append("")
            return results


class VoxtralRealtimeOnlineEvaluator(BaseEvaluator):
    """Evaluator for Voxtral Realtime in online/streaming mode.

    Simulates real-time streaming by splitting audio into chunks and feeding
    them via a generator to model.generate(). This evaluates the model's
    streaming transcription quality (which may differ from offline mode).
    """

    # Suffix appended to model name for CLI selection
    _MODEL_SUFFIX = "-online"

    def __init__(
        self,
        model_name: str = "mistralai/Voxtral-Mini-4B-Realtime-2602-online",
        language: str = "deu_Latn",
        batch_size: int = 1,
    ) -> None:
        # Strip the -online suffix to get the actual HF model name
        hf_model_name = model_name.removesuffix(self._MODEL_SUFFIX)
        # Streaming mode is per-audio, no batching possible
        super().__init__(hf_model_name, language, batch_size=1)
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load the Voxtral Realtime model and processor."""
        if self._model is None:
            logger.info(f"Loading Voxtral Realtime model (online mode): {self.model_name}")
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
                logger.info(f"Voxtral Realtime model loaded on {self._device} (online mode)")

            except ImportError as e:
                raise ImportError(
                    "Voxtral Realtime support requires transformers>=5.2.0. "
                    "Install with: pip install --upgrade 'transformers>=5.2.0'. "
                    f"Original error: {e}"
                ) from e

    def _proc_attr(self, name):
        """Resolve processor attribute, calling it if it's a method."""
        val = getattr(self._processor, name)
        return val() if callable(val) else val

    def _make_chunk_generator(self, audio: np.ndarray, first_inputs):
        """Create a generator that yields input_features for each audio chunk."""
        hop_length = self._processor.feature_extractor.hop_length
        win_length = self._processor.feature_extractor.win_length

        # First chunk features
        yield first_inputs.input_features

        # Subsequent chunks
        mel_frame_idx = self._proc_attr("num_mel_frames_first_audio_chunk")
        samples_per_chunk = self._proc_attr("num_samples_per_audio_chunk")
        audio_len_per_tok = self._proc_attr("audio_length_per_tok")
        start_idx = mel_frame_idx * hop_length - win_length // 2

        while start_idx + samples_per_chunk < audio.shape[0]:
            end_idx = start_idx + samples_per_chunk
            chunk_inputs = self._processor(
                audio[start_idx:end_idx],
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            chunk_inputs = chunk_inputs.to(self._device, dtype=self._model.dtype)
            yield chunk_inputs.input_features

            mel_frame_idx += audio_len_per_tok
            start_idx = mel_frame_idx * hop_length - win_length // 2

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using Voxtral Realtime in online/streaming mode."""
        import soundfile as sf

        self._load_model()

        results = []
        for audio_path in audio_paths:
            try:
                audio, _ = sf.read(audio_path, dtype="float32")
                if getattr(audio, "ndim", 1) > 1:
                    audio = audio.mean(axis=1)

                # Pad audio for right padding tokens required by the model
                right_pad = self._proc_attr("num_right_pad_tokens") * self._proc_attr("raw_audio_length_per_tok")
                audio = np.pad(audio, (0, right_pad))

                # Process first chunk
                first_chunk_size = self._proc_attr("num_samples_first_audio_chunk")
                first_chunk = audio[:first_chunk_size]
                first_inputs = self._processor(
                    first_chunk,
                    is_streaming=True,
                    is_first_audio_chunk=True,
                    return_tensors="pt",
                )
                first_inputs = first_inputs.to(self._device, dtype=self._model.dtype)

                # Generate with chunk generator
                outputs = self._model.generate(
                    input_ids=first_inputs.input_ids,
                    input_features=self._make_chunk_generator(audio, first_inputs),
                    num_delay_tokens=first_inputs.num_delay_tokens,
                )

                decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
                results.append(decoded[0].strip() if decoded else "")

            except Exception as e:
                logger.warning(f"Error transcribing {audio_path} (online mode): {e}")
                results.append("")

        return results
