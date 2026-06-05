"""Gemma 4 Multimodal Evaluator using HuggingFace Transformers with VAD chunking.

The Gemma 4 multimodal models (e.g. ``google/gemma-4-12B-it``) accept text, image
and audio input but support at most ~30 seconds of audio per inference. Longer
benchmark recordings are therefore split with Silero VAD: speech segments are
greedily grouped into chunks of <= ~28s, each chunk is transcribed individually,
and the partial transcripts are concatenated.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Language code mapping (shared convention with phi4_evaluator.py)
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

TARGET_SR = 16000
MAX_CHUNK_SEC = 28.0  # safety margin below the model's 30s audio limit
MIN_CHUNK_SEC = 1.0


class GemmaEvaluator(BaseEvaluator):
    """Evaluator for Google Gemma 4 multimodal models via HuggingFace Transformers.

    Audio longer than ~28s is segmented with Silero VAD and transcribed in chunks.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-12B-it",
        language: str = "deu_Latn",
        batch_size: int = 1,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._processor = None
        self._device = None
        self._vad_model = None
        self._get_speech_timestamps = None

        self._lang_name = LANGUAGE_MAP.get(language, "English")
        if language not in LANGUAGE_MAP:
            logger.warning(
                f"Language {language} not in supported list. "
                f"Using 'English'. Supported: {list(LANGUAGE_MAP.keys())}"
            )

    def _load_model(self):
        """Lazy-load the Gemma model and processor."""
        if self._model is not None:
            return
        logger.info(f"Loading Gemma model: {self.model_name}")
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForMultimodalLM

            self._processor = AutoProcessor.from_pretrained(self.model_name)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto" if device == "cuda" else None,
            )
            if device == "cpu":
                self._model = self._model.to("cpu")
            self._model.eval()
            self._device = device
            logger.info(f"Gemma model loaded on {device}")
        except ImportError as e:
            raise ImportError(
                "Gemma support requires a recent transformers (with "
                "AutoModelForMultimodalLM / Gemma 4 support), torch and soundfile. "
                "Install with: pip install -U transformers torch soundfile. "
                f"Original error: {e}"
            ) from e

    def _load_vad(self):
        """Lazy-load Silero VAD."""
        if self._vad_model is not None:
            return
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps

            self._vad_model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            logger.info("Silero VAD loaded")
        except ImportError as e:
            raise ImportError(
                "VAD chunking requires silero-vad. Install with: pip install silero-vad. "
                f"Original error: {e}"
            ) from e

    def _vad_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        """Split audio (mono, 16kHz float32) into <= MAX_CHUNK_SEC chunks via VAD.

        Speech segments detected by Silero VAD are greedily grouped so that each
        returned chunk spans at most MAX_CHUNK_SEC seconds. A single speech segment
        longer than MAX_CHUNK_SEC is hard-split into fixed windows. If VAD finds no
        speech, the audio is split into fixed MAX_CHUNK_SEC windows.
        """
        import torch

        self._load_vad()

        max_samples = int(MAX_CHUNK_SEC * TARGET_SR)

        segments = self._get_speech_timestamps(
            torch.from_numpy(audio),
            self._vad_model,
            sampling_rate=TARGET_SR,
            return_seconds=False,
        )

        # Fallback: no speech detected -> fixed windows over the whole audio
        if not segments:
            return [
                audio[i:i + max_samples]
                for i in range(0, len(audio), max_samples)
            ]

        # Hard-split any segment longer than the max window
        normalized = []
        for seg in segments:
            start, end = int(seg["start"]), int(seg["end"])
            if end - start > max_samples:
                for s in range(start, end, max_samples):
                    normalized.append((s, min(s + max_samples, end)))
            else:
                normalized.append((start, end))

        # Greedy group consecutive segments up to max window length
        chunks: List[np.ndarray] = []
        group_start = normalized[0][0]
        group_end = normalized[0][1]
        for start, end in normalized[1:]:
            if end - group_start <= max_samples:
                group_end = end
            else:
                chunks.append(audio[group_start:group_end])
                group_start, group_end = start, end
        chunks.append(audio[group_start:group_end])

        return [c for c in chunks if len(c) > 0]

    def _transcribe_one(self, audio: np.ndarray) -> str:
        """Transcribe a single <=30s audio chunk (mono, 16kHz float32)."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {
                        "type": "text",
                        "text": f"Transcribe the audio to text in {self._lang_name}.",
                    },
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generate_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self._processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files, VAD-chunking anything longer than ~28s."""
        import soundfile as sf
        import torch

        self._load_model()

        max_samples = int(MAX_CHUNK_SEC * TARGET_SR)
        results: List[str] = []

        for audio_path in audio_paths:
            try:
                audio, sr = sf.read(audio_path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)  # to mono

                if sr != TARGET_SR:
                    audio = torchaudio_resample(audio, sr, TARGET_SR)

                if len(audio) <= max_samples:
                    results.append(self._transcribe_one(audio))
                else:
                    chunks = self._vad_chunks(audio)
                    logger.info(
                        f"Split {audio_path} ({len(audio) / TARGET_SR:.1f}s) "
                        f"into {len(chunks)} VAD chunks"
                    )
                    texts = [self._transcribe_one(c) for c in chunks]
                    results.append(" ".join(t for t in texts if t))

            except Exception as e:
                import traceback

                logger.warning(f"Error transcribing {audio_path}: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                results.append("")

        return results


def torchaudio_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample a 1-D float32 numpy array using torchaudio."""
    import torch
    import torchaudio

    waveform = torch.from_numpy(audio).unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, src_sr, dst_sr)
    return resampled.squeeze(0).numpy()
