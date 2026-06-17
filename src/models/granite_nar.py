"""IBM Granite Speech NAR (non-autoregressive) evaluator.

``ibm-granite/granite-speech-4.1-2b-nar`` edits a CTC hypothesis in a single
forward pass with a bidirectional LLM. It uses a custom architecture and differs
from the autoregressive Granite Speech model:

- loaded with ``AutoModel`` + ``trust_remote_code=True`` (not AutoModelForSpeechSeq2Seq)
- inference via ``model.transcribe(**processor([waveforms], device=...))`` and
  ``processor.batch_decode(output.preds)`` — no ``generate()``, no chat prompt
- requires ``transformers>=5.5.3``, ``torch>=2.9`` and ``flash-attn`` (the model
  defaults to ``flash_attention_2``)

Because decoding is non-autoregressive (single pass over the CTC hypothesis),
there is no max-token cap and long-form utterances are not truncated.
"""

from __future__ import annotations

import logging
from typing import List

from .base import AsrModel

logger = logging.getLogger(__name__)


class GraniteNarEvaluator(AsrModel):
    """Evaluator for the non-autoregressive Granite Speech model."""

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-speech-4.1-2b-nar",
        language: str = "deu_Latn",
        batch_size: int = 1,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._attn = attn_implementation

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "Granite NAR needs transformers>=5.5.3 and torch>=2.9. "
                f"Original error: {e}"
            ) from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Prefer flash_attention_2 (model default) but fall back to sdpa/eager
        # when flash-attn is not installed — avoids a fragile source build.
        attn_candidates = []
        for attn in (self._attn, "sdpa", "eager"):
            if attn and attn not in attn_candidates:
                attn_candidates.append(attn)

        last_err = None
        for attn in attn_candidates:
            try:
                logger.info("Loading Granite NAR %s (attn=%s)", self.model_name, attn)
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    attn_implementation=attn,
                    device_map=device,
                    dtype=dtype,
                ).eval()
                self._attn = attn
                break
            except (ImportError, ValueError) as e:
                logger.warning("attn_implementation=%s unavailable (%s); trying next", attn, e)
                last_err = e

        if self._model is None:
            raise ImportError(
                f"Could not load Granite NAR with any attention implementation. Last error: {last_err}"
            )

        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        logger.info("Granite NAR model loaded on %s with %s (attn=%s)", device, dtype, self._attn)

    def _transcribe(self, waveforms: List["torch.Tensor"]) -> List[str]:  # noqa: F821
        import torch

        inputs = self._processor(waveforms, device=self._device)
        with torch.no_grad():
            output = self._model.transcribe(**inputs)
        return [t.strip() for t in self._processor.batch_decode(output.preds)]

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        import soundfile as sf
        import torch

        self._load_model()

        # Load + resample to 16 kHz mono. Use soundfile rather than
        # torchaudio.load (torchaudio>=2.9 routes load() through torchcodec,
        # which needs FFmpeg); soundfile reads these WAVs directly.
        waveforms = []
        for path in audio_paths:
            data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
            waveform = torch.from_numpy(data.T)  # (channels, frames)
            if sr != 16000:
                import torchaudio
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveforms.append(waveform.squeeze(0))

        # Try a single batched forward; fall back to per-utterance on error
        # (e.g. padding issues mixing very long and short clips).
        try:
            return self._transcribe(waveforms)
        except Exception as e:  # noqa: BLE001
            import traceback
            logger.warning("Batched NAR transcribe failed (%s); per-utterance fallback", e)
            logger.warning("Traceback: %s", traceback.format_exc())

        results = []
        for waveform in waveforms:
            try:
                results.append(self._transcribe([waveform])[0])
            except Exception as e:  # noqa: BLE001
                logger.warning("NAR transcribe failed for one utterance: %s", e)
                results.append("")
        return results
