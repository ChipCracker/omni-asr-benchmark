"""Parakeet Evaluator using NVIDIA NeMo."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class ParakeetEvaluator(BaseEvaluator):
    """Evaluator for NVIDIA Parakeet models via NeMo.

    Supports Parakeet CTC and TDT (Transducer) models from NVIDIA.
    Models are available on HuggingFace: nvidia/parakeet-ctc-1.1b, nvidia/parakeet-tdt-1.1b
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-ctc-1.1b",
        language: str = "deu_Latn",
        batch_size: int = 4,
    ) -> None:
        """Initialize the Parakeet evaluator.

        Args:
            model_name: HuggingFace model ID (e.g., "nvidia/parakeet-ctc-1.1b").
            language: Language code (note: Parakeet models are primarily English-focused).
            batch_size: Batch size for inference.
        """
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._device = "cpu"
        self._cuda_graphs_disabled = False

        # Warn if using non-English language with Parakeet
        if not language.startswith("eng"):
            logger.warning(
                f"Parakeet models are primarily trained on English. "
                f"Performance may be degraded for language: {language}"
            )

    def _set_cfg_value(self, cfg, key: str, value) -> bool:
        """Set a config key if it exists."""
        if cfg is None:
            return False
        try:
            if isinstance(cfg, dict):
                if key in cfg:
                    cfg[key] = value
                    return True
                return False
        except Exception:
            pass
        try:
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                return True
        except Exception:
            pass
        try:
            if key in cfg:
                cfg[key] = value
                return True
        except Exception:
            pass
        return False

    def _disable_cuda_graphs(self) -> bool:
        """Disable CUDA graphs in decoding config when available."""
        model = self._model
        if model is None:
            return False
        try:
            decoding_cfg = model.cfg.decoding
        except Exception:
            return False

        changed = False
        for key in ("use_cuda_graphs", "cuda_graphs", "cuda_graph"):
            changed |= self._set_cfg_value(decoding_cfg, key, False)

        for section_name in ("greedy", "beam", "tdt", "rnnt"):
            section = None
            try:
                section = getattr(decoding_cfg, section_name)
            except Exception:
                section = None
            if section is None:
                try:
                    section = decoding_cfg.get(section_name)
                except Exception:
                    section = None
            if section is not None:
                for key in ("use_cuda_graphs", "cuda_graphs", "cuda_graph"):
                    changed |= self._set_cfg_value(section, key, False)

        if changed and hasattr(model, "change_decoding_strategy"):
            try:
                model.change_decoding_strategy(decoding_cfg)
            except Exception as e:
                logger.debug(f"Failed to apply decoding config changes: {e}")

        if changed:
            self._cuda_graphs_disabled = True
        return changed

    def _is_cuda_failure(self, error: Exception) -> bool:
        text = str(error).lower()
        return (
            "cuda failure" in text
            or "cuda error" in text
            or "cuda graph" in text
            or "cudagraph" in text
        )

    def _maybe_recover_from_cuda_failure(self, error: Exception) -> bool:
        if not self._is_cuda_failure(error):
            return False

        if not self._cuda_graphs_disabled and self._disable_cuda_graphs():
            logger.warning(f"Disabled CUDA graphs after failure: {error}")
            return True

        if self._device == "cuda":
            try:
                import torch
                self._model = self._model.cpu()
                self._device = "cpu"
                torch.cuda.empty_cache()
                logger.warning(f"Falling back to CPU after CUDA failure: {error}")
                return True
            except Exception:
                return False

        return False

    def _get_model(self):
        """Lazy-load the NeMo ASR model."""
        if self._model is None:
            logger.info(f"Loading Parakeet model: {self.model_name}")
            try:
                import nemo.collections.asr as nemo_asr

                # Load model from HuggingFace
                self._model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name
                )
                self._model.eval()

                # Move to GPU if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._model = self._model.cuda()
                        self._device = "cuda"
                        if self._disable_cuda_graphs():
                            logger.info("Disabled CUDA graphs for Parakeet decoding")
                        logger.info("Parakeet model loaded on CUDA")
                        # Warmup to initialize CUDA context
                        self._warmup()
                    else:
                        self._device = "cpu"
                        logger.info("Parakeet model loaded on CPU")
                except Exception:
                    self._device = "cpu"
                    logger.info("Parakeet model loaded on CPU")

            except ImportError as e:
                raise ImportError(
                    "Parakeet support requires NVIDIA NeMo. "
                    "Install with: pip install nemo-toolkit[asr]"
                ) from e
        return self._model

    def _warmup(self) -> None:
        """Run a dummy inference to warm up CUDA context."""
        import tempfile
        import numpy as np
        import soundfile as sf

        # Create a short silent audio file for warmup
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            # 0.5 seconds of silence at 16kHz
            silence = np.zeros(8000, dtype=np.float32)
            sf.write(f.name, silence, 16000)

            try:
                self._model.transcribe([f.name], batch_size=1)
                logger.info("Parakeet warmup completed")
            except Exception as e:
                if self._maybe_recover_from_cuda_failure(e):
                    try:
                        self._model.transcribe([f.name], batch_size=1)
                        logger.info("Parakeet warmup completed after recovery")
                        return
                    except Exception as retry_error:
                        logger.warning(
                            f"Parakeet warmup failed after recovery (will retry on first batch): {retry_error}"
                        )
                        return
                logger.warning(f"Parakeet warmup failed (will retry on first batch): {e}")

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe a batch of audio files using Parakeet."""
        model = self._get_model()

        try:
            # Log audio file info for first file in batch to diagnose format issues
            if audio_paths:
                import soundfile as sf
                info = sf.info(audio_paths[0])
                logger.info(
                    f"Audio file info: {audio_paths[0]} - "
                    f"sample_rate={info.samplerate}Hz, channels={info.channels}, "
                    f"duration={info.duration:.2f}s, format={info.format}"
                )
                if info.samplerate != 16000:
                    logger.warning(
                        f"Audio sample rate is {info.samplerate}Hz, but Parakeet expects 16000Hz. "
                        "This may cause empty transcriptions."
                    )
                if info.channels != 1:
                    logger.warning(
                        f"Audio has {info.channels} channels, but Parakeet expects mono. "
                        "This may cause issues."
                    )

            try:
                transcriptions = model.transcribe(
                    audio_paths,
                    batch_size=self.batch_size,
                )
            except Exception as e:
                if self._maybe_recover_from_cuda_failure(e):
                    model = self._model
                    transcriptions = model.transcribe(
                        audio_paths,
                        batch_size=self.batch_size,
                    )
                else:
                    raise

            # INFO-level logging to see what the model returns
            logger.info(f"Transcribe returned type: {type(transcriptions)}")
            if transcriptions:
                if isinstance(transcriptions, (list, tuple)) and len(transcriptions) > 0:
                    first = transcriptions[0]
                    logger.info(f"First element type: {type(first)}")
                    logger.info(f"First element repr: {repr(first)[:200]}")
                    if hasattr(first, '__dict__'):
                        logger.info(f"First element attrs: {list(first.__dict__.keys())}")
                    if hasattr(first, 'text'):
                        logger.info(f"First element .text: '{first.text}'")

            # Handle tuple return type from RNNT models
            if isinstance(transcriptions, tuple) and len(transcriptions) >= 1:
                logger.info(f"Extracting from tuple of length {len(transcriptions)}")
                transcriptions = transcriptions[0]

            # Extract text from each transcription
            results = []
            for i, t in enumerate(transcriptions):
                if isinstance(t, str):
                    text = t.strip()
                elif hasattr(t, "text"):
                    text = t.text.strip() if t.text else ""
                else:
                    text = str(t).strip()

                if i == 0:
                    logger.info(f"Extracted text: '{text}' (from {type(t).__name__})")
                results.append(text)

            return results

        except Exception as e:
            logger.error(f"Error transcribing batch: {e}", exc_info=True)
            return [""] * len(audio_paths)
