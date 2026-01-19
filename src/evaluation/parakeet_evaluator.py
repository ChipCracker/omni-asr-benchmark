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

        # Warn if using non-English language with Parakeet
        if not language.startswith("eng"):
            logger.warning(
                f"Parakeet models are primarily trained on English. "
                f"Performance may be degraded for language: {language}"
            )

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

                # Move to GPU if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._model = self._model.cuda()
                        logger.info("Parakeet model loaded on CUDA")
                        # Warmup to initialize CUDA context
                        self._warmup()
                    else:
                        logger.info("Parakeet model loaded on CPU")
                except Exception:
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
                logger.warning(f"Parakeet warmup failed (will retry on first batch): {e}")

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe a batch of audio files using Parakeet.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of transcription strings.
        """
        model = self._get_model()

        try:
            # NeMo's transcribe method handles batching internally
            transcriptions = model.transcribe(
                audio_paths,
                batch_size=self.batch_size,
            )

            # Handle different return types from NeMo models
            if isinstance(transcriptions, tuple):
                # Some models return (text, logprobs) tuple
                transcriptions = transcriptions[0]

            # Ensure we return strings, not nested structures
            results = []
            for t in transcriptions:
                if isinstance(t, str):
                    results.append(t.strip())
                elif hasattr(t, "text"):
                    results.append(t.text.strip())
                else:
                    results.append(str(t).strip())

            return results

        except Exception as e:
            logger.error(f"Error transcribing batch: {e}")
            return [""] * len(audio_paths)
