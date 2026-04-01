"""NeMo STT Evaluator for standard NeMo speech-to-text models."""

from __future__ import annotations

import logging
from typing import List

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class NemoSTTEvaluator(BaseEvaluator):
    """Evaluator for standard NVIDIA NeMo STT models.

    Handles NeMo CTC/RNNT Conformer models like stt_de_conformer_ctc_large.
    Uses ASRModel.from_pretrained() + model.transcribe() API.
    """

    def __init__(
        self,
        model_name: str = "nvidia/stt_de_conformer_ctc_large",
        language: str = "deu_Latn",
        batch_size: int = 16,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self._model = None

    def _load_model(self):
        """Lazy-load the NeMo ASR model."""
        if self._model is not None:
            return

        logger.info(f"Loading NeMo STT model: {self.model_name}")
        try:
            import torch
            from nemo.collections.asr.models import ASRModel

            self._model = ASRModel.from_pretrained(model_name=self.model_name)
            self._model.eval()

            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info(f"NeMo STT model loaded on CUDA")
            else:
                logger.info(f"NeMo STT model loaded on CPU")

        except ImportError as e:
            raise ImportError(
                "NeMo STT support requires NVIDIA NeMo toolkit. "
                "Install with: pip install nemo_toolkit[asr]. "
                f"Original error: {e}"
            ) from e

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe audio files using NeMo STT model."""
        self._load_model()

        try:
            outputs = self._model.transcribe(
                audio_paths,
                batch_size=self.batch_size,
            )

            # Handle tuple return from RNNT models
            if isinstance(outputs, tuple) and len(outputs) >= 1:
                outputs = outputs[0]

            results = []
            for t in outputs:
                if isinstance(t, str):
                    results.append(t.strip())
                elif hasattr(t, "text"):
                    results.append(t.text.strip() if t.text else "")
                else:
                    results.append(str(t).strip())

            return results

        except Exception as e:
            logger.error(f"Error transcribing batch: {e}", exc_info=True)
            return [""] * len(audio_paths)
