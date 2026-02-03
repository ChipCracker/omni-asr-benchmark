"""Syllabic ASR Evaluator for CTC and RNNT decoding benchmarks."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class SyllabicASREvaluator(BaseEvaluator):
    """Evaluator for syllabic-asr models with CTC and RNNT decode modes.

    Uses the SyllabicASRPipeline from the syllabic-asr project.
    The pipeline is single-file only, so transcribe_batch loops sequentially.
    """

    def __init__(
        self,
        model_name: str,
        language: str,
        batch_size: int,
        decode_mode: str = "ctc",
        model_dir: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        if decode_mode not in ("ctc", "rnnt"):
            raise ValueError(f"decode_mode must be 'ctc' or 'rnnt', got '{decode_mode}'")
        self.decode_mode = decode_mode
        self._model_dir = model_dir
        self._pipeline = None

    def _resolve_model_dir(self) -> Path:
        """Resolve the model directory from CLI arg, env var, or raise."""
        if self._model_dir:
            p = Path(self._model_dir)
            if p.exists():
                return p
            raise ValueError(f"Provided model_dir does not exist: {p}")

        env_dir = os.getenv("SYLLABIC_ASR_MODEL_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists():
                return p
            raise ValueError(
                f"SYLLABIC_ASR_MODEL_DIR points to non-existent path: {p}"
            )

        raise ValueError(
            "No model directory specified. Use --model-dir or set "
            "SYLLABIC_ASR_MODEL_DIR environment variable."
        )

    def _resolve_syllabic_asr_root(self) -> Path:
        """Find the syllabic-asr project root for imports."""
        # 1. Explicit env var
        env_root = os.getenv("SYLLABIC_ASR_ROOT")
        if env_root:
            p = Path(env_root)
            if p.exists():
                return p

        # 2. Sibling directory heuristic (../syllabic-asr relative to omni-asr-test)
        omni_root = Path(__file__).resolve().parent.parent.parent
        sibling = omni_root.parent / "syllabic-asr"
        if sibling.exists():
            return sibling

        raise ImportError(
            "Cannot find syllabic-asr project. Set SYLLABIC_ASR_ROOT env var "
            "or place it as a sibling directory to omni-asr-test."
        )

    def _get_pipeline(self):
        """Lazy-load the SyllabicASRPipeline."""
        if self._pipeline is not None:
            return self._pipeline

        # Add syllabic-asr to sys.path for imports
        syllabic_root = self._resolve_syllabic_asr_root()
        root_str = str(syllabic_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            logger.info(f"Added syllabic-asr to sys.path: {root_str}")

        from utils.inference.pipeline import SyllabicASRPipeline

        model_dir = self._resolve_model_dir()

        # Device selection: cuda if available, else cpu (no MPS)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Loading SyllabicASRPipeline from {model_dir} "
            f"(device={device}, decode_mode={self.decode_mode})"
        )
        self._pipeline = SyllabicASRPipeline(model_dir=str(model_dir), device=device)
        return self._pipeline

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe a batch of audio files sequentially.

        For CTC mode, extracts result["text"].
        For RNNT mode, extracts result["rnnt_text"] with fallback to result["text"].
        """
        pipeline = self._get_pipeline()
        results = []

        for path in audio_paths:
            try:
                result = pipeline.transcribe(path)

                if self.decode_mode == "rnnt":
                    text = result.get("rnnt_text")
                    if not text:
                        logger.warning(
                            f"No rnnt_text for {path}, falling back to CTC text"
                        )
                        text = result.get("text", "")
                else:
                    text = result.get("text", "")

                results.append(text)
            except Exception as e:
                logger.error(f"Failed to transcribe {path}: {e}")
                results.append("")

        return results
