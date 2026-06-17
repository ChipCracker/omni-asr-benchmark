"""Model registry: map a model name to an :class:`AsrModel` instance.

Routing is by substring match on the (lower-cased) model name, mirroring the
historical ``get_evaluator`` factory. Each backend is imported lazily inside its
branch so optional heavy dependencies (transformers, NeMo, omnilingual-asr, ...)
are only loaded when that model is actually selected.
"""

from __future__ import annotations

from typing import Optional

from .base import AsrModel


def get_model(
    model_name: str,
    language: str,
    batch_size: int,
    model_dir: Optional[str] = None,
) -> AsrModel:
    """Instantiate the appropriate model for ``model_name``.

    Args:
        model_name: Model identifier (e.g. ``openai/whisper-large-v3``,
            ``nvidia/parakeet-tdt-0.6b-v3``, ``omniASR_LLM_Unlimited_7B_v2``).
        language: Language code for transcription (e.g. ``deu_Latn``).
        batch_size: Batch size for inference.
        model_dir: Local model directory (used by the syllabic-asr model).

    Returns:
        An :class:`AsrModel` instance. Falls back to OmniASR for unknown names.
    """
    model_lower = model_name.lower()

    # CrisperWhisper must be checked before generic whisper.
    if "crisperwhisper" in model_lower or "crisper" in model_lower:
        from .crisperwhisper import CrisperWhisperEvaluator
        return CrisperWhisperEvaluator(model_name, language, batch_size)
    elif "whisper" in model_lower:
        from .whisper import WhisperEvaluator
        return WhisperEvaluator(model_name, language, batch_size)
    elif "parakeet" in model_lower:
        from .parakeet import ParakeetEvaluator
        return ParakeetEvaluator(model_name, language, batch_size)
    elif "vibevoice" in model_lower:
        from .vibevoice import VibeVoiceEvaluator
        return VibeVoiceEvaluator(model_name, language, batch_size)
    elif "canary-1b" in model_lower:
        from .canary1b import Canary1bEvaluator
        return Canary1bEvaluator(model_name, language, batch_size)
    elif "canary" in model_lower:
        from .canary import CanaryEvaluator
        return CanaryEvaluator(model_name, language, batch_size)
    elif "voxtral" in model_lower and "realtime" in model_lower and "online" in model_lower:
        from .voxtral_realtime import VoxtralRealtimeOnlineEvaluator
        return VoxtralRealtimeOnlineEvaluator(model_name, language, batch_size)
    elif "voxtral" in model_lower and "realtime" in model_lower:
        from .voxtral_realtime import VoxtralRealtimeEvaluator
        return VoxtralRealtimeEvaluator(model_name, language, batch_size)
    elif "voxtral" in model_lower:
        from .voxtral import VoxtralEvaluator
        return VoxtralEvaluator(model_name, language, batch_size)
    elif "phi-4" in model_lower or "phi4" in model_lower:
        from .phi4 import Phi4Evaluator
        return Phi4Evaluator(model_name, language, batch_size)
    elif "qwen3-asr" in model_lower:
        from .qwen3asr import Qwen3ASREvaluator
        return Qwen3ASREvaluator(model_name, language, batch_size)
    elif "syllabic-asr" in model_lower or "syllabic_asr" in model_lower:
        from .syllabic_asr import SyllabicASREvaluator
        decode_mode = "rnnt" if "rnnt" in model_lower else "ctc"
        return SyllabicASREvaluator(
            model_name, language, batch_size,
            decode_mode=decode_mode, model_dir=model_dir,
        )
    elif "stt_" in model_lower and "conformer" in model_lower:
        from .nemo_stt import NemoSTTEvaluator
        return NemoSTTEvaluator(model_name, language, batch_size)
    elif "cohere" in model_lower and "transcribe" in model_lower:
        from .cohere_transcribe import CohereTranscribeEvaluator
        return CohereTranscribeEvaluator(model_name, language, batch_size)
    elif "granite" in model_lower and "nar" in model_lower:
        from .granite_nar import GraniteNarEvaluator
        return GraniteNarEvaluator(model_name, language, batch_size)
    elif "granite" in model_lower:
        from .granite import GraniteEvaluator
        return GraniteEvaluator(model_name, language, batch_size)
    elif "gemma" in model_lower:
        from .gemma import GemmaEvaluator
        return GemmaEvaluator(model_name, language, batch_size)
    elif "omniasr" in model_lower or "omni-asr" in model_lower or "omni_asr" in model_lower:
        from .omniasr import OmniASREvaluator
        return OmniASREvaluator(model_name, language, batch_size)
    else:
        # Default to OmniASR.
        from .omniasr import OmniASREvaluator
        return OmniASREvaluator(model_name, language, batch_size)
