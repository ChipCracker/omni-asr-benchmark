"""ASR evaluation metrics using jiwer."""

from typing import Dict, List, Any

from jiwer import wer, cer, process_words


def compute_asr_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, Any]:
    """Compute ASR evaluation metrics (WER, CER, and error counts).

    Args:
        hypotheses: List of predicted transcriptions.
        references: List of reference transcriptions.

    Returns:
        Dictionary containing:
            - wer: Word Error Rate
            - cer: Character Error Rate
            - substitutions: Number of word substitutions
            - deletions: Number of word deletions
            - insertions: Number of word insertions
            - num_samples: Number of samples evaluated
    """
    if not hypotheses or not references:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "num_samples": 0,
        }

    output = process_words(references, hypotheses)

    return {
        "wer": output.wer,
        "cer": cer(references, hypotheses),
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "num_samples": len(hypotheses),
    }


def compute_single_sample_metrics(
    hypothesis: str,
    reference: str,
) -> Dict[str, Any]:
    """Compute metrics for a single sample.

    Args:
        hypothesis: The predicted transcription.
        reference: The reference transcription.

    Returns:
        Dictionary containing WER and CER for the single sample.
    """
    if not hypothesis or not reference:
        return {"wer": 1.0, "cer": 1.0}

    return {
        "wer": wer(reference, hypothesis),
        "cer": cer(reference, hypothesis),
    }
