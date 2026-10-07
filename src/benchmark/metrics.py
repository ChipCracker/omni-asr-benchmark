"""ASR evaluation metrics using jiwer.

Moved unchanged from ``src/evaluation/metrics.py`` so the generic benchmark
engine and the (legacy) evaluation package share one implementation.
"""

import re
import string
import sys
import unicodedata
from typing import Dict, List, Any

from jiwer import wer, cer, process_words

from .number_norm import spell_digits_de, spell_numbers_de


# Characters deleted by normalize_text: ASCII string.punctuation (incl. the symbols
# $ + < = > ^ ` | ~, as before), every Unicode punctuation character (category P*:
# typographic quotes „ “ ‚ ‘ ’ « », dashes – — ‑, …, fullwidth ，。 etc.) and two
# apostrophe look-alikes outside P* (´ U+00B4 ACUTE ACCENT, ʼ U+02BC MODIFIER LETTER
# APOSTROPHE). Deleted, not replaced by a space -- like ASCII "-" ("E-Mail" -> "email").
_PUNCT_DELETE = dict.fromkeys(
    [ord(c) for c in string.punctuation]
    + [cp for cp in range(sys.maxunicode + 1) if unicodedata.category(chr(cp)).startswith("P")]
    + [0x00B4, 0x02BC]
)


def normalize_text(text: str) -> str:
    """Normalize text for ASR evaluation (the "standard" track).

    - Lowercase
    - Numbers -> German words on both sides (``number_norm.spell_numbers_de``: times,
      dates, ordinals, decimals, ranges, percent/euro, years, plain integers), BEFORE
      the punctuation is stripped because the dot/colon/comma of a number carry meaning;
      a second pass (``spell_digits_de``) catches integers that only became standalone
      after the stripping ("er kam 1.")
    - ß -> ss (after lowercasing and after the number words, which num2words writes with ß)
    - Remove punctuation: ASCII and all Unicode punctuation (see ``_PUNCT_DELETE``)
    - Normalize whitespace

    Numbers (2026-10-08): the references spell numbers out (BAS ORT, Kiel, Tuda) or keep
    digits (Common Voice), and models differ the same way (SelOSS v4 and Whisper write
    "18 Uhr", Qwen "achtzehn Uhr"). Neither is a recognition error, so both sides are
    canonicalised to the spoken form. The "raw" track (``normalize_text_raw``) keeps the
    digits as they are, i.e. the scoring used until 2026-10-07.
    ß/ss: BAS-RVG1-ORT is transcribed in the pre-1996 spelling ("daß", "muß",
    "bißchen"), current models write "dass", "muss", "bisschen". The spelling
    reform is not a recognition error, so both sides are folded to "ss".
    Unicode punctuation: Common Voice sentences and some model outputs carry
    typographic quotes, dashes and apostrophes ("geht’s") that ASCII
    string.punctuation left in place as word parts or stray tokens.
    """
    if not text:
        return ""

    text = text.lower()
    text = spell_numbers_de(text)
    text = text.translate(_PUNCT_DELETE)
    text = spell_digits_de(text)
    text = text.replace("ß", "ss")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_raw(text: str) -> str:
    """``normalize_text`` without the number canonicalisation (digits stay digits): the
    scoring of the leaderboard before 2026-10-08, kept as the "raw" track."""
    if not text:
        return ""

    text = text.lower()
    text = text.replace("ß", "ss")
    text = text.translate(_PUNCT_DELETE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_numbers(text: str) -> str:
    """Alias of :func:`normalize_text` (numbers are canonical in the standard track now)."""
    return normalize_text(text)


NORMALIZERS = {"std": normalize_text, "raw": normalize_text_raw}


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

    # Normalize texts before computing metrics
    hypotheses = [normalize_text(h) for h in hypotheses]
    references = [normalize_text(r) for r in references]

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

    # Normalize texts before computing metrics
    hypothesis = normalize_text(hypothesis)
    reference = normalize_text(reference)

    return {
        "wer": wer(reference, hypothesis),
        "cer": cer(reference, hypothesis),
    }
