"""German number canonicalisation for scoring: digits -> spoken words, on BOTH sides.

Speakers say numbers as words; the references spell them out (BAS ORT, Kiel, Tuda
``cleaned_sentence``) or keep digits (Common Voice), and models differ the same way
(SelOSS v4 and Whisper write "18 Uhr", Qwen writes "achtzehn Uhr"). None of that is a
recognition error, so :func:`spell_numbers_de` rewrites every digit token to the words a
German speaker would say, and ``normalize_text`` applies it to hypothesis and reference
before the comparison (the "standard" track; the "raw" track keeps digits as they are).

Rules, applied to the lower-cased text BEFORE punctuation is stripped (the dot, colon and
comma of a number carry meaning):

* ``18:30`` / ``8:00``            -> "achtzehn uhr dreissig" / "acht uhr"
* ``1.3.2024`` / ``15.8.``        -> "ersten dritten zweitausendvierundzwanzig" / "fünfzehnten achten"
* ``15.`` before a word           -> ordinal in the oblique case ("fünfzehnten"), the form dates take
* ``1,5`` / ``1.5``               -> "eins komma fünf"
* ``8-10`` / ``8–10``             -> "acht bis zehn"
* ``10 %`` / ``10%``              -> "zehn prozent";  ``5 €`` / ``5€`` -> "fünf euro"
* ``1963``, ``1100`` … ``1999``   -> year form "neunzehnhundertdreiundsechzig"
* any other integer              -> cardinal; a leading "ein" of "einhundert"/"eintausend"
                                     is dropped ("hundert", "tausendfünfhundert"), as spoken
* leading zero (``0171``)        -> digit by digit "null eins sieben eins"
* digits glued to letters (``3d``, ``g7``, ``mp3``) are left alone

num2words is REQUIRED (``pip install num2words``); without it scoring would silently
differ between machines, so the import error is raised, not swallowed.
"""

from __future__ import annotations

import re

from num2words import num2words

_SP = r"(?:(?<=\s)|^)"
_EP = r"(?=\s|$)"
_TIME_RE = re.compile(_SP + r"(\d{1,2}):(\d{2})(?:\s?uhr)?" + _EP)   # "18:30 Uhr" -> one "uhr"
_DATE_RE = re.compile(_SP + r"(\d{1,2})\.(\d{1,2})\.(\d{4})?" + _EP)
_DECIMAL_RE = re.compile(_SP + r"(\d+)[,.](\d+)" + _EP)
_RANGE_RE = re.compile(_SP + r"(\d+)\s?[-–]\s?(\d+)" + _EP)
_PERCENT_RE = re.compile(_SP + r"(\d+(?:[,.]\d+)?)\s?%" + _EP)
_EURO_RE = re.compile(_SP + r"(\d+(?:[,.]\d+)?)\s?€" + _EP)
_ORDINAL_RE = re.compile(_SP + r"(\d+)\.(?=\s+[a-zäöüß])")
_INT_RE = re.compile(_SP + r"\d+" + _EP)


def _card(n: int) -> str:
    """Cardinal as spoken: years 1100-1999 in the hundreds form, no leading 'ein' on 100/1000."""
    if 1100 <= n <= 1999:
        return num2words(n, lang="de", to="year")
    w = num2words(n, lang="de")
    if w.startswith("einhundert"):
        w = w[3:]
    elif w.startswith("eintausend"):
        w = w[3:]
    return w


def _ord(n: int) -> str:
    """Ordinal in the oblique case ("am fünfzehnten", "vom ersten bis zum dritten")."""
    return num2words(n, lang="de", to="ordinal") + "n"


def _digits(s: str) -> str:
    """Digit string spoken digit by digit ("05" -> "null fünf"), for decimals and leading zeros."""
    return " ".join(_card(int(c)) for c in s)


def _num(s: str) -> str:
    """Integer or decimal string -> words; leading zero (0171, 007) means digit by digit."""
    m = re.fullmatch(r"(\d+)[,.](\d+)", s)
    if m:
        return _card(int(m.group(1))) + " komma " + _digits(m.group(2))
    if len(s) > 1 and s[0] == "0":
        return _digits(s)
    return _card(int(s))


def spell_numbers_de(text: str) -> str:
    """Rewrite every digit token of ``text`` (lower-cased, punctuation still present) to words."""
    if not any(ch.isdigit() for ch in text):
        return text
    text = _TIME_RE.sub(lambda m: _card(int(m.group(1))) + " uhr"
                        + (" " + _card(int(m.group(2))) if int(m.group(2)) else ""), text)
    text = _DATE_RE.sub(lambda m: _ord(int(m.group(1))) + " " + _ord(int(m.group(2)))
                        + (" " + _card(int(m.group(3))) if m.group(3) else ""), text)
    text = _PERCENT_RE.sub(lambda m: _num(m.group(1)) + " prozent", text)
    text = _EURO_RE.sub(lambda m: _num(m.group(1)) + " euro", text)
    text = _DECIMAL_RE.sub(lambda m: _num(m.group(0)), text)
    text = _RANGE_RE.sub(lambda m: _num(m.group(1)) + " bis " + _num(m.group(2)), text)
    text = _ORDINAL_RE.sub(lambda m: _ord(int(m.group(1))), text)
    text = _INT_RE.sub(lambda m: _num(m.group(0)), text)
    return text


def spell_digits_de(text: str) -> str:
    """Standalone integers only -- the second pass after punctuation stripping ("er kam 1." -> "eins")."""
    return _INT_RE.sub(lambda m: _num(m.group(0)), text)


def available() -> bool:
    return True
