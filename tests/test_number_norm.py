"""Number canonicalisation in the scoring normalisation (2026-10-08): digits -> German words
on both sides, so "18 Uhr" vs "achtzehn Uhr" is not a recognition error."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.metrics import NORMALIZERS, compute_asr_metrics, normalize_text, normalize_text_raw  # noqa: E402
from src.benchmark.number_norm import spell_numbers_de  # noqa: E402


def test_digits_and_words_score_equal():
    hyp, ref = "Wir treffen uns um 18 Uhr am 15. August 1963.", "wir treffen uns um achtzehn uhr am fünfzehnten august neunzehnhundertdreiundsechzig"
    assert normalize_text(hyp) == normalize_text(ref)
    assert compute_asr_metrics([hyp], [ref])["wer"] == 0.0
    assert normalize_text_raw(hyp) != normalize_text_raw(ref)       # the raw track keeps the digits


def test_times_dates_decimals_ranges_units():
    assert spell_numbers_de("um 18:30 uhr") == "um achtzehn uhr dreißig"
    assert spell_numbers_de("um 8:00") == "um acht uhr"
    assert spell_numbers_de("vom 1.3.2024 bis 15.8.") == "vom ersten dritten zweitausendvierundzwanzig bis fünfzehnten achten"
    assert spell_numbers_de("1,5 € und 10 %") == "eins komma fünf euro und zehn prozent"
    assert spell_numbers_de("von 8-10 uhr") == "von acht bis zehn uhr"
    assert spell_numbers_de("0171 und 100 und 1000 und 2003") == "null eins sieben eins und hundert und tausend und zweitausenddrei"
    assert spell_numbers_de("mp3 und g7") == "mp3 und g7"


def test_second_pass_after_punctuation_and_eszett():
    assert normalize_text("Er kam 1.") == "er kam eins"
    assert normalize_text("Die 30 Jahre") == normalize_text("die dreißig Jahre") == "die dreissig jahre"
    assert normalize_text("1963, 2003") == "neunzehnhundertdreiundsechzig zweitausenddrei"


def test_tracks():
    assert set(NORMALIZERS) == {"std", "raw"}
    assert NORMALIZERS["std"] is normalize_text and NORMALIZERS["raw"] is normalize_text_raw
