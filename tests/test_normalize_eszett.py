"""ß/ss folding in the scoring normalisation and the result re-scoring tool."""

import copy
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark.metrics import (  # noqa: E402
    compute_asr_metrics,
    compute_single_sample_metrics,
    normalize_text,
)

_spec = importlib.util.spec_from_file_location("rescore_results", ROOT / "scripts" / "rescore_results.py")
rescore_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore_results)


def test_eszett_folds_to_ss_on_both_cases():
    assert normalize_text("Daß er muß") == "dass er muss"
    assert normalize_text("ein bißchen") == "ein bisschen"
    assert normalize_text("GROẞE Straße") == "grosse strasse"


def test_rest_of_the_normalisation_is_unchanged():
    assert normalize_text("Hallo, Welt!  Wie geht's?") == "hallo welt wie gehts"
    assert normalize_text("") == ""


def test_old_spelling_is_not_an_error():
    m = compute_single_sample_metrics("Ich weiß, dass er muss.", "ich weiss daß er muß")
    assert m["wer"] == 0.0 and m["cer"] == 0.0
    agg = compute_asr_metrics(["ein bisschen"], ["ein bißchen"])
    assert agg["wer"] == 0.0 and agg["substitutions"] == 0


def _v2():
    return {
        "schema_version": 2, "references": ["ort"], "primary_reference": "ort",
        "results": {"ort": {"wer": 0.5, "cer": 0.1, "substitutions": 1, "deletions": 0,
                            "insertions": 0, "num_samples": 1}},
        "speed": {"rtfx": 100.0},
        "per_sample": [{"hypothesis": "dass er", "references": {"ort": "daß er"},
                        "metrics": {"ort": {"wer": 0.5, "cer": 0.1}}}],
    }


def _v1():
    return {
        "results": {"ort_reference": {"wer": 0.5, "cer": 0.1, "substitutions": 1, "deletions": 0,
                                      "insertions": 0, "num_samples": 1}},
        "per_sample": [{"hypothesis": "muss ich", "ort_reference": "muß ich",
                        "ort_wer": 0.5, "ort_cer": 0.1}],
    }


def test_rescore_v2_in_place_keeps_other_fields():
    d = _v2()
    before = copy.deepcopy(d)
    delta, changed = rescore_results.rescore(d, write=True)
    assert delta == 0.5 and changed == 1
    assert d["results"]["ort"]["wer"] == 0.0 and d["per_sample"][0]["metrics"]["ort"]["wer"] == 0.0
    assert d["speed"] == before["speed"] and d["schema_version"] == 2


def test_rescore_v1_stays_v1():
    d = _v1()
    rescore_results.rescore(d, write=True)
    assert "schema_version" not in d
    assert d["results"]["ort_reference"]["wer"] == 0.0
    assert d["per_sample"][0]["ort_wer"] == 0.0 and d["per_sample"][0]["ort_cer"] == 0.0


def test_check_mode_writes_nothing():
    d = _v2()
    before = copy.deepcopy(d)
    rescore_results.rescore(d, write=False)
    assert d == before
