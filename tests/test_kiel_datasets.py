"""kiel_read / kiel_spon: bundled Kiel Corpus manifests (scripts/build_kiel_manifests.py)."""
import json
from pathlib import Path

from src.datasets.de_testsets import MANIFEST_DIR, KielReadSource, KielSponSource, manifest_labels
from src.datasets.registry import get_dataset


def _rows(name):
    return [json.loads(l) for l in (MANIFEST_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def test_sources_registered():
    for cls, name in ((KielReadSource, "kiel_read"), (KielSponSource, "kiel_spon")):
        src = get_dataset(name)
        assert isinstance(src, cls)
        assert src.reference_key == "text"
        assert src.manifest_path == MANIFEST_DIR / f"{name}.jsonl"


def test_manifests_well_formed():
    read, spon = _rows("kiel_read"), _rows("kiel_spon")
    assert len(read) == 5227 and len(spon) == 3609
    for rows, style in ((read, "read"), (spon, "spontaneous")):
        ids = {r["utt_id"] for r in rows}
        assert len(ids) == len(rows)
        for r in rows:
            assert r["style"] == style
            assert any(ch.isalpha() for ch in r["text"])
            assert r["duration"] >= 0.3
            assert r["audio_filepath"].endswith(".wav")
            assert any(l.startswith("realized: ") for l in r["labels"])
    assert all(r["subcorpus"].startswith("PhonDat") for r in read)
    assert {r["subcorpus"] for r in spon} == {"VMaddOrt", "VMaddSeg", "VerbMobil", "VideoTask"}


def test_labels_exposed_for_drilldown():
    labels = manifest_labels("kiel_spon")
    assert len(labels) == 3609
    assert all(any(l.startswith("subcorpus: ") for l in v) for v in labels.values())
