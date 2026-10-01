"""German test sets (Tuda-De, Common Voice, Verbmobil): sources, registry, manifest builder."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sf = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")

from src.datasets.registry import get_dataset  # noqa: E402
from src.datasets.de_testsets import (  # noqa: E402
    CvDeTestSource,
    TudaTestKinectRawSource,
    TudaTestYamahaSource,
    VerbmobilTestSource,
    manifest_labels,
)

_spec = importlib.util.spec_from_file_location("build_de_test_manifests",
                                              ROOT / "scripts" / "build_de_test_manifests.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def _wav(path: Path, seconds: float, sr: int = 16000, amp: float = 0.1, channels: int = 1) -> None:
    n = int(sr * seconds)
    sig = amp * np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, n)).astype("float32")
    if channels > 1:
        sig = np.stack([sig] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sig, sr, subtype="PCM_16")


def _tuda_xml(path: Path, cleaned: str, corpus: str = "PARL", gender: str = "male") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><recording>'
        f"<speaker_id>spk</speaker_id><gender>{gender}</gender><ageclass>21-30</ageclass>"
        f"<sentence_id>1</sentence_id><sentence>{cleaned}.</sentence>"
        f"<cleaned_sentence>{cleaned}</cleaned_sentence><corpus>{corpus}</corpus></recording>",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "cls,name,ref",
    [
        (TudaTestKinectRawSource, "tuda_test_kinect_raw", "ref"),
        (TudaTestYamahaSource, "tuda_test_yamaha", "ref"),
        (CvDeTestSource, "cv_de_test", "ref"),
        (VerbmobilTestSource, "verbmobil_test", "ort"),
    ],
)
def test_sources_yield_samples_and_resolve_in_registry(tmp_path, cls, name, ref):
    _wav(tmp_path / "a.wav", 1.5)
    manifest = tmp_path / "m.jsonl"
    rows = [
        {"audio_filepath": str(tmp_path / "a.wav"), "text": "Guten Morgen", "duration": 1.5,
         "utt_id": "u1", "labels": ["gender: female"]},
        {"audio_filepath": str(tmp_path / "b.wav"), "text": "", "duration": 1.0},
    ]
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    src = cls(manifest_path=manifest)
    assert src.name == name
    samples = list(src.iter_samples())
    assert len(samples) == 1  # empty-text row skipped
    s = samples[0]
    assert s.references == {ref: "Guten Morgen"}
    assert s.primary_reference == ref
    assert s.duration == pytest.approx(1.5)
    assert s.dataset_info["dataset_name"] == name

    assert isinstance(get_dataset(name, manifest_path=manifest), cls)
    assert isinstance(get_dataset(cls.__name__, manifest_path=manifest), cls)
    assert manifest_labels(name, manifest) == {str(tmp_path / "a.wav"): ["gender: female"]}


def test_default_manifest_paths_point_into_manifests_dir():
    for cls in (TudaTestKinectRawSource, TudaTestYamahaSource, CvDeTestSource, VerbmobilTestSource):
        assert cls().manifest_path == ROOT / "manifests" / f"{cls.name}.jsonl"
    assert manifest_labels("unknown_dataset") == {}


def test_build_tuda_keeps_same_recordings_for_all_mics(tmp_path):
    raw = tmp_path / "raw"
    test = raw / "test"
    # ok: both mics fine; its sentence also occurs in train
    _tuda_xml(test / "rec-ok.xml", "Das ist ein Satz", corpus="WIKI", gender="female")
    _wav(test / "rec-ok_Kinect-RAW.wav", 2.0)
    _wav(test / "rec-ok_Yamaha.wav", 2.0, sr=48000, channels=2)  # converted to 16 kHz mono
    # silent on the far-field mic -> dropped for both mics
    _tuda_xml(test / "rec-silent.xml", "Noch ein Satz")
    _wav(test / "rec-silent_Kinect-RAW.wav", 2.0, amp=0.0)
    _wav(test / "rec-silent_Yamaha.wav", 2.0)
    # too short on one mic -> dropped
    _tuda_xml(test / "rec-short.xml", "Kurz")
    _wav(test / "rec-short_Kinect-RAW.wav", 0.5)
    _wav(test / "rec-short_Yamaha.wav", 0.5)
    # no transcript -> dropped
    _tuda_xml(test / "rec-empty.xml", "")
    _wav(test / "rec-empty_Kinect-RAW.wav", 2.0)
    _wav(test / "rec-empty_Yamaha.wav", 2.0)
    _tuda_xml(raw / "train" / "t1.xml", "das ist ein  Satz")
    audit = tmp_path / "audit.txt"
    audit.write_text("# leak audit\ntuda_test_rec-ok_Kinect-RAW\n", encoding="utf-8")

    out = tmp_path / "manifests"
    report = tmp_path / "report.json"
    assert build.main(["tuda", "--root", str(raw), "--audio-out", str(tmp_path / "audio"),
                       "--manifest-dir", str(out), "--report", str(report),
                       "--audit-ids", str(audit)]) == 0

    far = [json.loads(l) for l in (out / "tuda_test_kinect_raw.jsonl").read_text().splitlines()]
    near = [json.loads(l) for l in (out / "tuda_test_yamaha.jsonl").read_text().splitlines()]
    assert [r["recording_id"] for r in far] == [r["recording_id"] for r in near] == ["rec-ok"]
    assert far[0]["text"] == "Das ist ein Satz"
    assert far[0]["audio_filepath"] == str(test / "rec-ok_Kinect-RAW.wav")  # used in place
    assert far[0]["labels"] == ["source: wiki", "gender: female", "sentence in SelOSS training data: yes"]
    assert far[0]["in_tuda_train"] is True
    assert near[0]["in_audit_list"] is True
    conv = Path(near[0]["audio_filepath"])
    assert conv.parent == tmp_path / "audio" / "Yamaha"
    info = sf.info(str(conv))
    assert (info.samplerate, info.channels) == (16000, 1)
    assert near[0]["duration"] == pytest.approx(2.0, abs=0.01)

    rep = json.loads(report.read_text())
    assert rep["recordings_in_split"] == 4 and rep["recordings_kept"] == 1
    assert set(rep["drop_reasons"]) == {"rec-silent", "rec-short", "rec-empty"}
    assert "silent" in rep["drop_reasons"]["rec-silent"]
    assert rep["kept_sentence_in_train"] == 1
    assert rep["kept_on_audit_list"] == 1


def test_build_tuda_rejects_unknown_mic(tmp_path):
    test = tmp_path / "raw" / "test"
    _tuda_xml(test / "r.xml", "Satz")
    _wav(test / "r_Yamaha.wav", 2.0)
    with pytest.raises(SystemExit, match="not in test split"):
        build.main(["tuda", "--root", str(tmp_path / "raw"), "--mics", "Kinect-RAW",
                    "--audio-out", str(tmp_path / "a"), "--manifest-dir", str(tmp_path / "m"),
                    "--report", str(tmp_path / "r.json")])


def test_build_cv_from_release_dir(tmp_path):
    pytest.importorskip("soxr")
    corpus = tmp_path / "de"
    _wav(corpus / "clips" / "common_voice_de_1.wav", 1.2, sr=48000)
    _wav(corpus / "clips" / "common_voice_de_2.wav", 0.8)
    (corpus / "test.tsv").write_text(
        "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\n"
        "c1\tcommon_voice_de_1.wav\t„Guten Tag“, sagte er.\t2\t0\tthirties\tmale\n"
        "c2\tcommon_voice_de_2.wav\tWie geht es dir?\t2\t0\t\t\n"
        "c3\tcommon_voice_de_3.wav\tFehlt.\t2\t0\t\t\n",
        encoding="utf-8",
    )
    train = tmp_path / "train.txt"
    train.write_text("wie geht es dir\n", encoding="utf-8")
    manifest = tmp_path / "cv.jsonl"
    report = tmp_path / "cv.json"
    assert build.main(["cv", "--version", "cvX", "--corpus-dir", str(corpus), "--audio-out",
                       str(tmp_path / "wav"), "--manifest", str(manifest), "--report", str(report),
                       "--train-sentences", str(train), "--workers", "1"]) == 0
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()]
    assert [r["utt_id"] for r in rows] == ["common_voice_de_1", "common_voice_de_2"]
    assert rows[0]["text"] == "„Guten Tag“, sagte er."  # reference kept verbatim
    assert rows[0]["labels"] == ["gender: male", "sentence in CV train: no"]
    assert rows[1]["labels"] == ["sentence in CV train: yes"]
    info = sf.info(rows[0]["audio_filepath"])
    assert (info.samplerate, info.channels) == (16000, 1)
    rep = json.loads(report.read_text())
    assert rep["clips"] == 2 and list(rep["failed"]) == ["common_voice_de_3"]
    assert rep["sentence_in_train"] == 1


def test_leaderboard_subsplits_read_manifest_labels(tmp_path, monkeypatch):
    import src.datasets.de_testsets as de_testsets
    from src.benchmark.result import SampleResult
    from src.leaderboard import aggregate

    if not hasattr(aggregate, "_subsplit_fns"):
        pytest.skip("leaderboard without sub-split drill-down")
    _subsplit_fns = aggregate._subsplit_fns

    row = {"audio_filepath": "/x/a.wav", "text": "t", "duration": 1.0,
           "labels": ["gender: male", "sentence in CV train: no"]}
    (tmp_path / "cv_de_test.jsonl").write_text(json.dumps(row), encoding="utf-8")
    monkeypatch.setattr(de_testsets, "MANIFEST_DIR", tmp_path)

    fns = _subsplit_fns({})
    assert {"parlo", "ksof", "nscc", "cv_de_test"} <= set(fns)
    assert "tuda_test_yamaha" not in fns  # no manifest -> no sub-split function
    hit = SampleResult(index=0, audio_path="/x/a.wav", hypothesis="", duration=1.0)
    miss = SampleResult(index=1, audio_path="/x/b.wav", hypothesis="", duration=1.0)
    assert fns["cv_de_test"](hit) == ["gender: male", "sentence in CV train: no"]
    assert fns["cv_de_test"](miss) == []


def _vm_turn(root: Path, part: str, turn: str, ort_tokens, seconds: float = 1.5, sr: int = 16000):
    d = root / part / part / turn[:5]
    _wav(d / f"{turn}.wav", seconds, sr=sr)
    lines = ["LHD: Partitur 1.3", f"SAM: {sr}"]
    lines += [f"ORT:\t{i}\t{tok}" for i, tok in enumerate(ort_tokens)]
    (d / f"{turn}.par").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_verbmobil_uses_bas_test_lists_and_rvg1_ort_cleaning(tmp_path):
    root = tmp_path / "Verbmobil"
    sets = root / "VM1" / "doc" / "doc" / "SETS"
    sets.mkdir(parents=True)
    (sets / "VM1_TEST").write_text("j532axx0_000_DEP\tVM14.1\nj532axx0_002_DEP\tVM14.1\n"
                                   "j532axx0_004_DEP\tVM14.1\n", encoding="utf-8")
    (sets / "VM2_TEST").write_text("g223acn2_001_AHV\tVM21.1\n", encoding="utf-8")
    _vm_turn(root, "VM1", "j532axx0_000_DEP", ["Herr", "Hoff", '<"ahm>', 'f"unft"agigen', "$A-$G-$T-$R"])
    _vm_turn(root, "VM1", "j532axx0_002_DEP", ['<"ah>', "<Schmatzen>"])  # only markers -> dropped
    # j532axx0_004_DEP is listed but missing on disk -> dropped
    _vm_turn(root, "VM2", "g223acn2_001_AHV", ['zw"olfter', 'w"urde', "gehen"], sr=8000)
    # a turn that is not on a TEST list is ignored
    _vm_turn(root, "VM1", "j532axx0_001_OSH", ["nein"])

    manifest = tmp_path / "vm.jsonl"
    report = tmp_path / "vm.json"
    assert build.main(["verbmobil", "--root", str(root), "--audio-out", str(tmp_path / "audio"),
                       "--manifest", str(manifest), "--report", str(report)]) == 0
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()]
    assert [r["turn_id"] for r in rows] == ["j532axx0_000_DEP", "g223acn2_001_AHV"]
    assert rows[0]["text"] == "Herr Hoff fünftägigen $A-$G-$T-$R"
    assert rows[0]["audio_filepath"] == str(root / "VM1" / "VM1" / "j532a" / "j532axx0_000_DEP.wav")
    assert rows[0]["labels"] == ["part: VM1"] and rows[0]["speaker_id"] == "DEP"
    assert rows[1]["text"] == "zwölfter würde gehen"
    conv = Path(rows[1]["audio_filepath"])  # 8 kHz source -> converted copy
    assert conv.parent == tmp_path / "audio" / "VM2"
    assert sf.info(str(conv)).samplerate == 16000
    rep = json.loads(report.read_text())
    assert rep["clips"] == 2 and rep["speakers"] == 2
    assert rep["dropped"] == {"j532axx0_002_DEP": "empty ORT after cleaning",
                              "j532axx0_004_DEP": "missing wav/par"}
    assert rep["parts"]["VM1"]["listed"] == 3 and rep["parts"]["VM2"]["kept"] == 1
