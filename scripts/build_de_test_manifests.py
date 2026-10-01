#!/usr/bin/env python3
"""Build the NeMo manifests for the German test sets: Tuda-De, Common Voice, Verbmobil.

One sub-command per corpus. Each writes one JSONL manifest per dataset and a
JSON report with the counts; audio ends up as 16 kHz mono PCM-16 WAV (files
that already are 16 kHz mono WAV are referenced in place, everything else is
converted into a writable directory). ``src/datasets/de_testsets.py`` reads the
manifests.

**tuda** -- Tuda-De v4 (``german-speechdata-package-v4.tar.gz``), test split.
Every recording exists once per microphone; the manifests hold the same
recordings for every selected microphone, so a far-field vs. close-talk
difference is the microphone alone. A recording is dropped for *all*
microphones when one of them is missing, unreadable, shorter than
``--min-duration`` or silent (peak below ``--min-peak``: the speaker did not
speak but the XML carries a transcript). Reference = ``cleaned_sentence``.
Labels per row: text source (``<corpus>`` of the XML), gender and, with
``--audit-ids``, whether a leak audit found the sentence in a model's training
data (list of recording ids / ``tuda_test_<rec>_<mic>`` utt ids). Whether the
sentence also occurs in the Tuda *train* split (same prompt, other speaker) is
kept as the row field ``in_tuda_train`` and counted in the report (v4: 1 of
1021, too few for a sub-split).

  # extract test/ (audio + XML) and train/*.xml once, then
  python scripts/build_de_test_manifests.py tuda \\
      --root /nfs1/scratch/staff/witzlch/tuda-bench/raw_v4 \\
      --mics Kinect-RAW,Yamaha --audio-out /nfs1/scratch/staff/witzlch/tuda-bench/audio

**cv** -- Common Voice German, full official ``test`` split. Input is either a
Common Voice release directory (``test.tsv`` + ``clips/*.mp3``) or a parquet
store with ``audio`` struct{bytes, path} + ``text`` columns (HELMA's
``eval-de/cv26_de/test``). Reference = ``sentence`` (case and punctuation kept).
``--train-sentences`` (one sentence per line) adds the label ``sentence in CV
train: yes/no`` -- Common Voice splits are speaker-disjoint, not
sentence-disjoint.

  python scripts/build_de_test_manifests.py cv --version cv26 \\
      --release "Common Voice Scripted Speech 26.0 German (Mozilla Data Collective)" \\
      --train-sentences cv26_train_sentences.txt --parquet-glob '/nfs1/scratch/staff/witzlch/cv-bench/src/cv26_de_test/*.parquet' \\
      --audio-out /nfs1/scratch/staff/witzlch/cv-bench/cv26_de_test

**verbmobil** -- Verbmobil German, the BAS-defined TEST set: the turn lists
``VM1_TEST`` and ``VM2_TEST`` of "Infos to VM Data Sets" v2.3 (F. Schiel, BAS,
2003; ``<root>/VM1/doc/doc/SETS``). Speaker-disjoint from the BAS TRAIN/DEV
sets; VM1_TEST comes from volume VM14.1, the last official VM1 evaluation set.
One clip per listed turn (``<root>/VM<n>/VM<n>/<dialog>/<turn>.wav``, whole
turn), reference = ORT tier of the turn's BAS Partitur, cleaned with BAS-RVG1's
ORT token cleaner (umlauts decoded, ``<...>`` hesitation/noise tokens removed).
Turns whose ORT is empty after cleaning are dropped and reported.

  python scripts/build_de_test_manifests.py verbmobil --root /nfs/data/Verbmobil \
      --audio-out /nfs1/scratch/staff/witzlch/vm-bench/audio
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "manifests"


def log(*args) -> None:
    print(f"[{time.strftime('%F %T')}]", *args, flush=True)


def sentence_key(text: str) -> str:
    """Overlap key: lower case, letters/digits only, single spaces."""
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def to_mono_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import soxr

        wav = soxr.resample(wav, sr, SAMPLE_RATE)
    return wav.astype("float32")


def write_manifest(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"wrote {path} ({len(rows)} rows, {sum(r['duration'] for r in rows) / 3600:.3f} h)")


# ---------------------------------------------------------------- Tuda-De


def find_split_dir(root: Path, split: str) -> Path:
    """``<root>/<split>`` or ``<root>/<package>/<split>`` (v2 has a top folder, v4 not)."""
    for cand in [root / split, *sorted(root.glob(f"*/{split}"))]:
        if cand.is_dir():
            return cand
    raise SystemExit(f"no '{split}' directory under {root}")


def read_tuda_xml(path: Path) -> Dict[str, str]:
    root = ET.fromstring(path.read_bytes().decode("utf-8", "ignore"))
    fields = ("cleaned_sentence", "sentence", "corpus", "gender", "ageclass", "speaker_id",
              "sentence_id")
    return {f: ((root.find(f".//{f}").text or "").strip() if root.find(f".//{f}") is not None
                else "") for f in fields}


def read_audit_ids(path: str) -> set:
    """Recording ids from an audit list: ``<rec>``, ``<rec>_<mic>`` or ``tuda_test_<rec>_<mic>``."""
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            tok = line.split("#", 1)[0].strip().split()
            if not tok:
                continue
            rid = tok[0]
            if rid.startswith("tuda_test_"):
                rid = rid[len("tuda_test_"):]
            out.add(rid.split("_", 1)[0])
    return out


def tuda_dataset_name(mic: str) -> str:
    return "tuda_test_" + mic.lower().replace("-", "_")


def probe_wav(path: Path) -> Tuple[Optional[np.ndarray], Optional[int], str]:
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # unreadable file
        return None, None, f"unreadable ({exc.__class__.__name__})"
    return wav, sr, ""


def build_tuda(args) -> int:
    root = Path(args.root)
    test_dir = find_split_dir(root, "test")
    mics = [m.strip() for m in args.mics.split(",") if m.strip()]
    xmls = sorted(test_dir.glob("*.xml"))
    log(f"{len(xmls)} XML in {test_dir}")

    train_keys: set = set()
    try:
        train_dir = find_split_dir(root, "train")
        for p in train_dir.glob("*.xml"):
            key = sentence_key(read_tuda_xml(p)["cleaned_sentence"])
            if key:
                train_keys.add(key)
        log(f"{len(train_keys)} distinct train sentences from {train_dir}")
    except SystemExit:
        log("no train/ directory: overlap label skipped")

    audit = read_audit_ids(args.audit_ids) if args.audit_ids else None
    if audit is not None:
        log(f"{len(audit)} recordings on the audit list {args.audit_ids}")

    audio_out = Path(args.audio_out)
    kept: List[Tuple[str, Dict[str, str], Dict[str, Dict]]] = []
    drops: Dict[str, str] = {}
    mic_seen: Dict[str, int] = {}
    for p in sorted(test_dir.glob("*.wav")):
        mic = p.stem.split("_", 1)[1] if "_" in p.stem else ""
        mic_seen[mic] = mic_seen.get(mic, 0) + 1
    unknown = [m for m in mics if m not in mic_seen]
    if unknown:
        raise SystemExit(f"microphone(s) {unknown} not in test split; present: {sorted(mic_seen)}")

    for xml in xmls:
        rec = xml.stem
        meta = read_tuda_xml(xml)
        if not meta["cleaned_sentence"]:
            drops[rec] = "no transcript"
            continue
        per_mic: Dict[str, Dict] = {}
        reason = ""
        for mic in mics:
            src = test_dir / f"{rec}_{mic}.wav"
            if not src.is_file():
                reason = f"{mic}: missing"
                break
            wav, sr, err = probe_wav(src)
            if wav is None:
                reason = f"{mic}: {err}"
                break
            converted = wav.ndim > 1 or sr != SAMPLE_RATE
            wav = to_mono_16k(wav, sr)
            dur = len(wav) / SAMPLE_RATE
            peak = float(np.abs(wav).max()) if len(wav) else 0.0
            if dur < args.min_duration:
                reason = f"{mic}: {dur:.2f} s < {args.min_duration} s"
                break
            if peak < args.min_peak:
                reason = f"{mic}: silent (peak {peak:.2e})"
                break
            path = src
            if converted:
                path = audio_out / mic / f"{rec}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), wav, SAMPLE_RATE, subtype="PCM_16")
            per_mic[mic] = {"path": str(path), "duration": round(dur, 3), "peak": peak,
                            "converted": converted}
        if reason:
            drops[rec] = reason
            continue
        kept.append((rec, meta, per_mic))

    n_overlap = 0
    reports: Dict[str, Dict] = {}
    for mic in mics:
        rows = []
        for rec, meta, per_mic in kept:
            in_train = sentence_key(meta["cleaned_sentence"]) in train_keys
            labels = [f"source: {meta['corpus'].lower() or 'unknown'}",
                      f"gender: {meta['gender'] or 'unknown'}"]
            in_audit = audit is not None and rec in audit
            if audit is not None:
                labels.append(f"{args.audit_label}: {'yes' if in_audit else 'no'}")
            rows.append({
                "audio_filepath": per_mic[mic]["path"],
                "text": meta["cleaned_sentence"],
                "duration": per_mic[mic]["duration"],
                "utt_id": f"tuda_test_{rec}_{mic}",
                "recording_id": rec,
                "mic": mic,
                "speaker_id": meta["speaker_id"],
                "sentence_id": meta["sentence_id"],
                "sentence_raw": meta["sentence"],
                "in_tuda_train": in_train,
                **({"in_audit_list": in_audit} if audit is not None else {}),
                "labels": labels,
            })
        name = tuda_dataset_name(mic)
        write_manifest(rows, Path(args.manifest_dir) / f"{name}.jsonl")
        n_overlap = sum(r["in_tuda_train"] for r in rows)
        durs = [r["duration"] for r in rows]
        reports[name] = {
            "mic": mic, "clips": len(rows), "hours": round(sum(durs) / 3600, 4),
            "words": sum(len(r["text"].split()) for r in rows),
            "duration_s": {"mean": round(float(np.mean(durs)), 2), "min": round(min(durs), 2),
                           "max": round(max(durs), 2),
                           "over_30s": int(sum(d > 30 for d in durs))},
            "converted_to_16k_mono": int(sum(pm[mic]["converted"] for _, _, pm in kept)),
        }
    report = {
        "corpus": "Tuda-De (german-speechdata-package-v4), split test",
        "root": str(root), "text_field": "cleaned_sentence", "mics": mics,
        "recordings_in_split": len(xmls), "recordings_kept": len(kept),
        "recordings_dropped": len(drops), "drop_reasons": drops,
        "filters": {"min_duration_s": args.min_duration, "min_peak": args.min_peak},
        "wavs_per_mic_in_split": mic_seen,
        "train_sentences": len(train_keys), "kept_sentence_in_train": n_overlap,
        "audit_ids": args.audit_ids, "audit_label": args.audit_label if audit is not None else None,
        "kept_on_audit_list": sum(rec in audit for rec, _, _ in kept) if audit is not None else None,
        "datasets": reports,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: v for k, v in report.items() if k != "drop_reasons"}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- Common Voice


def iter_cv_corpus(corpus_dir: Path) -> Iterable[Dict]:
    """Rows of ``test.tsv``: utt_id, sentence, audio source path."""
    csv.field_size_limit(sys.maxsize)
    with (corpus_dir / "test.tsv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            yield {"utt_id": Path(row["path"]).stem, "text": row["sentence"],
                   "src": str(corpus_dir / "clips" / row["path"]), "bytes": None,
                   "gender": row.get("gender") or "", "age": row.get("age") or ""}


def iter_cv_parquet(pattern: str) -> Iterable[Dict]:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet files match {pattern}")
    for f in files:
        for row in pq.read_table(f).to_pylist():
            audio = row.get("audio") or {}
            utt = row.get("utt_id") or Path(audio.get("path") or "").stem
            yield {"utt_id": utt, "text": row.get("text") or row.get("sentence") or "",
                   "src": audio.get("path") or "", "bytes": audio.get("bytes"),
                   "gender": row.get("gender") or "", "age": row.get("age") or ""}


def _convert_one(job: Tuple[str, Optional[bytes], str]) -> Tuple[str, float, str]:
    src, data, dst = job
    try:
        if os.path.isfile(dst) and os.path.getsize(dst) > 44:
            return dst, sf.info(dst).duration, ""
        if data is not None:
            wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        else:
            wav, sr = sf.read(src, dtype="float32", always_2d=False)
        wav = to_mono_16k(wav, sr)
        if not len(wav):
            return dst, 0.0, "empty audio"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, wav, SAMPLE_RATE, subtype="PCM_16")
        return dst, len(wav) / SAMPLE_RATE, ""
    except Exception as exc:
        return dst, 0.0, f"{exc.__class__.__name__}: {exc}"


def build_cv(args) -> int:
    if bool(args.corpus_dir) == bool(args.parquet_glob):
        raise SystemExit("give exactly one of --corpus-dir / --parquet-glob")
    rows = list(iter_cv_corpus(Path(args.corpus_dir)) if args.corpus_dir
                else iter_cv_parquet(args.parquet_glob))
    log(f"{len(rows)} test rows ({args.version})")
    train_keys: set = set()
    if args.train_sentences:
        with open(args.train_sentences, encoding="utf-8") as fh:
            train_keys = {sentence_key(l) for l in fh if l.strip()}
        log(f"{len(train_keys)} distinct train sentences from {args.train_sentences}")

    out = Path(args.audio_out)
    jobs = [(r["src"], r["bytes"], str(out / f"{r['utt_id']}.wav")) for r in rows]
    if args.workers <= 1:
        results = [_convert_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_convert_one, jobs, chunksize=64))

    manifest, failed = [], {}
    for r, (dst, dur, err) in zip(rows, results):
        text = (r["text"] or "").strip()
        if err or not text:
            failed[r["utt_id"]] = err or "no transcript"
            continue
        labels = []
        if r["gender"]:
            labels.append(f"gender: {r['gender']}")
        if train_keys:
            labels.append(f"sentence in CV train: {'yes' if sentence_key(text) in train_keys else 'no'}")
        manifest.append({"audio_filepath": dst, "text": text, "duration": round(dur, 3),
                         "utt_id": r["utt_id"], "cv_version": args.version, "labels": labels})
    write_manifest(manifest, Path(args.manifest))
    durs = [m["duration"] for m in manifest]
    report = {
        "corpus": f"Common Voice German {args.version}, split test",
        "release": args.release,
        "input": args.corpus_dir or args.parquet_glob, "text_field": "sentence",
        "rows_in_split": len(rows), "clips": len(manifest), "failed": failed,
        "hours": round(sum(durs) / 3600, 4),
        "words": sum(len(m["text"].split()) for m in manifest),
        "duration_s": {"mean": round(float(np.mean(durs)), 2), "min": round(min(durs), 2),
                       "max": round(max(durs), 2)},
        "train_sentences": len(train_keys),
        "sentence_in_train": sum("sentence in CV train: yes" in m["labels"] for m in manifest),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: v for k, v in report.items() if k != "failed"}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- Verbmobil

VM_PARTS = ("VM1", "VM2")


def read_vm_turn_list(path: Path) -> List[str]:
    """Turn ids of a BAS set list (``<turn-id> TAB <volume>`` per line)."""
    out = []
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            tok = line.split()
            if tok:
                out.append(tok[0])
    return out


def vm_ort_reference(par_path: Path) -> str:
    """ORT tier of a BAS Partitur, cleaned exactly like BAS-RVG1's ``ort`` reference."""
    sys.path.insert(0, str(REPO))
    from src.datasets.bas_rvg1 import BasRvg1Source

    # The RVG1 source keeps its tier parser as methods; they need no instance state.
    parser = BasRvg1Source.__new__(BasRvg1Source)
    ort, _tr2, _kan = parser._read_par_transcriptions(par_path)
    return ort or ""


def build_verbmobil(args) -> int:
    root = Path(args.root)
    sets = Path(args.sets_dir) if args.sets_dir else root / "VM1" / "doc" / "doc" / "SETS"
    audio_out = Path(args.audio_out)
    rows, drops, per_part = [], {}, {}
    for part in VM_PARTS:
        turns = read_vm_turn_list(sets / f"{part}_{args.set}")
        per_part[part] = {"listed": len(turns), "kept": 0, "hours": 0.0, "words": 0,
                          "speakers": set()}
        for turn in turns:
            dialog = turn[:5]
            wav = root / part / part / dialog / f"{turn}.wav"
            par = wav.with_suffix(".par")
            if not wav.is_file() or not par.is_file():
                drops[turn] = "missing wav/par"
                continue
            ref = vm_ort_reference(par)
            if not ref:
                drops[turn] = "empty ORT after cleaning"
                continue
            info = sf.info(str(wav))
            path, dur = wav, info.duration
            if info.samplerate != SAMPLE_RATE or info.channels != 1:
                data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                data = to_mono_16k(data, sr)
                path = audio_out / part / f"{turn}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), data, SAMPLE_RATE, subtype="PCM_16")
                dur = len(data) / SAMPLE_RATE
            speaker = turn.rsplit("_", 1)[-1]
            rows.append({
                "audio_filepath": str(path), "text": ref, "duration": round(dur, 3),
                "utt_id": f"verbmobil_{turn}", "turn_id": turn, "dialog_id": dialog,
                "speaker_id": speaker, "part": part, "labels": [f"part: {part}"],
            })
            pp = per_part[part]
            pp["kept"] += 1
            pp["hours"] += dur / 3600
            pp["words"] += len(ref.split())
            pp["speakers"].add(speaker)
    write_manifest(rows, Path(args.manifest))
    durs = [r["duration"] for r in rows]
    report = {
        "corpus": f"Verbmobil German (BAS), set {args.set} of 'Infos to VM Data Sets' v2.3",
        "root": str(root), "sets_dir": str(sets), "reference": "ORT tier, BAS-RVG1 ORT cleaning",
        "clips": len(rows), "hours": round(sum(durs) / 3600, 4),
        "words": sum(len(r["text"].split()) for r in rows),
        "speakers": len({r["speaker_id"] for r in rows}),
        "dialogs": len({r["dialog_id"] for r in rows}),
        "duration_s": {"mean": round(float(np.mean(durs)), 2), "min": round(min(durs), 2),
                       "max": round(max(durs), 2), "over_30s": int(sum(d > 30 for d in durs))},
        "parts": {k: {**{kk: (round(vv, 4) if isinstance(vv, float) else vv)
                         for kk, vv in v.items() if kk != "speakers"},
                      "speakers": len(v["speakers"])} for k, v in per_part.items()},
        "dropped": drops,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: v for k, v in report.items() if k != "dropped"}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="corpus", required=True)

    t = sub.add_parser("tuda", help="Tuda-De v4 test split")
    t.add_argument("--root", required=True, help="extracted package (contains test/ and train/)")
    t.add_argument("--mics", default="Kinect-RAW,Yamaha")
    t.add_argument("--audio-out", required=True, help="writable dir for converted WAVs")
    t.add_argument("--manifest-dir", default=str(MANIFEST_DIR))
    t.add_argument("--report", default=str(MANIFEST_DIR / "tuda_test_report.json"))
    t.add_argument("--min-duration", type=float, default=1.0)
    t.add_argument("--min-peak", type=float, default=1e-4)
    t.add_argument("--audit-ids", default=None,
                   help="leak-audit list: test recordings whose sentence is in a model's training data")
    t.add_argument("--audit-label", default="sentence in SelOSS training data")
    t.set_defaults(fn=build_tuda)

    c = sub.add_parser("cv", help="Common Voice German test split")
    c.add_argument("--version", required=True, help="short release tag stored in every row, e.g. cv26")
    c.add_argument("--release", default="", help="full release name for the report, e.g. "
                   "'Common Voice Scripted Speech 26.0 German (Mozilla Data Collective)'")
    c.add_argument("--corpus-dir", default=None, help="<release>/de with test.tsv and clips/")
    c.add_argument("--parquet-glob", default=None, help="parquet store with audio + text")
    c.add_argument("--audio-out", required=True)
    c.add_argument("--manifest", default=str(MANIFEST_DIR / "cv_de_test.jsonl"))
    c.add_argument("--report", default=str(MANIFEST_DIR / "cv_de_test_report.json"))
    c.add_argument("--train-sentences", default=None)
    c.add_argument("--workers", type=int, default=8)
    c.set_defaults(fn=build_cv)

    v = sub.add_parser("verbmobil", help="Verbmobil German, BAS TEST set (VM1 + VM2)")
    v.add_argument("--root", required=True, help="BAS release with VM1/VM1/<dialog>/ and VM2/VM2/<dialog>/")
    v.add_argument("--sets-dir", default=None, help="default <root>/VM1/doc/doc/SETS")
    v.add_argument("--set", default="TEST", choices=["TEST", "DEV", "TRAIN"])
    v.add_argument("--audio-out", required=True, help="writable dir for converted WAVs")
    v.add_argument("--manifest", default=str(MANIFEST_DIR / "verbmobil_test.jsonl"))
    v.add_argument("--report", default=str(MANIFEST_DIR / "verbmobil_test_report.json"))
    v.set_defaults(fn=build_verbmobil)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
