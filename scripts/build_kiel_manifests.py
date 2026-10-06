#!/usr/bin/env python3
"""Build the NeMo manifests for the Kiel Corpus (IPDS Kiel): ``kiel_read`` and ``kiel_spon``.

Input is the JSONL written by the ``kiel_ml`` pipeline (one row per utterance,
``audio_path`` relative to the raw ``Kiel_Corpus/`` tree, 16 kHz mono WAV per
speaker turn, ``text`` = normalised orthography with the Kiel markers removed).
Two leaderboard datasets come out of it, the WHOLE corpus each (no model on the
leaderboard trains on the Kiel Corpus, and its ML splits are not needed here):

* ``kiel_read`` -- KCRead: PhonDat90/92 read sentences and texts (Berlin and
  Marburg sentences, "Nordwind und Sonne", "Buttergeschichte", Erlangen/Siemens
  train-information prompts). 5.7 h, 5 227 utterances, 53 speakers.
* ``kiel_spon`` -- KCSpon: spontaneous appointment dialogues (Verbmobil
  recordings made in Kiel, plus the VMaddOrt/VMaddSeg additions) and the
  VideoTask dialogues. 10.0 h, ~3 600 turns, 64 speakers. One clip per speaker
  turn, close-talk microphone, the partner is on the other channel's file.
  Turns without a word (Kiel ".") or shorter than ``--min-duration`` are dropped.

Reference ``text`` as in the Kiel orthography (old spelling such as "daß",
apostrophes "hab'"); the scorer's normalisation removes case and punctuation.
Labels per row: ``subcorpus``, ``read_corpus`` (KCRead) / ``dialog`` style, the
kiel_ml split and whether hand-segmented realised phones exist
(``realized: yes/no``) -- the latter marks the subset usable for phone-level
scoring later on.

  python scripts/build_kiel_manifests.py --jsonl /path/kiel_corpus.jsonl \\
      --audio-root /nfs1/scratch/staff/witzlch/kiel-bench/Kiel_Corpus
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True, help="kiel_ml manifest (kiel_corpus.jsonl)")
    ap.add_argument("--audio-root", required=True,
                    help="absolute path of the Kiel_Corpus/ tree on the benchmark host (kiz0)")
    ap.add_argument("--manifest-dir", default=str(MANIFEST_DIR))
    ap.add_argument("--report", default=str(MANIFEST_DIR / "kiel_report.json"))
    ap.add_argument("--check-local-root", default=None,
                    help="local Kiel_Corpus/ tree to verify that every referenced WAV exists")
    ap.add_argument("--min-duration", type=float, default=0.3,
                    help="drop turns shorter than this (KCSpon has a few 50-ms fragments)")
    a = ap.parse_args(argv)

    out = {"kiel_read": [], "kiel_spon": []}
    stats = {k: collections.Counter() for k in out}
    speakers = {k: set() for k in out}
    missing = []
    dropped = collections.Counter()
    with open(a.jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            ds = "kiel_read" if r["corpus"] == "KCRead" else "kiel_spon"
            # Turns ohne Wort (nur ".") oder kuerzer als min_duration: kein Bewertungsgegenstand.
            if not any(ch.isalpha() for ch in text) or float(r["duration"]) < a.min_duration:
                dropped[ds] += 1
                continue
            if a.check_local_root and not (Path(a.check_local_root) / r["audio_path"]).is_file():
                missing.append(r["audio_path"])
                continue
            labels = [f"subcorpus: {r['subcorpus']}", f"split: {r.get('split', '?')}",
                      f"realized: {'yes' if r.get('has_realized') else 'no'}"]
            if r.get("read_corpus"):
                labels.append(f"read_corpus: {r['read_corpus']}")
            spk = (r.get("speaker") or {})
            row = {
                "audio_filepath": str(Path(a.audio_root) / r["audio_path"]),
                "text": text,
                "duration": round(float(r["duration"]), 4),
                "utt_id": f"kiel_{r['id']}",
                "speaker_id": spk.get("canonical_id") or spk.get("id"),
                "subcorpus": r["subcorpus"],
                "style": r.get("style"),
                "labels": labels,
            }
            if r.get("read_corpus"):
                row["read_corpus"] = r["read_corpus"]
            if r.get("dialog"):
                row["dialog_id"] = r["dialog"]
            out[ds].append(row)
            st = stats[ds]
            st["clips"] += 1
            st["hours"] += row["duration"] / 3600
            st["words"] += len(text.split())
            st[f"sub:{r['subcorpus']}"] += 1
            if r.get("has_realized"):
                st["realized"] += 1
            speakers[ds].add(row["speaker_id"])
    mdir = Path(a.manifest_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    report = {"corpus": "The Kiel Corpus of Read/Spontaneous Speech (IPDS Kiel), via kiel_ml",
              "audio_root": a.audio_root, "source_jsonl": str(Path(a.jsonl).resolve()),
              "missing_audio": len(missing), "dropped_empty_or_short": dict(dropped),
              "min_duration": a.min_duration, "datasets": {}}
    for ds, rows in out.items():
        p = mdir / f"{ds}.jsonl"
        with p.open("w", encoding="utf-8") as fo:
            for row in rows:
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        st = stats[ds]
        durs = [r["duration"] for r in rows]
        report["datasets"][ds] = {
            "manifest": str(p), "clips": int(st["clips"]), "hours": round(st["hours"], 3),
            "words": int(st["words"]), "speakers": len(speakers[ds]), "realized_phones": int(st["realized"]),
            "subcorpora": {k[4:]: int(v) for k, v in st.items() if k.startswith("sub:")},
            "duration_s": {"mean": round(sum(durs) / max(1, len(durs)), 2), "min": round(min(durs), 2),
                           "max": round(max(durs), 2), "over_30s": sum(d > 30 for d in durs)},
        }
        print(f"{ds}: {st['clips']} clips, {st['hours']:.2f} h, {st['words']} words, {len(speakers[ds])} speakers -> {p}")
    with open(a.report, "w", encoding="utf-8") as fo:
        json.dump(report, fo, ensure_ascii=False, indent=1)
    if missing:
        print(f"WARNUNG: {len(missing)} Audiodateien fehlen lokal, z. B. {missing[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
