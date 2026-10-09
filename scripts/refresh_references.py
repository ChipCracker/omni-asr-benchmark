#!/usr/bin/env python3
"""Replace the references inside stored result JSONs and re-score -- no model runs.

Two kinds of reference fixes need this (2026-10-09): a rebuilt manifest with corrected
texts (``kiel_spon``: the kiel_ml normaliser kept the ``<h"as>`` hesitation marker as a
pseudo-word and split words around in-word markers, 617 of 3 609 turns) and a clean-up rule
applied to the stored text (``mod9``: annotator marker words, see
``src/datasets/mod9_markers.py``). In both cases the hypotheses stay as they are; only the
reference text per utterance changes, and clips that lose their reference (empty after the
clean-up, or listed in an exclusion file) are dropped. Aggregates are recomputed with the
current scorer (``rescore_results.rescore``); originals go to ``--backup-dir`` first.

  python scripts/refresh_references.py --dataset kiel_spon --manifest manifests/kiel_spon.jsonl \\
      --dirs results results_tuda_cv --backup-dir results_pre_kielfix
  python scripts/refresh_references.py --dataset mod9 --clean mod9 \\
      --exclude manifests/mod9_exclude_markers.txt --dirs results results_tuda_cv \\
      --backup-dir results_pre_mod9markers
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _keys(sample: dict) -> list:
    """Kandidaten-Schluessel eines Samples: utt_id (falls vorhanden) und der Audio-Dateistamm."""
    ex = sample.get("extra") or {}
    keys = [str(ex["utt_id"])] if ex.get("utt_id") else []
    ap = sample.get("audio_path") or sample.get("audio_filepath") or ""
    if ap:
        keys.append(Path(ap).stem)
    return keys


def _load_manifest(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Manifest-Zeilen sind ueber utt_id UND Audio-Dateistamm erreichbar (aeltere Ergebnisdateien
            # tragen keine utt_id, Kiel-Manifeste haben utt_id "kiel_<id>" bei Dateistamm "<id>").
            if r.get("utt_id"):
                out[str(r["utt_id"])] = r["text"]
            out[Path(r["audio_filepath"]).stem] = r["text"]
    return out


def _load_exclude(path: Path) -> set:
    return {ln.split("\t", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", help="JSONL with utt_id + text: new reference per utterance")
    ap.add_argument("--clean", choices=["mod9"], help="clean-up rule applied to the stored reference")
    ap.add_argument("--exclude", nargs="*", default=[], help="text files, first column = utt_id")
    ap.add_argument("--keep-empty", action="store_true", help="keep clips whose new reference is empty")
    ap.add_argument("--dirs", nargs="+", default=["results", "results_tuda_cv"])
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if not a.manifest and not a.clean:
        ap.error("--manifest or --clean required")

    rescore = _load_script("rescore_results")
    new_text = _load_manifest(Path(a.manifest)) if a.manifest else None
    cleaner = None
    if a.clean == "mod9":
        from src.datasets.mod9_markers import clean_reference
        cleaner = clean_reference
    excluded = set()
    for p in a.exclude:
        excluded |= _load_exclude(Path(p))

    note = {"manifest": a.manifest, "clean": a.clean, "exclude": a.exclude, "date": "2026-10-09"}
    total_changed = total_dropped = n_files = 0
    for d in a.dirs:
        for f in sorted(Path(d).glob(f"*__{a.dataset}.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            samples = data.get("per_sample") or []
            pr = data.get("primary_reference", "ref")
            keep, changed, dropped = [], 0, 0
            for s in samples:
                keys = _keys(s)
                uid = next((k for k in keys if new_text is not None and k in new_text), None) \
                    or next((k for k in keys if k in excluded), None) or (keys[0] if keys else "")
                refs = s.get("references")
                if refs is None:  # legacy v1 schema
                    key = f"{pr}_reference"
                    old = s.get(key, "")
                else:
                    old = refs.get(pr, "")
                new = old
                if new_text is not None:
                    if uid not in new_text:
                        dropped += 1
                        continue
                    new = new_text[uid]
                if cleaner is not None:
                    new = cleaner(new)
                if uid in excluded or (not a.keep_empty and not new.strip()):
                    dropped += 1
                    continue
                if new != old:
                    changed += 1
                    if refs is None:
                        s[f"{pr}_reference"] = new
                    else:
                        refs[pr] = new
                keep.append(s)
            old_wer = (data.get("results") or {}).get(pr, {}).get("wer")
            if not a.check:
                if a.backup_dir:
                    bd = Path(a.backup_dir); bd.mkdir(parents=True, exist_ok=True)
                    dest = bd / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
                data["per_sample"] = keep
                data["num_samples"] = len(keep)
                rescore.rescore(data, write=True)
                data.setdefault("references_refreshed", []).append(
                    {**note, "changed": changed, "dropped": dropped})
                f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                new_wer = (data.get("results") or {}).get(pr, {}).get("wer")
            else:
                new_wer = None
            n_files += 1; total_changed += changed; total_dropped += dropped
            w0 = f"{100*old_wer:5.2f}" if old_wer is not None else "  n/a"
            w1 = f"{100*new_wer:5.2f}" if new_wer is not None else "  n/a"
            print(f"{f.name:80s} geändert {changed:5d} entfernt {dropped:4d}  WER {w0} -> {w1}")
    print(f"{n_files} Dateien, {total_changed} Referenzen geändert, {total_dropped} Clips entfernt"
          + ("  (check, nichts geschrieben)" if a.check else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
