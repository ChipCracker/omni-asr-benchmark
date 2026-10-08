#!/usr/bin/env python3
"""Drop excluded utterances from stored result JSONs and re-aggregate -- no model runs.

Used for the mod9 English exclusion (manifests/mod9_exclude_english.txt): every
``<model>__mod9.json`` loses the per-sample entries of the excluded clips, the per-reference
aggregates are recomputed with the current scorer (``rescore_results.rescore``), ``num_samples``
is updated and the exclusion is recorded under ``"excluded"``. Originals are copied to
``<backup-dir>/`` first. The leaderboard recomputes its numbers from the remaining per-sample
texts, so it follows automatically.

  python scripts/filter_results.py --dataset mod9 --exclude manifests/mod9_exclude_english.txt \\
      --dirs results results_tuda_cv --backup-dir results_pre_mod9filter
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

_spec = importlib.util.spec_from_file_location("rescore_results", ROOT / "scripts" / "rescore_results.py")
rescore_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore_results)


def _utt_id(sample: dict) -> str:
    md = sample.get("metadata") or {}
    if md.get("utt_id"):
        return str(md["utt_id"])
    return Path(sample.get("audio_path") or "").stem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exclude", required=True, help="text file, first column = utt_id, # = comment")
    ap.add_argument("--dirs", nargs="+", default=["results", "results_tuda_cv"])
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    excl = {ln.split("\t", 1)[0].strip() for ln in Path(a.exclude).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")}
    n_files = 0
    for d in a.dirs:
        for path in sorted(Path(d).glob(f"*__{a.dataset}.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            samples = data.get("per_sample") or []
            keep = [s for s in samples if _utt_id(s) not in excl]
            dropped = len(samples) - len(keep)
            before = {k: (v.get("wer") if isinstance(v, dict) else None) for k, v in (data.get("results") or {}).items()}
            if a.check:
                print(f"{path}: {dropped} of {len(samples)} would be dropped")
                continue
            if a.backup_dir:
                bd = Path(a.backup_dir) / d
                bd.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, bd / path.name)
            data["per_sample"] = keep
            data["num_samples"] = len(keep)
            data["excluded"] = {"file": str(Path(a.exclude).name), "dropped": dropped, "rule": "mod9 English references"}
            rescore_results.rescore(data, write=True)
            after = {k: (v.get("wer") if isinstance(v, dict) else None) for k, v in (data.get("results") or {}).items()}
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            n_files += 1
            prim = data.get("primary_reference") or next(iter(after), None)
            print(f"{path.name}: -{dropped} samples, WER {before.get(prim)} -> {after.get(prim)}")
    print(f"{n_files} files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
