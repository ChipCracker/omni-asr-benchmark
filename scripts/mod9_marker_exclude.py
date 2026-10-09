#!/usr/bin/env python3
"""Build ``manifests/mod9_exclude_markers.txt``: mod9 clips with no scorable reference.

A clip is listed when its reference is empty after ``mod9_markers.clean_reference`` (marker
words only: ``musik musik``, ``fremdsprache fremdsprache fremdsprache``, ``leeresaudio`` ...) or
is a ``dies ist blindtext`` placeholder. Applied together with the English exclusion by
``src/datasets/mod9.py`` (new runs) and ``scripts/refresh_references.py`` (stored results).

  python scripts/mod9_marker_exclude.py --manifest /nfs1/scratch/staff/witzlch/mod9-bench/manifest.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.mod9_markers import clean_reference, is_blindtext  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="/nfs1/scratch/staff/witzlch/mod9-bench/manifest.jsonl")
    ap.add_argument("--out", default=str(ROOT / "manifests" / "mod9_exclude_markers.txt"))
    a = ap.parse_args(argv)
    rows = []
    stats = collections.Counter()
    with open(a.manifest, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            uid = r.get("utt_id") or Path(r["audio_filepath"]).stem
            text = r.get("text", "")
            stats["clips"] += 1
            if is_blindtext(text):
                rows.append((uid, "blindtext", text)); stats["blindtext"] += 1
            elif not clean_reference(text).strip():
                rows.append((uid, "nur_marker", text)); stats["nur_marker"] += 1
            else:
                before, after = len(text.split()), len(clean_reference(text).split())
                stats["marker_woerter_entfernt"] += before - after
    with open(a.out, "w", encoding="utf-8") as fo:
        fo.write("# mod9 clips without a scorable reference (scripts/mod9_marker_exclude.py, 2026-10-09)\n")
        fo.write(f"# {stats['nur_marker']} marker-only, {stats['blindtext']} blindtext of {stats['clips']} clips; "
                 f"{stats['marker_woerter_entfernt']} marker words removed from the remaining references\n")
        fo.write("# utt_id\treason\treference\n")
        for uid, why, text in rows:
            fo.write(f"{uid}\t{why}\t{text[:80]}\n")
    print(dict(stats), "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
