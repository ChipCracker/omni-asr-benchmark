#!/usr/bin/env python3
"""List the mod9 clips whose reference is predominantly English (excluded from the leaderboard).

mod9 is scraped German talk/interview audio with a sizeable block of English-language
material (German-lesson videos narrated in English, interviews in English). Those clips do
not measure German recognition -- a German-only model gets 90 % WER there, the multilingual
LLM decoders about 45 % -- so they are left out of the dataset (decision 2026-10-08).

Rule (on the reference text, lower-cased, annotation markers removed): count English and
German stop words; a clip is English when en / (en + de) >= 0.5 and at least 3 English stop
words occur. Dialect tokens that collide with English ("i", "a", "is", "so", "in", "do") are
deliberately NOT in the English list, Bavarian/Austrian clips stay in. 419 of 17 398 clips
(2.4 %, 2.3 h) match; the borderline cases just below the threshold are English-narrated
German lessons with dictated German phrases.

  python scripts/mod9_english_exclude.py --manifest /nfs1/scratch/staff/witzlch/tuda-bench/mod9/mod9.jsonl

Output: manifests/mod9_exclude_english.txt (one utt_id per line, with the ratio and the start
of the reference as a comment), read by src/datasets/mod9.py and scripts/filter_results.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"
EN = set("the and you to of it that this for on are we they have with be your as what there can "
         "will would just if or not about from but i'm it's they're you're don't was were my me our".split())
DE = set("der die das und ist ich nicht es zu den ein eine dass mit auch sich auf für wir sie er aber "
         "da was noch dann wie man hat nur bei haben wenn oder sind von dem im also ja des i is a".split())
MARKERS = {"musik", "fremdsprache", "geräusch", "leeresaudio", "unverständlich"}


def english_ratio(text: str):
    words = [w for w in re.sub(r"[^\wäöüß' ]+", " ", text.lower()).split() if w not in MARKERS]
    en = sum(w in EN for w in words)
    de = sum(w in DE for w in words)
    return (en / (en + de) if en + de else 0.0), en, de, len(words)


def is_english(text: str, threshold: float = 0.5, min_en: int = 3) -> bool:
    ratio, en, _, _ = english_ratio(text)
    return ratio >= threshold and en >= min_en


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="mod9 NeMo manifest (utt_id, text, duration)")
    ap.add_argument("--out", default=str(MANIFEST_DIR / "mod9_exclude_english.txt"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-en", type=int, default=3)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    sel = []
    for r in rows:
        ratio, en, de, n = english_ratio(r["text"])
        if ratio >= a.threshold and en >= a.min_en:
            sel.append((r["utt_id"], ratio, n, float(r.get("duration") or 0), r["text"]))
    sel.sort()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# mod9 clips whose reference is predominantly English: excluded from the leaderboard (2026-10-08)\n")
        f.write(f"# rule: en/(en+de) stop words >= {a.threshold} and >= {a.min_en} English stop words; "
                f"{len(sel)} of {len(rows)} clips, {sum(s[2] for s in sel)} words, {sum(s[3] for s in sel)/3600:.2f} h\n")
        f.write("# utt_id\tratio\treference (start)\n")
        for utt, ratio, n, dur, text in sel:
            f.write(f"{utt}\t{ratio:.2f}\t{text[:80]}\n")
    print(f"{len(sel)} of {len(rows)} clips -> {a.out}")
    return 0


def load_exclude(path: str | Path = MANIFEST_DIR / "mod9_exclude_english.txt") -> set:
    """utt_ids of the excluded clips (empty set when the file is missing)."""
    p = Path(path)
    if not p.is_file():
        return set()
    return {ln.split("\t", 1)[0].strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")}


if __name__ == "__main__":
    raise SystemExit(main())
