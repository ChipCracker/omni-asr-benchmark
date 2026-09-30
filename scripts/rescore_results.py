#!/usr/bin/env python3
"""Re-score stored result JSONs with the current ``normalize_text`` — no model runs.

Every result file keeps the hypotheses and references per utterance, so a change
to the scoring normalisation (e.g. ß -> ss) only needs a re-computation:

* per sample: WER/CER for every reference (``compute_single_sample_metrics``)
* per reference: the aggregate (``compute_asr_metrics`` over the samples that
  have a non-empty reference, the same rule as the benchmark runner)

Both schemas are rewritten in place and stay in their schema: v2
(``references``/``metrics`` per sample) and legacy v1 (``<ref>_reference``,
``<ref>_wer``, ``<ref>_cer``; aggregates under ``<ref>_reference``). All other
fields (speed, hardware, timestamps, raw hypotheses) are left untouched. The
leaderboard's numbers track and bootstrap intervals are computed from the
per-sample texts at render time, so they follow automatically.

Examples:
    python scripts/rescore_results.py --check            # compare, write nothing
    python scripts/rescore_results.py                    # results/, speed/, failed_evidence/
    python scripts/rescore_results.py --dirs results/speed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.metrics import compute_asr_metrics, compute_single_sample_metrics  # noqa: E402

_SKIP_NAMES = {"leaderboard.json"}
_V1_REFS = ("dialect", "ort", "kan")
_AGG_KEYS = ("wer", "cer", "substitutions", "deletions", "insertions", "num_samples")


def _collect(samples):
    """-> {ref_name: (hyps, refs)} and new per-sample metrics, v2 or v1 layout."""
    groups, per = {}, []
    for s in samples:
        hyp = s.get("hypothesis") or ""
        if "references" in s or "metrics" in s:                       # v2
            refs = {k: v for k, v in (s.get("references") or {}).items()}
        else:                                                         # v1
            refs = {r: s.get(f"{r}_reference") for r in _V1_REFS if f"{r}_reference" in s}
        m = {}
        for name, ref in refs.items():
            if not ref:
                continue
            m[name] = compute_single_sample_metrics(hyp, ref)
            groups.setdefault(name, ([], []))
            groups[name][0].append(hyp)
            groups[name][1].append(ref)
        per.append(m)
    return groups, per


def rescore(data: dict, write: bool):
    """Returns (max |delta| over aggregate WER/CER, per-sample metrics changed)."""
    v1 = data.get("schema_version", 1) < 2
    samples = data.get("per_sample") or []
    groups, per = _collect(samples)
    max_delta, changed = 0.0, 0
    for s, m in zip(samples, per):
        for name, met in m.items():
            if v1:
                old = (s.get(f"{name}_wer"), s.get(f"{name}_cer"))
                if write:
                    s[f"{name}_wer"], s[f"{name}_cer"] = met["wer"], met["cer"]
            else:
                slot = s.setdefault("metrics", {}).setdefault(name, {})
                old = (slot.get("wer"), slot.get("cer"))
                if write:
                    slot["wer"], slot["cer"] = met["wer"], met["cer"]
            changed += old != (met["wer"], met["cer"])
    results = data.get("results") or {}
    for name, (hyps, refs) in groups.items():
        key = f"{name}_reference" if v1 else name
        if key not in results or not isinstance(results[key], dict):
            continue
        agg = compute_asr_metrics(hyps, refs)
        for k in ("wer", "cer"):
            if results[key].get(k) is not None:
                max_delta = max(max_delta, abs(results[key][k] - agg[k]))
        if write:
            results[key].update({k: agg[k] for k in _AGG_KEYS})
    return max_delta, changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+",
                    default=["results", "results/speed", "results/failed_evidence"])
    ap.add_argument("--check", action="store_true", help="only report deltas, write nothing")
    a = ap.parse_args()

    n_files = 0
    for d in a.dirs:
        for path in sorted(Path(d).glob("*.json")):
            if path.name in _SKIP_NAMES:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"skip {path} (unreadable: {e})")
                continue
            if not isinstance(data, dict) or "results" not in data or not data.get("per_sample"):
                print(f"skip {path} (no per-sample result)")
                continue
            delta, changed = rescore(data, write=not a.check)
            n_files += 1
            tag = "v1" if data.get("schema_version", 1) < 2 else "v2"
            print(f"{path} [{tag}] max |d agg| {delta:.5f}, per-sample changed {changed}")
            if not a.check:
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'checked' if a.check else 'rescored'} {n_files} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
