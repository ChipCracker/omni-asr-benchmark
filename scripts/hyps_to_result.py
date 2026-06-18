#!/usr/bin/env python3
"""Convert an external decoder's hypotheses JSONL into a v2 BenchmarkResult.

For models that can't run inside this framework (custom repos / venvs) but can
emit one ``{audio_filepath, text, hyp, ...}`` JSON line per utterance. WER/CER
are recomputed here with the shared metrics so the result drops straight into
the leaderboard.

Example:
    python scripts/hyps_to_result.py --hyps hyps_ksof.jsonl \\
        --model "XEUS-RNNT (PARLO)" --dataset ksof --ref-name ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.metrics import compute_asr_metrics, compute_single_sample_metrics  # noqa: E402
from src.benchmark.result import BenchmarkResult, SampleResult  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True, help="JSONL with audio_filepath/text/hyp per line.")
    ap.add_argument("--model", required=True, help="Display name for the leaderboard row.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ref-name", default="ref", help="Reference key (e.g. ref, ort).")
    ap.add_argument("--language", default="deu_Latn")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.hyps, encoding="utf-8") if l.strip()]
    hyps, refs, per = [], [], []
    for i, r in enumerate(rows):
        hyp = (r.get("hyp") or "").strip()
        ref = (r.get("ref") or r.get("text") or "").strip()
        hyps.append(hyp)
        refs.append(ref)
        m = compute_single_sample_metrics(hyp, ref) if ref else {"wer": None, "cer": None}
        per.append(SampleResult(
            index=r.get("index", i),
            audio_path=r.get("audio_filepath", ""),
            hypothesis=hyp,
            duration=r.get("duration") or 0.0,
            references={a.ref_name: ref},
            metrics={a.ref_name: m},
            raw_hypothesis=r.get("hyp_raw"),
            extra={"utt_id": r.get("utt_id")} if r.get("utt_id") else None,
        ))

    agg = compute_asr_metrics(hyps, refs)
    res = BenchmarkResult(
        model=a.model, dataset=a.dataset, language=a.language,
        num_samples=len(rows), references=[a.ref_name], primary_reference=a.ref_name,
        results={a.ref_name: agg}, per_sample=per,
    )
    model_safe = a.model.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    out = a.out or f"results/{model_safe}__{a.dataset}.json"
    res.save(Path(out))
    wer = agg.get("wer")
    print(f"wrote {out}  ({len(rows)} utts, {a.ref_name} WER {wer * 100:.2f}%)" if wer is not None
          else f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
