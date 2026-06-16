#!/usr/bin/env python3
"""Migrate legacy (v1) result JSONs to the v2 benchmark schema, in place.

The leaderboard already reads v1 results transparently (via
``BenchmarkResult.from_dict``); this tool *persists* the conversion so the files
on disk become canonical v2. The conversion is lossless — named references,
per-sample metrics, raw hypotheses and multi-speaker info are all preserved — and
each original is backed up to ``<results-dir>/v1_backup/`` first.

Examples:
    python scripts/migrate_results.py --dry-run
    python scripts/migrate_results.py
    python scripts/migrate_results.py --no-backup
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.result import BenchmarkResult  # noqa: E402

_SKIP_NAMES = {"leaderboard.json"}


def _is_v1(data: dict) -> bool:
    return isinstance(data, dict) and "results" in data and data.get("schema_version", 1) < 2


def _verify_roundtrip(original: dict, migrated: dict) -> bool:
    """Confirm the v2 file still parses and primary metrics are unchanged."""
    reparsed = BenchmarkResult.from_dict(migrated)
    source = BenchmarkResult.from_dict(original)
    for ref, metrics in source.results.items():
        sm = metrics.get("wer")
        rm = reparsed.results.get(ref, {}).get("wer")
        if sm is None and rm is None:
            continue
        if sm is None or rm is None or abs(sm - rm) > 1e-9:
            return False
    return reparsed.num_samples == source.num_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate legacy v1 result JSONs to the v2 schema (in place).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true", help="Report only, write nothing.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up originals.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}", file=sys.stderr)
        return 1

    backup_dir = args.results_dir / "v1_backup"
    migrated = skipped = failed = 0

    for path in sorted(args.results_dir.glob("*.json")):
        if path.name.startswith(".") or path.name in _SKIP_NAMES:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict) or "results" not in data or "model" not in data:
            continue

        if not _is_v1(data):
            print(f"  skip (already v2): {path.name}")
            skipped += 1
            continue

        result = BenchmarkResult.from_dict(data)
        result.schema_version = 2  # persist as canonical v2
        new_data = result.to_dict()

        if not _verify_roundtrip(data, new_data):
            print(f"  FAILED round-trip check, leaving untouched: {path.name}", file=sys.stderr)
            failed += 1
            continue

        refs = ", ".join(result.references)
        if args.dry_run:
            print(f"  would migrate: {path.name}  (refs: {refs}, primary: {result.primary_reference})")
            migrated += 1
            continue

        if not args.no_backup:
            backup_dir.mkdir(exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
        path.write_text(json.dumps(new_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  migrated: {path.name}  (refs: {refs}, primary: {result.primary_reference})")
        migrated += 1

    verb = "would migrate" if args.dry_run else "migrated"
    print(f"\n{verb}: {migrated}, skipped (already v2): {skipped}, failed: {failed}")
    if migrated and not args.dry_run and not args.no_backup:
        print(f"Originals backed up to: {backup_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
