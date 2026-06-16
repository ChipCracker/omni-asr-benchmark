#!/usr/bin/env python3
"""Render the color-coded ASR leaderboard from result JSONs.

Reads every benchmark result in ``results/`` (both the new v2 schema and the
legacy v1 schema), prints a green->red color-coded table to the terminal, and
optionally exports HTML and/or Markdown.

Examples:
    python scripts/leaderboard.py
    python scripts/leaderboard.py --export html,md
    python scripts/leaderboard.py --exclude-failed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.leaderboard import (  # noqa: E402
    build_leaderboard,
    load_results,
    render_html,
    render_markdown,
    render_terminal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the color-coded ASR leaderboard from result JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--export",
        type=str,
        default="",
        help="Comma-separated formats to write to results-dir: html, md.",
    )
    parser.add_argument(
        "--exclude-failed",
        action="store_true",
        help="Drop runs whose primary WER >= 0.99 (broken runs).",
    )
    parser.add_argument("--no-rtfx", action="store_true", help="Hide the RTFx column.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}", file=sys.stderr)
        return 1

    results = load_results(args.results_dir)
    if not results:
        print(f"No benchmark result JSONs found in {args.results_dir}", file=sys.stderr)
        return 1

    lb = build_leaderboard(results, exclude_failed=args.exclude_failed)
    show_rtfx = not args.no_rtfx
    render_terminal(lb, show_rtfx=show_rtfx)

    formats = {f.strip().lower() for f in args.export.split(",") if f.strip()}
    if "html" in formats:
        out = args.results_dir / "leaderboard.html"
        out.write_text(render_html(lb, show_rtfx=show_rtfx), encoding="utf-8")
        print(f"Wrote {out}")
    if "md" in formats or "markdown" in formats:
        out = args.results_dir / "leaderboard.md"
        out.write_text(render_markdown(lb, show_rtfx=show_rtfx), encoding="utf-8")
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
