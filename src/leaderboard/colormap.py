"""Green->yellow->red color mapping for WER cells.

Coloring is *per column*: each column gets its own ``vmin``/``vmax`` so the
gradient spreads across the actually-observed range. ``vmax`` is the 90th
percentile (not the max) so a few broken runs (WER ~ 1.0) don't squash every
other model into green.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

# Gradient anchors: green (good) -> yellow (mid) -> red (bad).
_GREEN = (0, 200, 0)
_YELLOW = (220, 200, 0)
_RED = (220, 0, 0)
_MISSING = (90, 90, 90)  # neutral grey for absent cells


def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile (``pct`` in 0..100), numpy-free."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def column_bounds(values: List[Optional[float]]) -> Tuple[float, float]:
    """Return ``(vmin, vmax)`` for a column, robust to outliers."""
    present = [v for v in values if v is not None]
    if not present:
        return 0.0, 1.0
    vmin = min(present)
    vmax = percentile(present, 90)
    if vmax <= vmin:
        vmax = max(present)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))  # type: ignore[return-value]


def wer_rgb(wer: Optional[float], vmin: float, vmax: float) -> Tuple[int, int, int]:
    """Map a WER value to an ``(r, g, b)`` color within the column bounds."""
    if wer is None:
        return _MISSING
    if vmax <= vmin:
        t = 0.0
    else:
        t = (wer - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return _lerp(_GREEN, _YELLOW, t / 0.5)
    return _lerp(_YELLOW, _RED, (t - 0.5) / 0.5)


def rich_style(wer: Optional[float], vmin: float, vmax: float, is_best: bool = False) -> str:
    """Return a rich style string for a WER cell."""
    if wer is None:
        return "dim"
    r, g, b = wer_rgb(wer, vmin, vmax)
    style = f"rgb({r},{g},{b})"
    if is_best:
        style = "bold " + style
    return style


def text_on(rgb: Tuple[int, int, int]) -> str:
    """Pick black or white text for readability on a colored background."""
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 140 else "#ffffff"
