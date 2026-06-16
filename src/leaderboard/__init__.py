"""Color-coded ASR leaderboard (HuggingFace Open ASR style).

Reads benchmark result JSONs, aggregates them into a model x (dataset, reference)
matrix, and renders a green->red color-coded table to the terminal (rich), HTML,
and Markdown.
"""

from .aggregate import Leaderboard, Row, build_leaderboard, load_results
from .render import render_html, render_markdown, render_terminal

__all__ = [
    "Leaderboard",
    "Row",
    "build_leaderboard",
    "load_results",
    "render_terminal",
    "render_html",
    "render_markdown",
]
