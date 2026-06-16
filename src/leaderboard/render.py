"""Render a :class:`Leaderboard` to the terminal, HTML, and Markdown."""

from __future__ import annotations

import html as _html
from typing import Dict, List, Optional, Tuple

from .aggregate import Column, Leaderboard
from .colormap import column_bounds, rich_style, text_on, wer_rgb


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "–"


def _fmt_rtfx(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "–"


def _column_header(column: Column, multi_dataset: bool) -> str:
    dataset, ref = column
    return f"{dataset}\n{ref.upper()}" if multi_dataset else ref.upper()


def _column_stats(lb: Leaderboard) -> Dict[Column, Tuple[float, float, Optional[float]]]:
    """Per column: (vmin, vmax, best_wer)."""
    stats: Dict[Column, Tuple[float, float, Optional[float]]] = {}
    for col in lb.columns:
        values = [row.cells[col].wer for row in lb.rows if col in row.cells]
        vmin, vmax = column_bounds(values)
        present = [v for v in values if v is not None]
        best = min(present) if present else None
        stats[col] = (vmin, vmax, best)
    # Average column shares the same idea over the average values.
    avgs = [row.average_wer for row in lb.rows if row.average_wer is not None]
    vmin, vmax = column_bounds(avgs)
    stats[("__avg__", "")] = (vmin, vmax, min(avgs) if avgs else None)
    return stats


# --------------------------------------------------------------------------- #
# Terminal (rich)
# --------------------------------------------------------------------------- #
def render_terminal(lb: Leaderboard, console=None, show_rtfx: bool = True) -> None:
    """Print the color-coded leaderboard to the terminal."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box

    console = console or Console()
    if not lb.rows:
        console.print("[yellow]No benchmark results found.[/yellow]")
        return

    multi = len(lb.datasets) > 1
    stats = _column_stats(lb)

    title = "ASR Leaderboard — WER by dataset/reference (green = better)"
    table = Table(title=title, box=box.ROUNDED, header_style="bold", title_style="bold")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Model", justify="left", no_wrap=True)
    for col in lb.columns:
        table.add_column(_column_header(col, multi), justify="right")
    table.add_column("Avg WER", justify="right", style="bold")
    if show_rtfx:
        table.add_column("RTFx", justify="right")

    for row in lb.rows:
        cells: List[Text] = [Text(str(row.rank)), Text(row.model)]
        for col in lb.columns:
            cell = row.cells.get(col)
            wer = cell.wer if cell else None
            vmin, vmax, best = stats[col]
            is_best = wer is not None and best is not None and abs(wer - best) < 1e-9
            cells.append(Text(_fmt_pct(wer), style=rich_style(wer, vmin, vmax, is_best)))
        vmin, vmax, best = stats[("__avg__", "")]
        is_best = row.average_wer is not None and best is not None and abs(row.average_wer - best) < 1e-9
        cells.append(Text(_fmt_pct(row.average_wer), style=rich_style(row.average_wer, vmin, vmax, is_best)))
        if show_rtfx:
            cells.append(Text(_fmt_rtfx(row.rtfx)))
        table.add_row(*cells)

    console.print(table)
    console.print(
        "[dim]Color scale is per-column (vmin=min, vmax=90th pct). "
        "Avg WER uses each dataset's primary reference. RTFx is indicative.[/dim]"
    )


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def render_html(lb: Leaderboard, show_rtfx: bool = True) -> str:
    """Return a standalone HTML document with colored cells."""
    multi = len(lb.datasets) > 1
    stats = _column_stats(lb)

    def cell_html(wer: Optional[float], col_key: Column) -> str:
        vmin, vmax, best = stats[col_key]
        if wer is None:
            return '<td class="missing">–</td>'
        rgb = wer_rgb(wer, vmin, vmax)
        fg = text_on(rgb)
        weight = "700" if best is not None and abs(wer - best) < 1e-9 else "400"
        bg = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        return (
            f'<td style="background-color:{bg};color:{fg};font-weight:{weight}">'
            f"{_fmt_pct(wer)}</td>"
        )

    head = ["#", "Model"] + [_column_header(c, multi).replace("\n", "<br>") for c in lb.columns] + ["Avg WER"]
    if show_rtfx:
        head.append("RTFx")
    header_cells = "".join(f"<th>{_html.escape(h)}</th>" for h in head)

    body_rows = []
    for row in lb.rows:
        tds = [f"<td>{row.rank}</td>", f'<td class="model">{_html.escape(row.model)}</td>']
        for col in lb.columns:
            cell = row.cells.get(col)
            tds.append(cell_html(cell.wer if cell else None, col))
        tds.append(cell_html(row.average_wer, ("__avg__", "")))
        if show_rtfx:
            tds.append(f'<td class="rtfx">{_fmt_rtfx(row.rtfx)}</td>')
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>ASR Leaderboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; background:#0d1117; color:#e6edf3; }}
  h1 {{ font-size: 1.3rem; }}
  table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 6px 12px; text-align: right; border-bottom: 1px solid #30363d; }}
  th {{ position: sticky; top: 0; background:#161b22; }}
  td.model {{ text-align: left; font-weight: 600; }}
  td.missing {{ color:#7d8590; }}
  td.rtfx {{ color:#9da7b3; }}
  .legend {{ margin-top: 1rem; color:#9da7b3; font-size: 0.85rem; }}
</style></head>
<body>
<h1>ASR Leaderboard — WER by dataset/reference</h1>
<table><thead><tr>{header_cells}</tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
<p class="legend">Green = lower WER (better), red = higher. Color scale is per-column
(vmin = min, vmax = 90th percentile). Avg WER uses each dataset's primary reference;
RTFx (real-time factor) is indicative.</p>
</body></html>
"""


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def render_markdown(lb: Leaderboard, show_rtfx: bool = True) -> str:
    """Return a GitHub-flavored Markdown table (best per column in **bold**)."""
    from tabulate import tabulate

    multi = len(lb.datasets) > 1
    stats = _column_stats(lb)
    headers = ["#", "Model"] + [_column_header(c, multi).replace("\n", " ") for c in lb.columns] + ["Avg WER"]
    if show_rtfx:
        headers.append("RTFx")

    def mark(wer: Optional[float], col_key: Column) -> str:
        s = _fmt_pct(wer)
        if wer is None:
            return s
        _, _, best = stats[col_key]
        return f"**{s}**" if best is not None and abs(wer - best) < 1e-9 else s

    table_rows = []
    for row in lb.rows:
        cols = [str(row.rank), row.model]
        for col in lb.columns:
            cell = row.cells.get(col)
            cols.append(mark(cell.wer if cell else None, col))
        cols.append(mark(row.average_wer, ("__avg__", "")))
        if show_rtfx:
            cols.append(_fmt_rtfx(row.rtfx))
        table_rows.append(cols)

    table = tabulate(table_rows, headers=headers, tablefmt="github")
    legend = (
        "\n\n_Lower WER is better. **Bold** = best in column. "
        "Avg WER uses each dataset's primary reference; RTFx is indicative._\n"
    )
    return "# ASR Leaderboard\n\n" + table + legend
