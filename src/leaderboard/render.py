"""Render a :class:`Leaderboard` to the terminal, HTML, and Markdown."""

from __future__ import annotations

import html as _html
import json
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
_BIG = "1e9"  # sentinel sort value for missing cells (always sort last)


def _corpus_banner(lb: Leaderboard) -> str:
    """Chips describing which corpus/corpora were benchmarked."""
    chips = []
    for dataset in lb.datasets:
        meta = lb.dataset_meta.get(dataset, {})
        refs = meta.get("references", [])
        primary = meta.get("primary", "")
        ref_str = ", ".join((f"{r}★" if r == primary else r) for r in refs)
        lang = meta.get("language", "")
        details = [f"{meta.get('samples', 0)} samples", f"{meta.get('models', 0)} models"]
        if lang:
            details.append(lang)
        if ref_str:
            details.append(f"refs: {ref_str}")
        chips.append(
            f'<div class="chip"><span class="chip-name">{_html.escape(dataset)}</span>'
            f'<span class="chip-meta">{_html.escape(" · ".join(details))}</span></div>'
        )
    return '<div class="corpus">' + "".join(chips) + "</div>"


def render_html(lb: Leaderboard, show_rtfx: bool = True) -> str:
    """Return a standalone, self-sorting HTML document with colored cells."""
    multi = len(lb.datasets) > 1
    stats = _column_stats(lb)

    def cell_html(wer: Optional[float], col_key: Column) -> str:
        vmin, vmax, best = stats[col_key]
        if wer is None:
            return f'<td class="missing" data-sort="{_BIG}">–</td>'
        rgb = wer_rgb(wer, vmin, vmax)
        fg = text_on(rgb)
        is_best = best is not None and abs(wer - best) < 1e-9
        cls = "wer best" if is_best else "wer"
        bg = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        return (
            f'<td class="{cls}" data-sort="{wer:.6f}">'
            f'<span class="pill" style="background:{bg};color:{fg}">{_fmt_pct(wer)}</span></td>'
        )

    # Header (with sort metadata).
    heads = [("#", "num"), ("Model", "str")]
    heads += [(_column_header(c, multi).replace("\n", " "), "num") for c in lb.columns]
    heads.append(("Avg WER", "num"))
    if show_rtfx:
        heads.append(("RTFx", "num"))
    header_cells = "".join(
        f'<th data-type="{t}" onclick="sortTable(this)"><span>{_html.escape(h)}</span>'
        f'<span class="arrow"></span></th>'
        for h, t in heads
    )

    # Rows.
    medals = {1: "\U0001F947", 2: "\U0001F948", 3: "\U0001F949"}  # 🥇🥈🥉
    body_rows = []
    for row in lb.rows:
        rank_label = medals.get(row.rank, str(row.rank))
        tds = [
            f'<td class="rank" data-sort="{row.rank}">{rank_label}</td>',
            f'<td class="model" data-sort="{_html.escape(row.model.lower())}">{_html.escape(row.model)}</td>',
        ]
        for col in lb.columns:
            cell = row.cells.get(col)
            tds.append(cell_html(cell.wer if cell else None, col))
        tds.append(cell_html(row.average_wer, ("__avg__", "")))
        if show_rtfx:
            rtfx_sort = f"{row.rtfx:.6f}" if row.rtfx is not None else _BIG
            tds.append(f'<td class="rtfx" data-sort="{rtfx_sort}">{_fmt_rtfx(row.rtfx)}</td>')
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    # Data payload for the client-side bar-chart view.
    col_defs = [
        {"key": f"{ds}::{ref}", "label": _column_header((ds, ref), multi).replace("\n", " ")}
        for (ds, ref) in lb.columns
    ]
    col_defs.append({"key": "__avg__", "label": "Avg WER"})
    bounds = {}
    for c in lb.columns:
        vmin, vmax, _ = stats[c]
        bounds[f"{c[0]}::{c[1]}"] = [vmin, vmax]
    avmin, avmax, _ = stats[("__avg__", "")]
    bounds["__avg__"] = [avmin, avmax]
    rows_data = []
    for row in lb.rows:
        cells = {f"{c[0]}::{c[1]}": (row.cells[c].wer if c in row.cells else None) for c in lb.columns}
        cells["__avg__"] = row.average_wer
        rows_data.append({"model": row.model, "rank": row.rank, "cells": cells})
    data_json = json.dumps(
        {"columns": col_defs, "bounds": bounds, "rows": rows_data, "defaultMetric": "__avg__"}
    ).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASR Leaderboard</title>
<style>
  :root {{
    --bg:#eef1f5; --card:#ffffff; --line:#d8dee4; --text:#1f2328; --muted:#59636e;
    --accent:#0969da;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, Inter, sans-serif;
          margin: 0; padding: 2.5rem 1.5rem; background: radial-gradient(1200px 600px at 50% -10%, #ffffff, var(--bg)); color: var(--text); }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  header h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; letter-spacing: -0.01em; }}
  header p.sub {{ margin: 0 0 1.1rem; color: var(--muted); font-size: .9rem; }}
  .corpus {{ display:flex; flex-wrap:wrap; gap:.6rem; margin-bottom: 1.4rem; }}
  .chip {{ display:flex; flex-direction:column; gap:.15rem; padding:.55rem .8rem; border:1px solid var(--line);
           border-radius:12px; background:var(--card); box-shadow: 0 1px 2px rgba(27,31,36,.06); }}
  .chip-name {{ font-weight:700; font-size:.92rem; }}
  .chip-meta {{ color:var(--muted); font-size:.78rem; }}
  .card {{ border:1px solid var(--line); border-radius:16px; overflow:hidden; background:var(--card);
           box-shadow: 0 8px 24px rgba(27,31,36,.10); }}
  table {{ width:100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  thead th {{ position: sticky; top:0; z-index:2; background:#f6f8fa; color:var(--muted);
              font-weight:600; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em;
              padding:.7rem .9rem; text-align:right; cursor:pointer; white-space:nowrap;
              border-bottom:1px solid var(--line); user-select:none; }}
  thead th:hover {{ color:var(--text); }}
  thead th .arrow {{ display:inline-block; width:.9em; color:var(--accent); }}
  th:nth-child(1), td.rank {{ text-align:center; width:3.2rem; }}
  th:nth-child(2), td.model {{ text-align:left; }}
  tbody td {{ padding:.5rem .9rem; text-align:right; border-bottom:1px solid #eaeef2; }}
  tbody tr:last-child td {{ border-bottom:none; }}
  tbody tr:nth-child(even) {{ background:#fbfcfd; }}
  tbody tr:hover {{ background: rgba(9,105,218,.07); }}
  td.model {{ font-weight:600; }}
  td.rank {{ font-size:1.05rem; }}
  td.missing {{ color:#8c959f; }}
  td.rtfx {{ color:var(--muted); }}
  .pill {{ display:inline-block; min-width:3.6rem; padding:.18rem .5rem; border-radius:999px;
           font-weight:600; font-size:.85rem; }}
  td.best .pill {{ outline:2px solid rgba(31,35,40,.45); font-weight:800; }}
  .legend {{ margin-top:1rem; color:var(--muted); font-size:.82rem; line-height:1.5; }}
  .legend .swatch {{ display:inline-block; width:120px; height:.6rem; border-radius:4px; vertical-align:middle;
                     background:linear-gradient(90deg, rgb(0,200,0), rgb(220,200,0), rgb(220,0,0)); }}
  footer {{ margin-top:1rem; color:var(--muted); font-size:.78rem; }}
  /* View toggle + controls */
  .toolbar {{ display:flex; align-items:center; gap:.8rem; flex-wrap:wrap; margin-bottom:1rem; }}
  .toggle {{ display:inline-flex; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:var(--card); }}
  .toggle button {{ border:0; background:transparent; color:var(--muted); font:inherit; font-size:.85rem;
                    padding:.45rem .9rem; cursor:pointer; }}
  .toggle button.active {{ background:var(--accent); color:#fff; }}
  .metric-pick {{ display:none; align-items:center; gap:.4rem; color:var(--muted); font-size:.85rem; }}
  .metric-pick select {{ font:inherit; padding:.35rem .5rem; border-radius:8px; border:1px solid var(--line);
                         background:var(--card); color:var(--text); }}
  /* Bar chart */
  #chartView {{ padding:1.2rem 1.3rem; }}
  .bar-row {{ display:grid; grid-template-columns:230px 1fr; align-items:center; gap:.8rem; margin:.32rem 0; }}
  .bar-label {{ font-size:.85rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .bar-label .rk {{ color:var(--muted); font-weight:500; margin-right:.35rem; }}
  .bar-track {{ position:relative; background:#eef1f5; border-radius:8px; height:1.5rem; }}
  .bar-fill {{ height:100%; border-radius:8px; min-width:2px; transition:width .35s ease; }}
  .bar-val {{ position:absolute; top:0; height:100%; display:flex; align-items:center; padding-left:.45rem;
              font-size:.78rem; font-weight:700; color:var(--text); }}
  @media (max-width:640px) {{ .bar-row {{ grid-template-columns:130px 1fr; }} }}
</style></head>
<body><div class="wrap">
<header>
  <h1>ASR Leaderboard</h1>
  <p class="sub">Word Error Rate by dataset / reference — lower is better.</p>
</header>
{_corpus_banner(lb)}
<div class="toolbar">
  <div class="toggle">
    <button id="btnTable" class="active" onclick="setView('table')">▦ Table</button>
    <button id="btnChart" onclick="setView('chart')">▮ Bar chart</button>
  </div>
  <label class="metric-pick" id="metricPick">Metric <select id="metric" onchange="renderChart()"></select></label>
</div>
<div class="card" id="tableView">
<table id="lb"><thead><tr>{header_cells}</tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
</div>
<div class="card" id="chartView" hidden></div>
<p class="legend">
  <span class="swatch"></span> low&nbsp;WER&nbsp;→&nbsp;high &nbsp;·&nbsp;
  Color scale is per column (min … 90th percentile). &nbsp;·&nbsp;
  <b>★</b> marks each dataset's primary reference (drives Avg&nbsp;WER &amp; ranking). &nbsp;·&nbsp;
  Bold/outlined = best in column. &nbsp;·&nbsp; RTFx (real-time factor) is indicative.
</p>
<footer id="gen"></footer>
</div>
<script>
const DATA = {data_json};

function sortTable(th) {{
  const table = th.closest('table');
  const tbody = table.tBodies[0];
  const idx = th.cellIndex;
  const type = th.dataset.type;
  const asc = !(table.dataset.col === String(idx) && table.dataset.dir === 'asc');
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {{
    let av = a.cells[idx].dataset.sort, bv = b.cells[idx].dataset.sort;
    if (type === 'num') {{ av = parseFloat(av); bv = parseFloat(bv); return asc ? av - bv : bv - av; }}
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
  table.dataset.col = idx; table.dataset.dir = asc ? 'asc' : 'desc';
  table.querySelectorAll('thead th .arrow').forEach(a => a.textContent = '');
  th.querySelector('.arrow').textContent = asc ? ' ▲' : ' ▼';
}}

function werRgb(wer, vmin, vmax) {{
  if (wer == null) return [140,140,140];
  let t = vmax > vmin ? (wer - vmin) / (vmax - vmin) : 0;
  t = Math.max(0, Math.min(1, t));
  const lerp = (a, b, u) => a.map((x, i) => Math.round(x + (b[i] - x) * u));
  const G=[0,200,0], Y=[220,200,0], R=[220,0,0];
  return t < 0.5 ? lerp(G, Y, t/0.5) : lerp(Y, R, (t-0.5)/0.5);
}}

function setView(view) {{
  const isChart = view === 'chart';
  document.getElementById('tableView').hidden = isChart;
  document.getElementById('chartView').hidden = !isChart;
  document.getElementById('metricPick').style.display = isChart ? 'inline-flex' : 'none';
  document.getElementById('btnChart').classList.toggle('active', isChart);
  document.getElementById('btnTable').classList.toggle('active', !isChart);
  if (isChart) renderChart();
}}

function renderChart() {{
  const key = document.getElementById('metric').value;
  const [vmin, vmax] = DATA.bounds[key] || [0, 1];
  const vals = DATA.rows.map(r => r.cells[key]).filter(v => v != null);
  const scaleMax = Math.max(vmax, ...vals, 0.0001);
  const sorted = [...DATA.rows].sort((a, b) => {{
    const av = a.cells[key], bv = b.cells[key];
    if (av == null) return 1; if (bv == null) return -1; return av - bv;
  }});
  const label = (DATA.columns.find(c => c.key === key) || {{}}).label || key;
  let html = '';
  sorted.forEach((r, i) => {{
    const wer = r.cells[key];
    const pct = wer == null ? '–' : (wer * 100).toFixed(1) + '%';
    const w = wer == null ? 0 : Math.max(1, (wer / scaleMax) * 100);
    const c = werRgb(wer, vmin, vmax);
    const bg = `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
    html += `<div class="bar-row"><div class="bar-label" title="${{r.model}}">`
          + `<span class="rk">${{i+1}}</span>${{r.model}}</div>`
          + `<div class="bar-track"><div class="bar-fill" style="width:${{w}}%;background:${{bg}}"></div>`
          + `<div class="bar-val">${{pct}}</div></div></div>`;
  }});
  document.getElementById('chartView').innerHTML =
    `<div style="padding:.2rem .2rem .8rem;color:var(--muted);font-size:.85rem">Sorted by ${{label}} (lower = better)</div>` + html;
}}

(function init() {{
  const sel = document.getElementById('metric');
  DATA.columns.forEach(c => {{
    const o = document.createElement('option');
    o.value = c.key; o.textContent = c.label; sel.appendChild(o);
  }});
  sel.value = DATA.defaultMetric;
  document.getElementById('gen').textContent = 'Generated ' + new Date().toLocaleString();
}})();
</script>
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
