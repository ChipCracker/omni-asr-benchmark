#!/usr/bin/env python3
"""Visualize ASR benchmark results (BAS RVG1: dialect vs ORT) as bar charts.

Reads result JSONs via :class:`BenchmarkResult.from_dict`, so both the legacy
(v1) and generic (v2) schemas work. For a color-coded leaderboard table across
arbitrary datasets, see ``scripts/leaderboard.py``.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.result import BenchmarkResult  # noqa: E402


def load_all_results(results_dir: Path) -> List[Tuple[BenchmarkResult, str]]:
    """Load all benchmark result JSON files as (result, filename) tuples."""
    results = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith(".") or f.name == "leaderboard.json":
            continue
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict) or "results" not in data:
            continue
        results.append((BenchmarkResult.from_dict(data), f.name))
    return results


def _display_name(result: BenchmarkResult, filename: str) -> str:
    """Append '(online)'/'(offline)' for Voxtral Realtime variants."""
    model = result.model
    if "-online" in filename:
        return model + " (online)"
    if "Realtime" in model and "online" not in filename:
        return model + " (offline)"
    return model


def _ref_mean_std(result: BenchmarkResult, ref: str, metric: str):
    mean = result.results.get(ref, {}).get(metric)
    per = [
        s.metrics[ref][metric]
        for s in result.per_sample
        if ref in s.metrics and s.metrics[ref].get(metric) is not None
    ]
    return mean, (np.std(per) if per else 0)


def extract_metrics(item: Tuple[BenchmarkResult, str]) -> Dict:
    """Extract dialect/ORT WER+CER means and per-sample std."""
    result, filename = item
    dialect_wer, dialect_std = _ref_mean_std(result, "dialect", "wer")
    ort_wer, ort_std = _ref_mean_std(result, "ort", "wer")
    dialect_cer, dialect_cer_std = _ref_mean_std(result, "dialect", "cer")
    ort_cer, ort_cer_std = _ref_mean_std(result, "ort", "cer")
    return {
        "model": _display_name(result, filename),
        "dialect_wer": dialect_wer or 0,
        "dialect_std": dialect_std,
        "ort_wer": ort_wer,
        "ort_std": ort_std,
        "dialect_cer": dialect_cer or 0,
        "dialect_cer_std": dialect_cer_std,
        "ort_cer": ort_cer,
        "ort_cer_std": ort_cer_std,
    }


def plot_results(metrics: List[Dict], output_path: Path = None):
    """Create grouped bar chart with error bars and symlog scale."""
    models = [m["model"].split("/")[-1] for m in metrics]  # Short names
    dialect_wers = [m["dialect_wer"] * 100 for m in metrics]
    dialect_stds = [m["dialect_std"] * 100 for m in metrics]
    ort_wers = [m["ort_wer"] * 100 if m["ort_wer"] else 0 for m in metrics]
    ort_stds = [m["ort_std"] * 100 for m in metrics]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars with error bars
    bars1 = ax.bar(x - width/2, dialect_wers, width,
                   yerr=dialect_stds, label="Dialect WER", capsize=5)
    bars2 = ax.bar(x + width/2, ort_wers, width,
                   yerr=ort_stds, label="ORT WER", capsize=5)

    # Symlog scale: linear below 100%, logarithmic above
    ax.set_yscale("symlog", linthresh=100, linscale=1)
    ax.set_ylim(0, None)

    # Y-axis ticks at every 10%
    max_val = max(max(dialect_wers), max(ort_wers))
    ticks = list(range(0, 101, 10))  # 0, 10, 20, ..., 100
    if max_val > 100:
        # Add ticks for logarithmic region
        ticks.extend([200, 300, 500, 1000])
        ticks = [t for t in ticks if t <= max_val * 1.5]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}%" for t in ticks])

    # Add value labels
    ax.bar_label(bars1, fmt="%.1f%%", padding=3)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3)

    ax.set_ylabel("Word Error Rate (%) - symlog scale")
    ax.set_title("ASR Model Comparison - WER on BAS RVG1 Dataset (default configuration)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_cer_results(metrics: List[Dict], output_path: Path = None):
    """Create grouped bar chart for CER with error bars and symlog scale."""
    models = [m["model"].split("/")[-1] for m in metrics]  # Short names
    dialect_cers = [m["dialect_cer"] * 100 for m in metrics]
    dialect_stds = [m["dialect_cer_std"] * 100 for m in metrics]
    ort_cers = [m["ort_cer"] * 100 if m["ort_cer"] else 0 for m in metrics]
    ort_stds = [m["ort_cer_std"] * 100 for m in metrics]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars with error bars
    bars1 = ax.bar(x - width/2, dialect_cers, width,
                   yerr=dialect_stds, label="Dialect CER", capsize=5)
    bars2 = ax.bar(x + width/2, ort_cers, width,
                   yerr=ort_stds, label="ORT CER", capsize=5)

    # Symlog scale: linear below 100%, logarithmic above
    ax.set_yscale("symlog", linthresh=100, linscale=1)
    ax.set_ylim(0, None)

    # Y-axis ticks at every 10%
    max_val = max(max(dialect_cers), max(ort_cers))
    ticks = list(range(0, 101, 10))  # 0, 10, 20, ..., 100
    if max_val > 100:
        # Add ticks for logarithmic region
        ticks.extend([200, 300, 500, 1000])
        ticks = [t for t in ticks if t <= max_val * 1.5]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}%" for t in ticks])

    # Add value labels
    ax.bar_label(bars1, fmt="%.1f%%", padding=3)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3)

    ax.set_ylabel("Character Error Rate (%) - symlog scale")
    ax.set_title("ASR Model Comparison - CER on BAS RVG1 Dataset (default configuration)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()


def main():
    results_dir = Path("results")
    all_data = load_all_results(results_dir)
    metrics = [extract_metrics(d) for d in all_data]

    # Sort by ORT WER (ascending)
    metrics.sort(key=lambda m: m["ort_wer"] or float("inf"))

    plot_results(metrics, Path("results/comparison_chart.png"))
    plot_cer_results(metrics, Path("results/comparison_chart_cer.png"))


if __name__ == "__main__":
    main()
