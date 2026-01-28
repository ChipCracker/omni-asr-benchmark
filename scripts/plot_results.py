#!/usr/bin/env python3
"""Visualize ASR evaluation results as bar chart."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all evaluation JSON files."""
    results = []
    for f in results_dir.glob("*evaluation*.json"):
        if f.name.startswith("."):
            continue
        with open(f) as fp:
            data = json.load(fp)
            data["_file"] = f.name
            results.append(data)
    return results


def extract_metrics(data: Dict) -> Dict:
    """Extract WER metrics and compute variance from per-sample data."""
    per_sample = data.get("per_sample", [])

    # Dialect WER
    dialect_wers = [s["dialect_wer"] for s in per_sample if s.get("dialect_wer") is not None]
    dialect_mean = data["results"]["dialect_reference"]["wer"]
    dialect_std = np.std(dialect_wers) if dialect_wers else 0

    # ORT WER
    ort_wers = [s["ort_wer"] for s in per_sample if s.get("ort_wer") is not None]
    ort_mean = data["results"]["ort_reference"].get("wer")
    ort_std = np.std(ort_wers) if ort_wers else 0

    return {
        "model": data.get("model", "Unknown"),
        "dialect_wer": dialect_mean,
        "dialect_std": dialect_std,
        "ort_wer": ort_mean,
        "ort_std": ort_std,
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

    # Add value labels
    ax.bar_label(bars1, fmt="%.1f%%", padding=3)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3)

    ax.set_ylabel("Word Error Rate (%) - symlog scale")
    ax.set_title("ASR Model Comparison - WER on BAS RVG1 Dataset")
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


if __name__ == "__main__":
    main()
