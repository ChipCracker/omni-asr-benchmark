"""Load and validate the benchmark matrix (``configs/matrix.yaml``).

The matrix lists the datasets and models to benchmark; :func:`iter_jobs`
expands it into the ``(model x dataset)`` cross product, merging per-model
overrides over the global ``defaults``. ``${ENV}`` references in string fields
are expanded at load time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _expand(value: Any) -> Any:
    return os.path.expandvars(value) if isinstance(value, str) else value


@dataclass
class DatasetSpec:
    name: str
    data_dir: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    name: str
    batch_size: Optional[int] = None
    language: Optional[str] = None
    extra_args: str = ""


@dataclass
class Job:
    """One concrete (model, dataset) benchmark to run."""

    model: str
    dataset: str
    data_dir: str
    language: str
    batch_size: int
    split: str
    params: Dict[str, Any]
    extra_args: str

    @property
    def model_tag(self) -> str:
        return self.model.replace("/", "_").replace("\\", "_")

    @property
    def output(self) -> str:
        return f"results/{self.model_tag}__{self.dataset}.json"

    def to_cli_args(self, max_samples: Optional[int] = None) -> List[str]:
        """Build the ``scripts/benchmark.py`` argument list for this job."""
        args = [
            "--model", self.model,
            "--dataset", self.dataset,
            "--language", self.language,
            "--batch-size", str(self.batch_size),
            "--split", self.split,
            "--output", self.output,
            "--no-leaderboard",
        ]
        if self.data_dir:
            args += ["--data-dir", self.data_dir]
        if "channel" in self.params:
            args += ["--channel", str(self.params["channel"])]
        if max_samples is not None:
            args += ["--max-samples", str(max_samples)]
        if self.extra_args:
            args += self.extra_args.split()
        return args


@dataclass
class MatrixConfig:
    defaults: Dict[str, Any]
    datasets: List[DatasetSpec]
    models: List[ModelSpec]


def load_config(path: str | Path) -> MatrixConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.get("defaults", {}) or {}

    datasets = []
    for d in raw.get("datasets", []) or []:
        datasets.append(
            DatasetSpec(
                name=d["name"],
                data_dir=_expand(d.get("data_dir")),
                params={k: _expand(v) for k, v in (d.get("params", {}) or {}).items()},
            )
        )

    models = []
    for m in raw.get("models", []) or []:
        models.append(
            ModelSpec(
                name=m["name"],
                batch_size=m.get("batch_size"),
                language=m.get("language"),
                extra_args=_expand(m.get("extra_args", "")) or "",
            )
        )

    if not datasets:
        raise ValueError("Matrix config has no 'datasets'.")
    if not models:
        raise ValueError("Matrix config has no 'models'.")

    return MatrixConfig(defaults=defaults, datasets=datasets, models=models)


def iter_jobs(config: MatrixConfig) -> List[Job]:
    """Expand the matrix into concrete (model x dataset) jobs."""
    d = config.defaults
    jobs: List[Job] = []
    for model in config.models:
        for dataset in config.datasets:
            jobs.append(
                Job(
                    model=model.name,
                    dataset=dataset.name,
                    data_dir=dataset.data_dir or _expand(d.get("data_dir", "")) or "",
                    language=model.language or d.get("language", "deu_Latn"),
                    batch_size=model.batch_size or d.get("batch_size", 16),
                    split=d.get("split", "test"),
                    params=dataset.params,
                    extra_args=model.extra_args,
                )
            )
    return jobs
