"""Dataset registry: map a config name to a :class:`DatasetSource` instance.

Keeps the YAML matrix (``configs/matrix.yaml``) decoupled from concrete dataset
classes. Register a new dataset by adding one entry to ``_DATASETS``.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import DatasetSource
from .bas_rvg1 import BasRvg1Source

# Both the canonical ``name`` and the class name resolve to the same factory,
# so YAML entries can use either ``bas_rvg1`` or ``BasRvg1Source``.
_DATASETS: Dict[str, Callable[..., DatasetSource]] = {
    "bas_rvg1": BasRvg1Source,
    "basrvg1source": BasRvg1Source,
}


def get_dataset(name: str, **params) -> DatasetSource:
    """Instantiate a dataset source by name.

    Args:
        name: Dataset name (e.g. ``bas_rvg1``) or class name (``BasRvg1Source``).
        **params: Constructor parameters (e.g. ``data_dir``, ``channel``).

    Returns:
        A :class:`DatasetSource` instance.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    key = name.lower().replace("-", "_")
    if key not in _DATASETS:
        available = ", ".join(sorted({v.name for v in _DATASETS.values() if hasattr(v, "name")}))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return _DATASETS[key](**params)


def available_datasets() -> list[str]:
    """Return the canonical names of registered datasets."""
    return sorted({getattr(v, "name", k) for k, v in _DATASETS.items()})
