"""Dataset sources for ASR evaluation."""

from .base import DatasetSource, Sample
from .bas_rvg1 import BasRvg1Source
from .de_testsets import (
    CvDeTestSource,
    TudaTestKinectRawSource,
    TudaTestYamahaSource,
    VerbmobilTestSource,
)
from .manifest import KsofSource, ManifestSource
from .registry import available_datasets, get_dataset

__all__ = [
    "DatasetSource",
    "Sample",
    "BasRvg1Source",
    "ManifestSource",
    "KsofSource",
    "TudaTestKinectRawSource",
    "TudaTestYamahaSource",
    "CvDeTestSource",
    "VerbmobilTestSource",
    "get_dataset",
    "available_datasets",
]
