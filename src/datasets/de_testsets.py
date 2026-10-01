"""German public test sets as bundled NeMo manifests: Tuda-De, Common Voice, Verbmobil.

Four leaderboard datasets, each a fixed JSONL manifest under ``manifests/``
built by ``scripts/build_de_test_manifests.py`` (audio as 16 kHz mono WAV on
kiz0, paths absolute):

* ``tuda_test_kinect_raw`` -- Tuda-De v4 test split, far-field Kinect microphone
  (raw channel, speaker ~1-2 m away). Reference ``ref``: ``cleaned_sentence`` of
  the recording XML (case kept, no punctuation, numbers spelled out).
* ``tuda_test_yamaha`` -- the *same* recordings through the close-talk Yamaha
  microphone. Both Tuda datasets hold exactly the same utterances, so their
  difference is the microphone alone.
* ``cv_de_test`` -- Common Voice German, the complete official ``test`` split of
  the release named in the manifest (``cv_version``). Reference ``ref``:
  ``sentence`` (case and punctuation kept; the scorer normalises both away).
* ``verbmobil_test`` -- Verbmobil German spontaneous appointment/travel
  dialogues, the BAS-defined speaker-disjoint TEST set (VM1_TEST + VM2_TEST,
  "Infos to VM Data Sets" v2.3), one clip per speaker turn, close-talk
  microphone. Reference ``ort``: the ORT tier of the BAS Partitur, cleaned like
  BAS-RVG1's ORT (umlauts decoded, ``<...>`` hesitation/noise markers removed).

Every row may carry a ``labels`` list (``"source: parl"``, ``"sentence in CV
train: yes"``, ``"part: VM2"``, ...). :func:`manifest_labels` exposes them to
the leaderboard's sub-split drill-down, keyed by audio path.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .manifest import ManifestSource

MANIFEST_DIR = Path(__file__).resolve().parents[2] / "manifests"


class BundledManifestSource(ManifestSource):
    """Manifest source with a fixed dataset name, reference key and default manifest."""

    name = "manifest"
    manifest_file = ""
    reference_key = "ref"

    def __init__(self, manifest_path: str | Path | None = None, **kwargs) -> None:
        kwargs.pop("name", None)
        kwargs.pop("reference_key", None)
        super().__init__(
            manifest_path or MANIFEST_DIR / self.manifest_file,
            name=self.name,
            reference_key=type(self).reference_key,
            **kwargs,
        )


class TudaTestKinectRawSource(BundledManifestSource):
    """Tuda-De v4 test split, far-field Kinect microphone (raw channel)."""

    name = "tuda_test_kinect_raw"
    manifest_file = "tuda_test_kinect_raw.jsonl"


class TudaTestYamahaSource(BundledManifestSource):
    """Tuda-De v4 test split, close-talk Yamaha microphone (same utterances)."""

    name = "tuda_test_yamaha"
    manifest_file = "tuda_test_yamaha.jsonl"


class CvDeTestSource(BundledManifestSource):
    """Common Voice German, full official test split."""

    name = "cv_de_test"
    manifest_file = "cv_de_test.jsonl"


class VerbmobilTestSource(BundledManifestSource):
    """Verbmobil German, BAS TEST set (VM1 + VM2), one clip per speaker turn."""

    name = "verbmobil_test"
    manifest_file = "verbmobil_test.jsonl"
    reference_key = "ort"


SOURCES = (TudaTestKinectRawSource, TudaTestYamahaSource, CvDeTestSource, VerbmobilTestSource)


@lru_cache(maxsize=None)
def _labels_from(manifest_path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    path = Path(manifest_path)
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels = row.get("labels") or []
            audio = row.get("audio_filepath")
            if audio and labels:
                out[str(audio)] = list(labels)
    return out


def manifest_labels(dataset: str, manifest_path: Optional[str | Path] = None) -> Dict[str, List[str]]:
    """``{audio_path: [label, ...]}`` for one of the bundled datasets.

    Unknown datasets and missing manifests give an empty mapping, so callers can
    use it unconditionally.
    """
    if manifest_path is None:
        cls = next((c for c in SOURCES if c.name == dataset), None)
        if cls is None:
            return {}
        manifest_path = MANIFEST_DIR / cls.manifest_file
    return _labels_from(str(manifest_path))
