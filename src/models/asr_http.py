"""Remote HTTP ASR evaluator.

Benchmarks an ASR server that exposes a ``/transcribe`` endpoint accepting
base64-encoded audio (and a ``/info`` endpoint reporting readiness). No local
GPU is needed — the server does the inference; this client only sends audio and
collects transcripts.

Select it by passing a server URL as the model, e.g.::

    python scripts/benchmark.py --model http://141.75.89.18:8099 --dataset bas_rvg1
"""

from __future__ import annotations

import base64
import logging
import time
from typing import List, Optional
from urllib.parse import urlparse

from .base import AsrModel

logger = logging.getLogger(__name__)

# Framework language code -> server language code.
LANGUAGE_MAP = {
    "deu_Latn": "de",
    "eng_Latn": "en",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "ita_Latn": "it",
}


class AsrHttpEvaluator(AsrModel):
    """Evaluator for a remote ASR HTTP server (base64 audio -> transcript)."""

    def __init__(
        self,
        model_name: str,
        language: str = "deu_Latn",
        batch_size: int = 1,
        served_model: str = "large-v3",
        vad: bool = True,
        sample_rate: int = 16000,
        timeout: int = 300,
    ) -> None:
        super().__init__(model_name, language, batch_size)
        self.server_url = model_name.rstrip("/")
        self.transcribe_endpoint = f"{self.server_url}/transcribe"
        self.info_endpoint = f"{self.server_url}/info"
        self.served_model = served_model
        self.vad = vad
        self.sample_rate = sample_rate
        self.timeout = timeout
        self._lang = LANGUAGE_MAP.get(language, language.split("_")[0].lower())

    @property
    def display_name(self) -> str:
        return f"{self.served_model} @{urlparse(self.server_url).hostname} (server)"

    # ------------------------------------------------------------------ #
    def _wait_for_ready(self, timeout: int = 300, poll_interval: int = 5) -> bool:
        import requests

        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(self.info_endpoint, timeout=60)
                if r.status_code == 200:
                    status = r.json().get("result", {}).get("status", "").upper()
                    if status == "RUNNING":
                        return True
                    if status == "BUSY":
                        time.sleep(poll_interval)
                        continue
                time.sleep(poll_interval)
            except requests.exceptions.RequestException as e:
                logger.warning("Could not reach %s: %s; retrying...", self.info_endpoint, e)
                time.sleep(poll_interval)
        logger.error("Timeout waiting for ASR server after %ss", timeout)
        return False

    @staticmethod
    def _encode_audio(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _parse_transcript(result) -> str:
        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            return ""
        # Direct text field (some servers).
        text = result.get("text") or result.get("transcript") or ""
        if text:
            return text
        # Otherwise the server returns a list of VAD segments — concatenate ALL
        # of them in order (taking only the first segment truncates long audio).
        rlist = result.get("result", [])
        if not isinstance(rlist, list):
            return ""
        segs = [s for s in rlist if isinstance(s, dict)]
        try:
            segs.sort(key=lambda s: s.get("result_index", 0))
        except (TypeError, ValueError):
            pass
        parts = [
            (s.get("transcript_formatted") or s.get("transcript") or "").strip()
            for s in segs
        ]
        return " ".join(p for p in parts if p)

    def _transcribe_one(self, audio_path: str) -> str:
        import requests

        if not self._wait_for_ready():
            return ""
        payload = {
            "file": self._encode_audio(audio_path),
            "model": self.served_model,
            "language": self._lang,
            "batch_size": self.batch_size,
            "vad": self.vad,
            "sample_rate": self.sample_rate,
        }
        try:
            r = requests.post(
                self.transcribe_endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            logger.error("Transcription request failed for %s: %s", audio_path, e)
            return ""
        if r.status_code != 200:
            logger.error("Server %s returned %s: %s", self.transcribe_endpoint, r.status_code, r.text[:300])
            return ""
        return self._parse_transcript(r.json()).strip()

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        # One request per file (server handles its own VAD/batching).
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise ImportError("AsrHttpEvaluator needs the 'requests' package.") from e
        return [self._transcribe_one(p) for p in audio_paths]
