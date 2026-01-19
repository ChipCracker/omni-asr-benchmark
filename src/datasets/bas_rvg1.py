"""BAS RVG1 dataset source."""

from __future__ import annotations

import logging
import re
import soundfile as sf
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zipfile import ZipFile

from .base import DatasetSource, Sample

logger = logging.getLogger(__name__)

_RE_REPEAT = re.compile(r"\+/(.+?)/\+")
_RE_SENTENCE_BREAK = re.compile(r"-/(.+?)/-")
_RE_VARIANT = re.compile(r"<!(?:\w+)\s+([^>]+)>")
_RE_TAG = re.compile(r"<[^>]+>")
_RE_UMlaut = re.compile(r'"([AOUaou])')
_RE_ESZETT = re.compile(r'"([sS])')
_RE_WHITESPACE = re.compile(r"\s+")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:?!])")
_RE_BRACKET_CONTENT = re.compile(r"<([^>]+)>")

_UMLAUT_MAP = {
    "a": "ä",
    "A": "Ä",
    "o": "ö",
    "O": "Ö",
    "u": "ü",
    "U": "Ü",
}

_METADATA_COLUMN_MAP = {
    "cdnr": "volume",
    "filenr": "speaker_number",
    "diaclass": "dialect_class",
    "diareg": "dialect_region",
    "age": "age",
    "sex": "sex",
    "height": "height_cm",
    "weight": "weight_kg",
    "born": "birth_location",
    "time": "longest_residence_location",
    "parents": "parents_dialect_relation",
    "mother": "mother_origin",
    "father": "father_origin",
    "school": "education_level",
    "profession": "profession",
    "diaself": "dialect_self",
}


class BasRvg1Source(DatasetSource):
    """Source for the BAS RVG1 (Regional Variants of German 1) corpus.

    Reads from the raw file structure containing .par (Partitur), .trl (Transliteration),
    and .wav/.nis audio files.
    """

    name = "bas_rvg1"

    def __init__(
        self,
        data_dir: str | Path,
        channel: str = "c",
        cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.data_dir = Path(data_dir)
        self.channel = channel
        self._speaker_metadata_cache: Optional[Dict[str, Dict[str, Optional[str]]]] = None

    def iter_samples(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> Iterable[Sample]:
        """Iterate over BAS RVG1 samples.

        Each sp1{channel}*.wav segment is treated as an individual sample.
        ORT from the PAR file is used as primary transcript since TR2 (dialect)
        is often missing in segment-level PAR files.

        Note: The dataset does not have explicit splits. All valid data found in
        directories matching the structure is yielded. Splitting should be handled
        externally if needed, or by selecting indices.
        """
        logger.info("Loading BAS RVG1 from %s (channel=%s)", self.data_dir, self.channel)

        if not self.data_dir.exists():
            logger.error(f"BAS RVG1 data directory not found: {self.data_dir}")
            return

        metadata_map = self._load_speaker_metadata(self.data_dir)

        current_idx = 0

        end_index = float("inf")
        if max_samples is not None:
            end_index = start_index + max_samples

        for speaker_dir in self._iter_speaker_directories(self.data_dir):
            if current_idx >= end_index:
                break

            speaker_id = speaker_dir.name

            # Find ALL sp1{channel}*.wav files (segment-based approach)
            pattern = f"sp1{self.channel.lower()}*.wav"
            audio_files = sorted(speaker_dir.glob(pattern))

            if not audio_files:
                # Fallback to .nis files
                nis_pattern = f"sp1{self.channel.lower()}*.nis"
                audio_files = sorted(speaker_dir.glob(nis_pattern))

            for audio_path in audio_files:
                if current_idx >= end_index:
                    break

                # Read ORT from segment's PAR file
                par_path = audio_path.with_suffix(".par")
                ort_transcription, dialect_transcription, kan_transcription = self._read_par_transcriptions(par_path)

                # Use dialect (TR2) if available, otherwise use ORT as primary transcript
                transcript = dialect_transcription or ort_transcription
                if not transcript:
                    continue

                # Skip if before start_index
                if current_idx < start_index:
                    current_idx += 1
                    continue

                try:
                    duration = sf.info(audio_path).duration
                except Exception as e:
                    logger.warning(f"Could not read duration for {audio_path}: {e}")
                    duration = 0.0

                speaker_meta = metadata_map.get(speaker_id, {})

                sample_metadata = {
                    "dataset": self.name,
                    "source_dataset": self.name,
                    "speaker_id": speaker_id,
                    "segment_id": audio_path.stem,
                    "ort_transcript": ort_transcription,
                    "audio_channel": self.channel,
                }

                for key, val in speaker_meta.items():
                    if val is not None:
                        sample_metadata[f"speaker_{key}"] = val

                yield Sample(
                    transcript=transcript,
                    duration=duration,
                    dataset_info={
                        "dataset_name": self.name,
                        "language": "de",
                        "split": split,
                        "index": current_idx,
                        "audio_path": str(audio_path),
                    },
                    metadata=sample_metadata
                )

                current_idx += 1

    def _replace_umlaut(self, match: re.Match[str]) -> str:
        char = match.group(1)
        return _UMLAUT_MAP.get(char, char)

    def _replace_eszett(self, match: re.Match[str]) -> str:
        return "ß"

    def _clean_trl_text(self, raw_text: str) -> str:
        text = _RE_REPEAT.sub(r"\1", raw_text)
        text = _RE_SENTENCE_BREAK.sub(r"\1", text)
        text = _RE_VARIANT.sub(r"\1", text)
        text = _RE_TAG.sub(" ", text)
        text = text.replace("*", "")
        text = text.replace("%", "")
        text = text.replace("$", "")
        text = text.replace("/", "")

        text = _RE_UMlaut.sub(self._replace_umlaut, text)
        text = _RE_ESZETT.sub(self._replace_eszett, text)
        text = text.replace("``", '"').replace("''", '"')
        text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _RE_WHITESPACE.sub(" ", text)
        return text.strip()

    def _clean_kan_text(self, raw_text: str) -> str:
        text = raw_text
        text = text.replace("+/", "").replace("/+", "")
        text = text.replace("-/", "").replace("/-", "")
        text = text.replace("+", "")
        text = text.replace("/", " ")
        text = text.replace("Q", "?")
        text = text.replace("#", " ")
        text = _RE_TAG.sub(" ", text)
        text = text.replace('"', "")
        text = _RE_WHITESPACE.sub(" ", text)
        return text.strip()

    def _read_trl_transcription(self, trl_path: Path) -> Optional[str]:
        if not trl_path.is_file():
            return None

        lines: List[str] = []
        try:
            with trl_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith(";"):
                        continue
                    lines.append(stripped)
        except Exception:
            return None

        if not lines:
            return None

        joined = " ".join(lines)
        if ":" in joined:
            _, joined = joined.split(":", 1)
        return self._clean_trl_text(joined)

    def _normalise_metadata_value(self, value: str) -> Optional[str]:
        value = value.strip()
        if not value or value == "-":
            return None
        value = _RE_UMlaut.sub(self._replace_umlaut, value)
        value = _RE_ESZETT.sub(self._replace_eszett, value)
        value = value.replace('"', "")
        value = _RE_WHITESPACE.sub(" ", value).strip()
        return value or None

    def _load_speaker_metadata(self, dataset_root: Path) -> Dict[str, Dict[str, Optional[str]]]:
        if self._speaker_metadata_cache is not None:
            return self._speaker_metadata_cache

        candidate_files = [
            dataset_root / "table" / "sprk_att.txt",
            dataset_root / "RVG1Docu" / "table" / "sprk_att.txt",
        ]

        raw_lines: Optional[List[str]] = None
        for candidate in candidate_files:
            if candidate.is_file():
                try:
                    raw_lines = candidate.read_text(encoding="latin-1").splitlines()
                    break
                except Exception:
                    continue

        if raw_lines is None:
            zip_path = dataset_root / "CLARINDocu.zip"
            if zip_path.is_file():
                try:
                    with ZipFile(zip_path) as zf:
                        for member in ("table/sprk_att.txt", "RVG1Docu/table/sprk_att.txt"):
                            if member in zf.namelist():
                                with zf.open(member) as fh:
                                    raw_lines = fh.read().decode("latin-1").splitlines()
                                break
                except Exception:
                    pass

        if not raw_lines:
            self._speaker_metadata_cache = {}
            return {}

        header = [col.strip() for col in raw_lines[0].split("\t")]
        index_map = {name: idx for idx, name in enumerate(header)}

        metadata: Dict[str, Dict[str, Optional[str]]] = {}
        for row in raw_lines[1:]:
            if not row.strip():
                continue
            parts = row.split("\t")
            filenr_idx = index_map.get("filenr")
            if filenr_idx is None or filenr_idx >= len(parts):
                continue
            speaker_raw = parts[filenr_idx].strip()
            if not speaker_raw:
                continue
            speaker_id = speaker_raw.zfill(3)

            entry: Dict[str, Optional[str]] = {}
            for source_key, target_key in _METADATA_COLUMN_MAP.items():
                idx = index_map.get(source_key)
                value = parts[idx] if idx is not None and idx < len(parts) else ""
                entry[target_key] = self._normalise_metadata_value(value)

            metadata[speaker_id] = entry

        self._speaker_metadata_cache = metadata
        return metadata

    def _clean_ort_token(self, token: str) -> str:
        token = token.strip()
        if not token or token == "#":
            return ""

        if token.startswith("<") and token.endswith(">"):
            return ""

        token = token.replace("*", "")
        token = token.replace("=", "")
        token = token.replace("#", "")
        token = token.replace(":>", "")
        token = token.replace("_", " ")
        token = token.replace("``", '"').replace("''", '"')
        token = _RE_TAG.sub(" ", token)
        token = token.replace(":>", "")
        token = _RE_UMlaut.sub(self._replace_umlaut, token)
        token = _RE_ESZETT.sub(self._replace_eszett, token)
        token = token.replace('"', "")
        token = token.strip()
        return token

    def _clean_tr2_token(self, token: str) -> str:
        token = token.strip()
        if not token:
            return ""

        token_stripped = token.strip()
        if token_stripped.startswith("+/") and token_stripped.endswith("/+"):
            return ""
        if token_stripped.startswith("-/") and token_stripped.endswith("/-"):
            return ""

        selected: Optional[str] = None
        variant_matches: List[str] = []
        non_variant_matches: List[str] = []

        for match in _RE_BRACKET_CONTENT.finditer(token):
            content = match.group(1).strip()
            if not content:
                continue
            if "#" in content:
                continue
            if content.startswith(":") or content.startswith(";"):
                continue
            if content.startswith("!"):
                stripped = re.sub(r"^!+\d*\s*", "", content).strip()
                if stripped:
                    variant_matches.append(stripped)
                continue

            non_variant_matches.append(content)

        if variant_matches:
            selected = variant_matches[-1]
        else:
            base_candidate = re.sub(r"<[^>]*>", " ", token)
            if any(ch.isalpha() for ch in base_candidate):
                selected = base_candidate
            else:
                for candidate in non_variant_matches:
                    candidate = candidate.lstrip("~")
                    candidate = candidate.strip()
                    if not candidate:
                        continue
                    if len(candidate) == 1 and candidate.upper() in {"A", "P", "Z"}:
                        continue
                    selected = candidate
                    break

        value = selected if selected else token

        value = value.replace(r"\'", "'")
        value = value.replace("~", "")
        value = value.replace("+/", "")
        value = value.replace("/+", "")
        value = value.replace("/", " ")
        value = value.replace("*", "")
        value = value.replace("=", "")
        value = value.replace("$", "")
        value = value.replace(":>", "")
        value = value.replace("#", "")
        value = value.replace("_", " ")
        value = value.replace("``", '"').replace("''", '"')
        value = _RE_TAG.sub(" ", value)
        value = re.sub(r"<\s*>", " ", value)
        value = _RE_UMlaut.sub(self._replace_umlaut, value)
        value = _RE_ESZETT.sub(self._replace_eszett, value)
        value = value.replace('"', "")
        value = value.replace("%", "")
        value = _RE_WHITESPACE.sub(" ", value).strip()
        value = value.replace(" ", "")
        return value

    def _read_par_transcriptions(self, par_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not par_path.is_file():
            return None, None, None

        tokens: List[str] = []
        tr2_tokens: List[str] = []
        kan_tokens: List[str] = []

        try:
            with par_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("ORT:\t"):
                        parts = line.rstrip("\n").split("\t", maxsplit=2)
                        if len(parts) < 3:
                            continue
                        token = self._clean_ort_token(parts[2])
                        if token:
                            tokens.append(token)
                    elif line.startswith("TR2:\t"):
                        parts = line.rstrip("\n").split("\t", maxsplit=2)
                        if len(parts) < 3:
                            continue
                        token = self._clean_tr2_token(parts[2])
                        if token:
                            tr2_tokens.append(token)
                    elif line.startswith("KAN:\t"):
                        parts = line.rstrip("\n").split("\t", maxsplit=2)
                        if len(parts) < 3:
                            continue
                        token = parts[2].strip()
                        if token:
                            kan_tokens.append(token)

        except Exception:
            return None, None, None

        ort_text = ""
        if tokens:
            ort_text = " ".join(tokens)
            ort_text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", ort_text)
            ort_text = _RE_WHITESPACE.sub(" ", ort_text)
            ort_text = ort_text.strip()

        tr2_text = ""
        if tr2_tokens:
            raw_tr2 = " ".join(tr2_tokens)
            tr2_text = self._clean_trl_text(raw_tr2)

        kan_text = ""
        if kan_tokens:
            kan_text = self._clean_kan_text(" ".join(kan_tokens))

        return (ort_text or None, tr2_text or None, kan_text or None)

    def _iter_speaker_directories(self, dataset_root: Path) -> Iterable[Path]:
        try:
            for entry in sorted(dataset_root.iterdir()):
                if entry.is_dir() and entry.name.isdigit():
                    yield entry
        except Exception as e:
            logger.error(f"Error iterating speaker directories in {dataset_root}: {e}")

