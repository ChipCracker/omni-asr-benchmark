"""VibeVoice Evaluator for Microsoft's VibeVoice-ASR model."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .base import AsrModel
from ..benchmark.result import BenchmarkResult, SampleResult

if TYPE_CHECKING:
    from ..datasets.base import DatasetSource

logger = logging.getLogger(__name__)


class VibeVoiceEvaluator(AsrModel):
    """Evaluator for Microsoft VibeVoice-ASR model.

    VibeVoice-ASR is a 9B parameter unified speech-to-text model that:
    - Processes up to 60 minutes of continuous audio in a single pass
    - Generates structured transcriptions with speaker diarization and timestamps
    - Supports customized hotwords for domain-specific terms

    Note: Requires vibevoice package. Install from:
        git clone https://github.com/microsoft/VibeVoice.git
        cd VibeVoice
        pip install -e .[asr]
    """

    def __init__(
        self,
        model_name: str = "microsoft/VibeVoice-ASR",
        language: str = "deu_Latn",
        batch_size: int = 1,
    ) -> None:
        """Initialize the VibeVoice evaluator.

        Args:
            model_name: HuggingFace model ID (default: "microsoft/VibeVoice-ASR").
            language: Language code for transcription (e.g., "deu_Latn").
            batch_size: Batch size for inference (default 1 due to model size).
        """
        super().__init__(model_name, language, batch_size)
        self._model = None
        self._processor = None

    def _get_model(self):
        """Lazy-load the VibeVoice model and processor."""
        if self._model is None:
            logger.info(f"Loading VibeVoice-ASR model: {self.model_name}")

            # Check for vibevoice package first
            try:
                from vibevoice.modular.modeling_vibevoice_asr import (
                    VibeVoiceASRForConditionalGeneration,
                )
                from vibevoice.processor.vibevoice_asr_processor import (
                    VibeVoiceASRProcessor,
                )
            except ImportError as e:
                raise ImportError(
                    "VibeVoice support requires the vibevoice package. "
                    "Install with:\n"
                    "  git clone https://github.com/microsoft/VibeVoice.git\n"
                    "  cd VibeVoice\n"
                    "  pip install -e .[asr]"
                ) from e

            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            # Load processor
            logger.info("Loading VibeVoice processor...")
            self._processor = VibeVoiceASRProcessor.from_pretrained(
                self.model_name,
                llm_model="Qwen/Qwen2.5-7B",
            )

            # Load model
            logger.info("Loading VibeVoice model...")
            attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
            try:
                self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load with {attn_impl}, trying eager attention: {e}"
                )
                self._model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    attn_implementation="eager",
                )
                attn_impl = "eager"

            self._model = self._model.to(device)
            self._model.eval()

            self._device = device
            self._dtype = dtype

            logger.info(
                f"VibeVoice-ASR loaded on {device} with dtype={dtype}, "
                f"attention={attn_impl}"
            )
        return self._model, self._processor

    def _truncate_generation_loop(self, text: str) -> str:
        """Detect and truncate repetitive generation loops.

        VibeVoice sometimes gets stuck in generation loops, producing thousands
        of repeated words/phrases. This detects such loops and truncates the
        output to before the loop started.

        Args:
            text: Text to check for repetitive loops.

        Returns:
            Truncated text if loop detected, otherwise original text.
        """
        words = text.split()
        if len(words) < 100:
            return text

        # Check if the last part shows repetitive patterns
        # Look for 3-word phrases repeated many times
        last_words = words[-60:]
        for phrase_len in [3, 4, 5]:
            if len(last_words) < phrase_len * 3:
                continue
            test_phrase = " ".join(last_words[-phrase_len:])
            if len(test_phrase) < 5:
                continue
            # Count occurrences in last 60 words
            last_text = " ".join(last_words)
            count = last_text.count(test_phrase)
            if count >= 5:
                # Found a loop - find where it starts in the full text
                full_text = " ".join(words)
                # Find first occurrence of the repeating phrase
                first_occurrence = full_text.find(test_phrase)
                if first_occurrence > 0:
                    # Truncate just before the loop
                    truncated = full_text[:first_occurrence].strip()
                    # Return truncated even if short - better than thousands of garbage words
                    if truncated:
                        return truncated
        return text

    def _extract_plain_text(self, raw_output: str) -> str:
        """Extract plain transcription text from VibeVoice structured output.

        VibeVoice outputs structured transcriptions with speaker tags and timestamps.
        This extracts just the spoken text for WER/CER evaluation.

        Args:
            raw_output: Raw model output with structure like:
                "[00:00.00 - 00:05.00] Speaker 1: Hello world"
                or JSON format with Content fields.

        Returns:
            Plain text without timestamps or speaker tags.
        """
        if not raw_output:
            return ""

        # Try to extract Content fields from JSON-like output
        # This handles both complete and incomplete JSON (e.g., missing closing ])
        content_matches = re.findall(r'"Content"\s*:\s*"([^"]*)"', raw_output)
        if content_matches:
            result = " ".join(content_matches)
            return self._truncate_generation_loop(result)

        # Fallback for truncated JSON: Content field without closing quote
        # This happens when the model output is cut off mid-Content
        truncated_match = re.search(r'"Content"\s*:\s*"([^"]+)$', raw_output, re.DOTALL)
        if truncated_match:
            result = truncated_match.group(1).strip()
            return self._truncate_generation_loop(result)

        # Fallback: Remove timestamps and speaker tags from plain text format
        lines = raw_output.strip().split("\n")
        text_parts = []

        for line in lines:
            # Remove timestamp patterns like [00:00.00 - 00:05.00]
            line = re.sub(r"\[\d{2}:\d{2}\.\d{2}\s*-\s*\d{2}:\d{2}\.\d{2}\]", "", line)
            # Remove speaker tags like "Speaker 1:" or "Speaker_1:"
            line = re.sub(r"Speaker[_\s]?\d+:\s*", "", line, flags=re.IGNORECASE)
            # Clean up extra whitespace
            line = line.strip()
            if line:
                text_parts.append(line)

        return " ".join(text_parts)

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe a batch of audio files using VibeVoice-ASR.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of transcription strings (plain text, without timestamps/speakers).
        """
        import torch

        model, processor = self._get_model()
        results = []

        # Process each audio file (batching handled by model internally)
        for audio_path in audio_paths:
            try:
                # Load audio
                import soundfile as sf

                audio_data, sample_rate = sf.read(audio_path)

                # Prepare inputs
                inputs = processor(
                    audio=audio_data,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                    add_generation_prompt=True,
                )

                # Move to device
                inputs = {
                    k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                # Generate transcription
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=32768,
                        temperature=0.0,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )

                # Decode output
                input_length = inputs["input_ids"].shape[1]
                generated_ids = output_ids[0, input_length:]
                raw_text = processor.decode(generated_ids, skip_special_tokens=True)

                # Extract plain text for evaluation
                plain_text = self._extract_plain_text(raw_text)
                results.append(plain_text)

                logger.debug(f"Transcribed {audio_path}: {plain_text[:100]}...")

            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                results.append("")

        return results

    def _transcribe_raw(self, audio_path: str) -> str:
        """Transcribe a single audio file and return raw output.

        Args:
            audio_path: Path to audio file.

        Returns:
            Raw model output string (may contain JSON with speakers).
        """
        import torch
        import soundfile as sf

        model, processor = self._get_model()

        audio_data, sample_rate = sf.read(audio_path)

        inputs = processor(
            audio=audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )

        inputs = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32768,
                temperature=0.0,
                do_sample=False,
                num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        raw_text = processor.decode(generated_ids, skip_special_tokens=True)

        return raw_text

    def _parse_speaker_transcripts(self, raw_output: str) -> Dict[int, str]:
        """Parse JSON output and group transcripts by speaker.

        VibeVoice can output structured JSON like:
        'assistant [{"Start":0.0,"End":14.04,"Speaker":0,"Content":"Text..."},...]'

        Args:
            raw_output: Raw model output string.

        Returns:
            Dict mapping speaker_id -> concatenated transcript text.
            Returns {0: full_text} if not in JSON format.
        """
        if not raw_output:
            return {0: ""}

        # Try to extract JSON array from output
        # Format: "assistant [{...},{...}]" or just "[{...},{...}]"
        json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if json_match:
            try:
                segments = json.loads(json_match.group())
                if isinstance(segments, list) and segments:
                    # Group by speaker
                    speaker_texts: Dict[int, List[str]] = {}
                    for segment in segments:
                        if not isinstance(segment, dict):
                            continue
                        speaker_id = segment.get("Speaker", 0)
                        content = segment.get("Content", "")
                        if speaker_id not in speaker_texts:
                            speaker_texts[speaker_id] = []
                        if content:
                            speaker_texts[speaker_id].append(content)

                    # Concatenate texts per speaker and truncate loops
                    if speaker_texts:
                        return {
                            speaker_id: self._truncate_generation_loop(" ".join(texts))
                            for speaker_id, texts in speaker_texts.items()
                        }
            except json.JSONDecodeError:
                pass  # Fall through to regex-based extraction

        # Fallback for incomplete JSON: extract Speaker and Content fields via regex
        # This handles cases where the closing ] is missing
        content_pattern = r'"Speaker"\s*:\s*(\d+)[^}]*"Content"\s*:\s*"([^"]*)"'
        matches = re.findall(content_pattern, raw_output)
        if matches:
            speaker_texts: Dict[int, List[str]] = {}
            for speaker_id_str, content in matches:
                speaker_id = int(speaker_id_str)
                if speaker_id not in speaker_texts:
                    speaker_texts[speaker_id] = []
                if content:
                    speaker_texts[speaker_id].append(content)
            if speaker_texts:
                return {
                    speaker_id: self._truncate_generation_loop(" ".join(texts))
                    for speaker_id, texts in speaker_texts.items()
                }

        # Fallback for truncated JSON: Content without closing quote (model output cut off)
        truncated_pattern = r'"Speaker"\s*:\s*(\d+)[^}]*"Content"\s*:\s*"([^"]+)$'
        truncated_match = re.search(truncated_pattern, raw_output, re.DOTALL)
        if truncated_match:
            speaker_id = int(truncated_match.group(1))
            content = self._truncate_generation_loop(truncated_match.group(2).strip())
            if content:
                return {speaker_id: content}

        # Last fallback: extract plain text
        return {0: self._extract_plain_text(raw_output)}

    def _select_best_speaker(
        self,
        speaker_transcripts: Dict[int, str],
        references: Dict[str, str],
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Select the speaker whose transcript best matches any reference.

        Oracle selection: for each parsed speaker, the candidate score is the
        minimum WER over all named references; the speaker with the lowest
        candidate score wins.

        Args:
            speaker_transcripts: Dict mapping speaker_id -> transcript text.
            references: Named references, e.g. ``{"ort": ..., "dialect": ...}``.

        Returns:
            Tuple of (best_transcript, best_speaker_id, all_speaker_metrics),
            where all_speaker_metrics maps speaker_id -> {text, <ref>_wer, <ref>_cer}.
        """
        from ..benchmark.metrics import compute_single_sample_metrics

        all_speaker_metrics: Dict[str, Any] = {}
        best_speaker_id = 0
        best_wer = float("inf")
        best_transcript = ""

        for speaker_id, transcript in speaker_transcripts.items():
            entry: Dict[str, Any] = {"text": transcript}
            candidate_wer = float("inf")
            for ref_name, ref_text in references.items():
                if not ref_text:
                    continue
                m = compute_single_sample_metrics(transcript, ref_text)
                entry[f"{ref_name}_wer"] = m["wer"]
                entry[f"{ref_name}_cer"] = m["cer"]
                candidate_wer = min(candidate_wer, m["wer"])

            all_speaker_metrics[str(speaker_id)] = entry

            if candidate_wer < best_wer:
                best_wer = candidate_wer
                best_speaker_id = speaker_id
                best_transcript = transcript

        return best_transcript, best_speaker_id, all_speaker_metrics

    def benchmark(
        self,
        dataset: "DatasetSource",
        max_samples: Optional[int] = None,
        split: str = "test",
        measure_speed: bool = True,
    ) -> BenchmarkResult:
        """Benchmark VibeVoice with multi-speaker oracle selection.

        Overrides the generic engine because VibeVoice emits per-speaker
        transcripts; for each sample it parses the speakers, scores each against
        every named reference, and keeps the best-matching speaker as the
        hypothesis. Metrics are then aggregated per reference, exactly like the
        generic runner.
        """
        import time

        from ..benchmark.metrics import compute_asr_metrics, compute_single_sample_metrics

        logger.info(
            "Starting benchmark: model=%s, dataset=%s, max_samples=%s",
            self.model_name,
            dataset.name,
            max_samples,
        )

        samples = list(dataset.iter_samples(split=split, max_samples=max_samples))
        if not samples:
            logger.warning("No samples found for benchmark")
            return BenchmarkResult(
                model=self.display_name,
                dataset=dataset.name,
                language=self.language,
                num_samples=0,
            )

        self.load()

        ref_names: List[str] = []
        for sample in samples:
            for name in sample.get_references():
                if name not in ref_names:
                    ref_names.append(name)
        primary = next(
            (s.primary_reference for s in samples if s.primary_reference in ref_names),
            ref_names[0] if ref_names else "",
        )

        hyps_by_ref = {n: [] for n in ref_names}
        refs_by_ref = {n: [] for n in ref_names}
        per_sample_results: List[SampleResult] = []
        total_infer_s = 0.0
        total_audio_s = 0.0

        for i, sample in enumerate(samples):
            logger.info("Processing sample %d/%d", i + 1, len(samples))
            refs = sample.get_references()
            total_audio_s += sample.duration or 0.0

            t0 = time.perf_counter()
            try:
                raw_output = self._transcribe_raw(sample.audio_path)
                speaker_transcripts = self._parse_speaker_transcripts(raw_output)
                best_transcript, best_speaker_id, all_speaker_metrics = (
                    self._select_best_speaker(speaker_transcripts, refs)
                )
            except Exception as e:  # noqa: BLE001
                logger.error("Error processing sample %d: %s", i, e)
                raw_output, best_transcript = "", ""
                best_speaker_id, all_speaker_metrics = None, {}
            total_infer_s += time.perf_counter() - t0

            sample_metrics = {}
            for name in ref_names:
                ref_text = refs.get(name)
                if ref_text:
                    hyps_by_ref[name].append(best_transcript)
                    refs_by_ref[name].append(ref_text)
                    sample_metrics[name] = compute_single_sample_metrics(best_transcript, ref_text)

            per_sample_results.append(
                SampleResult(
                    index=sample.dataset_info.get("index", i),
                    audio_path=sample.audio_path or "",
                    hypothesis=best_transcript,
                    duration=sample.duration,
                    references=refs,
                    metrics=sample_metrics,
                    speaker_id=sample.metadata.get("speaker_id"),
                    raw_hypothesis=raw_output,
                    extra={"selected_speaker": best_speaker_id, "all_speakers": all_speaker_metrics},
                )
            )

        results = {}
        for name in ref_names:
            if hyps_by_ref[name]:
                results[name] = compute_asr_metrics(hyps_by_ref[name], refs_by_ref[name])

        speed = {}
        if measure_speed and total_infer_s > 0:
            speed = {
                "rtfx": total_audio_s / total_infer_s,
                "total_audio_s": total_audio_s,
                "total_infer_s": total_infer_s,
            }

        result = BenchmarkResult(
            model=self.display_name,
            dataset=dataset.name,
            language=self.language,
            num_samples=len(samples),
            num_skipped=0,
            references=[n for n in ref_names if n in results],
            primary_reference=primary,
            results=results,
            speed=speed,
            per_sample=per_sample_results,
        )

        primary_wer = results.get(primary, {}).get("wer")
        logger.info(
            "Benchmark complete: primary=%s WER=%s",
            primary,
            f"{primary_wer:.2%}" if primary_wer is not None else "n/a",
        )
        return result
