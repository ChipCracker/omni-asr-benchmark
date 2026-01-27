"""VibeVoice Evaluator for Microsoft's VibeVoice-ASR model."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .base_evaluator import BaseEvaluator, EvaluationResult, SampleResult

if TYPE_CHECKING:
    from ..datasets.base import DatasetSource

logger = logging.getLogger(__name__)


class VibeVoiceEvaluator(BaseEvaluator):
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
        dialect_ref: str,
        ort_ref: Optional[str],
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Select speaker with lowest WER.

        Args:
            speaker_transcripts: Dict mapping speaker_id -> transcript text.
            dialect_ref: Dialect reference transcription.
            ort_ref: Optional orthographic reference transcription.

        Returns:
            Tuple of (best_transcript, best_speaker_id, all_speaker_metrics).
            all_speaker_metrics contains {speaker_id: {text, dialect_wer, ort_wer}}.
        """
        from .metrics import compute_single_sample_metrics

        all_speaker_metrics: Dict[str, Any] = {}
        best_speaker_id = 0
        best_wer = float("inf")
        best_transcript = ""

        for speaker_id, transcript in speaker_transcripts.items():
            dialect_metrics = compute_single_sample_metrics(transcript, dialect_ref)
            ort_metrics = (
                compute_single_sample_metrics(transcript, ort_ref)
                if ort_ref
                else {"wer": None, "cer": None}
            )

            all_speaker_metrics[str(speaker_id)] = {
                "text": transcript,
                "dialect_wer": dialect_metrics["wer"],
                "dialect_cer": dialect_metrics["cer"],
                "ort_wer": ort_metrics["wer"],
                "ort_cer": ort_metrics["cer"],
            }

            # Select based on minimum WER (prefer dialect, fall back to ort)
            candidate_wer = dialect_metrics["wer"]
            if ort_metrics["wer"] is not None:
                candidate_wer = min(candidate_wer, ort_metrics["wer"])

            if candidate_wer < best_wer:
                best_wer = candidate_wer
                best_speaker_id = speaker_id
                best_transcript = transcript

        return best_transcript, best_speaker_id, all_speaker_metrics

    def evaluate(
        self,
        dataset_source: "DatasetSource",
        max_samples: Optional[int] = None,
        split: str = "test",
    ) -> EvaluationResult:
        """Evaluate the ASR model on a dataset with multi-speaker handling.

        This override handles VibeVoice's multi-speaker output by:
        1. Parsing speaker segments from JSON output
        2. Computing WER for each speaker against references
        3. Selecting the speaker with lowest WER as the hypothesis

        Args:
            dataset_source: The dataset source to evaluate on.
            max_samples: Maximum number of samples to evaluate (None for all).
            split: Dataset split to use.

        Returns:
            EvaluationResult containing metrics and per-sample results.
        """
        from ..datasets.base import Sample
        from .metrics import compute_asr_metrics, compute_single_sample_metrics

        logger.info(
            f"Starting evaluation: model={self.model_name}, dataset={dataset_source.name}, "
            f"max_samples={max_samples}"
        )

        samples: List[Sample] = list(
            dataset_source.iter_samples(split=split, max_samples=max_samples)
        )

        if not samples:
            logger.warning("No samples found for evaluation")
            return EvaluationResult(
                model=self.model_name,
                dataset=dataset_source.name,
                language=self.language,
                num_samples=0,
                num_skipped=0,
            )

        logger.info(f"Evaluating {len(samples)} samples")

        all_hypotheses: List[str] = []
        dialect_references: List[str] = []
        ort_references: List[str] = []
        per_sample_results: List[SampleResult] = []

        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i + 1}/{len(samples)}")

            try:
                # Get raw transcription
                raw_output = self._transcribe_raw(sample.audio_path)

                # Parse speaker transcripts
                speaker_transcripts = self._parse_speaker_transcripts(raw_output)

                # Select best speaker based on WER
                best_transcript, best_speaker_id, all_speaker_metrics = (
                    self._select_best_speaker(
                        speaker_transcripts,
                        sample.transcript,
                        sample.ort_transcript,
                    )
                )

                # Compute final metrics for selected speaker
                dialect_metrics = compute_single_sample_metrics(
                    best_transcript, sample.transcript
                )
                ort_metrics = (
                    compute_single_sample_metrics(best_transcript, sample.ort_transcript)
                    if sample.ort_transcript
                    else {"wer": None, "cer": None}
                )

                all_hypotheses.append(best_transcript)
                dialect_references.append(sample.transcript)
                ort_references.append(sample.ort_transcript or "")

                per_sample_results.append(
                    SampleResult(
                        index=sample.dataset_info.get("index", i),
                        audio_path=sample.audio_path or "",
                        hypothesis=best_transcript,
                        dialect_reference=sample.transcript,
                        ort_reference=sample.ort_transcript,
                        dialect_wer=dialect_metrics["wer"],
                        dialect_cer=dialect_metrics["cer"],
                        ort_wer=ort_metrics["wer"],
                        ort_cer=ort_metrics["cer"],
                        duration=sample.duration,
                        speaker_id=sample.metadata.get("speaker_id"),
                        raw_hypothesis=raw_output,
                        selected_speaker=best_speaker_id,
                        all_speakers=all_speaker_metrics,
                    )
                )

                logger.debug(
                    f"Sample {i}: selected speaker {best_speaker_id}, "
                    f"dialect_wer={dialect_metrics['wer']:.2%}"
                )

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                all_hypotheses.append("")
                dialect_references.append(sample.transcript)
                ort_references.append(sample.ort_transcript or "")

                per_sample_results.append(
                    SampleResult(
                        index=sample.dataset_info.get("index", i),
                        audio_path=sample.audio_path or "",
                        hypothesis="",
                        dialect_reference=sample.transcript,
                        ort_reference=sample.ort_transcript,
                        dialect_wer=1.0,
                        dialect_cer=1.0,
                        ort_wer=1.0 if sample.ort_transcript else None,
                        ort_cer=1.0 if sample.ort_transcript else None,
                        duration=sample.duration,
                        speaker_id=sample.metadata.get("speaker_id"),
                    )
                )

        # Compute aggregate metrics
        dialect_metrics = compute_asr_metrics(all_hypotheses, dialect_references)

        valid_ort_pairs = [
            (hyp, ref) for hyp, ref in zip(all_hypotheses, ort_references) if ref
        ]
        if valid_ort_pairs:
            ort_hyps, ort_refs = zip(*valid_ort_pairs)
            ort_metrics = compute_asr_metrics(list(ort_hyps), list(ort_refs))
        else:
            ort_metrics = {
                "wer": None,
                "cer": None,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "num_samples": 0,
            }

        result = EvaluationResult(
            model=self.model_name,
            dataset=dataset_source.name,
            language=self.language,
            num_samples=len(samples),
            num_skipped=0,
            results={
                "dialect_reference": dialect_metrics,
                "ort_reference": ort_metrics,
            },
            per_sample=per_sample_results,
        )

        logger.info(
            f"Evaluation complete: "
            f"Dialect WER={dialect_metrics['wer']:.2%}"
            + (f", ORT WER={ort_metrics['wer']:.2%}" if ort_metrics["wer"] else "")
        )

        return result
