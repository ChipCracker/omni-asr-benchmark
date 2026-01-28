# OmniASR Evaluation Framework

A framework for evaluating Automatic Speech Recognition (ASR) models on dialect speech datasets.

## Supported Models

- **OmniASR** (default) - Facebook's omnilingual ASR model
- **Whisper** - OpenAI's Whisper models via HuggingFace (e.g., `openai/whisper-large-v3`)
- **CrisperWhisper** - nyrahealth's fine-tuned Whisper with verbatim transcription (e.g., `nyrahealth/CrisperWhisper`)
- **Parakeet** - NVIDIA NeMo Parakeet models (e.g., `nvidia/parakeet-ctc-1.1b`)
- **Canary-Qwen** - NVIDIA NeMo SALM model (e.g., `nvidia/canary-qwen-2.5b`, English-only)
- **Voxtral** - Mistral AI's Voxtral models (e.g., `mistralai/Voxtral-Mini-3B-2507`)
- **VibeVoice** - Microsoft's VibeVoice-ASR model (9B params, up to 60 min audio)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd omni-asr-test

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install OmniASR (for OmniASR models)
pip install git+https://github.com/facebookresearch/omnilingual-asr.git

# Install Whisper dependencies (optional)
pip install transformers torch accelerate

# Install CrisperWhisper dependencies (optional, for CrisperWhisper)
pip install git+https://github.com/nyrahealth/transformers.git@crisper_whisper

# Install NeMo dependencies (optional, for Parakeet)
pip install nemo-toolkit[asr]

# Install NeMo trunk (optional, for Canary-Qwen - requires PyTorch 2.6+)
pip install 'nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git'

# Install Voxtral dependencies (optional, for Voxtral)
pip install -U transformers
pip install --upgrade 'mistral-common[audio]'

# Install VibeVoice dependencies (optional, for VibeVoice-ASR)
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .[asr]
```

## Configuration

Copy the environment template and configure:

```bash
cp .env.template .env
```

Set the `BAS_RVG1_DATA_DIR` environment variable to point to your BAS RVG1 dataset directory.

## Usage

### Evaluate with OmniASR (default)

```bash
python scripts/evaluate_rvg1.py
```

### Evaluate with Whisper

```bash
python scripts/evaluate_rvg1.py --model-card openai/whisper-large-v3
```

### Evaluate with CrisperWhisper

```bash
python scripts/evaluate_rvg1.py --model-card nyrahealth/CrisperWhisper
```

Note: CrisperWhisper is a fine-tuned Whisper Large V3 that provides verbatim transcription (including filler words like "um", "uh") and improved word-level timestamps.

### Evaluate with Parakeet

```bash
python scripts/evaluate_rvg1.py --model-card nvidia/parakeet-ctc-1.1b
```

### Evaluate with Canary-Qwen

```bash
python scripts/evaluate_rvg1.py --model-card nvidia/canary-qwen-2.5b
```

Note: Canary-Qwen is English-only (5.63% mean WER on HuggingFace OpenASR Leaderboard). German evaluation will show degraded results.

### Evaluate with Voxtral

```bash
python scripts/evaluate_rvg1.py --model-card mistralai/Voxtral-Mini-3B-2507
```

Note: Voxtral supports German (de), English (en), French (fr), Spanish (es), Portuguese (pt), Italian (it), Dutch (nl), and Hindi (hi). Requires ~9.5 GB GPU RAM.

### Evaluate with VibeVoice

```bash
python scripts/evaluate_rvg1.py --model-card microsoft/VibeVoice-ASR --batch-size 1
```

Note: VibeVoice-ASR is a 9B parameter model requiring ~18-20 GB VRAM. It supports up to 60 minutes of continuous audio and includes speaker diarization (stripped for WER evaluation).

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Path to BAS RVG1 data directory | `BAS_RVG1_DATA_DIR` env var |
| `--model-card` | Model identifier | `omniASR_LLM_Unlimited_7B_v2` |
| `--language` | Language code (e.g., `deu_Latn`) | `deu_Latn` |
| `--batch-size` | Batch size for inference | `2` |
| `--max-samples` | Max samples to evaluate | All |
| `--channel` | Audio channel (`c`/`h`/`l`) | `c` |
| `--output` | Output file path | `results/<model>_evaluation.json` |
| `--verbose` | Enable verbose logging | `False` |

## Output

Results are saved as JSON files in the `results/` directory with the model name as prefix:

- `results/omniASR_LLM_Unlimited_7B_v2_evaluation.json`
- `results/openai_whisper-large-v3_evaluation.json`
- `results/nyrahealth_CrisperWhisper_evaluation.json`
- `results/nvidia_parakeet-ctc-1.1b_evaluation.json`
- `results/nvidia_canary-qwen-2.5b_evaluation.json`
- `results/mistralai_Voxtral-Mini-3B-2507_evaluation.json`
- `results/microsoft_VibeVoice-ASR_evaluation.json`

Each result file contains:
- Model and dataset metadata
- Aggregate WER/CER metrics
- Per-sample results with hypotheses and references

## Analyzing Results

Use the analysis script to print a minimal summary:

```bash
python scripts/analyze_results.py results/openai_whisper-large-v3_evaluation.json
```

The summary includes:
- Model, dataset, language, timestamp
- Total samples and skipped count
- Dialect WER/CER (and ORT WER/CER if available)

## Visualizing Results

Generate a comparison bar chart of all evaluation results:

```bash
python scripts/plot_results.py
```

This creates `results/comparison_chart.png` with:
- Grouped bars for Dialect WER and ORT WER per model
- Error bars showing standard deviation across samples
- Symlog scale to handle WER values >100%
- Models sorted by ORT WER (ascending)

## Project Structure

```
omni-asr-test/
├── src/
│   ├── datasets/
│   │   ├── base.py           # Base dataset classes
│   │   └── bas_rvg1.py       # BAS RVG1 dataset loader
│   └── evaluation/
│       ├── base_evaluator.py  # Abstract base evaluator
│       ├── evaluator.py       # OmniASR evaluator + factory
│       ├── whisper_evaluator.py
│       ├── crisperwhisper_evaluator.py
│       ├── parakeet_evaluator.py
│       ├── canary_evaluator.py
│       ├── voxtral_evaluator.py
│       ├── vibevoice_evaluator.py
│       └── metrics.py         # WER/CER computation
├── scripts/
│   ├── evaluate_rvg1.py      # Main evaluation script
│   ├── analyze_results.py    # Result analysis script
│   └── plot_results.py       # Visualization script
├── results/                   # Evaluation output
└── requirements.txt
```

## Adding New Models

1. Create a new evaluator class inheriting from `BaseEvaluator`
2. Implement the `transcribe_batch()` method
3. Register the model in `get_evaluator()` factory function in `evaluator.py`
