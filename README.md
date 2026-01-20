# OmniASR Evaluation Framework

A framework for evaluating Automatic Speech Recognition (ASR) models on dialect speech datasets.

## Supported Models

- **OmniASR** (default) - Facebook's omnilingual ASR model
- **Whisper** - OpenAI's Whisper models via HuggingFace (e.g., `openai/whisper-large-v3`)
- **Parakeet** - NVIDIA NeMo Parakeet models (e.g., `nvidia/parakeet-ctc-1.1b`)

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

# Install NeMo dependencies (optional, for Parakeet)
pip install nemo-toolkit[asr]
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

### Evaluate with Parakeet

```bash
python scripts/evaluate_rvg1.py --model-card nvidia/parakeet-ctc-1.1b
```

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
- `results/nvidia_parakeet-ctc-1.1b_evaluation.json`

Each result file contains:
- Model and dataset metadata
- Aggregate WER/CER metrics
- Per-sample results with hypotheses and references

## Analyzing Results

Use the analysis script to inspect evaluation results in detail:

```bash
python scripts/analyze_results.py results/openai_whisper-large-v3_evaluation.json
```

### Analysis Options

| Option | Description | Default |
|--------|-------------|---------|
| `result_file` | Path to evaluation JSON file | (required) |
| `--top-n` | Number of best/worst samples to show | `5` |
| `--show-speakers` | Show per-speaker WER statistics | `False` |
| `--show-examples` | Show reference vs. hypothesis examples | `False` |

### Example with All Options

```bash
python scripts/analyze_results.py results/openai_whisper-large-v3_evaluation.json \
    --top-n 10 --show-speakers --show-examples
```

The analysis includes:
- Overview (model, dataset, aggregate metrics)
- Distribution statistics (min, max, mean, median, std for WER/CER)
- Top-N best and worst samples by WER
- Per-speaker breakdown (with `--show-speakers`)
- Side-by-side REF vs HYP transcriptions (with `--show-examples`)

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
│       ├── parakeet_evaluator.py
│       └── metrics.py         # WER/CER computation
├── scripts/
│   ├── evaluate_rvg1.py      # Main evaluation script
│   └── analyze_results.py    # Result analysis script
├── results/                   # Evaluation output
└── requirements.txt
```

## Adding New Models

1. Create a new evaluator class inheriting from `BaseEvaluator`
2. Implement the `transcribe_batch()` method
3. Register the model in `get_evaluator()` factory function in `evaluator.py`
