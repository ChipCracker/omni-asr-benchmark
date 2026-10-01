#!/bin/bash
#SBATCH --job-name=de-testsets
#SBATCH --partition=p4,p2,p1
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-13
#SBATCH --output=/nfs1/scratch/staff/witzlch/omni-asr-test/slurm/logs/%x-%A_%a.out
# German public test sets (src/datasets/de_testsets.py): Tuda-De test (Kinect-RAW
# far field, Yamaha close talk), Common Voice de test and the Verbmobil BAS TEST
# set, for every model with a full leaderboard row. Rows and venvs are the
# verified mapping of slurm/nscc_bench.sbatch; the datasets of one job run one
# after the other (DATASETS, comma list). Each (model, dataset) writes
# $RESDIR/<model>__<dataset>.json (default results_tuda_cv/, a staging dir that
# is re-scored and moved into results/ before the leaderboard is rendered) and
# resumes via slurm/ckpt/ after a timeout.
#
#   sbatch slurm/de_testsets_bench.sbatch                                     # Tuda, all rows
#   DATASETS=cv_de_test sbatch --time=04:00:00 slurm/de_testsets_bench.sbatch
#   sbatch --array=3 --qos=advanced slurm/de_testsets_bench.sbatch            # one row
#   ROWLIST="2 3 12" sbatch --array=0 slurm/de_testsets_bench.sbatch          # several rows, one job
set -uo pipefail
PROJ=/nfs1/scratch/staff/witzlch/omni-asr-test
STUDENT=/nfs1/scratch/students/witzlch88229/projects/omni-asr-test
cd "$PROJ"
export TMPDIR=/var/tmp
export HF_HOME=/nfs1/scratch/staff/witzlch/hf-cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_TOKEN_PATH=/dev/null
export NUMBA_CACHE_DIR=/var/tmp/numba_$USER
mkdir -p "$NUMBA_CACHE_DIR"
STUDENT_HUB=/home/witzlch88229/.cache/huggingface/hub
DATASETS=${DATASETS:-tuda_test_kinect_raw,tuda_test_yamaha}
# New results are staged outside results/ until they are re-scored with the final
# normalize_text and the leaderboard is rendered once; then they move into results/.
RESDIR=${RESDIR:-results_tuda_cv}
mkdir -p "$RESDIR"

#        model | venv (under $STUDENT) | batch | batch for cv_de_test (all clips < 16 s) | extra env
ROWS=(
  "openai/whisper-large-v3|.venv-granite|1|16|"     # batch > 1 trips the pipeline padding on > 30 s utterances (Tuda 4, Verbmobil 17)
  "nyrahealth/CrisperWhisper|.venv-granite|4|16|"
  "nvidia/parakeet-tdt-0.6b-v3|.venv-nemo|8|16|NEMO_MODEL_FILE=$STUDENT_HUB/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/8844af98cdfda411074894c417dfaf0cfb9c6a73/parakeet-tdt-0.6b-v3.nemo"
  "nvidia/canary-1b-v2|.venv-nemo|8|16|"
  "mistralai/Voxtral-Mini-3B-2507|.venv-granite-nar|4|8|HF_HUB_OFFLINE=0"
  "Qwen/Qwen3-ASR-1.7B|.venv-granite|16|16|"
  "Qwen/Qwen3-ASR-0.6B|.venv-granite|16|16|"
  "CohereLabs/cohere-transcribe-03-2026|.venv-granite-nar|4|8|"
  "ibm-granite/granite-speech-4.1-2b|.venv-granite|1|1|"
  "ibm-granite/granite-speech-4.1-2b-nar|.venv-granite-nar|1|1|"
  "microsoft/Phi-4-multimodal-instruct|.venv-phi4|2|4|"
  "omniASR_CTC_1B_v2|.venv-omni|8|8|XDG_CACHE_HOME=/nfs1/scratch/staff/witzlch/.cache"
  "oscillator-rnnt/exp_prod110m_de_rnnt_ft_v3b_lr8e4_decay|OSC|8|8|"
  "http://141.75.89.18:8099|.venv-granite|8|8|"     # HTTP client (AsrHttpEvaluator); display name large-v3 @141.75.89.18 (server)
)
run_row() {  # run_row <row index>: all DATASETS for one model, in a subshell
  (
  IFS="|" read -r MODEL VENV BATCH BATCH_SHORT ROWENV <<< "${ROWS[$1]}"
  [ -n "$ROWENV" ] && export "$ROWENV"
  case "$MODEL" in
    http*) SAFE=whisper-large-v3-server ;;   # keep historical result-file name
    *)     SAFE=$(echo "$MODEL" | tr "/" "_") ;;
  esac
  EXTRA=""
  if [ "$VENV" = "OSC" ]; then
    PY=/nfs1/scratch/staff/witzlch/oscillator-asr-resonant/.venv/bin/python
    export PYTHONPATH=/nfs1/scratch/staff/witzlch/osc-v3b/pylib
    export OSC_ASR_SRC=/nfs1/scratch/staff/witzlch/osc-v3b/oscillator-asr-resonant/src
    export OSC_RNNT_ROOT=/nfs1/scratch/staff/witzlch/osc-v3b
    EXTRA="--model-dir /nfs1/scratch/staff/witzlch/osc-v3b/ckpt/v3b_lr8e4_decay_best.ckpt"
  else
    PY="$STUDENT/$VENV/bin/python"
  fi
  RC=0
  echo "### row $1 host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) model $MODEL venv $VENV datasets $DATASETS"
  for DS in ${DATASETS//,/ }; do
    B=$BATCH
    [ "$DS" = cv_de_test ] && B=$BATCH_SHORT
    OUT="$RESDIR/${SAFE}__${DS}.json"
    if [ -z "${FORCE:-}" ] && { [ -s "$OUT" ] || [ -s "results/${SAFE}__${DS}.json" ]; }; then
      echo "skip $SAFE $DS (result exists)"; continue
    fi
    echo "=== $SAFE $DS batch $B $(date +%T)"
    "$PY" scripts/benchmark.py --model "$MODEL" --dataset "$DS" --batch-size "$B" \
      --output "$OUT" --no-leaderboard --checkpoint "slurm/ckpt/${SAFE}__${DS}.json" $EXTRA \
      || { echo "FEHLER $SAFE $DS rc=$?"; RC=1; }
    echo "=== done $SAFE $DS $(date +%T)"
  done
  exit $RC
  )
}

mkdir -p slurm/ckpt
# ROWLIST="2 3 12" runs several rows one after the other in one job (saves job
# slots when the QoS submit limits are tight); otherwise the array index picks the row.
RC=0
for R in ${ROWLIST:-$SLURM_ARRAY_TASK_ID}; do
  run_row "$R" || RC=1
done
echo "ENDE $(date +%T) rc=$RC"
exit $RC
