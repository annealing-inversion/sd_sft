#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_PYTHON="${CONDA_PYTHON:-$PROJECT_ROOT/.conda/sdxl-lora/bin/python}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/sdxl-base-1.0}"
STYLE="${STYLE:-ghibli}"

case "$STYLE" in
  ghibli)
    DEFAULT_DATA_DIR="$PROJECT_ROOT/data/Ghibli"
    DEFAULT_PLACEHOLDER_TOKEN="<ghibli_headshot>"
    ;;
  persona_5)
    DEFAULT_DATA_DIR="$PROJECT_ROOT/data/persona_5"
    DEFAULT_PLACEHOLDER_TOKEN="<persona_5_headshot>"
    ;;
  eva_rei)
    DEFAULT_DATA_DIR="$PROJECT_ROOT/data/EVA_rei"
    DEFAULT_PLACEHOLDER_TOKEN="<eva_rei_headshot>"
    ;;
  *)
    DEFAULT_DATA_DIR="$PROJECT_ROOT/data/$STYLE"
    DEFAULT_PLACEHOLDER_TOKEN="<${STYLE}_headshot>"
    ;;
esac

DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/sdxl_lora/$STYLE}"
PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-$DEFAULT_PLACEHOLDER_TOKEN}"
DEFAULT_CAPTION="${DEFAULT_CAPTION:-anime portrait, close-up face, character headshot}"
NUM_PROCESSES="${NUM_PROCESSES:-3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

"$CONDA_PYTHON" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  training_scripts/train_sdxl_pti_lora.py \
  --pretrained-model-name-or-path "$MODEL_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --placeholder-token "$PLACEHOLDER_TOKEN" \
  --initializer-token style \
  --default-caption "$DEFAULT_CAPTION" \
  --resolution 1024 \
  --center-crop \
  --train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-train-steps 2000 \
  --ti-steps 500 \
  --checkpointing-steps 200 \
  --rank 16 \
  --lora-alpha 16 \
  --learning-rate 1e-4 \
  --text-encoder-lora-lr 5e-6 \
  --ti-lr 5e-4 \
  --train-text-encoder-lora \
  --gradient-checkpointing \
  --allow-tf32 \
  --mixed-precision fp16 \
  --seed 42
