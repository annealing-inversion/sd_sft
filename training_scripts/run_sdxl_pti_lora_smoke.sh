#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_PYTHON="${CONDA_PYTHON:-$PROJECT_ROOT/.conda/sdxl-lora/bin/python}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/sdxl-base-1.0}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/Ghibli}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/sdxl_lora_smoke}"
PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-<smoke_headshot>}"

"$CONDA_PYTHON" training_scripts/train_sdxl_pti_lora.py \
  --pretrained-model-name-or-path "$MODEL_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --placeholder-token "$PLACEHOLDER_TOKEN" \
  --initializer-token style \
  --default-caption "anime portrait, close-up face, smoke test" \
  --resolution 256 \
  --center-crop \
  --train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-train-steps 2 \
  --ti-steps 1 \
  --checkpointing-steps 0 \
  --rank 4 \
  --lora-alpha 4 \
  --learning-rate 1e-4 \
  --ti-lr 5e-4 \
  --gradient-checkpointing \
  --allow-tf32 \
  --mixed-precision fp16 \
  --dataloader-num-workers 0 \
  --seed 7
