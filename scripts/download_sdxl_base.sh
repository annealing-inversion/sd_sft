#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO_ID="${REPO_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/sdxl-base-1.0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

FILES=(
  "model_index.json"
  "scheduler/scheduler_config.json"
  "text_encoder/config.json"
  "text_encoder/model.fp16.safetensors"
  "text_encoder_2/config.json"
  "text_encoder_2/model.fp16.safetensors"
  "tokenizer/merges.txt"
  "tokenizer/special_tokens_map.json"
  "tokenizer/tokenizer_config.json"
  "tokenizer/vocab.json"
  "tokenizer_2/merges.txt"
  "tokenizer_2/special_tokens_map.json"
  "tokenizer_2/tokenizer_config.json"
  "tokenizer_2/vocab.json"
  "unet/config.json"
  "unet/diffusion_pytorch_model.fp16.safetensors"
  "vae/config.json"
  "vae/diffusion_pytorch_model.fp16.safetensors"
)

download_file() {
  local rel_path="$1"
  local dst="$MODEL_DIR/$rel_path"
  local url="$HF_ENDPOINT/$REPO_ID/resolve/main/$rel_path"

  mkdir -p "$(dirname "$dst")"
  if [[ "$FORCE_DOWNLOAD" != "1" && -s "$dst" ]]; then
    echo "skip existing $rel_path"
    return
  fi

  echo "download $rel_path"
  curl -L --fail --retry 5 --retry-delay 2 -o "$dst" "$url"
}

for rel_path in "${FILES[@]}"; do
  download_file "$rel_path"
done

echo "SDXL base ready: $MODEL_DIR"
