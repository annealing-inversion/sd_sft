#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_PYTHON="${CONDA_PYTHON:-$PROJECT_ROOT/.conda/sdxl-lora/bin/python}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/sdxl-base-1.0}"
CONFIG="${CONFIG:-$PROJECT_ROOT/experiments/report_prompts.json}"
SEED_BASE="${SEED_BASE:-1234}"
SEEDS_PER_PROMPT="${SEEDS_PER_PROMPT:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/report_experiments/seed_${SEED_BASE}}"
EXPERIMENTS="${EXPERIMENTS:-all}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
STEPS="${STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
DEFAULT_LORA_SCALE="${DEFAULT_LORA_SCALE:-0.65}"
MAKE_MERGES="${MAKE_MERGES:-1}"
FORCE_MERGE="${FORCE_MERGE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_IMAGES="${MAX_IMAGES:-0}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"

if [[ "$DRY_RUN" == "1" ]]; then
  MAKE_MERGES=0
fi

if [[ ! -x "$CONDA_PYTHON" ]]; then
  echo "Python not found: $CONDA_PYTHON" >&2
  echo "Run: bash scripts/setup_conda_env.sh" >&2
  exit 1
fi

merge_variant() {
  local name="$1"
  local output_dir="$2"
  local weight_ghibli="$3"
  local weight_persona="$4"
  local weight_rei="$5"

  if [[ "$MAKE_MERGES" != "1" ]]; then
    return
  fi
  if [[ "$FORCE_MERGE" != "1" && -s "$output_dir/pytorch_lora_weights.safetensors" && -s "$output_dir/learned_embeds.safetensors" ]]; then
    echo "skip existing merge: $name -> $output_dir"
    return
  fi

  echo "merge $name -> $output_dir"
  "$CONDA_PYTHON" scripts/merge_sdxl_loras.py \
    --lora-dir outputs/sdxl_lora/ghibli \
    --lora-dir outputs/sdxl_lora/persona_5 \
    --lora-dir outputs/sdxl_lora/eva_rei \
    --weight "$weight_ghibli" \
    --weight "$weight_persona" \
    --weight "$weight_rei" \
    --normalize-weights \
    --output-dir "$output_dir" \
    --method concat \
    --embed-merge keep \
    --save-dtype float16
}

merge_variant "equal" "outputs/sdxl_lora/report_merge_equal_keep" 1.0 1.0 1.0
merge_variant "ghibli_heavy" "outputs/sdxl_lora/report_merge_ghibli_heavy_keep" 0.6 0.2 0.2
merge_variant "persona_heavy" "outputs/sdxl_lora/report_merge_persona_heavy_keep" 0.2 0.6 0.2
merge_variant "rei_heavy" "outputs/sdxl_lora/report_merge_rei_heavy_keep" 0.2 0.2 0.6

cmd=(
  "$CONDA_PYTHON" experiments/generate_report_images.py
  --config "$CONFIG"
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --seed-base "$SEED_BASE"
  --seeds-per-prompt "$SEEDS_PER_PROMPT"
  --experiments "$EXPERIMENTS"
  --width "$WIDTH"
  --height "$HEIGHT"
  --steps "$STEPS"
  --guidance-scale "$GUIDANCE_SCALE"
  --default-lora-scale "$DEFAULT_LORA_SCALE"
)

if [[ "$SKIP_EXISTING" == "1" ]]; then
  cmd+=(--skip-existing)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "$MAX_IMAGES" != "0" ]]; then
  cmd+=(--max-images "$MAX_IMAGES")
fi
if [[ "$CPU_OFFLOAD" == "1" ]]; then
  cmd+=(--cpu-offload)
fi

"${cmd[@]}"

if [[ "$DRY_RUN" != "1" ]]; then
  "$CONDA_PYTHON" experiments/make_contact_sheets.py \
    --metadata "$OUTPUT_DIR/all_metadata.json" \
    --output-dir "$OUTPUT_DIR/contact_sheets"
fi

echo "report experiment output: $OUTPUT_DIR"
