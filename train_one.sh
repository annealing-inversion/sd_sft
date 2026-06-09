#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

usage() {
  echo "Usage: bash train_one.sh <style_or_character_name>"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

STYLE="$1"

if [[ ! "$STYLE" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Error: style directory name contains spaces or special characters: $STYLE" >&2
  echo "Please rename the directory under data/ to use only letters, numbers, underscore, or hyphen." >&2
  exit 2
fi

DATA_DIR="data/$STYLE"
OUTPUT_DIR="outputs/sdxl_lora/$STYLE"
PLACEHOLDER_TOKEN="<${STYLE}_style>"
FREE_MEM_THRESHOLD_MIB="${FREE_MEM_THRESHOLD_MIB:-5120}"
GPU_MODE="${GPU_MODE:-auto}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: dataset directory does not exist: $DATA_DIR" >&2
  exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Error: output path already exists, refusing to overwrite: $OUTPUT_DIR" >&2
  exit 1
fi

require_nvidia_smi() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found; cannot choose GPUs safely." >&2
    exit 1
  fi
}

query_a100_gpus() {
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits |
    awk -F, '{
      idx=$1; name=$2; used=$3;
      gsub(/^[ \t]+|[ \t]+$/, "", idx);
      gsub(/^[ \t]+|[ \t]+$/, "", name);
      gsub(/^[ \t]+|[ \t]+$/, "", used);
      if (name ~ /A100/) {
        print idx " " used;
      }
    }'
}

select_auto_gpus() {
  require_nvidia_smi

  mapfile -t gpu_rows < <(query_a100_gpus)
  if [[ "${#gpu_rows[@]}" -lt 1 ]]; then
    echo "Error: no A100 GPUs found. Current nvidia-smi status:" >&2
    nvidia-smi >&2
    exit 1
  fi

  free_gpu_ids=()
  for row in "${gpu_rows[@]}"; do
    gpu_id="${row%% *}"
    mem_used="${row##* }"
    if [[ "$mem_used" =~ ^[0-9]+$ ]] && (( mem_used < FREE_MEM_THRESHOLD_MIB )); then
      free_gpu_ids+=("$gpu_id")
    fi
  done

  if (( ${#free_gpu_ids[@]} >= 2 )); then
    CUDA_VISIBLE_DEVICES="${free_gpu_ids[0]},${free_gpu_ids[1]}"
    NUM_PROCESSES=2
  elif (( ${#free_gpu_ids[@]} == 1 )); then
    CUDA_VISIBLE_DEVICES="${free_gpu_ids[0]}"
    NUM_PROCESSES=1
  else
    echo "Error: no A100 GPU is below ${FREE_MEM_THRESHOLD_MIB} MiB used. Current nvidia-smi status:" >&2
    nvidia-smi >&2
    exit 1
  fi
}

select_manual_gpus() {
  require_nvidia_smi

  case "$GPU_MODE" in
    1gpu)
      if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        selected="$CUDA_VISIBLE_DEVICES"
      else
        mapfile -t gpu_rows < <(query_a100_gpus)
        selected=""
        for row in "${gpu_rows[@]}"; do
          gpu_id="${row%% *}"
          mem_used="${row##* }"
          if [[ "$mem_used" =~ ^[0-9]+$ ]] && (( mem_used < FREE_MEM_THRESHOLD_MIB )); then
            selected="$gpu_id"
            break
          fi
        done
      fi
      if [[ -z "$selected" ]]; then
        echo "Error: GPU_MODE=1gpu requested but no free A100 GPU was found." >&2
        nvidia-smi >&2
        exit 1
      fi
      CUDA_VISIBLE_DEVICES="$selected"
      NUM_PROCESSES=1
      ;;
    2gpu)
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
      NUM_PROCESSES=2
      ;;
    *)
      echo "Error: unsupported GPU_MODE=$GPU_MODE. Use auto, 1gpu, or 2gpu." >&2
      exit 2
      ;;
  esac
}

if [[ "$GPU_MODE" == "auto" ]]; then
  select_auto_gpus
else
  select_manual_gpus
fi

echo "STYLE=$STYLE"
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PLACEHOLDER_TOKEN=$PLACEHOLDER_TOKEN"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not starting training."
  exit 0
fi

export CUDA_VISIBLE_DEVICES
export STYLE
export DATA_DIR
export OUTPUT_DIR
export PLACEHOLDER_TOKEN
export NUM_PROCESSES
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"

bash training_scripts/run_sdxl_pti_lora_single.sh
