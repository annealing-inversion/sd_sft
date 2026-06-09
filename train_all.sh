#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d data ]]; then
  echo "Error: data directory does not exist." >&2
  exit 1
fi

found=0
while IFS= read -r -d '' task_dir; do
  found=1
  style="$(basename "$task_dir")"
  echo
  echo "Starting sequential training for: $style"
  bash train_one.sh "$style"
done < <(find data -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [[ "$found" -eq 0 ]]; then
  echo "Error: no first-level task directories found under data/." >&2
  exit 1
fi
