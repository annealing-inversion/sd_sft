#!/usr/bin/env python
"""Merge SDXL PTI+LoRA style checkpoints.

The default concat merge preserves the weighted sum of LoRA deltas by stacking
the rank dimension. For example, merging two rank-16 adapters produces one
rank-32 adapter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

LORA_FILE = "pytorch_lora_weights.safetensors"
EMBEDS_FILE = "learned_embeds.safetensors"
METADATA_FILE = "training_metadata.json"
LORA_SUFFIX_PAIRS = [
    (".lora.down.weight", ".lora.up.weight"),
    (".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SDXL LoRA directories saved by this project.")
    parser.add_argument("--lora-dir", action="append", required=True, help="Input LoRA directory. Repeat.")
    parser.add_argument("--weight", action="append", type=float, help="Input weight. Repeat in lora-dir order.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--placeholder-token", default="<merged_headshot>")
    parser.add_argument(
        "--embed-merge",
        choices=["average", "keep"],
        default="average",
        help="average creates one merged trigger token; keep preserves each input trigger token.",
    )
    parser.add_argument(
        "--method",
        choices=["concat", "linear"],
        default="concat",
        help="concat preserves weighted LoRA deltas with increased rank; linear is a same-rank heuristic.",
    )
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        help="Normalize provided weights so they sum to 1. Omitted weights are always equal-normalized.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["same", "float32", "float16"],
        default="same",
        help="Output tensor dtype. same uses the first adapter tensor dtype.",
    )
    return parser.parse_args()


def resolve_weights(num_inputs: int, raw_weights: list[float] | None, normalize: bool) -> list[float]:
    if raw_weights is None:
        return [1.0 / num_inputs] * num_inputs
    if len(raw_weights) != num_inputs:
        raise ValueError(f"Expected {num_inputs} weights, got {len(raw_weights)}")
    weights = list(raw_weights)
    if normalize:
        total = sum(weights)
        if total == 0:
            raise ValueError("Cannot normalize weights because their sum is zero.")
        weights = [w / total for w in weights]
    return weights


def output_dtype(first_dtype: torch.dtype, save_dtype: str) -> torch.dtype:
    if save_dtype == "same":
        return first_dtype
    if save_dtype == "float32":
        return torch.float32
    if save_dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported save dtype: {save_dtype}")


def is_alpha_key(key: str) -> bool:
    return key.endswith(".alpha") or key.endswith(".lora.alpha")


def lora_pair_keys(down_key: str) -> tuple[str, str] | None:
    for down_suffix, up_suffix in LORA_SUFFIX_PAIRS:
        if down_key.endswith(down_suffix):
            base_key = down_key[: -len(down_suffix)]
            return base_key, f"{base_key}{up_suffix}"
    return None


def alpha_scale(state: dict[str, torch.Tensor], base_key: str, rank: int) -> float:
    for alpha_key in (f"{base_key}.alpha", f"{base_key}.lora.alpha"):
        if alpha_key in state:
            return float(state[alpha_key].reshape(()).item()) / float(rank)
    return 1.0


def lora_down_keys(state: dict[str, torch.Tensor]) -> list[str]:
    return sorted(key for key in state if lora_pair_keys(key) is not None)


def validate_lora_states(states: list[dict[str, torch.Tensor]]) -> list[str]:
    reference_keys = set(states[0].keys())
    for index, state in enumerate(states[1:], start=2):
        keys = set(state.keys())
        missing = sorted(reference_keys - keys)
        extra = sorted(keys - reference_keys)
        if missing or extra:
            raise ValueError(
                f"LoRA key mismatch in input {index}: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

    down_keys = lora_down_keys(states[0])
    if not down_keys:
        raise ValueError("No LoRA down weights found.")

    supported_keys = set()
    for down_key in down_keys:
        pair = lora_pair_keys(down_key)
        if pair is None:
            raise ValueError(f"Unsupported down key suffix for {down_key}")
        _, up_key = pair
        if up_key not in states[0]:
            raise ValueError(f"Missing up weight for {down_key}")
        supported_keys.add(down_key)
        supported_keys.add(up_key)
    supported_keys.update(key for key in states[0] if is_alpha_key(key))

    unsupported = sorted(reference_keys - supported_keys)
    if unsupported:
        raise ValueError(f"Unsupported keys in LoRA file: {unsupported[:10]}")
    return down_keys


def merge_concat(
    states: list[dict[str, torch.Tensor]],
    down_keys: list[str],
    weights: list[float],
    save_dtype: str,
) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}

    for down_key in down_keys:
        pair = lora_pair_keys(down_key)
        if pair is None:
            raise ValueError(f"Unsupported down key suffix for {down_key}")
        base_key, up_key = pair

        down_parts = []
        up_parts = []
        first_dtype = states[0][down_key].dtype
        target_dtype = output_dtype(first_dtype, save_dtype)

        for state, weight in zip(states, weights, strict=True):
            down = state[down_key]
            up = state[up_key]
            rank = int(down.shape[0])
            if int(up.shape[1]) != rank:
                raise ValueError(f"Rank mismatch for {base_key}: down={down.shape}, up={up.shape}")
            scale = weight * alpha_scale(state, base_key, rank)
            down_parts.append(down.float())
            up_parts.append(up.float() * scale)

        merged[down_key] = torch.cat(down_parts, dim=0).to(dtype=target_dtype)
        merged[up_key] = torch.cat(up_parts, dim=1).to(dtype=target_dtype)

    return merged


def merge_linear(
    states: list[dict[str, torch.Tensor]],
    weights: list[float],
    save_dtype: str,
) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    keys = sorted(key for key in states[0] if not is_alpha_key(key))

    for key in keys:
        shape = states[0][key].shape
        first_dtype = states[0][key].dtype
        target_dtype = output_dtype(first_dtype, save_dtype)
        if any(state[key].shape != shape for state in states[1:]):
            raise ValueError(f"Linear merge requires identical tensor shapes for key {key}")
        value = torch.zeros_like(states[0][key], dtype=torch.float32)
        for state, weight in zip(states, weights, strict=True):
            value += state[key].float() * weight
        merged[key] = value.to(dtype=target_dtype)

    return merged


def merge_embeds_average(input_dirs: list[Path], weights: list[float]) -> dict[str, torch.Tensor]:
    embed_states = [load_file(directory / EMBEDS_FILE) for directory in input_dirs]
    reference_keys = set(embed_states[0].keys())
    for index, state in enumerate(embed_states[1:], start=2):
        if set(state.keys()) != reference_keys:
            raise ValueError(f"Embedding key mismatch in input {index}")

    merged = {}
    for key in sorted(reference_keys):
        shape = embed_states[0][key].shape
        if any(state[key].shape != shape for state in embed_states[1:]):
            raise ValueError(f"Embedding shape mismatch for {key}")
        value = torch.zeros_like(embed_states[0][key], dtype=torch.float32)
        for state, weight in zip(embed_states, weights, strict=True):
            value += state[key].float() * weight
        merged[key] = value
    return merged


def legacy_token_record(metadata: dict, state: dict[str, torch.Tensor], input_index: int) -> tuple[str, dict[str, str]]:
    token = metadata.get("placeholder_token")
    if not token:
        raise ValueError(f"Input {input_index} does not define placeholder_token")
    keys = {"clip_l": "clip_l", "clip_g": "clip_g"}
    missing = [key for key in keys.values() if key not in state]
    if missing:
        raise ValueError(f"Input {input_index} missing legacy embedding keys: {missing}")
    return token, keys


def embedding_records(
    metadata: dict,
    state: dict[str, torch.Tensor],
    input_index: int,
) -> list[tuple[str, dict[str, str]]]:
    token_embedding_keys = metadata.get("token_embedding_keys")
    if token_embedding_keys:
        tokens = metadata.get("placeholder_tokens") or list(token_embedding_keys)
        records = []
        for token in tokens:
            if token not in token_embedding_keys:
                raise ValueError(f"Input {input_index} missing embedding metadata for token {token}")
            keys = token_embedding_keys[token]
            missing_names = [name for name in ("clip_l", "clip_g") if name not in keys]
            if missing_names:
                raise ValueError(f"Input {input_index} token {token} missing entries: {missing_names}")
            missing_tensors = [key for key in keys.values() if key not in state]
            if missing_tensors:
                raise ValueError(f"Input {input_index} token {token} missing tensors: {missing_tensors}")
            records.append((token, {"clip_l": keys["clip_l"], "clip_g": keys["clip_g"]}))
        return records

    return [legacy_token_record(metadata, state, input_index)]


def merge_embeds_keep(
    input_dirs: list[Path],
    input_metadata: list[dict],
) -> tuple[dict[str, torch.Tensor], list[str], dict[str, dict[str, str]]]:
    merged: dict[str, torch.Tensor] = {}
    placeholder_tokens: list[str] = []
    token_embedding_keys: dict[str, dict[str, str]] = {}

    for input_index, (directory, metadata) in enumerate(zip(input_dirs, input_metadata, strict=True), start=1):
        state = load_file(directory / EMBEDS_FILE)
        for token, source_keys in embedding_records(metadata, state, input_index):
            if token in token_embedding_keys:
                raise ValueError(f"Duplicate placeholder token in inputs: {token}")

            output_keys = {}
            token_index = len(placeholder_tokens)
            for encoder_key in ("clip_l", "clip_g"):
                output_key = f"token_{token_index}.{encoder_key}"
                merged[output_key] = state[source_keys[encoder_key]]
                output_keys[encoder_key] = output_key

            placeholder_tokens.append(token)
            token_embedding_keys[token] = output_keys

    return merged, placeholder_tokens, token_embedding_keys


def read_metadata(input_dirs: list[Path]) -> list[dict]:
    metadata = []
    for directory in input_dirs:
        path = directory / METADATA_FILE
        if path.exists():
            metadata.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            metadata.append({})
    return metadata


def main() -> None:
    args = parse_args()
    input_dirs = [Path(path) for path in args.lora_dir]
    if len(input_dirs) < 2:
        raise ValueError("Provide at least two --lora-dir inputs to merge.")

    for directory in input_dirs:
        if not (directory / LORA_FILE).exists():
            raise FileNotFoundError(directory / LORA_FILE)
        if not (directory / EMBEDS_FILE).exists():
            raise FileNotFoundError(directory / EMBEDS_FILE)

    weights = resolve_weights(len(input_dirs), args.weight, args.normalize_weights)
    states = [load_file(directory / LORA_FILE) for directory in input_dirs]
    down_keys = validate_lora_states(states)

    if args.method == "concat":
        merged_lora = merge_concat(states, down_keys, weights, args.save_dtype)
    else:
        merged_lora = merge_linear(states, weights, args.save_dtype)
    input_metadata = read_metadata(input_dirs)
    if args.embed_merge == "average":
        merged_embeds = merge_embeds_average(input_dirs, weights)
        placeholder_tokens = [args.placeholder_token]
        token_embedding_keys = None
    else:
        merged_embeds, placeholder_tokens, token_embedding_keys = merge_embeds_keep(input_dirs, input_metadata)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged_lora, output_dir / LORA_FILE)
    save_file(merged_embeds, output_dir / EMBEDS_FILE)

    metadata = {
        "placeholder_tokens": placeholder_tokens,
        "embedding_merge": args.embed_merge,
        "merge_method": args.method,
        "weights": weights,
        "input_dirs": [str(directory) for directory in input_dirs],
        "input_metadata": input_metadata,
        "rank_note": "concat merge increases rank by summing input ranks",
    }
    if args.embed_merge == "average":
        metadata["placeholder_token"] = args.placeholder_token
    else:
        metadata["token_embedding_keys"] = token_embedding_keys
    (output_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "merge_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"saved merged LoRA to {output_dir}")
    print(f"method={args.method} weights={weights}")
    print(f"placeholder_tokens={placeholder_tokens}")
    print(f"lora_keys={len(merged_lora)} embed_keys={sorted(merged_embeds)}")


if __name__ == "__main__":
    main()
