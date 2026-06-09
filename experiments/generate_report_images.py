#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from safetensors.torch import load_file


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, bad anatomy, text, watermark, signature"
)


@dataclass
class Task:
    experiment_id: str
    experiment_title: str
    variant_id: str
    prompt_id: str
    prompt: str
    seed: int
    output_path: str
    lora_dir: str | None
    lora_scale: float
    width: int
    height: int
    steps: int
    guidance_scale: float
    metadata: dict[str, Any]


def slugify(text: str, max_len: int = 72) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "prompt"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_token_embedding(tokenizer, text_encoder, placeholder_token: str, embedding: torch.Tensor) -> None:
    if placeholder_token in tokenizer.get_vocab():
        token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    else:
        tokenizer.add_tokens(placeholder_token)
        try:
            text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            text_encoder.resize_token_embeddings(len(tokenizer))
        token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    weight = text_encoder.get_input_embeddings().weight
    weight.data[token_id] = embedding.to(device=weight.device, dtype=weight.dtype)


def token_embedding_specs(metadata: dict[str, Any], embeds: dict[str, torch.Tensor]) -> list[tuple[str, dict[str, str]]]:
    token_embedding_keys = metadata.get("token_embedding_keys")
    if token_embedding_keys:
        tokens = metadata.get("placeholder_tokens") or list(token_embedding_keys)
        specs = []
        for token in tokens:
            if token not in token_embedding_keys:
                raise ValueError(f"Missing embedding metadata for token {token}")
            keys = token_embedding_keys[token]
            missing_names = [name for name in ("clip_l", "clip_g") if name not in keys]
            if missing_names:
                raise ValueError(f"Token {token} missing embedding entries: {missing_names}")
            missing_tensors = [key for key in keys.values() if key not in embeds]
            if missing_tensors:
                raise ValueError(f"Token {token} missing learned embedding tensors: {missing_tensors}")
            specs.append((token, {"clip_l": keys["clip_l"], "clip_g": keys["clip_g"]}))
        return specs

    placeholder_token = metadata["placeholder_token"]
    missing = [key for key in ("clip_l", "clip_g") if key not in embeds]
    if missing:
        raise ValueError(f"Missing legacy learned embedding tensors: {missing}")
    return [(placeholder_token, {"clip_l": "clip_l", "clip_g": "clip_g"})]


def add_learned_tokens(pipe: StableDiffusionXLPipeline, lora_dir: Path) -> list[str]:
    metadata_path = lora_dir / "training_metadata.json"
    embeds_path = lora_dir / "learned_embeds.safetensors"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embeds = load_file(embeds_path)
    specs = token_embedding_specs(metadata, embeds)

    for tokenizer, text_encoder, key in [
        (pipe.tokenizer, pipe.text_encoder, "clip_l"),
        (pipe.tokenizer_2, pipe.text_encoder_2, "clip_g"),
    ]:
        for placeholder_token, embedding_keys in specs:
            set_token_embedding(tokenizer, text_encoder, placeholder_token, embeds[embedding_keys[key]])

    return [token for token, _ in specs]


def parse_experiments(raw: str) -> set[str] | None:
    if raw == "all":
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def make_seeds(seed_base: int, seeds_per_prompt: int) -> list[int]:
    return [seed_base + offset for offset in range(seeds_per_prompt)]


def add_task(
    tasks: list[Task],
    output_dir: Path,
    experiment_id: str,
    experiment_title: str,
    variant_id: str,
    prompt_id: str,
    prompt: str,
    seed: int,
    lora_dir: str | None,
    lora_scale: float,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    safe_name = f"{prompt_id}_seed-{seed}_scale-{str(lora_scale).replace('.', 'p')}_{slugify(variant_id)}.png"
    image_path = output_dir / experiment_id / variant_id / safe_name
    metadata = extra_metadata or {}
    tasks.append(
        Task(
            experiment_id=experiment_id,
            experiment_title=experiment_title,
            variant_id=variant_id,
            prompt_id=prompt_id,
            prompt=prompt,
            seed=seed,
            output_path=str(image_path),
            lora_dir=lora_dir,
            lora_scale=lora_scale,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            metadata=metadata,
        )
    )


def style_lora_dir(config: dict[str, Any], style: str) -> str:
    return config["styles"][style]["lora_dir"]


def style_lora_scale(config: dict[str, Any], style: str, fallback: float) -> float:
    return float(config["styles"][style].get("default_lora_scale", fallback))


def checkpoint_lora_dir(config: dict[str, Any], style: str, checkpoint: str) -> str:
    base_dir = style_lora_dir(config, style)
    if checkpoint == "final":
        return base_dir
    return str(Path(base_dir) / checkpoint)


def build_tasks(config: dict[str, Any], args: argparse.Namespace) -> list[Task]:
    selected = parse_experiments(args.experiments)
    seeds = make_seeds(args.seed_base, args.seeds_per_prompt)
    tasks: list[Task] = []
    output_dir = resolve_path(args.output_dir)
    exps = config["experiments"]

    def enabled(name: str) -> bool:
        return selected is None or name in selected

    if enabled("base_vs_single"):
        exp = exps["base_vs_single"]
        for item in exp["prompts"]:
            style = item["style"]
            for seed in seeds:
                add_task(
                    tasks,
                    output_dir,
                    "base_vs_single",
                    exp["title"],
                    f"{style}_base",
                    item["prompt_id"],
                    item["base_prompt"],
                    seed,
                    None,
                    0.0,
                    args.width,
                    args.height,
                    args.steps,
                    args.guidance_scale,
                    {"style": style, "comparison": "base"},
                )
                add_task(
                    tasks,
                    output_dir,
                    "base_vs_single",
                    exp["title"],
                    f"{style}_lora",
                    item["prompt_id"],
                    item["lora_prompt"],
                    seed,
                    style_lora_dir(config, style),
                    style_lora_scale(config, style, args.default_lora_scale),
                    args.width,
                    args.height,
                    args.steps,
                    args.guidance_scale,
                    {"style": style, "comparison": "single_lora"},
                )

    if enabled("scale_sweep"):
        exp = exps["scale_sweep"]
        for item in exp["prompts"]:
            style = item["style"]
            for scale in exp["scales"]:
                for seed in seeds:
                    add_task(
                        tasks,
                        output_dir,
                        "scale_sweep",
                        exp["title"],
                        f"{style}_scale_{str(scale).replace('.', 'p')}",
                        item["prompt_id"],
                        item["prompt"],
                        seed,
                        style_lora_dir(config, style),
                        float(scale),
                        args.width,
                        args.height,
                        args.steps,
                        args.guidance_scale,
                        {"style": style, "scale": scale},
                    )

    if enabled("checkpoint_dynamics"):
        exp = exps["checkpoint_dynamics"]
        for item in exp["prompts"]:
            style = item["style"]
            for checkpoint in exp["checkpoints"]:
                for seed in seeds:
                    add_task(
                        tasks,
                        output_dir,
                        "checkpoint_dynamics",
                        exp["title"],
                        f"{style}_{checkpoint}",
                        item["prompt_id"],
                        item["prompt"],
                        seed,
                        checkpoint_lora_dir(config, style, checkpoint),
                        style_lora_scale(config, style, args.default_lora_scale),
                        args.width,
                        args.height,
                        args.steps,
                        args.guidance_scale,
                        {"style": style, "checkpoint": checkpoint},
                    )

    if enabled("generalization"):
        exp = exps["generalization"]
        for item in exp["prompts"]:
            style = item["style"]
            for seed in seeds:
                add_task(
                    tasks,
                    output_dir,
                    "generalization",
                    exp["title"],
                    f"{style}_{item['domain']}",
                    item["prompt_id"],
                    item["prompt"],
                    seed,
                    style_lora_dir(config, style),
                    style_lora_scale(config, style, args.default_lora_scale),
                    args.width,
                    args.height,
                    args.steps,
                    args.guidance_scale,
                    {"style": style, "domain": item["domain"]},
                )

    if enabled("multitoken_merge"):
        exp = exps["multitoken_merge"]
        merge_variant = exp["merge_variant"]
        merge_dir = config["merge_variants"][merge_variant]["lora_dir"]
        for item in exp["prompts"]:
            for seed in seeds:
                add_task(
                    tasks,
                    output_dir,
                    "multitoken_merge",
                    exp["title"],
                    item["role"],
                    item["prompt_id"],
                    item["prompt"],
                    seed,
                    merge_dir,
                    args.default_lora_scale,
                    args.width,
                    args.height,
                    args.steps,
                    args.guidance_scale,
                    {"merge_variant": merge_variant, "role": item["role"]},
                )

    if enabled("merge_weights"):
        exp = exps["merge_weights"]
        for merge_variant in exp["merge_variants"]:
            merge_info = config["merge_variants"][merge_variant]
            for item in exp["prompts"]:
                role = merge_info.get("role")
                if role is not None and item["role"] != role:
                    continue
                pair = merge_info.get("pair")
                if pair is not None and not item["role"].endswith(f"_{pair}"):
                    continue
                for seed in seeds:
                    add_task(
                        tasks,
                        output_dir,
                        "merge_weights",
                        exp["title"],
                        f"{merge_variant}_{item['role']}",
                        item["prompt_id"],
                        item["prompt"],
                        seed,
                        merge_info["lora_dir"],
                        args.default_lora_scale,
                        args.width,
                        args.height,
                        args.steps,
                        args.guidance_scale,
                        {
                            "merge_variant": merge_variant,
                            "merge_label": merge_info["label"],
                            "pair": merge_info.get("pair"),
                            "main_style": merge_info.get("main_style"),
                            "other_style": merge_info.get("other_style"),
                            "weight_role": merge_info.get("weight_role", merge_variant),
                            "weights": merge_info["weights"],
                            "role": item["role"],
                        },
                    )

    if args.max_images > 0:
        tasks = tasks[: args.max_images]
    return tasks


def validate_lora_dirs(tasks: list[Task]) -> tuple[list[Task], list[dict[str, Any]]]:
    valid_tasks = []
    missing = []
    for task in tasks:
        if task.lora_dir is None:
            valid_tasks.append(task)
            continue
        lora_dir = resolve_path(task.lora_dir)
        required = [lora_dir / "pytorch_lora_weights.safetensors", lora_dir / "learned_embeds.safetensors"]
        missing_files = [str(path) for path in required if not path.exists()]
        if missing_files:
            missing.append({**asdict(task), "missing_files": missing_files})
            continue
        valid_tasks.append(task)
    return valid_tasks, missing


def load_pipeline(model_dir: Path, device: str, dtype: torch.dtype, cpu_offload: bool) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def generate_group(
    tasks: list[Task],
    model_dir: Path,
    lora_dir: str | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = load_pipeline(model_dir, device, dtype, args.cpu_offload)
    loaded_tokens: list[str] = []

    if lora_dir is not None:
        lora_path = resolve_path(lora_dir)
        loaded_tokens = add_learned_tokens(pipe, lora_path)
        pipe.load_lora_weights(str(lora_path), adapter_name="style")

    records = []
    for task in tasks:
        image_path = Path(task.output_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and image_path.exists():
            records.append({**asdict(task), "status": "skipped_existing", "loaded_tokens": loaded_tokens})
            print(f"skip existing {image_path}")
            continue

        if lora_dir is not None:
            pipe.set_adapters(["style"], adapter_weights=[task.lora_scale])

        generator = torch.Generator(device=device).manual_seed(task.seed)
        image = pipe(
            prompt=task.prompt,
            negative_prompt=args.negative_prompt,
            width=task.width,
            height=task.height,
            num_inference_steps=task.steps,
            guidance_scale=task.guidance_scale,
            generator=generator,
        ).images[0]
        image.save(image_path)
        records.append({**asdict(task), "status": "generated", "loaded_tokens": loaded_tokens})
        print(f"saved {image_path}")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all report experiment images.")
    parser.add_argument("--config", default="experiments/report_prompts.json")
    parser.add_argument("--model-dir", default="models/sdxl-base-1.0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--seeds-per-prompt", type=int, default=2)
    parser.add_argument("--experiments", default="all", help="all or comma-separated experiment ids")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--default-lora-scale", type=float, default=0.65)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    tasks = build_tasks(config, args)
    if args.dry_run:
        valid_tasks = tasks
        missing_tasks = []
    else:
        valid_tasks, missing_tasks = validate_lora_dirs(tasks)

    plan = {
        "config": str(config_path),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "seed_base": args.seed_base,
        "seeds": make_seeds(args.seed_base, args.seeds_per_prompt),
        "experiments": args.experiments,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "default_lora_scale": args.default_lora_scale,
        "num_tasks": len(tasks),
        "num_valid_tasks": len(valid_tasks),
        "num_missing_tasks": len(missing_tasks),
        "tasks": [asdict(task) for task in valid_tasks],
        "missing_tasks": missing_tasks,
    }
    write_json(output_dir / "run_plan.json", plan)
    if missing_tasks:
        write_json(output_dir / "missing_tasks.json", missing_tasks)
        print(f"warning: {len(missing_tasks)} tasks skipped because LoRA files are missing")

    if args.dry_run:
        print(json.dumps({k: plan[k] for k in plan if k != "tasks"}, indent=2, ensure_ascii=False))
        print(f"dry run tasks: {len(valid_tasks)}")
        return

    records: list[dict[str, Any]] = []
    grouped: dict[str | None, list[Task]] = defaultdict(list)
    for task in valid_tasks:
        grouped[task.lora_dir].append(task)

    for lora_dir, group in grouped.items():
        label = "base_sdxl" if lora_dir is None else lora_dir
        print(f"generate group: {label} ({len(group)} images)")
        records.extend(generate_group(group, model_dir, lora_dir, args))

    metadata = {
        "run_plan": {key: value for key, value in plan.items() if key != "tasks"},
        "images": records,
    }
    write_json(output_dir / "all_metadata.json", metadata)
    print(f"saved {output_dir / 'all_metadata.json'}")


if __name__ == "__main__":
    main()
