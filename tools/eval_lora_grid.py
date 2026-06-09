#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file


NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, bad anatomy, text, watermark, signature"
SCALES = ["0.55", "0.65", "0.75"]
SEED_BASE = int(os.environ.get("SEED_BASE", "501"))
SEEDS = [SEED_BASE, SEED_BASE + 1]


def slugify(text: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "prompt"


def humanize(style: str) -> str:
    return style.replace("_", " ").replace("-", " ")


def default_prompts(style: str, token: str) -> list[str]:
    label = humanize(style)
    return [
        f"{token}, {label}, anime character portrait, expressive eyes, clean line art, detailed face, soft background",
        f"{token}, {label}, full body anime character standing outdoors, dynamic pose, detailed outfit, bright daylight",
        f"{token}, {label}, close-up character holding a magical object, sparkling light, dramatic anime composition",
        f"{token}, {label}, action scene, flowing hair and clothing, vivid cel shaded anime style, cinematic background",
        f"{token}, {label}, cozy indoor scene, character reading near a window, warm pastel colors, detailed room",
    ]


def base_prompt(prompt: str, token: str) -> str:
    return prompt.replace(f"{token}, ", "").replace(token, "").strip(", ")


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


def token_embedding_specs(metadata: dict, embeds: dict[str, torch.Tensor]) -> list[tuple[str, dict[str, str]]]:
    token_embedding_keys = metadata.get("token_embedding_keys")
    if token_embedding_keys:
        tokens = metadata.get("placeholder_tokens") or list(token_embedding_keys)
        return [
            (token, {"clip_l": token_embedding_keys[token]["clip_l"], "clip_g": token_embedding_keys[token]["clip_g"]})
            for token in tokens
        ]
    return [(metadata["placeholder_token"], {"clip_l": "clip_l", "clip_g": "clip_g"})]


def add_learned_tokens(pipe: StableDiffusionXLPipeline, lora_dir: Path) -> list[str]:
    metadata = json.loads((lora_dir / "training_metadata.json").read_text(encoding="utf-8"))
    embeds = load_file(lora_dir / "learned_embeds.safetensors")
    specs = token_embedding_specs(metadata, embeds)
    for tokenizer, text_encoder, key in [
        (pipe.tokenizer, pipe.text_encoder, "clip_l"),
        (pipe.tokenizer_2, pipe.text_encoder_2, "clip_g"),
    ]:
        for placeholder_token, embedding_keys in specs:
            set_token_embedding(tokenizer, text_encoder, placeholder_token, embeds[embedding_keys[key]])
    return [token for token, _ in specs]


def load_pipe(model_dir: str) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    return pipe


def generate_set(pipe, prompts: list[str], out_dir: Path, scale: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "lora_scale": scale,
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "guidance_scale": 7.5,
        "negative_prompt": NEGATIVE_PROMPT,
        "images": [],
    }
    for prompt_index, prompt in enumerate(prompts, start=1):
        for seed in SEEDS:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=1024,
                height=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
            image_path = out_dir / f"{prompt_index:02d}_seed-{seed}_{slugify(prompt)}.png"
            image.save(image_path)
            metadata["images"].append({"file": str(image_path), "prompt": prompt, "seed": seed})
            print(f"saved {image_path}", flush=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_grid(eval_root: Path) -> Path:
    columns = [("Base", None, seed) for seed in SEEDS]
    for scale in SCALES:
        for seed in SEEDS:
            columns.append((f"LoRA {scale}", scale, seed))
    prompt_titles = ["P1 portrait", "P2 full body", "P3 object close-up", "P4 action", "P5 indoor"]

    metas = {"base": json.loads((eval_root / "base" / "metadata.json").read_text(encoding="utf-8"))}
    for scale in SCALES:
        metas[scale] = json.loads((eval_root / f"scale_{scale}" / "metadata.json").read_text(encoding="utf-8"))
    lookup = {}
    for key, meta in metas.items():
        for item in meta["images"]:
            file = Path(item["file"])
            lookup[(key, int(file.name.split("_", 1)[0]), int(item["seed"]))] = file

    thumb = 256
    row_label_w = 160
    header_h = 112
    caption_h = 30
    gap = 9
    width = row_label_w + len(columns) * thumb + (len(columns) + 1) * gap
    height = header_h + 5 * (thumb + caption_h) + 6 * gap
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font_header = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font_title = font_header = font_small = ImageFont.load_default()

    style = eval_root.name.removeprefix("eval_").rsplit("_", 2)[0]
    draw.text((12, 12), f"{style}: Base SDXL vs LoRA scale comparison", fill=(20, 20, 20), font=font_title)
    draw.text((12, 43), "Rows are prompts. Columns compare Base SDXL and LoRA scales 0.55 / 0.65 / 0.75 with seeds 501 and 502.", fill=(80, 80, 80), font=font_small)
    draw.text((12, 61), "All thumbnails are generated at 1024x1024 and downsampled here for scanning.", fill=(80, 80, 80), font=font_small)
    for col, (label, _scale, seed) in enumerate(columns):
        x = row_label_w + gap + col * (thumb + gap)
        draw.rectangle((x, 76, x + thumb, 106), fill=(244, 246, 248), outline=(210, 210, 210))
        draw.text((x + 8, 82), f"{label}  seed {seed}", fill=(25, 25, 25), font=font_header)

    for row in range(1, 6):
        y = header_h + gap + (row - 1) * (thumb + caption_h + gap)
        draw.text((12, y + 10), prompt_titles[row - 1], fill=(20, 20, 20), font=font_header)
        for col, (_, scale, seed) in enumerate(columns):
            key = scale if scale else "base"
            path = lookup[(key, row, seed)]
            x = row_label_w + gap + col * (thumb + gap)
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (thumb, thumb), (245, 245, 245))
                canvas.paste(img, ((thumb - img.width) // 2, (thumb - img.height) // 2))
            sheet.paste(canvas, (x, y))
            draw.rectangle((x, y, x + thumb, y + thumb), outline=(205, 205, 205), width=1)
            draw.text((x + 5, y + thumb + 7), "base" if not scale else f"scale {scale}", fill=(80, 80, 80), font=font_small)

    out = eval_root / "base_lora_scale_grid.png"
    sheet.save(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", required=True)
    parser.add_argument("--model-dir", default="models/sdxl-base-1.0")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    token = f"<{args.style}_style>"
    prompts = default_prompts(args.style, token)
    eval_root = Path(args.output_root) / f"eval_{args.style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_root.mkdir(parents=True, exist_ok=False)
    (eval_root / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")

    base_pipe = load_pipe(args.model_dir)
    generate_set(base_pipe, [base_prompt(p, token) for p in prompts], eval_root / "base", None)
    del base_pipe
    torch.cuda.empty_cache()

    for scale in SCALES:
        pipe = load_pipe(args.model_dir)
        add_learned_tokens(pipe, Path(args.lora_dir))
        pipe.load_lora_weights(args.lora_dir, adapter_name="style")
        pipe.set_adapters(["style"], adapter_weights=[float(scale)])
        generate_set(pipe, prompts, eval_root / f"scale_{scale}", scale)
        del pipe
        torch.cuda.empty_cache()

    grid = make_grid(eval_root)
    print(f"saved {grid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
