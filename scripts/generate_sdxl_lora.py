import argparse
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from safetensors.torch import load_file


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, bad anatomy, text, watermark, signature"
)


def slugify(text: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "prompt"


def set_token_embedding(
    tokenizer,
    text_encoder,
    placeholder_token: str,
    embedding: torch.Tensor,
) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with an SDXL PTI+LoRA checkpoint.")
    parser.add_argument("--model-dir", default="models/sdxl-base-1.0")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/sdxl_lora_eval")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lora_dir = Path(args.lora_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    placeholder_tokens = add_learned_tokens(pipe, lora_dir)
    pipe.load_lora_weights(lora_dir, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[args.lora_scale])

    if args.cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    trigger_prompt = ", ".join(placeholder_tokens)
    prompts = args.prompts or [
        f"{trigger_prompt}, anime portrait, close-up face, clean line art",
    ]

    metadata = {
        "model_dir": args.model_dir,
        "lora_dir": str(lora_dir),
        "placeholder_tokens": placeholder_tokens,
        "lora_scale": args.lora_scale,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "images": [],
    }

    for prompt_index, prompt in enumerate(prompts, start=1):
        for seed in args.seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            filename = f"{prompt_index:02d}_seed-{seed}_{slugify(prompt)}.png"
            image_path = output_dir / filename
            image.save(image_path)
            metadata["images"].append({"file": str(image_path), "prompt": prompt, "seed": seed})
            print(f"saved {image_path}")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {metadata_path}")


if __name__ == "__main__":
    main()
