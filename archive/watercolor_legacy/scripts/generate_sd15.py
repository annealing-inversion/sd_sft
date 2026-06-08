import argparse
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_PROMPTS = [
    "watercolor style, a quiet village beside a river",
    "watercolor style, a small boat on a misty lake",
    "watercolor style, a flower shop on a rainy street",
]
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, bad anatomy, text, watermark, signature"
)


def slugify(text: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "prompt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate test images with Stable Diffusion v1.5.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default="outputs/sd15_base")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    metadata = {
        "model_id": args.model_id,
        "device": device,
        "dtype": str(dtype),
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

            record = {
                "file": str(image_path),
                "prompt": prompt,
                "seed": seed,
            }
            metadata["images"].append(record)
            print(f"saved {image_path}")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"saved {metadata_path}")


if __name__ == "__main__":
    main()
