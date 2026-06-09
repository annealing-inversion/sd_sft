#!/usr/bin/env python
"""Two-stage SDXL Textual-Inversion + LoRA training.

Stage 1 trains only a new placeholder token in both SDXL text encoders.
Stage 2 freezes that token and trains LoRA adapters, optionally including
text-encoder LoRA adapters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageOps
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class Batch:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    input_ids_2: torch.Tensor
    captions: list[str]


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        tokenizer_2,
        placeholder_token: str,
        default_caption: str,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
        caption_extension: str,
    ) -> None:
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.placeholder_token = placeholder_token
        self.default_caption = default_caption
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.caption_extension = caption_extension

        self.images = sorted(
            p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _caption_for(self, image_path: Path) -> str:
        caption_path = image_path.with_suffix(self.caption_extension)
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
        else:
            caption = self.default_caption
        if self.placeholder_token not in caption:
            caption = f"{self.placeholder_token}, {caption}"
        return caption

    def _preprocess_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")

        if self.center_crop:
            image = ImageOps.fit(
                image,
                (self.resolution, self.resolution),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        else:
            width, height = image.size
            scale = self.resolution / min(width, height)
            new_size = (round(width * scale), round(height * scale))
            image = image.resize(new_size, resample=Image.Resampling.LANCZOS)
            left = random.randint(0, max(0, image.width - self.resolution))
            top = random.randint(0, max(0, image.height - self.resolution))
            image = image.crop((left, top, left + self.resolution, top + self.resolution))

        if self.random_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)

        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]
        caption = self._caption_for(image_path)
        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_2 = self.tokenizer_2(
            caption,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": self._preprocess_image(image_path),
            "input_ids": text_inputs.input_ids[0],
            "input_ids_2": text_inputs_2.input_ids[0],
            "caption": caption,
        }


def collate_fn(examples: list[dict]) -> Batch:
    return Batch(
        pixel_values=torch.stack([example["pixel_values"] for example in examples]).contiguous(),
        input_ids=torch.stack([example["input_ids"] for example in examples]),
        input_ids_2=torch.stack([example["input_ids_2"] for example in examples]),
        captions=[example["caption"] for example in examples],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SDXL with TI warmup followed by LoRA.")
    parser.add_argument("--pretrained-model-name-or-path", default="models/sdxl-base-1.0")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--placeholder-token", required=True)
    parser.add_argument("--initializer-token", default="style")
    parser.add_argument("--default-caption", default="anime portrait, close-up face, clean line art")
    parser.add_argument("--caption-extension", default=".txt")

    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--random-flip", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=2000)
    parser.add_argument("--ti-steps", type=int, default=500)
    parser.add_argument("--checkpointing-steps", type=int, default=200)

    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--text-encoder-lora-lr", type=float, default=5e-6)
    parser.add_argument("--ti-lr", type=float, default=5e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)

    parser.add_argument("--train-text-encoder-lora", action="store_true")
    parser.add_argument("--train-ti-after-warmup", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--report-to", default=None)
    return parser.parse_args()


def add_placeholder_token(tokenizer, text_encoder, placeholder_token: str, initializer_token: str) -> int:
    num_added = tokenizer.add_tokens(placeholder_token)
    if num_added == 0:
        raise ValueError(
            f"Token {placeholder_token!r} already exists in tokenizer. Use a fresh placeholder token."
        )

    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Initializer token {initializer_token!r} produced no token ids.")
    initializer_token_id = token_ids[0]

    try:
        text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    except TypeError:
        text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    embeddings = text_encoder.get_input_embeddings().weight
    embeddings.data[token_id] = embeddings.data[initializer_token_id].clone()
    return token_id


def set_lora_requires_grad(model: torch.nn.Module, enabled: bool) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(enabled)


def unwrap_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def set_token_embedding_requires_grad(text_encoder, enabled: bool) -> None:
    unwrap_module(text_encoder).get_input_embeddings().weight.requires_grad_(enabled)


def trainable_lora_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for name, param in model.named_parameters() if "lora_" in name]


def restore_non_placeholder_embeddings(text_encoder, original_embeddings: torch.Tensor, token_id: int) -> None:
    embeddings = unwrap_module(text_encoder).get_input_embeddings().weight
    mask = torch.ones(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    mask[token_id] = False
    embeddings.data[mask] = original_embeddings.to(device=embeddings.device, dtype=embeddings.dtype)[mask]


def encode_prompt_for_sdxl(
    text_encoder,
    text_encoder_2,
    input_ids: torch.Tensor,
    input_ids_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds_1 = text_encoder(input_ids, output_hidden_states=True)
    prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]

    prompt_embeds_2 = text_encoder_2(input_ids_2, output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds_2[0]
    prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

    prompt_embeds = torch.concat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def make_add_time_ids(batch_size: int, resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # original_size, crop_top_left, target_size
    time_ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        device=device,
        dtype=dtype,
    )
    return time_ids.unsqueeze(0).repeat(batch_size, 1)


def save_training_artifacts(
    output_dir: Path,
    accelerator: Accelerator,
    unet,
    text_encoder,
    text_encoder_2,
    token_id: int,
    token_id_2: int,
    args: argparse.Namespace,
) -> None:
    if not accelerator.is_main_process:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
    unwrapped_text_encoder_2 = accelerator.unwrap_model(text_encoder_2)

    unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

    text_encoder_lora_layers = None
    text_encoder_2_lora_layers = None
    if args.train_text_encoder_lora:
        text_encoder_lora_layers = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_text_encoder)
        )
        text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_text_encoder_2)
        )

    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=text_encoder_lora_layers,
        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        weight_name="pytorch_lora_weights.safetensors",
        safe_serialization=True,
    )

    learned_embeds = {
        "clip_l": unwrapped_text_encoder.get_input_embeddings().weight[token_id].detach().cpu(),
        "clip_g": unwrapped_text_encoder_2.get_input_embeddings().weight[token_id_2].detach().cpu(),
    }
    save_file(learned_embeds, output_dir / "learned_embeds.safetensors")

    metadata = {
        "placeholder_token": args.placeholder_token,
        "initializer_token": args.initializer_token,
        "token_id": token_id,
        "token_id_2": token_id_2,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "resolution": args.resolution,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "ti_steps": args.ti_steps,
        "max_train_steps": args.max_train_steps,
        "train_text_encoder_lora": args.train_text_encoder_lora,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with=args.report_to,
        project_dir=str(output_dir),
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model_path = args.pretrained_model_name_or_path
    if accelerator.is_main_process:
        print("loading tokenizers and scheduler", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

    if accelerator.is_main_process:
        print("loading text_encoder", flush=True)
    text_encoder = CLIPTextModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        variant=args.variant,
        local_files_only=args.local_files_only,
    )
    if accelerator.is_main_process:
        print("loading text_encoder_2", flush=True)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
        variant=args.variant,
        local_files_only=args.local_files_only,
    )
    if accelerator.is_main_process:
        print("loading vae", flush=True)
    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        variant=args.variant,
        local_files_only=args.local_files_only,
    )
    if accelerator.is_main_process:
        print("loading unet", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        variant=args.variant,
        local_files_only=args.local_files_only,
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    if accelerator.is_main_process:
        print("adding placeholder tokens", flush=True)
    token_id = add_placeholder_token(tokenizer, text_encoder, args.placeholder_token, args.initializer_token)
    token_id_2 = add_placeholder_token(tokenizer_2, text_encoder_2, args.placeholder_token, args.initializer_token)

    if accelerator.is_main_process:
        print("adding LoRA adapters", flush=True)
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        lora_dropout=args.lora_dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    if args.train_text_encoder_lora:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)
        text_encoder_2.add_adapter(text_lora_config)

    set_token_embedding_requires_grad(text_encoder, True)
    set_token_embedding_requires_grad(text_encoder_2, True)

    if weight_dtype != torch.float32:
        if accelerator.is_main_process:
            print("casting trainable params to fp32", flush=True)
        models_to_cast = [unet, text_encoder, text_encoder_2]
        cast_training_params(models_to_cast, dtype=torch.float32)

    if accelerator.is_main_process:
        print("building dataset", flush=True)
    original_embeddings = text_encoder.get_input_embeddings().weight.detach().clone()
    original_embeddings_2 = text_encoder_2.get_input_embeddings().weight.detach().clone()

    dataset = ImageCaptionDataset(
        data_dir=Path(args.data_dir),
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        placeholder_token=args.placeholder_token,
        default_caption=args.default_caption,
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        caption_extension=args.caption_extension,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    if args.gradient_checkpointing:
        if accelerator.is_main_process:
            print("enabling gradient checkpointing", flush=True)
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()

    set_lora_requires_grad(unet, False)
    set_lora_requires_grad(text_encoder, False)
    set_lora_requires_grad(text_encoder_2, False)

    optimizer_groups = [
        {
            "params": [
                text_encoder.get_input_embeddings().weight,
                text_encoder_2.get_input_embeddings().weight,
            ],
            "lr": args.ti_lr,
        },
        {"params": trainable_lora_parameters(unet), "lr": args.learning_rate},
    ]
    if args.train_text_encoder_lora:
        optimizer_groups.append(
            {
                "params": trainable_lora_parameters(text_encoder)
                + trainable_lora_parameters(text_encoder_2),
                "lr": args.text_encoder_lora_lr,
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if accelerator.is_main_process:
        print("preparing accelerator objects", flush=True)
    unet, text_encoder, text_encoder_2, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, text_encoder_2, optimizer, dataloader, lr_scheduler
    )
    if accelerator.is_main_process:
        print("moving vae to device", flush=True)
    vae.to(accelerator.device, dtype=torch.float32)
    vae.eval()

    global_step = 0
    epoch = 0
    running_loss = 0.0
    batches_per_epoch = max(1, math.ceil(len(dataset) / args.train_batch_size))

    if accelerator.is_main_process:
        config_path = output_dir / "training_config.json"
        config_path.write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"dataset={len(dataset)} images, batches_per_epoch={batches_per_epoch}, "
            f"max_steps={args.max_train_steps}, ti_steps={args.ti_steps}"
        )

    while global_step < args.max_train_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            stage = "ti" if global_step < args.ti_steps else "lora"
            train_ti_now = stage == "ti" or (stage == "lora" and args.train_ti_after_warmup)
            train_lora_now = stage == "lora"

            set_token_embedding_requires_grad(text_encoder, train_ti_now)
            set_token_embedding_requires_grad(text_encoder_2, train_ti_now)
            set_lora_requires_grad(unet, train_lora_now)
            if args.train_text_encoder_lora:
                set_lora_requires_grad(text_encoder, train_lora_now)
                set_lora_requires_grad(text_encoder_2, train_lora_now)

            with accelerator.accumulate(unet):
                pixel_values = batch.pixel_values.to(device=accelerator.device, dtype=torch.float32)
                input_ids = batch.input_ids.to(device=accelerator.device)
                input_ids_2 = batch.input_ids_2.to(device=accelerator.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encode_needs_grad = train_ti_now or (train_lora_now and args.train_text_encoder_lora)
                with torch.set_grad_enabled(encode_needs_grad):
                    prompt_embeds, pooled_prompt_embeds = encode_prompt_for_sdxl(
                        text_encoder, text_encoder_2, input_ids, input_ids_2
                    )
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
                add_time_ids = make_add_time_ids(
                    batch_size=latents.shape[0],
                    resolution=args.resolution,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids,
                    },
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    trainable_params = [
                        param for group in optimizer.param_groups for param in group["params"] if param.requires_grad
                    ]
                    if trainable_params:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if train_ti_now:
                    restore_non_placeholder_embeddings(text_encoder, original_embeddings, token_id)
                    restore_non_placeholder_embeddings(text_encoder_2, original_embeddings_2, token_id_2)

            if accelerator.sync_gradients:
                global_step += 1
                running_loss += loss.detach().item()

                if accelerator.is_main_process and (global_step == 1 or global_step % 10 == 0):
                    avg_loss = running_loss / (10 if global_step % 10 == 0 else 1)
                    print(f"step={global_step} stage={stage} loss={avg_loss:.6f}")
                    running_loss = 0.0

                if (
                    args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                    and global_step < args.max_train_steps
                ):
                    save_training_artifacts(
                        output_dir / f"checkpoint-{global_step}",
                        accelerator,
                        unet,
                        text_encoder,
                        text_encoder_2,
                        token_id,
                        token_id_2,
                        args,
                    )
                    accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    save_training_artifacts(
        output_dir,
        accelerator,
        unet,
        text_encoder,
        text_encoder_2,
        token_id,
        token_id_2,
        args,
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"saved final LoRA and learned embeddings to {output_dir}")


if __name__ == "__main__":
    main()
