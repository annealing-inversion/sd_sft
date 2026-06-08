# Codex Project Notes

This repository is now an SDXL PTI+LoRA anime portrait style project. Treat `archive/watercolor_legacy/` as read-only historical material unless the user explicitly asks about the old watercolor SD1.5 experiment.

## Active Scope

- Current model family: SDXL base 1.0.
- Current training route: Textual Inversion/PTI warmup for a placeholder token, then LoRA training.
- Active scripts:
  - `scripts/setup_conda_env.sh`
  - `scripts/download_sdxl_base.sh`
  - `training_scripts/run_sdxl_pti_lora_single.sh`
  - `training_scripts/train_sdxl_pti_lora.py`
  - `scripts/generate_sdxl_lora.py`
  - `scripts/merge_sdxl_loras.py`
- Default mirror: `HF_ENDPOINT=https://hf-mirror.com`.

## Active Datasets

Use these names and tokens unless the user says otherwise:

| Dataset | Token | Output |
|---|---|---|
| `data/Ghibli` | `<ghibli_headshot>` | `outputs/sdxl_lora/ghibli` |
| `data/persona_5` | `<persona_5_headshot>` | `outputs/sdxl_lora/persona_5` |
| `data/EVA_rei` | `<eva_rei_headshot>` | `outputs/sdxl_lora/eva_rei` |

Each image should have a same-stem `.txt` caption. `metadata.csv` is useful for audits but is not consumed by the training script.

## Operating Rules

- Do not commit model weights, generated images, raw datasets, zip bundles, or local cache/env folders unless explicitly requested.
- Before remote training, check `nvidia-smi`, existing logs, and existing output directories.
- On multi-GPU launches, set a distinct `MAIN_PROCESS_PORT` if another run may be active.
- Preserve user edits in data captions. If an image is blurry or off-style, report it rather than deleting it.
- For merge experiments, prefer `--method concat --embed-merge keep` when the user wants separate trigger-token control.
- For report comparisons, keep prompt, seed, resolution, scheduler, guidance scale, and LoRA scale explicit in the saved metadata.

## Remote Convention

The current remote working copy used in prior runs is:

```text
fstqwq:~/valencia/sdxl-style-lora
```

The remote conda env is usually:

```text
~/valencia/sdxl-style-lora/.conda/sdxl-lora
```

Verify these paths live before relying on them.

