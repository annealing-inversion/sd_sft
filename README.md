# SDXL Anime Portrait LoRA Project

这个仓库现在服务于课程作业的 SDXL 动漫人物头像风格微调实验。旧的 Stable Diffusion v1.5 水彩风景实验已经归档到 `archive/watercolor_legacy/`，日常不要再沿用那套 prompt、数据结构或脚本。

## 当前目标

我们用 SDXL base 训练多个小数据集风格 LoRA，每个风格单独训练，然后按需要做 LoRA merge。

当前主要数据集：

| 数据集 | 目标 | 默认训练 token | 当前本地数量 |
|---|---|---|---|
| `data/Ghibli` | 吉卜力感人物与场景头像 | `<ghibli_headshot>` | 24 张 |
| `data/persona_5` | Persona 5 红黑图形化人物风格 | `<persona_5_headshot>` | 24 张 |
| `data/EVA_rei` | Ayanami Rei 人物图像 | `<eva_rei_headshot>` | 23 张 |

数据集、模型权重和输出图片默认不进 git。clone 仓库后需要从小组共享位置复制数据集到上表路径，或者让 Codex 在当前机器上检查已有数据。

## 快速开始

```bash
bash scripts/setup_conda_env.sh
bash scripts/download_sdxl_base.sh
```

这两个脚本默认使用：

```text
.conda/sdxl-lora
models/sdxl-base-1.0
HF_ENDPOINT=https://hf-mirror.com
```

如果在远程机器或 NAS 上跑，可以显式指定路径：

```bash
CONDA_ENV_PREFIX=/mnt/nas-new/valencia/sdxl-style-lora/.conda/sdxl-lora \
  bash scripts/setup_conda_env.sh

MODEL_DIR=/mnt/nas-new/valencia/sdxl-style-lora/models/sdxl-base-1.0 \
  bash scripts/download_sdxl_base.sh
```

## 训练一个风格

默认训练配置面向 3 张以上 24GB GPU，例如 4090。每张卡 batch size 1，梯度累积 4，1024 分辨率，先做 500 step PTI，再训到总共 2000 step LoRA。

吉卜力：

```bash
STYLE=ghibli \
DATA_DIR=data/Ghibli \
OUTPUT_DIR=outputs/sdxl_lora/ghibli \
PLACEHOLDER_TOKEN='<ghibli_headshot>' \
NUM_PROCESSES=3 \
MAIN_PROCESS_PORT=29501 \
  bash training_scripts/run_sdxl_pti_lora_single.sh
```

Persona 5：

```bash
STYLE=persona_5 \
DATA_DIR=data/persona_5 \
OUTPUT_DIR=outputs/sdxl_lora/persona_5 \
PLACEHOLDER_TOKEN='<persona_5_headshot>' \
NUM_PROCESSES=3 \
MAIN_PROCESS_PORT=29502 \
  bash training_scripts/run_sdxl_pti_lora_single.sh
```

EVA Rei：

```bash
STYLE=eva_rei \
DATA_DIR=data/EVA_rei \
OUTPUT_DIR=outputs/sdxl_lora/eva_rei \
PLACEHOLDER_TOKEN='<eva_rei_headshot>' \
NUM_PROCESSES=3 \
MAIN_PROCESS_PORT=29503 \
  bash training_scripts/run_sdxl_pti_lora_single.sh
```

本地只想验证代码链路时：

```bash
bash training_scripts/run_sdxl_pti_lora_smoke.sh
```

smoke 输出不能代表质量，只用来确认环境、模型加载和保存逻辑没断。

## 生成图片

```bash
.conda/sdxl-lora/bin/python scripts/generate_sdxl_lora.py \
  --model-dir models/sdxl-base-1.0 \
  --lora-dir outputs/sdxl_lora/eva_rei \
  --output-dir outputs/eval_eva_rei \
  --prompt "<eva_rei_headshot>, close-up portrait of Ayanami Rei, blue hair, red eyes, clean classic anime style" \
  --seeds 501 502 \
  --steps 30 \
  --lora-scale 0.65
```

建议报告图使用同一组 prompt 和 seed 比较 base、单风格 LoRA、不同 step checkpoint、以及 merge 结果。

## LoRA Merge

合并多个 LoRA 权重：

```bash
.conda/sdxl-lora/bin/python scripts/merge_sdxl_loras.py \
  --lora-dir outputs/sdxl_lora/ghibli \
  --lora-dir outputs/sdxl_lora/persona_5 \
  --lora-dir outputs/sdxl_lora/eva_rei \
  --output-dir outputs/sdxl_lora/merged_multitoken_ghibli_persona5_eva_rei_equal \
  --method concat \
  --embed-merge keep \
  --save-dtype float16
```

`--method concat` 会拼接 LoRA rank，三个 rank-16 LoRA 合并后变成 rank 48。`--embed-merge keep` 会保留三个原触发词，因此同一个 merged adapter 可以分别用：

```text
<ghibli_headshot>
<persona_5_headshot>
<eva_rei_headshot>
```

如果改用默认 `--embed-merge average`，三个 learned embedding 会平均成一个新 token，控制性更差，但更适合展示“混合风格”。

## 给 Codex 的常用任务

可以直接这样让 Codex 接手：

```text
请检查 data/EVA_rei 的图片和 txt caption 是否一一对应，找出胡言乱语标注，但不要删除图片。
```

```text
请在 fstqwq 的 ~/valencia/sdxl-style-lora 下训练 persona_5，先检查 nvidia-smi 和已有输出，不要覆盖正在跑的任务。
```

```text
请用 outputs/sdxl_lora/eva_rei 生成 6 个 prompt、2 个 seed 的评估图，并拉回本地做 contact sheet。
```

```text
请把 ghibli、persona_5、eva_rei 做 multi-token concat merge，并验证三个 token 都能出图。
```

更多细节见：

- [docs/datasets.md](docs/datasets.md)
- [docs/experiment_playbook.md](docs/experiment_playbook.md)
- [docs/course_assignment.md](docs/course_assignment.md)
- [training_scripts/README.md](training_scripts/README.md)
