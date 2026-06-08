# 实验执行手册

这份文档给“不想细读代码、主要让 Codex 跑实验”的组员使用。先看命令块，再按需让 Codex 修改参数。

## 环境

```bash
bash scripts/setup_conda_env.sh
bash scripts/download_sdxl_base.sh
```

远程或 NAS 路径：

```bash
CONDA_ENV_PREFIX=/mnt/nas-new/valencia/sdxl-style-lora/.conda/sdxl-lora \
  bash scripts/setup_conda_env.sh

MODEL_DIR=/mnt/nas-new/valencia/sdxl-style-lora/models/sdxl-base-1.0 \
  bash scripts/download_sdxl_base.sh
```

## 训练前检查

```bash
nvidia-smi
find data/Ghibli data/persona_5 data/EVA_rei -maxdepth 1 -type f | sort | head
find outputs/sdxl_lora -maxdepth 2 -name training_metadata.json -print
```

如果远程已经有同名 `outputs/sdxl_lora/<style>`，先确认是不是要覆盖。正常情况下不要覆盖 final 输出，换一个 output dir 或先备份。

## 正式训练命令

三卡 4090 默认配置：

```bash
STYLE=ghibli \
DATA_DIR=data/Ghibli \
OUTPUT_DIR=outputs/sdxl_lora/ghibli \
PLACEHOLDER_TOKEN='<ghibli_headshot>' \
NUM_PROCESSES=3 \
MAIN_PROCESS_PORT=29501 \
  bash training_scripts/run_sdxl_pti_lora_single.sh
```

```bash
STYLE=persona_5 \
DATA_DIR=data/persona_5 \
OUTPUT_DIR=outputs/sdxl_lora/persona_5 \
PLACEHOLDER_TOKEN='<persona_5_headshot>' \
NUM_PROCESSES=3 \
MAIN_PROCESS_PORT=29502 \
  bash training_scripts/run_sdxl_pti_lora_single.sh
```

```bash
STYLE=eva_rei \
DATA_DIR=data/EVA_rei \
OUTPUT_DIR=outputs/sdxl_lora/eva_rei \
PLACEHOLDER_TOKEN='<eva_rei_headshot>' \
NUM_PROCESSES=3 \
MAIN_PROCESS_PORT=29503 \
  bash training_scripts/run_sdxl_pti_lora_single.sh
```

默认输出：

```text
outputs/sdxl_lora/<style>/
  pytorch_lora_weights.safetensors
  learned_embeds.safetensors
  training_config.json
  training_metadata.json
  checkpoint-200/
  checkpoint-400/
  ...
```

## 生成评估图

单风格 LoRA：

```bash
.conda/sdxl-lora/bin/python scripts/generate_sdxl_lora.py \
  --model-dir models/sdxl-base-1.0 \
  --lora-dir outputs/sdxl_lora/persona_5 \
  --output-dir outputs/eval_persona_5 \
  --prompt "<persona_5_headshot>, red haired female anime thief in a black outfit, bold red and black graphic background" \
  --prompt "<persona_5_headshot>, black haired anime boy in a school uniform, dramatic red shadow behind him" \
  --seeds 401 402 \
  --steps 30 \
  --guidance-scale 7.5 \
  --lora-scale 0.65
```

推荐扫 `--lora-scale 0.55 0.65 0.75`。风格太弱就加 scale，脸和构图开始崩就降 scale。

## Multi-token LoRA Merge

保留三个触发词的合并：

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

验证三个 token：

```bash
.conda/sdxl-lora/bin/python scripts/generate_sdxl_lora.py \
  --model-dir models/sdxl-base-1.0 \
  --lora-dir outputs/sdxl_lora/merged_multitoken_ghibli_persona5_eva_rei_equal \
  --output-dir outputs/eval_merged_multitoken_smoke \
  --prompt "<ghibli_headshot>, close-up anime portrait, soft forest background" \
  --prompt "<persona_5_headshot>, close-up anime thief portrait, red and black graphic poster style" \
  --prompt "<eva_rei_headshot>, close-up portrait of Ayanami Rei, blue hair, red eyes" \
  --seeds 701 \
  --width 768 \
  --height 768 \
  --steps 16 \
  --lora-scale 0.65
```

## 远程同步

当前常用远程路径：

```text
fstqwq:~/valencia/sdxl-style-lora
```

同步脚本：

```bash
rsync -av scripts/ training_scripts/ fstqwq:~/valencia/sdxl-style-lora/
```

拉回评估图：

```bash
rsync -av fstqwq:~/valencia/sdxl-style-lora/outputs/eval_persona_5/ \
  outputs/remote_eval_persona_5/
```

让 Codex 操作远程时，最好明确：

```text
先 ssh fstqwq，检查 ~/valencia/sdxl-style-lora、nvidia-smi 和 logs，不要直接覆盖已有 outputs。
```

