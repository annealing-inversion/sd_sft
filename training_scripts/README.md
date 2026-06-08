# SDXL PTI + LoRA Script Reference

这里是脚本层面的说明。项目总览和推荐命令见仓库根目录 `README.md`，数据集规范见 `docs/datasets.md`。

## Entrypoints

| 文件 | 用途 |
|---|---|
| `run_sdxl_pti_lora_single.sh` | 正式单风格多 GPU 训练入口 |
| `run_sdxl_pti_lora_smoke.sh` | 本地或远程小训练 smoke test |
| `train_sdxl_pti_lora.py` | 实际训练实现 |

## 训练流程

`train_sdxl_pti_lora.py` 分两阶段：

1. PTI/Textual Inversion warmup：只训练新的 placeholder token 在 SDXL 两个 text encoder 里的 embedding。
2. LoRA stage：冻结或可选继续训练 token，然后训练 UNet LoRA；正式配置还会训练 text encoder LoRA。

最终输出：

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

## 默认正式配置

`run_sdxl_pti_lora_single.sh` 默认假设至少 3 张 24GB GPU：

| 参数 | 默认值 |
|---|---:|
| resolution | 1024 |
| train batch size per GPU | 1 |
| gradient accumulation | 4 |
| max train steps | 2000 |
| TI/PTI steps | 500 |
| checkpoint interval | 200 |
| LoRA rank / alpha | 16 / 16 |
| UNet LoRA lr | `1e-4` |
| text encoder LoRA lr | `5e-6` |
| TI lr | `5e-4` |
| precision | fp16 |

## 当前三个风格

直接运行 `bash training_scripts/run_sdxl_pti_lora_single.sh` 时默认训练 `ghibli`。其他风格建议显式指定：

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

`STYLE` 只影响默认路径。token 大小写要显式写对，尤其 `data/Ghibli` 和 `data/EVA_rei` 的目录名不是最终 token 的大小写。

## Dataset Assumptions

训练脚本读取：

```text
data/<style>/
  image_0001.jpg
  image_0001.txt
```

如果 `.txt` 缺失，会使用 `DEFAULT_CAPTION`。如果 caption 中没有 placeholder token，脚本会自动加在开头。

`metadata.csv` 不参与训练，只是审查记录。

## Smoke Test

```bash
bash training_scripts/run_sdxl_pti_lora_smoke.sh
```

smoke test 默认 256 分辨率、2 个 step，只验证代码路径和 artifact 保存。它不验证最终质量。

## Direct Python Call

需要临时改参数时可以绕过 shell：

```bash
.conda/sdxl-lora/bin/python training_scripts/train_sdxl_pti_lora.py \
  --pretrained-model-name-or-path models/sdxl-base-1.0 \
  --data-dir data/EVA_rei \
  --output-dir outputs/sdxl_lora/eva_rei_debug \
  --placeholder-token '<eva_rei_headshot>' \
  --initializer-token style \
  --resolution 768 \
  --center-crop \
  --train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-train-steps 50 \
  --ti-steps 10 \
  --rank 8 \
  --lora-alpha 8 \
  --learning-rate 1e-4 \
  --mixed-precision fp16
```

## Generation and Merge

生成脚本在 `scripts/generate_sdxl_lora.py`：

```bash
.conda/sdxl-lora/bin/python scripts/generate_sdxl_lora.py \
  --model-dir models/sdxl-base-1.0 \
  --lora-dir outputs/sdxl_lora/eva_rei \
  --output-dir outputs/eval_eva_rei \
  --prompt "<eva_rei_headshot>, close-up portrait of Ayanami Rei, blue hair, red eyes" \
  --seeds 501 502 \
  --lora-scale 0.65
```

合并脚本在 `scripts/merge_sdxl_loras.py`。当前推荐 multi-token merge：

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

`concat` 会增加 rank，但更忠实地保留多个 LoRA 的加权增量。`--embed-merge keep` 保留每个输入 LoRA 的原始触发词。
