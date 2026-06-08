# Report Plan

这份文件是给组员和 Codex 用的报告写作计划。它可以记录实验流程、图片排版、复现路径和注意事项；每个人自己的正式报告正文通常放在本地 `reports/course_report.md`，只保留报告口吻的技术解释、实验结果和结论。

## 1. 当前状态

报告主题已经确定为：SDXL 小样本动漫人物关键词与风格关键词 LoRA 的可控性、泛化和合并实验。

仓库内保留的是可复现实验和写作流程，不保留任何人的成品报告。当前公开文件包括：

- `experiments/report_prompts.json`：统一 prompt 与实验定义。
- `experiments/run_report_experiments.sh`：一键生成报告实验图。
- `experiments/make_contact_sheets.py`：把实验输出整理成报告图。
- `reports/report_plan.md`：报告写作计划。

`reports/course_report.md` 和 `reports/figures/` 是每个组员自己的本地私有产物，已经被 `.gitignore` 忽略。不要把自己的报告和报告图提交到仓库，避免和其他组员的报告混在一起。

本轮报告实验只包含推理和 LoRA merge，不包含新的训练。训练参数只作为技术路线和已完成模型的背景信息写入各自报告。

## 2. 写作边界

正式报告应该像课程作业报告，不要像工作日志。可以写“图 1 展示了 LoRA 与 base SDXL 的对比”，不要写“这张 contact sheet 每行/每列表示什么、prompt 放在图片下方”这类排版说明。

下面这些内容可以放在本 plan 里，但不要放进 `course_report.md` 正文：

- contact sheet 的布局规则。
- 本地或远程输出目录。
- 服务器主机名、SSH alias、绝对路径。
- 临时调试命令、拉取图片的过程。
- “给 Codex 的指令”式文本。

报告正文允许保留必要的复现入口，例如统一实验脚本和公开仓库内的相对路径；不要写任何私人机器信息。

## 3. 固定实验设置

报告使用三个 token：

```text
<ghibli_headshot>
<persona_5_headshot>
<eva_rei_headshot>
```

对应数据集：

| 数据集 | token | 目标 |
|---|---|---|
| `data/Ghibli` | `<ghibli_headshot>` | 吉卜力式柔和手绘动画风格 |
| `data/persona_5` | `<persona_5_headshot>` | Persona 5 式红黑图形化风格 |
| `data/EVA_rei` | `<eva_rei_headshot>` | Ayanami Rei 人物身份 |

推理默认参数：

| 参数 | 默认值 |
|---|---|
| Resolution | `1024 x 1024` |
| Steps | `30` |
| Guidance scale | `7.5` |
| Default LoRA scale | `0.65` |
| Seeds | `SEED_BASE`, `SEED_BASE + 1` |
| Scheduler | `DPMSolverMultistepScheduler` |

建议每个组员选择自己的主 seed，例如学号后四位或日期加学号后四位：

```text
SEED_BASE=<your_seed>
```

组员要生成不同图片时，只改 `SEED_BASE`，不要改实验结构。这样报告结果不完全相同，但仍然可比较。

## 4. 复现实验命令

先做 dry-run，确认任务数量和路径：

```bash
SEED_BASE=1234 DRY_RUN=1 bash experiments/run_report_experiments.sh
```

只跑小规模 smoke：

```bash
SEED_BASE=1234 WIDTH=768 HEIGHT=768 STEPS=16 MAX_IMAGES=12 bash experiments/run_report_experiments.sh
```

正式生成报告图：

```bash
SEED_BASE=1234 WIDTH=1024 HEIGHT=1024 STEPS=30 bash experiments/run_report_experiments.sh
```

只跑某一个实验：

```bash
SEED_BASE=1234 EXPERIMENTS=multitoken_merge bash experiments/run_report_experiments.sh
```

可选实验名：

```text
base_vs_single
scale_sweep
checkpoint_dynamics
generalization
multitoken_merge
merge_weights
```

脚本会先生成需要的 merged LoRA，再调用 `experiments/generate_report_images.py` 出图，最后调用 `experiments/make_contact_sheets.py` 生成 contact sheets。

## 5. 报告图产物

生成报告后，通常会把这些图复制或生成到本地 `reports/figures/`，再嵌入本地 `reports/course_report.md`。这些文件默认不提交：

```text
reports/figures/base_vs_single_6col.jpg
reports/figures/scale_sweep_ghibli.jpg
reports/figures/scale_sweep_persona_5.jpg
reports/figures/scale_sweep_eva_rei.jpg
reports/figures/checkpoint_dynamics_ghibli.jpg
reports/figures/checkpoint_dynamics_persona_5.jpg
reports/figures/checkpoint_dynamics_eva_rei.jpg
reports/figures/generalization_6col.jpg
reports/figures/multitoken_merge_prompts.jpg
reports/figures/merge_weights_8col.jpg
```

图片排版规则保留在这里，方便以后重生成：

- 5.1 使用 6 列：每个风格两列，左侧 base SDXL，右侧单 LoRA。
- 5.2 按风格拆分，每张图展示不同 `lora_scale`。
- 5.3 按风格拆分，checkpoint 必须按数值顺序排列：`200, 600, 1000, 1400, 1800, final`。
- 5.4 使用 6 列：每个风格两列，分别是 in-domain 和 out-of-domain。
- 5.5 展示 equal merge 下的单 token 与多 token 组合，图片中可以标出 prompt。
- 5.6 使用 8 列：4 个 merge 权重设置，每个设置展示两个 seed，图片中可以标出 prompt。

注意：这些排版规则不要原样写进正式报告正文。报告里只写实验目的、结果观察和结论。

## 6. 实验写作要点

### E1 原始 SDXL vs 单 LoRA

目的：证明单 LoRA 相比 base SDXL 带来了可见的风格或人物变化。

报告需要回答：

- Base SDXL 是否已经能理解主体或风格 prompt？
- LoRA 具体强化了哪些可见特征？
- LoRA 是否带来了构图模板化或多样性下降？

### E2 LoRA Scale 扫描

目的：分析 `lora_scale` 对风格强度、身份稳定性和画质的影响。

固定测试：

```text
0.4 / 0.55 / 0.65 / 0.75 / 0.9
```

报告需要回答：

- 哪个 scale 最适合作为默认值？
- scale 过低时缺什么？
- scale 过高时是否出现过拟合感、背景拥挤、颜色过饱和或构图僵硬？

### E3 Checkpoint 动态

目的：观察训练过程中风格或身份何时出现、何时趋于稳定。

固定测试：

```text
checkpoint-200 / checkpoint-600 / checkpoint-1000 / checkpoint-1400 / checkpoint-1800 / final
```

报告需要回答：

- 早期 checkpoint 是否已经学到主要颜色、发色或背景？
- 中后期 checkpoint 是否更稳定？
- final 是否明显优于中期 checkpoint，还是只是更接近训练集偏好？

### E8 跨 Prompt 泛化

目的：测试 LoRA 是否只记住训练集构图，还是能迁移到新场景。

报告需要回答：

- 风格 LoRA 是否能迁移到新场景？
- Rei 人物 token 在新服装和新背景下是否仍保留身份特征？
- 失败主要发生在身份、风格、背景还是画质？

当前 out-of-domain prompt 使用：

```text
<ghibli_headshot>, close-up portrait of a young mechanic inside a cozy airship workshop, brass tools, cloudy sky outside, soft hand painted animation style
<persona_5_headshot>, stylish anime detective in a black coat, red gloves, neon city background, bold red and black graphic poster style
<eva_rei_headshot>, Ayanami Rei wearing a black winter coat in a snowy city street, blue hair, red eyes
```

### E9 Multi-token Merge 控制

目的：验证一个 merged adapter 中保留多个触发词后，能否组合人物 token 和风格 token。

重点组合：

```text
<eva_rei_headshot> + <persona_5_headshot>
<eva_rei_headshot> + <ghibli_headshot>
```

报告需要回答：

- Rei on Persona 5 是否同时保留 Rei 身份和红黑图形化风格？
- Rei on Ghibli 是否同时保留 Rei 身份和柔和手绘动画感？
- 多 token 同时出现时，是人物 token 更强，还是风格 token 更强？

### E10 Merge 权重比例

目的：测试 concat merge 中不同 LoRA 权重是否能改变合并模型倾向。

固定权重：

```text
equal = 1 / 1 / 1
ghibli-heavy = 0.6 / 0.2 / 0.2
persona-heavy = 0.2 / 0.6 / 0.2
rei-heavy = 0.2 / 0.2 / 0.6
```

报告需要回答：

- 加大某个 LoRA 的权重是否真的增强对应风格或身份？
- 风格权重提高时，Rei 身份是否被削弱？
- Rei 权重提高时，Persona 5 或 Ghibli 风格是否变弱？
- 权重控制是否接近线性，还是只能作为粗粒度倾向控制？

## 7. 报告正文检查清单

提交报告前检查：

- 是否包含项目目标、数据集、技术路线、实验参数、实验结果、结论、局限性和后续工作。
- 是否解释了为什么使用 SDXL、PTI warmup、LoRA 和 multi-token merge。
- 是否明确说明本轮报告实验不包含新的训练。
- 是否写出了训练参数和推理参数。
- 是否在本地报告中嵌入了结果图片，而不是只给图片路径。
- 是否删除了占位符、TODO、调试过程和 Codex 指令式文本。
- 是否避免了“每列表示”“每行表示”“图片下方标注”这类排版说明。
- 是否没有出现服务器主机名、SSH alias、远程绝对路径或本机用户名路径。

建议用下面的命令做文本检查：

```bash
rg -n "TODO|待填写|待实验完成|ssh|SSH|/mnt/|/home/|~/" reports experiments README.md AGENTS.md docs .gitignore
rg -n "每列|每一行|占两列|行表示|列表示|图片下方|采用 8 列|按风格拆分|contact sheet" reports/course_report.md
```

图片链接检查：

```bash
python - <<'PY'
from pathlib import Path
import re

report = Path("reports/course_report.md").read_text(encoding="utf-8")
links = re.findall(r"!\\[[^\\]]*\\]\\(([^)]+)\\)", report)
missing = [link for link in links if not (Path("reports") / link).exists()]
print(f"image_links={len(links)}")
print("missing=" + (", ".join(missing) if missing else "none"))
PY
```

## 8. 不要提交的内容

不要提交：

- `outputs/` 下的原始生成图片和 metadata。
- `.cache/`、`.conda/`、模型权重。
- 包含服务器路径、SSH alias、用户名路径的笔记。
- `reports/course_report.md` 和 `reports/figures/` 下的个人报告产物。
- 临时对比图和中间可视化结果。

如果为了自己记录远程运行信息，需要放到本地私有文件，先确认文件被 `.gitignore` 忽略。

## 9. 给 Codex 的推荐指令

继续修改报告时，可以直接这样说：

```text
请阅读 reports/report_plan.md 和本地 reports/course_report.md，检查报告正文是否还有工作日志口吻、排版说明或私人路径，并直接修正。不要提交 course_report.md 和 reports/figures/。
```

重新生成不同 seed 的报告图时：

```text
请用 SEED_BASE=你的种子 WIDTH=1024 HEIGHT=1024 STEPS=30 运行 experiments/run_report_experiments.sh，生成新的 report figures，然后根据图片更新本地 reports/course_report.md。不要把 outputs/、reports/course_report.md、reports/figures/ 或任何服务器信息提交到仓库。
```

只改某个实验图时：

```text
请只重跑 EXPERIMENTS=generalization，并把新的 contact sheet 更新到本地 reports/figures/，然后同步修改本地 reports/course_report.md 中对应小节的观察表。不要提交这些本地报告产物。
```
