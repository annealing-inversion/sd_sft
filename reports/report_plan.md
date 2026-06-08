# Report Plan

这份文件是后续写课程报告时给组员和 Codex 用的执行计划。每次完成脚本、跑完实验、整理结果或修改报告，都应该同步更新这里的状态。

## 当前状态

- [x] 明确报告主题：SDXL 小样本动漫人物关键词与风格关键词 LoRA 的可控性、泛化和合并实验。
- [x] 选定实验集合：原始 SDXL vs 单 LoRA、LoRA scale 扫描、checkpoint 动态、跨 prompt 泛化、multi-token merge、merge 权重比例。
- [x] 建立可复用 prompt 配置：`experiments/report_prompts.json`。
- [x] 建立可复用实验入口：`experiments/run_report_experiments.sh`。
- [x] 建立 contact sheet 生成脚本：`experiments/make_contact_sheets.py`。
- [x] 建立报告草稿：`reports/course_report.md`。
- [x] 完成 dry-run 验证：`SEED_BASE=1234` 时完整计划包含 144 张图，使用种子 `1234` 和 `1235`。
- [x] 完成子实验 dry-run 验证：`EXPERIMENTS=multitoken_merge SEED_BASE=2222` 时计划包含 10 张图。
- [x] 跑完整实验并生成图片：`SEED_BASE=20260608`，共 144 张 PNG。
- [x] 根据生成图填写每个实验的观察表。
- [x] 使用报告图作为 Markdown 嵌入图，其中 5.1 和 5.4 采用三风格六列合并图。
- [x] 写最终结论并清理占位内容。

## 当前实验产物

本轮完整实验使用：

```text
SEED_BASE=20260608
WIDTH=1024
HEIGHT=1024
STEPS=30
GUIDANCE_SCALE=7.5
DEFAULT_LORA_SCALE=0.65
```

本地产物路径：

```text
outputs/report_experiments/seed_20260608/
outputs/report_experiments/seed_20260608/all_metadata.json
outputs/report_experiments/seed_20260608/run_plan.json
outputs/report_experiments/seed_20260608/contact_sheets/
```

完整性检查：

```text
run_plan tasks: 144
metadata images: 144
generated PNG files: 144
report figures: 10 contact sheets in reports/figures/
```

报告已根据本轮结果图完成分析。当前仓库内报告图包括：

```text
reports/figures/base_vs_single_6col.jpg
reports/figures/scale_sweep_{ghibli,persona_5,eva_rei}.jpg
reports/figures/checkpoint_dynamics_{ghibli,persona_5,eva_rei}.jpg
reports/figures/generalization_6col.jpg
reports/figures/multitoken_merge_prompts.jpg
reports/figures/merge_weights_8col.jpg
```

原始生成产物仍保留在本地输出目录，可用于重新挑图或排查。注意：`outputs/` 下的元数据可能包含运行机器上的绝对路径，不要提交这些文件：

```text
outputs/report_experiments/seed_20260608/
```

## 推荐 Codex 指令

clone 仓库后，可以让 Codex 按下面的话继续：

```text
请阅读 reports/report_plan.md，然后检查 experiments/run_report_experiments.sh 是否能 dry-run。使用 SEED_BASE=你的学号后四位加日期 的方式运行实验，生成 contact sheets，并根据结果补全 reports/course_report.md。
```

如果只想先跑小规模试验：

```text
请用 SEED_BASE=1234 WIDTH=768 HEIGHT=768 STEPS=16 MAX_IMAGES=12 DRY_RUN=0 跑 report experiments smoke，并检查 contact sheets 是否生成。
```

如果要跑正式报告图：

```text
请在共享 GPU 机器上用 SEED_BASE=1234 WIDTH=1024 HEIGHT=1024 STEPS=30 运行完整 report experiments，然后把 outputs/report_experiments/seed_1234 拉回本地。不要把主机名和绝对路径写进仓库。
```

## 报告结构

报告文件：`reports/course_report.md`

必须包含：

1. 项目目标和问题定义。
2. 数据集说明：三个小数据集、人物关键词和风格关键词、caption 设计。
3. 技术路线理由：为什么用 SDXL、为什么用 PTI warmup、为什么用 LoRA、为什么做 multi-token merge。
4. 训练和推理参数：分辨率、batch、step、LoRA rank、learning rate、seed、scheduler、guidance scale、LoRA scale。
5. 六个实验的设置、结果和分析。
6. 总体结论、局限性和后续改进。

## 实验列表

### E1 原始 SDXL vs 单 LoRA

目的：证明每个 LoRA 相比 base SDXL 带来了可见的风格或人物变化。

输出路径：

```text
outputs/report_experiments/seed_<SEED_BASE>/base_vs_single/
outputs/report_experiments/seed_<SEED_BASE>/contact_sheets/base_vs_single_6col.jpg
```

报告要回答：

- Base SDXL 是否已经能理解 prompt 中的主体？
- LoRA 相比 base 强化了哪些特征？
- 单 LoRA 是否牺牲了 prompt 响应或多样性？

### E2 LoRA Scale 扫描

目的：找出推理强度对风格、人物一致性和画质的影响。

测试 scale：

```text
0.4 / 0.55 / 0.65 / 0.75 / 0.9
```

报告要回答：

- 哪个 scale 最平衡？
- scale 过高时是否出现过拟合、脸部僵硬或风格污染？
- 三个 LoRA 的最佳 scale 是否一致？

### E3 Checkpoint 动态

目的：分析训练进程中风格何时出现、何时趋于稳定。

测试 checkpoint：

```text
checkpoint-200 / checkpoint-600 / checkpoint-1000 / checkpoint-1400 / checkpoint-1800 / final
```

报告要回答：

- 早期 checkpoint 是否只学到色彩或线条？
- 中后期 checkpoint 是否更稳定？
- final 是否比 checkpoint-1800 明显更好，还是已经接近平台期？

### E8 跨 Prompt 泛化

目的：测试 LoRA 是学到可迁移风格/身份，还是只记住训练图。

prompt 分组：

```text
in_domain
out_of_domain
```

报告要回答：

- 风格 LoRA 是否能迁移到新场景？
- Rei 人物 LoRA 在新服装、新背景下是否仍保留蓝发红眼和身份特征？
- out-of-domain 失败时主要失败在身份、风格还是画质？

### E9 Multi-token Merge 控制

目的：验证一个 merged LoRA 中保留多个触发词后，能否组合人物 token 和风格 token。

重点组合：

```text
<eva_rei_headshot> + <persona_5_headshot>
<eva_rei_headshot> + <ghibli_headshot>
```

报告要回答：

- Rei on Persona 5 是否保留 Rei 身份，同时出现红黑图形化风格？
- Rei on Ghibli 是否保留 Rei 身份，同时出现柔和背景和手绘动画感？
- 多 token 同时出现时，是人物 token 主导还是风格 token 主导？

### E10 Merge 权重比例

目的：测试 concat merge 中不同 LoRA 权重是否能改变合并模型倾向。

权重：

```text
equal: 1 / 1 / 1
ghibli-heavy: 0.6 / 0.2 / 0.2
persona-heavy: 0.2 / 0.6 / 0.2
rei-heavy: 0.2 / 0.2 / 0.6
```

报告要回答：

- 权重增加是否真的增强对应风格？
- 权重偏向人物时，风格组合是否变弱？
- 权重偏向风格时，Rei 身份是否被削弱？

## 结果记录格式

跑完实验后，在报告里为每个实验填一个短表：

```markdown
| 设置 | 观察 | 结论 |
|---|---|---|
| scale=0.4 | 风格较弱，但 prompt 响应较好 | 不适合作为主图 |
| scale=0.65 | 风格明显，人物稳定 | 推荐默认值 |
```

不要只说“效果好/不好”，要写具体可见现象，例如发色、眼睛、线条、背景、构图、红黑色块、柔和手绘感、训练图记忆痕迹等。

## 当前脚本命令

dry-run：

```bash
SEED_BASE=1234 DRY_RUN=1 bash experiments/run_report_experiments.sh
```

小规模 smoke：

```bash
SEED_BASE=1234 WIDTH=768 HEIGHT=768 STEPS=16 MAX_IMAGES=12 bash experiments/run_report_experiments.sh
```

正式实验：

```bash
SEED_BASE=1234 WIDTH=1024 HEIGHT=1024 STEPS=30 bash experiments/run_report_experiments.sh
```
