# 当前数据集说明

训练脚本只关心一个目录下的图片和同名 `.txt` caption。目录里可以有 `metadata.csv`，但它只是人工审查和记录用，不参与训练。

## 数据集目录

```text
data/Ghibli/
  ghibli_0001.jpg
  ghibli_0001.txt
  ...

data/persona_5/
  persona_5_0001.jpg
  persona_5_0001.txt
  ...

data/EVA_rei/
  eva_rei_0001.jpg
  eva_rei_0001.txt
  ...
```

当前本地工作副本里的数量：

| 数据集 | 图片 | caption | 备注 |
|---|---:|---:|---|
| `data/Ghibli` | 24 | 24 | 吉卜力风格人物和场景头像 |
| `data/persona_5` | 24 | 24 | Persona 5 红黑图形化人物风格 |
| `data/EVA_rei` | 23 | 23 | Ayanami Rei 人物图像 |

clone 仓库后这些数据集可能不存在，因为当前数据不进 git。需要从组内共享目录、远程机器或本地 zip 复制到同样路径。

## Caption 规则

caption 应该短、具体、不要编故事。推荐结构：

```text
<style_token>, visual style phrase, subject phrase, important attributes
```

实际训练脚本会额外确保 `PLACEHOLDER_TOKEN` 出现在 caption 里，所以 `.txt` 可以写风格标签，也可以直接写自然语言描述。为了后续人工检查，建议仍然在 caption 开头保留一个可读风格标签：

```text
ghibli_style, soft watercolor background, young girl in a straw hat under a blue sky
persona_5_style, bold graphic anime colors, black haired anime boy in a school uniform, red background
eva_rei_style, clean anime character art, Ayanami Rei with blue hair and red eyes in a white plugsuit
```

不要写：

```text
beautiful image, masterpiece, best quality
```

也不要把模型臆测出的错误人物、错误动作、错误物品硬写进去。

## 审查清单

检查新数据集时按这个顺序：

1. 图片和 `.txt` 是否一一对应。
2. caption 是否描述了图里真实可见的主体、发色、服装、背景。
3. caption 是否出现胡言乱语、重复词、明显错误人物名。
4. 图片是否过糊、严重压缩、主体太小、画风和同目录差异过大。
5. 不确定是否保留的图片只汇报，不直接删除。

可让 Codex 执行：

```text
请检查 data/persona_5 的图片和 caption，一张图一张图审查；caption 胡言乱语就修正，图片质量问题只汇报不要删除。
```

## 1024 分辨率和裁剪

正式训练使用 `--resolution 1024 --center-crop`。非 1:1 图片会被中心裁剪到正方形，这会带来两个风险：

- 竖图可能裁掉头顶、手、道具或背景关键信息。
- 横图可能只留下中间人物，削弱原构图和场景风格。

因此数据准备阶段尽量使用主体居中、头像足够大、裁成 1:1 后仍完整的图片。需要保留宽构图时，可以先人工裁成 1024x1024 或接近正方形，再放入数据集。

