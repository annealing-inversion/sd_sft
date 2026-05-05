# Beamer 素材说明

`slides/midterm_report.tex` 会引用本目录中的图片。

## Base 生成结果

`base/` 中是 Stable Diffusion v1.5 Base 根据 5 个个人主题 prompt 生成的测试图，用于说明基础模型的水彩风格不稳定。

这些图片建议保留：

- `base/01_seed-42_watercolor-style-a-small-boat-on-a-mountain-lake.png`
- `base/02_seed-42_watercolor-style-a-misty-lake-after-rain.png`
- `base/03_seed-42_watercolor-style-a-quiet-village-with-a-stone-bridge.png`
- `base/04_seed-42_watercolor-style-pine-trees-beside-a-lake-at-sunset.png`
- `base/05_seed-42_watercolor-style-wildflowers-beside-a-peaceful-river.png`

## 数据集示例图

`examples/` 中是汇报用的数据集方向示例图。后续如果手动找到更合适的图片，只需要替换同名文件即可，不需要改 Beamer 源码。

需要保留的文件名：

- `examples/overall.jpg`：整体公共数据集示例
- `examples/member_a_boat_lake.jpg`：成员 A，山湖 + 小船
- `examples/member_b_rain_mist.jpg`：成员 B，雾景 + 雨后氛围
- `examples/member_c_bridge_village.jpg`：成员 C，小镇 + 建筑 + 桥
- `examples/member_d_pine_sunset.jpg`：成员 D，森林 + 松树 + 粉紫日落
- `examples/member_e_flowers_river.jpg`：成员 E，植物 + 花草 + 近景自然

如果替换图片，建议使用 `.jpg` 格式，并保持文件名不变。
