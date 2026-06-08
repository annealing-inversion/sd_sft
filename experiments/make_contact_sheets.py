#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


STYLE_ORDER = ["ghibli", "persona_5", "eva_rei"]
STYLE_LABELS = {
    "ghibli": "Ghibli",
    "persona_5": "Persona 5",
    "eva_rei": "EVA Rei",
}

PROMPT_LABELS = {
    "ghibli_forest": "forest",
    "ghibli_sky": "sky",
    "persona_thief": "thief",
    "persona_school": "school",
    "rei_plugsuit": "plugsuit",
    "rei_uniform": "uniform",
}

COMPARISON_ORDER = ["base", "single_lora"]
COMPARISON_LABELS = {
    "base": "Base SDXL",
    "single_lora": "LoRA",
}

CHECKPOINT_ORDER = [
    "checkpoint-200",
    "checkpoint-600",
    "checkpoint-1000",
    "checkpoint-1400",
    "checkpoint-1800",
    "final",
]

DOMAIN_ORDER = ["in_domain", "out_of_domain"]
DOMAIN_LABELS = {
    "in_domain": "in-domain",
    "out_of_domain": "out-of-domain",
}

ROLE_ORDER = [
    "rei_only",
    "persona_only",
    "ghibli_only",
    "rei_on_persona",
    "rei_on_ghibli",
    "general_portrait",
]
ROLE_LABELS = {
    "rei_only": "Rei only",
    "persona_only": "Persona only",
    "ghibli_only": "Ghibli only",
    "rei_on_persona": "Rei + Persona",
    "rei_on_ghibli": "Rei + Ghibli",
    "general_portrait": "general",
}

MERGE_VARIANT_ORDER = ["equal", "ghibli_heavy", "persona_heavy", "rei_heavy"]
MERGE_VARIANT_LABELS = {
    "equal": "equal",
    "ghibli_heavy": "ghibli-heavy",
    "persona_heavy": "persona-heavy",
    "rei_heavy": "rei-heavy",
}


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def style_sort_key(style: str) -> tuple[int, str]:
    return (STYLE_ORDER.index(style) if style in STYLE_ORDER else len(STYLE_ORDER), style)


def ordered_values(values: Iterable[Any], order: list[Any] | None = None) -> list[Any]:
    values = list(dict.fromkeys(values))
    if order is None:
        return sorted(values)
    return sorted(values, key=lambda item: (order.index(item) if item in order else len(order), str(item)))


def scale_label(value: float) -> str:
    return f"scale={value:g}"


def seed_label(seed: int) -> str:
    return f"seed\n{seed}"


def resolve_image_path(record: dict[str, Any], metadata_root: Path) -> Path:
    path = Path(record["output_path"])
    if path.exists():
        return path

    candidate = metadata_root / record["experiment_id"] / record["variant_id"] / path.name
    if candidate.exists():
        return candidate

    parts = path.parts
    if metadata_root.name in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index(metadata_root.name)
        candidate = metadata_root.joinpath(*parts[idx + 1 :])
        if candidate.exists():
            return candidate

    return path


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "#111111",
) -> None:
    lines = text.split("\n")
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [box_[3] - box_[1] for box_ in line_boxes]
    total_height = sum(line_heights) + max(0, len(lines) - 1) * 4
    y = box[1] + (box[3] - box[1] - total_height) // 2
    for line, line_box, line_height in zip(lines, line_boxes, line_heights):
        width = line_box[2] - line_box[0]
        x = box[0] + (box[2] - box[0] - width) // 2
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height + 4


def paste_image(sheet: Image.Image, image_path: Path, box: tuple[int, int, int, int]) -> None:
    if not image_path.exists():
        draw = ImageDraw.Draw(sheet)
        font = load_font(22, bold=True)
        draw.rectangle(box, fill="#f8f8f8", outline="#d0d0d0")
        draw_centered_text(draw, box, "missing", font, fill="#9a1f1f")
        return

    image = Image.open(image_path).convert("RGB")
    max_size = (box[2] - box[0], box[3] - box[1])
    image = ImageOps.contain(image, max_size, method=Image.Resampling.LANCZOS)
    x = box[0] + (max_size[0] - image.width) // 2
    y = box[1] + (max_size[1] - image.height) // 2
    sheet.paste(image, (x, y))


def make_matrix_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    row_key: Callable[[dict[str, Any]], Any],
    row_label: Callable[[Any], str],
    col_key: Callable[[dict[str, Any]], Any],
    col_label: Callable[[Any], str],
    image_path: Callable[[dict[str, Any]], Path],
    row_order: list[Any] | None = None,
    col_order: list[Any] | None = None,
    thumb_size: int = 288,
    row_label_width: int = 150,
    title_height: int = 58,
    header_height: int = 50,
    margin: int = 14,
) -> None:
    if not records:
        return

    rows = ordered_values([row_key(record) for record in records], row_order)
    cols = ordered_values([col_key(record) for record in records], col_order)
    lookup: dict[tuple[Any, Any], dict[str, Any]] = {}
    for record in records:
        lookup[(row_key(record), col_key(record))] = record

    width = margin * 2 + row_label_width + len(cols) * thumb_size
    height = margin * 2 + title_height + header_height + len(rows) * thumb_size
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    title_font = load_font(28, bold=True)
    header_font = load_font(23, bold=True)
    row_font = load_font(22, bold=True)

    draw.text((margin, margin + 6), title, fill="#111111", font=title_font)

    grid_x = margin + row_label_width
    grid_y = margin + title_height + header_height
    header_y = margin + title_height
    row_x = margin

    draw.rectangle((grid_x, header_y, grid_x + len(cols) * thumb_size, grid_y), fill="#f1f1f1")
    draw.rectangle((row_x, grid_y, grid_x, grid_y + len(rows) * thumb_size), fill="#f6f6f6")

    for col_idx, col in enumerate(cols):
        x0 = grid_x + col_idx * thumb_size
        x1 = x0 + thumb_size
        draw.rectangle((x0, header_y, x1, grid_y), outline="#d0d0d0")
        draw_centered_text(draw, (x0 + 4, header_y, x1 - 4, grid_y), col_label(col), header_font)

    for row_idx, row in enumerate(rows):
        y0 = grid_y + row_idx * thumb_size
        y1 = y0 + thumb_size
        draw.rectangle((row_x, y0, grid_x, y1), outline="#d0d0d0")
        draw_centered_text(draw, (row_x + 4, y0, grid_x - 4, y1), row_label(row), row_font)

        for col_idx, col in enumerate(cols):
            x0 = grid_x + col_idx * thumb_size
            x1 = x0 + thumb_size
            draw.rectangle((x0, y0, x1, y1), outline="#d0d0d0")
            record = lookup.get((row, col))
            if record is not None:
                paste_image(sheet, image_path(record), (x0 + 2, y0 + 2, x1 - 2, y1 - 2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def group_by_style(records: list[dict[str, Any]], experiment_id: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["experiment_id"] != experiment_id:
            continue
        style = (record.get("metadata") or {}).get("style")
        if style:
            grouped[style].append(record)
    return dict(sorted(grouped.items(), key=lambda item: style_sort_key(item[0])))


def write_index(paths: list[Path], output_dir: Path) -> None:
    lines = ["# Contact Sheets", ""]
    for path in sorted(paths):
        lines.append(f"- [{path.name}]({path.name})")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report_sheets(records: list[dict[str, Any]], output_dir: Path, metadata_root: Path, thumb_size: int) -> list[Path]:
    outputs: list[Path] = []

    def image_path(record: dict[str, Any]) -> Path:
        return resolve_image_path(record, metadata_root)

    for style, group in group_by_style(records, "base_vs_single").items():
        path = output_dir / f"base_vs_single_{style}.jpg"
        make_matrix_sheet(
            group,
            path,
            title=f"Base SDXL vs LoRA: {STYLE_LABELS.get(style, style)}",
            row_key=lambda r: (r["prompt_id"], r["seed"]),
            row_label=lambda key: f"{PROMPT_LABELS.get(key[0], key[0])}\n{seed_label(key[1])}",
            col_key=lambda r: (r.get("metadata") or {}).get("comparison"),
            col_label=lambda key: COMPARISON_LABELS.get(key, str(key)),
            image_path=image_path,
            row_order=[],
            col_order=COMPARISON_ORDER,
            thumb_size=thumb_size,
            row_label_width=170,
        )
        outputs.append(path)

    for style, group in group_by_style(records, "scale_sweep").items():
        path = output_dir / f"scale_sweep_{style}.jpg"
        make_matrix_sheet(
            group,
            path,
            title=f"LoRA Scale Sweep: {STYLE_LABELS.get(style, style)}",
            row_key=lambda r: r["seed"],
            row_label=seed_label,
            col_key=lambda r: float(r["lora_scale"]),
            col_label=scale_label,
            image_path=image_path,
            col_order=[0.4, 0.55, 0.65, 0.75, 0.9],
            thumb_size=thumb_size,
            row_label_width=150,
        )
        outputs.append(path)

    for style, group in group_by_style(records, "checkpoint_dynamics").items():
        path = output_dir / f"checkpoint_dynamics_{style}.jpg"
        make_matrix_sheet(
            group,
            path,
            title=f"Checkpoint Dynamics: {STYLE_LABELS.get(style, style)}",
            row_key=lambda r: r["seed"],
            row_label=seed_label,
            col_key=lambda r: (r.get("metadata") or {}).get("checkpoint"),
            col_label=lambda key: str(key),
            image_path=image_path,
            col_order=CHECKPOINT_ORDER,
            thumb_size=thumb_size,
            row_label_width=150,
        )
        outputs.append(path)

    for style, group in group_by_style(records, "generalization").items():
        path = output_dir / f"generalization_{style}.jpg"
        make_matrix_sheet(
            group,
            path,
            title=f"Cross-prompt Generalization: {STYLE_LABELS.get(style, style)}",
            row_key=lambda r: r["seed"],
            row_label=seed_label,
            col_key=lambda r: (r.get("metadata") or {}).get("domain"),
            col_label=lambda key: DOMAIN_LABELS.get(key, str(key)),
            image_path=image_path,
            col_order=DOMAIN_ORDER,
            thumb_size=thumb_size,
            row_label_width=150,
        )
        outputs.append(path)

    multitoken = [record for record in records if record["experiment_id"] == "multitoken_merge"]
    if multitoken:
        path = output_dir / "multitoken_merge.jpg"
        make_matrix_sheet(
            multitoken,
            path,
            title="Multi-token Merge Control",
            row_key=lambda r: r["seed"],
            row_label=seed_label,
            col_key=lambda r: (r.get("metadata") or {}).get("role"),
            col_label=lambda key: ROLE_LABELS.get(key, str(key)),
            image_path=image_path,
            col_order=ROLE_ORDER,
            thumb_size=thumb_size,
            row_label_width=150,
        )
        outputs.append(path)

    merge_weights = [record for record in records if record["experiment_id"] == "merge_weights"]
    roles = ordered_values([(record.get("metadata") or {}).get("role") for record in merge_weights], ROLE_ORDER)
    for role in roles:
        if role is None:
            continue
        group = [record for record in merge_weights if (record.get("metadata") or {}).get("role") == role]
        path = output_dir / f"merge_weights_{role}.jpg"
        make_matrix_sheet(
            group,
            path,
            title=f"Merge Weights: {ROLE_LABELS.get(role, role)}",
            row_key=lambda r: (r.get("metadata") or {}).get("merge_variant"),
            row_label=lambda key: MERGE_VARIANT_LABELS.get(key, str(key)),
            col_key=lambda r: r["seed"],
            col_label=seed_label,
            image_path=image_path,
            row_order=MERGE_VARIANT_ORDER,
            thumb_size=thumb_size,
            row_label_width=180,
        )
        outputs.append(path)

    write_index(outputs, output_dir)
    outputs.append(output_dir / "index.md")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report-ready contact sheets from report experiment metadata.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thumb-size", type=int, default=288)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_root = metadata_path.parent
    records = [
        record
        for record in metadata.get("images", [])
        if record.get("status") in {"generated", "skipped_existing"}
        and resolve_image_path(record, metadata_root).exists()
    ]

    outputs = build_report_sheets(records, output_dir, metadata_root, args.thumb_size)
    for path in outputs:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
