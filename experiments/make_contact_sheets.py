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


def wrap_text_pixels(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = word if not line else f"{line} {word}"
        if draw.textlength(trial, font=font) <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


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


def make_base_vs_single_6col_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    image_path: Callable[[dict[str, Any]], Path],
    thumb_size: int = 230,
    margin: int = 14,
) -> None:
    records = [record for record in records if record["experiment_id"] == "base_vs_single"]
    if not records:
        return

    title_font = load_font(30, bold=True)
    style_font = load_font(22, bold=True)
    col_font = load_font(21, bold=True)
    prompt_font = load_font(13)

    styles = [
        style
        for style in STYLE_ORDER
        if any((record.get("metadata") or {}).get("style") == style for record in records)
    ]
    seeds = ordered_values([record["seed"] for record in records])

    prompt_ids_by_style: dict[str, list[str]] = {}
    prompt_lookup: dict[tuple[str, str], str] = {}
    record_lookup: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for style in styles:
        prompt_ids: list[str] = []
        for record in records:
            metadata = record.get("metadata") or {}
            if metadata.get("style") != style:
                continue
            comparison = metadata.get("comparison")
            prompt_id = record["prompt_id"]
            if prompt_id not in prompt_ids:
                prompt_ids.append(prompt_id)
            record_lookup[(style, comparison, prompt_id, record["seed"])] = record
            if comparison == "single_lora":
                prompt_lookup[(style, prompt_id)] = record["prompt"]
        prompt_ids_by_style[style] = prompt_ids

    prompt_count = max(len(prompt_ids) for prompt_ids in prompt_ids_by_style.values())
    rows = [(prompt_idx, seed) for prompt_idx in range(prompt_count) for seed in seeds]

    pair_width = thumb_size * 2
    col_count = len(styles) * 2
    title_height = 52
    style_height = 36
    col_height = 40

    probe = Image.new("RGB", (10, 10), "white")
    probe_draw = ImageDraw.Draw(probe)
    max_prompt_lines = 1
    for prompt in prompt_lookup.values():
        max_prompt_lines = max(max_prompt_lines, len(wrap_text_pixels(probe_draw, prompt, prompt_font, pair_width - 18)))
    prompt_line_height = 16
    prompt_height = max(58, max_prompt_lines * prompt_line_height + 14)

    width = margin * 2 + col_count * thumb_size
    height = margin * 2 + title_height + style_height + col_height + len(rows) * (thumb_size + prompt_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((margin, margin + 4), "5.1 Base SDXL vs Single LoRA", fill="#111111", font=title_font)

    x0 = margin
    style_y = margin + title_height
    col_y = style_y + style_height
    grid_y = col_y + col_height

    for style_idx, style in enumerate(styles):
        gx0 = x0 + style_idx * pair_width
        gx1 = gx0 + pair_width
        draw.rectangle((gx0, style_y, gx1, col_y), fill="#f1f1f1", outline="#cccccc")
        draw_centered_text(draw, (gx0, style_y, gx1, col_y), STYLE_LABELS.get(style, style), style_font)

    for style_idx, _style in enumerate(styles):
        for offset, comparison in enumerate(COMPARISON_ORDER):
            cx0 = x0 + style_idx * pair_width + offset * thumb_size
            cx1 = cx0 + thumb_size
            draw.rectangle((cx0, col_y, cx1, grid_y), fill="#eeeeee", outline="#c9c9c9")
            draw_centered_text(draw, (cx0, col_y, cx1, grid_y), COMPARISON_LABELS.get(comparison, comparison), col_font)

    current_y = grid_y
    for prompt_idx, seed in rows:
        image_y = current_y
        prompt_y = image_y + thumb_size

        for style_idx, style in enumerate(styles):
            prompt_ids = prompt_ids_by_style[style]
            if prompt_idx >= len(prompt_ids):
                continue
            prompt_id = prompt_ids[prompt_idx]
            for offset, comparison in enumerate(COMPARISON_ORDER):
                cx0 = x0 + style_idx * pair_width + offset * thumb_size
                cx1 = cx0 + thumb_size
                draw.rectangle((cx0, image_y, cx1, image_y + thumb_size), outline="#d0d0d0")
                record = record_lookup.get((style, comparison, prompt_id, seed))
                if record is not None:
                    paste_image(sheet, image_path(record), (cx0 + 2, image_y + 2, cx1 - 2, image_y + thumb_size - 2))

        for style_idx, style in enumerate(styles):
            prompt_ids = prompt_ids_by_style[style]
            if prompt_idx >= len(prompt_ids):
                continue
            prompt = prompt_lookup.get((style, prompt_ids[prompt_idx]), "")
            gx0 = x0 + style_idx * pair_width
            gx1 = gx0 + pair_width
            draw.rectangle((gx0, prompt_y, gx1, prompt_y + prompt_height), fill="#f7f7f7", outline="#d0d0d0")
            y = prompt_y + 7
            for line in wrap_text_pixels(draw, prompt, prompt_font, pair_width - 18):
                draw.text((gx0 + 8, y), line, fill="#222222", font=prompt_font)
                y += prompt_line_height

        current_y += thumb_size + prompt_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_generalization_6col_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    image_path: Callable[[dict[str, Any]], Path],
    thumb_size: int = 230,
    margin: int = 14,
) -> None:
    records = [record for record in records if record["experiment_id"] == "generalization"]
    if not records:
        return

    title_font = load_font(30, bold=True)
    style_font = load_font(22, bold=True)
    col_font = load_font(21, bold=True)
    prompt_font = load_font(12)

    styles = [
        style
        for style in STYLE_ORDER
        if any((record.get("metadata") or {}).get("style") == style for record in records)
    ]
    seeds = ordered_values([record["seed"] for record in records])

    record_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    prompt_lookup: dict[tuple[str, str], str] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        style = metadata.get("style")
        domain = metadata.get("domain")
        if style is None or domain is None:
            continue
        record_lookup[(style, domain, record["seed"])] = record
        prompt_lookup[(style, domain)] = record["prompt"]

    pair_width = thumb_size * 2
    col_count = len(styles) * 2
    title_height = 52
    style_height = 36
    col_height = 40

    probe = Image.new("RGB", (10, 10), "white")
    probe_draw = ImageDraw.Draw(probe)
    max_prompt_lines = 1
    for prompt in prompt_lookup.values():
        max_prompt_lines = max(max_prompt_lines, len(wrap_text_pixels(probe_draw, prompt, prompt_font, thumb_size - 18)))
    prompt_line_height = 15
    prompt_height = max(82, max_prompt_lines * prompt_line_height + 16)

    width = margin * 2 + col_count * thumb_size
    height = margin * 2 + title_height + style_height + col_height + len(seeds) * (thumb_size + prompt_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((margin, margin + 4), "5.4 Cross-prompt Generalization", fill="#111111", font=title_font)

    x0 = margin
    style_y = margin + title_height
    col_y = style_y + style_height
    grid_y = col_y + col_height

    for style_idx, style in enumerate(styles):
        gx0 = x0 + style_idx * pair_width
        gx1 = gx0 + pair_width
        draw.rectangle((gx0, style_y, gx1, col_y), fill="#f1f1f1", outline="#cccccc")
        draw_centered_text(draw, (gx0, style_y, gx1, col_y), STYLE_LABELS.get(style, style), style_font)

    for style_idx, _style in enumerate(styles):
        for offset, domain in enumerate(DOMAIN_ORDER):
            cx0 = x0 + style_idx * pair_width + offset * thumb_size
            cx1 = cx0 + thumb_size
            draw.rectangle((cx0, col_y, cx1, grid_y), fill="#eeeeee", outline="#c9c9c9")
            draw_centered_text(draw, (cx0, col_y, cx1, grid_y), DOMAIN_LABELS.get(domain, domain), col_font)

    current_y = grid_y
    for seed in seeds:
        image_y = current_y
        prompt_y = image_y + thumb_size

        for style_idx, style in enumerate(styles):
            for offset, domain in enumerate(DOMAIN_ORDER):
                cx0 = x0 + style_idx * pair_width + offset * thumb_size
                cx1 = cx0 + thumb_size
                draw.rectangle((cx0, image_y, cx1, image_y + thumb_size), outline="#d0d0d0")
                record = record_lookup.get((style, domain, seed))
                if record is not None:
                    paste_image(sheet, image_path(record), (cx0 + 2, image_y + 2, cx1 - 2, image_y + thumb_size - 2))

        for style_idx, style in enumerate(styles):
            for offset, domain in enumerate(DOMAIN_ORDER):
                cx0 = x0 + style_idx * pair_width + offset * thumb_size
                cx1 = cx0 + thumb_size
                draw.rectangle((cx0, prompt_y, cx1, prompt_y + prompt_height), fill="#f7f7f7", outline="#d0d0d0")
                prompt = prompt_lookup.get((style, domain), "")
                y = prompt_y + 7
                for line in wrap_text_pixels(draw, prompt, prompt_font, thumb_size - 18):
                    draw.text((cx0 + 8, y), line, fill="#222222", font=prompt_font)
                    y += prompt_line_height

        current_y += thumb_size + prompt_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_multitoken_prompt_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    image_path: Callable[[dict[str, Any]], Path],
    thumb_size: int = 240,
    margin: int = 14,
) -> None:
    records = [record for record in records if record["experiment_id"] == "multitoken_merge"]
    if not records:
        return

    title_font = load_font(30, bold=True)
    role_font = load_font(20, bold=True)
    prompt_font = load_font(12)

    roles = ordered_values([(record.get("metadata") or {}).get("role") for record in records], ROLE_ORDER)
    roles = [role for role in roles if role is not None]
    seeds = ordered_values([record["seed"] for record in records])

    record_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    prompt_lookup: dict[str, str] = {}
    for record in records:
        role = (record.get("metadata") or {}).get("role")
        if role is None:
            continue
        record_lookup[(role, record["seed"])] = record
        prompt_lookup[role] = record["prompt"]

    title_height = 52
    header_height = 44
    probe = Image.new("RGB", (10, 10), "white")
    probe_draw = ImageDraw.Draw(probe)
    max_prompt_lines = 1
    for prompt in prompt_lookup.values():
        max_prompt_lines = max(max_prompt_lines, len(wrap_text_pixels(probe_draw, prompt, prompt_font, thumb_size - 18)))
    prompt_line_height = 15
    prompt_height = max(74, max_prompt_lines * prompt_line_height + 16)

    width = margin * 2 + len(roles) * thumb_size
    height = margin * 2 + title_height + header_height + len(seeds) * (thumb_size + prompt_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((margin, margin + 4), "5.5 Multi-token Merge Control", fill="#111111", font=title_font)

    x0 = margin
    header_y = margin + title_height
    grid_y = header_y + header_height

    for role_idx, role in enumerate(roles):
        cx0 = x0 + role_idx * thumb_size
        cx1 = cx0 + thumb_size
        draw.rectangle((cx0, header_y, cx1, grid_y), fill="#eeeeee", outline="#c9c9c9")
        draw_centered_text(draw, (cx0, header_y, cx1, grid_y), ROLE_LABELS.get(role, role), role_font)

    current_y = grid_y
    for seed in seeds:
        image_y = current_y
        prompt_y = image_y + thumb_size

        for role_idx, role in enumerate(roles):
            cx0 = x0 + role_idx * thumb_size
            cx1 = cx0 + thumb_size
            draw.rectangle((cx0, image_y, cx1, image_y + thumb_size), outline="#d0d0d0")
            record = record_lookup.get((role, seed))
            if record is not None:
                paste_image(sheet, image_path(record), (cx0 + 2, image_y + 2, cx1 - 2, image_y + thumb_size - 2))

        for role_idx, role in enumerate(roles):
            cx0 = x0 + role_idx * thumb_size
            cx1 = cx0 + thumb_size
            draw.rectangle((cx0, prompt_y, cx1, prompt_y + prompt_height), fill="#f7f7f7", outline="#d0d0d0")
            y = prompt_y + 7
            for line in wrap_text_pixels(draw, prompt_lookup.get(role, ""), prompt_font, thumb_size - 18):
                draw.text((cx0 + 8, y), line, fill="#222222", font=prompt_font)
                y += prompt_line_height

        current_y += thumb_size + prompt_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def make_merge_weights_8col_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    image_path: Callable[[dict[str, Any]], Path],
    thumb_size: int = 188,
    margin: int = 14,
) -> None:
    records = [record for record in records if record["experiment_id"] == "merge_weights"]
    if not records:
        return

    title_font = load_font(30, bold=True)
    variant_font = load_font(19, bold=True)
    seed_font = load_font(16, bold=True)
    prompt_font = load_font(16)

    variants = ordered_values([(record.get("metadata") or {}).get("merge_variant") for record in records], MERGE_VARIANT_ORDER)
    variants = [variant for variant in variants if variant is not None]
    roles = ordered_values([(record.get("metadata") or {}).get("role") for record in records], ROLE_ORDER)
    roles = [role for role in roles if role is not None]
    seeds = ordered_values([record["seed"] for record in records])

    record_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    prompt_lookup: dict[str, str] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        variant = metadata.get("merge_variant")
        role = metadata.get("role")
        if variant is None or role is None:
            continue
        record_lookup[(variant, role, record["seed"])] = record
        prompt_lookup[role] = record["prompt"]

    title_height = 52
    variant_header_height = 42
    seed_header_height = 34
    col_count = len(variants) * len(seeds)

    probe = Image.new("RGB", (10, 10), "white")
    probe_draw = ImageDraw.Draw(probe)
    max_prompt_lines = 1
    prompt_width = col_count * thumb_size - 18
    for prompt in prompt_lookup.values():
        max_prompt_lines = max(max_prompt_lines, len(wrap_text_pixels(probe_draw, prompt, prompt_font, prompt_width)))
    prompt_line_height = 20
    prompt_height = max(46, max_prompt_lines * prompt_line_height + 16)

    width = margin * 2 + col_count * thumb_size
    height = (
        margin * 2
        + title_height
        + variant_header_height
        + seed_header_height
        + len(roles) * (thumb_size + prompt_height)
    )
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((margin, margin + 4), "5.6 Merge Weight Ratio Sensitivity", fill="#111111", font=title_font)

    x0 = margin
    variant_y = margin + title_height
    seed_y = variant_y + variant_header_height
    grid_y = seed_y + seed_header_height

    for variant_idx, variant in enumerate(variants):
        gx0 = x0 + variant_idx * len(seeds) * thumb_size
        gx1 = gx0 + len(seeds) * thumb_size
        draw.rectangle((gx0, variant_y, gx1, seed_y), fill="#f1f1f1", outline="#c9c9c9")
        draw_centered_text(draw, (gx0, variant_y, gx1, seed_y), MERGE_VARIANT_LABELS.get(variant, variant), variant_font)
        for seed_idx, seed in enumerate(seeds):
            cx0 = gx0 + seed_idx * thumb_size
            cx1 = cx0 + thumb_size
            draw.rectangle((cx0, seed_y, cx1, grid_y), fill="#eeeeee", outline="#c9c9c9")
            draw_centered_text(draw, (cx0, seed_y, cx1, grid_y), str(seed), seed_font)

    current_y = grid_y
    for role in roles:
        image_y = current_y
        prompt_y = image_y + thumb_size

        for variant_idx, variant in enumerate(variants):
            for seed_idx, seed in enumerate(seeds):
                cx0 = x0 + (variant_idx * len(seeds) + seed_idx) * thumb_size
                cx1 = cx0 + thumb_size
                draw.rectangle((cx0, image_y, cx1, image_y + thumb_size), outline="#d0d0d0")
                record = record_lookup.get((variant, role, seed))
                if record is not None:
                    paste_image(sheet, image_path(record), (cx0 + 2, image_y + 2, cx1 - 2, image_y + thumb_size - 2))

        draw.rectangle((x0, prompt_y, x0 + col_count * thumb_size, prompt_y + prompt_height), fill="#f7f7f7", outline="#d0d0d0")
        y = prompt_y + 7
        for line in wrap_text_pixels(draw, prompt_lookup.get(role, ""), prompt_font, prompt_width):
            draw.text((x0 + 8, y), line, fill="#222222", font=prompt_font)
            y += prompt_line_height

        current_y += thumb_size + prompt_height

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

    if any(record["experiment_id"] == "base_vs_single" for record in records):
        base_vs_single_path = output_dir / "base_vs_single_6col.jpg"
        make_base_vs_single_6col_sheet(records, base_vs_single_path, image_path=image_path)
        outputs.append(base_vs_single_path)

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

    if any(record["experiment_id"] == "generalization" for record in records):
        generalization_path = output_dir / "generalization_6col.jpg"
        make_generalization_6col_sheet(records, generalization_path, image_path=image_path)
        outputs.append(generalization_path)

    multitoken = [record for record in records if record["experiment_id"] == "multitoken_merge"]
    if multitoken:
        path = output_dir / "multitoken_merge_prompts.jpg"
        make_multitoken_prompt_sheet(multitoken, path, image_path=image_path)
        outputs.append(path)

    merge_weights = [record for record in records if record["experiment_id"] == "merge_weights"]
    if merge_weights:
        path = output_dir / "merge_weights_8col.jpg"
        make_merge_weights_8col_sheet(merge_weights, path, image_path=image_path)
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
