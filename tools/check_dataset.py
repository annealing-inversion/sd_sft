#!/usr/bin/env python3
"""Audit SDXL LoRA datasets arranged as data/<style>/*.{png,jpg,jpeg,webp}+txt."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def iter_task_dirs(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir())


def image_key(path: Path) -> str:
    return path.stem


def audit_task(task_dir: Path) -> bool:
    images = sorted(
        p for p in task_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    captions = sorted(p for p in task_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")

    image_stems = {image_key(p) for p in images}
    caption_stems = {p.stem for p in captions}

    missing_captions = [p for p in images if image_key(p) not in caption_stems]
    orphan_captions = [p for p in captions if p.stem not in image_stems]

    resolution_counts: Counter[str] = Counter()
    small_images: list[tuple[Path, str]] = []
    unreadable_images: list[tuple[Path, str]] = []

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            unreadable_images.append((image_path, str(exc)))
            continue

        resolution = f"{width}x{height}"
        resolution_counts[resolution] += 1
        if width < 1024 or height < 1024:
            small_images.append((image_path, resolution))

    has_errors = bool(missing_captions or orphan_captions or unreadable_images)

    print(f"\n== {task_dir.name} ==")
    print(f"Path: {task_dir}")
    print(f"Images: {len(images)}")
    print(f"Captions: {len(captions)}")
    print(f"Missing same-stem .txt: {len(missing_captions)}")
    print(f"Orphan .txt without image: {len(orphan_captions)}")
    print(f"Unreadable images: {len(unreadable_images)}")
    print(f"Images smaller than 1024x1024: {len(small_images)}")

    if resolution_counts:
        print("Resolutions:")
        for resolution, count in sorted(
            resolution_counts.items(),
            key=lambda item: (
                tuple(int(part) for part in item[0].split("x")),
                item[0],
            ),
        ):
            print(f"  {resolution}: {count}")
    else:
        print("Resolutions: none")

    if missing_captions:
        print("Missing captions:")
        for path in missing_captions:
            print(f"  {path.name}")

    if orphan_captions:
        print("Orphan captions:")
        for path in orphan_captions:
            print(f"  {path.name}")

    if unreadable_images:
        print("Unreadable image files:")
        for path, error in unreadable_images:
            print(f"  {path.name}: {error}")

    if small_images:
        print("Small images:")
        for path, resolution in small_images:
            print(f"  {path.name}: {resolution}")

    status = "FAIL" if has_errors else "OK"
    print(f"Summary: {status} - {task_dir.name}")
    return not has_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check data/<style> SDXL LoRA datasets.")
    parser.add_argument(
        "data_root",
        nargs="?",
        default="data",
        help="Root directory whose first-level subdirectories are independent tasks.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        parser.error(f"data root does not exist: {data_root}")
    if not data_root.is_dir():
        parser.error(f"data root is not a directory: {data_root}")

    task_dirs = iter_task_dirs(data_root)
    if not task_dirs:
        print(f"No first-level task directories found under {data_root}")
        return 1

    print(f"Data root: {data_root}")
    print(f"Task directories: {len(task_dirs)}")

    ok_count = 0
    for task_dir in task_dirs:
        if audit_task(task_dir):
            ok_count += 1

    failed = len(task_dirs) - ok_count
    print("\n== Overall ==")
    print(f"OK tasks: {ok_count}")
    print(f"Tasks with errors: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
