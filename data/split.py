# data/split.py
"""
Creates a stratified train/test split as a symlink tree and optionally compresses it.

Directory structure produced:
    <output>/
        train/<region>/<image>.jpg -> ../../../../data/images/<region>/<image>.jpg
        test/<region>/<image>.jpg  -> ../../../../data/images/<region>/<image>.jpg

Usage:
    python data/split.py
    python data/split.py --test-ratio 0.15 --seed 42
    python data/split.py --compress /tmp/kaggle-upload/quebec-street-images.tar.gz
"""

from __future__ import annotations

import argparse
import os
import random
import tarfile
from pathlib import Path

IMAGE_DIR = Path("data/images")
OUTPUT_DIR = Path("data/dataset")
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42


def build_split(
    test_ratio: float,
    seed: int,
) -> dict[str, dict[str, list[Path]]]:
    rng = random.Random(seed)
    split: dict[str, dict[str, list[Path]]] = {"train": {}, "test": {}}

    for region_dir in sorted(IMAGE_DIR.iterdir()):
        if not region_dir.is_dir():
            continue
        images = sorted(region_dir.glob("*.jpg"))
        rng.shuffle(images)
        n_test = max(1, round(len(images) * test_ratio))
        split["test"][region_dir.name] = images[:n_test]
        split["train"][region_dir.name] = images[n_test:]

    return split


def create_symlinks(split: dict[str, dict[str, list[Path]]], output_dir: Path) -> None:
    if output_dir.exists():
        for p in output_dir.rglob("*"):
            if p.is_symlink():
                p.unlink()
        for p in sorted(output_dir.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
        output_dir.rmdir()

    for subset, regions in split.items():
        for region, images in regions.items():
            dest_dir = output_dir / subset / region
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                link = dest_dir / img.name
                link.symlink_to(img.resolve())

    print(f"Dataset tree created at {output_dir}/")
    for subset, regions in split.items():
        total = sum(len(imgs) for imgs in regions.values())
        print(f"  {subset}: {total} images across {len(regions)} regions")


def compress(output_dir: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Compressing to {dest} ...")
    with tarfile.open(dest, "w:gz") as tar:
        for path in sorted(output_dir.rglob("*")):
            if path.is_symlink():
                resolved = path.resolve()
                arcname = output_dir.name / path.relative_to(output_dir)
                tar.add(resolved, arcname=str(arcname))
            elif path.is_dir():
                arcname = output_dir.name / path.relative_to(output_dir)
                tar.add(path, arcname=str(arcname), recursive=False)
    size_mb = dest.stat().st_size / 1_000_000
    print(f"Done — {size_mb:.0f} MB written to {dest}")


def print_summary(split: dict[str, dict[str, list[Path]]]) -> None:
    header = f"{'Region':<40} {'Train':>6} {'Test':>6}"
    print(f"\n{header}")
    print("-" * len(header))
    for region in sorted(next(iter(split.values())).keys()):
        n_train = len(split["train"].get(region, []))
        n_test = len(split["test"].get(region, []))
        print(f"{region:<40} {n_train:>6} {n_test:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified train/test split with optional compression.")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="fraction held out for test (default: 0.15)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: 42)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="symlink tree output directory")
    parser.add_argument("--compress", type=Path, metavar="DEST", help="also compress to this .tar.gz path")
    args = parser.parse_args()

    split = build_split(args.test_ratio, args.seed)
    print_summary(split)
    print()
    create_symlinks(split, args.output)

    if args.compress:
        compress(args.output, args.compress)


if __name__ == "__main__":
    main()
