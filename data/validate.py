# data/validate.py
"""
Validates the downloaded image dataset before training.

Checks per region: image count, corrupt files, and spatial spread of coordinates.
Prints a summary table and exits with code 1 if any critical issues are found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_DIR = Path("data/images")
METADATA_FILE = Path("data/sample_metadata.csv")
TARGET = 750
MIN_ACCEPTABLE = 500
MIN_SPREAD_STD = 0.01  # degrees — flags regions where all images cluster in one spot

REGIONS = [
    "Bas-Saint-Laurent",
    "Saguenay-Lac-Saint-Jean",
    "Capitale-Nationale",
    "Mauricie",
    "Estrie",
    "Montreal",
    "Outaouais",
    "Abitibi-Temiscamingue",
    "Cote-Nord",
    "Gaspesie-Iles-de-la-Madeleine",
    "Chaudiere-Appalaches",
    "Laval",
    "Lanaudiere",
    "Laurentides",
    "Monteregie",
    "Centre-du-Quebec",
    "Nord-du-Quebec",
]


def check_corrupt(region_dir: Path) -> list[str]:
    bad = []
    for jpg in region_dir.glob("*.jpg"):
        try:
            with Image.open(jpg) as img:
                img.verify()
        except (UnidentifiedImageError, Exception):
            bad.append(jpg.name)
    return bad


def check_spatial_spread(df: pd.DataFrame, region: str) -> tuple[float, float]:
    sub = df[df["region"] == region].dropna(subset=["lon", "lat"])
    if sub.empty:
        return 0.0, 0.0
    return float(sub["lon"].std()), float(sub["lat"].std())


def main() -> None:
    if not METADATA_FILE.exists():
        sys.exit(f"Metadata file not found: {METADATA_FILE}")

    meta = pd.read_csv(METADATA_FILE)

    print(f"\n{'Region':<40} {'Files':>6} {'Meta':>6} {'Corrupt':>8} {'lon_std':>8} {'lat_std':>8}  Status")
    print("-" * 95)

    issues: list[str] = []

    for region in REGIONS:
        region_dir = IMAGE_DIR / region

        file_count = len(list(region_dir.glob("*.jpg"))) if region_dir.exists() else 0
        meta_count = int((meta["region"] == region).sum())
        corrupt = check_corrupt(region_dir) if region_dir.exists() else []
        lon_std, lat_std = check_spatial_spread(meta, region)

        flags = []
        if file_count < MIN_ACCEPTABLE:
            flags.append("LOW_COUNT")
        if corrupt:
            flags.append(f"{len(corrupt)}_CORRUPT")
        if lon_std < MIN_SPREAD_STD or lat_std < MIN_SPREAD_STD:
            flags.append("CLUSTERED")

        status = ", ".join(flags) if flags else "ok"
        print(
            f"{region:<40} {file_count:>6} {meta_count:>6} {len(corrupt):>8} "
            f"{lon_std:>8.4f} {lat_std:>8.4f}  {status}"
        )

        for name in corrupt:
            issues.append(f"  Corrupt: {region}/{name}")
        if "LOW_COUNT" in flags:
            issues.append(f"  Low count: {region} has only {file_count} images")
        if "CLUSTERED" in flags:
            issues.append(f"  Clustered: {region} lon_std={lon_std:.4f} lat_std={lat_std:.4f}")

    print()

    if issues:
        print("Issues found:")
        for issue in issues:
            print(issue)
        print()
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
