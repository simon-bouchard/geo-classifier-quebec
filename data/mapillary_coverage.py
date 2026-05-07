#!/usr/bin/env python3
"""
Counts Mapillary images per Quebec administrative region.

Each region is split into 0.09-degree grid cells (API limit: 0.010 sq degrees).
Regions are processed in parallel. Results are checkpointed after each region
so a crashed run can be resumed.

Output: terminal table + data/coverage_results.json

Setup:
    uv add requests   # python-dotenv not needed — direnv loads .env
    echo "MAPILLARY_TOKEN=your_token" > .env

Usage:
    python data/mapillary_coverage.py
    python data/mapillary_coverage.py --fresh   # ignore checkpoint, start over
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not TOKEN:
    sys.exit("MAPILLARY_TOKEN not set — add it to .env")

API_URL    = "https://graph.mapillary.com/images"
PAGE_SIZE  = 2000    # max images per Mapillary page
REGION_CAP = 50_000  # stop counting beyond this per region
CELL_SIZE  = 0.09    # degrees — keeps bbox area under the 0.010 sq degree API limit
DELAY      = 0.15    # seconds between requests per thread
WORKERS    = 4
CHECKPOINT = "data/coverage_results.json"


@dataclass
class Region:
    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float


REGIONS = [
    Region("Bas-Saint-Laurent",               -69.5,  47.0,  -63.5,  49.0),
    Region("Saguenay-Lac-Saint-Jean",         -76.5,  47.5,  -69.5,  52.5),
    Region("Capitale-Nationale",              -72.5,  46.5,  -70.0,  48.5),
    Region("Mauricie",                        -74.5,  46.0,  -72.0,  48.5),
    Region("Estrie",                          -72.5,  45.0,  -71.0,  46.2),
    Region("Montreal",                        -73.97, 45.40, -73.47, 45.70),
    Region("Outaouais",                       -77.5,  45.3,  -74.5,  47.5),
    Region("Abitibi-Temiscamingue",           -80.0,  47.0,  -76.0,  49.5),
    Region("Cote-Nord",                       -70.5,  49.0,  -57.0,  52.5),
    Region("Gaspesie-Iles-de-la-Madeleine",  -66.5,  47.5,  -61.5,  49.5),
    Region("Chaudiere-Appalaches",            -71.5,  45.8,  -70.0,  47.0),
    Region("Laval",                           -73.90, 45.52, -73.52, 45.70),
    Region("Lanaudiere",                      -74.5,  45.6,  -72.5,  47.5),
    Region("Laurentides",                     -76.5,  45.7,  -73.5,  47.5),
    Region("Monteregie",                      -74.5,  45.0,  -72.5,  45.6),
    Region("Centre-du-Quebec",                -73.0,  45.5,  -71.5,  46.5),
]

print_lock      = threading.Lock()
checkpoint_lock = threading.Lock()


def load_checkpoint() -> dict[str, dict]:
    if not os.path.exists(CHECKPOINT):
        return {}
    with open(CHECKPOINT, encoding="utf-8") as f:
        return {r["region"]: r for r in json.load(f)}


def save_checkpoint(results: list[dict]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def query_cell(lon0: float, lat0: float, lon1: float, lat1: float) -> int:
    params: dict = {
        "access_token": TOKEN,
        "fields": "id",
        "bbox": f"{lon0},{lat0},{lon1},{lat1}",
        "limit": PAGE_SIZE,
    }
    count = 0

    while True:
        while True:
            try:
                resp = requests.get(API_URL, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                with print_lock:
                    print(f" [timeout: {e.__class__.__name__}, retrying in 10s]", flush=True)
                time.sleep(10)
                continue
            if resp.status_code == 429:
                with print_lock:
                    print(" [rate limited, waiting 60s]", flush=True)
                time.sleep(60)
                continue
            if resp.status_code >= 500:
                return 0
            resp.raise_for_status()
            break

        data   = resp.json()
        images = data.get("data", [])
        count += len(images)

        cursor = data.get("paging", {}).get("cursors", {}).get("after")
        if not cursor or not images:
            return count

        params["after"] = cursor
        time.sleep(DELAY)


def count_region(region: Region) -> tuple[int, bool]:
    n_cols = math.ceil((region.lon_max - region.lon_min) / CELL_SIZE)
    n_rows = math.ceil((region.lat_max - region.lat_min) / CELL_SIZE)
    total  = 0

    total_cells = n_rows * n_cols
    queried     = 0

    for row in range(n_rows):
        for col in range(n_cols):
            lon0 = region.lon_min + col * CELL_SIZE
            lat0 = region.lat_min + row * CELL_SIZE
            lon1 = min(lon0 + CELL_SIZE, region.lon_max)
            lat1 = min(lat0 + CELL_SIZE, region.lat_max)

            total   += query_cell(lon0, lat0, lon1, lat1)
            queried += 1
            time.sleep(DELAY)

            if queried % 100 == 0:
                with print_lock:
                    print(f"  {region.name}: {queried}/{total_cells} cells, {total:,} images so far", flush=True)

            if total >= REGION_CAP:
                return total, True

    return total, False


def process_region(region: Region, completed: list[dict]) -> dict:
    count, capped = count_region(region)
    result = {"region": region.name, "images": count, "capped": capped}

    label = f"{count:,}+" if capped else f"{count:,}"
    with print_lock:
        print(f"  {region.name:<40} {label:>11}")

    with checkpoint_lock:
        completed.append(result)
        save_checkpoint(completed)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="ignore checkpoint and start over")
    args = parser.parse_args()

    done      = {} if args.fresh else load_checkpoint()
    completed = list(done.values())
    todo      = [r for r in REGIONS if r.name not in done]

    print(f"\n{'Region':<42} {'Images':>10}")
    print("-" * 55)

    for result in completed:
        label = f"{result['images']:,}+" if result["capped"] else f"{result['images']:,}"
        print(f"  {result['region']:<40} {label:>11}  (cached)")

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_region, r, completed): r for r in todo}
        for future in as_completed(futures):
            future.result()  # re-raise any exception from the worker

    print("-" * 55)
    region_order = {r.name: i for i, r in enumerate(REGIONS)}
    completed.sort(key=lambda r: region_order.get(r["region"], 99))
    total = sum(r["images"] for r in completed)
    print(f"  {'Total':<40} {total:>10,}")
    print(f"\nResults saved -> {CHECKPOINT}\n")


if __name__ == "__main__":
    main()
