#!/usr/bin/env python3
"""
Counts Mapillary images per Quebec administrative region.

Each region is split into 0.03-degree grid cells. All cells across all regions
are submitted as concurrent async tasks; CONCURRENCY caps simultaneous HTTP
requests globally. Results are checkpointed after each region completes.

Output: terminal table + data/coverage_results.json

Setup:
    uv add aiohttp   # python-dotenv not needed — direnv loads .env
    echo "MAPILLARY_TOKEN=your_token" > .env

Usage:
    python data/mapillary_coverage.py
    python data/mapillary_coverage.py --fresh   # ignore checkpoint, start over
"""

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import dataclass

import aiohttp

TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not TOKEN:
    sys.exit("MAPILLARY_TOKEN not set — add it to .env")

API_URL = "https://graph.mapillary.com/images"
PAGE_SIZE = 2000  # max images per Mapillary page
REGION_CAP = 50_000  # stop counting beyond this per region
CELL_SIZE = 0.03  # degrees — Mapillary tightened bbox limits; 0.04 fails, 0.03 confirmed working
DELAY = 0.15  # seconds between paginated requests within a cell
CONCURRENCY = 20  # max simultaneous HTTP requests
CHECKPOINT = "data/coverage_results.json"


@dataclass
class Region:
    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float


REGIONS = [
    Region("Bas-Saint-Laurent", -69.5, 47.0, -63.5, 49.0),
    Region("Saguenay-Lac-Saint-Jean", -76.5, 47.5, -69.5, 52.5),
    Region("Capitale-Nationale", -72.5, 46.5, -70.0, 48.5),
    Region("Mauricie", -74.5, 46.0, -72.0, 48.5),
    Region("Estrie", -72.5, 45.0, -71.0, 46.2),
    Region("Montreal", -73.97, 45.40, -73.47, 45.70),
    Region("Outaouais", -77.5, 45.3, -74.5, 47.5),
    Region("Abitibi-Temiscamingue", -80.0, 47.0, -76.0, 49.5),
    Region("Cote-Nord", -70.5, 49.0, -57.0, 52.5),
    Region("Gaspesie-Iles-de-la-Madeleine", -66.5, 47.5, -61.5, 49.5),
    Region("Chaudiere-Appalaches", -71.5, 45.8, -70.0, 47.0),
    Region("Laval", -73.90, 45.52, -73.52, 45.70),
    Region("Lanaudiere", -74.5, 45.6, -72.5, 47.5),
    Region("Laurentides", -76.5, 45.7, -73.5, 47.5),
    Region("Monteregie", -74.5, 45.0, -72.5, 45.6),
    Region("Centre-du-Quebec", -73.0, 45.5, -71.5, 46.5),
    Region("Nord-du-Quebec", -80.0, 49.5, -57.0, 63.0),
]


def load_checkpoint() -> dict[str, dict]:
    if not os.path.exists(CHECKPOINT):
        return {}
    with open(CHECKPOINT, encoding="utf-8") as f:
        return {r["region"]: r for r in json.load(f)}


def save_checkpoint(results: list[dict]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def query_cell(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    lon0: float,
    lat0: float,
    lon1: float,
    lat1: float,
    stop: asyncio.Event | None = None,
) -> int:
    params: dict = {
        "access_token": TOKEN,
        "fields": "id",
        "bbox": f"{lon0},{lat0},{lon1},{lat1}",
        "limit": PAGE_SIZE,
    }
    count = 0

    while True:
        async with semaphore:
            if stop is not None and stop.is_set():
                return 0
            while True:
                try:
                    async with session.get(API_URL, params=params) as resp:
                        if resp.status == 429:
                            print(" [rate limited, waiting 60s]", flush=True)
                            await asyncio.sleep(60)
                            continue
                        if resp.status >= 500:
                            return 0
                        resp.raise_for_status()
                        data = await resp.json()
                except asyncio.TimeoutError:
                    print(" [timeout, retrying in 10s]", flush=True)
                    await asyncio.sleep(10)
                    continue
                except aiohttp.ClientError as e:
                    print(f" [error: {e.__class__.__name__}, retrying in 10s]", flush=True)
                    await asyncio.sleep(10)
                    continue
                break

        images = data.get("data", [])
        count += len(images)

        cursor = data.get("paging", {}).get("cursors", {}).get("after")
        if not cursor or not images:
            return count

        params["after"] = cursor
        await asyncio.sleep(DELAY)


async def count_region(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    region: Region,
    print_lock: asyncio.Lock,
) -> tuple[int, bool]:
    n_cols = math.ceil((region.lon_max - region.lon_min) / CELL_SIZE)
    n_rows = math.ceil((region.lat_max - region.lat_min) / CELL_SIZE)
    total_cells = n_rows * n_cols

    total = 0
    queried = 0
    lock = asyncio.Lock()
    stop = asyncio.Event()

    async def process_cell(lon0: float, lat0: float, lon1: float, lat1: float) -> None:
        nonlocal total, queried
        if stop.is_set():
            return
        n = await query_cell(session, semaphore, lon0, lat0, lon1, lat1, stop)
        async with lock:
            total += n
            queried += 1
            if total >= REGION_CAP:
                stop.set()
            if queried % 100 == 0:
                async with print_lock:
                    print(
                        f"  {region.name}: {queried}/{total_cells} cells, {total:,} images so far",
                        flush=True,
                    )

    await asyncio.gather(
        *[
            asyncio.create_task(
                process_cell(
                    region.lon_min + col * CELL_SIZE,
                    region.lat_min + row * CELL_SIZE,
                    min(region.lon_min + (col + 1) * CELL_SIZE, region.lon_max),
                    min(region.lat_min + (row + 1) * CELL_SIZE, region.lat_max),
                )
            )
            for row in range(n_rows)
            for col in range(n_cols)
        ]
    )
    return total, stop.is_set()


async def process_region(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    region: Region,
    completed: list[dict],
    print_lock: asyncio.Lock,
    checkpoint_lock: asyncio.Lock,
) -> dict:
    count, capped = await count_region(session, semaphore, region, print_lock)
    result = {"region": region.name, "images": count, "capped": capped}

    label = f"{count:,}+" if capped else f"{count:,}"
    async with print_lock:
        print(f"  {region.name:<40} {label:>11}")

    async with checkpoint_lock:
        completed.append(result)
        save_checkpoint(completed)

    return result


async def main_async(fresh: bool) -> None:
    done = {} if fresh else load_checkpoint()
    completed = list(done.values())
    todo = [r for r in REGIONS if r.name not in done]

    print(f"\n{'Region':<42} {'Images':>10}")
    print("-" * 55)

    for result in completed:
        label = f"{result['images']:,}+" if result["capped"] else f"{result['images']:,}"
        print(f"  {result['region']:<40} {label:>11}  (cached)")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    print_lock = asyncio.Lock()
    checkpoint_lock = asyncio.Lock()

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(
            *[
                process_region(session, semaphore, r, completed, print_lock, checkpoint_lock)
                for r in todo
            ]
        )

    print("-" * 55)
    region_order = {r.name: i for i, r in enumerate(REGIONS)}
    completed.sort(key=lambda r: region_order.get(r["region"], 99))
    total = sum(r["images"] for r in completed)
    print(f"  {'Total':<40} {total:>10,}")
    print(f"\nResults saved -> {CHECKPOINT}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="ignore checkpoint and start over")
    args = parser.parse_args()
    asyncio.run(main_async(args.fresh))


if __name__ == "__main__":
    main()
