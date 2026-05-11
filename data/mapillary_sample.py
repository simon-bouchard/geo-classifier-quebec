# data/mapillary_sample.py
"""
Downloads a stratified spatial sample of Mapillary images per Quebec administrative region.

Grid cells (0.03°) are generated from each region's bounding box, then filtered to those
whose center falls within the official OSM boundary polygon. Cells are shuffled and each
cell yields up to BATCH_SIZE images. Deduplication is done globally per region by image ID
(preventing exact duplicates) and sequence ID (preventing near-duplicate frames from the
same dashcam pass). All regions are processed concurrently under a shared API semaphore.

Setup:
    uv add aiohttp geopandas osmnx shapely

Usage:
    python data/mapillary_sample.py
    python data/mapillary_sample.py --target 1000
    python data/mapillary_sample.py --fresh    # re-fetch OSM polygons, reset metadata
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not TOKEN:
    sys.exit("MAPILLARY_TOKEN not set — add it to .env")

API_URL = "https://graph.mapillary.com/images"
CELL_SIZE = 0.03
CONCURRENCY = 15  # max simultaneous Mapillary API requests across all regions
WORKERS_PER_REGION = 5
BATCH_SIZE = 50  # images fetched per cell query
OUTPUT_DIR = Path("data/images")
METADATA_FILE = Path("data/sample_metadata.csv")
POLYGON_CACHE = Path("data/region_polygons.gpkg")
API_FIELDS = "id,thumb_1024_url,computed_geometry,captured_at,sequence"
REQUEST_DELAY = 0.05  # seconds between requests per worker


@dataclass
class Region:
    name: str
    nominatim: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    cell_size: float = CELL_SIZE


REGIONS = [
    Region("Bas-Saint-Laurent", "Bas-Saint-Laurent, Québec, Canada", -69.5, 47.0, -63.5, 49.0),
    Region(
        "Saguenay-Lac-Saint-Jean",
        "Saguenay–Lac-Saint-Jean, Québec, Canada",
        -76.5,
        47.5,
        -69.5,
        52.5,
    ),
    Region("Capitale-Nationale", "Capitale-Nationale, Québec, Canada", -72.5, 46.5, -70.0, 48.5),
    Region("Mauricie", "Mauricie, Québec, Canada", -74.5, 46.0, -72.0, 48.5),
    Region("Estrie", "Estrie, Québec, Canada", -72.5, 45.0, -71.0, 46.2),
    Region("Montreal", "Montréal, Québec, Canada", -73.97, 45.40, -73.47, 45.70, cell_size=0.008),
    Region("Outaouais", "Outaouais, Québec, Canada", -77.5, 45.3, -74.5, 47.5),
    Region(
        "Abitibi-Temiscamingue",
        "Abitibi-Témiscamingue, Québec, Canada",
        -80.0,
        47.0,
        -76.0,
        49.5,
    ),
    Region("Cote-Nord", "Côte-Nord, Québec, Canada", -70.5, 49.0, -57.0, 52.5),
    Region(
        "Gaspesie-Iles-de-la-Madeleine",
        "Gaspésie–Îles-de-la-Madeleine, Québec, Canada",
        -66.5,
        47.5,
        -61.5,
        49.5,
    ),
    Region(
        "Chaudiere-Appalaches",
        "Chaudière-Appalaches, Québec, Canada",
        -71.5,
        45.8,
        -70.0,
        47.0,
    ),
    Region("Laval", "Laval, Québec, Canada", -73.90, 45.52, -73.52, 45.70, cell_size=0.008),
    Region("Lanaudiere", "Lanaudière, Québec, Canada", -74.5, 45.6, -72.5, 47.5),
    Region("Laurentides", "Laurentides, Québec, Canada", -76.5, 45.7, -73.5, 47.5),
    Region("Monteregie", "Montérégie, Québec, Canada", -74.5, 45.0, -72.5, 45.6),
    Region("Centre-du-Quebec", "Centre-du-Québec, Québec, Canada", -73.0, 45.5, -71.5, 46.5),
    Region("Nord-du-Quebec", "Nord-du-Québec, Québec, Canada", -80.0, 49.5, -57.0, 63.0),
]


@dataclass
class _RegionState:
    """Mutable state shared across workers for a single region."""

    remaining: int
    seen_ids: set[str]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def load_polygons(fresh: bool) -> dict[str, object]:
    """Load region polygons from local cache or fetch from OSM via Nominatim."""
    if fresh and POLYGON_CACHE.exists():
        POLYGON_CACHE.unlink()

    if POLYGON_CACHE.exists():
        gdf = gpd.read_file(POLYGON_CACHE)
        return {str(row["name"]): row.geometry for _, row in gdf.iterrows()}

    print("Fetching region boundaries from OSM (cached after first run)...")
    polygons: dict[str, object] = {}
    for region in REGIONS:
        print(f"  {region.name}...", end=" ", flush=True)
        try:
            gdf = ox.geocode_to_gdf(region.nominatim)
            polygons[region.name] = gdf.geometry.iloc[0]
            print("ok")
        except Exception as exc:
            sys.exit(f"\nFailed to fetch polygon for {region.name}: {exc}")

    rows = [{"name": name, "geometry": geom} for name, geom in polygons.items()]
    cache_gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    POLYGON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache_gdf.to_file(POLYGON_CACHE, driver="GPKG")
    print(f"Boundaries cached to {POLYGON_CACHE}\n")
    return polygons


def get_valid_cells(region: Region, polygon: object) -> list[tuple[float, float, float, float]]:
    """Return grid cells whose center point falls within the region polygon."""
    cs = region.cell_size
    n_cols = math.ceil((region.lon_max - region.lon_min) / cs)
    n_rows = math.ceil((region.lat_max - region.lat_min) / cs)
    cells = []
    for row in range(n_rows):
        for col in range(n_cols):
            lon0 = region.lon_min + col * cs
            lat0 = region.lat_min + row * cs
            lon1 = min(lon0 + cs, region.lon_max)
            lat1 = min(lat0 + cs, region.lat_max)
            center = Point((lon0 + lon1) / 2, (lat0 + lat1) / 2)
            if polygon.contains(center):  # type: ignore[union-attr]
                cells.append((lon0, lat0, lon1, lat1))
    return cells


def load_existing_ids(region_name: str) -> set[str]:
    """Return image IDs (file stems) already saved to disk for this region."""
    region_dir = OUTPUT_DIR / region_name
    if not region_dir.exists():
        return set()
    return {f.stem for f in region_dir.iterdir() if f.suffix == ".jpg"}


async def fetch_cell_images(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    lon0: float,
    lat0: float,
    lon1: float,
    lat1: float,
) -> list[dict]:
    """Fetch up to BATCH_SIZE images from a single cell. Returns only images with a download URL."""
    params = {
        "access_token": TOKEN,
        "fields": API_FIELDS,
        "bbox": f"{lon0},{lat0},{lon1},{lat1}",
        "limit": BATCH_SIZE,
    }
    async with semaphore:
        await asyncio.sleep(REQUEST_DELAY)
        while True:
            try:
                async with session.get(API_URL, params=params) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(60)
                        continue
                    if resp.status >= 500:
                        return []
                    resp.raise_for_status()
                    data = await resp.json()
            except asyncio.TimeoutError:
                await asyncio.sleep(10)
                continue
            except aiohttp.ClientError:
                return []
            break

    # One image per sequence within this cell — prevents consecutive frames from the
    # same dashcam pass within a 3km area, while allowing the sequence to appear in
    # other cells at different locations.
    seen_seqs: set[str] = set()
    result = []
    for img in data.get("data", []):
        if not img.get("thumb_1024_url"):
            continue
        seq: str = img.get("sequence", "")
        if seq and seq in seen_seqs:
            continue
        if seq:
            seen_seqs.add(seq)
        result.append(img)
    return result


async def download_image(session: aiohttp.ClientSession, url: str, dest: Path) -> bool:
    """Download image bytes to dest. Returns True on success."""
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(await resp.read())
        return True
    except (aiohttp.ClientError, OSError):
        return False


async def region_worker(
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    region_name: str,
    state: _RegionState,
    metadata_lock: asyncio.Lock,
    metadata_writer: csv.DictWriter,
) -> None:
    while True:
        try:
            cell = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        async with state.lock:
            if state.remaining <= 0:
                return

        images = await fetch_cell_images(session, semaphore, *cell)

        for img in images:
            image_id: str = img["id"]

            async with state.lock:
                if state.remaining <= 0:
                    return
                if image_id in state.seen_ids:
                    continue
                state.seen_ids.add(image_id)

            dest = OUTPUT_DIR / region_name / f"{image_id}.jpg"
            ok = await download_image(session, img["thumb_1024_url"], dest)
            if not ok:
                continue

            coords = (img.get("computed_geometry") or {}).get("coordinates", [None, None])

            async with state.lock:
                state.remaining -= 1

            async with metadata_lock:
                metadata_writer.writerow(
                    {
                        "image_id": image_id,
                        "region": region_name,
                        "lon": coords[0],
                        "lat": coords[1],
                        "captured_at": img.get("captured_at"),
                    }
                )


async def sample_region(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    region: Region,
    polygon: object,
    target: int,
    metadata_lock: asyncio.Lock,
    metadata_writer: csv.DictWriter,
    print_lock: asyncio.Lock,
) -> None:
    existing_ids = load_existing_ids(region.name)
    remaining = target - len(existing_ids)

    if remaining <= 0:
        async with print_lock:
            print(f"  {region.name:<40} already complete ({len(existing_ids)} images)")
        return

    cells = get_valid_cells(region, polygon)
    random.shuffle(cells)

    async with print_lock:
        print(
            f"  {region.name:<40} {len(cells):,} valid cells, "
            f"{len(existing_ids)} existing, need {remaining} more"
        )

    queue: asyncio.Queue = asyncio.Queue()
    for cell in cells:
        queue.put_nowait(cell)

    state = _RegionState(
        remaining=remaining,
        seen_ids=set(existing_ids),
    )

    workers = [
        asyncio.create_task(
            region_worker(
                queue, session, semaphore, region.name, state, metadata_lock, metadata_writer
            )
        )
        for _ in range(WORKERS_PER_REGION)
    ]
    await asyncio.gather(*workers)

    saved = remaining - max(state.remaining, 0)
    total = len(existing_ids) + saved

    async with print_lock:
        if total >= target:
            print(f"  {region.name:<40} done ({total} images)")
        else:
            print(f"  {region.name:<40} only {total}/{target} — not enough populated cells")


async def main_async(target: int, fresh: bool) -> None:
    polygons = load_polygons(fresh)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_header = fresh or not METADATA_FILE.exists()
    if fresh and METADATA_FILE.exists():
        METADATA_FILE.unlink()

    metadata_file = open(METADATA_FILE, "a", newline="", encoding="utf-8")
    fieldnames = ["image_id", "region", "lon", "lat", "captured_at"]
    writer = csv.DictWriter(metadata_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    metadata_lock = asyncio.Lock()
    print_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    print(f"\nTarget: {target} images per region — {len(REGIONS)} regions in parallel\n")

    timeout = aiohttp.ClientTimeout(total=60)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(
                *[
                    sample_region(
                        session,
                        semaphore,
                        region,
                        polygons[region.name],
                        target,
                        metadata_lock,
                        writer,
                        print_lock,
                    )
                    for region in REGIONS
                    if region.name in polygons
                ]
            )
    finally:
        metadata_file.close()

    print(f"\nImages  -> {OUTPUT_DIR}/")
    print(f"Metadata -> {METADATA_FILE}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Mapillary images per Quebec region.")
    parser.add_argument("--target", type=int, default=750, help="images per region (default: 750)")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="re-fetch OSM polygons and reset metadata (does not delete downloaded images)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.target, args.fresh))


if __name__ == "__main__":
    main()
