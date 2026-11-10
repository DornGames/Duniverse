#!/usr/bin/env python3
"""Build a stable, terrain-aware first-quadrant city boundary layer.

All geometry is in the official map's source-pixel coordinate system. City
territories are seeded by the SVG marker centers, kept inside their province,
and partitioned with a least-cost flood over the existing elevation and river
masks. Steep local relief and river pixels resist crossing, so the divides
tend to use ridges, river corridors, and a smaller number of irregular
intervening lines instead of straight radial cells.
"""

from __future__ import annotations

import base64
import json
import math
import re
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import LineString, MultiLineString, Point, box, mapping, shape
from shapely.ops import linemerge, unary_union


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WIDTH = 2048
HEIGHT = 1448
QUADRANT = (WIDTH / 2, 0.0, WIDTH, HEIGHT / 2)
QUADRANT_ID = "Q1"
QUADRANT_NAME = "东北象限"
SEED = 20261110
ELEVATION_MIN_M = -260.0
MAX_SEED_SNAP_PX = 24.0
RIVER_CROSSING_COST = 6.0
MIN_BOUNDARY_LENGTH_PX = 8.0
SIMPLIFY_TOLERANCE_PX = 1.4


def read_js_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig").strip()
    if "=" not in text:
        raise ValueError(f"Expected a JavaScript global assignment in {path}")
    return json.loads(text.split("=", 1)[1].strip().rstrip(";"))


def city_province_lookup() -> dict[str, str]:
    text = (DATA / "city_directory.js").read_text(encoding="utf-8-sig")
    lookup: dict[str, str] = {}
    for block in re.split(r"\n\s*\},?\s*\n\s*\{", text):
        city_id = re.search(r'id:\s*"([^"]+)"', block)
        province = re.search(r'parentId:\s*"province:([A-Z]+)"', block)
        if city_id and province:
            lookup[city_id.group(1)] = province.group(1)
    return lookup


def geometry_mask(geometry, width: int, height: int) -> tuple[np.ndarray, int, int]:
    min_x, min_y, max_x, max_y = geometry.bounds
    x0 = max(0, math.floor(min_x))
    y0 = max(0, math.floor(min_y))
    x1 = min(WIDTH, math.ceil(max_x) + 1)
    y1 = min(HEIGHT, math.ceil(max_y) + 1)
    local_width, local_height = max(1, x1 - x0), max(1, y1 - y0)
    image = Image.new("L", (local_width, local_height), 0)
    draw = ImageDraw.Draw(image)
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for polygon in polygons:
        exterior = [(x - x0, y - y0) for x, y in polygon.exterior.coords]
        draw.polygon(exterior, fill=255)
        for interior in polygon.interiors:
            hole = [(x - x0, y - y0) for x, y in interior.coords]
            draw.polygon(hole, fill=0)
    mask = np.asarray(image, dtype=np.uint8) > 0
    return mask[:height, :width], x0, y0


def nearest_mask_cell(mask: np.ndarray, x: float, y: float, x0: int, y0: int,
                      max_distance: float, occupied: set[tuple[int, int]] | None = None):
    local_x, local_y = x - x0, y - y0
    radius = int(math.ceil(max_distance))
    left, right = max(0, math.floor(local_x) - radius), min(mask.shape[1], math.ceil(local_x) + radius + 1)
    top, bottom = max(0, math.floor(local_y) - radius), min(mask.shape[0], math.ceil(local_y) + radius + 1)
    yy, xx = np.nonzero(mask[top:bottom, left:right])
    candidates = []
    for row, col in zip(yy + top, xx + left):
        if occupied and (int(col), int(row)) in occupied:
            continue
        distance = math.hypot(col + x0 - x, row + y0 - y)
        if distance <= max_distance:
            candidates.append((distance, int(row), int(col)))
    if not candidates:
        return None
    _, row, col = min(candidates)
    return row, col


def smooth_owner(owner: np.ndarray, domain: np.ndarray, label_count: int) -> np.ndarray:
    footprint = np.ones((3, 3), dtype=np.uint8)
    smoothed = owner.copy()
    best_count = ndimage.convolve((owner == 0).astype(np.uint8), footprint, mode="constant", cval=0)
    for label in range(1, label_count):
        count = ndimage.convolve((owner == label).astype(np.uint8), footprint, mode="constant", cval=0)
        replace = domain & (count > best_count)
        smoothed[replace] = label
        best_count[replace] = count[replace]
    smoothed[~domain] = -1
    return smoothed


def terrain_cost_partition(domain: np.ndarray, elevation: np.ndarray, river: np.ndarray,
                           ridge: np.ndarray, noise: np.ndarray,
                           seeds: list[tuple[int, int, int]]) -> np.ndarray:
    height, width = domain.shape
    owner = np.full((height, width), -1, dtype=np.int16)
    distance = np.full((height, width), np.inf, dtype=np.float32)
    queue: list[tuple[float, int, int, int]] = []
    for label, row, col in seeds:
        distance[row, col] = 0.0
        owner[row, col] = label
        heapq.heappush(queue, (0.0, label, row, col))

    neighbors = ((-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
                 (0, -1, 1.0), (0, 1, 1.0),
                 (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2)))
    while queue:
        cost, label, row, col = heapq.heappop(queue)
        if cost > float(distance[row, col]) + 1e-5 or owner[row, col] != label:
            continue
        for dy, dx, step in neighbors:
            next_row, next_col = row + dy, col + dx
            if (next_row < 0 or next_col < 0 or next_row >= height or next_col >= width
                    or not domain[next_row, next_col]):
                continue
            relief = min(abs(float(elevation[next_row, next_col] - elevation[row, col])) / 850.0, 2.4)
            ridge_cost = min(max(float(ridge[row, col] + ridge[next_row, next_col]) / 9000.0, 0.0), 0.38)
            river_cost = RIVER_CROSSING_COST if river[row, col] or river[next_row, next_col] else 1.0
            roughness = 1.0 + 0.022 * float(noise[row, col] + noise[next_row, next_col]) * 0.5
            edge_cost = step * (1.0 + 0.72 * relief + ridge_cost) * river_cost * roughness
            next_cost = cost + edge_cost
            old_cost = float(distance[next_row, next_col])
            if next_cost < old_cost - 1e-5 or (abs(next_cost - old_cost) <= 1e-5 and label < owner[next_row, next_col]):
                distance[next_row, next_col] = next_cost
                owner[next_row, next_col] = label
                heapq.heappush(queue, (next_cost, label, next_row, next_col))
    return owner


def collect_shared_edges(owner: np.ndarray, cities: list[dict], x0: int, y0: int) -> dict[tuple[int, int], list[LineString]]:
    grouped: dict[tuple[int, int], list[LineString]] = defaultdict(list)
    left, right = owner[:, :-1], owner[:, 1:]
    rows, cols = np.where((left >= 0) & (right >= 0) & (left != right))
    for row, col in zip(rows, cols):
        a, b = sorted((int(left[row, col]), int(right[row, col])))
        x, y = x0 + col + 0.5, y0 + row
        grouped[(a, b)].append(LineString([(x, y - 0.5), (x, y + 0.5)]))
    upper, lower = owner[:-1, :], owner[1:, :]
    rows, cols = np.where((upper >= 0) & (lower >= 0) & (upper != lower))
    for row, col in zip(rows, cols):
        a, b = sorted((int(upper[row, col]), int(lower[row, col])))
        x, y = x0 + col, y0 + row + 0.5
        grouped[(a, b)].append(LineString([(x - 0.5, y), (x + 0.5, y)]))
    return grouped


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        parts = []
        for item in geometry.geoms:
            parts.extend(line_parts(item))
        return parts
    return []


def main() -> None:
    cities_data = read_js_json(DATA / "relief_city_coordinates.js")["cities"]
    province_by_city = city_province_lookup()
    cities_by_province: dict[str, list[dict]] = defaultdict(list)
    for city in cities_data:
        province = province_by_city.get(city["id"])
        if province:
            city = dict(city)
            city["provinceCode"] = province
            cities_by_province[province].append(city)

    region_features = json.loads((DATA / "administrative_regions.geojson").read_text(encoding="utf-8"))["features"]
    region_parts: dict[str, list] = defaultdict(list)
    for feature in region_features:
        region_parts[feature["properties"]["region_code"]].append(shape(feature["geometry"]))
    regions = {code: unary_union(parts) for code, parts in region_parts.items()}
    quadrant = box(*QUADRANT)
    active_codes = [code for code, geom in regions.items()
                    if len(cities_by_province.get(code, [])) >= 2 and geom.intersects(quadrant)]

    rgba = np.asarray(Image.open(DATA / "relief_elevation.png").convert("RGBA"), dtype=np.uint8)
    if rgba.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError(f"Elevation sampler must be {WIDTH}x{HEIGHT}; received {rgba.shape[1]}x{rgba.shape[0]}")
    metadata = json.loads((DATA / "relief_model_metadata.json").read_text(encoding="utf-8"))
    maximum_m = float(metadata["rasterScaleMaximumM"])
    code = (rgba[:, :, 0].astype(np.uint16) << 8) | rgba[:, :, 1].astype(np.uint16)
    elevation = (ELEVATION_MIN_M + code.astype(np.float32) / 65535.0 * (maximum_m - ELEVATION_MIN_M))
    land_or_lake = rgba[:, :, 3] > 0

    relief_data = read_js_json(DATA / "relief_elevation.js")
    packed_river = np.frombuffer(base64.b64decode(relief_data["riverWaterMaskB64"]), dtype=np.uint8)
    river_mask = np.unpackbits(packed_river, bitorder="big")[:WIDTH * HEIGHT].reshape(HEIGHT, WIDTH).astype(bool)
    river_mask &= ~land_or_lake

    broad_relief = ndimage.gaussian_filter(elevation, sigma=13.0, mode="nearest")
    ridge = np.maximum(elevation - broad_relief, 0.0)
    rng = np.random.default_rng(SEED)
    noise = ndimage.gaussian_filter(rng.normal(size=(HEIGHT, WIDTH)).astype(np.float32), sigma=5.5, mode="nearest")
    noise /= max(float(noise.std()), 1e-6)
    noise = np.clip(noise, -2.0, 2.0)

    features = []
    snap_log = []
    total_city_count = 0
    for province in sorted(active_codes):
        all_cities = sorted(cities_by_province[province], key=lambda city: (city["name"], city["id"]))
        geom = regions[province]
        raw_mask, x0, y0 = geometry_mask(geom, WIDTH, HEIGHT)
        x1, y1 = x0 + raw_mask.shape[1], y0 + raw_mask.shape[0]
        local_land = land_or_lake[y0:y1, x0:x1]
        local_river = river_mask[y0:y1, x0:x1]
        domain = raw_mask & (local_land | local_river)
        if not np.any(domain):
            continue

        local_elevation = elevation[y0:y1, x0:x1]
        local_ridge = ridge[y0:y1, x0:x1]
        local_noise = noise[y0:y1, x0:x1]
        seeds = []
        occupied: set[tuple[int, int]] = set()
        for label, city in enumerate(all_cities):
            x, y = city["sourcePixelXY"]
            snapped = nearest_mask_cell(domain, x, y, x0, y0, MAX_SEED_SNAP_PX, occupied)
            if snapped is None:
                continue
            row, col = snapped
            occupied.add((col, row))
            distance_px = math.hypot((x0 + col) - x, (y0 + row) - y)
            if distance_px > 0.75:
                snap_log.append({"city": city["name"], "province": province, "snapPx": round(distance_px, 2)})
            seeds.append((label, row, col))
        if len(seeds) < 2:
            continue

        # Give every disconnected island group a source from its nearest city,
        # so no land component is silently left without an administrative owner.
        component_labels, component_count = ndimage.label(domain, structure=np.ones((3, 3), dtype=np.uint8))
        seeded_components = {int(component_labels[row, col]) for _, row, col in seeds}
        for component_id in range(1, component_count + 1):
            if component_id in seeded_components:
                continue
            component_rows, component_cols = np.where(component_labels == component_id)
            if not len(component_rows):
                continue
            center_x = float(component_cols.mean() + x0)
            center_y = float(component_rows.mean() + y0)
            nearest_city = min(range(len(all_cities)), key=lambda index:
                               math.hypot(all_cities[index]["sourcePixelXY"][0] - center_x,
                                          all_cities[index]["sourcePixelXY"][1] - center_y))
            x, y = all_cities[nearest_city]["sourcePixelXY"]
            distances = (component_cols + x0 - x) ** 2 + (component_rows + y0 - y) ** 2
            pick = int(np.argmin(distances))
            row, col = int(component_rows[pick]), int(component_cols[pick])
            seeds.append((nearest_city, row, col))
            seeded_components.add(component_id)

        owner = terrain_cost_partition(domain, local_elevation, local_river, local_ridge, local_noise, seeds)
        owner = smooth_owner(owner, domain, len(all_cities))

        # Keep a small land neighborhood around each settlement in its own cell.
        for label, row, col in seeds:
            yy, xx = np.ogrid[:owner.shape[0], :owner.shape[1]]
            lock = domain & ((yy - row) ** 2 + (xx - col) ** 2 <= 3.2 ** 2)
            owner[lock] = label

        edge_groups = collect_shared_edges(owner, all_cities, x0, y0)
        for (a, b), segments in edge_groups.items():
            if a >= len(all_cities) or b >= len(all_cities):
                continue
            network = unary_union(MultiLineString(segments))
            merged = linemerge(network) if network.geom_type in ("MultiLineString", "GeometryCollection") else network
            clipped = merged.intersection(quadrant)
            kept = [line.simplify(SIMPLIFY_TOLERANCE_PX, preserve_topology=False)
                    for line in line_parts(clipped) if line.length >= MIN_BOUNDARY_LENGTH_PX]
            kept = [line for line in kept if line.length >= MIN_BOUNDARY_LENGTH_PX and len(line.coords) >= 2]
            if not kept:
                continue
            geometry = kept[0] if len(kept) == 1 else MultiLineString(kept)
            city_a, city_b = all_cities[a], all_cities[b]
            features.append({
                "type": "Feature",
                "id": f"city-boundary:{province}:{city_a['id']}:{city_b['id']}",
                "geometry": mapping(geometry),
                "properties": {
                    "province_code": province,
                    "city_a_id": city_a["id"], "city_a": city_a["name"],
                    "city_b_id": city_b["id"], "city_b": city_b["name"],
                    "basis": "terrain_and_hydrology_least_cost_divide",
                    "quadrant": QUADRANT_ID,
                },
            })
        total_city_count += len(all_cities)

    features.sort(key=lambda feature: (feature["properties"]["province_code"],
                                       feature["properties"]["city_a_id"],
                                       feature["properties"]["city_b_id"]))
    collection = {
        "type": "FeatureCollection",
        "name": "Duniverse first-quadrant city boundaries",
        "coordinate_reference_note": "Official source-map pixel coordinates; origin at top-left; x east, y south.",
        "quadrant": {"id": QUADRANT_ID, "name": QUADRANT_NAME, "extent_px": list(QUADRANT)},
        "generation": {
            "seed": SEED,
            "city_anchor_source": "data/relief_city_coordinates.js SVG marker centers",
            "province_constraint_source": "data/administrative_regions.geojson",
            "terrain_source": "data/relief_elevation.png",
            "river_source": "data/relief_elevation.js river-water mask",
            "method": "Province-constrained multi-source least-cost partition. Steep local relief and mapped river cells increase crossing cost; a fixed smooth noise field supplies restrained irregularity. The map draws only the Q1 portion; full-province partitions keep later quadrant boundaries consistent.",
            "stroke": {"color": "#718a84", "opacity": 0.36, "width_px": 1.0, "dash_px": [3.2, 4.6]},
        },
        "features": features,
        "audit": {
            "provinces_processed": active_codes,
            "city_sites_processed": total_city_count,
            "boundary_pairs": len(features),
            "snapped_city_sites": snap_log,
            "minimum_boundary_length_px": MIN_BOUNDARY_LENGTH_PX,
        },
    }
    geojson_path = DATA / "city_boundaries_q1.geojson"
    js_path = DATA / "city_boundaries_q1.js"
    geojson_path.write_bytes((json.dumps(collection, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    js_path.write_bytes(("window.DUNIVERSE_CITY_BOUNDARIES_Q1=" +
                         json.dumps(collection, ensure_ascii=False, separators=(",", ":")) + ";\n").encode("utf-8"))
    print(f"Wrote {len(features)} city boundary pairs across {len(active_codes)} intersecting provinces: {geojson_path}")
    print(f"Processed {total_city_count} city anchors; snapped {len(snap_log)} to nearby province land pixels.")
    for item in snap_log:
        print(f"  snapped {item['city']} in {item['province']} by {item['snapPx']:.1f}px")


if __name__ == "__main__":
    main()
