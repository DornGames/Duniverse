#!/usr/bin/env python3
"""Build stable, terrain-aware city and province boundaries for the full map.

All geometry is in the official map's source-pixel coordinate system. City
territories are seeded by the SVG marker centers and kept inside their
province. Named range backbones are curved toward DEM crest signals, while
local crest lines come from anisotropic terrain curvature. Those ridges and
the mapped rivers guide the least-cost partition before fixed-seed,
relief-scaled curve detail is added to each city boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import heapq
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.interpolate import CubicSpline
from shapely.geometry import LineString, MultiLineString, Point, box, mapping, shape
from shapely.ops import linemerge, unary_union


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WIDTH = 2048
HEIGHT = 1448
MAP_EXTENT = (0.0, 0.0, float(WIDTH), float(HEIGHT))
SEED = 20261110
ELEVATION_MIN_M = -260.0
MAX_SEED_SNAP_PX = 24.0
RIVER_CROSSING_COST = 6.0
MIN_BOUNDARY_LENGTH_PX = 2.5
SIMPLIFY_TOLERANCE_PX = 0.12
RIDGE_BARRIER_WEIGHT = 3.6
RIDGE_DETECTION_PERCENTILE = 55.0
RIDGE_MIN_LOCAL_RELIEF_M = 22.0
RIDGE_MIN_SEGMENT_LENGTH_PX = 18.0
BOUNDARY_DETAIL_STEP_PX = 0.65
CITY_MARKER_CLEARANCE_PX = 4.8
BOUNDARY_SUMMIT_MIN_ELEVATION_M = 8000.0
BOUNDARY_SUMMIT_CAPTURE_DISTANCE_PX = 12.0
BOUNDARY_SUMMIT_CITY_CLEARANCE_PX = 10.0


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


def read_boundary_summits() -> list[dict]:
    collection = read_js_json(DATA / "mountain_peaks.js")
    summits = []
    for feature in collection.get("features", []):
        properties = feature.get("properties", {})
        elevation = float(properties.get("elevation_m") or 0.0)
        point = properties.get("source_map_pixel_xy")
        if elevation < BOUNDARY_SUMMIT_MIN_ELEVATION_M or not point or len(point) != 2:
            continue
        x, y = map(float, point)
        if not (0.0 <= x <= WIDTH and 0.0 <= y <= HEIGHT):
            continue
        summits.append({
            "id": feature.get("id", f"summit:{len(summits) + 1}"),
            "name": properties.get("name", "未命名高峰"),
            "elevation_m": elevation,
            "point": Point(x, y),
        })
    return sorted(summits, key=lambda item: (item["name"], item["id"]))


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
            roughness = 1.0 + 0.12 * float(noise[row, col] + noise[next_row, next_col]) * 0.5
            edge_cost = step * (1.0 + 1.0 * relief + ridge_cost) * river_cost * roughness
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


def named_ridge_guides(index_html: str, quadrant) -> list[dict]:
    marker = "const MOUNTAIN_RANGE_SEARCH_DIRECTORY=["
    if marker not in index_html:
        raise ValueError("Could not find the named mountain-range guides in index.html")
    block = index_html.split(marker, 1)[1].split("\n];", 1)[0]
    guides = []
    for line in block.splitlines():
        identity = re.search(r"id:'mountain-range:([^']+)'", line)
        name = re.search(r"name:'([^']+)'", line)
        point_block = re.search(r"points:\[(.*?)\],description:", line)
        if not identity or not name or not point_block:
            continue
        points = [(float(x), float(y)) for x, y in re.findall(
            r"\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]",
            point_block.group(1))]
        if len(points) < 2:
            continue
        clipped = LineString(points).intersection(quadrant)
        for index, part in enumerate(line_parts(clipped)):
            if part.length >= RIDGE_MIN_SEGMENT_LENGTH_PX:
                guides.append({
                    "id": f"named-ridge:{identity.group(1)}:{index + 1}",
                    "name": name.group(1),
                    "kind": "named_range_backbone",
                    "geometry": part,
                })
    return guides


def skeleton_segments(skeleton: np.ndarray, x0: int, y0: int) -> list[LineString]:
    rows, cols = np.nonzero(skeleton)
    pixels = set(zip(cols.tolist(), rows.tolist()))
    adjacency = {}
    for x, y in sorted(pixels):
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                other = (x + dx, y + dy)
                if other not in pixels:
                    continue
                if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                    continue
                neighbors.append(other)
        adjacency[(x, y)] = neighbors

    nodes = sorted(point for point, neighbors in adjacency.items() if len(neighbors) != 2)
    visited = set()
    lines = []
    for start in nodes:
        for neighbor in sorted(adjacency[start]):
            edge = frozenset((start, neighbor))
            if edge in visited:
                continue
            visited.add(edge)
            path = [start, neighbor]
            previous, current = start, neighbor
            while current not in nodes:
                following = sorted(point for point in adjacency[current] if point != previous)
                if not following:
                    break
                nxt = following[0]
                edge = frozenset((current, nxt))
                if edge in visited:
                    break
                visited.add(edge)
                path.append(nxt)
                previous, current = current, nxt
            if len(path) < 3:
                continue
            coords = [(x0 + x + 0.5, y0 + y + 0.5) for x, y in path]
            line = LineString(coords)
            min_x, min_y, max_x, max_y = line.bounds
            ratio = max(max_x - min_x, max_y - min_y) / max(1.0, min(max_x - min_x, max_y - min_y))
            if line.length >= RIDGE_MIN_SEGMENT_LENGTH_PX and ratio >= 1.22:
                lines.append(line)
    return lines


def thin_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce a binary crest corridor to one-pixel centerlines (Zhang-Suen)."""
    image = np.pad(mask.astype(np.uint8), 1)
    while True:
        removed = False
        for phase in (0, 1):
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            center = image[1:-1, 1:-1]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = sum(((neighbors[index] == 0) &
                               (neighbors[(index + 1) % 8] == 1)).astype(np.uint8)
                              for index in range(8))
            common = (center == 1) & (count >= 2) & (count <= 6) & (transitions == 1)
            if phase == 0:
                remove = common & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                remove = common & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            if np.any(remove):
                center[remove] = 0
                removed = True
        if not removed:
            break
    return image[1:-1, 1:-1] > 0


def extract_dem_ridges(elevation: np.ndarray, land_mask: np.ndarray,
                       local_relief: np.ndarray) -> tuple[list[LineString], np.ndarray, np.ndarray]:
    smoothed = ndimage.gaussian_filter(elevation, sigma=1.25, mode="nearest")
    ridge_curvatures = []
    offset = 5
    for dy, dx in ((0, offset), (offset, offset), (offset, 0), (offset, -offset)):
        before = ndimage.shift(smoothed, (dy, dx), order=1, mode="nearest", prefilter=False)
        after = ndimage.shift(smoothed, (-dy, -dx), order=1, mode="nearest", prefilter=False)
        ridge_curvatures.append(smoothed - 0.5 * (before + after))
    curvature = np.stack(ridge_curvatures, axis=0)
    strongest = np.max(curvature, axis=0)
    second = np.partition(curvature, -2, axis=0)[-2]
    anisotropic_crest = np.maximum(strongest - second, 0.0)
    candidate_domain = land_mask & (local_relief >= RIDGE_MIN_LOCAL_RELIEF_M)
    scores = anisotropic_crest[candidate_domain]
    if not scores.size:
        return [], np.zeros_like(land_mask, dtype=bool), anisotropic_crest
    threshold = float(np.percentile(scores, RIDGE_DETECTION_PERCENTILE))
    candidate = candidate_domain & (anisotropic_crest >= threshold)

    x0, y0, x1, y1 = map(int, MAP_EXTENT)
    skeleton = thin_binary_mask(candidate[y0:y1, x0:x1])
    lines = skeleton_segments(skeleton, x0, y0)
    retained = []
    for line in lines:
        clipped = line.intersection(box(x0, y0, x1, y1))
        retained.extend(part for part in line_parts(clipped)
                        if part.length >= RIDGE_MIN_SEGMENT_LENGTH_PX)

    ridge_mask = np.zeros_like(land_mask, dtype=bool)
    brush = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(brush)
    retained = [smooth_ridge_line(line, f"dem-ridge:{index}", local_relief, anisotropic_crest)
                for index, line in enumerate(retained)]
    for line in retained:
        draw.line([(round(x), round(y)) for x, y in line.coords], fill=255, width=2)
    ridge_mask = (np.asarray(brush, dtype=np.uint8) > 0) & land_mask
    return retained, ridge_mask, anisotropic_crest


def rasterize_ridge_guides(guides: list[dict], base_mask: np.ndarray) -> np.ndarray:
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for guide in guides:
        line = guide["geometry"]
        draw.line([(round(x), round(y)) for x, y in line.coords], fill=255, width=3)
    return (np.asarray(image, dtype=np.uint8) > 0) & base_mask


def stable_seed(identifier: str) -> int:
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def smooth_ridge_line(line: LineString, identifier: str, local_relief: np.ndarray,
                      crest_score: np.ndarray) -> LineString:
    if line.length < 4.0 or len(line.coords) < 3:
        return line
    source = np.asarray(line.coords, dtype=np.float64)
    segment_lengths = np.hypot(np.diff(source[:, 0]), np.diff(source[:, 1]))
    source_distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    samples = np.arange(0.0, line.length, 0.75)
    if not len(samples) or samples[-1] < line.length:
        samples = np.append(samples, line.length)
    spline = CubicSpline(source_distances, source, axis=0, bc_type="natural")
    base = spline(samples)
    sample_coords = np.vstack((base[:, 1], base[:, 0]))
    relief = map_coordinates(local_relief, sample_coords, order=1, mode="nearest")
    tangent = np.gradient(base, axis=0)
    norm = np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-6)
    normal_x, normal_y = -tangent[:, 1] / norm, tangent[:, 0] / norm

    # Shift each guide slightly toward the locally stronger crest, then add
    # restrained fixed-seed bends between its source control points.
    offsets = np.arange(-4.0, 4.01, 0.8)
    offset_scores = np.zeros((len(offsets), len(samples)), dtype=np.float32)
    for index, offset in enumerate(offsets):
        coords = np.vstack((base[:, 1] + normal_y * offset,
                            base[:, 0] + normal_x * offset))
        local_score = map_coordinates(crest_score, coords, order=1, mode="nearest")
        local_rel = map_coordinates(local_relief, coords, order=1, mode="nearest")
        offset_scores[index] = local_score + 0.06 * np.maximum(local_rel, 0.0)
    best_offsets = offsets[np.argmax(offset_scores, axis=0)]
    best_offsets = ndimage.gaussian_filter1d(best_offsets, sigma=2.2, mode="nearest")
    best_offsets = np.clip(best_offsets, -2.5, 2.5)

    rng = np.random.default_rng(stable_seed(identifier))
    raw = rng.normal(size=len(samples))
    broad = ndimage.gaussian_filter1d(raw, sigma=9.0, mode="reflect")
    detail = ndimage.gaussian_filter1d(raw, sigma=2.2, mode="reflect")
    broad /= max(float(broad.std()), 1e-6)
    detail /= max(float(detail.std()), 1e-6)
    amplitude = np.clip(0.6 + np.maximum(relief, 0.0) / 600.0, 0.6, 1.8)
    anchor_gap = np.min(np.abs(samples[:, None] - source_distances[None, :]), axis=1)
    taper = np.clip(anchor_gap / 5.0, 0.0, 1.0)
    taper *= np.clip(np.minimum(samples, line.length - samples) / 5.0, 0.0, 1.0)
    displacement = (0.52 * best_offsets + amplitude * (0.72 * broad + 0.42 * detail)) * taper
    curved = np.column_stack((base[:, 0] + normal_x * displacement,
                              base[:, 1] + normal_y * displacement))
    curved[0], curved[-1] = source[0], source[-1]
    return LineString(curved).simplify(0.08, preserve_topology=False)


def curve_boundary(line: LineString, identifier: str, local_relief: np.ndarray,
                   river_distance: np.ndarray, ridge_distance: np.ndarray,
                   city_points: list[tuple[float, float]],
                   summit_controls: list[dict] | None = None) -> LineString:
    if line.length < 3.0:
        return line
    count = max(3, int(math.ceil(line.length / BOUNDARY_DETAIL_STEP_PX)) + 1)
    distances = np.linspace(0.0, line.length, count)
    points = np.asarray([line.interpolate(distance).coords[0] for distance in distances], dtype=np.float64)
    base = points.copy()
    for axis in (0, 1):
        base[:, axis] = ndimage.gaussian_filter1d(base[:, axis], sigma=0.65, mode="nearest")
    base[0], base[-1] = points[0], points[-1]

    tangent = np.gradient(base, axis=0)
    tangent_norm = np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-6)
    normal_x, normal_y = -tangent[:, 1] / tangent_norm, tangent[:, 0] / tangent_norm
    sample_coords = np.vstack((base[:, 1], base[:, 0]))
    relief = map_coordinates(local_relief, sample_coords, order=1, mode="nearest")
    river_gap = map_coordinates(river_distance, sample_coords, order=1, mode="nearest")
    ridge_gap = map_coordinates(ridge_distance, sample_coords, order=1, mode="nearest")
    # Give plains visible but restrained bends; increase amplitude where the
    # local terrain is rough. River corridors keep tighter banks, while ridges
    # retain enough detail to avoid the old, overly smooth Voronoi look.
    amplitude = np.clip(1.8 + np.maximum(relief, 0.0) / 180.0, 1.8, 7.0)
    amplitude *= np.where(river_gap < 3.0, 0.38, np.where(river_gap < 7.0, 0.74, 1.0))
    amplitude *= np.where(ridge_gap < 1.5, 0.82, 1.0)
    city_safety = np.ones(count, dtype=np.float64)
    if city_points:
        city_gap = np.min(np.hypot(base[:, 0, None] - np.asarray(city_points)[:, 0],
                                   base[:, 1, None] - np.asarray(city_points)[:, 1]), axis=1)
        city_safety = np.clip((city_gap - 5.0) / 7.0, 0.0, 1.0)
        amplitude *= city_safety

    rng = np.random.default_rng(stable_seed(identifier))
    raw = rng.normal(size=count)
    def smooth_noise(sigma: float) -> np.ndarray:
        values = ndimage.gaussian_filter1d(raw, sigma=sigma, mode="reflect")
        return values / max(float(values.std()), 1e-6)
    displacement = 1.25 * (0.36 * smooth_noise(11.0) +
                           0.34 * smooth_noise(4.2) +
                           0.30 * smooth_noise(1.2))
    fade = np.minimum(1.0, np.minimum(distances / 5.0, (line.length - distances) / 5.0))
    fade = np.clip(fade, 0.0, 1.0)
    offset = np.clip(displacement * amplitude * fade, -10.5, 10.5)
    if city_points:
        city_array = np.asarray(city_points, dtype=np.float64)
        normal = np.column_stack((normal_x, normal_y))
        scale = np.ones(count, dtype=np.float64)
        for _ in range(8):
            candidates = base + normal * (offset * scale)[:, None]
            candidate_gap = np.min(np.hypot(candidates[:, None, 0] - city_array[None, :, 0],
                                             candidates[:, None, 1] - city_array[None, :, 1]), axis=1)
            scale[candidate_gap < 4.8] *= 0.5
        offset *= scale
    curved = np.column_stack((base[:, 0] + normal_x * offset,
                              base[:, 1] + normal_y * offset))
    curved[0], curved[-1] = points[0], points[-1]
    if summit_controls:
        path_distances = np.concatenate(([0.0], np.cumsum(
            np.hypot(np.diff(curved[:, 0]), np.diff(curved[:, 1])))))
        for summit in summit_controls:
            summit_point: Point = summit["point"]
            if city_points and min(math.hypot(summit_point.x - x, summit_point.y - y)
                                   for x, y in city_points) < BOUNDARY_SUMMIT_CITY_CLEARANCE_PX:
                continue
            current = LineString(curved)
            if current.distance(summit_point) > BOUNDARY_SUMMIT_CAPTURE_DISTANCE_PX + 2.5:
                continue
            station = current.project(summit_point)
            if min(station, current.length - station) < 2.0:
                continue
            nearest = current.interpolate(station)
            displacement = np.array([summit_point.x - nearest.x, summit_point.y - nearest.y])
            sigma = float(np.clip(5.0 + np.linalg.norm(displacement) * 0.72, 5.0, 10.0))
            influence = np.exp(-0.5 * ((path_distances - station) / sigma) ** 2)
            curved += influence[:, None] * displacement[None, :]
            summit_vertex = int(np.argmin(np.abs(path_distances - station)))
            curved[summit_vertex] = (summit_point.x, summit_point.y)
            path_distances = np.concatenate(([0.0], np.cumsum(
                np.hypot(np.diff(curved[:, 0]), np.diff(curved[:, 1])))))
    result = LineString(curved).simplify(0.08, preserve_topology=False)
    return result if result.length >= 2.0 else line


def write_feature_collection(path: Path, collection: dict) -> None:
    entries = []
    for key, value in collection.items():
        if key == "features":
            encoded = "[\n" + ",\n".join(
                "    " + json.dumps(feature, ensure_ascii=False, separators=(",", ":"))
                for feature in value) + "\n  ]"
        else:
            encoded = json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\n  ")
        entries.append(f"  {json.dumps(key, ensure_ascii=False)}: {encoded}")
    payload = "{\n" + ",\n".join(entries) + "\n}\n"
    path.write_bytes(payload.encode("utf-8"))


def main() -> None:
    cities_data = read_js_json(DATA / "relief_city_coordinates.js")["cities"]
    boundary_summits = read_boundary_summits()
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
    summit_provinces = {}
    for summit in boundary_summits:
        matches = [(geom.distance(summit["point"]), code)
                   for code, geom in regions.items() if geom.distance(summit["point"]) <= 1.5]
        if matches:
            summit_provinces[summit["id"]] = min(matches)[1]
    map_extent = box(*MAP_EXTENT)
    active_codes = [code for code, geom in regions.items()
                    if len(cities_by_province.get(code, [])) >= 2 and geom.intersects(map_extent)]

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
    local_relief = np.maximum(elevation - broad_relief, 0.0)
    index_html = (ROOT / "index.html").read_text(encoding="utf-8")
    named_guides = named_ridge_guides(index_html, map_extent)
    dem_ridges, dem_ridge_mask, crest_score = extract_dem_ridges(elevation, land_or_lake, local_relief)
    named_guides = [{**guide, "geometry": smooth_ridge_line(
        guide["geometry"], guide["id"], local_relief, crest_score)} for guide in named_guides]
    named_ridge_mask = rasterize_ridge_guides(named_guides, land_or_lake)
    ridge_mask = dem_ridge_mask | named_ridge_mask
    ridge_distance = ndimage.distance_transform_edt(~ridge_mask)
    ridge_barrier = np.minimum(local_relief / 9000.0, 0.38)
    ridge_barrier += RIDGE_BARRIER_WEIGHT * np.exp(-ridge_distance / 2.2)
    river_distance = ndimage.distance_transform_edt(~river_mask)
    rng = np.random.default_rng(SEED)
    raw_noise = rng.normal(size=(HEIGHT, WIDTH)).astype(np.float32)
    noise = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for weight, sigma in ((0.30, 2.5), (0.45, 8.0), (0.25, 22.0)):
        component = ndimage.gaussian_filter(raw_noise, sigma=sigma, mode="nearest")
        component /= max(float(component.std()), 1e-6)
        noise += weight * component
    del raw_noise
    noise /= max(float(noise.std()), 1e-6)
    noise = np.clip(noise, -2.0, 2.0)

    city_features = []
    ridge_features = []
    for guide in named_guides:
        ridge_features.append({
            "type": "Feature", "id": guide["id"],
            "geometry": mapping(guide["geometry"]),
            "properties": {"name": guide["name"], "ridge_type": guide["kind"],
                           "source": "MOUNTAIN_RANGE_SEARCH_DIRECTORY.points"},
        })
    for index, line in enumerate(dem_ridges, start=1):
        ridge_features.append({
            "type": "Feature", "id": f"dem-ridge:all:{index:04d}",
            "geometry": mapping(line),
            "properties": {"name": "", "ridge_type": "dem_local_crest",
                           "source": "anisotropic_cross_curvature_skeleton"},
        })
    snap_log = []
    total_city_count = 0
    used_summit_controls = []
    for province in sorted(active_codes):
        all_cities = sorted(cities_by_province[province], key=lambda city: (city["name"], city["id"]))
        city_points = [tuple(city["sourcePixelXY"]) for city in all_cities]
        geom = regions[province]
        city_clearance = unary_union([Point(point).buffer(CITY_MARKER_CLEARANCE_PX)
                                      for point in city_points])
        raw_mask, x0, y0 = geometry_mask(geom, WIDTH, HEIGHT)
        x1, y1 = x0 + raw_mask.shape[1], y0 + raw_mask.shape[0]
        local_land = land_or_lake[y0:y1, x0:x1]
        local_river = river_mask[y0:y1, x0:x1]
        domain = raw_mask & (local_land | local_river)
        if not np.any(domain):
            continue

        local_elevation = elevation[y0:y1, x0:x1]
        local_ridge_barrier = ridge_barrier[y0:y1, x0:x1]
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

        owner = terrain_cost_partition(domain, local_elevation, local_river,
                                       local_ridge_barrier, local_noise, seeds)
        owner = smooth_owner(owner, domain, len(all_cities))

        # Keep a small land neighborhood around each settlement in its own cell.
        for label, row, col in seeds:
            yy, xx = np.ogrid[:owner.shape[0], :owner.shape[1]]
            lock = domain & ((yy - row) ** 2 + (xx - col) ** 2 <= 5.5 ** 2)
            owner[lock] = label

        edge_groups = collect_shared_edges(owner, all_cities, x0, y0)
        boundary_sources = []
        for (a, b), segments in edge_groups.items():
            if a >= len(all_cities) or b >= len(all_cities):
                continue
            network = unary_union(MultiLineString(segments))
            merged = linemerge(network) if network.geom_type in ("MultiLineString", "GeometryCollection") else network
            clipped = merged.intersection(map_extent)
            city_a, city_b = all_cities[a], all_cities[b]
            boundary_id = f"city-boundary:{province}:{city_a['id']}:{city_b['id']}"
            for part_index, line in enumerate(line_parts(clipped)):
                if line.length >= MIN_BOUNDARY_LENGTH_PX:
                    boundary_sources.append({
                        "id": boundary_id, "part_index": part_index, "line": line,
                        "city_a": city_a, "city_b": city_b,
                    })

        summit_assignment: dict[int, list[dict]] = defaultdict(list)
        province_summits = [summit for summit in boundary_summits
                            if summit_provinces.get(summit["id"]) == province]
        for summit in province_summits:
            if min(math.hypot(summit["point"].x - x, summit["point"].y - y)
                   for x, y in city_points) < BOUNDARY_SUMMIT_CITY_CLEARANCE_PX:
                continue
            candidates = []
            for source_index, source in enumerate(boundary_sources):
                line = source["line"]
                distance = line.distance(summit["point"])
                station = line.project(summit["point"])
                if (distance <= BOUNDARY_SUMMIT_CAPTURE_DISTANCE_PX
                        and min(station, line.length - station) >= 6.0):
                    candidates.append((distance, source_index))
            if candidates:
                _, nearest_index = min(candidates)
                summit_assignment[nearest_index].append(summit)

        for source_index, source in enumerate(boundary_sources):
            boundary_id = source["id"]
            part_index = source["part_index"]
            line = source["line"]
            city_a, city_b = source["city_a"], source["city_b"]
            controls = summit_assignment.get(source_index, [])
            curve_id = f"{boundary_id}:{part_index}"
            curved = curve_boundary(line, curve_id, local_relief, river_distance,
                                    ridge_distance, city_points, controls)
            clipped_curves = curved.intersection(geom).intersection(map_extent).difference(city_clearance)
            simplify_tolerance = 0.0 if controls else SIMPLIFY_TOLERANCE_PX
            kept = [part.simplify(simplify_tolerance, preserve_topology=False)
                    for part in line_parts(clipped_curves)
                    if part.length >= MIN_BOUNDARY_LENGTH_PX and len(part.coords) >= 2]
            kept = [line for line in kept if line.length >= MIN_BOUNDARY_LENGTH_PX and len(line.coords) >= 2]
            if not kept:
                continue
            geometry = kept[0] if len(kept) == 1 else MultiLineString(kept)
            attached_controls = [summit for summit in controls
                                 if geometry.distance(summit["point"]) <= 0.25]
            city_features.append({
                "type": "Feature",
                "id": curve_id,
                "geometry": mapping(geometry),
                "properties": {
                    "boundary_type": "city",
                    "province_code": province,
                    "city_a_id": city_a["id"], "city_a": city_a["name"],
                    "city_b_id": city_b["id"], "city_b": city_b["name"],
                    "basis": "ridge_network_and_hydrology_least_cost_divide",
                    "summit_controls": [item["name"] for item in attached_controls],
                },
            })
            for summit in attached_controls:
                used_summit_controls.append({
                    "name": summit["name"], "elevation_m": summit["elevation_m"],
                    "province_code": province, "city_boundary_id": curve_id,
                    "capture_distance_px": round(line.distance(summit["point"]), 2),
                })
                ridge_features.append({
                    "type": "Feature",
                    "id": f"boundary-summit-control:{summit['id']}",
                    "geometry": mapping(summit["point"]),
                    "properties": {"name": summit["name"],
                                   "ridge_type": "boundary_summit_control",
                                   "elevation_m": summit["elevation_m"],
                                   "province_code": province,
                                   "city_boundary_id": curve_id},
                })
        total_city_count += len(all_cities)
        print(f"Partitioned province {province}: {len(all_cities)} city anchors, "
              f"{len(boundary_sources)} source divides.", flush=True)

    province_boundary_collection = json.loads(
        (DATA / "province_boundaries.geojson").read_text(encoding="utf-8"))
    province_features = []
    for boundary_index, feature in enumerate(province_boundary_collection.get("features", []), start=1):
        clipped = shape(feature["geometry"]).intersection(map_extent)
        source_properties = feature.get("properties", {})
        for part_index, line in enumerate(line_parts(clipped), start=1):
            if line.length < 3.0:
                continue
            province_features.append({
                "type": "Feature",
                "id": (f"province-boundary:{source_properties.get('region_a', 'NA')}-"
                       f"{source_properties.get('region_b', 'NA')}:{boundary_index}:{part_index}"),
                "geometry": mapping(line),
                "properties": {
                    "boundary_type": "province",
                    "province_a": source_properties.get("region_a"),
                    "province_b": source_properties.get("region_b"),
                    "basis": "shared_province_boundary",
                    "source": source_properties.get("source", "data/province_boundaries.geojson"),
                },
            })
    features = city_features + province_features
    features.sort(key=lambda feature: (
        feature["properties"]["boundary_type"] != "city",
        feature["properties"].get("province_code", ""),
        feature["properties"].get("city_a_id", feature["properties"].get("province_a", "")),
        feature["properties"].get("city_b_id", feature["properties"].get("province_b", "")),
        feature["id"],
    ))
    collection = {
        "type": "FeatureCollection",
        "name": "Duniverse full-map city and province boundaries",
        "coordinate_reference_note": "Official source-map pixel coordinates; origin at top-left; x east, y south.",
        "extent_px": list(MAP_EXTENT),
        "generation": {
            "seed": SEED,
            "city_anchor_source": "data/relief_city_coordinates.js SVG marker centers",
            "province_constraint_source": "data/administrative_regions.geojson",
            "terrain_source": "data/relief_elevation.png",
            "river_source": "data/relief_elevation.js river-water mask",
            "ridge_source": "data/ridge_network.geojson (named mountain-range backbones, anisotropic DEM crest skeletons, and selected high-summit boundary controls)",
            "method": "Across the full map, extract named mountain-range backbones and local DEM ridge crests, then guide province-constrained multi-source least-cost city partitions with ridges and mapped river corridors. High summits close to an existing city divide can become fixed-seed boundary anchors. City lines receive terrain-scaled curve detail; river-adjacent sections remain steadier. Shared province boundaries are included in this layer for close-up views.",
            "ridge_network_feature_count": len(ridge_features),
            "boundary_summit_minimum_elevation_m": BOUNDARY_SUMMIT_MIN_ELEVATION_M,
            "boundary_summit_capture_distance_px": BOUNDARY_SUMMIT_CAPTURE_DISTANCE_PX,
            "boundary_curve_detail": {"seed": SEED, "sample_step_px": BOUNDARY_DETAIL_STEP_PX,
                                      "plain_amplitude_px": 1.8, "maximum_amplitude_px": 7.0,
                                      "maximum_offset_px": 10.5,
                                      "ridge_barrier_weight": RIDGE_BARRIER_WEIGHT,
                                      "city_marker_clearance_px": CITY_MARKER_CLEARANCE_PX,
                                      "partition_cost_noise_weight": 0.12,
                                      "partition_noise_scales_px": [2.5, 8.0, 22.0]},
            "city_stroke": {"color": "#718a84", "opacity": 0.44, "width_px": 1.05, "dash_px": [6.0, 1.8]},
            "province_stroke": {"color": "#596b67", "opacity": 0.64, "width_px": 1.2, "dash_px": [4.2, 3.2]},
        },
        "features": features,
        "audit": {
            "provinces_processed": active_codes,
            "province_boundary_parts": len(province_features),
            "city_sites_total": len(cities_data),
            "city_sites_processed": total_city_count,
            "city_boundary_pairs": len(city_features),
            "boundary_summit_controls": used_summit_controls,
            "snapped_city_sites": snap_log,
            "minimum_boundary_length_px": MIN_BOUNDARY_LENGTH_PX,
        },
    }
    ridge_collection = {
        "type": "FeatureCollection",
        "name": "Duniverse full-map ridge network",
        "coordinate_reference_note": "Official source-map pixel coordinates; origin at top-left; x east, y south.",
        "extent_px": list(MAP_EXTENT),
        "generation": {
            "method": "Full-map named mountain-range backbones are combined with land-constrained, anisotropic cross-curvature crest skeletons from the elevation raster. Closed summit rings and short noisy fragments are discarded; selected high summits near city divides are included as boundary anchors.",
            "detection_percentile": RIDGE_DETECTION_PERCENTILE,
            "minimum_local_relief_m": RIDGE_MIN_LOCAL_RELIEF_M,
            "minimum_segment_length_px": RIDGE_MIN_SEGMENT_LENGTH_PX,
        },
        "features": ridge_features,
        "audit": {"named_backbones": len(named_guides), "dem_crest_segments": len(dem_ridges),
                  "boundary_summit_controls": used_summit_controls},
    }
    geojson_path = DATA / "city_boundaries.geojson"
    js_path = DATA / "city_boundaries.js"
    ridge_geojson_path = DATA / "ridge_network.geojson"
    write_feature_collection(geojson_path, collection)
    js_path.write_bytes(("window.DUNIVERSE_CITY_BOUNDARIES=" +
                         json.dumps(collection, ensure_ascii=False, separators=(",", ":")) + ";\n").encode("utf-8"))
    write_feature_collection(ridge_geojson_path, ridge_collection)
    print(f"Wrote {len(city_features)} city boundaries and {len(province_features)} province boundary parts across {len(active_codes)} provinces: {geojson_path}")
    print(f"Designed {len(ridge_features)} full-map ridge features ({len(named_guides)} named backbones + {len(dem_ridges)} DEM crest segments + {len(used_summit_controls)} summit controls): {ridge_geojson_path}")
    print(f"Processed {total_city_count} city anchors; snapped {len(snap_log)} to nearby province land pixels.")
    print("Boundary summits: " + (", ".join(
        f"{item['name']} ({item['capture_distance_px']:.1f}px)" for item in used_summit_controls) or "none"))
    for item in snap_log:
        print(f"  snapped {item['city']} in {item['province']} by {item['snapPx']:.1f}px")


if __name__ == "__main__":
    main()
