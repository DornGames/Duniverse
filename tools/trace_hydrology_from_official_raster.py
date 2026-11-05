"""Re-trace river centerlines from the official high-resolution water raster.

The reference raster is kept outside this repository. Pass it explicitly with
``--reference-raster``. Coordinates in the project GeoJSON are map-image pixels
with a top-left origin; the 12x reference uses X=12*x+5.5, Y=12*y+5.5.

Run without ``--write`` to review the extracted layers first. ``--preview-dir``
stores the result without changing project files. Add ``--write`` only after
reviewing that result to replace the river and cross-border river layers and
regenerate their render groups.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CROSS_BORDER_DOMAIN_SEED = ROOT / "tools" / "trace_sources" / "cross_border_rivers_seed.geojson"
SCALE = 12.0
OFFSET = 5.5
WATER_CLOSE_KERNEL = 3
SKELETON_TILE_SIZE = 1536
SKELETON_TILE_OVERLAP = 72
SIMPLIFY_TOLERANCE = 0.08
MIN_OUTPUT_LENGTH = 0.28
CANAL_TICK_MAX_SPUR_LENGTH = 5.8
CANAL_TICK_MIN_SPACING = 7.0
CANAL_TICK_MAX_SPACING = 22.0
CANAL_TICK_MAX_AXIS_DEVIATION = 30.0
CANAL_TICK_MAX_PERPENDICULAR_DEVIATION = 35.0
CANAL_TICK_GROUP_RADIUS = 45.0
CANAL_TICK_MIN_GROUP_SIZE = 3


def read_fc(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_parts(geometry: Any) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in line_parts(item)]
    return []


def polygon_parts(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in polygon_parts(item)]
    return []


def highres_points(coords: Iterable[Iterable[float]]) -> np.ndarray:
    points = np.asarray([[point[0], point[1]] for point in coords], dtype=np.float64)
    return np.rint(points * SCALE + OFFSET).astype(np.int32)


def fill_polygon(canvas: np.ndarray, polygon: Polygon, value: int) -> None:
    exterior = highres_points(polygon.exterior.coords)
    cv2.fillPoly(canvas, [exterior], value, lineType=cv2.LINE_8)
    for ring in polygon.interiors:
        hole = highres_points(ring.coords)
        cv2.fillPoly(canvas, [hole], 0 if value else 255, lineType=cv2.LINE_8)


def draw_geometry(canvas: np.ndarray, geometry: Any, value: int = 255) -> None:
    for polygon in polygon_parts(geometry):
        fill_polygon(canvas, polygon, value)


def draw_line(canvas: np.ndarray, geometry: Any, width: int, value: int = 255) -> None:
    for line in line_parts(geometry):
        points = highres_points(line.coords)
        if len(points) >= 2:
            cv2.polylines(canvas, [points], False, value, width, cv2.LINE_8)


def build_domain(shape_hw: tuple[int, int], outline_fc: dict[str, Any],
                 cross_border_fc: dict[str, Any], seasonal_fc: dict[str, Any],
                 ocean_fc: dict[str, Any]) -> np.ndarray:
    outline = np.zeros(shape_hw, np.uint8)
    outside_routes = np.zeros(shape_hw, np.uint8)
    ocean = np.zeros(shape_hw, np.uint8)
    for feature in outline_fc["features"]:
        draw_geometry(outline, shape(feature["geometry"]))
    for layer in (cross_border_fc, seasonal_fc):
        for feature in layer["features"]:
            route = shape(feature["geometry"]).buffer(16.0, cap_style="round", join_style="round")
            draw_geometry(outside_routes, route)
    for feature in ocean_fc["features"]:
        draw_geometry(ocean, shape(feature["geometry"]))
    domain = cv2.bitwise_or(outline, outside_routes)
    domain[ocean > 0] = 0
    return domain


def official_water_mask(image: np.ndarray, domain: np.ndarray) -> np.ndarray:
    height, width = domain.shape
    mask = np.zeros((height, width), np.uint8)
    for y0 in range(0, height, 1024):
        y1 = min(height, y0 + 1024)
        for x0 in range(0, width, 1024):
            x1 = min(width, x0 + 1024)
            tile = image[y0:y1, x0:x1]
            blue, green, red = tile[:, :, 0], tile[:, :, 1], tile[:, :, 2]
            blue_red = cv2.subtract(blue, red)
            green_red = cv2.subtract(green, red)
            candidate = (
                (blue_red > 20) & (green_red > 12) & (red < 245) & (blue > 170)
            )
            candidate &= domain[y0:y1, x0:x1] > 0
            mask[y0:y1, x0:x1] = candidate.astype(np.uint8) * 255
    return mask


def remove_filled_lakes(mask: np.ndarray, lakes_fc: dict[str, Any]) -> None:
    lake_mask = np.zeros_like(mask)
    for feature in lakes_fc["features"]:
        geometry = shape(feature["geometry"]).buffer(0.08)
        draw_geometry(lake_mask, geometry)
    mask[lake_mask > 0] = 0


def remove_seasonal_strokes(mask: np.ndarray, seasonal_fc: dict[str, Any]) -> None:
    seasonal_mask = np.zeros_like(mask)
    for feature in seasonal_fc["features"]:
        # Exclude only the dashed symbol and its antialiasing halo. The 16px
        # outer corridor in build_domain limits where seasonal rivers are read.
        draw_geometry(seasonal_mask, shape(feature["geometry"]).buffer(2.5, cap_style="round"))
    mask[seasonal_mask > 0] = 0


def remove_canal_ticks(mask: np.ndarray, canals_fc: dict[str, Any]) -> None:
    tick_mask = np.zeros_like(mask)
    for feature in canals_fc["features"]:
        geometry = shape(feature["geometry"])
        if feature.get("properties", {}).get("symbol_part") == "ruler_tick":
            coords = list(geometry.coords)
            if len(coords) < 2:
                continue
            # Leave the short stem nearest the channel. It is removed as a
            # residual symbol branch after skeletonizing; masking only the
            # outer tail prevents this perpendicular mark from cutting the
            # canal centerline at its anchor.
            tail_start = geometry.interpolate(min(1.25, geometry.length * 0.6))
            tail = LineString([tuple(tail_start.coords[0]), *coords[1:]])
            draw_line(tick_mask, tail, width=18)
    mask[tick_mask > 0] = 0


def skeletonize_tiled(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    skeleton = np.zeros((height, width), bool)
    tile = SKELETON_TILE_SIZE
    overlap = SKELETON_TILE_OVERLAP
    for y0 in range(0, height, tile):
        y1 = min(height, y0 + tile)
        for x0 in range(0, width, tile):
            x1 = min(width, x0 + tile)
            ex0, ey0 = max(0, x0 - overlap), max(0, y0 - overlap)
            ex1, ey1 = min(width, x1 + overlap), min(height, y1 + overlap)
            local = skeletonize(mask[ey0:ey1, ex0:ex1] > 0)
            core = local[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]
            skeleton[y0:y1, x0:x1] = core
    return skeleton


def build_skeleton_adjacency(points: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in points:
        neighbors = []
        for dy, dx in offsets:
            other = (y + dy, x + dx)
            if other not in points:
                continue
            if dy and dx and ((y, x + dx) in points or (y + dy, x) in points):
                continue
            neighbors.append(other)
        adjacency[(y, x)] = neighbors
    return adjacency


def axis_angle(vector: tuple[float, float]) -> float:
    """Return an undirected line angle in degrees (0 <= angle < 180)."""
    return math.degrees(math.atan2(vector[1], vector[0])) % 180.0


def angular_difference(first: float, second: float) -> float:
    return abs((first - second + 90.0) % 180.0 - 90.0)


def terminal_spurs(adjacency: dict[tuple[int, int], list[tuple[int, int]]]) -> list[dict[str, Any]]:
    spurs = []
    for tip, neighbors in adjacency.items():
        if len(neighbors) != 1:
            continue
        path = [tip]
        previous = tip
        current = neighbors[0]
        path.append(current)
        while len(adjacency[current]) == 2:
            following = next((node for node in adjacency[current] if node != previous), None)
            if following is None:
                break
            path.append(following)
            previous, current = current, following
        if len(adjacency[current]) < 3:
            continue
        length = sum(
            math.hypot(second[1] - first[1], second[0] - first[0])
            for first, second in zip(path, path[1:])
        ) / SCALE
        vector = (tip[1] - current[1], tip[0] - current[0])
        spurs.append({
            "tip": tip,
            "junction": current,
            "path": path,
            "length_px": length,
            "angle": axis_angle(vector),
        })
    return spurs


def prune_repeated_tick_spurs(points: set[tuple[int, int]], ticks: list[tuple[LineString, LineString]]) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    """Remove known tick remnants and repeated, regularly spaced unknown tick spurs."""
    adjacency = build_skeleton_adjacency(points)
    spurs = terminal_spurs(adjacency)
    candidates = [index for index, spur in enumerate(spurs)
                  if spur["length_px"] <= CANAL_TICK_MAX_SPUR_LENGTH]
    known_symbol_indices: set[int] = set()
    known_symbol_tick_ids: set[int] = set()
    known_symbol_lengths = []
    spur_geometries = {}
    for index in candidates:
        coords = [((x - OFFSET) / SCALE, (y - OFFSET) / SCALE)
                  for y, x in spurs[index]["path"]]
        spur_geometries[index] = LineString(coords)
        spur = spur_geometries[index]
        junction = Point((spurs[index]["junction"][1] - OFFSET) / SCALE,
                         (spurs[index]["junction"][0] - OFFSET) / SCALE)
        for tick_id, (tick, centerline) in enumerate(ticks):
            if centerline.distance(junction) > 3.0:
                continue
            if angular_difference(spurs[index]["angle"], axis_angle((
                    tick.coords[-1][0] - tick.coords[0][0],
                    tick.coords[-1][1] - tick.coords[0][1]))) > 28.0:
                continue
            if spur.distance(tick) <= 1.25:
                known_symbol_indices.add(index)
                known_symbol_tick_ids.add(tick_id)
                known_symbol_lengths.append(spurs[index]["length_px"])
                break

    unknown_candidates = [index for index in candidates if index not in known_symbol_indices]
    neighbors: dict[int, set[int]] = {index: set() for index in unknown_candidates}

    for position, first_index in enumerate(unknown_candidates):
        first = spurs[first_index]
        first_y, first_x = first["junction"]
        for second_index in unknown_candidates[position + 1:]:
            second = spurs[second_index]
            second_y, second_x = second["junction"]
            dx, dy = second_x - first_x, second_y - first_y
            spacing = math.hypot(dx, dy) / SCALE
            if not CANAL_TICK_MIN_SPACING <= spacing <= CANAL_TICK_MAX_SPACING:
                continue
            if angular_difference(first["angle"], second["angle"]) > CANAL_TICK_MAX_AXIS_DEVIATION:
                continue
            trunk_angle = axis_angle((dx, dy))
            if (abs(90.0 - angular_difference(first["angle"], trunk_angle))
                    > CANAL_TICK_MAX_PERPENDICULAR_DEVIATION):
                continue
            if (abs(90.0 - angular_difference(second["angle"], trunk_angle))
                    > CANAL_TICK_MAX_PERPENDICULAR_DEVIATION):
                continue
            neighbors[first_index].add(second_index)
            neighbors[second_index].add(first_index)

    # A spur is classified only if it belongs to a local group of at least
    # three short branches. Single and paired tributaries stay in the river.
    repeated_indices: set[int] = set()
    visited: set[int] = set()
    clusters = []
    for start in unknown_candidates:
        if start in visited:
            continue
        stack = [start]
        component = []
        visited.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for other in neighbors[current]:
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        if len(component) < CANAL_TICK_MIN_GROUP_SIZE:
            continue
        for index in component:
            y, x = spurs[index]["junction"]
            local_count = sum(
                math.hypot(spurs[other]["junction"][1] - x,
                           spurs[other]["junction"][0] - y) / SCALE <= CANAL_TICK_GROUP_RADIUS
                for other in component
            )
            if local_count >= CANAL_TICK_MIN_GROUP_SIZE:
                repeated_indices.add(index)
        if any(index in repeated_indices for index in component):
            clusters.append(len(component))

    removed_indices = known_symbol_indices | repeated_indices
    remove_points: set[tuple[int, int]] = set()
    for index in removed_indices:
        # Keep the junction itself so the mapped river remains continuous.
        remove_points.update(spurs[index]["path"][:-1])
    if remove_points:
        points.difference_update(remove_points)

    lengths = sorted(spurs[index]["length_px"] for index in candidates)
    summary = {
        "terminal_side_spurs_before_filter": len(spurs),
        "short_spurs_at_or_below_threshold": len(candidates),
        "known_tick_remnants_matched": len(known_symbol_indices),
        "known_ticks_with_a_matched_remnant": len(known_symbol_tick_ids),
        "known_tick_remnant_length_min_median_max_px": (
            [round(min(known_symbol_lengths), 3),
             round(statistics.median(known_symbol_lengths), 3),
             round(max(known_symbol_lengths), 3)]
            if known_symbol_lengths else []
        ),
        "repeated_tick_groups": len(clusters),
        "repeated_tick_group_sizes": sorted(clusters, reverse=True),
        "removed_repeated_unknown_tick_spurs": len(repeated_indices),
        "removed_total_tick_spurs": len(removed_indices),
        "retained_unknown_short_spurs": len(unknown_candidates) - len(repeated_indices),
        "retained_short_spurs": len(candidates) - len(removed_indices),
        "short_spur_length_min_median_max_px": (
            [round(lengths[0], 3), round(lengths[(len(lengths) - 1) // 2], 3), round(lengths[-1], 3)]
            if lengths else []
        ),
        "filter": {
            "maximum_spur_length_px": CANAL_TICK_MAX_SPUR_LENGTH,
            "threshold_basis": (
                "5.8 px rounds up the measured 5.762 px maximum among 52 known canal ticks; "
                "matched residual spurs measured 0.652-5.345 px with a 3.655 px median."
            ),
            "minimum_group_size": CANAL_TICK_MIN_GROUP_SIZE,
            "junction_spacing_px": [CANAL_TICK_MIN_SPACING, CANAL_TICK_MAX_SPACING],
            "maximum_arm_axis_difference_degrees": CANAL_TICK_MAX_AXIS_DEVIATION,
            "maximum_perpendicular_deviation_degrees": CANAL_TICK_MAX_PERPENDICULAR_DEVIATION,
            "maximum_local_group_radius_px": CANAL_TICK_GROUP_RADIUS,
            "known_tick_matching": "arm <= 1.25 px from a digitized tick, within 28 degrees of its axis, and junction <= 3 px from its canal centerline",
        },
    }
    return points, summary


def canal_tick_calibration(canals_fc: dict[str, Any]) -> dict[str, Any]:
    centerlines = {
        feature.get("properties", {}).get("canal_id"): shape(feature["geometry"])
        for feature in canals_fc["features"]
        if feature.get("properties", {}).get("symbol_part") == "centerline"
    }
    ticks: dict[Any, list[tuple[float, float]]] = {}
    for feature in canals_fc["features"]:
        properties = feature.get("properties", {})
        if properties.get("symbol_part") != "ruler_tick":
            continue
        canal_id = properties.get("canal_id")
        tick = shape(feature["geometry"])
        centerline = centerlines.get(canal_id)
        if centerline is None:
            continue
        ticks.setdefault(canal_id, []).append((centerline.project(tick.centroid), tick.length))

    def range_summary(values: list[float]) -> list[float]:
        return [round(min(values), 3), round(statistics.median(values), 3), round(max(values), 3)]

    by_canal = []
    all_lengths = []
    all_spacings = []
    for canal_id, measurements in sorted(ticks.items(), key=lambda item: str(item[0])):
        positions = sorted(position for position, _ in measurements)
        lengths = [length for _, length in measurements]
        spacings = [right - left for left, right in zip(positions, positions[1:])]
        all_lengths.extend(lengths)
        all_spacings.extend(spacings)
        by_canal.append({
            "canal_id": canal_id,
            "tick_count": len(lengths),
            "full_tick_length_min_median_max_px": range_summary(lengths),
            "along_canal_spacing_min_median_max_px": range_summary(spacings) if spacings else [],
        })
    return {
        "reference_tick_count": len(all_lengths),
        "full_tick_length_min_median_max_px": range_summary(all_lengths) if all_lengths else [],
        "along_canal_spacing_min_median_max_px": range_summary(all_spacings) if all_spacings else [],
        "by_canal": by_canal,
        "interpretation": (
            "The short-spur limit is calibrated against manually traced known ruler ticks; "
            "repeated spacing and orientation are required before any spur is removed."
        ),
    }


def trace_skeleton(points: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    adjacency = build_skeleton_adjacency(points)

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a < b else (b, a)

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths = []

    def walk(start: tuple[int, int], nxt: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start, nxt]
        visited.add(edge_key(start, nxt))
        previous, current = start, nxt
        while len(adjacency[current]) == 2:
            options = [point for point in adjacency[current] if point != previous]
            if not options:
                break
            following = options[0]
            edge = edge_key(current, following)
            if edge in visited:
                break
            visited.add(edge)
            path.append(following)
            previous, current = current, following
            if current == start:
                break
        return path

    for point, neighbors in adjacency.items():
        if len(neighbors) == 2:
            continue
        for neighbor in neighbors:
            edge = edge_key(point, neighbor)
            if edge not in visited:
                paths.append(walk(point, neighbor))
    for point, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = edge_key(point, neighbor)
            if edge not in visited:
                paths.append(walk(point, neighbor))
    return paths


def is_symbol_only(line: LineString, canals: list[LineString], ticks: list[LineString],
                   lake_boundaries: list[Any]) -> bool:
    if line.length <= 0:
        return True
    for tick in ticks:
        if line.length <= max(3.0, tick.length * 1.6) and line.distance(tick.buffer(0.2)) < 0.02:
            overlap = line.intersection(tick.buffer(0.2)).length / line.length
            if overlap >= 0.7:
                return True
    for canal in canals:
        if line.length <= max(3.0, canal.length * 1.15) and line.distance(canal.buffer(0.12)) < 0.02:
            overlap = line.intersection(canal.buffer(0.12)).length / line.length
            if overlap >= 0.85:
                return True
    for boundary in lake_boundaries:
        overlap = line.intersection(boundary.buffer(0.75)).length / line.length
        if overlap >= 0.7:
            return True
    return False


def write_fc(path: Path, name: str, features: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "name": name,
        "coordinate_reference_note": (
            "Input map image pixels; top-left origin; x right; y down; not georeferenced. "
            "Traced from the official 12x pure-river raster with X=12*x+5.5; Y=12*y+5.5."
        ),
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def feature_for_line(line: LineString, segment_id: int, territory: str) -> dict[str, Any]:
    properties = {
        "segment_id": segment_id,
        "length_px": round(float(line.length), 3),
        "source": "official_12x_river_raster_centerline_trace",
        "review_status": "traced_from_official_high_resolution_river_raster",
        "territory": territory,
        "raster_scale": 12,
        "geometry_simplification_tolerance_px": SIMPLIFY_TOLERANCE,
    }
    coordinates = [[round(float(x), 3), round(float(y), 3)] for x, y in line.coords]
    return {"type": "Feature", "properties": properties,
            "geometry": {"type": "LineString", "coordinates": coordinates}}


def trace_layers(reference_raster: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    image = cv2.imread(str(reference_raster), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read official raster: {reference_raster}")
    height, width = image.shape[:2]
    expected = (round(1448 * SCALE), round(2048 * SCALE))
    if abs(width - expected[1]) > 8 or abs(height - expected[0]) > 8:
        raise ValueError(f"Unexpected official 12x raster size {width}x{height}; expected near {expected[1]}x{expected[0]}")

    outline_fc = read_fc(DATA / "national_outline.geojson")
    # Keep the extraction corridor fixed across runs; the output layer replaces
    # cross_border_rivers.geojson, so it cannot also serve as its own input.
    cross_border_seed = CROSS_BORDER_DOMAIN_SEED if CROSS_BORDER_DOMAIN_SEED.exists() else DATA / "cross_border_rivers.geojson"
    cross_border_fc = read_fc(cross_border_seed)
    seasonal_fc = read_fc(DATA / "seasonal_rivers.geojson")
    ocean_fc = read_fc(DATA / "ocean_water.geojson")
    lakes_fc = read_fc(DATA / "lakes.geojson")
    canals_fc = read_fc(DATA / "canals.geojson")

    domain = build_domain((height, width), outline_fc, cross_border_fc, seasonal_fc, ocean_fc)
    water = official_water_mask(image, domain)
    del image, domain
    water_pixels = int(cv2.countNonZero(water))
    remove_filled_lakes(water, lakes_fc)
    remove_seasonal_strokes(water, seasonal_fc)
    remove_canal_ticks(water, canals_fc)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (WATER_CLOSE_KERNEL, WATER_CLOSE_KERNEL))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel)
    skeleton = skeletonize_tiled(water)
    del water

    ys, xs = np.nonzero(skeleton)
    skeleton_points = {(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())}
    del skeleton, ys, xs
    canal_centerlines = {
        feature.get("properties", {}).get("canal_id"): shape(feature["geometry"])
        for feature in canals_fc["features"]
        if feature.get("properties", {}).get("symbol_part") == "centerline"
    }
    known_tick_pairs = [
        (shape(feature["geometry"]), canal_centerlines[feature.get("properties", {}).get("canal_id")])
        for feature in canals_fc["features"]
        if feature.get("properties", {}).get("symbol_part") == "ruler_tick"
        and feature.get("properties", {}).get("canal_id") in canal_centerlines
    ]
    skeleton_points, spur_filter_summary = prune_repeated_tick_spurs(skeleton_points, known_tick_pairs)
    raw_paths = trace_skeleton(skeleton_points)
    del skeleton_points

    outline = unary_union([shape(feature["geometry"]) for feature in outline_fc["features"]])
    canals = [shape(feature["geometry"]) for feature in canals_fc["features"]
              if feature.get("properties", {}).get("symbol_part") == "centerline"]
    ticks = [shape(feature["geometry"]) for feature in canals_fc["features"]
             if feature.get("properties", {}).get("symbol_part") == "ruler_tick"]
    lake_boundaries = [shape(feature["geometry"]).boundary for feature in lakes_fc["features"]]
    river_lines: list[LineString] = []
    cross_border_lines: list[LineString] = []
    skipped_symbol_paths = 0
    skipped_short_parts = 0
    max_deviation = 0.0

    for path in raw_paths:
        coords = [((x - OFFSET) / SCALE, (y - OFFSET) / SCALE) for y, x in path]
        if len(coords) < 2:
            continue
        source_line = LineString(coords)
        if source_line.length < MIN_OUTPUT_LENGTH:
            skipped_short_parts += 1
            continue
        if is_symbol_only(source_line, canals, ticks, lake_boundaries):
            skipped_symbol_paths += 1
            continue
        simplified = source_line.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        if simplified.geom_type != "LineString" or len(simplified.coords) < 2:
            simplified = source_line
        max_deviation = max(max_deviation, float(source_line.hausdorff_distance(simplified)))
        inside = simplified.intersection(outline)
        outside = simplified.difference(outline)
        for part in line_parts(inside):
            if part.length >= MIN_OUTPUT_LENGTH:
                river_lines.append(part)
            else:
                skipped_short_parts += 1
        for part in line_parts(outside):
            if part.length >= MIN_OUTPUT_LENGTH:
                cross_border_lines.append(part)
            else:
                skipped_short_parts += 1

    # Deterministic ordering makes the traced GeoJSON reproducible across runs.
    order = lambda line: (round(line.bounds[1], 3), round(line.bounds[0], 3), round(line.length, 3))
    river_lines.sort(key=order)
    cross_border_lines.sort(key=order)
    rivers = [feature_for_line(line, i, "national") for i, line in enumerate(river_lines, 1)]
    cross_border = [feature_for_line(line, i, "neighboring_country") for i, line in enumerate(cross_border_lines, 1)]
    summary = {
        "reference_raster": str(reference_raster),
        "cross_border_domain_seed": str(cross_border_seed.relative_to(ROOT)),
        "coordinate_mapping": "X=12*x+5.5; Y=12*y+5.5",
        "raster_size_px": [width, height],
        "official_water_mask_pixels_before_exclusions": water_pixels,
        "raw_skeleton_paths": len(raw_paths),
        "stored_river_paths": len(rivers),
        "stored_cross_border_paths": len(cross_border),
        "total_stored_paths": len(rivers) + len(cross_border),
        "rivers_total_length_px": round(sum(f["properties"]["length_px"] for f in rivers), 3),
        "cross_border_total_length_px": round(sum(f["properties"]["length_px"] for f in cross_border), 3),
        "skipped_symbol_paths": skipped_symbol_paths,
        "skipped_short_parts": skipped_short_parts,
        "canal_tick_measurements": canal_tick_calibration(canals_fc),
        "short_branch_classification": spur_filter_summary,
        "canal_tick_features_removed_before_skeleton": sum(
            feature.get("properties", {}).get("symbol_part") == "ruler_tick"
            for feature in canals_fc["features"]
        ),
        "lake_shoreline_paths_excluded": True,
        "simplification_tolerance_px": SIMPLIFY_TOLERANCE,
        "maximum_simplification_deviation_px": round(max_deviation, 5),
        "method": (
            "12x official pure-river raster blue mask; national outline plus buffered current "
            "cross-border/seasonal corridors; ocean and known lake interiors removed; lake "
            "shoreline loops and dashed seasonal strokes excluded; outer tails of canal ruler "
            "ticks removed while retaining short stems for graph classification; repeated groups "
            "of at least three short, similarly oriented, regularly spaced perpendicular spurs "
            "are removed while isolated or paired short tributaries are retained; "
            "3x3 one-pixel closing; tiled "
            "raster skeleton; graph "
            "traced at shared nodes; paths clipped to the national outline and simplified "
            "with endpoint-preserving 0.08 map-pixel tolerance."
        ),
    }
    return rivers, cross_border, summary


def save_preview(preview_dir: Path, rivers: list[dict[str, Any]],
                 cross_border: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_fc(preview_dir / "rivers_12x_trace.geojson", "official_12x_river_centerlines", rivers)
    write_fc(preview_dir / "cross_border_rivers_12x_trace.geojson", "official_12x_cross_border_river_centerlines", cross_border)
    preview_dir.mkdir(parents=True, exist_ok=True)
    (preview_dir / "trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def apply_to_project(rivers: list[dict[str, Any]], cross_border: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_fc(DATA / "hydrology_lines.geojson", "official_12x_traced_river_centerlines", rivers)
    write_fc(DATA / "cross_border_rivers.geojson", "official_12x_traced_cross_border_river_centerlines", cross_border)

    log_path = DATA / "hydrology_repair_log.json"
    log = read_fc(log_path)
    log["official_12x_centerline_retrace"] = summary
    log["display_geometry_precision"] = {
        **log.get("display_geometry_precision", {}),
        "hydrology_lod_max_simplification_px": 1.0,
        "hydrology_corner_smoothing_radius_px": 0.35,
        "rationale": (
            "Higher-resolution river centerlines are traced from the official 12x raster; "
            "shared network nodes remain mandatory at every render simplification level."
        ),
    }
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    manifest_path = ROOT / "extraction_manifest.json"
    if manifest_path.exists():
        manifest = read_fc(manifest_path)
        manifest.update({
            "river_line_segments": len(rivers),
            "cross_border_river_segments": len(cross_border),
            "official_12x_centerline_retrace": True,
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    canals = read_fc(DATA / "canals.geojson")["features"]
    seasonal = read_fc(DATA / "seasonal_rivers.geojson")["features"]
    import sys
    sys.path.insert(0, str(ROOT))
    import extract_hydrology_styles
    extract_hydrology_styles.update_render_files(rivers, canals, seasonal, cross_border)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raster", required=True, type=Path)
    parser.add_argument("--preview-dir", type=Path, help="write traced GeoJSON here without changing project files")
    parser.add_argument("--write", action="store_true", help="replace the project river layers and render files")
    args = parser.parse_args()
    rivers, cross_border, summary = trace_layers(args.reference_raster.resolve())
    if args.preview_dir:
        save_preview(args.preview_dir.resolve(), rivers, cross_border, summary)
    if args.write:
        apply_to_project(rivers, cross_border, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
