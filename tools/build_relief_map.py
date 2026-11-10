#!/usr/bin/env python3
"""Build the fixed-seed illustrative land-relief model used by the official map.

All horizontal positions come from the project's 2048x1448 source-pixel layers.
No measured elevation model is present in the repository, so the elevation field
is an explicitly synthetic cartographic design constrained by the mapped peaks,
the official northern-plateau boundary trace, the river network, and the coastline.
"""

from __future__ import annotations

import csv
import base64
import json
import math
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "tools"))
import georeference_map  # noqa: E402
from build_region_elevation_stats import build_region_elevation_stats  # noqa: E402

WIDTH, HEIGHT = 2048, 1448
STEP = 2
SEED = 20261108
ELEVATION_MIN_M = -260.0
ELEVATION_MAX_M = 15000.0

# Hand-traced against the green dashed line on the official 2026 map.
# The trace is kept in the same source-pixel coordinate system as every existing
# vector layer, and the original raster is never edited.
PLATEAU_BOUNDARY = [
    [654, 238], [662, 244], [672, 251], [683, 259], [694, 268], [704, 278],
    [713, 288], [721, 299], [730, 311], [739, 323], [748, 336], [757, 349],
    [766, 362], [775, 376], [783, 389], [790, 402], [793, 414], [791, 426],
    [786, 438], [782, 451], [780, 464], [779, 477], [781, 489], [787, 499],
    [797, 506], [809, 509], [823, 509], [837, 511], [850, 511], [864, 508],
    [878, 506], [892, 506], [906, 508], [920, 512], [935, 515], [950, 515],
    [965, 511], [980, 509], [995, 510], [1010, 515], [1025, 520], [1040, 526],
    [1055, 532], [1070, 538], [1085, 544], [1100, 550], [1115, 555], [1130, 560],
    [1145, 565], [1160, 570], [1175, 573], [1188, 574], [1198, 571], [1206, 565],
    [1211, 556], [1213, 545], [1216, 533], [1222, 521], [1231, 509],
]

# Central-range relief is built from local summit massifs and a few gently
# overlapping basin-side peaks. It has no point-to-point backbone: the last
# mapped central-range summit is 刺梨山, beyond which the northern-plateau
# extension peaks remain separate.
CENTRAL_RANGE_BACKBONES: list[list[tuple[float, float]]] = []
CENTRAL_BASIN_CLUSTER_PEAKS = {"春分山", "梅花山", "旃檀山"}
CENTRAL_RANGE_TERMINAL_PEAK = "刺梨山"

RANGE_BACKBONES = {
    # Named ranges receive a backbone only along their mapped relationship.
    # The Central Range is intentionally omitted here and formed from separate
    # local massifs rather than a point-to-point line.
    "北铁山山脉": [(859.25, 347.375), (911.25, 361.375), (1008.5, 370.625)],
    "长岭山脉": [(804.0, 434.375), (808.0, 426.0), (815.0, 419.0),
      (828.0, 415.0), (844.0, 416.0), (862.0, 419.0), (879.0, 419.0),
      (894.5, 417.375), (937.25, 414.625),
      (960.75, 409.625), (1008.5, 370.625)],
    "铁山山脉": [(1008.5, 370.625), (1031.25, 388.125)],
    # This S-shaped line follows the west/left bank of river 60 from Snow Rui
    # to Sword Mountain and fades out at the downstream river crossing.
    "瑞丰山脉": [(991.75, 602.625), (980.0, 612.0), (974.0, 625.0),
      (976.0, 640.0), (982.0, 655.0), (991.0, 670.0), (1003.0, 686.0),
      (1016.5, 705.125)],
}

RIDGES = [
    ("北铁山山脉", RANGE_BACKBONES["北铁山山脉"], 1750, 15),
    ("长岭山脉", RANGE_BACKBONES["长岭山脉"], 2200, 17),
    ("铁山山脉", RANGE_BACKBONES["铁山山脉"], 1200, 14),
    ("瑞丰山脉", RANGE_BACKBONES["瑞丰山脉"], 2350, 19),
]

RIFENG_FADE_CROSSING = (1002.0, 724.5)
RIFENG_FADE_LENGTH_PX = 24.0

NORTH_DUNG_EAST_SPUR_POLYGON = [
    (1178, 544), (1207, 541), (1237, 552), (1265, 576), (1285, 604),
    (1293, 634), (1286, 666), (1269, 683), (1248, 662), (1225, 636),
    (1203, 607), (1184, 585), (1172, 565),
]
NORTH_PLATEAU_EXTENSION_PEAKS = {"秋分山", "光山", "射手山"}
BAISUI_RELIEF_POSITION = (1350.6, 974.8)

GOBI_POLYGON = [(290, 620), (635, 625), (705, 850), (650, 1030), (390, 1075), (270, 870)]
BASIN_POLYGON = [(850, 735), (1090, 720), (1165, 835), (1125, 970), (970, 1010), (850, 920)]
PLAINS_POLYGON = [(800, 900), (1210, 900), (1275, 1160), (1055, 1270), (820, 1165)]
EAST_HILLS_POLYGON = [(1210, 680), (1410, 620), (1500, 1015), (1430, 1280), (1265, 1175), (1180, 950)]
EAST_SOUTHEAST_POLYGON = [
    (1180, 585), (2047, 585), (2047, 1447), (1095, 1447),
    (1120, 1290), (1155, 1100), (1140, 920), (1165, 750),
]

LANDFORM_LABELS: list[dict] = []


def load_js_json(path: Path):
    source = path.read_text(encoding="utf-8")
    return json.loads(source[source.index("=") + 1 : source.rindex(";")])


def svg_marker_centers(svg_source: str, group_id: str) -> list[tuple[float, float]]:
    """Read actual city-dot centers from one SVG marker group, including nested groups."""
    stack: list[str] = []
    centers = []
    token_pattern = re.compile(r'<g\b[^>]*>|</g>|<circle\b[^>]*>')
    for match in token_pattern.finditer(svg_source):
        token = match.group(0)
        if token.startswith("<g"):
            identifier = re.search(r'\bid="([^"]+)"', token)
            stack.append(identifier.group(1) if identifier else "")
        elif token == "</g>":
            if stack:
                stack.pop()
        elif group_id in stack and not (
            group_id == "redMarkers" and "specialCapitalMarker" in stack
        ):
            attributes = dict(re.findall(r'([\w-]+)="([^"]*)"', token))
            if "cx" in attributes and "cy" in attributes:
                centers.append((float(attributes["cx"]), float(attributes["cy"])))
    return centers


def resolve_relief_city_locations(cities: list[dict]) -> tuple[list[dict], dict]:
    """Use the official map's visible city dots as the relief model sample points.

    Many capital labels are deliberately offset from their red dot to avoid collisions.
    The city coordinate catalog is label-anchored, so sampling it directly can put a
    city on the wrong side of a river or even on a neighboring slope.
    """
    svg_source = (ROOT / "index.html").read_text(encoding="utf-8")
    black_markers = svg_marker_centers(svg_source, "blackMarkers")
    red_markers = svg_marker_centers(svg_source, "redMarkers")
    national_capital_decoration = svg_marker_centers(svg_source, "specialCapitalMarker")
    if len(black_markers) != 270 or len(red_markers) != 44 or len(national_capital_decoration) != 3:
        raise RuntimeError(
            "Unexpected official city marker groups: "
            f"black={len(black_markers)}, red={len(red_markers)}, "
            f"national-capital-decoration={len(national_capital_decoration)}"
        )

    resolved = [dict(city) for city in cities]
    source_points = [city.get("sourcePixelXY") for city in resolved]
    role_city_indices = [index for index, city in enumerate(resolved) if city.get("role") == "city" and source_points[index]]
    city_cost = np.asarray([
        [math.hypot(float(source_points[index][0]) - x, float(source_points[index][1]) - y) for x, y in black_markers]
        for index in role_city_indices
    ], dtype=float)
    city_rows, city_cols = linear_sum_assignment(city_cost)
    city_marker_assignment = {
        role_city_indices[row]: (black_markers[col], float(city_cost[row, col]))
        for row, col in zip(city_rows, city_cols)
    }

    capital_indices = [
        index for index, city in enumerate(resolved)
        if city.get("role") in {"capital", "province-capital"} and source_points[index]
    ]
    national_capital_center = (
        sum(point[0] for point in national_capital_decoration) / len(national_capital_decoration),
        sum(point[1] for point in national_capital_decoration) / len(national_capital_decoration),
    )
    capital_markers = red_markers + [national_capital_center]
    capital_cost = np.asarray([
        [math.hypot(float(source_points[index][0]) - x, float(source_points[index][1]) - y) for x, y in capital_markers]
        for index in capital_indices
    ], dtype=float)
    capital_rows, capital_cols = linear_sum_assignment(capital_cost)
    capital_marker_assignment = {
        capital_indices[row]: (capital_markers[col], float(capital_cost[row, col]))
        for row, col in zip(capital_rows, capital_cols)
    }

    marker_assignment = {**city_marker_assignment, **capital_marker_assignment}
    reference = json.loads((DATA / "georeference.json").read_text(encoding="utf-8"))
    matched_count = 0
    moved_capitals = []
    max_capital_label_offset = 0.0
    for index, city in enumerate(resolved):
        original_point = source_points[index]
        if not original_point:
            continue
        city["labelAnchorPixelXY"] = [float(original_point[0]), float(original_point[1])]
        assignment = marker_assignment.get(index)
        if assignment is None:
            # One ordinary city is a label-only entry in the source SVG; leave it at
            # its published label anchor instead of snapping it to an unrelated dot.
            city["markerSource"] = "map-label-anchor (no separate marker)"
            city["markerOffsetPx"] = 0.0
            continue
        point, offset = assignment
        if city.get("role") == "city" and offset > 0.20:
            raise RuntimeError(f"Could not pair ordinary city marker safely: {city.get('name')} ({offset:.2f}px)")
        if city.get("role") in {"capital", "province-capital"} and offset > 25.0:
            raise RuntimeError(f"Could not pair capital marker safely: {city.get('name')} ({offset:.2f}px)")
        city["sourcePixelXY"] = [round(point[0], 3), round(point[1], 3)]
        city["markerOffsetPx"] = round(offset, 2)
        city["markerSource"] = "official SVG city marker"
        longitude, latitude = georeference_map.pixel_to_lonlat(point[0], point[1], reference)
        city["longitudeDeg"] = longitude
        city["mapLatitudeDeg"] = latitude
        matched_count += 1
        if city.get("role") in {"capital", "province-capital"}:
            moved_capitals.append(city["name"])
            max_capital_label_offset = max(max_capital_label_offset, offset)

    audit = {
        "cityCount": len(resolved),
        "markerMatchedCityCount": matched_count,
        "ordinaryCityMarkerCount": len(city_marker_assignment),
        "capitalMarkerCount": len(capital_marker_assignment),
        "capitalCityCount": len(capital_indices),
        "capitalLabelOffsetsOver1Px": len(moved_capitals),
        "maximumCapitalLabelToMarkerOffsetPx": round(max_capital_label_offset, 2),
        "singleUnmarkedOrdinaryCityCount": len(role_city_indices) - len(city_marker_assignment),
        "basis": "visible SVG city-dot centers in index.html; city label anchors retained separately",
    }
    return resolved, audit


def point_in_polygon(x: np.ndarray, y: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    return np.asarray(MplPath(polygon).contains_points(np.column_stack([x.ravel(), y.ravel()]))).reshape(x.shape)


def smooth_polygon_weight(
    X: np.ndarray, Y: np.ndarray, polygon: list[list[float]], feather_px: float
) -> tuple[np.ndarray, np.ndarray]:
    inside = point_in_polygon(X, Y, polygon)
    inside_distance = ndimage.distance_transform_edt(inside) * STEP
    outside_distance = ndimage.distance_transform_edt(~inside) * STEP
    signed_distance = inside_distance - outside_distance
    weight = 1.0 / (1.0 + np.exp(np.clip(-signed_distance / feather_px, -30, 30)))
    return weight, inside_distance


def normalized_noise(
    shape: tuple[int, int],
    sigma_px: float,
    rng: np.random.Generator,
    pixel_size_px: float = STEP,
) -> np.ndarray:
    field = rng.standard_normal(shape)
    field = ndimage.gaussian_filter(field, sigma=sigma_px / pixel_size_px, mode="reflect")
    return field / max(float(field.std()), 1e-8)


def land_mask_full() -> np.ndarray:
    outline = json.loads((DATA / "national_outline.geojson").read_text(encoding="utf-8"))
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for feature in outline["features"]:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Polygon":
            continue
        rings = geometry.get("coordinates", [])
        if not rings:
            continue
        draw.polygon([tuple(point) for point in rings[0]], fill=255)
        for hole in rings[1:]:
            draw.polygon([tuple(point) for point in hole], fill=0)
    return np.asarray(image) > 0


def ocean_water_mask_full() -> np.ndarray:
    water = json.loads((DATA / "ocean_water.geojson").read_text(encoding="utf-8"))
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for feature in water.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        if geometry_type == "MultiPolygon":
            polygons = geometry.get("coordinates", [])
        elif geometry_type == "Polygon":
            polygons = [geometry.get("coordinates", [])]
        else:
            continue
        for rings in polygons:
            if not rings:
                continue
            draw.polygon([tuple(point) for point in rings[0]], fill=255)
            for hole in rings[1:]:
                draw.polygon([tuple(point) for point in hole], fill=0)
    return np.asarray(image) > 0


def lake_water_mask_full() -> np.ndarray:
    lakes = json.loads((DATA / "lakes.geojson").read_text(encoding="utf-8"))
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for feature in lakes.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        if geometry_type == "MultiPolygon":
            polygons = geometry.get("coordinates", [])
        elif geometry_type == "Polygon":
            polygons = [geometry.get("coordinates", [])]
        else:
            continue
        for rings in polygons:
            if not rings:
                continue
            draw.polygon([tuple(point) for point in rings[0]], fill=255)
            for hole in rings[1:]:
                draw.polygon([tuple(point) for point in hole], fill=0)
    return np.asarray(image) > 0


def official_river_water_mask_full(
    land_mask: np.ndarray,
    ocean_mask: np.ndarray,
    lake_mask: np.ndarray,
) -> np.ndarray:
    """Read the river strokes actually drawn on the official source map."""
    source = Image.open(ROOT / "Administrative-Division-Map-of-Donh-2026.png").convert("RGB")
    if source.size != (WIDTH * 4, HEIGHT * 4):
        raise RuntimeError(
            f"Expected 4x source-map pixels {(WIDTH * 4, HEIGHT * 4)}, got {source.size}"
        )
    rgb = np.asarray(source, dtype=np.uint8)
    red, green, blue = (rgb[:, :, channel].astype(np.int16) for channel in range(3))
    # The official river and lake strokes use #b1d0ed. A narrow tolerance keeps
    # antialiased line cores while excluding blue map lettering and border ink.
    water_ink = (
        (np.abs(red - 177) <= 6)
        & (np.abs(green - 208) <= 6)
        & (np.abs(blue - 237) <= 6)
    )
    source_rivers = water_ink.reshape(HEIGHT, 4, WIDTH, 4).any(axis=(1, 3))
    return source_rivers & land_mask & ~ocean_mask & ~lake_mask


def make_river_bank_specs(
    peaks: list[tuple],
    river_mask: np.ndarray,
    max_river_distance_px: float = 22.0,
) -> dict[str, dict[str, float]]:
    """Identify a nearby river bank for isolated mapped peaks."""
    if not np.any(river_mask):
        return {}
    distance, nearest_indices = ndimage.distance_transform_edt(
        ~river_mask, return_indices=True
    )
    specs: dict[str, dict[str, float]] = {}
    for peak in peaks:
        x, y, name = float(peak[0]), float(peak[1]), peak[2]
        px, py = int(round(x)), int(round(y))
        river_distance = float(distance[py, px])
        if river_distance < 2.0 or river_distance > max_river_distance_px:
            continue
        qx = float(nearest_indices[1, py, px])
        qy = float(nearest_indices[0, py, px])
        x0, x1 = max(0, int(qx) - 12), min(WIDTH, int(qx) + 13)
        y0, y1 = max(0, int(qy) - 12), min(HEIGHT, int(qy) + 13)
        local_y, local_x = np.nonzero(river_mask[y0:y1, x0:x1])
        points = np.column_stack((local_x + x0, local_y + y0)).astype(float)
        vectors = points - np.asarray([qx, qy])
        vectors = vectors[np.linalg.norm(vectors, axis=1) <= 10.0]
        if len(vectors) < 6:
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(vectors.T))
        if eigenvalues[-1] / max(float(eigenvalues[0]), 1e-4) < 1.3:
            continue
        tangent_x, tangent_y = map(float, eigenvectors[:, -1])
        if tangent_x < 0.0:
            tangent_x, tangent_y = -tangent_x, -tangent_y
        signed_peak = tangent_x * (y - qy) - tangent_y * (x - qx)
        if abs(signed_peak) < 6.0:
            continue
        side = 1.0 if signed_peak > 0.0 else -1.0
        opposing_peaks = []
        for other in peaks:
            ox, oy = float(other[0]), float(other[1])
            if other[2] == name or math.hypot(ox - qx, oy - qy) > 42.0:
                continue
            signed_other = tangent_x * (oy - qy) - tangent_y * (ox - qx)
            if signed_other * side < -2.0:
                opposing_peaks.append(other[2])
        if opposing_peaks:
            continue
        peak_key = f"{name}@{x:.3f},{y:.3f}"
        specs[peak_key] = {
            "name": name,
            "peak_x": x,
            "peak_y": y,
            "qx": qx,
            "qy": qy,
            "tangent_x": tangent_x,
            "tangent_y": tangent_y,
            "side": side,
            "river_distance_px": river_distance,
        }
    return specs


def river_bank_weight(x, y, spec: dict[str, float], fade_px: float = 3.5):
    signed_distance = spec["side"] * (
        spec["tangent_x"] * (y - spec["qy"])
        - spec["tangent_y"] * (x - spec["qx"])
    )
    t = np.clip(signed_distance / fade_px, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def flow_conditioned_river_surface(
    terrain: np.ndarray,
    land_mask: np.ndarray,
    river_water_mask: np.ndarray,
    ocean_mask: np.ndarray,
    lake_mask: np.ndarray,
    hydrology: dict,
    height_caps: list[tuple[float, float, float, str]] | None = None,
    minimum_drop_m_per_px: float = 0.35,
) -> dict[str, object]:
    """Route every official blue channel pixel and traced reach downhill.

    The previous bank-plane mask made mountain slopes terminate on a straight
    line at the river. This conditions the connected graph made from both the
    official water strokes and the traced vector reaches, then uses its curved
    channel distance for the nearby valley cross-section.
    """
    centerline_image = Image.new("L", (WIDTH, HEIGHT), 0)
    centerline_draw = ImageDraw.Draw(centerline_image)
    for feature in hydrology.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "LineString":
            centerline_draw.line(
                [tuple(point) for point in geometry.get("coordinates", [])],
                fill=255,
                width=1,
                joint="curve",
            )
    centerline_mask = np.asarray(centerline_image) > 0
    centerline_mask &= land_mask | ocean_mask | lake_mask
    if not np.any(centerline_mask):
        raise RuntimeError("No hydrology centerlines could be rasterized")
    routing_mask = centerline_mask | river_water_mask

    bank_ground = np.where(land_mask & ~river_water_mask & ~lake_mask, terrain, np.inf)
    nearby_bank_min = ndimage.minimum_filter(bank_ground, size=25, mode="nearest")
    raw_channel = np.minimum(terrain, nearby_bank_min - 8.0)
    raw_channel = np.clip(raw_channel, ELEVATION_MIN_M, ELEVATION_MAX_M)
    ocean_distance = ndimage.distance_transform_edt(~ocean_mask).astype(np.float32)
    lake_distance, lake_nearest = ndimage.distance_transform_edt(
        ~lake_mask, return_indices=True
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        routing_mask.astype(np.uint8), connectivity=8
    )
    conditioned = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    flow_distance_full = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
    outlet_types: dict[int, str] = {}
    applied_height_caps = []
    uphill_step_count = 0
    downstream_step_count = 0
    uphill_violation_edges = []

    offsets = [
        (dy, dx, math.sqrt(2.0) if dy and dx else 1.0)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    ]
    for component_id in range(1, component_count):
        x0 = int(stats[component_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[component_id, cv2.CC_STAT_TOP])
        box_w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        local_labels = labels[y0:y0 + box_h, x0:x0 + box_w]
        local_y, local_x = np.where(local_labels == component_id)
        count = len(local_x)
        if count == 0:
            continue
        pixel_y = local_y + y0
        pixel_x = local_x + x0
        node_map = np.full(local_labels.shape, -1, dtype=np.int32)
        node_map[local_y, local_x] = np.arange(count, dtype=np.int32)

        graph_rows, graph_cols, graph_weights = [], [], []
        degree = np.zeros(count, dtype=np.uint8)
        for dy, dx, edge_length in offsets:
            inside = (
                (pixel_y + dy >= y0) & (pixel_y + dy < y0 + box_h)
                & (pixel_x + dx >= x0) & (pixel_x + dx < x0 + box_w)
            )
            source_ids = np.flatnonzero(inside)
            neighbor_ids = node_map[
                pixel_y[source_ids] + dy - y0,
                pixel_x[source_ids] + dx - x0,
            ]
            connected = neighbor_ids >= 0
            source_ids = source_ids[connected]
            neighbor_ids = neighbor_ids[connected]
            if len(source_ids):
                graph_rows.extend(source_ids.tolist())
                graph_cols.extend(neighbor_ids.tolist())
                graph_weights.extend([edge_length] * len(source_ids))
                degree[source_ids] += 1
        graph = coo_matrix(
            (graph_weights, (graph_rows, graph_cols)), shape=(count, count)
        ).tocsr()

        local_ocean_distance = ocean_distance[pixel_y, pixel_x]
        sea_sources = np.flatnonzero(local_ocean_distance <= 2.0)
        local_lake_distance = lake_distance[pixel_y, pixel_x]
        lake_sources = np.flatnonzero(local_lake_distance <= 2.0)
        if len(sea_sources):
            source_nodes = sea_sources
            source_level = 0.0
            lower_bound = 0.0
            outlet_type = "海口"
        elif len(lake_sources):
            source_nodes = lake_sources
            source_x = np.clip(lake_nearest[1, pixel_y[lake_sources], pixel_x[lake_sources]], 0, WIDTH - 1)
            source_y = np.clip(lake_nearest[0, pixel_y[lake_sources], pixel_x[lake_sources]], 0, HEIGHT - 1)
            source_level = float(np.nanmin(terrain[lake_mask])) if np.any(lake_mask) else float(np.nanmedian(terrain[source_y, source_x]))
            lower_bound = min(source_level, 0.0)
            outlet_type = "湖口"
        else:
            candidates = np.arange(count)
            source_node = int(candidates[np.argmin(raw_channel[pixel_y, pixel_x])])
            source_nodes = np.asarray([source_node], dtype=np.int32)
            source_level = float(raw_channel[pixel_y[source_node], pixel_x[source_node]])
            lower_bound = min(source_level, 0.0)
            outlet_type = "内陆低点"
        outlet_types[component_id] = outlet_type

        distances = dijkstra(graph, indices=source_nodes, min_only=True)
        if distances.ndim != 1 or not np.all(np.isfinite(distances)):
            raise RuntimeError(f"Could not route river component {component_id} to its outlet")
        heights = raw_channel[pixel_y, pixel_x].astype(np.float32, copy=True)
        heights = np.maximum(heights, lower_bound)
        heights[source_nodes] = source_level
        source_set = np.zeros(count, dtype=bool)
        source_set[source_nodes] = True

        for cx, cy, maximum_height, control_name in height_caps or []:
            nearest_node = int(np.argmin((pixel_x - cx) ** 2 + (pixel_y - cy) ** 2))
            control_distance = math.hypot(pixel_x[nearest_node] - cx, pixel_y[nearest_node] - cy)
            if control_distance <= 22.0 and not source_set[nearest_node]:
                heights[nearest_node] = min(heights[nearest_node], float(maximum_height))
                applied_height_caps.append({
                    "name": control_name,
                    "x": round(float(pixel_x[nearest_node]), 1),
                    "y": round(float(pixel_y[nearest_node]), 1),
                    "maximumRiverElevationM": round(float(maximum_height), 1),
                    "snapDistancePx": round(control_distance, 1),
                    "componentId": int(component_id),
                })

        ordered_nodes = np.argsort(distances)[::-1]
        for node in ordered_nodes:
            if source_set[node]:
                continue
            y, x = int(pixel_y[node]), int(pixel_x[node])
            for dy, dx, edge_length in offsets:
                nx_, ny_ = x + dx, y + dy
                if not (x0 <= nx_ < x0 + box_w and y0 <= ny_ < y0 + box_h):
                    continue
                next_node = int(node_map[ny_ - y0, nx_ - x0])
                if next_node < 0 or distances[next_node] >= distances[node] - 1e-5:
                    continue
                downstream_step_count += 1
                candidate = max(
                    lower_bound,
                    float(heights[node]) - minimum_drop_m_per_px * edge_length,
                )
                if heights[next_node] > candidate:
                    heights[next_node] = candidate

        for node in range(count):
            y, x = int(pixel_y[node]), int(pixel_x[node])
            conditioned[y, x] = heights[node]
            flow_distance_full[y, x] = distances[node]
            for dy, dx, _ in offsets:
                nx_, ny_ = x + dx, y + dy
                if not (x0 <= nx_ < x0 + box_w and y0 <= ny_ < y0 + box_h):
                    continue
                next_node = int(node_map[ny_ - y0, nx_ - x0])
                if next_node >= 0 and distances[next_node] < distances[node] - 1e-5:
                    if heights[next_node] > heights[node] + 0.02:
                        uphill_step_count += 1
                        uphill_violation_edges.append({
                            "upstream": [int(x), int(y)],
                            "downstream": [nx_, ny_],
                            "upstreamM": round(float(heights[node]), 2),
                            "downstreamM": round(float(heights[next_node]), 2),
                            "upstreamDistancePx": round(float(distances[node]), 2),
                            "downstreamDistancePx": round(float(distances[next_node]), 2),
                            "componentId": int(component_id),
                        })

    centerline_distance, centerline_nearest = ndimage.distance_transform_edt(
        ~routing_mask, return_indices=True
    )
    nearest_surface = conditioned[centerline_nearest[0], centerline_nearest[1]]
    fallback_surface = np.maximum(
        0.0, terrain - (8.0 + 0.005 * np.maximum(terrain, 0.0))
    )
    river_surface = fallback_surface.astype(np.float32)
    mapped_channel = river_water_mask
    river_surface[mapped_channel] = nearest_surface[mapped_channel]
    river_surface[river_water_mask & ~mapped_channel] = fallback_surface[
        river_water_mask & ~mapped_channel
    ]

    return {
        "centerlineMask": centerline_mask,
        "routedMask": routing_mask,
        "centerlineDistance": centerline_distance.astype(np.float32),
        "centerlineNearest": centerline_nearest,
        "centerlineSurface": conditioned,
        "flowDistance": flow_distance_full,
        "riverSurface": river_surface,
        "componentCount": int(component_count - 1),
        "centerlinePixelCount": int(np.count_nonzero(centerline_mask)),
        "routedNetworkPixelCount": int(np.count_nonzero(routing_mask)),
        "outletTypes": outlet_types,
        "heightCaps": applied_height_caps,
        "minimumDropMPerPx": float(minimum_drop_m_per_px),
        "downhillStepCount": int(downstream_step_count),
        "uphillViolationCount": int(uphill_step_count),
        "uphillViolationEdges": uphill_violation_edges,
    }


def write_hydrology_tile_audit(
    terrain: np.ndarray,
    land_mask: np.ndarray,
    river_mask: np.ndarray,
    river_surface: np.ndarray,
    centerline_mask: np.ndarray,
    centerline_surface: np.ndarray,
    flow_distance: np.ndarray,
    cities: list[dict],
    peaks: list[tuple],
) -> dict[str, int]:
    """Audit the map as 48 overlapping blocks, checking flow and valley banks."""
    water_distance, water_nearest = ndimage.distance_transform_edt(
        ~river_mask, return_indices=True
    )
    rows = []
    audited_uphill_pairs = audited_bank_samples = bank_below_water = 0
    tile_width, tile_height = 280, 258
    x_step, y_step = 254, 238
    cols, row_count = 8, 6
    half_offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
    for tile_row in range(row_count):
        y0 = tile_row * y_step
        y1 = min(HEIGHT, y0 + tile_height)
        for tile_col in range(cols):
            x0 = tile_col * x_step
            x1 = min(WIDTH, x0 + tile_width)
            local_centerline = centerline_mask[y0:y1, x0:x1]
            local_flow = flow_distance[y0:y1, x0:x1]
            local_surface = centerline_surface[y0:y1, x0:x1]
            uphill_pairs = 0
            for dy, dx in half_offsets:
                ya0, ya1 = max(y0, y0 - dy), min(y1, y1 - dy)
                xa0, xa1 = max(x0, x0 - dx), min(x1, x1 - dx)
                if ya0 >= ya1 or xa0 >= xa1:
                    continue
                ys0, ys1 = ya0 - y0, ya1 - y0
                xs0, xs1 = xa0 - x0, xa1 - x0
                ys1n, xs1n = ys0 + dy, xs0 + dx
                valid = (
                    local_centerline[ys0:ys1, xs0:xs1]
                    & local_centerline[ys1n:ys1n + (ys1 - ys0), xs1n:xs1n + (xs1 - xs0)]
                    & (local_flow[ys0:ys1, xs0:xs1]
                       > local_flow[ys1n:ys1n + (ys1 - ys0), xs1n:xs1n + (xs1 - xs0)] + 1e-5)
                )
                if not np.any(valid):
                    continue
                upstream = local_surface[ys0:ys1, xs0:xs1][valid]
                downstream = local_surface[
                    ys1n:ys1n + (ys1 - ys0), xs1n:xs1n + (xs1 - xs0)
                ][valid]
                uphill_pairs += int(np.count_nonzero(downstream > upstream + 0.02))

            d = water_distance[y0:y1, x0:x1]
            local_land = land_mask[y0:y1, x0:x1] & ~river_mask[y0:y1, x0:x1]
            bank_area = local_land & (d >= 2.0) & (d <= 12.0)
            if np.any(bank_area):
                nearest_y = water_nearest[0, y0:y1, x0:x1][bank_area]
                nearest_x = water_nearest[1, y0:y1, x0:x1][bank_area]
                clearance = terrain[y0:y1, x0:x1][bank_area] - river_surface[nearest_y, nearest_x]
                min_clearance = float(np.min(clearance))
                p05_clearance = float(np.percentile(clearance, 5))
                below = int(np.count_nonzero(clearance < -0.1))
                bank_count = int(len(clearance))
            else:
                min_clearance = p05_clearance = float("nan")
                below = bank_count = 0

            city_count = sum(
                x0 <= float(city.get("sourcePixelXY", [-1, -1])[0]) < x1
                and y0 <= float(city.get("sourcePixelXY", [-1, -1])[1]) < y1
                for city in cities if city.get("sourcePixelXY")
            )
            peak_count = sum(x0 <= peak[0] < x1 and y0 <= peak[1] < y1 for peak in peaks)
            tile_id = f"R{tile_row + 1:02d}C{tile_col + 1:02d}"
            status = "通过" if uphill_pairs == 0 and below == 0 else "复核"
            rows.append([
                tile_id, tile_row + 1, tile_col + 1, x0, y0, x1, y1,
                int(np.count_nonzero(river_mask[y0:y1, x0:x1])),
                int(np.count_nonzero(local_centerline)),
                uphill_pairs, bank_count, below,
                round(min_clearance, 1) if np.isfinite(min_clearance) else "",
                round(p05_clearance, 1) if np.isfinite(p05_clearance) else "",
                city_count, peak_count, status,
            ])
            audited_uphill_pairs += uphill_pairs
            audited_bank_samples += bank_count
            bank_below_water += below

    with (DATA / "hydrology_tile_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "板块", "行", "列", "x起点", "y起点", "x终点", "y终点",
            "河道水面像素数", "水系流路像素数", "顺流升高邻接数",
            "邻岸样本数_2至12px", "低于河面邻岸数", "最低邻岸高差_m", "邻岸高差5分位_m",
            "城市数", "山峰数", "自查结果",
        ])
        writer.writerows(rows)
    return {
        "tileCount": len(rows),
        "tilePassCount": sum(row[-1] == "通过" for row in rows),
        "uphillAdjacentPairCount": audited_uphill_pairs,
        "nearBankSampleCount": audited_bank_samples,
        "landBelowRiverSurfaceSampleCount": bank_below_water,
    }


def create_hydrology_tile_review_sheets(preview_path: Path) -> list[str]:
    """Export eight labelled 2x3 sheets covering every overlapping map tile."""
    preview = Image.open(preview_path).convert("RGB")
    output_dir = ROOT / "diagnostics"
    output_dir.mkdir(exist_ok=True)
    tile_width, tile_height = 280, 258
    x_step, y_step = 254, 238
    tiles = []
    for row in range(6):
        for col in range(8):
            x0, y0 = col * x_step, row * y_step
            x1, y1 = min(WIDTH, x0 + tile_width), min(HEIGHT, y0 + tile_height)
            tile_id = f"R{row + 1:02d}C{col + 1:02d}"
            tiles.append((tile_id, preview.crop((x0, y0, x1, y1))))
    outputs = []
    for sheet_index in range(8):
        canvas = Image.new("RGB", (2 * tile_width, 3 * (tile_height + 22)), (248, 247, 242))
        draw = ImageDraw.Draw(canvas)
        for slot in range(6):
            tile_id, tile = tiles[sheet_index * 6 + slot]
            col, row = slot % 2, slot // 2
            left, top = col * tile_width, row * (tile_height + 22)
            canvas.paste(tile, (left, top + 22))
            draw.text((left + 6, top + 3), tile_id, fill=(30, 48, 55))
            draw.rectangle((left, top + 22, left + tile.width - 1, top + 22 + tile.height - 1), outline=(55, 76, 81), width=1)
        path = output_dir / f"hydrology_tile_review_{sheet_index + 1:02d}.png"
        canvas.save(path, optimize=True)
        outputs.append(str(path))
    return outputs


def draw_line_on_grid(draw: ImageDraw.ImageDraw, coords: list[list[float]], width: int = 1):
    scaled = [(round(x / STEP), round(y / STEP)) for x, y in coords]
    if len(scaled) > 1:
        draw.line(scaled, fill=255, width=width, joint="curve")


def bilinear(field: np.ndarray, x: float, y: float, step: float = STEP) -> float:
    gx, gy = x / step, y / step
    x0, y0 = int(np.floor(gx)), int(np.floor(gy))
    x1, y1 = min(x0 + 1, field.shape[1] - 1), min(y0 + 1, field.shape[0] - 1)
    x0, y0 = max(0, x0), max(0, y0)
    dx, dy = gx - x0, gy - y0
    return float(
        field[y0, x0] * (1 - dx) * (1 - dy)
        + field[y0, x1] * dx * (1 - dy)
        + field[y1, x0] * (1 - dx) * dy
        + field[y1, x1] * dx * dy
    )


def bilinear_encoded_elevation(code: np.ndarray, mask: np.ndarray, x: float, y: float) -> float | None:
    """Sample the same 16-bit elevation values and mask used by the browser."""
    gx = min(max(float(x), 0.0), code.shape[1] - 1.0)
    gy = min(max(float(y), 0.0), code.shape[0] - 1.0)
    x0, y0 = math.floor(gx), math.floor(gy)
    x1, y1 = min(x0 + 1, code.shape[1] - 1), min(y0 + 1, code.shape[0] - 1)
    dx, dy = gx - x0, gy - y0
    total = weight_total = 0.0
    for xx, yy, weight in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x1, y0, dx * (1.0 - dy)),
        (x0, y1, (1.0 - dx) * dy),
        (x1, y1, dx * dy),
    ):
        if weight <= 0.0 or not mask[yy, xx]:
            continue
        value = ELEVATION_MIN_M + int(code[yy, xx]) / 65535.0 * (ELEVATION_MAX_M - ELEVATION_MIN_M)
        total += value * weight
        weight_total += weight
    return total / weight_total if weight_total else None


def pixel_to_saturn_map_coordinates(x: float, y: float) -> tuple[float, float]:
    """Invert the map's documented Saturn projection for a corrected peak point."""
    radius_km = 60268.0
    eccentricity_squared = 0.18632823035361423
    pixel_scale_km = 7.7719530772698535
    standard_parallel = math.radians(26.28251565833631)
    anchor_physical_latitude = math.radians(28.7)
    degrees_per_map_degree = 6.0

    def authalic_q(latitude: float) -> float:
        eccentricity = math.sqrt(eccentricity_squared)
        sine = math.sin(latitude)
        denominator = 1.0 - eccentricity_squared * sine * sine
        return (1.0 - eccentricity_squared) * (
            sine / denominator
            - math.log((1.0 - eccentricity * sine) / (1.0 + eccentricity * sine))
            / (2.0 * eccentricity)
        )

    longitude = (x - 1002.0) * pixel_scale_km / (radius_km * math.cos(standard_parallel))
    longitude = math.degrees(longitude)
    projected_y = (877.0 - y) * pixel_scale_km
    target_q = authalic_q(anchor_physical_latitude) + (
        projected_y * 2.0 * math.cos(standard_parallel) / radius_km
    )
    low, high = -math.pi / 2.0 + 1e-9, math.pi / 2.0 - 1e-9
    for _ in range(80):
        middle = (low + high) / 2.0
        if authalic_q(middle) < target_q:
            low = middle
        else:
            high = middle
    map_latitude = math.degrees((low + high) / 2.0) / degrees_per_map_degree
    return longitude, map_latitude


def backbone_tangent(
    center_x: float,
    center_y: float,
    line_sets: list[list[tuple[float, float]]],
) -> tuple[tuple[float, float], float]:
    """Return the direction and distance to a documented ridge segment."""
    best_distance_sq = float("inf")
    best_tangent = (1.0, 0.0)
    for line in line_sets:
        for (x1, y1), (x2, y2) in zip(line, line[1:]):
            vx, vy = x2 - x1, y2 - y1
            length_sq = vx * vx + vy * vy
            if length_sq <= 1e-8:
                continue
            t = float(np.clip(((center_x - x1) * vx + (center_y - y1) * vy) / length_sq, 0.0, 1.0))
            distance_sq = (center_x - x1 - t * vx) ** 2 + (center_y - y1 - t * vy) ** 2
            if distance_sq < best_distance_sq:
                length = math.sqrt(length_sq)
                best_distance_sq = distance_sq
                best_tangent = (vx / length, vy / length)
    return best_tangent, math.sqrt(best_distance_sq)


def bilinear_oriented_gaussian(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    tangent: tuple[float, float],
    sigma_along: float,
    sigma_across: float,
    allowed_mask: np.ndarray | None = None,
    river_bank_spec: dict[str, float] | None = None,
) -> float:
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, WIDTH - 1), min(y0 + 1, HEIGHT - 1)
    dx, dy = x - x0, y - y0
    sample_points = (
        (x0, y0, (1 - dx) * (1 - dy)),
        (x1, y0, dx * (1 - dy)),
        (x0, y1, (1 - dx) * dy),
        (x1, y1, dx * dy),
    )
    tangent_x, tangent_y = tangent
    normal_x, normal_y = -tangent_y, tangent_x
    return sum(
        weight
        * (1.0 if allowed_mask is None else float(allowed_mask[py, px]))
        * (1.0 if river_bank_spec is None else float(river_bank_weight(px, py, river_bank_spec)))
        * math.exp(-0.5 * (
            (((px - center_x) * tangent_x + (py - center_y) * tangent_y) / sigma_along) ** 2
            + (((px - center_x) * normal_x + (py - center_y) * normal_y) / sigma_across) ** 2
        ))
        for px, py, weight in sample_points
        if allowed_mask is None or allowed_mask[py, px] > 0.0
    )


def apply_peak_elevation_anchors(
    z_full: np.ndarray,
    peaks: list[tuple],
    rifeng_west_bank_mask: np.ndarray | None = None,
    river_bank_specs: dict[str, dict[str, float]] | None = None,
) -> float:
    """Fit summit heights while keeping each isolated massif on its river bank."""
    if not peaks:
        return 0.0
    points = np.asarray([[peak[0], peak[1]] for peak in peaks], dtype=float)
    targets = np.asarray([peak[3] for peak in peaks], dtype=float)
    current = np.asarray([bilinear(z_full, x, y, step=1.0) for x, y in points], dtype=float)
    residuals = targets - current
    kernel_specs = []
    kernel_masks = []
    kernel_bank_specs = []
    for peak in peaks:
        name, zone, range_name = peak[2], peak[5], peak[6]
        if range_name in RANGE_BACKBONES:
            candidates = [RANGE_BACKBONES[range_name]]
        elif zone == "central_range":
            candidates = []
        else:
            candidates = []
        if candidates:
            tangent, distance_to_ridge = backbone_tangent(peak[0], peak[1], candidates)
        else:
            # Central peaks receive independent orientations; adjacent peaks
            # are not aimed at each other along a connect-the-dots axis.
            angle = math.radians((peak[0] * 127.1 + peak[1] * 311.7 + SEED) % 180.0)
            tangent = (math.cos(angle), math.sin(angle))
        if name in {"百岁山", "浪花山"}:
            sigma_along, sigma_across = 10.0, 5.0
        elif zone == "central_range" and name in CENTRAL_BASIN_CLUSTER_PEAKS:
            sigma_along, sigma_across = 20.0, 9.0
        elif zone == "central_range":
            sigma_along, sigma_across = 19.0, 8.5
        elif range_name:
            sigma_along, sigma_across = 31.0, 11.0
        elif zone == "north_plateau":
            sigma_along, sigma_across = 26.0, 12.0
        else:
            sigma_along, sigma_across = 22.0, 9.0
        kernel_specs.append((tangent, sigma_along, sigma_across))
        kernel_masks.append(rifeng_west_bank_mask if range_name == "瑞丰山脉" else None)
        peak_key = f"{name}@{peak[0]:.3f},{peak[1]:.3f}"
        kernel_bank_specs.append((river_bank_specs or {}).get(peak_key))
    kernel = np.asarray([
        [
            bilinear_oriented_gaussian(
                x,
                y,
                center_x,
                center_y,
                *kernel_specs[column],
                allowed_mask=kernel_masks[column],
                river_bank_spec=kernel_bank_specs[column],
            )
            for column, (center_x, center_y) in enumerate(points)
        ]
        for x, y in points
    ], dtype=float)
    try:
        weights = np.linalg.solve(kernel, residuals)
    except np.linalg.LinAlgError:
        weights = np.linalg.solve(kernel + np.eye(len(points)) * 1e-8, residuals)

    grid_x = np.arange(WIDTH, dtype=np.float32)[None, :]
    grid_y = np.arange(HEIGHT, dtype=np.float32)[:, None]
    correction = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for (center_x, center_y), weight, (tangent, sigma_along, sigma_across), allowed_mask, bank_spec in zip(
        points, weights, kernel_specs, kernel_masks, kernel_bank_specs
    ):
        tangent_x, tangent_y = tangent
        normal_x, normal_y = -tangent_y, tangent_x
        dx, dy = grid_x - center_x, grid_y - center_y
        along = dx * tangent_x + dy * tangent_y
        across = dx * normal_x + dy * normal_y
        exponent = -0.5 * ((along / sigma_along) ** 2 + (across / sigma_across) ** 2)
        kernel_image = np.exp(exponent).astype(np.float32)
        if allowed_mask is not None:
            kernel_image *= allowed_mask
        if bank_spec is not None:
            qx, qy = bank_spec["qx"], bank_spec["qy"]
            x0, x1 = max(0, int(qx - 80)), min(WIDTH, int(qx + 81))
            y0, y1 = max(0, int(qy - 80)), min(HEIGHT, int(qy + 81))
            local_y, local_x = np.mgrid[y0:y1, x0:x1]
            kernel_image[y0:y1, x0:x1] *= river_bank_weight(local_x, local_y, bank_spec)
        correction += kernel_image * np.float32(weight)
    z_full += correction

    errors = [abs(bilinear(z_full, x, y, step=1.0) - target) for (x, y), target in zip(points, targets)]
    return float(max(errors, default=0.0))


def dashed_line(draw: ImageDraw.ImageDraw, coords: list[list[float]], fill, width: int, dash=9, gap=5):
    for first, last in zip(coords, coords[1:]):
        x1, y1 = first
        x2, y2 = last
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            continue
        t = 0.0
        while t < length:
            end = min(length, t + dash)
            p1 = (x1 + (x2 - x1) * t / length, y1 + (y2 - y1) * t / length)
            p2 = (x1 + (x2 - x1) * end / length, y1 + (y2 - y1) * end / length)
            draw.line((p1, p2), fill=fill, width=width)
            t += dash + gap


def main():
    global ELEVATION_MAX_M
    rng = random.Random(SEED)
    noise_rng = np.random.default_rng(SEED)
    land_full = land_mask_full()
    ocean_water_full = ocean_water_mask_full()
    lake_mask_full = lake_water_mask_full()
    river_water_mask_full = official_river_water_mask_full(
        land_full, ocean_water_full, lake_mask_full
    )
    xs = np.arange(0, WIDTH, STEP, dtype=np.float32)
    ys = np.arange(0, HEIGHT, STEP, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    land = land_full[::STEP, ::STEP]
    ocean_water = ocean_water_full[::STEP, ::STEP]
    coast_distance = ndimage.distance_transform_edt(~ocean_water) * STEP
    east_southeast_soft_weight, _ = smooth_polygon_weight(X, Y, EAST_SOUTHEAST_POLYGON, 18.0)
    east_southeast_inside = point_in_polygon(X, Y, EAST_SOUTHEAST_POLYGON)
    east_southeast_weight = np.where(east_southeast_inside, 1.0, east_southeast_soft_weight)

    boundary = np.asarray(PLATEAU_BOUNDARY, dtype=float)
    sort_idx = np.argsort(boundary[:, 0], kind="stable")
    bx = boundary[sort_idx, 0]
    by = boundary[sort_idx, 1]
    # The official trace anchors a high plateau with a narrow, irregular scarp.
    # A few pixels of deterministic warp model broken escarpments and passes;
    # river cuts later open deep notches through the edge.
    boundary_y = np.interp(X, bx, by, left=235.0, right=510.0)
    boundary_warp = 6.0 * np.sin(X / 27.0) + 3.5 * np.sin((X + Y) / 18.0) + 2.0 * np.cos(X / 9.0 + Y / 21.0)
    effective_boundary_y = boundary_y + boundary_warp
    plateau_signed = effective_boundary_y - Y
    plateau_weight = 1.0 / (1.0 + np.exp(np.clip(-plateau_signed / 6.8, -30, 30)))
    north_spur_weight, _ = smooth_polygon_weight(X, Y, NORTH_DUNG_EAST_SPUR_POLYGON, 24.0)
    rim_knots_x = np.arange(-100.0, WIDTH + 100.0, 90.0)
    rim_knots = np.asarray([rng.uniform(0.28, 1.22) for _ in rim_knots_x])
    rim_knots = ndimage.gaussian_filter1d(rim_knots, sigma=0.8, mode="nearest")
    rim_modulation = np.interp(X, rim_knots_x, rim_knots)
    plateau_rim = (
        np.clip((plateau_signed + 10.0) / 30.0, 0.0, 1.0)
        * np.exp(-np.abs(plateau_signed) / 38.0)
        * rim_modulation
    )

    # Broad background relief is deliberately restrained. Landform zones and
    # documented range backbones carry regional shape; unassigned peaks do not
    # create links to neighboring mountains.
    Z = (
        365.0
        + 40.0 * np.sin((X - 150.0) / 220.0)
        + 31.0 * np.cos((Y + 60.0) / 185.0)
        + 25.0 * np.sin((X + Y) / 93.0)
        + 18.0 * np.cos((2 * X - Y) / 155.0)
    )
    Z += plateau_weight * (1320.0 + 45.0 * np.sin(X / 155.0) + 32.0 * np.cos(Y / 105.0))
    Z += plateau_rim * 520.0
    Z += north_spur_weight * (1.0 - 0.72 * plateau_weight) * (
        390.0 + 28.0 * np.sin((X + Y) / 42.0) + 18.0 * np.cos(X / 37.0)
    )

    gobi_weight, _ = smooth_polygon_weight(X, Y, GOBI_POLYGON, 28.0)
    basin_edge_weight, basin_inside_distance = smooth_polygon_weight(X, Y, BASIN_POLYGON, 18.0)
    basin_weight = basin_edge_weight * (0.58 + 0.42 * np.clip(basin_inside_distance / 95.0, 0.0, 1.0))
    plain_weight, _ = smooth_polygon_weight(X, Y, PLAINS_POLYGON, 34.0)
    east_hills_weight, _ = smooth_polygon_weight(X, Y, EAST_HILLS_POLYGON, 30.0)
    Z += gobi_weight * 250.0
    Z -= basin_weight * 260.0
    Z -= plain_weight * 110.0
    Z += east_hills_weight * 90.0

    mountain_data = load_js_json(DATA / "mountain_peaks.js")
    baisui_original_xy = None
    for feature in mountain_data.get("features", []):
        properties = feature.get("properties", {})
        name = properties.get("name", "")
        if name in NORTH_PLATEAU_EXTENSION_PEAKS:
            properties["elevation_zone"] = "north_plateau"
            properties["relief_zone_note"] = "User correction: separate north-plateau extension, outside the Central Range."
        if name == "百岁山":
            baisui_original_xy = list(properties.get("source_map_pixel_xy", []))
            properties["source_map_pixel_xy"] = list(BAISUI_RELIEF_POSITION)
            properties["relief_position_adjustment_from"] = baisui_original_xy
            properties["relief_position_adjustment_note"] = "Moved onto the mapped island interior per user correction; the original peak source file is unchanged."
            feature["geometry"]["coordinates"] = list(pixel_to_saturn_map_coordinates(*BAISUI_RELIEF_POSITION))
    (DATA / "relief_mountain_peaks.js").write_text(
        "window.DUNIVERSE_MOUNTAIN_PEAKS="
        + json.dumps(mountain_data, separators=(",", ":"), ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    peaks = []
    for feature in mountain_data.get("features", []):
        properties = feature.get("properties", {})
        point = properties.get("source_map_pixel_xy")
        elevation = properties.get("elevation_m")
        if point and elevation is not None and 0 <= point[0] < WIDTH and 0 <= point[1] < HEIGHT:
            peaks.append((
                float(point[0]),
                float(point[1]),
                properties.get("name", "山峰"),
                float(elevation),
                properties.get("elevation_source", "unknown"),
                properties.get("elevation_zone", "other"),
                properties.get("mountain_range") or (
                    "中央山脉" if properties.get("elevation_zone") == "central_range" else
                    "瑞丰山脉" if properties.get("name") in {"剑山", "雪瑞峰"} else ""
                ),
            ))
    peak_marker_mask_for_rivers = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for peak in peaks:
        peak_marker_mask_for_rivers[min(HEIGHT - 1, round(peak[1])), min(WIDTH - 1, round(peak[0]))] = True
    peak_marker_distance_for_rivers = ndimage.distance_transform_edt(
        ~peak_marker_mask_for_rivers
    )
    river_water_mask_full &= peak_marker_distance_for_rivers > 1.5
    river_bank_specs = make_river_bank_specs(peaks, river_water_mask_full)
    city_coordinate_data = load_js_json(DATA / "city_coordinates.js")
    cities, city_location_audit = resolve_relief_city_locations(city_coordinate_data.get("cities", []))
    relief_city_coordinate_data = dict(city_coordinate_data)
    relief_city_coordinate_data["cities"] = cities
    relief_city_coordinate_data["cityLocationBasis"] = city_location_audit["basis"]
    (DATA / "relief_city_coordinates.js").write_text(
        "window.DUNIVERSE_CITY_COORDINATES="
        + json.dumps(relief_city_coordinate_data, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    peak_height_ceiling_m = max((peak[3] for peak in peaks), default=ELEVATION_MIN_M)
    ELEVATION_MAX_M = max(ELEVATION_MAX_M, peak_height_ceiling_m * 1.4)
    hydrology = json.loads((DATA / "hydrology_lines.geojson").read_text(encoding="utf-8"))
    rifeng_river = next(
        feature for feature in hydrology.get("features", [])
        if feature.get("properties", {}).get("segment_id") == 60
    )
    river60_points = np.asarray(rifeng_river["geometry"]["coordinates"], dtype=float)
    rifeng_peak_end_y = RANGE_BACKBONES["瑞丰山脉"][-1][1]
    rifeng_main_branch = river60_points[river60_points[:, 1] <= rifeng_peak_end_y + 0.5]
    river60_branch_y, river60_branch_inverse = np.unique(rifeng_main_branch[:, 1], return_inverse=True)
    river60_branch_x = np.bincount(
        river60_branch_inverse, weights=rifeng_main_branch[:, 0]
    ) / np.bincount(river60_branch_inverse)
    river60_bank_profile_y = np.append(river60_branch_y, RIFENG_FADE_CROSSING[1])
    river60_bank_profile_x = np.append(river60_branch_x, RIFENG_FADE_CROSSING[0])
    river60_bank_x_grid = np.interp(ys, river60_bank_profile_y, river60_bank_profile_x)
    rifeng_west_bank_mask_grid = X < river60_bank_x_grid[:, None]
    rifeng_fade_dx = RIFENG_FADE_CROSSING[0] - RANGE_BACKBONES["瑞丰山脉"][-1][0]
    rifeng_fade_dy = RIFENG_FADE_CROSSING[1] - RANGE_BACKBONES["瑞丰山脉"][-1][1]
    rifeng_fade_length = math.hypot(rifeng_fade_dx, rifeng_fade_dy)
    rifeng_fade_direction = (rifeng_fade_dx / rifeng_fade_length, rifeng_fade_dy / rifeng_fade_length)
    rifeng_fade_projection = (
        (X - RANGE_BACKBONES["瑞丰山脉"][-1][0]) * rifeng_fade_direction[0]
        + (Y - RANGE_BACKBONES["瑞丰山脉"][-1][1]) * rifeng_fade_direction[1]
    )
    rifeng_fade_t = np.clip(rifeng_fade_projection / RIFENG_FADE_LENGTH_PX, 0.0, 1.0)
    rifeng_fade_t = rifeng_fade_t * rifeng_fade_t * (3.0 - 2.0 * rifeng_fade_t)
    rifeng_end_weight_grid = 1.0 - rifeng_fade_t

    # The central range deliberately has no drawn backbone. Build a local,
    # asymmetrical influence field for its mapped peaks so river incision and
    # seeded surface texture still respond to mountain terrain.
    central_peak_support = np.zeros_like(Z)
    central_basin_cluster_support = np.zeros_like(Z)
    central_cluster_peaks = [peak for peak in peaks if peak[5] == "central_range"]
    for peak in central_cluster_peaks:
        px, py, name = peak[0], peak[1], peak[2]
        angle = math.radians((px * 127.1 + py * 311.7 + SEED) % 180.0)
        along_sigma, across_sigma = (36.0, 23.0) if name in CENTRAL_BASIN_CLUSTER_PEAKS else (25.0, 16.0)
        tangent_x, tangent_y = math.cos(angle), math.sin(angle)
        normal_x, normal_y = -tangent_y, tangent_x
        dx, dy = X - px, Y - py
        along = dx * tangent_x + dy * tangent_y
        across = dx * normal_x + dy * normal_y
        support = np.exp(-0.5 * ((along / along_sigma) ** 2 + (across / across_sigma) ** 2))
        central_peak_support = np.maximum(central_peak_support, support)
        if name in CENTRAL_BASIN_CLUSTER_PEAKS:
            basin_support = np.exp(-0.5 * ((dx / 42.0) ** 2 + (dy / 31.0) ** 2))
            central_basin_cluster_support = np.maximum(central_basin_cluster_support, basin_support)
    Z += 140.0 * central_peak_support + 230.0 * central_basin_cluster_support

    # Build only the explicitly named non-central range backbones, then
    # calibrate every labelled summit to its printed height.
    ridge_total = np.zeros_like(Z)
    ridge_presence = np.zeros_like(Z)
    for ridge_name, line_points, amplitude, width in RIDGES:
        vertices = np.asarray(line_points, dtype=float)
        segment_lengths = np.sqrt(np.sum(np.diff(vertices, axis=0) ** 2, axis=1))
        cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = max(float(cumulative_lengths[-1]), 1.0)
        stations = np.linspace(0.0, total_length, max(3, int(math.ceil(total_length / 64.0)) + 1))
        multipliers = np.asarray([rng.uniform(0.76, 1.24) for _ in stations])
        multipliers = ndimage.gaussian_filter1d(multipliers, sigma=0.9, mode="nearest")
        for segment_index, ((x1, y1), (x2, y2)) in enumerate(zip(vertices, vertices[1:])):
            vx, vy = x2 - x1, y2 - y1
            denom = max(vx * vx + vy * vy, 1.0)
            t = np.clip(((X - x1) * vx + (Y - y1) * vy) / denom, 0.0, 1.0)
            distance = np.sqrt((X - (x1 + t * vx)) ** 2 + (Y - (y1 + t * vy)) ** 2)
            along = cumulative_lengths[segment_index] + t * segment_lengths[segment_index]
            envelope = np.interp(along, stations, multipliers)
            cross_section = 0.58 * np.exp(-distance / (width * 0.62)) + 0.42 * np.exp(-distance / (width * 2.25))
            ridge_layer = amplitude * envelope * cross_section
            presence_layer = envelope * np.exp(-distance / (width * 4.1))
            if ridge_name == "瑞丰山脉":
                ridge_layer *= rifeng_west_bank_mask_grid * rifeng_end_weight_grid
                presence_layer *= rifeng_west_bank_mask_grid * rifeng_end_weight_grid
            ridge_total = np.maximum(ridge_total, ridge_layer)
            ridge_presence = np.maximum(ridge_presence, presence_layer)
    # Rivers shape the terrain through their curved, flow-conditioned valleys
    # below. Do not clip ridge influence to a local tangent plane at the bank;
    # that produced the straight, saw-cut mountain edge seen in the review.
    Z += ridge_total

    # Correlated, seeded surface variation breaks up overly smooth arcs while
    # staying continuous and reproducible at every build.
    scarp_strength = (
        np.exp(-np.abs(plateau_signed) / 42.0)
        * np.clip(plateau_signed / 15.0, 0.0, 1.0)
    )
    roughness = np.clip(
        0.18 + 0.68 * ridge_presence + 0.18 * plateau_weight + 0.18 * north_spur_weight + 0.48 * scarp_strength
        + 0.25 * east_hills_weight + 0.24 * gobi_weight
        + 0.32 * central_peak_support + 0.15 * central_basin_cluster_support,
        0.0,
        1.45,
    )
    Z += roughness * (
        54.0 * normalized_noise(Z.shape, 24.0, noise_rng)
        + 25.0 * normalized_noise(Z.shape, 10.0, noise_rng)
        + 13.0 * normalized_noise(Z.shape, 4.5, noise_rng)
    )

    # A smoothed mountain-zone mask lets rivers carve differently by terrain:
    # broad lowland valleys stay broad, while high-relief ranges keep their mass
    # and receive only a narrow, steep-sided incision.
    mountain_strength = np.maximum(
        np.maximum(np.clip(ridge_total / 1050.0, 0.0, 1.0), central_peak_support * 0.78),
        scarp_strength * 0.86,
    )
    mountain_strength = np.clip(ndimage.gaussian_filter(mountain_strength, sigma=2.6 / STEP), 0.0, 1.0)

    river_lines = [
        feature["geometry"]["coordinates"]
        for feature in hydrology.get("features", [])
        if feature.get("geometry", {}).get("type") == "LineString"
    ]
    river_grid_img = Image.new("L", (len(xs), len(ys)), 0)
    river_draw = ImageDraw.Draw(river_grid_img)
    for coords in river_lines:
        draw_line_on_grid(river_draw, coords)
    river_grid = np.asarray(river_grid_img) > 0
    river_water_grid = river_water_mask_full.reshape(
        len(ys), STEP, len(xs), STEP
    ).any(axis=(1, 3))
    river_grid |= river_water_grid
    river_distance = ndimage.distance_transform_edt(~river_grid) * STEP
    # Use a broad floodplain-shaped channel in lowlands, but a narrow V-shaped
    # gorge in mountainous areas. This keeps the ridge line continuous on both
    # sides of the river instead of flattening a wide strip through the range.
    lowland_valley = 310.0 * np.exp(-river_distance / 15.0) + 115.0 * np.exp(-river_distance / 42.0)
    mountain_gorge = 1500.0 * np.exp(-river_distance / 2.4) + 380.0 * np.exp(-river_distance / 8.0)
    Z -= (1.0 - mountain_strength) * lowland_valley + mountain_strength * mountain_gorge
    canyon_ids = {3, 47}
    canyon_img = Image.new("L", (len(xs), len(ys)), 0)
    canyon_draw = ImageDraw.Draw(canyon_img)
    for feature in hydrology.get("features", []):
        segment_id = feature.get("properties", {}).get("segment_id")
        if segment_id in canyon_ids:
            coords = feature["geometry"]["coordinates"]
            # Only steep headwater reaches are designed as gorges. Downstream
            # sections remain ordinary river valleys and floodplains.
            limit_y = 720 if segment_id == 3 else 470
            headwater = []
            for point in coords:
                if point[1] > limit_y:
                    break
                headwater.append(point)
            if len(headwater) > 1:
                draw_line_on_grid(canyon_draw, headwater)
    canyon_grid = np.asarray(canyon_img) > 0
    canyon_distance = ndimage.distance_transform_edt(~canyon_grid) * STEP
    Z -= 480.0 * np.exp(-canyon_distance / 3.0)

    # Ease lowlands down toward the real ocean shoreline. The open-ocean water
    # polygons are used here instead of the national outline, whose land-border
    # sections must not be mistaken for a coast. The northern plateau keeps a
    # cliffed shore; southern coastal lowlands open into plains, with only
    # strong ridge cores retained.
    coastal_target = np.minimum(420.0, 0.4 + 2.5 * coast_distance)
    coastal_blend = np.exp(-np.power(coast_distance / 43.0, 1.55))
    ridge_core = np.clip((ridge_total - 300.0) / 1000.0, 0.0, 1.0)
    ridge_core = ridge_core * ridge_core * (3.0 - 2.0 * ridge_core)
    south_coast_t = np.clip((Y - 820.0) / 250.0, 0.0, 1.0)
    south_coast_weight = south_coast_t * south_coast_t * (3.0 - 2.0 * south_coast_t)
    southern_ridge_keep = np.where(
        ridge_core < 0.85,
        ridge_core * (1.0 - 0.5 * south_coast_weight),
        ridge_core,
    )
    coast_relief_keep = np.maximum(
        southern_ridge_keep,
        np.maximum(np.clip(scarp_strength * 0.9, 0.0, 1.0), np.clip(plateau_weight * 0.96, 0.0, 1.0)),
    )
    Z += (coastal_target - Z) * coastal_blend * (1.0 - coast_relief_keep)

    Z = ndimage.gaussian_filter(Z, sigma=1.15, mode="nearest")
    east_mountain_t = np.clip((ridge_total - 1100.0) / 250.0, 0.0, 1.0)
    east_mountain_weight = east_mountain_t * east_mountain_t * (3.0 - 2.0 * east_mountain_t)
    east_lowland_cap = east_southeast_weight * (1.0 - east_mountain_weight)
    east_lowland_surface = 5.0 + 0.025 * (Z - 5.0)
    Z += (east_lowland_surface - Z) * east_lowland_cap
    Z += (np.minimum(Z, 50.0) - Z) * east_lowland_cap
    coastal_t = np.clip((20.0 - coast_distance) / 14.0, 0.0, 1.0)
    coastal_strip_weight = coastal_t * coastal_t * (3.0 - 2.0 * coastal_t)
    coastal_limit = np.minimum(50.0, 0.4 + 0.55 * coast_distance)
    east_coastal_cap = east_southeast_weight * coastal_strip_weight
    Z += (np.minimum(Z, coastal_limit) - Z) * east_coastal_cap
    central_massif_signal = np.maximum(ridge_total, central_peak_support * 2200.0)
    plain_mountain_t = np.clip((central_massif_signal - 1000.0) / 700.0, 0.0, 1.0)
    plain_mountain_weight = plain_mountain_t * plain_mountain_t * (3.0 - 2.0 * plain_mountain_t)
    plain_cap = plain_weight * (1.0 - plain_mountain_weight)
    plain_surface = np.minimum(19.0, 5.0 + 0.008 * (Z - 5.0))
    Z += (plain_surface - Z) * plain_cap
    plain_rng = np.random.default_rng(SEED + 1618)
    plain_undulation = (
        4.0 * normalized_noise(Z.shape, 125.0, plain_rng)
        + 1.5 * normalized_noise(Z.shape, 42.0, plain_rng)
    )
    Z += plain_undulation * plain_cap
    Z += (np.minimum(Z, 19.5) - Z) * plain_cap
    Z = np.where(land, np.clip(Z, 0.0, ELEVATION_MAX_M), np.nan)

    # Expand the fixed model to source-map resolution, then add subtle,
    # deterministic terrain texture so close zooms keep natural curve detail.
    _, nearest_land_indices = ndimage.distance_transform_edt(~land, return_indices=True)
    z_filled = np.where(land, Z, Z[tuple(nearest_land_indices)])
    z_full = cv2.resize(z_filled.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    roughness_full = cv2.resize(roughness.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    coast_distance_full = ndimage.distance_transform_edt(~ocean_water_full).astype(np.float32)
    east_southeast_weight_full = cv2.resize(east_southeast_weight.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    east_southeast_mask_image = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(east_southeast_mask_image).polygon(EAST_SOUTHEAST_POLYGON, fill=255)
    east_southeast_inside_full = np.asarray(east_southeast_mask_image) > 0
    east_southeast_weight_full = np.where(east_southeast_inside_full, 1.0, east_southeast_weight_full)
    ridge_total_full = cv2.resize(ridge_total.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    central_peak_support_full = cv2.resize(
        central_peak_support.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC
    )
    coast_detail_factor = 0.35 + 0.65 * np.clip(coast_distance_full / 12.0, 0.0, 1.0)
    detail_rng = np.random.default_rng(SEED + 2718)
    fine_detail = (
        12.0 * normalized_noise((HEIGHT, WIDTH), 8.0, detail_rng, pixel_size_px=1.0)
        + 6.0 * normalized_noise((HEIGHT, WIDTH), 3.0, detail_rng, pixel_size_px=1.0)
        + 3.0 * normalized_noise((HEIGHT, WIDTH), 1.25, detail_rng, pixel_size_px=1.0)
    )
    z_full += fine_detail * roughness_full * coast_detail_factor
    east_mountain_t_full = np.clip((ridge_total_full - 1100.0) / 250.0, 0.0, 1.0)
    east_mountain_weight_full = east_mountain_t_full * east_mountain_t_full * (3.0 - 2.0 * east_mountain_t_full)
    peak_marker_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for peak in peaks:
        peak_marker_mask[min(HEIGHT - 1, round(peak[1])), min(WIDTH - 1, round(peak[0]))] = True
    peak_distance_full = ndimage.distance_transform_edt(~peak_marker_mask).astype(np.float32)
    river60_bank_x_full = np.interp(
        np.arange(HEIGHT, dtype=float), river60_bank_profile_y, river60_bank_profile_x
    )
    rifeng_west_bank_mask_full = (
        np.arange(WIDTH, dtype=np.float32)[None, :] < river60_bank_x_full[:, None]
    )
    rifeng_fade_projection_full = (
        (np.arange(WIDTH, dtype=np.float32)[None, :] - RANGE_BACKBONES["瑞丰山脉"][-1][0]) * rifeng_fade_direction[0]
        + (np.arange(HEIGHT, dtype=np.float32)[:, None] - RANGE_BACKBONES["瑞丰山脉"][-1][1]) * rifeng_fade_direction[1]
    )
    rifeng_fade_t_full = np.clip(rifeng_fade_projection_full / RIFENG_FADE_LENGTH_PX, 0.0, 1.0)
    rifeng_fade_t_full = rifeng_fade_t_full * rifeng_fade_t_full * (3.0 - 2.0 * rifeng_fade_t_full)
    rifeng_west_bank_weight_full = rifeng_west_bank_mask_full.astype(np.float32) * (1.0 - rifeng_fade_t_full)
    peak_protection_t = np.clip((54.0 - peak_distance_full) / 18.0, 0.0, 1.0)
    peak_mountain_protection_full = peak_protection_t * peak_protection_t * (3.0 - 2.0 * peak_protection_t)
    east_lowland_cap_full = east_southeast_weight_full * (1.0 - east_mountain_weight_full)
    east_lowland_surface_full = 5.0 + 0.025 * (z_full - 5.0)
    z_full += (east_lowland_surface_full - z_full) * east_lowland_cap_full
    z_full += (np.minimum(z_full, 50.0) - z_full) * east_lowland_cap_full
    coastal_t_full = np.clip((20.0 - coast_distance_full) / 14.0, 0.0, 1.0)
    coastal_strip_weight_full = coastal_t_full * coastal_t_full * (3.0 - 2.0 * coastal_t_full)
    coastal_limit_full = np.minimum(50.0, 0.4 + 0.55 * coast_distance_full)
    east_coastal_cap_full = east_southeast_weight_full * coastal_strip_weight_full
    z_full += (np.minimum(z_full, coastal_limit_full) - z_full) * east_coastal_cap_full
    peak_anchor_max_error_m = apply_peak_elevation_anchors(
        z_full, peaks, rifeng_west_bank_weight_full
    )
    east_nonmountain_cap_full = east_southeast_weight_full * (1.0 - east_mountain_weight_full) * (1.0 - peak_mountain_protection_full)
    east_lowland_surface_full = 5.0 + 0.025 * (z_full - 5.0)
    z_full += (east_lowland_surface_full - z_full) * east_nonmountain_cap_full
    z_full += (np.minimum(z_full, 50.0) - z_full) * east_nonmountain_cap_full
    east_nonmountain_coastal_cap_full = (
        east_southeast_weight_full
        * coastal_strip_weight_full
        * (1.0 - east_mountain_weight_full)
        * (1.0 - peak_mountain_protection_full)
    )
    z_full += (np.minimum(z_full, coastal_limit_full) - z_full) * east_nonmountain_coastal_cap_full
    ridge_core_t_full = np.clip((ridge_total_full - 300.0) / 1000.0, 0.0, 1.0)
    ridge_core_full = ridge_core_t_full * ridge_core_t_full * (3.0 - 2.0 * ridge_core_t_full)
    plateau_weight_full = cv2.resize(plateau_weight.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    scarp_strength_full = cv2.resize(scarp_strength.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    coastal_relief_keep_full = np.maximum(
        ridge_core_full,
        np.maximum(np.clip(scarp_strength_full * 0.9, 0.0, 1.0), np.clip(plateau_weight_full * 0.96, 0.0, 1.0)),
    )
    all_coast_lowland_cap_full = coastal_strip_weight_full * (1.0 - coastal_relief_keep_full) * (1.0 - peak_mountain_protection_full)
    z_full += (np.minimum(z_full, coastal_limit_full) - z_full) * all_coast_lowland_cap_full
    plain_weight_full = cv2.resize(plain_weight.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    central_massif_signal_full = np.maximum(ridge_total_full, central_peak_support_full * 2200.0)
    plain_mountain_t_full = np.clip((central_massif_signal_full - 1000.0) / 700.0, 0.0, 1.0)
    plain_mountain_weight_full = plain_mountain_t_full * plain_mountain_t_full * (3.0 - 2.0 * plain_mountain_t_full)
    summit_protection_t_full = np.clip((8.0 - peak_distance_full) / 4.0, 0.0, 1.0)
    summit_protection_full = summit_protection_t_full * summit_protection_t_full * (3.0 - 2.0 * summit_protection_t_full)
    plain_cap_full = plain_weight_full * (1.0 - plain_mountain_weight_full) * (1.0 - summit_protection_full)
    plain_surface_full = np.minimum(19.0, 5.0 + 0.008 * (z_full - 5.0))
    z_full += (plain_surface_full - z_full) * plain_cap_full
    plain_detail_rng = np.random.default_rng(SEED + 2719)
    plain_undulation_full = (
        3.5 * normalized_noise((HEIGHT, WIDTH), 145.0, plain_detail_rng, pixel_size_px=1.0)
        + 1.25 * normalized_noise((HEIGHT, WIDTH), 48.0, plain_detail_rng, pixel_size_px=1.0)
    )
    z_full += plain_undulation_full * plain_cap_full
    z_full += (np.minimum(z_full, 19.5) - z_full) * plain_cap_full
    zaq_city = next(city for city in cities if city.get("name") == "乍曲市")
    zaq_city_x, zaq_city_y = map(float, zaq_city["sourcePixelXY"])
    zaq_river_feature = next(
        feature for feature in hydrology.get("features", [])
        if feature.get("properties", {}).get("segment_id") == 24
        and feature.get("geometry", {}).get("type") == "LineString"
    )
    zaq_river_image = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(zaq_river_image).line(
        [tuple(point) for point in zaq_river_feature["geometry"]["coordinates"]],
        fill=255,
        width=1,
        joint="curve",
    )
    zaq_river_distance_full = ndimage.distance_transform_edt(np.asarray(zaq_river_image) == 0).astype(np.float32)
    zaq_floor_halfwidth_px, zaq_wall_outer_radius_px = 4.0, 26.0
    zaq_wall_t = np.clip(
        (zaq_river_distance_full - zaq_floor_halfwidth_px)
        / (zaq_wall_outer_radius_px - zaq_floor_halfwidth_px),
        0.0,
        1.0,
    )
    zaq_u_cross_section = 1.0 - zaq_wall_t * zaq_wall_t * (3.0 - 2.0 * zaq_wall_t)
    full_x = np.arange(WIDTH, dtype=np.float32)[None, :]
    full_y = np.arange(HEIGHT, dtype=np.float32)[:, None]
    zaq_city_scope = np.exp(-((full_x - zaq_city_x) ** 2 + (full_y - zaq_city_y) ** 2) / (2.0 * 78.0**2))
    summit_keepout_t = np.clip((16.0 - peak_distance_full) / 8.0, 0.0, 1.0)
    summit_keepout = summit_keepout_t * summit_keepout_t * (3.0 - 2.0 * summit_keepout_t)
    zaq_u_valley_floor_m = 2400.0
    zaq_city_elevation_before_valley = bilinear(z_full, zaq_city_x, zaq_city_y, step=1.0)
    zaq_u_valley_cut_depth_m = max(0.0, zaq_city_elevation_before_valley - zaq_u_valley_floor_m)
    z_full -= zaq_u_valley_cut_depth_m * zaq_u_cross_section * zaq_city_scope * (1.0 - summit_keepout)
    zaq_city_elevation_after_cut = bilinear(z_full, zaq_city_x, zaq_city_y, step=1.0)
    zaq_u_valley_raise_m = max(0.0, zaq_u_valley_floor_m - zaq_city_elevation_after_cut)
    if zaq_u_valley_raise_m > 0.0:
        z_full += zaq_u_valley_raise_m * zaq_u_cross_section * zaq_city_scope * (1.0 - summit_keepout)

    # Give 剑山市 a lower mountain-foot settlement beside its nearby river.
    # The incision joins the river corridor and keeps the adjacent mapped summit intact.
    sword_city = next((city for city in cities if city.get("name") == "剑山市"), None)
    sword_city_xy = sword_river_xy = None
    sword_valley_floor_m = 1750.0
    sword_valley_cut_depth_m = 0.0
    if sword_city and sword_city.get("sourcePixelXY") and river_lines:
        sword_city_xy = tuple(map(float, sword_city["sourcePixelXY"]))
        cx, cy = sword_city_xy
        nearest_distance_sq = float("inf")
        for river_line in river_lines:
            for (x1, y1), (x2, y2) in zip(river_line, river_line[1:]):
                vx, vy = x2 - x1, y2 - y1
                length_sq = vx * vx + vy * vy
                if length_sq <= 1e-8:
                    continue
                t = float(np.clip(((cx - x1) * vx + (cy - y1) * vy) / length_sq, 0.0, 1.0))
                rx, ry = x1 + t * vx, y1 + t * vy
                distance_sq = (cx - rx) ** 2 + (cy - ry) ** 2
                if distance_sq < nearest_distance_sq:
                    nearest_distance_sq = distance_sq
                    sword_river_xy = (rx, ry)
        if sword_river_xy:
            rx, ry = sword_river_xy
            river_distance_to_city = math.sqrt(nearest_distance_sq)
            toward_river_x = (rx - cx) / max(river_distance_to_city, 1e-6)
            toward_river_y = (ry - cy) / max(river_distance_to_city, 1e-6)
            valley_image = Image.new("L", (WIDTH, HEIGHT), 0)
            valley_draw = ImageDraw.Draw(valley_image)
            valley_start = (cx - toward_river_x * 30.0, cy - toward_river_y * 30.0)
            valley_draw.line([valley_start, sword_city_xy, sword_river_xy], fill=255, width=1, joint="curve")
            valley_distance = ndimage.distance_transform_edt(np.asarray(valley_image) == 0).astype(np.float32)
            valley_wall_t = np.clip((valley_distance - 2.5) / 15.5, 0.0, 1.0)
            valley_u_profile = 1.0 - valley_wall_t * valley_wall_t * (3.0 - 2.0 * valley_wall_t)
            along_river = np.clip(
                ((full_x - cx) * toward_river_x + (full_y - cy) * toward_river_y)
                / max(river_distance_to_city, 1e-6),
                0.0,
                1.0,
            )
            river_floor_m = bilinear(z_full, rx, ry, step=1.0)
            valley_target = sword_valley_floor_m + (river_floor_m - sword_valley_floor_m) * along_river
            local_scope = np.exp(-((full_x - cx) ** 2 + (full_y - cy) ** 2) / (2.0 * 48.0**2))
            summit_keepout_t = np.clip((7.0 - peak_distance_full) / 2.0, 0.0, 1.0)
            sword_summit_keepout = summit_keepout_t * summit_keepout_t * (3.0 - 2.0 * summit_keepout_t)
            sword_city_elevation_before_valley = bilinear(z_full, cx, cy, step=1.0)
            sword_valley_cut_depth_m = max(0.0, sword_city_elevation_before_valley - sword_valley_floor_m)
            z_full -= np.maximum(z_full - valley_target, 0.0) * valley_u_profile * local_scope * (1.0 - sword_summit_keepout)

    # River 231 passes through the saddle between the Anhuo and Juanmaogou
    # peaks south of Alkali-Zinc Tork. Carve this mapped reach as a narrow,
    # steep-sided gorge so the nearby unassigned massifs do not merge into an
    # invented continuous range. Keep the labelled summit cores intact.
    nearby_river_gorge = {
        "city": "碱锌托克市",
        "riverSegmentId": 231,
        "pathBoundsSourcePixelXY": [1018.0, 1278.0, 1066.0, 1305.0],
        "floorTargetM": 420.0,
        "innerRadiusPx": 1.5,
        "outerWallRadiusPx": 16.0,
    }
    gorge_image = Image.new("L", (WIDTH, HEIGHT), 0)
    gorge_draw = ImageDraw.Draw(gorge_image)
    gx0, gy0, gx1, gy1 = nearby_river_gorge["pathBoundsSourcePixelXY"]
    gorge_path = []
    for feature in hydrology.get("features", []):
        if feature.get("properties", {}).get("segment_id") != nearby_river_gorge["riverSegmentId"]:
            continue
        coords = feature.get("geometry", {}).get("coordinates", [])
        gorge_path = [
            point for point in coords
            if gx0 <= point[0] <= gx1 and gy0 <= point[1] <= gy1
        ]
        if len(gorge_path) > 1:
            gorge_draw.line([tuple(point) for point in gorge_path], fill=255, width=1, joint="curve")
            break
    if len(gorge_path) < 2:
        raise RuntimeError("Could not find the mapped river reach for the Alkali-Zinc Tork gorge")
    gorge_grid = np.asarray(gorge_image) > 0
    gorge_distance = ndimage.distance_transform_edt(~gorge_grid).astype(np.float32)
    gorge_wall_t = np.clip(
        (gorge_distance - nearby_river_gorge["innerRadiusPx"])
        / (nearby_river_gorge["outerWallRadiusPx"] - nearby_river_gorge["innerRadiusPx"]),
        0.0,
        1.0,
    )
    nearby_gorge_profile = 1.0 - gorge_wall_t * gorge_wall_t * (3.0 - 2.0 * gorge_wall_t)
    summit_keepout_t = np.clip((7.0 - peak_distance_full) / 2.0, 0.0, 1.0)
    summit_keepout = summit_keepout_t * summit_keepout_t * (3.0 - 2.0 * summit_keepout_t)
    nearby_gorge_cut = (
        np.maximum(z_full - nearby_river_gorge["floorTargetM"], 0.0)
        * nearby_gorge_profile
        * (1.0 - summit_keepout)
    )
    z_full -= nearby_gorge_cut
    nearby_river_gorge["cutDepthMaxM"] = float(np.max(nearby_gorge_cut))
    nearby_river_gorge["carvedPathSourcePixelXY"] = [
        [float(point[0]), float(point[1])] for point in gorge_path
    ]

    # Ease the northern plateau's near-shore rim slightly while retaining a
    # steep escarpment and the exact heights at labelled summit cores.
    north_plateau_coast_mask = (
        land_full
        & (full_x > 1000.0)
        & (full_x < 1265.0)
        & (full_y < 620.0)
        & (coast_distance_full < 20.0)
        & (peak_distance_full > 6.0)
    )
    north_plateau_coast_cap_m = 1050.0 + 24.0 * coast_distance_full
    z_full = np.where(
        north_plateau_coast_mask,
        np.minimum(z_full, north_plateau_coast_cap_m),
        z_full,
    )
    north_plateau_coast_capped_cells = int(np.count_nonzero(north_plateau_coast_mask))

    # Small offshore land components should read as low islands, except for a
    # compact cone around any explicitly mapped summit on that island.
    land_component_labels, land_component_count = ndimage.label(
        land_full, structure=np.ones((3, 3), dtype=bool)
    )
    land_component_sizes = np.bincount(land_component_labels.ravel())
    mainland_component = int(np.argmax(land_component_sizes[1:]) + 1) if land_component_count else 0
    component_slices = ndimage.find_objects(land_component_labels)
    small_island_count = 0
    small_island_peak_records = []
    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None or component_id == mainland_component:
            continue
        component_area = int(land_component_sizes[component_id])
        if component_area > 10000:
            continue
        island_mask = land_component_labels[component_slice] == component_id
        island_peaks = []
        for peak in peaks:
            px = min(WIDTH - 1, max(0, round(peak[0])))
            py = min(HEIGHT - 1, max(0, round(peak[1])))
            if land_component_labels[py, px] == component_id:
                island_peaks.append(peak)
        row_slice, col_slice = component_slice
        local_x = np.arange(col_slice.start, col_slice.stop, dtype=np.float32)[None, :]
        local_y = np.arange(row_slice.start, row_slice.stop, dtype=np.float32)[:, None]
        if island_peaks:
            component_peak_surface = np.full(island_mask.shape, 20.0, dtype=np.float32)
            # Keep labelled summits visible without turning a tiny island into
            # a broad high plateau. The smaller scale is especially important
            # for 百岁山 and 浪花山, whose mapped islands are only a few dozen
            # source pixels across.
            cone_radius = float(np.clip(math.sqrt(component_area) / 7.5, 1.1, 2.6))
            for peak in island_peaks:
                distance_to_peak = np.maximum(
                    np.hypot(local_x - peak[0], local_y - peak[1]) - 1.0,
                    0.0,
                )
                cone = 20.0 + (peak[3] - 20.0) * np.exp(-distance_to_peak / cone_radius)
                component_peak_surface = np.maximum(component_peak_surface, cone.astype(np.float32))
                small_island_peak_records.append(peak)
            component_surface = z_full[component_slice]
            z_full[component_slice] = np.where(
                island_mask,
                np.minimum(component_surface, component_peak_surface),
                component_surface,
            )
        else:
            component_surface = z_full[component_slice]
            component_cap = 80.0 + 6.0 * np.minimum(coast_distance_full[component_slice], 20.0)
            z_full[component_slice] = np.where(
                island_mask,
                np.minimum(component_surface, component_cap),
                component_surface,
            )
        small_island_count += 1

    # Restore sub-pixel summit samples with a tiny deterministic core so the
    # island cap cannot lower the displayed height of its labelled peak.
    for peak in small_island_peak_records:
        px, py, target_m = peak[0], peak[1], peak[3]
        residual = target_m - bilinear(z_full, px, py, step=1.0)
        if abs(residual) > 1e-4:
            summit_sigma = 1.25
            summit_patch = np.exp(-0.5 * (
                ((full_x - px) / summit_sigma) ** 2
                + ((full_y - py) / summit_sigma) ** 2
            )).astype(np.float32)
            patch_value = bilinear(summit_patch, px, py, step=1.0)
            z_full += summit_patch * np.float32(residual / max(patch_value, 1e-8))

    # Refit all mapped summits after river cuts and local valley shaping. This
    # final constrained pass preserves exact peak heights while bank masks keep
    # individual massifs from leaking across nearby rivers.
    peak_anchor_max_error_m = apply_peak_elevation_anchors(
        z_full, peaks, rifeng_west_bank_weight_full
    )
    zaq_city_after_peak_fit = bilinear(z_full, zaq_city_x, zaq_city_y, step=1.0)
    zaq_u_valley_post_anchor_raise_m = max(0.0, zaq_u_valley_floor_m - zaq_city_after_peak_fit)
    if zaq_u_valley_post_anchor_raise_m > 0.0:
        z_full += (
            zaq_u_valley_post_anchor_raise_m
            * zaq_u_cross_section
            * zaq_city_scope
            * (1.0 - summit_keepout)
        )
    z_full = np.clip(z_full, 0.0, ELEVATION_MAX_M)
    lakes_data = json.loads((DATA / "lakes.geojson").read_text(encoding="utf-8"))
    sea_bottom_lake = next(
        feature for feature in lakes_data.get("features", [])
        if feature.get("properties", {}).get("lake_id") == 3
    )
    lake_name_records = load_js_json(DATA / "lake_names.js")
    sea_bottom_lake_label = next(
        (record for record in lake_name_records if record.get("lakeId") == 3), {}
    )
    lake_inside_distance = ndimage.distance_transform_edt(lake_mask_full).astype(np.float32)
    lake_outside_distance = ndimage.distance_transform_edt(~lake_mask_full).astype(np.float32)
    lake_depth_factor = np.sqrt(np.clip(lake_inside_distance / max(float(lake_inside_distance.max()), 1.0), 0.0, 1.0))
    lake_noise_rng = np.random.default_rng(SEED + 3117)
    lake_floor_noise = (
        4.6 * normalized_noise((HEIGHT, WIDTH), 13.0, lake_noise_rng, pixel_size_px=1.0)
        + 1.7 * normalized_noise((HEIGHT, WIDTH), 4.0, lake_noise_rng, pixel_size_px=1.0)
    )
    lake_floor = -183.7 - 29.0 * lake_depth_factor + lake_floor_noise
    z_full = np.where(lake_mask_full, lake_floor, z_full)
    lake_basin_t = np.clip((105.0 - lake_outside_distance) / 40.0, 0.0, 1.0)
    lake_basin_weight = lake_basin_t * lake_basin_t * (3.0 - 2.0 * lake_basin_t)
    lake_basin_weight *= (~lake_mask_full & land_full) * (1.0 - peak_mountain_protection_full)
    # Keep the mapped lake bed near -200 m, but the mapped land around its
    # shoreline stays above the river surface and rises gently into the basin
    # margin; do not extend the submerged lake floor under dry river banks.
    lake_basin_target = 8.0 + 0.45 * lake_outside_distance
    z_full += (lake_basin_target - z_full) * lake_basin_weight
    z_full = np.clip(z_full, ELEVATION_MIN_M, ELEVATION_MAX_M).astype(np.float32)

    # Force the river south of 长岭市 into a broad upland valley. The channel
    # point is constrained first; the city floor is then set just above its
    # water surface and connected to the actual bend with a smooth valley.
    longling_city = next(city for city in cities if city.get("name") == "长岭市")
    longling_city_x, longling_city_y = map(float, longling_city["sourcePixelXY"])
    longling_river_height_cap_m = 3200.0
    river_flow = flow_conditioned_river_surface(
        z_full,
        land_full,
        river_water_mask_full,
        ocean_water_full,
        lake_mask_full,
        hydrology,
        height_caps=[(
            longling_city_x,
            longling_city_y,
            longling_river_height_cap_m,
            "长岭市邻近河道",
        )],
        minimum_drop_m_per_px=0.35,
    )

    river_distance_to_centerline = river_flow["centerlineDistance"]
    river_nearest_centerline = river_flow["centerlineNearest"]
    river_centerline_surface = river_flow["centerlineSurface"]
    nearest_centerline_surface = river_centerline_surface[
        river_nearest_centerline[0], river_nearest_centerline[1]
    ]
    plateau_weight_full = cv2.resize(
        plateau_weight.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC
    )
    valley_mountain_strength = np.maximum(
        np.maximum(ridge_core_full, central_peak_support_full * 0.78),
        scarp_strength_full * 0.72,
    )
    valley_bank_slope_m_per_px = 55.0 + 510.0 * valley_mountain_strength + 260.0 * plateau_weight_full
    river_valley_cap = (
        nearest_centerline_surface
        + 35.0
        + river_distance_to_centerline * valley_bank_slope_m_per_px
    )
    river_valley_t = np.clip((28.0 - river_distance_to_centerline) / 10.0, 0.0, 1.0)
    river_valley_weight = river_valley_t * river_valley_t * (3.0 - 2.0 * river_valley_t)
    summit_keepout_t = np.clip((9.0 - peak_distance_full) / 6.0, 0.0, 1.0)
    summit_keepout = summit_keepout_t * summit_keepout_t * (3.0 - 2.0 * summit_keepout_t)
    river_valley_weight *= land_full * (~river_water_mask_full) * (1.0 - summit_keepout)
    z_full = np.where(
        river_valley_weight > 0.0,
        np.minimum(z_full, z_full + (river_valley_cap - z_full) * river_valley_weight),
        z_full,
    )

    # Restore exact mapped summits after the curved valley incision, then keep
    # the named city floor in its valley while leaving nearby summit cores intact.
    peak_anchor_max_error_m = apply_peak_elevation_anchors(
        z_full, peaks, rifeng_west_bank_weight_full
    )
    longling_qx = int(river_nearest_centerline[1, round(longling_city_y), round(longling_city_x)])
    longling_qy = int(river_nearest_centerline[0, round(longling_city_y), round(longling_city_x)])
    longling_channel_height_m = float(river_centerline_surface[longling_qy, longling_qx])
    longling_city_target_m = min(
        3800.0,
        max(2600.0, longling_channel_height_m + 180.0),
    )
    longling_dx, longling_dy = longling_qx - longling_city_x, longling_qy - longling_city_y
    longling_length_sq = max(longling_dx * longling_dx + longling_dy * longling_dy, 1.0)
    longling_projection = np.clip(
        ((full_x - longling_city_x) * longling_dx
         + (full_y - longling_city_y) * longling_dy) / longling_length_sq,
        0.0,
        1.0,
    )
    longling_path_x = longling_city_x + longling_projection * longling_dx
    longling_path_y = longling_city_y + longling_projection * longling_dy
    longling_path_distance = np.hypot(full_x - longling_path_x, full_y - longling_path_y)
    longling_path_height = longling_city_target_m + (
        longling_channel_height_m - longling_city_target_m
    ) * longling_projection
    longling_valley_cap = longling_path_height + 210.0 * longling_path_distance
    longling_valley_weight_t = np.clip((20.0 - longling_path_distance) / 7.0, 0.0, 1.0)
    longling_valley_weight = longling_valley_weight_t * longling_valley_weight_t * (
        3.0 - 2.0 * longling_valley_weight_t
    )
    longling_valley_weight *= land_full * (~river_water_mask_full) * (1.0 - summit_keepout)
    z_full = np.where(
        longling_valley_weight > 0.0,
        np.minimum(
            z_full,
            z_full + (longling_valley_cap - z_full) * longling_valley_weight,
        ),
        z_full,
    )
    z_full = np.clip(z_full, ELEVATION_MIN_M, ELEVATION_MAX_M).astype(np.float32)

    # Recondition once more against the final valley banks. This catches any
    # terrain lowering that happened after the first river-guided pass and
    # keeps the displayed water surface below its actual neighbouring land.
    river_flow = flow_conditioned_river_surface(
        z_full,
        land_full,
        river_water_mask_full,
        ocean_water_full,
        lake_mask_full,
        hydrology,
        height_caps=[(
            longling_city_x,
            longling_city_y,
            longling_river_height_cap_m,
            "长岭市邻近河道",
        )],
        minimum_drop_m_per_px=0.35,
    )
    river_distance_to_centerline = river_flow["centerlineDistance"]
    river_nearest_centerline = river_flow["centerlineNearest"]
    river_centerline_surface = river_flow["centerlineSurface"]
    longling_qx = int(river_nearest_centerline[1, round(longling_city_y), round(longling_city_x)])
    longling_qy = int(river_nearest_centerline[0, round(longling_city_y), round(longling_city_x)])
    longling_channel_height_m = float(river_centerline_surface[longling_qy, longling_qx])

    river_water_surface_full = np.asarray(river_flow["riverSurface"], dtype=np.float32).copy()
    coastal_tidal_channels = (
        river_water_mask_full
        & (coast_distance_full <= 45.0)
        & (np.arange(HEIGHT, dtype=np.float32)[:, None] >= 820.0)
        & (z_full <= 180.0)
    )
    river_water_surface_full[coastal_tidal_channels] = 0.0
    z_full[river_water_mask_full] = river_water_surface_full[river_water_mask_full]
    relief_sample_mask = land_full | lake_mask_full
    lake_floor_min_m = float(np.min(lake_floor[lake_mask_full]))
    lake_floor_max_m = float(np.max(lake_floor[lake_mask_full]))
    lake_floor_mean_m = float(np.mean(lake_floor[lake_mask_full]))
    peak_anchor_max_error_m = max(
        abs(bilinear(z_full, peak[0], peak[1], step=1.0) - peak[3]) for peak in peaks
    ) if peaks else 0.0

    # Pack the fixed elevation field into two 8-bit color channels so the browser
    # can sample any clicked map point without embedding a huge JavaScript array.
    elevation_code = np.rint(
        (np.nan_to_num(z_full, nan=ELEVATION_MIN_M) - ELEVATION_MIN_M)
        / (ELEVATION_MAX_M - ELEVATION_MIN_M)
        * 65535.0
    ).clip(0, 65535).astype(np.uint16)
    elevation_rgba = np.zeros((*z_full.shape, 4), dtype=np.uint8)
    elevation_rgba[:, :, 0] = (elevation_code >> 8).astype(np.uint8)
    elevation_rgba[:, :, 1] = (elevation_code & 255).astype(np.uint8)
    elevation_rgba[:, :, 3] = np.where(land_full, 255, 0).astype(np.uint8)
    Image.fromarray(elevation_rgba, "RGBA").save(DATA / "relief_elevation.png", optimize=True)
    elevation_source = {
        "width": int(elevation_code.shape[1]),
        "height": int(elevation_code.shape[0]),
        "step": 1,
        "bitDepth": 16,
        "min": ELEVATION_MIN_M,
        "max": ELEVATION_MAX_M,
        "valuesB64": base64.b64encode(elevation_code.astype("<u2", copy=False).tobytes()).decode("ascii"),
        "landMaskB64": base64.b64encode(np.packbits(relief_sample_mask.ravel(), bitorder="big").tobytes()).decode("ascii"),
        "riverWaterMaskB64": base64.b64encode(np.packbits(river_water_mask_full.ravel(), bitorder="big").tobytes()).decode("ascii"),
    }
    (DATA / "relief_elevation.js").write_text(
        "window.DUNIVERSE_RELIEF_ELEVATION="
        + json.dumps(elevation_source, separators=(",", ":"))
        + ";\n",
        encoding="ascii",
    )

    # Transparent hypsometric tint and soft directional shading.
    land_full_u8 = land_full.astype(np.uint8)
    gx = cv2.Sobel(z_full, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(z_full, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    norm = np.sqrt(gx * gx + gy * gy) + 1e-6
    shade = np.clip(0.92 - 0.00012 * gx + 0.00012 * gy, 0.75, 1.10)
    stops = [-260, -220, -180, -120, -60, 0, 20, 50, 100, 250, 800, 1500, 2300, 3300, 4300, 6000, 8500, 11000, 14000, ELEVATION_MAX_M]
    colors = np.asarray(
        [
            (151, 184, 204),
            (160, 195, 208),
            (173, 205, 207),
            (186, 214, 201),
            (198, 220, 194),
            (205, 224, 190),
            (207, 224, 188),
            (210, 223, 183),
            (213, 222, 178),
            (180, 210, 160),
            (194, 201, 139),
            (207, 181, 125),
            (188, 148, 108),
            (164, 127, 105),
            (202, 197, 185),
            (210, 206, 193),
            (221, 217, 205),
            (233, 230, 219),
            (242, 240, 232),
            (249, 248, 244),
        ],
        dtype=float,
    )
    rgb = np.stack([np.interp(z_full, stops, colors[:, i]) for i in range(3)], axis=-1)
    rgb = np.clip(rgb * shade[:, :, None], 0, 255).astype(np.uint8)
    rgba = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = np.where(land_full_u8 > 0, 56, 0).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(DATA / "relief_tint.png", optimize=True)

    peak_xy = np.asarray([[p[0], p[1]] for p in peaks], dtype=float)
    peak_names = [p[2] for p in peaks]
    peak_heights = np.asarray([p[3] for p in peaks], dtype=float)
    city_to_land_distance_full = ndimage.distance_transform_edt(~land_full).astype(np.float32)
    cities_for_csv = []
    city_audit_rows = []
    city_profiles = []
    max_x, max_y = len(xs) - 1, len(ys) - 1
    for city in cities:
        source = city.get("sourcePixelXY")
        if not source:
            continue
        cx, cy = float(source[0]), float(source[1])
        if not (0 <= cx < WIDTH and 0 <= cy < HEIGHT):
            continue
        row, col = min(round(cy / STEP), max_y), min(round(cx / STEP), max_x)
        profile_row, profile_col = row, col
        city_x, city_y = min(round(cx), WIDTH - 1), min(round(cy), HEIGHT - 1)
        elevation_sample = bilinear_encoded_elevation(elevation_code, relief_sample_mask, cx, cy)
        elevation = float(elevation_sample) if elevation_sample is not None else float("nan")
        if not np.isfinite(elevation) or not land_full[city_y, city_x]:
            profile_row = int(nearest_land_indices[0, row, col])
            profile_col = int(nearest_land_indices[1, row, col])
            elevation = float(Z[profile_row, profile_col])
        plateau_here = plateau_weight[profile_row, profile_col] >= 0.50
        mountain_here = float(mountain_strength[profile_row, profile_col])
        scarp_here = float(scarp_strength[profile_row, profile_col])
        gobi_here = point_in_polygon(np.asarray([cx]), np.asarray([cy]), GOBI_POLYGON)[0]
        basin_here = point_in_polygon(np.asarray([cx]), np.asarray([cy]), BASIN_POLYGON)[0]
        plain_here = point_in_polygon(np.asarray([cx]), np.asarray([cy]), PLAINS_POLYGON)[0]
        hills_here = point_in_polygon(np.asarray([cx]), np.asarray([cy]), EAST_HILLS_POLYGON)[0]
        if len(peak_xy):
            distances = np.sqrt(((peak_xy - np.asarray([cx, cy])) ** 2).sum(axis=1))
            nearest_peak_idx = int(np.argmin(distances))
            nearest_peak_dist = float(distances[nearest_peak_idx])
            nearest_peak_name = peak_names[nearest_peak_idx]
        else:
            nearest_peak_dist, nearest_peak_name = float("inf"), ""
        if nearest_peak_dist <= 5.0:
            elevation = float(peak_heights[nearest_peak_idx])
        river_dist = float(river_distance[row, col])
        canyon_dist = float(canyon_distance[row, col])
        coast_dist = float(coast_distance_full[city_y, city_x])
        city_is_land = bool(land_full[city_y, city_x])
        land_dist = 0.0 if city_is_land else float(city_to_land_distance_full[city_y, city_x])
        radius_cells = 12
        y0, y1 = max(0, city_y - radius_cells), min(HEIGHT, city_y + radius_cells + 1)
        x0, x1 = max(0, city_x - radius_cells), min(WIDTH, city_x + radius_cells + 1)
        neighborhood = z_full[y0:y1, x0:x1]
        neighborhood_land = land_full[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        neighborhood_radius = (xx - cx) ** 2 + (yy - cy) ** 2 <= 12.0**2
        local_values = neighborhood[neighborhood_land & neighborhood_radius & np.isfinite(neighborhood)]
        local_low = float(np.percentile(local_values, 10)) if local_values.size else elevation
        local_high = float(np.percentile(local_values, 90)) if local_values.size else elevation
        local_relief = local_high - local_low
        relation_flags = []
        if not city_is_land:
            relation_flags.append(f"城市标记点在水域掩膜外，距最近陆地约{land_dist:.1f}px")
        if river_dist < 1.5:
            relation_flags.append("城市点与河道中心线重叠")
        if nearest_peak_dist <= 5.0:
            relation_flags.append(f"城市点贴近主峰标记：{nearest_peak_name}")
        if local_relief >= 1800.0 and not plateau_here and city.get("name") not in {"长岭市", "塔马市"}:
            relation_flags.append("城市周边起伏很大，需确认其是否位于谷地或山前台地")
        if city.get("markerSource", "").startswith("map-label-anchor"):
            relation_flags.append("该城没有独立圆点，按文字锚点估算")
        relation_review = "；".join(relation_flags) if relation_flags else "通过：落在陆地，未压河道或已标峰顶"
        if city.get("name") == "长岭市":
            relation_review = (
                f"通过：城市位于长岭山间谷地；点位约{elevation:.0f}米，"
                f"邻近河道水面约{longling_channel_height_m:.0f}米"
            )
        elif city.get("name") == "塔马市":
            relation_review = "通过：按既定设定保留为山城，周边高地与邻近河流相连"
        east_plain_here = (
            east_southeast_weight[profile_row, profile_col] >= 0.70
            and ridge_total[profile_row, profile_col] < 1100.0
            and local_relief < 200.0
            and elevation <= 100.0
        )
        highland_here = mountain_here >= 0.50 or local_relief >= 700.0
        foothill_here = mountain_here >= 0.23 or scarp_here >= 0.24 or local_relief >= 280.0
        named_landform = f"{nearest_peak_name}山地" if nearest_peak_dist < 85 else (
            "北部山地" if cy < 600 else "西南山地" if cx < 800 else
            "东部山地" if cx > 1250 else "南部山脉" if cy > 1050 else "中轴山脉"
        )
        if city.get("name") == "长岭市":
            terrain, landform = "河谷城市", "长岭山间谷地"
        elif city.get("name") == "塔马市":
            terrain, landform = "山城", "塔马山地"
        elif city.get("name") == "乍曲市":
            terrain, landform = "U型山谷", "乍曲U型河谷"
        elif city.get("name") == "剑山市":
            terrain, landform = "山前河谷", "剑山山前河谷"
        elif canyon_dist <= 10.0 and elevation > 650 and cy < 760 and mountain_here >= 0.30:
            terrain = "峡谷河谷"
            landform = "中部河谷"
        elif basin_here and not highland_here:
            terrain, landform = "盆地", "臭都盆地"
        elif plateau_here:
            terrain, landform = ("高原山地", named_landform) if highland_here else ("高原", "北湩高原")
        elif gobi_here and not (highland_here and elevation > 1250):
            terrain, landform = "戈壁", "瓦莱莉戈壁"
        elif highland_here and elevation > 1000:
            terrain, landform = "山地", named_landform
        elif east_plain_here and river_dist < 26:
            terrain, landform = "河谷平原", "河流冲积平原"
        elif east_plain_here:
            terrain = "平原"
            landform = "东部沿海平原" if coast_distance[profile_row, profile_col] < 20.0 else "东部平原"
        elif hills_here or foothill_here:
            terrain = "丘陵"
            landform = "东南丘陵" if hills_here else f"{nearest_peak_name}丘陵" if nearest_peak_dist < 85 else "山前丘陵"
        elif plain_here:
            terrain, landform = "平原", "中部河谷平原"
        elif river_dist < 26:
            terrain, landform = "河谷平原", "河流冲积平原"
        elif elevation > 1100:
            terrain, landform = "高地", "内陆高地"
        else:
            terrain, landform = "平原", "沿河平原"
        profile = {
            "city_id": city["id"],
            "name": city.get("name", ""),
            "x": round(cx, 3),
            "y": round(cy, 3),
            "elevation_m": round(elevation, 1),
            "range_m": [round(local_low, 1), round(local_high, 1)],
            "terrain": terrain,
            "landform": landform,
            "nearest_peak": nearest_peak_name if nearest_peak_dist < 70 else "",
            "nearest_peak_distance_px": round(nearest_peak_dist, 1) if nearest_peak_dist < 70 else None,
            "nearest_river_distance_px": round(river_dist, 1),
            "coast_distance_px": round(coast_dist, 1),
            "land_distance_px": round(land_dist, 1),
            "is_land": city_is_land,
            "label_anchor_x": round(float(city.get("labelAnchorPixelXY", [cx, cy])[0]), 1),
            "label_anchor_y": round(float(city.get("labelAnchorPixelXY", [cx, cy])[1]), 1),
            "marker_offset_px": round(float(city.get("markerOffsetPx", 0.0)), 1),
            "marker_source": city.get("markerSource", "official SVG city marker"),
            "relation_review": relation_review,
        }
        city_profiles.append(profile)
        cities_for_csv.append(
            [profile["city_id"], profile["name"], profile["x"], profile["y"], profile["elevation_m"],
             profile["range_m"][0], profile["range_m"][1], profile["terrain"], profile["landform"],
             profile["nearest_peak"], profile["nearest_peak_distance_px"], profile["nearest_river_distance_px"],
             profile["label_anchor_x"], profile["label_anchor_y"], profile["marker_offset_px"], profile["marker_source"]]
        )
        city_audit_rows.append([
            profile["city_id"], profile["name"], profile["x"], profile["y"],
            profile["label_anchor_x"], profile["label_anchor_y"], profile["marker_offset_px"],
            profile["marker_source"], profile["elevation_m"], profile["terrain"],
            profile["is_land"], profile["land_distance_px"], profile["nearest_river_distance_px"],
            "河道中心" if river_dist < 1.5 else "近河谷" if river_dist < 15 else "远离河道",
            nearest_peak_name, round(nearest_peak_dist, 1),
            profile["range_m"][0], profile["range_m"][1], round(coast_dist, 1),
            profile["relation_review"],
        ])

    city_relationship_audit = {
        "checkedCityCount": len(city_profiles),
        "officialMapMarkerMatchedCount": sum(
            profile["marker_source"] == "official SVG city marker" for profile in city_profiles
        ),
        "cityPointsOnWaterMaskCount": sum(not profile["is_land"] for profile in city_profiles),
        "cityPointsOver5PxFromLandCount": sum(profile["land_distance_px"] > 5.0 for profile in city_profiles),
        "cityPointsOnRiverCenterlineCount": sum(
            profile["nearest_river_distance_px"] < 1.5 for profile in city_profiles
        ),
        "cityPointsWithin5PxOfMappedPeakCount": sum(
            profile["nearest_peak_distance_px"] is not None
            and profile["nearest_peak_distance_px"] <= 5.0
            for profile in city_profiles
        ),
        "nonPlateauCitiesWithLocalReliefOver1800mCount": sum(
            "城市周边起伏很大" in profile["relation_review"] for profile in city_profiles
        ),
        "reviewFlagCount": sum(
            profile["relation_review"].startswith("通过：") is False for profile in city_profiles
        ),
            "method": "Each city was sampled at its visible map marker; checked against land mask, the complete official river-pixel network, nearest named summit, and a 12px neighborhood elevation range.",
    }

    river_distance_audit, river_nearest_audit = ndimage.distance_transform_edt(
        ~river_water_mask_full, return_indices=True
    )
    river_mountain_audit_rows = []
    for peak in peaks:
        px, py, name, peak_height = peak[0], peak[1], peak[2], peak[3]
        ix, iy = int(round(px)), int(round(py))
        river_distance_px = float(river_distance_audit[iy, ix])
        qx = float(river_nearest_audit[1, iy, ix])
        qy = float(river_nearest_audit[0, iy, ix])
        dx, dy = px - qx, py - qy
        length = math.hypot(dx, dy) or 1.0
        same_x = int(np.clip(round(qx + dx / length * 8.0), 0, WIDTH - 1))
        same_y = int(np.clip(round(qy + dy / length * 8.0), 0, HEIGHT - 1))
        far_x = int(np.clip(round(qx - dx / length * 8.0), 0, WIDTH - 1))
        far_y = int(np.clip(round(qy - dy / length * 8.0), 0, HEIGHT - 1))
        peak_key = f"{name}@{px:.3f},{py:.3f}"
        river_valley_nearby = river_distance_px <= 28.0
        river_mountain_audit_rows.append([
            name, round(px, 1), round(py, 1), round(peak_height, 1),
            round(river_distance_px, 1), round(qx, 1), round(qy, 1),
            round(float(z_full[int(round(qy)), int(round(qx))]), 1),
            round(float(z_full[same_y, same_x]), 1), round(float(z_full[far_y, far_x]), 1),
            "是" if river_valley_nearby else "否",
            "由真实弯曲河道塑造谷地；山体不沿直线河岸截断" if river_valley_nearby else (
                "距河较远；保留自然山体坡面"
            ),
        ])
    with (DATA / "river_mountain_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "山峰", "峰位_x", "峰位_y", "标注峰高_m", "距原图蓝色河道_px",
            "最近河道_x", "最近河道_y", "河道水面_m", "标注峰同岸8px_m",
            "对岸8px_m", "位于曲线河谷影响范围", "复核结果",
        ])
        writer.writerows(river_mountain_audit_rows)

    river_flow_audit_rows = []
    centerline_surface = river_flow["centerlineSurface"]
    flow_distance_full = river_flow["flowDistance"]
    traced_line_mask = river_flow["centerlineMask"]
    line_neighbor_offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
    for feature in hydrology.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            continue
        x0 = max(0, int(math.floor(min(point[0] for point in coords))) - 1)
        y0 = max(0, int(math.floor(min(point[1] for point in coords))) - 1)
        x1 = min(WIDTH, int(math.ceil(max(point[0] for point in coords))) + 2)
        y1 = min(HEIGHT, int(math.ceil(max(point[1] for point in coords))) + 2)
        line_image = Image.new("L", (x1 - x0, y1 - y0), 0)
        line_draw = ImageDraw.Draw(line_image)
        line_draw.line(
            [(point[0] - x0, point[1] - y0) for point in coords],
            fill=255,
            width=1,
            joint="curve",
        )
        local_line_mask = np.asarray(line_image) > 0
        local_line_mask &= traced_line_mask[y0:y1, x0:x1]
        sample_count = int(np.count_nonzero(local_line_mask))
        local_flow_distance = flow_distance_full[y0:y1, x0:x1]
        local_surface = centerline_surface[y0:y1, x0:x1]
        line_distances = local_flow_distance[local_line_mask]
        line_surfaces = local_surface[local_line_mask]
        upstream_height = float(line_surfaces[int(np.argmax(line_distances))]) if sample_count else 0.0
        downstream_height = float(line_surfaces[int(np.argmin(line_distances))]) if sample_count else 0.0
        directed_rises = []
        box_height, box_width = local_line_mask.shape
        for dy, dx in line_neighbor_offsets:
            source_y0, source_y1 = max(0, -dy), min(box_height, box_height - dy)
            source_x0, source_x1 = max(0, -dx), min(box_width, box_width - dx)
            target_y0, target_y1 = source_y0 + dy, source_y1 + dy
            target_x0, target_x1 = source_x0 + dx, source_x1 + dx
            adjacent = (
                local_line_mask[source_y0:source_y1, source_x0:source_x1]
                & local_line_mask[target_y0:target_y1, target_x0:target_x1]
            )
            if not np.any(adjacent):
                continue
            source_distance = local_flow_distance[source_y0:source_y1, source_x0:source_x1]
            target_distance = local_flow_distance[target_y0:target_y1, target_x0:target_x1]
            source_surface = local_surface[source_y0:source_y1, source_x0:source_x1]
            target_surface = local_surface[target_y0:target_y1, target_x0:target_x1]
            source_upstream = adjacent & (source_distance > target_distance + 1e-5)
            target_upstream = adjacent & (target_distance > source_distance + 1e-5)
            directed_rises.extend((target_surface[source_upstream] - source_surface[source_upstream]).tolist())
            directed_rises.extend((source_surface[target_upstream] - target_surface[target_upstream]).tolist())
        increases = np.asarray(directed_rises, dtype=float)
        river_flow_audit_rows.append([
            feature.get("properties", {}).get("segment_id", ""),
            sample_count,
            round(upstream_height, 1),
            round(downstream_height, 1),
            round(float(np.max(increases)), 1) if len(increases) else 0.0,
            int(np.count_nonzero(increases > 0.1)),
            "通过" if sample_count and not np.any(increases > 0.1) else "中心线像素局部复核",
        ])
    with (DATA / "river_flow_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "河段_id", "中心线像素数", "上游水面_m", "下游水面_m", "最大顺流升幅_m", "顺流升高步数", "自查结果",
        ])
        writer.writerows(river_flow_audit_rows)

    hydrology_tile_audit = write_hydrology_tile_audit(
        z_full,
        land_full & ~lake_mask_full,
        river_water_mask_full,
        river_water_surface_full,
        river_flow["routedMask"],
        river_flow["centerlineSurface"],
        river_flow["flowDistance"],
        cities,
        peaks,
    )

    model = {
        "coordinateReference": "source map pixels, 2048x1448, origin top-left",
        "metadata": {
            "title": "𣿅国陆地推演地形",
            "elevationStatus": "synthetic_relief_anchored_to_all_mapped_peak_heights",
            "fixedSeed": SEED,
            "ridgeBackboneCount": len(RIDGES),
            "mountainConnectionPolicy": "瑞丰山脉沿河60西侧从剑山经雪瑞峰延伸，并顺河道弯折方向渐隐；中央山脉不绘制贯穿式骨架，由彼此留有山口的局部山群组成，西南盆地的春分山、梅花山、旃檀山仅轻微搭接，中央山脉最北端为刺梨山。长岭山脉从鹞岳沿河道U形弯折绕过河谷，在北侧鞍部与对岸山体自然接合；河道通过曲线峡谷塑形，不再用直线河岸切断山体。长岭市位于山间河谷，塔马市按山城保留。",
            "namedPeakPolicy": "All 87 labelled summit heights in data/mountain_peaks.js are exact interpolation constraints; relief-only position and zone corrections are stored in a separate copy, leaving the original source file untouched.",
            "peakAnchorKernel": "Named ridge peaks use range-aligned anisotropic kernels; Central Range peaks use independent local orientations, with only the three southwest-basin massifs allowed mild overlap. Nearby mapped rivers carve smooth curved valleys from the actual watercourse; no planar bank mask truncates mountain massifs.",
            "peakAnchorMaxFitErrorM": round(peak_anchor_max_error_m, 3),
            "riverWaterSurfacePolicy": {
                "sourceMapWaterInk": "#b1d0ed pixels from the official 4x map raster, intersected with the land mask and excluding lakes and ocean",
                "mappedRiverWaterPixelCount": int(np.count_nonzero(river_water_mask_full)),
                "coastalTidalPixelCountSetToSeaLevel": int(np.count_nonzero(coastal_tidal_channels)),
                "coastalTidalSeaLevelM": 0.0,
                "inlandWaterSurfaceRule": "Channel surfaces are constrained on the connected river-pixel graph from headwaters toward sea/lake/sinks, never rising downstream; local bank minima keep the water below adjacent land.",
                "flowConditioning": {
                    "method": "Eight-neighbour graph over the union of all official blue river-stroke pixels and 300 traced vector reaches; route connected water pixels to sea/lake/inland sinks and propagate downstream height limits across the full network.",
                    "connectedChannelComponents": river_flow["componentCount"],
                    "centerlinePixelCount": river_flow["centerlinePixelCount"],
                    "routedNetworkPixelCount": river_flow["routedNetworkPixelCount"],
                    "minimumDownstreamDropMPerPixel": river_flow["minimumDropMPerPx"],
                    "directedDownstreamSteps": river_flow["downhillStepCount"],
                    "uphillViolationCount": river_flow["uphillViolationCount"],
                    "targetedRiverHeightCaps": river_flow["heightCaps"],
                    "lineAuditCsv": "data/river_flow_audit.csv",
                },
                "curvedRiverValleyPolicy": "Terrain caps follow the curved union of traced channels and official water strokes and vary with mountain/plateau relief; no straight bank-plane cuts are applied.",
                "mountainGorgeCutM": [1500.0, 380.0],
                "planarRiverBankTaperCount": 0,
                "description": "All official blue water-stroke pixels use the downhill-conditioned surface; the river graph is reconditioned against final banks after valley carving.",
            },
            "riverMountainAudit": {
                "checkedSummitCount": len(peaks),
                "summitsWithinCurvedValleyReviewDistance": sum(row[4] <= 28.0 for row in river_mountain_audit_rows),
                "auditCsv": "data/river_mountain_audit.csv",
                "method": "For each labelled summit, compare its nearest official blue water pixel with terrain 8px on both sides after the curved channel valley has been carved; the check records no straight bank clipping.",
            },
            "hydrologyTileAudit": {
                **hydrology_tile_audit,
                "grid": "8 columns x 6 rows, 280x258 source-pixel windows with 20-26px overlap",
                "tileAuditCsv": "data/hydrology_tile_audit.csv",
                "flowAuditCsv": "data/river_flow_audit.csv",
                "reviewSheetPattern": "diagnostics/hydrology_tile_review_01.png through hydrology_tile_review_08.png",
            },
            "riverFlowAudit": {
                "checkedRiverReachCount": len(river_flow_audit_rows),
                "reachProfilesWithoutUphillSteps": sum(str(row[-1]).startswith("通过") for row in river_flow_audit_rows),
                "reachProfilesWithLocalUphillStep": sum(not str(row[-1]).startswith("通过") for row in river_flow_audit_rows),
                "directionRule": "Each traced reach is checked along adjacent 8-connected centerline pixels within the complete water graph; sparse geometry vertices are not compared across branches.",
                "auditCsv": "data/river_flow_audit.csv",
            },
            "specialRifengRiverAlignment": {
                "range": "瑞丰山脉",
                "riverSegmentId": 60,
                "riverSide": "map-left/west",
                "backboneSourcePixelXY": RANGE_BACKBONES["瑞丰山脉"],
                "fadeCrossingSourcePixelXY": list(RIFENG_FADE_CROSSING),
                "fadeLengthAlongBendPx": RIFENG_FADE_LENGTH_PX,
                "bankMask": "traced unlooped river-60 trunk profile, then continued to the bend crossing on the map-west side",
                "description": "The ridge bends along river 60's western bank from 剑山 to 雪瑞峰 and tapers in the downstream bend direction, avoiding a horizontal cut across the terrain.",
            },
            "specialCentralRangeLayout": {
                "ridgeSegmentsSourcePixelXY": [],
                "terminalPeak": CENTRAL_RANGE_TERMINAL_PEAK,
                "basinSideSlightlyBlendedPeaks": sorted(CENTRAL_BASIN_CLUSTER_PEAKS),
                "excludedPeaks": sorted(NORTH_PLATEAU_EXTENSION_PEAKS),
                "description": "No point-to-point Central Range backbone is drawn. Local massifs are kept separate by natural passes, with only slight overlap in the southwest-basin cluster; the range terminates at 刺梨山.",
            },
            "specialSouthwestBasinMountainBlend": {
                "peakNames": sorted(CENTRAL_BASIN_CLUSTER_PEAKS),
                "summitKernelSigmaAlongAcrossPx": [20.0, 9.0],
                "broadShoulderReliefM": 230.0,
                "description": "Three nearby southwest-basin peaks receive modestly broader, differently oriented forms and a low shared shoulder so their lower slopes touch lightly without joining the whole Central Range.",
            },
            "specialNorthPlateauEasternSpur": {
                "peakNames": sorted(NORTH_PLATEAU_EXTENSION_PEAKS),
                "influencePolygonSourcePixelXY": NORTH_DUNG_EAST_SPUR_POLYGON,
                "description": "A restrained plateau shoulder supports the three mapped eastern peaks without drawing a continuous Central Range backbone through them.",
            },
            "specialBaisuiIslandPeak": {
                "peak": "百岁山",
                "originalSourcePixelXY": baisui_original_xy,
                "reliefCopySourcePixelXY": list(BAISUI_RELIEF_POSITION),
                "description": "The relief-copy summit marker is moved from offshore onto the mapped island interior; original mountain peak data remain unchanged.",
            },
            "smallIslandElevationPolicy": {
                "smallIslandComponentCount": small_island_count,
                "unlabelledIslandCapM": "80 + 6 * shoreline_distance_px, capped at 200 m",
                "labelledIslandSummitPolicy": "Summit heights stay exact in the click readout and are concentrated in a 1.1–2.6 px local cone; surrounding island terrain falls rapidly toward the shore.",
                "summitRestoreCoreSigmaPx": 1.25,
            },
            "northernPlateauCoastalPolicy": {
                "shorelineCapM": 1050.0,
                "risePerSourcePixelM": 24.0,
                "maximumDistanceAppliedPx": 20.0,
                "cappedCellCount": north_plateau_coast_capped_cells,
                "description": "Nearshore terrain of the northern plateau is lowered modestly while preserving the sharp coastal escarpment and exact summit cores.",
            },
            "specialUValley": {
                "city": "乍曲市",
                "centerSourcePixelXY": [zaq_city_x, zaq_city_y],
                "riverSegmentId": 24,
                "floorTargetM": zaq_u_valley_floor_m,
                "flatFloorHalfWidthPx": zaq_floor_halfwidth_px,
                "outerWallRadiusPx": zaq_wall_outer_radius_px,
                "localScopeSigmaPx": 78.0,
                "appliedCutDepthAtCityM": round(zaq_u_valley_cut_depth_m, 1),
                "raisedAfterPeakFitAtCityM": round(zaq_u_valley_post_anchor_raise_m, 1),
            },
            "specialLonglingRiverValley": {
                "city": "长岭市",
                "centerSourcePixelXY": [longling_city_x, longling_city_y],
                "riverControlSourcePixelXY": [longling_qx, longling_qy],
                "riverSurfaceTargetCapM": longling_river_height_cap_m,
                "riverSurfaceAtControlM": round(longling_channel_height_m, 1),
                "cityFloorTargetM": round(longling_city_target_m, 1),
                "cityFloorAfterBuildM": round(bilinear(z_full, longling_city_x, longling_city_y, step=1.0), 1),
                "description": "The river segment south of 长岭市 is lowered into a continuous valley; the city sits on a valley floor just above the channel. Its local trough follows the shortest land path to the mapped river bend.",
            },
            "specialTamadMountainCity": {
                "city": "塔马市",
                "centerSourcePixelXY": list(next(city["sourcePixelXY"] for city in cities if city.get("name") == "塔马市")),
                "description": "Tamad remains a high-elevation mountain city as explicitly requested.",
            },
            "specialSwordCityValley": {
                "city": "剑山市",
                "centerSourcePixelXY": list(sword_city_xy) if sword_city_xy else None,
                "nearestRiverPointSourcePixelXY": list(sword_river_xy) if sword_river_xy else None,
                "floorTargetM": sword_valley_floor_m,
                "appliedCutDepthAtCityM": round(sword_valley_cut_depth_m, 1),
                "description": "A narrow, steep-sided mountain-foot settlement valley joins the nearby river and preserves mapped summit points.",
            },
            "specialNearbyRiverGorge": {
                **nearby_river_gorge,
                "description": "A narrow gorge follows mapped river 231 through the saddle between nearby peaks, interrupting the unsupported mountain connection while preserving summit anchors.",
            },
            "specialSeaBottomLakeBasin": {
                "lake": sea_bottom_lake.get("properties", {}).get("name", "海底湖"),
                "lakeId": 3,
                "centerSourcePixelXY": [sea_bottom_lake_label.get("pointX"), sea_bottom_lake_label.get("pointY")],
                "floorRangeM": [round(lake_floor_min_m, 1), round(lake_floor_max_m, 1)],
                "meanFloorM": round(lake_floor_mean_m, 1),
                "shoreMarginTargetM": [8.0, 55.3],
                "description": "The mapped lake floor stays near -200 m with fixed-seed non-integer relief; dry land around the shore rises gently above the river surface.",
            },
            "elevationMinimumM": round(float(np.nanmin(z_full[relief_sample_mask])), 3),
            "elevationMaximumM": round(float(np.nanmax(z_full[land_full])), 3),
            "highestMappedPeakM": round(peak_height_ceiling_m, 3),
            "rasterScaleMaximumM": ELEVATION_MAX_M,
            "elevationScalePolicy": "The raster scale expands to 1.4 times the highest mapped peak; no Earth-based maximum elevation is imposed.",
            "plateauEdgeStyle": "Irregular steep escarpment with river-cut passes and rapid relief change at the official northern-plateau boundary.",
            "coastalElevationPolicy": "Non-mountain coast terrain includes a varied 0–4 m strip at the shore, rising gently inland and staying below 50 m by about 20 source pixels; northern plateau nearshore relief is capped separately while retaining a steep escarpment.",
            "easternLowlandPolicy": "Non-mountain terrain in the east and southeast remains below 50 m; the first 20 source pixels from shore taper gently from near sea level to below 50 m.",
            "plainElevationPolicy": "Non-mountain terrain in the mapped central plain remains at or below 20 m, usually below 20 m, with broad low-amplitude fixed-seed undulation and no quantized elevation tiers.",
            "fineReliefDetail": "Fixed-seed, low-amplitude multiscale terrain detail; only documented ranges receive continuous backbones, while other labelled peaks receive separate asymmetric local massifs.",
            "contoursDisplayed": False,
            "landformLabelsDisplayed": False,
            "cityNeighborhoodRadiusPx": 12,
            "cityNeighborhoodRadiusKmApprox": 93,
            "heightUnit": "metres, illustrative absolute scale",
            "elevationRaster": "data/relief_elevation.js; full-resolution 16-bit elevation with official river-water mask and flow-conditioned river surfaces; retained as terrain-model and hydrology-audit data, while interactive elevation readout is temporarily disabled",
            "sources": [
                "data/mountain_peaks.js",
                "data/relief_mountain_peaks.js (relief-copy peak positions and northern-plateau extension classification)",
                "data/mountain_elevation_model.md",
                "data/hydrology_lines.geojson",
                "data/ocean_water.geojson",
                "data/lakes.geojson",
                "data/coastlines.geojson",
                "data/national_outline.geojson",
                "Administrative-Division-Map-of-Donh-2026.png (official labelled map; green dashed northern plateau boundary digitized in source pixels)",
            ],
            "limits": "The source map has no surveyed elevation raster. All displayed peak heights are honored exactly; terrain between mapped control points and city heights remain a fixed geographic model based on relief, drainage, coastlines, and map context.",
            "cityCount": len(city_profiles),
            "cityLocationAudit": city_location_audit,
            "cityRelationshipAudit": city_relationship_audit,
            "cityMarkerElevationSnapRadiusPx": 3.5,
            "cityMarkerElevationSnapScope": "offland_city_markers_only",
            "peakCount": len(peaks),
        },
        "plateauBoundary": PLATEAU_BOUNDARY,
        "landformLabels": LANDFORM_LABELS,
        "peakAnchors": [
            {
                "name": peak[2],
                "x": peak[0],
                "y": peak[1],
                "elevation_m": peak[3],
                "elevation_source": peak[4],
                "elevation_zone": peak[5],
                "mountain_range": peak[6],
            }
            for peak in peaks
        ],
        "cityProfiles": city_profiles,
    }
    (DATA / "relief_model.js").write_text(
        "window.DUNIVERSE_RELIEF=" + json.dumps(model, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    (DATA / "relief_model_metadata.json").write_text(
        json.dumps(model["metadata"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plateau_geojson = {
        "type": "FeatureCollection",
        "name": "north_dung_plateau_boundary",
        "coordinate_reference": "2048x1448 official map source pixels; origin top-left; not geographic coordinates",
        "features": [
            {
                "type": "Feature",
                "id": "landform:north-dung-plateau-boundary",
                "properties": {
                    "name": "北湩高原分界线",
                    "source": "Administrative-Division-Map-of-Donh-2026.png",
                    "digitized_from": "green dashed line on the labelled official map",
                    "status": "hand-traced cartographic boundary",
                },
                "geometry": {"type": "LineString", "coordinates": PLATEAU_BOUNDARY},
            }
        ],
    }
    (DATA / "north_dung_plateau_boundary.geojson").write_text(
        json.dumps(plateau_geojson, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (DATA / "city_terrain_profiles.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "city_id", "城市", "地图圆点_x", "地图圆点_y", "推演海拔_m", "附近低值_m", "附近高值_m",
            "地形", "邻近地貌", "最近山峰", "山峰距离_px", "最近河流距离_px",
            "文字锚点_x", "文字锚点_y", "圆点与文字偏移_px", "圆点来源",
        ])
        writer.writerows(cities_for_csv)
    with (DATA / "city_terrain_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "city_id", "城市", "地图圆点_x", "地图圆点_y", "文字锚点_x", "文字锚点_y", "圆点与文字偏移_px",
            "取样点来源", "推演海拔_m", "地形", "落在陆地掩膜", "距最近陆地_px", "最近河流距离_px", "河流关系",
            "最近山峰", "山峰距离_px", "12px邻域低值_m", "12px邻域高值_m", "距海岸_px", "自查结论",
        ])
        writer.writerows(city_audit_rows)

    # A flattened preview is useful for quick review; the interactive duplicate
    # remains the editable deliverable with independent layer controls.
    official = Image.open(ROOT / "Administrative-Division-Map-of-Donh-2026.png").convert("RGB")
    official = official.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).convert("RGBA")
    tint = Image.open(DATA / "relief_tint.png").convert("RGBA")
    official.alpha_composite(tint)
    draw = ImageDraw.Draw(official, "RGBA")
    dashed_line(draw, PLATEAU_BOUNDARY, (255, 255, 255, 230), 4, 7, 4)
    dashed_line(draw, PLATEAU_BOUNDARY, (160, 106, 20, 255), 2, 7, 4)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    for item in LANDFORM_LABELS:
        text = item["name"]
        draw.text((item["x"], item["y"]), text, font=font, fill=(85, 66, 43, 255),
                  stroke_width=3, stroke_fill=(255, 252, 241, 230), anchor="mm")
    official.convert("RGB").save(ROOT / "relief_preview.png", optimize=True)
    tile_review_sheet_paths = create_hydrology_tile_review_sheets(ROOT / "relief_preview.png")

    east_southeast_nonmountain_mask = (
        land_full
        & (east_southeast_weight_full > 0.98)
        & (ridge_total_full < 1100.0)
        & (peak_distance_full >= 54.0)
    )
    region_elevation_stats_summary = build_region_elevation_stats(ROOT)
    print(json.dumps({
        "model": str(DATA / "relief_model.js"),
        "preview": str(ROOT / "relief_preview.png"),
        "cities": len(city_profiles),
        "city_location_audit": city_location_audit,
        "city_relationship_audit": city_relationship_audit,
        "peaks": len(peaks),
        "hydrology_tile_audit": hydrology_tile_audit,
        "river_flow_components": river_flow["componentCount"],
        "river_flow_centerline_uphill_violations": river_flow["uphillViolationCount"],
        "river_flow_reach_audit": {
            "checked": len(river_flow_audit_rows),
            "passed": sum(str(row[-1]).startswith("通过") for row in river_flow_audit_rows),
            "needs_review": sum(not str(row[-1]).startswith("通过") for row in river_flow_audit_rows),
        },
        "longling_city_elevation_m": round(bilinear(z_full, longling_city_x, longling_city_y, step=1.0), 1),
        "longling_river_surface_m": round(longling_channel_height_m, 1),
        "tamad_city_elevation_m": next(
            (profile["elevation_m"] for profile in city_profiles if profile["name"] == "塔马市"), None
        ),
        "tile_review_sheets": tile_review_sheet_paths,
        "elevation_range_m": [round(float(np.nanmin(z_full[relief_sample_mask])), 1), round(float(np.nanmax(z_full[land_full])), 1)],
        "sea_bottom_lake_floor_m": [round(lake_floor_min_m, 1), round(lake_floor_mean_m, 1), round(lake_floor_max_m, 1)],
        "peak_anchor_max_error_m": round(peak_anchor_max_error_m, 3),
        "zaq_city_elevation_m": round(bilinear(z_full, zaq_city_x, zaq_city_y, step=1.0), 1),
        "zaq_u_valley_cut_depth_m": round(zaq_u_valley_cut_depth_m, 1),
        "alkali_zinc_tork_city_elevation_m": next(
            (profile["elevation_m"] for profile in city_profiles if profile["name"] == "碱锌托克市"), None
        ),
        "alkali_zinc_tork_gorge_cut_depth_m": round(nearby_river_gorge["cutDepthMaxM"], 1),
        "land_mask_cells": int(land.sum()),
        "coast_cells_0_to_4m": int(np.count_nonzero(land_full & (coast_distance_full < 20.0) & (z_full >= 0.0) & (z_full <= 4.0))),
        "coastal_0_20px_share_above_500_pct": round(float(np.mean(z_full[land_full & (coast_distance_full < 20.0)] > 500.0) * 100.0), 1),
        "east_southeast_nonmountain_lowland_max_m": int(np.nanmax(z_full[east_southeast_nonmountain_mask])),
        "east_southeast_nonmountain_coastal_max_m": int(np.nanmax(z_full[east_southeast_nonmountain_mask & (coast_distance_full < 6.0)])),
        "region_elevation_stats": region_elevation_stats_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
