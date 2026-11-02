#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract ocean, neighboring borders, and cross-border rivers from the source map."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, LineString, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import skeletonize

SOURCE = Path("input_map.png")
HIGH_RES_SOURCE = Path("Administrative-Division-Map-of-Donh-2026.png")
DATA = Path("data")
DIAGNOSTICS = Path("diagnostics")
SEA_RGB = np.array([177, 208, 237], dtype=np.uint8)
HIGH_RES_SCALE = 4
COASTLINE_CROP_LEFT = 1000
COASTLINE_OPEN_KERNEL_PX = 13


def write_fc(path, name, features):
    Path(path).write_text(json.dumps({
        "type": "FeatureCollection", "name": name,
        "coordinate_reference_note": "Image pixel coordinates; origin top-left; x right; y down. Not georeferenced.",
        "features": features,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def mask_to_polygons(mask, epsilon=0.65, minimum_hole_area=12):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    polygons = []
    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1 or cv2.contourArea(contour) < 1:
            continue
        outer = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(outer) < 3:
            continue
        holes = []
        child = hierarchy[index][2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= minimum_hole_area:
                ring = cv2.approxPolyDP(contours[child], epsilon, True).reshape(-1, 2)
                if len(ring) >= 3:
                    holes.append([(float(x), float(y)) for x, y in ring])
            child = hierarchy[child][0]
        poly = Polygon([(float(x), float(y)) for x, y in outer], holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 1:
            polygons.append(poly)
    return polygons


def coast_segments(ring, width, height, minimum_length=10, simplify=0.65):
    points = [(float(x), float(y)) for x, y in ring]
    on_frame = lambda p: p[0] <= 0 or p[0] >= width - 1 or p[1] <= 0 or p[1] >= height - 1
    if not any(on_frame(p) for p in points):
        line = LineString(points).simplify(simplify, preserve_topology=False)
        return [line] if line.length >= minimum_length else []
    start = next(i for i, p in enumerate(points) if on_frame(p))
    points = points[start + 1:] + points[:start + 1]
    segments, current = [], []
    for point in points:
        if on_frame(point):
            if len(current) > 1:
                line = LineString(current).simplify(simplify, preserve_topology=False)
                if line.length >= minimum_length:
                    segments.append(line)
            current = []
        else:
            current.append(point)
    if len(current) > 1:
        line = LineString(current).simplify(simplify, preserve_topology=False)
        if line.length >= minimum_length:
            segments.append(line)
    return segments


def extract_high_resolution_coastlines(region_features):
    with Image.open(HIGH_RES_SOURCE) as image:
        high_width, high_height = image.size
        if high_width % HIGH_RES_SCALE or high_height % HIGH_RES_SCALE:
            raise RuntimeError("High-resolution map dimensions must divide evenly into source-map pixels.")
        source_width, source_height = high_width // HIGH_RES_SCALE, high_height // HIGH_RES_SCALE
        if COASTLINE_CROP_LEFT >= source_width:
            raise RuntimeError("Coastline crop starts outside the source-map width.")
        crop_left = COASTLINE_CROP_LEFT * HIGH_RES_SCALE
        pixels = np.asarray(image.crop((crop_left, 0, high_width, high_height)).convert("RGB"))

    sea = np.all(pixels == SEA_RGB, axis=2).astype(np.uint8)
    raw_sea = sea.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (COASTLINE_OPEN_KERNEL_PX, COASTLINE_OPEN_KERNEL_PX),
    )
    sea = cv2.morphologyEx(sea, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(sea, 8)
    edge_ids = [int(value) for value in np.unique(labels[:, -1]) if value]
    if not edge_ids:
        raise RuntimeError("Could not find the high-resolution ocean component along the right edge.")
    ocean_id = max(edge_ids, key=lambda value: stats[value, cv2.CC_STAT_AREA])
    ocean = (labels == ocean_id).astype(np.uint8) * 255
    del labels, stats, sea
    coast_sea_neighborhood = cv2.dilate(
        raw_sea,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    ) > 0
    del raw_sea

    # The opening disconnects river channels; raw shoreline proximity below
    # removes inland label counters from the resulting contour tree.
    # Fill lettering and marker cutouts in the open water, then restore every
    # mapped land-color pixel so small islands and the broken archipelago coast
    # remain holes in the ocean mask.
    outside = cv2.bitwise_not(ocean)
    exterior = outside.copy()
    cv2.floodFill(exterior, None, (0, 0), 0)
    ocean = cv2.bitwise_or(ocean, exterior)
    del outside, exterior

    color_codes = {
        (red << 16) | (green << 8) | blue
        for feature in region_features
        for red, green, blue in [feature["properties"]["fill_rgb"]]
    }
    packed_pixels = (
        (pixels[:, :, 0].astype(np.uint32) << 16)
        | (pixels[:, :, 1].astype(np.uint32) << 8)
        | pixels[:, :, 2].astype(np.uint32)
    )
    mapped_land = np.isin(packed_pixels, np.fromiter(color_codes, dtype=np.uint32))
    del packed_pixels
    ocean[mapped_land] = 0

    polygons = mask_to_polygons(ocean, epsilon=0.85, minimum_hole_area=8)
    del ocean
    outline_path = DATA / "national_outline.geojson"
    if outline_path.exists():
        outline_features = json.loads(outline_path.read_text(encoding="utf-8"))["features"]
        national_outline = unary_union([shape(feature["geometry"]) for feature in outline_features])
    else:
        national_outline = unary_union([shape(feature["geometry"]) for feature in region_features])
    coastline_band = national_outline.boundary.buffer(8.0)
    coast_features = []
    for polygon in polygons:
        parts = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
        for part in parts:
            rings = [(part.exterior, "coastline")]
            rings.extend((ring, "island_shoreline") for ring in part.interiors)
            for ring, role in rings:
                high_res_lines = coast_segments(
                    [(x + crop_left, y) for x, y in ring.coords],
                    high_width,
                    high_height,
                    minimum_length=8,
                    simplify=0.8,
                )
                for high_res_line in high_res_lines:
                    sample_coords = list(high_res_line.coords)
                    sample_step = max(1, len(sample_coords) // 512)
                    sea_support = []
                    for x, y in sample_coords[::sample_step]:
                        local_x, local_y = int(round(x - crop_left)), int(round(y))
                        if 0 <= local_x < coast_sea_neighborhood.shape[1] and 0 <= local_y < coast_sea_neighborhood.shape[0]:
                            sea_support.append(coast_sea_neighborhood[local_y, local_x])
                    if not sea_support or sum(sea_support) / len(sea_support) < 0.55:
                        continue
                    line = LineString(
                        [(x / HIGH_RES_SCALE, y / HIGH_RES_SCALE) for x, y in high_res_line.coords]
                    ).simplify(0.12, preserve_topology=False)
                    if line.length < 2:
                        continue
                    coastline_parts = [line]
                    if line.length > 1000:
                        clipped = line.intersection(coastline_band)
                        coastline_parts = []
                        pending = [clipped]
                        while pending:
                            part = pending.pop()
                            if part.geom_type == "LineString":
                                coastline_parts.append(part)
                            elif hasattr(part, "geoms"):
                                pending.extend(part.geoms)
                    for coastline_part in coastline_parts:
                        if coastline_part.length < 2:
                            continue
                        coast_features.append({
                            "type": "Feature",
                            "properties": {
                                "role": role,
                                "length_px": round(coastline_part.length, 2),
                                "source": "8192x5792_high_resolution_ocean_boundary",
                            },
                            "geometry": mapping(coastline_part),
                        })
    coast_features.extend(high_resolution_island_ink_features(pixels, mapped_land, crop_left))
    del mapped_land, pixels
    return coast_features


def high_resolution_island_ink_features(pixels, mapped_land, crop_left):
    """Recover tiny unfilled islands from their original ink outlines."""
    height, width = pixels.shape[:2]
    sea = np.all(pixels == SEA_RGB, axis=2).astype(np.uint8)
    sea_neighborhood = cv2.dilate(sea, np.ones((7, 7), dtype=np.uint8)) > 0
    del sea
    land_neighborhood = cv2.dilate(
        mapped_land.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    ) > 0
    del mapped_land

    dark_ink = np.max(pixels, axis=2) <= 55
    muted_shore_ink = cv2.inRange(
        pixels,
        np.array([167, 157, 157], dtype=np.uint8),
        np.array([203, 193, 193], dtype=np.uint8),
    ) > 0
    candidates = []
    for ink_kind, ink_mask in (("dark", dark_ink), ("muted", muted_shore_ink)):
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ink_mask.astype(np.uint8), 8
        )
        for component_id in range(1, component_count):
            x, y, component_width, component_height, area = map(int, stats[component_id])
            largest_dimension = max(component_width, component_height)
            if area < 3 or area > 8000 or largest_dimension > 180:
                continue
            aspect = largest_dimension / max(1, min(component_width, component_height))
            if aspect > 8 and area > 32:
                continue
            # The source attribution is typography in open water, not an island.
            if crop_left + x >= 6350 and y >= 5600:
                continue

            component = labels[y:y + component_height, x:x + component_width] == component_id
            area_pixels = int(component.sum())
            nearby_sea = np.count_nonzero(component & sea_neighborhood[y:y + component_height, x:x + component_width])
            nearby_land = np.count_nonzero(component & land_neighborhood[y:y + component_height, x:x + component_width])
            if nearby_sea / area_pixels < 0.15 or nearby_land / area_pixels > 0.25:
                continue
            candidates.append((ink_kind, x, y, component_width, component_height, area, component, centroids[component_id]))
        del labels, stats, centroids, ink_mask
    del dark_ink, muted_shore_ink, sea_neighborhood, land_neighborhood

    features = []
    for ink_kind, x, y, component_width, component_height, area, component, centroid in candidates:
        skeleton = skeletonize(component)
        contours, _ = cv2.findContours(
            skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        paths = []
        for contour in contours:
            points = contour.reshape(-1, 2)
            if len(points) >= 3:
                path = [(float(px + x + crop_left) / HIGH_RES_SCALE,
                         float(py + y) / HIGH_RES_SCALE) for px, py in points]
                path.append(path[0])
                paths.append(path)
            elif len(points) == 2:
                paths.append([
                    (float(points[0][0] + x + crop_left) / HIGH_RES_SCALE,
                     float(points[0][1] + y) / HIGH_RES_SCALE),
                    (float(points[1][0] + x + crop_left) / HIGH_RES_SCALE,
                     float(points[1][1] + y) / HIGH_RES_SCALE),
                ])
        if not paths and area <= 40:
            center_x, center_y = centroid
            radius_x = max(0.45, component_width / (2 * HIGH_RES_SCALE))
            radius_y = max(0.45, component_height / (2 * HIGH_RES_SCALE))
            paths.append([
                ((crop_left + x + center_x + radius_x * HIGH_RES_SCALE * math.cos(angle)) / HIGH_RES_SCALE,
                 (y + center_y + radius_y * HIGH_RES_SCALE * math.sin(angle)) / HIGH_RES_SCALE)
                for angle in np.linspace(0, 2 * math.pi, 13)[:-1]
            ] + [
                ((crop_left + x + center_x + radius_x * HIGH_RES_SCALE) / HIGH_RES_SCALE,
                 (y + center_y) / HIGH_RES_SCALE)
            ])
        for points in paths:
            line = LineString(points).simplify(0.03, preserve_topology=True)
            if line.length < 0.2:
                continue
            features.append({
                "type": "Feature",
                "properties": {
                    "role": "island_shoreline",
                    "length_px": round(line.length, 2),
                    "source": f"8192x5792_high_resolution_{ink_kind}_island_ink",
                },
                "geometry": mapping(line),
            })
    return features


def trace_lines(skeleton, min_length=24):
    ys, xs = np.where(skeleton)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    adjacency = {}
    for y, x in pixels:
        neighbors = []
        for dy, dx in offsets:
            q = (y + dy, x + dx)
            if q not in pixels:
                continue
            if dy and dx and ((y, x + dx) in pixels or (y + dy, x) in pixels):
                continue
            neighbors.append(q)
        adjacency[(y, x)] = neighbors
    degree = {p: len(ns) for p, ns in adjacency.items()}
    visited, result = set(), []

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    def trace(start, nxt):
        path = [start, nxt]
        visited.add(edge(start, nxt))
        prev, cur = start, nxt
        while degree.get(cur, 0) == 2:
            options = [q for q in adjacency[cur] if q != prev]
            if not options or edge(cur, options[0]) in visited:
                break
            q = options[0]
            visited.add(edge(cur, q))
            path.append(q)
            prev, cur = cur, q
            if cur == start:
                break
        return path

    for point, d in degree.items():
        if d != 2:
            for nxt in adjacency[point]:
                if edge(point, nxt) not in visited:
                    path = trace(point, nxt)
                    length = sum(math.hypot(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0]) for i in range(len(path) - 1))
                    if length >= min_length:
                        line = LineString([(float(x), float(y)) for y, x in path]).simplify(1.15, preserve_topology=False)
                        if len(line.coords) >= 2:
                            result.append(line)
    return result


def main():
    DATA.mkdir(exist_ok=True)
    DIAGNOSTICS.mkdir(exist_ok=True)
    arr = np.asarray(Image.open(SOURCE).convert("RGB"))
    h, w = arr.shape[:2]
    feature_collection = json.loads((DATA / "administrative_regions.geojson").read_text(encoding="utf-8"))
    colors = {tuple(f["properties"]["fill_rgb"]) for f in feature_collection["features"]}
    land = np.zeros((h, w), dtype=np.uint8)
    palette = np.array(sorted(colors), dtype=np.int32)
    for color in colors:
        land[np.all(arr == color, axis=2)] = 255
    land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # The connected sea fill is identified from the source pixels and opened only
    # enough to disconnect thin rivers from the ocean color.
    sea_color = np.all(arr == SEA_RGB, axis=2).astype(np.uint8)
    sea_color = cv2.morphologyEx(sea_color, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(sea_color, 8)
    edge_ids = [int(v) for v in np.unique(labels[:, -1]) if v]
    if not edge_ids:
        raise RuntimeError("Could not find the ocean-color component along the right edge.")
    ocean_id = max(edge_ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    ocean = (labels == ocean_id).astype(np.uint8) * 255
    # Fill holes caused by labels and symbols; then restore mapped land/islands.
    inv = cv2.bitwise_not(ocean)
    flood = inv.copy()
    cv2.floodFill(flood, None, (0, 0), 0)
    ocean = cv2.bitwise_or(ocean, flood)
    ocean[land > 0] = 0

    ocean_polygons = mask_to_polygons(ocean)
    ocean_features = [{
        "type": "Feature",
        "properties": {"source": "source_image_ocean_color_component", "fill_rgb": SEA_RGB.tolist(), "role": "ocean_water"},
        "geometry": mapping(poly),
    } for poly in ocean_polygons]
    write_fc(DATA / "ocean_water.geojson", "ocean_water", ocean_features)
    coast_features = extract_high_resolution_coastlines(feature_collection["features"])
    write_fc(DATA / "coastlines.geojson", "coastlines", coast_features)

    # Water-color linework is traced over the whole map. The ocean component is
    # removed first, then only segments outside the national land mask are saved.
    flat = arr.reshape(-1, 3).astype(np.int32)
    min_d2 = np.full(flat.shape[0], 999999, dtype=np.int32)
    for start in range(0, len(flat), 200000):
        q = flat[start:start + 200000]
        d2 = ((q[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        min_d2[start:start + len(q)] = d2.min(axis=1)
    distance_to_fill = np.sqrt(min_d2).reshape(h, w)
    blue = ((arr[:, :, 2].astype(int) - arr[:, :, 0].astype(int) > 30) &
            (arr[:, :, 2].astype(int) - arr[:, :, 1].astype(int) > 10) &
            (arr[:, :, 2] > 160) & (distance_to_fill > 12))
    ocean_buffer = cv2.dilate((labels == ocean_id).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    blue &= ~ocean_buffer
    water_distance = cv2.distanceTransform(blue.astype(np.uint8), cv2.DIST_L2, 5)
    lake_core = water_distance > 2.2
    lake_dilated = cv2.dilate(lake_core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    river_skeleton = skeletonize(blue & ~lake_dilated)
    outside_rivers = river_skeleton & (land == 0)
    river_lines = trace_lines(outside_rivers, 20)
    river_features = [{
        "type": "Feature", "properties": {"segment_id": i + 1, "length_px": round(line.length, 1), "source": "source_image_blue_hydrology_outside_national_outline", "review_status": "candidate_cross_border_river"},
        "geometry": mapping(line),
    } for i, line in enumerate(river_lines)]
    write_fc(DATA / "cross_border_rivers.geojson", "cross_border_river_candidates", river_features)

    # Keep the darker neutral strokes used for neighboring countries' borders.
    # Faint gray separators inside neighboring territories are province lines
    # and background artifacts; blue strokes are waterways and are kept in the
    # river layer instead.
    rgb_max = arr.max(axis=2).astype(np.int16)
    rgb_min = arr.min(axis=2).astype(np.int16)
    gray = arr.mean(axis=2)
    neighbor_mask = ((rgb_max - rgb_min <= 24) & (gray >= 145) & (gray <= 210) & (land == 0))
    neighbor_mask[:, 1050:] = False
    neighbor_mask &= ~ocean_buffer
    river_halo = cv2.dilate(blue.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    neighbor_mask &= ~river_halo
    neighbor_mask = skeletonize(neighbor_mask)
    border_lines = trace_lines(neighbor_mask, 42)
    border_features = [{
        "type": "Feature", "properties": {"segment_id": i + 1, "length_px": round(line.length, 1), "source": "dark_neutral_stroke_outside_national_land", "review_status": "candidate_neighbor_country_border"},
        "geometry": mapping(line),
    } for i, line in enumerate(border_lines)]
    write_fc(DATA / "neighbor_borders.geojson", "neighbor_border_candidates", border_features)

    # Save review overlays from the original raster.
    overlay = arr.copy()
    for line in river_lines:
        pts = np.round(np.array(line.coords)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], False, (255, 40, 40), 2, cv2.LINE_AA)
    for line in border_lines:
        pts = np.round(np.array(line.coords)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], False, (30, 80, 255), 2, cv2.LINE_AA)
    Image.fromarray(overlay).save(DIAGNOSTICS / "context_layers_overlay.png")
    Image.fromarray(ocean).save(DIAGNOSTICS / "ocean_mask.png")
    summary = {"ocean_features": len(ocean_features), "coastline_segments": len(coast_features), "cross_border_river_segments": len(river_features), "neighbor_border_segments": len(border_features), "ocean_pixels": int((ocean > 0).sum())}
    manifest_path = Path("extraction_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(summary)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
