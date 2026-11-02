#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract ocean, neighboring borders, and cross-border rivers from the source map."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, LineString, mapping
from skimage.morphology import skeletonize

SOURCE = Path("input_map.png")
DATA = Path("data")
DIAGNOSTICS = Path("diagnostics")
SEA_RGB = np.array([177, 208, 237], dtype=np.uint8)


def write_fc(path, name, features):
    Path(path).write_text(json.dumps({
        "type": "FeatureCollection", "name": name,
        "coordinate_reference_note": "Image pixel coordinates; origin top-left; x right; y down. Not georeferenced.",
        "features": features,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def mask_to_polygons(mask, epsilon=0.65):
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
            if cv2.contourArea(contours[child]) >= 12:
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


def coast_segments(ring, width, height):
    points = [(float(x), float(y)) for x, y in ring]
    on_frame = lambda p: p[0] <= 0 or p[0] >= width - 1 or p[1] <= 0 or p[1] >= height - 1
    if not any(on_frame(p) for p in points):
        line = LineString(points)
        return [line] if line.length >= 10 else []
    start = next(i for i, p in enumerate(points) if on_frame(p))
    points = points[start + 1:] + points[:start + 1]
    segments, current = [], []
    for point in points:
        if on_frame(point):
            if len(current) > 1:
                line = LineString(current).simplify(0.65, preserve_topology=False)
                if line.length >= 10:
                    segments.append(line)
            current = []
        else:
            current.append(point)
    if len(current) > 1:
        line = LineString(current).simplify(0.65, preserve_topology=False)
        if line.length >= 10:
            segments.append(line)
    return segments


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
    coast_features = []
    for poly in ocean_polygons:
        polygons = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
        for part in polygons:
            rings = [(part.exterior, "coastline")] + [(ring, "island_shoreline") for ring in part.interiors]
            for ring, role in rings:
                for line in coast_segments(ring.coords, w, h):
                    coast_features.append({
                        "type": "Feature", "properties": {"role": role, "length_px": round(line.length, 1), "source": "source_image_ocean_color_boundary"},
                        "geometry": mapping(line),
                    })
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

    # Neighbor borders are thin neutral-gray strokes outside the mapped land.
    rgb_max = arr.max(axis=2).astype(np.int16)
    rgb_min = arr.min(axis=2).astype(np.int16)
    gray = arr.mean(axis=2)
    neighbor_mask = ((rgb_max - rgb_min <= 24) & (gray >= 145) & (gray <= 248) & (land == 0))
    neighbor_mask[:, 1050:] = False
    neighbor_mask &= ~ocean_buffer
    neighbor_mask = skeletonize(neighbor_mask)
    border_lines = trace_lines(neighbor_mask, 42)
    border_features = [{
        "type": "Feature", "properties": {"segment_id": i + 1, "length_px": round(line.length, 1), "source": "source_image_neutral_boundary_stroke", "review_status": "candidate_neighbor_border"},
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
