#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit existing coastline lines against pixels on both sides in the source map.

This tool does not trace, extract, or create coastlines. It only evaluates the
current GeoJSON against the original high-resolution raster so false shoreline
loops over land, lettering, and inland water can be reviewed separately.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, shape


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIAGNOSTICS = ROOT / "diagnostics" / "coastline_pixel_audit"
SOURCE_IMAGE = ROOT / "Administrative-Division-Map-of-Donh-2026.png"
FEATURES_PATH = DATA / "coastlines.geojson"
SEA_RGB = np.array([177, 208, 237], dtype=np.int16)
MAP_TO_SOURCE = 4
SAMPLE_STEP_MAP_PX = 1.5
SIDE_OFFSETS_MAP_PX = (0.6, 0.9, 1.3, 1.8, 2.5, 3.5)
COLOR_TOLERANCE = 7


def load_land_colors() -> np.ndarray:
    regions = json.loads((DATA / "administrative_regions.geojson").read_text(encoding="utf-8"))["features"]
    return np.array(sorted({tuple(feature["properties"]["fill_rgb"]) for feature in regions}), dtype=np.int16)


def pixel_class(image: np.ndarray, land_colors: np.ndarray, x: float, y: float) -> str:
    sx = int(round(x * MAP_TO_SOURCE))
    sy = int(round(y * MAP_TO_SOURCE))
    if not (0 <= sx < image.shape[1] and 0 <= sy < image.shape[0]):
        return "unknown"
    rgb = image[sy, sx].astype(np.int16)
    if np.max(np.abs(rgb - SEA_RGB)) <= COLOR_TOLERANCE:
        return "sea"
    if np.max(rgb) >= 246 and np.min(rgb) >= 239:
        return "land"
    if np.any(np.max(np.abs(land_colors - rgb), axis=1) <= COLOR_TOLERANCE):
        return "land"
    # Inland rivers and lakes are deliberately kept separate from the sea.
    if rgb[2] - rgb[0] > 24 and rgb[2] - rgb[1] > 7 and rgb[2] > 150:
        return "inland_water"
    return "other"


def line_parts(geometry) -> list[LineString]:
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [LineString(geometry.coords)]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in line_parts(item)]
    return []


def nearest_side_class(
    image: np.ndarray,
    land_colors: np.ndarray,
    x: float,
    y: float,
    nx: float,
    ny: float,
    sign: float,
) -> str:
    for offset in SIDE_OFFSETS_MAP_PX:
        kind = pixel_class(image, land_colors, x + nx * offset * sign, y + ny * offset * sign)
        if kind != "other" and kind != "unknown":
            return kind
    return "other"


def sample_feature(image: np.ndarray, land_colors: np.ndarray, feature: dict) -> dict:
    geometry = shape(feature["geometry"])
    left = {key: 0 for key in ("sea", "land", "inland_water", "other")}
    right = {key: 0 for key in ("sea", "land", "inland_water", "other")}
    sample_count = 0
    for line in line_parts(geometry):
        coordinates = list(line.coords)
        for start, end in zip(coordinates, coordinates[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            steps = max(1, math.ceil(length / SAMPLE_STEP_MAP_PX))
            nx, ny = -dy / length, dx / length
            for index in range(steps):
                t = (index + 0.5) / steps
                x, y = start[0] + dx * t, start[1] + dy * t
                left[nearest_side_class(image, land_colors, x, y, nx, ny, -1)] += 1
                right[nearest_side_class(image, land_colors, x, y, nx, ny, 1)] += 1
                sample_count += 1

    denominator = max(1, sample_count)
    left_ratio = {key: value / denominator for key, value in left.items()}
    right_ratio = {key: value / denominator for key, value in right.items()}
    shore_support = max(
        min(left_ratio["sea"], right_ratio["land"]),
        min(right_ratio["sea"], left_ratio["land"]),
    )
    same_sea_support = min(left_ratio["sea"], right_ratio["sea"])
    same_land_support = min(left_ratio["land"], right_ratio["land"])
    both_inland_water = min(left_ratio["inland_water"], right_ratio["inland_water"])

    if shore_support >= 0.16:
        verdict = "shore_supported"
    elif same_land_support >= 0.28:
        verdict = "land_artifact"
    elif same_sea_support >= 0.40:
        verdict = "sea_artifact"
    elif both_inland_water >= 0.28:
        verdict = "inland_water_artifact"
    else:
        verdict = "needs_visual_review"

    bounds = geometry.bounds
    return {
        "verdict": verdict,
        "shore_support": round(shore_support, 4),
        "same_sea_support": round(same_sea_support, 4),
        "same_land_support": round(same_land_support, 4),
        "inland_water_support": round(both_inland_water, 4),
        "left": {key: round(value, 4) for key, value in left_ratio.items()},
        "right": {key: round(value, 4) for key, value in right_ratio.items()},
        "sample_count": sample_count,
        "bounds": [round(value, 3) for value in bounds],
        "length_px": round(geometry.length, 3),
    }


def draw_review_overlay(image_path: Path, features: list[dict], audits: list[dict], output: Path) -> None:
    image = Image.open(image_path).convert("RGB").resize((2048, 1448), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")
    colors = {
        "shore_supported": (0, 180, 95, 230),
        "land_artifact": (235, 45, 70, 245),
        "sea_artifact": (190, 25, 180, 245),
        "inland_water_artifact": (242, 130, 25, 245),
        "needs_visual_review": (255, 205, 0, 245),
    }
    for feature, audit in zip(features, audits):
        color = (
            (220, 35, 64, 250)
            if feature["properties"].get("review_status") == "rejected"
            else colors[audit["verdict"]]
        )
        geometry = shape(feature["geometry"])
        for line in line_parts(geometry):
            points = [(round(x), round(y)) for x, y in line.coords]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
    image.save(output)


def main() -> None:
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    collection = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    features = collection["features"]
    land_colors = load_land_colors()
    image = np.asarray(Image.open(SOURCE_IMAGE).convert("RGB"), dtype=np.uint8)
    audits = [sample_feature(image, land_colors, feature) for feature in features]
    result = {
        "method": "read-only bilateral pixel audit; no coastline extraction",
        "source_image": SOURCE_IMAGE.name,
        "sea_rgb": SEA_RGB.tolist(),
        "feature_count": len(features),
        "features": [
            {"feature_index": index, **audit, "role": feature["properties"].get("role"),
             "source": feature["properties"].get("source"),
             "review_id": feature["properties"].get("review_id"),
             "review_status": feature["properties"].get("review_status", "pending"),
             "review_note": feature["properties"].get("review_note")}
            for index, (feature, audit) in enumerate(zip(features, audits))
        ],
    }
    (DIAGNOSTICS / "audit.json").write_bytes(
        (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    draw_review_overlay(SOURCE_IMAGE, features, audits, DIAGNOSTICS / "audit_overlay.png")
    counts: dict[str, int] = {}
    for audit in audits:
        counts[audit["verdict"]] = counts.get(audit["verdict"], 0) + 1
    print(f"Audited {len(features)} existing lines without extracting any new coastline.")
    for verdict, count in sorted(counts.items()):
        print(f"{verdict}: {count}")
    rejected_count = sum(feature["properties"].get("review_status") == "rejected" for feature in features)
    print(f"manually rejected non-coast candidates: {rejected_count}")
    print(f"Review files: {DIAGNOSTICS}")


if __name__ == "__main__":
    main()
