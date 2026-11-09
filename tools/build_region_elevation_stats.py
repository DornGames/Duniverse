#!/usr/bin/env python3
"""Precompute elevation summaries for every province and major region."""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WIDTH, HEIGHT = 2048, 1448
POLYGON_SHIFT = 4


def load_js_assignment(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    _, payload = text.split("=", 1)
    return json.loads(payload.strip().removesuffix(";"))


def iter_polygons(geometry: dict):
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        yield coordinates
    elif geometry_type == "MultiPolygon":
        yield from coordinates
    elif geometry_type == "GeometryCollection":
        for child in geometry.get("geometries", []):
            yield from iter_polygons(child)


def fill_ring(mask: np.ndarray, ring: list[list[float]], value: int) -> None:
    if len(ring) < 3:
        return
    points = np.rint(np.asarray(ring, dtype=np.float64) * (1 << POLYGON_SHIFT)).astype(np.int32)
    cv2.fillPoly(mask, [points], value, shift=POLYGON_SHIFT)


def rasterize_features(features: list[dict]) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for feature in features:
        for polygon in iter_polygons(feature.get("geometry", {})):
            if not polygon:
                continue
            fill_ring(mask, polygon[0], 1)
            for hole in polygon[1:]:
                fill_ring(mask, hole, 0)
    return mask.astype(bool)


def point_extreme(elevation: np.ndarray, mask: np.ndarray, maximum: bool) -> dict:
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        raise ValueError("An administrative area has no valid elevation cells")
    values = elevation.ravel()[selected]
    offset = int(np.argmax(values) if maximum else np.argmin(values))
    flat_index = int(selected[offset])
    y, x = divmod(flat_index, WIDTH)
    rounded_elevation = round(float(elevation[y, x]), 1)
    return {
        "elevationM": 0.0 if rounded_elevation == 0 else rounded_elevation,
        "x": int(x),
        "y": int(y),
    }


def build_region_elevation_stats(root: Path = ROOT) -> dict:
    data_dir = root / "data"
    raster = load_js_assignment(data_dir / "relief_elevation.js")
    width, height = int(raster["width"]), int(raster["height"])
    if (width, height) != (WIDTH, HEIGHT):
        raise ValueError(f"Unexpected elevation raster size: {width}x{height}")
    cells = width * height
    codes = np.frombuffer(base64.b64decode(raster["valuesB64"]), dtype="<u2").reshape(height, width)
    elevation = raster["min"] + codes.astype(np.float32) / 65535.0 * (raster["max"] - raster["min"])
    sampled_surface = np.unpackbits(
        np.frombuffer(base64.b64decode(raster["landMaskB64"]), dtype=np.uint8), bitorder="big"
    )[:cells].reshape(height, width).astype(bool)
    river_mask = np.unpackbits(
        np.frombuffer(base64.b64decode(raster["riverWaterMaskB64"]), dtype=np.uint8), bitorder="big"
    )[:cells].reshape(height, width).astype(bool)

    administrative = json.loads((data_dir / "administrative_regions.geojson").read_text(encoding="utf-8"))
    regions = json.loads((data_dir / "region_directory.js").read_text(encoding="utf-8").split("=", 1)[1].strip().removesuffix(";"))
    membership = json.loads((data_dir / "major_region_membership.json").read_text(encoding="utf-8"))
    lakes = json.loads((data_dir / "lakes.geojson").read_text(encoding="utf-8"))

    expected_codes = {entry["code"] for entry in regions if entry["kind"] == "province"}
    features_by_code: dict[str, list[dict]] = {}
    for feature in administrative["features"]:
        code = feature["properties"]["region_code"]
        features_by_code.setdefault(code, []).append(feature)
    actual_codes = set(features_by_code)
    if actual_codes != expected_codes:
        raise ValueError(
            f"Administrative geometry mismatch: missing={sorted(expected_codes - actual_codes)}, "
            f"unexpected={sorted(actual_codes - expected_codes)}"
        )

    lake_mask = rasterize_features(lakes.get("features", []))
    eligible_extrema = sampled_surface & ~river_mask
    dry_land = eligible_extrema & ~lake_mask
    stats: dict[str, dict] = {}

    for entry in regions:
        if entry["kind"] == "province":
            code = entry["code"]
            area_mask = rasterize_features(features_by_code[code])
        elif entry["kind"] == "major":
            major_id = entry["id"].removeprefix("major:")
            declared_codes = next(item["codes"] for item in membership["regions"] if item["id"] == major_id)
            # The source directory documents ND as belonging to the South region;
            # keep its first listed assignment when a membership list repeats it.
            member_codes = []
            for code in declared_codes:
                if code not in features_by_code:
                    continue
                prior_owner = next(
                    (item["id"] for item in membership["regions"] if code in item["codes"]), None
                )
                if prior_owner == major_id:
                    member_codes.append(code)
            member_features = [feature for code in member_codes for feature in features_by_code[code]]
            area_mask = rasterize_features(member_features)
        else:
            continue

        extrema_mask = area_mask & eligible_extrema
        mean_mask = area_mask & dry_land
        if not np.any(mean_mask):
            raise ValueError(f"{entry['id']} contains no dry-land elevation cells")
        mean_elevation = round(float(np.mean(elevation[mean_mask], dtype=np.float64)), 1)
        stats[entry["id"]] = {
            "kind": entry["kind"],
            "meanElevationM": 0.0 if mean_elevation == 0 else mean_elevation,
            "highest": point_extreme(elevation, extrema_mask, maximum=True),
            "lowest": point_extreme(elevation, extrema_mask, maximum=False),
            "meanCellCount": int(np.count_nonzero(mean_mask)),
            "extremaCellCount": int(np.count_nonzero(extrema_mask)),
        }

    output = {
        "source": "data/relief_elevation.js",
        "method": "fixed 16-bit elevation raster; area-weighted pixel mean; administrative polygons",
        "meanMask": "land excluding river and lake water",
        "extremaMask": "land and inland lake basins excluding river water",
        "coordinateReference": "2048x1448 source-map pixels; origin top-left",
        "regions": stats,
    }
    (data_dir / "region_elevation_stats.js").write_bytes(
        (
            "window.DUNIVERSE_REGION_ELEVATION_STATS="
            + json.dumps(output, ensure_ascii=False, separators=(",", ":"))
            + ";\n"
        ).encode("utf-8")
    )

    by_id = {entry["id"]: entry for entry in regions}
    with (data_dir / "region_elevation_stats.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "region_id", "类型", "地区", "平均海拔_m", "最高点海拔_m", "最高点_x", "最高点_y",
            "最低点海拔_m", "最低点_x", "最低点_y", "均值样本数", "极值样本数",
        ])
        for region_id, stat in stats.items():
            entry = by_id[region_id]
            writer.writerow([
                region_id, entry["kind"], entry["name"], stat["meanElevationM"],
                stat["highest"]["elevationM"], stat["highest"]["x"], stat["highest"]["y"],
                stat["lowest"]["elevationM"], stat["lowest"]["x"], stat["lowest"]["y"],
                stat["meanCellCount"], stat["extremaCellCount"],
            ])

    return {
        "regionCount": len(stats),
        "provinceCount": sum(stat["kind"] == "province" for stat in stats.values()),
        "majorRegionCount": sum(stat["kind"] == "major" for stat in stats.values()),
        "meanElevationRangeM": [
            min(stat["meanElevationM"] for stat in stats.values()),
            max(stat["meanElevationM"] for stat in stats.values()),
        ],
        "dataFile": str(data_dir / "region_elevation_stats.js"),
        "reviewFile": str(data_dir / "region_elevation_stats.csv"),
    }


if __name__ == "__main__":
    print(json.dumps(build_region_elevation_stats(), ensure_ascii=False, indent=2))
