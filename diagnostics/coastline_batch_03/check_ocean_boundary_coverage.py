#!/usr/bin/env python3
"""Recheck the three strongest raster shoreline candidates at 12x resolution."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.affinity import translate
from shapely.geometry import box, shape


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CANDIDATES = [
    ("SHORE-GAP-0002", 1597.7885, 443.6394, 1595.75, 443.5, 1600.25, 445.75),
    ("SHORE-GAP-0003", 1610.0250, 445.4000, 1609.75, 445.25, 1611.25, 446.5),
    ("SHORE-GAP-0005", 1596.3015, 446.7279, 1596.0, 446.5, 1597.75, 448.25),
]


def normalized_geometry(feature: dict):
    source = feature.get("properties", {}).get("source", "")
    offset = -11 / 24 if source.endswith("_open_ocean_boundary") else -0.5
    return translate(shape(feature["geometry"]), xoff=offset, yoff=offset)


def local_pixels(geom, left: int, top: int, size: int) -> list[list[tuple[int, int]]]:
    bounds = box((left - 5.5) / 12, (top - 5.5) / 12,
                 (left + size - 5.5) / 12, (top + size - 5.5) / 12)
    if not geom.intersects(bounds):
        return []
    clipped = geom.intersection(bounds)
    parts = [clipped] if clipped.geom_type == "LineString" else list(getattr(clipped, "geoms", []))
    result = []
    for line in parts:
        if line.geom_type != "LineString" or len(line.coords) < 2:
            continue
        points: list[tuple[int, int]] = []
        coords = list(line.coords)
        for a, b in zip(coords, coords[1:]):
            length = math.hypot((b[0] - a[0]) * 12, (b[1] - a[1]) * 12)
            steps = max(1, math.ceil(length))
            for i in range(steps + 1):
                t = i / steps
                x = 12 * (a[0] + t * (b[0] - a[0])) + 5.5 - left
                y = 12 * (a[1] + t * (b[1] - a[1])) + 5.5 - top
                points.append((int(round(x)), int(round(y))))
        if len(points) > 1:
            result.append(points)
    return result


def ocean_mask(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = (rgb[..., i].astype(np.int16) for i in range(3))
    return (
        (red >= 158) & (red <= 194)
        & (green >= 188) & (green <= 225)
        & (blue >= 216) & (blue <= 255)
        & (blue - red >= 35)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--uncovered-threshold-reference-px", type=float, default=9.0)
    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    features = json.loads(
        (ROOT / "data" / "coastlines.geojson").read_text(encoding="utf-8")
    )["features"]
    layers = []
    for filename, label in (
        ("Administrative Division Map of Donh 2026-no-river-super-big.png", "no-river"),
        ("Administrative Division Map of Donh 2026-no-text-super-big.png", "no-text"),
    ):
        layers.append((label, Image.open(args.reference_dir / filename).convert("RGB")))

    crop_size = 384
    panel_size = 768
    title_h = 34
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (panel_size * 2, (panel_size + title_h) * len(CANDIDATES)), "#f4f2ec")
    sheet_draw = ImageDraw.Draw(sheet)
    results = []

    for row, (issue, x, y, bx0, by0, bx1, by1) in enumerate(CANDIDATES):
        cx, cy = 12 * x + 5.5, 12 * y + 5.5
        left, top = int(round(cx - crop_size / 2)), int(round(cy - crop_size / 2))
        center_box = (
            math.ceil(12 * bx0 - left - 0.5), math.ceil(12 * by0 - top - 0.5),
            math.ceil(12 * bx1 - left - 0.5), math.ceil(12 * by1 - top - 0.5),
        )
        for col, (label, source) in enumerate(layers):
            crop = source.crop((left, top, left + crop_size, top + crop_size))
            rgb = np.asarray(crop)
            water = ocean_mask(rgb).astype(np.uint8)
            count, labels = cv2.connectedComponents(water, connectivity=8)
            right_labels = set(np.unique(labels[:, -1]).tolist()) - {0}
            ocean = np.isin(labels, list(right_labels)) if right_labels else np.zeros_like(water, dtype=bool)
            edge = ocean & (cv2.erode(ocean.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
            lines = []
            vector_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
            crop_bounds = box(
                (left - 5.5) / 12,
                (top - 5.5) / 12,
                (left + crop_size - 5.5) / 12,
                (top + crop_size - 5.5) / 12,
            )
            for feature in features:
                geom = normalized_geometry(feature)
                if not geom.intersects(crop_bounds):
                    continue
                for points in local_pixels(geom, left, top, crop_size):
                    arr = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vector_mask, [arr], False, 1, 1, lineType=cv2.LINE_8)
                    lines.append((feature.get("properties", {}).get("role", "coastline"), points))
            distance = cv2.distanceTransform((vector_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
            x0, y0, x1, y1 = center_box
            x0, x1 = max(0, x0), min(crop_size, x1)
            y0, y1 = max(0, y0), min(crop_size, y1)
            boundary_roi = edge[y0:y1, x0:x1]
            distance_roi = distance[y0:y1, x0:x1]
            uncovered = boundary_roi & (distance_roi > args.uncovered_threshold_reference_px)
            distances = distance_roi[boundary_roi]
            centroids = []
            if uncovered.any():
                yy, xx = np.where(uncovered)
                centroids = [[round(float(x0 + xx.mean()), 2), round(float(y0 + yy.mean()), 2)]]
            results.append({
                "issue_id": issue,
                "reference_layer": label,
                "right_edge_ocean_component_count": len(right_labels),
                "candidate_ocean_boundary_pixel_count": int(boundary_roi.sum()),
                "candidate_boundary_pixels_over_threshold": int(uncovered.sum()),
                "threshold_reference_px": args.uncovered_threshold_reference_px,
                "candidate_boundary_distance_p50_reference_px": round(float(np.median(distances)), 3) if len(distances) else "",
                "candidate_boundary_distance_p95_reference_px": round(float(np.quantile(distances, .95)), 3) if len(distances) else "",
                "uncovered_centroids_local_px": json.dumps(centroids),
            })

            # Orange marks only ocean-connected boundary pixels outside tolerance.
            draw = ImageDraw.Draw(crop)
            if uncovered.any():
                yy, xx = np.where(uncovered)
                draw_points = [(int(x0 + dx), int(y0 + dy)) for dx, dy in zip(xx, yy)]
                for point in draw_points:
                    draw.point(point, fill="#ff8a00")
            for role, points in lines:
                draw.line(points, fill="#9a35dd" if role == "island_shoreline" else "#ed243e", width=1)
            px, py = int(round(cx - left)), int(round(cy - top))
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#ffc200", outline="#191400")
            crop = crop.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
            x_out = col * panel_size
            y_out = row * (panel_size + title_h)
            sheet.paste(crop, (x_out, y_out + title_h))
            sheet_draw.text(
                (x_out + 8, y_out + 10),
                f"{issue} | {label} | orange=uncovered ocean boundary",
                fill="#111111",
                font=font,
            )
            crop.close()

    for _, layer in layers:
        layer.close()
    sheet.save(OUT / "uncovered_ocean_boundary_review.png")
    with (OUT / "uncovered_ocean_boundary_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {OUT / 'uncovered_ocean_boundary_review.png'}")
    print(f"Wrote {OUT / 'uncovered_ocean_boundary_metrics.csv'}")


if __name__ == "__main__":
    main()
