#!/usr/bin/env python3
"""Measure and visualize local coastline placement against high-res shore ink."""

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
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CANDIDATES = [
    ("SHORE-GAP-0002", 1597.7885, 443.6394),
    ("SHORE-GAP-0003", 1610.0250, 445.4000),
    ("SHORE-GAP-0005", 1596.3015, 446.7279),
]


def normalized_geometry(feature: dict):
    source = feature.get("properties", {}).get("source", "")
    offset = -11 / 24 if source.endswith("_open_ocean_boundary") else -0.5
    return translate(shape(feature["geometry"]), xoff=offset, yoff=offset)


def to_reference(x: float, y: float, left: int, top: int) -> tuple[float, float]:
    # input_map pixel-centre indices -> 12x raster pixel-centre indices
    return 12 * x + 5.5 - left, 12 * y + 5.5 - top


def sample_geometry(features: list[dict], bounds, left: int, top: int, size: int):
    samples: list[tuple[float, float]] = []
    draw_lines: list[list[tuple[float, float]]] = []
    for feature in features:
        geom = normalized_geometry(feature)
        if not geom.intersects(bounds):
            continue
        clipped = geom.intersection(bounds)
        parts = [clipped] if clipped.geom_type == "LineString" else list(
            getattr(clipped, "geoms", [])
        )
        for line in parts:
            if line.geom_type != "LineString" or len(line.coords) < 2:
                continue
            points = [to_reference(x, y, left, top) for x, y, *_ in line.coords]
            draw_lines.append(points)
            coords = list(line.coords)
            for a, b in zip(coords, coords[1:]):
                length = math.hypot((b[0] - a[0]) * 12, (b[1] - a[1]) * 12)
                steps = max(1, math.ceil(length))
                for i in range(steps + 1):
                    t = i / steps
                    px, py = to_reference(
                        a[0] + t * (b[0] - a[0]),
                        a[1] + t * (b[1] - a[1]),
                        left,
                        top,
                    )
                    if 0 <= px < size and 0 <= py < size:
                        samples.append((px, py))
    return np.asarray(samples, dtype=np.float64), draw_lines


def stats(distance: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    values = distance[y, x]
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.95)),
        float(np.quantile(values, 0.99)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--threshold", type=int, default=70,
        help="grayscale cutoff used to identify dark cartographic ink",
    )
    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    features = json.loads(
        (ROOT / "data" / "coastlines.geojson").read_text(encoding="utf-8")
    )["features"]
    crop_size = 768
    display_size = 768
    title_h = 34
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (display_size, (display_size + title_h) * len(CANDIDATES)), "#f4f2ec")
    sheet_draw = ImageDraw.Draw(sheet)
    result_rows = []

    source = Image.open(args.reference).convert("RGB")
    for row, (issue, x, y) in enumerate(CANDIDATES):
        center_x, center_y = 12 * x + 5.5, 12 * y + 5.5
        left, top = int(round(center_x - crop_size / 2)), int(round(center_y - crop_size / 2))
        crop = source.crop((left, top, left + crop_size, top + crop_size))
        rgb = np.asarray(crop)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dark = gray < args.threshold
        centerline = skeletonize(dark)
        dark_distance = cv2.distanceTransform((~dark).astype(np.uint8), cv2.DIST_L2, 5)
        centerline_distance = cv2.distanceTransform(
            (~centerline).astype(np.uint8), cv2.DIST_L2, 5
        )

        bounds = box(
            (left - 5.5) / 12,
            (top - 5.5) / 12,
            (left + crop_size - 5.5) / 12,
            (top + crop_size - 5.5) / 12,
        )
        points, draw_lines = sample_geometry(features, bounds, left, top, crop_size)
        if len(points) == 0:
            raise SystemExit(f"No coastline samples in local reference crop for {issue}.")
        xi = np.clip(np.rint(points[:, 0]).astype(int), 0, crop_size - 1)
        yi = np.clip(np.rint(points[:, 1]).astype(int), 0, crop_size - 1)
        ink_p50, ink_p95, ink_p99 = stats(dark_distance, xi, yi)
        skel_p50, skel_p95, skel_p99 = stats(centerline_distance, xi, yi)

        # Search a small translation grid as a registration check, not as a
        # geometry correction. A non-zero optimum would suggest map alignment error.
        shifts = []
        for dx in range(-12, 13, 2):
            for dy in range(-12, 13, 2):
                xs = np.clip(np.rint(points[:, 0] + dx).astype(int), 0, crop_size - 1)
                ys = np.clip(np.rint(points[:, 1] + dy).astype(int), 0, crop_size - 1)
                values = centerline_distance[ys, xs]
                shifts.append((float(np.quantile(values, 0.95)), dx, dy))
        best_p95, best_dx, best_dy = min(shifts)

        # Cyan is the thresholded black-ink skeleton; red is the current vector.
        overlay = crop.copy()
        overlay_pixels = np.asarray(overlay).copy()
        overlay_pixels[centerline] = (0, 220, 220)
        overlay = Image.fromarray(overlay_pixels)
        draw = ImageDraw.Draw(overlay)
        for line in draw_lines:
            if len(line) > 1:
                draw.line(line, fill="#ed243e", width=1, joint="curve")
        overlay = overlay.resize((display_size, display_size), Image.Resampling.LANCZOS)
        y_out = row * (display_size + title_h)
        sheet.paste(overlay, (0, y_out + title_h))
        sheet_draw.text(
            (8, y_out + 10),
            f"{issue} ({x:.4f}, {y:.4f}) | red=vector, cyan=dark-ink skeleton",
            fill="#111111",
            font=font,
        )
        result_rows.append(
            {
                "issue_id": issue,
                "x_center_index_px": x,
                "y_center_index_px": y,
                "sample_count": len(points),
                "dark_ink_distance_p50_reference_px": round(ink_p50, 3),
                "dark_ink_distance_p95_reference_px": round(ink_p95, 3),
                "dark_ink_distance_p99_reference_px": round(ink_p99, 3),
                "dark_ink_distance_p95_input_map_px": round(ink_p95 / 12, 4),
                "ink_skeleton_distance_p50_reference_px": round(skel_p50, 3),
                "ink_skeleton_distance_p95_reference_px": round(skel_p95, 3),
                "ink_skeleton_distance_p95_input_map_px": round(skel_p95 / 12, 4),
                "best_grid_shift_x_reference_px": best_dx,
                "best_grid_shift_y_reference_px": best_dy,
                "best_shift_skeleton_p95_reference_px": round(best_p95, 3),
                "ink_threshold_grayscale_lt": args.threshold,
            }
        )
        crop.close()
        overlay.close()

    source.close()
    sheet.save(OUT / "coastline_stroke_centerline_review.png")
    with (OUT / "coastline_stroke_fit_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"Wrote {OUT / 'coastline_stroke_centerline_review.png'}")
    print(f"Wrote {OUT / 'coastline_stroke_fit_metrics.csv'}")


if __name__ == "__main__":
    main()
