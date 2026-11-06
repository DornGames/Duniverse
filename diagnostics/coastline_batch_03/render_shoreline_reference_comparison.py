#!/usr/bin/env python3
"""Compare candidate shoreline pixels in aligned high-resolution reference exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from shapely.affinity import translate
from shapely.geometry import box, shape

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CANDIDATES = [
    ("SHORE-GAP-0002", 1597.7885, 443.6394),
    ("SHORE-GAP-0003", 1610.0250, 445.4000),
    ("SHORE-GAP-0005", 1596.3015, 446.7279),
]


def load_features(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def normalized_coast(feature: dict):
    source = feature.get("properties", {}).get("source", "")
    offset = -11 / 24 if source.endswith("_open_ocean_boundary") else -0.5
    return translate(shape(feature["geometry"]), xoff=offset, yoff=offset)


def to_local(x: float, y: float, left: int, top: int, scale: float) -> tuple[float, float]:
    # The stored, normalized vector uses input-map pixel-centre indices.
    # Map those centres to 12x raster pixel-centre indices, not image edges.
    return ((12 * x + 5.5 - left) * scale, (12 * y + 5.5 - top) * scale)


def draw_overlays(draw: ImageDraw.ImageDraw, bounds: tuple[float, float, float, float], left: int, top: int, scale: float, coasts: list[dict], rivers: list[dict]) -> None:
    region = box(bounds[0] - 2, bounds[1] - 2, bounds[2] + 2, bounds[3] + 2)
    for feature in coasts:
        geom = normalized_coast(feature)
        if not geom.intersects(region):
            continue
        role = feature.get("properties", {}).get("role", "coastline")
        color = "#ed243e" if role == "coastline" else "#9a35dd"
        points = [to_local(x, y, left, top, scale) for x, y, *_ in geom.coords]
        if len(points) > 1:
            draw.line(points, fill=color, width=3, joint="curve")
    for feature in rivers:
        geom = shape(feature["geometry"])
        if not geom.intersects(region):
            continue
        points = [to_local(x, y, left, top, scale) for x, y, *_ in geom.coords]
        if len(points) > 1:
            draw.line(points, fill="#1678e7", width=3, joint="curve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True, help="Folder containing the local 12x reference PNG files")
    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    no_river_path = args.reference_dir / "Administrative Division Map of Donh 2026-no-river-super-big.png"
    no_text_path = args.reference_dir / "Administrative Division Map of Donh 2026-no-text-super-big.png"
    no_river = Image.open(no_river_path).convert("RGB")
    no_text = Image.open(no_text_path).convert("RGB")
    if no_river.size != no_text.size:
        raise SystemExit(f"Reference grids differ: no-river={no_river.size}, no-text={no_text.size}")

    coasts = load_features(ROOT / "data" / "coastlines.geojson")
    rivers: list[dict] = []
    for name in ("hydrology_lines.geojson", "cross_border_rivers.geojson", "seasonal_rivers.geojson"):
        path = ROOT / "data" / name
        if path.exists():
            rivers.extend(load_features(path))

    crop_raw = 192
    panel = 768
    title_h = 34
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (panel * 2, (panel + title_h) * len(CANDIDATES)), "#f4f2ec")
    draw_sheet = ImageDraw.Draw(sheet)
    for row, (issue, x, y) in enumerate(CANDIDATES):
        cx, cy = 12 * (x + 0.5), 12 * (y + 0.5)
        left, top = int(round(cx - crop_raw / 2)), int(round(cy - crop_raw / 2))
        bounds = (x - crop_raw / 24, y - crop_raw / 24, x + crop_raw / 24, y + crop_raw / 24)
        for col, (label, source) in enumerate((("no-river", no_river), ("no-text (rivers visible)", no_text))):
            crop = source.crop((left, top, left + crop_raw, top + crop_raw)).resize((panel, panel), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(crop, "RGBA")
            draw_overlays(draw, bounds, left, top, panel / crop_raw, coasts, rivers)
            px, py = to_local(x, y, left, top, panel / crop_raw)
            draw.ellipse((px - 10, py - 10, px + 10, py + 10), fill=(255, 194, 0, 220), outline=(25, 20, 0, 255), width=3)
            x_out = col * panel
            y_out = row * (panel + title_h)
            sheet.paste(crop, (x_out, y_out + title_h))
            draw_sheet.text((x_out + 8, y_out + 10), f"{issue} ({x:.4f}, {y:.4f}) | {label}", fill="#111111", font=font)
    out_path = OUT / "shore_gap_source_comparison.png"
    sheet.save(out_path)
    no_river.close()
    no_text.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
