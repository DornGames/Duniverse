#!/usr/bin/env python3
"""Trace the open-ocean coastline from small, independently processed tiles.

The official source is a very large PNG, so this tool never creates a
full-resolution mask for the whole map. It labels 1024 px tiles separately,
joins only their edge labels to find water connected to the map's right edge,
then extracts contours from overlapping tile windows and clips each result to
its non-overlapping core. The no-text export is used to distinguish land from
labels in the no-river export.

Example:
  python tools/extract_official_coastline_tiled.py \
    --source-dir "../official_map" --diagnostics-dir "../diagnostics/coastline_tiles"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import LineString, box
from shapely.ops import linemerge, unary_union

Image.MAX_IMAGE_PIXELS = None


SOURCE_NO_RIVER = "Administrative Division Map of Donh 2026-no-river-super-big.png"
SOURCE_NO_TEXT = "Administrative Division Map of Donh 2026-no-text-super-big.png"
SOURCE_SCALE = 12.0
CANVAS_WIDTH = 2048
CANVAS_HEIGHT = 1448
MASK_RGB_LOW = (158, 188, 216)
MASK_RGB_HIGH = (194, 225, 255)
MIN_BLUE_RED_DIFFERENCE = 35
INK_THRESHOLD = 72
DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 48


@dataclass(frozen=True)
class Tile:
    column: int
    row: int
    x: int
    y: int
    width: int
    height: int


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    def add(self, item: tuple[int, int, int]) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: tuple[int, int, int]) -> tuple[int, int, int]:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(
        self,
        first: tuple[int, int, int],
        second: tuple[int, int, int],
    ) -> None:
        if first not in self.parent or second not in self.parent:
            return
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a != root_b:
            self.parent[root_b] = root_a


def sea_mask(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (
        (red >= MASK_RGB_LOW[0])
        & (red <= MASK_RGB_HIGH[0])
        & (green >= MASK_RGB_LOW[1])
        & (green <= MASK_RGB_HIGH[1])
        & (blue >= MASK_RGB_LOW[2])
        & (blue <= MASK_RGB_HIGH[2])
        & ((blue.astype(np.int16) - red.astype(np.int16)) >= MIN_BLUE_RED_DIFFERENCE)
    )


def tile_grid(width: int, height: int, tile_size: int) -> list[Tile]:
    return [
        Tile(column, row, x, y, min(tile_size, width - x), min(tile_size, height - y))
        for row, y in enumerate(range(0, height, tile_size))
        for column, x in enumerate(range(0, width, tile_size))
    ]


def node(tile: Tile, label: int) -> tuple[int, int, int]:
    return tile.column, tile.row, int(label)


def union_edge_pairs(
    union_find: UnionFind,
    tile_a: Tile,
    labels_a: np.ndarray,
    tile_b: Tile,
    labels_b: np.ndarray,
    *,
    vertical: bool,
) -> None:
    """Join local water components that touch across neighboring tile edges."""
    # The caller supplies the one-dimensional edge label arrays. ``vertical``
    # is retained in the signature to make call sites self-documenting.
    edge_a, edge_b = labels_a, labels_b
    shifts = (-1, 0, 1)

    length = min(len(edge_a), len(edge_b))
    edge_a, edge_b = edge_a[:length], edge_b[:length]
    for shift in shifts:
        if shift < 0:
            a, b = edge_a[1:], edge_b[:-1]
        elif shift > 0:
            a, b = edge_a[:-1], edge_b[1:]
        else:
            a, b = edge_a, edge_b
        valid = (a > 0) & (b > 0)
        for label_a, label_b in zip(a[valid].tolist(), b[valid].tolist()):
            union_find.union(node(tile_a, label_a), node(tile_b, label_b))


def local_components(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    count, labels = cv2.connectedComponents(sea_mask(rgb).astype(np.uint8), connectivity=8)
    return labels, count


def ocean_component_roots(
    source: Image.Image,
    tiles: list[Tile],
    tile_size: int,
    *,
    seed_side: str = "right",
) -> tuple[UnionFind, set[tuple[int, int, int]], dict[tuple[int, int], dict[str, np.ndarray]]]:
    union_find = UnionFind()
    edge_labels: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    tile_by_position = {(tile.column, tile.row): tile for tile in tiles}
    seeds: set[tuple[int, int, int]] = set()

    for tile in tiles:
        crop = source.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height)).convert("RGB")
        rgb = np.asarray(crop)
        labels, count = local_components(rgb)
        for label in range(1, count):
            union_find.add(node(tile, label))

        edges = {
            "top": labels[0, :].copy(),
            "bottom": labels[-1, :].copy(),
            "left": labels[:, 0].copy(),
            "right": labels[:, -1].copy(),
        }
        edge_labels[(tile.column, tile.row)] = edges

        left_tile = tile_by_position.get((tile.column - 1, tile.row))
        if left_tile:
            union_edge_pairs(
                union_find,
                left_tile,
                edge_labels[(left_tile.column, left_tile.row)]["right"],
                tile,
                edges["left"],
                vertical=False,
            )
        upper_tile = tile_by_position.get((tile.column, tile.row - 1))
        if upper_tile:
            union_edge_pairs(
                union_find,
                upper_tile,
                edge_labels[(upper_tile.column, upper_tile.row)]["bottom"],
                tile,
                edges["top"],
                vertical=True,
            )

        # A diagonal touch can occur exactly at a four-tile corner. Preserve
        # 8-connected sea without creating one giant image-wide label array.
        upper_left = tile_by_position.get((tile.column - 1, tile.row - 1))
        if upper_left:
            label_a = int(edge_labels[(upper_left.column, upper_left.row)]["bottom"][-1])
            label_b = int(edges["top"][0])
            union_find.union(node(upper_left, label_a), node(tile, label_b))
        upper_right = tile_by_position.get((tile.column + 1, tile.row - 1))
        if upper_right:
            label_a = int(edge_labels[(upper_right.column, upper_right.row)]["bottom"][0])
            label_b = int(edges["top"][-1])
            union_find.union(node(upper_right, label_a), node(tile, label_b))

        if seed_side == "right" and tile.column == max(t.column for t in tiles):
            seeds.update(node(tile, label) for label in np.unique(edges["right"]) if label > 0)
        elif seed_side == "left" and tile.column == min(t.column for t in tiles):
            seeds.update(node(tile, label) for label in np.unique(edges["left"]) if label > 0)
        elif seed_side == "top" and tile.row == min(t.row for t in tiles):
            seeds.update(node(tile, label) for label in np.unique(edges["top"]) if label > 0)
        elif seed_side == "bottom" and tile.row == max(t.row for t in tiles):
            seeds.update(node(tile, label) for label in np.unique(edges["bottom"]) if label > 0)

    if not seeds:
        raise RuntimeError(f"No sea-color components touch the requested {seed_side} seed edge")
    roots = {union_find.find(seed) for seed in seeds}
    return union_find, roots, edge_labels


def selected_core_mask(
    rgb: np.ndarray,
    tile: Tile,
    union_find: UnionFind,
    ocean_roots: set[tuple[int, int, int]],
) -> np.ndarray:
    labels, count = local_components(rgb)
    selected_ids = [
        label
        for label in range(1, count)
        if node(tile, label) in union_find.parent
        and union_find.find(node(tile, label)) in ocean_roots
    ]
    return np.isin(labels, selected_ids)


def has_hollow_dark_outline(rgb: np.ndarray, component: np.ndarray) -> bool:
    ys, xs = np.where(component)
    if len(xs) == 0:
        return False
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    dark = (rgb[y0:y1, x0:x1].max(axis=2) < INK_THRESHOLD).astype(np.uint8)
    dark &= component[y0:y1, x0:x1].astype(np.uint8)
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return bool(hierarchy is not None and np.any(hierarchy[0, :, 3] >= 0))


def remove_text_and_filled_markers(
    source_rgb: np.ndarray,
    no_text_rgb: np.ndarray,
    selected_ocean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill false land holes made by labels and solid city symbols."""
    selected = selected_ocean.copy()
    removed = np.zeros_like(selected, dtype=bool)
    no_text_water = sea_mask(no_text_rgb)
    gaps = (~selected).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=8)
    height, width = selected.shape

    for label in range(1, count):
        x, y, box_width, box_height, _area = map(int, stats[label])
        # A piece touching a tile window edge may be mainland or an unfinished
        # segment. Do not classify it as an offshore island or marker.
        if x == 0 or y == 0 or x + box_width >= width or y + box_height >= height:
            continue
        component = labels == label
        sample = source_rgb[component]
        if sample.size == 0:
            continue
        comparison_water_fraction = float(no_text_water[component].mean())
        dark_fraction = float((sample.max(axis=1) < INK_THRESHOLD).mean())
        if comparison_water_fraction >= 0.55 or (
            dark_fraction >= 0.5 and not has_hollow_dark_outline(source_rgb, component)
        ):
            selected[component] = True
            removed[component] = True
    return selected, removed


def contour_segments(
    mask: np.ndarray,
    origin_x: int,
    origin_y: int,
    core: tuple[int, int, int, int],
    full_size: tuple[int, int],
) -> list[LineString]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    core_x, core_y, core_width, core_height = core
    core_box = box(core_x, core_y, core_x + core_width, core_y + core_height)
    full_width, full_height = full_size
    lines: list[LineString] = []

    for contour in contours:
        points = contour[:, 0, :]
        if len(points) < 2:
            continue
        pairs = zip(points.tolist(), points[1:].tolist() + points[:1].tolist())
        segments = []
        mask_height, mask_width = mask.shape
        for first, second in pairs:
            x1, y1 = first
            x2, y2 = second
            along_crop_edge = (
                (x1 == x2 == 0)
                or (x1 == x2 == mask_width - 1)
                or (y1 == y2 == 0)
                or (y1 == y2 == mask_height - 1)
            )
            if along_crop_edge:
                continue
            gx1, gy1 = origin_x + x1, origin_y + y1
            gx2, gy2 = origin_x + x2, origin_y + y2
            # The source raster frame is not a shoreline. Drop any contour run
            # drawn exactly along its outermost rows or columns.
            on_image_frame = (
                (gx1 == gx2 == 0)
                or (gx1 == gx2 == full_width - 1)
                or (gy1 == gy2 == 0)
                or (gy1 == gy2 == full_height - 1)
            )
            if on_image_frame:
                continue
            segments.append(LineString([(gx1, gy1), (gx2, gy2)]))
        if not segments:
            continue
        merged = linemerge(unary_union(segments))
        parts = [merged] if merged.geom_type == "LineString" else list(getattr(merged, "geoms", []))
        for part in parts:
            if part.geom_type != "LineString" or part.length < 1:
                continue
            clipped = part.intersection(core_box)
            if clipped.is_empty:
                continue
            clipped_parts = [clipped] if clipped.geom_type == "LineString" else list(getattr(clipped, "geoms", []))
            lines.extend(p for p in clipped_parts if p.geom_type == "LineString" and p.length >= 1)
    return lines


def tile_window(
    source: Image.Image,
    tile: Tile,
    overlap: int,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, int, int]:
    x0 = max(0, tile.x - overlap)
    y0 = max(0, tile.y - overlap)
    x1 = min(image_width, tile.x + tile.width + overlap)
    y1 = min(image_height, tile.y + tile.height + overlap)
    rgb = np.asarray(source.crop((x0, y0, x1, y1)).convert("RGB"))
    return rgb, x0, y0


def trace_tile(
    source: Image.Image,
    no_text: Image.Image,
    tile: Tile,
    union_find: UnionFind,
    ocean_roots: set[tuple[int, int, int]],
    overlap: int,
    image_width: int,
    image_height: int,
    diagnostics_dir: Path | None,
) -> list[LineString]:
    core_rgb = np.asarray(
        source.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height)).convert("RGB")
    )
    core_ocean = selected_core_mask(core_rgb, tile, union_find, ocean_roots)
    if not core_ocean.any():
        return []

    window_rgb, x0, y0 = tile_window(source, tile, overlap, image_width, image_height)
    window_no_text, _, _ = tile_window(no_text, tile, overlap, image_width, image_height)
    window_labels, window_count = local_components(window_rgb)
    local_x = tile.x - x0
    local_y = tile.y - y0
    window_core_labels = window_labels[
        local_y : local_y + tile.height,
        local_x : local_x + tile.width,
    ]
    selected_local_ids = np.unique(window_core_labels[core_ocean])
    selected_local_ids = selected_local_ids[selected_local_ids != 0]
    if len(selected_local_ids) == 0:
        return []
    selected_window = np.isin(window_labels, selected_local_ids)
    selected_window, removed = remove_text_and_filled_markers(
        window_rgb, window_no_text, selected_window
    )

    lines = contour_segments(
        selected_window,
        x0,
        y0,
        (tile.x, tile.y, tile.width, tile.height),
        (image_width, image_height),
    )

    if diagnostics_dir is not None and lines:
        overlay = core_rgb.copy()
        contours, _ = cv2.findContours(
            selected_window.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        offset = (x0 - tile.x, y0 - tile.y)
        cv2.drawContours(overlay, contours, -1, (255, 24, 180), 2, offset=offset)
        local_removed = removed[
            local_y : local_y + tile.height,
            local_x : local_x + tile.width,
        ]
        overlay[local_removed] = (
            overlay[local_removed].astype(np.float32) * 0.55
            + np.array([255, 220, 0], dtype=np.float32) * 0.45
        ).astype(np.uint8)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(
            diagnostics_dir / f"tile_{tile.x:05d}_{tile.y:05d}.png"
        )
    return lines


def merged_features(lines: Iterable[LineString], tile_size: int) -> list[dict]:
    line_list = list(lines)
    if not line_list:
        return []
    merged = linemerge(unary_union(line_list))
    parts = [merged] if merged.geom_type == "LineString" else list(getattr(merged, "geoms", []))
    features: list[dict] = []
    review_id = 0
    for part in parts:
        if part.geom_type != "LineString" or part.length < 0.05:
            continue
        canvas_line = LineString([(x / SOURCE_SCALE, y / SOURCE_SCALE) for x, y in part.coords])
        canvas_line = canvas_line.simplify(0.08, preserve_topology=True)
        if canvas_line.is_empty or canvas_line.length < 0.05:
            continue
        role = "island_shoreline" if canvas_line.is_ring else "coastline"
        coordinates = [[round(x, 3), round(y, 3)] for x, y in canvas_line.coords]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "role": role,
                    "length_px": round(canvas_line.length, 3),
                    "source": (
                        "official_no_river_highres_tiled_island_ink"
                        if role == "island_shoreline"
                        else "official_no_river_highres_tiled_open_ocean_boundary"
                    ),
                    "tile_size_px": tile_size,
                    "review_id": f"official-tile-{review_id:04d}",
                },
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
        review_id += 1
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/coastlines.geojson"))
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--seed-side", choices=("right", "left", "top", "bottom"), default="right")
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument(
        "--bounds",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional image-pixel window for small-area review; right-edge seeding applies within it.",
    )
    args = parser.parse_args()

    if args.tile_size < 128 or args.overlap < 2:
        parser.error("tile-size must be at least 128 and overlap at least 2 pixels")

    source_path = args.source_dir / SOURCE_NO_RIVER
    no_text_path = args.source_dir / SOURCE_NO_TEXT
    source = Image.open(source_path)
    no_text = Image.open(no_text_path)
    if source.size != no_text.size:
        raise RuntimeError(f"Source variants differ in size: {source.size} vs {no_text.size}")

    map_width = (CANVAS_WIDTH * int(SOURCE_SCALE))
    map_height = (CANVAS_HEIGHT * int(SOURCE_SCALE))
    image_width = min(source.width, map_width)
    image_height = min(source.height, map_height)
    if args.bounds:
        x0, y0, x1, y1 = args.bounds
        x0, y0 = max(0, x0), max(0, y0)
        image_width, image_height = min(image_width, x1), min(image_height, y1)
        if x0 >= image_width or y0 >= image_height:
            raise RuntimeError("Requested review bounds do not intersect the source image")
        # Crop origins remain aligned to the local tile grid, allowing edge
        # stitching to work within a small review region.
        source = source.crop((x0, y0, image_width, image_height))
        no_text = no_text.crop((x0, y0, image_width, image_height))
        image_width -= x0
        image_height -= y0
        coordinate_offset_x, coordinate_offset_y = x0, y0
    else:
        coordinate_offset_x = coordinate_offset_y = 0

    tiles = tile_grid(image_width, image_height, args.tile_size)
    print(
        f"Processing {len(tiles)} tiles ({args.tile_size}px cores, {args.overlap}px contour overlap); "
        f"each tile is classified independently."
    )
    union_find, ocean_roots, _edge_labels = ocean_component_roots(
        source, tiles, args.tile_size, seed_side=args.seed_side
    )
    print(f"Found {len(ocean_roots)} tile-connected ocean component(s) at the seed edge.")

    all_lines: list[LineString] = []
    for index, tile in enumerate(tiles, start=1):
        shifted_tile = Tile(
            tile.column,
            tile.row,
            tile.x + coordinate_offset_x,
            tile.y + coordinate_offset_y,
            tile.width,
            tile.height,
        )
        lines = trace_tile(
            source,
            no_text,
            tile,
            union_find,
            ocean_roots,
            args.overlap,
            image_width,
            image_height,
            args.diagnostics_dir,
        )
        if coordinate_offset_x or coordinate_offset_y:
            translated = []
            for line in lines:
                translated.append(
                    LineString(
                        [(x + coordinate_offset_x, y + coordinate_offset_y) for x, y in line.coords]
                    )
                )
            lines = translated
        all_lines.extend(lines)
        print(f"  tile {index:>3}/{len(tiles)} at {shifted_tile.x},{shifted_tile.y}: {len(lines)} coast segment(s)")

    features = merged_features(all_lines, args.tile_size)
    collection = {
        "type": "FeatureCollection",
        "name": "official_high_resolution_tiled_coastlines",
        "coordinate_reference_note": (
            "Image pixel coordinates; origin top-left; x right; y down. "
            "Not georeferenced. Source coordinates scaled from 24576x17376 by 12."
        ),
        "source_layers": [SOURCE_NO_RIVER, SOURCE_NO_TEXT],
        "tile_size_px": args.tile_size,
        "tile_overlap_px": args.overlap,
        "ocean_seed_side": args.seed_side,
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Wrote {len(features)} coastline feature(s), "
        f"including {sum(f['properties']['role'] == 'island_shoreline' for f in features)} closed islands, "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
