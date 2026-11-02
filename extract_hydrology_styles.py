#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Separate the hand-checked seasonal rivers and the ruler-marked canal.

The map is not georeferenced. Coordinates below are traced in source-image
pixels after checking the original raster. The southeast coastal mesh stays in
the ordinary river layer because it is an estuary, not a canal.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import LineString, shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOURCE = ROOT / "input_map.png"
DIAGNOSTICS = ROOT / "diagnostics"

SEASONAL_RIVER_PATHS = [
    # Traced along the centers of the blue dashes in input_map.png. The old
    # trace wandered tens of pixels off the source around the reservoir and
    # the southern bend.
    [
        (128, 333), (148, 331), (165, 333), (180, 340), (194, 351),
        (208, 355), (227, 355), (244, 355), (258, 362), (270, 375),
        (281, 384), (289, 385), (296, 378), (301, 362), (308, 350),
        (321, 344), (337, 343), (354, 344), (370, 346), (383, 341),
        (394, 331), (405, 324), (416, 321), (426, 325),
        (438, 337), (451, 345), (466, 352), (483, 361), (501, 370),
        (518, 379), (529, 392), (532, 408), (530, 425), (526, 441),
        (528, 458), (538, 476), (554, 492), (574, 507), (588, 520),
        (591, 533),
    ],
    [
        (232, 812), (220, 807), (208, 804), (195, 804), (184, 807),
        (177, 813), (174, 820), (177, 831), (180, 842),
        (180, 850), (175, 861), (170, 874), (169, 887), (174, 897),
        (187, 903), (199, 910), (206, 919), (204, 928), (195, 938),
        (187, 948), (182, 960), (181, 972), (185, 984), (196, 993),
        (207, 999),
    ],
]

# The source has two separate ruler-marked canals. Keep their actual straight
# centerlines and draw the ruler ticks above the line, as in the raster.
CANAL_CENTERLINES = [
    {
        "canal_id": 1,
        "name": "南部运河",
        "line": LineString([
            (1127, 1144), (1160, 1144), (1190, 1144), (1220, 1144),
            (1250, 1144), (1280, 1144), (1310, 1144), (1340, 1144),
            (1370, 1144), (1400, 1144), (1420, 1144),
        ]),
    },
    {
        "canal_id": 2,
        "name": "南部底部运河",
        "line": LineString([(970, 1375), (1030, 1375), (1100, 1375), (1170, 1375),
                            (1240, 1375), (1310, 1375), (1365, 1375)]),
    },
]
CANAL_TICK_SPACING_PX = 16.0
CANAL_TICK_LENGTH_PX = 9.0
CANAL_RIVER_EXCLUSION_BUFFER_PX = 3.5
SEASONAL_CROSS_BORDER_EXCLUSION_BUFFER_PX = 2.5


def read_fc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_fc(path: Path, name: str, features: list[dict]) -> None:
    path.write_text(
        json.dumps({
            "type": "FeatureCollection",
            "name": name,
            "coordinate_reference_note": (
                "Image pixel coordinates; origin top-left; x right; y down. "
                "Not georeferenced."
            ),
            "features": features,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def line_parts(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in line_parts(item)]
    return []


def round_line(line: LineString) -> dict:
    return {
        "type": "LineString",
        "coordinates": [[round(x, 2), round(y, 2)] for x, y in line.coords],
    }


def seasonal_features() -> list[dict]:
    features = []
    for river_id, coordinates in enumerate(SEASONAL_RIVER_PATHS, start=1):
        line = LineString(coordinates).simplify(0.4, preserve_topology=False)
        features.append({
            "type": "Feature",
            "properties": {
                "river_id": river_id,
                "name": f"邻国季节性河流 {river_id}",
                "territory": "neighboring_country",
                "flow_regime": "seasonal",
                "symbol": "blue_dashed_river",
                "source": "manual_trace_from_source_image",
                "review_status": "visually_checked_against_dashed_blue_strokes",
                "length_px": round(line.length, 1),
            },
            "geometry": round_line(line),
        })
    return features


def canal_features() -> list[dict]:
    features = []
    for canal in CANAL_CENTERLINES:
        centerline = canal["line"]
        common = {
            "canal_id": canal["canal_id"],
            "name": canal["name"],
            "source": "manual_trace_from_ruler_marked_symbol",
        }
        features.append({
            "type": "Feature",
            "properties": {
                **common,
                "symbol_part": "centerline",
                "review_status": "traced_from_source_raster",
                "length_px": round(centerline.length, 1),
            },
            "geometry": round_line(centerline),
        })
        count = max(0, int(centerline.length // CANAL_TICK_SPACING_PX) - 1)
        for tick_id in range(1, count + 1):
            distance = tick_id * CANAL_TICK_SPACING_PX
            point = centerline.interpolate(distance)
            before = centerline.interpolate(max(0, distance - 1))
            after = centerline.interpolate(min(centerline.length, distance + 1))
            dx, dy = after.x - before.x, after.y - before.y
            magnitude = math.hypot(dx, dy) or 1.0
            # Image coordinates grow downward. This normal points to the upper
            # side of the horizontal source symbol.
            nx, ny = dy / magnitude, -dx / magnitude
            tick = LineString([
                (point.x, point.y),
                (point.x + nx * CANAL_TICK_LENGTH_PX,
                 point.y + ny * CANAL_TICK_LENGTH_PX),
            ])
            features.append({
                "type": "Feature",
                "properties": {
                    **common,
                    "tick_id": tick_id,
                    "symbol_part": "ruler_tick",
                },
                "geometry": round_line(tick),
            })
    return features


def remove_canal_from_rivers(features: list[dict]) -> list[dict]:
    corridor = unary_union([
        canal["line"].buffer(CANAL_RIVER_EXCLUSION_BUFFER_PX, cap_style="flat")
        for canal in CANAL_CENTERLINES
    ])
    result = []
    for feature in features:
        geometry = shape(feature["geometry"]).difference(corridor)
        for part in line_parts(geometry):
            if part.length < 2:
                continue
            result.append({
                "type": "Feature",
                "properties": {
                    **feature["properties"],
                    "segment_id": len(result) + 1,
                    "length_px": round(part.length, 1),
                },
                "geometry": round_line(part),
            })
    return result


def remove_seasonal_from_cross_border(features: list[dict], seasonal: list[dict]) -> list[dict]:
    corridor = unary_union([
        shape(feature["geometry"]).buffer(SEASONAL_CROSS_BORDER_EXCLUSION_BUFFER_PX,
                                           cap_style="round")
        for feature in seasonal
    ])
    result = []
    for feature in features:
        geometry = shape(feature["geometry"]).difference(corridor)
        for part in line_parts(geometry):
            if part.length < 2:
                continue
            result.append({
                "type": "Feature",
                "properties": {
                    **feature["properties"],
                    "segment_id": len(result) + 1,
                    "length_px": round(part.length, 1),
                },
                "geometry": round_line(part),
            })
    return result


def path_data(geometry: dict) -> str:
    commands = []
    for line in line_parts(shape(geometry)):
        coordinates = list(line.coords)
        if len(coordinates) < 2:
            continue
        point_text = lambda p: f"{p[0]:.2f},{p[1]:.2f}"
        if len(coordinates) < 3:
            commands.append(f"M {point_text(coordinates[0])} L {point_text(coordinates[-1])}")
            continue
        radius = 2.0
        toward = lambda point, target: (
            point[0] + (target[0] - point[0]) * min(radius / (math.dist(point, target) or 1), 0.4),
            point[1] + (target[1] - point[1]) * min(radius / (math.dist(point, target) or 1), 0.4),
        )
        data = f"M {point_text(coordinates[0])}"
        for index in range(1, len(coordinates) - 1):
            point = coordinates[index]
            data += f" L {point_text(toward(point, coordinates[index - 1]))}"
            data += f" Q {point_text(point)} {point_text(toward(point, coordinates[index + 1]))}"
        data += f" L {point_text(coordinates[-1])}"
        commands.append(data)
    return " ".join(commands)


def render_group(group_id: str, layer_id: str, features: list[dict], style: str) -> str:
    paths = []
    for feature in features:
        d = path_data(feature["geometry"])
        if d:
            paths.append(f'<path d="{d}"/>')
    return (
        f'<g id="{group_id}" data-layer="{layer_id}" {style} '
        f'pointer-events="none">\n' + "\n".join(paths) + "\n</g>"
    )


def render_preview_group(group_id: str, features: list[dict], style: str) -> str:
    paths = []
    for feature in features:
        d = path_data(feature["geometry"])
        if d:
            paths.append(f'<path d="{d}"/>')
    return f'<g id="{group_id}" {style} pointer-events="none">\n' + "\n".join(paths) + "\n</g>"


def upsert_group(source: str, group: str, group_id: str, after_id: str | None) -> str:
    pattern = rf'<g id="{re.escape(group_id)}"[^>]*>.*?</g>'
    updated, count = re.subn(pattern, lambda _: group, source, count=1, flags=re.DOTALL)
    if count:
        return updated
    if after_id is None:
        raise RuntimeError(f"Cannot insert {group_id} without an insertion point")
    target = re.search(rf'<g id="{re.escape(after_id)}"[^>]*>.*?</g>', source, flags=re.DOTALL)
    if not target:
        raise RuntimeError(f"Could not find insertion group {after_id}")
    return source[:target.end()] + group + source[target.end():]


def update_render_files(rivers: list[dict], canals: list[dict], seasonal: list[dict], cross_border: list[dict]) -> None:
    index_groups = [
        render_group("rivers", "rivers", rivers,
                     'fill="none" stroke="#83b8e8" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"'),
        render_group("canals", "canals", [f for f in canals if f["properties"].get("symbol_part") == "centerline"],
                     'fill="none" stroke="#347ca5" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"'),
        render_group("canalTicks", "canals", [f for f in canals if f["properties"].get("symbol_part") == "ruler_tick"],
                     'fill="none" stroke="#347ca5" stroke-width="1.6" stroke-linecap="round"'),
        render_group("seasonalRivers", "seasonalRivers", seasonal,
                     'fill="none" stroke="#82b8e9" stroke-width="1.8" stroke-dasharray="7 6" stroke-linecap="round" stroke-linejoin="round"'),
        render_group("crossBorderRivers", "crossBorderRivers", cross_border,
                     'fill="none" stroke="#82b8e9" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"'),
    ]
    index_path = ROOT / "index.html"
    index = index_path.read_text(encoding="utf-8")
    index = upsert_group(index, index_groups[0], "rivers", None)
    index = upsert_group(index, index_groups[1], "canals", "rivers")
    index = upsert_group(index, index_groups[2], "canalTicks", "canals")
    index = upsert_group(index, index_groups[3], "seasonalRivers", "canalTicks")
    index = upsert_group(index, index_groups[4], "crossBorderRivers", "seasonalRivers")
    index_path.write_text(index, encoding="utf-8")

    preview_path = ROOT / "vector_preview.svg"
    preview = preview_path.read_text(encoding="utf-8")
    preview_groups = [
        render_preview_group("rivers", rivers,
                            'fill="none" stroke="#83b8e8" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"'),
        render_preview_group("canals", [f for f in canals if f["properties"].get("symbol_part") == "centerline"],
                            'fill="none" stroke="#347ca5" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"'),
        render_preview_group("canal-ticks", [f for f in canals if f["properties"].get("symbol_part") == "ruler_tick"],
                            'fill="none" stroke="#347ca5" stroke-width="1.6" stroke-linecap="round"'),
        render_preview_group("seasonal-rivers", seasonal,
                            'fill="none" stroke="#82b8e9" stroke-width="1.8" stroke-dasharray="7 6" stroke-linecap="round" stroke-linejoin="round"'),
        render_preview_group("cross-border-rivers", cross_border,
                            'fill="none" stroke="#82b8e9" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"'),
    ]
    preview = upsert_group(preview, preview_groups[0], "rivers", None)
    preview = upsert_group(preview, preview_groups[1], "canals", "rivers")
    preview = upsert_group(preview, preview_groups[2], "canal-ticks", "canals")
    preview = upsert_group(preview, preview_groups[3], "seasonal-rivers", "canal-ticks")
    preview = upsert_group(preview, preview_groups[4], "cross-border-rivers", "seasonal-rivers")
    preview_path.write_text(preview, encoding="utf-8")


def write_overlay(seasonal: list[dict], canals: list[dict]) -> None:
    image = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(image)
    for feature in seasonal:
        coords = [tuple(point) for point in feature["geometry"]["coordinates"]]
        draw.line(coords, fill=(220, 40, 65), width=2)
    for feature in canals:
        coords = [tuple(point) for point in feature["geometry"]["coordinates"]]
        width = 3 if feature["properties"].get("symbol_part") == "centerline" else 2
        draw.line(coords, fill=(28, 95, 140), width=width)
    DIAGNOSTICS.mkdir(exist_ok=True)
    image.save(DIAGNOSTICS / "waterway_styles_overlay.png")


def main() -> None:
    river_fc = read_fc(DATA / "hydrology_lines.geojson")
    river_features = remove_canal_from_rivers(river_fc["features"])
    canals = canal_features()
    seasonal = seasonal_features()
    cross_border_fc = read_fc(DATA / "cross_border_rivers.geojson")
    cross_border = remove_seasonal_from_cross_border(cross_border_fc["features"], seasonal)

    write_fc(DATA / "hydrology_lines.geojson", "hydrology_rivers_without_canal", river_features)
    write_fc(DATA / "canals.geojson", "canal_centerline_and_ruler_ticks", canals)
    write_fc(DATA / "seasonal_rivers.geojson", "neighbor_seasonal_rivers", seasonal)
    write_fc(DATA / "cross_border_rivers.geojson", "cross_border_rivers_without_seasonal_rivers", cross_border)
    update_render_files(river_features, canals, seasonal, cross_border)
    write_overlay(seasonal, canals)

    manifest_path = ROOT / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "river_line_segments": len(river_features),
        "canal_centerlines": len(CANAL_CENTERLINES),
        "canal_ruler_ticks": sum(f["properties"].get("symbol_part") == "ruler_tick" for f in canals),
        "seasonal_river_features": len(seasonal),
        "cross_border_river_segments": len(cross_border),
        "national_seasonal_river_candidates": 0,
    })
    label_features = read_fc(DATA / "region_code_labels.geojson")["features"]
    manifest["region_code_label_instances"] = len(label_features)
    manifest["unique_region_codes"] = len({f["properties"].get("region_code") for f in label_features})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Kept {len(river_features)} regular river segments; separated {len(CANAL_CENTERLINES)} canals "
        f"with {manifest['canal_ruler_ticks']} ruler ticks; traced {len(seasonal)} "
        "neighbor seasonal rivers."
    )


if __name__ == "__main__":
    main()
