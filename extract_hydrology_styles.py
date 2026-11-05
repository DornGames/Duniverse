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
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOURCE = ROOT / "input_map.png"
HIGH_RES_SOURCE = ROOT / "Administrative-Division-Map-of-Donh-2026.png"
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

# Traced from the 8192x5792 source image, then scaled into input_map.png
# coordinates. Canal 1 bends around the lake outlet; canal 2 ends at its
# eastern river junction. Tick locations are points on the centerline.
CANAL_CENTERLINES = [
    {
        "canal_id": 1,
        "name": "南部运河",
        # From the reservoir outlet, the channel dips south, turns northeast,
        # then meets the eastern river network before the coastline.
        "line": LineString([
            (1277, 1145.0), (1281, 1145.5), (1285, 1145.8),
            (1289, 1146.2), (1293, 1146.7), (1297, 1147.1),
            (1301, 1147.6), (1305, 1148.0), (1310, 1148.4),
            (1315, 1148.7), (1320, 1148.7), (1325, 1148.7),
            (1330, 1148.6), (1335, 1148.2), (1340, 1147.7),
            (1344, 1147.0), (1348, 1145.8), (1352, 1144.3),
            (1356, 1142.5), (1360, 1140.7), (1364, 1138.5),
            (1368, 1136.2), (1372, 1134.2), (1376, 1132.5),
            (1380, 1131.2), (1385, 1130.3), (1390, 1129.8),
            (1395, 1129.7), (1400, 1129.0),
        ]),
        "tick_positions": [
            (1295, 1147.0), (1306, 1148.1), (1318, 1148.7),
            (1329, 1148.6), (1340, 1147.7), (1352, 1144.3),
            (1364, 1138.5), (1375, 1132.8), (1387, 1130.0),
            (1398, 1129.2),
        ],
        "tick_length_px": 5.5,
    },
    {
        "canal_id": 2,
        "name": "南部底部运河",
        # The long southern canal runs from the western water junction to the
        # eastern river junction; its marked reach is straight in the source.
        "line": LineString([
            (1027, 1373.5), (1030, 1375.0), (1034, 1375.8),
            (1038, 1376.0), (1310, 1376.0),
        ]),
        "tick_positions": [
            (1036, 1376), (1048, 1376), (1062, 1376), (1074, 1376),
            (1087, 1376), (1097, 1376), (1109, 1376), (1122, 1376),
            (1134, 1376), (1145, 1376), (1157, 1376), (1169, 1376),
            (1179, 1376), (1192, 1376), (1204, 1376), (1214, 1376),
            (1224, 1376), (1237, 1376), (1247, 1376), (1259, 1376),
            (1270, 1376), (1279, 1376), (1288, 1376), (1300, 1376),
            (1310, 1376),
        ],
        "tick_length_px": 4.5,
    },
    {
        "canal_id": 3,
        "name": "湖区南向运河",
        "source": "official_map/Administrative Division Map of Donh 2026-river-super.png",
        "review_status": "vectorized_and_manually_checked_against_official_12x_river_raster",
        "exclude_river_overlap": False,
        # Centerline skeletonized from the pure-river 12x reference. The short
        # ruler marks are stored separately below, so they do not distort the
        # centerline or get misclassified as river branches.
        "line": LineString([
            (1263.79, 767.96), (1261.71, 768.21), (1261.29, 768.62),
            (1261.04, 770.29), (1258.79, 774.88), (1258.54, 776.21),
            (1257.54, 776.88), (1253.96, 781.71), (1252.54, 783.79),
            (1252.12, 785.29), (1251.46, 785.62), (1250.29, 787.21),
            (1247.62, 789.79), (1245.04, 792.96), (1244.71, 794.12),
            (1243.79, 794.46), (1242.96, 795.29), (1240.38, 798.46),
            (1239.04, 800.79), (1236.79, 803.62), (1236.54, 804.71),
            (1235.62, 805.29), (1234.79, 806.62), (1232.88, 810.21),
            (1231.88, 813.04), (1230.71, 818.12), (1230.54, 822.54),
            (1231.04, 828.54), (1230.79, 836.54), (1231.79, 842.54),
            (1231.96, 846.71), (1232.21, 847.96), (1232.54, 848.21),
            (1232.71, 849.62), (1234.38, 853.46), (1237.96, 860.88),
            (1238.54, 861.29), (1238.71, 862.62), (1240.04, 866.54),
            (1240.54, 869.29), (1240.54, 872.21), (1239.46, 875.12),
            (1238.54, 881.62), (1238.54, 883.54), (1239.21, 888.04),
            (1239.21, 891.54), (1238.04, 895.29), (1237.12, 895.88),
            (1234.46, 899.88), (1231.96, 902.79), (1229.38, 905.21),
            (1229.21, 905.79), (1227.46, 906.88), (1223.71, 910.96),
            (1221.62, 914.71), (1220.71, 918.29), (1220.88, 923.88),
            (1221.96, 927.04), (1223.54, 928.79), (1226.38, 930.29),
            (1227.79, 931.54), (1228.62, 933.12), (1229.29, 933.62),
            (1229.54, 934.96), (1230.71, 936.88), (1237.62, 943.46),
            (1239.54, 946.62), (1240.21, 950.96), (1240.29, 955.71),
            (1240.12, 958.88), (1239.62, 960.46), (1239.04, 964.46),
            (1239.12, 967.71), (1239.62, 971.62), (1239.29, 978.62),
            (1240.12, 982.04), (1241.79, 986.46), (1242.46, 987.38),
            (1243.12, 991.71), (1243.88, 994.04),
        ]),
        # Individual ruler marks traced from their visible centerline anchors
        # and tips in the official raster.
        "ticks": [
            LineString([(1258.54, 776.21), (1262.21, 778.29)]),
            LineString([(1252.04, 785.38), (1255.62, 787.54)]),
            LineString([(1244.71, 794.12), (1249.04, 796.79)]),
            LineString([(1236.46, 804.79), (1241.21, 807.38)]),
            LineString([(1231.04, 817.46), (1235.79, 818.38)]),
            LineString([(1230.96, 832.71), (1234.71, 832.62)]),
            LineString([(1232.54, 848.21), (1236.46, 847.62)]),
            LineString([(1238.54, 861.29), (1242.71, 859.96)]),
            LineString([(1239.29, 877.29), (1244.71, 876.88)]),
            LineString([(1238.04, 895.29), (1241.71, 896.71)]),
            LineString([(1229.12, 905.88), (1232.62, 908.21)]),
            LineString([(1220.62, 920.54), (1225.71, 920.54)]),
            LineString([(1229.21, 933.54), (1233.29, 930.96)]),
            LineString([(1239.54, 946.62), (1244.71, 945.54)]),
            LineString([(1240.04, 959.21), (1243.71, 959.71)]),
            LineString([(1239.62, 973.38), (1243.04, 973.54)]),
            LineString([(1242.38, 987.12), (1248.12, 986.62)]),
        ],
    },
    {
        "canal_id": 4,
        "name": "中北部运河段",
        "source": "official_map/Administrative Division Map of Donh 2026-river-super.png",
        "review_status": "manually_checked_against_official_12x_river_raster_and_latest_overlay",
        "subtract_from_rivers": True,
        # Reclassified from the ordinary river vectors so this ruler-marked
        # reach can be toggled independently. The trace follows the official
        # 12x raster from its first ruler mark to the eastern junction.
        "line": LineString([
            (935.21, 917.79), (935.38, 918.04), (935.88, 918.21),
            (936.88, 918.38), (938.21, 918.79), (938.96, 919.12),
            (940.79, 920.29), (942.12, 920.71), (943.12, 920.71),
            (944.71, 920.38), (946.88, 919.62), (948.62, 919.29),
            (948.79, 919.04), (950.54, 918.96), (953.46, 918.38),
            (955.21, 917.79), (958.54, 916.21), (958.88, 915.62),
            (960.38, 915.46), (961.29, 915.12), (962.21, 914.96),
            (963.04, 915.04), (963.62, 915.46), (966.12, 920.04),
            (966.46, 920.38), (967.21, 921.62), (967.71, 922.12),
            (968.46, 922.38), (968.71, 923.46), (970.54, 925.29),
            (972.54, 926.71), (977.38, 928.79), (977.54, 928.96),
            (979.21, 929.21), (979.54, 929.71), (981.88, 930.71),
            (982.29, 931.04), (985.04, 932.38), (988.04, 933.38),
            (989.04, 933.46), (989.12, 933.62), (991.04, 933.96),
            (991.38, 934.12), (994.46, 934.71), (995.21, 934.71),
            (995.46, 934.88), (996.38, 934.88), (996.46, 935.04),
            (998.29, 935.21), (998.79, 935.38), (1001.04, 935.54),
            (1001.54, 935.71), (1003.62, 935.79),
        ]),
        # Seven ruler marks are visible along this reach. Four had already
        # been digitized as river branches; the three at x≈935, 949 and 959
        # were visibly absent from the overlay.
        "ticks": [
            LineString([(935.21, 917.79), (935.46, 914.04)]),
            LineString([(948.79, 919.12), (947.46, 915.21)]),
            LineString([(958.88, 915.62), (957.29, 911.96)]),
            LineString([(968.46, 922.38), (971.38, 919.29)]),
            LineString([(979.21, 929.21), (981.12, 924.38)]),
            LineString([(989.04, 933.46), (990.62, 928.29)]),
            LineString([(1003.62, 935.79), (1003.54, 930.96)]),
        ],
    },
]
CANAL_RIVER_EXCLUSION_BUFFER_PX = 3.5
SEASONAL_CROSS_BORDER_EXCLUSION_BUFFER_PX = 2.5
# These short segments were manually checked against the official 12x raster
# and are duplicate fragments of the two dashed seasonal traces, not separate
# neighboring rivers. Keep the nearby real confluences (90, 93) and lake mouth
# (119) intact.
SEASONAL_DASH_RESIDUAL_SEGMENT_IDS = frozenset({
    *range(43, 54), 60, 61, 64, 72, 75, 88, 89,
    97, *range(111, 119),
})


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
        newline="\n",
    )


def line_parts(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in line_parts(item)]
    return []


def round_line(line: LineString, digits: int = 2) -> dict:
    return {
        "type": "LineString",
        "coordinates": [[round(x, digits), round(y, digits)] for x, y in line.coords],
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


def perpendicular_tick(centerline: LineString, location: tuple[float, float],
                       length: float) -> LineString:
    distance = centerline.project(Point(location))
    anchor = centerline.interpolate(distance)
    tangent_span = 3.0
    before = centerline.interpolate(max(0, distance - tangent_span))
    after = centerline.interpolate(min(centerline.length, distance + tangent_span))
    dx, dy = after.x - before.x, after.y - before.y
    magnitude = math.hypot(dx, dy) or 1.0
    nx, ny = dy / magnitude, -dx / magnitude
    if ny > 0:
        nx, ny = -nx, -ny
    return LineString([
        (anchor.x, anchor.y),
        (anchor.x + nx * length, anchor.y + ny * length),
    ])


def canal_features() -> list[dict]:
    features = []
    for canal in CANAL_CENTERLINES:
        centerline = canal["line"]
        common = {
            "canal_id": canal["canal_id"],
            "name": canal["name"],
            "source": canal.get("source", HIGH_RES_SOURCE.name),
            "separate_from_rivers": bool(canal.get("subtract_from_rivers", False)),
        }
        features.append({
            "type": "Feature",
            "properties": {
                **common,
                "symbol_part": "centerline",
                "review_status": canal.get(
                    "review_status", "manually_traced_from_high_resolution_source"
                ),
                "length_px": round(centerline.length, 1),
            },
            "geometry": round_line(centerline),
        })
        if "ticks" in canal:
            ticks = canal["ticks"]
        else:
            ticks = [
                perpendicular_tick(centerline, location, canal["tick_length_px"])
                for location in canal["tick_positions"]
            ]
        for tick_id, tick in enumerate(ticks, start=1):
            features.append({
                "type": "Feature",
                "properties": {
                    **common,
                    "tick_id": tick_id,
                    "symbol_part": "ruler_tick",
                    "orientation": "perpendicular_to_local_tangent",
                },
                "geometry": round_line(tick),
            })
    return features


def remove_canal_from_rivers(features: list[dict]) -> list[dict]:
    corridor = unary_union([
        canal["line"].buffer(CANAL_RIVER_EXCLUSION_BUFFER_PX, cap_style="flat")
        for canal in CANAL_CENTERLINES
        if canal.get("exclude_river_overlap", True)
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


def remove_separated_canal_reaches(features: list[dict], canals: list[dict]) -> list[dict]:
    """Keep verified, independently toggled canal geometry out of rivers."""
    corridors = []
    for canal in canals:
        properties = canal["properties"]
        if not properties.get("separate_from_rivers"):
            continue
        geometry = shape(canal["geometry"])
        if properties.get("symbol_part") == "centerline":
            corridors.append(geometry.buffer(0.9, cap_style="flat"))
        elif properties.get("symbol_part") == "ruler_tick":
            corridors.append(geometry.buffer(0.8, cap_style="flat"))
    if not corridors:
        return features
    corridor = unary_union(corridors)
    next_segment_id = max(
        (feature["properties"].get("segment_id", 0) for feature in features),
        default=0,
    ) + 1
    result = []
    for feature in features:
        source_geometry = shape(feature["geometry"])
        if not source_geometry.intersects(corridor):
            result.append(feature)
            continue
        geometry = source_geometry.difference(corridor)
        if source_geometry.equals(geometry):
            result.append(feature)
            continue
        for part in line_parts(geometry):
            if part.length < 2:
                continue
            properties = {
                **feature["properties"],
                "length_px": round(part.length, 3),
            }
            if result and properties.get("segment_id") in {
                prior["properties"].get("segment_id") for prior in result
            }:
                properties["segment_id"] = next_segment_id
                next_segment_id += 1
            result.append({
                "type": "Feature",
                "properties": properties,
                "geometry": round_line(part, digits=3),
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


def remove_verified_seasonal_dash_residuals(features: list[dict]) -> list[dict]:
    return [
        feature for feature in features
        if feature["properties"].get("segment_id") not in SEASONAL_DASH_RESIDUAL_SEGMENT_IDS
    ]


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
        radius = 0.35
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


def polygon_path_data(geometry: dict) -> str:
    geom = shape(geometry)
    polygons = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    commands = []
    for polygon in polygons:
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = list(ring.coords)
            if len(coordinates) < 4:
                continue
            if coordinates[0] == coordinates[-1]:
                coordinates = coordinates[:-1]
            point_text = lambda point: f"{point[0]:.2f},{point[1]:.2f}"
            commands.append(
                "M " + point_text(coordinates[0]) + " L "
                + " L ".join(point_text(point) for point in coordinates[1:]) + " Z"
            )
    return " ".join(commands)


def render_lakes_group(features: list[dict], *, preview: bool = False) -> str:
    paths = []
    for feature in features:
        path = polygon_path_data(feature["geometry"])
        if path:
            paths.append(f'<path d="{path}"/>')
    style = 'fill="#b1d0ed" fill-rule="evenodd" stroke="#83b8e8" stroke-width="1.1"'
    if preview:
        style = 'fill="#b1d0ed" fill-rule="evenodd" stroke="#83b8e8" stroke-width="1.2"'
    return f'<g id="lakes" {style} pointer-events="none">\n' + "\n".join(paths) + "\n</g>"


def update_hydrology_lod_precision(index: str) -> str:
    hydrology_lod = "const HYDROLOGY_LOD_TOLERANCES=[.06,.12,.24,.48,.96];"
    lod_pattern = r"const HYDROLOGY_LOD_TOLERANCES=\[[^\]]+\];"
    if "const HYDROLOGY_LAYER_IDS=" in index:
        index, count = re.subn(lod_pattern, hydrology_lod, index, count=1)
        if count != 1:
            raise RuntimeError("Could not find the existing hydrology LOD precision marker")
    else:
        marker = "const LOD_TOLERANCES=[.12,.25,.5,1,2,4,8];"
        if index.count(marker) != 1:
            raise RuntimeError("Could not locate the base LOD precision marker")
        index = index.replace(
            marker,
            marker + "\n"
            "const HYDROLOGY_LAYER_IDS=new Set(['lakes','rivers','canals','canalTicks','seasonalRivers','crossBorderRivers']);\n"
            + hydrology_lod,
            1,
        )
    legacy_radius = "['rivers','canals','canalTicks','seasonalRivers','crossBorderRivers'].includes(layerId)?2:0;"
    legacy_radius_updated = "HYDROLOGY_LAYER_IDS.has(layerId)?0.35:0;"
    old_hydrology_radius = "HYDROLOGY_LAYER_IDS.has(layerId)?0.75:0;"
    if legacy_radius in index:
        index = index.replace(legacy_radius, legacy_radius_updated, 1)
    elif old_hydrology_radius in index:
        index = index.replace(old_hydrology_radius, legacy_radius_updated, 1)
    elif legacy_radius_updated not in index:
        raise RuntimeError("Could not locate the hydrology curve-smoothing marker")
    replacements = [
        (
            "const tolerances=['oceanFill','islandFills'].includes(layerId)?COASTAL_FILL_LOD_TOLERANCES:LOD_TOLERANCES;",
            "const tolerances=['oceanFill','islandFills'].includes(layerId)?COASTAL_FILL_LOD_TOLERANCES:\n"
            "      HYDROLOGY_LAYER_IDS.has(layerId)?HYDROLOGY_LOD_TOLERANCES:LOD_TOLERANCES;",
        ),
        (
            "const path=record.variants[toleranceIndex];",
            "const path=record.variants[Math.min(toleranceIndex,record.variants.length-1)];",
        ),
        (
            "const pathToleranceIndex=isCoastalFill?coastalFillToleranceIndex:toleranceIndex;\n"
            "    const pathTolerance=isCoastalFill\n"
            "      ?COASTAL_FILL_LOD_TOLERANCES[pathToleranceIndex]\n"
            "      :LOD_TOLERANCES[pathToleranceIndex];",
            "const isHydrology=HYDROLOGY_LAYER_IDS.has(record.layerId);\n"
            "    const pathToleranceIndex=isCoastalFill?coastalFillToleranceIndex:\n"
            "      isHydrology?Math.min(toleranceIndex,HYDROLOGY_LOD_TOLERANCES.length-1):toleranceIndex;\n"
            "    const pathTolerance=isCoastalFill?COASTAL_FILL_LOD_TOLERANCES[pathToleranceIndex]:\n"
            "      isHydrology?HYDROLOGY_LOD_TOLERANCES[pathToleranceIndex]:LOD_TOLERANCES[pathToleranceIndex];",
        ),
    ]
    for old, new in replacements:
        if new in index:
            continue
        count = index.count(old)
        if count != 1:
            raise RuntimeError(f"Expected one map-render precision marker, found {count}: {old[:72]}")
        index = index.replace(old, new, 1)
    return index


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
    lakes = read_fc(DATA / "lakes.geojson")["features"]
    index_groups = [
        render_group("rivers", "rivers", rivers,
                     'fill="none" stroke="#83b8e8" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"'),
        render_group("canals", "canals", [f for f in canals if f["properties"].get("symbol_part") == "centerline"],
                     'fill="none" stroke="#347ca5" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'),
        render_group("canalTicks", "canalTicks", [f for f in canals if f["properties"].get("symbol_part") == "ruler_tick"],
                     'fill="none" stroke="#347ca5" stroke-width="1.5" stroke-linecap="round"'),
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
    index = upsert_group(index, render_lakes_group(lakes), "lakes", None)
    index = update_hydrology_lod_precision(index)
    index_path.write_text(index, encoding="utf-8", newline="\n")

    preview_path = ROOT / "vector_preview.svg"
    preview = preview_path.read_text(encoding="utf-8")
    preview_groups = [
        render_preview_group("rivers", rivers,
                            'fill="none" stroke="#83b8e8" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"'),
        render_preview_group("canals", [f for f in canals if f["properties"].get("symbol_part") == "centerline"],
                            'fill="none" stroke="#347ca5" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'),
        render_preview_group("canal-ticks", [f for f in canals if f["properties"].get("symbol_part") == "ruler_tick"],
                            'fill="none" stroke="#347ca5" stroke-width="1.5" stroke-linecap="round"'),
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
    preview = upsert_group(preview, render_lakes_group(lakes, preview=True), "lakes", None)
    preview_path.write_text(preview, encoding="utf-8", newline="\n")


def write_overlay(seasonal: list[dict], canals: list[dict]) -> None:
    image = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(image)
    for feature in seasonal:
        coords = [tuple(point) for point in feature["geometry"]["coordinates"]]
        draw.line(coords, fill=(220, 40, 65), width=2)
    for feature in canals:
        coords = [tuple(point) for point in feature["geometry"]["coordinates"]]
        width = 2
        draw.line(coords, fill=(28, 95, 140), width=width)
    DIAGNOSTICS.mkdir(exist_ok=True)
    image.save(DIAGNOSTICS / "waterway_styles_overlay.png")


def main() -> None:
    with Image.open(SOURCE) as source_image, Image.open(HIGH_RES_SOURCE) as high_res:
        if high_res.width != source_image.width * 4 or high_res.height != source_image.height * 4:
            raise ValueError("The high-resolution trace source must be exactly 4x input_map.png")
    river_source_fc = read_fc(DATA / "hydrology_lines.geojson")
    # Subtract only reaches marked for the independent canal layer. The
    # geometric difference is idempotent, so rerunning this generator keeps
    # the river/canal separation stable.
    canals = canal_features()
    canal_additions_fc = read_fc(DATA / "canal_additions.geojson")
    canals.extend(canal_additions_fc["features"])
    river_features = remove_separated_canal_reaches(river_source_fc["features"], canals)
    seasonal = seasonal_features()
    cross_border_fc = read_fc(DATA / "cross_border_rivers.geojson")
    # Remove only short fragments manually matched to the dashed seasonal
    # traces on the official raster. True river junctions and the lake mouth
    # remain in the cross-border layer.
    original_cross_border_count = len(cross_border_fc["features"])
    cross_border = remove_verified_seasonal_dash_residuals(cross_border_fc["features"])
    cross_border_fc["features"] = cross_border

    write_fc(DATA / "canals.geojson", "canal_centerline_and_ruler_ticks", canals)
    write_fc(DATA / "rivers.geojson", "rivers_excluding_independent_canals", river_features)
    write_fc(DATA / "seasonal_rivers.geojson", "neighbor_seasonal_rivers", seasonal)
    (DATA / "cross_border_rivers_clean.geojson").write_text(
        json.dumps(cross_border_fc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    update_render_files(river_features, canals, seasonal, cross_border)
    write_overlay(seasonal, canals)

    manifest_path = ROOT / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "river_line_segments": len(river_features),
        "canal_centerlines": len({f["properties"]["canal_id"] for f in canals}),
        "canal_centerline_reaches": sum(
            f["properties"].get("symbol_part") == "centerline" for f in canals
        ),
        "canal_trace_source_image": HIGH_RES_SOURCE.name,
        "canal_trace_source_images": sorted({
            f["properties"].get("source", HIGH_RES_SOURCE.name)
            for f in canals
        }),
        "canal_tick_orientation": "perpendicular_to_local_centerline_tangent",
        "canal_ruler_ticks": sum(f["properties"].get("symbol_part") == "ruler_tick" for f in canals),
        "seasonal_river_features": len(seasonal),
        "cross_border_river_segments": len(cross_border),
        "seasonal_dash_residual_segments_removed": original_cross_border_count - len(cross_border),
        "national_seasonal_river_candidates": 0,
    })
    label_features = read_fc(DATA / "region_code_labels.geojson")["features"]
    manifest["region_code_label_instances"] = len(label_features)
    manifest["unique_region_codes"] = len({f["properties"].get("region_code") for f in label_features})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"Kept {len(river_features)} regular river segments; separated {manifest['canal_centerlines']} canals "
        f"across {manifest['canal_centerline_reaches']} reaches with {manifest['canal_ruler_ticks']} ruler ticks; traced {len(seasonal)} "
        "neighbor seasonal rivers."
    )


if __name__ == "__main__":
    main()
