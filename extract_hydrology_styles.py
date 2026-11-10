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
import base64
import statistics
import sys
from array import array
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOURCE = ROOT / "input_map.png"
HIGH_RES_SOURCE = ROOT / "Administrative-Division-Map-of-Donh-2026.png"
DIAGNOSTICS = ROOT / "diagnostics"

SEASONAL_RIVER_PATHS = [
    # Centerline traced through the dash-stroke centers on the official 12x
    # river raster, then mapped into input_map.png pixels using the established
    # high-resolution pixel transform. This replaces the visibly offset path.
    [
        (129.82, 336.55), (137.40, 334.56), (148.38, 333.88),
        (159.70, 335.05), (171.08, 340.03), (180.88, 347.83),
        (189.43, 354.05), (198.55, 356.05), (209.23, 355.61),
        (221.68, 355.55), (232.13, 356.62), (243.42, 360.98),
        (252.46, 366.10), (259.40, 371.34), (263.84, 378.99),
        (268.45, 385.11), (276.27, 385.20), (283.25, 381.85),
        (288.26, 376.98), (293.01, 369.29), (296.77, 360.46),
        (299.87, 354.05), (306.70, 347.18), (315.60, 343.47),
        (323.84, 342.30), (333.75, 341.47), (344.97, 342.82),
        (352.27, 345.81), (359.28, 353.34), (367.72, 356.00),
        (374.10, 351.08), (379.84, 341.49), (386.74, 332.81),
        (395.63, 326.60), (405.24, 324.83), (415.19, 322.43),
        (431.68, 326.75), (438.83, 335.84), (450.01, 344.30),
        (462.24, 350.42), (474.05, 355.96), (485.98, 360.56),
        (498.69, 365.33), (511.71, 370.00), (525.59, 374.91),
        (536.76, 383.04), (543.80, 394.98), (543.37, 408.61),
        (538.33, 421.23), (526.98, 433.32), (520.37, 445.27),
        (520.85, 458.42), (523.65, 469.72), (531.23, 482.76),
        (542.99, 494.31), (553.21, 502.95), (564.01, 509.62),
        (575.11, 515.81), (579.73, 524.37), (590.07, 533.77),
    ],
    [
        (223.07, 812.48), (214.40, 810.16), (206.31, 809.15),
        (197.03, 810.79), (189.16, 814.87), (184.53, 822.11),
        (186.63, 830.67), (190.12, 840.10), (190.02, 850.02),
        (186.79, 859.14), (182.92, 867.97), (179.74, 876.96),
        (179.73, 885.72), (183.01, 893.96), (190.02, 899.09),
        (200.39, 902.61), (209.13, 909.24), (208.59, 916.60),
        (200.94, 923.83), (195.13, 933.46), (190.82, 942.71),
        (187.00, 954.76), (186.01, 967.71), (187.92, 981.08),
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
            (1282.581, 1145.619), (1285, 1145.8),
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
        "subtract_from_rivers": True,
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
        "subtract_from_rivers": True,
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
# These raster fragments were manually checked against the official 12x source.
# They duplicate strokes on the two seasonal routes. ID 76 is included because
# the high-resolution overlay confirms it lies on seasonal trace 1; keep IDs
# 90/93 as real headwater junctions. Cross-border river geometry is clipped to
# verified neighboring lake fills below, so lake inlets/outlets remain at the
# shoreline without drawing broken centerlines across the lake surface.
SEASONAL_DASH_RESIDUAL_SEGMENT_IDS = frozenset({
    *range(43, 90), *range(94, 117),
})
CROSS_BORDER_LAKE_EDGE_RESIDUAL_SEGMENT_IDS = frozenset({118})
MIN_RIVER_CONTRIBUTING_REACH_PX = 2.5


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


def smooth_river_centerline(line: LineString) -> LineString:
    """Remove raster-trace jitter while preserving the reach's broad bends."""
    line = line.simplify(0.45, preserve_topology=True)
    coords = [(float(x), float(y)) for x, y, *rest in line.coords]
    if len(coords) < 3:
        return line
    smoothed = coords
    for _ in range(2):
        refined = [smoothed[0]]
        for start, end in zip(smoothed, smoothed[1:]):
            refined.append((0.75 * start[0] + 0.25 * end[0], 0.75 * start[1] + 0.25 * end[1]))
            refined.append((0.25 * start[0] + 0.75 * end[0], 0.25 * start[1] + 0.75 * end[1]))
        refined.append(smoothed[-1])
        smoothed = refined
    return LineString(smoothed)


def round_line(line: LineString, digits: int = 2) -> dict:
    return {
        "type": "LineString",
        "coordinates": [[round(x, digits), round(y, digits)] for x, y in line.coords],
    }


def q2_river_width_config() -> dict:
    config = read_fc(DATA / "river_widths_q2.json")
    outline_features = read_fc(DATA / "national_outline.geojson")["features"]
    outline = unary_union([shape(feature["geometry"]) for feature in outline_features])
    min_x, min_y, max_x, max_y = outline.bounds
    expected_extent = (min_x, min_y, (min_x + max_x) / 2, (min_y + max_y) / 2)
    actual_extent = tuple(float(value) for value in config["extent_px"])
    if len(actual_extent) != 4 or any(
        abs(actual - expected) > 0.1
        for actual, expected in zip(actual_extent, expected_extent)
    ):
        raise ValueError(
            "Q2 river styling extent no longer matches the upper-left quarter "
            f"of the national outline: configured={actual_extent}, expected={expected_extent}"
        )
    return config


def width_profiles(config: dict) -> dict[int, dict]:
    profiles = {}
    for segment_id, raw_profile in config["reach_profiles_px"].items():
        profile = {
            **raw_profile,
            "start_width_px": float(raw_profile["start_width_px"]),
            "end_width_px": float(raw_profile["end_width_px"]),
            "network_id": str(raw_profile["network_id"]),
            "clip_to_quadrant": bool(raw_profile.get("clip_to_quadrant", True)),
        }
        knots = raw_profile.get("width_knots_px") or [
            [0.0, profile["start_width_px"]],
            [1.0, profile["end_width_px"]],
        ]
        profile["width_knots_px"] = [
            (float(fraction), float(width)) for fraction, width in knots
        ]
        if (
            len(profile["width_knots_px"]) < 2
            or abs(profile["width_knots_px"][0][0]) > 1e-6
            or abs(profile["width_knots_px"][-1][0] - 1.0) > 1e-6
            or any(
                right[0] <= left[0]
                for left, right in zip(profile["width_knots_px"], profile["width_knots_px"][1:])
            )
        ):
            raise ValueError(f"River width knots for segment {segment_id} must increase from 0 to 1")
        profiles[int(segment_id)] = profile
    return profiles


def profile_width_at(profile: dict, fraction: float) -> float:
    fraction = max(0.0, min(1.0, fraction))
    knots = profile["width_knots_px"]
    for (left_fraction, left_width), (right_fraction, right_width) in zip(knots, knots[1:]):
        if fraction <= right_fraction:
            local = (fraction - left_fraction) / (right_fraction - left_fraction)
            return left_width + (right_width - left_width) * local
    return knots[-1][1]


def offset_centerline(line: LineString, definition: dict) -> LineString:
    """Apply a smooth, bounded lateral meander while keeping both endpoints fixed."""
    start_fraction = float(definition["start_fraction"])
    end_fraction = float(definition["end_fraction"])
    amplitude = float(definition["amplitude_px"])
    if not (0.0 <= start_fraction < end_fraction <= 1.0) or amplitude < 0:
        raise ValueError(f"Invalid centerline adjustment {definition.get('id')}")
    if line.length <= 0 or amplitude == 0:
        return line

    sample_count = max(2, math.ceil(line.length / 1.25), len(line.coords) - 1)
    tangent_span = min(2.5, max(0.75, line.length / sample_count))
    adjusted = []
    for index in range(sample_count + 1):
        fraction = index / sample_count
        distance = line.length * fraction
        before = line.interpolate(max(0.0, distance - tangent_span))
        center = line.interpolate(distance)
        after = line.interpolate(min(line.length, distance + tangent_span))
        tangent_x, tangent_y = after.x - before.x, after.y - before.y
        tangent_length = math.hypot(tangent_x, tangent_y) or 1.0
        normal_x, normal_y = -tangent_y / tangent_length, tangent_x / tangent_length

        offset = 0.0
        if start_fraction < fraction < end_fraction:
            local = (fraction - start_fraction) / (end_fraction - start_fraction)
            envelope = math.sin(math.pi * local) ** 2
            # A dominant sine pair gives an S bend; the smaller third wave
            # makes its lobes unequal, as in natural meanders.
            wave = math.sin(2 * math.pi * local) + 0.12 * math.sin(3 * math.pi * local)
            offset = amplitude * 1.5 * envelope * wave
        adjusted.append((center.x + normal_x * offset, center.y + normal_y * offset))
    adjusted[0] = (float(line.coords[0][0]), float(line.coords[0][1]))
    adjusted[-1] = (float(line.coords[-1][0]), float(line.coords[-1][1]))
    return LineString(adjusted)


def apply_centerline_adjustments(features: list[dict], config: dict) -> list[dict]:
    """Apply documented local bends to selected reaches before deriving channels."""
    definitions_by_segment: dict[int, list[dict]] = {}
    for definition in config.get("centerline_adjustments", []):
        definitions_by_segment.setdefault(int(definition["segment_id"]), []).append(definition)
    if not definitions_by_segment:
        return features

    grouped_indices: dict[int, list[int]] = {}
    for index, feature in enumerate(features):
        segment_id = int(feature["properties"].get("segment_id", 0))
        if segment_id in definitions_by_segment:
            grouped_indices.setdefault(segment_id, []).append(index)
    missing = sorted(set(definitions_by_segment) - set(grouped_indices))
    duplicated = sorted(segment_id for segment_id, indices in grouped_indices.items() if len(indices) != 1)
    if missing or duplicated:
        raise ValueError(f"Centerline adjustments have missing or split segments: missing={missing}, split={duplicated}")

    adjusted_features = list(features)
    for segment_id, definitions in definitions_by_segment.items():
        index = grouped_indices[segment_id][0]
        feature = adjusted_features[index]
        lines = line_parts(shape(feature["geometry"]))
        if len(lines) != 1:
            raise ValueError(f"Centerline adjustment requires one line for segment {segment_id}")
        line = lines[0]
        for definition in definitions:
            line = offset_centerline(line, definition)
        properties = {
            **feature["properties"],
            "cartographic_adjustment_ids": [item["id"] for item in definitions],
        }
        adjusted_features[index] = {
            **feature,
            "properties": properties,
            "geometry": mapping(line),
        }
    return adjusted_features


def reach_endpoint(features_by_id: dict[int, dict], endpoint: dict) -> tuple[Point, float]:
    segment_id = int(endpoint["segment_id"])
    side = str(endpoint["endpoint"])
    if side not in {"start", "end"}:
        raise ValueError(f"Unknown endpoint {side!r} on segment {segment_id}")
    feature = features_by_id.get(segment_id)
    if feature is None:
        raise ValueError(f"Width topology references missing segment {segment_id}")
    lines = line_parts(shape(feature["geometry"]))
    if len(lines) != 1:
        raise ValueError(f"Width topology requires one line for segment {segment_id}")
    line = lines[0]
    fraction = 0.0 if side == "start" else 1.0
    coordinate = line.coords[0 if side == "start" else -1]
    return Point(coordinate), fraction


def validate_channel_topology(features: list[dict], config: dict) -> list[dict]:
    """Reject Q2 profiles where confluences pinch or continuous reaches jump."""
    profiles = width_profiles(config)
    features_by_id = {
        int(feature["properties"]["segment_id"]): feature
        for feature in features
        if "segment_id" in feature["properties"]
    }
    checks = []
    for definition in config.get("confluence_width_checks", []):
        tributaries = definition["upstream_reaches"]
        downstream = definition["downstream_reach"]
        points_and_widths = []
        for endpoint in [*tributaries, downstream]:
            point, fraction = reach_endpoint(features_by_id, endpoint)
            profile = profiles.get(int(endpoint["segment_id"]))
            if profile is None:
                raise ValueError(f"Confluence check {definition['id']} has no width profile for {endpoint['segment_id']}")
            points_and_widths.append((point, profile_width_at(profile, fraction)))

        max_gap = max(
            left[0].distance(right[0])
            for index, left in enumerate(points_and_widths)
            for right in points_and_widths[index + 1:]
        )
        allowed_gap = float(definition.get("max_node_gap_px", 0.1))
        if max_gap > allowed_gap:
            raise ValueError(f"Confluence {definition['id']} endpoints do not meet: gap={max_gap:.3f}px")

        combined_upstream_width = math.sqrt(sum(width ** 2 for _, width in points_and_widths[:-1]))
        downstream_width = points_and_widths[-1][1]
        required_width = combined_upstream_width * float(definition.get("minimum_ratio", 1.0))
        if downstream_width + 1e-6 < required_width:
            raise ValueError(
                f"Confluence {definition['id']} pinches downstream: "
                f"{downstream_width:.3f}px < {required_width:.3f}px"
            )
        checks.append({
            "id": definition["id"],
            "upstream_widths_px": [round(width, 3) for _, width in points_and_widths[:-1]],
            "combined_upstream_width_px": round(combined_upstream_width, 3),
            "downstream_width_px": round(downstream_width, 3),
            "node_gap_px": round(max_gap, 4),
        })

    for definition in config.get("channel_continuity_checks", []):
        endpoints = [definition["first_reach"], definition["second_reach"]]
        values = [
            reach_endpoint(features_by_id, endpoint)
            for endpoint in endpoints
        ]
        widths = [
            profile_width_at(profiles[int(endpoint["segment_id"])], fraction)
            for endpoint, (_, fraction) in zip(endpoints, values)
        ]
        gap = values[0][0].distance(values[1][0])
        if gap > float(definition.get("max_node_gap_px", 0.1)):
            raise ValueError(f"Continuity {definition['id']} endpoints do not meet: gap={gap:.3f}px")
        width_difference = abs(widths[0] - widths[1])
        if width_difference > float(definition.get("max_width_difference_px", 0.05)):
            raise ValueError(
                f"Continuity {definition['id']} has a width jump: "
                f"{width_difference:.3f}px"
            )
        checks.append({
            "id": definition["id"],
            "continuous_widths_px": [round(width, 3) for width in widths],
            "node_gap_px": round(gap, 4),
        })
    return checks


def relief_elevation_grid() -> dict:
    """Read the map's stored 16-bit terrain model for outlet and plain checks."""
    source = (DATA / "relief_elevation.js").read_text(encoding="utf-8")
    payload = json.loads(source[source.index("{"):source.rindex("}") + 1])
    values = array("H")
    values.frombytes(base64.b64decode(payload["valuesB64"]))
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != int(payload["width"]) * int(payload["height"]):
        raise ValueError("The terrain height grid does not match its stored dimensions")
    return {
        "width": int(payload["width"]),
        "height": int(payload["height"]),
        "minimum_m": float(payload["min"]),
        "maximum_m": float(payload["max"]),
        "values": values,
    }


def terrain_height_at(grid: dict, x: float, y: float) -> float:
    width, height = grid["width"], grid["height"]
    ix = max(0, min(width - 1, round(x)))
    iy = max(0, min(height - 1, round(y)))
    encoded = grid["values"][iy * width + ix]
    return grid["minimum_m"] + encoded / 65535 * (grid["maximum_m"] - grid["minimum_m"])


def river_network_forest(
    rivers: list[dict],
    country_outline,
    lakes: list[dict],
    grid: dict,
) -> list[dict]:
    """Build directed tree reaches by rooting each mapped network at its low outlet."""
    adjacency: dict[tuple[float, float], list[tuple[int, tuple[float, float]]]] = {}
    node_coordinates: dict[tuple[float, float], tuple[float, float]] = {}
    line_by_id: dict[int, LineString] = {}
    edge_nodes: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    for feature in rivers:
        segment_id = int(feature["properties"]["segment_id"])
        lines = line_parts(shape(feature["geometry"]))
        if len(lines) != 1:
            raise ValueError(f"National width modelling requires one line per reach: {segment_id}")
        if segment_id in line_by_id:
            raise ValueError(f"National width modelling found duplicate segment id {segment_id}")
        line = lines[0]
        start = (float(line.coords[0][0]), float(line.coords[0][1]))
        end = (float(line.coords[-1][0]), float(line.coords[-1][1]))
        start_node = (round(start[0], 1), round(start[1], 1))
        end_node = (round(end[0], 1), round(end[1], 1))
        line_by_id[segment_id] = line
        edge_nodes[segment_id] = (start_node, end_node)
        node_coordinates.setdefault(start_node, start)
        node_coordinates.setdefault(end_node, end)
        adjacency.setdefault(start_node, []).append((segment_id, end_node))
        adjacency.setdefault(end_node, []).append((segment_id, start_node))

    lake_shapes = [shape(feature["geometry"]) for feature in lakes]

    def sink_proximity(node: tuple[float, float]) -> float:
        point = Point(node_coordinates[node])
        distances = [point.distance(country_outline.boundary)]
        distances.extend(item.boundary.distance(point) for item in lake_shapes)
        return min(distances)

    components = []
    visited_nodes = set()
    for initial_node in adjacency:
        if initial_node in visited_nodes:
            continue
        stack = [initial_node]
        nodes = set()
        segment_ids = set()
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            visited_nodes.add(node)
            for segment_id, neighbour in adjacency[node]:
                segment_ids.add(segment_id)
                stack.append(neighbour)

        leaves = [node for node in nodes if len(adjacency[node]) == 1]
        if not leaves:
            raise ValueError(f"National river network has no outlet candidate: segments={sorted(segment_ids)}")
        substantial_leaves = [
            node for node in leaves
            if line_by_id[adjacency[node][0][0]].length >= MIN_RIVER_CONTRIBUTING_REACH_PX
        ]
        if substantial_leaves:
            leaves = substantial_leaves
        leaf_heights = {node: terrain_height_at(grid, *node_coordinates[node]) for node in leaves}
        lowest_height = min(leaf_heights.values())
        outlet_candidates = [
            node for node in leaves if leaf_heights[node] <= lowest_height + 20.0
        ]
        root = min(
            outlet_candidates,
            key=lambda node: (sink_proximity(node), leaf_heights[node], node[1], node[0]),
        )

        parent: dict[tuple[float, float], tuple[float, float] | None] = {root: None}
        parent_segment: dict[tuple[float, float], int] = {}
        order = [root]
        for node in order:
            for segment_id, neighbour in adjacency[node]:
                if neighbour == parent[node]:
                    continue
                if neighbour in parent:
                    raise ValueError(
                        f"National river network has a loop near {node}: segments={sorted(segment_ids)}"
                    )
                parent[neighbour] = node
                parent_segment[neighbour] = segment_id
                order.append(neighbour)
        if len(parent) != len(nodes) or len(parent_segment) != len(segment_ids):
            raise ValueError(f"Could not orient all reaches in basin {min(segment_ids)}")

        source_counts = {}
        for node in reversed(order):
            upstream_nodes = [
                neighbour for _, neighbour in adjacency[node]
                if parent.get(neighbour) == node
            ]
            if upstream_nodes:
                source_counts[node] = sum(source_counts[upstream] for upstream in upstream_nodes)
            else:
                incoming_id = parent_segment.get(node)
                source_counts[node] = int(
                    incoming_id is not None
                    and line_by_id[incoming_id].length >= MIN_RIVER_CONTRIBUTING_REACH_PX
                )
        reach_source_counts = {
            parent_segment[child]: source_counts[child]
            for child in parent_segment
        }
        components.append({
            "segment_ids": sorted(segment_ids),
            "nodes": nodes,
            "root": root,
            "root_height_m": round(leaf_heights[root], 2),
            "parent": parent,
            "parent_segment": parent_segment,
            "source_counts": source_counts,
            "reach_source_counts": reach_source_counts,
            "adjacency": adjacency,
            "node_coordinates": node_coordinates,
            "edge_nodes": edge_nodes,
            "line_by_id": line_by_id,
        })
    return sorted(components, key=lambda item: item["segment_ids"][0])


def segment_endpoint_side(edge_nodes: dict, segment_id: int, node: tuple[float, float]) -> str:
    start_node, end_node = edge_nodes[segment_id]
    if start_node == node:
        return "start"
    if end_node == node:
        return "end"
    raise ValueError(f"Segment {segment_id} is not connected to node {node}")


def set_profile_endpoint_width(profile: dict, side: str, width: float) -> None:
    width = round(float(width), 3)
    profile[f"{side}_width_px"] = width
    knots = profile["width_knots_px"]
    knots[0 if side == "start" else -1][1] = width


def integrated_profile_width(profile: dict) -> float:
    knots = profile["width_knots_px"]
    return sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2
        for left, right in zip(knots, knots[1:])
    )


def bank_terrain_samples(line: LineString, grid: dict) -> list[float]:
    samples = []
    for index in range(1, 10):
        distance = line.length * index / 10
        point = line.interpolate(distance)
        before = line.interpolate(max(0.0, distance - 1.5))
        after = line.interpolate(min(line.length, distance + 1.5))
        tangent_x, tangent_y = after.x - before.x, after.y - before.y
        tangent_length = math.hypot(tangent_x, tangent_y) or 1.0
        normal_x, normal_y = -tangent_y / tangent_length, tangent_x / tangent_length
        for direction in (-1.0, 1.0):
            samples.append(terrain_height_at(
                grid,
                point.x + direction * normal_x * 4.0,
                point.y + direction * normal_y * 4.0,
            ))
    return samples


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def plain_width_factor(elevation_m: float) -> float:
    """Widen channels gradually as highland rivers descend into lowlands."""
    anchors = [
        (0.0, 1.35),
        (500.0, 1.25),
        (1000.0, 1.16),
        (2000.0, 1.08),
        (3000.0, 1.0),
    ]
    elevation_m = max(0.0, float(elevation_m))
    if elevation_m >= anchors[-1][0]:
        return 1.0
    for (low_elevation, low_factor), (high_elevation, high_factor) in zip(anchors, anchors[1:]):
        if elevation_m <= high_elevation:
            fraction = (elevation_m - low_elevation) / (high_elevation - low_elevation)
            return low_factor + (high_factor - low_factor) * fraction
    return 1.0


def region_for_point(point: Point, region_features: list[dict]) -> dict | None:
    matches = [
        feature for feature in region_features
        if shape(feature["geometry"]).buffer(0.05).covers(point)
    ]
    if matches:
        return matches[0]
    nearest = min(
        region_features,
        key=lambda feature: shape(feature["geometry"]).distance(point),
        default=None,
    )
    if nearest is None or shape(nearest["geometry"]).distance(point) > 8.0:
        return None
    return nearest


def build_national_river_width_config(
    rivers: list[dict],
    q2_config: dict,
    country_outline,
    lakes: list[dict],
    regions: list[dict],
    grid: dict,
) -> dict:
    """Extend the reviewed Q2 profiles to every national river reach."""
    config = json.loads(json.dumps(q2_config))
    q2_profiles = q2_config["reach_profiles_px"]
    forest = river_network_forest(rivers, country_outline, lakes, grid)
    reach_profiles = {}
    reach_component = {}
    adjustment_candidates = []
    auto_islands = []
    summaries = []
    confluence_checks = json.loads(json.dumps(q2_config.get("confluence_width_checks", [])))
    continuity_checks = json.loads(json.dumps(q2_config.get("channel_continuity_checks", [])))
    rivers_by_id = {
        int(feature["properties"]["segment_id"]): feature
        for feature in rivers
    }
    manually_oriented_nodes = set()
    for check in q2_config.get("confluence_width_checks", []):
        endpoints = [*check["upstream_reaches"], check["downstream_reach"]]
        points = [
            reach_endpoint(rivers_by_id, endpoint)[0]
            for endpoint in endpoints
        ]
        if max(point.distance(points[0]) for point in points[1:]) <= 0.15:
            manually_oriented_nodes.add((round(points[0].x, 1), round(points[0].y, 1)))

    for component_index, component in enumerate(forest, start=1):
        segment_ids = component["segment_ids"]
        manual_ids = [segment_id for segment_id in segment_ids if str(segment_id) in q2_profiles]
        manual_networks = sorted({
            q2_profiles[str(segment_id)]["network_id"] for segment_id in manual_ids
        })
        if len(manual_networks) == 1:
            network_id = manual_networks[0]
        else:
            network_id = f"national_basin_{segment_ids[0]:03d}"
        for segment_id in segment_ids:
            reach_component[segment_id] = network_id

        scale_candidates = [
            integrated_profile_width(width_profiles(q2_config)[segment_id])
            / math.sqrt(component["reach_source_counts"][segment_id])
            for segment_id in manual_ids
            if component["reach_source_counts"][segment_id] > 0
        ]
        base_width = statistics.median(scale_candidates) if scale_candidates else 0.20
        base_width = min(0.34, max(0.14, base_width))

        base_width = round(base_width, 4)
        parent = component["parent"]
        parent_segment = component["parent_segment"]
        source_counts = component["source_counts"]
        adjacency = component["adjacency"]
        edge_nodes = component["edge_nodes"]
        manual_set = set(manual_ids)

        for segment_id in segment_ids:
            if segment_id in manual_set:
                profile = json.loads(json.dumps(q2_profiles[str(segment_id)]))
                profile["flow_accumulation_count"] = component["reach_source_counts"][segment_id]
                profile["cartographic_profile_source"] = "reviewed_q2_profile"
            else:
                source_count = component["reach_source_counts"][segment_id]
                line = component["line_by_id"][segment_id]
                start_node, end_node = edge_nodes[segment_id]
                if source_count > 0:
                    width_knots = []
                    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                        point = line.interpolate(line.length * fraction)
                        if fraction == 0.0:
                            height = terrain_height_at(grid, *component["node_coordinates"][start_node])
                        elif fraction == 1.0:
                            height = terrain_height_at(grid, *component["node_coordinates"][end_node])
                        else:
                            height = terrain_height_at(grid, point.x, point.y)
                        local_width = min(
                            0.8,
                            base_width * math.sqrt(source_count) * plain_width_factor(height),
                        )
                        width_knots.append([fraction, round(local_width, 3)])
                else:
                    width_knots = [[0.0, 0.12], [1.0, 0.12]]
                width = width_knots[0][1]
                profile = {
                    "network_id": network_id,
                    "start_width_px": width,
                    "end_width_px": width_knots[-1][1],
                    "width_knots_px": width_knots,
                    "clip_to_quadrant": False,
                    "flow_accumulation_count": source_count,
                    "cartographic_profile_source": "basin_tree_and_lowland_transition",
                }
            profile["network_id"] = network_id
            profile["clip_to_quadrant"] = False
            reach_profiles[str(segment_id)] = profile

        def endpoint_width(segment_id: int, node: tuple[float, float]) -> float:
            side = segment_endpoint_side(edge_nodes, segment_id, node)
            return profile_width_at(
                width_profiles({"reach_profiles_px": {str(segment_id): reach_profiles[str(segment_id)]}})[segment_id],
                0.0 if side == "start" else 1.0,
            )

        # Make adjacent reaches meet at simple bends/junctions.
        for node in component["nodes"]:
            incident = [segment_id for segment_id, _ in adjacency[node]]
            if len(incident) != 2:
                continue
            manual_incident = [segment_id for segment_id in incident if segment_id in manual_set]
            if manual_incident:
                target_width = statistics.median(endpoint_width(segment_id, node) for segment_id in manual_incident)
                if len(manual_incident) > 1:
                    for segment_id in manual_incident:
                        side = segment_endpoint_side(edge_nodes, segment_id, node)
                        set_profile_endpoint_width(reach_profiles[str(segment_id)], side, target_width)
                for segment_id in incident:
                    if segment_id not in manual_set:
                        side = segment_endpoint_side(edge_nodes, segment_id, node)
                        set_profile_endpoint_width(reach_profiles[str(segment_id)], side, target_width)

        # At every bifurcation, grow the downstream bank width from the sum
        # of the incoming channel capacities while holding tributary mouths.
        for node in component["nodes"]:
            if node in manually_oriented_nodes:
                continue
            upstream_nodes = [
                neighbour for _, neighbour in adjacency[node]
                if parent.get(neighbour) == node
                and source_counts[neighbour] > 0
            ]
            if parent[node] is None:
                continue
            downstream_id = parent_segment[node]
            if len(upstream_nodes) == 1:
                # A short traced spur can make a normal bend look like a
                # three-way junction. Preserve continuity along the real flow.
                incoming_id = parent_segment[upstream_nodes[0]]
                pair = {incoming_id, downstream_id}
                manual_pair = pair & manual_set
                if manual_pair:
                    target_width = statistics.median(
                        endpoint_width(segment_id, node) for segment_id in manual_pair
                    )
                    for segment_id in pair - manual_pair:
                        side = segment_endpoint_side(edge_nodes, segment_id, node)
                        set_profile_endpoint_width(
                            reach_profiles[str(segment_id)], side, target_width,
                        )
                if not any(
                    {item["first_reach"]["segment_id"], item["second_reach"]["segment_id"]} == pair
                    for item in continuity_checks
                ):
                    continuity_checks.append({
                        "id": f"{network_id}_through_{round(node[0],1)}_{round(node[1],1)}",
                        "first_reach": {
                            "segment_id": incoming_id,
                            "endpoint": segment_endpoint_side(edge_nodes, incoming_id, node),
                        },
                        "second_reach": {
                            "segment_id": downstream_id,
                            "endpoint": segment_endpoint_side(edge_nodes, downstream_id, node),
                        },
                        "max_node_gap_px": 0.15,
                        "max_width_difference_px": 0.035,
                    })
                continue
            if len(upstream_nodes) < 2:
                continue
            incoming = [parent_segment[upstream] for upstream in upstream_nodes]
            incoming_widths = [endpoint_width(segment_id, node) for segment_id in incoming]
            required_width = math.sqrt(sum(width ** 2 for width in incoming_widths))
            if downstream_id not in manual_set:
                downstream_side = segment_endpoint_side(edge_nodes, downstream_id, node)
                current_width = endpoint_width(downstream_id, node)
                set_profile_endpoint_width(
                    reach_profiles[str(downstream_id)], downstream_side,
                    max(current_width, required_width + 0.003),
                )
            confluence_checks.append({
                "id": f"{network_id}_to_{parent_segment[node]}_at_{round(node[0],1)}_{round(node[1],1)}",
                "upstream_reaches": [
                    {"segment_id": segment_id, "endpoint": segment_endpoint_side(edge_nodes, segment_id, node)}
                    for segment_id in incoming
                ],
                "downstream_reach": {
                    "segment_id": downstream_id,
                    "endpoint": segment_endpoint_side(edge_nodes, downstream_id, node),
                },
                "minimum_ratio": 1.0,
                "max_node_gap_px": 0.15,
            })

        for node in component["nodes"]:
            incident = [segment_id for segment_id, _ in adjacency[node]]
            if len(incident) == 2:
                continuity_checks.append({
                    "id": f"{network_id}_through_{round(node[0],1)}_{round(node[1],1)}",
                    "first_reach": {"segment_id": incident[0], "endpoint": segment_endpoint_side(edge_nodes, incident[0], node)},
                    "second_reach": {"segment_id": incident[1], "endpoint": segment_endpoint_side(edge_nodes, incident[1], node)},
                    "max_node_gap_px": 0.15,
                    "max_width_difference_px": 0.035,
                })

        summaries.append({
            "network_id": network_id,
            "segment_ids": segment_ids,
            "headwater_count": source_counts[component["root"]],
            "outlet_xy": [round(value, 2) for value in component["node_coordinates"][component["root"]]],
            "outlet_height_m": component["root_height_m"],
            "base_width_px": base_width,
        })

        # Add one broad meander and a small point bar only where neighboring
        # land heights identify a low, low-relief plain reach.
        eligible = []
        for segment_id in segment_ids:
            line = component["line_by_id"][segment_id]
            source_count = component["reach_source_counts"][segment_id]
            if segment_id in manual_set or source_count < 2 or line.length < 60:
                continue
            samples = bank_terrain_samples(line, grid)
            p75 = percentile(samples, 0.75)
            relief = percentile(samples, 0.9) - percentile(samples, 0.1)
            chord = math.dist(line.coords[0], line.coords[-1])
            sinuosity = line.length / max(chord, 1.0)
            if p75 > 500 or relief > 500 or sinuosity > 1.22:
                continue
            score = line.length * (1.25 - sinuosity)
            eligible.append((score, segment_id, p75, relief, sinuosity))
        if not eligible:
            continue
        _, segment_id, p75, relief, sinuosity = max(eligible)
        width = profile_width_at(
            width_profiles({"reach_profiles_px": {str(segment_id): reach_profiles[str(segment_id)]}})[segment_id],
            0.5,
        )
        adjustment = {
            "id": f"national_plain_s_bend_{segment_id}",
            "segment_id": segment_id,
            "start_fraction": 0.06,
            "end_fraction": 0.94,
            "amplitude_px": round(min(0.85, max(0.35, width * 1.45)), 3),
            "note": (
                "Broad, asymmetric S bend on a low-relief plain reach; "
                "bank terrain p75=" + str(round(p75)) + "m, sinuosity=" + str(round(sinuosity, 3)) + "."
            ),
        }
        adjustment_candidates.append(adjustment)
        if component["reach_source_counts"][segment_id] >= 3 and width >= 0.32:
            center_fraction = 0.68
            center = component["line_by_id"][segment_id].interpolate(
                component["line_by_id"][segment_id].length * center_fraction,
            )
            region = region_for_point(center, regions)
            if region and region["properties"].get("region_code") and region["properties"].get("fill_rgb"):
                island_width = round(width * 0.29, 3)
                auto_islands.append({
                    "id": f"national-bar-{segment_id}",
                    "segment_id": segment_id,
                    "fraction_along_reach": center_fraction,
                    "length_px": round(max(1.2, min(2.2, width * 3.8)), 3),
                    "width_px": island_width,
                    "expected_region_code": region["properties"]["region_code"],
                    "shape_phase": round((segment_id * 2.399963229728653) % math.tau, 3),
                    "lateral_offset_fraction": 0.06,
                    "region_snap_tolerance_px": 8.0,
                })

    config.update({
        "quadrant": "National",
        "definition": "All mapped national river reaches, with upper-left Q2 profiles retained and remaining reaches scaled by routed headwater contribution.",
        "extent_px": list(country_outline.bounds),
        "width_semantics": (
            "Display-only widths in source image pixels. Q2 keeps its reviewed profiles; "
            "other basins use a compact width per headwater, combined quadratically at each confluence. "
            "No width is a literal surveyed kilometre value."
        ),
        "reach_profiles_px": {key: reach_profiles[key] for key in sorted(reach_profiles, key=int)},
        "confluence_width_checks": confluence_checks,
        "channel_continuity_checks": continuity_checks,
        "centerline_adjustments": [
            *q2_config.get("centerline_adjustments", []),
            *adjustment_candidates,
        ],
        "islands": [*q2_config.get("islands", []), *auto_islands],
        "national_network_summaries": summaries,
        "national_width_method": {
            "outlet_rule": "Each endpoint tree is rooted at its lowest mapped water surface, with coast or lake proximity resolving near-equal elevations.",
            "confluence_rule": "Downstream width is at least the square root of the summed squared upstream reach widths.",
            "default_headwater_width_px": 0.2,
            "lowland_width_factors": [
                {"elevation_m": 3000, "factor": 1.0},
                {"elevation_m": 2000, "factor": 1.08},
                {"elevation_m": 1000, "factor": 1.16},
                {"elevation_m": 500, "factor": 1.25},
                {"elevation_m": 0, "factor": 1.35},
            ],
            "plain_meander_selection": "One long, comparatively straight reach per basin when both banks remain below 500m with less than 500m relief spread.",
        },
    })
    config["reach_notes"] = {
        **q2_config.get("reach_notes", {}),
        **{
            str(segment_id): f"National basin reach; mapped upstream source count {reach_profiles[str(segment_id)]['flow_accumulation_count']}."
            for segment_id in reach_profiles
            if reach_profiles[str(segment_id)].get("cartographic_profile_source") != "reviewed_q2_profile"
        },
    }
    return config


def variable_width_channel(
    reach: LineString,
    start_width: float,
    end_width: float,
    sample_step: float = 0.55,
    phase: float = 0.0,
    width_knots: list[tuple[float, float]] | None = None,
):
    """Build a tapered channel with independently varying, bend-aware banks."""
    if reach.length <= 0:
        return Polygon()
    sample_count = max(2, math.ceil(reach.length / sample_step))
    left_bank = []
    right_bank = []
    start_bank_widths = None
    end_bank_widths = None
    width_knots = sorted(width_knots or [(0.0, start_width), (1.0, end_width)])

    def width_at(fraction: float) -> float:
        for (left_fraction, left_width), (right_fraction, right_width) in zip(width_knots, width_knots[1:]):
            if fraction <= right_fraction:
                local = (fraction - left_fraction) / (right_fraction - left_fraction)
                return left_width + (right_width - left_width) * local
        return width_knots[-1][1]

    for index in range(sample_count + 1):
        fraction = index / sample_count
        distance = reach.length * fraction
        tangent_span = min(2.5, max(0.75, reach.length / (sample_count * 2)))
        before = reach.interpolate(max(0.0, distance - tangent_span))
        center = reach.interpolate(distance)
        after = reach.interpolate(min(reach.length, distance + tangent_span))
        tangent_x = after.x - before.x
        tangent_y = after.y - before.y
        tangent_length = math.hypot(tangent_x, tangent_y) or 1.0
        normal_x = -tangent_y / tangent_length
        normal_y = tangent_x / tangent_length
        width = width_at(fraction)
        half_width = width / 2
        incoming_x, incoming_y = center.x - before.x, center.y - before.y
        outgoing_x, outgoing_y = after.x - center.x, after.y - center.y
        turn = math.atan2(
            incoming_x * outgoing_y - incoming_y * outgoing_x,
            incoming_x * outgoing_x + incoming_y * outgoing_y,
        )
        curvature = turn / max(2 * tangent_span, 0.75)
        # Keep bank asymmetry gradual: outer bends ease outward, while inner
        # bends tighten without adding sharp vertical-looking wiggles.
        bend_bias = max(-0.18, min(0.18, curvature * 4.5))
        left_waves = (
            0.075 * math.sin(2 * math.pi * 1.15 * fraction + phase)
            + 0.035 * math.sin(2 * math.pi * 3.1 * fraction + phase * 0.61 + 1.1)
        )
        right_waves = (
            0.095 * math.sin(2 * math.pi * 1.3 * fraction + phase + 1.65)
            + 0.035 * math.sin(2 * math.pi * 3.5 * fraction + phase * 0.83 + 0.4)
        )
        left_width = half_width * max(0.72, 1 - bend_bias + left_waves)
        right_width = half_width * max(0.72, 1 + bend_bias + right_waves)
        if index == 0:
            start_bank_widths = (left_width, right_width)
        if index == sample_count:
            end_bank_widths = (left_width, right_width)
        left_bank.append((
            round(center.x + normal_x * left_width, 3),
            round(center.y + normal_y * left_width, 3),
        ))
        right_bank.append((
            round(center.x - normal_x * right_width, 3),
            round(center.y - normal_y * right_width, 3),
        ))

    ribbon = Polygon(left_bank + list(reversed(right_bank)))
    if not ribbon.is_valid:
        ribbon = ribbon.buffer(0)
    start_radius = sum(start_bank_widths or (start_width / 2, start_width / 2)) / 2
    end_radius = sum(end_bank_widths or (end_width / 2, end_width / 2)) / 2
    start_cap = Point(reach.coords[0]).buffer(start_radius, resolution=8)
    end_cap = Point(reach.coords[-1]).buffer(end_radius, resolution=8)
    return unary_union([ribbon, start_cap, end_cap]).simplify(0.06, preserve_topology=True)


def river_base_features(features: list[dict], config: dict) -> list[dict]:
    """Keep ordinary centerlines outside Q2; Q2 profiles use filled channels."""
    q2 = box(*config["extent_px"])
    profiles = width_profiles(config)
    result = []
    for feature in features:
        segment_id = int(feature["properties"].get("segment_id", 0))
        profile = profiles.get(segment_id)
        if profile is None:
            result.append(feature)
            continue
        if not profile["clip_to_quadrant"]:
            continue
        outside = shape(feature["geometry"]).difference(q2)
        if line_parts(outside):
            result.append({**feature, "geometry": mapping(outside)})
    return result


def river_channel_features(rivers: list[dict], config: dict) -> list[dict]:
    q2 = box(*config["extent_px"])
    min_x, min_y, max_x, max_y = q2.bounds
    edge_width = float(config.get("quadrant_edge_transition_width_px", 1.25))
    profiles = width_profiles(config)
    rivers_by_id = {
        int(feature["properties"]["segment_id"]): feature
        for feature in rivers
    }
    missing = sorted(set(profiles) - set(rivers_by_id))
    if missing:
        raise ValueError(f"River channel profiles reference missing segments: {missing}")

    network_geometries: dict[str, list] = {}
    network_segments: dict[str, list[int]] = {}
    for segment_id, profile in sorted(profiles.items()):
        lines = line_parts(shape(rivers_by_id[segment_id]["geometry"]))
        if not lines:
            raise ValueError(f"River channel profile has no line reach: segment {segment_id}")
        reach = smooth_river_centerline(max(lines, key=lambda line: line.length))
        if profile["clip_to_quadrant"]:
            channel_parts = []
            clipped_reaches = line_parts(reach.intersection(q2))
            for clipped_reach in clipped_reaches:
                if clipped_reach.length <= 0:
                    continue
                clipped_start = Point(clipped_reach.coords[0])
                clipped_end = Point(clipped_reach.coords[-1])
                start_fraction = reach.project(clipped_start) / reach.length
                end_fraction = reach.project(clipped_end) / reach.length
                start_width = profile_width_at(profile, start_fraction)
                end_width = profile_width_at(profile, end_fraction)
                interval = end_fraction - start_fraction
                local_width_knots = [(0.0, start_width), (1.0, end_width)]
                if abs(interval) > 1e-9:
                    for knot_fraction, knot_width in profile["width_knots_px"][1:-1]:
                        local_fraction = (knot_fraction - start_fraction) / interval
                        if 1e-6 < local_fraction < 1.0 - 1e-6:
                            local_width_knots.append((local_fraction, knot_width))
                local_width_knots.sort()
                for point, at_start in ((clipped_start, True), (clipped_end, False)):
                    if (
                        abs(point.x - min_x) < 0.01
                        or abs(point.x - max_x) < 0.01
                        or abs(point.y - min_y) < 0.01
                        or abs(point.y - max_y) < 0.01
                    ):
                        if at_start:
                            start_width = edge_width
                        else:
                            end_width = edge_width
                local_width_knots[0] = (0.0, start_width)
                local_width_knots[-1] = (1.0, end_width)
                channel_parts.append(variable_width_channel(
                    clipped_reach,
                    start_width,
                    end_width,
                    phase=(segment_id * 2.399963229728653 + start_fraction * math.tau) % math.tau,
                    width_knots=local_width_knots,
                ))
            channel = unary_union(channel_parts) if channel_parts else Polygon()
        else:
            channel = variable_width_channel(
                reach,
                profile["start_width_px"],
                profile["end_width_px"],
                phase=(segment_id * 2.399963229728653) % math.tau,
                width_knots=profile["width_knots_px"],
            )
        if profile["clip_to_quadrant"]:
            channel = channel.intersection(q2)
        if channel.is_empty:
            continue
        network = profile["network_id"]
        network_geometries.setdefault(network, []).append(channel)
        network_segments.setdefault(network, []).append(segment_id)

    features = []
    for network in sorted(network_geometries):
        channel = unary_union(network_geometries[network])
        if not channel.is_valid:
            channel = channel.buffer(0)
        segment_ids = sorted(network_segments[network])
        features.append({
            "type": "Feature",
            "properties": {
                "network_id": network,
                "segment_ids": segment_ids,
                "source": "hydrology_reach_connectivity_and_downstream_width_profile",
                "review_status": "variable_width_channel_polygon_with_asymmetric_banks",
            },
            "geometry": mapping(channel),
        })
    return features


def island_polygon(
    reach: LineString,
    fraction: float,
    length: float,
    width: float,
    phase: float = 0.0,
    lateral_offset_fraction: float = 0.0,
) -> Polygon:
    """Make a gently curved, asymmetrical mid-channel bar following the reach."""
    center_distance = reach.length * fraction
    start_distance = max(0.0, center_distance - length / 2)
    end_distance = min(reach.length, center_distance + length / 2)
    span = max(0.01, end_distance - start_distance)
    sample_count = max(12, math.ceil(span / 0.35))
    left_bank = []
    right_bank = []
    for index in range(sample_count + 1):
        along = index / sample_count
        distance = start_distance + span * along
        tangent_span = min(2.0, max(0.5, span / sample_count))
        before = reach.interpolate(max(0.0, distance - tangent_span))
        center = reach.interpolate(distance)
        after = reach.interpolate(min(reach.length, distance + tangent_span))
        tangent_x = after.x - before.x
        tangent_y = after.y - before.y
        tangent_length = math.hypot(tangent_x, tangent_y) or 1.0
        normal_x = -tangent_y / tangent_length
        normal_y = tangent_x / tangent_length
        envelope_exponent = 0.9 - 0.4 * along
        envelope = math.sin(math.pi * along) ** envelope_exponent
        envelope *= 0.84 + 0.16 * along
        broad_shoulder = math.exp(-((along - (0.66 + 0.06 * math.sin(phase))) / 0.16) ** 2)
        shallow_notch = math.exp(-((along - (0.38 + 0.05 * math.cos(phase))) / 0.13) ** 2)
        left_variation = (
            0.18 * math.sin(2 * math.pi * along + phase)
            + 0.07 * math.sin(4 * math.pi * along + phase * 0.7 + 1.25)
            + 0.13 * broad_shoulder
        )
        right_variation = (
            0.15 * math.sin(2 * math.pi * along + phase + 1.8)
            + 0.06 * math.sin(5 * math.pi * along + phase * 0.9 + 0.45)
            - 0.08 * shallow_notch
        )
        left_width = width * 0.5 * envelope * max(0.48, 1 + left_variation)
        right_width = width * 0.5 * envelope * max(0.48, 1 + right_variation)
        lateral_shift = width * (
            lateral_offset_fraction + 0.1 * math.sin(2 * math.pi * along + phase)
        )
        center_x = center.x + normal_x * lateral_shift
        center_y = center.y + normal_y * lateral_shift
        left_bank.append((
            round(center_x + normal_x * left_width, 3),
            round(center_y + normal_y * left_width, 3),
        ))
        right_bank.append((
            round(center_x - normal_x * right_width, 3),
            round(center_y - normal_y * right_width, 3),
        ))
    polygon = Polygon(left_bank + list(reversed(right_bank)))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def river_island_features(
    rivers: list[dict],
    river_channels: list[dict],
    region_features: list[dict],
    config: dict,
) -> list[dict]:
    rivers_by_id = {
        int(feature["properties"]["segment_id"]): feature
        for feature in rivers
    }
    region_shapes = [
        (feature["properties"].get("region_code"), feature["properties"].get("fill_rgb"), shape(feature["geometry"]))
        for feature in region_features
    ]
    channels_by_network = {
        str(feature["properties"]["network_id"]): shape(feature["geometry"])
        for feature in river_channels
    }
    q2 = box(*config["extent_px"])
    profiles = width_profiles(config)
    islands = []
    for definition in config["islands"]:
        segment_id = int(definition["segment_id"])
        feature = rivers_by_id.get(segment_id)
        if feature is None:
            raise ValueError(f"Q2 river island {definition['id']} references missing segment {segment_id}")
        lines = line_parts(shape(feature["geometry"]))
        if not lines:
            raise ValueError(f"Q2 river island {definition['id']} has no line reach to follow")
        reach = smooth_river_centerline(max(lines, key=lambda line: line.length))
        point = reach.interpolate(reach.length * float(definition["fraction_along_reach"]))
        expected_code = definition["expected_region_code"]
        region = next(
            (item for item in region_shapes if item[0] == expected_code and item[2].covers(point)),
            None,
        )
        if region is None:
            expected_regions = [item for item in region_shapes if item[0] == expected_code]
            region = min(expected_regions, key=lambda item: item[2].distance(point), default=None)
            if region is None or region[2].distance(point) > float(definition.get("region_snap_tolerance_px", 8.0)):
                raise ValueError(f"Q2 river island {definition['id']} has no nearby fill region {expected_code}")
        rgb = region[1]
        fill = "#" + "".join(f"{int(channel):02x}" for channel in rgb)
        geometry = island_polygon(
            reach,
            float(definition["fraction_along_reach"]),
            float(definition["length_px"]),
            float(definition["width_px"]),
            float(definition.get("shape_phase", 0.0)),
            float(definition.get("lateral_offset_fraction", 0.0)),
        )
        profile = profiles.get(segment_id)
        if profile is None:
            raise ValueError(f"Q2 river island {definition['id']} has no matching channel profile")
        fraction = float(definition["fraction_along_reach"])
        channel_width = profile_width_at(profile, fraction)
        if float(definition["width_px"]) * 1.25 >= channel_width:
            raise ValueError(f"Q2 river island {definition['id']} is wider than its river channel")
        network_channel = channels_by_network.get(profile["network_id"])
        if network_channel is None:
            raise ValueError(f"Q2 river island {definition['id']} references missing channel network {profile['network_id']}")
        if (
            not q2.covers(geometry)
            or not network_channel.buffer(1e-4).covers(geometry)
            or not reach.buffer(channel_width / 2, cap_style=1, join_style=1).covers(geometry)
        ):
            raise ValueError(
                f"Q2 river island {definition['id']} crosses a quadrant or river-bank boundary"
            )
        islands.append({
            "type": "Feature",
            "properties": {
                "island_id": definition["id"],
                "segment_id": segment_id,
                "fraction_along_reach": definition["fraction_along_reach"],
                "width_px": definition["width_px"],
                "length_px": definition["length_px"],
                "region_code": expected_code,
                "fill": fill,
                "review_status": "hand_placed_asymmetric_midchannel_bar",
            },
            "geometry": mapping(geometry),
        })
    return islands


def lake_island_geometry(definition: dict) -> Polygon:
    center_x, center_y = (float(value) for value in definition["center_xy"])
    major = float(definition["major_radius_px"])
    minor = float(definition["minor_radius_px"])
    rotation = math.radians(float(definition.get("rotation_deg", 0.0)))
    phase = float(definition.get("shape_phase", 0.0))
    points = []
    for index in range(64):
        angle = 2 * math.pi * index / 64
        radial = (
            1.0
            + 0.09 * math.sin(3 * angle + phase)
            + 0.045 * math.sin(5 * angle + phase * 0.73 + 1.1)
            + 0.055 * math.cos(angle - phase)
        )
        along = major * math.cos(angle) * radial
        across = minor * math.sin(angle) * radial
        x = center_x + along * math.cos(rotation) - across * math.sin(rotation)
        y = center_y + along * math.sin(rotation) + across * math.cos(rotation)
        points.append((round(x, 3), round(y, 3)))
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def lake_island_features(
    lakes: list[dict], region_features: list[dict], config: dict,
) -> list[dict]:
    lakes_by_id = {
        int(feature["properties"]["lake_id"]): shape(feature["geometry"])
        for feature in lakes
    }
    regions = [
        (feature["properties"].get("region_code"), feature["properties"].get("fill_rgb"))
        for feature in region_features
    ]
    islands = []
    for definition in config.get("lake_islands", []):
        lake_id = int(definition["lake_id"])
        lake = lakes_by_id.get(lake_id)
        if lake is None:
            raise ValueError(f"Lake island {definition['id']} references missing lake {lake_id}")
        geometry = lake_island_geometry(definition)
        shore_margin = float(definition.get("shore_margin_px", 1.5))
        if not lake.buffer(-shore_margin).covers(geometry):
            raise ValueError(f"Lake island {definition['id']} is too close to the lake shore")
        region_code = definition["fill_region_code"]
        region = next((item for item in regions if item[0] == region_code), None)
        if region is None or not region[1]:
            raise ValueError(f"Lake island {definition['id']} references missing fill region {region_code}")
        fill = "#" + "".join(f"{int(channel):02x}" for channel in region[1])
        islands.append({
            "type": "Feature",
            "properties": {
                "island_id": definition["id"],
                "lake_id": lake_id,
                "region_code": region_code,
                "fill": fill,
                "creek_id": f"{definition['id']}-creek",
                "review_status": "irregular_lake_center_island_with_meandering_creek",
            },
            "geometry": mapping(geometry),
        })
    return islands


def lake_island_creek_features(
    lake_islands: list[dict], config: dict,
) -> list[dict]:
    definitions = {item["id"]: item for item in config.get("lake_islands", [])}
    creeks = []
    for island in lake_islands:
        properties = island["properties"]
        definition = definitions[properties["island_id"]]
        line = smooth_river_centerline(LineString(definition["creek_coordinates"]))
        channel = variable_width_channel(
            line,
            float(definition["creek_start_width_px"]),
            float(definition["creek_end_width_px"]),
            sample_step=0.3,
            phase=float(definition.get("shape_phase", 0.0)) + 1.7,
        )
        island_shape = shape(island["geometry"])
        if not island_shape.buffer(0.02).covers(channel):
            raise ValueError(f"Creek {properties['creek_id']} leaves its lake-center island")
        creeks.append({
            "type": "Feature",
            "properties": {
                "creek_id": properties["creek_id"],
                "island_id": properties["island_id"],
                "lake_id": properties["lake_id"],
                "network_id": f"lake_island_creek_{properties['lake_id']}",
                "source": "hand_placed_meander_on_q2_lake_island",
                "review_status": "meandering_creek_inside_lake_island",
            },
            "geometry": mapping(channel),
        })
    return creeks


def render_river_islands_group(features: list[dict], group_id: str = "riverIslands") -> str:
    paths = []
    for feature in features:
        properties = feature["properties"]
        path = polygon_path_data(feature["geometry"])
        if path:
            paths.append(
                f'<path d="{path}" fill="{properties["fill"]}" fill-rule="evenodd" '
                f'data-river-island-id="{properties["island_id"]}"/>'
            )
    return f'<g id="{group_id}" data-layer="rivers" pointer-events="none">\n' + "\n".join(paths) + "\n</g>"


def render_river_channels_group(features: list[dict], group_id: str = "riverChannels") -> str:
    paths = []
    for feature in features:
        properties = feature["properties"]
        path = polygon_path_data(feature["geometry"])
        if path:
            paths.append(
                f'<path d="{path}" fill="#b1d0ed" fill-rule="evenodd" '
                f'data-river-channel="true" data-river-network-id="{properties["network_id"]}"/>'
            )
    return f'<g id="{group_id}" data-layer="rivers" pointer-events="none">\n' + "\n".join(paths) + "\n</g>"


def render_lake_islands_group(features: list[dict]) -> str:
    paths = []
    for feature in features:
        properties = feature["properties"]
        path = polygon_path_data(feature["geometry"])
        if path:
            paths.append(
                f'<path d="{path}" fill="{properties["fill"]}" fill-rule="evenodd" '
                f'data-lake-island-id="{properties["island_id"]}"/>'
            )
    return '<g id="lakeIslandsQ2" data-layer="rivers" pointer-events="none">\n' + "\n".join(paths) + "\n</g>"


def render_lake_island_creeks_group(features: list[dict]) -> str:
    paths = []
    for feature in features:
        properties = feature["properties"]
        path = polygon_path_data(feature["geometry"])
        if path:
            paths.append(
                f'<path d="{path}" fill="#b1d0ed" fill-rule="evenodd" '
                f'data-river-channel="true" data-river-network-id="{properties["network_id"]}" '
                f'data-river-creek-id="{properties["creek_id"]}"/>'
            )
    return '<g id="lakeIslandCreeksQ2" data-layer="rivers" pointer-events="none">\n' + "\n".join(paths) + "\n</g>"


def update_river_channel_hit_testing(index: str) -> str:
    record_marker = "isRiverChannel:element.dataset.riverChannel==='true'"
    record_old = "fixedFontSize:Number.parseFloat(element.dataset.fixedFontSize)||0"
    record_new = record_old + ",\n    " + record_marker
    if record_marker not in index:
        if index.count(record_old) != 1:
            raise RuntimeError("Could not locate the SVG record parser for river channel hit testing")
        index = index.replace(record_old, record_new, 1)

    hit_old = (
        "    hitCtx.lineWidth=Math.max(record.lineWidth,8/scale);\n"
        "    if(!hitCtx.isPointInStroke(record.variants[0],point.x,point.y))continue;"
    )
    hit_new = (
        "    const isRiverHit=record.isRiverChannel\n"
        "      ?hitCtx.isPointInPath(record.variants[0],point.x,point.y,'evenodd')\n"
        "      :(hitCtx.lineWidth=Math.max(record.lineWidth,8/scale),\n"
        "        hitCtx.isPointInStroke(record.variants[0],point.x,point.y));\n"
        "    if(!isRiverHit)continue;"
    )
    if hit_new not in index:
        count = index.count(hit_old)
        if count != 1:
            raise RuntimeError(f"Expected one river hit-test marker, found {count}")
        index = index.replace(hit_old, hit_new, 1)
    return index


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
                "source": "official_map/Administrative Division Map of Donh 2026-river-super.png",
                "review_status": "centerline_retraced_from_official_12x_dash_strokes",
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


def remove_cross_border_lake_edge_residuals(features: list[dict]) -> list[dict]:
    return [
        feature for feature in features
        if feature["properties"].get("segment_id") not in CROSS_BORDER_LAKE_EDGE_RESIDUAL_SEGMENT_IDS
    ]


def clip_cross_border_rivers_to_neighbor_lakes(
    features: list[dict], lakes: list[dict]
) -> tuple[list[dict], list[int], list[int], float]:
    """Remove misclassified lake interiors while retaining river reaches to shore."""
    neighboring_lakes = [
        feature for feature in lakes
        if feature.get("properties", {}).get("territory") == "neighboring_country"
    ]
    if not neighboring_lakes:
        return features, [], [], 0.0

    lake_geometries = {
        int(feature["properties"]["lake_id"]): shape(feature["geometry"])
        for feature in neighboring_lakes
    }
    lake_area = unary_union(list(lake_geometries.values()))
    lake_clip_area = lake_area.buffer(0.001)
    result = []
    removed_ids = []
    clipped_ids = []
    total_removed_length = 0.0

    for feature in features:
        source = shape(feature["geometry"])
        if source.disjoint(lake_area):
            result.append(feature)
            continue

        outside = source.difference(lake_clip_area)
        parts = [part for part in line_parts(outside) if part.length >= 2.0]
        kept_length = sum(part.length for part in parts)
        removed_length = max(0.0, source.length - kept_length)
        if removed_length <= 0.02:
            result.append(feature)
            continue

        segment_id = int(feature["properties"].get("segment_id", 0))
        total_removed_length += removed_length
        if not parts:
            removed_ids.append(segment_id)
            continue

        clipped_ids.append(segment_id)
        lake_ids = sorted(
            lake_id for lake_id, geometry in lake_geometries.items()
            if source.intersects(geometry)
        )
        if len(parts) == 1:
            geometry = round_line(parts[0], digits=6)
        else:
            geometry = {
                "type": "MultiLineString",
                "coordinates": [
                    [[round(x, 6), round(y, 6)] for x, y in part.coords]
                    for part in parts
                ],
            }
        result.append({
            **feature,
            "properties": {
                **feature["properties"],
                "length_px": round(kept_length, 3),
                "lake_overlap_removed_px": round(removed_length, 3),
                "lake_clip_ids": lake_ids,
            },
            "geometry": geometry,
        })

    return result, sorted(removed_ids), sorted(clipped_ids), total_removed_length


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


def render_group(
    group_id: str,
    layer_id: str,
    features: list[dict],
    style: str,
    extra_paths: list[str] | None = None,
) -> str:
    paths = []
    for feature in features:
        d = path_data(feature["geometry"])
        if d:
            paths.append(f'<path d="{d}"/>')
    paths.extend(extra_paths or [])
    return (
        f'<g id="{group_id}" data-layer="{layer_id}" {style} '
        f'pointer-events="none">\n' + "\n".join(paths) + "\n</g>"
    )


def render_preview_group(
    group_id: str,
    features: list[dict],
    style: str,
    extra_paths: list[str] | None = None,
) -> str:
    paths = []
    for feature in features:
        d = path_data(feature["geometry"])
        if d:
            paths.append(f'<path d="{d}"/>')
    paths.extend(extra_paths or [])
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
    style = 'fill="#b1d0ed" fill-rule="evenodd" stroke="#b1d0ed" stroke-width="1.1"'
    if preview:
        style = 'fill="#b1d0ed" fill-rule="evenodd" stroke="#b1d0ed" stroke-width="1.2"'
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


def remove_group(source: str, group_id: str) -> str:
    pattern = rf'<g id="{re.escape(group_id)}"[^>]*>.*?</g>'
    return re.sub(pattern, "", source, count=1, flags=re.DOTALL)


def update_render_files(
    rivers: list[dict],
    river_channels: list[dict],
    canals: list[dict],
    seasonal: list[dict],
    cross_border: list[dict],
    river_islands: list[dict],
    lake_islands: list[dict],
    lake_island_creeks: list[dict],
    width_config: dict,
) -> None:
    lakes = read_fc(DATA / "lakes.geojson")["features"]
    base_rivers = river_base_features(rivers, width_config)
    index_groups = [
        render_group("rivers", "rivers", base_rivers,
                     'fill="none" stroke="#b1d0ed" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"'),
        render_river_channels_group(river_channels),
        render_group("canals", "canals", [f for f in canals if f["properties"].get("symbol_part") == "centerline"],
                     'fill="none" stroke="#b1d0ed" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'),
        render_group("canalTicks", "canalTicks", [f for f in canals if f["properties"].get("symbol_part") == "ruler_tick"],
                     'fill="none" stroke="#b1d0ed" stroke-width="1.5" stroke-linecap="round"'),
        render_group("seasonalRivers", "seasonalRivers", seasonal,
                     'fill="none" stroke="#b1d0ed" stroke-width="1.8" stroke-dasharray="7 6" stroke-linecap="round" stroke-linejoin="round"'),
        render_group("crossBorderRivers", "crossBorderRivers", cross_border,
                     'fill="none" stroke="#b1d0ed" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"'),
    ]
    index_path = ROOT / "index.html"
    index = index_path.read_text(encoding="utf-8")
    index = remove_group(index, "riverChannelsQ2")
    index = remove_group(index, "riverIslandsQ2")
    index = upsert_group(index, index_groups[0], "rivers", None)
    index = upsert_group(index, index_groups[1], "riverChannels", "rivers")
    index = upsert_group(index, index_groups[2], "canals", "riverChannels")
    index = upsert_group(index, index_groups[3], "canalTicks", "canals")
    index = upsert_group(index, index_groups[4], "seasonalRivers", "canalTicks")
    index = upsert_group(index, index_groups[5], "crossBorderRivers", "seasonalRivers")
    index = upsert_group(index, render_river_islands_group(river_islands), "riverIslands", "crossBorderRivers")
    index = upsert_group(index, render_lakes_group(lakes), "lakes", None)
    index = upsert_group(index, render_lake_islands_group(lake_islands), "lakeIslandsQ2", "lakes")
    index = upsert_group(index, render_lake_island_creeks_group(lake_island_creeks), "lakeIslandCreeksQ2", "lakeIslandsQ2")
    index = update_hydrology_lod_precision(index)
    index = update_river_channel_hit_testing(index)
    index_path.write_text(index, encoding="utf-8", newline="\n")

    preview_path = ROOT / "vector_preview.svg"
    preview = preview_path.read_text(encoding="utf-8")
    preview = remove_group(preview, "riverChannelsQ2")
    preview = remove_group(preview, "riverIslandsQ2")
    preview_groups = [
        render_preview_group("rivers", base_rivers,
                            'fill="none" stroke="#b1d0ed" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"'),
        render_river_channels_group(river_channels),
        render_preview_group("canals", [f for f in canals if f["properties"].get("symbol_part") == "centerline"],
                            'fill="none" stroke="#b1d0ed" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'),
        render_preview_group("canal-ticks", [f for f in canals if f["properties"].get("symbol_part") == "ruler_tick"],
                            'fill="none" stroke="#b1d0ed" stroke-width="1.5" stroke-linecap="round"'),
        render_preview_group("seasonal-rivers", seasonal,
                            'fill="none" stroke="#b1d0ed" stroke-width="1.8" stroke-dasharray="7 6" stroke-linecap="round" stroke-linejoin="round"'),
        render_preview_group("cross-border-rivers", cross_border,
                            'fill="none" stroke="#b1d0ed" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"'),
    ]
    preview = upsert_group(preview, preview_groups[0], "rivers", None)
    preview = upsert_group(preview, preview_groups[1], "riverChannels", "rivers")
    preview = upsert_group(preview, preview_groups[2], "canals", "riverChannels")
    preview = upsert_group(preview, preview_groups[3], "canal-ticks", "canals")
    preview = upsert_group(preview, preview_groups[4], "seasonal-rivers", "canal-ticks")
    preview = upsert_group(preview, preview_groups[5], "cross-border-rivers", "seasonal-rivers")
    preview = upsert_group(preview, render_river_islands_group(river_islands), "riverIslands", "cross-border-rivers")
    preview = upsert_group(preview, render_lakes_group(lakes, preview=True), "lakes", None)
    preview = upsert_group(preview, render_lake_islands_group(lake_islands), "lakeIslandsQ2", "lakes")
    preview = upsert_group(preview, render_lake_island_creeks_group(lake_island_creeks), "lakeIslandCreeksQ2", "lakeIslandsQ2")
    preview_path.write_text(preview, encoding="utf-8", newline="\n")


def write_overlay(seasonal: list[dict], canals: list[dict]) -> None:
    image = Image.open(SOURCE).convert("RGB")
    draw = ImageDraw.Draw(image)
    for feature in seasonal:
        coords = [tuple(point) for point in feature["geometry"]["coordinates"]]
        draw.line(coords, fill=(177, 208, 237), width=2)
    for feature in canals:
        coords = [tuple(point) for point in feature["geometry"]["coordinates"]]
        width = 2
        draw.line(coords, fill=(52, 124, 165), width=width)
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
    q2_width_config = q2_river_width_config()
    lakes = read_fc(DATA / "lakes.geojson")["features"]
    region_features = read_fc(DATA / "administrative_regions.geojson")["features"]
    outline_features = read_fc(DATA / "national_outline.geojson")["features"]
    country_outline = unary_union([shape(feature["geometry"]) for feature in outline_features])
    terrain_grid = relief_elevation_grid()
    national_width_config = build_national_river_width_config(
        river_features, q2_width_config, country_outline, lakes,
        region_features, terrain_grid,
    )
    river_features = apply_centerline_adjustments(river_features, national_width_config)
    national_topology_checks = validate_channel_topology(river_features, national_width_config)
    q2_topology_checks = validate_channel_topology(river_features, q2_width_config)
    (DATA / "river_widths_national.json").write_text(
        json.dumps(national_width_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    seasonal = seasonal_features()
    cross_border_fc = read_fc(DATA / "cross_border_rivers.geojson")
    # Exclude raster fragments that exactly repeat the seasonal dash strokes.
    # Then clip any river traces that entered the two verified neighboring
    # lake fills, retaining the reaches that connect to the shoreline.
    original_cross_border_count = len(cross_border_fc["features"])
    cross_border = remove_verified_seasonal_dash_residuals(cross_border_fc["features"])
    seasonal_dash_residual_count = original_cross_border_count - len(cross_border)
    (
        cross_border,
        lake_fill_segment_ids_removed,
        lake_fill_segment_ids_clipped,
        lake_fill_line_length_removed,
    ) = clip_cross_border_rivers_to_neighbor_lakes(cross_border, lakes)
    cross_border_ids_before_edge_filter = {
        int(feature["properties"].get("segment_id", 0)) for feature in cross_border
    }
    cross_border = remove_cross_border_lake_edge_residuals(cross_border)
    cross_border_ids_after_edge_filter = {
        int(feature["properties"].get("segment_id", 0)) for feature in cross_border
    }
    lake_edge_residual_ids_removed = sorted(
        cross_border_ids_before_edge_filter - cross_border_ids_after_edge_filter
    )
    lake_edge_residual_count = len(lake_edge_residual_ids_removed)
    cross_border_fc["features"] = cross_border

    q2_river_channels = river_channel_features(river_features, q2_width_config)
    q2_river_islands = river_island_features(
        river_features, q2_river_channels, region_features, q2_width_config,
    )
    river_channels = river_channel_features(river_features, national_width_config)
    river_islands = river_island_features(
        river_features, river_channels, region_features, national_width_config,
    )
    lake_islands = lake_island_features(lakes, region_features, q2_width_config)
    lake_island_creeks = lake_island_creek_features(lake_islands, q2_width_config)
    write_fc(DATA / "river_channels_q2.geojson", "quadrant_ii_variable_width_river_channels", q2_river_channels)
    write_fc(DATA / "river_islands_q2.geojson", "quadrant_ii_river_islands", q2_river_islands)
    write_fc(DATA / "river_channels_national.geojson", "national_variable_width_river_channels", river_channels)
    write_fc(DATA / "river_islands_national.geojson", "national_river_midchannel_bars", river_islands)
    write_fc(DATA / "lake_islands_q2.geojson", "quadrant_ii_lake_center_islands", lake_islands)
    write_fc(DATA / "lake_island_creeks_q2.geojson", "quadrant_ii_lake_island_creeks", lake_island_creeks)

    write_fc(DATA / "canals.geojson", "canal_centerline_and_ruler_ticks", canals)
    write_fc(DATA / "rivers.geojson", "rivers_excluding_independent_canals", river_features)
    write_fc(DATA / "seasonal_rivers.geojson", "neighbor_seasonal_rivers", seasonal)
    (DATA / "cross_border_rivers_clean.geojson").write_text(
        json.dumps(cross_border_fc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    update_render_files(
        river_features, river_channels, canals, seasonal, cross_border,
        river_islands, lake_islands, lake_island_creeks, national_width_config,
    )
    write_overlay(seasonal, canals)

    manifest_path = ROOT / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "river_line_segments": len(river_features),
        "quadrant_ii_river_width_reaches": [
            {
                "segment_id": int(segment_id),
                "network_id": profile["network_id"],
                "start_width_px": profile["start_width_px"],
                "end_width_px": profile["end_width_px"],
                "width_knots_px": profile["width_knots_px"],
                "clip_to_quadrant": profile["clip_to_quadrant"],
                "note": q2_width_config["reach_notes"].get(str(segment_id), ""),
            }
            for segment_id, profile in sorted(width_profiles(q2_width_config).items())
        ],
        "quadrant_ii_river_centerline_adjustments": q2_width_config.get("centerline_adjustments", []),
        "quadrant_ii_river_topology_checks": q2_topology_checks,
        "quadrant_ii_river_channel_networks": [
            feature["properties"]["network_id"] for feature in q2_river_channels
        ],
        "quadrant_ii_river_islands": [
            feature["properties"]["island_id"] for feature in q2_river_islands
        ],
        "national_river_width_reach_count": len(national_width_config["reach_profiles_px"]),
        "national_river_channel_networks": [
            feature["properties"]["network_id"] for feature in river_channels
        ],
        "national_river_centerline_adjustments": national_width_config.get("centerline_adjustments", []),
        "national_river_topology_checks": national_topology_checks,
        "national_river_islands": [
            feature["properties"]["island_id"] for feature in river_islands
        ],
        "national_river_network_summaries": national_width_config.get("national_network_summaries", []),
        "quadrant_ii_lake_islands": [
            feature["properties"]["island_id"] for feature in lake_islands
        ],
        "quadrant_ii_lake_island_creeks": [
            feature["properties"]["creek_id"] for feature in lake_island_creeks
        ],
        "lake_candidates": len(lakes),
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
        "seasonal_river_trace_source": "official_map/Administrative Division Map of Donh 2026-river-super.png",
        "seasonal_river_trace_method": "centerline_through_centers_of_official_12x_dash_strokes",
        "seasonal_river_trace_coordinate_space": "input_map.png image pixels",
        "cross_border_input_river_segments": original_cross_border_count,
        "cross_border_river_segments": len(cross_border),
        "seasonal_dash_residual_segments_removed": seasonal_dash_residual_count,
        "seasonal_dash_residual_segment_ids_removed": sorted(SEASONAL_DASH_RESIDUAL_SEGMENT_IDS),
        "neighboring_lake_features": sum(
            feature.get("properties", {}).get("territory") == "neighboring_country"
            for feature in lakes
        ),
        "neighboring_lake_ids": sorted(
            int(feature["properties"]["lake_id"])
            for feature in lakes
            if feature.get("properties", {}).get("territory") == "neighboring_country"
        ),
        "cross_border_lake_fill_segments_removed": len(lake_fill_segment_ids_removed),
        "cross_border_lake_fill_segment_ids_removed": lake_fill_segment_ids_removed,
        "cross_border_lake_fill_segments_clipped": len(lake_fill_segment_ids_clipped),
        "cross_border_lake_fill_segment_ids_clipped": lake_fill_segment_ids_clipped,
        "cross_border_lake_fill_line_length_removed_px": round(lake_fill_line_length_removed, 3),
        "cross_border_lake_edge_residual_segments_removed": lake_edge_residual_count,
        "cross_border_lake_edge_residual_segment_ids_removed": lake_edge_residual_ids_removed,
        "national_seasonal_river_candidates": 0,
    })
    label_features = read_fc(DATA / "region_code_labels.geojson")["features"]
    manifest["region_code_label_instances"] = len(label_features)
    manifest["unique_region_codes"] = len({f["properties"].get("region_code") for f in label_features})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"Kept {len(river_features)} regular river segments; separated {manifest['canal_centerlines']} canals "
        f"across {manifest['canal_centerline_reaches']} reaches with {manifest['canal_ruler_ticks']} ruler ticks; "
        f"removed {seasonal_dash_residual_count} duplicated seasonal dashes, "
        f"clipped {len(lake_fill_segment_ids_clipped)} river traces at neighboring lake shores, "
        f"removed {len(lake_fill_segment_ids_removed)} lake-interior fragments and {lake_edge_residual_count} lake-edge fragment; "
        f"traced {len(seasonal)} neighbor seasonal rivers."
    )


if __name__ == "__main__":
    main()
