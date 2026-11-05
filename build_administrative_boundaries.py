#!/usr/bin/env python3
"""Build shared province boundaries and a roster-based major-region overlay.

The source map stores provinces as separate filled areas. Their contours are
therefore offset on opposite sides of the printed border and, around rivers,
on opposite riverbanks. This builder assigns each pixel inside the national
outline to its nearest province polygon and vectorizes label transitions once.
Water boundaries are consequently placed midway between the adjoining regions.

Six lake crossings were visually checked against input_map.png and are made
straight chords: lakes 2, 3, 4, 5, 6, and 10. The other extracted lakes sit
inside a single region and do not receive an administrative line. National
and province borders are clipped back from detailed coastline paths so the
land and sea fills define the shore without a separate outline stroke.
Major-region boundaries are shared province-boundary segments whose codes
belong to different groups in data/major_region_membership.json.
"""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Point, shape
from shapely.ops import linemerge, nearest_points, unary_union


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1448
STRAIGHTENED_LAKE_IDS = {2, 3, 4, 5, 6, 10}
LAKE_BUFFER_PX = 2.5
LINE_SIMPLIFY_PX = 0.8
MIN_LINE_LENGTH_PX = 3.0
# The source sea-color contour is offset slightly from the center of the
# printed shoreline. Leave a wider landward clearance so coast and national
# outline never read as two parallel boundary lines.
COASTLINE_CLIP_BUFFER_PX = 7.0
COAST_JUNCTION_MAX_GAP_PX = COASTLINE_CLIP_BUFFER_PX * 2


def read_features(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def load_major_region_membership() -> tuple[
    dict[str, str], dict[str, str], dict[str, str]
]:
    source = json.loads(
        (DATA / "major_region_membership.json").read_text(encoding="utf-8")
    )
    membership = {}
    names = {}
    tongue_names = {}
    for region in source["regions"]:
        region_id = region["id"]
        names[region_id] = region["name"]
        tongue_names[region_id] = region["tongue_name"]
        for code in region["codes"]:
            if code in membership:
                raise RuntimeError(f"Region code {code} belongs to multiple major regions")
            membership[code] = region_id
    return membership, names, tongue_names


def build_major_region_boundaries(
    province_boundaries: list[dict],
    membership: dict[str, str],
    major_region_names: dict[str, str],
) -> list[dict]:
    major_boundaries = []
    for feature in province_boundaries:
        properties = feature["properties"]
        region_a = properties["region_a"]
        region_b = properties["region_b"]
        if region_a not in membership or region_b not in membership:
            missing = sorted({region_a, region_b} - membership.keys())
            raise RuntimeError(
                "Missing major-region assignments for region codes: "
                + ", ".join(missing)
            )
        major_a = membership[region_a]
        major_b = membership[region_b]
        if major_a == major_b:
            continue

        major_properties = dict(properties)
        major_properties.update(
            {
                "boundary_type": "shared_macro_region_boundary",
                "major_region_a": major_a,
                "major_region_b": major_b,
                "major_region_name_a": major_region_names[major_a],
                "major_region_name_b": major_region_names[major_b],
            }
        )
        major_boundaries.append(
            {
                "type": "Feature",
                "properties": major_properties,
                "geometry": feature["geometry"],
            }
        )
    return major_boundaries


def build_major_region_label_anchors(
    region_features: list[dict],
    membership: dict[str, str],
    major_region_names: dict[str, str],
    major_region_tongue_names: dict[str, str],
) -> list[dict]:
    geometries_by_code: dict[str, list] = defaultdict(list)
    for feature in region_features:
        code = feature["properties"].get("region_code", "")
        geometries_by_code[code].append(shape(feature["geometry"]))

    geometries_by_region: dict[str, list] = defaultdict(list)
    missing_codes = []
    for code, major_region_id in membership.items():
        geometries = geometries_by_code.get(code)
        if not geometries:
            missing_codes.append(code)
            continue
        geometries_by_region[major_region_id].extend(geometries)
    if missing_codes:
        raise RuntimeError(
            "Cannot place major-region labels; map geometry is missing for codes: "
            + ", ".join(sorted(missing_codes))
        )

    labels = []
    for major_region_id, name in major_region_names.items():
        region_geometry = unary_union(geometries_by_region[major_region_id])
        anchor = region_geometry.centroid
        if not region_geometry.covers(anchor):
            anchor = region_geometry.representative_point()
        labels.append(
            {
                "type": "Feature",
                "properties": {
                    "major_region_id": major_region_id,
                    "major_region_name": name,
                    "major_region_name_tongue": major_region_tongue_names[major_region_id],
                    "source": "major_region_geometry_center",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(anchor.x, 2), round(anchor.y, 2)],
                },
            }
        )
    return labels


def polygon_rings(geometry: dict) -> Iterable[list[list[float]]]:
    if geometry["type"] == "Polygon":
        yield from geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield from polygon
    else:
        raise ValueError(f"Expected polygon geometry, got {geometry['type']}")


def fill_polygon_mask(mask: np.ndarray, geometry: dict, value: int) -> None:
    for ring in polygon_rings(geometry):
        points = np.rint(ring).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], value)


def line_parts(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in line_parts(item)]
    return []


def clip_border_from_coastline(geometry, coastline_exclusion):
    clipped = geometry.difference(coastline_exclusion)
    parts = [part for part in line_parts(clipped) if part.length >= 0.5]
    if not parts:
        return None
    merged = unary_union(parts)
    if merged.geom_type == "LineString":
        return merged
    return linemerge(merged)


def build_national_land_borders(
    outline_features: list[dict],
    coastline_union,
    coastline_exclusion,
):
    features = []
    for feature in outline_features:
        # Offshore outlines are islands: their boundary is coastline, not a
        # land border. Keep the mainland's actual land-border arc only.
        if feature["properties"].get("role") != "mainland":
            continue
        source_line = shape(feature["geometry"]).boundary
        land_line = clip_border_from_coastline(source_line, coastline_exclusion)
        if land_line is None:
            continue
        land_parts = line_parts(land_line)
        if not land_parts:
            continue
        # The coastline contour and the printed national outline differ by a
        # few pixels. Clipping can therefore leave tiny coastal fragments;
        # the longest remaining arc is the continuous inland national border.
        land_line = max(land_parts, key=lambda part: part.length)
        coordinates = list(land_line.coords)
        junctions = 0
        for at_start in (True, False):
            endpoint = Point(coordinates[0] if at_start else coordinates[-1])
            _, coast_point = nearest_points(endpoint, coastline_union)
            if endpoint.distance(coast_point) > COAST_JUNCTION_MAX_GAP_PX:
                continue
            coast_coordinate = (coast_point.x, coast_point.y)
            if at_start:
                coordinates.insert(0, coast_coordinate)
            else:
                coordinates.append(coast_coordinate)
            junctions += 1
        land_line = LineString(coordinates)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "outline_id": feature["properties"].get("outline_id"),
                    "role": feature["properties"].get("role"),
                    "source": "mainland_border_clipped_and_joined_to_coastline",
                    "coast_junctions_connected": junctions,
                },
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[round(x, 2), round(y, 2)] for x, y in line.coords]
                        for line in line_parts(land_line)
                    ],
                },
            }
        )
    return features


def build_labels(region_features: list[dict], outline_features: list[dict]):
    codes = sorted({feature["properties"]["region_code"] for feature in region_features})
    code_to_id = {code: index + 1 for index, code in enumerate(codes)}
    labels = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    for feature in region_features:
        fill_polygon_mask(
            labels,
            feature["geometry"],
            code_to_id[feature["properties"]["region_code"]],
        )

    inside = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    for feature in outline_features:
        fill_polygon_mask(inside, feature["geometry"], 1)

    missing = (inside > 0) & (labels == 0)
    _, nearest_indices = distance_transform_edt(labels == 0, return_indices=True)
    nearest_labels = labels[nearest_indices[0], nearest_indices[1]]
    labels[missing] = nearest_labels[missing]
    return labels, inside, codes, int(missing.sum())


def collect_boundary_edges(labels: np.ndarray, inside: np.ndarray, codes: list[str]):
    edge_segments: dict[tuple[int, int], list[LineString]] = defaultdict(list)

    vertical = (
        (labels[:, :-1] != labels[:, 1:])
        & (labels[:, :-1] > 0)
        & (labels[:, 1:] > 0)
        & (inside[:, :-1] > 0)
        & (inside[:, 1:] > 0)
    )
    ys, xs = np.where(vertical)
    for y, x in zip(ys.tolist(), xs.tolist()):
        pair = tuple(sorted((int(labels[y, x]), int(labels[y, x + 1]))))
        edge_segments[pair].append(LineString([(x + 0.5, y - 0.5), (x + 0.5, y + 0.5)]))

    horizontal = (
        (labels[:-1, :] != labels[1:, :])
        & (labels[:-1, :] > 0)
        & (labels[1:, :] > 0)
        & (inside[:-1, :] > 0)
        & (inside[1:, :] > 0)
    )
    ys, xs = np.where(horizontal)
    for y, x in zip(ys.tolist(), xs.tolist()):
        pair = tuple(sorted((int(labels[y, x]), int(labels[y + 1, x]))))
        edge_segments[pair].append(LineString([(x - 0.5, y + 0.5), (x + 0.5, y + 0.5)]))

    boundaries = {}
    for pair, segments in edge_segments.items():
        merged = linemerge(unary_union(segments))
        boundaries[(codes[pair[0] - 1], codes[pair[1] - 1])] = merged
    return boundaries, sum(len(segments) for segments in edge_segments.values())


def straighten_lake_crossings(geometry, lakes: list[dict]):
    parts = line_parts(geometry)
    straightened_ids: set[int] = set()
    for lake_feature in lakes:
        lake_id = int(lake_feature["properties"]["lake_id"])
        if lake_id not in STRAIGHTENED_LAKE_IDS:
            continue
        water_area = shape(lake_feature["geometry"]).buffer(LAKE_BUFFER_PX)
        next_parts: list[LineString] = []
        for line in parts:
            intersections = line_parts(line.intersection(water_area))
            crossings = [part for part in intersections if part.length >= MIN_LINE_LENGTH_PX]
            if not crossings:
                next_parts.append(line)
                continue

            next_parts.extend(line_parts(line.difference(water_area)))
            for crossing in crossings:
                start, end = crossing.coords[0], crossing.coords[-1]
                if crossing.is_ring or np.hypot(end[0] - start[0], end[1] - start[1]) < 1:
                    next_parts.append(crossing)
                    continue
                next_parts.append(LineString([start, end]))
                straightened_ids.add(lake_id)

            # Preserve very short contacts that were removed by the lake buffer.
            for contact in intersections:
                if contact not in crossings:
                    next_parts.append(contact)
        parts = next_parts

    simplified = []
    for line in parts:
        line = line.simplify(LINE_SIMPLIFY_PX, preserve_topology=False)
        if not line.is_empty and line.length >= MIN_LINE_LENGTH_PX:
            simplified.append(line)
    if not simplified:
        return None, sorted(straightened_ids)
    merged = unary_union(simplified)
    if merged.geom_type == "LineString":
        return merged, sorted(straightened_ids)
    return linemerge(merged), sorted(straightened_ids)


def geometry_to_path_data(geometry) -> str:
    commands = []
    for line in line_parts(geometry):
        coordinates = list(line.coords)
        if len(coordinates) < 2:
            continue
        commands.append(
            "M "
            + f"{coordinates[0][0]:.2f},{coordinates[0][1]:.2f}"
            + " L "
            + " L ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates[1:])
        )
    return " ".join(commands)


def path_tag(properties: dict, path_data: str, *, standalone: bool) -> str:
    pair = f"{properties['region_a']}-{properties['region_b']}"
    major_pair = properties.get("boundary_type") == "shared_macro_region_boundary"
    pair_attributes = (
        f'data-region-pair="{pair}" data-major-region-pair="{pair}"'
        if major_pair
        else f'data-region-pair="{pair}"'
    )
    if standalone:
        return f'<path {pair_attributes} d="{path_data}"/>'
    return f'<path {pair_attributes} d="{path_data}"/>'


def build_region_label_anchors(
    region_features: list[dict], label_features: list[dict]
) -> list[dict]:
    regions_by_code: dict[str, list] = defaultdict(list)
    regions_by_component: dict[str, list[tuple[str | None, object, int | None]]] = defaultdict(list)
    for feature in region_features:
        code = feature["properties"].get("region_code")
        geometry = shape(feature["geometry"])
        if code:
            regions_by_code[code].append((geometry, feature["properties"].get("component_id")))
        component_id = feature["properties"].get("component_id")
        if component_id is not None:
            regions_by_component[str(component_id)].append((code, geometry, component_id))

    anchors = []
    sw_added = False
    for feature in label_features:
        code = feature["properties"].get("region_code")
        if code == "SW":
            if sw_added:
                continue
            sw_candidates = regions_by_code.get("SW", [])
            if not sw_candidates:
                continue
            region, component_id = max(sw_candidates, key=lambda item: item[0].area)
            sw_added = True
            matched_component_ids = [component_id] if component_id is not None else []
        else:
            component_ids = feature["properties"].get("matched_component_ids", [])
            matches = [
                (geometry, matched_component_id)
                for component_id in component_ids
                for matched_code, geometry, matched_component_id in regions_by_component.get(str(component_id), [])
                if matched_code == code
            ]
            if not matches:
                matches = regions_by_code.get(code, [])
            if not matches:
                continue
            region = unary_union([geometry for geometry, _ in matches])
            matched_component_ids = [component_id for _, component_id in matches if component_id is not None]
        center = region.centroid
        if not region.covers(center):
            center = region.representative_point()
        anchors.append(
            {
                "type": "Feature",
                "properties": {
                    **feature["properties"],
                    "matched_component_ids": matched_component_ids,
                    "match_distance_px": 0.0 if code == "SW" else feature["properties"].get("match_distance_px", 0.0),
                    "transcription_method": (
                        "largest_component_geometric_center"
                        if code == "SW"
                        else "unioned_region_geometric_center"
                    ),
                    "review_status": "derived_from_region_geometry",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(center.x, 2), round(center.y, 2)],
                },
            }
        )
    return anchors


def replace_group(source: str, pattern: str, replacement: str, *, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Could not replace {label} group (matches={count})")
    return updated


def write_geojson(filename: str, name: str, features: list[dict]) -> None:
    collection = {
        "type": "FeatureCollection",
        "name": name,
        "coordinate_system": "source image pixels; not georeferenced",
        "features": features,
    }
    (DATA / filename).write_text(
        json.dumps(collection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_label_groups(
    label_features: list[dict],
    province_names: dict[str, str],
    province_tongue_names: dict[str, str],
) -> tuple[str, str, str, str, str]:
    canvas_code_tags = []
    canvas_name_tags = []
    canvas_tongue_tags = []
    preview_code_tags = []
    preview_name_tags = []
    missing_codes = set()
    missing_tongue_codes = set()

    for feature in label_features:
        code = feature["properties"].get("region_code", "")
        name = province_names.get(code)
        tongue_name = province_tongue_names.get(code)
        if not name or not tongue_name:
            if not name:
                missing_codes.add(code or "<empty>")
            if not tongue_name:
                missing_tongue_codes.add(code or "<empty>")
            continue

        x, y = feature["geometry"]["coordinates"]
        safe_code = html.escape(code, quote=True)
        safe_name = html.escape(name, quote=True)
        safe_tongue_name = html.escape(tongue_name, quote=True)
        safe_name_with_code = html.escape(f"{name}（{code}）", quote=True)
        canvas_code_tags.append(
            f'<text x="{x:g}" y="{y:g}" data-pair-text="{safe_name}" '
            f'dominant-baseline="central" stroke="#fff" stroke-width=".7" '
            f'paint-order="stroke">{safe_code}</text>'
        )
        canvas_name_tags.append(
            f'<text x="{x:g}" y="{y:g}" data-pair-text="{safe_code}" '
            f'dominant-baseline="central" stroke="#fff" stroke-width=".6" '
            f'paint-order="stroke">{safe_name}</text>'
        )
        canvas_tongue_tags.append(
            f'<text x="{x:g}" y="{y:g}" data-language="tongue" '
            f'dominant-baseline="central" stroke="#fff" stroke-width=".55" '
            f'paint-order="stroke">{safe_tongue_name}</text>'
        )

        preview_name_tags.append(
            f'<text x="{x:g}" y="{y - 6:g}" font-size="11" font-weight="600" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'stroke="#fff" stroke-width=".6" paint-order="stroke">'
            f'{safe_name_with_code}</text>'
        )
        preview_name_tags.append(
            f'<text x="{x:g}" y="{y + 6:g}" font-size="10" font-weight="600" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'stroke="#fff" stroke-width=".55" paint-order="stroke">'
            f'{safe_tongue_name}</text>'
        )

    if missing_codes or missing_tongue_codes:
        errors = []
        if missing_codes:
            errors.append(
                "missing Chinese names for codes: " + ", ".join(sorted(missing_codes))
            )
        if missing_tongue_codes:
            errors.append(
                "missing Tongue names for codes: "
                + ", ".join(sorted(missing_tongue_codes))
            )
        raise RuntimeError(
            "Missing administrative label names; " + "; ".join(errors)
        )

    canvas_codes = (
        '<g id="regionCodes" data-layer="regionCodes" pointer-events="none" '
        'font-family="Arial,sans-serif" font-size="21" font-weight="600" '
        'text-anchor="middle" fill="#00a98e" fill-opacity=".78">'
        + "".join(canvas_code_tags)
        + "</g>"
    )
    canvas_names = (
        '<g id="regionNames" data-layer="regionNames" pointer-events="none" '
        'font-family="Arial,sans-serif" font-size="11" font-weight="600" '
        'text-anchor="middle" fill="#168d7b" fill-opacity=".9">'
        + "".join(canvas_name_tags)
        + "</g>"
    )
    canvas_tongue_names = (
        '<g id="regionTongueNames" data-layer="regionTongueNames" '
        'pointer-events="none" font-family="Arial,sans-serif" font-size="10" '
        'font-weight="600" text-anchor="middle" fill="#168d7b" fill-opacity=".9">'
        + "".join(canvas_tongue_tags)
        + "</g>"
    )
    preview_codes = (
        '<g id="region-codes" pointer-events="none" '
        'font-family="Arial,sans-serif" font-size="21" font-weight="600" '
        'fill="#00a98e" fill-opacity=".78">'
        + "".join(preview_code_tags)
        + "</g>"
    )
    preview_names = (
        '<g id="region-names" pointer-events="none" '
        'font-family="Arial,sans-serif" font-size="11" font-weight="600" '
        'fill="#168d7b" fill-opacity=".9">'
        + "".join(preview_name_tags)
        + "</g>"
    )
    return canvas_codes, canvas_names, canvas_tongue_names, preview_codes, preview_names


def build_province_capital_tongue_group(
    source: str, province_capital_tongue_names: dict[str, str]
) -> str:
    match = re.search(
        r'<g id="provinceCapitalLabels"[^>]*>(.*?)</g>', source, flags=re.DOTALL
    )
    if not match:
        raise RuntimeError("Could not find the province-capital Chinese labels")

    tags = []
    missing_names = set()
    for attributes, text in re.findall(
        r'<text\b([^>]*)>(.*?)</text>', match.group(1), flags=re.DOTALL
    ):
        name = html.unescape(text).strip()
        tongue_name = province_capital_tongue_names.get(name)
        if not tongue_name:
            missing_names.add(name or "<empty>")
            continue
        values = {}
        for attribute in ("x", "y", "text-anchor"):
            attribute_match = re.search(
                rf'\b{re.escape(attribute)}="([^"]+)"', attributes
            )
            if attribute_match:
                values[attribute] = attribute_match.group(1)
        if not all(attribute in values for attribute in ("x", "y", "text-anchor")):
            raise RuntimeError(f"Province-capital label is missing position/alignment: {name}")
        safe_tongue_name = html.escape(tongue_name, quote=True)
        tags.append(
            f'<text x="{values["x"]}" y="{values["y"]}" '
            f'text-anchor="{values["text-anchor"]}" data-language="tongue" '
            f'dominant-baseline="central" stroke="#fff" stroke-width="1.2" '
            f'paint-order="stroke">{safe_tongue_name}</text>'
        )

    if missing_names:
        raise RuntimeError(
            "Missing Tongue names for province-capital labels: "
            + ", ".join(sorted(missing_names))
        )

    return (
        '<g id="provinceCapitalTongueLabels" '
        'data-layer="provinceCapitalTongueLabels" pointer-events="none" '
        'font-family="Arial,sans-serif" font-size="10" font-weight="600" '
        'fill="#c32631">'
        + "".join(tags)
        + "</g>"
    )


def update_province_capital_tongue_group(
    source: str, province_capital_tongue_names: dict[str, str]
) -> str:
    group = build_province_capital_tongue_group(source, province_capital_tongue_names)
    if re.search(r'<g id="provinceCapitalTongueLabels"(?=[ >])', source):
        return replace_group(
            source,
            r'<g id="provinceCapitalTongueLabels"[^>]*>.*?</g>',
            group,
            label="Canvas province-capital Tongue names",
        )

    updated, count = re.subn(
        r'(<g id="provinceCapitalLabels"[^>]*>.*?</g>)',
        lambda match: match.group(1) + group,
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Could not insert Canvas province-capital Tongue names")
    return updated


def update_city_tongue_names_constant(
    source: str, city_tongue_names: dict[str, str]
) -> str:
    declaration = (
        "const CITY_TONGUE_NAMES="
        + json.dumps(city_tongue_names, ensure_ascii=False, indent=2)
        + ";"
    )
    updated, _ = re.subn(
        r"const CITY_TONGUE_NAMES\s*=\s*[^;]*;",
        "",
        source,
        flags=re.DOTALL,
    )
    updated, count = re.subn(
        r"(?=const CITY_LABELS=\[)",
        lambda _: declaration + "\n",
        updated,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not replace the Canvas city Tongue-name mapping")
    return updated


def move_group_after(source: str, moving_id: str, target_id: str) -> str:
    moving_pattern = rf'<g id="{re.escape(moving_id)}"[^>]*>.*?</g>'
    moving_match = re.search(moving_pattern, source, flags=re.DOTALL)
    if not moving_match:
        raise RuntimeError(f"Could not find SVG group {moving_id}")
    moving_group = moving_match.group(0)
    without_moving = source[: moving_match.start()] + source[moving_match.end() :]
    target_pattern = rf'<g id="{re.escape(target_id)}"[^>]*>.*?</g>'
    target_match = re.search(target_pattern, without_moving, flags=re.DOTALL)
    if not target_match:
        raise RuntimeError(f"Could not find SVG group {target_id}")
    insertion = target_match.end()
    return without_moving[:insertion] + moving_group + without_moving[insertion:]


def update_render_files(
    features: list[dict],
    major_region_boundaries: list[dict],
    major_region_label_features: list[dict],
    national_land_borders: list[dict],
    neighbor_borders: list[dict],
    label_features: list[dict],
    province_names: dict[str, str],
    province_tongue_names: dict[str, str],
    province_capital_tongue_names: dict[str, str],
    city_tongue_names: dict[str, str],
    coastlines: list[dict],
    ocean_surface_path: str,
) -> None:
    standalone_tags = []
    canvas_tags = []
    for feature in features:
        path_data = geometry_to_path_data(shape(feature["geometry"]))
        if not path_data:
            continue
        props = feature["properties"]
        standalone_tags.append(path_tag(props, path_data, standalone=True))
        canvas_tags.append(path_tag(props, path_data, standalone=False))
    major_standalone_tags = []
    major_canvas_tags = []
    for feature in major_region_boundaries:
        path_data = geometry_to_path_data(shape(feature["geometry"]))
        if not path_data:
            continue
        props = feature["properties"]
        major_standalone_tags.append(path_tag(props, path_data, standalone=True))
        major_canvas_tags.append(path_tag(props, path_data, standalone=False))

    standalone_group = (
        '<g id="administrative-regions" fill="none" stroke="#454545" '
        'stroke-width="0.9" stroke-dasharray="5 4" stroke-linecap="round" '
        'stroke-linejoin="round">'
        + "".join(standalone_tags)
        + "</g>"
    )
    major_standalone_group = (
        '<g id="major-region-boundaries" fill="none" stroke="#454545" '
        'stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round">'
        + "".join(major_standalone_tags)
        + "</g>"
    )
    major_name_canvas_tags = []
    major_name_preview_tags = []
    for feature in major_region_label_features:
        x, y = feature["geometry"]["coordinates"]
        name = html.escape(feature["properties"]["major_region_name"], quote=True)
        tongue_name = html.escape(
            feature["properties"]["major_region_name_tongue"], quote=True
        )
        major_name_canvas_tags.append(
            f'<text x="{x:g}" y="{y - 6:g}" data-letter-spacing="8" '
            f'dominant-baseline="central" stroke="#f5f8f9" stroke-width="2" '
            f'paint-order="stroke">{name}</text>'
        )
        major_name_canvas_tags.append(
            f'<text x="{x:g}" y="{y + 9:g}" data-language="tongue" '
            f'font-size="10" font-weight="600" data-letter-spacing="0" '
            f'dominant-baseline="central" stroke="#f5f8f9" stroke-width="0.8" '
            f'paint-order="stroke">{tongue_name}</text>'
        )
        major_name_preview_tags.append(
            f'<text x="{x:g}" y="{y - 6:g}" text-anchor="middle" '
            f'dominant-baseline="central" letter-spacing="8" '
            f'stroke="#f5f8f9" stroke-width="2" paint-order="stroke">{name}</text>'
        )
        major_name_preview_tags.append(
            f'<text x="{x:g}" y="{y + 9:g}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="10" font-weight="600" '
            f'letter-spacing="0" stroke="#f5f8f9" stroke-width="0.8" '
            f'paint-order="stroke">{tongue_name}</text>'
        )
    major_names_canvas_group = (
        '<g id="majorRegionNames" data-layer="majorRegionBoundaries" '
        'pointer-events="none" font-family="Microsoft YaHei,system-ui,sans-serif" '
        'font-size="12" font-weight="600" text-anchor="middle" '
        'fill="#53646b" fill-opacity=".82">'
        + "".join(major_name_canvas_tags)
        + "</g>"
    )
    major_names_preview_group = (
        '<g id="major-region-names" pointer-events="none" '
        'font-family="Microsoft YaHei,system-ui,sans-serif" '
        'font-size="12" font-weight="600" fill="#53646b" fill-opacity=".82">'
        + "".join(major_name_preview_tags)
        + "</g>"
    )
    national_standalone_tags = []
    national_canvas_tags = []
    for feature in national_land_borders:
        path_data = geometry_to_path_data(shape(feature["geometry"]))
        if not path_data:
            continue
        outline_id = feature["properties"].get("outline_id", "")
        national_path_tag = f'<path data-outline-id="{outline_id}" d="{path_data}"/>'
        national_standalone_tags.append(national_path_tag)
        national_canvas_tags.append(national_path_tag)
    national_standalone_group = (
        '<g id="national-border" fill="none" stroke="#293840" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round">'
        + "".join(national_standalone_tags)
        + "</g>"
    )
    neighbor_tags = []
    for feature in neighbor_borders:
        path_data = geometry_to_path_data(shape(feature["geometry"]))
        if path_data:
            segment_id = feature["properties"].get("segment_id", "")
            neighbor_tags.append(
                f'<path data-segment-id="{segment_id}" d="{path_data}"/>'
            )
    neighbor_standalone_group = (
        '<g id="neighbor-borders" fill="none" stroke="#665b56" '
        'stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" '
        'pointer-events="none">'
        + "".join(neighbor_tags)
        + "</g>"
    )
    island_fill_tags = []
    for feature in coastlines:
        if feature["properties"].get("review_status") == "rejected":
            continue
        coast_geometry = shape(feature["geometry"])
        path_data = geometry_to_path_data(coast_geometry)
        if (
            feature["properties"].get("role") == "island_shoreline"
            and coast_geometry.geom_type == "LineString"
            and coast_geometry.is_ring
        ):
            island_fill_tags.append(f'<path d="{path_data}"/>')
    coastline_canvas_group = (
        '<g id="coastline" data-layer="coastline" fill="none" '
        'pointer-events="none"></g>'
    )
    coastline_preview_group = (
        '<g id="coastlines" fill="none" pointer-events="none"></g>'
    )
    (
        canvas_codes_group,
        canvas_names_group,
        canvas_tongue_names_group,
        preview_codes_group,
        preview_names_group,
    ) = build_label_groups(label_features, province_names, province_tongue_names)
    preview_path = ROOT / "vector_preview.svg"
    preview = preview_path.read_text(encoding="utf-8")
    preview = replace_group(
        preview,
        r'<g id="administrative-regions"[^>]*>.*?</g>',
        standalone_group,
        label="standalone administrative-regions",
    )
    if re.search(r'<g id="major-region-boundaries"(?=[ >])', preview):
        preview = replace_group(
            preview,
            r'<g id="major-region-boundaries"[^>]*>.*?</g>',
            major_standalone_group,
            label="standalone major-region-boundaries",
        )
    else:
        preview, major_insert_count = re.subn(
            r'(<g id="administrative-regions"[^>]*>.*?</g>)',
            lambda match: match.group(1) + major_standalone_group,
            preview,
            count=1,
            flags=re.DOTALL,
        )
        if major_insert_count != 1:
            raise RuntimeError("Could not insert the standalone major-region-boundaries group")
    if re.search(r'<g id="major-region-names"(?=[ >])', preview):
        preview = replace_group(
            preview,
            r'<g id="major-region-names"[^>]*>.*?</g>',
            major_names_preview_group,
            label="standalone major-region-names",
        )
    else:
        preview, major_name_insert_count = re.subn(
            r'(<g id="major-region-boundaries"[^>]*>.*?</g>)',
            lambda match: match.group(1) + major_names_preview_group,
            preview,
            count=1,
            flags=re.DOTALL,
        )
        if major_name_insert_count != 1:
            raise RuntimeError("Could not insert the standalone major-region-names group")
    preview = replace_group(
        preview,
        r'<g id="national-border"[^>]*>.*?</g>',
        national_standalone_group,
        label="standalone national-border",
    )
    preview = replace_group(
        preview,
        r'<g id="coastlines"[^>]*>.*?</g>',
        coastline_preview_group,
        label="standalone coastlines",
    )
    preview = replace_group(
        preview,
        r'<g id="neighbor-borders"[^>]*>.*?</g>',
        neighbor_standalone_group,
        label="standalone neighbor-borders",
    )
    preview = replace_group(
        preview,
        r'(?:<g id="region-labels"[^>]*>.*?</g>|'
        r'<g id="region-codes"[^>]*>.*?</g><g id="region-names"[^>]*>.*?</g>)',
        preview_codes_group + preview_names_group,
        label="standalone labels",
    )
    preview = move_group_after(preview, "neighbor-borders", "cross-border-rivers")
    preview = move_group_after(preview, "administrative-regions", "neighbor-borders")
    preview = move_group_after(preview, "major-region-names", "major-region-boundaries")
    preview = move_group_after(preview, "coastlines", "national-border")
    preview_path.write_text(preview, encoding="utf-8", newline="\n")

    canvas_group = (
        '<g id="provinceBoundaries" data-layer="regions" fill="none" '
        'stroke="#454545" stroke-width="0.9" stroke-dasharray="5 4" '
        'stroke-linecap="round" stroke-linejoin="round" pointer-events="none">'
        + "".join(canvas_tags)
        + "</g>"
    )
    major_canvas_group = (
        '<g id="majorRegionBoundaries" data-layer="majorRegionBoundaries" fill="none" '
        'stroke="#454545" stroke-width="1.1" stroke-linecap="round" '
        'stroke-linejoin="round" pointer-events="none">'
        + "".join(major_canvas_tags)
        + "</g>"
    )
    neighbor_canvas_group = (
        '<g id="neighborBorders" data-layer="neighborBorders" fill="none" '
        'stroke="#665b56" stroke-width="2.6" stroke-linecap="round" '
        'stroke-linejoin="round" pointer-events="none">'
        + "".join(neighbor_tags)
        + "</g>"
    )
    index_path = ROOT / "index.html"
    index = index_path.read_text(encoding="utf-8")
    index = re.sub(
        r'<g id="coastlineFills"[^>]*>.*?</g>',
        "",
        index,
        count=1,
        flags=re.DOTALL,
    )
    coastline_fills_group = (
        '<g id="coastlineFills" data-layer="islandFills" fill="#fff" '
        'fill-rule="evenodd" stroke="none" pointer-events="none">'
        + "".join(island_fill_tags)
        + "</g>"
    )
    ocean_fill_tag = (
        f'<path id="oceanFill" d="{ocean_surface_path}" fill="#b1d0ed" '
        'fill-rule="evenodd"/>'
    )
    index, ocean_fill_count = re.subn(
        r'<path id="oceanFill"[^>]*?/>',
        lambda _match: ocean_fill_tag,
        index,
        count=1,
    )
    if ocean_fill_count != 1:
        raise RuntimeError("Could not replace the old oceanFill path with tiled ocean data")
    ocean_fill_match = re.search(r'<path id="oceanFill"[^>]*?/>', index)
    if not ocean_fill_match:
        raise RuntimeError("Could not find the new oceanFill path to place island land fills")
    index = (
        index[: ocean_fill_match.end()]
        + coastline_fills_group
        + index[ocean_fill_match.end() :]
    )
    index = replace_group(
        index,
        r'<g id="provinceBoundaries"[^>]*>.*?</g>',
        "",
        label="old Canvas provinceBoundaries",
    ) if re.search(r'<g id="provinceBoundaries"(?=[ >])', index) else index
    index = replace_group(
        index,
        r'<g id="majorRegionBoundaries"[^>]*>.*?</g>',
        "",
        label="old Canvas majorRegionBoundaries",
    ) if re.search(r'<g id="majorRegionBoundaries"(?=[ >])', index) else index
    index = replace_group(
        index,
        r'<g id="majorRegionNames"[^>]*>.*?</g>',
        "",
        label="old Canvas majorRegionNames",
    ) if re.search(r'<g id="majorRegionNames"(?=[ >])', index) else index
    index = replace_group(
        index,
        r'<g id="neighborBorders"[^>]*>.*?</g>',
        "",
        label="old Canvas neighborBorders",
    ) if re.search(r'<g id="neighborBorders"(?=[ >])', index) else index

    index = replace_group(
        index,
        r'<g id="regions"([^>]*)>',
        lambda match: '<g id="regions"' + re.sub(
            r'\s(?:stroke|stroke-width|stroke-dasharray)="[^"]*"',
            "",
            match.group(1),
        ) + ' stroke="none" stroke-width="0">',
        label="Canvas region hit-test",
    )
    draw_order_replacement = (
        "['provinceBoundaries',9],['majorRegionBoundaries',10],['graticule',10.5],"
        "['majorRegionNames',11],"
        "['nationalBorder',12],['blackMarkers',13],['redMarkers',14],"
        "['specialCapitalMarker',14],['nationalCapitalMarker',14],"
        "['regionCodes',15],['regionNames',16],['regionTongueNames',17],"
        "['provinceCapitalLabels',18],['provinceCapitalTongueLabels',19],"
        "['nationalCapitalLabels',20],['cityLabels',21],['cityTongueLabels',22]"
    )
    draw_order_match = re.search(
        r"\['provinceBoundaries',9\].*?\['regionNames',(?:15|16)\]"
        r"(?:,\['regionTongueNames',17\])?"
        r"(?:,\['provinceCapitalLabels',(?:17|18)\])?"
        r"(?:,\['provinceCapitalTongueLabels',(?:18|19)\])?"
        r"(?:,\['nationalCapitalLabels',(?:18|19|20)\])?"
        r"(?:,\['cityLabels',(?:19|20|21)\])?"
        r"(?:,\['cityTongueLabels',(?:20|21|22)\])?",
        index,
        re.DOTALL,
    )
    if draw_order_match:
        index = (
            index[: draw_order_match.start()]
            + draw_order_replacement
            + index[draw_order_match.end() :]
        )
    elif draw_order_replacement not in index:
        raise RuntimeError("Could not update the Canvas major-region draw order")
    old_radius_expression = r"drawGroup==='provinceBoundaries'\?1\.1:"
    new_radius_expression = (
        "drawGroup==='provinceBoundaries'||drawGroup==='majorRegionBoundaries'?1.1:"
    )
    if re.search(old_radius_expression, index):
        index = re.sub(old_radius_expression, new_radius_expression, index, count=1)
    elif new_radius_expression not in index:
        raise RuntimeError("Could not update the Canvas major-region line detail")
    index = replace_group(
        index,
        r'(<g id="crossBorderRivers"[^>]*>.*?</g>)',
        lambda match: match.group(1) + neighbor_canvas_group + canvas_group
        + major_canvas_group + major_names_canvas_group,
        label="Canvas boundary insertion point",
    )
    national_canvas_group = (
        '<g id="nationalBorder" data-layer="regions" fill="none" stroke="#293840" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" pointer-events="none">'
        + "".join(national_canvas_tags)
        + "</g>"
    )
    index = replace_group(
        index,
        r'<g id="nationalBorder"[^>]*>.*?</g>',
        national_canvas_group,
        label="Canvas national-border",
    )
    index = replace_group(
        index,
        r'<g id="coastline"[^>]*>.*?</g>',
        coastline_canvas_group,
        label="Canvas coastline",
    )
    index = replace_group(
        index,
        r'(?:<g id="labels"[^>]*>.*?</g>|'
        r'<g id="regionCodes"[^>]*>.*?</g><g id="regionNames"[^>]*>.*?</g>'
        r'<g id="regionTongueNames"[^>]*>.*?</g>)',
        canvas_codes_group + canvas_names_group + canvas_tongue_names_group,
        label="Canvas geometric-center labels",
    )
    index = update_province_capital_tongue_group(
        index, province_capital_tongue_names
    )
    index = update_city_tongue_names_constant(index, city_tongue_names)
    index_path.write_text(index, encoding="utf-8", newline="\n")


def main() -> None:
    regions = read_features(DATA / "administrative_regions.geojson")
    outlines = read_features(DATA / "national_outline.geojson")
    lakes = read_features(DATA / "lakes.geojson")
    coastline_candidates = read_features(DATA / "coastlines.geojson")
    coastlines = [
        feature for feature in coastline_candidates
        if feature["properties"].get("review_status") != "rejected"
    ]
    print(f"Loaded {len(coastlines)} coastline features; preparing shared coastline clip.", flush=True)
    neighbor_borders = read_features(DATA / "neighbor_borders.geojson")
    label_candidates = read_features(DATA / "region_code_labels.geojson")
    label_features = build_region_label_anchors(regions, label_candidates)
    province_names = json.loads(
        (DATA / "province_names.json").read_text(encoding="utf-8")
    )
    province_tongue_names = json.loads(
        (DATA / "province_tongue_names.json").read_text(encoding="utf-8")
    )
    province_capital_tongue_names = json.loads(
        (DATA / "province_capital_tongue_names.json").read_text(encoding="utf-8")
    )
    city_tongue_names = json.loads(
        (DATA / "city_tongue_names.json").read_text(encoding="utf-8")
    )
    (
        major_region_membership,
        major_region_names,
        major_region_tongue_names,
    ) = load_major_region_membership()
    unmapped_codes = set(province_names) - set(major_region_membership)
    unknown_codes = set(major_region_membership) - set(province_names)
    if unmapped_codes or unknown_codes:
        raise RuntimeError(
            "Major-region memberships must cover the map's administrative codes; "
            f"unmapped={sorted(unmapped_codes)}, unknown={sorted(unknown_codes)}"
        )
    major_region_label_features = build_major_region_label_anchors(
        regions,
        major_region_membership,
        major_region_names,
        major_region_tongue_names,
    )
    label_codes = {feature["properties"].get("region_code", "") for feature in label_features}
    missing_names = label_codes - province_names.keys()
    if missing_names:
        raise RuntimeError(
            "Missing Chinese administrative names for region codes: "
            + ", ".join(sorted(missing_names))
        )
    missing_tongue_names = label_codes - province_tongue_names.keys()
    if missing_tongue_names:
        raise RuntimeError(
            "Missing Tongue administrative names for region codes: "
            + ", ".join(sorted(missing_tongue_names))
        )
    ocean_surface_path_data = json.loads(
        (DATA / "ocean_surface_path.json").read_text(encoding="utf-8")
    )
    ocean_surface_path = ocean_surface_path_data.get("path", "")
    if not ocean_surface_path:
        raise RuntimeError("Tiled ocean surface path is missing or empty")
    coastline_union = unary_union([
        shape(feature["geometry"])
        for feature in coastlines
        if feature["properties"].get("role") != "island_shoreline"
        if "_island_ink" not in feature["properties"].get("source", "")
    ])
    # The same shoreline buffer is used to clip every shared region border.
    # Build it once; buffering the detailed coastline separately for each of
    # the ~100 province boundaries was needlessly expensive.
    coastline_exclusion = coastline_union.buffer(COASTLINE_CLIP_BUFFER_PX)
    print("Prepared the coastline exclusion buffer.", flush=True)
    labels, inside, codes, filled_pixels = build_labels(regions, outlines)
    boundaries, edge_count = collect_boundary_edges(labels, inside, codes)
    print(f"Derived {len(boundaries)} shared province-boundary networks.", flush=True)

    features = []
    for (region_a, region_b), geometry in sorted(boundaries.items()):
        geometry, straightened_lakes = straighten_lake_crossings(geometry, lakes)
        if geometry is not None:
            geometry = clip_border_from_coastline(geometry, coastline_exclusion)
        if geometry is None:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "region_a": region_a,
                    "region_b": region_b,
                    "boundary_type": "shared_administrative_midline",
                    "source": "nearest_region_label_transition",
                    "straightened_lake_ids": straightened_lakes,
                    "coastline_clipped": True,
                },
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[round(x, 2), round(y, 2)] for x, y in line.coords]
                        for line in line_parts(geometry)
                    ],
                },
            }
        )

    major_region_features = build_major_region_boundaries(
        features,
        major_region_membership,
        major_region_names,
    )
    print(
        f"Derived {len(major_region_features)} shared major-region boundaries from the roster.",
        flush=True,
    )

    national_land_borders = build_national_land_borders(
        outlines,
        coastline_union,
        coastline_exclusion,
    )
    print("Clipped land borders and prepared render layers.", flush=True)
    write_geojson("province_boundaries.geojson", "shared_province_boundaries", features)
    write_geojson(
        "major_region_boundaries.geojson",
        "shared_major_region_boundaries_from_roster",
        major_region_features,
    )
    write_geojson(
        "major_region_labels.geojson",
        "major_region_geometry_center_labels",
        major_region_label_features,
    )
    write_geojson("national_border_land.geojson", "national_border_land_only", national_land_borders)
    write_geojson("region_code_labels.geojson", "region_geometric_center_labels", label_features)
    update_render_files(
        features,
        major_region_features,
        major_region_label_features,
        national_land_borders,
        neighbor_borders,
        label_features,
        province_names,
        province_tongue_names,
        province_capital_tongue_names,
        city_tongue_names,
        coastlines,
        ocean_surface_path,
    )
    print(
        f"Built {len(features)} shared province boundaries from {edge_count} pixel edges; "
        f"filled {filled_pixels} gaps inside the national outline; "
        f"straightened lake crossings: {', '.join(map(str, sorted(STRAIGHTENED_LAKE_IDS)))}; "
        f"clipped {len(national_land_borders)} national outline components against coastlines; "
        f"centered {len(label_features)} province labels; "
        f"excluded {len(coastline_candidates) - len(coastlines)} manually rejected coastline candidates."
    )


if __name__ == "__main__":
    main()
