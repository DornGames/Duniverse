#!/usr/bin/env python3
"""Build browser-ready city coordinates and administrative extents."""

from __future__ import annotations

import ast
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import georeference_map  # noqa: E402


def read_js_array(path: Path, marker: str) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    start = source.find("[", source.find(marker))
    if start < 0:
        raise ValueError(f"Could not find array assigned to {marker} in {path}")
    value, _ = json.JSONDecoder().raw_decode(source[start:])
    return value


def iter_vertices(coordinates: Any) -> Iterator[tuple[float, float]]:
    if (
        isinstance(coordinates, list)
        and len(coordinates) >= 2
        and isinstance(coordinates[0], (int, float))
        and isinstance(coordinates[1], (int, float))
    ):
        yield float(coordinates[0]), float(coordinates[1])
        return
    if isinstance(coordinates, list):
        for part in coordinates:
            yield from iter_vertices(part)


def capital_names_from_map() -> set[str]:
    source = (ROOT / "index.html").read_text(encoding="utf-8")
    match = re.search(
        r'<g\s+id="provinceCapitalLabels"[^>]*>(.*?)</g>', source, re.DOTALL
    )
    if not match:
        raise ValueError("Could not find province capital labels in index.html")
    names = {
        html.unescape(re.sub(r"<[^>]+>", "", label)).strip()
        for label in re.findall(r"<text\b[^>]*>(.*?)</text>", match.group(1), re.DOTALL)
    }
    names.discard("")
    return names


def city_label_names_from_map() -> set[str]:
    source = (ROOT / "index.html").read_text(encoding="utf-8")
    match = re.search(r"const CITY_LABELS=\[", source)
    if not match:
        raise ValueError("Could not find CITY_LABELS in index.html")
    end = source.find("];", match.end())
    if end < 0:
        raise ValueError("Could not parse CITY_LABELS in index.html")
    labels = ast.literal_eval(source[match.end() - 1 : end + 1])
    return {str(label[5]) for label in labels}


def transformed_point(x: float, y: float, reference: dict[str, Any]) -> dict[str, Any]:
    longitude, latitude = georeference_map.pixel_to_lonlat(x, y, reference)
    return {
        "longitudeDeg": longitude,
        "mapLatitudeDeg": latitude,
        "sourcePixelXY": [x, y],
    }


def city_point(city: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    point = transformed_point(float(city["pointX"]), float(city["pointY"]), reference)
    return {
        "id": city["id"],
        "name": city["name"],
        "longitudeDeg": point["longitudeDeg"],
        "mapLatitudeDeg": point["mapLatitudeDeg"],
        "sourcePixelXY": point["sourcePixelXY"],
    }


def select_boundary_extremes(
    vertices: list[tuple[float, float, float, float]], reference: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    # Resolve exact boundary ties consistently: E/W choose the northernmost tied
    # vertex; N/S choose the westernmost tied vertex.
    selections = {
        "east": max(vertices, key=lambda p: (p[0], p[1], -p[2], -p[3])),
        "south": max(vertices, key=lambda p: (-p[1], -p[0], -p[2], p[3])),
        "west": max(vertices, key=lambda p: (-p[0], p[1], p[2], -p[3])),
        "north": max(vertices, key=lambda p: (p[1], -p[0], -p[2], p[3])),
    }
    return {
        direction: {
            "longitudeDeg": point[0],
            "mapLatitudeDeg": point[1],
            "sourcePixelXY": [point[2], point[3]],
        }
        for direction, point in selections.items()
    }


def extreme_city(cities: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    selectors = {
        "east": lambda c: (-c["longitudeDeg"], c["name"], c["id"]),
        "south": lambda c: (c["mapLatitudeDeg"], c["name"], c["id"]),
        "west": lambda c: (c["longitudeDeg"], c["name"], c["id"]),
        "north": lambda c: (-c["mapLatitudeDeg"], c["name"], c["id"]),
    }
    city = min(cities, key=selectors[direction])
    return {key: city[key] for key in ("id", "name", "longitudeDeg", "mapLatitudeDeg")}


def write_js(path: Path, global_name: str, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    path.write_bytes(f"window.{global_name}={payload};\n".encode("utf-8"))


def main() -> None:
    reference = json.loads((ROOT / "data/georeference.json").read_text(encoding="utf-8"))
    regions = read_js_array(ROOT / "data/region_directory.js", "DUNIVERSE_REGION_DIRECTORY")
    cities = read_js_array(ROOT / "data/city_directory.js", "DUNIVERSE_CITY_DIRECTORY")
    capital_names = capital_names_from_map()
    city_by_name = {city["name"]: city for city in cities}
    missing_capitals = capital_names - set(city_by_name)
    if missing_capitals:
        raise ValueError(f"Capital labels missing from city directory: {sorted(missing_capitals)}")
    missing_map_cities = city_label_names_from_map() - set(city_by_name)
    if missing_map_cities:
        raise ValueError(f"Map city labels missing from city directory: {sorted(missing_map_cities)}")

    national_capital = reference["coordinate_system"]["prime_meridian"]["capital"]
    national_capital_id = city_by_name[national_capital]["id"]
    province_capital_ids = {city_by_name[name]["id"] for name in capital_names}
    city_catalog = []
    for city in cities:
        point = city_point(city, reference)
        role = (
            "capital"
            if city["id"] == national_capital_id
            else "province-capital"
            if city["id"] in province_capital_ids
            else "city"
        )
        source = "map-label-anchor" if city["id"].startswith("city:map-label:") else "city-directory"
        city_catalog.append({**point, "role": role, "source": source})

    cities_by_province: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for city in city_catalog:
        city_entry = city_by_name[city["name"]]
        cities_by_province[city_entry["parentId"]].append(city)

    feature_collection = json.loads(
        (ROOT / "data/administrative_regions.geojson").read_text(encoding="utf-8")
    )
    features_by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mask_area_by_code: dict[str, float] = defaultdict(float)
    for feature in feature_collection["features"]:
        properties = feature["properties"]
        code = str(properties["region_code"])
        features_by_code[code].append(feature)
        mask_area_by_code[code] += float(properties["area_px2"])

    calibration = reference["calibration"]
    target_country_area = float(calibration["target_country_area_km2"])
    pixel_area = float(calibration["pixel_area_km2"])
    raw_mask_area_px2 = sum(mask_area_by_code.values())
    national_outline_area_px2 = float(calibration["national_outline_area_px2"])
    area_normalization_factor = target_country_area / (raw_mask_area_px2 * pixel_area)
    mask_coverage_ratio = raw_mask_area_px2 / national_outline_area_px2
    expected_reconciliation = calibration.get("administrative_region_area_reconciliation", {})
    for key, calculated in (
        ("raw_mask_area_px2", raw_mask_area_px2),
        ("area_normalization_factor", area_normalization_factor),
        ("mask_coverage_ratio", mask_coverage_ratio),
    ):
        if key in expected_reconciliation and abs(float(expected_reconciliation[key]) - calculated) > 1e-10:
            raise ValueError(f"Georeference calibration mismatch for {key}")

    province_entries = [region for region in regions if region["kind"] == "province"]
    region_catalog = []
    for region in regions:
        if region["kind"] == "province":
            member_provinces = [region]
        else:
            member_provinces = [
                province for province in province_entries if province["parentId"] == region["id"]
            ]

        features = [
            feature
            for province in member_provinces
            for feature in features_by_code.get(str(province["code"]), [])
        ]
        vertices = []
        for feature in features:
            for x, y in iter_vertices(feature["geometry"]["coordinates"]):
                point = transformed_point(x, y, reference)
                vertices.append(
                    (
                        point["longitudeDeg"],
                        point["mapLatitudeDeg"],
                        x,
                        y,
                    )
                )
        if not vertices:
            raise ValueError(f"No boundary vertices found for {region['id']}")

        region_mask_area_px2 = sum(
            mask_area_by_code[str(province["code"])] for province in member_provinces
        )
        normalized_area_km2 = target_country_area * region_mask_area_px2 / raw_mask_area_px2
        normalized_area_rounded = round(normalized_area_km2, 3)
        directory_area = float(region.get("areaKm2") or 0)
        if abs(directory_area - normalized_area_rounded) > 0.001:
            raise ValueError(
                f"Area directory disagrees with normalized mask area for {region['id']}: "
                f"{directory_area} vs {normalized_area_rounded}"
            )

        member_city_ids = {province["id"] for province in member_provinces}
        region_cities = [
            city
            for province_id in member_city_ids
            for city in cities_by_province.get(province_id, [])
        ]
        if not region_cities:
            raise ValueError(f"No directory cities found for {region['id']}")

        boundary = select_boundary_extremes(vertices, reference)
        extrema = {
            direction: {
                "boundaryPoint": boundary[direction],
                "extremeCity": extreme_city(region_cities, direction),
            }
            for direction in ("east", "south", "west", "north")
        }
        region_catalog.append(
            {
                "id": region["id"],
                "name": region["name"],
                "code": region.get("code"),
                "parentId": region.get("parentId"),
                "areaKm2": normalized_area_rounded,
                "normalizedAreaKm2": normalized_area_km2,
                "rawMaskAreaPx2": region_mask_area_px2,
                "rawMaskAreaUnderNationalScaleKm2": region_mask_area_px2 * pixel_area,
                "extrema": extrema,
            }
        )

    metadata = {
        "coordinateReference": reference["identifier"],
        "displayDigits": 2,
        "longitudeDomainDeg": reference["coordinate_system"]["longitude_domain_deg"],
        "latitudeBoundsDeg": reference["coordinate_system"]["atlas_latitude_bounds_deg"],
    }
    write_js(ROOT / "data/city_coordinates.js", "DUNIVERSE_CITY_COORDINATES", {**metadata, "cities": city_catalog})
    region_metadata = {
        **metadata,
        "boundarySource": "data/administrative_regions.geojson (automatic candidate polygons)",
        "areaAttributeSource": "data/region_directory.js",
        "areaReconciliation": {
            "targetCountryAreaKm2": target_country_area,
            "nationalOutlineAreaPx2": national_outline_area_px2,
            "rawMaskAreaPx2": raw_mask_area_px2,
            "rawMaskAreaUnderNationalScaleKm2": raw_mask_area_px2 * pixel_area,
            "maskCoverageRatio": mask_coverage_ratio,
            "areaNormalizationFactor": area_normalization_factor,
            "allocationFormula": "A_i = target_country_area_km2 * province_area_px2_i / sum(province_area_px2)",
        },
        "tieBreakRule": "E/W choose the northernmost tied vertex; N/S choose the westernmost tied vertex.",
    }
    write_js(ROOT / "data/region_extrema.js", "DUNIVERSE_REGION_EXTREMA", {**region_metadata, "regions": region_catalog})
    print(
        f"Wrote {len(city_catalog)} city coordinate records and "
        f"{len(region_catalog)} region extrema records."
    )


if __name__ == "__main__":
    main()
