#!/usr/bin/env python3
"""Convert Duniverse image-space coordinates to its custom Saturn map frame."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "data" / "georeference.json"


def load_reference() -> dict[str, Any]:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def authalic_q(phi_rad: float, eccentricity: float, eccentricity_squared: float) -> float:
    sine = math.sin(phi_rad)
    denominator = 1.0 - eccentricity_squared * sine * sine
    return (1.0 - eccentricity_squared) * (
        sine / denominator
        - math.log((1.0 - eccentricity * sine) / (1.0 + eccentricity * sine))
        / (2.0 * eccentricity)
    )


def inverse_authalic_q(target_q: float, eccentricity: float, eccentricity_squared: float) -> float:
    low = -math.pi / 2.0
    high = math.pi / 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if authalic_q(middle, eccentricity, eccentricity_squared) < target_q:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def pixel_to_lonlat(x_px: float, y_px: float, reference: dict[str, Any]) -> tuple[float, float]:
    projection = reference["projection"]
    ellipsoid = reference["reference_surface"]
    anchor_x, anchor_y = projection["anchor_pixel_xy"]
    semi_major = float(ellipsoid["equatorial_radius_km"])
    eccentricity_squared = float(ellipsoid["eccentricity_squared"])
    eccentricity = math.sqrt(eccentricity_squared)
    scale = float(projection["pixel_scale_km_per_px"])

    phi0 = math.radians(float(projection["anchor_physical_planetographic_latitude_deg"]))
    phi_ts = math.radians(float(projection["standard_parallel_physical_planetographic_deg"]))
    cos_ts = math.cos(phi_ts)
    east_km = (float(x_px) - float(anchor_x)) * scale
    north_km = (float(anchor_y) - float(y_px)) * scale

    lon0 = math.radians(float(projection["anchor_longitude_deg"]))
    longitude = lon0 + east_km / (semi_major * cos_ts)
    y0 = semi_major * authalic_q(phi0, eccentricity, eccentricity_squared) / (2.0 * cos_ts)
    q_target = 2.0 * cos_ts * (y0 + north_km) / semi_major
    latitude = inverse_authalic_q(q_target, eccentricity, eccentricity_squared)

    longitude_deg = (math.degrees(longitude) + 180.0) % 360.0 - 180.0
    physical_latitude_deg = math.degrees(latitude)
    map_latitude_deg = physical_latitude_deg / float(
        reference["coordinate_system"]["physical_planetographic_degrees_per_map_degree"]
    )
    return longitude_deg, map_latitude_deg


def transform_coordinate_tree(value: Any, reference: dict[str, Any]) -> Any:
    if not isinstance(value, list):
        return value
    if len(value) >= 2 and all(isinstance(part, (int, float)) for part in value[:2]):
        longitude, latitude = pixel_to_lonlat(value[0], value[1], reference)
        return [longitude, latitude, *value[2:]]
    return [transform_coordinate_tree(part, reference) for part in value]


def transform_geometry(geometry: dict[str, Any] | None, reference: dict[str, Any]) -> None:
    if not geometry:
        return
    geometry.pop("bbox", None)
    if geometry.get("type") == "GeometryCollection":
        for child in geometry.get("geometries", []):
            transform_geometry(child, reference)
    elif "coordinates" in geometry:
        geometry["coordinates"] = transform_coordinate_tree(geometry["coordinates"], reference)


def remove_pixel_bboxes(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("bbox", None)
        for child in value.values():
            remove_pixel_bboxes(child)
    elif isinstance(value, list):
        for child in value:
            remove_pixel_bboxes(child)


def convert_geojson(document: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(document)
    remove_pixel_bboxes(converted)
    source_note = converted.pop("coordinate_reference_note", None)
    if source_note:
        converted["source_coordinate_reference_note"] = source_note
    if converted.get("type") == "FeatureCollection":
        for feature in converted.get("features", []):
            transform_geometry(feature.get("geometry"), reference)
    elif converted.get("type") == "Feature":
        transform_geometry(converted.get("geometry"), reference)
    else:
        transform_geometry(converted, reference)

    converted["coordinate_reference_note"] = "Geometry coordinates use the custom Saturn reference in data/georeference.json; not WGS 84 / EPSG:4326."
    converted["coordinate_reference"] = reference_descriptor(reference)
    return converted


def reference_descriptor(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "identifier": reference["identifier"],
        "body": reference["body"],
        "latitude_type": reference["coordinate_system"]["latitude_type"],
        "physical_planetographic_degrees_per_map_degree": reference["coordinate_system"]["physical_planetographic_degrees_per_map_degree"],
        "atlas_latitude_bounds_deg": reference["coordinate_system"]["atlas_latitude_bounds_deg"],
        "longitude_positive_direction": reference["coordinate_system"]["longitude_positive_direction"],
        "prime_meridian": reference["coordinate_system"]["prime_meridian"],
        "reference_file": "data/georeference.json",
        "warning": "Custom Saturn coordinates; these values are not WGS 84 / EPSG:4326."
    }


def convert_svg_path_document(document: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Convert the project's M/L/Z ocean outline path into custom lon/lat rings."""
    converted = copy.deepcopy(document)
    source_path = converted.pop("path")
    tokens = re.findall(r"[MLZ]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", source_path)
    rings: list[list[list[float]]] = []
    ring: list[list[float]] = []
    index = 0
    while index < len(tokens):
        command = tokens[index]
        index += 1
        if command == "Z":
            if ring:
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                rings.append(ring)
                ring = []
            continue
        if command not in {"M", "L"} or index + 1 >= len(tokens):
            raise ValueError(f"Unsupported or malformed SVG path command near token {index - 1}")
        x_px = float(tokens[index])
        y_px = float(tokens[index + 1])
        index += 2
        lon, lat = pixel_to_lonlat(x_px, y_px, reference)
        point = [lon, lat]
        if command == "M" and ring:
            rings.append(ring)
            ring = []
        ring.append(point)
    if ring:
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        rings.append(ring)

    converted["type"] = "tiled_official_ocean_surface_path_saturn"
    converted["source_pixel_path"] = source_path
    converted["paths_lonlat"] = rings
    converted["coordinate_reference"] = reference_descriptor(reference)
    return converted


def convert_city_directory(source_path: Path, reference: dict[str, Any]) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    assignment = source.find("window.DUNIVERSE_CITY_DIRECTORY")
    equals = source.find("=", assignment)
    start = source.find("[", equals)
    records, _ = json.JSONDecoder().raw_decode(source[start:])
    for record in records:
        x_px = record.get("pointX")
        y_px = record.get("pointY")
        if isinstance(x_px, (int, float)) and isinstance(y_px, (int, float)):
            longitude, latitude = pixel_to_lonlat(x_px, y_px, reference)
            record["saturnCoordinates"] = {
                "longitudeDeg": longitude,
                "mapLatitudeDeg": latitude,
                "physicalPlanetographicLatitudeDeg": latitude * float(
                    reference["coordinate_system"]["physical_planetographic_degrees_per_map_degree"]
                )
            }
        else:
            record["saturnCoordinates"] = None
    return {
        "type": "DuniverseCityDirectory",
        "source_file": "data/city_directory.js",
        "coordinate_reference": reference_descriptor(reference),
        "records": records
    }


def convert_all(input_dir: Path, output_dir: Path, reference: dict[str, Any]) -> int:
    files = sorted(input_dir.glob("*.geojson"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        document = json.loads(source.read_text(encoding="utf-8"))
        converted = convert_geojson(document, reference)
        destination = output_dir / f"{source.stem}.saturn.json"
        destination.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    ocean_path = input_dir / "ocean_surface_path.json"
    if ocean_path.exists():
        document = json.loads(ocean_path.read_text(encoding="utf-8"))
        converted = convert_svg_path_document(document, reference)
        destination = output_dir / "ocean_surface_path.saturn.json"
        destination.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        files.append(ocean_path)

    city_directory = input_dir / "city_directory.js"
    if city_directory.exists():
        converted = convert_city_directory(city_directory, reference)
        destination = output_dir / "city_directory.saturn.json"
        destination.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        files.append(city_directory)

    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", nargs=2, type=float, metavar=("X_PX", "Y_PX"), help="convert one image-space point")
    parser.add_argument("--geojson", type=Path, help="convert one image-space GeoJSON file")
    parser.add_argument("--output", type=Path, help="output path for --geojson; use .saturn.json to keep the custom CRS explicit")
    parser.add_argument("--all", action="store_true", help="convert all spatial files in data/ (GeoJSON, ocean path, city directory)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "georeferenced", help="output directory for --all")
    args = parser.parse_args()
    reference = load_reference()

    if args.point:
        longitude, latitude = pixel_to_lonlat(args.point[0], args.point[1], reference)
        latitude_scale = float(reference["coordinate_system"]["physical_planetographic_degrees_per_map_degree"])
        print(json.dumps({"pixel_xy": args.point, "longitude_deg": longitude, "map_latitude_deg": latitude,
                          "physical_planetographic_latitude_deg": latitude * latitude_scale,
                          "coordinate_reference": reference["identifier"]}, ensure_ascii=False, indent=2))
        return

    if args.all:
        count = convert_all(ROOT / "data", args.output_dir, reference)
        print(f"Converted {count} spatial datasets to {args.output_dir}")
        return

    if args.geojson:
        if not args.output:
            parser.error("--geojson requires --output")
        document = json.loads(args.geojson.read_text(encoding="utf-8"))
        converted = convert_geojson(document, reference)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote custom Saturn coordinates to {args.output}")
        return

    parser.error("choose --point, --geojson, or --all")


if __name__ == "__main__":
    main()
