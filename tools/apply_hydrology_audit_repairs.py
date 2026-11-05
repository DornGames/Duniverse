#!/usr/bin/env python3
"""Apply the manually reviewed 2026-11-04 hydrology audit to project layers.

The audit directory is passed explicitly because the source imagery and review
tiles are kept outside the Git repository. Coordinates are in input-map pixels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_review_decisions(audit_dir: Path) -> dict[str, dict[str, Any]]:
    review_files = [
        audit_dir / "manual_review_RCH-0001-0112.csv",
        audit_dir / "manual_review_center" / "review_RCH-0113_0224.csv",
        audit_dir / "manual_review_RCH-0225-0336.csv",
    ]
    decisions: dict[str, dict[str, Any]] = {}
    for path in review_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing manual review file: {path}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                issue_id = row["issue_id"]
                if "manual_decision" in row:
                    decision = "repair" if row["manual_decision"] == "repair_candidate" else "exclude"
                    basis = row.get("manual_reason", "")
                elif "decision" in row:
                    decision = "repair" if row["decision"] == "\u53ef\u4fee" else "exclude"
                    basis = row.get("round2_basis") or row.get("visual_basis", "")
                else:
                    decision = "repair" if row.get("manual_verdict") == "fix" else "exclude"
                    basis = row.get("manual_basis", "")
                if issue_id in decisions:
                    raise ValueError(f"Duplicate manual decision for {issue_id}")
                decisions[issue_id] = {
                    "issue_id": issue_id,
                    "tile_id": row.get("tile_id", ""),
                    "decision": decision,
                    "review_basis": basis,
                    "review_file": str(path.name),
                }
    expected = {f"RCH-{i:04d}" for i in range(1, 337)}
    if set(decisions) != expected:
        missing = sorted(expected - set(decisions))
        extra = sorted(set(decisions) - expected)
        raise ValueError(f"Manual review coverage mismatch; missing={missing[:8]}, extra={extra[:8]}")
    return decisions


def feature_by_segment(fc: dict[str, Any], feature_id: str) -> dict[str, Any]:
    matches = [
        feature for feature in fc["features"]
        if f"segment-{feature.get('properties', {}).get('segment_id')}" == feature_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {feature_id} in {fc.get('name')}, got {len(matches)}")
    return matches[0]


def snap_feature_endpoint(feature: dict[str, Any], old_xy: list[float], new_xy: list[float]) -> dict[str, Any]:
    geometry = feature["geometry"]
    if geometry["type"] != "LineString":
        raise ValueError("Endpoint repair currently expects a LineString feature")
    coordinates = [list(point) for point in geometry["coordinates"]]
    choices = [(Point(point).distance(Point(old_xy)), index) for index, point in ((0, coordinates[0]), (-1, coordinates[-1]))]
    distance, index = min(choices)
    if distance > 0.01:
        raise ValueError(f"Expected endpoint {old_xy}; closest current endpoint is {distance:.4f}px away")
    original = coordinates[index]
    coordinates[index] = [float(new_xy[0]), float(new_xy[1]), *original[2:]]
    feature["geometry"]["coordinates"] = coordinates
    return {"original_xy": original[:2], "corrected_xy": coordinates[index][:2]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path, help="previously generated hydrology audit directory")
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()

    decisions = load_review_decisions(audit_dir)
    gap_fc = read_json(audit_dir / "river_gap_candidates.geojson")
    reach_fc = read_json(audit_dir / "unmatched_official_reaches.geojson")
    lake_candidate_fc = read_json(audit_dir / "official_lake_candidates.geojson")
    lake_metrics = {
        str(row["lake_id"]): row
        for row in csv.DictReader((audit_dir / "lake_boundary_metrics.csv").open(encoding="utf-8-sig", newline=""))
    }

    river_fc = read_json(DATA / "hydrology_lines.geojson")
    cross_border_fc = read_json(DATA / "cross_border_rivers.geojson")
    lakes_fc = read_json(DATA / "lakes.geojson")
    outline_fc = read_json(DATA / "national_outline.geojson")
    outline = unary_union([shape(feature["geometry"]) for feature in outline_fc["features"]])
    layer_files = {"river": river_fc, "cross_border_river": cross_border_fc}

    gap_log = []
    for candidate in gap_fc["features"]:
        props = candidate["properties"]
        a_xy, b_xy = props["endpoint_a_px_xy"], props["endpoint_b_px_xy"]
        target = [
            (a_xy[0] + b_xy[0]) / 2.0,
            (a_xy[1] + b_xy[1]) / 2.0,
        ]
        endpoints = []
        for feature_key, layer_key, old_xy in (
            ("feature_a", "feature_a_layer", a_xy),
            ("feature_b", "feature_b_layer", b_xy),
        ):
            layer_fc = layer_files[props[layer_key]]
            feature = feature_by_segment(layer_fc, props[feature_key])
            changed = snap_feature_endpoint(feature, old_xy, target)
            audit_ids = feature["properties"].setdefault("audit_gap_issue_ids", [])
            if props["issue_id"] not in audit_ids:
                audit_ids.append(props["issue_id"])
            endpoints.append({
                "layer": props[layer_key], "feature_id": props[feature_key], **changed,
            })
        gap_log.append({
            "issue_id": props["issue_id"],
            "tile_id": props["tile_id"],
            "gap_px": props["gap_px"],
            "official_blue_support_fraction": props["official_blue_support_fraction"],
            "corrected_join_xy": target,
            "endpoints": endpoints,
        })

    lake_candidates_by_id: dict[str, dict[str, Any]] = {}
    for feature in lake_candidate_fc["features"]:
        props = feature["properties"]
        if props["issue_type"] == "matches_current_lake" and props.get("matches_current_lake_id") is not None:
            lake_candidates_by_id[str(props["matches_current_lake_id"])] = feature

    lake_log = []
    for feature in lakes_fc["features"]:
        props = feature["properties"]
        lake_id = str(props["lake_id"])
        candidate = lake_candidates_by_id.get(lake_id)
        if candidate is None:
            raise ValueError(f"No high-resolution boundary candidate for lake {lake_id}")
        old_geometry = shape(feature["geometry"])
        new_geometry = shape(candidate["geometry"])
        if not new_geometry.is_valid:
            new_geometry = new_geometry.buffer(0)
        if new_geometry.is_empty or new_geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Invalid raster-derived shoreline for lake {lake_id}")
        if not outline.buffer(0.25).covers(new_geometry.centroid):
            raise ValueError(f"Lake {lake_id} candidate lies outside the national outline")
        feature["geometry"] = mapping(new_geometry)
        bounds = new_geometry.bounds
        props["area_px2"] = round(float(new_geometry.area), 2)
        props["bbox_px"] = [
            round(bounds[0]), round(bounds[1]), round(bounds[2] - bounds[0]), round(bounds[3] - bounds[1]),
        ]
        props["detection_method"] = "4x_high_resolution_outer_blue_water_boundary"
        props["boundary_review_status"] = "manually_confirmed_against_4x_reference"
        props["audit_boundary_candidate_id"] = candidate["properties"]["candidate_id"]
        metric = lake_metrics[lake_id]
        props["median_inward_boundary_offset_px"] = float(metric["median_boundary_distance_inside_visible_water_px"])
        lake_log.append({
            "lake_id": int(lake_id),
            "tile_id": metric["tile_id"],
            "audit_candidate_id": candidate["properties"]["candidate_id"],
            "old_area_px2": round(float(old_geometry.area), 2),
            "new_area_px2": round(float(new_geometry.area), 2),
            "area_ratio_new_to_old": round(float(new_geometry.area / old_geometry.area), 4),
            "old_boundary_inward_offset_median_px": float(metric["median_boundary_distance_inside_visible_water_px"]),
            "new_geometry_valid": bool(new_geometry.is_valid),
        })

    added_log = []
    for candidate in reach_fc["features"]:
        candidate_props = candidate["properties"]
        issue_id = candidate_props["issue_id"]
        if decisions[issue_id]["decision"] != "repair":
            continue
        source_geometry = shape(candidate["geometry"])
        ratio_inside = source_geometry.intersection(outline).length / (source_geometry.length or 1.0)
        if 0.4 < ratio_inside < 0.6:
            raise ValueError(f"Could not assign river/cross-border layer from boundary position: {issue_id}")
        target_name = "river" if ratio_inside >= 0.6 else "cross_border_river"
        target_fc = layer_files[target_name]
        line = source_geometry.simplify(0.25, preserve_topology=False)
        if line.geom_type != "LineString" or len(line.coords) < 2:
            line = source_geometry
        deviation = source_geometry.hausdorff_distance(line)
        if deviation > 0.250001:
            raise ValueError(f"Precision simplification exceeded 0.25px for {issue_id}")
        next_id = max(int(f["properties"]["segment_id"]) for f in target_fc["features"]) + 1
        feature = {
            "type": "Feature",
            "properties": {
                "segment_id": next_id,
                "length_px": round(float(line.length), 3),
                "source": "manually_reviewed_4x_high_resolution_blue_water_centerline",
                "review_status": "manually_verified_missing_reach",
                "audit_candidate_id": issue_id,
                "audit_tile_id": candidate_props["tile_id"],
                "audit_machine_confidence": candidate_props["confidence"],
                "nearest_current_feature_id": candidate_props["nearest_current_feature_id"],
                "geometry_simplification_tolerance_px": 0.25,
            },
            "geometry": mapping(line),
        }
        target_fc["features"].append(feature)
        added_log.append({
            "audit_candidate_id": issue_id,
            "tile_id": candidate_props["tile_id"],
            "layer": target_name,
            "new_segment_id": next_id,
            "length_px": round(float(line.length), 3),
            "source_vertex_count": len(source_geometry.coords),
            "stored_vertex_count": len(line.coords),
            "max_simplification_deviation_px": round(float(deviation), 4),
        })

    river_fc["name"] = "hydrology_rivers_with_manually_verified_high_resolution_repairs"
    cross_border_fc["name"] = "cross_border_rivers_with_manually_verified_high_resolution_repairs"
    lakes_fc["name"] = "lakes_with_high_resolution_shoreline_boundaries"
    write_json(DATA / "hydrology_lines.geojson", river_fc)
    write_json(DATA / "cross_border_rivers.geojson", cross_border_fc)
    write_json(DATA / "lakes.geojson", lakes_fc)

    review_log = []
    for issue_id in sorted(decisions):
        row = decisions[issue_id]
        review_log.append({
            "issue_id": issue_id,
            "tile_id": row["tile_id"],
            "decision": row["decision"],
            "review_file": row["review_file"],
        })

    log = {
        "title": "2026-11-04 high-resolution hydrology repair log",
        "coordinate_system": "input_map.png image pixels; top-left origin; x right; y down; not georeferenced",
        "reference": "Project-local 4x no-text and 12x super-big map rasters, compared tile by tile",
        "manual_reach_review": {
            "total_candidates": len(decisions),
            "repaired_centerline_reaches": len(added_log),
            "excluded_false_positive_candidates": sum(row["decision"] == "exclude" for row in decisions.values()),
            "unresolved_candidates": sum(row["decision"] == "unresolved" for row in decisions.values()),
            "decisions": review_log,
        },
        "raster_supported_endpoint_gap_repairs": gap_log,
        "added_reaches": added_log,
        "lake_boundary_repairs": lake_log,
        "display_geometry_precision": {
            "hydrology_lod_max_simplification_px": 1.5,
            "hydrology_corner_smoothing_radius_px": 0.75,
            "rationale": "Preserve visible river and lake curvature at overview scales while keeping other map layers on their existing LOD settings.",
        },
        "review_sources": [
            "manual_review_RCH-0001-0112.csv",
            "manual_review_center/review_RCH-0113_0224.csv",
            "manual_review_RCH-0225-0336.csv",
        ],
    }
    write_json(DATA / "hydrology_repair_log.json", log)

    manifest_path = ROOT / "extraction_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest.update({
            "river_line_segments": len(river_fc["features"]),
            "cross_border_river_segments": len(cross_border_fc["features"]),
            "high_resolution_manual_reach_repairs": len(added_log),
            "raster_supported_endpoint_gap_repairs": len(gap_log),
            "lake_boundary_refinements": len(lake_log),
            "manual_review_false_positive_candidates": log["manual_reach_review"]["excluded_false_positive_candidates"],
        })
        write_json(manifest_path, manifest)

    import sys
    sys.path.insert(0, str(ROOT))
    import extract_hydrology_styles

    canals_fc = read_json(DATA / "canals.geojson")
    seasonal_fc = read_json(DATA / "seasonal_rivers.geojson")
    extract_hydrology_styles.update_render_files(
        river_fc["features"],
        canals_fc["features"],
        seasonal_fc["features"],
        cross_border_fc["features"],
    )
    print(json.dumps({
        "repaired_gaps": len(gap_log),
        "added_reaches": len(added_log),
        "reaches_by_layer": {
            "river": sum(item["layer"] == "river" for item in added_log),
            "cross_border_river": sum(item["layer"] == "cross_border_river" for item in added_log),
        },
        "excluded_candidates": log["manual_reach_review"]["excluded_false_positive_candidates"],
        "lakes_refined": len(lake_log),
        "unresolved": log["manual_reach_review"]["unresolved_candidates"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
