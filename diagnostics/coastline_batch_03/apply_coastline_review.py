#!/usr/bin/env python3
"""Apply the final reviewed closures and reject one non-coast map symbol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "coastlines.geojson"
LOG = ROOT / "data" / "coastline_correction_batch_03.json"
EXPECTED_INPUT_SHA256 = "36d06e5c7f3db67db96c01b424325b6180c8594ad85710f8eaf41b014a2ef73f"

REVIEWED_FEATURES = {
    123: {
        "review_id": "official-tile-0127",
        "center_xy": [1302.8332, 511.6247],
        "gap_px": 0.083,
        "action": "close_as_island_shoreline",
        "evidence": "12x reference has a continuous dark oval shoreline enclosing land-colored fill.",
    },
    359: {
        "review_id": "official-tile-0379",
        "center_xy": [1353.0002, 988.2087],
        "gap_px": 0.083,
        "action": "close_as_island_shoreline",
        "evidence": "12x reference has a continuous dark oval shoreline enclosing land-colored fill.",
    },
    421: {
        "review_id": "official-tile-0443",
        "center_xy": [1364.8747, 98.1667],
        "gap_px": 0.084,
        "action": "reject_nonshore_symbol",
        "evidence": "12x reference shows a pale gray marker-like blob with a white center, without dark shoreline ink.",
    },
    522: {
        "review_id": "official-tile-0547",
        "center_xy": [1457.0832, 938.1247],
        "gap_px": 0.083,
        "action": "close_as_island_shoreline",
        "evidence": "12x reference has a continuous dark oval shoreline enclosing land-colored fill.",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    input_hash = sha256(SOURCE)
    if input_hash != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            f"Unexpected coastline input hash {input_hash}; review before applying."
        )

    collection = json.loads(SOURCE.read_text(encoding="utf-8"))
    features = collection["features"]
    actions = []
    for index, review in REVIEWED_FEATURES.items():
        feature = features[index]
        props = feature["properties"]
        geom = shape(feature["geometry"])
        if geom.geom_type != "LineString" or props.get("source") != (
            "official_no_river_highres_tiled_open_ocean_boundary"
        ):
            raise SystemExit(f"Unexpected feature type/source at array index {index}.")
        raw = feature["geometry"]["coordinates"]
        first, last = raw[0], raw[-1]
        if review["action"] == "reject_nonshore_symbol":
            if index != 421 or props.get("role") != "coastline":
                raise SystemExit("The reviewed symbol candidate changed unexpectedly.")
            props["review_status"] = "rejected"
            props["review_note"] = review["evidence"]
            actions.append(
                {
                    "review_id": review["review_id"],
                    "feature_array_index_zero_based": index,
                    "center_index_xy_px": review["center_xy"],
                    "original_start_xy": first,
                    "original_end_xy": last,
                    "gap_length_px": review["gap_px"],
                    "action": review["action"],
                    "reason": review["evidence"],
                }
            )
            continue

        if props.get("role") != "coastline":
            raise SystemExit(f"Expected an open coastline role at array index {index}.")
        if first == last:
            raise SystemExit(f"Feature at array index {index} is already closed.")
        raw[-1] = list(first)
        props["role"] = "island_shoreline"
        props["review_status"] = "reviewed_closed_highres_island"
        new_length_px = round(shape(feature["geometry"]).length, 3)
        props["length_px"] = new_length_px
        actions.append(
            {
                "review_id": review["review_id"],
                "feature_array_index_zero_based": index,
                "center_index_xy_px": review["center_xy"],
                "original_start_xy": first,
                "original_end_xy": last,
                "gap_length_px": review["gap_px"],
                "closed_xy": raw[-1],
                "new_length_px": new_length_px,
                "action": review["action"],
                "reason": review["evidence"],
            }
        )

    output_bytes = (json.dumps(collection, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    SOURCE.write_bytes(output_bytes)
    updated = json.loads(output_bytes)
    role_counts = Counter(f["properties"].get("role", "") for f in updated["features"])
    closed_counts = Counter(
        f["properties"].get("role", "")
        for f in updated["features"]
        if shape(f["geometry"]).geom_type == "LineString"
        and shape(f["geometry"]).is_ring
    )
    log = {
        "schema_version": 1,
        "correction_batch": "3/3",
        "created_date": "2026-11-06",
        "scope": {
            "remaining_candidate_total": 59,
            "reviewed_candidate_total": 59,
            "confirmed_geometry_edits": 3,
            "rejected_nonshore_candidates": 1,
            "no_geometry_change_candidates": 55,
        },
        "coordinate_mapping": {
            "coordinate_frame": "input_map.png pixel-centre indices, not longitude/latitude",
            "base_index_to_12x_pixel_center_index": "X12=12*x+5.5; Y12=12*y+5.5",
            "open_coastline_stored_to_normalized_index": "subtract 11/24 from both stored x and y",
        },
        "reference": {
            "file": "official_map/Administrative Division Map of Donh 2026-no-river-super-big.png",
            "provenance_limit": "Local project reference; the available files do not independently authenticate it as the official published original.",
            "evidence_image": "diagnostics/coastline_batch_03/near_close_raw_reference.png",
        },
        "input": {
            "file": "data/coastlines.geojson",
            "sha256": input_hash,
            "feature_count": len(features),
        },
        "actions": actions,
        "output": {
            "file": "data/coastlines.geojson",
            "sha256": hashlib.sha256(output_bytes).hexdigest(),
            "feature_count": len(updated["features"]),
            "role_counts": dict(role_counts),
            "closed_geometry_counts": dict(closed_counts),
            "rejected_feature_count": sum(
                f["properties"].get("review_status") == "rejected"
                for f in updated["features"]
            ),
        },
        "candidate_disposition_file": "diagnostics/coastline_batch_03/candidate_dispositions.csv",
        "method_note": "Open-line candidates were changed only when the high-resolution reference showed a closed, dark shoreline around land-colored fill. The gray marker candidate is retained in source data with review_status=rejected so the boundary builder excludes it from rendering.",
    }
    LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Applied 3 island closures and rejected 1 non-shore symbol candidate.")
    print(f"Coastline output SHA-256: {log['output']['sha256']}")


if __name__ == "__main__":
    main()
