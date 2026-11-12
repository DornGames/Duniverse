"""Sync traced white eastern island silhouettes to the map layers."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISLAND_SHAPES_FILE = ROOT / "data" / "eastern_island_shapes.geojson"
HTML_FILES = (ROOT / "index.html", ROOT / "relief_map.html")
VECTOR_PREVIEW = ROOT / "vector_preview.svg"

GROUP_PATTERN = re.compile(
    r'(<g id="coastlineFills"[^>]*>)(.*?)(</g>)', re.DOTALL
)
ELEMENT_PATTERN = re.compile(r"<(?:path|circle)\b[^>]*/?>")
OLD_TOLERANCE_PATTERN = re.compile(
    r"const pathToleranceIndex=isCoastalFill\?coastalFillToleranceIndex:\s*"
    r"(?:(isHydrology\?Math\.min\(toleranceIndex,"
    r"HYDROLOGY_LOD_TOLERANCES\.length-1\):))?toleranceIndex;"
)
EAST_ISLAND_TOLERANCE_PATTERN = re.compile(
    r"const eastIsland=record\.layerId==='islandFills'&&"
    r"record\.bounds\.x\+record\.bounds\.width/2>=\d+;"
    r"const pathToleranceIndex=isCoastalFill\?"
    r"\(eastIsland\?0:coastalFillToleranceIndex\):"
)
OLD_VISIBILITY = (
    "const hideIsland=record.layerId==='islandFills'&&"
    "(!layerVisible||islandExtent<MIN_ISLAND_SCREEN_EXTENT);"
)
NEW_VISIBILITY = (
    "const hideIsland=record.layerId==='islandFills'&&"
    "(!layerVisible||(!eastIsland&&islandExtent<MIN_ISLAND_SCREEN_EXTENT));"
)
EAST_ISLAND_TOLERANCE = (
    "const eastIsland=record.layerId==='islandFills'&&"
    "record.bounds.x+record.bounds.width/2>=1400;"
    "const pathToleranceIndex=isCoastalFill?"
    "(eastIsland?0:coastalFillToleranceIndex):"
)


def write_text_preserving_newlines(path: Path, text: str) -> None:
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_bytes(normalized.replace("\n", newline.decode("ascii")).encode("utf-8"))


def geometry_path(
    geometry: dict, display_scale: float, center: tuple[float, float]
) -> str:
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise ValueError(f"Unexpected island geometry: {geometry['type']}")

    commands = []
    for polygon in polygons:
        for ring in polygon:
            if len(ring) < 4:
                raise ValueError("An island ring has fewer than four coordinates")
            scaled_ring = [
                (
                    center[0] + (x - center[0]) * display_scale,
                    center[1] + (y - center[1]) * display_scale,
                )
                for x, y in ring
            ]
            commands.append(f"M {scaled_ring[0][0]:.4f},{scaled_ring[0][1]:.4f}")
            commands.extend(
                f"L {x:.4f},{y:.4f}" for x, y in scaled_ring[1:]
            )
            commands.append("Z")
    return " ".join(commands)


def load_island_shapes() -> list[str]:
    collection = json.loads(ISLAND_SHAPES_FILE.read_text(encoding="utf-8"))
    elements = []
    for feature in collection["features"]:
        properties = feature.get("properties", {})
        if properties.get("role") != "island_surface":
            continue
        shape_id = html.escape(str(feature["id"]), quote=True)
        display_scale = float(properties.get("display_scale_factor", 1.0))
        source_center = properties.get("source_pixel_centroid")
        if not source_center:
            raise ValueError(f"Missing source centroid for {shape_id}")
        source_scale = float(collection.get("source_scale", 12))
        center = (source_center[0] / source_scale, source_center[1] / source_scale)
        path_data = geometry_path(feature["geometry"], display_scale, center)
        elements.append(
            f'<path data-eastern-island-shape="{shape_id}" '
            f'd="{path_data}" fill="#fff"/>'
        )
    if not elements:
        raise ValueError("No reviewed eastern island silhouettes were found")
    return elements


def sync_html(path: Path, shapes: list[str]) -> int:
    source = path.read_text(encoding="utf-8")
    match = GROUP_PATTERN.search(source)
    if not match:
        raise ValueError(f"Could not find the island fill group in {path.name}")

    kept = []
    removed = 0
    for tag in ELEMENT_PATTERN.findall(match.group(2)):
        is_old_dot = tag.startswith("<circle") and "data-island-dot=" in tag
        is_old_shape = tag.startswith("<path") and "data-eastern-island-shape=" in tag
        if is_old_dot or is_old_shape:
            removed += 1
        else:
            kept.append(tag)

    updated_group = match.group(1) + "".join(kept + shapes) + match.group(3)
    source = source[: match.start()] + updated_group + source[match.end() :]

    if EAST_ISLAND_TOLERANCE not in source:
        if EAST_ISLAND_TOLERANCE_PATTERN.search(source):
            source = EAST_ISLAND_TOLERANCE_PATTERN.sub(
                EAST_ISLAND_TOLERANCE, source, count=1
            )
        else:
            matches = list(OLD_TOLERANCE_PATTERN.finditer(source))
            if len(matches) != 1:
                raise ValueError(f"Unexpected island precision logic in {path.name}")
            hydrology_branch = matches[0].group(1) or ""
            replacement = (
                EAST_ISLAND_TOLERANCE + hydrology_branch + "toleranceIndex;"
            )
            source = OLD_TOLERANCE_PATTERN.sub(replacement, source, count=1)

    if OLD_VISIBILITY in source:
        source = source.replace(OLD_VISIBILITY, NEW_VISIBILITY, 1)
    elif NEW_VISIBILITY not in source:
        raise ValueError(f"Unexpected island visibility logic in {path.name}")

    write_text_preserving_newlines(path, source)
    return removed


def sync_vector_preview(path: Path, shapes: list[str]) -> None:
    source = path.read_text(encoding="utf-8")
    group = (
        '<g id="eastern-island-shapes" fill="#fff" pointer-events="none">'
        + "".join(shapes)
        + "</g>"
    )
    pattern = re.compile(
        r'<g id="eastern-island-(?:dots|shapes)"[^>]*>.*?</g>', re.DOTALL
    )
    if pattern.search(source):
        source = pattern.sub(group, source, count=1)
    elif "</svg>" in source:
        source = source.replace("</svg>", group + "</svg>", 1)
    else:
        raise ValueError(f"Could not find the end of {path.name}")
    write_text_preserving_newlines(path, source)


def main() -> None:
    shapes = load_island_shapes()
    html_files = [path for path in HTML_FILES if path.exists()]
    if not html_files:
        raise FileNotFoundError("No interactive map HTML file was found")
    for html_file in html_files:
        removed = sync_html(html_file, shapes)
        print(
            f"{html_file.relative_to(ROOT)}: retained all shoreline paths, "
            f"replaced {removed} prior markers with {len(shapes)} traced white silhouettes"
        )
    sync_vector_preview(VECTOR_PREVIEW, shapes)
    print(f"{VECTOR_PREVIEW.relative_to(ROOT)}: synced {len(shapes)} island silhouettes")


if __name__ == "__main__":
    main()
