"""Keep reviewed point-like eastern islets visible in the map layers."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOT_DATA_FILE = ROOT / "data" / "eastern_island_dots.geojson"
HTML_FILES = (ROOT / "index.html", ROOT / "relief_map.html")
VECTOR_PREVIEW = ROOT / "vector_preview.svg"

GROUP_PATTERN = re.compile(
    r'(<g id="coastlineFills"[^>]*>)(.*?)(</g>)', re.DOTALL
)
ELEMENT_PATTERN = re.compile(r"<(?:path|circle)\b[^>]*/?>")
ATTRIBUTE_PATTERN = re.compile(r'\b(data-island-dot|cx|cy)="([^"]+)"')
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


def load_island_dots() -> tuple[list[str], set[tuple[str, str]]]:
    collection = json.loads(DOT_DATA_FILE.read_text(encoding="utf-8"))
    elements = []
    coordinates = set()
    for feature in collection["features"]:
        if feature.get("properties", {}).get("role") != "island_dot":
            continue
        dot_id = html.escape(str(feature["id"]), quote=True)
        x, y = feature["geometry"]["coordinates"]
        x_text, y_text = f"{x:.4f}", f"{y:.4f}"
        radius = feature["properties"]["render_radius_px"]
        elements.append(
            f'<circle data-island-dot="{dot_id}" cx="{x_text}" '
            f'cy="{y_text}" r="{radius:.4f}" fill="#111"/>'
        )
        coordinates.add((x_text, y_text))
    if not elements:
        raise ValueError("No reviewed point-like eastern islets were found")
    return elements, coordinates


def sync_html(path: Path, dots: list[str], coordinates: set[tuple[str, str]]) -> int:
    source = path.read_text(encoding="utf-8")
    match = GROUP_PATTERN.search(source)
    if not match:
        raise ValueError(f"Could not find the island fill group in {path.name}")

    kept = []
    removed = 0
    for tag in ELEMENT_PATTERN.findall(match.group(2)):
        if not tag.startswith("<circle"):
            kept.append(tag)
            continue
        attributes = dict(ATTRIBUTE_PATTERN.findall(tag))
        key = (attributes.get("cx", ""), attributes.get("cy", ""))
        if "data-island-dot" in attributes or key in coordinates:
            removed += 1
        else:
            kept.append(tag)

    updated_group = match.group(1) + "".join(kept + dots) + match.group(3)
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


def sync_vector_preview(path: Path, dots: list[str]) -> None:
    source = path.read_text(encoding="utf-8")
    group = (
        '<g id="eastern-island-dots" fill="#111" pointer-events="none">'
        + "".join(dots)
        + "</g>"
    )
    pattern = re.compile(r'<g id="eastern-island-dots"[^>]*>.*?</g>', re.DOTALL)
    if pattern.search(source):
        source = pattern.sub(group, source, count=1)
    elif "</svg>" in source:
        source = source.replace("</svg>", group + "</svg>", 1)
    else:
        raise ValueError(f"Could not find the end of {path.name}")
    write_text_preserving_newlines(path, source)


def main() -> None:
    dots, coordinates = load_island_dots()
    html_files = [path for path in HTML_FILES if path.exists()]
    if not html_files:
        raise FileNotFoundError("No interactive map HTML file was found")
    for html_file in html_files:
        removed = sync_html(html_file, dots, coordinates)
        print(
            f"{html_file.relative_to(ROOT)}: retained all shoreline paths, "
            f"replaced {removed} prior dot markers with {len(dots)} reviewed islets"
        )
    sync_vector_preview(VECTOR_PREVIEW, dots)
    print(f"{VECTOR_PREVIEW.relative_to(ROOT)}: synced {len(dots)} point islets")


if __name__ == "__main__":
    main()
