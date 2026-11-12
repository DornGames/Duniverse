from __future__ import annotations

import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point, shape

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCALE_KM_PER_PIXEL = json.loads((DATA / "georeference.json").read_text(encoding="utf-8"))["projection"]["pixel_scale_km_per_px"]


def read_js_assignment(path: Path, prefix: str):
    text = path.read_text(encoding="utf-8").strip()
    if not text.startswith(prefix):
        raise ValueError(f"Unexpected assignment in {path}")
    return json.loads(text[len(prefix):].strip().rstrip(";"))


def distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def controls_for_segment(points, closed, index):
    count = len(points)
    p1 = points[index]
    p2 = points[(index + 1) % count]
    p0 = points[(index - 1) % count] if closed or index > 0 else (2 * p1[0] - p2[0], 2 * p1[1] - p2[1])
    p3 = points[(index + 2) % count] if closed or index + 2 < count else (2 * p2[0] - p1[0], 2 * p2[1] - p1[1])

    def delta(a, b):
        return math.sqrt(max(distance(a, b), 1e-4))

    d01, d12, d23 = delta(p0, p1), delta(p1, p2), delta(p2, p3)
    tangent1 = (
        d12 * ((p1[0] - p0[0]) / d01 - (p2[0] - p0[0]) / (d01 + d12) + (p2[0] - p1[0]) / d12),
        d12 * ((p1[1] - p0[1]) / d01 - (p2[1] - p0[1]) / (d01 + d12) + (p2[1] - p1[1]) / d12),
    )
    tangent2 = (
        d12 * ((p2[0] - p1[0]) / d12 - (p3[0] - p1[0]) / (d12 + d23) + (p3[0] - p2[0]) / d23),
        d12 * ((p2[1] - p1[1]) / d12 - (p3[1] - p1[1]) / (d12 + d23) + (p3[1] - p2[1]) / d23),
    )
    c1 = (p1[0] + tangent1[0] / 3, p1[1] + tangent1[1] / 3)
    c2 = (p2[0] - tangent2[0] / 3, p2[1] - tangent2[1] / 3)
    return p1, c1, c2, p2


def sample_part(points_json, closed):
    points = [(float(p[0]), float(p[1])) for p in points_json]
    if len(points) < 2:
        return points
    samples = [points[0]]
    count = len(points) if closed else len(points) - 1
    for index in range(count):
        p1, c1, c2, p2 = controls_for_segment(points, closed, index)
        approximate_length = distance(p1, c1) + distance(c1, c2) + distance(c2, p2)
        steps = max(8, min(80, math.ceil(approximate_length / 2.0)))
        for step in range(1, steps + 1):
            t = step / steps
            u = 1 - t
            samples.append((
                u**3 * p1[0] + 3 * u**2 * t * c1[0] + 3 * u * t**2 * c2[0] + t**3 * p2[0],
                u**3 * p1[1] + 3 * u**2 * t * c1[1] + 3 * u * t**2 * c2[1] + t**3 * p2[1],
            ))
    return samples


def canonical_geometry_key(points_json, closed):
    points = [(round(float(p[0]), 1), round(float(p[1]), 1)) for p in points_json]

    def encode(sequence):
        return tuple(sequence)

    variants = [points, list(reversed(points))]
    if closed:
        for source in (points, list(reversed(points))):
            variants.extend(source[i:] + source[:i] for i in range(1, len(source)))
    return bool(closed), min(encode(v) for v in variants)


def intersection_position(route_line, intersection):
    if intersection.is_empty:
        return 0.0
    if intersection.geom_type in ("Point", "MultiPoint"):
        points = [intersection] if intersection.geom_type == "Point" else list(intersection.geoms)
        return min(route_line.project(point) for point in points)
    return route_line.project(intersection.representative_point())


routes = read_js_assignment(DATA / "highways.js", "window.DUNIVERSE_HIGHWAY_ROUTES =")
city_features = json.loads((DATA / "city_boundaries.geojson").read_text(encoding="utf-8"))["features"]
province_features = json.loads((DATA / "administrative_regions.geojson").read_text(encoding="utf-8"))["features"]
province_names = json.loads((DATA / "province_names.json").read_text(encoding="utf-8"))
major_membership = json.loads((DATA / "major_region_membership.json").read_text(encoding="utf-8"))["regions"]
province_to_major = {}
for major in major_membership:
    for code in major["codes"]:
        province_to_major.setdefault(code, major["name"])
province_order = {code: index for index, code in enumerate(province_names)}
major_order = {major["name"]: index for index, major in enumerate(major_membership)}
city_geometries = [(shape(feature["geometry"]), feature["properties"]) for feature in city_features]
province_geometries = [(shape(feature["geometry"]), feature["properties"]["region_code"]) for feature in province_features]

details = {}
for route in routes:
    parts = []
    unique_lines = {}
    stop_events = []
    for part_index, part in enumerate(route.get("parts", [])):
        points = part.get("points", [])
        if len(points) < 2:
            continue
        closed = bool(part.get("closed"))
        samples = sample_part(points, closed)
        line = LineString(samples)
        parts.append((part_index, line))
        key = canonical_geometry_key(points, closed)
        unique_lines.setdefault(key, line)
        stops = []
        for stop in part.get("stops", []):
            name = stop if isinstance(stop, str) else stop.get("city")
            if name:
                stops.append(name)
        if stops:
            for stop_index, name in enumerate(stops):
                fraction = stop_index / max(1, len(stops) - 1)
                stop_events.append((part_index, line.length * fraction, name))

    route_line = LineString()
    if unique_lines:
        from shapely.ops import unary_union
        route_line = unary_union(list(unique_lines.values()))
    length_px = route_line.length

    city_events = {}
    for part_index, part_line in parts:
        for event_index, _, city in stop_events:
            if event_index == part_index:
                city_events.setdefault(city, (part_index, _))
        for boundary_line, properties in city_geometries:
            if not part_line.intersects(boundary_line):
                continue
            position = intersection_position(part_line, part_line.intersection(boundary_line))
            for city in (properties.get("city_a"), properties.get("city_b")):
                if city and ("?" in city or chr(0xfffd) in city or city in ("未命名", "暂无")):
                    continue
                if city:
                    current = city_events.get(city)
                    candidate = (part_index, position)
                    if current is None or candidate < current:
                        city_events[city] = candidate

    province_events = {}
    for part_index, part_line in parts:
        for polygon, code in province_geometries:
            if not part_line.intersects(polygon):
                continue
            position = intersection_position(part_line, part_line.intersection(polygon))
            candidate = (part_index, position)
            if code not in province_events or candidate < province_events[code]:
                province_events[code] = candidate

    province_codes = sorted(province_events, key=lambda code: (province_events[code], province_order.get(code, 999)))
    major_events = {}
    for code, position in province_events.items():
        major = province_to_major.get(code)
        if major and (major not in major_events or position < major_events[major]):
            major_events[major] = position
    major_names = sorted(major_events, key=lambda name: (major_events[name], major_order.get(name, 999)))
    city_names = [name for name, _ in sorted(city_events.items(), key=lambda item: (item[1], item[0]))]

    details[route["id"]] = {
        "name": route.get("name", route["id"]),
        "lengthKm": round(length_px * SCALE_KM_PER_PIXEL, 1),
        "lengthSource": "data/georeference.json projection.pixel_scale_km_per_px",
        "majorRegions": major_names,
        "provinces": [province_names.get(code, code) for code in province_codes],
        "cities": city_names,
        "stops": list(dict.fromkeys(
            stop if isinstance(stop, str) else stop.get("city")
            for part in route.get("parts", [])
            for stop in part.get("stops", [])
            if (stop if isinstance(stop, str) else stop.get("city"))
        )),
    }

output_data = {"projectionScaleKmPerPixel": SCALE_KM_PER_PIXEL, "routes": details}
output = "window.DUNIVERSE_HIGHWAY_GEOGRAPHY = " + json.dumps(output_data, ensure_ascii=False, indent=2) + ";\n"
(DATA / "highway_geography.js").write_text(output, encoding="utf-8", newline="\n")
for route_id, item in details.items():
    print(f"{route_id}: {item['lengthKm']:.1f} km, {len(item['majorRegions'])} macro-regions, {len(item['provinces'])} provinces, {len(item['cities'])} cities")