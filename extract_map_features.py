#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semi-automatic feature extraction for a pastel-filled administrative map.

Outputs:
  administrative_regions.geojson
  national_outline.geojson
  hydrology_lines.geojson
  lakes.geojson
  settlement_markers.geojson
  region_code_labels.geojson
  region_codes.csv
  ocr_candidates.csv
  map_metadata.json
  diagnostics/*.png

All coordinates are source-image pixels (origin at top-left), not geographic coordinates.
"""
import argparse
import json
import math
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon, Point, LineString, mapping
from skimage.morphology import skeletonize
import pytesseract

SPECIAL_COLORS = [(213, 160, 207), (168, 139, 180), (235, 181, 240)]

# Blue province-code lettering on the large DD island resembles a pair of
# short waterways at source-image resolution. Keep those glyph pixels out of
# hydrology extraction; there is no river through the label.
BLUE_HYDROLOGY_TEXT_EXCLUSION_RECTS = [
    (1778, 700, 1805, 718),
]

# Visually transcribed code-label anchors; coordinates are approximate image pixels.
CODE_LABELS = [
    ("BD",900,135),("OH",1000,240),("HX",1150,250),("DE",680,350),("ZC",850,420),("TM",1000,360),
    ("FN",1348,490),("LK",710,600),("XD",470,700),("VA",650,800),("MM",850,800),("GA",950,680),
    ("RF",1080,660),("ST",1190,660),("BP",1320,580),("PP",1320,730),("MT",1100,800),("FS",1290,900),
    ("GR",1000,920),("BG",1110,910),("LL",1120,1000),("PN",820,1000),("WZ",940,1060),("NG",1080,1040),
    ("TL",340,1150),("HB",640,1120),("TT",434,1196),("TT",566,1314),("SW",550,1249),("SW",414,1340),
    ("GW",850,1270),("SP",890,1245),("CS",990,1270),("FL",1190,1200),("TG",1180,1310),("SB",1350,1060),
    ("UE",1350,1170),("KT",1350,1270),("SD",1450,1200),("CM",1480,1310),("SE",1460,1370),
    ("OC",850,1340),("ND",1200,1400),("SM",840,1420),("TO",1600,450),("DD",1800,680),("WT",1360,975)
]

def write_fc(path, name, features):
    fc = {
        "type": "FeatureCollection",
        "name": name,
        "coordinate_reference_note": "Image pixel coordinates; origin top-left; x right; y down. Not georeferenced.",
        "features": features
    }
    Path(path).write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")

def contour_polygon(contour, epsilon=1.0):
    pts = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(pts) < 3:
        return None
    poly = Polygon([(float(x), float(y)) for x, y in pts])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly.simplify(epsilon, preserve_topology=True)

def geom_parts(geom):
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]

def pixel_path_length(path):
    return sum(
        math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        for i in range(len(path)-1)
    )

def extract_river_paths(skeleton, min_length=20.0):
    ys, xs = np.where(skeleton)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    adjacency = {}
    for y, x in pixels:
        neighbors = []
        for dy, dx in offsets:
            q = (y+dy, x+dx)
            if q not in pixels:
                continue
            # Avoid redundant diagonal edges when an orthogonal connection exists.
            if dy != 0 and dx != 0 and ((y, x+dx) in pixels or (y+dy, x) in pixels):
                continue
            neighbors.append(q)
        adjacency[(y,x)] = neighbors

    degree = {p: len(ns) for p, ns in adjacency.items()}
    visited = set()
    paths = []

    def edge_key(a, b):
        return (a,b) if a < b else (b,a)

    def trace(start, nxt):
        path = [start, nxt]
        visited.add(edge_key(start, nxt))
        prev, cur = start, nxt
        while degree.get(cur, 0) == 2:
            options = [q for q in adjacency[cur] if q != prev]
            if not options:
                break
            q = options[0]
            ek = edge_key(cur, q)
            if ek in visited:
                break
            visited.add(ek)
            path.append(q)
            prev, cur = cur, q
            if cur == start:
                break
        return path

    for p, d in degree.items():
        if d != 2:
            for q in adjacency[p]:
                if edge_key(p, q) not in visited:
                    paths.append(trace(p, q))
    for p, neighbors in adjacency.items():
        for q in neighbors:
            if edge_key(p, q) not in visited:
                paths.append(trace(p, q))

    lines = []
    for path in paths:
        length = pixel_path_length(path)
        if length < min_length:
            continue
        xy = np.array([[p[1], p[0]] for p in path], dtype=np.float32).reshape(-1,1,2)
        approx = cv2.approxPolyDP(xy, 1.5, False).reshape(-1,2)
        if len(approx) < 2:
            continue
        line = LineString([(float(x),float(y)) for x,y in approx])
        if line.length >= 10:
            lines.append(line)
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input PNG/JPG map")
    parser.add_argument("--out", default="map_extraction", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    data_dir = out / "data"
    diag_dir = out / "diagnostics"
    data_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    arr = np.array(image)
    h, w = arr.shape[:2]
    color_counts = Counter(map(tuple, arr.reshape(-1,3)))

    # A. Dominant pastel administrative fill colors.
    palette = []
    for rgb, count in color_counts.items():
        c = np.array(rgb, dtype=np.int32)
        excluded = rgb in [(255,255,255),(177,208,237),(0,0,0),(255,0,0),(147,147,147)]
        if count >= 1200 and not excluded and c.min() >= 130 and c.max()-c.min() <= 130:
            palette.append(tuple(map(int,c)))
    all_region_colors = palette + SPECIAL_COLORS

    # B. Polygonize each fill color.
    region_components = []
    for color in all_region_colors:
        mask = np.all(arr == color, axis=2).astype(np.uint8) * 255
        k = 5 if color in palette else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        threshold = 300 if color in palette else 40
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < threshold:
                continue
            poly = contour_polygon(contour, 1.0)
            if poly is None:
                continue
            x,y,bw,bh = cv2.boundingRect(contour)
            m = cv2.moments(contour)
            cx = m["m10"]/m["m00"] if m["m00"] else x+bw/2
            cy = m["m01"]/m["m00"] if m["m00"] else y+bh/2
            region_components.append({
                "color": color, "area": area, "bbox": [int(x),int(y),int(bw),int(bh)],
                "centroid": [float(cx),float(cy)], "polygon": poly
            })

    # C. Associate code labels with candidate polygon components.
    special_groups = {
        "FN":[i for i,r in enumerate(region_components) if r["color"]==(213,160,207)],
        "TO":[i for i,r in enumerate(region_components) if r["color"]==(168,139,180)],
        "WT":[i for i,r in enumerate(region_components) if r["color"]==(235,181,240)],
        "DD":[i for i,r in enumerate(region_components) if r["color"]==(210,225,255)]
    }
    assigned, used, label_features = {}, set(), []
    for code, x, y in CODE_LABELS:
        p = Point(x,y)
        if code in special_groups and special_groups[code]:
            indices = special_groups[code]
            nearest = min(indices, key=lambda i: region_components[i]["polygon"].distance(p))
            distance = region_components[nearest]["polygon"].distance(p)
            component_ids = [i+1 for i in indices]
            assigned.setdefault(code,set()).update(indices)
        else:
            candidates = [i for i in range(len(region_components)) if i not in used]
            nearest = min(candidates, key=lambda i: region_components[i]["polygon"].distance(p))
            distance = region_components[nearest]["polygon"].distance(p)
            used.add(nearest)
            component_ids = [nearest+1]
            assigned.setdefault(code,set()).add(nearest)
        label_features.append({
            "type":"Feature",
            "properties":{
                "region_code":code,"matched_component_ids":component_ids,
                "match_distance_px":round(float(distance),2),
                "transcription_method":"visual_inspection_with_OCR_assistance",
                "review_status":"candidate_requires_final_human_confirmation"
            },
            "geometry":{"type":"Point","coordinates":[int(x),int(y)]}
        })
    component_to_code = {i:code for code,inds in assigned.items() for i in inds}

    # Keep both disconnected SW components assigned to the same province code,
    # but place its map label only on the larger component.
    sw_components = assigned.get("SW", set())
    if sw_components:
        largest_sw_component_id = max(
            sw_components, key=lambda i: region_components[i]["polygon"].area
        ) + 1
        sw_component_ids = sorted(i + 1 for i in sw_components)
        for feature in label_features:
            if feature["properties"].get("region_code") == "SW":
                feature["properties"]["matched_component_ids"] = sw_component_ids
        label_features = [
            feature for feature in label_features
            if feature["properties"].get("region_code") != "SW"
            or largest_sw_component_id in feature["properties"].get("matched_component_ids", [])
        ]

    admin_features = []
    for i, comp in enumerate(region_components):
        geom = comp["polygon"]
        admin_features.append({
            "type":"Feature",
            "properties":{
                "component_id":i+1,"region_code":component_to_code.get(i),
                "fill_rgb":list(comp["color"]),"area_px2":round(float(geom.area),1),
                "bbox_px":comp["bbox"],
                "centroid_px":[round(comp["centroid"][0],1),round(comp["centroid"][1],1)],
                "extraction_method":"exact_color_mask_morphological_closing_contour_vectorization",
                "review_status":"automatic_candidate"
            },
            "geometry":mapping(geom)
        })
    write_fc(data_dir/"administrative_regions.geojson","administrative_regions",admin_features)
    write_fc(data_dir/"region_code_labels.geojson","region_code_labels",label_features)

    # D. National outline from union of fill masks.
    all_mask = np.zeros((h,w),np.uint8)
    for color in all_region_colors:
        all_mask[np.all(arr==color,axis=2)] = 255
    all_mask = cv2.morphologyEx(
        all_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=1
    )
    outline_contours,_ = cv2.findContours(all_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    outline_features=[]
    land_silhouette=np.zeros((h,w),np.uint8)
    for contour in outline_contours:
        area=float(cv2.contourArea(contour))
        if area < 40:
            continue
        poly=contour_polygon(contour,1.0)
        if poly is None: continue
        x,y,bw,bh=cv2.boundingRect(contour)
        cv2.drawContours(land_silhouette,[contour],-1,255,thickness=cv2.FILLED)
        outline_features.append({
            "type":"Feature",
            "properties":{
                "outline_id":len(outline_features)+1,
                "role":"mainland" if area>10000 else "island_or_coastal_fragment",
                "area_px2":round(area,1),"bbox_px":[int(x),int(y),int(bw),int(bh)]
            },
            "geometry":mapping(poly)
        })
    write_fc(data_dir/"national_outline.geojson","national_outline",outline_features)

    # E. Hydrology: blue mask, lake cores, river skeleton.
    palette_arr=np.array(all_region_colors,dtype=np.int32)
    flat=arr.reshape(-1,3).astype(np.int32)
    min_d2=np.full(flat.shape[0],999999,dtype=np.int32)
    for start in range(0,len(flat),200000):
        q=flat[start:start+200000]
        d2=((q[:,None,:]-palette_arr[None,:,:])**2).sum(axis=2)
        min_d2[start:start+len(q)]=d2.min(axis=1)
    distance_to_fill=np.sqrt(min_d2).reshape(h,w)
    blue_mask=(
        (arr[:,:,2].astype(int)-arr[:,:,0].astype(int)>30) &
        (arr[:,:,2].astype(int)-arr[:,:,1].astype(int)>10) &
        (arr[:,:,2]>160) & (land_silhouette>0) & (distance_to_fill>12)
    ).astype(np.uint8)*255
    for left, top, right, bottom in BLUE_HYDROLOGY_TEXT_EXCLUSION_RECTS:
        blue_mask[top:bottom, left:right] = 0
    water_distance=cv2.distanceTransform((blue_mask>0).astype(np.uint8),cv2.DIST_L2,5)
    lake_core=(water_distance>2.2).astype(np.uint8)*255
    lake_contours,_=cv2.findContours(lake_core,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    lake_features=[]
    for contour in lake_contours:
        area=float(cv2.contourArea(contour))
        if area<20: continue
        poly=contour_polygon(contour,1.0)
        if poly is None: continue
        x,y,bw,bh=cv2.boundingRect(contour)
        lake_features.append({
            "type":"Feature",
            "properties":{"lake_id":len(lake_features)+1,"area_px2":round(area,1),
                          "bbox_px":[int(x),int(y),int(bw),int(bh)],
                          "detection_method":"blue_hydrology_mask_distance_transform_core"},
            "geometry":mapping(poly)
        })
    write_fc(data_dir/"lakes.geojson","lake_candidates",lake_features)
    lake_dilated=cv2.dilate(lake_core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    river_mask=cv2.bitwise_and(blue_mask,cv2.bitwise_not(lake_dilated))
    river_skeleton=skeletonize(river_mask>0)
    lines=extract_river_paths(river_skeleton,20.0)
    river_features=[]
    for line in lines:
        river_features.append({
            "type":"Feature",
            "properties":{"segment_id":len(river_features)+1,"length_px":round(float(line.length),1),
                          "source":"blue_hydrology_mask_skeleton","review_status":"candidate_line_segment"},
            "geometry":mapping(line)
        })
    write_fc(data_dir/"hydrology_lines.geojson","hydrology_line_candidates",river_features)

    # F. Black and red point-symbol candidates.
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    black_mask=(gray<60).astype(np.uint8)
    n, labels, stats, centroids=cv2.connectedComponentsWithStats(black_mask,8)
    black_points=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if not (38<=area<=45 and 6<=bw<=8 and 6<=bh<=8): continue
        cx,cy=centroids[i]
        if land_silhouette[int(cy),int(cx)]==0: continue
        local=(labels[y:y+bh,x:x+bw]==i).astype(np.uint8)
        cs,_=cv2.findContours(local,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        perimeter=cv2.arcLength(cs[0],True) if cs else 0
        circularity=4*math.pi*area/(perimeter*perimeter+1e-6)
        if circularity<0.9: continue
        black_points.append((float(cx),float(cy),int(area),float(circularity)))

    red_mask=((arr[:,:,0]>160)&(arr[:,:,1]<100)&(arr[:,:,2]<100)).astype(np.uint8)
    n, labels, stats, centroids=cv2.connectedComponentsWithStats(red_mask,8)
    red_points=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if not (38<=area<=70 and 6<=bw<=10 and 6<=bh<=10): continue
        cx,cy=centroids[i]
        local=(labels[y:y+bh,x:x+bw]==i).astype(np.uint8)
        cs,_=cv2.findContours(local,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        perimeter=cv2.arcLength(cs[0],True) if cs else 0
        circularity=4*math.pi*area/(perimeter*perimeter+1e-6)
        if circularity<0.9: continue
        red_points.append((float(cx),float(cy),int(area),float(circularity)))

    marker_features=[]
    for marker_type, points, color in [
        ("city_marker_candidate",black_points,"#111111"),
        ("provincial_capital_marker_candidate",red_points,"#e52323")
    ]:
        for x,y,area,circ in points:
            marker_features.append({
                "type":"Feature",
                "properties":{"marker_type":marker_type,"color":color,"component_area_px":area,
                              "shape_score":round(float(circ),3),
                              "review_status":"candidate_location_unverified"},
                "geometry":{"type":"Point","coordinates":[round(x,2),round(y,2)]}
            })
    write_fc(data_dir/"settlement_markers.geojson","point_marker_candidates",marker_features)

    # G. OCR text boxes.
    ocr_rows=[]
    try:
        ocr=pytesseract.image_to_data(
            image,lang="chi_sim+eng",config="--oem 1 --psm 11",
            output_type=pytesseract.Output.DATAFRAME
        )
        ocr=ocr.dropna(subset=["text"]).copy()
        ocr["text"]=ocr["text"].astype(str).str.strip()
        ocr=ocr[ocr["text"]!=""]
        for _,r in ocr.iterrows():
            ocr_rows.append({
                "text":r["text"],"confidence":float(r["conf"]),
                "left":int(r["left"]),"top":int(r["top"]),
                "width":int(r["width"]),"height":int(r["height"]),
                "center_x":round(float(r["left"]+r["width"]/2),1),
                "center_y":round(float(r["top"]+r["height"]/2),1)
            })
    except Exception as exc:
        (data_dir/"ocr_error.txt").write_text(str(exc),encoding="utf-8")
    pd.DataFrame(ocr_rows).to_csv(data_dir/"ocr_candidates.csv",index=False,encoding="utf-8-sig")
    code_rows=[]
    for f in label_features:
        p=f["properties"]; x,y=f["geometry"]["coordinates"]
        code_rows.append({
            "region_code":p["region_code"],"label_x_px":x,"label_y_px":y,
            "matched_component_ids":";".join(map(str,p["matched_component_ids"])),
            "match_distance_px":p["match_distance_px"],"review_status":p["review_status"]
        })
    pd.DataFrame(code_rows).to_csv(data_dir/"region_codes.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([
        {"marker_type":"city_marker_candidate","count":len(black_points)},
        {"marker_type":"provincial_capital_marker_candidate","count":len(red_points)}
    ]).to_csv(data_dir/"marker_detection_summary.csv",index=False,encoding="utf-8-sig")

    # H. Metadata and diagnostics.
    metadata={
        "map_title":"湩国行政区划图","edition":"2026年 第六版",
        "english_subtitle":"ADMINISTRATIVE DIVISION MAP OF DONH (visual transcription; verify spelling)",
        "producer":"Dornhub","copyright":"© 2022–2026 Dornhub, All Rights Reserved.",
        "image_width_px":w,"image_height_px":h,
        "coordinate_system":"image pixel coordinates; origin top-left; x right; y down",
        "georeferenced":False,"scale_bar_detected":False
    }
    (data_dir/"map_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    overlay=arr.copy()
    for comp in region_components:
        for g in geom_parts(comp["polygon"]):
            coords=np.array([[[int(x),int(y)] for x,y in g.exterior.coords]],dtype=np.int32)
            cv2.polylines(overlay,coords,True,(235,45,45),1,cv2.LINE_AA)
    for code,x,y in CODE_LABELS:
        cv2.putText(overlay,code,(x-8,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.42,(20,20,20),1,cv2.LINE_AA)
    Image.fromarray(overlay).save(diag_dir/"region_contours_overlay.png")
    Image.fromarray(land_silhouette).save(diag_dir/"national_land_mask.png")
    Image.fromarray(blue_mask).save(diag_dir/"hydrology_mask.png")
    Image.fromarray(lake_core).save(diag_dir/"lake_core_mask.png")
    marker_overlay=arr.copy()
    for x,y,_,_ in black_points: cv2.circle(marker_overlay,(int(round(x)),int(round(y))),5,(255,0,0),1,cv2.LINE_AA)
    for x,y,_,_ in red_points: cv2.circle(marker_overlay,(int(round(x)),int(round(y))),6,(0,0,0),1,cv2.LINE_AA)
    Image.fromarray(marker_overlay).save(diag_dir/"marker_candidates_overlay.png")

    manifest={
        "source_image_size_px":[w,h],"dominant_region_fill_colors":len(palette),
        "special_island_fill_colors":len(SPECIAL_COLORS),"administrative_polygon_components":len(region_components),
        "unique_region_codes":len(set(x[0] for x in CODE_LABELS)),"region_code_label_instances":len(label_features),
        "national_outline_components":len(outline_features),"river_line_segments":len(river_features),
        "lake_candidates":len(lake_features),"black_marker_candidates":len(black_points),
        "red_marker_candidates":len(red_points),"ocr_text_candidates":len(ocr_rows),
        "coordinate_system":"source image pixels; not georeferenced"
    }
    (out/"extraction_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
