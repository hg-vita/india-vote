#!/usr/bin/env python3
import json
import math
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from shapely.geometry import shape, Point, Polygon, LineString, mapping
from shapely import prepared
try:
    from shapely.algorithms.polylabel import polylabel as _polylabel
except Exception:
    _polylabel = None
from pyproj import CRS, Transformer
import h3

U = np.array([
    [1.0, 0.0],
    [0.5,  math.sqrt(3)/2.0],
    [-0.5, math.sqrt(3)/2.0],
])

def hex_radius_from_apothem(s: float) -> float:
    return 2.0 * s / math.sqrt(3.0)

def hex_vertices_xy(cx: float, cy: float, s: float) -> List[Tuple[float, float]]:
    R = hex_radius_from_apothem(s)
    verts = []
    for k in range(6):
        theta = math.radians(60.0 * k)
        verts.append((cx + R*math.cos(theta), cy + R*math.sin(theta)))
    return verts

def hex_polygon_xy(cxy: Tuple[float, float], s: float) -> Polygon:
    cx, cy = cxy
    return Polygon(hex_vertices_xy(cx, cy, s))

def make_local_transform(lon_lat_pairs: List[Tuple[float, float]]):
    if not lon_lat_pairs:
        c_lon, c_lat = 78.0, 22.0
    else:
        lons, lats = zip(*lon_lat_pairs)
        c_lon, c_lat = float(np.mean(lons)), float(np.mean(lats))
    proj_metric = CRS.from_proj4(
        f"+proj=aeqd +lat_0={c_lat} +lon_0={c_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    proj_geo = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(proj_geo, proj_metric, always_xy=True)
    inv = Transformer.from_crs(proj_metric, proj_geo, always_xy=True)
    return fwd, inv

def geom_to_xy(geom, fwd: Transformer):
    gj = mapping(geom)
    def rec_coords(coords):
        if isinstance(coords[0], (float, int)):
            x, y = fwd.transform(coords[0], coords[1])
            return (x, y)
        return [rec_coords(c) for c in coords]
    gj["coordinates"] = rec_coords(gj["coordinates"])
    return shape(gj)

def geom_to_lonlat(geom, inv: Transformer):
    gj = mapping(geom)
    def rec_coords(coords):
        if isinstance(coords[0], (float, int)):
            lon, lat = inv.transform(coords[0], coords[1])
            return (lon, lat)
        return [rec_coords(c) for c in coords]
    gj["coordinates"] = rec_coords(gj["coordinates"])
    return shape(gj)

def first_hit_distance_along_ray(poly: Polygon, cxy: Tuple[float, float], v: np.ndarray, far: float) -> float:
    cx, cy = cxy
    end = (cx + far*v[0], cy + far*v[1])
    ray = LineString([cxy, end])
    dists = []
    inter = ray.intersection(poly.exterior)
    def collect_dist(g):
        if g.is_empty:
            return
        if g.geom_type == "Point":
            dists.append(Point(cxy).distance(g))
        elif g.geom_type in ("MultiPoint", "GeometryCollection"):
            for gg in g.geoms:
                if gg.geom_type == "Point":
                    dists.append(Point(cxy).distance(gg))
    collect_dist(inter)
    for hole in poly.interiors:
        inter_h = ray.intersection(hole)
        collect_dist(inter_h)
    if not dists:
        return far
    dpos = [d for d in dists if d > 1e-9]
    return min(dpos) if dpos else far

def apothem_upper_bound(poly_xy: Polygon, centroid_xy: Tuple[float, float]) -> float:
    if poly_xy.is_empty:
        return 0.0
    minx, miny, maxx, maxy = poly_xy.bounds
    far = 4.0 * math.hypot(maxx - minx, maxy - miny)
    dirs = [U[0], U[1], U[2], -U[0], -U[1], -U[2]]
    hits = []
    for v in dirs:
        d = first_hit_distance_along_ray(poly_xy, centroid_xy, v, far)
        hits.append(d)
    return float(max(0.0, min(hits)))

def max_inscribed_apothem(poly_xy: Polygon, cxy: Tuple[float, float], s_hi: float, iters: int = 22) -> float:
    if s_hi <= 0:
        return 0.0
    try:
        poly_chk = poly_xy.buffer(-1e-7)
        if poly_chk.is_empty:
            poly_chk = poly_xy
    except Exception:
        poly_chk = poly_xy
    pprep = prepared.prep(poly_chk)
    lo, hi = 0.0, float(s_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mid <= 0:
            lo = mid
            continue
        hx = hex_polygon_xy(cxy, mid)
        if pprep.contains(hx):
            lo = mid
        else:
            hi = mid
    return lo

_H3_AVG_AREA_M2 = {
    0: 4.357e13, 1: 6.081e12, 2: 8.687e11, 3: 1.241e11,
    4: 1.773e10, 5: 2.533e9,  6: 3.619e8,  7: 5.170e7,
    8: 7.386e6,  9: 1.055e6, 10: 150_000.0, 11: 21_400.0,
    12: 3_060.0, 13: 436.0,   14: 62.3,     15: 8.9
}

def best_h3_res_for_area(area_m2: float, res_min: int = 4, res_max: int = 13) -> int:
    best_res, best_err = res_min, float("inf")
    for r in range(res_min, res_max + 1):
        a = _H3_AVG_AREA_M2[r]
        err = abs(a - area_m2)
        if err < best_err:
            best_err, best_res = err, r
    return best_res

def _h3_point_to_cell(lat: float, lon: float, res: int) -> str:
    try:
        return h3.geo_to_h3(lat, lon, res)
    except AttributeError:
        return h3.latlng_to_cell(lat, lon, res)

def _h3_boundary(h: str):
    if hasattr(h3, "h3_to_geo_boundary"):
        return h3.h3_to_geo_boundary(h, geo_json=True)
    if hasattr(h3, "cell_to_boundary"):
        coords = h3.cell_to_boundary(h)
        return [[lon, lat] for (lat, lon) in coords]
    raise AttributeError("No suitable H3 boundary function found on 'h3' module")

def main():
    ap = argparse.ArgumentParser(description="Greedy non-overlap hex selection and per-row H3 mapping.")
    ap.add_argument("--in", dest="infile", required=True, help="Input GeoJSON")
    ap.add_argument("--out_hex", required=True, help="Output hexagons GeoJSON (selected subset)")
    ap.add_argument("--out_h3", required=True, help="Output H3 GeoJSON for selected rows")
    ap.add_argument("--verbose", action="store_true", help="Print timings and summaries")
    ap.add_argument("--min_apothem_m", type=float, default=0.0, help="Discard candidates below this apothem")
    args = ap.parse_args()

    t0 = time.time()
    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    features = data["features"]
    if args.verbose:
        print(f"[load] Features: {len(features)} from {args.infile}")

    all_lonlat: List[Tuple[float, float]] = []
    polys_ll: List[Polygon] = []
    orig_idx_for_poly: List[int] = []
    for fi, f in enumerate(features):
        geom = shape(f["geometry"])
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys_ll.append(geom)
            orig_idx_for_poly.append(fi)
            all_lonlat.extend(list(geom.exterior.coords))
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                polys_ll.append(g)
                orig_idx_for_poly.append(fi)
                all_lonlat.extend(list(g.exterior.coords))

    t1 = time.time()
    fwd, inv = make_local_transform(all_lonlat)
    polys_xy: List[Polygon] = [geom_to_xy(p, fwd) for p in polys_ll]
    t2 = time.time()
    if args.verbose:
        print(f"[proj] Projected {len(polys_xy)} polygons in {t2 - t1:.2f}s")

    centers_ll: List[Tuple[float, float]] = []
    for pll in polys_ll:
        try:
            if _polylabel is not None:
                pt = _polylabel(pll, tolerance=1e-6)
                centers_ll.append((pt.x, pt.y))
            else:
                rp = pll.representative_point()
                centers_ll.append((rp.x, rp.y))
        except Exception:
            c = pll.centroid
            centers_ll.append((c.x, c.y))

    centers_xy: List[Tuple[float, float]] = []
    for lon, lat in centers_ll:
        x, y = fwd.transform(lon, lat)
        centers_xy.append((x, y))

    s_upper_raw = [apothem_upper_bound(p, c) for p, c in zip(polys_xy, centers_xy)]
    s_max = [max_inscribed_apothem(p, c, s) for p, c, s in zip(polys_xy, centers_xy, s_upper_raw)]
    if args.min_apothem_m > 0:
        for i in range(len(s_max)):
            if s_max[i] < args.min_apothem_m:
                s_max[i] = 0.0
    t3 = time.time()
    if args.verbose:
        print(f"[bound] Refined inscribed apothems in {t3 - t2:.2f}s | positive={sum(1 for s in s_max if s>0)}/{len(s_max)}")

    order = sorted(range(len(s_max)), key=lambda i: s_max[i], reverse=True)

    acc_idx: List[int] = []
    acc_centers: List[Tuple[float, float]] = []
    acc_R: List[float] = []
    acc_hex_prepared: List[Any] = []
    grid: Dict[Tuple[int, int], List[int]] = {}
    max_R_acc = 0.0

    all_R = [hex_radius_from_apothem(s) if s > 0 else 0.0 for s in s_max]
    max_R_global = max(all_R) if all_R else 0.0
    cell_size = max(max_R_global, 1.0)

    def cell_of(pt: Tuple[float, float]) -> Tuple[int, int]:
        return (int(math.floor(pt[0] / cell_size)), int(math.floor(pt[1] / cell_size)))

    for i in order:
        s = s_max[i]
        if s <= 0:
            continue
        cxy = centers_xy[i]
        Ri = hex_radius_from_apothem(s)
        if not acc_idx:
            hx = hex_polygon_xy(cxy, s)
            acc_idx.append(i)
            acc_centers.append(cxy)
            acc_R.append(Ri)
            acc_hex_prepared.append(prepared.prep(hx))
            max_R_acc = max(max_R_acc, Ri)
            cell = cell_of(cxy)
            grid.setdefault(cell, []).append(len(acc_idx)-1)
            continue

        k = int(math.ceil((Ri + max_R_acc) / cell_size))
        if k < 1:
            k = 1
        cell = cell_of(cxy)
        neighbors: List[int] = []
        for dx in range(-k, k+1):
            for dy in range(-k, k+1):
                neighbors.extend(grid.get((cell[0]+dx, cell[1]+dy), []))

        conflict = False
        hx = None
        for j in neighbors:
            cj = acc_centers[j]
            Rj = acc_R[j]
            dx = cxy[0] - cj[0]
            dy = cxy[1] - cj[1]
            if dx*dx + dy*dy >= (Ri + Rj)*(Ri + Rj):
                continue
            if hx is None:
                hx = hex_polygon_xy(cxy, s)
            if acc_hex_prepared[j].intersects(hx):
                conflict = True
                break
        if conflict:
            continue

        if hx is None:
            hx = hex_polygon_xy(cxy, s)
        acc_idx.append(i)
        acc_centers.append(cxy)
        acc_R.append(Ri)
        acc_hex_prepared.append(prepared.prep(hx))
        max_R_acc = max(max_R_acc, Ri)
        grid.setdefault(cell, []).append(len(acc_idx)-1)

    hex_features: List[Dict[str, Any]] = []
    total_area_m2 = 0.0
    selected_rows: List[Tuple[int, Polygon, Tuple[float, float], float, float, Dict[str, Any]]] = []
    for k_idx, i in enumerate(acc_idx):
        cxy = acc_centers[k_idx]
        s = s_max[i]
        verts_xy = hex_vertices_xy(cxy[0], cxy[1], s)
        hex_xy = Polygon(verts_xy)
        area = float(hex_xy.area)
        total_area_m2 += area
        hex_ll = geom_to_lonlat(hex_xy, inv)
        src_props = dict(features[orig_idx_for_poly[i]].get("properties", {}))
        hex_features.append({
            "type": "Feature",
            "properties": {**src_props, **{
                "id": i,
                "apothem_m": s,
                "side_m": (2.0/math.sqrt(3.0))*s,
                "area_m2": area
            }},
            "geometry": mapping(hex_ll)
        })
        selected_rows.append((i, polys_ll[i], centers_ll[i], s, area, src_props))

    Path(args.out_hex).write_text(json.dumps({"type": "FeatureCollection","features": hex_features}), encoding="utf-8")
    if args.verbose:
        total_area_km2 = total_area_m2 / 1e6
        print(f"[hex] Wrote {args.out_hex} with {len(hex_features)} hexagons | total_area={total_area_km2:.2f} km^2 | elapsed {time.time() - t0:.2f}s")
    else:
        print(f"Wrote {args.out_hex} with {len(hex_features)} hexagons")

    if len(hex_features) == 0:
        Path(args.out_h3).write_text(json.dumps({"type": "FeatureCollection","features":[]}), encoding="utf-8")
        print("No selected hexagons; wrote empty H3 output.")
        return

    def refine_h3_cover(pll: Polygon, center_ll: Tuple[float, float], start_res: int, min_res: int = 4) -> Tuple[int, str, list, bool]:
        res = int(start_res)
        last_h, last_boundary = None, None
        while res >= min_res:
            h = _h3_point_to_cell(center_ll[1], center_ll[0], res)
            boundary = _h3_boundary(h)
            hex_poly_ll = Polygon(boundary + [boundary[0]])
            if pll.within(hex_poly_ll.buffer(1e-12)):
                return res, h, boundary, True
            last_h, last_boundary = h, boundary
            res -= 1
        return max(min_res, 0), (last_h or ""), (last_boundary or []), False

    h3_feats = []
    refined_down = 0
    containment_floor_hits = 0
    for idx, (orig_idx, poly_ll, center_ll, s, area_m2, src_props) in enumerate(selected_rows):
        start_res = best_h3_res_for_area(area_m2, res_min=4, res_max=13)
        res_final, h, boundary, contained = refine_h3_cover(poly_ll, center_ll, start_res, min_res=4)
        if res_final < start_res:
            refined_down += 1
        if not contained:
            containment_floor_hits += 1
        poly = {"type": "Polygon", "coordinates": [boundary + ([boundary[0]] if boundary else [])]}
        props = {**src_props, **{
            "row_id": orig_idx,
            "h3": h,
            "h3_res": res_final,
            "hex_apothem_m": s,
            "hex_area_m2": area_m2,
        }}
        h3_feats.append({"type": "Feature", "properties": props, "geometry": poly})

    Path(args.out_h3).write_text(json.dumps({"type":"FeatureCollection","features":h3_feats}), encoding="utf-8")
    if args.verbose:
        res_counts: Dict[int, int] = {}
        for f in h3_feats:
            r = int(f["properties"]["h3_res"])
            res_counts[r] = res_counts.get(r, 0) + 1
        res_summary = ", ".join(f"r{r}:{c}" for r, c in sorted(res_counts.items()))
        extra = f" | refined_down={refined_down}" + (f" | containment_at_floor={containment_floor_hits}" if containment_floor_hits else "")
        print(f"[h3] Wrote {args.out_h3} with {len(h3_feats)} H3 features | res mix: {res_summary}{extra}")
    else:
        print(f"Wrote {args.out_h3} with {len(h3_feats)} H3 features (selected subset)")

if __name__ == "__main__":
    main()

