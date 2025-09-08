#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Part→Village Matcher

Features
- AC-first candidate subset (falls back to district if AC id missing/mismatch)
- Hybrid scoring:
    * Name similarity (RapidFuzz WRatio)
    * Neighbor continuity by partNumber (contiguous parts often same/near village)
    * Optional distance similarity (activates if you provide booth/part lat/lon)
- Many parts → one village allowed (common and intended)
- CSV/Excel auto-detection and optional delimiter/sheet/encoding controls
- Flexible CLI overrides for column names

Dependencies:
  pip install pandas rapidfuzz openpyxl
  (For legacy .xls: pip install xlrd==1.2.0)
"""

import argparse
import os
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from math import radians, sin, cos, asin, sqrt
from typing import Optional, List


# ==============================
# Tunable weights & knobs
# ==============================
W_NAME = 0.55            # weight for name similarity [0..1]
W_DIST = 0.30            # weight for distance similarity [0..1] (0 if you don't pass coordinates)
W_NEIGHBOR = 0.15        # weight for neighbor continuity by part order [0..1]
TOP_K = 10               # number of village name candidates to evaluate per part
NEIGHBOR_RADIUS_KM = 2.0 # full neighbor bonus within this distance of previous chosen village
DIST_SCALE_KM = 5.0      # distance shaping scale (~0.5 similarity around this distance)


# ==============================
# IO helpers
# ==============================
def load_tabular(path: str,
                 sep: Optional[str] = None,
                 sheet_name: Optional[str | int] = None,
                 encoding: Optional[str] = None) -> pd.DataFrame:
    """
    Load CSV or Excel by extension.
    - For CSV: if sep is None, use sep=None + engine='python' to sniff delimiter.
      (Do NOT pass low_memory to python engine.)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".txt", ".tsv"}:
        read_kwargs = {}
        if encoding:
            read_kwargs["encoding"] = encoding
        if sep is None:
            # Delimiter sniffing requires python engine; no low_memory here
            return pd.read_csv(path, sep=None, engine="python", **read_kwargs)
        else:
            return pd.read_csv(path, sep=sep, **read_kwargs)
    elif ext in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return pd.read_excel(path, engine="openpyxl", sheet_name=sheet_name or 0)
    elif ext in {".xls"}:
        # For legacy .xls: pip install xlrd==1.2.0
        return pd.read_excel(path, engine="xlrd", sheet_name=sheet_name or 0)
    else:
        raise ValueError(f"Unsupported file extension for {path!r}. Use CSV or Excel.")


# ==============================
# Utility functions
# ==============================
def norm(s: Optional[str]) -> Optional[str]:
    if pd.isna(s):
        return None
    return (str(s).strip()
            .replace("–", "-").replace("—", "-")
            .replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .upper())


def to_int_safe(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return pd.NA


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


def dist_to_similarity_km(km: float, k: float = DIST_SCALE_KM) -> float:
    """Monotonically decreasing; ~0.5 at k km."""
    if pd.isna(km):
        return 0.0
    return 1.0 / (1.0 + km / k)


# ==============================
# Core matching
# ==============================
def fuzzy_name_candidates(query_norm: str,
                          candidate_norm_names: List[str],
                          limit: int = TOP_K):
    """RapidFuzz WRatio top-k (returns tuples: (name_norm, score_raw, index))."""
    return process.extract(query_norm, candidate_norm_names, scorer=fuzz.WRatio, limit=limit)


def match_ac_group(ac_parts_df: pd.DataFrame,
                   ac_villages_df: pd.DataFrame,
                   part_lat_col: Optional[str] = None,
                   part_lon_col: Optional[str] = None) -> pd.DataFrame:
    """
    Match parts to villages within one AC group.

    ac_parts_df required columns:
        partNumber, partName, partName_norm
    optionally (for distance):
        <part_lat_col>, <part_lon_col>

    ac_villages_df required columns:
        village_name, village_name_norm
    optional:
        village_latitude, village_longitude

    Returns: DataFrame with
        partNumber, partName, matched_village,
        score_total, score_name, score_dist, score_neighbor
    """
    vill_names_norm = ac_villages_df["village_name_norm"].fillna("").tolist()
    vill_names_orig = ac_villages_df["village_name"].tolist()
    vill_lat = ac_villages_df["village_latitude"].tolist() if "village_latitude" in ac_villages_df.columns else [np.nan] * len(vill_names_orig)
    vill_lon = ac_villages_df["village_longitude"].tolist() if "village_longitude" in ac_villages_df.columns else [np.nan] * len(vill_names_orig)

    # Sort by part number to enforce sequential continuity
    ac_parts_df = ac_parts_df.sort_values("partNumber").reset_index(drop=True)

    prev_choice_idx = None
    prev_choice_lat = None
    prev_choice_lon = None

    out_rows = []

    for _, row in ac_parts_df.iterrows():
        pnum = row["partNumber"]
        pname = row["partName"]
        pname_norm = row["partName_norm"] or ""

        booth_lat = row[part_lat_col] if part_lat_col and part_lat_col in ac_parts_df.columns else np.nan
        booth_lon = row[part_lon_col] if part_lon_col and part_lon_col in ac_parts_df.columns else np.nan

        # 1) Name shortlist
        name_cands = fuzzy_name_candidates(pname_norm, vill_names_norm, limit=TOP_K)

        candidate_scored = []
        for cand_name_norm, sim_raw, pos in name_cands:
            sim_name = (sim_raw or 0) / 100.0

            v_idx = pos
            v_name = vill_names_orig[v_idx]
            vlat = vill_lat[v_idx]
            vlon = vill_lon[v_idx]

            # 2) Distance similarity (only if booth coords provided & village has coords)
            if not pd.isna(booth_lat) and not pd.isna(booth_lon) and not pd.isna(vlat) and not pd.isna(vlon):
                km = haversine_km(booth_lat, booth_lon, vlat, vlon)
                sim_dist = dist_to_similarity_km(km)
            else:
                sim_dist = 0.0

            # 3) Neighbor continuity bonus
            neighbor_bonus = 0.0
            if prev_choice_idx is not None:
                if v_idx == prev_choice_idx:
                    neighbor_bonus = 1.0
                elif not pd.isna(vlat) and not pd.isna(vlon) and not pd.isna(prev_choice_lat) and not pd.isna(prev_choice_lon):
                    dkm = haversine_km(prev_choice_lat, prev_choice_lon, vlat, vlon)
                    if not pd.isna(dkm):
                        if dkm <= NEIGHBOR_RADIUS_KM:
                            neighbor_bonus = 1.0
                        else:
                            neighbor_bonus = max(0.0, 1.0 - (dkm / (NEIGHBOR_RADIUS_KM * 3)))

            score = W_NAME * sim_name + W_DIST * sim_dist + W_NEIGHBOR * neighbor_bonus
            candidate_scored.append((v_idx, v_name, sim_name, sim_dist, neighbor_bonus, score))

        if candidate_scored:
            candidate_scored.sort(key=lambda x: x[-1], reverse=True)
            best_idx, best_name, s_name, s_dist, s_neighbor, s_total = candidate_scored[0]

            prev_choice_idx = best_idx
            prev_choice_lat = vill_lat[best_idx]
            prev_choice_lon = vill_lon[best_idx]

            out_rows.append({
                "partNumber": pnum,
                "partName": pname,
                "matched_village": best_name,
                "score_total": round(float(s_total), 4),
                "score_name": round(float(s_name), 4),
                "score_dist": round(float(s_dist), 4),
                "score_neighbor": round(float(s_neighbor), 4),
            })
        else:
            out_rows.append({
                "partNumber": pnum,
                "partName": pname,
                "matched_village": None,
                "score_total": np.nan,
                "score_name": np.nan,
                "score_dist": np.nan,
                "score_neighbor": np.nan,
            })

    return pd.DataFrame(out_rows)


def run_hybrid_match(
    parts_path: str,
    ma_path: str,
    out_csv_path: str,
    # Optional column overrides (parts)
    parts_ac_col: str = "acNumber",
    parts_district_col: str = "districtName",
    parts_partnum_col: str = "partNumber",
    parts_partname_col: str = "partName",
    parts_lat_col: Optional[str] = None,
    parts_lon_col: Optional[str] = None,
    parts_state_col: Optional[str] = "stateCd",
    parts_ac_label_col: Optional[str] = "acLabel",
    # Optional column overrides (MA)
    ma_village_col: str = "village_name",
    ma_village_lat_col: str = "village_latitude",
    ma_village_lon_col: str = "village_longitude",
    ma_district_col: str = "district_name",
    ma_ac_col: Optional[str] = None,     # e.g., "ac_code" (auto-detected if None)
    ma_state_col: Optional[str] = "state_name",
    ma_ac_name_col: Optional[str] = None,  # e.g., "ac_name" if available
    # IO options
    parts_sep: Optional[str] = None,
    parts_encoding: Optional[str] = None,
    ma_sep: Optional[str] = None,
    ma_sheet: Optional[str | int] = None,
    ma_encoding: Optional[str] = None
):
    # ---- Read files (auto)
    parts_raw = load_tabular(parts_path, sep=parts_sep, encoding=parts_encoding)
    ma_raw = load_tabular(ma_path, sep=ma_sep, sheet_name=ma_sheet, encoding=ma_encoding)

    # ---- Autodetect MA AC col if not provided
    if ma_ac_col is None:
        candidates = [c for c in ma_raw.columns if c.lower() in
                      ["ac_code", "acnumber", "ac_no", "ac_id", "assembly_constituency_code"]]
        ma_ac_col = candidates[0] if candidates else None

    # ---- Validate required columns
    for col in [parts_ac_col, parts_district_col, parts_partnum_col, parts_partname_col]:
        if col not in parts_raw.columns:
            raise ValueError(f"Part list missing required column: {col!r}")
    if ma_village_col not in ma_raw.columns:
        raise ValueError(f"MA file missing required column: {ma_village_col!r}")

    # ---- Prepare parts frame
    parts = (parts_raw[[parts_ac_col, parts_district_col, parts_partnum_col, parts_partname_col] +
                       ([parts_lat_col] if parts_lat_col else []) +
                       ([parts_lon_col] if parts_lon_col else [])]
             .dropna(subset=[parts_ac_col, parts_partnum_col, parts_partname_col])
             .drop_duplicates())

    parts = parts.rename(columns={
        parts_ac_col: "acNumber",
        parts_district_col: "districtName",
        parts_partnum_col: "partNumber",
        parts_partname_col: "partName",
    })
    if parts_lat_col:
        parts = parts.rename(columns={parts_lat_col: "__booth_lat"})
        parts_lat_col = "__booth_lat"
    if parts_lon_col:
        parts = parts.rename(columns={parts_lon_col: "__booth_lon"})
        parts_lon_col = "__booth_lon"

    parts["partName_norm"] = parts["partName"].map(norm)
    parts["district_norm"] = parts["districtName"].map(norm)

    # ---- Prepare village frame
    vill_cols = [ma_village_col]
    if ma_state_col and ma_state_col in ma_raw.columns:
        vill_cols.append(ma_state_col)
    if ma_ac_name_col and ma_ac_name_col in ma_raw.columns:
        vill_cols.append(ma_ac_name_col)
    if ma_village_lat_col in ma_raw.columns:
        vill_cols.append(ma_village_lat_col)
    if ma_village_lon_col in ma_raw.columns:
        vill_cols.append(ma_village_lon_col)
    if ma_district_col in ma_raw.columns:
        vill_cols.append(ma_district_col)
    if ma_ac_col and ma_ac_col in ma_raw.columns:
        vill_cols.append(ma_ac_col)

    villages = ma_raw[vill_cols].drop_duplicates()

    # rename to internal names
    ren = {ma_village_col: "village_name"}
    if ma_village_lat_col in villages.columns:
        ren[ma_village_lat_col] = "village_latitude"
    if ma_village_lon_col in villages.columns:
        ren[ma_village_lon_col] = "village_longitude"
    if ma_district_col in villages.columns:
        ren[ma_district_col] = "district_name"
    if ma_ac_col and ma_ac_col in villages.columns:
        ren[ma_ac_col] = "_ac_src"
    if ma_state_col and ma_state_col in villages.columns:
        ren[ma_state_col] = "state_name"
    if ma_ac_name_col and ma_ac_name_col in villages.columns:
        ren[ma_ac_name_col] = "_ac_name_src"
    villages = villages.rename(columns=ren)

    villages["village_name_norm"] = villages["village_name"].map(norm)
    if "district_name" in villages.columns:
        villages["district_norm"] = villages["district_name"].map(norm)
    # Preferred AC assignment by names: (state, ac_name)
    # 1) Build a lookup from parts: (state_norm, ac_name_norm) -> most common acNumber
    parts_lookup = None
    try:
        def norm_state_from_parts(s):
            if pd.isna(s):
                return None
            v = str(s).strip()
            if len(v) >= 3 and v[0].upper() in {"S","U"} and v[1:].isdigit():
                code = v.upper()
                state_map = {
                    "U01":"ANDAMAN & NICOBAR ISLANDS","S01":"ANDHRA PRADESH","S02":"ARUNACHAL PRADESH","S03":"ASSAM","S04":"BIHAR",
                    "U02":"CHANDIGARH","S26":"CHATTISGARH","U03":"DADRA & NAGAR HAVELI AND DAMAN & DIU","S05":"GOA","S06":"GUJARAT",
                    "S07":"HARYANA","S08":"HIMACHAL PRADESH","U08":"JAMMU AND KASHMIR","S27":"JHARKHAND","S10":"KARNATAKA",
                    "S11":"KERALA","U09":"LADAKH","U06":"LAKSHADWEEP","S12":"MADHYA PRADESH","S13":"MAHARASHTRA",
                    "S14":"MANIPUR","S15":"MEGHALAYA","S16":"MIZORAM","S17":"NAGALAND","U05":"NCT OF DELHI",
                    "S18":"ODISHA","U07":"PUDUCHERRY","S19":"PUNJAB","S20":"RAJASTHAN","S21":"SIKKIM",
                    "S22":"TAMIL NADU","S29":"TELANGANA","S23":"TRIPURA","S24":"UTTAR PRADESH","S28":"UTTARAKHAND","S25":"WEST BENGAL",
                }
                return state_map.get(code)
            return str(s).strip()

        pr = parts_raw.copy()
        pr_cols = []
        if parts_state_col and parts_state_col in pr.columns:
            pr_cols.append(parts_state_col)
        if parts_ac_label_col and parts_ac_label_col in pr.columns:
            pr_cols.append(parts_ac_label_col)
        pr_cols.append(parts_ac_col)
        pr = pr[pr_cols].dropna(subset=[parts_ac_col]).copy()
        if parts_state_col and parts_state_col in pr.columns:
            pr["state_norm"] = pr[parts_state_col].map(norm_state_from_parts).map(norm)
        else:
            pr["state_norm"] = None
        if parts_ac_label_col and parts_ac_label_col in pr.columns:
            pr["ac_name_norm"] = pr[parts_ac_label_col].astype(str).str.split(" - ", 1).str[-1].map(norm)
        elif "ac_name" in pr.columns:
            pr["ac_name_norm"] = pr["ac_name"].map(norm)
        else:
            pr["ac_name_norm"] = None
        if "ac_name_norm" in pr.columns:
            pr = pr.dropna(subset=["ac_name_norm"]).copy()
            parts_lookup = (pr.groupby(["state_norm","ac_name_norm"])[parts_ac_col]
                            .agg(lambda s: pd.Series(s).astype("Int64").mode().iloc[0] if len(pd.Series(s).astype("Int64").mode())>0 else pd.NA)
                            .reset_index().rename(columns={parts_ac_col: "__ac_from_name"}))
    except Exception:
        parts_lookup = None

    if ma_state_col and "state_name" in villages.columns:
        villages["state_norm"] = villages["state_name"].map(norm)
    if ma_ac_name_col and "_ac_name_src" in villages.columns:
        villages["ac_name_norm"] = villages["_ac_name_src"].map(norm)

    villages["acNumber"] = pd.Series([pd.NA] * len(villages))
    if parts_lookup is not None and "state_norm" in villages.columns and "ac_name_norm" in villages.columns:
        villages = villages.merge(parts_lookup, on=["state_norm","ac_name_norm"], how="left")
        if "__ac_from_name" in villages.columns:
            try:
                villages["acNumber"] = villages["__ac_from_name"].astype("Int64")
            except Exception:
                villages["acNumber"] = villages["__ac_from_name"]
            villages.drop(columns=["__ac_from_name"], inplace=True)

    if villages["acNumber"].isna().any() and "_ac_src" in villages.columns:
        fill = villages.loc[villages["acNumber"].isna(), "_ac_src"].map(to_int_safe)
        villages.loc[villages["acNumber"].isna(), "acNumber"] = fill

    # ---- Per-AC matching
    all_matches = []
    ac_values = parts["acNumber"].dropna().unique()
    try:
        ac_values = pd.Series(ac_values).astype(int).unique().tolist()
    except Exception:
        ac_values = parts["acNumber"].dropna().unique().tolist()

    for ac in sorted(ac_values):
        ac_parts = parts[parts["acNumber"] == ac].copy()

        # Prefer AC subset; fallback to district
        ac_vill = pd.DataFrame()
        scope = None
        if "acNumber" in villages.columns:
            try:
                ac_vill = villages[villages["acNumber"].astype("Int64") == int(ac)].copy()
            except Exception:
                ac_vill = villages[villages["acNumber"] == ac].copy()

        if ac_vill.empty and "district_norm" in villages.columns:
            dists = sorted(ac_parts["district_norm"].dropna().unique().tolist())
            ac_vill = villages[villages["district_norm"].isin(dists)].copy()
            scope = "DISTRICT"
        else:
            scope = "AC"

        if ac_vill.empty:
            tmp = ac_parts[["acNumber", "districtName", "partNumber", "partName"]].copy()
            tmp["matched_village"] = None
            tmp["score_total"] = np.nan
            tmp["score_name"] = np.nan
            tmp["score_dist"] = np.nan
            tmp["score_neighbor"] = np.nan
            tmp["match_scope"] = None
        else:
            cols_needed = ["partNumber", "partName", "partName_norm"]
            if parts_lat_col and parts_lat_col in ac_parts.columns:
                cols_needed.append(parts_lat_col)
            if parts_lon_col and parts_lon_col in ac_parts.columns:
                cols_needed.append(parts_lon_col)

            matched_df = match_ac_group(
                ac_parts[cols_needed].copy().reset_index(drop=True),
                ac_villages_df=ac_vill.reset_index(drop=True),
                part_lat_col=parts_lat_col,
                part_lon_col=parts_lon_col
            )

            tmp = ac_parts[["acNumber", "districtName", "partNumber", "partName"]].merge(
                matched_df, on=["partNumber", "partName"], how="left"
            )
            tmp["match_scope"] = scope

        all_matches.append(tmp)

    mapping_df = pd.concat(all_matches, ignore_index=True)

    # diagnostics
    mapping_df["weight_name"] = W_NAME
    mapping_df["weight_dist"] = W_DIST
    mapping_df["weight_neighbor"] = W_NEIGHBOR
    mapping_df["top_k"] = TOP_K
    mapping_df["neighbor_radius_km"] = NEIGHBOR_RADIUS_KM
    mapping_df["dist_scale_km"] = DIST_SCALE_KM

    mapping_df.to_csv(out_csv_path, index=False)
    print(f"Saved: {out_csv_path}")


# ==============================
# CLI
# ==============================
def main():
    p = argparse.ArgumentParser(description="Hybrid match part_name → village_name within same AC (district fallback).")
    # Required IO
    p.add_argument("--parts", required=True, help="Path to part list CSV/Excel")
    p.add_argument("--ma", required=True, help="Path to MA CSV/Excel")
    p.add_argument("--out", required=True, help="Output CSV path")

    # Optional parts column overrides
    p.add_argument("--parts_ac_col", default="acNumber", help="Parts AC column (default: acNumber)")
    p.add_argument("--parts_district_col", default="districtName", help="Parts district column (default: districtName)")
    p.add_argument("--parts_partnum_col", default="partNumber", help="Parts part number column (default: partNumber)")
    p.add_argument("--parts_partname_col", default="partName", help="Parts part name column (default: partName)")
    p.add_argument("--parts_lat_col", default=None, help="Optional parts latitude column (booth/part)")
    p.add_argument("--parts_lon_col", default=None, help="Optional parts longitude column (booth/part)")
    p.add_argument("--parts_state_col", default="stateCd", help="Parts state column (default: stateCd; accepts SXX/UXY or name)")
    p.add_argument("--parts_ac_label_col", default="acLabel", help="Parts AC name label column (e.g., '7 - Adilabad')")

    # Optional MA column overrides
    p.add_argument("--ma_village_col", default="village_name", help="MA village name column (default: village_name)")
    p.add_argument("--ma_village_lat_col", default="village_latitude", help="MA village latitude column (default: village_latitude)")
    p.add_argument("--ma_village_lon_col", default="village_longitude", help="MA village longitude column (default: village_longitude)")
    p.add_argument("--ma_district_col", default="district_name", help="MA district column (default: district_name)")
    p.add_argument("--ma_ac_col", default=None, help="MA AC column (e.g., ac_code). If omitted, auto-detected.")
    p.add_argument("--ma_state_col", default="state_name", help="MA state name column (default: state_name)")
    p.add_argument("--ma_ac_name_col", default=None, help="MA AC name column (e.g., ac_name). If provided, AC will be matched by (state, ac name) first.")

    # IO aids
    p.add_argument("--parts_sep", default=None, help="CSV separator for --parts if CSV (e.g., ',', '\\t'). If omitted, auto-detect.")
    p.add_argument("--parts_encoding", default=None, help="Encoding for --parts if CSV (e.g., 'utf-8', 'latin1').")
    p.add_argument("--ma_sep", default=None, help="CSV separator for --ma if CSV (e.g., ',', '\\t'). If omitted, auto-detect.")
    p.add_argument("--ma_sheet", default=None, help="Excel sheet name/index for --ma if Excel")
    p.add_argument("--ma_encoding", default=None, help="Encoding for --ma if CSV (e.g., 'utf-8', 'latin1').")

    args = p.parse_args()

    run_hybrid_match(
        parts_path=args.parts,
        ma_path=args.ma,
        out_csv_path=args.out,
        parts_ac_col=args.parts_ac_col,
        parts_district_col=args.parts_district_col,
        parts_partnum_col=args.parts_partnum_col,
        parts_partname_col=args.parts_partname_col,
        parts_lat_col=args.parts_lat_col,
        parts_lon_col=args.parts_lon_col,
        parts_state_col=args.parts_state_col,
        parts_ac_label_col=args.parts_ac_label_col,
        ma_village_col=args.ma_village_col,
        ma_village_lat_col=args.ma_village_lat_col,
        ma_village_lon_col=args.ma_village_lon_col,
        ma_district_col=args.ma_district_col,
        ma_ac_col=args.ma_ac_col,
        ma_state_col=args.ma_state_col,
        ma_ac_name_col=args.ma_ac_name_col,
        parts_sep=args.parts_sep,
        parts_encoding=args.parts_encoding,
        ma_sep=args.ma_sep,
        ma_sheet=args.ma_sheet,
        ma_encoding=args.ma_encoding
    )


if __name__ == "__main__":
    main()
