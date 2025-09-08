#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit app to visualize Telangana Assembly Constituencies (ACs) as H3 grids
with options to select sampled ACs and control a margin-of-error slider.

Run:
  streamlit run India-VOTE.py
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import streamlit as st

import pydeck as pdk
import geopandas as gpd
# Prefer the pyogrio engine (wheels available, avoids GDAL/Fiona install issues)
try:  # noqa: SIM105
    import pyogrio  # type: ignore  # noqa: F401

    try:
        gpd.options.io_engine = "pyogrio"  # GeoPandas >=0.13
    except Exception:
        pass
except Exception:
    # pyogrio not installed; GeoPandas will fall back to default engine
    pass
from shapely.geometry import mapping, MultiPolygon, Polygon
import h3


# ----------------------
# Page setup
# ----------------------
# ----------------------
# Page setup
# ----------------------
st.set_page_config(page_title="Telangana ACs | H3 Sampler", layout="wide")
st.title("Telangana ACs — H3 Grid Sampler")
st.caption("Visualize ACs as H3 hexagons; pick sampled ACs and set margin of error.")
# Data loading
# ----------------------
@st.cache_data(show_spinner=False)
def load_telangana(paths: Iterable[str] = ("India_AC.geojson", "eci_check_reports/India_AC.geojson")) -> "gpd.GeoDataFrame":
    last_err = None
    for p in paths:
        try:
            gdf = gpd.read_file(p)
            cols = {c.lower(): c for c in gdf.columns}
            # Resolve column names
            state_col = cols.get("st_name", cols.get("state", "ST_NAME"))
            dist_col = cols.get("dist_name", cols.get("district", "DIST_NAME"))
            acno_col = cols.get("ac_no", "AC_NO")
            acname_col = cols.get("ac_name", "AC_NAME")

            need = [state_col, dist_col, acno_col, acname_col, "geometry"]
            gdf = gdf[need].rename(
                columns={
                    state_col: "state",
                    dist_col: "district",
                    acno_col: "ac_no",
                    acname_col: "ac_name",
                }
            )
            # Primary filter: explicit TELANGANA by state name
            mask_tg = gdf["state"].astype(str).str.strip().str.upper() == "TELANGANA"
            tg = gdf[mask_tg].copy()

            # Fallback for older datasets: Telangana districts within ANDHRA PRADESH
            if tg.empty:
                telangana_dists = {
                    "ADILABAD",
                    "KARIMNAGAR",
                    "KHAMMAM",
                    "HYDERABAD",
                    "MAHBUBNAGAR",
                    "MEDAK",
                    "NALGONDA",
                    "NIZAMABAD",
                    "RANGAREDDI",
                    "WARANGAL",
                }
                state_is_ap = gdf["state"].astype(str).str.strip().str.upper() == "ANDHRA PRADESH"
                dist_is_tg = gdf["district"].astype(str).str.strip().str.upper().isin(telangana_dists)
                tg = gdf[state_is_ap & dist_is_tg].copy()
            tg["ac_no"] = pd.to_numeric(tg["ac_no"], errors="coerce").astype(int)
            tg.sort_values("ac_no", inplace=True)
            tg.reset_index(drop=True, inplace=True)
            if not tg.empty:
                return tg
            # If empty, try next path
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(f"India_AC.geojson not found or no Telangana ACs present. Last error: {last_err}")


tg = load_telangana()


# ----------------------
# H3 helpers
# ----------------------
def geom_to_geojson_coords(geom: Polygon | MultiPolygon) -> Dict:
    return mapping(geom)


@st.cache_data(show_spinner=False)
def polyfill_ac_hexes(tg_wkb: List[bytes], res: int) -> List[List[str]]:
    # We cache on WKB bytes to avoid large shapely objects in cache key
    out: List[List[str]] = []

    def polyfill_geojson(geojson_obj, resolution: int) -> List[str]:
        # Support h3 v3 and v4 APIs
        if hasattr(h3, "polyfill"):
            return list(h3.polyfill(geojson_obj, resolution, geo_json_conformant=True))
        # v4 path: prefer geo_to_cells which accepts __geo_interface__ or dict
        if hasattr(h3, "geo_to_cells"):
            return list(h3.geo_to_cells(geojson_obj, resolution))
        # Fallback: build H3Shape then convert
        if hasattr(h3, "geo_to_h3shape") and hasattr(h3, "h3shape_to_cells"):
            shape = h3.geo_to_h3shape(geojson_obj)
            return list(h3.h3shape_to_cells(shape, resolution))
        raise RuntimeError("h3 library missing polyfill/geo_to_cells APIs")

    for w in tg_wkb:
        geom = gpd.GeoSeries.from_wkb([w]).iloc[0]
        gj = geom_to_geojson_coords(geom)
        hexes = polyfill_geojson(gj, res)
        out.append(hexes)
    return out


# ----------------------
# Reference CSV + sampling helpers
# ----------------------
@st.cache_data(show_spinner=False)
def load_parts_reference_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8")
    # Normalize expected columns
    if "acNumber" in df.columns:
        df["acNumber"] = pd.to_numeric(df["acNumber"], errors="coerce").astype("Int64")
    if "partNumber" in df.columns:
        df["partNumber"] = pd.to_numeric(df["partNumber"], errors="coerce").astype("Int64")
    for col in ["acLabel", "districtName", "partName", "districtCd"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def pick_residues(k: int, m: int, rng: np.random.Generator) -> List[int]:
    k = max(0, min(int(k), int(m)))
    if m <= 0 or k <= 0:
        return []
    return sorted(rng.choice(np.arange(m), size=k, replace=False).tolist())


def systematic_pick_indices(n: int, residues: List[int], m: int) -> np.ndarray:
    if m <= 0 or n <= 0 or not residues:
        return np.array([], dtype=int)
    idx = np.arange(n)
    mask = np.isin(idx % m, residues)
    return idx[mask]


# ----------------------
# Hex data loader (one hex per AC)
# ----------------------
@st.cache_data(show_spinner=False)
def load_telangana_hex(paths: Iterable[str] = ("India_AC_hex.geojson", "eci_check_reports/India_AC_hex.geojson")) -> "gpd.GeoDataFrame":
    last_err = None
    for p in paths:
        try:
            gdf = gpd.read_file(p)
            cols = {c.lower(): c for c in gdf.columns}
            st_col = cols.get("st_name", "ST_NAME")
            dist_col = cols.get("dist_name", cols.get("district", "DIST_NAME"))
            acno_col = cols.get("ac_no", "AC_NO")
            acname_col = cols.get("ac_name", "AC_NAME")
            need = [st_col, dist_col, acno_col, acname_col, "geometry"]
            gdf = gdf[need].rename(columns={st_col: "state", dist_col: "district", acno_col: "ac_no", acname_col: "ac_name"})

            # Filter to Telangana; fallback for older AP dataset with TG districts
            mask_tg = gdf["state"].astype(str).str.strip().str.upper() == "TELANGANA"
            hex_tg = gdf[mask_tg].copy()
            if hex_tg.empty:
                telangana_dists = {
                    "ADILABAD","KARIMNAGAR","KHAMMAM","HYDERABAD","MAHBUBNAGAR",
                    "MEDAK","NALGONDA","NIZAMABAD","RANGAREDDI","WARANGAL",
                }
                state_is_ap = gdf["state"].astype(str).str.strip().str.upper() == "ANDHRA PRADESH"
                dist_is_tg = gdf["district"].astype(str).str.strip().str.upper().isin(telangana_dists)
                hex_tg = gdf[state_is_ap & dist_is_tg].copy()

            hex_tg["ac_no"] = pd.to_numeric(hex_tg["ac_no"], errors="coerce").astype(int)
            hex_tg.sort_values("ac_no", inplace=True)
            hex_tg.reset_index(drop=True, inplace=True)
            if not hex_tg.empty:
                return hex_tg
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(f"India_AC_hex.geojson not found or no Telangana rows present. Last error: {last_err}")

# ----------------------
# Sidebar: Sampling Planner + map options
# ----------------------
with st.sidebar:
    st.header("Sampling Planner")
    level = st.selectbox("Level of analysis", ["State", "PC", "AC", "Mandal", "Booth"], index=2)
    # Fixed 95% confidence
    z = 1.96
    moe_pct = st.slider("Target margin of error (+/-% )", min_value=1, max_value=15, value=3, step=1)
    # Hidden defaults
    p_est = 0.5
    N_input = 0
    resp_rate = st.number_input("Expected response rate (0-1)", min_value=0.10, max_value=1.0, value=0.80, step=0.05, format="%.2f")
    cluster_size = st.number_input("Interviews per booth (cluster size m)", min_value=1, max_value=500, value=20, step=1)
    deff = st.slider("Design effect (DEFF)", min_value=1.0, max_value=2.0, value=1.5, step=0.05)

    st.markdown("Example cohorts (proportion of population). Add/edit rows as needed.")
    default_rows = pd.DataFrame([
        {"cohort": "Women", "proportion": 0.50},
        {"cohort": "Urban", "proportion": 0.40},
    ])
    cohorts_df = st.data_editor(default_rows, num_rows="dynamic", use_container_width=True, key="cohorts_editor_main")
    apply_cohort_constraint = st.checkbox("Ensure MOE holds within smallest cohort", value=True,
                                          help="Upscales total sample so even the smallest cohort meets the target MOE")
    cycles = st.number_input("Fieldwork cycles", min_value=1, max_value=12, value=1, step=1,
                             help="If sampling is spread across cycles, shows per-cycle cluster counts")

    st.divider()
    st.subheader("Map Display")
    show_only_sampled = st.checkbox("Show only sampled ACs", value=False)
    overlay_outline = st.checkbox("Show AC boundary outline", value=True)

# ----------------------
# Main layout
# ----------------------
def color_for(sampled: bool, moe: int) -> List[int]:
    base = [70, 130, 180] if not sampled else [233, 133, 37]
    a = int(np.interp(moe, [0, 20], [90, 230]))
    return [base[0], base[1], base[2], a]

# Sample size calculation (from sidebar inputs)
e = max(1e-9, float(moe_pct) / 100.0)
p = min(0.999999, max(1e-6, float(p_est)))
n0 = (z*z) * p * (1.0 - p) / (e*e)
n_deff = deff * n0
n_fpc = (n_deff / (1.0 + (n_deff - 1.0) / float(N_input))) if int(N_input) > 0 else n_deff
n_resp = n_fpc / max(1e-6, float(resp_rate))

n_scaled = n_resp
try:
    props = [float(x) for x in cohorts_df.get("proportion", pd.Series(dtype=float)).fillna(0.0).tolist() if float(x) > 0]
    if apply_cohort_constraint and props:
        p_min = min(props)
        if p_min > 0:
            n_scaled = max(n_resp, n_resp / p_min)
except Exception:
    pass

n_final = int(np.ceil(n_scaled))
clusters_total = int(np.ceil(n_final / float(cluster_size)))
clusters_per_cycle = int(np.ceil(clusters_total / float(cycles)))

# Main layout: map on left, sampling controls/results on right
left_map, right_controls = st.columns([2, 1])

with right_controls:
    st.header("AC/Booth Sampling (from Reference CSV)")
    default_csv_path = r"eci_check_reports\part_list_S29_20250907_202919.csv"
    csv_path = st.text_input("Reference CSV path", value=default_csv_path, help="Output from get-part-list.py")

    col_ac, col_pt, col_seed = st.columns([1, 1, 1])
    with col_ac:
        st.subheader("AC sampling")
        ac_k = st.slider("AC: k out of 5 (m fixed)", min_value=0, max_value=5, value=1, step=1)
        ac_m = 5
    with col_pt:
        st.subheader("Part (Booth) sampling")
        pt_m = st.slider("Parts: 1 out of m (m up to 300)", min_value=1, max_value=300, value=10, step=1)
        pt_k = 1
    with col_seed:
        st.subheader("Reproducibility")
        seed_txt = st.text_input("Random seed", value="12345")
        rng = np.random.default_rng(None if seed_txt.strip()=="" else int(seed_txt))
        do_sample = st.button("Sample ACs & Parts", use_container_width=True)

    # Load CSV and sample when requested
    ref_df = pd.DataFrame()
    sampled_ac_numbers: List[int] = []
    parts_sample_df = pd.DataFrame()
    # Normalize Windows-style backslashes to forward slashes for Linux containers
    csv_path_norm = csv_path.replace("\\", "/")
    if csv_path.strip():
        try:
            ref_df = load_parts_reference_csv(csv_path_norm)
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
    if not ref_df.empty:
        total_parts = int(ref_df.shape[0])
        ac_series = ref_df.get("acNumber").dropna().astype(int)
        unique_acs = sorted(pd.unique(ac_series))
        st.caption(f"Reference contains {len(unique_acs)} ACs and {total_parts:,} parts/booths.")
        if do_sample:
            ac_res = pick_residues(ac_k, ac_m, rng)
            idx_keep = systematic_pick_indices(len(unique_acs), ac_res, ac_m)
            sampled_ac_numbers = [unique_acs[i] for i in idx_keep.tolist()]
            # parts within each AC
            rows = []
            for acno in sampled_ac_numbers:
                df_ac = ref_df[ref_df["acNumber"].astype("Int64") == int(acno)].copy()
                if "partNumber" in df_ac.columns:
                    df_ac.sort_values(["partNumber", "partName"], inplace=True)
                pt_res = pick_residues(pt_k, pt_m, rng)
                sel = systematic_pick_indices(df_ac.shape[0], pt_res, pt_m)
                if sel.size > 0:
                    rows.append(df_ac.iloc[sel].copy())
            parts_sample_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=ref_df.columns.tolist())
            # align to interviews
            try:
                booths_needed = int(np.ceil(n_final / float(cluster_size))) if int(cluster_size) > 0 else 0
            except Exception:
                booths_needed = 0
            if booths_needed > 0 and parts_sample_df.shape[0] > booths_needed:
                parts_sample_df = parts_sample_df.iloc[:booths_needed].copy()
            # Persist for map/metrics
            st.session_state["sample_acnos"] = set(int(x) for x in sampled_ac_numbers)
            st.session_state["sample_booth_count"] = int(parts_sample_df.shape[0])

    # Show sampled ACs table only on right
    if not ref_df.empty and len(sampled_ac_numbers) > 0:
        st.subheader("Sampled ACs")
        ac_info_cols = [c for c in ["acNumber", "acLabel", "districtName", "districtCd"] if c in ref_df.columns]
        if ac_info_cols:
            ac_list_df = (
                ref_df[ref_df["acNumber"].astype("Int64").isin(pd.Series(sampled_ac_numbers, dtype="Int64"))][ac_info_cols]
                .drop_duplicates().sort_values(by=["acNumber"]).reset_index(drop=True)
            )
            st.dataframe(ac_list_df, use_container_width=True)

with left_map:
    # Map metrics and visualization
    ac_count = len(st.session_state.get("sample_acnos", set()))
    part_count = int(st.session_state.get("sample_booth_count", 0))
    st.metric("Required sample (total)", f"{n_final:,} | ACs {ac_count} | Parts {part_count}")
    st.caption(f"Clusters: {clusters_total:,} (m={int(cluster_size)}), per-cycle: {clusters_per_cycle:,}")

    # Use India_AC_hex polygons for visualization
    hex_tg = load_telangana_hex()
    # Prefer ACs sampled from CSV; fall back to sidebar selection
    if "sample_acnos" in st.session_state and st.session_state.get("sample_acnos"):
        sample_acnos = set(int(x) for x in st.session_state.get("sample_acnos"))
    else:
        sample_acnos = set(tg.loc[list(sampled_indices), "ac_no"].astype(int).tolist()) if 'sampled_indices' in globals() else set()
    hex_tg["sampled"] = hex_tg["ac_no"].astype(int).isin(sample_acnos)
    hex_tg["moe"] = int(moe_pct)
    hex_tg["fill"] = [color_for(bool(s), int(moe_pct)) for s in hex_tg["sampled"].tolist()]
    hex_plot = hex_tg if not show_only_sampled else hex_tg[hex_tg["sampled"]]

    if hex_plot.empty:
        st.info("No hexes to display at current settings.")
    else:
        feats_hex = json.loads(hex_plot.to_json())
        bounds = hex_plot.total_bounds
        view_state = pdk.ViewState(
            longitude=float((bounds[0] + bounds[2]) / 2.0),
            latitude=float((bounds[1] + bounds[3]) / 2.0),
            zoom=6.5,
        )
        layers = [
            pdk.Layer(
                "GeoJsonLayer",
                feats_hex,
                get_fill_color="properties.fill",
                get_line_color=[255, 255, 255],
                line_width_min_pixels=0.5,
                pickable=True,
                auto_highlight=True,
            )
        ]
        if overlay_outline and not tg.empty:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    json.loads(tg.to_json()),
                    get_fill_color=[0, 0, 0, 0],
                    get_line_color=[0, 0, 0, 160],
                    line_width_min_pixels=1,
                    pickable=False,
                )
            )
        tooltip = {
            "html": "<b>AC {properties.ac_no}: {properties.ac_name}</b><br/>District: {properties.district}<br/>Sampled: {properties.sampled}<br/>MOE: +/-{properties.moe}%",
            "style": {"backgroundColor": "#1f2630", "color": "white"},
        }
        st.pydeck_chart(pdk.Deck(initial_view_state=view_state, layers=layers, tooltip=tooltip, map_style=None), use_container_width=True)

    # Below the map: show Sampled Parts table (if any)
    try:
        if not parts_sample_df.empty:
            st.subheader("Sampled Parts")
            cols_show = [c for c in ["acNumber", "acLabel", "districtName", "partNumber", "partName"] if c in parts_sample_df.columns]
            st.dataframe(parts_sample_df[cols_show].sort_values(["acNumber", "partNumber"]).reset_index(drop=True), use_container_width=True)
    except Exception:
        pass

    # Results tables removed on left; use right column tables only
