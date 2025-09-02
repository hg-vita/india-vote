
import io
import os
import re
import glob
import json
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk
import h3
import geopandas as gpd
import numpy as np

# -----------------
# Page config
# -----------------
st.set_page_config(page_title="AC-wise Tracking Dashboard", page_icon="🗳️", layout="wide")
st.title("🗳️ AC-wise Tracking Dashboard")
st.caption("Load your `aggregate_summary.csv` and explore Assembly Constituency (AC) level metrics.")

# -----------------
# Helpers
# -----------------
AC_NAME_CANDIDATES = [
    "ac", "ac_name", "assembly_constituency", "assemblyconstituency", "acname",
    "assembly_constituency_name", "ac_title", "constituency_name", "aclabel"
]

AC_NUMBER_CANDIDATES = [
    "ac_no", "ac_number", "acnum", "acno", "assembly_constituency_no", "constituency_no", "ac_code", "acnumber"
]

DISTRICT_CANDIDATES = ["district", "district_name", "districtcd"]
STATE_CANDIDATES = ["state", "state_name", "st_name", "statename", "statecd"]

DATE_LIKE = ["last_updated", "updated_at", "created_at", "crawl_date", "timestamp", "generated_at"]

BOOTH_COUNT_CANDIDATES = [
    "booths","booth_count","num_booths",
    "parts","parts_count","part_count","no_of_parts","total_parts","total_booths","parts_total"
]

TOTAL_FILES_CANDIDATES = [
    "files_total","total_files","expected_files","booth_files_total","total_booth_files","total_pdfs","pdfs_total"
]
AVAILABLE_FILES_CANDIDATES = [
    "files_available","available_files","booth_files_available","available_booth_files",
    "pdfs_found","files_found","pdfs_available"
]
MISSING_FILES_CANDIDATES = [
    "files_missing","missing_files","booth_files_missing","missing_booth_files","pdfs_missing","missing_count"
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r'[^a-zA-Z0-9_]+', '_', c.strip()).lower().strip("_") for c in df.columns]
    return df

def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def combine_ac_key(df: pd.DataFrame, ac_no_col: Optional[str], ac_name_col: Optional[str]) -> pd.Series:
    if ac_no_col and ac_name_col:
        def fmt_no(x):
            try:
                xi = int(pd.to_numeric(x, errors='coerce'))
                return f"{xi:03d}"
            except Exception:
                return str(x)
        return df[ac_no_col].map(fmt_no).astype(str) + " - " + df[ac_name_col].astype(str)
    if ac_name_col:
        return df[ac_name_col].astype(str)
    if ac_no_col:
        return df[ac_no_col].astype(str)
    return df.iloc[:, 0].astype(str)

def detect_date_cols(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        if any(k in c for k in DATE_LIKE) or c.endswith("_at") or "date" in c:
            out.append(c)
    return out

def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def agg_numeric(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        likely_numeric = [c for c in df.columns if any(k in c for k in ["count", "total", "sum", "size", "male", "female", "pdf", "error", "elector", "parts","booth","file"])]
        for c in likely_numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return df[group_cols].drop_duplicates().assign(__no_numeric_cols__=True)
    g = df.groupby(group_cols, dropna=False)[num_cols].sum(min_count=1).reset_index()
    return g

def pick_default_metric(df: pd.DataFrame) -> Optional[str]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    preferred = ["electors", "total_electors", "total_voters", "parts_count", "parts_total", "booth_count", "pdfs_found", "files_found", "files_missing"]
    for p in preferred:
        if p in num_cols:
            return p
    return num_cols[0] if num_cols else None

def detect_booth_column(df: pd.DataFrame) -> Optional[str]:
    for c in BOOTH_COUNT_CANDIDATES:
        if c in df.columns:
            return c
    return None

def detect_file_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    total = first_existing(df, TOTAL_FILES_CANDIDATES)
    avail = first_existing(df, AVAILABLE_FILES_CANDIDATES)
    miss = first_existing(df, MISSING_FILES_CANDIDATES)
    return total, avail, miss

def _detect_ac_cols_gdf(gdf: 'gpd.GeoDataFrame'):
    cols = [c.lower() for c in gdf.columns]
    num_cands = ["ac_no","acno","acnum","acnumber","ac_code","const_no"]
    name_cands = ["ac_name","acname","aclabel","constituency","ac","ac_title","assembly","acname_1"]
    num_col = None
    name_col = None
    for c in num_cands:
        if c in cols:
            num_col = gdf.columns[cols.index(c)]
            break
    for c in name_cands:
        if c in cols:
            name_col = gdf.columns[cols.index(c)]
            break
    return num_col, name_col

def _color_from_pct(p):
    try:
        p = max(0.0, min(100.0, float(p)))
    except Exception:
        p = 0.0
    r = int(255 * (p / 100.0))
    g = int(255 * (1 - (p / 100.0)))
    b = 60
    return [r, g, b]

# -----------------
# Data loading
# -----------------
st.sidebar.header("Load data")
opt = st.sidebar.radio("How would you like to load data?", ["Upload CSV", "Use path"], index=1, horizontal=True)

df: Optional[pd.DataFrame] = None
source_desc = ""

if opt == "Upload CSV":
    up = st.sidebar.file_uploader("Upload aggregate_summary.csv", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            source_desc = f"Uploaded file: `{up.name}`"
        except Exception:
            up.seek(0)
            try:
                df = pd.read_csv(up, encoding="utf-8-sig")
                source_desc = f"Uploaded file: `{up.name}` (utf-8-sig)"
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
else:
    default_path = st.sidebar.text_input("CSV path", value="eci_check_reports/aggregate_summary.csv")
    path_candidates = [
        default_path,
        default_path.replace("/", "\\"),
        "aggregate_summary.csv",
        "/mnt/data/aggregate_summary.csv",
    ]
    loaded = False
    for pth in path_candidates:
        if os.path.exists(pth):
            try:
                df = pd.read_csv(pth)
                source_desc = f"Loaded from path: `{pth}`"
                loaded = True
                break
            except Exception as e:
                st.error(f"Failed to read CSV at {pth}: {e}")
                break
    if not loaded:
        if st.sidebar.button("Load from path"):
            if os.path.exists(default_path):
                df = pd.read_csv(default_path)
                source_desc = f"Loaded from path: `{default_path}`"
            else:
                st.error(f"File not found: {default_path}")

if df is None:
    st.info("👆 Load your CSV to begin. The app auto-detects AC columns and available metrics.")
    st.stop()

df = normalize_cols(df)
st.success(f"Data loaded • {source_desc}")
st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
with st.expander("Preview columns", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# -----------------
# Column detection
# -----------------
ac_name_col = first_existing(df, AC_NAME_CANDIDATES)
ac_no_col = first_existing(df, AC_NUMBER_CANDIDATES)
district_col = first_existing(df, DISTRICT_CANDIDATES)
state_col = first_existing(df, STATE_CANDIDATES)
date_cols = detect_date_cols(df)

df = df.copy()
df["ac_key"] = combine_ac_key(df, ac_no_col, ac_name_col)

# Searchable AC name (by name)
if ac_name_col and ac_name_col in df.columns:
    df["__ac_name_search__"] = df[ac_name_col].astype(str)
else:
    df["__ac_name_search__"] = df["ac_key"].astype(str).str.split(" - ", n=1).str[-1]

# Booth/files columns
booth_col = detect_booth_column(df)
total_files_col, available_files_col, missing_files_col = detect_file_columns(df)
ensure_numeric(df, [c for c in [booth_col, total_files_col, available_files_col, missing_files_col] if c])

# -----------------
# Sidebar filters
# -----------------
st.sidebar.header("Filters")
if state_col:
    states = ["(All)"] + sorted([x for x in df[state_col].dropna().astype(str).unique()])
    sel_state = st.sidebar.selectbox("State", states, index=0)
    if sel_state != "(All)":
        df = df[df[state_col].astype(str) == sel_state]

if district_col:
    dists = ["(All)"] + sorted([x for x in df[district_col].dropna().astype(str).unique()])
    sel_dist = st.sidebar.selectbox("District", dists, index=0)
    if sel_dist != "(All)":
        df = df[df[district_col].astype(str) == sel_dist]

# AC search by name + select
search_q = st.sidebar.text_input("🔎 Search AC (by name)", value="")
acs_all = df[["ac_key", "__ac_name_search__"]].dropna().drop_duplicates()
if search_q.strip():
    mask = acs_all["__ac_name_search__"].str.contains(search_q.strip(), case=False, na=False)
    acs_filtered = acs_all[mask]
else:
    acs_filtered = acs_all
ac_options = ["(All)"] + acs_filtered["ac_key"].astype(str).sort_values().tolist()
sel_ac = st.sidebar.selectbox("Select AC", ac_options, index=0, help="Choose an AC to focus. Use the search box above to narrow results.")
if sel_ac != "(All)":
    df = df[df["ac_key"] == sel_ac]

# -----------------
# Group AC-wise
# -----------------
group_cols = ["ac_key"]
gdf = agg_numeric(df, group_cols)

if "__no_numeric_cols__" in gdf.columns:
    st.warning("No numeric columns detected to aggregate. Showing distinct ACs only.")
    st.dataframe(gdf[group_cols].drop_duplicates(), use_container_width=True)
    st.stop()

# Map aggregated columns
ag_total = first_existing(gdf, TOTAL_FILES_CANDIDATES) or ("files_total" if "files_total" in gdf.columns else None)
ag_avail = first_existing(gdf, AVAILABLE_FILES_CANDIDATES) or ("files_available" if "files_available" in gdf.columns else None)
ag_miss = first_existing(gdf, MISSING_FILES_CANDIDATES) or ("files_missing" if "files_missing" in gdf.columns else None)
ag_booth = detect_booth_column(gdf)

if ag_miss is None and ag_total and ag_avail:
    gdf["files_missing"] = pd.to_numeric(gdf[ag_total], errors="coerce") - pd.to_numeric(gdf[ag_avail], errors="coerce")
    ag_miss = "files_missing"
if ag_total is None and ag_avail and ag_miss:
    gdf["files_total"] = pd.to_numeric(gdf[ag_avail], errors="coerce") + pd.to_numeric(gdf[ag_miss], errors="coerce")
    ag_total = "files_total"

# -----------------
# Tabs
# -----------------
tab_overview, tab_ac_detail, tab_duplicates, tab_h3 = st.tabs(["Overview", "AC Detail", "Duplicates", "H3 Map"])

with tab_overview:
    # Metric selection
    num_cols = [c for c in gdf.columns if c not in group_cols]
    default_metric = pick_default_metric(gdf)
    if default_metric and default_metric in num_cols:
        default_idx = num_cols.index(default_metric)
    else:
        default_idx = 0 if num_cols else 0
    metric = st.selectbox("Primary metric", options=num_cols, index=default_idx if num_cols else 0)

    # KPIs
    left, mid, right = st.columns(3)
    with left:
        st.metric("Total ACs (after filters)", f"{gdf['ac_key'].nunique():,}")
    with mid:
        st.metric(f"Sum of {metric}", f"{pd.to_numeric(gdf[metric], errors='coerce').sum():,.0f}")
    with right:
        # Overall Missing % across filtered data
        _sum_avail = pd.to_numeric(gdf[ag_avail], errors="coerce").sum() if ag_avail else 0.0
        _sum_miss  = pd.to_numeric(gdf[ag_miss], errors="coerce").sum() if ag_miss else 0.0
        _sum_total = pd.to_numeric(gdf[ag_total], errors="coerce").sum() if ag_total else (_sum_avail + _sum_miss)
        _pct_missing = ((_sum_miss / _sum_total) * 100.0) if _sum_total else 0.0
        st.metric("Overall Missing %", f"{_pct_missing:.2f}%")

    # Freshness
    if date_cols:
        for dc in date_cols:
            try:
                df[dc] = pd.to_datetime(df[dc], errors="coerce")
            except Exception:
                pass
        best_dc = max(date_cols, key=lambda c: df[c].notna().sum())
        min_dt = pd.to_datetime(df[best_dc]).min()
        max_dt = pd.to_datetime(df[best_dc]).max()
        st.caption(f"Data freshness: **{best_dc}** from **{pd.to_datetime(min_dt)}** to **{pd.to_datetime(max_dt)}**")

    st.divider()

    # Missing % pie (Available vs Missing) on home
    if sel_ac != "(All)":
        row = gdf[gdf["ac_key"] == sel_ac]
        if not row.empty:
            r = row.iloc[0]
            v_avail = float(r.get(ag_avail, 0)) if ag_avail else None
            v_miss = float(r.get(ag_miss, 0)) if ag_miss else None
            v_total = float(r.get(ag_total, 0)) if ag_total else None
        else:
            v_avail = v_miss = v_total = None
    else:
        v_avail = pd.to_numeric(gdf[ag_avail], errors="coerce").sum() if ag_avail else None
        v_miss = pd.to_numeric(gdf[ag_miss], errors="coerce").sum() if ag_miss else None
        v_total = pd.to_numeric(gdf[ag_total], errors="coerce").sum() if ag_total else None

    if v_miss is None and (v_total is not None and v_avail is not None):
        v_miss = max(v_total - v_avail, 0.0)
    if v_total is None and (v_avail is not None and v_miss is not None):
        v_total = v_avail + v_miss

    if v_avail is None and v_miss is None and v_total is None:
        st.info("No file availability columns detected to compute Missing %.")
    else:
        denom = v_total if (v_total and v_total > 0) else ((v_avail or 0.0) + (v_miss or 0.0))
        p_avail = (v_avail or 0.0) / denom * 100 if denom else 0.0
        p_miss = (v_miss or 0.0) / denom * 100 if denom else 0.0
        pie_df = pd.DataFrame({"status": ["Available", "Missing"], "percent": [p_avail, p_miss]})
        pie = (
            alt.Chart(pie_df)
            .mark_arc()
            .encode(theta="percent:Q", color="status:N", tooltip=["status:N", alt.Tooltip("percent:Q", format=".2f")])
            .properties(width=350, height=350, title="Booth Files: Missing %")
        )
        st.altair_chart(pie, use_container_width=False)

    # Top ACs chart
    st.subheader("Top ACs by metric")
    top_n = st.slider("Show top N", min_value=5, max_value=50, value=20, step=1)
    chart_data = gdf.sort_values(metric, ascending=False).head(top_n)
    bar = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("ac_key:N", sort='-y', title="AC"),
            y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
            tooltip=[alt.Tooltip("ac_key:N", title="AC"), alt.Tooltip(f"{metric}:Q", title=metric.replace("_", " ").title(), format=",.0f")],
        )
        .properties(height=400)
    )
    st.altair_chart(bar, use_container_width=True)

    st.subheader("AC-wise table")
    st.dataframe(gdf.sort_values(metric, ascending=False).reset_index(drop=True), use_container_width=True)

    csv_buf = io.StringIO()
    gdf.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download AC-wise CSV", data=csv_buf.getvalue(), file_name="ac_wise_summary.csv", mime="text/csv")

with tab_ac_detail:
    st.subheader("AC Detail")
    st.caption("Select an AC from the left sidebar to see focused numbers.")
    if sel_ac != "(All)":
        row = gdf[gdf["ac_key"] == sel_ac].copy()
        if row.empty:
            st.warning("No aggregated row for the selected AC.")
        else:
            r = row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("AC", r["ac_key"])
            with c2:
                st.metric("Number of booths", f"{int(r.get(ag_booth, 0)) if (ag_booth and pd.notna(r.get(ag_booth, 0))) else 0:,}" if ag_booth else "N/A")
            with c3:
                st.metric("Files available", f"{int(r.get(ag_avail, 0)) if (ag_avail and pd.notna(r.get(ag_avail, 0))) else 0:,}" if ag_avail else "N/A")
            with c4:
                st.metric("Files missing", f"{int(r.get(ag_miss, 0)) if (ag_miss and pd.notna(r.get(ag_miss, 0))) else 0:,}" if ag_miss else "N/A")

            # Roll summary from first-page JSONs
            st.markdown("### Roll summary (from first-page JSON files)")
            default_dl = os.path.expanduser("~/Downloads")
            default_dl_win = os.path.join(os.path.expanduser("~"), "Downloads")
            dl_path = st.text_input(
                "Folder containing files like 'AC_128_Part_2_first_page_english.json'",
                value=default_dl if os.path.isdir(default_dl) else default_dl_win
            )
            ac_num_selected = None
            try:
                ac_num_selected = int(str(sel_ac).split(" - ")[0].strip())
            except Exception:
                ac_num_selected = None

            parts_rows = []
            if os.path.isdir(dl_path):
                pattern = os.path.join(dl_path, "**", "*first_page_english.json")
                for fp in glob.iglob(pattern, recursive=True):
                    try:
                        with open(fp, "r", encoding="utf-8") as fjson:
                            data = json.load(fjson)
                        er = data.get("electoral_roll_2025", {})
                        ac_no = er.get("assembly_constituency", {}).get("number")
                        try:
                            ac_no_int = int(str(ac_no))
                        except Exception:
                            ac_no_int = None
                        if ac_num_selected is not None and ac_no_int != ac_num_selected:
                            continue
                        part_no = er.get("part_number")
                        elec = er.get("number_of_electors", {})
                        start_serial = elec.get("starting_serial", {}).get("number")
                        end_serial = elec.get("ending_serial", {}).get("number")
                        male = elec.get("male", 0) or 0
                        female = elec.get("female", 0) or 0
                        third = elec.get("third_gender", 0) or 0
                        total = elec.get("total", 0) or 0
                        parts_rows.append({
                            "part_number": part_no,
                            "first_serial": start_serial,
                            "last_serial": end_serial,
                            "male": male,
                            "female": female,
                            "third_gender": third,
                            "total": total,
                            "file": os.path.basename(fp),
                        })
                    except Exception:
                        continue

            if parts_rows:
                parts_df = pd.DataFrame(parts_rows).sort_values("part_number").reset_index(drop=True)
                sum_booths = int(parts_df["part_number"].nunique())
                sum_male = int(pd.to_numeric(parts_df["male"], errors="coerce").fillna(0).sum())
                sum_female = int(pd.to_numeric(parts_df["female"], errors="coerce").fillna(0).sum())
                sum_total = int(pd.to_numeric(parts_df["total"], errors="coerce").fillna(0).sum())
                cs1, cs2, cs3, cs4 = st.columns(4)
                with cs1: st.metric("Booths (from JSON)", f"{sum_booths:,}")
                with cs2: st.metric("Male", f"{sum_male:,}")
                with cs3: st.metric("Female", f"{sum_female:,}")
                with cs4: st.metric("Total voters", f"{sum_total:,}")
                st.dataframe(parts_df[["part_number","first_serial","last_serial","total","male","female","third_gender","file"]], use_container_width=True)
                csv2 = parts_df.to_csv(index=False)
                st.download_button("⬇️ Download part-wise table (JSON roll)", data=csv2, file_name=f"ac_{ac_num_selected}_part_summary.csv", mime="text/csv")
            else:
                st.info("No matching first-page JSON files found in the folder for the selected AC.")
    else:
        st.info("Select a specific AC to view details.")

with tab_duplicates:
    st.subheader("Duplicates")
    st.caption("This sheet is reserved for duplicate detection (coming soon). It is empty for now.")
    placeholder = pd.DataFrame(columns=["ac_key", "booth_id", "reason"])
    st.dataframe(placeholder, use_container_width=True)

with tab_h3:
    st.subheader("🗺️ H3 Map: Missing Booth Files % by AC")
    st.sidebar.header("H3 map data")
    shp_default = st.sidebar.text_input(
        "Boundary file path (.geojson/.shp)",
        value="eci_check_reports/India_AC.geojson",
        help="Path to AC boundaries (GeoJSON preferred; Shapefile also works)"
    )
    only_bihar = st.sidebar.checkbox("Filter map to Bihar", value=True)

    shp_candidates = [shp_default, shp_default.replace("/", "\\"), "/mnt/data/India_AC.geojson", "/mnt/data/India_AC.shp"]
    gdf_shp = None
    for sp in shp_candidates:
        if os.path.exists(sp):
            try:
                gdf_shp = gpd.read_file(sp)
                break
            except Exception as e:
                st.error(f"Failed to read boundary file at {sp}: {e}")
                break
    if gdf_shp is None:
        st.info("Provide a valid shapefile/geojson path in the sidebar to render the H3 map.")
        st.stop()

    try:
        if gdf_shp.crs is None:
            st.warning("Boundary file has no CRS; assuming EPSG:4326. Set CRS in your source data if incorrect.")
            gdf_shp.set_crs(4326, inplace=True)
        else:
            gdf_shp = gdf_shp.to_crs(4326)
    except Exception as e:
        st.error(f"Failed to project boundaries to EPSG:4326: {e}")
        st.stop()

    # Optional: filter to Bihar
    if only_bihar:
        state_col_geo = None
        for c in STATE_CANDIDATES:
            match = [col for col in gdf_shp.columns if col.lower() == c]
            if match:
                state_col_geo = match[0]
                break
        if state_col_geo:
            gdf_shp = gdf_shp[gdf_shp[state_col_geo].astype(str).str.contains("Bihar", case=False, na=False)]
            st.caption(f"Showing only Bihar (rows after filter: {len(gdf_shp)})")
        else:
            st.warning("Could not find a state column in boundaries; showing all states.")

    # Detect AC cols
    shp_num_col, shp_name_col = _detect_ac_cols_gdf(gdf_shp)
    if shp_num_col is None and shp_name_col is None:
        st.error("Could not detect AC number/name columns in boundaries. Expected something like AC_NO/AC_NAME.")
        st.stop()

    # CSV side: build join keys & explicit acLabel
    df_csv = df.copy()
    csv_num = first_existing(df_csv, ["acnumber","ac_no","acnum","acno"])
    csv_name = first_existing(df_csv, ["aclabel","ac_name","acname","assembly_constituency","ac"])
    has_aclabel = "aclabel" in df_csv.columns

    def _num_from_label(val):
        if pd.isna(val):
            return None
        s = str(val)
        try:
            return int(s.split(" - ")[0].strip())
        except Exception:
            try:
                return int(pd.to_numeric(s, errors="coerce"))
            except Exception:
                return None

    df_csv["__ac_no_csv__"] = None
    if csv_num:
        df_csv["__ac_no_csv__"] = pd.to_numeric(df_csv[csv_num], errors="coerce").astype("Int64")
    elif csv_name:
        df_csv["__ac_no_csv__"] = df_csv[csv_name].map(_num_from_label).astype("Int64")

    def _name_from_label(val):
        if pd.isna(val):
            return None
        s = str(val)
        if " - " in s:
            return s.split(" - ", 1)[1].strip()
        return s.strip()

    # Preferred AC label for map: acLabel from CSV if present, else derived
    if has_aclabel:
        df_csv["__ac_label_for_map__"] = df_csv["aclabel"].astype(str)
    else:
        if csv_name:
            df_csv["__ac_label_for_map__"] = df_csv[csv_name].map(_name_from_label).astype(str)
        elif csv_num:
            df_csv["__ac_label_for_map__"] = df_csv[csv_num].astype(str)
        else:
            df_csv["__ac_label_for_map__"] = df_csv["ac_key"].astype(str)

    # Missing % from CSV
    df_csv["files_total"] = pd.to_numeric(df_csv.get("files_found", 0), errors="coerce").fillna(0) + pd.to_numeric(df_csv.get("missing_count", 0), errors="coerce").fillna(0)
    df_csv["missing_pct"] = (pd.to_numeric(df_csv.get("missing_count", 0), errors="coerce").fillna(0) / df_csv["files_total"].replace(0, pd.NA)) * 100

    # Shapefile join keys
    gdf_shp["__ac_no_shp__"] = None
    if shp_num_col:
        gdf_shp["__ac_no_shp__"] = pd.to_numeric(gdf_shp[shp_num_col], errors="coerce").astype("Int64")
    gdf_shp["__ac_name_shp__"] = None
    if shp_name_col:
        gdf_shp["__ac_name_shp__"] = gdf_shp[shp_name_col].astype(str).str.strip()

    # Join by number then fallback to name (case-insensitive), keeping acLabel from CSV
    joined = None
    if "__ac_no_shp__" in gdf_shp.columns and df_csv["__ac_no_csv__"].notna().any():
        joined = gdf_shp.merge(
            df_csv[["__ac_no_csv__","missing_pct","__ac_label_for_map__"]],
            left_on="__ac_no_shp__", right_on="__ac_no_csv__", how="left"
        )
    if joined is None or joined["missing_pct"].isna().all():
        gdf_shp["__name_l__"] = gdf_shp["__ac_name_shp__"].str.lower() if "__ac_name_shp__" in gdf_shp.columns else None
        df_csv["__name_l__"] = df_csv["__ac_label_for_map__"].str.lower()
        joined = gdf_shp.merge(
            df_csv[["__name_l__","missing_pct","__ac_label_for_map__"]],
            on="__name_l__", how="left"
        )

    # Centroids
    joined = joined.set_geometry("geometry")
    try:
        centroids = joined.geometry.centroid
    except Exception:
        joined["geometry"] = joined.buffer(0)
        centroids = joined.geometry.centroid
    joined["lat"] = centroids.y
    joined["lon"] = centroids.x

    # H3 resolution slider (default=6)
    RES = st.slider("H3 resolution", min_value=4, max_value=8, value=6, step=1, help="Lower=coarser, Higher=finer")

    # Compute H3 index
    joined["h3"] = joined.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lon"], RES) if pd.notna(r["lat"]) and pd.notna(r["lon"]) else None, axis=1)

    # Aggregate by H3 cell (mean missing % and concat labels from CSV acLabel)
    h3_df = joined.dropna(subset=["h3"]).groupby("h3", as_index=False).agg(
        missing_pct=("missing_pct","mean"),
        acLabel=("__ac_label_for_map__", lambda s: ", ".join(pd.Series(s).dropna().astype(str).unique()[:3]))
    )
    h3_df["missing_pct"] = h3_df["missing_pct"].fillna(0)
    # ---- Color gradient controls ----
    st.sidebar.subheader("H3 styling")
    _scheme = st.sidebar.selectbox("Color scheme", ["Green → Red", "Viridis"], index=0)
    _norm = st.sidebar.selectbox("Normalize colors by", ["0–100%", "Data min–max", "5–95% quantiles"], index=2)
    vals = h3_df["missing_pct"].dropna()
    if _norm == "0–100%":
        vmin, vmax = 0.0, 100.0
    elif _norm == "Data min–max" and not vals.empty:
        vmin, vmax = float(vals.min()), float(vals.max())
    elif not vals.empty:
        vmin, vmax = float(vals.quantile(0.05)), float(vals.quantile(0.95))
    else:
        vmin, vmax = 0.0, 100.0
    if vmin == vmax:
        vmin, vmax = max(0.0, vmin - 1.0), min(100.0, vmax + 1.0)
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    if _scheme == "Green → Red":
        colors = [
            [0, 200, 0],      # green
            [173, 255, 47],  # green-yellow
            [255, 215, 0],   # gold
            [255, 140, 0],   # dark orange
            [220, 20, 60],   # crimson
        ]
    else:  # Viridis-like
        colors = [
            [68, 1, 84],     # dark purple
            [59, 82, 139],   # blue
            [33, 144, 141],  # teal
            [94, 201, 98],   # green
            [253, 231, 37],  # yellow
        ]
    def _lerp(a, b, t):
        return a + (b - a) * t
    def _interp_color(val):
        try:
            v = float(val)
        except Exception:
            v = 0.0
        # normalize to [0,1] using vmin/vmax
        t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        t = 0.0 if t < 0 else (1.0 if t > 1 else t)
        # find segment
        for i in range(len(stops)-1):
            if stops[i] <= t <= stops[i+1]:
                local_t = (t - stops[i]) / (stops[i+1] - stops[i])
                c0, c1 = colors[i], colors[i+1]
                r = int(_lerp(c0[0], c1[0], local_t))
                g = int(_lerp(c0[1], c1[1], local_t))
                b = int(_lerp(c0[2], c1[2], local_t))
                return [r, g, b]
        return colors[-1]
    h3_df["fill_color"] = h3_df["missing_pct"].apply(_interp_color)
    h3_df["elevation"] = h3_df["missing_pct"].fillna(0) * 10.0

    # Center and zoom from bounds
    if joined["lat"].notna().any() and joined["lon"].notna().any():
        lat_min, lat_max = float(joined["lat"].min()), float(joined["lat"].max())
        lon_min, lon_max = float(joined["lon"].min()), float(joined["lon"].max())
        lat0 = (lat_min + lat_max) / 2.0
        lon0 = (lon_min + lon_max) / 2.0
        lat_span = max(0.0001, lat_max - lat_min)
        lon_span = max(0.0001, lon_max - lon_min)
        span = max(lat_span, lon_span)
        if span > 20:
            zoom0 = 4.0
        elif span > 10:
            zoom0 = 5.0
        elif span > 5:
            zoom0 = 6.0
        elif span > 2:
            zoom0 = 7.0
        elif span > 1:
            zoom0 = 8.0
        else:
            zoom0 = 9.0
    else:
        lat0, lon0, zoom0 = 23.5, 80.9, 5.0

    # Layer & render
    layer = pdk.Layer(
        "H3HexagonLayer",
        data=h3_df,
        pickable=True,
        get_hexagon="h3",
        get_fill_color="fill_color",
        get_elevation="elevation",
        elevation_scale=1,
        extruded=True,
    )
    tooltip = {"html": "<b>Missing %:</b> {missing_pct}<br/><b>AC:</b> {acLabel}", "style": {"backgroundColor": "steelblue", "color": "white"}}
    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom0, pitch=20)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style=None))

    # Histogram of Missing %
    if not h3_df.empty:
        st.subheader("Distribution of Missing % (by H3 hex)")
        _hist = (
            alt.Chart(h3_df)
            .mark_bar()
            .encode(
                x=alt.X("missing_pct:Q", bin=alt.Bin(step=5), title="Missing %"),
                y=alt.Y("count():Q", title="Hex count"),
                tooltip=[alt.Tooltip("count():Q", title="Hexes")]
            )
            .properties(height=240)
        )
        st.altair_chart(_hist, use_container_width=True)


    st.caption("H3 labels use acLabel from aggregate_summary (when present). Missing % is from aggregate_summary (files_found & missing_count) aggregated by hex.")
