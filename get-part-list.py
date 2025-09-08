#!/usr/bin/env python3
"""
Fetch part list for ALL ACs in a state directly via ECI APIs (no UI),
and export a single CSV with State/District/AC context plus Part Number,
Part Name, and all additional fields present in the API payload.
Usage:
  python get-part-list.py --state S29 [--roll-code g] [--lang ENG]
Notes:
  - Uses public endpoints:
      - /api/v1/common/districts/{stateCd}
      - /api/v1/common/acs/{districtCd}
      - /api/v1/rolls/get-part-list?acId=...&rollType=...&lang=...
  - Default rollType is 'g' (general/normal roll). Use --roll-code to override
    if you need final ('f'), supplement ('o'), or draft ('d') where applicable.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union
import requests
def info(*args):
    print(*args)
SAVE_ROOT = "eci_check_reports"
os.makedirs(SAVE_ROOT, exist_ok=True)
def ist_now_str() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(tz=ist).strftime("%Y%m%d_%H%M%S")
def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\\/:*?\"<>|]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", "_")
    return re.sub(r"[^\w\-\. ]+", "_", s).strip().replace(" ", "_")
BASE_PAGE = "https://voters.eci.gov.in/download-eroll"
BASE_HEADERS = {
    "Origin": "https://voters.eci.gov.in",
    "Referer": BASE_PAGE,
    "applicationname": "VSP",
    "channelidobo": "VSP",
    "platform-type": "ECIWEB",
    "CurrentRole": "citizen",
    "Accept": "*/*",
}
def http_json(sess: requests.Session, url: str, headers: Dict[str, str], params: Optional[Dict] = None, timeout: int = 60):
    r = sess.get(url, headers=headers, params=params or {}, timeout=timeout)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None
def list_districts(sess: requests.Session, state_cd: str) -> List[Dict]:
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    urls = [
        f"https://gateway-voters.eci.gov.in/api/v1/common/districts/{state_cd}",
        f"https://gateway-voters.eci.gov.in/api/v1/common/district/{state_cd}",
    ]
    for url in urls:
        data = http_json(sess, url, headers=hdrs)
        if not data:
            continue
        payload = data.get("payload") if isinstance(data, dict) else data
        if isinstance(payload, list) and payload:
            return payload
    return []
def list_district_names_from_constituencies(sess: requests.Session, state_cd: str) -> Dict[str, str]:
    """Fetch additional district names from the constituencies endpoint and build a map.
    Endpoint hinted by user: /api/v1/common/constituencies?stateCode=Sxx
    Returns a mapping of districtCd -> districtName (best effort).
    """
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    url = f"https://gateway-voters.eci.gov.in/api/v1/common/constituencies"
    data = http_json(sess, url, headers=hdrs, params={"stateCode": state_cd}, timeout=60)
    out: Dict[str, str] = {}
    if not data:
        return out
    payload = data.get("payload") if isinstance(data, dict) else data
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            dcd = str(
                row.get("districtCd")
                or row.get("districtCode")
                or row.get("district")
                or row.get("code")
                or ""
            ).strip().upper()
            dname = (row.get("districtName") or row.get("district") or row.get("name") or "").strip()
            if dcd and dname and dcd not in out:
                out[dcd] = dname
    return out
def list_district_names_from_page(sess: requests.Session, state_cd: str, verbose: bool = False) -> Dict[str, str]:
    """Parse the voters portal HTML to extract <select name="district"> options.
    This often contains a static list of district codes → names.
    """
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    page_url = f"{BASE_PAGE}?stateCode={state_cd}"
    try:
        r = sess.get(page_url, headers=hdrs, timeout=60)
        if r.status_code != 200:
            if verbose:
                info(f"HTML fetch failed HTTP {r.status_code} for {page_url}")
            return {}
        html = r.text or ""
        # Extract the district select block (handle both single and double quotes)
        m = re.search(r"<select[^>]*\bname\s*=\s*['\"]district['\"][^>]*>(.*?)</select>", html, flags=re.I | re.S)
        if not m:
            # Try alternative attribute name
            m = re.search(r"<select[^>]*\bname\s*=\s*['\"]districtCd['\"][^>]*>(.*?)</select>", html, flags=re.I | re.S)
        if not m:
            # Fallback: scan entire HTML for option tags that look like district codes
            if verbose:
                info("Could not locate <select name='district'> block in HTML; scanning <option> tags globally")
            option_pattern = re.compile(r"<option[^>]*\bvalue\s*=\s*['\"](S\d{4})['\"][^>]*>(.*?)</option>", flags=re.I | re.S)
            out: Dict[str, str] = {}
            for opt_val, opt_text in option_pattern.findall(html):
                code = (opt_val or "").strip().upper()
                name = re.sub(r"<[^>]+>", "", (opt_text or "")).strip()
                if not code or not name:
                    continue
                # Heuristic: only keep codes for the current state prefix
                if state_cd and code.startswith(state_cd.upper()):
                    out[code] = name
            if out:
                if verbose:
                    sample = list(out.items())[:10]
                    info(f"Parsed {len(out)} district <option> entries globally. Sample:")
                    for k, v in sample:
                        info(f"  {k}: {v}")
                return out
            if verbose:
                info("No district <option> entries found globally either")
            return {}
        block = m.group(1)
        out: Dict[str, str] = {}
        # Parse options within the located select
        for opt_val, opt_text in re.findall(r"<option[^>]*\bvalue\s*=\s*['\"]([^'\"]+)['\"][^>]*>(.*?)</option>", block, flags=re.I | re.S):
            code = (opt_val or "").strip()
            name = re.sub(r"<[^>]+>", "", (opt_text or "")).strip()
            if not code or not name or code == "":
                continue
            out[code.upper()] = name
        # Drop placeholder entries if present
        out.pop("", None)
        if verbose:
            sample = list(out.items())[:10]
            info(f"Parsed {len(out)} district options from HTML. Sample:")
            for k, v in sample:
                info(f"  {k}: {v}")
        return out
    except Exception:
        return {}
def list_district_names_via_selenium(state_cd: str, headless: bool = True, verbose: bool = False) -> Dict[str, str]:
    """Use Selenium to load the page and scrape the rendered district <select> options.
    This works even when the HTML is populated client-side.
    """
    try:
        # Lazy import so script can run without Selenium unless needed
        from selenium import webdriver  # type: ignore
        from selenium.webdriver.chrome.options import Options  # type: ignore
        from selenium.webdriver.common.by import By  # type: ignore
        import time as _t
    except Exception as e:
        if verbose:
            info("Selenium not available:", e)
        return {}
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1200,900")
    try:
        driver = webdriver.Chrome(options=opts)
    except Exception as e:
        if verbose:
            info("Could not start Chrome WebDriver:", e)
        return {}
    try:
        url = f"{BASE_PAGE}?stateCode={state_cd}"
        driver.get(url)
        # Give time for client-side rendering
        _t.sleep(1.0)
        # Try common names
        selects = []
        for css in ["select[name='district']", "select[name='districtCd']", "select#district", "select#districtCd", "select.form-select[name='district']"]:
            try:
                eles = driver.find_elements(By.CSS_SELECTOR, css)
                if eles:
                    selects.extend(eles)
            except Exception:
                pass
        # If still none, search by label proximity
        if not selects:
            try:
                label = driver.find_element(By.XPATH, "//label[contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'DISTRICT')]")
                sel = label.find_element(By.XPATH, "following::select[1]")
                selects.append(sel)
            except Exception:
                pass
        out: Dict[str, str] = {}
        for sel in selects:
            try:
                opts_elems = sel.find_elements(By.TAG_NAME, "option")
                for o in opts_elems:
                    code = (o.get_attribute("value") or "").strip().upper()
                    name = (o.text or "").strip()
                    if not code or not name or code == "":
                        continue
                    if code.startswith("S") and len(code) >= 5:
                        out[code] = name
            except Exception:
                continue
        if verbose:
            info(f"Selenium scraped {len(out)} district options")
            for k, v in list(out.items())[:10]:
                info(f"  {k}: {v}")
        return out
    finally:
        try:
            driver.quit()
        except Exception:
            pass
def fetch_constituencies_meta(sess: requests.Session, state_cd: str) -> Tuple[Dict[Union[str,int], Dict], Dict[Tuple[str, int], Dict], List[str]]:
    """Fetch full constituencies metadata and return lookup maps and key list.
    - by_acid: acId -> metadata dict
    - by_dt_ac: (districtCd, asmblyNo) -> metadata dict
    - keys: union of all keys across rows (to include as CSV columns)
    """
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    url = "https://gateway-voters.eci.gov.in/api/v1/common/constituencies"
    data = http_json(sess, url, headers=hdrs, params={"stateCode": state_cd}, timeout=60)
    by_acid: Dict[Union[str,int], Dict] = {}
    by_dt_ac: Dict[Tuple[str, int], Dict] = {}
    keys: List[str] = []
    seen = set()
    payload = data.get("payload") if isinstance(data, dict) else data
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            r = dict(row)
            # Normalize types
            try:
                if isinstance(r.get("asmblyNo"), str) and r["asmblyNo"].isdigit():
                    r["asmblyNo"] = int(r["asmblyNo"])
            except Exception:
                pass
            acid = r.get("acId")
            if acid is not None:
                by_acid[acid] = r
            dtcd = str(r.get("districtCd") or r.get("districtCode") or "").strip().upper()
            acno = r.get("asmblyNo")
            if dtcd and isinstance(acno, int):
                by_dt_ac[(dtcd, acno)] = r
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
    return by_acid, by_dt_ac, keys
def list_acs_for_district(sess: requests.Session, state_cd: str, district_cd: str) -> List[Tuple[int, str, Optional[Union[str,int]]]]:
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    url = f"https://gateway-voters.eci.gov.in/api/v1/common/acs/{district_cd}"
    data = http_json(sess, url, headers=hdrs)
    if not data:
        return []
    payload = data.get("payload") if isinstance(data, dict) else data
    out: List[Tuple[int, str, Optional[Union[str,int]]]] = []
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            ac_no = row.get("asmblyNo") or row.get("acNumber") or row.get("asmbly") or row.get("no")
            name = (row.get("asmblyName") or row.get("name") or "").strip()
            ac_id = row.get("acId") or row.get("id") or row.get("acID")
            try:
                ac_no = int(ac_no)
            except Exception:
                continue
            label = f"{ac_no} - {name}" if name else str(ac_no)
            out.append((ac_no, label, ac_id))
    out.sort(key=lambda t: t[0])
    return out
def fetch_parts_by_acid(sess: requests.Session, state_cd: str, ac_id: Optional[Union[str,int]], roll_code: str, lang: str = "ENG", verbose: bool = False, debug_dir: Optional[str] = None, context: Optional[Dict] = None) -> List[Dict]:
    if not ac_id:
        return []
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    url = "https://gateway-voters.eci.gov.in/api/v1/rolls/get-part-list"
    params = {"acId": ac_id, "rollType": roll_code, "lang": lang}
    if verbose:
        info("GET", url, params)
    data = http_json(sess, url, headers=hdrs, params=params, timeout=60)
    payload = data.get("payload") if isinstance(data, dict) else data
    out: List[Dict] = []
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            r = dict(row)
            pn = r.get("partNumber")
            if isinstance(pn, str) and pn.isdigit():
                r["partNumber"] = int(pn)
            name = (
                r.get("partName")
                or r.get("partname")
                or r.get("partNameEnglish")
                or r.get("partnameEnglish")
                or r.get("part_name")
                or r.get("name")
            )
            if name is not None:
                r["partName"] = str(name)
            out.append(r)
    if debug_dir and context is not None:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            fp = os.path.join(debug_dir, f"parts_acid_{safe_name(str(context.get('districtCd','')))}_AC{safe_name(str(context.get('acNumber','')))}_{safe_name(str(roll_code))}.json")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump({"params": params, "data": data}, f, ensure_ascii=False)
        except Exception:
            pass
    return out
def http_json_post(sess: requests.Session, url: str, headers: Dict[str, str], json_body: Dict, timeout: int = 60):
    r = sess.post(url, headers={**headers, "Content-Type": "application/json"}, json=json_body, timeout=timeout)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None
def fetch_parts_printing_paged(sess: requests.Session, state_cd: str, district_cd: str, ac_number: int, lang: str = "ENG", page_size: int = 1000, verbose: bool = False, debug_dir: Optional[str] = None) -> List[Dict]:
    """POST to printing-publish/get-part-list with paging, as seen in sample payload.
    Loops pages until an empty payload is returned.
    """
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    url = "https://gateway-voters.eci.gov.in/api/v1/printing-publish/get-part-list"
    page = 0
    all_rows: List[Dict] = []
    while True:
        body = {
            "stateCd": state_cd,
            "districtCd": district_cd,
            "acNumber": int(ac_number),
            "pageNumber": int(page),
            "pageSize": int(page_size),
        }
        if verbose:
            info("POST", url, body)
        data = http_json_post(sess, url, headers=hdrs, json_body=body, timeout=60)
        payload = data.get("payload") if isinstance(data, dict) else data
        if not isinstance(payload, list) or not payload:
            break
        # Normalize fields
        for row in payload:
            if not isinstance(row, dict):
                continue
            r = dict(row)
            pn = r.get("partNumber")
            if isinstance(pn, str) and pn.isdigit():
                r["partNumber"] = int(pn)
            name = (
                r.get("partName")
                or r.get("partname")
                or r.get("partNameEnglish")
                or r.get("partnameEnglish")
                or r.get("part_name")
                or r.get("name")
            )
            if name is not None:
                r["partName"] = str(name)
            all_rows.append(r)
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                fp = os.path.join(debug_dir, f"parts_print_paged_{safe_name(district_cd)}_AC{int(ac_number)}_p{page}.json")
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump({"body": body, "data": data}, f, ensure_ascii=False)
            except Exception:
                pass
        # If we received less than page_size, assume done
        if len(payload) < page_size:
            break
        page += 1
    return all_rows
def main():
    ap = argparse.ArgumentParser(description="Fetch ECI part list for all ACs in a state and export CSV")
    ap.add_argument("--state", dest="state_cd", default="S29", help="State code (e.g., S29)")
    ap.add_argument("--roll-code", dest="roll_code", default="g", help="rollType code: g (default), f, o, d, sir")
    ap.add_argument("--lang", dest="lang", default="ENG", help="Language code (ENG/HIN/etc.)")
    ap.add_argument("--verbose", action="store_true", help="Print debug info while fetching")
    ap.add_argument("--debug-dir", default="debug", help="Directory to save raw debug JSON responses")
    args = ap.parse_args()
    state_cd = (args.state_cd or "").strip().upper()
    roll_code = (args.roll_code or "g").strip().lower()
    lang = (args.lang or "ENG").strip().upper()
    verbose = bool(args.verbose)
    debug_dir = args.debug_dir or None
    sess = requests.Session()
    # Build district list (prefer DOM selector when API is empty)
    districts = list_districts(sess, state_cd)
    dist_name_map: Dict[str, str] = {}
    dist_codes: List[str] = []
    if not districts:
        # Try DOM parse first
        if verbose:
            info("District API returned 0; falling back to DOM selector parse...")
        html_names = list_district_names_from_page(sess, state_cd, verbose=verbose) or {}
        if not html_names:
            # Selenium fallback (headless)
            html_names = list_district_names_via_selenium(state_cd, headless=True, verbose=verbose) or {}
        if html_names:
            dist_name_map = dict(html_names)
            dist_codes = list(dist_name_map.keys())
            if verbose:
                info(f"Using {len(dist_codes)} districts from DOM selector")
        else:
            print("No districts available from API or DOM; aborting.")
            return
    else:
        if verbose:
            info(f"Found {len(districts)} districts for {state_cd}")
        # Map district code -> name (best effort) from API payload
        for d in districts:
            dcd = str(d.get("districtCd") or d.get("districtCode") or d.get("code") or "").strip().upper()
            if not dcd:
                # Compose from districtNo when full code not present
                try:
                    dno = int(str(d.get("districtNo") or "").strip())
                    dcd = f"{state_cd}{dno:02d}"
                except Exception:
                    dcd = ""
            dname = (d.get("districtName") or d.get("districtValue") or d.get("name") or d.get("district") or "").strip()
            if dcd:
                dist_codes.append(dcd)
                dist_name_map[dcd] = dname
    if verbose:
        try:
            sample_d = districts[:3]
            info("District API sample (first 3 rows):")
            for row in sample_d:
                if isinstance(row, dict):
                    info("  keys=", list(row.keys()))
        except Exception:
            pass
    # Enrich names via constituencies endpoint (fills missing or overrides blanks)
    try:
        extra_names = list_district_names_from_constituencies(sess, state_cd)
        for k, v in extra_names.items():
            if v and (k not in dist_name_map or not dist_name_map[k]):
                dist_name_map[k] = v
    except Exception:
        pass
    # Parse static names from the HTML page as another source (often most reliable)
    try:
        html_names = list_district_names_from_page(sess, state_cd, verbose=verbose)
        if verbose:
            info(f"HTML district names count: {len(html_names)}")
        # If HTML parsing yielded nothing (likely client-rendered), try Selenium scrape
        if not html_names:
            sel_names = list_district_names_via_selenium(state_cd, headless=True, verbose=verbose)
            if verbose:
                info(f"Selenium district names count: {len(sel_names)}")
            html_names = sel_names
        for k, v in (html_names or {}).items():
            if v:
                # Prefer HTML/Selenium-provided names (override API if different)
                dist_name_map[k] = v
                if k not in dist_codes:
                    dist_codes.append(k)
        if verbose:
            # Print a few merged entries to verify
            merged_sample = list(dist_name_map.items())[:10]
            info("Merged district names sample:")
            for k, v in merged_sample:
                info(f"  {k}: {v}")
    except Exception:
        pass
    # Collect all rows first to compute full union of payload + metadata keys
    rows: List[Dict] = []
    payload_keys_set = set()
    roll_try_order = []
    for rc in [roll_code, "g", "f", "o", "d", "sir"]:
        if rc and rc not in roll_try_order:
            roll_try_order.append(rc)
    # Fetch AC metadata once for enrichment
    acmeta_by_id, acmeta_by_dtac, acmeta_keys = fetch_constituencies_meta(sess, state_cd)
    for district_cd in dist_codes:
        if verbose:
            info(f"District {district_cd} - fetching ACs...")
        acs = list_acs_for_district(sess, state_cd, district_cd)
        if verbose:
            info(f"  Found {len(acs)} ACs")
        for ac_no, ac_label, ac_id in acs:
            parts: List[Dict] = []
            # Try multiple roll codes until we find parts
            for rc in roll_try_order:
                parts = fetch_parts_by_acid(sess, state_cd, ac_id, roll_code=rc, lang=lang, verbose=verbose, debug_dir=debug_dir, context={"districtCd": district_cd, "acNumber": ac_no})
                if parts:
                    break
            # Fallback to printing endpoint (POST paging)
            if not parts:
                try:
                    parts = fetch_parts_printing_paged(sess, state_cd, district_cd, int(ac_no), lang=lang, verbose=verbose, debug_dir=debug_dir)
                except Exception:
                    parts = []
            if not parts:
                continue
            # Identify AC metadata
            ac_meta = {}
            if ac_id is not None and ac_id in acmeta_by_id:
                ac_meta = acmeta_by_id.get(ac_id) or {}
            if not ac_meta:
                ac_meta = acmeta_by_dtac.get((district_cd, int(ac_no)), {})
            for p in parts:
                payload_keys_set.update(p.keys())
                # Include metadata keys in set as well
                for k in ac_meta.keys():
                    payload_keys_set.add(k)
                row = {
                    "stateCd": state_cd,
                    "districtCd": district_cd,
                    "districtName": dist_name_map.get(district_cd, ""),
                    "acNumber": ac_no,
                    "acLabel": ac_label,
                    "acId": ac_id if ac_id is not None else (ac_meta.get("acId") if isinstance(ac_meta, dict) else ""),
                }
                # First the part payload
                # Avoid clobbering prefilled identifiers with empty payload values
                if isinstance(p, dict):
                    for k, v in p.items():
                        if k in {"stateCd", "districtCd", "acNumber", "acLabel", "acId", "districtName"}:
                            # Keep existing non-empty value
                            if str(row.get(k, "")).strip() and (v is None or str(v).strip() == ""):
                                continue
                        row[k] = v
                # Then merge AC metadata, but do not override existing non-empty cells
                if isinstance(ac_meta, dict):
                    for k, v in ac_meta.items():
                        if k not in row or row[k] in (None, ""):
                            row[k] = v
                # Restore districtName from map if it was overwritten by blank payload/meta
                if not (isinstance(row.get("districtName"), str) and row["districtName"].strip()):
                    row["districtName"] = dist_name_map.get(district_cd, "")
                rows.append(row)
    if not rows:
        print("No part rows found across districts/ACs.")
        return
    payload_keys: List[str] = list(payload_keys_set)
    # Keep partNumber and partName first among payload keys
    def promote(cols: List[str], key: str) -> None:
        if key in cols:
            cols.remove(key)
            cols.insert(0, key)
    promote(payload_keys, "partName")
    promote(payload_keys, "partNumber")
    base_cols = ["stateCd", "districtCd", "districtName", "acNumber", "acLabel", "acId"]
    headers = base_cols + payload_keys
    ts = ist_now_str()
    out_name = f"part_list_{safe_name(state_cd)}_{ts}.csv"
    out_path = os.path.join(SAVE_ROOT, out_name)
    # Use UTF-8 with BOM for better compatibility with Excel and regional scripts
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            row = {h: r.get(h, "") for h in headers}
            w.writerow(row)
    print("Saved:", out_path)
if __name__ == "__main__":
    main()
