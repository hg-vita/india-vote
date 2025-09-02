"""
ECI e-roll downloader – iterate all ACs within a given District.

Usage examples:
  python eroll_dt_all.py --s S02 --dt 05
  python eroll_dt_all.py --s 02 --dt S0205

Behavior:
- Opens the ECI Download E-Roll page for the given state.
- Programmatically selects the District.
- Iterates through each AC listed for that District.
- For each AC, prefers English, then loops available Roll Types.
- For each Roll Type, prompts you to solve the captcha, then downloads all parts.

This script reuses helpers from eroll_ac_auto.py to avoid duplication.
"""

import os
import re
import time
import argparse
from typing import List, Dict, Tuple, Optional, Set, Union

import requests
from seleniumwire import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from scripts.download.eroll_ac_auto import (
    # Constants and small utils
    BASE_PAGE,
    SAVE_ROOT,
    BASE_HEADERS,
    info,
    safe_name,
    normalize_state,
    # Selenium helpers
    get_cookie_session,
    ui_select_by_value,
    ui_select_match_ac,
    ui_react_select_pick_ac,
    set_language_preference_to_english,
    enumerate_rolltype_labels,
    select_roll_type_by_label,
    capture_district_code_from_network,
    capture_parts_from_network,
    capture_captcha_id,
    get_ac_number_and_label,
    infer_lang,
    # Generate & download helpers
    generate_once,
    derive_roll_type_from_items,
    download_item,
    choose_endpoint_and_flags_from_label,
)


def _http_json(sess: requests.Session, url: str, headers: Dict[str, str], timeout: int = 60):
    try:
        r = sess.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def normalize_district_code(state_cd: str, dt: str) -> str:
    """Accepts dt as '05' or full 'S0205' and returns canonical 'S02DD'."""
    dt = (dt or "").strip().upper()
    if dt.startswith("S"):
        # Assume already like S0205; validate length and digits
        if re.fullmatch(r"S\d{4}", dt):
            return dt
        # Fallback: try to coerce S + digits
        m = re.match(r"S(\d+)", dt)
        if m and len(m.group(1)) >= 3:
            digits = m.group(1)[:4]
            return "S" + digits
    # dt is expected as 1-2 digit district number
    s_num = int(state_cd[1:]) if state_cd.upper().startswith("S") else int(state_cd)
    d_num = int(dt)
    return f"S{s_num:02d}{d_num:02d}"


def list_acs_for_district(sess: requests.Session, state_cd: str, district_cd: str) -> List[Tuple[int, str, Optional[str]]]:
    """Returns [(acNumber, acLabel, acId)] for given district via /api/v1/common/acs/{district_cd}."""
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    url = f"https://gateway-voters.eci.gov.in/api/v1/common/acs/{district_cd}"
    data = _http_json(sess, url, headers=hdrs)
    if not data:
        return []
    payload = data.get("payload") if isinstance(data, dict) else data
    out: List[Tuple[int, str, Optional[str]]] = []
    if isinstance(payload, list):
        for row in payload:
            ac_no = row.get("asmblyNo") or row.get("acNumber") or row.get("asmbly") or row.get("no")
            name = (row.get("asmblyName") or row.get("name") or "").strip()
            ac_id = row.get("acId") or row.get("id") or row.get("acID")
            try:
                ac_no = int(ac_no)
            except Exception:
                continue
            label = f"{ac_no} - {name}" if name else str(ac_no)
            out.append((ac_no, label, ac_id))
    # Sort by AC number
    out.sort(key=lambda t: t[0])
    return out


def list_districts(sess: requests.Session, state_cd: str) -> List[Dict]:
    """Fetch districts for a state. Tries a couple of likely endpoints."""
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    urls = [
        f"https://gateway-voters.eci.gov.in/api/v1/common/districts/{state_cd}",
        f"https://gateway-voters.eci.gov.in/api/v1/common/district/{state_cd}",
    ]
    for url in urls:
        data = _http_json(sess, url, headers=hdrs)
        if not data:
            continue
        payload = data.get("payload") if isinstance(data, dict) else data
        if isinstance(payload, list) and payload:
            return payload
    return []


def select_district_best_effort(driver, district_cd: str, district_name: Optional[str]) -> bool:
    """Try native <select> first, then react-select by visible district name or code."""
    # Native select by value (districtCd)
    if ui_select_by_value(driver, "select[name='district']", district_cd) \
       or ui_select_by_value(driver, "select[name='districtCd']", district_cd) \
       or ui_select_by_value(driver, "select#district", district_cd):
        return True
    # React-select: find the District control
    try:
        ctrl = driver.find_element(By.XPATH, "//label[contains(.,'District') or contains(.,'DISTRICT')]/following::*[contains(@class,'control')][1]")
        ctrl.click()
        time.sleep(0.2)
        # Prefer selecting by known district name
        picked = False
        if district_name:
            up = district_name.strip().upper()
            opts = driver.find_elements(By.XPATH, "//div[contains(@class,'option')]")
            for o in opts:
                t = (o.text or '').strip().upper()
                if t and up in t:
                    o.click(); picked = True; break
            if not picked:
                # Type and enter
                try:
                    inp = driver.find_element(By.CSS_SELECTOR, "input[id^='react-select-'][id$='-input']")
                    inp.send_keys(district_name)
                    time.sleep(0.2)
                    inp.send_keys(Keys.ENTER)
                    picked = True
                except Exception:
                    pass
        if not picked:
            # Fallback: try with code, e.g., S0205 or the last 2 digits
            code_try = district_cd
            try:
                inp = driver.find_element(By.CSS_SELECTOR, "input[id^='react-select-'][id$='-input']")
                inp.clear(); inp.send_keys(code_try)
                time.sleep(0.2); inp.send_keys(Keys.ENTER)
                picked = True
            except Exception:
                pass
        return picked
    except Exception:
        return False


def wait_for_acs_load(driver, expected_district_cd: str, timeout: float = 8.0) -> bool:
    """Wait until the page triggers the AC list API for the chosen district."""
    try:
        from scripts.download.eroll_ac_auto import capture_district_code_from_network
    except Exception:
        return False
    import time as _t
    start = _t.time()
    while _t.time() - start < timeout:
        got = capture_district_code_from_network(driver)
        if got and got.upper() == expected_district_cd.upper():
            return True
        _t.sleep(0.2)
    return False


def select_ac_best_effort(driver, ac_number: int) -> bool:
    """Try native selects, then a more general react-select strategy near the 'Assembly Constituency' label."""
    # Native select attempts first
    if ui_select_match_ac(driver, "select[name='constituency']", int(ac_number)) \
       or ui_select_match_ac(driver, "select[name='acNumber']", int(ac_number)) \
       or ui_select_match_ac(driver, "select#acNumber", int(ac_number)):
        return True

    # React-select near label variants
    try:
        # Locate by label text: Assembly, Assembly Constituency, AC
        ctrl = None
        label_xps = [
            "//label[contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'ASSEMBLY CONSTITUENCY')]",
            "//label[contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'ASSEMBLY')]",
            "//label[contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'AC')]",
        ]
        for xp in label_xps:
            try:
                lab = driver.find_element(By.XPATH, xp)
                ctrl = lab.find_element(By.XPATH, "following::*[@role='combobox' or contains(@class,'control')][1]")
                if ctrl:
                    break
            except Exception:
                continue
        if not ctrl:
            return False
        ctrl.click()
        time.sleep(0.2)
        # Type number into input
        try:
            inp = driver.find_element(By.CSS_SELECTOR, "input[id^='react-select-'][id$='-input']")
            inp.clear(); inp.send_keys(str(int(ac_number)))
            time.sleep(0.25)
        except Exception:
            pass

        num = str(int(ac_number))
        opt_xpath = (
            "//div[contains(@class,'option')][starts-with(normalize-space(.), '" + num + " ')]"
            + " | //div[contains(@class,'option')][starts-with(normalize-space(.), '" + num + "-')]"
            + " | //div[contains(@class,'option')][starts-with(normalize-space(.), '" + num + " –')]"
            + " | //div[contains(@class,'option')][contains(normalize-space(.), '(" + num + ")') or contains(normalize-space(.), '[" + num + "]')]"
        )
        opts = driver.find_elements(By.XPATH, opt_xpath)
        if opts:
            opts[0].click()
            return True
        # Fallback: Enter to accept top suggestion
        try:
            inp = driver.find_element(By.CSS_SELECTOR, "input[id^='react-select-'][id$='-input']")
            inp.send_keys(Keys.ENTER)
            return True
        except Exception:
            pass
    except Exception:
        pass
    return False


def roll_label_to_code(label: str) -> str:
    """Map visible roll label to short rollCode used by parts API: g,d,sir,f,o."""
    low = (label or "").lower()
    if "general" in low and "election" in low:
        return "g"
    if "sir" in low or "special" in low:
        return "sir"
    if "draft" in low and "sir" not in low:
        return "d"
    if "final" in low:
        return "f"
    if "supplement" in low:
        return "o"
    return "g"


def fetch_parts_by_acid(sess: requests.Session, state_cd: str, ac_id: Optional[Union[str,int]], roll_label: str, lang: str = "ENG") -> List[int]:
    """Query parts using rolls/get-part-list with acId + rollType + lang, bypassing UI.

    Returns sorted list of part numbers or [].
    """
    if not ac_id:
        return []


def ensure_district_confirmed(driver, sess: requests.Session, state_cd: str, district_cd: str, district_name: Optional[str], attempts: int = 3, wait_timeout: float = 10.0) -> bool:
    """Ensure District is selected in UI and the AC list load for that district is observed.

    - Tries programmatic select + wait up to `attempts` times.
    - If still not confirmed, prompts user to select manually and waits again.
    Returns True if confirmed, else False.
    """
    for i in range(attempts):
        if select_district_best_effort(driver, district_cd, district_name):
            if wait_for_acs_load(driver, district_cd, timeout=wait_timeout):
                return True
        time.sleep(0.5)
    # Manual fallback
    try:
        target = district_name or district_cd
        input(f"Please select District '{target}' manually in the page, then press ENTER…\n")
        if wait_for_acs_load(driver, district_cd, timeout=max(wait_timeout, 12.0)):
            return True
    except Exception:
        pass
    return False
    roll_code = roll_label_to_code(roll_label)
    hdrs = {**BASE_HEADERS, "Referer": f"{BASE_PAGE}?stateCode={state_cd}"}
    url = "https://gateway-voters.eci.gov.in/api/v1/rolls/get-part-list"
    try:
        r = sess.get(url, headers=hdrs, params={"acId": ac_id, "rollType": roll_code, "lang": lang}, timeout=60)
        if r.status_code != 200:
            return []
        j = r.json()
        payload = j.get("payload") if isinstance(j, dict) else j
        parts: List[int] = []
        if isinstance(payload, list):
            for row in payload:
                pn = row.get("partNumber")
                if isinstance(pn, int):
                    parts.append(pn)
        return sorted(set(parts))
    except Exception:
        return []


def ensure_language_english(driver, attempts: int = 3) -> bool:
    """Best-effort to set Language to English and verify visually."""
    for _ in range(attempts):
        try:
            set_language_preference_to_english(driver)
        except Exception:
            pass
        time.sleep(0.2)
        # Verify via native select value or react-select singleValue near Language label
        try:
            sel = driver.find_element(By.CSS_SELECTOR, "select[name='language'], select[name='langCd'], select[name='lang'], select#language")
            # Check selected option text
            txt = ""
            for opt in sel.find_elements(By.TAG_NAME, "option"):
                if opt.get_attribute("selected"):
                    txt = (opt.text or "").strip().lower(); break
            if not txt:
                txt = (sel.get_attribute("value") or "").strip().lower()
            if "eng" in txt or "english" in txt:
                return True
        except Exception:
            pass
        # React-select: read singleValue near Language label
        try:
            val = driver.find_element(By.XPATH, "//label[contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'LANG')]/following::*[contains(@class,'singleValue')][1]").text.strip().lower()
            if "english" in val:
                return True
        except Exception:
            pass
    return False


def ac_control_enabled(driver) -> bool:
    """Detect whether the Assembly react-select/native control is enabled."""
    # native
    for css in ("select[name='constituency']", "select[name='acNumber']", "select#acNumber"):
        try:
            if driver.find_elements(By.CSS_SELECTOR, css):
                el = driver.find_element(By.CSS_SELECTOR, css)
                if not el.get_attribute("disabled"):
                    return True
        except Exception:
            pass
    # react-select near label
    try:
        ctrl = driver.find_element(By.XPATH, "//label[contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'ASSEMBLY')]/following::*[@role='combobox' or contains(@class,'control')][1]")
        cls = ctrl.get_attribute("class") or ""
        aria = ctrl.get_attribute("aria-disabled") or "false"
        return ("is-disabled" not in cls) and (aria != "true")
    except Exception:
        return False


def _wait_until(pred, timeout: float = 5.0, interval: float = 0.2) -> bool:
    import time as _t
    start = _t.time()
    while _t.time() - start < timeout:
        try:
            if pred():
                return True
        except Exception:
            pass
        _t.sleep(interval)
    return False


def is_ac_selected(driver, expected_ac: int) -> bool:
    try:
        ac_no, _ = get_ac_number_and_label(driver)
        return str(ac_no or "").strip() == str(int(expected_ac))
    except Exception:
        return False


def wait_until_ac_selected(driver, expected_ac: int, timeout: float = 6.0) -> bool:
    return _wait_until(lambda: is_ac_selected(driver, expected_ac), timeout=timeout)


def is_language_english(driver) -> bool:
    try:
        return (infer_lang(driver) or "").upper().startswith("EN")
    except Exception:
        return False


def wait_until_language_english(driver, timeout: float = 4.0) -> bool:
    return _wait_until(lambda: is_language_english(driver), timeout=timeout)


def run(args: argparse.Namespace) -> None:
    state_cd = normalize_state(args.state)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    page_url = f"{BASE_PAGE}?stateCode={state_cd}"

    UI_STEP_DELAY = 1.0
    DOWNLOAD_DELAY = 0.4
    MAX_GEN_RETRIES = 2

    chrome_opts = Options()
    chrome_opts.add_argument("--start-maximized")
    if args.headless:
        chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_opts)
    driver.scopes = ['.*']
    driver.get(page_url)

    info("\n✅ Browser opened.")
    district_cd = normalize_district_code(state_cd, args.dt)
    info(f"Target: stateCd={state_cd}, districtCd={district_cd} (iterate all ACs)")

    # Build a session for discovery
    sess = get_cookie_session(driver)
    ac_list = list_acs_for_district(sess, state_cd, district_cd)
    if not ac_list:
        info("❌ Could not list ACs for the district (API returned no data).")
        driver.quit(); return
    info(f"📍 Found {len(ac_list)} AC(s) for district {district_cd}.")

    # Set District once (native or react-select)
    district_name: Optional[str] = None
    try:
        # Try to resolve district name from API for better matching in react-select
        drows = list_districts(sess, state_cd)
        for row in drows:
            dcd = row.get('districtCd') or row.get('districtCode') or row.get('code') or row.get('district')
            if str(dcd).strip().upper() == district_cd:
                district_name = (row.get('districtName') or row.get('name') or '').strip()
                break
    except Exception:
        pass
    if not ensure_district_confirmed(driver, sess, state_cd, district_cd, district_name, attempts=3, wait_timeout=10.0):
        info("❌ District could not be confirmed; aborting to avoid wrong selections.")
        driver.quit(); return

    time.sleep(UI_STEP_DELAY)

    grand_saved = grand_failed = grand_skipped = 0

    for ac_no, ac_label_hint, ac_id in ac_list:
        info(f"\n================= AC {ac_no} =================")

        # Use the same selectors/flow as eroll_ac_auto.py to pick AC first
        ac_selected = select_ac_best_effort(driver, int(ac_no)) or ui_react_select_pick_ac(driver, int(ac_no))
        if not ac_selected:
            info("⚠️ Could not programmatically select AC via UI. Will try API fallback for parts.")
        # Verify AC reflectively appears in the UI
        if not wait_until_ac_selected(driver, int(ac_no), timeout=6.0):
            # One more attempt if mismatch
            _ = select_ac_best_effort(driver, int(ac_no)) or ui_react_select_pick_ac(driver, int(ac_no))
            if not wait_until_ac_selected(driver, int(ac_no), timeout=4.0):
                info("ℹ️ AC not visibly confirmed; proceeding with API fallback for parts.")
        time.sleep(UI_STEP_DELAY)

        # Prefer English using the same helper from eroll_ac_auto
        try:
            set_language_preference_to_english(driver)
        except Exception:
            pass
        # Verify language reflects as English and retry once if needed
        if not wait_until_language_english(driver, timeout=4.0):
            try:
                set_language_preference_to_english(driver)
            except Exception:
                pass
            if not wait_until_language_english(driver, timeout=3.0):
                info("ℹ️ Language not visibly confirmed as English; continuing anyway.")
        time.sleep(UI_STEP_DELAY)

        # Roll types available for this AC
        roll_labels = enumerate_rolltype_labels(driver)
        if not roll_labels:
            # Fall back to current selection text, if any
            try:
                from scripts.download.eroll_ac_auto import get_roll_type_text
                rt = get_roll_type_text(driver)
                roll_labels = [rt] if rt else []
            except Exception:
                roll_labels = []
        if not roll_labels:
            info("ℹ️ No Roll Type options detected; skipping AC.")
            continue

        # For each roll type, capture + generate + download
        for lab in roll_labels:
            select_roll_type_by_label(driver, lab)
            time.sleep(0.6)
            info(f"➡️ Roll Type: {lab}")

            input("Solve captcha for this roll type, then press ENTER…\n")

            # Capture from network (same approach as eroll_ac_auto), then fallback to API by acId
            sess = get_cookie_session(driver)
            district_cd_net = capture_district_code_from_network(driver) or district_cd
            parts = capture_parts_from_network(driver)
            if not parts:
                parts = fetch_parts_by_acid(sess, state_cd, ac_id, lab, lang="ENG")
            captcha_id = capture_captcha_id(driver)
            ac_ui_no, ac_ui_label = get_ac_number_and_label(driver)
            if not ac_ui_no:
                ac_ui_no, ac_ui_label = str(ac_no), ac_label_hint
            # Read typed captcha value
            captcha_text = ""
            try:
                for css in ["input[name*='captcha']", "input[aria-label*='Captcha']", "input[placeholder*='Captcha']"]:
                    try:
                        els = driver.find_elements(By.CSS_SELECTOR, css)
                        if els:
                            captcha_text = els[0].get_attribute("value") or ""
                            if captcha_text:
                                break
                    except Exception:
                        pass
            except Exception:
                pass

            if not (ac_ui_no and parts and captcha_id and captcha_text):
                info("❌ Missing required fields (ac/parts/captcha); skipping this roll type.")
                continue

            # Build generate payload
            body_base = {
                "stateCd": state_cd,
                "districtCd": district_cd_net or district_cd,
                "acNumber": int(ac_ui_no),
                "partNumberList": [int(p) for p in parts],
                "captcha": captcha_text,
                "captchaId": captcha_id,
                "langCd": "ENG",
            }
            url, flags = choose_endpoint_and_flags_from_label(lab)
            body_base.update(flags)

            # Minimal captcha retry loop
            items: List[Dict] = []
            for attempt in range(MAX_GEN_RETRIES + 1):
                items, had_auth_error = generate_once(
                    sess=sess,
                    referer=page_url,
                    endpoint_url=url,
                    body=body_base,
                )
                if items:
                    break
                if had_auth_error and attempt < MAX_GEN_RETRIES:
                    prev = captcha_id
                    try:
                        # try UI refresh if available
                        from scripts.download.eroll_ac_auto import refresh_captcha_ui, wait_for_new_captcha
                        refresh_captcha_ui(driver)
                        input("Captcha invalid. Refresh and type new code, then ENTER…\n")
                        captcha_id, captcha_text = wait_for_new_captcha(driver, prev, timeout=10.0)
                        body_base["captchaId"] = captcha_id
                        body_base["captcha"] = captcha_text
                        continue
                    except Exception:
                        pass
                # no items or cannot recover
                items = []
                break

            if not items:
                info("ℹ️ No items returned for this Roll Type.")
                continue

            inferred_label = derive_roll_type_from_items(items, lab)
            ac_slug = safe_name(f"{ac_ui_no}_{ac_ui_label}")
            roll_slug = safe_name(inferred_label)
            out_dir = os.path.join(SAVE_ROOT, ac_slug, roll_slug)
            os.makedirs(out_dir, exist_ok=True)
            info(f"📂 Saving under: {out_dir}")

            # Cache existing parts to skip duplicates efficiently
            existing_parts_set: Set[int] = set()
            try:
                for name in os.listdir(out_dir):
                    m = re.search(r"p-(\d+)_", name)
                    if m:
                        existing_parts_set.add(int(m.group(1)))
            except Exception:
                pass

            saved = failed = skipped = 0
            for i, it in enumerate(items, 1):
                try:
                    res = download_item(sess, out_dir, ac_slug, it, existing_parts=existing_parts_set)
                    if res == "saved":
                        saved += 1
                    elif res == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                except Exception as e:
                    info(f"❌ [{i}] Error: {e}")
                    failed += 1
                if res != "skipped":
                    time.sleep(DOWNLOAD_DELAY)

            info(f"✅ Done AC {ac_ui_no} / Roll '{lab}'. Saved {saved}, skipped {skipped}, failed {failed}.")
            grand_saved += saved; grand_skipped += skipped; grand_failed += failed

    info(f"\n🎉 District complete. Saved {grand_saved}, skipped {grand_skipped}, failed {grand_failed}.")
    driver.quit()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ECI e-roll downloader: iterate all ACs in a District.")
    p.add_argument("--s", "--state", dest="state", required=True, help="State code (e.g., 02 or S02).")
    p.add_argument("--dt", dest="dt", required=True, help="District number (e.g., 05) or full code (e.g., S0205).")
    p.add_argument("--headless", action="store_true", help="Run Chrome in headless mode.")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
