import os
import re
import json
import csv
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple

import requests
from seleniumwire import webdriver  # important: selenium-wire
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# =====================
# CONSTANTS / CONFIG
# =====================
STATE_CD = "S04"  # Bihar example; the URL also carries this
PAGE_URL = f"https://voters.eci.gov.in/download-eroll?stateCode={STATE_CD}"

GEN_URLS = {
    "geroll": "https://gateway-voters.eci.gov.in/api/v1/printing-publish/generate-published-geroll",
    "eroll": "https://gateway-voters.eci.gov.in/api/v1/printing-publish/generate-published-eroll",
    "supplement": "https://gateway-voters.eci.gov.in/api/v1/printing-publish/generate-published-supplement",
}
CAPTCHA_URL = "https://gateway-voters.eci.gov.in/api/v1/captcha-service/generateCaptcha/EROLL"

COMMON_HEADERS = {
    "Origin": "https://voters.eci.gov.in",
    "Referer": PAGE_URL,
    "applicationname": "VSP",
    "channelidobo": "VSP",
    "platform-type": "ECIWEB",
    "CurrentRole": "citizen",
    "Accept": "*/*",
    "Content-Type": "application/json",
}

SAVE_ROOT = "eci_check_reports"
os.makedirs(SAVE_ROOT, exist_ok=True)

# =====================
# HELPER: names/time
# =====================
def safe_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\\/:*?\"<>|]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", "_")
    return re.sub(r"[^\w\-\. ]+", "_", s).strip().replace(" ", "_")

def _ist_now_str() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(tz=ist).strftime("%Y-%m-%d %H:%M:%S %Z")

def sanitize_items(items_raw):
    return [x for x in (items_raw or []) if isinstance(x, dict)]

# =====================
# SELENIUM-WIRE UTILS
# =====================
def last_request(driver, predicate) -> Optional[object]:
    for req in reversed(driver.requests):
        try:
            if predicate(req):
                return req
        except Exception:
            continue
    return None

def get_cookie_session(driver) -> requests.Session:
    s = requests.Session()
    for c in driver.get_cookies():
        try:
            s.cookies.set(c["name"], c["value"], domain=".eci.gov.in")
            s.cookies.set(c["name"], c["value"], domain="gateway-voters.eci.gov.in")
        except Exception:
            pass
    return s

# =====================
# CAPTURE FROM NETWORK / UI
# =====================
def capture_district_code_from_network(driver) -> Optional[str]:
    req = last_request(driver, lambda r: "/api/v1/common/acs/" in (r.url or ""))
    if not req:
        return None
    m = re.search(r"/acs/(S\d{4,6})", req.url or "", flags=re.I)
    return m.group(1) if m else None

def capture_parts_from_network(driver) -> List[int]:
    req = last_request(
        driver,
        lambda r: "/api/v1/printing-publish/get-part-list" in (r.url or "") and r.response,
    )
    if not req or not req.response:
        return []
    try:
        txt = req.response.body.decode("utf-8", "ignore")
        data = json.loads(txt)
        payload = data.get("payload", data)
        parts = []
        if isinstance(payload, list):
            for row in payload:
                pn = row.get("partNumber")
                if isinstance(pn, int):
                    parts.append(pn)
        return sorted(set(parts))
    except Exception:
        return []

def capture_captcha_id(driver) -> Optional[str]:
    req = last_request(
        driver,
        lambda r: "/api/v1/captcha-service/generateCaptcha/EROLL" in (r.url or "") and r.response,
    )
    if not req or not req.response:
        return None
    txt = ""
    try:
        txt = req.response.body.decode("utf-8", "ignore")
        data = json.loads(txt)
        if isinstance(data, dict):
            if isinstance(data.get("payload"), dict) and data["payload"].get("captchaId"):
                return data["payload"]["captchaId"]
            if data.get("captchaId"):
                return data["captchaId"]
    except Exception:
        pass
    try:
        m = re.search(r"([A-F0-9]{16,})", txt, flags=re.I)
        return m.group(1) if m else None
    except Exception:
        return None

def get_ac_number_and_label(driver) -> Tuple[Optional[str], str]:
    ac_no = None
    label = None
    if driver.find_elements(By.CSS_SELECTOR, "input[name='constituency']"):
        el = driver.find_element(By.CSS_SELECTOR, "input[name='constituency']")
        ac_no = (el.get_attribute("value") or "").strip() or None

    candidates = [
        "//label[contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'ASSEMBLY') or contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'AC')]/following::*[contains(@class,'singleValue')][1]",
        "(//div[@role='combobox']//div[contains(@class,'singleValue')])[position()=2 or position()=3]",
    ]
    for xp in candidates:
        try:
            t = driver.find_element(By.XPATH, xp).text.strip()
            if t:
                label = t
            if not ac_no:
                m = re.match(r"(\d+)", t)
                if m:
                    ac_no = m.group(1)
            break
        except Exception:
            pass
    return ac_no, (label or f"AC_{ac_no or 'UNKNOWN'}")

def get_lang_from_ui(driver) -> str:
    try:
        sel = driver.find_element(By.CSS_SELECTOR, "select[name='langCd']")
        val = (sel.get_attribute("value") or "").strip().upper()
        if val:
            return val
        for opt in sel.find_elements(By.TAG_NAME, "option"):
            if opt.get_attribute("selected"):
                vv = (opt.get_attribute("value") or opt.text or "").strip().upper()
                if vv:
                    return vv
    except Exception:
        pass
    try:
        txt = driver.find_element(
            By.XPATH,
            "//*[@id='textContent']/div[2]/div[2]/div[1]/div[4]/div/div[2]/div/select",
        ).get_attribute("value") or ""
        txt = txt.strip().upper()
        if txt:
            return txt
    except Exception:
        pass
    return "HIN"

def get_rolltype_from_ui(driver) -> Tuple[str, str]:
    try:
        sel = driver.find_element(By.CSS_SELECTOR, "select[name='roleType']")
        code = (sel.get_attribute("value") or "").strip()
        txt = ""
        if not code:
            for opt in sel.find_elements(By.TAG_NAME, "option"):
                if opt.get_attribute("selected"):
                    code = (opt.get_attribute("value") or "").strip()
                    txt = (opt.text or "").strip()
                    break
        if not txt:
            for opt in sel.find_elements(By.TAG_NAME, "option"):
                if (opt.get_attribute("value") or "").strip() == code:
                    txt = (opt.text or "").strip()
                    break
        return code, (txt or "ROLL_Unknown")
    except Exception:
        try:
            sel = driver.find_element(
                By.XPATH,
                "//*[@id='textContent']/div[2]/div[2]/div[1]/div[5]/div/div[2]/div/select",
            )
            code = (sel.get_attribute("value") or "").strip()
            txt = ""
            if not code:
                for opt in sel.find_elements(By.TAG_NAME, "option"):
                    if opt.get_attribute("selected"):
                        code = (opt.get_attribute("value") or "").strip()
                        txt = (opt.text or "").strip()
                        break
            if not txt:
                for opt in sel.find_elements(By.TAG_NAME, "option"):
                    if (opt.get_attribute("value") or "").strip() == code:
                        txt = (opt.text or "").strip()
                        break
            return code, (txt or "ROLL_Unknown")
        except Exception:
            return "", "ROLL_Unknown"

def choose_endpoint_and_flags(roll_code: str) -> Tuple[str, Dict[str, bool]]:
    roll_code = (roll_code or "").strip().lower()
    flags: Dict[str, bool] = {}
    if roll_code == "g":
        return GEN_URLS["geroll"], flags
    if roll_code == "d":
        return GEN_URLS["eroll"], flags
    if roll_code == "sir":
        flags["isSir"] = True
        return GEN_URLS["eroll"], flags
    if roll_code == "f":
        flags["isSupplement"] = False
        return GEN_URLS["supplement"], flags
    if roll_code == "o":
        flags["isSupplement"] = True
        return GEN_URLS["supplement"], flags
    return GEN_URLS["geroll"], flags

# =====================
# CAPTCHA REFRESH FLOW
# =====================
def read_captcha_from_ui(driver) -> Tuple[str, Optional[str]]:
    captcha_text = ""
    for css in ["input[name*='captcha']", "input[aria-label*='Captcha']", "input[placeholder*='Captcha']"]:
        if driver.find_elements(By.CSS_SELECTOR, css):
            captcha_text = driver.find_element(By.CSS_SELECTOR, css).get_attribute("value") or ""
            break
    captcha_id = capture_captcha_id(driver)
    return captcha_text, captcha_id

def try_click_captcha_refresh(driver) -> bool:
    """
    Try multiple heuristics to refresh captcha in the page.
    Returns True if we *think* we clicked a refresh control.
    """
    xpaths = [
        "//button[contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'REFRESH')]",
        "//*[@role='button' and contains(translate(.,'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'REFRESH')]",
        "//span[contains(@class,'icon') or contains(@class,'refresh')]/ancestor::button[1]",
        "//img[contains(@alt,'captcha') or contains(@src,'captcha') or contains(@class,'captcha')]|//canvas[contains(@class,'captcha')]",
    ]
    # click refresh-like buttons
    for xp in xpaths[:3]:
        els = driver.find_elements(By.XPATH, xp)
        if els:
            try:
                els[0].click()
                return True
            except Exception:
                pass
    # last resort: click captcha image/canvas (some sites refresh on click)
    els = driver.find_elements(By.XPATH, xpaths[3])
    if els:
        try:
            els[0].click()
            return True
        except Exception:
            pass
    return False

def generate_new_captcha(sess: requests.Session) -> Optional[str]:
    """
    Call the backend captcha API to ensure a new captcha is minted server-side.
    Returns newest captchaId (string) if response contains it; else None.
    """
    try:
        r = sess.get(CAPTCHA_URL, headers=COMMON_HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"⚠️ Captcha GET HTTP {r.status_code}")
            return None
        data = r.json()
        cid = None
        if isinstance(data, dict):
            if isinstance(data.get("payload"), dict):
                cid = data["payload"].get("captchaId")
            if not cid:
                cid = data.get("captchaId")
        print(f"🔄 Captcha generated (server): captchaId={cid}")
        return cid
    except Exception as e:
        print(f"⚠️ Captcha GET error: {e}")
        return None

# =====================
# OUTPUT PATHS & CSV
# =====================
def make_out_dir(save_root: str, state_cd: str, district_cd: str, ac_no: str, roll_label: str) -> str:
    part1 = f"{safe_name(state_cd)}_{safe_name(district_cd)}_AC{safe_name(str(ac_no))}"
    roll = safe_name(roll_label or "ROLL")
    out_dir = os.path.join(save_root, part1, roll)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def write_ac_filelist_csv(items, parts_expected, out_dir):
    path = os.path.join(out_dir, "_filelist.csv")
    index = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        file_uuid = it.get("fileUuid") or it.get("imagePath") or ""
        ref_id = it.get("refId") or ""
        bucket = it.get("bucketName") or ""
        m = re.search(r"/p-(\d+)_", file_uuid) or re.search(r"[-_](\d+)[-_]\w*\.pdf$", ref_id)
        part_num = int(m.group(1)) if m else None
        if part_num:
            index[part_num] = {"refId": ref_id, "bucketName": bucket, "fileUuid": file_uuid}

    headers = ["partNumber", "refId", "bucketName", "fileUuid", "status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for pn in parts_expected:
            if pn in index:
                row = index[pn]
                w.writerow({"partNumber": pn, "refId": row["refId"], "bucketName": row["bucketName"], "fileUuid": row["fileUuid"], "status": "OK"})
            else:
                w.writerow({"partNumber": pn, "refId": "", "bucketName": "", "fileUuid": "", "status": "MISSING"})

    files_found = len(index)
    print(f"📝 File list saved: {path} (found={files_found}, expected={len(parts_expected)}, missing={len(parts_expected)-files_found})")
    return files_found

def write_ac_summary_csv(out_dir: str, state_cd: str, district_cd: str, ac_number: str,
                         roll_code: str, roll_text: str, parts_total: int, files_found: int):
    path = os.path.join(out_dir, "_summary.csv")
    headers = ["stateCd","districtCd","acNumber","rollCode","rollText","parts_total","files_found","missing_count","timestamp"]
    row = {
        "stateCd": state_cd, "districtCd": district_cd, "acNumber": ac_number,
        "rollCode": roll_code, "rollText": roll_text,
        "parts_total": int(parts_total), "files_found": int(files_found),
        "missing_count": max(0, int(parts_total) - int(files_found)),
        "timestamp": _ist_now_str(),
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers); w.writeheader(); w.writerow(row)
    print(f"📝 Summary saved: {path}")

def append_master_summary_row(save_root: str, state_cd: str, district_cd: str, ac_number: str,
                              roll_code: str, roll_text: str, parts_total: int, files_found: int):
    path = os.path.join(save_root, "master_summary.csv")
    headers = ["stateCd","districtCd","acNumber","rollCode","rollText","parts_total","files_found","missing_count","timestamp"]
    row = {
        "stateCd": state_cd, "districtCd": district_cd, "acNumber": ac_number,
        "rollCode": roll_code, "rollText": roll_text,
        "parts_total": int(parts_total), "files_found": int(files_found),
        "missing_count": max(0, int(parts_total) - int(files_found)),
        "timestamp": _ist_now_str(),
    }
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if new_file: w.writeheader()
        w.writerow(row)
    print(f"🧾 Master summary appended: {path}")

# =====================
# CORE RUN FOR CURRENT AC (with your captcha flow)
# =====================
def run_for_current_ac(driver, sess):
    # Stable fields
    district_cd = capture_district_code_from_network(driver)
    ac_no, ac_label = get_ac_number_and_label(driver)
    lang_cd = get_lang_from_ui(driver)
    roll_code, roll_label = get_rolltype_from_ui(driver)
    endpoint, extra_flags = choose_endpoint_and_flags(roll_code)

    def build_body(parts, captcha_text, captcha_id):
        body = {
            "stateCd": STATE_CD,
            "districtCd": district_cd,
            "acNumber": int(ac_no),
            "partNumberList": [int(p) for p in parts],
            "captcha": captcha_text,
            "captchaId": captcha_id,
            "langCd": lang_cd,
        }
        for k, v in extra_flags.items():
            body[k] = v
        return body

    # Always capture parts (after you’ve waited for table)
    parts = capture_parts_from_network(driver)

    # 1) Try with whatever captcha is currently typed
    captcha_text, captcha_id = read_captcha_from_ui(driver)
    body = build_body(parts, captcha_text, captcha_id)

    print(f"\n🧭 {driver.current_url}")
    print(f"🏳️ stateCd={STATE_CD}  📌 districtCd={district_cd}  🏷️ ACLabel='{ac_label}'")
    print(f"🏛️ acNumber={ac_no}")
    print(f"🌐 langCd={lang_cd}  🗂 rollTypeUI='{roll_label}'  code='{roll_code}'")
    print(f"🔐 captcha='{captcha_text}'  captchaId='{captcha_id}'")
    print(f"📄 parts from network: {len(parts)}")
    print("\n=== GENERATE DEBUG (initial) ===")
    print(f"Endpoint: {endpoint}")
    print(f"Body    : {json.dumps(body, ensure_ascii=False)}")
    print("===============================\n")

    gen = sess.post(endpoint, headers=COMMON_HEADERS, json=body, timeout=180)
    if gen.status_code == 200:
        # go save
        return _save_payload_csvs(gen, parts, STATE_CD, district_cd, ac_no, roll_code, roll_label)
    else:
        # 2) If invalid captcha → follow your flow:
        preview = ""
        try: preview = gen.text[:400]
        except: pass

        if gen.status_code == 400 and ("Invalid Catpcha" in preview or "Invalid Captcha" in preview):
            print("❌ Invalid captcha → generating a NEW captcha now (server) and refreshing UI…")

            # (a) Try clicking refresh in page (best UX)
            clicked = try_click_captcha_refresh(driver)
            if clicked:
                print("🔁 Clicked captcha refresh in page.")

            # (b) ALSO generate on backend to ensure new captchaId exists
            new_cid = generate_new_captcha(sess)
            if not new_cid:
                print("⚠️ Could not confirm server-side captcha generation (still OK if UI refreshed).")

            print("👉 Now type the NEW captcha text in the page input, wait for parts table, then press ENTER here.")
            input("Press ENTER to continue… ")

            # Re-read parts (in case the list refreshed) and new captcha fields
            parts = capture_parts_from_network(driver)
            captcha_text, captcha_id = read_captcha_from_ui(driver)
            body = build_body(parts, captcha_text, captcha_id)
            print("\n=== GENERATE DEBUG (retry after new captcha) ===")
            print(f"Body    : {json.dumps(body, ensure_ascii=False)}")
            print("==============================================\n")
            gen2 = sess.post(endpoint, headers=COMMON_HEADERS, json=body, timeout=180)
            if gen2.status_code == 200:
                return _save_payload_csvs(gen2, parts, STATE_CD, district_cd, ac_no, roll_code, roll_label)

            print(f"❌ Still failed (HTTP {gen2.status_code}).")
            try: print(f"↳ Response: {gen2.text[:1000]}")
            except: pass
            return
        else:
            print(f"❌ Generate HTTP {gen.status_code}.")
            try: print(f"↳ Response: {gen.text[:1000]}")
            except: pass
            return

def _save_payload_csvs(gen_response, parts, state_cd, district_cd, ac_no, roll_code, roll_label):
    print(f"↳ HTTP 200 OK")
    try:
        print(f"↳ Response body: {gen_response.text[:1000]}{'...' if len(gen_response.text) > 1000 else ''}")
    except: pass

    gen_json = gen_response.json()
    raw_items = gen_json.get("payload") or []
    items = sanitize_items(raw_items)
    print(f"ℹ️ payload size: {len(raw_items)}, valid dicts: {len(items)}, null/other: {len(raw_items)-len(items)}")

    out_dir = make_out_dir(SAVE_ROOT, state_cd, district_cd, str(ac_no), str(roll_label))
    print(f"📂 Writing CSVs under: {out_dir}")

    files_found = write_ac_filelist_csv(items, parts, out_dir)
    write_ac_summary_csv(out_dir, state_cd, district_cd or "", str(ac_no or ""),
                         str(roll_code or ""), str(roll_label or ""), len(parts or []), files_found)
    append_master_summary_row(SAVE_ROOT, state_cd, district_cd or "", str(ac_no or ""),
                              str(roll_code or ""), str(roll_label or ""), len(parts or []), files_found)
    print("✅ Done for AC", ac_no)

# =====================
# MAIN LOOP (keeps browser open)
# =====================
def main():
    chrome_opts = Options()
    chrome_opts.add_argument("--start-maximized")
    chrome_opts.add_experimental_option("detach", True)

    driver = webdriver.Chrome(options=chrome_opts)
    driver.get(PAGE_URL)
    sess = get_cookie_session(driver)

    print(
        "\n✅ Browser opened.\n"
        "Flow per AC:\n"
        "  • Change AC and wait for the PARTS TABLE.\n"
        "  • Press ENTER when prompted. Script tries the CURRENT captcha.\n"
        "  • If invalid, it GENERATES a new captcha (server) and tries to refresh in-page.\n"
        "  • You type the new captcha in the page, press ENTER → CSVs saved.\n"
        "Repeat for more ACs. Controls: ENTER=run, 's'=skip, 'q'=quit.\n"
    )

    last_ac = None
    try:
        while True:
            ac_no, ac_label = get_ac_number_and_label(driver)
            ac_key = f"{ac_no or ''}::{ac_label or ''}"

            if ac_no and ac_key != last_ac:
                print(f"\n🔎 Detected AC change → {ac_no}  ({ac_label})")
                choice = input("Press ENTER to capture this AC, or type 's' to skip, 'q' to quit: ").strip().lower()
                if choice == "q":
                    print("👋 Exiting loop (browser stays open).")
                    break
                if choice == "s":
                    print("⏭️ Skipped.")
                    last_ac = ac_key
                    continue

                print("⏳ Capturing… ensure parts table is visible.")
                run_for_current_ac(driver, sess)
                last_ac = ac_key
                continue

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user (Ctrl+C). Browser remains open.")
    # Do not close browser.

if __name__ == "__main__":
    main()
