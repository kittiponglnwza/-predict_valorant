"""
╔══════════════════════════════════════════════════════════════╗
║   xG SCRAPER v3 — Selenium + Understat                       ║
║   เปิด Chrome จริงๆ รอ JavaScript โหลด แล้วดึง JSON         ║
╚══════════════════════════════════════════════════════════════╝

ติดตั้งก่อน:
    pip install selenium webdriver-manager

รัน:
    python xg_scraper_v3.py
"""

import json
import os
import re
import shutil
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

DATA_DIR   = "data_set"
DELAY      = 3.0   # วินาทีรอหน้าโหลด

SEASON_MAP = {
    "season 2020.csv": 2020,
    "season 2021.csv": 2021,
    "season 2022.csv": 2022,
    "season 2023.csv": 2023,
    "season 2024.csv": 2024,
    "season 2025.csv": 2025,
}

TEAM_MAP = {
    "Manchester City":         "Man City",
    "Manchester United":       "Man United",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham":               "Tottenham",
    "Sheffield United":        "Sheffield Utd",
    "Leeds United":            "Leeds",
    "Nottingham Forest":       "Nott'm Forest",
    "Brighton":                "Brighton",
    "Aston Villa":             "Aston Villa",
    "Newcastle United":        "Newcastle",
    "West Bromwich Albion":    "West Brom",
    "West Ham":                "West Ham",
    "Leicester":               "Leicester",
    "Brentford":               "Brentford",
    "Burnley":                 "Burnley",
    "Crystal Palace":          "Crystal Palace",
    "Everton":                 "Everton",
    "Fulham":                  "Fulham",
    "Southampton":             "Southampton",
    "Arsenal":                 "Arsenal",
    "Chelsea":                 "Chelsea",
    "Liverpool":               "Liverpool",
    "Watford":                 "Watford",
    "Norwich":                 "Norwich",
    "Bournemouth":             "Bournemouth",
    "Ipswich":                 "Ipswich",
    "Luton":                   "Luton",
    "Sunderland":              "Sunderland",
}

# ══════════════════════════════════════════════════════════════
# SELENIUM DRIVER SETUP
# ══════════════════════════════════════════════════════════════

def make_driver(headless=True):
    """สร้าง Chrome driver"""
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    # ปิด popup ต่างๆ
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    opts.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2
    })

    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=opts)
    return driver


# ══════════════════════════════════════════════════════════════
# FETCH xG จาก UNDERSTAT ด้วย Selenium
# ══════════════════════════════════════════════════════════════

def fetch_xg_selenium(driver, year: int) -> pd.DataFrame:
    """
    เปิด Understat EPL page ด้วย Selenium
    รอ JS โหลด แล้วดึง datesData ออกมา
    """
    url = f"https://understat.com/league/EPL/{year}"
    print(f"  🌐 เปิด {url}")

    driver.get(url)
    time.sleep(DELAY)  # รอ JS โหลด

    # ดึง page source หลัง JS render แล้ว
    html = driver.page_source
    print(f"  📄 HTML size: {len(html):,} bytes")

    # ลอง extract datesData จาก JavaScript variables
    patterns = [
        r"var datesData\s*=\s*JSON\.parse\('(.+?)'\)",
        r'var datesData\s*=\s*JSON\.parse\("(.+?)"\)',
        r"datesData\s*=\s*JSON\.parse\('(.+?)'\)",
        r'datesData\s*=\s*JSON\.parse\("(.+?)"\)',
        # บางครั้ง Understat inject ตรงๆ ไม่ผ่าน JSON.parse
        r'"datesData"\s*:\s*(\[.*?\])\s*[,}]',
    ]

    raw_data = None
    for pat in patterns:
        m = re.search(pat, html, re.DOTALL)
        if m:
            print(f"  ✅ พบ pattern: {pat[:50]}...")
            raw_data = m.group(1)
            break

    if raw_data is None:
        # ลองดึง via JavaScript execution
        print("  🔄 ลอง execute JavaScript...")
        try:
            result = driver.execute_script(
                "return typeof datesData !== 'undefined' ? JSON.stringify(datesData) : null"
            )
            if result:
                raw_data = result
                print("  ✅ ได้จาก JavaScript execution")
        except Exception as e:
            print(f"  ❌ JS execution failed: {e}")

    if raw_data is None:
        print(f"  ❌ ไม่พบ datesData ใน HTML ({len(html):,} bytes)")
        # Debug: แสดงตัวแปรที่มี
        try:
            vars_found = driver.execute_script(
                "return Object.keys(window).filter(k => k.includes('Data') || k.includes('data'))"
            )
            print(f"  💡 Variables ที่พบ: {vars_found[:10]}")
        except Exception:
            pass
        return pd.DataFrame()

    # Parse JSON
    try:
        # unescape unicode
        try:
            raw_data = raw_data.encode('utf-8').decode('unicode_escape')
        except Exception:
            pass
        data = json.loads(raw_data)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error: {e}")
        print(f"  Raw (first 200): {raw_data[:200]}")
        return pd.DataFrame()

    # Extract rows
    rows = []
    items = data if isinstance(data, list) else []
    for f in items:
        if not f.get("isResult"):
            continue
        try:
            rows.append({
                "date":         f["datetime"][:10],
                "home_team_us": f["h"]["title"],
                "away_team_us": f["a"]["title"],
                "home_goals":   int(f["goals"]["h"]),
                "away_goals":   int(f["goals"]["a"]),
                "home_xg":      float(f["xG"]["h"]),
                "away_xg":      float(f["xG"]["a"]),
            })
        except (KeyError, ValueError, TypeError):
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        print(f"  ✅ ดึงได้ {len(df)} แมตช์  "
              f"| xG range: {df['home_xg'].min():.2f}–{df['home_xg'].max():.2f}")
    else:
        print(f"  ⚠️  ได้ data แต่ไม่มีแมตช์ที่เสร็จแล้ว")

    return df


# ══════════════════════════════════════════════════════════════
# MERGE xG → CSV
# ══════════════════════════════════════════════════════════════

def merge_xg_into_csv(csv_path: str, xg_df: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_csv(csv_path)
    original["Date"] = pd.to_datetime(original["Date"], dayfirst=True, errors="coerce")

    xg_df = xg_df.copy()
    xg_df["HomeTeam_norm"] = xg_df["home_team_us"].apply(lambda x: TEAM_MAP.get(x, x))
    xg_df["AwayTeam_norm"] = xg_df["away_team_us"].apply(lambda x: TEAM_MAP.get(x, x))

    xg_slim = xg_df[["date","HomeTeam_norm","AwayTeam_norm","home_xg","away_xg"]].rename(columns={
        "date":          "Date",
        "HomeTeam_norm": "HomeTeam",
        "AwayTeam_norm": "AwayTeam",
        "home_xg":       "HomeXG",
        "away_xg":       "AwayXG",
    })

    merged  = original.merge(xg_slim, on=["Date","HomeTeam","AwayTeam"], how="left")
    matched = merged["HomeXG"].notna().sum()
    total   = len(original)
    print(f"  📊 Merge: {matched}/{total} ({matched/total*100:.1f}%)")

    # แสดงทีมที่ match ไม่ได้
    miss = merged[merged["HomeXG"].isna() & merged["FTHG"].notna()]
    if not miss.empty:
        bad = set(miss["HomeTeam"].tolist() + miss["AwayTeam"].tolist())
        # กรองเฉพาะที่ไม่มีใน xg_df เลย
        xg_teams = set(xg_df["HomeTeam_norm"].tolist() + xg_df["AwayTeam_norm"].tolist())
        missing  = bad - xg_teams
        if missing:
            print(f"  ⚠️  ทีม mismatch (เพิ่มใน TEAM_MAP): {sorted(missing)}")

    return merged


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  xG SCRAPER v3 — Selenium + Understat            ║")
    print("╚══════════════════════════════════════════════════╝\n")
    print("  🚀 เริ่ม Chrome driver...")

    try:
        driver = make_driver(headless=True)
        print("  ✅ Chrome driver พร้อม\n")
    except Exception as e:
        print(f"  ❌ Chrome driver error: {e}")
        print("  💡 ลองรันแบบ headless=False แทน")
        try:
            driver = make_driver(headless=False)
            print("  ✅ Chrome (visible mode) พร้อม\n")
        except Exception as e2:
            print(f"  ❌ ไม่สามารถเริ่ม Chrome ได้: {e2}")
            return

    total_matched = 0
    total_rows    = 0

    try:
        for csv_file, year in SEASON_MAP.items():
            csv_path = os.path.join(DATA_DIR, csv_file)
            if not os.path.exists(csv_path):
                print(f"\n⚠️  ไม่พบ: {csv_path}  (ข้าม)")
                continue

            print(f"\n{'='*54}")
            print(f"  📁  {csv_file}  →  EPL {year}/{year+1}")
            print(f"{'='*54}")

            xg_df = fetch_xg_selenium(driver, year)

            if xg_df.empty:
                print(f"  ❌ ไม่ได้ xG สำหรับ {year} — ข้าม")
                time.sleep(1)
                continue

            merged = merge_xg_into_csv(csv_path, xg_df)

            # Backup ก่อน save
            backup = csv_path.replace(".csv", "_backup.csv")
            if not os.path.exists(backup):
                shutil.copy(csv_path, backup)
                print(f"  💾 Backup → {os.path.basename(backup)}")

            merged.to_csv(csv_path, index=False)
            matched = int(merged["HomeXG"].notna().sum())
            total_matched += matched
            total_rows    += len(merged)
            print(f"  ✅ Saved → {csv_path}")

            time.sleep(1.5)

    finally:
        driver.quit()
        print("\n  🔒 Chrome ปิดแล้ว")

    # Summary
    print(f"\n{'█'*54}")
    print(f"  🎉  เสร็จสิ้น!")
    print(f"  📊  xG matched: {total_matched}/{total_rows} แมตช์")

    if total_matched > 0:
        print(f"\n  ▶️  รัน app.py ได้เลย — Phase 1 จะ activate อัตโนมัติ!")
        print(f"       คาด accuracy เพิ่ม ~2-4%")
    else:
        print(f"\n  ⚠️  ยังไม่ได้ xG — Understat อาจ block IP ไทย")
        print(f"  💡  ลอง VPN แล้วรันใหม่")
    print(f"{'█'*54}")


if __name__ == "__main__":
    main()