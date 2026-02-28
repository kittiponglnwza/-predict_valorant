"""
ui/page_fixtures.py — Upcoming Fixtures + Prediction History
Football AI Nexus Engine
"""
import streamlit as st
import json
import os
from datetime import datetime, timedelta, timezone

from src.predict import predict_match, predict_score, show_next_pl_fixtures
from src.config import API_KEY
from utils import silent

DEFAULT_LOGO = "https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg"
HISTORY_FILE = "data/prediction_history.json"

TEAM_LOGOS = {
    "Arsenal":          "https://crests.football-data.org/57.png",
    "Aston Villa":      "https://crests.football-data.org/58.png",
    "Bournemouth":      "https://crests.football-data.org/1044.png",
    "Brentford":        "https://crests.football-data.org/402.png",
    "Brighton":         "https://crests.football-data.org/397.png",
    "Brighton Hove":    "https://crests.football-data.org/397.png",
    "Chelsea":          "https://crests.football-data.org/61.png",
    "Crystal Palace":   "https://crests.football-data.org/354.png",
    "Everton":          "https://crests.football-data.org/62.png",
    "Fulham":           "https://crests.football-data.org/63.png",
    "Ipswich":          "https://crests.football-data.org/678.png",
    "Leicester":        "https://crests.football-data.org/338.png",
    "Liverpool":        "https://crests.football-data.org/64.png",
    "Man City":         "https://crests.football-data.org/65.png",
    "Man United":       "https://crests.football-data.org/66.png",
    "Newcastle":        "https://crests.football-data.org/67.png",
    "Nott'm Forest":    "https://crests.football-data.org/351.png",
    "Nottingham":       "https://crests.football-data.org/351.png",
    "Southampton":      "https://crests.football-data.org/340.png",
    "Spurs":            "https://crests.football-data.org/73.png",
    "Tottenham":        "https://crests.football-data.org/73.png",
    "West Ham":         "https://crests.football-data.org/563.png",
    "Wolves":           "https://crests.football-data.org/76.png",
    "Leeds":            "https://crests.football-data.org/341.png",
    "Leeds United":     "https://crests.football-data.org/341.png",
    "Burnley":          "https://crests.football-data.org/328.png",
    "Sheffield Utd":    "https://crests.football-data.org/356.png",
    "Luton":            "https://crests.football-data.org/389.png",
    "Watford":          "https://crests.football-data.org/346.png",
    "Norwich":          "https://crests.football-data.org/68.png",
    "West Brom":        "https://crests.football-data.org/74.png",
    "Huddersfield":     "https://crests.football-data.org/394.png",
    "Cardiff":          "https://crests.football-data.org/715.png",
    "Swansea":          "https://crests.football-data.org/72.png",
    "Stoke":            "https://crests.football-data.org/70.png",
    "Middlesbrough":    "https://crests.football-data.org/343.png",
    "Sunderland":       "https://crests.football-data.org/71.png",
    "Hull":             "https://crests.football-data.org/322.png",
}


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY JSON HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_history(history):
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _add_prediction(home, away, date, time, pred_score, h_prob, d_prob, a_prob, match_id=None):
    """บันทึก — เรียกเฉพาะตอนกด Analyse เท่านั้น"""
    history = _load_history()
    key = f"{home}_vs_{away}_{date}"
    if any(e.get("key") == key for e in history):
        return
    if h_prob >= d_prob and h_prob >= a_prob:
        pred_winner = "Home"
    elif d_prob >= a_prob:
        pred_winner = "Draw"
    else:
        pred_winner = "Away"
    entry = {
        "key":           key,
        "match_id":      match_id,
        "home":          home,
        "away":          away,
        "date":          date,
        "time":          time,
        "pred_score":    pred_score,
        "h_prob":        h_prob,
        "d_prob":        d_prob,
        "a_prob":        a_prob,
        "pred_winner":   pred_winner,
        "real_score":    None,
        "real_winner":   None,
        "score_correct": None,
        "winner_correct": None,
        "saved_at":      datetime.now(timezone.utc).isoformat(),
    }
    history.append(entry)
    _save_history(history)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH REAL SCORES FOR FINISHED MATCHES IN HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_real_scores():
    history = _load_history()
    pending = [e for e in history if e.get("real_score") is None]
    if not pending:
        return history
    try:
        import requests
        date_from = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://api.football-data.org/v4/competitions/PL/matches",
            headers={"X-Auth-Token": API_KEY},
            params={"status": "FINISHED", "dateFrom": date_from,
                    "dateTo": datetime.now(timezone.utc).strftime("%Y-%m-%d")},
            timeout=10,
        )
        if not r.ok:
            return history
        finished = r.json().get("matches", [])
        changed = False
        for entry in pending:
            for m in finished:
                h_api = (m["homeTeam"].get("shortName") or m["homeTeam"]["name"]).lower()
                a_api = (m["awayTeam"].get("shortName") or m["awayTeam"]["name"]).lower()
                h_saved = entry["home"].lower()
                a_saved = entry["away"].lower()
                if (h_saved in h_api or h_api in h_saved) and (a_saved in a_api or a_api in a_saved):
                    ft = m.get("score", {}).get("fullTime", {})
                    sh, sa = ft.get("home"), ft.get("away")
                    if sh is not None and sa is not None:
                        real_score  = f"{sh}-{sa}"
                        real_winner = "Home" if sh > sa else ("Away" if sa > sh else "Draw")
                        entry["real_score"]     = real_score
                        entry["real_winner"]    = real_winner
                        entry["winner_correct"] = (entry["pred_winner"] == real_winner)
                        entry["score_correct"]  = (entry["pred_score"] == real_score)
                        changed = True
                    break
        if changed:
            _save_history(history)
    except Exception:
        pass
    return history


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FULL SEASON HISTORY — ดึงทุกแมตจบตั้งแต่ต้นฤดูกาล + AI re-predict
# ─────────────────────────────────────────────────────────────────────────────

SEASON_START = "2025-08-01"   # ต้นฤดูกาล 2025/26
SEASON_CACHE_FILE = "data/season_history_cache.json"
CACHE_VERSION = "v2_poisson"  # เปลี่ยนเมื่อ logic เปลี่ยน → force re-predict


def _load_season_cache():
    try:
        if os.path.exists(SEASON_CACHE_FILE):
            with open(SEASON_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_season_cache(cache: dict):
    try:
        os.makedirs(os.path.dirname(SEASON_CACHE_FILE), exist_ok=True)
        with open(SEASON_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _build_season_history(ctx, status_placeholder=None):
    """
    ดึงแมตจบทั้งหมดตั้งแต่ต้นฤดูกาลจาก API
    แล้ว AI re-predict ทุกแมตที่ยังไม่เคย predict
    cache ลงไฟล์ JSON — แมตที่ predict แล้วจะไม่ predict ซ้ำ
    คืน list เรียงวันที่ใหม่สุดก่อน
    """
    try:
        import requests
        from src.config import TEAM_NAME_MAP

        # ── 1. ดึงแมตทั้งหมดฤดูกาล 2024/25 ──────────────────────────
        r = requests.get(
            "https://api.football-data.org/v4/competitions/PL/matches",
            headers={"X-Auth-Token": API_KEY},
            params={"season": "2025"},
            timeout=15,
        )
        if not r.ok:
            return []

        all_finished = [m for m in r.json().get("matches", []) if m.get("status") == "FINISHED"]

        # ── 2. โหลด cache ──────────────────────────────────────────
        cache = _load_season_cache()   # key → entry dict
        changed = False

        total = len(all_finished)
        for idx, m in enumerate(all_finished):
            _utc = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            _th  = _utc + timedelta(hours=7)
            date_str = _th.strftime("%d/%m/%Y")

            home_raw = TEAM_NAME_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"])
            away_raw = TEAM_NAME_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"])

            cache_key = f"{home_raw}_vs_{away_raw}_{date_str}"

            # ถ้า cache มีแล้ว → ตรวจ version + consistency ก่อน skip
            if cache_key in cache and cache[cache_key].get("pred_score") not in (None, "—"):
                _cached_entry = cache[cache_key]
                # version check — ถ้า version เก่า → re-predict
                if _cached_entry.get("cache_version") != CACHE_VERSION:
                    pass  # ตกลงมา re-predict
                else:
                    _cs = _cached_entry.get("pred_score", "")
                    _cw = _cached_entry.get("pred_winner", "")
                    # consistency check — pred_score กับ pred_winner ต้องตรงกัน
                    _ok = False
                    try:
                        _ch, _ca = map(int, _cs.split("-"))
                        _derived_w = "Home" if _ch > _ca else ("Away" if _ca > _ch else "Draw")
                        _ok = (_derived_w == _cw)
                    except Exception:
                        _ok = True  # parse ไม่ได้ → ไม่บังคับ re-predict
                    if _ok:
                        continue
                # ไม่ผ่าน → re-predict

            # ── 3. AI predict ──────────────────────────────────────
            if status_placeholder:
                status_placeholder.markdown(
                    f'<div class="hst-empty">🔄 AI predicting... {idx+1}/{total} — {home_raw} vs {away_raw}</div>',
                    unsafe_allow_html=True,
                )

            _r = silent(predict_match, home_raw, away_raw, ctx)
            _s = silent(predict_score, home_raw, away_raw, ctx)

            ft = m.get("score", {}).get("fullTime", {})
            sh, sa = ft.get("home"), ft.get("away")
            real_score  = f"{sh}-{sa}" if sh is not None else None
            real_winner = ("Home" if sh > sa else ("Away" if sa > sh else "Draw")) if sh is not None else None

            if _r and _s:
                # ── ใช้ Poisson prob (จาก predict_score) เป็น source of truth ──
                # เพราะ pred_score มาจาก Poisson ตัวเดียวกัน → ไม่ขัดกันแน่นอน
                # predict_match (Hybrid) เก็บไว้แสดง bar chart เท่านั้น ไม่ใช้ decide winner
                h_prob = _s.get("poisson_home_win", _r["Home Win"])
                d_prob = _s.get("poisson_draw",     _r["Draw"])
                a_prob = _s.get("poisson_away_win", _r["Away Win"])

                # pred_winner derive จาก most_likely_score โดยตรง → consistent 100%
                pred_score = _s["most_likely_score"]
                try:
                    _mh, _ma = map(int, pred_score.split("-"))
                    pred_winner = "Home" if _mh > _ma else ("Away" if _ma > _mh else "Draw")
                except Exception:
                    pred_winner = "Home" if h_prob >= d_prob and h_prob >= a_prob \
                                  else ("Draw" if d_prob >= a_prob else "Away")

                winner_correct = (pred_winner == real_winner) if real_winner else None
                score_correct  = (pred_score == real_score)   if real_score  else None
            else:
                h_prob = d_prob = a_prob = None
                pred_score = pred_winner = "—"
                winner_correct = score_correct = None

            cache[cache_key] = {
                "cache_version":  CACHE_VERSION,
                "key":            cache_key,
                "home":           home_raw,
                "away":           away_raw,
                "date":           date_str,
                "time":           _th.strftime("%H:%M"),
                "pred_score":     pred_score,
                "h_prob":         h_prob,
                "d_prob":         d_prob,
                "a_prob":         a_prob,
                "pred_winner":    pred_winner,
                "real_score":     real_score,
                "real_winner":    real_winner,
                "winner_correct": winner_correct,
                "score_correct":  score_correct,
            }
            changed = True

        if changed:
            _save_season_cache(cache)

        if status_placeholder:
            status_placeholder.empty()

        # ── 4. เรียงวันที่ใหม่สุดก่อน ──────────────────────────────
        def _pdate(e):
            try:
                return datetime.strptime(e["date"], "%d/%m/%Y")
            except Exception:
                return datetime.min

        return sorted(cache.values(), key=_pdate, reverse=True)

    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# LOGO HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_team_logo(team):
    t = (team or "").lower()
    for name, logo in TEAM_LOGOS.items():
        n = name.lower()
        if n in t or t in n:
            return logo
    return DEFAULT_LOGO


def _crest_url(team, logo_url, team_id):
    if logo_url:
        return logo_url
    if team_id:
        return f"https://crests.football-data.org/{team_id}.png"
    return _fallback_team_logo(team)


def _filter_future_fixtures(fixtures):
    if not fixtures:
        return fixtures
    now_utc = datetime.now(timezone.utc)
    future = []
    for f in fixtures:
        try:
            date_str = f.get("Date", "")
            time_str = f.get("Time", "00:00")
            if not date_str or date_str == "—":
                future.append(f)
                continue
            try:
                kickoff_local = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            except ValueError:
                year = now_utc.year
                kickoff_local = datetime.strptime(f"{date_str} {year} {time_str}", "%d %b %Y %H:%M")
                if kickoff_local.month < now_utc.month - 3:
                    kickoff_local = kickoff_local.replace(year=year + 1)
            # Time จาก API เป็น UTC+7 (ไทย) → แปลงกลับเป็น UTC
            kickoff_utc = kickoff_local.replace(tzinfo=timezone.utc) - timedelta(hours=7)
            # ให้ buffer 3 ชั่วโมงหลังเตะ (ไม่ตัดแมตที่เพิ่งจบ)
            if kickoff_utc >= now_utc - timedelta(hours=3):
                future.append(f)
        except Exception:
            future.append(f)
    return future


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def _navigate_to_predict(home_team, away_team):
    st.session_state["nav_page"]     = "Predict Match"
    st.session_state["pred_home"]    = home_team
    st.session_state["pred_away"]    = away_team
    st.session_state["auto_predict"] = True
    for key in list(st.session_state.keys()):
        if key.startswith("sel_home_") or key.startswith("sel_away_"):
            del st.session_state[key]


def _save_and_predict(home, away, date, time, pred_score, h_prob, d_prob, a_prob, match_id):
    _add_prediction(home, away, date, time, pred_score, h_prob, d_prob, a_prob, match_id)
    _navigate_to_predict(home, away)


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');

.main .block-container { padding-top:2rem; max-width:1280px; margin:0 auto; }
.fx-page-header { margin-bottom:28px; }
.fx-eyebrow { font-family:'Rajdhani',sans-serif; font-size:0.8rem; font-weight:700; letter-spacing:4px; text-transform:uppercase; color:#38BDF8; margin-bottom:8px; }
.fx-title { font-family:'Orbitron',sans-serif; font-size:2.8rem; font-weight:900; color:#F0F6FF; letter-spacing:1px; line-height:1.1; }
.fx-title em { font-style:normal; color:#38BDF8; }
.fx-subtitle { font-family:'Rajdhani',sans-serif; font-size:1.05rem; color:rgba(148,187,233,0.55); letter-spacing:1px; margin-top:6px; }
.fx-section-label { font-family:'Rajdhani',sans-serif; font-size:0.75rem; font-weight:700; letter-spacing:3.5px; text-transform:uppercase; color:rgba(148,187,233,0.45); margin-bottom:14px; }

.main [data-testid="stHorizontalBlock"] {
    background:rgba(255,255,255,0.06) !important; border:1px solid rgba(255,255,255,0.14) !important;
    border-radius:8px !important; padding:18px 24px !important;
    max-width:1100px !important; margin:0 auto 10px !important;
    transition:border-color 0.15s ease, background 0.15s ease !important;
}
.main [data-testid="stHorizontalBlock"]:hover { background:rgba(56,189,248,0.09) !important; border-color:rgba(56,189,248,0.35) !important; }
.fx-error { background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25); border-radius:8px; padding:16px 20px; font-family:'Rajdhani',sans-serif; font-size:0.9rem; color:rgba(252,165,165,0.8); letter-spacing:0.5px; }
.stNumberInput label { font-family:'Rajdhani',sans-serif !important; font-size:0.85rem !important; letter-spacing:2px !important; text-transform:uppercase !important; color:rgba(148,187,233,0.5) !important; font-weight:700 !important; }
.stButton > button { font-family:'Rajdhani',sans-serif !important; font-weight:700 !important; letter-spacing:1.5px !important; font-size:0.92rem !important; border-radius:6px !important; }

/* Fixture card */
.fxc-row {
    display: grid;
    grid-template-columns: 64px 1fr 24px 1fr 110px 110px 56px 48px;
    align-items: center; gap: 0 14px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px 10px 0 0; padding: 14px 22px; margin-bottom: 0;
    transition: border-color 0.2s, background 0.2s;
}
.fxc-row:hover { background: rgba(255,255,255,0.055); border-color: rgba(255,255,255,0.14); }
.fxc-date { font-family:'Rajdhani',sans-serif; font-size:0.7rem; font-weight:700; letter-spacing:0.5px; color:rgba(148,187,233,0.7); background:rgba(148,187,233,0.07); border:1px solid rgba(148,187,233,0.15); border-radius:5px; padding:4px 0; text-align:center; white-space:nowrap; }
.fxc-home { display:flex; align-items:center; justify-content:flex-end; gap:9px; }
.fxc-away { display:flex; align-items:center; justify-content:flex-start; gap:9px; }
.fxc-vs { font-family:'Rajdhani',sans-serif; font-size:0.65rem; font-weight:600; color:rgba(148,187,233,0.2); text-align:center; letter-spacing:1px; }
.fxc-logo { width:24px; height:24px; object-fit:contain; flex-shrink:0; opacity:0.9; }
.fxc-name { font-family:'Rajdhani',sans-serif; font-size:0.98rem; font-weight:700; color:#E8F0FF; white-space:nowrap; }
.fxc-probs { display:flex; align-items:center; justify-content:flex-end; gap:4px; font-family:'Rajdhani',sans-serif; font-size:0.82rem; font-weight:700; white-space:nowrap; }
.fxc-plbl { font-size:0.6rem; color:rgba(148,187,233,0.3); letter-spacing:0.5px; margin-right:1px; }
.fxc-sep  { color:rgba(148,187,233,0.12); margin:0 1px; }
.fxc-score { font-family:'Orbitron',sans-serif; font-size:0.85rem; font-weight:700; color:rgba(148,187,233,0.35); letter-spacing:1px; text-align:center; }
.fxc-time { font-family:'Rajdhani',sans-serif; font-size:0.82rem; font-weight:600; color:rgba(148,187,233,0.3); text-align:right; letter-spacing:0.5px; }
.fxc-pred { text-align:center; line-height:1.25; }
.fxc-pred-team { font-family:'Rajdhani',sans-serif; font-size:0.88rem; font-weight:700; white-space:nowrap; }
.fxc-pred-pct  { font-family:'Rajdhani',sans-serif; font-size:0.68rem; color:rgba(148,187,233,0.4); }

/* Analyse button */
.stButton:has(button[kind="secondary"]) button {
    background: transparent !important; border: 1px solid rgba(148,187,233,0.12) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
    color: rgba(148,187,233,0.4) !important; font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.72rem !important; font-weight: 700 !important; letter-spacing: 2.5px !important;
    text-transform: uppercase !important; height: 32px !important; margin-top: -6px !important;
    transition: all 0.2s !important;
}
.stButton:has(button[kind="secondary"]) button:hover {
    background: rgba(148,187,233,0.06) !important; border-color: rgba(148,187,233,0.25) !important;
    color: rgba(148,187,233,0.75) !important;
}

/* History */
.acc-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:18px 22px; text-align:center; }
.acc-num  { font-family:'Orbitron',sans-serif; font-size:2rem; font-weight:700; line-height:1; }
.acc-lbl  { font-family:'Rajdhani',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:rgba(148,187,233,0.4); margin-top:6px; }

.hst-section-title {
    font-family:'Rajdhani',sans-serif; font-size:0.65rem; font-weight:700;
    letter-spacing:3px; text-transform:uppercase; color:rgba(148,187,233,0.35);
    border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px; margin-bottom:10px;
}
.hst-header {
    display:grid; grid-template-columns:72px 1fr 28px 1fr 76px 100px 76px 96px 72px;
    gap:0 10px; padding:0 22px 8px;
    font-family:'Rajdhani',sans-serif; font-size:0.68rem; font-weight:700;
    letter-spacing:2px; color:rgba(148,187,233,0.38); text-transform:uppercase;
    border-bottom:1px solid rgba(255,255,255,0.07); margin-bottom:6px;
}
.hst-row {
    display:grid; grid-template-columns:72px 1fr 28px 1fr 76px 100px 76px 96px 72px;
    align-items:center; gap:0 10px;
    background:rgba(255,255,255,0.035); border:1px solid rgba(255,255,255,0.08);
    border-radius:9px; padding:11px 22px; margin-bottom:6px;
    transition:background 0.15s, border-color 0.15s;
}
.hst-row:hover { background:rgba(56,189,248,0.055); }
.hst-date-col { display:flex; flex-direction:column; gap:2px; }
.hst-date-badge { font-family:'Rajdhani',sans-serif; font-size:0.72rem; font-weight:700; color:#F0F6FF; background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.28); border-radius:5px; padding:2px 0; text-align:center; letter-spacing:0.5px; }
.hst-time { font-family:'Rajdhani',sans-serif; font-size:0.7rem; color:rgba(148,187,233,0.35); text-align:center; }
.hst-home { display:flex; align-items:center; justify-content:flex-end; gap:8px; }
.hst-away { display:flex; align-items:center; justify-content:flex-start; gap:8px; }
.hst-logo { width:22px; height:22px; object-fit:contain; flex-shrink:0; }
.hst-name { font-family:'Rajdhani',sans-serif; font-size:0.92rem; font-weight:700; color:#F0F6FF; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.hst-vs   { font-family:'Rajdhani',sans-serif; font-size:0.68rem; font-weight:700; color:rgba(148,187,233,0.3); text-align:center; }
.hst-score-pred { font-family:'Orbitron',sans-serif; font-size:0.88rem; font-weight:700; color:rgba(148,187,233,0.4); text-align:center; letter-spacing:1px; }
.hst-score-real { font-family:'Orbitron',sans-serif; font-size:0.88rem; font-weight:700; text-align:center; letter-spacing:1px; }
.hst-badge-wrap { display:flex; justify-content:center; }
.hst-badge { display:inline-block; padding:3px 9px; border-radius:20px; font-family:'Rajdhani',sans-serif; font-size:0.68rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; white-space:nowrap; }
.hst-upcoming { background:rgba(56,189,248,0.10); border:1px solid rgba(56,189,248,0.28); color:#38BDF8; }
.hst-correct  { background:rgba(0,230,118,0.10);  border:1px solid rgba(0,230,118,0.28);  color:#00E676; }
.hst-wrong    { background:rgba(239,68,68,0.09);   border:1px solid rgba(239,68,68,0.22);  color:#EF4444; }
.hst-pending  { background:rgba(148,187,233,0.05); border:1px solid rgba(148,187,233,0.13);color:rgba(148,187,233,0.38); }
.hst-empty    { font-family:'Rajdhani',sans-serif; font-size:0.78rem; color:rgba(148,187,233,0.28); letter-spacing:1px; padding:12px 4px; }
</style>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_fixtures(ctx):
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown("""
        <div class="fx-page-header">
            <div class="fx-eyebrow">Nexus Engine · Premier League</div>
            <div class="fx-title">Next <em>Fixtures</em></div>
            <div class="fx-subtitle">Upcoming matches with AI win probability &amp; score prediction</div>
        </div>
    """, unsafe_allow_html=True)

    tab_fix, tab_hist = st.tabs(["  Fixtures  ", "  Prediction History  "])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — FIXTURES
    # ══════════════════════════════════════════════════════════════════════════
    with tab_fix:
        _, lane_c, _ = st.columns([0.06, 0.88, 0.06])
        with lane_c:
            _, c_mid, _ = st.columns([1.4, 1.8, 1.4])
            with c_mid:
                n = st.number_input("Matches to fetch", min_value=1, max_value=20, value=5)

        cache_age = (
            datetime.now(timezone.utc) -
            st.session_state.get("fx_fetched_at", datetime.min.replace(tzinfo=timezone.utc))
        ).total_seconds()
        needs_refresh = (
            "fx_upcoming" not in st.session_state
            or st.session_state.get("fx_n") != n
            or cache_age > 120
            or any("HomeID" not in m for m in st.session_state.get("fx_upcoming", []) if isinstance(m, dict))
        )
        if needs_refresh:
            with st.spinner("Fetching fixtures..."):
                raw = silent(show_next_pl_fixtures, ctx, num_matches=n + 10)
            st.session_state["fx_upcoming"]   = raw
            st.session_state["fx_n"]          = n
            st.session_state["fx_fetched_at"] = datetime.now(timezone.utc)
        else:
            raw = st.session_state["fx_upcoming"]

        upcoming = _filter_future_fixtures(raw)[:n] if raw else raw
        st.session_state["fx_upcoming_filtered"] = _filter_future_fixtures(raw) if raw else []

        if upcoming:
            st.markdown("")
            st.markdown('<div class="fx-section-label">Upcoming Matches</div>', unsafe_allow_html=True)
            for i, f in enumerate(upcoming):
                r = silent(predict_match, f["HomeTeam"], f["AwayTeam"], ctx)
                s = silent(predict_score, f["HomeTeam"], f["AwayTeam"], ctx)
                if r and s:
                    # ใช้ Poisson prob → consistent กับ score ที่มาจาก Poisson เดียวกัน
                    h_prob = s.get("poisson_home_win", r["Home Win"])
                    d_prob = s.get("poisson_draw",     r["Draw"])
                    a_prob = s.get("poisson_away_win", r["Away Win"])
                    max_prob = max(h_prob, d_prob, a_prob)

                    # derive score และ pred_winner จาก most_likely_score โดยตรง
                    score = s["most_likely_score"]
                    try:
                        _sh, _sa = map(int, score.split("-"))
                        pred_winner = "Home" if _sh > _sa else ("Away" if _sa > _sh else "Draw")
                    except Exception:
                        pred_winner = "Home" if h_prob == max_prob else ("Draw" if d_prob == max_prob else "Away")

                    h_color = "#38BDF8" if pred_winner == "Home" else "#64748B"
                    d_color = "#F59E0B" if pred_winner == "Draw" else "#64748B"
                    a_color = "#A78BFA" if pred_winner == "Away" else "#64748B"
                    home_logo     = _crest_url(f.get("HomeTeam"), f.get("HomeLogo"), f.get("HomeID"))
                    away_logo     = _crest_url(f.get("AwayTeam"), f.get("AwayLogo"), f.get("AwayID"))
                    home_fallback = _fallback_team_logo(f.get("HomeTeam"))
                    away_fallback = _fallback_team_logo(f.get("AwayTeam"))
                    raw_date = f.get("Date", "")
                    try:
                        date_label = datetime.strptime(raw_date, "%d/%m/%Y").strftime("%d %b").upper()
                    except Exception:
                        date_label = raw_date

                    # ── AI PRED — derive จาก pred_winner (consistent กับ score) ──
                    if pred_winner == "Home":
                        pred_team  = f["HomeTeam"].split()[-1]
                        pred_pct   = h_prob
                        pred_color = "#38BDF8"
                        pred_sub   = "Win"
                    elif pred_winner == "Draw":
                        pred_team  = "Draw"
                        pred_pct   = d_prob
                        pred_color = "#F59E0B"
                        pred_sub   = ""
                    else:
                        pred_team  = f["AwayTeam"].split()[-1]
                        pred_pct   = a_prob
                        pred_color = "#A78BFA"
                        pred_sub   = "Win"

                    st.markdown(f"""
                    <div class="fxc-row">
                        <div class="fxc-date">{date_label}</div>
                        <div class="fxc-home">
                            <span class="fxc-name">{f['HomeTeam']}</span>
                            <img class="fxc-logo" src="{home_logo}" onerror="this.src='{home_fallback}'"/>
                        </div>
                        <div class="fxc-vs">vs</div>
                        <div class="fxc-away">
                            <img class="fxc-logo" src="{away_logo}" onerror="this.src='{away_fallback}'"/>
                            <span class="fxc-name">{f['AwayTeam']}</span>
                        </div>
                        <div class="fxc-probs">
                            <span class="fxc-plbl">H</span><span style="color:{h_color};">{h_prob}%</span>
                            <span class="fxc-sep">|</span>
                            <span class="fxc-plbl">D</span><span style="color:{d_color};">{d_prob}%</span>
                            <span class="fxc-sep">|</span>
                            <span class="fxc-plbl">A</span><span style="color:{a_color};">{a_prob}%</span>
                        </div>
                        <div class="fxc-pred">
                            <div class="fxc-pred-team" style="color:{pred_color};">
                                {pred_team} {pred_sub}
                            </div>
                            <div class="fxc-pred-pct">{pred_pct}%</div>
                        </div>
                        <div class="fxc-score">{score}</div>
                        <div class="fxc-time">{f.get('Time','—')}</div>
                    </div>
                    <div class="fxc-analyse-row" id="fxc-ar-{i}"></div>
                    """, unsafe_allow_html=True)
                    st.button("Analyse", key=f"fx_nav_{i}",
                              use_container_width=True,
                              on_click=_save_and_predict,
                              args=(f["HomeTeam"], f["AwayTeam"],
                                    f.get("Date",""), f.get("Time",""),
                                    score, h_prob, d_prob, a_prob,
                                    f.get("MatchID")))
        elif upcoming is not None:
            st.markdown('<div class="fx-error">Unable to fetch upcoming fixtures. Check your API connection.</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — PREDICTION HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_hist:
        st.markdown("")

        # ── ดึงข้อมูล ──────────────────────────────────────────────────────────

        @st.cache_data(ttl=120, show_spinner=False)
        def _fetch_finished_api():
            """ดึง FINISHED 60 วันย้อนหลัง เพื่อเอาผลจริง"""
            try:
                import requests as _rq
                from src.config import TEAM_NAME_MAP
                _from = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")
                _to   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                _r = _rq.get(
                    "https://api.football-data.org/v4/competitions/PL/matches",
                    headers={"X-Auth-Token": API_KEY},
                    params={"status": "FINISHED", "dateFrom": _from, "dateTo": _to},
                    timeout=10,
                )
                if not _r.ok:
                    return []
                _out = []
                for _m in _r.json().get("matches", []):
                    _utc = datetime.fromisoformat(_m["utcDate"].replace("Z", "+00:00"))
                    _th  = _utc + timedelta(hours=7)
                    _ft  = _m.get("score", {}).get("fullTime", {})
                    _out.append({
                        "home": TEAM_NAME_MAP.get(_m["homeTeam"]["name"], _m["homeTeam"]["name"]),
                        "away": TEAM_NAME_MAP.get(_m["awayTeam"]["name"], _m["awayTeam"]["name"]),
                        "date": _th.strftime("%d/%m/%Y"),
                        "real_score": f"{_ft.get('home','?')}-{_ft.get('away','?')}",
                    })
                return _out
            except Exception:
                return []

        with st.spinner("Loading..."):
            history      = _fetch_real_scores()
            finished_api = _fetch_finished_api()

        # ── ดึง 5 แมตถัดไปจาก session_state ที่โหลดไว้แล้ว (ไม่ยิง API เพิ่ม) ──
        raw_upcoming = st.session_state.get("fx_upcoming_filtered") or _filter_future_fixtures(st.session_state.get("fx_upcoming") or [])
        next5 = [
            {"home": f["HomeTeam"], "away": f["AwayTeam"],
             "date": f.get("Date",""), "time": f.get("Time",""),
             "home_id": f.get("HomeID"), "away_id": f.get("AwayID")}
            for f in raw_upcoming[:5]
        ]

        # history เรียงใหม่สุดก่อน
        hist_all = sorted(history, key=lambda x: x.get("saved_at",""), reverse=True) if history else []

        # ── helpers ────────────────────────────────────────────────────────────
        def _fmt_date(d):
            try:
                return datetime.strptime(d, "%d/%m/%Y").strftime("%d %b").upper()
            except Exception:
                return d

        def _get_real_score(home, away, date):
            _k = f"{home}_{away}_{date}"
            for _m in finished_api:
                if f"{_m['home']}_{_m['away']}_{_m['date']}" == _k:
                    return _m["real_score"]
            return None

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── TABLE HEADER ───────────────────────────────────────────────────────
        def _hdr():
            st.markdown("""
            <div class="hst-header">
                <div>DATE</div>
                <div style="text-align:right;">HOME</div>
                <div></div>
                <div>AWAY</div>
                <div style="text-align:center;">AI SCORE</div>
                <div style="text-align:center;">AI PRED</div>
                <div style="text-align:center;">REAL</div>
                <div style="text-align:center;">WINNER</div>
                <div style="text-align:center;">SCORE</div>
            </div>""", unsafe_allow_html=True)

        # ── ROW RENDERER ───────────────────────────────────────────────────────
        def _row_upcoming(m, pred_score, h_prob, d_prob, a_prob, pred_winner=None):
            """แถว upcoming — ดึงผลทำนายจาก AI"""
            hl = _fallback_team_logo(m["home"])
            al = _fallback_team_logo(m["away"])
            dl = _fmt_date(m["date"])
            max_p = max(h_prob, d_prob, a_prob)
            hc = "#38BDF8" if h_prob == max_p else "rgba(148,187,233,0.35)"
            dc = "#F59E0B" if d_prob == max_p else "rgba(148,187,233,0.35)"
            ac = "#A78BFA" if a_prob == max_p else "rgba(148,187,233,0.35)"

            # AI PRED cell — ใช้ pred_winner (ที่ derive จาก score แล้ว) ถ้ามี
            # เพื่อให้ pred_label สอดคล้องกับ AI SCORE เสมอ
            if pred_winner is None:
                pred_winner = "Home" if h_prob == max_p else ("Draw" if d_prob == max_p else "Away")

            if pred_winner == "Home":
                pred_label = m["home"].split()[-1]
                pred_pct   = h_prob
                pred_color = "#38BDF8"
            elif pred_winner == "Draw":
                pred_label = "Draw"
                pred_pct   = d_prob
                pred_color = "#F59E0B"
            else:
                pred_label = m["away"].split()[-1]
                pred_pct   = a_prob
                pred_color = "#A78BFA"

            st.markdown(f"""
            <div class="hst-row" style="border-color:rgba(56,189,248,0.15);">
                <div class="hst-date-col">
                    <div class="hst-date-badge" style="color:#38BDF8;background:rgba(56,189,248,0.08);border-color:rgba(56,189,248,0.2);">{dl}</div>
                    <div class="hst-time">{m.get("time","")}</div>
                </div>
                <div class="hst-home">
                    <span class="hst-name">{m["home"]}</span>
                    <img class="hst-logo" src="{hl}" onerror="this.src='{DEFAULT_LOGO}'"/>
                </div>
                <div class="hst-vs">vs</div>
                <div class="hst-away">
                    <img class="hst-logo" src="{al}" onerror="this.src='{DEFAULT_LOGO}'"/>
                    <span class="hst-name">{m["away"]}</span>
                </div>
                <div class="hst-score-pred">{pred_score}</div>
                <div style="text-align:center;line-height:1.2">
                    <div style="font-family:Rajdhani,sans-serif;font-size:0.82rem;font-weight:700;color:{pred_color};">{pred_label}</div>
                    <div style="font-family:Rajdhani,sans-serif;font-size:0.68rem;color:rgba(148,187,233,0.4);">
                        <span style="color:{hc};">{h_prob}%</span>
                        <span style="color:rgba(148,187,233,0.2);"> / </span>
                        <span style="color:{dc};">{d_prob}%</span>
                        <span style="color:rgba(148,187,233,0.2);"> / </span>
                        <span style="color:{ac};">{a_prob}%</span>
                    </div>
                </div>
                <div class="hst-score-real">
                    <span style="font-family:'Rajdhani',sans-serif;font-size:0.65rem;color:rgba(148,187,233,0.3);letter-spacing:1px;">TBD</span>
                </div>
                <div class="hst-badge-wrap">
                    <span class="hst-badge hst-upcoming">⏳ Upcoming</span>
                </div>
                <div class="hst-badge-wrap">
                    <span class="hst-badge hst-pending">—</span>
                </div>
            </div>""", unsafe_allow_html=True)

        def _row_history(e, real_score):
            """แถว history — แมตที่กด Analyse แล้ว"""
            hl = _fallback_team_logo(e["home"])
            al = _fallback_team_logo(e["away"])
            dl = _fmt_date(e.get("date",""))
            has_r = bool(real_score)
            if has_r:
                wc = e.get("winner_correct")
                if wc is True:
                    w_badge    = '<span class="hst-badge hst-correct">✓ Correct</span>'
                    row_accent = "rgba(0,230,118,0.10)"
                elif wc is False:
                    w_badge    = '<span class="hst-badge hst-wrong">✗ Wrong</span>'
                    row_accent = "rgba(239,68,68,0.10)"
                else:
                    w_badge    = '<span class="hst-badge hst-pending">Pending</span>'
                    row_accent = "rgba(255,255,255,0.05)"
                sc = e.get("score_correct")
                s_badge = (
                    '<span class="hst-badge hst-correct">Exact</span>' if sc is True
                    else '<span class="hst-badge hst-wrong">Off</span>' if sc is False
                    else '<span class="hst-badge hst-pending">—</span>'
                )
                real_cell = f'<span style="color:#E8F0FF;font-weight:700;">{real_score}</span>'
            else:
                w_badge    = '<span class="hst-badge hst-pending">Pending</span>'
                s_badge    = '<span class="hst-badge hst-pending">—</span>'
                row_accent = "rgba(255,255,255,0.05)"
                real_cell  = '<span style="color:rgba(148,187,233,0.2);">—</span>'

            # ── AI PRED cell — แสดงว่า AI ทายใครชนะ + % ──
            pw  = e.get("pred_winner")
            hp  = e.get("h_prob")
            dp  = e.get("d_prob")
            ap  = e.get("a_prob")
            if pw and hp is not None:
                if pw == "Home":
                    pred_label = e["home"].split()[-1]   # ชื่อทีมสั้น
                    pred_pct   = hp
                    pred_color = "#38BDF8"
                elif pw == "Draw":
                    pred_label = "Draw"
                    pred_pct   = dp
                    pred_color = "#F59E0B"
                else:
                    pred_label = e["away"].split()[-1]
                    pred_pct   = ap
                    pred_color = "#A78BFA"
                ai_pred_cell = (
                    f'<div style="text-align:center;line-height:1.2">'
                    f'<div style="font-family:Rajdhani,sans-serif;font-size:0.82rem;'
                    f'font-weight:700;color:{pred_color};">{pred_label}</div>'
                    f'<div style="font-family:Rajdhani,sans-serif;font-size:0.68rem;'
                    f'color:rgba(148,187,233,0.4);">{pred_pct}%</div>'
                    f'</div>'
                )
            else:
                ai_pred_cell = '<div style="text-align:center;color:rgba(148,187,233,0.2);">—</div>'

            st.markdown(f"""
            <div class="hst-row" style="border-color:{row_accent};">
                <div class="hst-date-col">
                    <div class="hst-date-badge">{dl}</div>
                    <div class="hst-time">{e.get("time","")}</div>
                </div>
                <div class="hst-home">
                    <span class="hst-name">{e["home"]}</span>
                    <img class="hst-logo" src="{hl}" onerror="this.src='{DEFAULT_LOGO}'"/>
                </div>
                <div class="hst-vs">vs</div>
                <div class="hst-away">
                    <img class="hst-logo" src="{al}" onerror="this.src='{DEFAULT_LOGO}'"/>
                    <span class="hst-name">{e["away"]}</span>
                </div>
                <div class="hst-score-pred">{e.get("pred_score","—")}</div>
                {ai_pred_cell}
                <div class="hst-score-real" style="text-align:center;">{real_cell}</div>
                <div class="hst-badge-wrap">{w_badge}</div>
                <div class="hst-badge-wrap">{s_badge}</div>
            </div>""", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════
        # SECTION 1 — 5 แมตถัดไป + ผลทำนาย AI (จาก model ที่เทรนแล้ว)
        # ════════════════════════════════════════════════════════════
        st.markdown('<div class="hst-section-title">⏳ NEXT 5 FIXTURES — AI PREDICTION</div>', unsafe_allow_html=True)

        if not next5:
            st.markdown('<div class="hst-empty">Go to Fixtures tab first to load upcoming matches.</div>',
                        unsafe_allow_html=True)
        else:
            _hdr()
            for m in next5:
                _r = silent(predict_match, m["home"], m["away"], ctx)
                _s = silent(predict_score, m["home"], m["away"], ctx)
                if _r and _s:
                    # ใช้ Poisson prob → consistent กับ pred_score ที่มาจาก Poisson เดียวกัน
                    _h = _s.get("poisson_home_win", _r["Home Win"])
                    _d = _s.get("poisson_draw",     _r["Draw"])
                    _a = _s.get("poisson_away_win", _r["Away Win"])
                    # derive pred_winner จาก most_likely_score โดยตรง
                    _ps = _s["most_likely_score"]
                    try:
                        _mh, _ma = map(int, _ps.split("-"))
                        _pw = "Home" if _mh > _ma else ("Away" if _ma > _mh else "Draw")
                    except Exception:
                        _pw = "Home" if _h >= _d and _h >= _a else ("Draw" if _d >= _a else "Away")
                    _row_upcoming(m, _ps, _h, _d, _a, pred_winner=_pw)
                else:
                    st.markdown(f'<div class="hst-empty">{m["home"]} vs {m["away"]} — prediction unavailable</div>',
                                unsafe_allow_html=True)

        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════
        # SECTION 2 — ทุกแมตที่จบแล้วตั้งแต่ต้นฤดูกาล + AI re-predict
        # ════════════════════════════════════════════════════════════
        st.markdown('<div class="hst-section-title">📋 AI SEASON PREDICTIONS — FINISHED MATCHES</div>',
                    unsafe_allow_html=True)

        # ── โหลด cache ก่อน ถ้ามีแล้วแสดงทันที ──────────────────────
        _cached = _load_season_cache()
        def _pdate(e):
            try: return datetime.strptime(e["date"], "%d/%m/%Y")
            except Exception: return datetime.min

        if _cached:
            # fix on-the-fly: แก้ entry ที่ pred_score กับ pred_winner ไม่ consistent
            # (เกิดจาก cache เก่าที่ใช้ LightGBM hybrid ทาย winner)
            _fixed = False
            for _e in _cached.values():
                _cs = _e.get("pred_score", "")
                _cw = _e.get("pred_winner", "")
                if not _cs or _cs == "—" or not _cw or _cw == "—":
                    continue
                try:
                    _ch, _ca = map(int, _cs.split("-"))
                    _correct_w = "Home" if _ch > _ca else ("Away" if _ca > _ch else "Draw")
                    if _correct_w != _cw:
                        _e["pred_winner"] = _correct_w
                        _rw = _e.get("real_winner")
                        _e["winner_correct"] = (_correct_w == _rw) if _rw else None
                        _fixed = True
                except Exception:
                    pass
            if _fixed:
                _save_season_cache(_cached)
            season_entries = sorted(_cached.values(), key=_pdate, reverse=True)
        else:
            season_entries = []

        # ── ปุ่ม refresh + status placeholder ──────────────────────
        _col_ref, _col_info = st.columns([1, 4])
        with _col_ref:
            do_refresh = st.button("🔄 Refresh / Load All", key="btn_season_refresh",
                                   use_container_width=True)
        with _col_info:
            _cached_count = len(_cached)
            st.markdown(
                f'<div style="font-family:Rajdhani,sans-serif;font-size:0.75rem;'
                f'color:rgba(148,187,233,0.45);padding-top:10px;">'
                f'{"Cached" if _cached_count else "No cache yet"} · {_cached_count} matches · '
                f'Click Refresh to fetch all finished matches this season (uses API quota)</div>',
                unsafe_allow_html=True,
            )

        _status = st.empty()

        if do_refresh or not _cached:
            with st.spinner("Fetching all finished matches from API and running AI predictions..."):
                season_entries = _build_season_history(ctx, status_placeholder=_status)

        # ── Stats bar ──────────────────────────────────────────────
        if season_entries:
            _s_total  = len(season_entries)
            _s_w_ok   = sum(1 for e in season_entries if e.get("winner_correct") is True)
            _s_sc_ok  = sum(1 for e in season_entries if e.get("score_correct")  is True)
            _s_w_acc  = round(_s_w_ok  / _s_total * 100, 1)
            _s_sc_acc = round(_s_sc_ok / _s_total * 100, 1)

            sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
            for _c, _v, _l, _col in [
                (sc1, _s_total,       "TOTAL MATCHES",  "#38BDF8"),
                (sc2, _s_w_ok,        "WINNER CORRECT", "#00E676"),
                (sc3, f"{_s_w_acc}%", "WINNER ACC",     "#00E676"),
                (sc4, f"{_s_sc_acc}%","SCORE ACC",      "#A78BFA"),
            ]:
                with _c:
                    st.markdown(f"""<div class="acc-card">
                        <div class="acc-num" style="color:{_col};">{_v}</div>
                        <div class="acc-lbl">{_l}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── Pagination ─────────────────────────────────────────
            PAGE_SIZE = 20
            total_pages = max(1, (len(season_entries) + PAGE_SIZE - 1) // PAGE_SIZE)
            if "season_page" not in st.session_state:
                st.session_state["season_page"] = 0

            page_idx = st.session_state["season_page"]
            page_entries = season_entries[page_idx * PAGE_SIZE : (page_idx + 1) * PAGE_SIZE]

            _hdr()
            for e in page_entries:
                _row_history(e, e.get("real_score"))

            # ── page nav ───────────────────────────────────────────
            if total_pages > 1:
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                pn_cols = st.columns([1, 2, 1])
                with pn_cols[0]:
                    if st.button("← Prev", key="pg_prev", disabled=(page_idx == 0)):
                        st.session_state["season_page"] = max(0, page_idx - 1)
                        st.rerun()
                with pn_cols[1]:
                    st.markdown(
                        f'<div style="text-align:center;font-family:Rajdhani,sans-serif;'
                        f'font-size:0.8rem;color:rgba(148,187,233,0.45);padding-top:8px;">'
                        f'Page {page_idx+1} / {total_pages}</div>',
                        unsafe_allow_html=True,
                    )
                with pn_cols[2]:
                    if st.button("Next →", key="pg_next", disabled=(page_idx >= total_pages - 1)):
                        st.session_state["season_page"] = min(total_pages - 1, page_idx + 1)
                        st.rerun()
        else:
            st.markdown(
                '<div class="hst-empty">Click "Refresh / Load All" to fetch all finished matches this season and run AI predictions.</div>',
                unsafe_allow_html=True,
            )