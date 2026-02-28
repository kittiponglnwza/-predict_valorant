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
    """Save one prediction entry — skip if already exists for same match."""
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
        "winner_correct":None,
        "saved_at":      datetime.now(timezone.utc).isoformat(),
    }
    history.append(entry)
    _save_history(history)


# ─────────────────────────────────────────────────────────────────────────────
# REAL SCORE FETCHER  (football-data.org FINISHED matches)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_real_scores():
    history = _load_history()
    pending = [e for e in history if e.get("real_score") is None]
    if not pending:
        return history
    try:
        import requests
        date_from = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://api.football-data.org/v4/competitions/PL/matches",
            headers={"X-Auth-Token": API_KEY},
            params={"status": "FINISHED", "dateFrom": date_from},
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
                        entry["real_score"]    = real_score
                        entry["real_winner"]   = real_winner
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
# TEAM LOGO + FIXTURE FILTER HELPERS
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
            # รองรับทั้ง dd/mm/YYYY และ dd Mon (เช่น 01 Mar)
            try:
                kickoff = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            except ValueError:
                year = now_utc.year
                kickoff = datetime.strptime(f"{date_str} {year} {time_str}", "%d %b %Y %H:%M")
                if kickoff.month < now_utc.month - 3:
                    kickoff = kickoff.replace(year=year + 1)
            # เทียบกับ UTC+7 (เวลาไทย)
            kickoff_aware = kickoff.replace(tzinfo=timezone.utc) - timedelta(hours=7)
            if kickoff_aware >= now_utc - timedelta(minutes=5):
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


def _save_and_predict(home, away, date, time, pred_score, h_prob, d_prob, a_prob, match_id):
    _add_prediction(home, away, date, time, pred_score, h_prob, d_prob, a_prob, match_id)
    _navigate_to_predict(home, away)


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');

.main .block-container { padding-top:2rem; max-width:1280px; margin:0 auto; }

.fx-page-header  { margin-bottom:28px; }
.fx-eyebrow {
    font-family:'Rajdhani',sans-serif; font-size:0.8rem; font-weight:700;
    letter-spacing:4px; text-transform:uppercase; color:#38BDF8; margin-bottom:8px;
}
.fx-title { font-family:'Orbitron',sans-serif; font-size:2.8rem; font-weight:900; color:#F0F6FF; letter-spacing:1px; line-height:1.1; }
.fx-title em { font-style:normal; color:#38BDF8; }
.fx-subtitle { font-family:'Rajdhani',sans-serif; font-size:1.05rem; color:rgba(148,187,233,0.55); letter-spacing:1px; margin-top:6px; }

.fx-section-label { font-family:'Rajdhani',sans-serif; font-size:0.75rem; font-weight:700; letter-spacing:3.5px; text-transform:uppercase; color:rgba(148,187,233,0.45); margin-bottom:14px; }

.fx-date-badge { display:inline-block; background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.18); border-radius:4px; padding:2px 8px; font-family:'Rajdhani',sans-serif; font-size:0.85rem; font-weight:700; letter-spacing:1px; color:#38BDF8; }
.fx-time { font-family:'Rajdhani',sans-serif; font-size:0.9rem; color:rgba(148,187,233,0.5); margin-top:3px; letter-spacing:1px; }

.fx-teams { font-family:'Rajdhani',sans-serif; }
.fx-team-line { display:flex; align-items:center; gap:8px; }
.fx-team-logo { width:24px; height:24px; object-fit:contain; flex-shrink:0; filter:drop-shadow(0 2px 6px rgba(0,0,0,0.45)); }
.fx-home { font-size:1.15rem; font-weight:700; color:#F0F6FF; letter-spacing:0.5px; }
.fx-away { font-size:1.0rem; font-weight:600; color:rgba(148,187,233,0.65); margin-top:5px; letter-spacing:0.5px; }
.fx-vs   { font-size:0.7rem; color:rgba(56,189,248,0.5); font-weight:700; letter-spacing:2px; margin:3px 0; }

.fx-team-tag { font-family:'Rajdhani',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:1.5px; padding:1px 5px; border-radius:3px; margin-right:5px; vertical-align:middle; }
.fx-tag-home { background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); color:#38BDF8; }
.fx-tag-away { background:rgba(167,139,250,0.12); border:1px solid rgba(167,139,250,0.3); color:#A78BFA; }

.fx-prob-wrap { margin-top:2px; }
.fx-prob-row  { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
.fx-prob-label { font-family:'Rajdhani',sans-serif; font-size:0.75rem; font-weight:700; letter-spacing:1px; width:16px; color:rgba(148,187,233,0.5); }
.fx-prob-bar-bg { flex:1; height:6px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden; min-width:60px; }
.fx-prob-bar-fill { height:100%; border-radius:2px; }
.fx-prob-val { font-family:'Rajdhani',sans-serif; font-size:0.88rem; font-weight:700; width:40px; text-align:right; }

.fx-score       { font-family:'Orbitron',sans-serif; font-size:1.4rem; font-weight:700; color:#F0F6FF; letter-spacing:2px; line-height:1; }
.fx-score-label { font-family:'Rajdhani',sans-serif; font-size:0.58rem; font-weight:600; letter-spacing:2px; color:rgba(148,187,233,0.35); text-transform:uppercase; margin-top:3px; }

.fx-th { font-family:'Rajdhani',sans-serif; font-size:0.72rem; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:rgba(148,187,233,0.45); padding:4px 0; }

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

/* ── History ── */
.hist-date  { font-family:'Rajdhani',sans-serif; font-size:0.82rem; color:#38BDF8; font-weight:700; letter-spacing:1px; }
.hist-pred  { font-family:'Orbitron',sans-serif; font-size:1rem; font-weight:700; color:rgba(148,187,233,0.6); text-align:center; letter-spacing:1px; }
.hist-real  { font-family:'Orbitron',sans-serif; font-size:1rem; font-weight:700; color:#F0F6FF; text-align:center; letter-spacing:1px; }
.hist-real.pending { font-family:'Rajdhani',sans-serif; font-size:0.75rem; letter-spacing:2px; color:rgba(148,187,233,0.3); }
.hist-badge { display:inline-block; padding:3px 10px; border-radius:20px; font-family:'Rajdhani',sans-serif; font-size:0.7rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; }
.badge-correct { background:rgba(0,230,118,0.12); border:1px solid rgba(0,230,118,0.3); color:#00E676; }
.badge-wrong   { background:rgba(239,68,68,0.10); border:1px solid rgba(239,68,68,0.25); color:#EF4444; }
.badge-pending { background:rgba(148,187,233,0.06); border:1px solid rgba(148,187,233,0.15); color:rgba(148,187,233,0.4); }

.acc-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:18px 22px; text-align:center; }
.acc-num  { font-family:'Orbitron',sans-serif; font-size:2rem; font-weight:700; line-height:1; }
.acc-lbl  { font-family:'Rajdhani',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:rgba(148,187,233,0.4); margin-top:6px; }
/* ── Fixture Card ── */
.fxc-row {
    display: grid;
    grid-template-columns: 64px 1fr 24px 1fr 150px 56px 48px;
    align-items: center;
    gap: 0 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px 22px;
    margin-bottom: 0;
    border-radius: 10px 10px 0 0;
    transition: border-color 0.2s, background 0.2s;
}
.fxc-row:hover {
    background: rgba(255,255,255,0.055);
    border-color: rgba(255,255,255,0.14);
}
.fxc-date {
    font-family:'Rajdhani',sans-serif; font-size:0.7rem; font-weight:700;
    letter-spacing:0.5px; color:rgba(148,187,233,0.7);
    background:rgba(148,187,233,0.07); border:1px solid rgba(148,187,233,0.15);
    border-radius:5px; padding:4px 0; text-align:center; white-space:nowrap;
}
.fxc-home { display:flex; align-items:center; justify-content:flex-end; gap:9px; }
.fxc-away { display:flex; align-items:center; justify-content:flex-start; gap:9px; }
.fxc-vs {
    font-family:'Rajdhani',sans-serif; font-size:0.65rem; font-weight:600;
    color:rgba(148,187,233,0.2); text-align:center; letter-spacing:1px;
}
.fxc-logo { width:24px; height:24px; object-fit:contain; flex-shrink:0; opacity:0.9; }
.fxc-name { font-family:'Rajdhani',sans-serif; font-size:0.98rem; font-weight:700; color:#E8F0FF; white-space:nowrap; }
.fxc-probs {
    display:flex; align-items:center; justify-content:flex-end; gap:4px;
    font-family:'Rajdhani',sans-serif; font-size:0.82rem; font-weight:700; white-space:nowrap;
}
.fxc-plbl { font-size:0.6rem; color:rgba(148,187,233,0.3); letter-spacing:0.5px; margin-right:1px; }
.fxc-sep  { color:rgba(148,187,233,0.12); margin:0 1px; }
.fxc-score {
    font-family:'Orbitron',sans-serif; font-size:0.85rem; font-weight:700;
    color:rgba(148,187,233,0.35); letter-spacing:1px; text-align:center;
}
.fxc-time {
    font-family:'Rajdhani',sans-serif; font-size:0.82rem; font-weight:600;
    color:rgba(148,187,233,0.3); text-align:right; letter-spacing:0.5px;
}
/* ── History Grid ── */
.hst-header {
    display: grid;
    grid-template-columns: 72px 1fr 28px 1fr 76px 76px 96px 72px;
    gap: 0 10px;
    padding: 0 22px 8px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: rgba(148,187,233,0.38);
    text-transform: uppercase;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 6px;
}
.hst-row {
    display: grid;
    grid-template-columns: 72px 1fr 28px 1fr 76px 76px 96px 72px;
    align-items: center;
    gap: 0 10px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 9px;
    padding: 11px 22px;
    margin-bottom: 6px;
    transition: background 0.15s, border-color 0.15s;
}
.hst-row:hover { background: rgba(56,189,248,0.055); }
.hst-date-col { display:flex; flex-direction:column; gap:2px; }
.hst-date-badge {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem; font-weight: 700;
    color: #F0F6FF;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.28);
    border-radius: 5px;
    padding: 2px 0;
    text-align: center;
    letter-spacing: 0.5px;
}
.hst-time {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.7rem;
    color: rgba(148,187,233,0.35);
    text-align: center;
    letter-spacing: 0.5px;
}
.hst-home {
    display: flex; align-items: center;
    justify-content: flex-end; gap: 8px;
}
.hst-away {
    display: flex; align-items: center;
    justify-content: flex-start; gap: 8px;
}
.hst-logo { width:22px; height:22px; object-fit:contain; flex-shrink:0; }
.hst-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.92rem; font-weight: 700;
    color: #F0F6FF;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.hst-vs {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.68rem; font-weight: 700;
    color: rgba(148,187,233,0.3);
    text-align: center;
}
.hst-score-pred {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.88rem; font-weight: 700;
    color: rgba(148,187,233,0.4);
    text-align: center; letter-spacing: 1px;
}
.hst-score-real {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.88rem; font-weight: 700;
    text-align: center; letter-spacing: 1px;
}
.hst-badge-wrap { display:flex; justify-content:center; }
.hst-badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    white-space: nowrap;
}
.hst-correct { background:rgba(0,230,118,0.10); border:1px solid rgba(0,230,118,0.28); color:#00E676; }
.hst-wrong   { background:rgba(239,68,68,0.09); border:1px solid rgba(239,68,68,0.22); color:#EF4444; }
.hst-pending { background:rgba(148,187,233,0.05); border:1px solid rgba(148,187,233,0.13); color:rgba(148,187,233,0.38); }


/* ── Analyse button — full width ใต้ card ── */
/* ซ่อน Streamlit default button wrapper margin */
.fxc-row + div[data-testid="stVerticalBlock"] > div,
.fxc-row ~ div > div[data-testid="stVerticalBlock"] > div {
    margin: 0 !important;
    padding: 0 !important;
}
/* style ปุ่ม Analyse ที่อยู่ใต้ card */
.stButton:has(button[kind="secondary"]) button {
    background: transparent !important;
    border: 1px solid rgba(148,187,233,0.12) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    color: rgba(148,187,233,0.4) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    height: 32px !important;
    margin-top: -6px !important;
    transition: all 0.2s !important;
}
.stButton:has(button[kind="secondary"]) button:hover {
    background: rgba(148,187,233,0.06) !important;
    border-color: rgba(148,187,233,0.25) !important;
    color: rgba(148,187,233,0.75) !important;
}

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
            or any(
                "HomeID" not in m
                for m in st.session_state.get("fx_upcoming", [])
                if isinstance(m, dict)
            )
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

        if upcoming:
            st.markdown("")
            st.markdown('<div class="fx-section-label">Upcoming Matches</div>', unsafe_allow_html=True)

            for i, f in enumerate(upcoming):
                r = silent(predict_match, f["HomeTeam"], f["AwayTeam"], ctx)
                s = silent(predict_score, f["HomeTeam"], f["AwayTeam"], ctx)

                if r and s:
                    h_prob = r["Home Win"]
                    d_prob = r["Draw"]
                    a_prob = r["Away Win"]
                    score  = s["most_likely_score"]
                    max_prob = max(h_prob, d_prob, a_prob)
                    h_color = "#38BDF8" if h_prob == max_prob else "#64748B"
                    d_color = "#F59E0B" if d_prob == max_prob else "#64748B"
                    a_color = "#A78BFA" if a_prob == max_prob else "#64748B"
                    home_logo     = _crest_url(f.get("HomeTeam"), f.get("HomeLogo"), f.get("HomeID"))
                    away_logo     = _crest_url(f.get("AwayTeam"), f.get("AwayLogo"), f.get("AwayID"))
                    home_fallback = _fallback_team_logo(f.get("HomeTeam"))
                    away_fallback = _fallback_team_logo(f.get("AwayTeam"))

                    # แปลงวันที่เป็น "01 MAR"
                    raw_date = f.get("Date", "")
                    try:
                        from datetime import datetime as _dt2
                        _d = _dt2.strptime(raw_date, "%d/%m/%Y")
                        date_label = _d.strftime("%d %b").upper()
                    except Exception:
                        date_label = raw_date

                    # Auto-save ทุกแมตช์
                    _add_prediction(
                        f["HomeTeam"], f["AwayTeam"],
                        f.get("Date", ""), f.get("Time", ""),
                        score, h_prob, d_prob, a_prob,
                        f.get("MatchID"),
                    )

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
                        <div class="fxc-score">{score}</div>
                        <div class="fxc-time">{f.get('Time','—')}</div>
                    </div>
                    <div class="fxc-analyse-row" id="fxc-ar-{i}"></div>
                    """, unsafe_allow_html=True)
                    st.button("Analyse", key=f"fx_nav_{i}",
                              use_container_width=True,
                              on_click=_navigate_to_predict,
                              args=(f["HomeTeam"], f["AwayTeam"]))

        elif upcoming is not None:
            st.markdown('<div class="fx-error">Unable to fetch upcoming fixtures. Check your API connection.</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — PREDICTION HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_hist:
        st.markdown("")
        with st.spinner("Checking for real scores..."):
            history = _fetch_real_scores()

        if not history:
            st.markdown("""
                <div style="text-align:center;padding:3rem 0;font-family:'Rajdhani',sans-serif;
                     font-size:1rem;color:rgba(148,187,233,0.35);letter-spacing:2px;">
                    NO PREDICTIONS SAVED YET<br>
                    <span style="font-size:0.8rem;opacity:0.6;">
                        Press "Save & Predict" on any fixture to start tracking
                    </span>
                </div>""", unsafe_allow_html=True)
        else:
            history_sorted = sorted(history, key=lambda x: x.get("saved_at", ""), reverse=True)
            finished   = [e for e in history_sorted if e.get("real_score") is not None]
            total_f    = len(finished)
            w_correct  = sum(1 for e in finished if e.get("winner_correct"))
            s_correct  = sum(1 for e in finished if e.get("score_correct"))
            w_acc = round(w_correct / total_f * 100, 1) if total_f else 0
            s_acc = round(s_correct / total_f * 100, 1) if total_f else 0

            # Summary cards
            a1, a2, a3, a4 = st.columns(4, gap="small")
            for col, val, lbl, color in [
                (a1, len(history_sorted), "TOTAL SAVED",     "#38BDF8"),
                (a2, total_f,             "RESULTS IN",       "#F59E0B"),
                (a3, f"{w_acc}%",         "WINNER ACCURACY",  "#00E676"),
                (a4, f"{s_acc}%",         "SCORE ACCURACY",   "#A78BFA"),
            ]:
                with col:
                    st.markdown(f"""
                        <div class="acc-card">
                            <div class="acc-num" style="color:{color};">{val}</div>
                            <div class="acc-lbl">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin:16px 0 6px;'></div>", unsafe_allow_html=True)

            # Header
            st.markdown("""
            <div class="hst-header">
                <div>DATE</div>
                <div style="text-align:right;">HOME</div>
                <div></div>
                <div>AWAY</div>
                <div style="text-align:center;">PREDICTED</div>
                <div style="text-align:center;">REAL</div>
                <div style="text-align:center;">WINNER</div>
                <div style="text-align:center;">SCORE</div>
            </div>""", unsafe_allow_html=True)

            # Rows
            for e in history_sorted:
                home_logo = _fallback_team_logo(e["home"])
                away_logo = _fallback_team_logo(e["away"])
                real_text = e["real_score"] if e.get("real_score") else "—"
                real_style = "color:#F0F6FF;font-weight:700;" if e.get("real_score") else "color:rgba(148,187,233,0.22);"
                if e.get("winner_correct") is True:
                    w_badge = '<span class="hst-badge hst-correct">✓ Correct</span>'
                elif e.get("winner_correct") is False:
                    w_badge = '<span class="hst-badge hst-wrong">✗ Wrong</span>'
                else:
                    w_badge = '<span class="hst-badge hst-pending">Pending</span>'
                if e.get("score_correct") is True:
                    s_badge = '<span class="hst-badge hst-correct">Exact</span>'
                elif e.get("score_correct") is False:
                    s_badge = '<span class="hst-badge hst-wrong">Off</span>'
                else:
                    s_badge = '<span class="hst-badge hst-pending">—</span>'
                raw_date = e.get("date", "")
                try:
                    from datetime import datetime as _dt3
                    _d = _dt3.strptime(raw_date, "%d/%m/%Y")
                    date_label = _d.strftime("%d %b").upper()
                except Exception:
                    date_label = raw_date
                if e.get("winner_correct") is True:
                    row_accent = "rgba(0,230,118,0.22)"
                elif e.get("winner_correct") is False:
                    row_accent = "rgba(239,68,68,0.18)"
                else:
                    row_accent = "rgba(255,255,255,0.08)"
                st.markdown(f"""
                <div class="hst-row" style="border-color:{row_accent};">
                    <div class="hst-date-col">
                        <div class="hst-date-badge">{date_label}</div>
                        <div class="hst-time">{e.get('time','')}</div>
                    </div>
                    <div class="hst-home">
                        <span class="hst-name">{e['home']}</span>
                        <img class="hst-logo" src="{home_logo}" onerror="this.src='{DEFAULT_LOGO}'"/>
                    </div>
                    <div class="hst-vs">vs</div>
                    <div class="hst-away">
                        <img class="hst-logo" src="{away_logo}" onerror="this.src='{DEFAULT_LOGO}'"/>
                        <span class="hst-name">{e['away']}</span>
                    </div>
                    <div class="hst-score-pred">{e.get('pred_score','—')}</div>
                    <div class="hst-score-real" style="{real_style}">{real_text}</div>
                    <div class="hst-badge-wrap">{w_badge}</div>
                    <div class="hst-badge-wrap">{s_badge}</div>
                </div>
                """, unsafe_allow_html=True)