"""
ui/page_predict.py — Match Prediction page for Football AI Nexus Engine
"""
import os
import streamlit as st
import pandas as pd

from src.config import DATA_DIR, NEW_TEAMS_BOOTSTRAPPED
from src.predict import predict_match, predict_score
from utils import silent

TEAM_LOGOS = {
    "Arsenal":          "https://crests.football-data.org/57.png",
    "Aston Villa":      "https://crests.football-data.org/58.png",
    "Bournemouth":      "https://crests.football-data.org/1044.png",
    "Brentford":        "https://crests.football-data.org/402.png",
    "Brighton":         "https://crests.football-data.org/397.png",
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
    "Southampton":      "https://crests.football-data.org/340.png",
    "Spurs":            "https://crests.football-data.org/73.png",
    "West Ham":         "https://crests.football-data.org/563.png",
    "Wolves":           "https://crests.football-data.org/76.png",
    "Leeds":            "https://crests.football-data.org/341.png",
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
DEFAULT_LOGO = "https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg"

TEAM_API_IDS = {
    "Arsenal": 57, "Aston Villa": 58, "Bournemouth": 1044, "Brentford": 402,
    "Brighton": 397, "Chelsea": 61, "Crystal Palace": 354, "Everton": 62,
    "Fulham": 63, "Ipswich": 678, "Leicester": 338, "Liverpool": 64,
    "Man City": 65, "Man United": 66, "Newcastle": 67, "Nott'm Forest": 351,
    "Southampton": 340, "Spurs": 73, "West Ham": 563, "Wolves": 76,
    "Leeds": 341, "Burnley": 328, "Sheffield Utd": 356, "Luton": 389,
    "Watford": 346, "Norwich": 68, "West Brom": 74, "Huddersfield": 394,
    "Cardiff": 715, "Swansea": 72, "Stoke": 70, "Middlesbrough": 343,
    "Sunderland": 71, "Hull": 322,
}

def _get_logo(team: str) -> str:
    for key, url in TEAM_LOGOS.items():
        if key.lower() in team.lower() or team.lower() in key.lower():
            return url
    return DEFAULT_LOGO

def _get_team_id(team: str):
    for key, tid in TEAM_API_IDS.items():
        if key.lower() in team.lower() or team.lower() in key.lower():
            return tid
    return None


def navigate_to_predict(home_team, away_team):
    st.session_state['nav_page']     = "Predict Match"
    st.session_state['pred_home']    = home_team
    st.session_state['pred_away']    = away_team
    st.session_state['auto_predict'] = True


def page_predict(ctx):
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600;700&display=swap');

    /* ── VS Banner ── */
    .vs-banner {
        background: linear-gradient(135deg, #0a1628, #0d1f3c);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 2.2rem 3rem;
        display: flex; align-items: center; justify-content: space-between;
        margin: 1.2rem 0 1.6rem;
        position: relative; overflow: hidden;
    }
    .vs-banner::before {
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(ellipse at 50% 50%, rgba(0,176,255,0.06), transparent 70%);
        pointer-events: none;
    }
    .vs-team {
        display: flex; flex-direction: column;
        align-items: center; gap: 12px; z-index: 1; flex: 1;
    }
    .vs-team img {
        width: 96px; height: 96px; object-fit: contain;
        filter: drop-shadow(0 6px 20px rgba(0,0,0,0.7));
        transition: transform 0.3s ease;
    }
    .vs-team img:hover { transform: scale(1.1); }
    .vs-team-name {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.9rem; letter-spacing: 0.06em; line-height: 1;
    }
    .vs-home-name { color: #7DD3FC; }
    .vs-away-name { color: #FED7AA; }
    .vs-center {
        font-family: 'Bebas Neue', cursive; font-size: 3.5rem;
        color: rgba(255,255,255,0.08); z-index: 1; padding: 0 2rem; letter-spacing: 0.1em;
    }
    .role-tag {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.68rem; font-weight: 700;
        padding: 4px 16px; border-radius: 20px;
        letter-spacing: 0.14em; text-transform: uppercase;
    }
    .role-home { background: rgba(0,176,255,0.15); color: #00B0FF; border: 1px solid rgba(0,176,255,0.35); }
    .role-away { background: rgba(249,115,22,0.15); color: #F97316; border: 1px solid rgba(249,115,22,0.35); }

    /* ── Section Label ── */
    .eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 0.55rem;
    }

    /* ── Prob Card ── */
    .prob-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px; padding: 1.4rem 1.6rem;
    }
    .prob-row { display: flex; align-items: center; gap: 14px; margin-bottom: 0.8rem; }
    .prob-logo { width: 28px; height: 28px; object-fit: contain; flex-shrink: 0; }
    .prob-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem; font-weight: 600; color: #94A3B8;
        width: 110px; flex-shrink: 0;
    }
    .prob-track {
        flex: 1; height: 10px;
        background: rgba(255,255,255,0.06); border-radius: 5px; overflow: hidden;
    }
    .prob-fill { height: 100%; border-radius: 5px; }
    .prob-pct {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.3rem; width: 56px; text-align: right; letter-spacing: 0.04em;
    }
    .pred-badge {
        display: inline-block; padding: 0.5rem 1.6rem; border-radius: 30px;
        font-family: 'Bebas Neue', cursive; font-size: 1.1rem;
        letter-spacing: 0.1em; margin-top: 0.7rem;
    }

    /* ── xG Card ── */
    .xg-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px; padding: 1.4rem 1.6rem;
    }

    /* ── Form Pills ── */
    .form-pill {
        display: inline-block; width: 28px; height: 28px; border-radius: 50%;
        text-align: center; line-height: 28px;
        font-size: 0.72rem; font-weight: 700; margin-right: 5px;
    }
    .pill-W { background: rgba(0,230,118,0.2);  color: #00E676; border: 1px solid rgba(0,230,118,0.35); }
    .pill-D { background: rgba(245,158,11,0.2); color: #F59E0B; border: 1px solid rgba(245,158,11,0.35); }
    .pill-L { background: rgba(239,68,68,0.2);  color: #EF4444; border: 1px solid rgba(239,68,68,0.35); }

    /* ── Form Table ── */
    .form-table { width: 100%; border-collapse: collapse; margin-top: 0.4rem; }
    .form-table thead tr {
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    .form-table thead th {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase;
        color: #475569; padding: 6px 8px; text-align: left;
    }
    .form-table thead th:last-child { text-align: center; }
    .form-table tbody tr {
        border-bottom: 1px solid rgba(255,255,255,0.05);
        transition: background 0.15s;
    }
    .form-table tbody tr:hover { background: rgba(255,255,255,0.03); }
    .form-table tbody td {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem; color: #94A3B8;
        padding: 9px 8px; vertical-align: middle;
    }
    .form-table .td-date  { color: #475569; font-size: 0.75rem; white-space: nowrap; }
    .form-table .td-comp  {
        font-size: 0.65rem; font-weight: 700;
        background: rgba(255,255,255,0.07); color: #64748B;
        border-radius: 4px; padding: 2px 7px; white-space: nowrap;
        display: inline-block;
    }
    .form-table .td-venue {
        font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
        padding: 2px 7px; border-radius: 4px;
        display: inline-block; white-space: nowrap;
    }
    .venue-h { background: rgba(0,176,255,0.12); color: #00B0FF; }
    .venue-a { background: rgba(249,115,22,0.12); color: #F97316; }
    .form-table .td-opp  { color: #CBD5E1; font-weight: 500; }
    .form-table .td-score {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.05rem; color: #E2E8F0; letter-spacing: 0.08em;
        white-space: nowrap;
    }
    .form-table .td-result { text-align: center; }
    .res-badge {
        display: inline-block; width: 26px; height: 26px;
        border-radius: 6px; text-align: center; line-height: 26px;
        font-family: 'DM Sans', sans-serif; font-size: 0.75rem; font-weight: 800;
    }
    .res-W { background: rgba(0,230,118,0.15); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }
    .res-D { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
    .res-L { background: rgba(239,68,68,0.15);  color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }

    /* ── Selectbox override ── */
    div[data-testid="stSelectbox"] > div > div {
        background: linear-gradient(145deg, #131d2e, #0d1520) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        color: #E2E8F0 !important;
    }
    div[data-testid="stSelectbox"] > div > div:hover {
        border-color: rgba(0,176,255,0.45) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-family:'DM Sans',sans-serif;font-size:0.68rem;font-weight:700;
                  letter-spacing:0.22em;text-transform:uppercase;color:#00B0FF;margin-bottom:0.25rem">
        NEXUS ENGINE · FOOTBALL AI v9.0
      </div>
      <div style="font-family:'Bebas Neue',cursive;font-size:3.2rem;letter-spacing:0.04em;line-height:1;
                  background:linear-gradient(90deg,#fff,#94A3B8);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block">
        MATCH PREDICTION
      </div>
      <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#4B6080;margin-top:0.25rem">
        Select home and away team, then click Generate Prediction to see AI analysis.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Team list ────────────────────────────────────────────────
    all_teams = sorted(
        set(ctx['match_df_clean']['HomeTeam'].tolist() +
            ctx['match_df_clean']['AwayTeam'].tolist()) |
        set(NEW_TEAMS_BOOTSTRAPPED.keys())
    )
    default_h = st.session_state.get('pred_home', 'Arsenal')
    default_a = st.session_state.get('pred_away', 'Chelsea')
    if default_h not in all_teams: default_h = all_teams[0]
    if default_a not in all_teams: default_a = all_teams[1]

    # ── Dropdowns ────────────────────────────────────────────────
    col_h, col_vs, col_a = st.columns([5, 1, 5], gap="small")

    with col_h:
        st.markdown('<div class="eyebrow" style="color:#00B0FF">Home Team</div>',
                    unsafe_allow_html=True)
        home = st.selectbox("home_team", all_teams,
                            index=all_teams.index(default_h),
                            label_visibility="collapsed", key="sel_home")
        st.session_state['pred_home'] = home

    with col_vs:
        st.markdown(
            '<div style="text-align:center;margin-top:1.9rem;font-family:Bebas Neue,cursive;'
            'font-size:1.5rem;color:rgba(255,255,255,0.18);letter-spacing:0.1em">VS</div>',
            unsafe_allow_html=True)

    with col_a:
        st.markdown('<div class="eyebrow" style="color:#F97316">Away Team</div>',
                    unsafe_allow_html=True)
        away_opts = [t for t in all_teams if t != home]
        idx_a = away_opts.index(default_a) if default_a in away_opts else 0
        away = st.selectbox("away_team", away_opts, index=idx_a,
                            label_visibility="collapsed", key="sel_away")
        st.session_state['pred_away'] = away

    # ── VS Banner ────────────────────────────────────────────────
    h_logo = _get_logo(home)
    a_logo = _get_logo(away)

    st.markdown(f"""
    <div class="vs-banner">
      <div class="vs-team">
        <img src="{h_logo}" onerror="this.src='{DEFAULT_LOGO}'"/>
        <div class="vs-team-name vs-home-name">{home}</div>
        <span class="role-tag role-home">Home</span>
      </div>
      <div class="vs-center">VS</div>
      <div class="vs-team">
        <img src="{a_logo}" onerror="this.src='{DEFAULT_LOGO}'"/>
        <div class="vs-team-name vs-away-name">{away}</div>
        <span class="role-tag role-away">Away</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Predict Button ───────────────────────────────────────────
    auto_run = st.session_state.pop('auto_predict', False)

    if st.button("GENERATE PREDICTION", type="primary",
                 use_container_width=True) or auto_run:
        if home == away:
            st.warning("Please select two different teams.")
            return
        with st.spinner("Analyzing match data..."):
            r = silent(predict_match, home, away, ctx)
            s = silent(predict_score, home, away, ctx)
        if r:
            st.divider()
            _render_results(home, away, h_logo, a_logo, r, s)
            st.divider()
            _render_recent_form(home, away)


# ─────────────────────────────────────────────────────────────
def _render_results(home, away, h_logo, a_logo, r, s):
    st.markdown('<div class="eyebrow" style="color:#3D5068;margin-top:0.5rem">Prediction Analysis</div>',
                unsafe_allow_html=True)
    c_prob, c_score = st.columns([1.3, 1], gap="large")

    with c_prob:
        hw, dr, aw = r['Home Win'], r['Draw'], r['Away Win']
        pred = r['Prediction']
        pred_color = "#00B0FF" if "Home" in pred else ("#F59E0B" if "Draw" in pred else "#F97316")
        pred_bg    = ("rgba(0,176,255,0.12)"  if "Home" in pred else
                      "rgba(245,158,11,0.12)" if "Draw" in pred else
                      "rgba(249,115,22,0.12)")
        st.markdown(f"""
        <div class="prob-card">
          <div class="prob-row">
            <img class="prob-logo" src="{h_logo}" onerror="this.style.opacity='0.3'"/>
            <div class="prob-label">{home[:18]}</div>
            <div class="prob-track">
              <div class="prob-fill" style="width:{hw}%;background:linear-gradient(90deg,#00B0FF,#38BDF8)"></div>
            </div>
            <div class="prob-pct" style="color:#00B0FF">{hw}%</div>
          </div>
          <div class="prob-row">
            <div class="prob-logo" style="text-align:center;line-height:28px;font-size:0.85rem">—</div>
            <div class="prob-label">Draw</div>
            <div class="prob-track">
              <div class="prob-fill" style="width:{dr}%;background:linear-gradient(90deg,#F59E0B,#FCD34D)"></div>
            </div>
            <div class="prob-pct" style="color:#F59E0B">{dr}%</div>
          </div>
          <div class="prob-row" style="margin-bottom:0">
            <img class="prob-logo" src="{a_logo}" onerror="this.style.opacity='0.3'"/>
            <div class="prob-label">{away[:18]}</div>
            <div class="prob-track">
              <div class="prob-fill" style="width:{aw}%;background:linear-gradient(90deg,#F97316,#FB923C)"></div>
            </div>
            <div class="prob-pct" style="color:#F97316">{aw}%</div>
          </div>
          <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06)">
            <span style="font-family:'DM Sans',sans-serif;font-size:0.78rem;color:#475569;font-weight:600">
              PREDICTED OUTCOME
            </span>
            <span class="pred-badge"
                  style="background:{pred_bg};color:{pred_color};
                         border:1px solid {pred_color}44;margin-left:8px">
              {pred}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c_score:
        if s:
            rows_html = "".join(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.05)">'
                f'<span style="font-family:Bebas Neue,cursive;font-size:1.15rem;'
                f'color:#E2E8F0;letter-spacing:0.08em">{sc[0]}</span>'
                f'<span style="font-family:DM Sans,sans-serif;font-size:0.82rem;'
                f'font-weight:600;color:#475569">{sc[1]}%</span>'
                f'</div>'
                for sc in s['top5_scores']
            )
            st.markdown(f"""
            <div class="xg-card">
              <div style="display:flex;justify-content:space-around;align-items:center;margin-bottom:1.2rem">
                <div style="text-align:center">
                  <img src="{h_logo}" style="width:44px;height:44px;object-fit:contain"/>
                  <div style="font-family:'Bebas Neue',cursive;font-size:2.6rem;
                              color:#7DD3FC;line-height:1.1;margin-top:6px">{s['home_xg']}</div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                              color:#3D5068;text-transform:uppercase;letter-spacing:0.14em">xG Home</div>
                </div>
                <div style="font-family:'Bebas Neue',cursive;font-size:2rem;color:rgba(255,255,255,0.08)">:</div>
                <div style="text-align:center">
                  <img src="{a_logo}" style="width:44px;height:44px;object-fit:contain"/>
                  <div style="font-family:'Bebas Neue',cursive;font-size:2.6rem;
                              color:#FED7AA;line-height:1.1;margin-top:6px">{s['away_xg']}</div>
                  <div style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                              color:#3D5068;text-transform:uppercase;letter-spacing:0.14em">xG Away</div>
                </div>
              </div>
              <div style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:700;
                          letter-spacing:0.16em;text-transform:uppercase;color:#3D5068;margin-bottom:0.6rem">
                Top Predicted Scores
              </div>
              {rows_html}
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
def _render_recent_form(home, away):
    from src.config import API_KEY
    import requests

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_api(team_id: int):
        if not team_id:
            return None
        try:
            r = requests.get(
                f"https://api.football-data.org/v4/teams/{team_id}/matches",
                headers={"X-Auth-Token": API_KEY},
                params={"status": "FINISHED", "limit": 5},
                timeout=8,
            )
            if not r.ok:
                return None
            rows = []
            for m in r.json().get("matches", []):
                is_home  = m["homeTeam"]["id"] == team_id
                opp_id   = m["awayTeam"]["id"] if is_home else m["homeTeam"]["id"]
                opp_name = (m["awayTeam"].get("shortName") or m["awayTeam"]["name"]) if is_home \
                           else (m["homeTeam"].get("shortName") or m["homeTeam"]["name"])
                gf = m["score"]["fullTime"]["home"] if is_home else m["score"]["fullTime"]["away"]
                ga = m["score"]["fullTime"]["away"] if is_home else m["score"]["fullTime"]["home"]
                if gf is None or ga is None:
                    continue
                rows.append({
                    "Date":        m["utcDate"][:10],
                    "Competition": m.get("competition", {}).get("name", ""),
                    "Opponent":    opp_name,
                    "Opp_ID":      opp_id,
                    "Venue":       "H" if is_home else "A",
                    "GF": int(gf), "GA": int(ga),
                    "Result":      "W" if gf > ga else ("D" if gf == ga else "L"),
                })
            rows.sort(key=lambda x: x["Date"], reverse=True)
            return rows[:5]
        except Exception:
            return None

    @st.cache_data(ttl=300, show_spinner=False)
    def _fallback_csv(team_name: str):
        import glob as _g
        dfs = []
        for f in _g.glob(os.path.join(DATA_DIR, "*.csv")):
            if "backup" in f.lower(): continue
            try:
                _df = pd.read_csv(f)
                _df["FTHG"] = pd.to_numeric(_df["FTHG"], errors="coerce")
                _df["FTAG"] = pd.to_numeric(_df["FTAG"], errors="coerce")
                _df["Date"] = pd.to_datetime(_df["Date"], dayfirst=True, errors="coerce")
                dfs.append(_df)
            except Exception: pass
        if not dfs: return []
        c = pd.concat(dfs, ignore_index=True)
        c = c.drop_duplicates(subset=["Date","HomeTeam","AwayTeam"], keep="last")
        c = c.dropna(subset=["FTHG","FTAG"]).sort_values("Date")
        hm = c[c["HomeTeam"]==team_name].copy()
        hm["Venue"]="H"; hm["GF"]=hm["FTHG"]; hm["GA"]=hm["FTAG"]; hm["Opponent"]=hm["AwayTeam"]
        am = c[c["AwayTeam"]==team_name].copy()
        am["Venue"]="A"; am["GF"]=am["FTAG"];  am["GA"]=am["FTHG"]; am["Opponent"]=am["HomeTeam"]
        all_m = pd.concat([hm, am]).sort_values("Date", ascending=False).head(5)
        rows = []
        for _, row in all_m.iterrows():
            gf, ga = int(row["GF"]), int(row["GA"])
            rows.append({
                "Date": row["Date"].strftime("%Y-%m-%d"),
                "Competition": "Premier League",
                "Opponent": row["Opponent"], "Opp_ID": None,
                "Venue": row["Venue"], "GF": gf, "GA": ga,
                "Result": "W" if gf > ga else ("D" if gf == ga else "L"),
            })
        return rows

    # ── Section header ───────────────────────────────────────────
    st.markdown(
        '<div class="eyebrow" style="color:#3D5068;margin-bottom:0.8rem">'
        'Recent Form — Last 5 Matches</div>',
        unsafe_allow_html=True)

    COMP_SHORT = {
        "Premier League": "PL", "FA Cup": "FA Cup",
        "EFL Cup": "League Cup", "UEFA Champions League": "UCL",
        "UEFA Europa League": "UEL", "UEFA Conference League": "UECL",
    }

    ch, ca = st.columns(2, gap="large")

    for team, col, accent in [(home, ch, "#7DD3FC"), (away, ca, "#FED7AA")]:
        with col:
            logo    = _get_logo(team)
            team_id = _get_team_id(team)

            # Team header
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.7rem">
              <img src="{logo}" style="width:32px;height:32px;object-fit:contain"/>
              <span style="font-family:'Bebas Neue',cursive;font-size:1.4rem;
                           letter-spacing:0.05em;color:{accent}">{team}</span>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner(f"Fetching {team} matches..."):
                rows = _fetch_api(team_id)

            source = "Real-time · API"
            if not rows:
                rows = _fallback_csv(team)
                source = "Local CSV"

            if not rows:
                st.warning("No match data available.")
                continue

            # Form pills + source
            pills = "".join(
                f'<span class="form-pill pill-{r["Result"]}">{r["Result"]}</span>'
                for r in rows
            )
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.8rem">
              {pills}
              <span style="font-family:'DM Sans',sans-serif;font-size:0.62rem;
                           color:#3D5068;font-weight:500">{source}</span>
            </div>
            """, unsafe_allow_html=True)

            # Table
            rows_html = ""
            for r in rows:
                opp_logo = (f"https://crests.football-data.org/{r['Opp_ID']}.png"
                            if r["Opp_ID"] else DEFAULT_LOGO)
                res_cls  = f"res-{r['Result']}"
                venue_cls = "venue-h" if r["Venue"] == "H" else "venue-a"
                venue_txt = "Home" if r["Venue"] == "H" else "Away"
                comp = COMP_SHORT.get(r["Competition"], r["Competition"][:6])
                date_disp = r["Date"][5:].replace("-", "/")   # MM/DD

                rows_html += f"""
                <tr>
                  <td class="td-date">{date_disp}</td>
                  <td><span class="td-comp">{comp}</span></td>
                  <td><span class="td-venue {venue_cls}">{venue_txt}</span></td>
                  <td class="td-opp">
                    <div style="display:flex;align-items:center;gap:7px">
                      <img src="{opp_logo}" style="width:20px;height:20px;object-fit:contain"
                           onerror="this.style.opacity='0.2'"/>
                      {r["Opponent"]}
                    </div>
                  </td>
                  <td class="td-score">{r["GF"]} – {r["GA"]}</td>
                  <td class="td-result"><span class="res-badge {res_cls}">{r["Result"]}</span></td>
                </tr>"""

            st.markdown(f"""
            <table class="form-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Comp</th>
                  <th>Venue</th>
                  <th>Opponent</th>
                  <th>Score</th>
                  <th>Res</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)