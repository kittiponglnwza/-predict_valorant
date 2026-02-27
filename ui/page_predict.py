"""
ui/page_predict.py — Match Prediction page for Football AI Nexus Engine
"""
import os
import streamlit as st
import pandas as pd

from src.config import DATA_DIR, NEW_TEAMS_BOOTSTRAPPED
from src.predict import predict_match, predict_score
from utils import silent

# ── โลโก้ทีม Premier League ───────────────────────────────────
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

def _get_logo(team_name: str) -> str:
    for key, url in TEAM_LOGOS.items():
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return url
    return DEFAULT_LOGO


def navigate_to_predict(home_team, away_team):
    st.session_state['nav_page']    = "Predict Match"
    st.session_state['pred_home']   = home_team
    st.session_state['pred_away']   = away_team
    st.session_state['auto_predict'] = True


def page_predict(ctx):
    # ── CSS ──────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    /* VS Banner */
    .vs-banner {
        background: linear-gradient(135deg, #0a1628, #0d1f3c);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 1.8rem 2.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1rem 0 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .vs-banner::before {
        content: '';
        position: absolute; inset: 0;
        background: radial-gradient(ellipse at 50% 50%, rgba(0,176,255,0.05), transparent 70%);
        pointer-events: none;
    }
    .vs-team {
        display: flex; flex-direction: column;
        align-items: center; gap: 10px;
        z-index: 1; flex: 1;
    }
    .vs-team img {
        width: 80px; height: 80px;
        object-fit: contain;
        filter: drop-shadow(0 4px 16px rgba(0,0,0,0.6));
        transition: transform 0.3s ease;
    }
    .vs-team img:hover { transform: scale(1.08); }
    .vs-team-name {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.6rem; letter-spacing: 0.06em; line-height: 1;
    }
    .vs-home-name { color: #7DD3FC; }
    .vs-away-name { color: #FED7AA; }
    .vs-center {
        font-family: 'Bebas Neue', cursive;
        font-size: 3rem;
        color: rgba(255,255,255,0.1);
        z-index: 1; padding: 0 1.5rem;
        letter-spacing: 0.1em;
    }
    .role-tag {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.58rem; font-weight: 700;
        padding: 3px 12px; border-radius: 20px;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .role-home { background: rgba(0,176,255,0.15); color: #00B0FF; border: 1px solid rgba(0,176,255,0.3); }
    .role-away { background: rgba(249,115,22,0.15); color: #F97316; border: 1px solid rgba(249,115,22,0.3); }

    /* Custom Selectbox styling */
    div[data-testid="stSelectbox"] > div > div {
        background: linear-gradient(145deg, #131d2e, #0d1520) !important;
        border: 1.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
        padding: 0.2rem 0.5rem !important;
        color: #E2E8F0 !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-testid="stSelectbox"] > div > div:hover {
        border-color: rgba(0,176,255,0.4) !important;
    }

    /* Prob bars */
    .prob-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
    }
    .prob-row {
        display: flex; align-items: center; gap: 12px;
        margin-bottom: 0.65rem;
    }
    .prob-logo { width: 24px; height: 24px; object-fit: contain; flex-shrink: 0; }
    .prob-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem; color: #94A3B8; width: 100px; flex-shrink: 0;
    }
    .prob-track {
        flex: 1; height: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 4px; overflow: hidden;
    }
    .prob-fill { height: 100%; border-radius: 4px; }
    .prob-pct {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.15rem; width: 50px; text-align: right;
    }
    .pred-badge {
        display: inline-block;
        padding: 0.45rem 1.4rem;
        border-radius: 30px;
        font-family: 'Bebas Neue', cursive;
        font-size: 1.05rem; letter-spacing: 0.1em;
        margin-top: 0.6rem;
    }

    /* xG card */
    .xg-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
    }

    /* Form pills */
    .form-pill {
        display: inline-block;
        width: 24px; height: 24px; border-radius: 50%;
        text-align: center; line-height: 24px;
        font-size: 0.65rem; font-weight: 700;
        margin-right: 4px;
    }
    .pill-W { background: rgba(0,230,118,0.2); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }
    .pill-D { background: rgba(245,158,11,0.2); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
    .pill-L { background: rgba(239,68,68,0.2);  color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }

    .section-eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.62rem; font-weight: 700;
        letter-spacing: 0.2em; text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1rem">
        <div style="font-family:'DM Sans',sans-serif;font-size:0.62rem;font-weight:700;
                    letter-spacing:0.22em;text-transform:uppercase;color:#00B0FF;margin-bottom:0.2rem">
            ⚡ Nexus Engine
        </div>
        <div style="font-family:'Bebas Neue',cursive;font-size:2.8rem;letter-spacing:0.04em;line-height:1;
                    background:linear-gradient(90deg,#fff,#94A3B8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block">
            MATCH PREDICTION
        </div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#4B6080;margin-top:0.2rem">
            เลือกทีมเหย้าและทีมเยือน แล้วกด Predict เพื่อดูผลวิเคราะห์จาก AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Build team list ──────────────────────────────────────────
    all_teams = sorted(
        set(ctx['match_df_clean']['HomeTeam'].tolist() +
            ctx['match_df_clean']['AwayTeam'].tolist()) |
        set(NEW_TEAMS_BOOTSTRAPPED.keys())
    )

    # defaults
    default_h = st.session_state.get('pred_home', 'Arsenal')
    default_a = st.session_state.get('pred_away', 'Chelsea')
    if default_h not in all_teams: default_h = all_teams[0]
    if default_a not in all_teams: default_a = all_teams[1]

    # ── Dropdown selectors ───────────────────────────────────────
    col_h, col_vs, col_a = st.columns([5, 1, 5], gap="small")

    with col_h:
        st.markdown('<div class="section-eyebrow" style="color:#00B0FF">🏠 ทีมเหย้า</div>',
                    unsafe_allow_html=True)
        home = st.selectbox(
            "home_team", all_teams,
            index=all_teams.index(default_h),
            label_visibility="collapsed",
            key="sel_home"
        )
        st.session_state['pred_home'] = home

    with col_vs:
        st.markdown('<div style="text-align:center;margin-top:2rem;'
                    'font-family:Bebas Neue,cursive;font-size:1.4rem;'
                    'color:rgba(255,255,255,0.2);letter-spacing:0.1em">VS</div>',
                    unsafe_allow_html=True)

    with col_a:
        st.markdown('<div class="section-eyebrow" style="color:#F97316">✈️ ทีมเยือน</div>',
                    unsafe_allow_html=True)
        away_opts = [t for t in all_teams if t != home]
        idx_a = away_opts.index(default_a) if default_a in away_opts else 0
        away = st.selectbox(
            "away_team", away_opts,
            index=idx_a,
            label_visibility="collapsed",
            key="sel_away"
        )
        st.session_state['pred_away'] = away

    # ── VS Banner with logos ─────────────────────────────────────
    h_logo = _get_logo(home)
    a_logo = _get_logo(away)

    st.markdown(f"""
    <div class="vs-banner">
        <div class="vs-team">
            <img src="{h_logo}"
                 onerror="this.src='{DEFAULT_LOGO}'"/>
            <div class="vs-team-name vs-home-name">{home}</div>
            <span class="role-tag role-home">Home</span>
        </div>
        <div class="vs-center">VS</div>
        <div class="vs-team">
            <img src="{a_logo}"
                 onerror="this.src='{DEFAULT_LOGO}'"/>
            <div class="vs-team-name vs-away-name">{away}</div>
            <span class="role-tag role-away">Away</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Predict Button ───────────────────────────────────────────
    auto_run = st.session_state.pop('auto_predict', False)

    if st.button("🚀  GENERATE PREDICTION", type="primary",
                 use_container_width=True) or auto_run:
        if home == away:
            st.warning("⚠️ กรุณาเลือกทีมคนละทีม")
            return

        with st.spinner("🤖 AI กำลังวิเคราะห์..."):
            r = silent(predict_match, home, away, ctx)
            s = silent(predict_score, home, away, ctx)

        if r:
            st.divider()
            _render_results(home, away, h_logo, a_logo, r, s)
            st.divider()
            _render_recent_form(home, away)


# ─────────────────────────────────────────────────────────────
def _render_results(home, away, h_logo, a_logo, r, s):
    st.markdown('<div class="section-eyebrow" style="color:#3D5068">📊 ผลการวิเคราะห์</div>',
                unsafe_allow_html=True)

    c_prob, c_score = st.columns([1.3, 1], gap="large")

    with c_prob:
        hw = r['Home Win']
        dr = r['Draw']
        aw = r['Away Win']
        pred = r['Prediction']

        pred_color = "#00B0FF" if "Home" in pred else ("#F59E0B" if "Draw" in pred else "#F97316")
        pred_bg    = ("rgba(0,176,255,0.12)"  if "Home" in pred else
                      "rgba(245,158,11,0.12)" if "Draw" in pred else
                      "rgba(249,115,22,0.12)")

        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-row">
                <img class="prob-logo" src="{h_logo}" onerror="this.style.opacity='0.3'"/>
                <div class="prob-label">{home[:16]}</div>
                <div class="prob-track">
                    <div class="prob-fill"
                         style="width:{hw}%;background:linear-gradient(90deg,#00B0FF,#38BDF8)"></div>
                </div>
                <div class="prob-pct" style="color:#00B0FF">{hw}%</div>
            </div>
            <div class="prob-row">
                <div class="prob-logo" style="text-align:center;line-height:24px;font-size:0.8rem">🤝</div>
                <div class="prob-label">Draw</div>
                <div class="prob-track">
                    <div class="prob-fill"
                         style="width:{dr}%;background:linear-gradient(90deg,#F59E0B,#FCD34D)"></div>
                </div>
                <div class="prob-pct" style="color:#F59E0B">{dr}%</div>
            </div>
            <div class="prob-row" style="margin-bottom:0">
                <img class="prob-logo" src="{a_logo}" onerror="this.style.opacity='0.3'"/>
                <div class="prob-label">{away[:16]}</div>
                <div class="prob-track">
                    <div class="prob-fill"
                         style="width:{aw}%;background:linear-gradient(90deg,#F97316,#FB923C)"></div>
                </div>
                <div class="prob-pct" style="color:#F97316">{aw}%</div>
            </div>
            <div style="margin-top:0.8rem;padding-top:0.8rem;
                        border-top:1px solid rgba(255,255,255,0.05)">
                <span style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#475569">
                    💡 Predicted:
                </span>
                <span class="pred-badge"
                      style="background:{pred_bg};color:{pred_color};
                             border:1px solid {pred_color}44;margin-left:6px">
                    {pred}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_score:
        if s:
            st.markdown(f"""
            <div class="xg-card">
                <div style="display:flex;justify-content:space-around;
                            align-items:center;margin-bottom:1rem">
                    <div style="text-align:center">
                        <img src="{h_logo}" style="width:36px;height:36px;object-fit:contain"/>
                        <div style="font-family:'Bebas Neue',cursive;font-size:2.2rem;
                                    color:#7DD3FC;line-height:1.1;margin-top:4px">
                            {s['home_xg']}
                        </div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.58rem;
                                    color:#3D5068;text-transform:uppercase;letter-spacing:0.12em">
                            xG Home
                        </div>
                    </div>
                    <div style="font-family:'Bebas Neue',cursive;font-size:1.8rem;
                                color:rgba(255,255,255,0.1)">:</div>
                    <div style="text-align:center">
                        <img src="{a_logo}" style="width:36px;height:36px;object-fit:contain"/>
                        <div style="font-family:'Bebas Neue',cursive;font-size:2.2rem;
                                    color:#FED7AA;line-height:1.1;margin-top:4px">
                            {s['away_xg']}
                        </div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.58rem;
                                    color:#3D5068;text-transform:uppercase;letter-spacing:0.12em">
                            xG Away
                        </div>
                    </div>
                </div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.6rem;font-weight:700;
                            letter-spacing:0.15em;text-transform:uppercase;color:#3D5068;
                            margin-bottom:0.5rem">
                    🎯 Top Predicted Scores
                </div>
            """, unsafe_allow_html=True)

            for sc_row in s['top5_scores']:
                score_str, prob = sc_row[0], sc_row[1]
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
                    <span style="font-family:'Bebas Neue',cursive;font-size:1.05rem;
                                 color:#E2E8F0;letter-spacing:0.08em">{score_str}</span>
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.75rem;
                                 color:#475569">{prob}%</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
def _render_recent_form(home, away):
    @st.cache_data(ttl=300)
    def _load_fresh():
        import glob as _g
        dfs = []
        for f in _g.glob(os.path.join(DATA_DIR, "*.csv")):
            if 'backup' in f.lower(): continue
            try:
                _df = pd.read_csv(f)
                _df['FTHG'] = pd.to_numeric(_df['FTHG'], errors='coerce')
                _df['FTAG'] = pd.to_numeric(_df['FTAG'], errors='coerce')
                _df['Date'] = pd.to_datetime(_df['Date'], dayfirst=True, errors='coerce')
                dfs.append(_df)
            except Exception: pass
        if not dfs: return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date','HomeTeam','AwayTeam'], keep='last')
        return combined.dropna(subset=['FTHG','FTAG']).sort_values('Date').reset_index(drop=True)

    def _form(team, data):
        hm = data[data['HomeTeam']==team].copy()
        hm['Venue']='H'; hm['GF']=hm['FTHG']; hm['GA']=hm['FTAG']; hm['Opponent']=hm['AwayTeam']
        am = data[data['AwayTeam']==team].copy()
        am['Venue']='A'; am['GF']=am['FTAG'];  am['GA']=am['FTHG']; am['Opponent']=am['HomeTeam']
        all_m = pd.concat([hm, am]).sort_values('Date', ascending=False).head(5)
        def rl(r):
            if r['GF']>r['GA']: return 'W'
            elif r['GF']==r['GA']: return 'D'
            else: return 'L'
        all_m['Result'] = all_m.apply(rl, axis=1)
        return all_m

    fresh = _load_fresh()
    latest = fresh['Date'].max() if len(fresh)>0 else None

    st.markdown('<div class="section-eyebrow" style="color:#3D5068">📋 Recent Form — Last 5 Matches</div>',
                unsafe_allow_html=True)
    if latest is not None and pd.notna(latest):
        st.caption(f"ข้อมูลล่าสุดใน CSV: **{latest.strftime('%d %b %Y')}**")

    ch, ca = st.columns(2, gap="large")
    for team, col, color in [(home, ch, "#7DD3FC"), (away, ca, "#FED7AA")]:
        with col:
            logo = _get_logo(team)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.6rem">
                <img src="{logo}" style="width:26px;height:26px;object-fit:contain"/>
                <span style="font-family:'DM Sans',sans-serif;font-weight:700;
                             font-size:0.9rem;color:{color}">{team}</span>
            </div>
            """, unsafe_allow_html=True)
            if len(fresh) > 0:
                try:
                    d = _form(team, fresh)
                    pills = "".join(
                        f'<span class="form-pill pill-{row["Result"]}">{row["Result"]}</span>'
                        for _, row in d.iterrows()
                    )
                    st.markdown(f'<div style="margin-bottom:0.5rem">{pills}</div>',
                                unsafe_allow_html=True)
                    d = d[['Date','Opponent','Venue','GF','GA','Result']].copy()
                    d['Date'] = d['Date'].dt.strftime('%d/%m/%y')
                    st.dataframe(d, hide_index=True, use_container_width=True)
                except Exception:
                    st.warning("No recent data available.")
            else:
                st.warning("No recent data available.")