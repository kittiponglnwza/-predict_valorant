"""
ui/page_predict.py — Match Prediction page for Football AI Nexus Engine
"""
import os
import streamlit as st
import pandas as pd

from src.config import DATA_DIR, NEW_TEAMS_BOOTSTRAPPED
from src.predict import predict_match, predict_score
from utils import silent

# ── โลโก้ทีม Premier League (Football-Data.org crest URLs) ────
TEAM_LOGOS = {
    "Arsenal":              "https://crests.football-data.org/57.png",
    "Aston Villa":          "https://crests.football-data.org/58.png",
    "Bournemouth":          "https://crests.football-data.org/1044.png",
    "Brentford":            "https://crests.football-data.org/402.png",
    "Brighton":             "https://crests.football-data.org/397.png",
    "Chelsea":              "https://crests.football-data.org/61.png",
    "Crystal Palace":       "https://crests.football-data.org/354.png",
    "Everton":              "https://crests.football-data.org/62.png",
    "Fulham":               "https://crests.football-data.org/63.png",
    "Ipswich":              "https://crests.football-data.org/678.png",
    "Leicester":            "https://crests.football-data.org/338.png",
    "Liverpool":            "https://crests.football-data.org/64.png",
    "Man City":             "https://crests.football-data.org/65.png",
    "Man United":           "https://crests.football-data.org/66.png",
    "Newcastle":            "https://crests.football-data.org/67.png",
    "Nott'm Forest":        "https://crests.football-data.org/351.png",
    "Southampton":          "https://crests.football-data.org/340.png",
    "Spurs":                "https://crests.football-data.org/73.png",
    "West Ham":             "https://crests.football-data.org/563.png",
    "Wolves":               "https://crests.football-data.org/76.png",
    # ทีมที่เคยอยู่ Premier League
    "Leeds":                "https://crests.football-data.org/341.png",
    "Burnley":              "https://crests.football-data.org/328.png",
    "Sheffield Utd":        "https://crests.football-data.org/356.png",
    "Luton":                "https://crests.football-data.org/389.png",
    "Watford":              "https://crests.football-data.org/346.png",
    "Norwich":              "https://crests.football-data.org/68.png",
    "West Brom":            "https://crests.football-data.org/74.png",
    "Huddersfield":         "https://crests.football-data.org/394.png",
    "Cardiff":              "https://crests.football-data.org/715.png",
    "Swansea":              "https://crests.football-data.org/72.png",
    "Stoke":                "https://crests.football-data.org/70.png",
    "Middlesbrough":        "https://crests.football-data.org/343.png",
    "Sunderland":           "https://crests.football-data.org/71.png",
    "Hull":                 "https://crests.football-data.org/322.png",
}

DEFAULT_LOGO = "https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg"

def _get_logo(team_name: str) -> str:
    """หาโลโก้ทีม — ถ้าไม่เจอใช้โลโก้ PL default"""
    for key, url in TEAM_LOGOS.items():
        if key.lower() in team_name.lower() or team_name.lower() in key.lower():
            return url
    return DEFAULT_LOGO


def navigate_to_predict(home_team, away_team):
    st.session_state['nav_page'] = "Predict Match"
    st.session_state['pred_home'] = home_team
    st.session_state['pred_away'] = away_team
    st.session_state['auto_predict'] = True


def page_predict(ctx):
    # ── CSS ──────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

    /* ── Team Card Grid ── */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
        gap: 10px;
        margin: 0.5rem 0 1.2rem;
    }
    .team-card {
        background: linear-gradient(145deg, #131d2e, #0d1520);
        border: 1.5px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 14px 8px 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.22s cubic-bezier(0.25,0.8,0.25,1);
        position: relative;
        overflow: hidden;
    }
    .team-card::before {
        content: '';
        position: absolute; inset: 0;
        background: radial-gradient(circle at 50% 0%, rgba(0,176,255,0.08), transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .team-card:hover { 
        transform: translateY(-4px);
        border-color: rgba(0,176,255,0.4);
        box-shadow: 0 8px 24px rgba(0,176,255,0.12);
    }
    .team-card:hover::before { opacity: 1; }
    .team-card.selected-home {
        border-color: #00B0FF !important;
        background: linear-gradient(145deg, #0d2035, #091828) !important;
        box-shadow: 0 0 0 2px rgba(0,176,255,0.25), 0 8px 24px rgba(0,176,255,0.2) !important;
    }
    .team-card.selected-away {
        border-color: #F97316 !important;
        background: linear-gradient(145deg, #201208, #160d06) !important;
        box-shadow: 0 0 0 2px rgba(249,115,22,0.25), 0 8px 24px rgba(249,115,22,0.2) !important;
    }
    .team-card img {
        width: 44px; height: 44px;
        object-fit: contain;
        margin-bottom: 6px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.4));
    }
    .team-card .team-name {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.62rem;
        font-weight: 600;
        color: #94A3B8;
        line-height: 1.2;
        letter-spacing: 0.01em;
    }
    .team-card.selected-home .team-name { color: #7DD3FC; }
    .team-card.selected-away .team-name { color: #FED7AA; }
    .home-badge {
        position: absolute; top: 5px; right: 5px;
        background: #00B0FF; color: #000;
        font-size: 0.45rem; font-weight: 800;
        padding: 1px 5px; border-radius: 6px;
        letter-spacing: 0.08em;
    }
    .away-badge {
        position: absolute; top: 5px; right: 5px;
        background: #F97316; color: #fff;
        font-size: 0.45rem; font-weight: 800;
        padding: 1px 5px; border-radius: 6px;
        letter-spacing: 0.08em;
    }

    /* ── VS Banner ── */
    .vs-banner {
        background: linear-gradient(135deg, #0a1628, #0d1f3c);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 1.6rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0.8rem 0 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .vs-banner::before {
        content: '';
        position: absolute; inset: 0;
        background: radial-gradient(ellipse at 50% 50%, rgba(0,176,255,0.05), transparent 70%);
    }
    .vs-team {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
        z-index: 1;
        flex: 1;
    }
    .vs-team img {
        width: 64px; height: 64px;
        object-fit: contain;
        filter: drop-shadow(0 4px 12px rgba(0,0,0,0.5));
    }
    .vs-team-name {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.4rem;
        letter-spacing: 0.06em;
        line-height: 1;
    }
    .vs-home-name { color: #7DD3FC; }
    .vs-away-name { color: #FED7AA; }
    .vs-center {
        font-family: 'Bebas Neue', cursive;
        font-size: 2.5rem;
        color: rgba(255,255,255,0.15);
        z-index: 1;
        letter-spacing: 0.1em;
        padding: 0 1rem;
    }
    .vs-placeholder {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.75rem;
        color: #3D5068;
        font-style: italic;
    }

    /* ── Section Labels ── */
    .section-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .label-home { color: #00B0FF; }
    .label-away { color: #F97316; }
    .label-neutral { color: #3D5068; }

    /* ── Result Cards ── */
    .prob-bar-wrap {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .prob-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.5rem;
    }
    .prob-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        color: #94A3B8;
        width: 80px;
        flex-shrink: 0;
    }
    .prob-track {
        flex: 1;
        height: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s cubic-bezier(0.25,0.8,0.25,1);
    }
    .prob-pct {
        font-family: 'Bebas Neue', cursive;
        font-size: 1.1rem;
        width: 48px;
        text-align: right;
        letter-spacing: 0.05em;
    }
    .prediction-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 30px;
        font-family: 'Bebas Neue', cursive;
        font-size: 1.05rem;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }
    .form-table-wrap { margin-top: 0.5rem; }
    .form-pill {
        display: inline-block;
        width: 22px; height: 22px;
        border-radius: 50%;
        text-align: center;
        line-height: 22px;
        font-size: 0.65rem;
        font-weight: 700;
        margin-right: 3px;
    }
    .pill-W { background: rgba(0,230,118,0.2); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }
    .pill-D { background: rgba(245,158,11,0.2); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
    .pill-L { background: rgba(239,68,68,0.2); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.2rem">
        <div style="font-family:'DM Sans',sans-serif;font-size:0.62rem;font-weight:700;
                    letter-spacing:0.22em;text-transform:uppercase;color:#00B0FF;margin-bottom:0.2rem">
            ⚡ Nexus Engine
        </div>
        <div style="font-family:'Bebas Neue',cursive;font-size:2.8rem;letter-spacing:0.04em;
                    line-height:1;background:linear-gradient(90deg,#fff,#94A3B8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block">
            MATCH PREDICTION
        </div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:#4B6080;margin-top:0.2rem">
            เลือกทีมเหย้าและทีมเยือน แล้วกด Predict เพื่อดูผลการวิเคราะห์จาก AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Build team list ───────────────────────────────────────────
    all_teams = sorted(
        set(ctx['match_df_clean']['HomeTeam'].tolist() +
            ctx['match_df_clean']['AwayTeam'].tolist()) |
        set(NEW_TEAMS_BOOTSTRAPPED.keys())
    )

    # Session state defaults
    if 'pred_home' not in st.session_state:
        st.session_state['pred_home'] = "Arsenal"
    if 'pred_away' not in st.session_state:
        st.session_state['pred_away'] = "Chelsea"

    # ── VS Banner ────────────────────────────────────────────────
    h_team = st.session_state['pred_home'] if st.session_state['pred_home'] in all_teams else all_teams[0]
    a_team = st.session_state['pred_away'] if st.session_state['pred_away'] in all_teams else all_teams[1]

    h_logo = _get_logo(h_team)
    a_logo = _get_logo(a_team)

    st.markdown(f"""
    <div class="vs-banner">
        <div class="vs-team">
            <img src="{h_logo}" onerror="this.style.opacity='0.3'"/>
            <div class="vs-team-name vs-home-name">{h_team}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.6rem;
                        color:#00B0FF;background:rgba(0,176,255,0.1);
                        padding:2px 10px;border-radius:20px;font-weight:600">HOME</div>
        </div>
        <div class="vs-center">VS</div>
        <div class="vs-team">
            <img src="{a_logo}" onerror="this.style.opacity='0.3'"/>
            <div class="vs-team-name vs-away-name">{a_team}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.6rem;
                        color:#F97316;background:rgba(249,115,22,0.1);
                        padding:2px 10px;border-radius:20px;font-weight:600">AWAY</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Team Selector Grid ────────────────────────────────────────
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-label label-home">🏠 เลือกทีมเหย้า</div>', unsafe_allow_html=True)
        _render_team_grid(all_teams, "home", h_team)

    with col_right:
        st.markdown('<div class="section-label label-away">✈️ เลือกทีมเยือน</div>', unsafe_allow_html=True)
        _render_team_grid(all_teams, "away", a_team)

    st.write("")

    # ── Predict Button ────────────────────────────────────────────
    auto_run = st.session_state.pop('auto_predict', False)

    if st.button("🚀  GENERATE PREDICTION", type="primary", use_container_width=True) or auto_run:
        home = st.session_state['pred_home']
        away = st.session_state['pred_away']
        if home == away:
            st.warning("⚠️ กรุณาเลือกทีมคนละทีม")
            return

        with st.spinner("🤖 AI กำลังวิเคราะห์..."):
            r = silent(predict_match, home, away, ctx)
            s = silent(predict_score, home, away, ctx)

        if r:
            _render_results(home, away, r, s)
            st.divider()
            _render_recent_form(home, away)


def _render_team_grid(all_teams, role, selected):
    """Render clickable team cards grid."""
    cols_per_row = 5
    teams_chunked = [all_teams[i:i+cols_per_row] for i in range(0, len(all_teams), cols_per_row)]

    for row in teams_chunked:
        cols = st.columns(cols_per_row)
        for i, team in enumerate(row):
            with cols[i]:
                is_selected = (team == selected)
                css_class = f"selected-{role}" if is_selected else ""
                badge = f'<div class="{role}-badge">{"🏠 HOME" if role=="home" else "✈ AWAY"}</div>' if is_selected else ""
                logo = _get_logo(team)
                short = team[:11] + "…" if len(team) > 11 else team

                st.markdown(f"""
                <div class="team-card {css_class}" id="card-{role}-{team}">
                    {badge}
                    <img src="{logo}" onerror="this.src='https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg'"/>
                    <div class="team-name">{short}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("　", key=f"btn_{role}_{team}", use_container_width=True,
                             help=team):
                    st.session_state[f'pred_{role}'] = team
                    st.rerun()


def _render_results(home, away, r, s):
    """Render prediction results section."""
    h_logo = _get_logo(home)
    a_logo = _get_logo(away)

    st.markdown('<div class="section-label label-neutral" style="margin-top:1rem">📊 ผลการวิเคราะห์</div>',
                unsafe_allow_html=True)

    c_prob, c_score = st.columns([1.3, 1], gap="large")

    with c_prob:
        hw = r['Home Win']
        dr = r['Draw']
        aw = r['Away Win']
        pred = r['Prediction']

        # Color mapping
        pred_color = "#00B0FF" if "Home" in pred else ("#F59E0B" if "Draw" in pred else "#F97316")
        pred_bg    = "rgba(0,176,255,0.12)" if "Home" in pred else ("rgba(245,158,11,0.12)" if "Draw" in pred else "rgba(249,115,22,0.12)")

        st.markdown(f"""
        <div class="prob-bar-wrap">
            <div class="prob-row">
                <img src="{h_logo}" style="width:22px;height:22px;object-fit:contain;flex-shrink:0"/>
                <div class="prob-label">{home[:14]}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{hw}%;background:linear-gradient(90deg,#00B0FF,#38BDF8)"></div>
                </div>
                <div class="prob-pct" style="color:#00B0FF">{hw}%</div>
            </div>
            <div class="prob-row">
                <div style="width:22px;height:22px;display:flex;align-items:center;justify-content:center;
                            font-size:0.75rem;flex-shrink:0">🤝</div>
                <div class="prob-label">Draw</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{dr}%;background:linear-gradient(90deg,#F59E0B,#FCD34D)"></div>
                </div>
                <div class="prob-pct" style="color:#F59E0B">{dr}%</div>
            </div>
            <div class="prob-row">
                <img src="{a_logo}" style="width:22px;height:22px;object-fit:contain;flex-shrink:0"/>
                <div class="prob-label">{away[:14]}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{aw}%;background:linear-gradient(90deg,#F97316,#FB923C)"></div>
                </div>
                <div class="prob-pct" style="color:#F97316">{aw}%</div>
            </div>
            <div style="margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid rgba(255,255,255,0.06)">
                <span style="font-family:'DM Sans',sans-serif;font-size:0.72rem;color:#475569">💡 Predicted:</span>
                <span class="prediction-badge" 
                      style="background:{pred_bg};color:{pred_color};
                             border:1px solid {pred_color}33;margin-left:6px">
                    {pred}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_score:
        if s:
            st.markdown(f"""
            <div class="prob-bar-wrap" style="height:100%">
                <div style="display:flex;justify-content:space-between;margin-bottom:1rem">
                    <div style="text-align:center;flex:1">
                        <img src="{h_logo}" style="width:32px;height:32px;object-fit:contain"/>
                        <div style="font-family:'Bebas Neue',cursive;font-size:2rem;
                                    color:#7DD3FC;line-height:1.1">{s['home_xg']}</div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.6rem;
                                    color:#3D5068;text-transform:uppercase;letter-spacing:0.1em">xG Home</div>
                    </div>
                    <div style="font-family:'Bebas Neue',cursive;font-size:1.5rem;
                                color:rgba(255,255,255,0.12);align-self:center;padding:0 0.5rem">:</div>
                    <div style="text-align:center;flex:1">
                        <img src="{a_logo}" style="width:32px;height:32px;object-fit:contain"/>
                        <div style="font-family:'Bebas Neue',cursive;font-size:2rem;
                                    color:#FED7AA;line-height:1.1">{s['away_xg']}</div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.6rem;
                                    color:#3D5068;text-transform:uppercase;letter-spacing:0.1em">xG Away</div>
                    </div>
                </div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.62rem;font-weight:700;
                            letter-spacing:0.15em;text-transform:uppercase;color:#3D5068;margin-bottom:0.5rem">
                    Top Scores
                </div>
            """, unsafe_allow_html=True)

            for sc_row in s['top5_scores']:
                score_str, prob = sc_row[0], sc_row[1]
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
                    <span style="font-family:'Bebas Neue',cursive;font-size:1rem;
                                 color:#E2E8F0;letter-spacing:0.08em">{score_str}</span>
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.75rem;
                                 color:#475569">{prob}%</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


def _render_recent_form(home, away):
    """Render recent form (last 5 matches) for both teams."""
    @st.cache_data(ttl=300)
    def _load_fresh_match_data():
        import glob as _g
        dfs = []
        for f in _g.glob(os.path.join(DATA_DIR, "*.csv")):
            if 'backup' in f.lower():
                continue
            try:
                _df = pd.read_csv(f)
                _df['FTHG'] = pd.to_numeric(_df['FTHG'], errors='coerce')
                _df['FTAG'] = pd.to_numeric(_df['FTAG'], errors='coerce')
                _df['Date'] = pd.to_datetime(_df['Date'], dayfirst=True, errors='coerce')
                dfs.append(_df)
            except Exception:
                pass
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
        return combined.dropna(subset=['FTHG', 'FTAG']).sort_values('Date').reset_index(drop=True)

    def _get_team_form(team, valid_data):
        hm = valid_data[valid_data['HomeTeam'] == team].copy()
        hm['Venue'] = 'H'; hm['GF'] = hm['FTHG']; hm['GA'] = hm['FTAG']; hm['Opponent'] = hm['AwayTeam']
        am = valid_data[valid_data['AwayTeam'] == team].copy()
        am['Venue'] = 'A'; am['GF'] = am['FTAG']; am['GA'] = am['FTHG']; am['Opponent'] = am['HomeTeam']
        all_m = pd.concat([hm, am]).sort_values('Date', ascending=False).head(5)
        def rl(r):
            if r['GF'] > r['GA']:    return 'W'
            elif r['GF'] == r['GA']: return 'D'
            else:                    return 'L'
        all_m['Result'] = all_m.apply(rl, axis=1)
        return all_m

    fresh_data = _load_fresh_match_data()
    latest_date = fresh_data['Date'].max() if len(fresh_data) > 0 else None

    st.markdown('<div class="section-label label-neutral">📋 Recent Form (Last 5 Matches)</div>',
                unsafe_allow_html=True)
    if latest_date is not None and pd.notna(latest_date):
        st.caption(f"ข้อมูลล่าสุดใน CSV: **{latest_date.strftime('%d %b %Y')}**")

    ch, ca = st.columns(2, gap="large")
    for team, col in [(home, ch), (away, ca)]:
        with col:
            role_color = "#7DD3FC" if team == home else "#FED7AA"
            logo = _get_logo(team)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.5rem">
                <img src="{logo}" style="width:24px;height:24px;object-fit:contain"/>
                <span style="font-family:'DM Sans',sans-serif;font-weight:700;
                             font-size:0.88rem;color:{role_color}">{team}</span>
            </div>
            """, unsafe_allow_html=True)

            if len(fresh_data) > 0:
                try:
                    d = _get_team_form(team, fresh_data)
                    # Form pills
                    pills = ""
                    for _, row in d.iterrows():
                        res = row['Result']
                        pills += f'<span class="form-pill pill-{res}">{res}</span>'
                    st.markdown(f'<div style="margin-bottom:0.5rem">{pills}</div>', unsafe_allow_html=True)

                    d = d[['Date', 'Opponent', 'Venue', 'GF', 'GA', 'Result']].copy()
                    d['Date'] = d['Date'].dt.strftime('%d/%m/%y')
                    st.dataframe(d, hide_index=True, use_container_width=True)
                except Exception:
                    st.warning("No recent data available.")
            else:
                st.warning("No recent data available.")