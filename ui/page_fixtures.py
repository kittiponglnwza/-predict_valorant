"""
ui/page_fixtures.py — Upcoming Fixtures page for Football AI Nexus Engine
"""
import streamlit as st
from datetime import datetime, timedelta, timezone

from src.predict import predict_match, predict_score, show_next_pl_fixtures
from utils import silent


def _filter_future_fixtures(fixtures):
    """Remove fixtures whose kickoff time has already passed (with 5-min grace)."""
    if not fixtures:
        return fixtures
    now_th = datetime.now(timezone.utc) + timedelta(hours=7)  # Thailand time
    future = []
    for f in fixtures:
        try:
            date_str = f.get('Date', '')   # e.g. "01 Mar"
            time_str = f.get('Time', '00:00')  # e.g. "21:00"
            if not date_str or date_str == '—':
                future.append(f)
                continue
            # Parse kickoff in TH time
            year = now_th.year
            dt_str = f"{date_str} {year} {time_str}"
            kickoff = datetime.strptime(dt_str, "%d %b %Y %H:%M")
            # If month seems in the past (e.g. Dec when now is Jan), try next year
            if kickoff.month < now_th.month - 3:
                kickoff = kickoff.replace(year=year + 1)
            # Keep if kickoff is in the future (allow 5 min grace period)
            if kickoff >= now_th.replace(tzinfo=None) - timedelta(minutes=5):
                future.append(f)
        except Exception:
            future.append(f)  # Keep if can't parse
    return future

DEFAULT_LOGO = "https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg"
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


def page_fixtures(ctx):

    # ── STYLES ────────────────────────────────────────────────────────────────
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');

        .main .block-container { padding-top: 2rem; max-width: 1280px; margin: 0 auto; }

        /* Page header */
        .fx-page-header  { margin-bottom: 28px; }
        .fx-eyebrow {
            font-family: 'Rajdhani', sans-serif; font-size: 0.6rem; font-weight: 700;
            letter-spacing: 4px; text-transform: uppercase; color: #38BDF8; margin-bottom: 8px; font-size: 0.8rem;
        }
        .fx-title {
            font-family: 'Orbitron', sans-serif; font-size: 2.8rem; font-weight: 900;
            color: #F0F6FF; letter-spacing: 1px; line-height: 1.1;
        }
        .fx-title em { font-style: normal; color: #38BDF8; }
        .fx-subtitle {
            font-family: 'Rajdhani', sans-serif; font-size: 0.85rem;
            color: rgba(148,187,233,0.55); letter-spacing: 1px; margin-top: 6px; font-size: 1.05rem;
        }

        /* Section label */
        .fx-section-label {
            font-family: 'Rajdhani', sans-serif; font-size: 0.6rem; font-weight: 700;
            letter-spacing: 3.5px; text-transform: uppercase;
            color: rgba(148,187,233,0.45); margin-bottom: 14px; font-size: 0.75rem;
        }

        /* Date badge */
        .fx-date-badge {
            display: inline-block;
            background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.18);
            border-radius: 4px; padding: 2px 8px;
            font-family: 'Rajdhani', sans-serif; font-size: 0.85rem;
            font-weight: 700; letter-spacing: 1px; color: #38BDF8;
        }
        .fx-time {
            font-family: 'Rajdhani', sans-serif; font-size: 0.9rem;
            color: rgba(148,187,233,0.5); margin-top: 3px; letter-spacing: 1px;
        }

        /* Teams */
        .fx-teams { font-family: 'Rajdhani', sans-serif; }
        .fx-team-line { display: flex; align-items: center; gap: 8px; }
        .fx-team-logo {
            width: 24px; height: 24px; object-fit: contain; flex-shrink: 0;
            filter: drop-shadow(0 2px 6px rgba(0,0,0,0.45));
        }
        .fx-home  { font-size: 1.15rem; font-weight: 700; color: #F0F6FF; letter-spacing: 0.5px; }
        .fx-away  { font-size: 1.0rem; font-weight: 600; color: rgba(148,187,233,0.65); margin-top: 5px; letter-spacing: 0.5px; }
        .fx-vs    { font-size: 0.7rem; color: rgba(56,189,248,0.5); font-weight: 700; letter-spacing: 2px; margin: 3px 0; }

        /* Team tags */
        .fx-team-tag {
            font-family: 'Rajdhani', sans-serif; font-size: 0.65rem; font-weight: 700;
            letter-spacing: 1.5px; padding: 1px 5px; border-radius: 3px;
            margin-right: 5px; vertical-align: middle;
        }
        .fx-tag-home { background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3); color: #38BDF8; }
        .fx-tag-away { background: rgba(167,139,250,0.12); border: 1px solid rgba(167,139,250,0.3); color: #A78BFA; }

        /* Probability bars */
        .fx-prob-wrap { margin-top: 2px; }
        .fx-prob-row  { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
        .fx-prob-label {
            font-family: 'Rajdhani', sans-serif; font-size: 0.75rem; font-weight: 700;
            letter-spacing: 1px; width: 16px; color: rgba(148,187,233,0.5);
        }
        .fx-prob-bar-bg   { flex: 1; height: 6px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; min-width: 60px; }
        .fx-prob-bar-fill { height: 100%; border-radius: 2px; }
        .fx-prob-val {
            font-family: 'Rajdhani', sans-serif; font-size: 0.88rem;
            font-weight: 700; width: 40px; text-align: right;
        }

        /* Score */
        .fx-score       { font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 700; color: #F0F6FF; letter-spacing: 2px; line-height: 1; }
        .fx-score-label { font-family: 'Rajdhani', sans-serif; font-size: 0.58rem; font-weight: 600; letter-spacing: 2px; color: rgba(148,187,233,0.35); text-transform: uppercase; margin-top: 3px; }


        /* ── Table header ── */
        .fx-th {
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.58rem; font-weight: 700;
            letter-spacing: 2.5px; text-transform: uppercase;
            color: rgba(148,187,233,0.45);
            padding: 4px 0;
            font-size: 0.72rem;
        }

        /* ── Row: style Streamlit's stHorizontalBlock inside main ── */
        .main [data-testid="stHorizontalBlock"] {
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 8px !important;
            padding: 18px 24px !important;
            max-width: 1100px !important;
            margin: 0 auto 10px !important;
            transition: border-color 0.15s ease, background 0.15s ease !important;
        }
        .main [data-testid="stHorizontalBlock"]:hover {
            background: rgba(56,189,248,0.09) !important;
            border-color: rgba(56,189,248,0.35) !important;
        }
        /* Error */
        .fx-error {
            background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25);
            border-radius: 8px; padding: 16px 20px;
            font-family: 'Rajdhani', sans-serif; font-size: 0.9rem;
            color: rgba(252,165,165,0.8); letter-spacing: 0.5px;
        }

        /* Number input label */
        .stNumberInput label {
            font-family: 'Rajdhani', sans-serif !important; font-size: 0.85rem !important;
            letter-spacing: 2px !important; text-transform: uppercase !important;
            color: rgba(148,187,233,0.5) !important; font-weight: 700 !important;
        }

        /* Button */
        .stButton > button {
            font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important;
            letter-spacing: 1.5px !important; font-size: 0.92rem !important;
            border-radius: 6px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── PAGE HEADER ───────────────────────────────────────────────────────────
    st.markdown("""
        <div class="fx-page-header">
            <div class="fx-eyebrow">Nexus Engine · Premier League</div>
            <div class="fx-title">Next <em>Fixtures</em></div>
            <div class="fx-subtitle">Upcoming matches with AI win probability &amp; score prediction</div>
        </div>
    """, unsafe_allow_html=True)

    # ── CONTROLS ──────────────────────────────────────────────────────────────
    lane_l, lane_c, lane_r = st.columns([0.06, 0.88, 0.06])
    with lane_c:
        c_left, c_mid, c_right = st.columns([1.4, 1.8, 1.4])
        with c_mid:
            n = st.number_input("Matches to fetch", min_value=1, max_value=20, value=5)

    # ── FETCH + FILTER ────────────────────────────────────────────────────────
    # Always re-fetch if n changed or cache is stale (> 2 min)
    cache_age = (datetime.now(timezone.utc) - st.session_state.get("fx_fetched_at", datetime.min.replace(tzinfo=timezone.utc))).total_seconds()
    needs_id_refresh = any(
        ("HomeID" not in m or "AwayID" not in m or "HomeLogo" not in m or "AwayLogo" not in m) or
        (not (m.get("HomeLogo") or m.get("HomeID"))) or
        (not (m.get("AwayLogo") or m.get("AwayID")))
        for m in st.session_state.get("fx_upcoming", [])
        if isinstance(m, dict)
    )
    if "fx_upcoming" not in st.session_state or st.session_state.get("fx_n") != n or needs_id_refresh or cache_age > 120:
        with st.spinner("Fetching fixtures..."):
            raw = silent(show_next_pl_fixtures, ctx, num_matches=n + 10)  # fetch extra to compensate filtering
        st.session_state["fx_upcoming"] = raw
        st.session_state["fx_n"] = n
        st.session_state["fx_fetched_at"] = datetime.now(timezone.utc)
    else:
        raw = st.session_state["fx_upcoming"]

    # Filter out past matches in real-time (done every render, not cached)
    upcoming = _filter_future_fixtures(raw)[:n] if raw else raw

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if upcoming:
        st.markdown("")
        st.markdown('<div class="fx-section-label">Upcoming Matches</div>', unsafe_allow_html=True)

        # ── Column headers ──
        hc1, hc2, hc3, hc4, hc5 = st.columns([1.2, 2.8, 2.6, 1.0, 0.9])
        hc1.markdown('<div class="fx-th">Date / Time</div>', unsafe_allow_html=True)
        hc2.markdown('<div class="fx-th">Match</div>', unsafe_allow_html=True)
        hc3.markdown('<div class="fx-th">Win Probability</div>', unsafe_allow_html=True)
        hc4.markdown('<div class="fx-th">Exp. Score</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='height:1px; background:rgba(255,255,255,0.1); margin:4px 0 8px;'></div>",
            unsafe_allow_html=True
        )

        for i, f in enumerate(upcoming):
            r = silent(predict_match, f['HomeTeam'], f['AwayTeam'], ctx)
            s = silent(predict_score, f['HomeTeam'], f['AwayTeam'], ctx)

            if r and s:
                h_prob = r['Home Win']
                d_prob = r['Draw']
                a_prob = r['Away Win']
                score  = s['most_likely_score']

                max_prob = max(h_prob, d_prob, a_prob)
                h_color  = "#38BDF8" if h_prob == max_prob else "#475569"
                d_color  = "#F59E0B" if d_prob == max_prob else "#475569"
                a_color  = "#A78BFA" if a_prob == max_prob else "#475569"
                home_logo = _crest_url(f.get("HomeTeam"), f.get("HomeLogo"), f.get("HomeID"))
                away_logo = _crest_url(f.get("AwayTeam"), f.get("AwayLogo"), f.get("AwayID"))
                home_fallback = _fallback_team_logo(f.get("HomeTeam"))
                away_fallback = _fallback_team_logo(f.get("AwayTeam"))

                with st.container():
                    col_date, col_match, col_prob, col_score, col_btn = st.columns([1.2, 2.8, 2.6, 1.0, 0.9])

                    with col_date:
                        st.markdown(f"""
                            <div style="padding-top:6px;">
                                <div class="fx-date-badge">{f['Date']}</div>
                                <div class="fx-time">{f.get('Time', '—')}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col_match:
                        st.markdown(f"""
                            <div class="fx-teams">
                                <div class="fx-home fx-team-line">
                                    <img class="fx-team-logo" src="{home_logo}" onerror="this.src='{home_fallback}'"/>
                                    <span class="fx-team-tag fx-tag-home">HOME</span>{f['HomeTeam']}
                                </div>
                                <div class="fx-vs">VS</div>
                                <div class="fx-away fx-team-line">
                                    <img class="fx-team-logo" src="{away_logo}" onerror="this.src='{away_fallback}'"/>
                                    <span class="fx-team-tag fx-tag-away">AWAY</span>{f['AwayTeam']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col_prob:
                        st.markdown(f"""
                            <div class="fx-prob-wrap">
                                <div class="fx-prob-row">
                                    <span class="fx-prob-label">H</span>
                                    <div class="fx-prob-bar-bg">
                                        <div class="fx-prob-bar-fill" style="width:{h_prob}%; background:{h_color};"></div>
                                    </div>
                                    <span class="fx-prob-val" style="color:{h_color};">{h_prob}%</span>
                                </div>
                                <div class="fx-prob-row">
                                    <span class="fx-prob-label">D</span>
                                    <div class="fx-prob-bar-bg">
                                        <div class="fx-prob-bar-fill" style="width:{d_prob}%; background:{d_color};"></div>
                                    </div>
                                    <span class="fx-prob-val" style="color:{d_color};">{d_prob}%</span>
                                </div>
                                <div class="fx-prob-row">
                                    <span class="fx-prob-label">A</span>
                                    <div class="fx-prob-bar-bg">
                                        <div class="fx-prob-bar-fill" style="width:{a_prob}%; background:{a_color};"></div>
                                    </div>
                                    <span class="fx-prob-val" style="color:{a_color};">{a_prob}%</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col_score:
                        st.markdown(f"""
                            <div style="padding-top:6px; padding-left:8px;">
                                <div class="fx-score">{score}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col_btn:
                        st.write("")
                        st.button(
                            "Predict",
                            key=f"btn_pred_{i}",
                            use_container_width=True,
                            on_click=_navigate_to_predict,
                            args=(f['HomeTeam'], f['AwayTeam'])
                        )

                st.markdown(
                    "<div style='height:1px; background:rgba(255,255,255,0.07); margin:4px 0 10px;'></div>",
                    unsafe_allow_html=True
                )

    elif upcoming is not None:
        st.markdown("""
            <div class="fx-error">
                ⚠️ &nbsp; Unable to fetch upcoming fixtures. Check your API connection and try again.
            </div>
        """, unsafe_allow_html=True)


def _navigate_to_predict(home_team, away_team):
    """Callback to navigate to Predict page with pre-filled teams."""
    st.session_state['nav_page']     = "Predict Match"
    st.session_state['pred_home']    = home_team
    st.session_state['pred_away']    = away_team
    st.session_state['auto_predict'] = True