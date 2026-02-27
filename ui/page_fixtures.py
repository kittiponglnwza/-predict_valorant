"""
ui/page_fixtures.py — Upcoming Fixtures page for Football AI Nexus Engine
"""
import streamlit as st

from src.predict import predict_match, predict_score, show_next_pl_fixtures
from utils import silent


def page_fixtures(ctx):

    # ── STYLES ────────────────────────────────────────────────────────────────
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');

        .main .block-container { padding-top: 2rem; max-width: 960px; }

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
        .fx-home  { font-size: 1.15rem; font-weight: 700; color: #F0F6FF; letter-spacing: 0.5px; }
        .fx-away  { font-size: 1.0rem; font-weight: 600; color: rgba(148,187,233,0.65); margin-top: 3px; letter-spacing: 0.5px; }
        .fx-vs    { font-size: 0.7rem; color: rgba(56,189,248,0.5); font-weight: 700; letter-spacing: 2px; margin: 1px 0; }

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
        .fx-prob-row  { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
        .fx-prob-label {
            font-family: 'Rajdhani', sans-serif; font-size: 0.75rem; font-weight: 700;
            letter-spacing: 1px; width: 14px; color: rgba(148,187,233,0.5);
        }
        .fx-prob-bar-bg   { flex: 1; height: 6px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; }
        .fx-prob-bar-fill { height: 100%; border-radius: 2px; }
        .fx-prob-val {
            font-family: 'Rajdhani', sans-serif; font-size: 0.88rem;
            font-weight: 700; width: 34px; text-align: right;
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
            padding: 14px 18px !important;
            margin-bottom: 6px !important;
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
            <div class="fx-eyebrow">⚡ Nexus Engine · Premier League</div>
            <div class="fx-title">Next <em>Fixtures</em></div>
            <div class="fx-subtitle">Upcoming matches with AI win probability &amp; score prediction</div>
        </div>
    """, unsafe_allow_html=True)

    # ── CONTROLS ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1.2, 1.2, 3])
    with c1:
        n = st.number_input("Matches to fetch", min_value=1, max_value=20, value=5)
    with c2:
        st.write("")
        st.write("")
        refetch = st.button("Refresh", type="primary", use_container_width=True)

    # ── AUTO-FETCH on first load, n change, or manual refresh ─────────────────
    if refetch or "fx_upcoming" not in st.session_state or st.session_state.get("fx_n") != n:
        with st.spinner("Fetching fixtures..."):
            upcoming = silent(show_next_pl_fixtures, ctx, num_matches=n)
        st.session_state["fx_upcoming"] = upcoming
        st.session_state["fx_n"] = n
    else:
        upcoming = st.session_state["fx_upcoming"]

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if upcoming:
        st.markdown("")
        st.markdown('<div class="fx-section-label">Upcoming Matches</div>', unsafe_allow_html=True)

        # ── Column headers ──
        hc1, hc2, hc3, hc4, hc5 = st.columns([1.1, 2.6, 2.4, 1.1, 1.0])
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

                with st.container():
                    col_date, col_match, col_prob, col_score, col_btn = st.columns([1.1, 2.6, 2.4, 1.1, 1.0])

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
                                <div class="fx-home">
                                    <span class="fx-team-tag fx-tag-home">HOME</span>{f['HomeTeam']}
                                </div>
                                <div class="fx-vs">VS</div>
                                <div class="fx-away">
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
                            <div style="padding-top:4px;">
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
                    "<div style='height:1px; background:rgba(255,255,255,0.07); margin:2px 0 6px;'></div>",
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