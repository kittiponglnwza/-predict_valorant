"""
ui/page_fixtures.py — Upcoming Fixtures page for Football AI Nexus Engine
"""
import streamlit as st

from src.predict import predict_match, predict_score, show_next_pl_fixtures
from utils import silent


def page_fixtures(ctx):
    st.title("Upcoming Fixtures")

    c1, c2 = st.columns([1, 3])
    n = c1.number_input("Number of matches to fetch", min_value=1, max_value=20, value=5)

    st.write("")
    if st.button("📡 Fetch Fixtures", type="primary"):
        with st.spinner("Fetching data from API..."):
            upcoming = silent(show_next_pl_fixtures, ctx, num_matches=n)

        if upcoming:
            st.write("")
            st.markdown("### 🏟️ Select Match to Analyze")

            h1, h2, h3, h4, h5 = st.columns([1.5, 3.5, 3, 1.5, 1.5])
            h1.markdown("⏱️ **Date / Time**")
            h2.markdown("⚔️ **Match**")
            h3.markdown("📊 **Win Prob (H - D - A)**")
            h4.markdown("⚽ **Exp. Score**")
            h5.markdown("⚡ **Action**")
            st.divider()

            for i, f in enumerate(upcoming):
                r = silent(predict_match, f['HomeTeam'], f['AwayTeam'], ctx)
                s = silent(predict_score, f['HomeTeam'], f['AwayTeam'], ctx)

                if r and s:
                    c1, c2, c3, c4, c5 = st.columns([1.5, 3.5, 3, 1.5, 1.5])

                    c1.caption(f"{f['Date']}  \n{f.get('Time', '')}")
                    c2.markdown(f"🏠 **{f['HomeTeam']}** \n✈️ **{f['AwayTeam']}**")
                    c3.caption(f"H: **{r['Home Win']}%** | D: **{r['Draw']}%** | A: **{r['Away Win']}%**")
                    c4.markdown(f"**{s['most_likely_score']}**")

                    c5.button(
                        "🎯 Predict",
                        key=f"btn_pred_{i}",
                        use_container_width=True,
                        on_click=_navigate_to_predict,
                        args=(f['HomeTeam'], f['AwayTeam'])
                    )

                    st.markdown("<hr style='margin: 0.5em 0; opacity: 0.15;'>", unsafe_allow_html=True)
        else:
            st.error("Unable to fetch upcoming fixtures.")


def _navigate_to_predict(home_team, away_team):
    """Callback to navigate to Predict page with pre-filled teams."""
    st.session_state['nav_page'] = "Predict Match"
    st.session_state['pred_home'] = home_team
    st.session_state['pred_away'] = away_team
    st.session_state['auto_predict'] = True