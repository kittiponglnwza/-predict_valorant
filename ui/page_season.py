"""
ui/page_season.py — Season Table page for Football AI Nexus Engine
"""
import streamlit as st
import pandas as pd

from src.config import TODAY
from src.predict import run_season_simulation, update_season_csv_from_api, get_pl_standings_from_api
from utils import silent, make_styled_table, zone_label, find_team_col


def page_season(ctx):
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');
    .season-hero {
        background: linear-gradient(135deg,#0f1923 0%,#0a1628 50%,#0d1f3c 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 2rem 2.5rem 1.8rem;
        margin-bottom: 1.2rem;
        position: relative; overflow: hidden;
    }
    .season-hero::before {
        content:''; position:absolute; top:-80px; right:-80px;
        width:300px; height:300px;
        background:radial-gradient(circle,rgba(0,176,255,0.10) 0%,transparent 70%);
        border-radius:50%; pointer-events:none;
    }
    .hero-eyebrow { font-family:'DM Sans',sans-serif; font-size:0.65rem; font-weight:600;
        letter-spacing:0.22em; text-transform:uppercase; color:#00B0FF; margin-bottom:0.3rem; }
    .hero-heading { font-family:'Bebas Neue',cursive; font-size:3.2rem; letter-spacing:0.04em;
        line-height:1; color:#fff; margin:0 0 0.3rem; }
    .hero-heading span { background:linear-gradient(90deg,#00B0FF,#00E676);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .hero-sub { font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#4B6080; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="season-hero">
        <div class="hero-eyebrow">📊 Premier League</div>
        <div class="hero-heading">SEASON <span>TABLE</span></div>
        <div class="hero-sub">ดูตารางคะแนนปัจจุบัน หรือให้ AI จำลองผลถึงสิ้นฤดูกาล</div>
    </div>
    """, unsafe_allow_html=True)

    tab_current, tab_sim = st.tabs(["📋  ตารางคะแนนปัจจุบัน", "🔮  AI Simulation"])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — Current Standings
    # ════════════════════════════════════════════════════════════════
    with tab_current:
        st.write("")

        current_year = TODAY.year
        current_season_year = current_year if TODAY.month >= 8 else current_year - 1
        season_options = list(range(current_season_year, current_season_year - 6, -1))
        season_labels  = {y: f"{y}/{str(y + 1)[2:]}" for y in season_options}

        sc1, sc2 = st.columns([1, 3])
        selected_year = sc1.selectbox(
            "🗓️ ฤดูกาล",
            options=season_options,
            format_func=lambda y: season_labels[y],
            index=0,
            key="season_year_select"
        )

        bsync, _ = sc2.columns([1, 2])
        if bsync.button("⟳  Refresh", use_container_width=True, key="sync_cur"):
            st.cache_data.clear()
            st.rerun()

        @st.cache_data(ttl=300, show_spinner=False)
        def _fetch_standings(year):
            return silent(get_pl_standings_from_api, year)

        with st.spinner("📡 กำลังดึงข้อมูลตารางคะแนนจาก API..."):
            rows = _fetch_standings(selected_year)

        if rows is None or len(rows) == 0:
            st.error("❌ ดึงข้อมูลจาก API ไม่ได้ — ตรวจสอบ API Key หรือการเชื่อมต่ออินเทอร์เน็ต")
        else:
            tbl = pd.DataFrame(rows)
            tbl = tbl.drop(columns=['pos'], errors='ignore')

            def _form_emoji(form_str):
                if not form_str: return ""
                mapping = {'W': '🟢', 'D': '🟡', 'L': '🔴'}
                return " ".join(mapping.get(c, '⚪') for c in form_str.split(',') if c)

            if 'Form' in tbl.columns:
                tbl['Form'] = tbl['Form'].apply(_form_emoji)

            tbl.index = range(1, len(tbl) + 1)
            tbl.index.name = '#'
            tbl['Zone'] = [zone_label(i) for i in tbl.index]

            max_pts_cur = max(tbl['PTS'].max(), 1)
            styled_cur = make_styled_table(tbl, 'PTS', max_pts_cur)
            styled_cur = (
                styled_cur
                .applymap(lambda v: 'font-weight:700;color:#fff', subset=['PTS'])
                .bar(subset=['GD'],
                     color=['rgba(239,68,68,0.3)', 'rgba(0,230,118,0.3)'],
                     vmin=-30, vmax=30)
            )
            st.dataframe(styled_cur, use_container_width=True, height=720)
            st.caption(
                f"📡 ข้อมูลจาก football-data.org · ฤดูกาล {season_labels[selected_year]} · "
                f"👑 Champion · ⚽ Top 4 UCL · 🌍 Top 6 Europa · 🏅 Top 7 Conference · 🔻 Bottom 3 Relegation"
            )

    # ════════════════════════════════════════════════════════════════
    # TAB 2 — AI Simulation
    # ════════════════════════════════════════════════════════════════
    with tab_sim:
        st.write("")

        b1, b2, _ = st.columns([1, 1, 2], gap="medium")
        sim_clicked = b1.button("🔮  Run Simulation", type="primary",
                                use_container_width=True, key="run_sim")
        sync2 = b2.button("⟳  Sync Season Data", use_container_width=True, key="sync_sim")

        if sync2:
            with st.spinner("Syncing..."):
                silent(update_season_csv_from_api)
            st.success("✅ Synced!")

        if sim_clicked:
            with st.spinner("🔮 Simulating remaining fixtures..."):
                ctx_new = silent(run_season_simulation, ctx)
                st.session_state['ctx'] = ctx_new
                ctx = ctx_new

        ft = ctx.get('final_table')

        if ft is not None:
            df = ft.sort_values('FinalPoints', ascending=False).reset_index(drop=False)
            df.index = range(1, len(df) + 1)
            team_col = find_team_col(df)
            max_pts = max(df['FinalPoints'].max(), 1)

            display_df = pd.DataFrame({
                'Club':  [str(df.loc[p, team_col]) for p in df.index],
                'PTS':   [int(df.loc[p, 'RealPoints']) for p in df.index],
                '+PROJ': [int(df.loc[p, 'FinalPoints'] - df.loc[p, 'RealPoints']) for p in df.index],
                'FINAL': [int(df.loc[p, 'FinalPoints']) for p in df.index],
                'Zone':  [zone_label(p) for p in df.index],
            }, index=df.index)
            display_df.index.name = '#'

            def _color_proj(val):
                if val > 0: return 'color:#00E676;font-weight:600'
                if val < 0: return 'color:#EF4444;font-weight:600'
                return 'color:#475569'

            styled_sim = (
                make_styled_table(display_df, 'FINAL', max_pts)
                .applymap(_color_proj, subset=['+PROJ'])
                .applymap(lambda v: 'font-weight:700;color:#fff', subset=['FINAL'])
                .format({'+PROJ': lambda x: f'+{x}' if x > 0 else str(x)})
            )

            st.dataframe(styled_sim, use_container_width=True, height=720)
            st.caption("👑 Champion · ⚽ Top 4 UCL · 🌍 Top 6 Europa · 🏅 Top 7 Conference · 🔻 Bottom 3 Relegation")

            st.write("")
            st.markdown('<p style="font-size:0.68rem;font-weight:600;letter-spacing:0.18em;'
                        'text-transform:uppercase;color:#3D5068;margin-bottom:0.5rem">'
                        'Simulation Highlights</p>', unsafe_allow_html=True)

            champion  = df.iloc[0]
            ucl_teams = df[df.index <= 4]
            relegated = df[df.index >= 18]
            s1, s2, s3 = st.columns(3, gap="medium")
            s1.metric("🏆 Predicted Champion",
                      str(champion.get(team_col, '—')),
                      f"{int(champion['FinalPoints'])} pts")
            if not ucl_teams.empty:
                names = ", ".join(str(r.get(team_col, '?')) for _, r in ucl_teams.iterrows())
                s2.metric("⚽ UCL (Top 4)", f"{len(ucl_teams)} clubs",
                          names[:32] + "…" if len(names) > 32 else names)
            if not relegated.empty:
                names = ", ".join(str(r.get(team_col, '?')) for _, r in relegated.iterrows())
                s3.metric("🔻 Relegated", f"{len(relegated)} clubs",
                          names[:32] + "…" if len(names) > 32 else names)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 2rem;
                        background:linear-gradient(135deg,#0f1923,#0b1320);
                        border:1px dashed rgba(0,176,255,0.18);border-radius:18px;margin-top:0.5rem">
                <div style="font-size:2.8rem;margin-bottom:0.7rem;opacity:0.45">🔮</div>
                <div style="font-family:'Bebas Neue',cursive;font-size:1.8rem;color:#E2E8F0;letter-spacing:0.06em">
                    SIMULATION READY
                </div>
                <div style="font-family:'DM Sans',sans-serif;color:#3D5068;font-size:0.82rem;margin-top:0.35rem">
                    กด <strong style="color:#00B0FF">Run Simulation</strong> เพื่อให้ AI คาดการณ์ผลจนจบฤดูกาล
                </div>
            </div>
            """, unsafe_allow_html=True)