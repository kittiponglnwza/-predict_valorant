"""
ui/page_overview.py — Overview / Dashboard page for Football AI Nexus Engine
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.config import TODAY, API_KEY
from utils import silent


def page_overview(ctx):
    from sklearn.metrics import confusion_matrix, accuracy_score

    # ── CSS ───────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');
    .ov-hero {
        background: linear-gradient(135deg,#0f1923 0%,#0a1628 50%,#0d1f3c 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px; padding: 1.8rem 2.5rem 1.6rem;
        margin-bottom: 1.2rem; position:relative; overflow:hidden;
    }
    .ov-hero::before { content:''; position:absolute; top:-80px; right:-80px;
        width:300px; height:300px;
        background:radial-gradient(circle,rgba(0,176,255,0.10) 0%,transparent 70%);
        border-radius:50%; pointer-events:none; }
    .ov-eyebrow { font-family:'DM Sans',sans-serif; font-size:0.65rem; font-weight:600;
        letter-spacing:0.22em; text-transform:uppercase; color:#00B0FF; margin-bottom:0.3rem; }
    .ov-heading { font-family:'Bebas Neue',cursive; font-size:3rem; letter-spacing:0.04em;
        line-height:1; color:#fff; margin:0 0 0.3rem; }
    .ov-heading span { background:linear-gradient(90deg,#00B0FF,#00E676);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .ov-sub { font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#4B6080; }

    .match-card {
        background: linear-gradient(135deg,#0f1923,#0b1320);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .match-card.live { border-left: 3px solid #00E676; }
    .match-card.upcoming { border-left: 3px solid #00B0FF; }
    .match-live-badge {
        display:inline-block; background:rgba(0,230,118,0.15); color:#00E676;
        border:1px solid rgba(0,230,118,0.3); border-radius:20px;
        font-family:'DM Sans',sans-serif; font-size:0.62rem; font-weight:700;
        letter-spacing:0.1em; padding:2px 8px; text-transform:uppercase;
    }
    .match-upcoming-badge {
        display:inline-block; background:rgba(0,176,255,0.12); color:#00B0FF;
        border:1px solid rgba(0,176,255,0.25); border-radius:20px;
        font-family:'DM Sans',sans-serif; font-size:0.62rem; font-weight:600;
        padding:2px 8px;
    }
    .match-teams {
        font-family:'Bebas Neue',cursive; font-size:1.25rem; color:#E2E8F0;
        letter-spacing:0.05em; margin:0.2rem 0;
    }
    .match-score {
        font-family:'Bebas Neue',cursive; font-size:1.5rem; color:#00E676;
    }
    .match-time {
        font-family:'DM Sans',sans-serif; font-size:0.75rem; color:#475569;
    }
    .stat-ring-wrap {
        background: linear-gradient(135deg,#0f1923,#0b1320);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 1.2rem 1.5rem; text-align:center;
    }
    .stat-big {
        font-family:'Bebas Neue',cursive; font-size:2.8rem; line-height:1;
    }
    .stat-label {
        font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:600;
        letter-spacing:0.1em; text-transform:uppercase; color:#475569;
        margin-top:0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ov-hero">
        <div class="ov-eyebrow">⚡ Nexus Engine · Football AI v9.0</div>
        <div class="ov-heading">DASHBOARD <span>OVERVIEW</span></div>
        <div class="ov-sub">Today: {TODAY.strftime('%A, %d %b %Y')} · Latest data: {ctx['data']['Date'].max().strftime('%d %b %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top Metrics ────────────────────────────────────────────────
    acc = round(accuracy_score(ctx['y_test'], ctx['y_pred_final']) * 100, 1)
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("🎯 Model Accuracy", f"{acc}%", "Validated")
    m2.metric("📚 Training Matches", f"{len(ctx['train']):,}")
    m3.metric("🧪 Test Matches",     f"{len(ctx['test']):,}")
    m4.metric("⚙️ Features",         f"{len(ctx['FEATURES'])}")

    st.write("")

    # ── Main 2-column layout ───────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1], gap="large")

    # ════════════════════════════════════════
    # LEFT — Live & Upcoming Matches
    # ════════════════════════════════════════
    with col_left:
        st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:0.68rem;font-weight:600;'
                    'letter-spacing:0.18em;text-transform:uppercase;color:#3D5068;margin-bottom:0.6rem">'
                    '📡 Live &amp; Upcoming Matches</p>', unsafe_allow_html=True)

        @st.cache_data(ttl=60, show_spinner=False)
        def _fetch_live_and_upcoming():
            import requests as _req
            headers = {"X-Auth-Token": API_KEY}
            live, upcoming = [], []
            try:
                r = _req.get("https://api.football-data.org/v4/competitions/PL/matches",
                             headers=headers, params={"status": "IN_PLAY,PAUSED"}, timeout=8)
                if r.ok:
                    for m in r.json().get("matches", []):
                        utc = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
                        th  = utc + timedelta(hours=7)
                        home_name = m["homeTeam"].get("shortName") or m["homeTeam"].get("name", "?")
                        away_name = m["awayTeam"].get("shortName") or m["awayTeam"].get("name", "?")
                        score_h = m['score']['fullTime'].get('home', 0) or 0
                        score_a = m['score']['fullTime'].get('away', 0) or 0
                        live.append({
                            "home":  home_name,
                            "away":  away_name,
                            "score": f"{score_h} - {score_a}",
                            "min":   m.get("minute", ""),
                            "time":  th.strftime("%H:%M"),
                        })
                r2 = _req.get("https://api.football-data.org/v4/competitions/PL/matches",
                              headers=headers, params={"status": "SCHEDULED"}, timeout=8)
                if r2.ok:
                    matches = sorted(r2.json().get("matches", []), key=lambda x: x["utcDate"])[:5]
                    for m in matches:
                        utc = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
                        th  = utc + timedelta(hours=7)
                        home_name = m["homeTeam"].get("shortName") or m["homeTeam"].get("name", "?")
                        away_name = m["awayTeam"].get("shortName") or m["awayTeam"].get("name", "?")
                        upcoming.append({
                            "home": home_name,
                            "away": away_name,
                            "date": th.strftime("%d %b"),
                            "time": th.strftime("%H:%M"),
                        })
            except Exception:
                pass
            return live, upcoming

        with st.spinner("📡 กำลังดึงข้อมูลแมตช์..."):
            live_matches, upcoming_matches = _fetch_live_and_upcoming()

        if live_matches:
            for m in live_matches:
                st.markdown(f"""
                <div class="match-card live">
                    <span class="match-live-badge">🔴 LIVE {m['min']}'</span>
                    <div class="match-teams">{m['home']}  <span style="color:#475569">vs</span>  {m['away']}</div>
                    <div class="match-score">{m['score']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(0,230,118,0.04);border:1px dashed rgba(0,230,118,0.15);
                        border-radius:12px;padding:0.8rem 1.2rem;margin-bottom:0.6rem">
                <span style="font-family:DM Sans,sans-serif;font-size:0.82rem;color:#3D5068">
                    ⚽ ไม่มีแมตช์ที่กำลังแข่งอยู่ขณะนี้
                </span>
            </div>
            """, unsafe_allow_html=True)

        if upcoming_matches:
            for m in upcoming_matches:
                st.markdown(f"""
                <div class="match-card upcoming">
                    <span class="match-upcoming-badge">📅 {m['date']} · {m['time']}</span>
                    <div class="match-teams">{m['home']}  <span style="color:#475569">vs</span>  {m['away']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("ไม่พบตารางแข่งถัดไป")

    # ════════════════════════════════════════
    # RIGHT — Prediction Stats + Confusion Matrix
    # ════════════════════════════════════════
    with col_right:
        st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:0.68rem;font-weight:600;'
                    'letter-spacing:0.18em;text-transform:uppercase;color:#3D5068;margin-bottom:0.6rem">'
                    '🤖 Prediction Performance</p>', unsafe_allow_html=True)

        y_test = ctx['y_test']
        y_pred = ctx['y_pred_final']
        total   = len(y_test)
        correct = int((y_test == y_pred).sum())
        wrong   = total - correct
        label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

        cm = confusion_matrix(y_test, y_pred)
        per_class = {label_map[i]: int(cm[i][i]) for i in range(3)}
        per_total = {label_map[i]: int(cm[i].sum()) for i in range(3)}

        s1, s2, s3 = st.columns(3, gap="small")
        s1.markdown(f"""
        <div class="stat-ring-wrap">
            <div class="stat-big" style="color:#00E676">{correct:,}</div>
            <div class="stat-label">ทำนายถูก</div>
        </div>""", unsafe_allow_html=True)
        s2.markdown(f"""
        <div class="stat-ring-wrap">
            <div class="stat-big" style="color:#EF4444">{wrong:,}</div>
            <div class="stat-label">ทำนายผิด</div>
        </div>""", unsafe_allow_html=True)
        s3.markdown(f"""
        <div class="stat-ring-wrap">
            <div class="stat-big" style="color:#00B0FF">{acc}%</div>
            <div class="stat-label">Accuracy</div>
        </div>""", unsafe_allow_html=True)

        st.write("")

        st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:0.68rem;font-weight:600;'
                    'letter-spacing:0.15em;text-transform:uppercase;color:#3D5068;margin-bottom:0.4rem">'
                    'แยกตามประเภทผล</p>', unsafe_allow_html=True)

        outcome_colors = {"Home Win": "#00B0FF", "Draw": "#F59E0B", "Away Win": "#A855F7"}
        for outcome, color in outcome_colors.items():
            c = per_class[outcome]
            t = per_total[outcome]
            pct = round(c / t * 100, 1) if t > 0 else 0
            bar_w = round(pct)
            st.markdown(f"""
            <div style="margin-bottom:0.55rem">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
                    <span style="font-family:DM Sans,sans-serif;font-size:0.8rem;color:#94A3B8">{outcome}</span>
                    <span style="font-family:DM Sans,sans-serif;font-size:0.78rem;font-weight:600;color:{color}">{c}/{t} ({pct}%)</span>
                </div>
                <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;overflow:hidden">
                    <div style="width:{bar_w}%;height:100%;background:{color};border-radius:4px;opacity:0.8"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        with st.expander("📊 Confusion Matrix"):
            df_cm = pd.DataFrame(cm,
                index=['Actual Away', 'Actual Draw', 'Actual Home'],
                columns=['Pred Away', 'Pred Draw', 'Pred Home'])
            st.dataframe(df_cm.style.background_gradient(cmap='Blues', axis=None),
                         use_container_width=True)

        st.write("")

        st.markdown('<p style="font-family:DM Sans,sans-serif;font-size:0.68rem;font-weight:600;'
                    'letter-spacing:0.15em;text-transform:uppercase;color:#3D5068;margin-bottom:0.4rem">'
                    '🏆 Top 5 Elo Ratings</p>', unsafe_allow_html=True)
        elo_top = sorted(ctx['final_elo'].items(), key=lambda x: x[1], reverse=True)[:5]
        max_elo = elo_top[0][1]
        for i, (team, elo_val) in enumerate(elo_top, 1):
            bar_w = round(elo_val / max_elo * 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.45rem">
                <span style="font-family:Bebas Neue,cursive;font-size:1rem;color:#3D5068;width:16px">{i}</span>
                <span style="font-family:DM Sans,sans-serif;font-size:0.82rem;color:#E2E8F0;width:140px">{team}</span>
                <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:3px;height:5px">
                    <div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,#00B0FF,#00E676);border-radius:3px"></div>
                </div>
                <span style="font-family:DM Sans,sans-serif;font-size:0.78rem;color:#64748B;width:40px;text-align:right">{round(elo_val)}</span>
            </div>
            """, unsafe_allow_html=True)