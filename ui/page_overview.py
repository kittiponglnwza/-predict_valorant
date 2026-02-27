"""
ui/page_overview.py — Overview / Dashboard page for Football AI Nexus Engine
Redesigned: Broadcast-grade tactical board aesthetic. Full English. Larger typography.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.config import TODAY, API_KEY
from utils import silent


def page_overview(ctx):
    from sklearn.metrics import confusion_matrix, accuracy_score

    default_logo = "https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg"

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@400;500;600&display=swap');

    :root {
        --blue:   #00B4FF;
        --green:  #00E676;
        --amber:  #F59E0B;
        --red:    #EF4444;
        --purple: #A78BFA;
        --bg0:    #080D14;
        --bg1:    #0D1521;
        --bg2:    #121E2F;
        --border: rgba(255,255,255,0.07);
        --text1:  #F1F5F9;
        --text2:  #64748B;
        --text3:  #334155;
    }

    /* ── HERO ── */
    .hero-wrap {
        position: relative;
        background: var(--bg1);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2rem 2.5rem 1.8rem;
        margin-bottom: 1.6rem;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse 60% 80% at 90% 50%, rgba(0,180,255,0.06) 0%, transparent 100%),
            radial-gradient(ellipse 40% 60% at 10% 80%, rgba(0,230,118,0.04) 0%, transparent 100%);
        pointer-events: none;
    }
    .hero-grid-line {
        position: absolute; right: 3rem; top: 50%;
        transform: translateY(-50%);
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 7rem; font-weight: 800; letter-spacing: -0.04em;
        color: rgba(255,255,255,0.025);
        pointer-events: none; user-select: none; line-height: 1;
    }
    .hero-tag {
        font-family: 'Barlow', sans-serif;
        font-size: 0.65rem; font-weight: 600;
        letter-spacing: 0.24em; text-transform: uppercase;
        color: var(--blue); margin-bottom: 0.4rem;
        display: flex; align-items: center; gap: 6px;
    }
    .hero-tag::before {
        content: ''; display: inline-block;
        width: 18px; height: 2px; background: var(--blue);
        border-radius: 2px;
    }
    .hero-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 3.6rem; font-weight: 800;
        letter-spacing: 0.01em; line-height: 1;
        color: var(--text1); margin: 0 0 0.3rem;
    }
    .hero-title span {
        background: linear-gradient(90deg, var(--blue), var(--green));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-meta {
        font-family: 'Barlow', sans-serif;
        font-size: 0.85rem; color: var(--text2); font-weight: 500;
    }
    .hero-meta b { color: #94A3B8; font-weight: 600; }

    /* ── STAT CARD (top metrics) ── */
    .stat-card {
        background: var(--bg1);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        position: relative; overflow: hidden;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stat-card:hover {
        border-color: rgba(0,180,255,0.3);
        box-shadow: 0 4px 20px rgba(0,180,255,0.08);
    }
    .stat-card-accent {
        position: absolute; top: 0; left: 0;
        width: 3px; height: 100%; border-radius: 14px 0 0 14px;
    }
    .stat-card-label {
        font-family: 'Barlow', sans-serif;
        font-size: 0.65rem; font-weight: 600;
        letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--text2); margin-bottom: 0.4rem;
    }
    .stat-card-value {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.4rem; font-weight: 800;
        letter-spacing: -0.01em; line-height: 1; color: var(--text1);
    }
    .stat-card-sub {
        font-family: 'Barlow', sans-serif;
        font-size: 0.7rem; color: var(--text3);
        margin-top: 0.2rem; font-weight: 500;
    }

    /* ── SECTION LABEL ── */
    .sec-label {
        font-family: 'Barlow', sans-serif;
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 0.22em; text-transform: uppercase;
        color: var(--text3); margin-bottom: 0.8rem;
        display: flex; align-items: center; gap: 8px;
    }
    .sec-label::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(90deg, var(--border), transparent);
    }

    /* ── MATCH CARD ── */
    .match-block {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.55rem;
        display: flex; align-items: center; gap: 12px;
        transition: border-color 0.2s;
    }
    .match-block:hover { border-color: rgba(255,255,255,0.12); }
    .match-block.live { border-left: 3px solid var(--green); }
    .match-block.upcoming { border-left: 3px solid var(--blue); }

    .match-badge {
        font-family: 'Barlow', sans-serif;
        font-size: 0.6rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        padding: 2px 8px; border-radius: 20px;
        flex-shrink: 0;
    }
    .badge-live {
        background: rgba(0,230,118,0.15); color: var(--green);
        border: 1px solid rgba(0,230,118,0.3);
        animation: pulse 2s infinite;
    }
    .badge-sched {
        background: rgba(0,180,255,0.12); color: var(--blue);
        border: 1px solid rgba(0,180,255,0.25);
    }
    @keyframes pulse {
        0%,100% { opacity: 1; } 50% { opacity: 0.6; }
    }
    .match-teams {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.1rem; font-weight: 700;
        color: var(--text1); letter-spacing: 0.02em; flex: 1;
    }
    .match-teams-logos {
        display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    }
    .match-team {
        display: inline-flex; align-items: center; gap: 6px;
    }
    .match-logo {
        width: 20px; height: 20px; object-fit: contain; flex-shrink: 0;
        filter: drop-shadow(0 1px 4px rgba(0,0,0,0.35));
    }
    .match-vs { color: var(--text3); font-weight: 400; margin: 0 4px; }
    .match-score {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.3rem; font-weight: 800;
        color: var(--green); letter-spacing: 0.04em; flex-shrink: 0;
    }
    .match-time {
        font-family: 'Barlow', sans-serif;
        font-size: 0.72rem; color: var(--text2);
        text-align: right; flex-shrink: 0; min-width: 40px;
    }
    .no-match {
        background: rgba(0,230,118,0.03);
        border: 1px dashed rgba(0,230,118,0.12);
        border-radius: 12px; padding: 1rem 1.2rem;
        font-family: 'Barlow', sans-serif;
        font-size: 0.85rem; color: var(--text3);
    }

    /* ── PERF STATS ── */
    .perf-big-wrap {
        background: var(--bg2);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .perf-big-num {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 3.2rem; font-weight: 800; line-height: 1;
    }
    .perf-big-lbl {
        font-family: 'Barlow', sans-serif;
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 0.16em; text-transform: uppercase;
        color: var(--text3); margin-top: 0.25rem;
    }

    /* ── OUTCOME BARS ── */
    .outcome-row {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 0.6rem;
    }
    .outcome-name {
        font-family: 'Barlow', sans-serif;
        font-size: 0.82rem; font-weight: 600;
        color: #94A3B8; width: 80px; flex-shrink: 0;
    }
    .outcome-track {
        flex: 1; height: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 4px; overflow: hidden;
    }
    .outcome-fill { height: 100%; border-radius: 4px; }
    .outcome-pct {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.95rem; font-weight: 700;
        color: #94A3B8; width: 60px; text-align: right;
    }

    /* ── ELO BARS ── */
    .elo-row {
        display: flex; align-items: center;
        gap: 10px; margin-bottom: 0.5rem;
    }
    .elo-rank {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1rem; font-weight: 700;
        color: var(--text3); width: 18px; flex-shrink: 0;
    }
    .elo-name {
        font-family: 'Barlow', sans-serif;
        font-size: 0.85rem; font-weight: 600;
        color: var(--text1); width: 130px; flex-shrink: 0;
    }
    .elo-track {
        flex: 1; height: 5px;
        background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden;
    }
    .elo-fill {
        height: 100%; border-radius: 3px;
        background: linear-gradient(90deg, var(--blue), var(--green));
    }
    .elo-val {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.9rem; font-weight: 700;
        color: var(--text2); width: 40px; text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

    acc = round(accuracy_score(ctx['y_test'], ctx['y_pred_final']) * 100, 1)

    # ── HERO ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-grid-line">AI</div>
        <div class="hero-tag">Nexus Engine · Football AI v9.0</div>
        <div class="hero-title">DASHBOARD <span>OVERVIEW</span></div>
        <div class="hero-meta">
            <b>{TODAY.strftime('%A, %d %b %Y')}</b>
            &nbsp;·&nbsp; Latest data: <b>{ctx['data']['Date'].max().strftime('%d %b %Y')}</b>
            &nbsp;·&nbsp; Model accuracy: <b style="color:#00E676">{acc}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TOP STAT CARDS ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")
    cards = [
        (c1, f"{acc}%",                   "MODEL ACCURACY",    "Validated on test set",  "#00E676"),
        (c2, f"{len(ctx['train']):,}",     "TRAINING MATCHES",  "Historical match data",  "#00B4FF"),
        (c3, f"{len(ctx['test']):,}",      "TEST MATCHES",      "Held-out evaluation",    "#F59E0B"),
        (c4, f"{len(ctx['FEATURES'])}",    "FEATURES",          "Engineered predictors",  "#A78BFA"),
    ]
    for col, val, lbl, sub, color in cards:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-accent" style="background:{color}"></div>
                <div class="stat-card-label">{lbl}</div>
                <div class="stat-card-value" style="color:{color}">{val}</div>
                <div class="stat-card-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # ── MAIN 2-COL ────────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1], gap="large")

    # ════════════════════════════════════
    # LEFT — Live & Upcoming
    # ════════════════════════════════════
    with col_left:
        st.markdown('<div class="sec-label">Live &amp; Upcoming Matches</div>',
                    unsafe_allow_html=True)

        @st.cache_data(ttl=60, show_spinner=False)
        def _fetch_matches():
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
                        sh  = m['score']['fullTime'].get('home') or 0
                        sa  = m['score']['fullTime'].get('away') or 0
                        home_id = m["homeTeam"].get("id")
                        away_id = m["awayTeam"].get("id")
                        live.append({
                            "home":  m["homeTeam"].get("shortName") or m["homeTeam"]["name"],
                            "away":  m["awayTeam"].get("shortName") or m["awayTeam"]["name"],
                            "score": f"{sh}  –  {sa}",
                            "min":   m.get("minute", ""),
                            "time":  th.strftime("%H:%M"),
                            "home_logo": m["homeTeam"].get("crest") or (f"https://crests.football-data.org/{home_id}.png" if home_id else default_logo),
                            "away_logo": m["awayTeam"].get("crest") or (f"https://crests.football-data.org/{away_id}.png" if away_id else default_logo),
                        })
                r2 = _req.get("https://api.football-data.org/v4/competitions/PL/matches",
                              headers=headers, params={"status": "SCHEDULED"}, timeout=8)
                if r2.ok:
                    for m in sorted(r2.json().get("matches", []),
                                    key=lambda x: x["utcDate"])[:6]:
                        utc = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
                        th  = utc + timedelta(hours=7)
                        home_id = m["homeTeam"].get("id")
                        away_id = m["awayTeam"].get("id")
                        upcoming.append({
                            "home": m["homeTeam"].get("shortName") or m["homeTeam"]["name"],
                            "away": m["awayTeam"].get("shortName") or m["awayTeam"]["name"],
                            "date": th.strftime("%d %b"),
                            "time": th.strftime("%H:%M"),
                            "home_logo": m["homeTeam"].get("crest") or (f"https://crests.football-data.org/{home_id}.png" if home_id else default_logo),
                            "away_logo": m["awayTeam"].get("crest") or (f"https://crests.football-data.org/{away_id}.png" if away_id else default_logo),
                        })
            except Exception:
                pass
            return live, upcoming

        with st.spinner("Fetching live data..."):
            live_matches, upcoming_matches = _fetch_matches()

        if live_matches:
            for m in live_matches:
                st.markdown(f"""
                <div class="match-block live">
                    <span class="match-badge badge-live">LIVE {m['min']}'</span>
                    <div class="match-teams match-teams-logos">
                        <span class="match-team">
                            <img class="match-logo" src="{m.get('home_logo', default_logo)}" onerror="this.src='{default_logo}'"/>
                            {m['home']}
                        </span>
                        <span class="match-vs">vs</span>
                        <span class="match-team">
                            <img class="match-logo" src="{m.get('away_logo', default_logo)}" onerror="this.src='{default_logo}'"/>
                            {m['away']}
                        </span>
                    </div>
                    <div class="match-score">{m['score']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-match">No matches currently in play.</div>
            """, unsafe_allow_html=True)

        if upcoming_matches:
            st.markdown('<div class="sec-label" style="margin-top:1rem">Next Fixtures</div>',
                        unsafe_allow_html=True)
            for m in upcoming_matches:
                st.markdown(f"""
                <div class="match-block upcoming">
                    <span class="match-badge badge-sched">{m['date']}</span>
                    <div class="match-teams match-teams-logos">
                        <span class="match-team">
                            <img class="match-logo" src="{m.get('home_logo', default_logo)}" onerror="this.src='{default_logo}'"/>
                            {m['home']}
                        </span>
                        <span class="match-vs">vs</span>
                        <span class="match-team">
                            <img class="match-logo" src="{m.get('away_logo', default_logo)}" onerror="this.src='{default_logo}'"/>
                            {m['away']}
                        </span>
                    </div>
                    <div class="match-time">{m['time']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No upcoming fixtures found.")

    # ════════════════════════════════════
    # RIGHT — Prediction Performance
    # ════════════════════════════════════
    with col_right:
        st.markdown('<div class="sec-label">Prediction Performance</div>',
                    unsafe_allow_html=True)

        y_test = ctx['y_test']
        y_pred = ctx['y_pred_final']
        total   = len(y_test)
        correct = int((y_test == y_pred).sum())
        wrong   = total - correct
        label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

        cm = confusion_matrix(y_test, y_pred)
        per_class = {label_map[i]: int(cm[i][i]) for i in range(3)}
        per_total = {label_map[i]: int(cm[i].sum()) for i in range(3)}

        # Big 3 stat cards
        s1, s2, s3 = st.columns(3, gap="small")
        for col, val, lbl, color in [
            (s1, f"{correct:,}", "CORRECT",  "#00E676"),
            (s2, f"{wrong:,}",   "MISSED",   "#EF4444"),
            (s3, f"{acc}%",      "ACCURACY", "#00B4FF"),
        ]:
            with col:
                st.markdown(f"""
                <div class="perf-big-wrap">
                    <div class="perf-big-num" style="color:{color}">{val}</div>
                    <div class="perf-big-lbl">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")

        # Per-outcome accuracy bars
        st.markdown('<div class="sec-label">Accuracy by Outcome</div>',
                    unsafe_allow_html=True)

        outcome_cfg = [
            ("Home Win", "#00B4FF"),
            ("Draw",     "#F59E0B"),
            ("Away Win", "#A78BFA"),
        ]
        for outcome, color in outcome_cfg:
            c  = per_class[outcome]
            t  = per_total[outcome]
            pct = round(c / t * 100, 1) if t > 0 else 0
            st.markdown(f"""
            <div class="outcome-row">
                <div class="outcome-name">{outcome}</div>
                <div class="outcome-track">
                    <div class="outcome-fill"
                         style="width:{pct}%;background:{color};opacity:0.85"></div>
                </div>
                <div class="outcome-pct" style="color:{color}">{pct}%</div>
            </div>
            <div style="font-family:Barlow,sans-serif;font-size:0.68rem;
                        color:#334155;margin-top:-4px;margin-bottom:0.5rem;
                        padding-left:90px">{c} / {t} matches correct</div>
            """, unsafe_allow_html=True)

        st.write("")

        # Confusion matrix
        with st.expander("Confusion Matrix"):
            df_cm = pd.DataFrame(cm,
                index=['Actual Away', 'Actual Draw', 'Actual Home'],
                columns=['Pred Away', 'Pred Draw', 'Pred Home'])
            st.dataframe(df_cm.style.background_gradient(cmap='Blues', axis=None),
                         use_container_width=True)

        st.write("")

        # Top 5 Elo
        st.markdown('<div class="sec-label">Top 5 Elo Ratings</div>',
                    unsafe_allow_html=True)

        elo_top = sorted(ctx['final_elo'].items(), key=lambda x: x[1], reverse=True)[:5]
        max_elo = elo_top[0][1]
        for i, (team, elo_val) in enumerate(elo_top, 1):
            bar_w = round(elo_val / max_elo * 100)
            st.markdown(f"""
            <div class="elo-row">
                <span class="elo-rank">{i}</span>
                <span class="elo-name">{team}</span>
                <div class="elo-track">
                    <div class="elo-fill" style="width:{bar_w}%"></div>
                </div>
                <span class="elo-val">{round(elo_val)}</span>
            </div>
            """, unsafe_allow_html=True)
