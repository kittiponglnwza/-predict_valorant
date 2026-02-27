"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI v9.0 — STREAMLIT UI (Interactive Edition)      ║
║   รัน: streamlit run ui/app_ui.py                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, os

_UI_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_UI_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
import numpy as np

from src.config import *
from src.features import run_feature_pipeline
from src.model import run_training_pipeline, load_model
from src.predict import (
    predict_match, predict_score, get_last_5_results,
    run_season_simulation, update_season_csv_from_api,
    show_next_pl_fixtures, get_pl_standings_from_api,
)
from src.analysis import (
    run_monte_carlo, backtest_roi,
    analyze_draw_calibration, run_feature_importance,
)

# ── 1. Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Football AI | Nexus Engine", 
    page_icon="⚡",
    layout="wide", 
    initial_sidebar_state="expanded",
)

# ── 2. Modern Cyber Sports CSS ────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    .stApp { 
        background-color: #0B0F19; 
        color: #E2E8F0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #151B2B, #0B0F19);
        border: 1px solid #2A3441;
        padding: 1.25rem;
        border-radius: 16px;
        border-left: 4px solid #00B0FF;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 176, 255, 0.15);
        border-color: #00B0FF;
    }
    
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111622;
        border-radius: 12px;
        padding: 6px;
        gap: 8px;
        border: 1px solid #2A3441;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 8px;
        color: #94A3B8;
        padding: 0 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F2937 !important;
        color: #00B0FF !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00B0FF 0%, #0081CB 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 176, 255, 0.2);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(0, 176, 255, 0.4);
        background: linear-gradient(90deg, #1AD6FF 0%, #00B0FF 100%);
    }
    
    [data-testid="stSidebar"] {
        background-color: #0E131F !important;
        border-right: 1px solid #1F2937;
    }
    
    h1, h2, h3 {
        color: #F8FAFC;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00B0FF, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ── load/train ────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ กำลังเร่งเครื่องยนต์ AI...")
def load_or_train():
    feat = run_feature_pipeline()
    try:
        bundle = load_model()
        ctx = _ctx_from_bundle(feat, bundle)
    except Exception:
        mr = run_training_pipeline(
            feat['match_df_clean'], feat['FEATURES'],
            feat['home_stats'], feat['away_stats'],
            feat['final_elo'], feat['final_elo_home'], feat['final_elo_away'],
        )
        ctx = _ctx_from_result(feat, mr)
    return ctx


def _ctx_from_result(f, mr):
    return {
        'data': f['data'], 'match_df': f['match_df'],
        'match_df_clean': f['match_df_clean'],
        'FEATURES': f['FEATURES'], 'XG_AVAILABLE': f['XG_AVAILABLE'],
        'ODDS_AVAILABLE': f['ODDS_AVAILABLE'],
        'scaler': mr['scaler'], 'ensemble': mr['ensemble'],
        'stage1_cal': mr['stage1_cal'], 'stage2_cal': mr['stage2_cal'],
        'calibrated_single': mr['calibrated_single'],
        'proba_hybrid': mr['proba_hybrid'], 'proba_2stage': mr['proba_2stage'],
        'y_test': mr['y_test'], 'y_pred_final': mr['y_pred_final'],
        'train': mr['train'], 'test': mr['test'],
        'X_train_sc': mr['X_train_sc'], 'X_test_sc': mr['X_test_sc'],
        'OPT_T_HOME': mr['OPT_T_HOME'], 'OPT_T_DRAW': mr['OPT_T_DRAW'],
        'DRAW_SUPPRESS_FACTOR': mr['DRAW_SUPPRESS_FACTOR'],
        'final_elo': f['final_elo'], 'final_elo_home': f['final_elo_home'],
        'final_elo_away': f['final_elo_away'],
        'home_stats': f['home_stats'], 'away_stats': f['away_stats'],
        'draw_stats_home': f['draw_stats_home'],
        'POISSON_HYBRID_READY': mr['POISSON_HYBRID_READY'],
        'POISSON_MODEL_READY': mr['POISSON_MODEL_READY'],
        'best_alpha': mr['best_alpha'],
        'home_poisson_model': mr['home_poisson_model'],
        'away_poisson_model': mr['away_poisson_model'],
        'poisson_scaler': mr['poisson_scaler'],
        'poisson_features_used': mr['poisson_features_used'],
    }


def _ctx_from_bundle(f, b):
    from src.model import predict_2stage, apply_thresholds, suppress_draw_proba, TwoStageEnsemble, split_train_test
    train, test, Xtr, ytr, Xte, yte = split_train_test(f['match_df_clean'], f['FEATURES'])
    sc = b['scaler']
    Xte_sc = sc.transform(Xte)
    s1, s2 = b['stage1'], b['stage2']
    p2s = predict_2stage(Xte_sc, s1, s2)
    factor = b.get('draw_suppress_factor', 0.92)
    alpha  = b.get('poisson_alpha', 0.5)
    ph = suppress_draw_proba(p2s.copy(), draw_factor=factor)
    yp = apply_thresholds(ph, b['opt_t_home'], b['opt_t_draw'])
    ens = TwoStageEnsemble(s1, s2, b['opt_t_home'], b['opt_t_draw'])
    return {
        'data': f['data'], 'match_df': f['match_df'],
        'match_df_clean': f['match_df_clean'],
        'FEATURES': f['FEATURES'], 'XG_AVAILABLE': f['XG_AVAILABLE'],
        'ODDS_AVAILABLE': f['ODDS_AVAILABLE'],
        'scaler': sc, 'ensemble': ens,
        'stage1_cal': s1, 'stage2_cal': s2,
        'calibrated_single': b.get('fallback_single', s1),
        'proba_hybrid': ph, 'proba_2stage': p2s,
        'y_test': yte, 'y_pred_final': yp,
        'train': train, 'test': test,
        'X_train_sc': sc.transform(Xtr), 'X_test_sc': Xte_sc,
        'OPT_T_HOME': b['opt_t_home'], 'OPT_T_DRAW': b['opt_t_draw'],
        'DRAW_SUPPRESS_FACTOR': factor,
        'final_elo': b.get('elo', f['final_elo']),
        'final_elo_home': b.get('elo_home', f['final_elo_home']),
        'final_elo_away': b.get('elo_away', f['final_elo_away']),
        'home_stats': f['home_stats'], 'away_stats': f['away_stats'],
        'draw_stats_home': f['draw_stats_home'],
        'POISSON_HYBRID_READY': False, 'POISSON_MODEL_READY': False,
        'best_alpha': alpha,
        'home_poisson_model': b.get('poisson_model_home'),
        'away_poisson_model': b.get('poisson_model_away'),
        'poisson_scaler': b.get('poisson_scaler'),
        'poisson_features_used': b.get('poisson_features', []),
    }


def _silent(fn, *args, **kwargs):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result


# ── Sidebar ──────────────────────────────────────────────────
def render_sidebar(ctx):
    from sklearn.metrics import accuracy_score
    with st.sidebar:
        st.title("⚡ Nexus Engine")
        st.caption("FOOTBALL AI v9.0")
        st.divider()
        
        # เชื่อม Sidebar กับ session_state
        page = st.radio("Navigation", [
            "Overview", "Predict Match", "Next Fixtures",
            "Season Table", "Analysis", "Update Data",
        ], key="nav_page", label_visibility="collapsed")
        
        st.divider()
        st.markdown("### 📊 System Status")
        acc = round(accuracy_score(ctx['y_test'], ctx['y_pred_final']) * 100, 1)
        st.metric("Model Accuracy", f"{acc}%")
        
        c1, c2 = st.columns(2)
        c1.metric("Hybrid Mode", "ON" if ctx['POISSON_HYBRID_READY'] else "OFF")
        c2.metric("α Value", f"{ctx['best_alpha']:.2f}")
        
        st.write("")
        st.caption(f"**Features:** `{len(ctx['FEATURES'])}` | **xG:** `{'Active' if ctx['XG_AVAILABLE'] else 'Inactive'}`")
        st.caption(f"**T_Home:** `{ctx['OPT_T_HOME']:.2f}` | **T_Draw:** `{ctx['OPT_T_DRAW']:.2f}`")


# ── Pages ─────────────────────────────────────────────────────
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
                # Live matches
                r = _req.get("https://api.football-data.org/v4/competitions/PL/matches",
                             headers=headers, params={"status": "IN_PLAY,PAUSED"}, timeout=8)
                if r.ok:
                    for m in r.json().get("matches", []):
                        utc = datetime.fromisoformat(m["utcDate"].replace("Z","+00:00"))
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
                # Upcoming (next 5)
                r2 = _req.get("https://api.football-data.org/v4/competitions/PL/matches",
                              headers=headers, params={"status": "SCHEDULED"}, timeout=8)
                if r2.ok:
                    matches = sorted(r2.json().get("matches", []), key=lambda x: x["utcDate"])[:5]
                    for m in matches:
                        utc = datetime.fromisoformat(m["utcDate"].replace("Z","+00:00"))
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
        total  = len(y_test)
        correct= int((y_test == y_pred).sum())
        wrong  = total - correct
        label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

        # Per-class accuracy
        cm = confusion_matrix(y_test, y_pred)
        per_class = {label_map[i]: int(cm[i][i]) for i in range(3)}
        per_total = {label_map[i]: int(cm[i].sum()) for i in range(3)}

        # Big stat cards
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

        # Per-outcome breakdown
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

        # Confusion Matrix
        with st.expander("📊 Confusion Matrix"):
            df_cm = pd.DataFrame(cm,
                index=['Actual Away','Actual Draw','Actual Home'],
                columns=['Pred Away','Pred Draw','Pred Home'])
            st.dataframe(df_cm.style.background_gradient(cmap='Blues', axis=None),
                         use_container_width=True)

        st.write("")

        # Top 5 Elo
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


def page_predict(ctx):
    st.title("Match Prediction")
    st.write("เลือกทีมเหย้าและทีมเยือนเพื่อดูการวิเคราะห์ความน่าจะเป็นและสกอร์ที่คาดการณ์")
    
    all_teams = sorted(set(ctx['match_df_clean']['HomeTeam'].tolist() +
                           ctx['match_df_clean']['AwayTeam'].tolist()) |
                       set(NEW_TEAMS_BOOTSTRAPPED.keys()))
                       
    st.write("") 
    
    # ดึงค่าที่รับมาจากหน้า Fixtures (ถ้ามี)
    default_h = st.session_state.get('pred_home', "Arsenal")
    if default_h not in all_teams: default_h = all_teams[0]
    
    default_a = st.session_state.get('pred_away', "Chelsea")
    if default_a not in all_teams: default_a = all_teams[1]
    
    with st.container():
        c1, c2 = st.columns(2, gap="medium")
        home = c1.selectbox("🏠 Home Team", all_teams, index=all_teams.index(default_h))
        
        away_opts = [t for t in all_teams if t != home]
        idx_a = away_opts.index(default_a) if default_a in away_opts else 0
        away = c2.selectbox("✈️ Away Team", away_opts, index=idx_a)

    st.write("")
    
    # ถ้ารับคำสั่งให้วิเคราะห์อัตโนมัติมาจากหน้าตาราง ให้ pop ค่าออกเพื่อไม่ให้ค้าง
    auto_run = st.session_state.pop('auto_predict', False)
    
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True) or auto_run:
        with st.spinner("Analyzing match data..."):
            r = _silent(predict_match, home, away, ctx)
            s = _silent(predict_score, home, away, ctx)
            
        if r:
            st.divider()
            cr, cs = st.columns([1.2, 1], gap="large")
            with cr:
                st.subheader("Win Probability")
                m1, m2, m3 = st.columns(3)
                m1.metric(home, f"{r['Home Win']}%")
                m2.metric("Draw", f"{r['Draw']}%")
                m3.metric(away, f"{r['Away Win']}%")
                
                st.write("")
                st.progress(r['Home Win']/100, text=f"🏠 Home ({r['Home Win']}%)")
                st.progress(r['Draw']/100, text=f"🤝 Draw ({r['Draw']}%)")
                st.progress(r['Away Win']/100, text=f"✈️ Away ({r['Away Win']}%)")
                
                st.write("")
                st.success(f"**💡 Predicted Outcome:** {r['Prediction']}")
                
            with cs:
                if s:
                    st.subheader("Expected Goals (xG)")
                    x1, x2 = st.columns(2)
                    x1.metric(f"xG {home}", s['home_xg'])
                    x2.metric(f"xG {away}", s['away_xg'])
                    
                    st.write("Most Likely Scores")
                    df_scores = pd.DataFrame(s['top5_scores'], columns=['Score','Probability (%)'])
                    st.dataframe(df_scores, hide_index=True, use_container_width=True)
                    
        st.divider()

        # ── โหลดข้อมูลสดจาก CSV โดยตรง (ไม่พึ่ง ctx['data'] ที่ค้างตั้งแต่ app start) ──
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
            combined = combined.drop_duplicates(subset=['Date','HomeTeam','AwayTeam'], keep='last')
            return combined.dropna(subset=['FTHG','FTAG']).sort_values('Date').reset_index(drop=True)

        def _get_team_form(team, valid_data):
            hm = valid_data[valid_data['HomeTeam'] == team].copy()
            hm['Venue'] = 'H'; hm['GF'] = hm['FTHG']; hm['GA'] = hm['FTAG']; hm['Opponent'] = hm['AwayTeam']
            am = valid_data[valid_data['AwayTeam'] == team].copy()
            am['Venue'] = 'A'; am['GF'] = am['FTAG']; am['GA'] = am['FTHG']; am['Opponent'] = am['HomeTeam']
            all_m = pd.concat([hm, am]).sort_values('Date', ascending=False).head(5)
            def rl(r):
                if r['GF'] > r['GA']: return 'W'
                elif r['GF'] == r['GA']: return 'D'
                else: return 'L'
            all_m['Result'] = all_m.apply(rl, axis=1)
            return all_m

        fresh_data = _load_fresh_match_data()
        latest_date = fresh_data['Date'].max() if len(fresh_data) > 0 else None

        st.subheader("Recent Form (Last 5 Matches)")
        if latest_date is not None and pd.notna(latest_date):
            st.caption(f"แมตช์ล่าสุดใน CSV: **{latest_date.strftime('%d %b %Y')}**")

        ch, ca = st.columns(2, gap="large")
        for team, col in [(home, ch), (away, ca)]:
            with col:
                st.markdown(f"**{team}**")
                if len(fresh_data) > 0:
                    try:
                        d = _get_team_form(team, fresh_data)
                        d = d[['Date','Opponent','Venue','GF','GA','Result']].copy()
                        d['Date'] = d['Date'].dt.strftime('%d/%m/%y')
                        st.dataframe(d, hide_index=True, use_container_width=True)
                    except Exception:
                        st.warning("No recent data available.")
                else:
                    st.warning("No recent data available.")


# ── ฟังก์ชันนี้ใช้สำหรับเปลี่ยนหน้าจากปุ่มตาราง ──
def navigate_to_predict(home_team, away_team):
    st.session_state['nav_page'] = "Predict Match"
    st.session_state['pred_home'] = home_team
    st.session_state['pred_away'] = away_team
    st.session_state['auto_predict'] = True


def page_fixtures(ctx):
    st.title("Upcoming Fixtures")
    
    c1, c2 = st.columns([1, 3])
    n = c1.number_input("Number of matches to fetch", min_value=1, max_value=20, value=5)
    
    st.write("")
    if st.button("📡 Fetch Fixtures", type="primary"):
        with st.spinner("Fetching data from API..."):
            upcoming = _silent(show_next_pl_fixtures, ctx, num_matches=n)
            
        if upcoming:
            st.write("")
            st.markdown("### 🏟️ Select Match to Analyze")
            
            # สร้างตารางส่วนหัวด้วยคอลัมน์
            h1, h2, h3, h4, h5 = st.columns([1.5, 3.5, 3, 1.5, 1.5])
            h1.markdown("⏱️ **Date / Time**")
            h2.markdown("⚔️ **Match**")
            h3.markdown("📊 **Win Prob (H - D - A)**")
            h4.markdown("⚽ **Exp. Score**")
            h5.markdown("⚡ **Action**")
            st.divider()
            
            # ดึงข้อมูลแต่ละแถวมาแสดงพร้อมปุ่มคลิก
            for i, f in enumerate(upcoming):
                r = _silent(predict_match, f['HomeTeam'], f['AwayTeam'], ctx)
                s = _silent(predict_score, f['HomeTeam'], f['AwayTeam'], ctx)
                
                if r and s:
                    c1, c2, c3, c4, c5 = st.columns([1.5, 3.5, 3, 1.5, 1.5])
                    
                    # ข้อมูลเวลา
                    c1.caption(f"{f['Date']}  \n{f.get('Time','')}")
                    
                    # ชื่อทีม
                    c2.markdown(f"🏠 **{f['HomeTeam']}** \n✈️ **{f['AwayTeam']}**")
                    
                    # ความน่าจะเป็น
                    c3.caption(f"H: **{r['Home Win']}%** | D: **{r['Draw']}%** | A: **{r['Away Win']}%**")
                    
                    # สกอร์ที่คาด
                    c4.markdown(f"**{s['most_likely_score']}**")
                    
                    # ปุ่ม Action สำหรับพาไปหน้า Predict (ใช้ on_click แบบถูกต้อง)
                    c5.button(
                        "🎯 Predict", 
                        key=f"btn_pred_{i}", 
                        use_container_width=True,
                        on_click=navigate_to_predict,
                        args=(f['HomeTeam'], f['AwayTeam'])
                    )
                    
                    # เส้นคั่นบางๆ ระหว่างแถว
                    st.markdown("<hr style='margin: 0.5em 0; opacity: 0.15;'>", unsafe_allow_html=True)
        else:
            st.error("Unable to fetch upcoming fixtures.")


def _make_styled_table(display_df, pts_col, max_pts):
    """Shared Pandas Styler สำหรับตารางทั้ง 2 แบบ"""
    def _row_bg(row):
        p = row.name
        if p == 1:   bg = 'rgba(255,215,0,0.07)'
        elif p <= 4: bg = 'rgba(0,176,255,0.06)'
        elif p <= 6: bg = 'rgba(249,115,22,0.06)'
        elif p <= 7: bg = 'rgba(168,85,247,0.05)'
        elif p >= 18:bg = 'rgba(239,68,68,0.07)'
        else:        bg = 'transparent'
        return [f'background-color:{bg}'] * len(row)

    def _color_zone(val):
        m = {'👑': '#FFD700','⚽': '#00B0FF','🌍': '#F97316','🏅': '#A855F7','🔻': '#EF4444'}
        for emoji, color in m.items():
            if emoji in str(val):
                return f'color:{color};font-weight:600'
        return 'color:#475569'

    styled = (
        display_df.style
        .apply(_row_bg, axis=1)
        .applymap(_color_zone, subset=['Zone'])
        .bar(subset=[pts_col], color='rgba(0,176,255,0.2)', vmin=0, vmax=max_pts)
    )
    return styled


def _zone_label(pos):
    if pos == 1:   return "👑 Champion"
    if pos <= 4:   return "⚽ Champions League"
    if pos <= 6:   return "🌍 Europa League"
    if pos <= 7:   return "🏅 Conference"
    if pos >= 18:  return "🔻 Relegation"
    return "➖ Mid-table"


def _find_team_col(df):
    for c in ['Team', 'team', 'index', 'Club', 'club', 'HomeTeam']:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0]


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

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="season-hero">
        <div class="hero-eyebrow">📊 Premier League</div>
        <div class="hero-heading">SEASON <span>TABLE</span></div>
        <div class="hero-sub">ดูตารางคะแนนปัจจุบัน หรือให้ AI จำลองผลถึงสิ้นฤดูกาล</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2 Tabs ────────────────────────────────────────────────────────────
    tab_current, tab_sim = st.tabs(["📋  ตารางคะแนนปัจจุบัน", "🔮  AI Simulation"])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — ตารางคะแนนปัจจุบัน (จาก CSV จริง)
    # ════════════════════════════════════════════════════════════════
    with tab_current:
        st.write("")

        # ── Season selector (ปีที่ football-data.org ใช้ = ปีเริ่มต้น เช่น 2024 = 2024/25) ──
        current_year = TODAY.year
        # ถ้าเดือน >= 8 ฤดูกาลปัจจุบันเริ่มแล้ว ไม่งั้นยังเป็นฤดูกาลที่แล้ว
        current_season_year = current_year if TODAY.month >= 8 else current_year - 1
        season_options = list(range(current_season_year, current_season_year - 6, -1))
        season_labels  = {y: f"{y}/{str(y+1)[2:]}" for y in season_options}

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

        # ── ดึงตารางจาก API ──────────────────────────────────────
        @st.cache_data(ttl=300, show_spinner=False)
        def _fetch_standings(year):
            return _silent(get_pl_standings_from_api, year)

        with st.spinner("📡 กำลังดึงข้อมูลตารางคะแนนจาก API..."):
            rows = _fetch_standings(selected_year)

        if rows is None or len(rows) == 0:
            st.error("❌ ดึงข้อมูลจาก API ไม่ได้ — ตรวจสอบ API Key หรือการเชื่อมต่ออินเทอร์เน็ต")
        else:
            tbl = pd.DataFrame(rows)
            tbl = tbl.drop(columns=['pos'], errors='ignore')

            # แปลง Form string → emoji
            def _form_emoji(form_str):
                if not form_str: return ""
                mapping = {'W': '🟢', 'D': '🟡', 'L': '🔴'}
                return " ".join(mapping.get(c, '⚪') for c in form_str.split(',') if c)

            if 'Form' in tbl.columns:
                tbl['Form'] = tbl['Form'].apply(_form_emoji)

            tbl.index = range(1, len(tbl) + 1)
            tbl.index.name = '#'
            tbl['Zone'] = [_zone_label(i) for i in tbl.index]

            max_pts_cur = max(tbl['PTS'].max(), 1)
            styled_cur = _make_styled_table(tbl, 'PTS', max_pts_cur)
            styled_cur = (
                styled_cur
                .applymap(lambda v: 'font-weight:700;color:#fff', subset=['PTS'])
                .bar(subset=['GD'],
                     color=['rgba(239,68,68,0.3)', 'rgba(0,230,118,0.3)'],
                     vmin=-30, vmax=30)
            )
            st.dataframe(styled_cur, use_container_width=True, height=720)
            st.caption(f"📡 ข้อมูลจาก football-data.org · ฤดูกาล {season_labels[selected_year]} · "
                       f"👑 Champion · ⚽ Top 4 UCL · 🌍 Top 6 Europa · 🏅 Top 7 Conference · 🔻 Bottom 3 Relegation")

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
                _silent(update_season_csv_from_api)
            st.success("✅ Synced!")

        if sim_clicked:
            with st.spinner("🔮 Simulating remaining fixtures..."):
                ctx_new = _silent(run_season_simulation, ctx)
                st.session_state['ctx'] = ctx_new
                ctx = ctx_new

        ft = ctx.get('final_table')

        if ft is not None:
            df = ft.sort_values('FinalPoints', ascending=False).reset_index(drop=False)
            df.index = range(1, len(df) + 1)
            team_col = _find_team_col(df)
            max_pts = max(df['FinalPoints'].max(), 1)

            display_df = pd.DataFrame({
                'Club':   [str(df.loc[p, team_col]) for p in df.index],
                'PTS':    [int(df.loc[p, 'RealPoints']) for p in df.index],
                '+PROJ':  [int(df.loc[p, 'FinalPoints'] - df.loc[p, 'RealPoints']) for p in df.index],
                'FINAL':  [int(df.loc[p, 'FinalPoints']) for p in df.index],
                'Zone':   [_zone_label(p) for p in df.index],
            }, index=df.index)
            display_df.index.name = '#'

            def _color_proj(val):
                if val > 0: return 'color:#00E676;font-weight:600'
                if val < 0: return 'color:#EF4444;font-weight:600'
                return 'color:#475569'

            styled_sim = (
                _make_styled_table(display_df, 'FINAL', max_pts)
                .applymap(_color_proj, subset=['+PROJ'])
                .applymap(lambda v: 'font-weight:700;color:#fff', subset=['FINAL'])
                .format({'+PROJ': lambda x: f'+{x}' if x > 0 else str(x)})
            )

            st.dataframe(styled_sim, use_container_width=True, height=720)
            st.caption("👑 Champion · ⚽ Top 4 UCL · 🌍 Top 6 Europa · 🏅 Top 7 Conference · 🔻 Bottom 3 Relegation")

            # ── Summary cards ──────────────────────────────────────
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


def page_analysis(ctx):
    st.title("Model Analysis & Insights")
    st.write("")
    
    t1, t2, t3, t4 = st.tabs(["🎲 Monte Carlo", "💰 ROI Backtest", "⚖️ Calibration", "🧠 Feature Importance"])
    
    with t1:
        st.subheader("Monte Carlo Simulation")
        n_sim = st.slider("Number of Simulations", 100, 2000, 500, 100)
        if st.button("Run Monte Carlo", type="primary", key="btn_mc"):
            with st.spinner(f"Simulating {n_sim:,} paths..."):
                mc = _silent(run_monte_carlo, ctx, n_simulations=n_sim)
            if mc:
                df = pd.DataFrame(mc).T.sort_values('expected_pts', ascending=False)
                st.dataframe(df, use_container_width=True)
                
    with t2:
        st.subheader("Betting ROI Backtest")
        c1, c2 = st.columns(2)
        me = c1.slider("Minimum Edge (%)", 1, 10, 3) / 100
        kf = c2.slider("Kelly Fraction (%)", 5, 30, 15) / 100
        
        if st.button("Run Backtest", type="primary", key="btn_roi"):
            with st.spinner("Running historical backtest..."):
                roi = _silent(backtest_roi, ctx, min_edge=me, kelly_fraction=kf)
            if roi:
                st.write("")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("ROI", f"{roi['roi']:+.1f}%")
                r2.metric("Win Rate", f"{roi['win_rate']:.1f}%")
                r3.metric("Net P&L", f"£{roi['net_pnl']:+,.0f}")
                r4.metric("Max Drawdown", f"{roi['max_dd']:.1f}%")
                
    with t3:
        st.subheader("Draw Probability Calibration")
        if st.button("Analyze Calibration", type="primary", key="btn_cal"):
            with st.spinner("Analyzing draw rates..."):
                cal = _silent(analyze_draw_calibration, ctx)
            if cal:
                st.write("")
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Draw Rate", f"{cal['predicted_rate']:.1%}")
                c2.metric("Actual Draw Rate", f"{cal['actual_rate']:.1%}")
                c3.metric("Bias", f"{cal['bias']:+.1f}%", delta_color="inverse")
                
    with t4:
        st.subheader("Feature Importance (Top 20)")
        if st.button("Show Importance", type="primary", key="btn_fi"):
            with st.spinner("Calculating tree weights..."):
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    run_feature_importance(ctx, max_display=20)
                st.code(buf.getvalue(), language="text")


def page_update(ctx):
    st.title("Data Management")
    st.caption(f"**DATA_DIR:** `{DATA_DIR}`  |  **MODEL_PATH:** `{MODEL_PATH}`")
    st.divider()
    
    st.subheader("System Update")
    if st.button("☁️ Sync Season 2025 via API", type="primary"):
        with st.spinner("Connecting to API..."):
            df_new = _silent(update_season_csv_from_api)
        if df_new is not None:
            st.success(f"Update successful — {len(df_new):,} matches indexed.")
            st.dataframe(df_new.head(10), use_container_width=True)
        else:
            st.error("Failed to fetch update.")
            
    st.divider()
    st.subheader("Local Datasets (`/data`)")
    import glob
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if files:
        for f in files:
            try:
                df_tmp = pd.read_csv(f)
                st.markdown(f"- 📄 **{os.path.basename(f)}** — `{len(df_tmp):,}` rows")
            except:
                st.markdown(f"- ⚠️ **{os.path.basename(f)}** — *Unable to read*")
    else:
        st.info("No CSV files found in data directory.")


# ── Main ─────────────────────────────────────────────────────

# กำหนดหน้าเริ่มต้นใน session_state ถ้ายังไม่มี
if 'nav_page' not in st.session_state:
    st.session_state['nav_page'] = "Overview"

if 'ctx' not in st.session_state:
    st.session_state['ctx'] = load_or_train()

ctx  = st.session_state['ctx']

# เรนเดอร์ Sidebar
render_sidebar(ctx)

pages = {
    "Overview":       page_overview,
    "Predict Match":  page_predict,
    "Next Fixtures":  page_fixtures,
    "Season Table":   page_season,
    "Analysis":       page_analysis,
    "Update Data":    page_update,
}

# ดึงชื่อหน้าจาก session_state มาใช้ และรันฟังก์ชันของหน้านั้น
active_page = st.session_state.get('nav_page', "Overview")
pages[active_page](ctx)