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
    show_next_pl_fixtures,
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
    st.title("Dashboard Overview")
    st.caption(f"**Today:** {TODAY.date()} | **Latest Data:** {ctx['data']['Date'].max().date()}")
    
    st.write("")
    c1, c2, c3, c4 = st.columns(4, gap="large")
    acc = round(accuracy_score(ctx['y_test'], ctx['y_pred_final']) * 100, 1)
    
    c1.metric("🎯 Target Accuracy", f"{acc}%", "Validated")
    c2.metric("📚 Training Size", f"{len(ctx['train']):,}", "Matches")
    c3.metric("🧪 Test Size", f"{len(ctx['test']):,}", "Matches")
    c4.metric("⚙️ Active Features", f"{len(ctx['FEATURES'])}", "Variables")
    
    st.divider()
    
    ca, cb = st.columns([1.5, 1], gap="large")
    with ca:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(ctx['y_test'], ctx['y_pred_final'])
        df_cm = pd.DataFrame(cm, index=['Away Win','Draw','Home Win'], columns=['Pred Away','Pred Draw','Pred Home'])
        st.dataframe(df_cm, use_container_width=True)
        
    with cb:
        st.subheader("Top 10 Elo Ratings")
        elo = sorted(ctx['final_elo'].items(), key=lambda x: x[1], reverse=True)[:10]
        df_elo = pd.DataFrame(elo, columns=['Team','Elo']).assign(**{'#': range(1,11)}).set_index('#')
        st.dataframe(df_elo, use_container_width=True)


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

        rcol1, rcol2 = st.columns([5, 1])
        rcol1.subheader("Recent Form (Last 5 Matches)")
        if latest_date is not None and pd.notna(latest_date):
            rcol1.caption(f"แมตช์ล่าสุดใน CSV: **{latest_date.strftime('%d %b %Y')}**")

        if rcol2.button("\U0001f504 Refresh", help="ดึงข้อมูลแมตช์ล่าสุดจาก API"):
            with st.spinner("กำลัง sync ข้อมูล..."):
                try:
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        result = update_season_csv_from_api()
                    if result is not None:
                        result['FTHG'] = pd.to_numeric(result['FTHG'], errors='coerce')
                        st.success(f"\u2705 Sync สำเร็จ! มีแมตช์ที่เตะแล้ว {result['FTHG'].notna().sum()} นัด")
                    else:
                        st.error(f"\u274c Sync ล้มเหลว\n```\n{buf.getvalue()}\n```")
                except Exception as _e:
                    st.error(f"\u274c Error: {_e}")
            st.cache_data.clear()
            st.rerun()

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


def page_season(ctx):
    st.title("Season Simulation")
    
    c1, c2 = st.columns(2, gap="large")
    if c1.button("🔄 Sync Season Data", use_container_width=True):
        with st.spinner("Syncing..."):
            _silent(update_season_csv_from_api)
        st.success("Season data synced successfully.")
        
    if c2.button("🔮 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Simulating remaining matches..."):
            ctx_new = _silent(run_season_simulation, ctx)
            st.session_state['ctx'] = ctx_new
            ctx = ctx_new
            
    st.divider()
    ft = ctx.get('final_table')
    if ft is not None:
        st.subheader("Predicted Final League Table")
        df = ft.sort_values('FinalPoints', ascending=False).reset_index()
        df.index = range(1, len(df)+1)
        
        def get_zone(r):
            if r <= 4: return "🏆 Champions League"
            if r <= 6: return "🌍 Europa"
            if r >= 18: return "🔻 Relegation"
            return "➖ Mid-table"
            
        df['Zone'] = [get_zone(i) for i in range(1, len(df)+1)]
        st.dataframe(df[['Team','RealPoints','PredictedPoints','FinalPoints','Zone']], use_container_width=True)
    else:
        st.info("Click 'Run Simulation' to generate the final league table projection.")


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