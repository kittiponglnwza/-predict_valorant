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

from src.features import run_feature_pipeline
from src.model import run_training_pipeline, load_model

from styles       import inject_global_css
from sidebar      import render_sidebar
from page_overview  import page_overview
from page_predict   import page_predict
from page_fixtures  import page_fixtures
from page_season    import page_season
from page_analysis  import page_analysis
from page_update    import page_update

# ── 1. Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Football AI | Nexus Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Global CSS ─────────────────────────────────────────────
inject_global_css()


# ── 3. Load / Train model ─────────────────────────────────────
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


# ── 4. Session State ─────────────────────────────────────────
if 'nav_page' not in st.session_state:
    st.session_state['nav_page'] = "Overview"

if 'ctx' not in st.session_state:
    st.session_state['ctx'] = load_or_train()

ctx = st.session_state['ctx']

# ── 5. Render Sidebar ────────────────────────────────────────
render_sidebar(ctx)

# ── 6. Route to Active Page ──────────────────────────────────
pages = {
    "Overview":      page_overview,
    "Predict Match": page_predict,
    "Next Fixtures": page_fixtures,
    "Season Table":  page_season,
    "Analysis":      page_analysis,
    "Update Data":   page_update,
}

active_page = st.session_state.get('nav_page', "Overview")
pages[active_page](ctx)