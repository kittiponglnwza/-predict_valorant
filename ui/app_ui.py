"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI v9.0 — STREAMLIT UI (Interactive Edition)      ║
║   รัน: streamlit run ui/app_ui.py                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import sys, os
from pathlib import Path
import numpy as np

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

STABILIZE_REPORT_PATH = Path(_ROOT) / "artifacts" / "reports" / "stabilize_backtest_report.json"


def _load_stabilize_report():
    if not STABILIZE_REPORT_PATH.exists():
        return None
    try:
        return json.loads(STABILIZE_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _bind_stabilize_to_ctx(ctx):
    ctx['stabilize_connected'] = False
    ctx['stabilize_report_path'] = str(STABILIZE_REPORT_PATH)
    ctx['stabilize_summary'] = {}
    ctx['stabilize_settings'] = {}
    ctx['stabilize_selected_profile'] = None
    ctx['stabilize_metric_views'] = {}
    ctx['stabilize_mode'] = "monitoring_only"

    report = _load_stabilize_report()
    if not report:
        return ctx

    ctx['stabilize_connected'] = True
    ctx['stabilize_summary'] = report.get('summary', {})
    cfg = report.get('selected_settings', {}).get('global_from_validation', {})
    ctx['stabilize_settings'] = cfg
    ctx['stabilize_selected_profile'] = report.get('selected_profile')
    ctx['stabilize_metric_views'] = report.get('metric_views', {})
    return ctx


def _apply_stabilize_thresholds(ctx):
    report = _load_stabilize_report()
    if not report:
        return ctx
    cfg = report.get('selected_settings', {}).get('global_from_validation', {})
    t_home = cfg.get('t_home')
    t_draw = cfg.get('t_draw')
    if t_home is not None and t_draw is not None:
        ctx['OPT_T_HOME'] = float(t_home)
        ctx['OPT_T_DRAW'] = float(t_draw)
    return ctx


def _ensure_runtime_prediction_consistency(ctx):
    """
    Keep UI metrics tied to runtime model thresholds only.
    This also fixes stale session values from older UI runs.
    """
    try:
        from src.model import apply_thresholds
        proba = ctx.get('proba_hybrid')
        t_home = ctx.get('OPT_T_HOME')
        t_draw = ctx.get('OPT_T_DRAW')
        if proba is not None and t_home is not None and t_draw is not None:
            ctx['y_pred_final'] = apply_thresholds(proba, float(t_home), float(t_draw))
    except Exception:
        pass
    return ctx

# ── 1. Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Football AI | Nexus Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Global CSS ─────────────────────────────────────────────
inject_global_css()

# Force sidebar always visible
st.markdown(
    """<style>
    section[data-testid="stSidebar"] {
        transform: translateX(0) !important;
        visibility: visible !important;
        display: block !important;
        margin-left: 0 !important;
        left: 0 !important;
    }
    </style>""",
    unsafe_allow_html=True,
)


# ── 3. Load / Train model ─────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ กำลังเร่งเครื่องยนต์ AI...")
def load_or_train():
    feat = run_feature_pipeline()
    try:
        bundle = load_model()
        bundle_version = float(bundle.get('version', 0))
        if bundle_version < 9.3 or ('mlp2_blend_weight' not in bundle):
            raise ValueError("Model bundle is outdated, retraining required.")
        ctx = _ctx_from_bundle(feat, bundle)
    except Exception:
        mr = run_training_pipeline(
            feat['match_df_clean'], feat['FEATURES'],
            feat['home_stats'], feat['away_stats'],
            feat['final_elo'], feat['final_elo_home'], feat['final_elo_away'],
        )
        ctx = _ctx_from_result(feat, mr)
    # ── Build remaining_fixtures + final_table for Monte Carlo ──
    try:
        from src.predict import run_season_simulation
        if not ctx.get('remaining_fixtures'):
            updated_ctx = run_season_simulation(ctx)  # returns ctx directly
            if updated_ctx:
                ctx = updated_ctx
    except Exception as _e:
        pass
    ctx.setdefault('remaining_fixtures', [])
    ctx.setdefault('final_table', None)
    return _bind_stabilize_to_ctx(ctx)


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
        'MLP_MODEL_READY': mr.get('MLP_MODEL_READY', False),
        'mlp_blend_weight': mr.get('mlp_blend_weight', 0.0),
        'MLP2_MODEL_READY': mr.get('MLP2_MODEL_READY', False),
        'mlp2_blend_weight': mr.get('mlp2_blend_weight', 0.0),
        'best_alpha': mr['best_alpha'],
        'home_poisson_model': mr['home_poisson_model'],
        'away_poisson_model': mr['away_poisson_model'],
        'poisson_scaler': mr['poisson_scaler'],
        'poisson_features_used': mr['poisson_features_used'],
        'remaining_fixtures': mr.get('remaining_fixtures', []),
        'final_table': mr.get('final_table', None),
    }


def _ctx_from_bundle(f, b):
    from src.model import (
        predict_2stage,
        apply_thresholds,
        suppress_draw_proba,
        align_proba_to_classes,
        TwoStageEnsemble,
        split_train_test,
    )
    train, test, Xtr, ytr, Xte, yte = split_train_test(f['match_df_clean'], f['FEATURES'])
    sc = b['scaler']
    Xte_sc = sc.transform(Xte)
    s1, s2 = b['stage1'], b['stage2']
    p2s = predict_2stage(Xte_sc, s1, s2)
    factor = b.get('draw_suppress_factor', 0.92)
    alpha  = b.get('poisson_alpha', 0.5)
    ph = suppress_draw_proba(p2s.copy(), draw_factor=factor)
    mlp_model = b.get('mlp_model')
    mlp_weight = float(b.get('mlp_blend_weight', 0.0) or 0.0)
    if mlp_model is not None and mlp_weight > 0:
        try:
            p_mlp = mlp_model.predict_proba(Xte_sc)
            p_mlp = align_proba_to_classes(p_mlp, mlp_model.classes_)
            ph = (1.0 - mlp_weight) * ph + mlp_weight * p_mlp
            row_sums = ph.sum(axis=1, keepdims=True)
            ph = ph / np.where(row_sums > 0, row_sums, 1)
        except Exception:
            pass
    mlp2_model = b.get('mlp2_model')
    mlp2_weight = float(b.get('mlp2_blend_weight', 0.0) or 0.0)
    if mlp2_model is not None and mlp2_weight > 0:
        try:
            p_mlp2 = mlp2_model.predict_proba(Xte_sc)
            p_mlp2 = align_proba_to_classes(p_mlp2, mlp2_model.classes_)
            ph = (1.0 - mlp2_weight) * ph + mlp2_weight * p_mlp2
            row_sums = ph.sum(axis=1, keepdims=True)
            ph = ph / np.where(row_sums > 0, row_sums, 1)
        except Exception:
            pass
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
        'MLP_MODEL_READY': bool(b.get('mlp_ready', False)),
        'mlp_blend_weight': mlp_weight,
        'MLP2_MODEL_READY': bool(b.get('mlp2_ready', False)),
        'mlp2_blend_weight': mlp2_weight,
        'best_alpha': alpha,
        'home_poisson_model': b.get('poisson_model_home'),
        'away_poisson_model': b.get('poisson_model_away'),
        'poisson_scaler': b.get('poisson_scaler'),
        'poisson_features_used': b.get('poisson_features', []),
        'remaining_fixtures': b.get('remaining_fixtures', []),
        'final_table': b.get('final_table', None),
    }


# ── 4. Session State ─────────────────────────────────────────
if 'nav_page' not in st.session_state:
    st.session_state['nav_page'] = "Overview"

if 'ctx' not in st.session_state:
    st.session_state['ctx'] = load_or_train()

ctx = st.session_state['ctx']
ctx = _bind_stabilize_to_ctx(ctx)
ctx = _apply_stabilize_thresholds(ctx)
ctx = _ensure_runtime_prediction_consistency(ctx)
st.session_state['ctx'] = ctx

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