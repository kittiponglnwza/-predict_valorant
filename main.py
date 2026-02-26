"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI v9.0 — MAIN ENTRY POINT                       ║
║                                                              ║
║   โครงสร้างโปรเจกต์:                                        ║
║     project_ai/                                              ║
║     ├── data/           ← ไฟล์ CSV ทุก season               ║
║     ├── models/         ← football_model_v9.pkl             ║
║     ├── src/            ← source code                        ║
║     │   ├── config.py                                        ║
║     │   ├── features.py                                      ║
║     │   ├── model.py                                         ║
║     │   ├── predict.py                                       ║
║     │   └── analysis.py                                      ║
║     ├── ui/                                                  ║
║     │   └── app_ui.py   ← Streamlit UI                       ║
║     └── main.py         ← ไฟล์นี้                           ║
║                                                              ║
║   วิธีรัน:                                                   ║
║     python main.py                   (CLI mode)             ║
║     streamlit run ui/app_ui.py       (UI mode)              ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import os

# ── เพิ่ม project root เข้า sys.path เพื่อให้ import src.* ได้ ──
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Imports ───────────────────────────────────────────────────
from src.config import *
from src.features import run_feature_pipeline
from src.model import run_training_pipeline
from src.predict import (
    update_season_csv_from_api,
    run_season_simulation,
    predict_with_api,
    show_next_pl_fixtures,
)


# ══════════════════════════════════════════════════════════════
# STEP 1 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

feat = run_feature_pipeline()

data            = feat['data']
match_df        = feat['match_df']
match_df_clean  = feat['match_df_clean']
FEATURES        = feat['FEATURES']
CORE_FEATURES   = feat['CORE_FEATURES']
XG_AVAILABLE    = feat['XG_AVAILABLE']
ODDS_AVAILABLE  = feat['ODDS_AVAILABLE']
final_elo       = feat['final_elo']
final_elo_home  = feat['final_elo_home']
final_elo_away  = feat['final_elo_away']
home_stats      = feat['home_stats']
away_stats      = feat['away_stats']
draw_stats_home = feat['draw_stats_home']


# ══════════════════════════════════════════════════════════════
# STEP 2 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════

model_result = run_training_pipeline(
    match_df_clean, FEATURES,
    home_stats, away_stats,
    final_elo, final_elo_home, final_elo_away,
)

train                = model_result['train']
test                 = model_result['test']
scaler               = model_result['scaler']
ensemble             = model_result['ensemble']
stage1_cal           = model_result['stage1_cal']
stage2_cal           = model_result['stage2_cal']
proba_hybrid         = model_result['proba_hybrid']
proba_2stage         = model_result['proba_2stage']
y_pred_final         = model_result['y_pred_final']
y_train              = model_result['y_train']
y_test               = model_result['y_test']
X_train_sc           = model_result['X_train_sc']
X_test_sc            = model_result['X_test_sc']
OPT_T_HOME           = model_result['OPT_T_HOME']
OPT_T_DRAW           = model_result['OPT_T_DRAW']
DRAW_SUPPRESS_FACTOR = model_result['DRAW_SUPPRESS_FACTOR']
POISSON_HYBRID_READY = model_result['POISSON_HYBRID_READY']
POISSON_MODEL_READY  = model_result['POISSON_MODEL_READY']
best_alpha           = model_result['best_alpha']
home_poisson_model   = model_result['home_poisson_model']
away_poisson_model   = model_result['away_poisson_model']
poisson_scaler       = model_result['poisson_scaler']
poisson_features_used= model_result['poisson_features_used']


# ── Context dict ────────────────────────────────────────────
ctx = {
    # data
    'data':                data,
    'match_df':            match_df,
    'match_df_clean':      match_df_clean,
    # features
    'FEATURES':            FEATURES,
    'XG_AVAILABLE':        XG_AVAILABLE,
    'ODDS_AVAILABLE':      ODDS_AVAILABLE,
    # model
    'scaler':              scaler,
    'ensemble':            ensemble,
    'stage1_cal':          stage1_cal,
    'stage2_cal':          stage2_cal,
    'calibrated_single':   model_result['calibrated_single'],
    'proba_hybrid':        proba_hybrid,
    'proba_2stage':        proba_2stage,
    'y_test':              y_test,
    'y_pred_final':        y_pred_final,
    'train':               train,
    'test':                test,
    'X_train_sc':          X_train_sc,
    'X_test_sc':           X_test_sc,
    # thresholds
    'OPT_T_HOME':          OPT_T_HOME,
    'OPT_T_DRAW':          OPT_T_DRAW,
    'DRAW_SUPPRESS_FACTOR':DRAW_SUPPRESS_FACTOR,
    # elo
    'final_elo':           final_elo,
    'final_elo_home':      final_elo_home,
    'final_elo_away':      final_elo_away,
    # stats
    'home_stats':          home_stats,
    'away_stats':          away_stats,
    'draw_stats_home':     draw_stats_home,
    # poisson
    'POISSON_HYBRID_READY':POISSON_HYBRID_READY,
    'POISSON_MODEL_READY': POISSON_MODEL_READY,
    'best_alpha':          best_alpha,
    'home_poisson_model':  home_poisson_model,
    'away_poisson_model':  away_poisson_model,
    'poisson_scaler':      poisson_scaler,
    'poisson_features_used':poisson_features_used,
}


# ══════════════════════════════════════════════════════════════
# STEP 3 — UPDATE DATA + SEASON SIMULATION
# ══════════════════════════════════════════════════════════════

update_season_csv_from_api()
ctx = run_season_simulation(ctx)


# ══════════════════════════════════════════════════════════════
# STEP 4 — PREDICT TEAM FIXTURES
# แก้ชื่อทีมตามต้องการ
# ══════════════════════════════════════════════════════════════

predict_with_api("Arsenal", ctx)
# predict_with_api("Liverpool", ctx)
# predict_with_api("Man City", ctx)
# predict_with_api("Chelsea", ctx)


# ══════════════════════════════════════════════════════════════
# STEP 5 — NEXT PL FIXTURES
# ══════════════════════════════════════════════════════════════

show_next_pl_fixtures(ctx, num_matches=5)


# ══════════════════════════════════════════════════════════════
# STEP 6-7 — ANALYSIS (optional)
# uncomment ที่ต้องการ
# ══════════════════════════════════════════════════════════════

# from src.analysis import run_phase2, run_phase3
# run_phase2(ctx, n_simulations=1000)
# run_phase3(ctx)
