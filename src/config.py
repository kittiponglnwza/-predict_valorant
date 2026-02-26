"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI v9.0 — CONFIG & IMPORTS                       ║
║   Central configuration, flags, and library imports         ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import glob
import os
import pickle
import warnings
import datetime
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    brier_score_loss, log_loss, f1_score
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import poisson
import requests

# ══════════════════════════════════════════════════════════════
# CONFIG FLAGS — แก้ตรงนี้เพื่อเปิด/ปิด features
# ══════════════════════════════════════════════════════════════

# แกนหลัก: Football model ไม่พึ่ง market odds
USE_MARKET_FEATURES = False

# API key สำหรับดึงข้อมูลแมตช์
API_KEY = "745c5b802b204590bfa05c093f00bd43"

# ── Optional Libraries ────────────────────────────────────────

# 🔥 LightGBM — core model v3.0
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    lgb = None
    LGBM_AVAILABLE = False
    print("⚠️  LightGBM not found — pip install lightgbm  (falling back to GBT)")

# 🔥 SHAP — feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

# 🔥 Optuna — hyperparameter tuning
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    print("✅ Optuna available")
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not found — pip install optuna  (skipping tuning)")

# 🔥 imbalanced-learn — SMOTE for Draw class
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
    print("✅ imbalanced-learn available")
except ImportError:
    SMOTE = None
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not found — pip install imbalanced-learn  (skipping SMOTE)")

# ── Date & Paths ──────────────────────────────────────────────

TODAY        = pd.Timestamp(datetime.date.today())
# ── Resolve root path (project_ai/) ──────────────────────────
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parent.parent  # src/ → project_ai/

DATA_DIR   = str(_ROOT / "data")
MODEL_DIR  = str(_ROOT / "models")
MODEL_PATH = str(_ROOT / "models" / "football_model_v9.pkl")

# ── Team Name Mapping (API → CSV) ────────────────────────────

TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth", "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton", "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea", "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton", "Fulham FC": "Fulham",
    "Leeds United FC": "Leeds", "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nott'm Forest",
    "Sunderland AFC": "Sunderland", "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham", "Wolverhampton Wanderers FC": "Wolves",
}

def normalize_team_name(name):
    return TEAM_NAME_MAP.get(name, name)


# ── New / Promoted Teams (Season 2025) ───────────────────────
# ทีมที่เพิ่งเลื่อนชั้นมา PL — ไม่มีข้อมูลเก่า
# bootstrap stats จาก league average แทน
NEW_TEAMS_BOOTSTRAPPED = {
    # ชื่อทีม: {'tier': 'lower'|'mid'|'upper', 'elo': ค่า Elo เริ่มต้น}
    "Sunderland":    {'tier': 'lower', 'elo': 1450},
    "Leicester":     {'tier': 'lower', 'elo': 1460},
    "Ipswich":       {'tier': 'lower', 'elo': 1430},
}

# Bootstrap feature defaults ตาม tier
_BOOTSTRAP_DEFAULTS = {
    'lower': {
        'GF5': 1.0, 'GA5': 1.8, 'GD5': -0.8, 'Pts5': 1.0,
        'Streak3': 1.0, 'Win5': 0.3, 'Draw5': 0.25, 'CS5': 0.1,
        'Scored5': 0.5, 'GF_ewm': 1.0, 'GA_ewm': 1.8,
        'Pts_ewm': 1.0, 'GD_ewm': -0.8, 'Pts10': 1.0,
        'DaysRest': 7, 'GD_std': 1.8, 'Elo_HA': 1450,
        'xGF5': np.nan, 'xGA5': np.nan, 'xGD5': np.nan,
        'xGF_ewm': np.nan, 'xGA_ewm': np.nan,
        'xG_overperf': np.nan, 'xGF_slope': np.nan,
    },
    'mid': {
        'GF5': 1.3, 'GA5': 1.5, 'GD5': -0.2, 'Pts5': 1.3,
        'Streak3': 1.5, 'Win5': 0.4, 'Draw5': 0.25, 'CS5': 0.15,
        'Scored5': 0.55, 'GF_ewm': 1.3, 'GA_ewm': 1.5,
        'Pts_ewm': 1.3, 'GD_ewm': -0.2, 'Pts10': 1.3,
        'DaysRest': 7, 'GD_std': 1.6, 'Elo_HA': 1500,
        'xGF5': np.nan, 'xGA5': np.nan, 'xGD5': np.nan,
        'xGF_ewm': np.nan, 'xGA_ewm': np.nan,
        'xG_overperf': np.nan, 'xGF_slope': np.nan,
    },
    'upper': {
        'GF5': 1.6, 'GA5': 1.2, 'GD5': 0.4, 'Pts5': 1.6,
        'Streak3': 2.0, 'Win5': 0.5, 'Draw5': 0.25, 'CS5': 0.2,
        'Scored5': 0.65, 'GF_ewm': 1.6, 'GA_ewm': 1.2,
        'Pts_ewm': 1.6, 'GD_ewm': 0.4, 'Pts10': 1.6,
        'DaysRest': 7, 'GD_std': 1.4, 'Elo_HA': 1530,
        'xGF5': np.nan, 'xGA5': np.nan, 'xGD5': np.nan,
        'xGF_ewm': np.nan, 'xGA_ewm': np.nan,
        'xG_overperf': np.nan, 'xGF_slope': np.nan,
    },
}

def get_bootstrap_features(team):
    """คืน bootstrap feature dict สำหรับทีมที่เพิ่งเลื่อนชั้น"""
    info = NEW_TEAMS_BOOTSTRAPPED.get(team, {'tier': 'lower', 'elo': 1440})
    tier = info.get('tier', 'lower')
    feats = _BOOTSTRAP_DEFAULTS[tier].copy()
    feats['Elo_HA'] = info.get('elo', feats['Elo_HA'])
    return feats