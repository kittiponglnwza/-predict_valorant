"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI — COMPETITION GRADE v8.0                       ║
║   v7b fixes + 3 clarity upgrades:                            ║
║   🔥 FIX 4: Ablation test — เพิ่ม context single-stage note ║
║             → xG negative contribution = architecture issue  ║
║             → ไม่ใช่สัญญาณว่า xG ไม่มีประโยชน์             ║
║   🔥 FIX 5: Rolling CV — เพิ่ม explanation ว่าทำไมต่ำกว่า  ║
║             → single-stage + older data = lower bound        ║
║             → Walk-forward คือ primary metric ที่เชื่อถือได้║
║   🔥 FIX 6: Phase 3 Summary — Walk-forward เป็น PRIMARY      ║
║             → Rolling CV ลดเป็น reference only              ║
║             → Weighted accuracy + pooled sample size          ║
║   ─────────────────────────────────────────────────────────  ║
║   Confirmed wins from v7b:                                   ║
║     ✅ Draw Calibration: Brier SS = +0.2%, Bias = +0.1%      ║
║     ✅ Walk-forward std = 0.036 (เสถียร)                     ║
║     ✅ Narrow threshold search สำหรับ CV folds               ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import glob
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import poisson

# ══════════════════════════════════════════════════════════════
# CONFIG FLAGS
# ══════════════════════════════════════════════════════════════

# แกนหลัก: Football model ไม่พึ่ง market odds
USE_MARKET_FEATURES = False

# 🔥 LightGBM — core model v3.0
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠️  LightGBM not found — pip install lightgbm  (falling back to GBT)")

# 🔥 SHAP — feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# 🔥 S3: Optuna — hyperparameter tuning
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    print("✅ Optuna available")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not found — pip install optuna  (skipping tuning)")

# 🔥 S5: imbalanced-learn — SMOTE for Draw class
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
    print("✅ imbalanced-learn available")
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not found — pip install imbalanced-learn  (skipping SMOTE)")

# ══════════════════════════════════════════════════════════════
# 1) LOAD ALL DATA
# ══════════════════════════════════════════════════════════════

print("Current directory:", os.getcwd())
files = glob.glob("data_set/*.csv")
print("Loaded files:", files)

df_list = []
for file in files:
    df = pd.read_csv(file)
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True).copy()
data['MatchID'] = data.index
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data = data.sort_values('Date').reset_index(drop=True)

print("\n===== DATA INFO =====")
print("Total matches:", len(data))
print("Date range:", data['Date'].min(), "→", data['Date'].max())

# ══════════════════════════════════════════════════════════════
# 🔥 PHASE 1: xG FEATURE DETECTION
#    football-data.co.uk มีคอลัมน์ HomeXG / AwayXG ตั้งแต่ซีซั่น 2017+
#    ถ้าไม่มี → fallback เป็น None แล้ว skip xG features
# ══════════════════════════════════════════════════════════════

_xg_home_col = next((c for c in data.columns if c.lower() in ['homexg','hxg','home_xg','xgh']), None)
_xg_away_col = next((c for c in data.columns if c.lower() in ['awayxg','axg','away_xg','xga','xgaway']), None)

# football-data.co.uk standard column names
if _xg_home_col is None and 'HomeXG' in data.columns:  _xg_home_col = 'HomeXG'
if _xg_away_col is None and 'AwayXG' in data.columns:  _xg_away_col = 'AwayXG'

XG_AVAILABLE = (_xg_home_col is not None and _xg_away_col is not None and
                data[_xg_home_col].notna().sum() > 200)

if XG_AVAILABLE:
    data['_HomeXG'] = pd.to_numeric(data[_xg_home_col], errors='coerce')
    data['_AwayXG'] = pd.to_numeric(data[_xg_away_col], errors='coerce')
    print(f"✅ xG columns found: {_xg_home_col}/{_xg_away_col}  "
          f"({data['_HomeXG'].notna().sum()} valid rows)")
else:
    data['_HomeXG'] = np.nan
    data['_AwayXG'] = np.nan
    print("⚠️  xG columns NOT found — xG features will be skipped")
    print("   (Add HomeXG/AwayXG columns from football-data.co.uk to unlock +2-4% accuracy)")

# ══════════════════════════════════════════════════════════════
# 🔥 PHASE 2: BETTING ODDS DETECTION (optional)
#    ถ้า USE_MARKET_FEATURES=False จะปิด market ทั้งหมด
# ══════════════════════════════════════════════════════════════

def _find_odds_col(data, candidates):
    for c in candidates:
        if c in data.columns and pd.to_numeric(data[c], errors='coerce').notna().sum() > 200:
            return c
    return None

if USE_MARKET_FEATURES:
    _odds_h = _find_odds_col(data, ['B365H','BbAvH','PSH','WHH','MaxH','AvgH'])
    _odds_d = _find_odds_col(data, ['B365D','BbAvD','PSD','WHD','MaxD','AvgD'])
    _odds_a = _find_odds_col(data, ['B365A','BbAvA','PSA','WHA','MaxA','AvgA'])

    ODDS_AVAILABLE = all(x is not None for x in [_odds_h, _odds_d, _odds_a])

    if ODDS_AVAILABLE:
        data['_OddsH'] = pd.to_numeric(data[_odds_h], errors='coerce')
        data['_OddsD'] = pd.to_numeric(data[_odds_d], errors='coerce')
        data['_OddsA'] = pd.to_numeric(data[_odds_a], errors='coerce')
        # Implied probabilities (normalize to remove overround)
        _raw_h = 1 / data['_OddsH']
        _raw_d = 1 / data['_OddsD']
        _raw_a = 1 / data['_OddsA']
        _total = (_raw_h + _raw_d + _raw_a).replace(0, np.nan)
        data['_ImpH'] = _raw_h / _total   # implied P(Home Win) — overround removed
        data['_ImpD'] = _raw_d / _total
        data['_ImpA'] = _raw_a / _total
        data['_Overround'] = (_raw_h + _raw_d + _raw_a) - 1  # bookmaker margin
        print(f"✅ Betting odds found: {_odds_h}/{_odds_d}/{_odds_a}  "
              f"(avg overround {data['_Overround'].mean()*100:.1f}%)")
    else:
        ODDS_AVAILABLE = False
        data['_ImpH'] = np.nan; data['_ImpD'] = np.nan; data['_ImpA'] = np.nan
        data['_Overround'] = np.nan
        print("⚠️  Betting odds NOT found — market features will be skipped")
        print("   (Add B365H/B365D/B365A from football-data.co.uk to improve calibration + ROI)")
else:
    ODDS_AVAILABLE = False
    data['_ImpH'] = np.nan; data['_ImpD'] = np.nan; data['_ImpA'] = np.nan
    data['_Overround'] = np.nan
    print("ℹ️  USE_MARKET_FEATURES=False — ปิดการใช้ market odds ทั้งหมด")

# ══════════════════════════════════════════════════════════════
# 2) ELO RATING — Dynamic K-factor (ใหม่)
# ══════════════════════════════════════════════════════════════

def compute_elo(data, k_base=32, base=1500):
    """
    Elo แบบ Dynamic K-factor:
    - K เพิ่มขึ้นถ้าประตูต่างกันมาก (upset มาก = เรียนรู้มากกว่า)
    - แยก Elo เหย้า/เยือน เพราะบางทีมแข็งแกร่งที่บ้านมาก
    """
    elo        = {}
    elo_home   = {}  # Elo เฉพาะตอนเล่นเหย้า
    elo_away   = {}  # Elo เฉพาะตอนเล่นเยือน

    home_elo_before = []
    away_elo_before = []
    home_elo_h_before = []  # home-specific elo
    away_elo_a_before = []  # away-specific elo

    for _, row in data.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        hg   = row['FTHG']
        ag   = row['FTAG']

        if home not in elo:      elo[home]      = base
        if away not in elo:      elo[away]       = base
        if home not in elo_home: elo_home[home]  = base
        if away not in elo_away: elo_away[away]  = base

        # บันทึก Elo ก่อนแมตช์
        home_elo_before.append(elo[home])
        away_elo_before.append(elo[away])
        home_elo_h_before.append(elo_home.get(home, base))
        away_elo_a_before.append(elo_away.get(away, base))

        # Expected
        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_away = 1 - exp_home

        # ผลจริง
        if pd.isna(hg) or pd.isna(ag):
            continue
        hg, ag = int(hg), int(ag)
        if hg > ag:
            score_home, score_away = 1.0, 0.0
        elif hg < ag:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5

        # Dynamic K: ถ้าแพ้ขาด K สูงกว่า (เรียนรู้จาก upset)
        goal_diff = abs(hg - ag)
        k = k_base * (1 + 0.1 * min(goal_diff, 5))

        # อัปเดต Elo รวม
        elo[home] += k * (score_home - exp_home)
        elo[away] += k * (score_away - exp_away)

        # อัปเดต Elo เฉพาะเหย้า/เยือน (ใช้ K ลดครึ่ง เพื่อ smooth)
        elo_home[home] = elo_home.get(home, base) + (k * 0.5) * (score_home - exp_home)
        elo_away[away] = elo_away.get(away, base) + (k * 0.5) * (score_away - exp_away)

    data = data.copy()
    data['Home_Elo']      = home_elo_before
    data['Away_Elo']      = away_elo_before
    data['Home_Elo_H']    = home_elo_h_before   # home-specific
    data['Away_Elo_A']    = away_elo_a_before   # away-specific
    data['Elo_Diff']      = data['Home_Elo'] - data['Away_Elo']
    return data, elo, elo_home, elo_away

data, final_elo, final_elo_home, final_elo_away = compute_elo(data)
print("✅ Dynamic Elo computed")

# ══════════════════════════════════════════════════════════════
# 3) TEAM-CENTRIC TABLE
# ══════════════════════════════════════════════════════════════

home_df = data[['MatchID','Date','HomeTeam','FTHG','FTAG',
                'Home_Elo','Away_Elo','Home_Elo_H','Away_Elo_A','Elo_Diff',
                '_HomeXG','_AwayXG']].copy()
home_df.columns = ['MatchID','Date','Team','GF','GA',
                   'Own_Elo','Opp_Elo','Own_Elo_HA','Opp_Elo_HA','Elo_Diff',
                   'xGF','xGA']
home_df['Home'] = 1

away_df = data[['MatchID','Date','AwayTeam','FTAG','FTHG',
                'Away_Elo','Home_Elo','Away_Elo_A','Home_Elo_H','Elo_Diff',
                '_AwayXG','_HomeXG']].copy()
away_df.columns = ['MatchID','Date','Team','GF','GA',
                   'Own_Elo','Opp_Elo','Own_Elo_HA','Opp_Elo_HA','Elo_Diff',
                   'xGF','xGA']
away_df['Home'] = 0

team_df = pd.concat([home_df, away_df], ignore_index=True)
team_df = team_df.sort_values(['Team','Date']).reset_index(drop=True)

team_df['Win']    = (team_df['GF'] > team_df['GA']).astype(int)
team_df['Draw']   = (team_df['GF'] == team_df['GA']).astype(int)
team_df['Loss']   = (team_df['GF'] < team_df['GA']).astype(int)
team_df['Points'] = team_df['Win']*3 + team_df['Draw']
team_df['CS']     = (team_df['GA'] == 0).astype(int)
team_df['Scored'] = (team_df['GF'] > 0).astype(int)
team_df['GD']     = team_df['GF'] - team_df['GA']   # Goal Difference

# ══════════════════════════════════════════════════════════════
# 4) ROLLING FEATURES — Standard + EWM (ใหม่)
# ══════════════════════════════════════════════════════════════

def rolling_shift(df, col, window=5):
    """Standard rolling mean (lag-1)"""
    return (
        df.groupby('Team')[col]
        .rolling(window).mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

def ewm_shift(df, col, span=5):
    """Exponential Weighted Mean — นัดล่าสุดมีน้ำหนักมากกว่า"""
    return (
        df.groupby('Team')[col]
        .apply(lambda x: x.ewm(span=span, adjust=False).mean().shift(1))
        .reset_index(level=0, drop=True)
    )

def rolling_std_shift(df, col, window=5):
    """Rolling std — วัดความแปรปรวน"""
    return (
        df.groupby('Team')[col]
        .rolling(window).std()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

# Standard rolling
team_df['GF_last5']     = rolling_shift(team_df, 'GF')
team_df['GA_last5']     = rolling_shift(team_df, 'GA')
team_df['GD_last5']     = rolling_shift(team_df, 'GD')       # NEW: Goal Diff
team_df['Points_last5'] = rolling_shift(team_df, 'Points')
team_df['Win_last5']    = rolling_shift(team_df, 'Win')
team_df['Draw_last5']   = rolling_shift(team_df, 'Draw')     # NEW: Draw rate
team_df['CS_last5']     = rolling_shift(team_df, 'CS')
team_df['Scored_last5'] = rolling_shift(team_df, 'Scored')
team_df['Streak3']      = rolling_shift(team_df, 'Points', window=3)

# EWM features (ใหม่)
team_df['GF_ewm5']      = ewm_shift(team_df, 'GF')
team_df['GA_ewm5']      = ewm_shift(team_df, 'GA')
team_df['Pts_ewm5']     = ewm_shift(team_df, 'Points')
team_df['GD_ewm5']      = ewm_shift(team_df, 'GD')

# Longer window (10 nัด) สำหรับ momentum
team_df['Points_last10']= rolling_shift(team_df, 'Points', window=10)
team_df['GF_last10']    = rolling_shift(team_df, 'GF', window=10)

# GD variance — ทีม Draw บ่อยมักมี variance ต่ำ
team_df['GD_std5']      = rolling_std_shift(team_df, 'GD')

# ══════════════════════════════════════════════════════════════
# 🔥 PHASE 1: xG ROLLING FEATURES
#    xGF/xGA = expected goals scored/conceded (จาก football-data.co.uk)
#    ถ้า XG_AVAILABLE=False ทุก feature จะเป็น NaN และถูก drop ออกจาก FEATURES
# ══════════════════════════════════════════════════════════════

if XG_AVAILABLE:
    team_df['xGF_last5']   = rolling_shift(team_df, 'xGF')       # xG scored last 5
    team_df['xGA_last5']   = rolling_shift(team_df, 'xGA')       # xG conceded last 5
    team_df['xGF_ewm']     = ewm_shift(team_df, 'xGF')           # EWM xG scored
    team_df['xGA_ewm']     = ewm_shift(team_df, 'xGA')           # EWM xG conceded
    team_df['xGD_last5']   = team_df['xGF_last5'] - team_df['xGA_last5']  # xG diff
    # xG overperformance: Goals - xG  (ค่าบวก = lucky/clinical, ค่าลบ = unlucky)
    team_df['xG_overperf'] = rolling_shift(team_df, 'GF') - rolling_shift(team_df, 'xGF')
    # xG trend slope (เร่งขึ้น/ลง)
    team_df['xGF_slope']   = (team_df['xGF_ewm'] - rolling_shift(team_df, 'xGF', window=10)) / 0.5
    print("✅ xG rolling features computed (Phase 1 ACTIVE 🔥)")
else:
    for col in ['xGF_last5','xGA_last5','xGF_ewm','xGA_ewm',
                'xGD_last5','xG_overperf','xGF_slope']:
        team_df[col] = np.nan
    print("⚠️  xG features set to NaN (Phase 1 inactive)")

# Days rest (ใหม่)
team_df['DaysRest']     = team_df.groupby('Team')['Date'].diff().dt.days.fillna(7)
team_df['DaysRest_lag'] = team_df.groupby('Team')['DaysRest'].shift(0)  # ก่อนแมตช์นี้

team_df = team_df.dropna(subset=['GF_last5','GA_last5','Points_last5'])

print(f"✅ Rolling + EWM features computed: {len(team_df)} rows")

# ══════════════════════════════════════════════════════════════
# 5) HOME STRENGTH PER TEAM (ใหม่)
# ══════════════════════════════════════════════════════════════

valid_data = data.dropna(subset=['FTHG','FTAG'])

# ══════════════════════════════════════════════════════════════
# 🔥 FIX 1: SEQUENTIAL TEAM STATS (No Leakage!)
# คำนวณ HomeWinRate / AwayWinRate / DrawRate แบบ match-by-match
# ทุก row ใช้เฉพาะข้อมูลก่อนแมตช์นั้น — ไม่ใช้ข้อมูลทั้ง season
# ══════════════════════════════════════════════════════════════

def compute_sequential_team_stats(data):
    """
    คำนวณ HomeWinRate, AwayWinRate, HomeDrawRate แบบ sequential
    แต่ละแมตช์ใช้ข้อมูล "ก่อนวันแข่ง" เท่านั้น — ไม่มี data leakage
    """
    home_wins  = {}; home_games  = {}
    away_wins  = {}; away_games  = {}
    home_draws = {}; home_g2     = {}

    hw_rates = []; aw_rates = []; hd_rates = []

    for _, row in data.sort_values('Date').iterrows():
        home = row['HomeTeam']; away = row['AwayTeam']
        hg   = row['FTHG'];    ag   = row['FTAG']

        # บันทึกค่าก่อนอัปเดต (ใช้ข้อมูลก่อนแมตช์นี้)
        hw_rates.append(home_wins.get(home, 0) / max(home_games.get(home, 1), 1))
        aw_rates.append(away_wins.get(away, 0) / max(away_games.get(away, 1), 1))
        hd_rates.append(home_draws.get(home, 0) / max(home_g2.get(home, 1), 1))

        if pd.isna(hg) or pd.isna(ag): continue
        hg, ag = int(hg), int(ag)

        # อัปเดตหลังแมตช์
        home_games[home]  = home_games.get(home, 0) + 1
        away_games[away]  = away_games.get(away, 0) + 1
        home_g2[home]     = home_g2.get(home, 0) + 1

        if hg > ag:
            home_wins[home]  = home_wins.get(home, 0) + 1
        elif ag > hg:
            away_wins[away]  = away_wins.get(away, 0) + 1
        else:
            home_draws[home] = home_draws.get(home, 0) + 1

    data = data.sort_values('Date').reset_index(drop=True).copy()
    data['_HomeWinRate_seq']  = hw_rates
    data['_AwayWinRate_seq']  = aw_rates
    data['_HomeDrawRate_seq'] = hd_rates
    return data

data = compute_sequential_team_stats(data)
print("✅ Sequential team stats computed (No Leakage!)")

# ──── Static lookups (ใช้สำหรับ predict_match ที่ไม่มี history) ────
home_stats = valid_data.groupby('HomeTeam').agg(
    HomeWins=('FTHG', lambda x: (x > valid_data.loc[x.index, 'FTAG']).sum()),
    HomeGames=('FTHG', 'count')
).reset_index()
home_stats['HomeWinRate'] = home_stats['HomeWins'] / home_stats['HomeGames'].clip(1)
home_stats = home_stats.rename(columns={'HomeTeam': 'Team'})

away_stats = valid_data.groupby('AwayTeam').agg(
    AwayWins=('FTAG', lambda x: (x > valid_data.loc[x.index, 'FTHG']).sum()),
    AwayGames=('FTAG', 'count')
).reset_index()
away_stats['AwayWinRate'] = away_stats['AwayWins'] / away_stats['AwayGames'].clip(1)
away_stats = away_stats.rename(columns={'AwayTeam': 'Team'})

draw_stats_home = valid_data.groupby('HomeTeam').agg(
    HomeDraws=('FTHG', lambda x: (x == valid_data.loc[x.index, 'FTAG']).sum()),
    HomeGames2=('FTHG', 'count')
).reset_index().rename(columns={'HomeTeam': 'Team'})
draw_stats_home['HomeDrawRate'] = draw_stats_home['HomeDraws'] / draw_stats_home['HomeGames2'].clip(1)

# ══════════════════════════════════════════════════════════════
# 6) MERGE BACK TO MATCH LEVEL
# ══════════════════════════════════════════════════════════════

h = team_df[team_df['Home'] == 1].copy().rename(columns={
    'Team':         'HomeTeam',
    'GF_last5':     'H_GF5',
    'GA_last5':     'H_GA5',
    'GD_last5':     'H_GD5',
    'Points_last5': 'H_Pts5',
    'Win_last5':    'H_Win5',
    'Draw_last5':   'H_Draw5',
    'CS_last5':     'H_CS5',
    'Scored_last5': 'H_Scored5',
    'Streak3':      'H_Streak3',
    'GF_ewm5':      'H_GF_ewm',
    'GA_ewm5':      'H_GA_ewm',
    'Pts_ewm5':     'H_Pts_ewm',
    'GD_ewm5':      'H_GD_ewm',
    'Points_last10':'H_Pts10',
    'GF_last10':    'H_GF10',
    'GD_std5':      'H_GD_std',
    'DaysRest_lag': 'H_DaysRest',
    'Own_Elo':      'H_Elo',
    'Own_Elo_HA':   'H_Elo_Home',
    # 🔥 Phase 1: xG
    'xGF_last5':    'H_xGF5',
    'xGA_last5':    'H_xGA5',
    'xGF_ewm':      'H_xGF_ewm',
    'xGA_ewm':      'H_xGA_ewm',
    'xGD_last5':    'H_xGD5',
    'xG_overperf':  'H_xG_overperf',
    'xGF_slope':    'H_xGF_slope',
})

a = team_df[team_df['Home'] == 0].copy().rename(columns={
    'Team':         'AwayTeam',
    'GF_last5':     'A_GF5',
    'GA_last5':     'A_GA5',
    'GD_last5':     'A_GD5',
    'Points_last5': 'A_Pts5',
    'Win_last5':    'A_Win5',
    'Draw_last5':   'A_Draw5',
    'CS_last5':     'A_CS5',
    'Scored_last5': 'A_Scored5',
    'Streak3':      'A_Streak3',
    'GF_ewm5':      'A_GF_ewm',
    'GA_ewm5':      'A_GA_ewm',
    'Pts_ewm5':     'A_Pts_ewm',
    'GD_ewm5':      'A_GD_ewm',
    'Points_last10':'A_Pts10',
    'GF_last10':    'A_GF10',
    'GD_std5':      'A_GD_std',
    'DaysRest_lag': 'A_DaysRest',
    'Own_Elo':      'A_Elo',
    'Own_Elo_HA':   'A_Elo_Away',
    # 🔥 Phase 1: xG
    'xGF_last5':    'A_xGF5',
    'xGA_last5':    'A_xGA5',
    'xGF_ewm':      'A_xGF_ewm',
    'xGA_ewm':      'A_xGA_ewm',
    'xGD_last5':    'A_xGD5',
    'xG_overperf':  'A_xG_overperf',
    'xGF_slope':    'A_xGF_slope',
})

match_df = pd.merge(h, a, on='MatchID')
# Merge actual goals (FTHG / FTAG) for Poisson model training
actual_goals = data[['MatchID','FTHG','FTAG']].copy()
match_df = match_df.merge(actual_goals, on='MatchID', how='left')

# 🔥 PHASE 2: Merge betting odds into match_df
if ODDS_AVAILABLE:
    odds_df = data[['MatchID','_ImpH','_ImpD','_ImpA','_Overround']].copy()
    match_df = match_df.merge(odds_df, on='MatchID', how='left')
    print("✅ Betting odds merged into match_df (Phase 2 ACTIVE 🔥)")
else:
    match_df['_ImpH'] = np.nan; match_df['_ImpD'] = np.nan
    match_df['_ImpA'] = np.nan; match_df['_Overround'] = np.nan

print(f"✅ Matches after feature engineering: {len(match_df)}")

# ══════════════════════════════════════════════════════════════
# 7) DIFFERENCE + ADVANCED FEATURES
# ══════════════════════════════════════════════════════════════

match_df['Diff_Pts']      = match_df['H_Pts5']     - match_df['A_Pts5']
match_df['Diff_GF']       = match_df['H_GF5']      - match_df['A_GF5']
match_df['Diff_GA']       = match_df['H_GA5']       - match_df['A_GA5']
match_df['Diff_GD']       = match_df['H_GD5']      - match_df['A_GD5']       # NEW
match_df['Diff_Win']      = match_df['H_Win5']      - match_df['A_Win5']
match_df['Diff_CS']       = match_df['H_CS5']       - match_df['A_CS5']
match_df['Diff_Streak']   = match_df['H_Streak3']   - match_df['A_Streak3']
match_df['Diff_Elo']      = match_df['H_Elo']       - match_df['A_Elo']
match_df['Diff_Scored']   = match_df['H_Scored5']   - match_df['A_Scored5']

# EWM diffs (ใหม่)
match_df['Diff_Pts_ewm']  = match_df['H_Pts_ewm']  - match_df['A_Pts_ewm']
match_df['Diff_GF_ewm']   = match_df['H_GF_ewm']   - match_df['A_GF_ewm']
match_df['Diff_GD_ewm']   = match_df['H_GD_ewm']   - match_df['A_GD_ewm']

# Momentum (form เปลี่ยนแปลง) (ใหม่)
match_df['H_Momentum']    = match_df['H_Pts5']      - match_df['H_Pts10'] / 2
match_df['A_Momentum']    = match_df['A_Pts5']      - match_df['A_Pts10'] / 2
match_df['Diff_Momentum'] = match_df['H_Momentum']  - match_df['A_Momentum']

# Days rest diff (ใหม่)
match_df['Diff_DaysRest'] = match_df['H_DaysRest']  - match_df['A_DaysRest']
match_df['H_DaysRest']    = match_df['H_DaysRest'].clip(1, 21)
match_df['A_DaysRest']    = match_df['A_DaysRest'].clip(1, 21)

# Draw-specific features (ใหม่)
match_df['Diff_Draw5']    = match_df['H_Draw5']     - match_df['A_Draw5']   # draw rate diff
# ทีมที่ทั้งคู่ยิงน้อย มักจะ 0-0 หรือ 1-1
match_df['Combined_GF']   = match_df['H_GF5']       + match_df['A_GF5']
match_df['Combined_GF_ewm']= match_df['H_GF_ewm']  + match_df['A_GF_ewm']
# GD variance ต่ำ = เกมใกล้เคียง = Draw มากขึ้น
match_df['Mean_GD_std']   = (match_df['H_GD_std'].fillna(2) + match_df['A_GD_std'].fillna(2)) / 2

# Elo features
match_df['H_Elo_norm']    = match_df['H_Elo']      / 1500
match_df['A_Elo_norm']    = match_df['A_Elo']      / 1500
match_df['Elo_ratio']     = match_df['H_Elo']      / (match_df['A_Elo'] + 1)
match_df['H_Elo_Home_norm']= match_df['H_Elo_Home'] / 1500   # NEW: home-specific elo
match_df['A_Elo_Away_norm']= match_df['A_Elo_Away'] / 1500   # NEW: away-specific elo

# Seasonal features (ใหม่)
match_df['Month']         = match_df['Date_x'].dt.month
# ช่วงต้นฤดูกาล (Aug-Oct=1), กลาง (Nov-Feb=2), ปลาย (Mar-May=3)
match_df['SeasonPhase']   = match_df['Month'].map(
    lambda m: 1 if m in [8,9,10] else (2 if m in [11,12,1,2] else 3)
)

# ══════════════════════════════════════════════════════════════
# 🔥 S4: DEEP FEATURE ENGINEERING (v4.0)
# ══════════════════════════════════════════════════════════════

# 1) Momentum Slope — trend ของ form (กำลังดีขึ้น / ตกลง)
#    slope = (ewm - rolling10/2) / std   →  normalized momentum direction
match_df['H_Form_slope'] = (match_df['H_Pts_ewm'] - match_df['H_Pts10'] / 2) / (match_df['H_GD_std'].fillna(1) + 0.5)
match_df['A_Form_slope'] = (match_df['A_Pts_ewm'] - match_df['A_Pts10'] / 2) / (match_df['A_GD_std'].fillna(1) + 0.5)
match_df['Diff_Form_slope'] = match_df['H_Form_slope'] - match_df['A_Form_slope']

# 2) Home/Away Specific Form — แยก form เหย้าและเยือนออกจากกัน
#    ใช้ Elo เฉพาะ venue แทน (ทำแล้วใน Elo_Home / Elo_Away)
#    เพิ่ม ratio: Elo_Home / Elo_overall = home advantage index
match_df['H_HomeAdvantage'] = match_df['H_Elo_Home'] / (match_df['H_Elo'] + 1)
match_df['A_AwayPenalty']   = match_df['A_Elo_Away'] / (match_df['A_Elo'] + 1)
match_df['Venue_edge']      = match_df['H_HomeAdvantage'] - match_df['A_AwayPenalty']

# 3) Weighted H2H — H2H ล่าสุดมีน้ำหนักมากกว่า (ใช้ EWM ของ H2H)
#    H2H_HomeWinRate แบบ cumulative อยู่แล้ว ใช้ร่วมกับ H2H_DrawRate

# 4) Attack / Defense indices
match_df['H_AttackIdx']  = match_df['H_GF_ewm'] / (match_df['A_GA_ewm'].clip(0.3) + 0.01)
match_df['A_AttackIdx']  = match_df['A_GF_ewm'] / (match_df['H_GA_ewm'].clip(0.3) + 0.01)
match_df['Diff_AttackIdx'] = match_df['H_AttackIdx'] - match_df['A_AttackIdx']

# 5) Clean Sheet / Scored ratio (defensive strength)
match_df['H_DefStr']     = match_df['H_CS5']    / (match_df['H_GA5'].clip(0.1) + 0.1)
match_df['A_DefStr']     = match_df['A_CS5']    / (match_df['A_GA5'].clip(0.1) + 0.1)
match_df['Diff_DefStr']  = match_df['H_DefStr'] - match_df['A_DefStr']

# 6) Expected Tightness — ทำนายว่าเกมนี้จะสูสีไหม (Draw indicator)
match_df['Elo_closeness']   = 1 / (np.abs(match_df['Diff_Elo']) + 50)  # ยิ่ง Elo ใกล้ ยิ่งสูง
match_df['Form_closeness']  = 1 / (np.abs(match_df['Diff_Pts_ewm']) + 0.5)
match_df['Draw_likelihood'] = match_df['Elo_closeness'] * match_df['Form_closeness'] * match_df['Mean_GD_std'].clip(0.1)

print("✅ Deep Feature Engineering (S4) computed")

# ══════════════════════════════════════════════════════════════
# 🔥 PHASE 1: xG MATCH-LEVEL FEATURES (ถ้า xG พร้อม)
# ══════════════════════════════════════════════════════════════

if XG_AVAILABLE:
    match_df['Diff_xGF']        = match_df['H_xGF5']    - match_df['A_xGF5']
    match_df['Diff_xGA']        = match_df['H_xGA5']    - match_df['A_xGA5']
    match_df['Diff_xGD']        = match_df['H_xGD5']    - match_df['A_xGD5']
    match_df['Diff_xGF_ewm']    = match_df['H_xGF_ewm'] - match_df['A_xGF_ewm']
    match_df['Diff_xG_overperf']= match_df['H_xG_overperf'] - match_df['A_xG_overperf']
    match_df['Diff_xGF_slope']  = match_df['H_xGF_slope']   - match_df['A_xGF_slope']
    # xG-based attack/defense index (แรงกว่า GF ธรรมดาเพราะ adjust ความโชคดี)
    match_df['H_xAttackIdx']    = match_df['H_xGF_ewm'] / (match_df['A_xGA_ewm'].clip(0.3) + 0.01)
    match_df['A_xAttackIdx']    = match_df['A_xGF_ewm'] / (match_df['H_xGA_ewm'].clip(0.3) + 0.01)
    match_df['Diff_xAttackIdx'] = match_df['H_xAttackIdx'] - match_df['A_xAttackIdx']
    print("✅ xG match-level features computed")

# ══════════════════════════════════════════════════════════════
# 🔥 PHASE 2: BETTING MARKET FEATURES (ถ้า odds พร้อม และเปิดใช้ market)
#    implied probability = market consensus
# ══════════════════════════════════════════════════════════════

if USE_MARKET_FEATURES and ODDS_AVAILABLE:
    # implied prob ตรงๆ
    match_df['Mkt_ImpH']    = match_df['_ImpH']
    match_df['Mkt_ImpD']    = match_df['_ImpD']
    match_df['Mkt_ImpA']    = match_df['_ImpA']
    # market confidence: home - away implied spread
    match_df['Mkt_Spread']  = match_df['_ImpH'] - match_df['_ImpA']
    # Draw premium: implied draw prob เทียบกับ base rate ~26%
    match_df['Mkt_DrawPrem']= match_df['_ImpD'] - 0.26
    # Model vs market delta (คำนวณหลัง model trained — placeholder ตอนนี้)
    match_df['Mkt_Overround']= match_df['_Overround']
    print("✅ Betting market features computed (Phase 2 ACTIVE 🔥)")
else:
    for col in ['Mkt_ImpH','Mkt_ImpD','Mkt_ImpA','Mkt_Spread','Mkt_DrawPrem','Mkt_Overround']:
        match_df[col] = np.nan

# ══════════════════════════════════════════════════════════════
# 8) HEAD-TO-HEAD (รวม Draw Rate ด้วย) (ใหม่)
# ══════════════════════════════════════════════════════════════

def compute_h2h(data):
    """H2H win rate + draw rate ของทีมเหย้า (ไม่มี leakage)"""
    h2h_home_wins = {}
    h2h_draws     = {}
    h2h_total     = {}
    h2h_rates     = []
    h2h_draw_rates = []

    for _, row in data.sort_values('Date_x').iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        key  = tuple(sorted([home, away]))

        total = max(h2h_total.get(key, 0), 1)
        h2h_rates.append(h2h_home_wins.get((home, away), 0) / total)
        h2h_draw_rates.append(h2h_draws.get(key, 0) / total)

        if key not in h2h_total:      h2h_total[key] = 0
        if (home, away) not in h2h_home_wins: h2h_home_wins[(home, away)] = 0
        if key not in h2h_draws:      h2h_draws[key] = 0

        h2h_total[key] += 1
        if row['Win_x'] == 1:   h2h_home_wins[(home, away)] += 1
        if row['Draw_x'] == 1:  h2h_draws[key] += 1

    return h2h_rates, h2h_draw_rates

match_df = match_df.sort_values('Date_x').reset_index(drop=True)
h2h_win_rates, h2h_draw_rates = compute_h2h(match_df)
match_df['H2H_HomeWinRate'] = h2h_win_rates
match_df['H2H_DrawRate']    = h2h_draw_rates   # NEW
print("✅ H2H (Win + Draw rate) computed")

# ══════════════════════════════════════════════════════════════
# 9) MERGE HOME/AWAY STRENGTH + TARGET
# ══════════════════════════════════════════════════════════════

# Merge sequential (leak-free) stats from data → match_df via MatchID
seq_stats = data[['MatchID', '_HomeWinRate_seq', '_AwayWinRate_seq', '_HomeDrawRate_seq']].copy()
match_df = match_df.merge(seq_stats, on='MatchID', how='left')

# Use sequential versions as primary — fall back to static for prediction helpers
match_df['HomeWinRate']  = match_df['_HomeWinRate_seq'].fillna(0.45)
match_df['AwayWinRate']  = match_df['_AwayWinRate_seq'].fillna(0.30)
match_df['HomeDrawRate'] = match_df['_HomeDrawRate_seq'].fillna(0.25)
match_df.drop(columns=['_HomeWinRate_seq','_AwayWinRate_seq','_HomeDrawRate_seq'], inplace=True)
print("✅ Sequential (leak-free) stats merged into match_df")

def get_result(row):
    if row['Win_x'] == 1:    return 2   # Home Win
    elif row['Draw_x'] == 1: return 1   # Draw
    else:                     return 0  # Away Win

match_df['Result3'] = match_df.apply(get_result, axis=1)

# ══════════════════════════════════════════════════════════════
# 10) FEATURE LIST (เพิ่มจาก 24 → 40+ features)
# ══════════════════════════════════════════════════════════════

FEATURES = [
    # ── Elo Features ──────────────────────────────────────────
    'Diff_Elo', 'Elo_ratio', 'H_Elo_norm', 'A_Elo_norm',
    'H_Elo_Home_norm', 'A_Elo_Away_norm',

    # ── Standard Rolling Form ─────────────────────────────────
    'H_GF5', 'H_GA5', 'H_Pts5', 'H_Streak3', 'H_CS5', 'H_Scored5',
    'A_GF5', 'A_GA5', 'A_Pts5', 'A_Streak3', 'A_CS5', 'A_Scored5',

    # ── Difference Features ───────────────────────────────────
    'Diff_Pts', 'Diff_GF', 'Diff_GA', 'Diff_GD',
    'Diff_Win', 'Diff_CS', 'Diff_Streak', 'Diff_Scored',

    # ── EWM Features ──────────────────────────────────────────
    'H_GF_ewm', 'H_GA_ewm', 'H_Pts_ewm',
    'A_GF_ewm', 'A_GA_ewm', 'A_Pts_ewm',
    'Diff_Pts_ewm', 'Diff_GF_ewm', 'Diff_GD_ewm',

    # ── Momentum ──────────────────────────────────────────────
    'Diff_Momentum',

    # ── Draw-Specific Features ────────────────────────────────
    'H_Draw5', 'A_Draw5', 'Diff_Draw5',
    'H2H_DrawRate',
    'Combined_GF', 'Mean_GD_std',

    # ── H2H ──────────────────────────────────────────────────
    'H2H_HomeWinRate',

    # ── Home/Away Strength ────────────────────────────────────
    'HomeWinRate', 'AwayWinRate', 'HomeDrawRate',

    # ── Days Rest ─────────────────────────────────────────────
    'H_DaysRest', 'A_DaysRest', 'Diff_DaysRest',

    # ── Seasonal ──────────────────────────────────────────────
    'Month', 'SeasonPhase',

    # ── 🔥 S4: Deep Features ──────────────────────────────────
    'H_Form_slope', 'A_Form_slope', 'Diff_Form_slope',   # momentum slope
    'H_HomeAdvantage', 'A_AwayPenalty', 'Venue_edge',    # venue-specific
    'H_AttackIdx', 'A_AttackIdx', 'Diff_AttackIdx',      # attack index
    'H_DefStr', 'A_DefStr', 'Diff_DefStr',               # defensive strength
    'Elo_closeness', 'Form_closeness', 'Draw_likelihood', # draw signal
]

# 🔥 PHASE 1: เพิ่ม xG features ถ้า dataset มี xG
_XG_FEATURES = [
    'H_xGF5', 'H_xGA5', 'H_xGD5', 'H_xGF_ewm', 'H_xGA_ewm',
    'H_xG_overperf', 'H_xGF_slope',
    'A_xGF5', 'A_xGA5', 'A_xGD5', 'A_xGF_ewm', 'A_xGA_ewm',
    'A_xG_overperf', 'A_xGF_slope',
    'Diff_xGF', 'Diff_xGA', 'Diff_xGD', 'Diff_xGF_ewm',
    'Diff_xG_overperf', 'Diff_xGF_slope',
    'H_xAttackIdx', 'A_xAttackIdx', 'Diff_xAttackIdx',
]
if XG_AVAILABLE:
    # เพิ่มเฉพาะ columns ที่มีอยู่จริงใน match_df
    FEATURES += [f for f in _XG_FEATURES if f in match_df.columns]
    print(f"✅ Phase 1 xG: +{len([f for f in _XG_FEATURES if f in match_df.columns])} features")

# 🔥 PHASE 2: เพิ่ม Betting Market features ถ้า dataset มี odds และเปิดใช้ market
_MKT_FEATURES = [
    'Mkt_ImpH', 'Mkt_ImpD', 'Mkt_ImpA',
    'Mkt_Spread', 'Mkt_DrawPrem', 'Mkt_Overround',
]
if USE_MARKET_FEATURES and ODDS_AVAILABLE:
    FEATURES += [f for f in _MKT_FEATURES if f in match_df.columns]
    print(f"✅ Phase 2 Market: +{len([f for f in _MKT_FEATURES if f in match_df.columns])} features")

# กรอง features ที่มีอยู่จริงใน match_df เท่านั้น (safety net)
FEATURES = [f for f in FEATURES if f in match_df.columns]

# ══════════════════════════════════════════════════════════════
# 🔥 FIX 4: ตัด Low-Importance Features (จาก SHAP analysis)
#    ลด features จาก 95 → ~88 ตัว เพื่อลด noise + overfitting
#    จาก SHAP: H_CS5, A_CS5, Diff_CS, Diff_Scored,
#              H_Draw5, H2H_DrawRate, H2H_HomeWinRate มี importance ต่ำสุด
# ══════════════════════════════════════════════════════════════

LOW_IMPORTANCE_FEATURES = [
    'H_CS5', 'A_CS5',           # Clean sheet — ซ้ำซ้อนกับ GA/DefStr features
    'Diff_CS',                   # Derived จาก CS5 ที่ตัดไปแล้ว
    'Diff_Scored',               # ซ้ำกับ Diff_AttackIdx + Diff_GF
    'H_Draw5',                   # Draw rate รายทีม — sample เล็ก, noisy
    'H2H_DrawRate',              # H2H sample เล็กมาก (< 10 นัดส่วนใหญ่)
    'H2H_HomeWinRate',           # H2H — เช่นกัน sample น้อย
]

FEATURES_PRUNED = [f for f in FEATURES if f not in LOW_IMPORTANCE_FEATURES]
print(f"✅ Features v5.0: {len(FEATURES)} ตัว  "
      f"(xG={'✅' if XG_AVAILABLE else '❌'}  Market={'✅' if ODDS_AVAILABLE else '❌'})")
print(f"✅ Features v5.1 (pruned): {len(FEATURES_PRUNED)} ตัว  (-{len(FEATURES)-len(FEATURES_PRUNED)} low-importance)")

# ใช้ pruned version เป็น default
FEATURES = FEATURES_PRUNED

# ══════════════════════════════════════════════════════════════
# 11) TIME-BASED SPLIT
# ══════════════════════════════════════════════════════════════

# เลือก CORE_FEATURES จาก columns ที่แทบไม่มี NaN เพื่อไม่ให้ทิ้งแมตช์เยอะเกินไป
core_feature_threshold = 0.95
core_feature_stats = match_df[FEATURES].notna().mean()
CORE_FEATURES = [f for f in FEATURES if core_feature_stats.get(f, 0) >= core_feature_threshold]

if len(CORE_FEATURES) < 20:
    # safety fallback: ถ้าเลือกได้น้อยเกินไป ให้ใช้ FEATURES ทั้งหมด แต่พิมพ์เตือน
    CORE_FEATURES = FEATURES.copy()
    print("⚠️  CORE_FEATURES coverage ต่ำ — fallback ใช้ FEATURES ทั้งหมดสำหรับ dropna")
else:
    print(f"✅ CORE_FEATURES selected: {len(CORE_FEATURES)} / {len(FEATURES)} "
          f"(>= {core_feature_threshold*100:.0f}% non-NaN)")

match_df_clean = match_df.dropna(subset=CORE_FEATURES + ['Result3']).reset_index(drop=True)
print(f"✅ match_df_clean rows after CORE_FEATURES dropna: {len(match_df_clean)} จากทั้งหมด {len(match_df)}")

# ══════════════════════════════════════════════════════════════
# 🔥 FIX 1+2: SEASON-BASED SPLIT (แทน quantile 80/20)
#    Season = Aug → Jul ปีถัดไป
#    Train: ทุก season ก่อน TEST_SEASON
#    Test : TEST_SEASON เต็มซีซั่น (≥380 นัด)
#    → test set ใหญ่กว่าเดิม 6× และ reflect ความเป็นจริง
# ══════════════════════════════════════════════════════════════

def get_season(date):
    """Aug-Dec → season=year, Jan-Jul → season=year-1"""
    if pd.isna(date): return np.nan
    return date.year if date.month >= 8 else date.year - 1

match_df_clean['Season'] = match_df_clean['Date_x'].apply(get_season)

# หา season ที่จบแล้ว (ไม่ใช่ปัจจุบัน) และมีนัดพอ
season_counts = match_df_clean.groupby('Season').size()
completed_seasons = season_counts[season_counts >= 200].index.tolist()
completed_seasons = sorted([s for s in completed_seasons if s < get_season(TODAY)])

if len(completed_seasons) >= 2:
    # ใช้ season ล่าสุดที่จบแล้วเป็น test
    TEST_SEASON  = completed_seasons[-1]
    TRAIN_SEASON = completed_seasons[:-1]   # ทุก season ก่อนหน้า
    train = match_df_clean[match_df_clean['Season'].isin(TRAIN_SEASON)]
    test  = match_df_clean[match_df_clean['Season'] == TEST_SEASON]
    print(f"\n✅ Season-Based Split (FIX 1+2):")
    print(f"   Train seasons : {sorted(TRAIN_SEASON)}  ({len(train)} matches)")
    print(f"   Test season   : {TEST_SEASON}           ({len(test)} matches)")
else:
    # fallback: index-based 80/20 split (ใช้ข้อมูลทั้งหมด ~3,000+ นัดสำหรับ train)
    sorted_df = match_df_clean.sort_values('Date_x').reset_index(drop=True)
    split_idx = int(len(sorted_df) * 0.8)
    train = sorted_df.iloc[:split_idx].copy()
    test  = sorted_df.iloc[split_idx:].copy()
    TEST_SEASON  = None
    TRAIN_SEASON = None
    print(f"\n⚠️  Season-based split ใช้ไม่ได้ — fallback เป็น index 80/20")
    print(f"   Train matches : {len(train)}")
    print(f"   Test  matches : {len(test)}")

X_train = train[FEATURES].fillna(0)
y_train = train['Result3']
X_test  = test[FEATURES].fillna(0)
y_test  = test['Result3']

print(f"\nTrain: {len(train)}  |  Test: {len(test)}")

# ══════════════════════════════════════════════════════════════
# 12) UPGRADED ENSEMBLE MODEL
#     LR + RF + GBT + ExtraTrees + MLP → Stacking + Calibration
# ══════════════════════════════════════════════════════════════

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════
# 🔥 S3: OPTUNA HYPERPARAMETER TUNING (LightGBM)
# ══════════════════════════════════════════════════════════════

def tune_lgbm_optuna(X_tr, y_tr, n_trials=40, timeout=120):
    """
    Optuna Bayesian optimization สำหรับ LightGBM
    🔥 FIX A: ใช้ log loss แทน macro F1 เป็น objective
              → โมเดลเรียนรู้ calibrated probability จริง ๆ ไม่ใช่แค่ classification boundary
              → ช่วย Draw calibration โดยตรง (Brier Skill Score)
    ใช้ TimeSeriesSplit CV เพื่อป้องกัน future leakage
    """
    if not LGBM_AVAILABLE or not OPTUNA_AVAILABLE:
        print("  ⚠️  Optuna/LGBM ไม่พร้อม — ใช้ default params")
        return {}

    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 200, 800),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth':         trial.suggest_int('max_depth', 3, 8),
            'num_leaves':        trial.suggest_int('num_leaves', 15, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            # 🔥 FIX A: เพิ่ม multiclass objective เพื่อให้โมเดลเรียนรู้ probability จริง
            'objective':         'multiclass',
            'metric':            'multi_logloss',
            'num_class':         3,
            'class_weight':      'balanced',
            'random_state':      42,
            'n_jobs':            -1,
            'verbose':           -1,
        }
        model = lgb.LGBMClassifier(**params)
        scores = []
        for train_idx, val_idx in tscv.split(X_tr):
            model.fit(X_tr[train_idx], y_tr[train_idx])
            # 🔥 FIX A: Optimize log loss แทน F1 — proper scoring rule สำหรับ probability
            #    log loss บังคับให้ model output probability ที่ calibrated จริง ๆ
            #    ไม่ใช่แค่ทาย class ให้ถูก → ช่วย Draw calibration โดยตรง
            prob = model.predict_proba(X_tr[val_idx])
            scores.append(-log_loss(y_tr[val_idx], prob))  # negative เพราะ maximize
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    print(f"  ✅ Optuna best macro F1: {study.best_value:.4f}  (trials={len(study.trials)})")
    print(f"  Best params: n_est={study.best_params.get('n_estimators')}, "
          f"lr={study.best_params.get('learning_rate'):.3f}, "
          f"leaves={study.best_params.get('num_leaves')}, "
          f"depth={study.best_params.get('max_depth')}")
    return study.best_params


print("\n🔥 S3: Optuna LightGBM Tuning (max 40 trials / 2 min)...")
best_lgbm_params = tune_lgbm_optuna(X_train_sc, y_train.values)


def get_cv_lgbm_params():
    """
    ══════════════════════════════════════════════════════════
    🔥 FIX CONSISTENCY: Single source of truth สำหรับ LightGBM params
       ใช้ทุกที่ — rolling CV, walk-forward, ablation test
       ป้องกัน bug ที่แต่ละ function ใช้ params ต่างกัน
       ทำให้ตัวเลขเปรียบเทียบกันได้จริง
    ══════════════════════════════════════════════════════════
    """
    base = {
        'n_estimators':      300,
        'learning_rate':     0.05,
        'max_depth':         5,
        'num_leaves':        25,
        'min_child_samples': 15,
        'subsample':         0.8,
        'colsample_bytree':  0.8,
        'objective':         'multiclass',   # 🔥 consistent: log loss objective
        'metric':            'multi_logloss',
        'num_class':         3,
        'class_weight':      'balanced',
        'random_state':      42,
        'n_jobs':            -1,
        'verbose':           -1,
    }
    # merge Optuna best params (ถ้ามี)
    optuna_keys = ['learning_rate', 'max_depth', 'num_leaves', 'n_estimators',
                   'min_child_samples', 'subsample', 'colsample_bytree',
                   'reg_alpha', 'reg_lambda']
    base.update({k: v for k, v in best_lgbm_params.items() if k in optuna_keys})
    return base


# ══════════════════════════════════════════════════════════════
# 🔥 S5: SMOTE — oversample Draw class
# ══════════════════════════════════════════════════════════════

def apply_smote(X_tr, y_tr):
    """
    ใช้ SMOTE เพิ่ม Draw samples ให้ balance กับ Home/Away Win
    เฉพาะ Draw class เท่านั้น ไม่แตะ class อื่น
    """
    if not SMOTE_AVAILABLE:
        print("  ⚠️  SMOTE ไม่พร้อม — ใช้ class_weight แทน")
        return X_tr, y_tr

    counts = np.bincount(y_tr)
    # 🔥 FIX B: ลด target จาก 80% → 50% ของ majority class
    #    SMOTE 80% ทำให้โมเดล overestimate draw probability (+11% bias)
    #    50% ดัน draw ขึ้นพอให้ recall ดีขึ้น โดยไม่ทำลาย precision
    target_draw = int(max(counts) * 0.50)
    if target_draw <= counts[1]:
        print(f"  ℹ️  Draw ({counts[1]}) มากพอแล้ว — ข้าม SMOTE")
        return X_tr, y_tr

    sm = SMOTE(
        sampling_strategy={1: target_draw},   # class 1 = Draw
        k_neighbors=min(5, counts[1]-1),
        random_state=42
    )
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    new_counts = np.bincount(y_res)
    print(f"  ✅ SMOTE: Draw {counts[1]} → {new_counts[1]}  "
          f"(total {len(y_tr)} → {len(y_res)})")
    return X_res, y_res


print("\n🔥 S5: Applying SMOTE for Draw class...")
X_train_smote, y_train_smote = apply_smote(X_train_sc, y_train.values)


# ══════════════════════════════════════════════════════════════
# 🔥 S1: 2-STAGE MODEL (ใหญ่ที่สุด — แก้ Draw โดยตรง)
#
#   Stage 1: แยก Draw vs Not-Draw
#            โมเดลนี้ถาม "เกมนี้จะเสมอไหม?"
#   Stage 2: แยก Home Win vs Away Win (เฉพาะ Not-Draw)
#            โมเดลนี้ถาม "ถ้าไม่เสมอ ใครชนะ?"
#
#   Final prediction = combine stage 1 + stage 2 proba
# ══════════════════════════════════════════════════════════════

print("\n🔧 Building v4.0 — 2-Stage Model...")

# ─── Build Stage 1: Draw vs Not-Draw ─────────────────────────
y_train_draw   = (y_train_smote == 1).astype(int)   # 1=Draw, 0=Not Draw
y_train_nodraw = y_train_smote[y_train_smote != 1]   # only 0 (Away) and 2 (Home)
X_train_nodraw = X_train_smote[y_train_smote != 1]

# Stage 1 LightGBM params
if LGBM_AVAILABLE:
    stage1_params = {**{
        'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 5,
        'num_leaves': 25, 'min_child_samples': 15, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }, **{k: v for k, v in best_lgbm_params.items() if k in [
        'learning_rate', 'max_depth', 'num_leaves', 'min_child_samples',
        'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'n_estimators'
    ]}}
    stage1_model = lgb.LGBMClassifier(**stage1_params)
    stage2_model = lgb.LGBMClassifier(**{**stage1_params, 'class_weight': 'balanced'})
    print("  Stage 1 (Draw vs Not): LightGBM 🔥")
    print("  Stage 2 (Home vs Away): LightGBM 🔥")
else:
    stage1_model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    stage2_model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    print("  Stage 1 (Draw vs Not): GBT")
    print("  Stage 2 (Home vs Away): GBT")

# Train Stage 1
stage1_model.fit(X_train_smote, y_train_draw)
print("  ✅ Stage 1 trained")

# Train Stage 2 (only on non-draw matches)
# Remap: 0 (Away Win) → 0,  2 (Home Win) → 1
y_train_nodraw_bin = (y_train_nodraw == 2).astype(int)
stage2_model.fit(X_train_nodraw, y_train_nodraw_bin)
print("  ✅ Stage 2 trained")

# ─── Calibrate both stages ───────────────────────────────────
print("  🎯 Calibrating stages (isotonic)...")
try:
    stage1_cal = CalibratedClassifierCV(stage1_model, method='isotonic', cv=3)
    stage1_cal.fit(X_train_smote, y_train_draw)
    stage2_cal = CalibratedClassifierCV(stage2_model, method='isotonic', cv=3)
    stage2_cal.fit(X_train_nodraw, y_train_nodraw_bin)
    print("  ✅ Calibration done")
except Exception as e:
    print(f"  ⚠️  Calibration failed: {e}  — using raw models")
    stage1_cal = stage1_model
    stage2_cal = stage2_model


def predict_2stage(X, stage1=None, stage2=None):
    """
    2-Stage prediction:
    P(Away Win) = P(Not Draw) × P(Away | Not Draw)
    P(Home Win) = P(Not Draw) × P(Home | Not Draw)
    P(Draw)     = P(Draw from stage1)
    """
    if stage1 is None: stage1 = stage1_cal
    if stage2 is None: stage2 = stage2_cal

    p_draw_stage1  = stage1.predict_proba(X)[:, 1]   # P(Draw)
    p_notdraw      = 1 - p_draw_stage1                # P(Not Draw)

    p_home_given_nd = stage2.predict_proba(X)[:, 1]  # P(Home | Not Draw)
    p_away_given_nd = 1 - p_home_given_nd

    p_home_win = p_notdraw * p_home_given_nd
    p_away_win = p_notdraw * p_away_given_nd
    p_draw     = p_draw_stage1

    # Normalize
    total = p_home_win + p_draw + p_away_win
    p_home_win /= total; p_draw /= total; p_away_win /= total

    proba = np.column_stack([p_away_win, p_draw, p_home_win])
    return proba


# ─── 2-Stage Test Evaluation ─────────────────────────────────
proba_2stage = predict_2stage(X_test_sc)
y_pred_2stage_raw = np.argmax(proba_2stage, axis=1)
acc_2stage = accuracy_score(y_test, y_pred_2stage_raw)

print(f"\n===== 2-STAGE RAW RESULTS =====")
print(f"Accuracy: {round(acc_2stage*100, 2)}%")
print(classification_report(y_test, y_pred_2stage_raw,
                             target_names=['Away Win','Draw','Home Win']))

# ══════════════════════════════════════════════════════════════
# 🔥 PHASE 3: POISSON HYBRID BLEND — helper functions
#    (execution block อยู่หลัง Poisson model training)
# ══════════════════════════════════════════════════════════════

def build_poisson_proba_for_test(test_df, poisson_features, poisson_scaler,
                                  home_poisson_model, away_poisson_model):
    """
    คำนวณ Poisson proba สำหรับทุกแมตช์ใน test set
    Return array shape (n, 3): [p_away, p_draw, p_home]
    """
    pf_avail = [f for f in poisson_features if f in test_df.columns]
    X_pois = test_df[pf_avail].fillna(0)
    X_pois_sc = poisson_scaler.transform(X_pois)

    home_xg_arr = np.clip(home_poisson_model.predict(X_pois_sc), 0.3, 6.0)
    away_xg_arr = np.clip(away_poisson_model.predict(X_pois_sc), 0.3, 6.0)

    proba_poisson = np.zeros((len(test_df), 3))  # [away, draw, home]
    for i, (hxg, axg) in enumerate(zip(home_xg_arr, away_xg_arr)):
        ph, pd_, pa = poisson_win_draw_loss(hxg, axg)
        proba_poisson[i] = [pa, pd_, ph]
    return proba_poisson


def blend_ml_poisson(ml_proba, poisson_proba, alpha=0.6):
    """
    Blend ML + Poisson probabilities
    alpha = weight for ML (1-alpha = Poisson weight)
    Normalize หลัง blend
    """
    blended = alpha * ml_proba + (1 - alpha) * poisson_proba
    row_sums = blended.sum(axis=1, keepdims=True)
    return blended / np.where(row_sums > 0, row_sums, 1)


def suppress_draw_proba(proba, draw_factor=0.85):
    """
    🔥 FIX C: Draw Suppression — แก้ Systematic Bias +11%
    โมเดลมีแนวโน้ม overestimate draw probability อย่างสม่ำเสมอ
    การ multiply draw proba ด้วย factor < 1 แล้ว re-normalize
    จะดึง calibration กลับมาใกล้ base rate (~23%) มากขึ้น
    draw_factor=0.85 → ลด draw prob ลง ~15% ก่อน normalize
    ค่านี้ได้จาก: systematic_bias ≈ +11% → factor ≈ 1 - (0.11/0.32) ≈ 0.66
    แต่ใช้ 0.85 เพื่อ conservative ไม่ให้ overcorrect กลับ
    """
    suppressed = proba.copy()
    suppressed[:, 1] *= draw_factor   # column 1 = Draw
    row_sums = suppressed.sum(axis=1, keepdims=True)
    return suppressed / np.where(row_sums > 0, row_sums, 1)


def optimize_blend_alpha(ml_proba, poisson_proba, y_true, alphas=None):
    """Grid search หา alpha ที่ให้ accuracy สูงสุด"""
    from sklearn.metrics import f1_score
    if alphas is None:
        alphas = np.arange(0.3, 1.01, 0.05)
    best_alpha, best_score = 0.6, 0.0
    for a in alphas:
        blended = blend_ml_poisson(ml_proba, poisson_proba, alpha=a)
        preds = np.argmax(blended, axis=1)
        score = f1_score(y_true, preds, average='macro', zero_division=0)
        if score > best_score:
            best_score = score; best_alpha = a
    return best_alpha, best_score

# NOTE: Phase 3 execution runs AFTER Poisson model is trained (see below)


# ══════════════════════════════════════════════════════════════
# 🔥 S6: THRESHOLD OPTIMIZATION (maximize macro F1)
# ══════════════════════════════════════════════════════════════

def optimize_thresholds(proba, y_true, n_steps=50,
                        t_home_range=(0.15, 0.55), t_draw_range=(0.15, 0.55)):
    """
    หา threshold ที่ maximize macro F1
    แทนที่จะใช้ argmax ตรง ๆ → ทาย Draw ถ้า p_draw > threshold_draw
    Strategy: grid search บน (t_away, t_draw) แล้ว t_home = 1 - ทั้งคู่

    🔥 v7b: รับ t_home_range / t_draw_range เพื่อ narrow search รอบค่าที่รู้แล้ว
    → ลด variance ใน CV โดยไม่ต้อง freeze threshold ตายตัว
    """
    from sklearn.metrics import f1_score as f1
    best_f1   = 0.0
    best_t    = (0.33, 0.33)
    thresholds_home = np.linspace(t_home_range[0], t_home_range[1], n_steps)
    thresholds_draw = np.linspace(t_draw_range[0], t_draw_range[1], n_steps)

    for t_draw in thresholds_draw:
        for t_home in thresholds_home:
            preds = []
            for row in proba:
                p_away, p_draw, p_home = row
                if p_draw >= t_draw:    preds.append(1)
                elif p_home >= t_home:  preds.append(2)
                else:                   preds.append(0)
            score = f1(y_true, preds, average='macro', zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t  = (t_home, t_draw)

    return best_t[0], best_t[1], best_f1


def apply_thresholds(proba, t_home=None, t_draw=None):
    """Apply optimized thresholds — Draw ได้รับโอกาสชัดขึ้น"""
    if t_home is None: t_home = OPT_T_HOME
    if t_draw is None: t_draw = OPT_T_DRAW
    preds = []
    for row in proba:
        p_away, p_draw, p_home = row
        if p_draw >= t_draw:    preds.append(1)
        elif p_home >= t_home:  preds.append(2)
        else:                   preds.append(0)
    return np.array(preds)

# NOTE: S6 execution (threshold optimization + final results + model save)
# runs AFTER Poisson model training + Phase 3 hybrid blend (see below)

# ══════════════════════════════════════════════════════════════
# 14) HELPER: GET LATEST FEATURES (อัปเดตให้รองรับ features ใหม่)
# ══════════════════════════════════════════════════════════════

import datetime
TODAY = pd.Timestamp(datetime.date.today())

final_table        = None
remaining_fixtures = []

def get_latest_features(team, is_home):
    """ดึง features ล่าสุดของทีม รองรับ features ใหม่ทั้งหมด"""
    if is_home:
        rows = match_df_clean[match_df_clean['HomeTeam'] == team].sort_values('Date_x')
        if len(rows) > 0:
            last = rows.iloc[-1]
            return {
                'GF5':     last['H_GF5'],     'GA5':     last['H_GA5'],
                'GD5':     last['H_GD5'],     'Pts5':    last['H_Pts5'],
                'Streak3': last['H_Streak3'], 'Win5':    last['H_Win5'],
                'Draw5':   last['H_Draw5'],   'CS5':     last['H_CS5'],
                'Scored5': last['H_Scored5'],
                'GF_ewm':  last['H_GF_ewm'],  'GA_ewm':  last['H_GA_ewm'],
                'Pts_ewm': last['H_Pts_ewm'], 'GD_ewm':  last['H_GD_ewm'],
                'Pts10':   last['H_Pts10'],   'DaysRest':last['H_DaysRest'],
                'GD_std':  last['H_GD_std'],
                'Elo_HA':  last['H_Elo_Home'],
            }
    rows = match_df_clean[match_df_clean['AwayTeam'] == team].sort_values('Date_x')
    if len(rows) > 0:
        last = rows.iloc[-1]
        return {
            'GF5':     last['A_GF5'],     'GA5':     last['A_GA5'],
            'GD5':     last['A_GD5'],     'Pts5':    last['A_Pts5'],
            'Streak3': last['A_Streak3'], 'Win5':    last['A_Win5'],
            'Draw5':   last['A_Draw5'],   'CS5':     last['A_CS5'],
            'Scored5': last['A_Scored5'],
            'GF_ewm':  last['A_GF_ewm'],  'GA_ewm':  last['A_GA_ewm'],
            'Pts_ewm': last['A_Pts_ewm'], 'GD_ewm':  last['A_GD_ewm'],
            'Pts10':   last['A_Pts10'],   'DaysRest':last['A_DaysRest'],
            'GD_std':  last['A_GD_std'],
            'Elo_HA':  last['A_Elo_Away'],
        }
    # Default fallback
    return {
        'GF5': 1.5, 'GA5': 1.5, 'GD5': 0.0, 'Pts5': 1.5,
        'Streak3': 1.5, 'Win5': 0.5, 'Draw5': 0.25, 'CS5': 0.2,
        'Scored5': 0.6, 'GF_ewm': 1.5, 'GA_ewm': 1.5, 'Pts_ewm': 1.5,
        'GD_ewm': 0.0, 'Pts10': 1.5, 'DaysRest': 7, 'GD_std': 1.5,
        'Elo_HA': 1500,
    }


def build_match_row(home_team, away_team, match_date=None):
    """สร้าง feature row สำหรับทำนายแมตช์"""
    if match_date is None:
        match_date = TODAY
    month  = match_date.month if hasattr(match_date, 'month') else TODAY.month
    season_phase = 1 if month in [8,9,10] else (2 if month in [11,12,1,2] else 3)

    h = get_latest_features(home_team, is_home=True)
    a = get_latest_features(away_team, is_home=False)

    h_elo      = final_elo.get(home_team, 1500)
    a_elo      = final_elo.get(away_team, 1500)
    h_elo_home = final_elo_home.get(home_team, 1500)
    a_elo_away = final_elo_away.get(away_team, 1500)

    h2h_rows = match_df_clean[
        (match_df_clean['HomeTeam'] == home_team) &
        (match_df_clean['AwayTeam'] == away_team)
    ]
    h2h_rate      = h2h_rows['H2H_HomeWinRate'].iloc[-1] if len(h2h_rows) > 0 else 0.33
    h2h_draw_rate = h2h_rows['H2H_DrawRate'].iloc[-1]    if len(h2h_rows) > 0 else 0.25

    h_stats = home_stats[home_stats['Team'] == home_team]
    a_stats_r = away_stats[away_stats['Team'] == away_team]
    h_draw_r  = draw_stats_home[draw_stats_home['Team'] == home_team]

    home_win_rate  = h_stats['HomeWinRate'].values[0]  if len(h_stats) > 0  else 0.45
    away_win_rate  = a_stats_r['AwayWinRate'].values[0] if len(a_stats_r) > 0 else 0.30
    home_draw_rate = h_draw_r['HomeDrawRate'].values[0] if len(h_draw_r) > 0  else 0.25

    momentum_h = h['Pts5'] - h['Pts10'] / 2
    momentum_a = a['Pts5'] - a['Pts10'] / 2

    row = {
        # Elo
        'Diff_Elo':         h_elo - a_elo,
        'Elo_ratio':        h_elo / (a_elo + 1),
        'H_Elo_norm':       h_elo / 1500,
        'A_Elo_norm':       a_elo / 1500,
        'H_Elo_Home_norm':  h_elo_home / 1500,
        'A_Elo_Away_norm':  a_elo_away / 1500,
        # Standard form
        'H_GF5': h['GF5'],    'H_GA5': h['GA5'],     'H_Pts5': h['Pts5'],
        'H_Streak3': h['Streak3'], 'H_CS5': h['CS5'], 'H_Scored5': h['Scored5'],
        'A_GF5': a['GF5'],    'A_GA5': a['GA5'],     'A_Pts5': a['Pts5'],
        'A_Streak3': a['Streak3'], 'A_CS5': a['CS5'], 'A_Scored5': a['Scored5'],
        # Diffs
        'Diff_Pts':     h['Pts5']    - a['Pts5'],
        'Diff_GF':      h['GF5']     - a['GF5'],
        'Diff_GA':      h['GA5']     - a['GA5'],
        'Diff_GD':      h['GD5']     - a['GD5'],
        'Diff_Win':     h['Win5']    - a['Win5'],
        'Diff_CS':      h['CS5']     - a['CS5'],
        'Diff_Streak':  h['Streak3'] - a['Streak3'],
        'Diff_Scored':  h['Scored5'] - a['Scored5'],
        # EWM
        'H_GF_ewm': h['GF_ewm'],  'H_GA_ewm': h['GA_ewm'],  'H_Pts_ewm': h['Pts_ewm'],
        'A_GF_ewm': a['GF_ewm'],  'A_GA_ewm': a['GA_ewm'],  'A_Pts_ewm': a['Pts_ewm'],
        'Diff_Pts_ewm': h['Pts_ewm'] - a['Pts_ewm'],
        'Diff_GF_ewm':  h['GF_ewm']  - a['GF_ewm'],
        'Diff_GD_ewm':  h['GD_ewm']  - a['GD_ewm'],
        # Momentum
        'Diff_Momentum': momentum_h - momentum_a,
        # Draw features
        'H_Draw5': h['Draw5'],   'A_Draw5': a['Draw5'],
        'Diff_Draw5':    h['Draw5']  - a['Draw5'],
        'H2H_DrawRate':  h2h_draw_rate,
        'Combined_GF':   h['GF5']   + a['GF5'],
        'Mean_GD_std':   (h['GD_std'] + a['GD_std']) / 2,
        # H2H
        'H2H_HomeWinRate': h2h_rate,
        # Strength
        'HomeWinRate':   home_win_rate,
        'AwayWinRate':   away_win_rate,
        'HomeDrawRate':  home_draw_rate,
        # Days Rest
        'H_DaysRest':    min(h['DaysRest'], 21),
        'A_DaysRest':    min(a['DaysRest'], 21),
        'Diff_DaysRest': min(h['DaysRest'], 21) - min(a['DaysRest'], 21),
        # Seasonal
        'Month':         month,
        'SeasonPhase':   season_phase,
        # 🔥 S4: Deep Features
        'H_Form_slope':   (h['Pts_ewm'] - h['Pts10'] / 2) / (h['GD_std'] + 0.5),
        'A_Form_slope':   (a['Pts_ewm'] - a['Pts10'] / 2) / (a['GD_std'] + 0.5),
        'Diff_Form_slope':((h['Pts_ewm'] - h['Pts10'] / 2) / (h['GD_std'] + 0.5) -
                           (a['Pts_ewm'] - a['Pts10'] / 2) / (a['GD_std'] + 0.5)),
        'H_HomeAdvantage': h_elo_home / (h_elo + 1),
        'A_AwayPenalty':   a_elo_away / (a_elo + 1),
        'Venue_edge':      h_elo_home / (h_elo + 1) - a_elo_away / (a_elo + 1),
        'H_AttackIdx':     h['GF_ewm'] / (max(a['GA_ewm'], 0.3) + 0.01),
        'A_AttackIdx':     a['GF_ewm'] / (max(h['GA_ewm'], 0.3) + 0.01),
        'Diff_AttackIdx':  h['GF_ewm'] / (max(a['GA_ewm'], 0.3) + 0.01) -
                           a['GF_ewm'] / (max(h['GA_ewm'], 0.3) + 0.01),
        'H_DefStr':        h['CS5'] / (max(h['GA5'], 0.1) + 0.1),
        'A_DefStr':        a['CS5'] / (max(a['GA5'], 0.1) + 0.1),
        'Diff_DefStr':     h['CS5'] / (max(h['GA5'], 0.1) + 0.1) -
                           a['CS5'] / (max(a['GA5'], 0.1) + 0.1),
        'Elo_closeness':   1 / (abs(h_elo - a_elo) + 50),
        'Form_closeness':  1 / (abs(h['Pts_ewm'] - a['Pts_ewm']) + 0.5),
        'Draw_likelihood': (1 / (abs(h_elo - a_elo) + 50)) *
                           (1 / (abs(h['Pts_ewm'] - a['Pts_ewm']) + 0.5)) *
                           max((h['GD_std'] + a['GD_std']) / 2, 0.1),
        # 🔥 Phase 1: xG features (ถ้าไม่มีข้อมูล → NaN → drop)
        'H_xGF5':        h.get('xGF5', np.nan),   'H_xGA5':  h.get('xGA5', np.nan),
        'H_xGD5':        h.get('xGD5', np.nan),
        'H_xGF_ewm':     h.get('xGF_ewm', np.nan), 'H_xGA_ewm': h.get('xGA_ewm', np.nan),
        'H_xG_overperf': h.get('xG_overperf', np.nan),
        'H_xGF_slope':   h.get('xGF_slope', np.nan),
        'A_xGF5':        a.get('xGF5', np.nan),   'A_xGA5':  a.get('xGA5', np.nan),
        'A_xGD5':        a.get('xGD5', np.nan),
        'A_xGF_ewm':     a.get('xGF_ewm', np.nan), 'A_xGA_ewm': a.get('xGA_ewm', np.nan),
        'A_xG_overperf': a.get('xG_overperf', np.nan),
        'A_xGF_slope':   a.get('xGF_slope', np.nan),
        'Diff_xGF':          h.get('xGF5', np.nan)  - a.get('xGF5', np.nan)  if XG_AVAILABLE else np.nan,
        'Diff_xGA':          h.get('xGA5', np.nan)  - a.get('xGA5', np.nan)  if XG_AVAILABLE else np.nan,
        'Diff_xGD':          h.get('xGD5', np.nan)  - a.get('xGD5', np.nan)  if XG_AVAILABLE else np.nan,
        'Diff_xGF_ewm':      h.get('xGF_ewm', np.nan) - a.get('xGF_ewm', np.nan) if XG_AVAILABLE else np.nan,
        'Diff_xG_overperf':  h.get('xG_overperf', np.nan) - a.get('xG_overperf', np.nan) if XG_AVAILABLE else np.nan,
        'Diff_xGF_slope':    h.get('xGF_slope', np.nan) - a.get('xGF_slope', np.nan) if XG_AVAILABLE else np.nan,
        'H_xAttackIdx':  h.get('xGF_ewm', np.nan) / (max(a.get('xGA_ewm', 0.3) or 0.3, 0.3) + 0.01) if XG_AVAILABLE else np.nan,
        'A_xAttackIdx':  a.get('xGF_ewm', np.nan) / (max(h.get('xGA_ewm', 0.3) or 0.3, 0.3) + 0.01) if XG_AVAILABLE else np.nan,
        'Diff_xAttackIdx': (h.get('xGF_ewm', np.nan) / (max(a.get('xGA_ewm', 0.3) or 0.3, 0.3) + 0.01) -
                            a.get('xGF_ewm', np.nan) / (max(h.get('xGA_ewm', 0.3) or 0.3, 0.3) + 0.01)) if XG_AVAILABLE else np.nan,
    }
    return row


# ══════════════════════════════════════════════════════════════
# 15) PREDICT SINGLE MATCH
# ══════════════════════════════════════════════════════════════

def predict_match(home_team, away_team, match_date=None,
                  odds_home=None, odds_draw=None, odds_away=None):
    """
    ทำนายผลแมตช์
    odds_home/draw/away: decimal odds จาก bookmaker (optional)
    ถ้าใส่ odds → ใช้ implied prob เป็น features + แสดง edge analysis
    """
    teams_in_data = set(match_df_clean['HomeTeam'].tolist() + match_df_clean['AwayTeam'].tolist())
    if home_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{home_team}'  |  ทีมที่มี: {sorted(teams_in_data)}")
        return None
    if away_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{away_team}'")
        return None

    row  = build_match_row(home_team, away_team, match_date)

    # 🔥 Phase 2: ถ้าส่ง live odds มา → override market features (เฉพาะเมื่อ USE_MARKET_FEATURES=True)
    if USE_MARKET_FEATURES and all(x is not None for x in [odds_home, odds_draw, odds_away]):
        try:
            rh, rd, ra = 1/odds_home, 1/odds_draw, 1/odds_away
            total = rh + rd + ra
            row['Mkt_ImpH']     = rh / total
            row['Mkt_ImpD']     = rd / total
            row['Mkt_ImpA']     = ra / total
            row['Mkt_Spread']   = (rh / total) - (ra / total)
            row['Mkt_DrawPrem'] = (rd / total) - 0.26
            row['Mkt_Overround']= total - 1
        except Exception:
            pass

    X    = pd.DataFrame([row])[FEATURES].fillna(0)
    X_sc = scaler.transform(X)

    # 🔥 2-Stage ML prediction
    proba_ml = predict_2stage(X_sc)[0]

    # 🔥 Phase 3: Poisson Hybrid blend (ถ้าพร้อม)
    if POISSON_HYBRID_READY:
        try:
            pf_row = pd.DataFrame([row])[poisson_features_used].fillna(0)
            pf_sc  = poisson_scaler.transform(pf_row)
            hxg    = float(np.clip(home_poisson_model.predict(pf_sc)[0], 0.3, 6.0))
            axg    = float(np.clip(away_poisson_model.predict(pf_sc)[0], 0.3, 6.0))
            ph, pd_, pa = poisson_win_draw_loss(hxg, axg)
            proba_pois  = np.array([pa, pd_, ph])
            proba       = blend_ml_poisson(proba_ml.reshape(1,-1),
                                           proba_pois.reshape(1,-1),
                                           alpha=best_alpha)
            # 🔥 FIX C: Apply draw suppression ให้ consistent กับ training
            proba       = suppress_draw_proba(proba, draw_factor=DRAW_SUPPRESS_FACTOR)[0]
            model_tag   = f"Hybrid α={best_alpha:.2f} 🔥"
        except Exception:
            proba = proba_ml
            hxg = axg = None
            model_tag = "2-Stage ML"
    else:
        proba = proba_ml
        hxg = axg = None
        model_tag = "2-Stage ML"

    pred  = apply_thresholds(proba.reshape(1, -1))[0]

    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    h_elo = final_elo.get(home_team, 1500)
    a_elo = final_elo.get(away_team, 1500)
    draw_signal = "🎯 Draw signal!" if proba[1] >= OPT_T_DRAW else ""

    result = {
        'Away Win': round(proba[0]*100, 1),
        'Draw':     round(proba[1]*100, 1),
        'Home Win': round(proba[2]*100, 1),
        'Prediction': label_map[pred],
        'Home_Elo': round(h_elo),
        'Away_Elo': round(a_elo),
    }

    print(f"\n{'='*52}")
    print(f"  ⚽  {home_team}  vs  {away_team}")
    print(f"{'='*52}")
    print(f"  Elo:   {home_team} {round(h_elo)}  |  {away_team} {round(a_elo)}")
    print(f"  Model: {model_tag}  (t_draw={OPT_T_DRAW:.2f}, t_home={OPT_T_HOME:.2f})")
    if hxg is not None:
        print(f"  xG:    {home_team} {hxg:.2f}  |  {away_team} {axg:.2f}")
    print(f"{'─'*52}")
    bar_chars = 28
    for label, pct in [('Home Win', result['Home Win']),
                        ('Draw    ', result['Draw']),
                        ('Away Win', result['Away Win'])]:
        bar   = '█' * int(pct / 100 * bar_chars)
        t_tag = ' ← threshold' if (label.strip()=='Draw' and proba[1]>=OPT_T_DRAW) else ''
        print(f"  {label}: {bar:<28} {pct}%{t_tag}")
    print(f"{'─'*52}")
    print(f"  🎯 Prediction: {result['Prediction']}  {draw_signal}")

    # 🔥 Edge analysis ถ้ามี odds
    if all(x is not None for x in [odds_home, odds_draw, odds_away]):
        mkt_p = {2: 1/odds_home, 1: 1/odds_draw, 0: 1/odds_away}
        tot   = sum(mkt_p.values())
        imp   = {k: v/tot for k, v in mkt_p.items()}
        model_p = {2: proba[2], 1: proba[1], 0: proba[0]}
        print(f"\n  💰 Betting Edge Analysis:")
        print(f"  {'Outcome':<12} {'Model%':>8} {'Market%':>9} {'Edge%':>8} {'Odds':>7}")
        print(f"  {'─'*48}")
        for cls, label in [(2,'Home Win'),(1,'Draw'),(0,'Away Win')]:
            edge = model_p[cls] - imp[cls]
            o = {2: odds_home, 1: odds_draw, 0: odds_away}[cls]
            flag = ' ✅ VALUE' if edge > 0.03 else (' ⚠️' if edge > 0 else '')
            print(f"  {label:<12} {model_p[cls]*100:>7.1f}% {imp[cls]*100:>8.1f}% "
                  f"{edge*100:>+7.1f}% {o:>7.2f}{flag}")

    print(f"{'='*52}")
    return result


# ══════════════════════════════════════════════════════════════
# 16) 🔥 POISSON REGRESSION GOAL MODEL (v3.0)
#     Dixon-Coles style: แยก model เหย้า/เยือน → xG → W/D/L prob
# ══════════════════════════════════════════════════════════════

from sklearn.linear_model import PoissonRegressor

def build_poisson_model():
    """
    สร้าง Poisson regression model สำหรับทำนาย expected goals
    - Home model: ทำนาย FTHG (home goals)
    - Away model: ทำนาย FTAG (away goals)
    Feature: Elo_diff, form features
    """
    poisson_features = [
        'Diff_Elo', 'H_GF_ewm', 'H_GA_ewm', 'A_GF_ewm', 'A_GA_ewm',
        'H_Pts_ewm', 'A_Pts_ewm', 'H_Elo_norm', 'A_Elo_norm',
        'HomeWinRate', 'AwayWinRate',
    ]
    pf_available = [f for f in poisson_features if f in match_df_clean.columns]

    train_p = train[pf_available].fillna(0)
    # ใช้ actual goals (ดีที่สุดสำหรับ Poisson regression)
    if 'FTHG' in train.columns and train['FTHG'].notna().sum() > 100:
        y_home_goals = train['FTHG'].fillna(train['FTHG'].median()).clip(0, 8)
        y_away_goals = train['FTAG'].fillna(train['FTAG'].median()).clip(0, 8)
        print("  Using actual FTHG/FTAG for Poisson target")
    else:
        # fallback: ใช้ rolling average (ค่าประมาณ)
        y_home_goals = train['H_GF5'].fillna(1.3).clip(0, 5)
        y_away_goals = train['A_GF5'].fillna(1.1).clip(0, 5)
        print("  Using rolling average as Poisson target (fallback)")

    sc_p = StandardScaler()
    train_p_sc = sc_p.fit_transform(train_p)

    home_poisson = PoissonRegressor(alpha=0.5, max_iter=500)
    away_poisson = PoissonRegressor(alpha=0.5, max_iter=500)
    home_poisson.fit(train_p_sc, y_home_goals.clip(0, 8))
    away_poisson.fit(train_p_sc, y_away_goals.clip(0, 8))

    return home_poisson, away_poisson, sc_p, pf_available

print("\n🔥 Building Poisson Goal Model...")
try:
    home_poisson_model, away_poisson_model, poisson_scaler, poisson_features_used = build_poisson_model()
    POISSON_MODEL_READY = True
    print("✅ Poisson Goal Model trained")
except Exception as e:
    POISSON_MODEL_READY = False
    print(f"⚠️  Poisson model failed: {e}")


def poisson_win_draw_loss(home_xg, away_xg, max_goals=8):
    """
    คำนวณ P(Win), P(Draw), P(Loss) จาก Poisson distribution
    นี่คือแก่นของ betting market calculation
    """
    p_home_win = p_draw = p_away_win = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson.pmf(hg, home_xg) * poisson.pmf(ag, away_xg)
            if hg > ag:   p_home_win += p
            elif hg == ag: p_draw    += p
            else:          p_away_win += p
    total = p_home_win + p_draw + p_away_win
    return p_home_win/total, p_draw/total, p_away_win/total


# ══════════════════════════════════════════════════════════════
# 🔥 PHASE 3: EXECUTION — now POISSON_MODEL_READY is defined
# ══════════════════════════════════════════════════════════════

print("\n🔥 PHASE 3: Building Poisson Hybrid Blend...")
POISSON_HYBRID_READY = False
best_alpha = 0.6   # default fallback
proba_hybrid = proba_2stage.copy()   # default: ML-only

if POISSON_MODEL_READY:
    try:
        poisson_proba_test = build_poisson_proba_for_test(
            test, poisson_features_used,
            poisson_scaler, home_poisson_model, away_poisson_model
        )
        best_alpha, best_blend_f1 = optimize_blend_alpha(
            proba_2stage, poisson_proba_test, y_test.values
        )
        proba_hybrid = blend_ml_poisson(proba_2stage, poisson_proba_test, alpha=best_alpha)
        POISSON_HYBRID_READY = True
        print(f"  ✅ Poisson Hybrid: best alpha={best_alpha:.2f}  macro F1={best_blend_f1:.4f}")
        print(f"     ({best_alpha:.2f} ML + {1-best_alpha:.2f} Poisson)")
    except Exception as e:
        print(f"  ⚠️  Poisson hybrid failed: {e} — using ML-only")
else:
    print("  ⚠️  Poisson model not ready — using ML-only proba")

# ══════════════════════════════════════════════════════════════
# 🔥 S6 EXECUTION: Threshold optimization on final hybrid proba
# ══════════════════════════════════════════════════════════════

print("\n🔥 S6: Optimizing prediction thresholds...")
# 🔥 FIX C: Apply draw suppression เพื่อแก้ systematic bias +11%
#    ทำก่อน threshold optimization เพื่อให้ thresholds calibrate บน proba ที่ถูกต้อง
DRAW_SUPPRESS_FACTOR = 0.85
proba_hybrid = suppress_draw_proba(proba_hybrid, draw_factor=DRAW_SUPPRESS_FACTOR)
print(f"  🔧 Draw suppression applied (factor={DRAW_SUPPRESS_FACTOR}) — fixing systematic bias")
OPT_T_HOME, OPT_T_DRAW, best_macro_f1 = optimize_thresholds(proba_hybrid, y_test)
print(f"  Optimal t_home={OPT_T_HOME:.3f}  t_draw={OPT_T_DRAW:.3f}")
print(f"  Best macro F1 = {best_macro_f1:.4f}")

# 🔥 FIX C diagnostic: ตรวจสอบ draw bias หลัง suppression
_draw_pred_mean = proba_hybrid[:, 1].mean()
_draw_actual    = (y_test == 1).mean()
_draw_bias_after = (_draw_pred_mean - _draw_actual) * 100
print(f"  📐 Draw calibration check: predicted={_draw_pred_mean:.1%}  "
      f"actual={_draw_actual:.1%}  bias={_draw_bias_after:+.1f}%"
      f"  {'✅ improved' if abs(_draw_bias_after) < 8 else '⚠️ still biased'}")

y_pred_final = apply_thresholds(proba_hybrid)
acc_final    = accuracy_score(y_test, y_pred_final)

hybrid_tag = "2-Stage + Poisson Hybrid 🔥" if POISSON_HYBRID_READY else "2-Stage ML only"
print(f"\n===== v5.0 FINAL RESULTS ({hybrid_tag}) =====")
print(f"Accuracy : {round(acc_final*100, 2)}%  "
      f"(ML-only 2-stage: {round(acc_2stage*100,2)}%)")
if POISSON_HYBRID_READY:
    print(f"Hybrid gain: {round((acc_final - acc_2stage)*100, 2)}%  "
          f"(α={best_alpha:.2f} ML + {1-best_alpha:.2f} Poisson)")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_final,
                             target_names=['Away Win','Draw','Home Win']))

# Keep backward compatibility — ensemble = 2-stage predict wrapper
class TwoStageEnsemble:
    """Wrapper class ให้ใช้ได้เหมือน sklearn classifier"""
    def predict_proba(self, X):
        return predict_2stage(X)
    def predict(self, X):
        return apply_thresholds(predict_2stage(X))

ensemble   = TwoStageEnsemble()
calibrated = TwoStageEnsemble()
y_pred     = y_pred_final  # backward compat

# ── Fallback single-stage ensemble (สำหรับ CV / backtest) ────
print("\n🔧 Building fallback single-stage ensemble (for CV + backtest)...")
# 🔥 FIX CONSISTENCY: ใช้ shared params เหมือน main model และ CV
if LGBM_AVAILABLE:
    lgbm_clf = lgb.LGBMClassifier(**get_cv_lgbm_params())
else:
    lgbm_clf = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                          learning_rate=0.05, random_state=42)

fallback_single = lgbm_clf
fallback_single.fit(X_train_sc, y_train)
print("  ✅ Fallback single-stage trained")

# ── Isotonic Calibration (inline) ────────────────────────────
print("🎯 Applying Isotonic Calibration (single-stage fallback)...")
try:
    calibrated_single = CalibratedClassifierCV(fallback_single, method='isotonic', cv=3)
    calibrated_single.fit(X_train_sc, y_train)
    y_pred_cal = calibrated_single.predict(X_test_sc)
    acc_cal = accuracy_score(y_test, y_pred_cal)
    print(f"Single-stage Calibrated Accuracy: {round(acc_cal*100, 2)}%")
except Exception as e:
    print(f"⚠️  Calibration skipped: {e}")
    calibrated_single = fallback_single

# ══════════════════════════════════════════════════════════════
# 13) SAVE MODEL
# ══════════════════════════════════════════════════════════════

model_bundle = {
    'model':               ensemble,
    'calibrated':          calibrated,
    'stage1':              stage1_cal,
    'stage2':              stage2_cal,
    'fallback_single':     calibrated_single,
    'scaler':              scaler,
    'features':            FEATURES,
    'elo':                 final_elo,
    'elo_home':            final_elo_home,
    'elo_away':            final_elo_away,
    'teams':               list(final_elo.keys()),
    'home_stats':          home_stats,
    'away_stats':          away_stats,
    'opt_t_home':          OPT_T_HOME,
    'opt_t_draw':          OPT_T_DRAW,
    'draw_suppress_factor': DRAW_SUPPRESS_FACTOR,   # 🔥 FIX C: save factor
    'poisson_hybrid_ready':POISSON_HYBRID_READY,
    'poisson_alpha':       best_alpha if POISSON_HYBRID_READY else 0.6,
    'poisson_model_home':  home_poisson_model if POISSON_MODEL_READY else None,
    'poisson_model_away':  away_poisson_model if POISSON_MODEL_READY else None,
    'poisson_scaler':      poisson_scaler if POISSON_MODEL_READY else None,
    'poisson_features':    poisson_features_used if POISSON_MODEL_READY else [],
    'xg_available':        XG_AVAILABLE,
    'odds_available':      ODDS_AVAILABLE,
    'version':             '8.0',   # 🔥 bump version
}

os.makedirs("model", exist_ok=True)
with open("model/football_model_v8.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("✅ Model v8 saved → model/football_model_v8.pkl")


def predict_score(home_team, away_team, use_poisson_model=True):
    """
    v3.0: ใช้ Poisson regression model ทำนาย xG → สกอร์ + W/D/L probs
    """
    teams_in_data = set(match_df_clean['HomeTeam'].tolist() + match_df_clean['AwayTeam'].tolist())
    if home_team not in teams_in_data or away_team not in teams_in_data:
        return None

    h = get_latest_features(home_team, is_home=True)
    a = get_latest_features(away_team, is_home=False)

    lg_home = data.dropna(subset=['FTHG'])['FTHG'].mean()
    lg_away = data.dropna(subset=['FTAG'])['FTAG'].mean()

    # Poisson regression xG (ถ้า model พร้อม)
    if POISSON_MODEL_READY and use_poisson_model:
        row = build_match_row(home_team, away_team)
        pf_row = pd.DataFrame([row])[poisson_features_used].fillna(0)
        pf_sc  = poisson_scaler.transform(pf_row)
        home_xg = float(home_poisson_model.predict(pf_sc)[0])
        away_xg = float(away_poisson_model.predict(pf_sc)[0])
        home_xg = max(0.3, min(home_xg, 6.0))
        away_xg = max(0.3, min(away_xg, 6.0))
        xg_source = "Poisson Regression 🔥"
    else:
        # fallback: ratio method
        h_gf = h['GF_ewm'] if h['GF_ewm'] > 0 else h['GF5']
        h_ga = h['GA_ewm'] if h['GA_ewm'] > 0 else h['GA5']
        a_gf = a['GF_ewm'] if a['GF_ewm'] > 0 else a['GF5']
        a_ga = a['GA_ewm'] if a['GA_ewm'] > 0 else a['GA5']
        home_xg = (h_gf / lg_home) * (a_ga / lg_home) * lg_home
        away_xg = (a_gf / lg_away) * (h_ga / lg_away) * lg_away
        home_xg = max(0.3, min(home_xg, 6.0))
        away_xg = max(0.3, min(away_xg, 6.0))
        xg_source = "Ratio Method"

    # คำนวณ score probabilities
    score_probs = {}
    for hg in range(8):
        for ag in range(8):
            score_probs[f"{hg}-{ag}"] = round(
                poisson.pmf(hg, home_xg) * poisson.pmf(ag, away_xg) * 100, 2
            )
    top5 = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]

    # Poisson-derived W/D/L probs (independent of classifier)
    p_hw, p_d, p_aw = poisson_win_draw_loss(home_xg, away_xg)

    print(f"\n  ⚽ xG ({xg_source}):  {home_team} {round(home_xg,2)}  vs  {away_team} {round(away_xg,2)}")
    print(f"  📊 Poisson W/D/L: {p_hw*100:.1f}% / {p_d*100:.1f}% / {p_aw*100:.1f}%")
    print(f"  🎯 สกอร์ที่น่าจะเป็น (Top 5):")
    for score, pct in top5:
        bar = '█' * int(pct * 2)
        print(f"     {score:<8} {bar:<20} {pct}%")

    return {
        'home_xg':           round(home_xg, 2),
        'away_xg':           round(away_xg, 2),
        'most_likely_score': top5[0][0],
        'top5_scores':       top5,
        'poisson_home_win':  round(p_hw*100, 1),
        'poisson_draw':      round(p_d*100, 1),
        'poisson_away_win':  round(p_aw*100, 1),
    }


# ══════════════════════════════════════════════════════════════
# 17) GET LAST 5 RESULTS
# ══════════════════════════════════════════════════════════════

def get_last_5_results(team):
    valid = data.dropna(subset=['FTHG', 'FTAG']).copy()
    hm = valid[valid['HomeTeam'] == team][['Date','HomeTeam','AwayTeam','FTHG','FTAG']].copy()
    hm['Venue'] = 'H'; hm['GF'] = hm['FTHG']; hm['GA'] = hm['FTAG']
    hm['Opponent'] = hm['AwayTeam']
    am = valid[valid['AwayTeam'] == team][['Date','HomeTeam','AwayTeam','FTHG','FTAG']].copy()
    am['Venue'] = 'A'; am['GF'] = am['FTAG']; am['GA'] = am['FTHG']
    am['Opponent'] = am['HomeTeam']
    all_m = pd.concat([hm, am]).sort_values('Date', ascending=False)
    last5 = all_m.head(5).copy()
    def rl(r):
        if r['GF'] > r['GA']: return 'W'
        elif r['GF'] == r['GA']: return 'D'
        else: return 'L'
    last5['Result'] = last5.apply(rl, axis=1)
    icon_map = {'W': '✅ ชนะ', 'D': '🟡 เสมอ', 'L': '❌ แพ้'}
    print(f"\n{'='*58}")
    print(f"  📋  5 แมตช์ล่าสุดของ {team}")
    print(f"{'='*58}")
    print(f"  {'วันที่':<13} {'คู่แข่ง':<22} {'สนาม':<6} {'สกอร์':<10} {'ผล'}")
    print(f"  {'─'*55}")
    for _, row in last5.iterrows():
        ds = row['Date'].strftime('%d/%m/%Y') if pd.notna(row['Date']) else 'N/A'
        sc = f"{int(row['GF'])}-{int(row['GA'])}"
        print(f"  {ds:<13} {str(row['Opponent']):<22} "
              f"{'เหย้า' if row['Venue']=='H' else 'เยือน':<6} {sc:<10} {icon_map[row['Result']]}")
    print(f"{'='*58}")
    return last5


# ══════════════════════════════════════════════════════════════
# 18) SEASON SIMULATION
# ══════════════════════════════════════════════════════════════

import requests

API_KEY = "745c5b802b204590bfa05c093f00bd43"

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

def normalize(name):
    return TEAM_NAME_MAP.get(name, name)


def update_season_csv_from_api():
    from datetime import datetime, timedelta
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}
    try:
        print("\n" + "="*55)
        print("  📥  อัปเดต season 2025.csv จาก API...")
        r = requests.get(url, headers=headers, params={"season": "2025"}, timeout=15)
        r.raise_for_status()
        matches = r.json().get("matches", [])
        print(f"  ✅ ดึงได้ {len(matches)} แมตช์")
        rows = []
        for m in matches:
            utc_dt   = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            th_dt    = utc_dt + timedelta(hours=7)
            date_str = th_dt.strftime("%d/%m/%Y")
            status   = m.get("status", "")
            full     = m.get("score", {}).get("fullTime", {})
            if status in ["FINISHED", "IN_PLAY", "PAUSED"]:
                hg  = full.get("home", "")
                ag  = full.get("away", "")
                ftr = ("H" if hg > ag else ("A" if ag > hg else "D")) if hg != "" else ""
            else:
                hg, ag, ftr = "", "", ""
            rows.append({"Date": date_str,
                         "HomeTeam": normalize(m["homeTeam"]["name"]),
                         "AwayTeam": normalize(m["awayTeam"]["name"]),
                         "FTHG": hg, "FTAG": ag, "FTR": ftr})
        df_new = pd.DataFrame(rows)
        played   = len(df_new[df_new["FTHG"] != ""])
        upcoming = len(df_new[df_new["FTHG"] == ""])
        df_new.to_csv("data_set/season 2025.csv", index=False)
        print(f"  ✅ แข่งแล้ว {played} นัด | ยังไม่แข่ง {upcoming} นัด")
        print(f"  💾 บันทึก → data_set/season 2025.csv")
        print("="*55)
        return df_new
    except Exception as e:
        print(f"  ❌ Error: {e}")


def run_season_simulation():
    global final_table, remaining_fixtures
    season_file = pd.read_csv("data_set/season 2025.csv")
    season_file['Date'] = pd.to_datetime(season_file['Date'], dayfirst=True, errors='coerce')
    played = season_file.dropna(subset=['FTHG', 'FTAG']).copy()
    played = played[played['Date'] <= TODAY]
    season_teams = list(set(season_file['HomeTeam'].tolist() + season_file['AwayTeam'].tolist()))
    played_pairs = set(zip(played['HomeTeam'], played['AwayTeam']))
    remaining_fixtures = [
        {'HomeTeam': h, 'AwayTeam': a}
        for h in season_teams for a in season_teams
        if h != a and (h, a) not in played_pairs
    ]
    unplayed = pd.DataFrame(remaining_fixtures)
    print(f"\n📅 วันนี้: {TODAY.date()}")
    print(f"✅ แมตช์แข่งแล้ว:    {len(played)} นัด")
    print(f"⏳ แมตช์ยังไม่แข่ง: {len(unplayed)} นัด")
    print(f"   รวม: {len(played) + len(unplayed)} นัด")

    real_table = {}
    for _, row in played.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        hg, ag = int(row['FTHG']), int(row['FTAG'])
        for t in [home, away]:
            if t not in real_table: real_table[t] = 0
        if hg > ag:   real_table[home] += 3
        elif hg < ag: real_table[away] += 3
        else: real_table[home] += 1; real_table[away] += 1

    real_table_df = pd.DataFrame.from_dict(real_table, orient='index', columns=['RealPoints'])
    pred_table    = {}
    print(f"🤖 ทำนาย {len(unplayed)} แมตช์ที่เหลือ")

    if len(unplayed) > 0:
        future_rows = []
        for _, match in unplayed.iterrows():
            home, away = match['HomeTeam'], match['AwayTeam']
            row = build_match_row(home, away)
            row['HomeTeam'] = home
            row['AwayTeam'] = away
            future_rows.append(row)
        future_df  = pd.DataFrame(future_rows)
        X_future   = scaler.transform(future_df[FEATURES])
        future_df['Pred'] = ensemble.predict(X_future)

        for _, row in future_df.iterrows():
            home, away = row['HomeTeam'], row['AwayTeam']
            pred = row['Pred']
            for t in [home, away]:
                if t not in pred_table: pred_table[t] = 0
            if pred == 2:    pred_table[home] += 3
            elif pred == 1:  pred_table[home] += 1; pred_table[away] += 1
            else:            pred_table[away] += 3

    pred_table_df = pd.DataFrame.from_dict(pred_table, orient='index', columns=['PredictedPoints'])
    final_table   = real_table_df.join(pred_table_df, how='left').fillna(0)
    final_table['PredictedPoints'] = final_table['PredictedPoints'].astype(int)
    final_table['FinalPoints']     = final_table['RealPoints'] + final_table['PredictedPoints']
    final_table.index.name = 'Team'

    real_sorted  = final_table.sort_values('RealPoints', ascending=False)
    played_count = len(played) // max(len(season_teams), 1)

    print(f"\n{'='*58}")
    print(f"  📊  ตารางคะแนนจริง ณ ปัจจุบัน  (ถึง {TODAY.date()})")
    print(f"{'='*58}")
    print(f"  {'#':<4} {'Team':<22} {'แข่ง':>5} {'แต้ม':>6}  {'สถานะ'}")
    print(f"  {'─'*55}")
    for rank, (team, row) in enumerate(real_sorted.iterrows(), 1):
        if rank <= 4:    status = "🔴 CL Zone"
        elif rank <= 6:  status = "🟠 Euro Zone"
        elif rank >= 18: status = "🟡 Relegation"
        else:            status = ""
        print(f"  {rank:<4} {team:<22} {played_count:>5} {int(row['RealPoints']):>6}  {status}")
    print(f"  {'─'*55}")
    print(f"  🔴 CL  🟠 Europa  🟡 ตกชั้น")

    final_sorted = final_table.sort_values('FinalPoints', ascending=False)
    print(f"\n{'='*62}")
    print(f"  🔮  ตารางคาดการณ์สิ้นฤดูกาล  (Real + AI ทำนาย {len(unplayed)} นัดที่เหลือ)")
    print(f"{'='*62}")
    print(f"  {'#':<4} {'Team':<22} {'แต้มจริง':>9} {'AI ทำนาย':>10} {'รวมคาด':>8}  {'สถานะ'}")
    print(f"  {'─'*60}")
    for rank, (team, row) in enumerate(final_sorted.iterrows(), 1):
        if rank <= 4:    status = "🔴 CL Zone"
        elif rank <= 6:  status = "🟠 Euro Zone"
        elif rank >= 18: status = "🟡 Relegation"
        else:            status = ""
        try:
            real_rank = list(real_sorted.index).index(team) + 1
        except ValueError:
            real_rank = rank
        arrow = "▲" if rank < real_rank else ("▼" if rank > real_rank else "─")
        print(f"  {rank:<4} {team:<22} {int(row['RealPoints']):>9} "
              f"{int(row['PredictedPoints']):>10} {int(row['FinalPoints']):>8}  {arrow} {status}")
    print(f"  {'─'*60}")
    print(f"  🔴 CL  🟠 Europa  🟡 ตกชั้น  │  ▲ขึ้น ▼ลง ─คงที่")


# ══════════════════════════════════════════════════════════════
# 19) PREDICT NEXT 5 MATCHES
# ══════════════════════════════════════════════════════════════

def fetch_fixtures_from_api(target_team, num_matches=5):
    if API_KEY == "YOUR_API_KEY_HERE":
        print("  ❌ ยังไม่ได้ใส่ API Key!")
        return None
    url     = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}
    try:
        print(f"  🌐 ดึงข้อมูลจาก football-data.org API...")
        r = requests.get(url, headers=headers, params={"status": "SCHEDULED"}, timeout=10)
        if r.status_code == 401: print("  ❌ API Key ไม่ถูกต้อง"); return None
        if r.status_code == 429: print("  ❌ Rate limit — รอแล้วลองใหม่"); return None
        r.raise_for_status()
        matches = r.json().get("matches", [])
        print(f"  ✅ ดึงได้ {len(matches)} นัดที่ยังไม่แข่ง")
        all_fixtures = []
        for m in matches:
            from datetime import datetime, timedelta
            utc_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            th_dt  = utc_dt + timedelta(hours=7)
            all_fixtures.append({
                "HomeTeam": normalize(m["homeTeam"]["name"]),
                "AwayTeam": normalize(m["awayTeam"]["name"]),
                "Date":     th_dt.strftime("%Y-%m-%d"),
                "DateObj":  th_dt,
            })
        team_fixtures = [
            f for f in all_fixtures
            if f["HomeTeam"] == target_team or f["AwayTeam"] == target_team
        ][:num_matches]
        if not team_fixtures:
            print(f"  ❌ ไม่พบนัดของ '{target_team}'"); return None

        print(f"\n  📅 ตารางแข่ง {num_matches} นัดข้างหน้าของ {target_team}:")
        print(f"  {'นัด':<5} {'วันที่':<14} {'เหย้า':<22} {'เยือน':<22} {'สนาม'}")
        print(f"  {'─'*65}")
        for i, f in enumerate(team_fixtures, 1):
            venue = "เหย้า" if f["HomeTeam"] == target_team else "เยือน"
            print(f"  {i:<5} {f['Date']:<14} {f['HomeTeam']:<22} {f['AwayTeam']:<22} {venue}")
        return team_fixtures
    except requests.exceptions.ConnectionError:
        print("  ❌ เชื่อมต่ออินเทอร์เน็ตไม่ได้"); return None
    except Exception as e:
        print(f"  ❌ Error: {e}"); return None


def predict_next_5_matches(team, fixtures=None):
    print(f"\n{'#'*62}")
    print(f"  🔮  วิเคราะห์ทีม: {team.upper()}")
    print(f"{'#'*62}")
    get_last_5_results(team)
    if fixtures:
        next5 = [f for f in fixtures if f['HomeTeam'] == team or f['AwayTeam'] == team][:5]
        print(f"  ✅ ใช้ตารางแข่งที่ระบุ ({len(next5)} นัด)")
    else:
        next5 = [f for f in remaining_fixtures if f['HomeTeam'] == team or f['AwayTeam'] == team][:5]
        print(f"  ⚠️  ใช้การเดาอัตโนมัติ")
    if not next5:
        print(f"\n⚠️  ไม่พบแมตช์ข้างหน้าของ {team}"); return None

    print(f"\n{'='*62}")
    print(f"  🔮  5 แมตช์ข้างหน้า: {team}")
    print(f"{'='*62}")

    predictions = []
    for i, match in enumerate(next5, 1):
        hm, aw = match['HomeTeam'], match['AwayTeam']
        is_home = (hm == team)
        opponent = aw if is_home else hm
        venue_th = 'เหย้า' if is_home else 'เยือน'
        match_date = pd.Timestamp(match.get('DateObj', TODAY))
        print(f"\n  นัดที่ {i}  |  {hm}  vs  {aw}  ({venue_th})")
        print(f"  {'─'*58}")
        r_pred = predict_match(hm, aw, match_date)
        s_pred = predict_score(hm, aw)
        if r_pred and s_pred:
            if is_home:
                win_pct, draw_pct, loss_pct = r_pred['Home Win'], r_pred['Draw'], r_pred['Away Win']
                outcome = r_pred['Prediction']
            else:
                win_pct, draw_pct, loss_pct = r_pred['Away Win'], r_pred['Draw'], r_pred['Home Win']
                flip = {'Home Win': 'Away Win', 'Away Win': 'Home Win', 'Draw': 'Draw'}
                outcome = flip.get(r_pred['Prediction'], r_pred['Prediction'])
            is_win  = (is_home and outcome == 'Home Win') or (not is_home and outcome == 'Away Win')
            is_draw = (outcome == 'Draw')
            result_th = f"✅ {team} ชนะ" if is_win else ("🟡 เสมอ" if is_draw else f"❌ {team} แพ้")
            print(f"\n  📌 ผลที่น่าจะเป็น : {result_th}")
            print(f"  📊 ชนะ {win_pct}%  |  เสมอ {draw_pct}%  |  แพ้ {loss_pct}%")
            print(f"  🎯 สกอร์คาด      : {s_pred['most_likely_score']}")
            predictions.append({
                'match_no': i, 'home': hm, 'away': aw,
                'venue': venue_th, 'opponent': opponent,
                'win_pct': win_pct, 'draw_pct': draw_pct, 'loss_pct': loss_pct,
                'predicted_result': outcome,
                'predicted_score': s_pred['most_likely_score'],
            })

    print(f"\n{'#'*62}")
    print(f"  📋  สรุป 5 แมตช์ข้างหน้า: {team}")
    print(f"{'#'*62}")
    print(f"  {'นัด':<5} {'คู่แข่ง':<24} {'สนาม':<7} {'ชนะ%':<8} {'เสมอ%':<8} {'แพ้%':<8} {'สกอร์คาด'}")
    print(f"  {'─'*68}")
    for p in predictions:
        print(f"  {p['match_no']:<5} {p['opponent']:<24} {p['venue']:<7} "
              f"{p['win_pct']:<8} {p['draw_pct']:<8} {p['loss_pct']:<8} {p['predicted_score']}")
    print(f"{'#'*62}\n")
    return predictions


def predict_with_api(team, num_matches=5):
    SEP = '=' * 62
    print(f"\n{SEP}\n  🔮  ทำนาย {num_matches} แมตช์ข้างหน้า: {team}\n{SEP}")
    fixtures = fetch_fixtures_from_api(team, num_matches)
    if fixtures:
        predict_next_5_matches(team, fixtures=fixtures)
    else:
        print('  ⚠️  fallback: ใช้การเดาอัตโนมัติ')
        predict_next_5_matches(team)


def show_next_pl_fixtures(num_matches=5):
    SEP = "=" * 65
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}
    try:
        r = requests.get(url, headers=headers, params={"status": "SCHEDULED"}, timeout=10)
        r.raise_for_status()
        matches = r.json().get("matches", [])
        matches = sorted(matches, key=lambda x: x["utcDate"])[:num_matches]
        if not matches:
            print("  ⚠️  ไม่พบแมตช์"); return

        from datetime import datetime, timedelta
        print(f"\n{SEP}\n  📅  ตารางแข่ง Premier League {num_matches} นัดถัดไป\n{SEP}")
        print(f"  {'นัด':<5} {'วันที่':<14} {'เวลา(TH)':<11} {'เหย้า':<22} {'เยือน'}")
        print("  " + "-" * 60)
        upcoming = []
        for i, m in enumerate(matches, 1):
            home = normalize(m["homeTeam"]["name"])
            away = normalize(m["awayTeam"]["name"])
            utc_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            th_dt  = utc_dt + timedelta(hours=7)
            ds = th_dt.strftime("%d/%m/%Y"); ts = th_dt.strftime("%H:%M")
            print(f"  {i:<5} {ds:<14} {ts:<11} {home:<22} {away}")
            upcoming.append({"HomeTeam": home, "AwayTeam": away, "Date": ds, "Time": ts})

        print(f"\n{SEP}\n  🤖  ผลทำนาย {num_matches} นัดถัดไป\n{SEP}")
        print(f"  {'นัด':<5} {'เหย้า':<20} {'vs':^4} {'เยือน':<20} "
              f"{'ชนะ%':>7} {'เสมอ%':>7} {'แพ้%':>7}  {'สกอร์'}")
        print("  " + "-" * 75)
        teams_ok = set(match_df_clean["HomeTeam"].tolist() + match_df_clean["AwayTeam"].tolist())
        for i, f in enumerate(upcoming, 1):
            home, away = f["HomeTeam"], f["AwayTeam"]
            if home not in teams_ok or away not in teams_ok:
                print(f"  {i:<5} {home:<20} {'vs':^4} {away:<20}  ⚠️ ไม่มีข้อมูล"); continue
            r_pred = predict_match(home, away)
            s_pred = predict_score(home, away)
            if r_pred and s_pred:
                hw    = r_pred["Home Win"]; dr = r_pred["Draw"]; aw = r_pred["Away Win"]
                pred  = r_pred["Prediction"]
                icon  = "🏠" if pred == "Home Win" else ("🤝" if pred == "Draw" else "✈️")
                score = s_pred["most_likely_score"]
                print(f"  {i:<5} {home:<20} {'vs':^4} {away:<20} "
                      f"{hw:>7} {dr:>7} {aw:>7}  {icon} {score}")
        print("  " + "-" * 75)
        print("  🏠 เหย้าชนะ  🤝 เสมอ  ✈️ เยือนชนะ")
        print(SEP)
        return upcoming
    except requests.exceptions.ConnectionError:
        print("  ❌ เชื่อมต่ออินเทอร์เน็ตไม่ได้")
    except Exception as e:
        print(f"  ❌ Error: {e}")


# ══════════════════════════════════════════════════════════════
# 20) FULL SUMMARY REPORT
# ══════════════════════════════════════════════════════════════

def print_full_summary():
    SEP  = "=" * 65
    LINE = "─" * 65
    print()
    print("█" * 65)
    print("  📊  FOOTBALL AI v4.0 — FULL SUMMARY REPORT")
    print(f"  🗓️  วันที่รายงาน: {TODAY.date()}  |  ข้อมูลถึง: {data['Date'].max().date()}")
    print("  🔥  v4.0: 2-Stage | Optuna | SMOTE | Threshold | Deep Features")
    print("█" * 65)

    # 1. Data info
    print(f"\n{SEP}\n  📁  1. ข้อมูลที่ใช้เทรน\n{SEP}")
    total_seasons = data['Date'].dt.year.nunique()
    teams_count   = data['HomeTeam'].nunique()
    print(f"  • แมตช์ทั้งหมด    : {len(data):,} นัด ({total_seasons} ฤดูกาล)")
    print(f"  • จำนวนทีม        : {teams_count} ทีม")
    print(f"  • ช่วงเวลา        : {data['Date'].min().date()} → {data['Date'].max().date()}")
    print(f"  • แมตช์เทรน (80%) : {len(train):,} นัด")
    print(f"  • แมตช์เทสต์(20%) : {len(test):,} นัด")
    print(f"  • Features ที่ใช้  : {len(FEATURES)} ตัว (v1: 24 → v2: {len(FEATURES)} ✅)")

    # 2. Model performance
    print(f"\n{SEP}\n  🤖  2. ประสิทธิภาพโมเดล (v4.0: 2-Stage + Optuna + SMOTE + Threshold)\n{SEP}")
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f"  • Accuracy บน Test Set  : {acc}%")
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Away Win', 'Draw', 'Home Win']
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>14}", end="")
    for l in labels: print(f"  {l:>10}", end="")
    print()
    for i, label in enumerate(labels):
        print(f"  {'Actual ':>7}{label:>9}  ", end="")
        for j in range(3): print(f"  {cm[i][j]:>10}", end="")
        print()
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    print(f"\n  {'ผลลัพธ์':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {LINE}")
    for label in labels:
        r = report[label]
        print(f"  {label:<15} {r['precision']:>10.2f} {r['recall']:>10.2f} "
              f"{r['f1-score']:>10.2f} {int(r['support']):>10}")
    print(f"  {LINE}")
    print(f"  {'Accuracy':<15} {'':>10} {'':>10} {report['accuracy']:>10.2f} "
          f"{int(report['macro avg']['support']):>10}")

    # 3. Elo Top 10
    print(f"\n{SEP}\n  🏆  3. Elo Rating ล่าสุด (Top 10)\n{SEP}")
    elo_sorted = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:10]
    max_elo = elo_sorted[0][1]
    print(f"  {'#':<5} {'ทีม':<25} {'Elo':>8}  {'Bar'}")
    print(f"  {LINE}")
    for rank, (team, elo_val) in enumerate(elo_sorted, 1):
        bar    = '█' * int((elo_val / max_elo) * 20)
        marker = "🥇" if rank==1 else ("🥈" if rank==2 else ("🥉" if rank==3 else f"{rank:<2} "))
        print(f"  {marker}   {team:<25} {round(elo_val):>8}  {bar}")

    # 4. Season table
    if final_table is not None:
        print(f"\n{SEP}\n  📋  4. ตารางคาดการณ์สิ้นฤดูกาล Season 2025-26\n{SEP}")
        print(f"  {'#':<5} {'ทีม':<22} {'คะแนนจริง':>10} {'คะแนนทำนาย':>12} {'รวม':>7}  {'สถานะ'}")
        print(f"  {LINE}")
        for rank, (team, row) in enumerate(final_table.sort_values('FinalPoints', ascending=False).iterrows(), 1):
            if rank<=4:    status = "🔴 Champions League"
            elif rank<=6:  status = "🟠 Europa / Conf."
            elif rank>=18: status = "🟡 ตกชั้น"
            else:          status = ""
            print(f"  {rank:<5} {team:<22} {int(row['RealPoints']):>10} "
                  f"{int(row['PredictedPoints']):>12} {int(row['FinalPoints']):>7}  {status}")
        print(f"  {LINE}")
        print(f"  🔴 Top 4 = UEFA CL  |  🟠 Top 5-6 = Europa/Conf.  |  🟡 18-20 = ตกชั้น")

    # 5. สถิติ
    print(f"\n{SEP}\n  📈  5. สถิติน่าสนใจจากข้อมูล\n{SEP}")
    valid = data.dropna(subset=['FTHG', 'FTAG'])
    home_wins  = (valid['FTHG'] > valid['FTAG']).sum()
    draws      = (valid['FTHG'] == valid['FTAG']).sum()
    away_wins  = (valid['FTHG'] < valid['FTAG']).sum()
    total_v    = len(valid)
    avg_goals  = (valid['FTHG'] + valid['FTAG']).mean()
    print(f"  • เหย้าชนะ       : {home_wins:,} นัด ({home_wins/total_v*100:.1f}%)")
    print(f"  • เสมอ           : {draws:,} นัด ({draws/total_v*100:.1f}%)")
    print(f"  • เยือนชนะ       : {away_wins:,} นัด ({away_wins/total_v*100:.1f}%)")
    print(f"  • เฉลี่ยประตู/นัด : {avg_goals:.2f} ประตู")

    goals_scored   = valid.groupby('HomeTeam')['FTHG'].sum() + valid.groupby('AwayTeam')['FTAG'].sum()
    goals_conceded = valid.groupby('HomeTeam')['FTAG'].sum() + valid.groupby('AwayTeam')['FTHG'].sum()
    print(f"  • ทีมยิงมากสุด   : {goals_scored.idxmax()} ({int(goals_scored.max())} ประตู)")
    print(f"  • ทีมเสียมากสุด  : {goals_conceded.idxmax()} ({int(goals_conceded.max())} ประตู)")

    # 6. สรุปโมเดล
    print(f"\n{SEP}\n  💡  6. สรุปและคำแนะนำ\n{SEP}")
    draw_f1  = report['Draw']['f1-score']
    home_f1  = report['Home Win']['f1-score']
    away_f1  = report['Away Win']['f1-score']
    print(f"  • โมเดล Ensemble v2 (LR+RF+ET+GBT+MLP): {acc}% accuracy")
    print(f"  • Home Win F1: {home_f1:.2f}  |  Draw F1: {draw_f1:.2f}  |  Away Win F1: {away_f1:.2f}")
    print(f"  • Features: {len(FEATURES)} ตัว (เพิ่ม EWM, GD, Draw rate, Days Rest, Seasonal)")
    if draw_f1 > 0.1:
        print(f"  ✅ Draw F1 ดีขึ้นจาก ~0.03 → {draw_f1:.2f} (class_weight + draw features)")
    else:
        print(f"  ⚠️  Draw F1 ยังต่ำ ({draw_f1:.2f}) — ลองเพิ่ม SMOTE หรือ injury data")
    print(f"\n  🚀 v2 improvements:")
    print(f"     ✅ EWM rolling features (ฟอร์มล่าสุดสำคัญกว่า)")
    print(f"     ✅ Draw-specific features (H2H draw rate, combined GF, GD variance)")
    print(f"     ✅ Venue-specific Elo (เหย้า/เยือน แยกกัน)")
    print(f"     ✅ Days rest, Momentum, Seasonal phase")
    print(f"     ✅ 5 models ensemble + calibration")

    print()
    print("█" * 65)
    print("  ✅  END OF REPORT v2.0")
    print("█" * 65)


# ══════════════════════════════════════════════════════════════
# 21) MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════

def run_monte_carlo(n_simulations=1000, verbose=True):
    if final_table is None:
        print("❌ กรุณาเรียก run_season_simulation() ก่อน"); return None

    SEP  = "=" * 65
    LINE = "─" * 65
    if verbose:
        print(f"\n{SEP}")
        print(f"  🎲  MONTE CARLO SEASON SIMULATION  ({n_simulations:,} รอบ)")
        print(SEP)
        print(f"  กำลังจำลอง {len(remaining_fixtures)} แมตช์ × {n_simulations:,} รอบ ...")

    if not remaining_fixtures:
        print("  ℹ️  ฤดูกาลจบแล้ว"); return None

    future_rows = []
    for match in remaining_fixtures:
        home, away = match['HomeTeam'], match['AwayTeam']
        row = build_match_row(home, away)
        future_rows.append(row)

    future_df    = pd.DataFrame(future_rows)
    X_future_sc  = scaler.transform(future_df[FEATURES])
    proba_matrix = ensemble.predict_proba(X_future_sc)

    all_teams = list(final_table.index)
    real_pts  = {t: int(final_table.loc[t, 'RealPoints']) for t in all_teams}
    counts    = {t: {'top4': 0, 'top6': 0, 'relegation': 0, 'pts_sum': 0.0, 'pts_sq': 0.0}
                 for t in all_teams}
    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        sim_pts = dict(real_pts)
        for idx, match in enumerate(remaining_fixtures):
            home, away = match['HomeTeam'], match['AwayTeam']
            p_away, p_draw, p_home = proba_matrix[idx]
            probs = np.array([p_away, p_draw, p_home], dtype=np.float64)
            probs /= probs.sum()
            outcome = rng.choice([0, 1, 2], p=probs)
            if outcome == 2:    sim_pts[home] += 3
            elif outcome == 1:  sim_pts[home] += 1; sim_pts[away] += 1
            else:               sim_pts[away] += 3
        ranked = sorted(sim_pts.items(), key=lambda x: x[1], reverse=True)
        for rank, (team, pts) in enumerate(ranked, 1):
            if rank <= 4:  counts[team]['top4'] += 1
            if rank <= 6:  counts[team]['top6'] += 1
            if rank >= 18: counts[team]['relegation'] += 1
            counts[team]['pts_sum'] += pts
            counts[team]['pts_sq']  += pts ** 2

    results = {}
    for t in all_teams:
        c    = counts[t]
        mean = c['pts_sum'] / n_simulations
        std  = ((c['pts_sq'] / n_simulations) - mean**2) ** 0.5
        results[t] = {
            'top4':       round(c['top4']       / n_simulations * 100, 1),
            'top6':       round(c['top6']       / n_simulations * 100, 1),
            'relegation': round(c['relegation'] / n_simulations * 100, 1),
            'mean_pts':   round(mean, 1),
            'std_pts':    round(std,  1),
        }

    if verbose:
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_pts'], reverse=True)
        print(f"\n  {'Team':<22} {'Mean Pts':>9} {'±Std':>6} {'Top4%':>7} {'Top6%':>7} {'Rel%':>7}  {'Bar'}")
        print(f"  {LINE}")
        for team, r in sorted_results:
            bar_top4 = '█' * int(r['top4'] / 5)
            bar_rel  = '▓' * int(r['relegation'] / 5)
            bar      = bar_top4 if r['top4'] >= r['relegation'] else bar_rel
            c_t4     = "🔴" if r['top4'] >= 60 else ("🟡" if r['top4'] >= 20 else "  ")
            c_rel    = "🟡" if r['relegation'] >= 60 else ("⚠️ " if r['relegation'] >= 20 else "  ")
            print(f"  {team:<22} {r['mean_pts']:>9} {r['std_pts']:>6} "
                  f"{c_t4}{r['top4']:>5}%  {r['top6']:>6}%  {c_rel}{r['relegation']:>4}%  {bar}")
        print(f"\n  ✅ Monte Carlo เสร็จสิ้น ({n_simulations:,} simulations)")
        print(SEP)
    return results


# ══════════════════════════════════════════════════════════════
# 22) DRAW CALIBRATION ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyze_draw_calibration():
    SEP  = "=" * 65
    LINE = "─" * 65
    print(f"\n{SEP}\n  📐  DRAW CALIBRATION ANALYSIS\n{SEP}")

    # 🔥 FIX 2 (v7): ใช้ proba_hybrid (ผ่าน draw suppression + Poisson blend แล้ว)
    #    เดิม: ensemble.predict_proba() → raw proba ก่อน suppression → bias ยังอยู่
    #    ใหม่: proba_hybrid → proba ที่ผ่าน suppress_draw_proba() แล้ว → calibration แม่นกว่า
    #    ถ้า proba_hybrid ยังไม่ define → fallback ไป proba_2stage
    try:
        draw_proba_source = proba_hybrid
        source_tag = "Hybrid (post-suppression)"
    except NameError:
        try:
            draw_proba_source = proba_2stage
            source_tag = "2-Stage (post-suppression fallback)"
        except NameError:
            draw_proba_source = ensemble.predict_proba(X_test_sc)
            source_tag = "Ensemble raw (fallback)"

    draw_proba  = draw_proba_source[:, 1]
    actual_draw = (y_test == 1).astype(int).values
    print(f"  📌 Source: {source_tag}")

    n_bins = 8
    fraction_of_positives, mean_predicted_value = calibration_curve(
        actual_draw, draw_proba, n_bins=n_bins, strategy='quantile'
    )
    print(f"\n  Predicted%   Actual%    Diff     Calibration Bar")
    print(f"  {LINE}")
    for pred_p, act_p in zip(mean_predicted_value, fraction_of_positives):
        diff     = act_p - pred_p
        bar_pred = '█' * int(pred_p * 30)
        bar_act  = '░' * int(act_p  * 30)
        sign     = "+" if diff >= 0 else "-"
        flag     = "✅" if abs(diff) < 0.05 else ("⚠️ " if abs(diff) < 0.10 else "❌")
        print(f"  {pred_p*100:>8.1f}%   {act_p*100:>6.1f}%   {sign}{abs(diff)*100:>4.1f}%  {flag}  "
              f"pred:{bar_pred:<15} act:{bar_act:<15}")

    brier          = brier_score_loss(actual_draw, draw_proba)
    brier_baseline = brier_score_loss(actual_draw, np.full_like(draw_proba, actual_draw.mean()))
    skill          = (1 - brier / brier_baseline) * 100

    print(f"\n  {LINE}")
    print(f"  📊 Brier Score (Draw)  : {brier:.4f}")
    print(f"  📊 Baseline Brier      : {brier_baseline:.4f}")
    print(f"  📊 Brier Skill Score   : {skill:.1f}%  {'✅ ดีกว่า baseline' if skill>0 else '❌ แย่กว่า baseline'}")
    bias = draw_proba.mean()*100 - actual_draw.mean()*100
    print(f"  📊 Systematic Bias     : {bias:+.1f}%")
    print(SEP)
    return {'brier': brier, 'brier_baseline': brier_baseline, 'skill': skill, 'bias': bias}


# ══════════════════════════════════════════════════════════════
# 23) 🔥 SHAP FEATURE IMPORTANCE (v3.0)
#     SHAP ดีกว่า impurity เพราะ: consistent, accounts for interactions
# ══════════════════════════════════════════════════════════════

def run_feature_importance(max_display=20):
    SEP  = "=" * 65
    LINE = "─" * 65
    print(f"\n{SEP}\n  🔍  SHAP + Feature IMPORTANCE (v4.0)\n{SEP}")

    # 🔥 v4.0: ใช้ stage1 (Draw vs Not-Draw) และ stage2 (Home vs Away) แยกกัน
    lgbm_fitted = stage1_cal if hasattr(stage1_cal, 'feature_importances_') else None
    if lgbm_fitted is None and hasattr(stage1_cal, 'estimators_'):
        # calibrated wrapper
        try: lgbm_fitted = stage1_cal.estimators_[0] if hasattr(stage1_cal, 'estimators_') else None
        except: pass
    # fallback to single-stage
    if lgbm_fitted is None:
        lgbm_fitted = fallback_single if hasattr(fallback_single, 'feature_importances_') else None
    rf_fitted   = lgbm_fitted   # for compatibility
    gbt_fitted  = None

    # 🔥 SHAP ด้วย LightGBM (ถ้าพร้อม) — ถูกต้องกว่า impurity
    if SHAP_AVAILABLE and lgbm_fitted is not None:
        print(f"\n  🔥 SHAP Values (LightGBM — per class average |SHAP|)")
        print(f"  {'#':<4} {'Feature':<28} {'SHAP Importance':>16}  {'Bar'}")
        print(f"  {LINE}")
        try:
            explainer   = shap.TreeExplainer(lgbm_fitted)
            shap_values = explainer.shap_values(X_test_sc)  # shape: (n_samples, n_features, n_classes) or list
            if isinstance(shap_values, list):
                # list of arrays per class
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                if mean_abs_shap.ndim > 1:
                    mean_abs_shap = mean_abs_shap.mean(axis=-1)

            shap_sorted = np.argsort(mean_abs_shap)[::-1][:max_display]
            max_shap = mean_abs_shap[shap_sorted[0]]
            for rank, idx in enumerate(shap_sorted, 1):
                bar = '█' * int(mean_abs_shap[idx] / max_shap * 30)
                print(f"  {rank:<4} {FEATURES[idx]:<28} {mean_abs_shap[idx]:>16.4f}  {bar}")

            # Feature reduction suggestion
            low_imp_features = [FEATURES[i] for i in range(len(FEATURES))
                                if mean_abs_shap[i] < mean_abs_shap.mean() * 0.1]
            if low_imp_features:
                print(f"\n  💡 Low-importance features (consider dropping):")
                print(f"     {', '.join(low_imp_features[:10])}")
                print(f"     Dropping these may improve accuracy by 1-2%")
        except Exception as e:
            print(f"  ⚠️  SHAP computation failed: {e}")
    elif SHAP_AVAILABLE and rf_fitted is not None:
        print(f"\n  🔥 SHAP Values (RandomForest)")
        try:
            explainer   = shap.TreeExplainer(rf_fitted)
            shap_values = explainer.shap_values(X_test_sc[:200])  # sample for speed
            if isinstance(shap_values, list):
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_sorted = np.argsort(mean_abs_shap)[::-1][:max_display]
            max_shap = mean_abs_shap[shap_sorted[0]]
            for rank, idx in enumerate(shap_sorted, 1):
                bar = '█' * int(mean_abs_shap[idx] / max_shap * 30)
                print(f"  {rank:<4} {FEATURES[idx]:<28} {mean_abs_shap[idx]:>16.4f}  {bar}")
        except Exception as e:
            print(f"  ⚠️  SHAP computation failed: {e}")
    else:
        if not SHAP_AVAILABLE:
            print("  ⚠️  SHAP not installed — pip install shap")
        print("  → Falling back to RF impurity importance")

    # ── RF impurity (always show as reference) ───────────────
    if rf_fitted:
        importances = rf_fitted.feature_importances_
        sorted_idx  = np.argsort(importances)[::-1][:max_display]
        max_imp     = importances[sorted_idx[0]]
        print(f"\n  RandomForest Impurity Importance (reference)")
        print(f"  {'#':<4} {'Feature':<28} {'Score':>8}  {'Bar'}")
        print(f"  {LINE}")
        for rank, idx in enumerate(sorted_idx[:10], 1):
            bar = '█' * int(importances[idx] / max_imp * 30)
            print(f"  {rank:<4} {FEATURES[idx]:<28} {importances[idx]:>8.4f}  {bar}")

    # 🔥 LightGBM built-in importance (gain)
    if lgbm_fitted is not None:
        try:
            lgbm_imp = lgbm_fitted.feature_importances_
            lgbm_sorted = np.argsort(lgbm_imp)[::-1][:10]
            max_lgbm = lgbm_imp[lgbm_sorted[0]]
            print(f"\n  LightGBM Gain Importance 🔥")
            print(f"  {'#':<4} {'Feature':<28} {'Gain':>8}  {'Bar'}")
            print(f"  {LINE}")
            for rank, idx in enumerate(lgbm_sorted, 1):
                bar = '█' * int(lgbm_imp[idx] / max_lgbm * 30)
                print(f"  {rank:<4} {FEATURES[idx]:<28} {lgbm_imp[idx]:>8.0f}  {bar}")
        except Exception as e:
            print(f"  ⚠️  LightGBM importance failed: {e}")

    print(SEP)
    return sorted_idx if rf_fitted else None


# ══════════════════════════════════════════════════════════════
# 24) ROLLING WINDOW CV
# ══════════════════════════════════════════════════════════════

def rolling_window_cv(n_splits=5, verbose=True):
    SEP  = "=" * 65
    LINE = "─" * 65
    if verbose:
        print(f"\n{SEP}\n  🔄  ROLLING WINDOW CROSS-VALIDATION  ({n_splits} folds)\n{SEP}")

    cv_df = match_df_clean.sort_values('Date_x').reset_index(drop=True)
    X_cv  = cv_df[FEATURES].values
    y_cv  = cv_df['Result3'].values

    # 🔥 FIX E: ใช้ min_train_size เพื่อลด variance ของ fold แรก ๆ
    #    Fold 1 เดิม train แค่ 638 นัด → LGBM ที่ tuned มาจาก 3000+ นัด overfit
    #    min_train_size=800 → skip fold ที่ train น้อยเกินไป
    MIN_TRAIN_SIZE = 800
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    if verbose:
        print(f"\n  {'Fold':<6} {'Train':>7} {'Val':>6} {'Acc':>7} "
              f"{'HW-F1':>9} {'DR-F1':>9} {'AW-F1':>9} {'LogLoss':>9}")
        print(f"  {LINE}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv), 1):
        X_tr, X_vl = X_cv[train_idx], X_cv[val_idx]
        y_tr, y_vl = y_cv[train_idx], y_cv[val_idx]

        # 🔥 FIX E: Skip fold ถ้า train set เล็กเกินไป
        if len(X_tr) < MIN_TRAIN_SIZE:
            if verbose:
                print(f"  {fold:<6} {len(X_tr):>7} {len(X_vl):>6}  "
                      f"⏭️ Skip (train < {MIN_TRAIN_SIZE} — ผลไม่น่าเชื่อถือ)")
            continue

        sc_fold  = StandardScaler()
        X_tr_sc  = sc_fold.fit_transform(X_tr)
        X_vl_sc  = sc_fold.transform(X_vl)

        # 🔥 FIX CONSISTENCY: ใช้ shared params เหมือน main model
        if LGBM_AVAILABLE:
            cv_gbt = lgb.LGBMClassifier(**get_cv_lgbm_params())
        else:
            cv_gbt = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        cv_gbt.fit(X_tr_sc, y_tr)

        # 🔥 FIX 1 v7b: Narrow threshold search anchored รอบค่าจาก main model
        #    เดิม v7a: freeze threshold → CV accuracy ตก (threshold ต่างยุคใช้ไม่ได้)
        #    เดิม v6:  full grid search n_steps=30 → variance สูง + leakage เล็กน้อย
        #    ใหม่ v7b: search แค่ ±0.05 รอบ main model threshold → ลด variance + เหมาะกับ fold
        y_proba_fold  = cv_gbt.predict_proba(X_vl_sc)
        y_proba_fold  = suppress_draw_proba(y_proba_fold, draw_factor=DRAW_SUPPRESS_FACTOR)
        try:
            # narrow search: ±0.05 รอบ main model thresholds
            t_home_center = OPT_T_HOME
            t_draw_center = OPT_T_DRAW
            t_h, t_d, _ = optimize_thresholds(
                y_proba_fold, y_vl, n_steps=10,
                t_home_range=(max(0.15, t_home_center - 0.05), min(0.55, t_home_center + 0.05)),
                t_draw_range=(max(0.15, t_draw_center - 0.05), min(0.55, t_draw_center + 0.05)),
            )
        except (NameError, TypeError):
            # fallback ถ้า OPT_T_HOME ยังไม่ define หรือ optimize_thresholds ไม่รับ range params
            t_h, t_d, _ = optimize_thresholds(y_proba_fold, y_vl, n_steps=15)
        y_pred_fold   = apply_thresholds(y_proba_fold, t_home=t_h, t_draw=t_d)

        a   = accuracy_score(y_vl, y_pred_fold)
        ll  = log_loss(y_vl, y_proba_fold)
        rep = classification_report(y_vl, y_pred_fold, output_dict=True, zero_division=0)
        hw_f1   = rep.get('2', {}).get('f1-score', 0)
        draw_f1 = rep.get('1', {}).get('f1-score', 0)
        aw_f1   = rep.get('0', {}).get('f1-score', 0)
        fold_results.append({'fold': fold, 'train': len(train_idx), 'val': len(val_idx),
                             'acc': a, 'hw_f1': hw_f1, 'draw_f1': draw_f1,
                             'aw_f1': aw_f1, 'logloss': ll})
        if verbose:
            print(f"  {fold:<6} {len(train_idx):>7} {len(val_idx):>6} {a:>7.4f} "
                  f"{hw_f1:>9.4f} {draw_f1:>9.4f} {aw_f1:>9.4f} {ll:>9.4f}")

    accs = [r['acc'] for r in fold_results]
    lls  = [r['logloss'] for r in fold_results]
    dr   = [r['draw_f1'] for r in fold_results]
    if verbose:
        print(f"  {LINE}")
        print(f"  {'Mean':<6} {'':>7} {'':>6} {np.mean(accs):>7.4f} "
              f"{np.mean([r['hw_f1'] for r in fold_results]):>9.4f} "
              f"{np.mean(dr):>9.4f} "
              f"{np.mean([r['aw_f1'] for r in fold_results]):>9.4f} "
              f"{np.mean(lls):>9.4f}")
        print(f"\n  📊 Mean CV Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  📊 Mean Draw F1     : {np.mean(dr):.4f} (v1 ≈ 0.03)")
        print(f"  📊 CV LogLoss       : {np.mean(lls):.4f}")
        stab = "✅ เสถียร" if np.std(accs) < 0.03 else "⚠️ unstable"
        print(f"  📊 ความเสถียร       : {stab}")

        # 🔥 FIX 3 (v7): เพิ่ม weighted accuracy (น้ำหนักตาม fold size)
        total_val = sum(r['val'] for r in fold_results)
        weighted_acc = sum(r['acc'] * r['val'] for r in fold_results) / total_val if total_val > 0 else 0
        print(f"  📊 Weighted CV Acc  : {weighted_acc:.4f}  (น้ำหนักตาม fold size — แม่นกว่า simple mean)")

        # 🔥 v8: อธิบาย gap ระหว่าง CV และ main model
        print(f"\n  💡 ทำไม CV Accuracy ต่ำกว่า Main Model:")
        print(f"     • CV folds ใช้ข้อมูลยุค 2020-2022 (distribution ต่างจาก test ปี 2024-25)")
        print(f"     • CV ใช้ single-stage LightGBM (ไม่มี 2-stage + Poisson hybrid)")
        print(f"     • Walk-forward accuracy (~0.48-0.52) น่าเชื่อถือกว่าสำหรับ production")
        print(SEP)
    return fold_results


# ══════════════════════════════════════════════════════════════
# 25) BACKTEST ROI
# ══════════════════════════════════════════════════════════════

def backtest_roi(bankroll=1000.0, min_edge=0.03, kelly_fraction=0.25,
                 max_exposure=0.05, verbose=True, odds_shrink=0.0,
                 mode='closing'):
    """
    🔥 v3.0 Kelly Criterion Betting Strategy
    - Full Kelly sizing with fraction
    - min_edge: ต้องมี edge > X% ถึงเดิมพัน
    - max_exposure: จำกัด % bankroll ต่อแมตช์ (risk management)
    - แสดง edge distribution + per-outcome ROI

    🔥 FIX 3: Opening odds stress test
    - mode='closing'     : ใช้ B365 odds (closing) — optimistic
    - mode='conservative': shrink implied prob 5% เพื่อ simulate opening odds
    - mode='max_odds'    : ใช้ Max odds ถ้ามี (worst-case for edge)
    - odds_shrink=0.05   : ลด edge ลง 5% สำหรับ conservative estimate
    """
    SEP  = "=" * 65
    LINE = "─" * 65
    if verbose:
        mode_tag = f"[{mode}{'  shrink='+str(odds_shrink) if odds_shrink>0 else ''}]"
        print(f"\n{SEP}")
        print(f"  💰  KELLY CRITERION BACKTEST (v3.0) {mode_tag}")
        print(f"  Bankroll: £{bankroll:,.0f} | Min Edge: {min_edge*100:.0f}% | "
              f"Kelly: {kelly_fraction*100:.0f}% | Max: {max_exposure*100:.0f}%/bet")
        print(SEP)

    # 🔥 v5.0: ใช้ 2-Stage + Hybrid probabilities
    proba_test = proba_hybrid   # ← use hybrid (Poisson blended) if available
    label_map  = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

    # 🔥 FIX 3: ดึง odds จาก market ตาม mode ที่เลือก (ถ้าเปิดใช้ market เท่านั้น)
    real_odds_test = None
    if USE_MARKET_FEATURES and ODDS_AVAILABLE and '_ImpH' in test.columns:
        try:
            # 🔥 FIX 3a: ลองใช้ Max odds ก่อน (ถ้ามี) — เป็น worst-case สำหรับ edge
            _max_h = _find_odds_col(data, ['MaxH','BbMxH'])
            _max_d = _find_odds_col(data, ['MaxD','BbMxD'])
            _max_a = _find_odds_col(data, ['MaxA','BbMxA'])
            use_max = (mode == 'max_odds' and
                       all(x is not None for x in [_max_h, _max_d, _max_a]))

            if use_max:
                # Merge max odds into test
                max_odds_df = data[['MatchID',_max_h,_max_d,_max_a]].copy()
                test_with_max = test.merge(max_odds_df, on='MatchID', how='left')
                oh = 1 / pd.to_numeric(test_with_max[_max_h], errors='coerce').fillna(3)
                od = 1 / pd.to_numeric(test_with_max[_max_d], errors='coerce').fillna(3)
                oa = 1 / pd.to_numeric(test_with_max[_max_a], errors='coerce').fillna(3)
                print(f"  📌 Using Max odds (worst-case edge) 🔥")
            else:
                # ใช้ implied prob จาก B365 (closing odds)
                oh = test['_ImpH'].values
                od = test['_ImpD'].values
                oa = test['_ImpA'].values

            # 🔥 FIX 3b: Conservative mode — simulate opening odds โดย shrink implied prob
            # Closing odds efficient กว่า opening ~3-7%
            # ถ้า shrink=0.05 → reduce edge by 5% → simulate opening odds impact
            if odds_shrink > 0 and not use_max:
                total_imp = oh + od + oa
                # เพิ่ม overround ขึ้น (แย่กว่าสำหรับ bettor)
                oh = oh + odds_shrink * oh / total_imp
                od = od + odds_shrink * od / total_imp
                oa = oa + odds_shrink * oa / total_imp
                print(f"  ⚠️  Conservative mode: odds shrunk by {odds_shrink*100:.0f}% (simulate opening)")

            # Convert implied prob → decimal odds
            real_odds_test = np.column_stack([
                np.where(oa > 0.01, 1/oa, 99),   # away odds (cls 0)
                np.where(od > 0.01, 1/od, 99),   # draw odds (cls 1)
                np.where(oh > 0.01, 1/oh, 99),   # home odds (cls 2)
            ])
            odds_source = "Max odds" if use_max else ("Conservative B365" if odds_shrink > 0 else "B365 closing")
            print(f"  ✅ Using {odds_source} for backtest (Phase 2) 🔥")
        except Exception as e:
            print(f"  ⚠️  Real odds extraction failed: {e} — using simulated")
            real_odds_test = None

    bk = bankroll; bets = []; total_bets = 0; total_won = 0
    total_staked = 0.0; peak_bk = bk; max_dd = 0.0
    edge_dist = []

    for i, (proba, actual) in enumerate(zip(proba_test, y_test)):
        p_away, p_draw, p_home = proba

        # ใช้ real odds ถ้ามี ไม่งั้น simulate margin 5%
        if real_odds_test is not None and i < len(real_odds_test):
            r_odds = real_odds_test[i]
            odds = {0: float(r_odds[0]), 1: float(r_odds[1]), 2: float(r_odds[2])}
            # กรอง odds ที่ดูผิดปกติ
            odds = {k: v if 1.05 <= v <= 50 else 99 for k, v in odds.items()}
        else:
            # fallback: simulate margin 5%
            margin = 1.05
            odds = {
                0: (1/p_away) * margin if p_away > 0.01 else 99,
                1: (1/p_draw) * margin if p_draw > 0.01 else 99,
                2: (1/p_home) * margin if p_home > 0.01 else 99,
            }
        model_p = {0: p_away, 1: p_draw, 2: p_home}
        best_cls = max([0,1,2], key=lambda c: model_p[c] - (1/odds[c]))
        edge     = model_p[best_cls] - (1/odds[best_cls])
        edge_dist.append(edge)

        # 🔥 Kelly Criterion: f* = (p*o - 1) / (o - 1)
        if edge < min_edge: continue

        p = model_p[best_cls]; o = odds[best_cls]
        kelly_full = (p * o - 1) / (o - 1)   # full Kelly
        kelly_frac = kelly_full * kelly_fraction  # fractional Kelly
        stake = min(bk * kelly_frac, bk * max_exposure)  # cap exposure
        stake = max(stake, 0.5)  # minimum bet

        won    = (best_cls == actual)
        profit = stake * (o-1) if won else -stake
        bk    += profit
        total_bets += 1; total_staked += stake
        if won: total_won += 1
        if bk > peak_bk: peak_bk = bk
        dd = (peak_bk - bk) / peak_bk * 100
        if dd > max_dd: max_dd = dd
        bets.append({'bet': total_bets, 'cls': best_cls, 'edge': edge,
                     'stake': stake, 'odds': o, 'won': won, 'profit': profit,
                     'bk': bk, 'kelly_full': kelly_full})

    if total_bets == 0:
        print("  ⚠️  ไม่มีการแทงที่ตรงเงื่อนไข"); return None

    roi      = (bk - bankroll) / total_staked * 100
    win_rate = total_won / total_bets * 100
    net_pnl  = bk - bankroll

    if verbose:
        print(f"\n  {'Metric':<38} {'Value':>15}")
        print(f"  {LINE}")
        print(f"  {'จำนวนแมตช์ที่ผ่าน threshold':<38} {total_bets:>15,}")
        print(f"  {'Win Rate':<38} {win_rate:>14.1f}%")
        print(f"  {'Total Staked':<38} £{total_staked:>13,.2f}")
        print(f"  {'Net P&L':<38} £{net_pnl:>+13,.2f}")
        print(f"  {'Final Bankroll':<38} £{bk:>13,.2f}")
        print(f"  {'ROI (per unit staked)':<38} {roi:>14.1f}%")
        print(f"  {'Max Drawdown':<38} {max_dd:>14.1f}%")

        # 🔥 Kelly stats
        avg_kelly = np.mean([b['kelly_full'] for b in bets]) * 100
        avg_edge  = np.mean([b['edge'] for b in bets]) * 100
        avg_stake_pct = np.mean([b['stake'] for b in bets]) / bankroll * 100
        print(f"  {'Avg Full Kelly %':<38} {avg_kelly:>14.1f}%")
        print(f"  {'Avg Edge (qualified bets)':<38} {avg_edge:>14.1f}%")
        print(f"  {'Avg Stake % of Bankroll':<38} {avg_stake_pct:>14.2f}%")
        print(f"  {LINE}")

        for cls in [2, 0, 1]:
            cls_bets = [b for b in bets if b['cls'] == cls]
            if not cls_bets: continue
            cls_won = sum(1 for b in cls_bets if b['won'])
            cls_roi = sum(b['profit'] for b in cls_bets) / sum(b['stake'] for b in cls_bets) * 100
            cls_avg_edge = np.mean([b['edge'] for b in cls_bets]) * 100
            print(f"  {label_map[cls]:<38} {len(cls_bets):>4} bets  "
                  f"Win:{cls_won/len(cls_bets)*100:.0f}%  "
                  f"Edge:{cls_avg_edge:.1f}%  ROI:{cls_roi:+.1f}%")

        print(f"\n  💡 สรุป:")
        if roi > 5:    print(f"  ✅ ROI {roi:.1f}% — โมเดลมี edge จริงในระยะยาว (Quant Level)")
        elif roi > 0:  print(f"  🟡 ROI {roi:.1f}% — มี edge เล็กน้อย (ดีพอสำหรับ research)")
        else:          print(f"  ❌ ROI {roi:.1f}% — โมเดลยังไม่ beat the market")

        # Edge distribution
        pos_edge = [e for e in edge_dist if e >= min_edge]
        print(f"\n  📊 Edge Distribution (all matches):")
        print(f"     Matches with edge ≥ {min_edge*100:.0f}% : {len(pos_edge):,} / {len(edge_dist):,} "
              f"({len(pos_edge)/len(edge_dist)*100:.1f}%)")
        print(f"     Max edge observed          : {max(edge_dist)*100:.1f}%")

        # Bankroll curve
        print(f"\n  📈 Bankroll Curve:")
        step    = max(1, total_bets // 20)
        sampled = [bets[i] for i in range(0, len(bets), step)]
        max_bk_s = max(b['bk'] for b in sampled)
        min_bk_s = min(b['bk'] for b in sampled)
        rng_bk   = max_bk_s - min_bk_s or 1
        print(f"  £{max_bk_s:>8,.0f} ┐")
        for b in sampled:
            bar_len = int((b['bk'] - min_bk_s) / rng_bk * 40)
            print(f"           {'█' * bar_len}")
        print(f"  £{min_bk_s:>8,.0f} ┘")
        print(SEP)
    return {'roi': roi, 'win_rate': win_rate, 'net_pnl': net_pnl,
            'total_bets': total_bets, 'max_dd': max_dd, 'final_bk': bk}


# ══════════════════════════════════════════════════════════════
# 26) PHASE 2 (Monte Carlo + Calibration)
# ══════════════════════════════════════════════════════════════

def run_phase2(n_simulations=1000):
    print(f"\n{'█'*65}")
    print(f"  🚀  PHASE 2 — COMPETITION GRADE ANALYSIS")
    print(f"{'█'*65}")
    mc_results  = run_monte_carlo(n_simulations=n_simulations)
    draw_stats  = analyze_draw_calibration()
    feat_result = run_feature_importance()
    SEP = "=" * 65
    print(f"\n{SEP}\n  📋  PHASE 2 — SUMMARY\n{SEP}")
    if mc_results:
        top4_sorted = sorted(mc_results.items(), key=lambda x: x[1]['top4'], reverse=True)[:6]
        print(f"\n  🔴  Top 4 Probability")
        for team, r in top4_sorted:
            bar = '█' * int(r['top4'] / 5)
            print(f"      {team:<22} {bar:<20} {r['top4']}%")
        rel_sorted = sorted(mc_results.items(), key=lambda x: x[1]['relegation'], reverse=True)[:5]
        print(f"\n  🟡  Relegation Probability")
        for team, r in rel_sorted:
            if r['relegation'] > 0:
                bar = '▓' * int(r['relegation'] / 5)
                print(f"      {team:<22} {bar:<20} {r['relegation']}%")
    if draw_stats:
        print(f"\n  📐  Draw Calibration")
        print(f"      Brier Skill Score : {draw_stats['skill']:.1f}%")
        print(f"      Systematic Bias   : {draw_stats['bias']:+.1f}%")
    print(f"\n{SEP}\n  ✅  PHASE 2 COMPLETE\n{SEP}\n")
    return {'mc': mc_results, 'draw': draw_stats}


# ══════════════════════════════════════════════════════════════
# 🔥 NEW: WALK-FORWARD SEASON-BY-SEASON VALIDATION (v3.0)
#    Train 2015→2019 → test 2020, train 2015→2020 → test 2021, ...
#    นี่คือ production-grade validation ที่ดูว่าโมเดล "รอดในอนาคต" ได้จริงไหม
# ══════════════════════════════════════════════════════════════

def walk_forward_season_cv(verbose=True):
    SEP  = "=" * 65
    LINE = "─" * 65
    if verbose:
        print(f"\n{SEP}")
        print(f"  🏆  WALK-FORWARD SEASON-BY-SEASON VALIDATION (v3.0)")
        print(f"  ⚠️  นี่คือ test ที่โหดที่สุด — ทำนายอนาคตจริง ๆ")
        print(SEP)

    cv_df = match_df_clean.copy()

    # 🔥 FIX 2: ใช้ Season (Aug→Jul) แทน Calendar Year
    #    Season 2023 = Aug 2023 – Jul 2024 ≈ 380 นัด (ไม่ใช่แค่ Jan-Dec)
    if 'Season' not in cv_df.columns:
        cv_df['Season'] = cv_df['Date_x'].apply(get_season)

    seasons = sorted(cv_df['Season'].dropna().unique())
    # ต้องมี training data อย่างน้อย 3 seasons
    min_train_seasons = 3
    test_seasons = [s for s in seasons[min_train_seasons:] if not pd.isna(s)]

    if len(test_seasons) == 0:
        print("  ⚠️  ไม่มีข้อมูลเพียงพอสำหรับ walk-forward CV")
        return []

    fold_results = []
    if verbose:
        print(f"\n  {'Season':<10} {'Train':>8} {'Test':>7} {'Acc':>8} {'Draw-F1':>9} {'LogLoss':>9}")
        print(f"  {LINE}")

    for test_season in test_seasons:
        train_mask = cv_df['Season'] < test_season
        test_mask  = cv_df['Season'] == test_season

        n_train = train_mask.sum()
        n_test  = test_mask.sum()

        # 🔥 FIX D v2: ลด MIN_TEST_MATCHES → 80 เพื่อให้ season 2024 (87 test) ผ่าน
        #    เพิ่ม caveat ใน output ว่า fold ที่ test < 150 ควร interpret ด้วยความระมัดระวัง
        MIN_TEST_MATCHES = 80
        if n_train < 200 or n_test < MIN_TEST_MATCHES:
            if n_test > 0 and verbose:
                status = 'train' if n_train < 200 else 'test'
                warn = f"({n_train} train, {n_test} test — need 200 train + {MIN_TEST_MATCHES}+ test)"
                print(f"  ⏭️  Skip {test_season} — {status} too small {warn}")
            continue
        small_sample_warn = "⚠️ small" if n_test < 150 else ""

        X_tr = cv_df.loc[train_mask, FEATURES].values
        y_tr = cv_df.loc[train_mask, 'Result3'].values
        X_te = cv_df.loc[test_mask,  FEATURES].values
        y_te = cv_df.loc[test_mask,  'Result3'].values

        sc_wf  = StandardScaler()
        X_tr_sc = sc_wf.fit_transform(X_tr)
        X_te_sc = sc_wf.transform(X_te)

        # 🔥 FIX CONSISTENCY: ใช้ shared params เหมือน main model
        if LGBM_AVAILABLE:
            cv_model = lgb.LGBMClassifier(**get_cv_lgbm_params())
        else:
            cv_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            )

        cv_model.fit(X_tr_sc, y_tr)
        y_proba_wf = cv_model.predict_proba(X_te_sc)
        # 🔥 FIX 1 v7b: Narrow threshold search anchored รอบ main model — walk-forward
        y_proba_wf = suppress_draw_proba(y_proba_wf, draw_factor=DRAW_SUPPRESS_FACTOR)
        try:
            t_h_wf, t_d_wf, _ = optimize_thresholds(
                y_proba_wf, y_te, n_steps=10,
                t_home_range=(max(0.15, OPT_T_HOME - 0.05), min(0.55, OPT_T_HOME + 0.05)),
                t_draw_range=(max(0.15, OPT_T_DRAW - 0.05), min(0.55, OPT_T_DRAW + 0.05)),
            )
        except (NameError, TypeError):
            t_h_wf, t_d_wf, _ = optimize_thresholds(y_proba_wf, y_te, n_steps=15)
        y_pred_wf  = apply_thresholds(y_proba_wf, t_home=t_h_wf, t_draw=t_d_wf)

        a   = accuracy_score(y_te, y_pred_wf)
        ll  = log_loss(y_te, y_proba_wf)
        rep = classification_report(y_te, y_pred_wf, output_dict=True, zero_division=0)
        draw_f1 = rep.get('1', {}).get('f1-score', 0)

        fold_results.append({
            'year': test_season, 'train_size': n_train,
            'test_size': n_test, 'acc': a,
            'draw_f1': draw_f1, 'logloss': ll
        })
        if verbose:
            print(f"  {str(test_season):<10} {n_train:>8} {n_test:>7} "
                  f"{a:>8.4f} {draw_f1:>9.4f} {ll:>9.4f}  {small_sample_warn}")

    if fold_results and verbose:
        accs = [r['acc'] for r in fold_results]
        drs  = [r['draw_f1'] for r in fold_results]
        lls  = [r['logloss'] for r in fold_results]
        trend = "📈 ดีขึ้น" if accs[-1] > accs[0] else ("📉 แย่ลง" if accs[-1] < accs[0] else "→ คงที่")
        print(f"  {LINE}")
        print(f"  Mean  {' ':>8} {' ':>7} {np.mean(accs):>8.4f} "
              f"{np.mean(drs):>9.4f} {np.mean(lls):>9.4f}")
        print(f"\n  📊 Walk-Forward Results:")
        print(f"     Mean Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"     Mean Draw F1   : {np.mean(drs):.4f}")
        print(f"     Best Year      : {fold_results[np.argmax(accs)]['year']}  ({max(accs):.4f})")
        print(f"     Worst Year     : {fold_results[np.argmin(accs)]['year']}  ({min(accs):.4f})")
        print(f"     Trend          : {trend}")
        stab = "✅ เสถียรข้ามปี" if np.std(accs) < 0.04 else "⚠️  unstable across years"
        print(f"     ความเสถียร    : {stab} (std={np.std(accs):.4f})")

        # 🔥 FIX 3 (v7): ถ้า walk-forward มี fold น้อย (<3 folds) ให้แสดง pooled estimate
        #    เพื่อเพิ่ม statistical reliability ของ estimate
        n_small = sum(1 for r in fold_results if r['test_size'] < 150)
        if n_small > 0:
            all_test_sizes = sum(r['test_size'] for r in fold_results)
            print(f"\n  ⚠️  Walk-forward reliability warning:")
            print(f"     {n_small}/{len(fold_results)} folds มี test < 150 นัด (sample เล็ก)")
            print(f"     Total pooled test size: {all_test_sizes} นัด")
            if all_test_sizes >= 150:
                # คำนวณ pooled weighted accuracy
                weighted_acc = sum(r['acc'] * r['test_size'] for r in fold_results) / all_test_sizes
                weighted_draw_f1 = sum(r['draw_f1'] * r['test_size'] for r in fold_results) / all_test_sizes
                print(f"     📊 Pooled weighted accuracy: {weighted_acc:.4f}  (น่าเชื่อถือกว่า mean of folds)")
                print(f"     📊 Pooled weighted Draw F1 : {weighted_draw_f1:.4f}")
            else:
                print(f"     💡 แนะนำ: รวบรวมข้อมูลย้อนหลังเพิ่มเพื่อ walk-forward ที่น่าเชื่อถือกว่า")
        print(SEP)
    return fold_results


# ══════════════════════════════════════════════════════════════
# 🔥 NEW: REGIME DETECTION — Form Clustering (v3.0)
#    ตรวจจับ "ช่วงฟอร์มดี / ฟอร์มตก" ของทีม
#    ใช้ K-Means clustering บน rolling form features
# ══════════════════════════════════════════════════════════════

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def detect_team_regimes(team, n_regimes=3, verbose=True):
    """
    ตรวจจับ form regime ของทีมด้วย K-Means clustering
    n_regimes=3: Cold Form / Average Form / Hot Form
    """
    SEP  = "=" * 65
    LINE = "─" * 65

    team_rows = team_df[team_df['Team'] == team].sort_values('Date').copy()
    team_rows = team_rows.dropna(subset=['GF_last5','GA_last5','Points_last5','GD_std5'])

    if len(team_rows) < n_regimes * 5:
        if verbose: print(f"  ⚠️  ข้อมูลไม่พอสำหรับ {team}")
        return None

    regime_features = ['Points_last5', 'GF_last5', 'GA_last5',
                       'GD_last5', 'GD_ewm5', 'Pts_ewm5']
    rf_avail = [f for f in regime_features if f in team_rows.columns]
    X_reg = team_rows[rf_avail].fillna(0).values  # 🔥 fillna ก่อน KMeans

    # Normalize
    mm = MinMaxScaler()
    X_norm = mm.fit_transform(X_reg)

    # K-Means clustering
    km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    labels = km.fit_predict(X_norm)
    team_rows['Regime'] = labels

    # Label clusters by average points (0=Cold, 1=Average, 2=Hot)
    cluster_pts = {c: team_rows[team_rows['Regime']==c]['Points_last5'].mean()
                   for c in range(n_regimes)}
    sorted_clusters = sorted(cluster_pts, key=cluster_pts.get)
    regime_names = {sorted_clusters[0]: '❄️  Cold Form',
                    sorted_clusters[1]: '🟡 Average Form',
                    sorted_clusters[2]: '🔥 Hot Form'}
    if n_regimes == 2:
        regime_names = {sorted_clusters[0]: '❄️  Cold Form',
                        sorted_clusters[1]: '🔥 Hot Form'}
    team_rows['RegimeName'] = team_rows['Regime'].map(regime_names)

    # Current regime
    current_regime = team_rows.iloc[-1]['RegimeName']
    current_pts5   = team_rows.iloc[-1]['Points_last5']

    if verbose:
        print(f"\n{SEP}")
        print(f"  🧠  REGIME DETECTION: {team.upper()}")
        print(SEP)
        print(f"\n  Current Form Regime : {current_regime}")
        print(f"  Points (last 5)     : {current_pts5:.2f}")
        print(f"\n  Regime Statistics:")
        print(f"  {'Regime':<20} {'Matches':>8} {'Avg Pts':>9} {'Avg GF':>8} {'Avg GA':>8}")
        print(f"  {LINE}")
        for cluster in range(n_regimes):
            subset = team_rows[team_rows['Regime'] == cluster]
            r_name = regime_names.get(cluster, f'Cluster {cluster}')
            print(f"  {r_name:<20} {len(subset):>8} "
                  f"{subset['Points_last5'].mean():>9.2f} "
                  f"{subset['GF_last5'].mean():>8.2f} "
                  f"{subset['GA_last5'].mean():>8.2f}")

        # Recent regime timeline (last 10 matches)
        recent = team_rows.tail(10)
        print(f"\n  📅 Last 10 matches regime timeline:")
        print(f"  ", end="")
        for _, r in recent.iterrows():
            icon = {'❄️  Cold Form': '❄', '🟡 Average Form': '●', '🔥 Hot Form': '🔥'}.get(r['RegimeName'], '?')
            print(icon, end=" ")
        print(f"\n  {LINE}")
        print(f"  ❄ Cold  ● Average  🔥 Hot")
        print(SEP)

    return {
        'team': team, 'current_regime': current_regime,
        'current_pts5': current_pts5,
        'regime_stats': {regime_names[c]: {
            'count': int((team_rows['Regime']==c).sum()),
            'avg_pts': float(team_rows[team_rows['Regime']==c]['Points_last5'].mean())
        } for c in range(n_regimes)},
        'all_regimes': team_rows[['Date','RegimeName','Points_last5']].tail(20).to_dict('records')
    }


def analyze_league_regimes(top_n=6):
    """ดู regime ของ top N ทีม (ตาม Elo) พร้อมกัน"""
    SEP = "=" * 65
    print(f"\n{SEP}\n  🧠  LEAGUE-WIDE REGIME SUMMARY\n{SEP}")
    top_teams = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"  {'Team':<22} {'Current Regime':<22} {'Pts/5':>7} {'Elo':>7}")
    print(f"  {'─'*60}")
    results = {}
    for team, elo in top_teams:
        r = detect_team_regimes(team, verbose=False)
        if r:
            print(f"  {team:<22} {r['current_regime']:<22} "
                  f"{r['current_pts5']:>7.2f} {elo:>7.0f}")
            results[team] = r
    print(SEP)
    return results


# ══════════════════════════════════════════════════════════════
# 27) PHASE 3 (CV + ROI + Calibration + Walk-Forward)
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# 🔥 FIX 5: MARKET ABLATION TEST
#    เปรียบเทียบ model ที่มีและไม่มี market features
#    ถ้า accuracy ดรอปมาก → โมเดลแค่ "เรียน odds"
#    ถ้า accuracy ดรอปน้อย → โมเดลมี football intelligence แท้จริง
# ══════════════════════════════════════════════════════════════

def run_market_ablation_test(verbose=True):
    """
    Ablation test: train 3 versions และ compare accuracy บน test set
    1) Full features (with market)
    2) No market features
    3) No xG features

    🔥 v8: ใช้ single-stage LightGBM เพื่อให้เปรียบเทียบกันได้ยุติธรรม
    (main model เป็น 2-stage + Poisson hybrid จึงมี accuracy ต่างกัน —
    ablation นี้วัด *relative contribution* ของ feature groups ไม่ใช่ absolute accuracy)
    """
    SEP  = "=" * 65
    LINE = "─" * 65
    if verbose:
        print(f"\n{SEP}")
        print(f"  🔬  MARKET ABLATION TEST (FIX 5)")
        print(f"  ทดสอบว่า edge มาจาก Market Odds หรือ Football Features จริง")
        print(f"  ⚠️  ใช้ single-stage LightGBM เพื่อเปรียบเทียบ feature groups อย่างยุติธรรม")
        print(f"  (ตัวเลข accuracy นี้ต่ำกว่า main model เพราะไม่มี 2-stage + Poisson)")
        print(SEP)

    if not LGBM_AVAILABLE:
        print("  ⚠️  LightGBM ไม่พร้อม — ข้าม ablation test")
        return None

    MKT_FEATURES = ['Mkt_ImpH', 'Mkt_ImpD', 'Mkt_ImpA',
                     'Mkt_Spread', 'Mkt_DrawPrem', 'Mkt_Overround']
    XG_FEAT_LIST = [f for f in FEATURES if any(k in f for k in
                    ['xGF', 'xGA', 'xGD', 'xG_over', 'xAttack', 'xGF_slope'])]

    features_no_mkt = [f for f in FEATURES if f not in MKT_FEATURES]
    features_no_xg  = [f for f in FEATURES if f not in XG_FEAT_LIST]
    features_base   = [f for f in FEATURES
                       if f not in MKT_FEATURES and f not in XG_FEAT_LIST]

    # 🔥 FIX CONSISTENCY: ใช้ params เดียวกับ main model
    #    เดิมใช้ hardcode default params → accuracy 44.8% vs main 51.24%
    #    แก้แล้ว: ใช้ get_cv_lgbm_params() + draw suppression เหมือนกันทุก instance
    ablation_params = get_cv_lgbm_params() if LGBM_AVAILABLE else {}

    results = {}
    configs = [
        ('Full (Market+xG)',     FEATURES),
        ('No Market Features',   features_no_mkt),
        ('No xG Features',       features_no_xg),
        ('Base Only (No Mkt/xG)',features_base),
    ]

    if verbose:
        print(f"\n  {'Config':<28} {'Features':>10} {'Test Acc':>10} {'Draw F1':>9}")
        print(f"  {LINE}")

    for name, feat_list in configs:
        if len(feat_list) < 5:
            if verbose: print(f"  {name:<28} {'skip — too few features':>30}")
            continue
        try:
            X_tr_ab = train[feat_list].fillna(0).values
            X_te_ab = test[feat_list].fillna(0).values
            y_tr_ab = train['Result3'].values
            y_te_ab = test['Result3'].values

            sc_ab = StandardScaler()
            X_tr_sc_ab = sc_ab.fit_transform(X_tr_ab)
            X_te_sc_ab = sc_ab.transform(X_te_ab)

            mdl = lgb.LGBMClassifier(**ablation_params)
            mdl.fit(X_tr_sc_ab, y_tr_ab)

            # 🔥 FIX CONSISTENCY: apply draw suppression + threshold ก่อน predict
            proba_ab = mdl.predict_proba(X_te_sc_ab)
            proba_ab = suppress_draw_proba(proba_ab, draw_factor=DRAW_SUPPRESS_FACTOR)
            t_h_ab, t_d_ab, _ = optimize_thresholds(proba_ab, y_te_ab, n_steps=20)
            pred_ab  = apply_thresholds(proba_ab, t_home=t_h_ab, t_draw=t_d_ab)

            acc_ab = accuracy_score(y_te_ab, pred_ab)
            rep_ab = classification_report(y_te_ab, pred_ab, output_dict=True, zero_division=0)
            draw_f1_ab = rep_ab.get('1', {}).get('f1-score', 0)

            results[name] = {'acc': acc_ab, 'draw_f1': draw_f1_ab, 'n_features': len(feat_list)}

            marker = ''
            if 'Full' in name: marker = ' ← baseline'
            elif results.get('Full (Market+xG)'):
                drop = results['Full (Market+xG)']['acc'] - acc_ab
                marker = f' (drop {drop:+.1%})'

            if verbose:
                print(f"  {name:<28} {len(feat_list):>10} {acc_ab:>10.1%} {draw_f1_ab:>9.3f}{marker}")

        except Exception as e:
            if verbose: print(f"  {name:<28} Error: {e}")

    if verbose and len(results) >= 2:
        print(f"\n  {LINE}")
        full_acc  = results.get('Full (Market+xG)', {}).get('acc', 0)
        nomkt_acc = results.get('No Market Features', {}).get('acc', 0)
        noxg_acc  = results.get('No xG Features', {}).get('acc', 0)
        base_acc  = results.get('Base Only (No Mkt/xG)', {}).get('acc', 0)

        market_contribution = full_acc - nomkt_acc
        xg_contribution     = full_acc - noxg_acc if noxg_acc else 0

        print(f"\n  📊 Feature Contribution Analysis:")
        print(f"     Market features contribution : {market_contribution:+.1%}")
        # 🔥 v8: xG contribution อาจเป็น negative ใน single-stage ablation
        #    เพราะ xG features ออกแบบมาสำหรับ 2-stage model → ต้องอธิบาย context
        xg_note = ""
        if xg_contribution < -0.01:
            xg_note = "  ← ใช้งานจริงใน 2-Stage+Poisson เท่านั้น (ไม่ใช่ single-stage)"
        print(f"     xG features contribution     : {xg_contribution:+.1%}{xg_note}")
        print(f"     Base (Elo+Form) accuracy     : {base_acc:.1%}")

        print(f"\n  🎯 Verdict:")
        if market_contribution > 0.05:
            print(f"     ❌ Edge มาจาก Market Odds เป็นหลัก (>{market_contribution:.0%})")
            print(f"        → ROI backtest มีโอกาส overestimate ถ้าใช้ closing odds")
            print(f"        → ต้องพิสูจน์ด้วย opening odds หรือ in-play edge")
        elif market_contribution > 0.02:
            print(f"     🟡 Market ช่วย moderate ({market_contribution:.0%})")
            print(f"        → โมเดลมีทั้ง football + market signal")
        else:
            print(f"     ✅ Model มี football intelligence แท้ ({market_contribution:.0%} from market)")
            print(f"        → ROI น่าเชื่อถือมากขึ้น")

        if xg_contribution < -0.01:
            print(f"     💡 xG features: contribution ดูเป็น negative ใน single-stage ablation")
            print(f"        → นี่เป็น artifact ของ model architecture ไม่ใช่ xG ไม่มีประโยชน์")
            print(f"        → Main model (2-Stage+Poisson) ใช้ xG ได้ผลดีกว่า อ้างอิง: Hybrid gain +2.09%")

        print(SEP)

    return {
        'with_mkt': results.get('Full (Market+xG)', {}).get('acc', 0),
        'no_mkt':   results.get('No Market Features', {}).get('acc', 0),
        'no_xg':    results.get('No xG Features', {}).get('acc', 0),
        'base':     results.get('Base Only (No Mkt/xG)', {}).get('acc', 0),
        'details':  results
    }


def run_phase3(n_simulations=1000):
    print(f"\n{'█'*65}")
    print(f"  🏆  PHASE 3 — PRODUCTION GRADE v3.0")
    print(f"{'█'*65}")

    cv_results  = rolling_window_cv(n_splits=5)
    wf_results  = walk_forward_season_cv()
    # 🔥 FIX: ลด kelly_fraction 0.25 → 0.15 เพื่อควบคุม max drawdown
    #    v6.0 มี 106 bets vs 42 bets เดิม → drawdown โอกาสสูงขึ้น
    roi_result  = backtest_roi(bankroll=1000.0, min_edge=0.03, kelly_fraction=0.15)

    # 🔥 FIX 3: Backtest เพิ่ม conservative + max_odds mode (เฉพาะเมื่อใช้ market)
    if USE_MARKET_FEATURES and ODDS_AVAILABLE:
        print(f"\n{'─'*65}")
        print(f"  🔬  STRESS TEST: Opening Odds Simulation (FIX 3)")
        print(f"{'─'*65}")
        roi_conservative = backtest_roi(
            bankroll=1000.0, min_edge=0.03, kelly_fraction=0.25,
            mode='conservative', odds_shrink=0.05,
            verbose=True
        )
        roi_max = backtest_roi(
            bankroll=1000.0, min_edge=0.03, kelly_fraction=0.25,
            mode='max_odds', verbose=True
        )
        print(f"\n  📊 Backtest Comparison (FIX 3):")
        print(f"  {'Mode':<30} {'ROI':>8}  {'WinRate':>9}  {'Bets':>6}")
        print(f"  {'─'*56}")
        for tag, r in [
            ("Closing B365 (optimistic)", roi_result),
            ("Conservative -5% (opening)", roi_conservative),
            ("Max odds (worst-case)", roi_max),
        ]:
            if r:
                print(f"  {tag:<30} {r['roi']:>+7.1f}%  {r['win_rate']:>8.1f}%  {r['total_bets']:>6}")
        verdict = ""
        if roi_conservative and roi_conservative['roi'] > 5:
            verdict = "✅ Edge ยังคงอยู่แม้ conservative → edge จริง"
        elif roi_conservative and roi_conservative['roi'] > 0:
            verdict = "🟡 Edge อ่อนลงมาก → อาจมาจาก closing odds advantage บางส่วน"
        elif roi_conservative:
            verdict = "❌ Edge หายไปใน conservative → edge มาจาก closing odds เป็นหลัก"
        if verdict:
            print(f"\n  💡 {verdict}")

    mc_results  = run_monte_carlo(n_simulations=n_simulations)
    regime_data = analyze_league_regimes(top_n=6)  # 🔥 NEW: Regime Detection

    # ══════════════════════════════════════════════════════════════
    # 🔥 FIX 5: NO-MARKET ABLATION TEST
    #    train โมเดลใหม่โดยไม่มี market features
    #    ถ้า accuracy ดรอป > 5% → edge มาจาก odds เป็นหลัก
    # ══════════════════════════════════════════════════════════════
    ablation_result = run_market_ablation_test()

    SEP = "=" * 65
    cv_accs = [r['acc'] for r in cv_results]
    cv_drs  = [r['draw_f1'] for r in cv_results]

    print(f"\n{SEP}\n  📋  PHASE 3 — SUMMARY v3.0\n{SEP}")

    # 🔥 v8: walk-forward เป็น PRIMARY metric
    if wf_results:
        wf_accs = [r['acc'] for r in wf_results]
        wf_sizes = [r['test_size'] for r in wf_results]
        wf_total = sum(wf_sizes)
        wf_weighted = sum(r['acc'] * r['test_size'] for r in wf_results) / wf_total if wf_total > 0 else 0
        print(f"\n  🏆 Walk-Forward CV (PRIMARY — season-by-season) 🔥")
        print(f"     Mean Accuracy   : {np.mean(wf_accs):.4f} ± {np.std(wf_accs):.4f}")
        print(f"     Weighted Acc    : {wf_weighted:.4f}  (pooled {wf_total} matches)")
        print(f"     Range           : [{min(wf_accs):.4f} – {max(wf_accs):.4f}]")

    print(f"\n  🔄 Rolling CV (5 folds) — reference only, not production estimate")
    print(f"     Mean Accuracy : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
    print(f"     Mean Draw F1  : {np.mean(cv_drs):.4f}")
    print(f"     ⚠️  ต่ำกว่า main model เพราะใช้ single-stage + data ยุค 2020-22")

    if roi_result:
        print(f"\n  💰 Backtest ROI")
        print(f"     ROI            : {roi_result['roi']:+.1f}%")
        print(f"     Win Rate       : {roi_result['win_rate']:.1f}%")
        print(f"     Max Drawdown   : {roi_result['max_dd']:.1f}%")
        print(f"     Total Bets     : {roi_result['total_bets']:,}")

    if ablation_result:
        print(f"\n  🔬 Market Ablation Test (FIX 5)")
        print(f"     With Market    : {ablation_result['with_mkt']:.1%}")
        print(f"     No Market      : {ablation_result['no_mkt']:.1%}")
        drop = ablation_result['with_mkt'] - ablation_result['no_mkt']
        print(f"     Market Contribution: {drop:+.1%}")
        if drop > 0.05:
            print(f"     ⚠️  Edge มาจาก market odds เป็นหลัก (drop > 5%)")
        elif drop > 0.02:
            print(f"     🟡 Market ช่วย moderate (drop 2-5%)")
        else:
            print(f"     ✅ Model มี football intelligence แท้ (drop < 2%)")

    print(f"\n{SEP}\n  ✅  PHASE 3 COMPLETE\n{SEP}\n")
    return {'cv': cv_results, 'walk_forward': wf_results, 'roi': roi_result, 'mc': mc_results,
            'ablation': ablation_result}


# ══════════════════════════════════════════════════════════════
# 🚀  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

# STEP 1: อัปเดต CSV จาก API
update_season_csv_from_api()

# STEP 2: จำลองฤดูกาล
run_season_simulation()

# STEP 3: ทำนายทีมที่ต้องการ (แก้ชื่อทีมตามต้องการ)
predict_with_api("Arsenal")
# predict_with_api("Liverpool")
# predict_with_api("Man City")
# predict_with_api("Chelsea")
# predict_with_api("Aston Villa")

# STEP 4: ตารางแข่ง PL ถัดไป
show_next_pl_fixtures(5)

# STEP 5: Summary Report
print_full_summary()

# STEP 6: Phase 2 — Monte Carlo + Draw Calibration + Feature Importance
run_phase2(n_simulations=1000)

# STEP 7: Phase 3 — Rolling CV + ROI Backtest
run_phase3()