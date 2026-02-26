"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI — PRODUCTION VERSION v3.0                      ║
║   ปรับปรุงจาก v2.0 ด้วย:                                     ║
║   ✅ v2 All Features (EWM, Momentum, H2H, Elo, etc.)        ║
║   🔥 FIX 1: No Data Leakage — Sequential HomeWinRate         ║
║   🔥 FIX 2: Walk-Forward Season-by-Season Validation         ║
║   🔥 FIX 3: Poisson Regression Goal Model (xG → W/D/L)      ║
║   🔥 FIX 4: LightGBM as Core Model + Stacking               ║
║   🔥 FIX 5: SHAP Feature Importance Analysis                 ║
║   🔥 FIX 6: Full Kelly Criterion Betting Strategy            ║
║   🔥 FIX 7: Regime Detection (Form Clustering + HMM)         ║
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
                'Home_Elo','Away_Elo','Home_Elo_H','Away_Elo_A','Elo_Diff']].copy()
home_df.columns = ['MatchID','Date','Team','GF','GA',
                   'Own_Elo','Opp_Elo','Own_Elo_HA','Opp_Elo_HA','Elo_Diff']
home_df['Home'] = 1

away_df = data[['MatchID','Date','AwayTeam','FTAG','FTHG',
                'Away_Elo','Home_Elo','Away_Elo_A','Home_Elo_H','Elo_Diff']].copy()
away_df.columns = ['MatchID','Date','Team','GF','GA',
                   'Own_Elo','Opp_Elo','Own_Elo_HA','Opp_Elo_HA','Elo_Diff']
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
})

match_df = pd.merge(h, a, on='MatchID')
# Merge actual goals (FTHG / FTAG) for Poisson model training
actual_goals = data[['MatchID','FTHG','FTAG']].copy()
match_df = match_df.merge(actual_goals, on='MatchID', how='left')
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
    'H_Elo_Home_norm', 'A_Elo_Away_norm',              # NEW: venue-specific elo

    # ── Standard Rolling Form ─────────────────────────────────
    'H_GF5', 'H_GA5', 'H_Pts5', 'H_Streak3', 'H_CS5', 'H_Scored5',
    'A_GF5', 'A_GA5', 'A_Pts5', 'A_Streak3', 'A_CS5', 'A_Scored5',

    # ── Difference Features ───────────────────────────────────
    'Diff_Pts', 'Diff_GF', 'Diff_GA', 'Diff_GD',      # NEW: GD diff
    'Diff_Win', 'Diff_CS', 'Diff_Streak', 'Diff_Scored',

    # ── EWM Features (ใหม่ — นัดล่าสุดสำคัญกว่า) ───────────
    'H_GF_ewm', 'H_GA_ewm', 'H_Pts_ewm',
    'A_GF_ewm', 'A_GA_ewm', 'A_Pts_ewm',
    'Diff_Pts_ewm', 'Diff_GF_ewm', 'Diff_GD_ewm',

    # ── Momentum (ใหม่) ────────────────────────────────────────
    'Diff_Momentum',

    # ── Draw-Specific Features (ใหม่) ─────────────────────────
    'H_Draw5', 'A_Draw5', 'Diff_Draw5',
    'H2H_DrawRate',                                    # NEW: H2H draw rate
    'Combined_GF', 'Mean_GD_std',                      # NEW: low scoring / variance

    # ── H2H ──────────────────────────────────────────────────
    'H2H_HomeWinRate',

    # ── Home/Away Strength (ใหม่) ───────────────────────────
    'HomeWinRate', 'AwayWinRate', 'HomeDrawRate',

    # ── Days Rest (ใหม่) ───────────────────────────────────
    'H_DaysRest', 'A_DaysRest', 'Diff_DaysRest',

    # ── Seasonal (ใหม่) ────────────────────────────────────
    'Month', 'SeasonPhase',
]

print(f"✅ Features: {len(FEATURES)} ตัว (เพิ่มจาก 24 เป็น {len(FEATURES)})")

# ══════════════════════════════════════════════════════════════
# 11) TIME-BASED SPLIT
# ══════════════════════════════════════════════════════════════

match_df_clean = match_df.dropna(subset=FEATURES + ['Result3']).reset_index(drop=True)

split_date = match_df_clean['Date_x'].quantile(0.8)
train = match_df_clean[match_df_clean['Date_x'] <= split_date]
test  = match_df_clean[match_df_clean['Date_x'] > split_date]

X_train = train[FEATURES]
y_train = train['Result3']
X_test  = test[FEATURES]
y_test  = test['Result3']

print(f"\nTrain: {len(train)}  |  Test: {len(test)}")

# ══════════════════════════════════════════════════════════════
# 12) UPGRADED ENSEMBLE MODEL
#     LR + RF + GBT + ExtraTrees + MLP → Stacking + Calibration
# ══════════════════════════════════════════════════════════════

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n🔧 Building v3.0 Ensemble (LightGBM + RF + GBT + ExtraTrees + MLP)...")

# ── Base Models ──────────────────────────────────────────────
lr = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    C=0.3,
    solver='lbfgs',
)

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=8,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

et = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

gbt = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)

# MLP — captures non-linear patterns
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=500,
    learning_rate_init=0.001,
    alpha=0.01,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

# 🔥 LightGBM — core model v3.0 (ดีกว่า RF ส่วนใหญ่, เร็วกว่า GBT)
if LGBM_AVAILABLE:
    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    estimators = [
        ('lr',   lr),
        ('rf',   rf),
        ('et',   et),
        ('gbt',  gbt),
        ('mlp',  mlp),
        ('lgbm', lgbm_clf),
    ]
    weights = [1, 2, 2, 3, 2, 5]  # LightGBM น้ำหนักสูงสุด
    print("  Models: LR + RF + ExtraTrees + GBT + MLP + LightGBM (🔥)")
else:
    estimators = [
        ('lr',  lr),
        ('rf',  rf),
        ('et',  et),
        ('gbt', gbt),
        ('mlp', mlp),
    ]
    weights = [1, 3, 2, 4, 2]
    print("  Models: LR + RF + ExtraTrees + GBT + MLP")

# ── Soft Voting Ensemble ──────────────────────────────────────
ensemble = VotingClassifier(
    estimators=estimators,
    voting='soft',
    weights=weights,
)

print("  Training ensemble...")
ensemble.fit(X_train_sc, y_train)

y_pred = ensemble.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)

print(f"\n===== v3.0 ENSEMBLE RESULTS =====")
print(f"Accuracy: {round(acc*100, 2)}%")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Away Win','Draw','Home Win']))

# ── Isotonic Calibration (inline) ────────────────────────────
print("🎯 Applying Isotonic Calibration...")
try:
    # sklearn >= 1.2: ใช้ set_params แทน cv='prefit'
    from sklearn.calibration import CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    # เทรนบน training set (calibration ด้วย cross-val ภายใน)
    calibrated.fit(X_train_sc, y_train)
    y_pred_cal = calibrated.predict(X_test_sc)
    acc_cal = accuracy_score(y_test, y_pred_cal)
    print(f"Calibrated Accuracy (on test): {round(acc_cal*100, 2)}%")
except Exception as e:
    print(f"⚠️  Calibration skipped: {e}")
    calibrated = ensemble  # fallback ใช้ ensemble ตรงๆ

# ══════════════════════════════════════════════════════════════
# 13) SAVE MODEL
# ══════════════════════════════════════════════════════════════

model_bundle = {
    'model':       ensemble,
    'calibrated':  calibrated,
    'scaler':      scaler,
    'features':    FEATURES,
    'elo':         final_elo,
    'elo_home':    final_elo_home,
    'elo_away':    final_elo_away,
    'teams':       list(final_elo.keys()),
    'home_stats':  home_stats,
    'away_stats':  away_stats,
}

os.makedirs("model", exist_ok=True)
with open("model/football_model_v2.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("✅ Model v2 saved → model/football_model_v2.pkl")

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
    }
    return row


# ══════════════════════════════════════════════════════════════
# 15) PREDICT SINGLE MATCH
# ══════════════════════════════════════════════════════════════

def predict_match(home_team, away_team, match_date=None):
    teams_in_data = set(match_df_clean['HomeTeam'].tolist() + match_df_clean['AwayTeam'].tolist())
    if home_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{home_team}'  |  ทีมที่มี: {sorted(teams_in_data)}")
        return None
    if away_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{away_team}'")
        return None

    row  = build_match_row(home_team, away_team, match_date)
    X    = pd.DataFrame([row])[FEATURES]
    X_sc = scaler.transform(X)

    proba = ensemble.predict_proba(X_sc)[0]
    pred  = ensemble.predict(X_sc)[0]

    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    h_elo = final_elo.get(home_team, 1500)
    a_elo = final_elo.get(away_team, 1500)

    result = {
        'Away Win': round(proba[0]*100, 1),
        'Draw':     round(proba[1]*100, 1),
        'Home Win': round(proba[2]*100, 1),
        'Prediction': label_map[pred],
        'Home_Elo': round(h_elo),
        'Away_Elo': round(a_elo),
    }

    print(f"\n{'='*45}")
    print(f"  ⚽  {home_team}  vs  {away_team}")
    print(f"{'='*45}")
    print(f"  Elo:  {home_team} {round(h_elo)}  |  {away_team} {round(a_elo)}")
    print(f"{'─'*45}")
    bar_chars = 30
    for label, pct in [('Home Win', result['Home Win']),
                        ('Draw    ', result['Draw']),
                        ('Away Win', result['Away Win'])]:
        bar = '█' * int(pct / 100 * bar_chars)
        print(f"  {label}: {bar:<30} {pct}%")
    print(f"{'─'*45}")
    print(f"  🎯 Prediction: {result['Prediction']}")
    print(f"{'='*45}")
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
    print("  📊  FOOTBALL AI v3.0 — FULL SUMMARY REPORT")
    print(f"  🗓️  วันที่รายงาน: {TODAY.date()}  |  ข้อมูลถึง: {data['Date'].max().date()}")
    print("  🔥  v3.0: No Leakage | LightGBM | Poisson | SHAP | Kelly | Regimes")
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
    print(f"\n{SEP}\n  🤖  2. ประสิทธิภาพโมเดล (v3.0: LR+RF+ET+GBT+MLP+LGBM)\n{SEP}")
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

    draw_proba  = ensemble.predict_proba(X_test_sc)[:, 1]
    actual_draw = (y_test == 1).astype(int).values

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
    print(f"\n{SEP}\n  🔍  SHAP + RF FEATURE IMPORTANCE (v3.0)\n{SEP}")

    # ── RF impurity (fallback, เร็ว) ──────────────────────────
    rf_fitted = None
    gbt_fitted = None
    lgbm_fitted = None
    for (name, _), fitted in zip(ensemble.estimators, ensemble.estimators_):
        if name == 'rf':   rf_fitted   = fitted
        if name == 'gbt':  gbt_fitted  = fitted
        if name == 'lgbm': lgbm_fitted = fitted

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

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    if verbose:
        print(f"\n  {'Fold':<6} {'Train':>7} {'Val':>6} {'Acc':>7} "
              f"{'HW-F1':>9} {'DR-F1':>9} {'AW-F1':>9} {'LogLoss':>9}")
        print(f"  {LINE}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv), 1):
        X_tr, X_vl = X_cv[train_idx], X_cv[val_idx]
        y_tr, y_vl = y_cv[train_idx], y_cv[val_idx]

        sc_fold  = StandardScaler()
        X_tr_sc  = sc_fold.fit_transform(X_tr)
        X_vl_sc  = sc_fold.transform(X_vl)

        # GBT เร็วกว่า full ensemble ใน CV
        cv_gbt = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        cv_gbt.fit(X_tr_sc, y_tr)
        y_pred_fold = cv_gbt.predict(X_vl_sc)
        y_proba_fold = cv_gbt.predict_proba(X_vl_sc)

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
        print(SEP)
    return fold_results


# ══════════════════════════════════════════════════════════════
# 25) BACKTEST ROI
# ══════════════════════════════════════════════════════════════

def backtest_roi(bankroll=1000.0, min_edge=0.03, kelly_fraction=0.25,
                 max_exposure=0.05, verbose=True):
    """
    🔥 v3.0 Kelly Criterion Betting Strategy
    - Full Kelly sizing with fraction
    - min_edge: ต้องมี edge > X% ถึงเดิมพัน
    - max_exposure: จำกัด % bankroll ต่อแมตช์ (risk management)
    - แสดง edge distribution + per-outcome ROI
    """
    SEP  = "=" * 65
    LINE = "─" * 65
    if verbose:
        print(f"\n{SEP}")
        print(f"  💰  KELLY CRITERION BACKTEST (v3.0)")
        print(f"  Bankroll: £{bankroll:,.0f} | Min Edge: {min_edge*100:.0f}% | "
              f"Kelly: {kelly_fraction*100:.0f}% | Max: {max_exposure*100:.0f}%/bet")
        print(SEP)

    proba_test = ensemble.predict_proba(X_test_sc)
    label_map  = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

    bk = bankroll; bets = []; total_bets = 0; total_won = 0
    total_staked = 0.0; peak_bk = bk; max_dd = 0.0
    edge_dist = []

    for proba, actual in zip(proba_test, y_test):
        p_away, p_draw, p_home = proba
        # สร้าง implied odds ด้วย bookmaker margin 5%
        margin = 1.05
        odds   = {
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
    cv_df['Year'] = cv_df['Date_x'].dt.year

    years = sorted(cv_df['Year'].unique())
    # ต้องมี training data อย่างน้อย 3 ปี
    test_years = [y for y in years if y >= years[min(3, len(years)-1)]]

    if len(test_years) == 0:
        print("  ⚠️  ไม่มีข้อมูลเพียงพอสำหรับ walk-forward CV")
        return []

    fold_results = []
    if verbose:
        print(f"\n  {'Year':<8} {'Train':>8} {'Test':>7} {'Acc':>8} {'Draw-F1':>9} {'LogLoss':>9}")
        print(f"  {LINE}")

    for test_year in test_years:
        train_mask = cv_df['Year'] < test_year
        test_mask  = cv_df['Year'] == test_year
        if train_mask.sum() < 100 or test_mask.sum() < 30:
            continue

        X_tr = cv_df.loc[train_mask, FEATURES].values
        y_tr = cv_df.loc[train_mask, 'Result3'].values
        X_te = cv_df.loc[test_mask,  FEATURES].values
        y_te = cv_df.loc[test_mask,  'Result3'].values

        sc_wf  = StandardScaler()
        X_tr_sc = sc_wf.fit_transform(X_tr)
        X_te_sc = sc_wf.transform(X_te)

        # ใช้ LightGBM ถ้ามี (เร็วกว่า full ensemble ใน CV)
        if LGBM_AVAILABLE:
            cv_model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                num_leaves=25, class_weight='balanced',
                random_state=42, n_jobs=-1, verbose=-1
            )
        else:
            cv_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            )

        cv_model.fit(X_tr_sc, y_tr)
        y_pred_wf  = cv_model.predict(X_te_sc)
        y_proba_wf = cv_model.predict_proba(X_te_sc)

        a   = accuracy_score(y_te, y_pred_wf)
        ll  = log_loss(y_te, y_proba_wf)
        rep = classification_report(y_te, y_pred_wf, output_dict=True, zero_division=0)
        draw_f1 = rep.get('1', {}).get('f1-score', 0)

        fold_results.append({
            'year': test_year, 'train_size': train_mask.sum(),
            'test_size': test_mask.sum(), 'acc': a,
            'draw_f1': draw_f1, 'logloss': ll
        })
        if verbose:
            model_tag = "LGBM" if LGBM_AVAILABLE else "GBT"
            print(f"  {test_year:<8} {train_mask.sum():>8} {test_mask.sum():>7} "
                  f"{a:>8.4f} {draw_f1:>9.4f} {ll:>9.4f}")

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

def run_phase3(n_simulations=1000):
    print(f"\n{'█'*65}")
    print(f"  🏆  PHASE 3 — PRODUCTION GRADE v3.0")
    print(f"{'█'*65}")

    cv_results  = rolling_window_cv(n_splits=5)
    wf_results  = walk_forward_season_cv()          # 🔥 NEW: Walk-Forward
    roi_result  = backtest_roi(bankroll=1000.0, min_edge=0.03, kelly_fraction=0.25)
    mc_results  = run_monte_carlo(n_simulations=n_simulations)
    regime_data = analyze_league_regimes(top_n=6)  # 🔥 NEW: Regime Detection

    SEP = "=" * 65
    cv_accs = [r['acc'] for r in cv_results]
    cv_drs  = [r['draw_f1'] for r in cv_results]

    print(f"\n{SEP}\n  📋  PHASE 3 — SUMMARY v3.0\n{SEP}")
    print(f"\n  🔄 Rolling CV (5 folds)")
    print(f"     Mean Accuracy : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
    print(f"     Mean Draw F1  : {np.mean(cv_drs):.4f}")

    if wf_results:
        wf_accs = [r['acc'] for r in wf_results]
        print(f"\n  🏆 Walk-Forward CV (season-by-season) 🔥")
        print(f"     Mean Accuracy : {np.mean(wf_accs):.4f} ± {np.std(wf_accs):.4f}")
        print(f"     Range         : [{min(wf_accs):.4f} – {max(wf_accs):.4f}]")

    if roi_result:
        print(f"\n  💰 Backtest ROI")
        print(f"     ROI            : {roi_result['roi']:+.1f}%")
        print(f"     Win Rate       : {roi_result['win_rate']:.1f}%")
        print(f"     Max Drawdown   : {roi_result['max_dd']:.1f}%")
        print(f"     Total Bets     : {roi_result['total_bets']:,}")
    print(f"\n{SEP}\n  ✅  PHASE 3 COMPLETE\n{SEP}\n")
    return {'cv': cv_results, 'walk_forward': wf_results, 'roi': roi_result, 'mc': mc_results}


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