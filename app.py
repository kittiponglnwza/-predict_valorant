import pandas as pd
import numpy as np
import glob
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ==============================
# 1) LOAD ALL DATA
# ==============================

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

# ==============================
# 2) ELO RATING
# ==============================

def compute_elo(data, k=32, base=1500):
    """คำนวณ Elo Rating แบบ time-series (ไม่มี leakage)"""
    elo = {}
    home_elo_before = []
    away_elo_before = []

    for _, row in data.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        hg   = row['FTHG']
        ag   = row['FTAG']

        # กำหนด Elo เริ่มต้น
        if home not in elo: elo[home] = base
        if away not in elo: elo[away] = base

        # บันทึก Elo ก่อนแมตช์
        home_elo_before.append(elo[home])
        away_elo_before.append(elo[away])

        # คำนวณ Expected
        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_away = 1 - exp_home

        # ผลจริง
        if hg > ag:
            score_home, score_away = 1.0, 0.0
        elif hg < ag:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5

        # อัปเดต Elo
        elo[home] += k * (score_home - exp_home)
        elo[away] += k * (score_away - exp_away)

    data = data.copy()
    data['Home_Elo'] = home_elo_before
    data['Away_Elo'] = away_elo_before
    data['Elo_Diff'] = data['Home_Elo'] - data['Away_Elo']
    return data, elo

data, final_elo = compute_elo(data)
print("\n✅ Elo Rating computed")

# ==============================
# 3) TEAM-CENTRIC TABLE
# ==============================

home_df = data[['MatchID','Date','HomeTeam','FTHG','FTAG','Home_Elo','Away_Elo','Elo_Diff']].copy()
home_df.columns = ['MatchID','Date','Team','GF','GA','Own_Elo','Opp_Elo','Elo_Diff']
home_df['Home'] = 1

away_df = data[['MatchID','Date','AwayTeam','FTAG','FTHG','Away_Elo','Home_Elo','Elo_Diff']].copy()
away_df.columns = ['MatchID','Date','Team','GF','GA','Own_Elo','Opp_Elo','Elo_Diff']
away_df['Home'] = 0

team_df = pd.concat([home_df, away_df], ignore_index=True)
team_df = team_df.sort_values(['Team','Date']).reset_index(drop=True)

team_df['Win']    = (team_df['GF'] > team_df['GA']).astype(int)
team_df['Draw']   = (team_df['GF'] == team_df['GA']).astype(int)
team_df['Loss']   = (team_df['GF'] < team_df['GA']).astype(int)
team_df['Points'] = team_df['Win']*3 + team_df['Draw']
team_df['CS']     = (team_df['GA'] == 0).astype(int)   # Clean Sheet
team_df['Scored'] = (team_df['GF'] > 0).astype(int)   # Scored at least 1

# ==============================
# 4) ROLLING FEATURES (NO LEAKAGE)
# ==============================

def rolling_shift(df, col, window=5):
    return (
        df.groupby('Team')[col]
        .rolling(window).mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

team_df['GF_last5']     = rolling_shift(team_df, 'GF')
team_df['GA_last5']     = rolling_shift(team_df, 'GA')
team_df['Points_last5'] = rolling_shift(team_df, 'Points')
team_df['Win_last5']    = rolling_shift(team_df, 'Win')
team_df['CS_last5']     = rolling_shift(team_df, 'CS')
team_df['Scored_last5'] = rolling_shift(team_df, 'Scored')

# Streak: ผลแมตช์ล่าสุด 3 นัด (cumulative points)
team_df['Streak3']      = rolling_shift(team_df, 'Points', window=3)

team_df = team_df.dropna()

# ==============================
# 5) MERGE BACK TO MATCH LEVEL
# ==============================

h = team_df[team_df['Home'] == 1].copy().rename(columns={
    'Team': 'HomeTeam',
    'GF_last5':     'H_GF5',
    'GA_last5':     'H_GA5',
    'Points_last5': 'H_Pts5',
    'Win_last5':    'H_Win5',
    'CS_last5':     'H_CS5',
    'Scored_last5': 'H_Scored5',
    'Streak3':      'H_Streak3',
    'Own_Elo':      'H_Elo',
})

a = team_df[team_df['Home'] == 0].copy().rename(columns={
    'Team': 'AwayTeam',
    'GF_last5':     'A_GF5',
    'GA_last5':     'A_GA5',
    'Points_last5': 'A_Pts5',
    'Win_last5':    'A_Win5',
    'CS_last5':     'A_CS5',
    'Scored_last5': 'A_Scored5',
    'Streak3':      'A_Streak3',
    'Own_Elo':      'A_Elo',
})

match_df = pd.merge(h, a, on='MatchID')
print(f"\n✅ Matches after feature engineering: {len(match_df)}")

# ==============================
# 6) DIFFERENCE FEATURES
# ==============================

match_df['Diff_Pts']    = match_df['H_Pts5']    - match_df['A_Pts5']
match_df['Diff_GF']     = match_df['H_GF5']     - match_df['A_GF5']
match_df['Diff_GA']     = match_df['H_GA5']     - match_df['A_GA5']
match_df['Diff_Win']    = match_df['H_Win5']    - match_df['A_Win5']
match_df['Diff_CS']     = match_df['H_CS5']     - match_df['A_CS5']
match_df['Diff_Streak'] = match_df['H_Streak3'] - match_df['A_Streak3']
match_df['Diff_Elo']    = match_df['H_Elo']     - match_df['A_Elo']

# ==============================
# 7) HEAD-TO-HEAD (H2H)
# ==============================

def compute_h2h(data):
    """คำนวณ H2H win rate ของทีมเหย้าในแมตช์ที่เคยเจอกัน (ไม่มี leakage)"""
    h2h_home_wins = {}
    h2h_total     = {}
    h2h_rates     = []

    for _, row in data.sort_values('Date_x').iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        key  = tuple(sorted([home, away]))

        rate = h2h_home_wins.get((home, away), 0) / max(h2h_total.get(key, 1), 1)
        h2h_rates.append(rate)

        # อัปเดตหลังบันทึก
        if key not in h2h_total:
            h2h_total[key] = 0
        h2h_total[key] += 1

        if (home, away) not in h2h_home_wins:
            h2h_home_wins[(home, away)] = 0
        if row['Win_x'] == 1:
            h2h_home_wins[(home, away)] += 1

    return h2h_rates

match_df = match_df.sort_values('Date_x').reset_index(drop=True)
match_df['H2H_HomeWinRate'] = compute_h2h(match_df)
print("✅ H2H computed")

# ==============================
# 8) TARGET VARIABLE
# ==============================

def get_result(row):
    if row['Win_x'] == 1:   return 2   # Home Win
    elif row['Draw_x'] == 1: return 1  # Draw
    else:                    return 0  # Away Win

match_df['Result3'] = match_df.apply(get_result, axis=1)

FEATURES = [
    'Diff_Pts', 'Diff_GF', 'Diff_GA', 'Diff_Win',
    'Diff_CS', 'Diff_Streak', 'Diff_Elo', 'H2H_HomeWinRate',
    'H_GF5', 'H_GA5', 'H_Pts5', 'H_Streak3',
    'A_GF5', 'A_GA5', 'A_Pts5', 'A_Streak3',
]

# ==============================
# 9) TIME-BASED SPLIT
# ==============================

split_date = match_df['Date_x'].quantile(0.8)
train = match_df[match_df['Date_x'] <= split_date]
test  = match_df[match_df['Date_x'] > split_date]

X_train = train[FEATURES]
y_train = train['Result3']
X_test  = test[FEATURES]
y_test  = test['Result3']

print(f"\nTrain: {len(train)}  |  Test: {len(test)}")

# ==============================
# 10) ENSEMBLE MODEL
# ==============================

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0
)

ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    voting='soft'
)

ensemble.fit(X_train_sc, y_train)
y_pred = ensemble.predict(X_test_sc)

print("\n===== ENSEMBLE MODEL RESULTS =====")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 4)}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Away Win','Draw','Home Win']))

# ==============================
# 11) SAVE MODEL (PICKLE)
# ==============================

model_bundle = {
    'model':    ensemble,
    'scaler':   scaler,
    'features': FEATURES,
    'elo':      final_elo,
    'teams':    list(final_elo.keys()),
}

os.makedirs("model", exist_ok=True)
with open("model/football_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("\n✅ Model saved → model/football_model.pkl")

# ==============================
# 12) PREDICT SINGLE MATCH
# ==============================

def predict_match(home_team, away_team, model_path="model/football_model.pkl"):
    """
    ทำนายแมตช์เดี่ยว โดยใช้ข้อมูลฟอร์มล่าสุดจาก match_df
    และ Elo จาก final_elo

    Returns:
        dict: { 'Home Win': %, 'Draw': %, 'Away Win': %, 'Prediction': str }
    """
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    mdl      = bundle['model']
    scl      = bundle['scaler']
    feats    = bundle['features']
    elo_dict = bundle['elo']

    teams_in_data = set(match_df['HomeTeam'].tolist() + match_df['AwayTeam'].tolist())
    if home_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{home_team}' ในข้อมูล")
        print(f"   ทีมที่มี: {sorted(teams_in_data)}")
        return None
    if away_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{away_team}' ในข้อมูล")
        return None

    # ดึงฟอร์มล่าสุดของแต่ละทีมจาก match_df
    def latest_home_stats(team):
        rows = match_df[match_df['HomeTeam'] == team].sort_values('Date_x')
        if len(rows) == 0:
            rows = match_df[match_df['AwayTeam'] == team].sort_values('Date_x')
            last = rows.iloc[-1]
            return {
                'GF5': last['A_GF5'], 'GA5': last['A_GA5'],
                'Pts5': last['A_Pts5'], 'Streak3': last['A_Streak3'],
                'Win5': last['A_Win5'], 'CS5': last['A_CS5'],
            }
        last = rows.iloc[-1]
        return {
            'GF5': last['H_GF5'], 'GA5': last['H_GA5'],
            'Pts5': last['H_Pts5'], 'Streak3': last['H_Streak3'],
            'Win5': last['H_Win5'], 'CS5': last['H_CS5'],
        }

    h_stats = latest_home_stats(home_team)
    a_stats = latest_home_stats(away_team)

    h_elo = elo_dict.get(home_team, 1500)
    a_elo = elo_dict.get(away_team, 1500)

    # H2H rate ล่าสุด
    h2h_rows = match_df[
        (match_df['HomeTeam'] == home_team) & (match_df['AwayTeam'] == away_team)
    ]
    h2h_rate = h2h_rows['H2H_HomeWinRate'].iloc[-1] if len(h2h_rows) > 0 else 0.33

    row = {
        'Diff_Pts':          h_stats['Pts5']    - a_stats['Pts5'],
        'Diff_GF':           h_stats['GF5']     - a_stats['GF5'],
        'Diff_GA':           h_stats['GA5']     - a_stats['GA5'],
        'Diff_Win':          h_stats['Win5']    - a_stats['Win5'],
        'Diff_CS':           h_stats['CS5']     - a_stats['CS5'],
        'Diff_Streak':       h_stats['Streak3'] - a_stats['Streak3'],
        'Diff_Elo':          h_elo - a_elo,
        'H2H_HomeWinRate':   h2h_rate,
        'H_GF5':             h_stats['GF5'],
        'H_GA5':             h_stats['GA5'],
        'H_Pts5':            h_stats['Pts5'],
        'H_Streak3':         h_stats['Streak3'],
        'A_GF5':             a_stats['GF5'],
        'A_GA5':             a_stats['GA5'],
        'A_Pts5':            a_stats['Pts5'],
        'A_Streak3':         a_stats['Streak3'],
    }

    X = pd.DataFrame([row])[feats]
    X_sc = scl.transform(X)

    proba = mdl.predict_proba(X_sc)[0]
    pred  = mdl.predict(X_sc)[0]

    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
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


# ==============================
# 13) SEASON SIMULATION 2025-26 (FIXED)
# ==============================
# ปัญหาเดิม: season 2025.csv มีแมตช์ปี 2026 ปนอยู่
# แก้: แยก "แมตช์แข่งแล้ว" (มีผล FTHG/FTAG จริง)
#      vs "แมตช์ยังไม่แข่ง" (FTHG เป็น NaN หรือวันในอนาคต)

import datetime
TODAY = pd.Timestamp(datetime.date.today())

season_file = pd.read_csv("data_set/season 2025.csv")
season_file['Date'] = pd.to_datetime(season_file['Date'], dayfirst=True)

# ✅ แมตช์ที่แข่งแล้ว = มีผลจริง (FTHG ไม่ใช่ NaN)
played   = season_file.dropna(subset=['FTHG', 'FTAG']).copy()
played   = played[played['Date'] <= TODAY]

# ⏳ แมตช์ที่ยังไม่แข่ง = FTHG เป็น NaN หรือวันหน้า
unplayed = season_file[
    season_file['FTHG'].isna() | (season_file['Date'] > TODAY)
].copy()

print(f"\n📅 วันนี้: {TODAY.date()}")
print(f"✅ แมตช์แข่งแล้ว:    {len(played)} นัด")
print(f"⏳ แมตช์ยังไม่แข่ง: {len(unplayed)} นัด")
print(f"   รวม: {len(played) + len(unplayed)} นัด (ฤดูกาล 38 นัด × 20 ทีม = 380 นัด)")

# --- คะแนนจริงจากแมตช์ที่แข่งแล้ว ---
real_table = {}
for _, row in played.iterrows():
    home, away = row['HomeTeam'], row['AwayTeam']
    hg, ag     = int(row['FTHG']), int(row['FTAG'])
    for t in [home, away]:
        if t not in real_table: real_table[t] = 0
    if hg > ag:   real_table[home] += 3
    elif hg < ag: real_table[away] += 3
    else:
        real_table[home] += 1
        real_table[away] += 1

real_table_df = pd.DataFrame.from_dict(
    real_table, orient='index', columns=['RealPoints']
)

# --- ทำนายแมตช์ที่ยังไม่แข่ง ---
# ใช้ match_df ที่มี rolling features ครบ กรอง Date > TODAY
future_matches = match_df[match_df['Date_x'] > TODAY].copy()
pred_table = {}

if len(future_matches) > 0:
    X_future = scaler.transform(future_matches[FEATURES])
    future_matches = future_matches.copy()
    future_matches['Pred'] = ensemble.predict(X_future)
    print(f"🤖 ทำนาย {len(future_matches)} แมตช์ที่เหลือ")

    for _, row in future_matches.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        pred       = row['Pred']
        for t in [home, away]:
            if t not in pred_table: pred_table[t] = 0
        if pred == 2:   pred_table[home] += 3
        elif pred == 1: pred_table[home] += 1; pred_table[away] += 1
        else:           pred_table[away] += 3
else:
    print("ℹ️  ไม่มีแมตช์ที่เหลือให้ทำนาย (จบฤดูกาลแล้ว)")

pred_table_df = pd.DataFrame.from_dict(
    pred_table, orient='index', columns=['PredictedPoints']
)

# --- รวมตาราง ---
final_table = real_table_df.join(pred_table_df, how='left').fillna(0)
final_table['PredictedPoints'] = final_table['PredictedPoints'].astype(int)
final_table['FinalPoints']     = final_table['RealPoints'] + final_table['PredictedPoints']
final_table = final_table.sort_values('FinalPoints', ascending=False)
final_table.index.name = 'Team'

print(f"\n{'='*55}")
print(f"  🏆  FULL SEASON 2025-26 PREDICTED TABLE")
print(f"      (Real ถึง {TODAY.date()} + AI ทำนายส่วนที่เหลือ)")
print(f"{'='*55}")
print(f"  {'#':<4} {'Team':<20} {'Real':>6} {'Pred':>6} {'Total':>7}")
print(f"  {'─'*50}")
for rank, (team, row) in enumerate(final_table.iterrows(), 1):
    marker = "🔴" if rank <= 3 else ("🟡" if rank >= 18 else "  ")
    print(f"{marker} {rank:<4} {team:<20} {int(row['RealPoints']):>6} "
          f"{int(row['PredictedPoints']):>6} {int(row['FinalPoints']):>7}")
print(f"  {'─'*50}")
print(f"  🔴 = Top 3 (UEFA CL)  |  🟡 = Relegation Zone")

# ==============================
# 14) EXAMPLE: PREDICT A SINGLE MATCH
# ==============================
# แก้ชื่อทีมให้ตรงกับข้อมูลของคุณ เช่น:
# predict_match("Man City", "Arsenal")
# predict_match("Liverpool", "Chelsea")


# ==============================
# 15) GET LAST 5 RESULTS (ผลย้อนหลัง 5 แมตช์)
# ==============================

def get_last_5_results(team):
    """
    แสดงผล 5 แมตช์ล่าสุดของทีม
    Returns: DataFrame พร้อม print สวยงาม
    """
    # หาแมตช์ทั้งหมดที่ทีมเล่น (ทั้งเหย้าและเยือน)
    home_matches = data[data['HomeTeam'] == team][['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
    home_matches['Venue'] = 'H'
    home_matches['GF'] = home_matches['FTHG']
    home_matches['GA'] = home_matches['FTAG']
    home_matches['Opponent'] = home_matches['AwayTeam']

    away_matches = data[data['AwayTeam'] == team][['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
    away_matches['Venue'] = 'A'
    away_matches['GF'] = away_matches['FTAG']
    away_matches['GA'] = away_matches['FTHG']
    away_matches['Opponent'] = away_matches['HomeTeam']

    all_matches = pd.concat([home_matches, away_matches]).sort_values('Date', ascending=False)
    last5 = all_matches.head(5).copy()

    def get_result_label(row):
        if row['GF'] > row['GA']:   return 'W'
        elif row['GF'] == row['GA']: return 'D'
        else:                        return 'L'

    last5['Result'] = last5.apply(get_result_label, axis=1)

    print(f"\n{'='*55}")
    print(f"  📋  5 แมตช์ล่าสุดของ {team}")
    print(f"{'='*55}")
    print(f"  {'วันที่':<12} {'คู่แข่ง':<22} {'H/A':<5} {'สกอร์':<10} {'ผล'}")
    print(f"  {'─'*52}")
    for _, row in last5.iterrows():
        icon = {'W': '✅', 'D': '🟡', 'L': '❌'}[row['Result']]
        date_str = row['Date'].strftime('%d/%m/%Y') if pd.notna(row['Date']) else 'N/A'
        score = f"{int(row['GF'])}-{int(row['GA'])}"
        print(f"  {date_str:<12} {row['Opponent']:<22} {row['Venue']:<5} {score:<10} {icon} {row['Result']}")
    print(f"{'='*55}")

    return last5[['Date', 'Opponent', 'Venue', 'GF', 'GA', 'Result']]


# ==============================
# 16) PREDICT SCORE (ทำนายสกอร์ + เปอร์เซนต์)
# ==============================

def predict_score(home_team, away_team):
    """
    ทำนายสกอร์ที่น่าจะเป็น โดยใช้ฟอร์มเฉลี่ยล่าสุด
    คำนวณจาก Expected Goals (xG) อย่างง่าย แล้วคำนวณ Poisson probability

    Returns:
        dict: { 'home_xg', 'away_xg', 'most_likely_score', 'score_probabilities' (top 5) }
    """
    from scipy.stats import poisson

    teams_in_data = set(match_df['HomeTeam'].tolist() + match_df['AwayTeam'].tolist())
    if home_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{home_team}' ในข้อมูล")
        return None
    if away_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{away_team}' ในข้อมูล")
        return None

    # ดึงฟอร์มล่าสุด (ค่าเฉลี่ยประตูย้อนหลัง 5 แมตช์)
    def get_team_avg(team, is_home):
        if is_home:
            rows = match_df[match_df['HomeTeam'] == team].sort_values('Date_x')
            if len(rows) > 0:
                last = rows.iloc[-1]
                return last['H_GF5'], last['H_GA5']
        else:
            rows = match_df[match_df['AwayTeam'] == team].sort_values('Date_x')
            if len(rows) > 0:
                last = rows.iloc[-1]
                return last['A_GF5'], last['A_GA5']
        return 1.5, 1.5  # default

    h_gf_avg, h_ga_avg = get_team_avg(home_team, is_home=True)
    a_gf_avg, a_ga_avg = get_team_avg(away_team, is_home=False)

    # xG โดยประมาณ: ค่าเฉลี่ยของ attack ทีมนั้น vs defense ฝั่งตรงข้าม
    # normalize ด้วย league average
    league_avg_goals = data['FTHG'].mean() + data['FTAG'].mean()
    league_avg_home  = data['FTHG'].mean()
    league_avg_away  = data['FTAG'].mean()

    home_attack  = h_gf_avg / league_avg_home if league_avg_home > 0 else 1.0
    home_defense = h_ga_avg / league_avg_away if league_avg_away > 0 else 1.0
    away_attack  = a_gf_avg / league_avg_away if league_avg_away > 0 else 1.0
    away_defense = a_ga_avg / league_avg_home if league_avg_home > 0 else 1.0

    home_xg = home_attack * away_defense * league_avg_home
    away_xg = away_attack * home_defense * league_avg_away

    # Poisson: คำนวณโอกาสแต่ละสกอร์
    max_goals = 6
    score_probs = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            prob = poisson.pmf(hg, home_xg) * poisson.pmf(ag, away_xg)
            score_probs[f"{hg}-{ag}"] = round(prob * 100, 2)

    # เรียงลำดับโอกาสสูงสุด
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    top5_scores   = sorted_scores[:5]
    best_score    = top5_scores[0][0]

    print(f"\n{'='*55}")
    print(f"  ⚽  ทำนายสกอร์: {home_team}  vs  {away_team}")
    print(f"{'='*55}")
    print(f"  📊 Expected Goals:  {home_team} {round(home_xg, 2)} xG  |  {away_team} {round(away_xg, 2)} xG")
    print(f"  🎯 สกอร์ที่น่าจะเป็นที่สุด: {best_score}")
    print(f"\n  {'สกอร์':<12} {'โอกาส (%)'}")
    print(f"  {'─'*25}")
    for score, pct in top5_scores:
        bar = '█' * int(pct * 2)
        print(f"  {score:<12} {bar:<20} {pct}%")
    print(f"{'='*55}")

    return {
        'home_xg':          round(home_xg, 2),
        'away_xg':          round(away_xg, 2),
        'most_likely_score': best_score,
        'top5_scores':       top5_scores,
    }


# ==============================
# 17) PREDICT NEXT 5 MATCHES (ทำนาย 5 แมตช์ข้างหน้า)
# ==============================

def predict_next_5_matches(team, model_path="model/football_model.pkl"):
    """
    ทำนาย 5 แมตช์ข้างหน้าของทีม พร้อม:
    - เปอร์เซนต์ชนะ/เสมอ/แพ้
    - สกอร์ที่น่าจะเป็น
    - ผลย้อนหลัง 5 แมตช์

    Parameters:
        team (str): ชื่อทีม เช่น "Arsenal", "Man City"

    Returns:
        dict: { 'next_5_predictions': [...], 'last_5_results': DataFrame }
    """
    import datetime

    print(f"\n{'#'*60}")
    print(f"  🔮  วิเคราะห์ทีม: {team.upper()}")
    print(f"{'#'*60}")

    # ─── STEP 1: ผลย้อนหลัง 5 แมตช์ ───
    last5_df = get_last_5_results(team)

    # ─── STEP 2: หา 5 แมตช์ข้างหน้า ───
    today = pd.Timestamp(datetime.date.today())

    # ดึงจาก season file (แมตช์ที่ยังไม่แข่ง)
    season_file_reload = pd.read_csv("data_set/season 2025.csv")
    season_file_reload['Date'] = pd.to_datetime(season_file_reload['Date'], dayfirst=True)

    upcoming = season_file_reload[
        ((season_file_reload['HomeTeam'] == team) | (season_file_reload['AwayTeam'] == team)) &
        (season_file_reload['FTHG'].isna() | (season_file_reload['Date'] > today))
    ].sort_values('Date').head(5)

    # fallback: ถ้าไม่มีใน season file ให้ใช้ match_df แมตช์ล่าสุด 5 นัด
    if len(upcoming) == 0:
        upcoming_from_match = match_df[
            ((match_df['HomeTeam'] == team) | (match_df['AwayTeam'] == team)) &
            (match_df['Date_x'] > today)
        ].sort_values('Date_x').head(5)
        if len(upcoming_from_match) > 0:
            upcoming = upcoming_from_match[['Date_x', 'HomeTeam', 'AwayTeam']].rename(
                columns={'Date_x': 'Date'}
            )

    if len(upcoming) == 0:
        print(f"\n⚠️  ไม่พบแมตช์ข้างหน้าของ {team} ในข้อมูล")
        print(f"   อาจเป็นเพราะข้อมูลฤดูกาลครบแล้ว หรือชื่อทีมไม่ตรง")
        return None

    print(f"\n{'='*60}")
    print(f"  🔮  5 แมตช์ข้างหน้า: {team}")
    print(f"{'='*60}")

    predictions = []

    for i, (_, match) in enumerate(upcoming.iterrows(), 1):
        home_team_m = match['HomeTeam']
        away_team_m = match['AwayTeam']
        match_date  = match['Date'].strftime('%d/%m/%Y') if pd.notna(match['Date']) else 'TBD'
        is_home     = (home_team_m == team)
        opponent    = away_team_m if is_home else home_team_m
        venue       = '(H)' if is_home else '(A)'

        print(f"\n  {'─'*56}")
        print(f"  นัดที่ {i} | {match_date}  |  {home_team_m} vs {away_team_m}")
        print(f"  {'─'*56}")

        # ทำนายผล (win/draw/loss probability)
        result_pred = predict_match(home_team_m, away_team_m, model_path)
        # ทำนายสกอร์ (score probability)
        score_pred  = predict_score(home_team_m, away_team_m)

        if result_pred and score_pred:
            # แปลผลให้เป็นมุมมองของทีมที่เราสนใจ
            if is_home:
                win_pct  = result_pred['Home Win']
                draw_pct = result_pred['Draw']
                loss_pct = result_pred['Away Win']
                likely_outcome = result_pred['Prediction']
            else:
                win_pct  = result_pred['Away Win']
                draw_pct = result_pred['Draw']
                loss_pct = result_pred['Home Win']
                pred_map = {'Home Win': 'Away Win', 'Away Win': 'Home Win', 'Draw': 'Draw'}
                likely_outcome = pred_map.get(result_pred['Prediction'], result_pred['Prediction'])

            team_wins = 'Win' in likely_outcome and (
                (is_home and 'Home' in likely_outcome) or
                (not is_home and 'Away' in likely_outcome)
            )
            team_result_label = (
                f"✅ {team} ชนะ" if team_wins else
                (f"🟡 เสมอ" if 'Draw' in likely_outcome else f"❌ {team} แพ้")
            )

            print(f"\n  📌 ผลที่น่าจะเป็น:  {team_result_label}")
            print(f"  📊 ชนะ {win_pct}%  |  เสมอ {draw_pct}%  |  แพ้ {loss_pct}%")
            print(f"  🎯 สกอร์คาด: {score_pred['most_likely_score']}")
            print(f"  📈 xG: {home_team_m} {score_pred['home_xg']}  vs  {away_team_m} {score_pred['away_xg']}")

            predictions.append({
                'match_no':        i,
                'date':            match_date,
                'home':            home_team_m,
                'away':            away_team_m,
                'venue':           venue,
                'opponent':        opponent,
                'win_pct':         win_pct,
                'draw_pct':        draw_pct,
                'loss_pct':        loss_pct,
                'predicted_result': likely_outcome,
                'predicted_score': score_pred['most_likely_score'],
                'top_scores':      score_pred['top5_scores'],
                'home_xg':         score_pred['home_xg'],
                'away_xg':         score_pred['away_xg'],
            })

    print(f"\n{'#'*60}")
    print(f"  📋  สรุปการทำนาย 5 แมตช์ข้างหน้า: {team}")
    print(f"{'#'*60}")
    print(f"  {'นัด':<5} {'วันที่':<12} {'คู่แข่ง':<22} {'H/A':<5} {'ชนะ%':<8} {'เสมอ%':<8} {'แพ้%':<8} {'สกอร์คาด'}")
    print(f"  {'─'*80}")
    for p in predictions:
        print(f"  {p['match_no']:<5} {p['date']:<12} {p['opponent']:<22} {p['venue']:<5} "
              f"{p['win_pct']:<8} {p['draw_pct']:<8} {p['loss_pct']:<8} {p['predicted_score']}")
    print(f"{'#'*60}\n")

    return {
        'next_5_predictions': predictions,
        'last_5_results':     last5_df,
    }


# ==============================
# 18) EXAMPLE USAGE
# ==============================
# เรียกใช้ฟังก์ชันใหม่ได้ดังนี้:
#
# ดูผลย้อนหลัง 5 แมตช์:
#   get_last_5_results("Arsenal")
#
# ทำนายสกอร์แมตช์เดียว:
#   predict_score("Man City", "Liverpool")
#
# ทำนาย 5 แมตช์ข้างหน้า + ผลย้อนหลัง (ฟังก์ชันหลัก):
#   predict_next_5_matches("Arsenal")
#   predict_next_5_matches("Man City")
#   predict_next_5_matches("Liverpool")