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
# 14) PREDICT SINGLE MATCH (helper)
# ==============================
# ใช้เรียกแบบ manual ได้:  predict_match("Man City", "Arsenal")
# ตัวอย่าง:
# predict_match("Liverpool", "Chelsea")
# predict_match("Arsenal", "Man City")

def predict_match(home_team, away_team, model_path="model/football_model.pkl"):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    mdl      = bundle['model']
    scl      = bundle['scaler']
    feats    = bundle['features']
    elo_dict = bundle['elo']

    teams_in_data = set(match_df['HomeTeam'].tolist() + match_df['AwayTeam'].tolist())
    if home_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{home_team}'"); return None
    if away_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{away_team}'"); return None

    def latest_home_stats(team):
        rows = match_df[match_df['HomeTeam'] == team].sort_values('Date_x')
        if len(rows) == 0:
            rows = match_df[match_df['AwayTeam'] == team].sort_values('Date_x')
            last = rows.iloc[-1]
            return {'GF5':last['A_GF5'],'GA5':last['A_GA5'],'Pts5':last['A_Pts5'],
                    'Streak3':last['A_Streak3'],'Win5':last['A_Win5'],'CS5':last['A_CS5']}
        last = rows.iloc[-1]
        return {'GF5':last['H_GF5'],'GA5':last['H_GA5'],'Pts5':last['H_Pts5'],
                'Streak3':last['H_Streak3'],'Win5':last['H_Win5'],'CS5':last['H_CS5']}

    h_stats = latest_home_stats(home_team)
    a_stats = latest_home_stats(away_team)
    h_elo   = elo_dict.get(home_team, 1500)
    a_elo   = elo_dict.get(away_team, 1500)
    h2h_rows = match_df[(match_df['HomeTeam']==home_team)&(match_df['AwayTeam']==away_team)]
    h2h_rate = h2h_rows['H2H_HomeWinRate'].iloc[-1] if len(h2h_rows) > 0 else 0.33

    row = {
        'Diff_Pts':h_stats['Pts5']-a_stats['Pts5'],'Diff_GF':h_stats['GF5']-a_stats['GF5'],
        'Diff_GA':h_stats['GA5']-a_stats['GA5'],'Diff_Win':h_stats['Win5']-a_stats['Win5'],
        'Diff_CS':h_stats['CS5']-a_stats['CS5'],'Diff_Streak':h_stats['Streak3']-a_stats['Streak3'],
        'Diff_Elo':h_elo-a_elo,'H2H_HomeWinRate':h2h_rate,
        'H_GF5':h_stats['GF5'],'H_GA5':h_stats['GA5'],'H_Pts5':h_stats['Pts5'],
        'H_Streak3':h_stats['Streak3'],'A_GF5':a_stats['GF5'],'A_GA5':a_stats['GA5'],
        'A_Pts5':a_stats['Pts5'],'A_Streak3':a_stats['Streak3'],
    }
    X    = pd.DataFrame([row])[feats]
    X_sc = scl.transform(X)
    proba = mdl.predict_proba(X_sc)[0]
    pred  = mdl.predict(X_sc)[0]
    label_map = {0:'Away Win',1:'Draw',2:'Home Win'}
    result = {'Away Win':round(proba[0]*100,1),'Draw':round(proba[1]*100,1),
              'Home Win':round(proba[2]*100,1),'Prediction':label_map[pred],
              'Home_Elo':round(h_elo),'Away_Elo':round(a_elo)}
    print(f"\n{'='*45}")
    print(f"  ⚽  {home_team}  vs  {away_team}")
    print(f"{'='*45}")
    for label, pct in [('Home Win',result['Home Win']),('Draw    ',result['Draw']),('Away Win',result['Away Win'])]:
        bar = '█' * int(pct / 100 * 30)
        print(f"  {label}: {bar:<30} {pct}%")
    print(f"  🎯 Prediction: {result['Prediction']}")
    print(f"{'='*45}")
    return result


# ==============================
# 15) GW28 PREDICTION + FORM (last 5 scores)
# ==============================

# ── โปรแกรม GW28 — แก้ไขได้ที่นี่ ──────────────────────────
GW_FIXTURES = [
    {"date": "28/02/2026 03:00",  "home": "Wolves",         "away": "Aston Villa"},
    {"date": "28/02/2026 19:30",  "home": "Bournemouth",    "away": "Sunderland"},
    {"date": "28/02/2026 22:00",  "home": "Liverpool",      "away": "West Ham"},
    {"date": "28/02/2026 22:00",  "home": "Newcastle",      "away": "Everton"},
    {"date": "28/02/2026 22:00",  "home": "Burnley",        "away": "Brentford"},
    {"date": "01/03/2026 00:30",  "home": "Leeds",          "away": "Man City"},
    {"date": "01/03/2026 21:00",  "home": "Brighton",       "away": "Nott'm Forest"},
    {"date": "01/03/2026 21:00",  "home": "Fulham",         "away": "Tottenham"},
    {"date": "01/03/2026 21:00",  "home": "Man United",     "away": "Crystal Palace"},
    {"date": "01/03/2026 23:30",  "home": "Arsenal",        "away": "Chelsea"},
]

def get_last5_scores(team, raw_data):
    """
    คืน list ของ 5 แมตช์หลังสุด พร้อม opponent, score, result icon
    raw_data = dataframe ที่มี Date, HomeTeam, AwayTeam, FTHG, FTAG
    """
    mask = (raw_data['HomeTeam'] == team) | (raw_data['AwayTeam'] == team)
    matches = raw_data[mask].sort_values('Date').dropna(subset=['FTHG','FTAG']).tail(5)
    results = []
    for _, r in matches.iterrows():
        is_home = r['HomeTeam'] == team
        gf = int(r['FTHG']) if is_home else int(r['FTAG'])
        ga = int(r['FTAG']) if is_home else int(r['FTHG'])
        opp = r['AwayTeam'] if is_home else r['HomeTeam']
        venue = "H" if is_home else "A"
        icon = "✅" if gf > ga else ("🟡" if gf == ga else "❌")
        date_str = r['Date'].strftime('%d/%m')
        results.append(f"{icon} {date_str} {venue} vs {opp:<18} {gf}-{ga}")
    return results


def get_team_stats_for_gw(team):
    """ดึง rolling stats ล่าสุดของทีมจาก match_df"""
    hr = match_df[match_df['HomeTeam'] == team].sort_values('Date_x')
    if len(hr):
        r = hr.iloc[-1]
        return {'GF5':r['H_GF5'],'GA5':r['H_GA5'],'Pts5':r['H_Pts5'],
                'Win5':r['H_Win5'],'CS5':r['H_CS5'],'Streak3':r['H_Streak3']}
    ar = match_df[match_df['AwayTeam'] == team].sort_values('Date_x')
    if len(ar):
        r = ar.iloc[-1]
        return {'GF5':r['A_GF5'],'GA5':r['A_GA5'],'Pts5':r['A_Pts5'],
                'Win5':r['A_Win5'],'CS5':r['A_CS5'],'Streak3':r['A_Streak3']}
    return None


LABEL_MAP = {0: 'Away Win ✈️', 1: 'Draw 🤝', 2: 'Home Win 🏠'}
BAR_LEN   = 28

print("\n")
print("╔" + "═"*65 + "╗")
print("║   ⚽  GW28 — Premier League 2025-26                           ║")
print("║   AI ทำนาย + ฟอร์ม 5 นัดย้อนหลัง                            ║")
print("╚" + "═"*65 + "╝")

for i, fix in enumerate(GW_FIXTURES, 1):
    home, away, dt = fix['home'], fix['away'], fix['date']

    h_stats = get_team_stats_for_gw(home)
    a_stats = get_team_stats_for_gw(away)

    if h_stats is None or a_stats is None:
        missing = home if h_stats is None else away
        print(f"\n  ❌ ไม่พบข้อมูลทีม '{missing}'")
        continue

    h_elo = final_elo.get(home, 1500)
    a_elo = final_elo.get(away, 1500)

    h2h_rows = match_df[(match_df['HomeTeam']==home)&(match_df['AwayTeam']==away)]
    h2h_rate = float(h2h_rows['H2H_HomeWinRate'].iloc[-1]) if len(h2h_rows) > 0 else 0.33

    feat_row = {
        'Diff_Pts':    h_stats['Pts5']    - a_stats['Pts5'],
        'Diff_GF':     h_stats['GF5']     - a_stats['GF5'],
        'Diff_GA':     h_stats['GA5']     - a_stats['GA5'],
        'Diff_Win':    h_stats['Win5']    - a_stats['Win5'],
        'Diff_CS':     h_stats['CS5']     - a_stats['CS5'],
        'Diff_Streak': h_stats['Streak3'] - a_stats['Streak3'],
        'Diff_Elo':    h_elo - a_elo,
        'H2H_HomeWinRate': h2h_rate,
        'H_GF5':     h_stats['GF5'],  'H_GA5':  h_stats['GA5'],
        'H_Pts5':    h_stats['Pts5'], 'H_Streak3': h_stats['Streak3'],
        'A_GF5':     a_stats['GF5'],  'A_GA5':  a_stats['GA5'],
        'A_Pts5':    a_stats['Pts5'], 'A_Streak3': a_stats['Streak3'],
    }

    X     = pd.DataFrame([feat_row])[FEATURES]
    X_sc  = scaler.transform(X)
    proba = ensemble.predict_proba(X_sc)[0]
    pred  = ensemble.predict(X_sc)[0]
    pw, pd_prob, pa = proba[2]*100, proba[1]*100, proba[0]*100

    # --- last 5 scores ---
    h_form = get_last5_scores(home, data)
    a_form = get_last5_scores(away, data)

    print(f"\n{'─'*67}")
    print(f"  แมตช์ที่ {i:2d}  │  📅 {dt}")
    print(f"  🏠 {home:<20}  vs  ✈️  {away}")
    print(f"  Elo:  {home} {round(h_elo)}  |  {away} {round(a_elo)}")
    print(f"{'─'*67}")

    # ── ฟอร์ม 5 นัดย้อนหลัง ──
    max_rows = max(len(h_form), len(a_form))
    h_form_padded = h_form + [''] * (max_rows - len(h_form))
    a_form_padded = a_form + [''] * (max_rows - len(a_form))

    print(f"  {'🏠 ' + home + ' (5 นัดล่าสุด)':<34}  {'✈️  ' + away + ' (5 นัดล่าสุด)'}")
    print(f"  {'─'*32}  {'─'*32}")
    for hrow, arow in zip(h_form_padded, a_form_padded):
        print(f"  {hrow:<34}  {arow}")

    print(f"{'─'*67}")

    # ── ผลทำนาย ──
    for lbl, pct, idx in [
        (f'Home Win 🏠 {home}', pw,      2),
        ('Draw          🤝',    pd_prob, 1),
        (f'Away Win ✈️  {away}', pa,     0),
    ]:
        bar    = '█' * int(pct / 100 * BAR_LEN)
        marker = "  ◀ PICK" if idx == pred else ""
        print(f"    {lbl:<30}: {bar:<{BAR_LEN}} {pct:5.1f}%{marker}")

    print(f"{'─'*67}")
    print(f"  🎯 ผลที่ทำนาย: {LABEL_MAP[pred]}")

print(f"\n{'═'*67}")
print("  ✅ ครบทั้ง GW28  |  Model: LR + RF + XGB Ensemble")
print("  ⚠️  เพื่อความบันเทิงเท่านั้น — ไม่รับประกันความแม่นยำ")
print(f"{'═'*67}\n")