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

# ── เพิ่ม Features ใหม่: Elo ของแต่ละทีม + Scored/CS rate ──
match_df['Diff_Scored'] = match_df['H_Scored5'] - match_df['A_Scored5']
match_df['H_Elo_norm']  = match_df['H_Elo'] / 1500
match_df['A_Elo_norm']  = match_df['A_Elo'] / 1500
match_df['Elo_ratio']   = match_df['H_Elo'] / (match_df['A_Elo'] + 1)

FEATURES = [
    # Difference features
    'Diff_Pts', 'Diff_GF', 'Diff_GA', 'Diff_Win',
    'Diff_CS', 'Diff_Streak', 'Diff_Elo', 'Diff_Scored',
    # H2H
    'H2H_HomeWinRate',
    # Home team stats
    'H_GF5', 'H_GA5', 'H_Pts5', 'H_Streak3', 'H_CS5', 'H_Scored5',
    # Away team stats
    'A_GF5', 'A_GA5', 'A_Pts5', 'A_Streak3', 'A_CS5', 'A_Scored5',
    # Elo
    'H_Elo_norm', 'A_Elo_norm', 'Elo_ratio',
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
                'Scored5': last['A_Scored5'],
            }
        last = rows.iloc[-1]
        return {
            'GF5': last['H_GF5'], 'GA5': last['H_GA5'],
            'Pts5': last['H_Pts5'], 'Streak3': last['H_Streak3'],
            'Win5': last['H_Win5'], 'CS5': last['H_CS5'],
            'Scored5': last['H_Scored5'],
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
        'Diff_Scored':       h_stats['Scored5'] - a_stats['Scored5'],
        'H2H_HomeWinRate':   h2h_rate,
        'H_GF5':             h_stats['GF5'],
        'H_GA5':             h_stats['GA5'],
        'H_Pts5':            h_stats['Pts5'],
        'H_Streak3':         h_stats['Streak3'],
        'H_CS5':             h_stats['CS5'],
        'H_Scored5':         h_stats['Scored5'],
        'A_GF5':             a_stats['GF5'],
        'A_GA5':             a_stats['GA5'],
        'A_Pts5':            a_stats['Pts5'],
        'A_Streak3':         a_stats['Streak3'],
        'A_CS5':             a_stats['CS5'],
        'A_Scored5':         a_stats['Scored5'],
        'H_Elo_norm':        h_elo / 1500,
        'A_Elo_norm':        a_elo / 1500,
        'Elo_ratio':         h_elo / (a_elo + 1),
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
# 13) SEASON SIMULATION 2025-26
# ==============================

import datetime
TODAY = pd.Timestamp(datetime.date.today())

# ตัวแปร global สำหรับ final_table (ใช้ใน summary)
final_table      = None
remaining_fixtures = []

def get_latest_features(team, is_home):
    if is_home:
        rows = match_df[match_df['HomeTeam'] == team].sort_values('Date_x')
        if len(rows) > 0:
            last = rows.iloc[-1]
            return {'GF5': last['H_GF5'], 'GA5': last['H_GA5'],
                    'Pts5': last['H_Pts5'], 'Streak3': last['H_Streak3'],
                    'Win5': last['H_Win5'], 'CS5': last['H_CS5'],
                    'Scored5': last['H_Scored5']}
    rows = match_df[match_df['AwayTeam'] == team].sort_values('Date_x')
    if len(rows) > 0:
        last = rows.iloc[-1]
        return {'GF5': last['A_GF5'], 'GA5': last['A_GA5'],
                'Pts5': last['A_Pts5'], 'Streak3': last['A_Streak3'],
                'Win5': last['A_Win5'], 'CS5': last['A_CS5'],
                'Scored5': last['A_Scored5']}
    return {'GF5': 1.5, 'GA5': 1.5, 'Pts5': 1.5, 'Streak3': 1.5,
            'Win5': 0.5, 'CS5': 0.2, 'Scored5': 0.6}


def run_season_simulation():
    """รันการจำลองฤดูกาล — เรียกหลัง update_season_csv_from_api() เสมอ"""
    global final_table, remaining_fixtures

    season_file = pd.read_csv("data_set/season 2025.csv")
    season_file['Date'] = pd.to_datetime(season_file['Date'], dayfirst=True, errors='coerce')

    # แมตช์ที่แข่งแล้ว = มีผลจริง
    played = season_file.dropna(subset=['FTHG', 'FTAG']).copy()
    played = played[played['Date'] <= TODAY]

    # สร้าง remaining fixtures อัตโนมัติ
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
    print(f"⏳ แมตช์ยังไม่แข่ง: {len(unplayed)} นัด (สร้างจาก remaining fixtures อัตโนมัติ)")
    print(f"   รวม: {len(played) + len(unplayed)} นัด (ฤดูกาล 38 นัด × 20 ทีม = 380 นัด)")

    # คะแนนจริงจากแมตช์ที่แข่งแล้ว
    real_table = {}
    for _, row in played.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        hg, ag = int(row['FTHG']), int(row['FTAG'])
        for t in [home, away]:
            if t not in real_table: real_table[t] = 0
        if hg > ag:   real_table[home] += 3
        elif hg < ag: real_table[away] += 3
        else:
            real_table[home] += 1
            real_table[away] += 1

    real_table_df = pd.DataFrame.from_dict(real_table, orient='index', columns=['RealPoints'])

    pred_table = {}

    if len(unplayed) > 0:
        future_rows = []
        for _, match in unplayed.iterrows():
            home, away = match['HomeTeam'], match['AwayTeam']
            h = get_latest_features(home, is_home=True)
            a = get_latest_features(away, is_home=False)
            h_elo = final_elo.get(home, 1500)
            a_elo = final_elo.get(away, 1500)
            h2h_rows = match_df[(match_df['HomeTeam'] == home) & (match_df['AwayTeam'] == away)]
            h2h_rate = h2h_rows['H2H_HomeWinRate'].iloc[-1] if len(h2h_rows) > 0 else 0.33
            future_rows.append({
                'HomeTeam': home, 'AwayTeam': away,
                'Diff_Pts':     h['Pts5']     - a['Pts5'],
                'Diff_GF':      h['GF5']      - a['GF5'],
                'Diff_GA':      h['GA5']      - a['GA5'],
                'Diff_Win':     h['Win5']     - a['Win5'],
                'Diff_CS':      h['CS5']      - a['CS5'],
                'Diff_Streak':  h['Streak3']  - a['Streak3'],
                'Diff_Elo':     h_elo - a_elo,
                'Diff_Scored':  h['Scored5']  - a['Scored5'],
                'H2H_HomeWinRate': h2h_rate,
                'H_GF5': h['GF5'],     'H_GA5': h['GA5'],
                'H_Pts5': h['Pts5'],   'H_Streak3': h['Streak3'],
                'H_CS5': h['CS5'],     'H_Scored5': h['Scored5'],
                'A_GF5': a['GF5'],     'A_GA5': a['GA5'],
                'A_Pts5': a['Pts5'],   'A_Streak3': a['Streak3'],
                'A_CS5': a['CS5'],     'A_Scored5': a['Scored5'],
                'H_Elo_norm': h_elo / 1500,
                'A_Elo_norm': a_elo / 1500,
                'Elo_ratio':  h_elo / (a_elo + 1),
            })

        future_df = pd.DataFrame(future_rows)
        X_future = scaler.transform(future_df[FEATURES])
        future_df['Pred'] = ensemble.predict(X_future)
        print(f"🤖 ทำนาย {len(future_df)} แมตช์ที่เหลือ")

        for _, row in future_df.iterrows():
            home, away = row['HomeTeam'], row['AwayTeam']
            pred = row['Pred']
            for t in [home, away]:
                if t not in pred_table: pred_table[t] = 0
            if pred == 2:   pred_table[home] += 3
            elif pred == 1: pred_table[home] += 1; pred_table[away] += 1
            else:           pred_table[away] += 3
    else:
        print("ℹ️  ฤดูกาลจบแล้ว ไม่มีแมตช์ที่เหลือ")

    pred_table_df = pd.DataFrame.from_dict(pred_table, orient='index', columns=['PredictedPoints'])

    # รวมตาราง
    final_table = real_table_df.join(pred_table_df, how='left').fillna(0)
    final_table['PredictedPoints'] = final_table['PredictedPoints'].astype(int)
    final_table['FinalPoints']     = final_table['RealPoints'] + final_table['PredictedPoints']
    final_table.index.name = 'Team'

    # ── ตารางคะแนนจริงปัจจุบัน (เรียงตาม RealPoints) ──
    real_sorted = final_table.sort_values('RealPoints', ascending=False)
    played_count = len(played) // len(season_teams) if len(season_teams) > 0 else 0

    print(f"\n{'='*58}")
    print(f"  📊  ตารางคะแนนจริง ณ ปัจจุบัน  (ถึง {TODAY.date()})")
    print(f"{'='*58}")
    print(f"  {'#':<4} {'Team':<22} {'แข่ง':>5} {'แต้ม':>6}  {'สถานะ'}")
    print(f"  {'─'*55}")
    for rank, (team, row) in enumerate(real_sorted.iterrows(), 1):
        if rank <= 4:
            status = "🔴 CL Zone"
        elif rank <= 6:
            status = "🟠 Euro Zone"
        elif rank >= 18:
            status = "🟡 Relegation"
        else:
            status = ""
        print(f"  {rank:<4} {team:<22} {played_count:>5} {int(row['RealPoints']):>6}  {status}")
    print(f"  {'─'*55}")
    print(f"  🔴 CL  🟠 Europa  🟡 ตกชั้น")

    # ── ตารางคาดการณ์สิ้นฤดูกาล (เรียงตาม FinalPoints) ──
    final_sorted = final_table.sort_values('FinalPoints', ascending=False)

    print(f"\n{'='*62}")
    print(f"  🔮  ตารางคาดการณ์สิ้นฤดูกาล  (Real + AI ทำนาย {len(unplayed)} นัดที่เหลือ)")
    print(f"{'='*62}")
    print(f"  {'#':<4} {'Team':<22} {'แต้มจริง':>9} {'AI ทำนาย':>10} {'รวมคาด':>8}  {'สถานะ'}")
    print(f"  {'─'*60}")
    for rank, (team, row) in enumerate(final_sorted.iterrows(), 1):
        if rank <= 4:
            status = "🔴 CL Zone"
        elif rank <= 6:
            status = "🟠 Euro Zone"
        elif rank >= 18:
            status = "🟡 Relegation"
        else:
            status = ""
        arrow = "▲" if rank < list(real_sorted.index).index(team) + 1 else \
                ("▼" if rank > list(real_sorted.index).index(team) + 1 else "─")
        print(f"  {rank:<4} {team:<22} {int(row['RealPoints']):>9} {int(row['PredictedPoints']):>10} "
              f"{int(row['FinalPoints']):>8}  {arrow} {status}")
    print(f"  {'─'*60}")
    print(f"  🔴 CL  🟠 Europa  🟡 ตกชั้น  │  ▲ขึ้น ▼ลง ─คงที่ (เทียบตารางจริง)")

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
    valid_data = data.dropna(subset=['FTHG', 'FTAG']).copy()

    home_matches = valid_data[valid_data['HomeTeam'] == team][['Date','HomeTeam','AwayTeam','FTHG','FTAG']].copy()
    home_matches['Venue']    = 'H'
    home_matches['GF']       = home_matches['FTHG']
    home_matches['GA']       = home_matches['FTAG']
    home_matches['Opponent'] = home_matches['AwayTeam']

    away_matches = valid_data[valid_data['AwayTeam'] == team][['Date','HomeTeam','AwayTeam','FTHG','FTAG']].copy()
    away_matches['Venue']    = 'A'
    away_matches['GF']       = away_matches['FTAG']
    away_matches['GA']       = away_matches['FTHG']
    away_matches['Opponent'] = away_matches['HomeTeam']

    all_matches = pd.concat([home_matches, away_matches]).sort_values('Date', ascending=False)
    last5 = all_matches.head(5).copy()

    def result_label(row):
        if   row['GF'] > row['GA']: return 'W'
        elif row['GF'] == row['GA']: return 'D'
        else:                        return 'L'

    last5['Result'] = last5.apply(result_label, axis=1)

    icon_map = {'W': '✅ ชนะ', 'D': '🟡 เสมอ', 'L': '❌ แพ้'}
    print(f"\n{'='*58}")
    print(f"  📋  5 แมตช์ล่าสุดของ {team}")
    print(f"{'='*58}")
    print(f"  {'วันที่':<13} {'คู่แข่ง':<22} {'สนาม':<6} {'สกอร์':<10} {'ผล'}")
    print(f"  {'─'*55}")
    for _, row in last5.iterrows():
        date_str = row['Date'].strftime('%d/%m/%Y') if pd.notna(row['Date']) else 'N/A'
        score    = f"{int(row['GF'])}-{int(row['GA'])}"
        print(f"  {date_str:<13} {str(row['Opponent']):<22} {'เหย้า' if row['Venue']=='H' else 'เยือน':<6} {score:<10} {icon_map[row['Result']]}")
    print(f"{'='*58}")
    return last5[['Date','Opponent','Venue','GF','GA','Result']]


# ==============================
# 16) PREDICT SCORE (ทำนายสกอร์ + เปอร์เซนต์)
# ==============================

def predict_score(home_team, away_team):
    from scipy.stats import poisson

    teams_in_data = set(match_df['HomeTeam'].tolist() + match_df['AwayTeam'].tolist())
    if home_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{home_team}'"); return None
    if away_team not in teams_in_data:
        print(f"❌ ไม่พบทีม '{away_team}'"); return None

    def get_avg(team, is_home):
        if is_home:
            rows = match_df[match_df['HomeTeam'] == team].sort_values('Date_x')
            if len(rows) > 0:
                last = rows.iloc[-1]
                return last['H_GF5'], last['H_GA5']
        rows = match_df[match_df['AwayTeam'] == team].sort_values('Date_x')
        if len(rows) > 0:
            last = rows.iloc[-1]
            return last['A_GF5'], last['A_GA5']
        return 1.5, 1.5

    h_gf, h_ga = get_avg(home_team, True)
    a_gf, a_ga = get_avg(away_team, False)

    lg_home = data['FTHG'].mean()
    lg_away = data['FTAG'].mean()

    home_xg = (h_gf / lg_home) * (a_ga / lg_home) * lg_home
    away_xg = (a_gf / lg_away) * (h_ga / lg_away) * lg_away

    score_probs = {}
    for hg in range(7):
        for ag in range(7):
            score_probs[f"{hg}-{ag}"] = round(poisson.pmf(hg, home_xg) * poisson.pmf(ag, away_xg) * 100, 2)

    top5 = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n  ⚽ xG คาด:  {home_team} {round(home_xg,2)}  vs  {away_team} {round(away_xg,2)}")
    print(f"  🎯 สกอร์ที่น่าจะเป็น (Top 5):")
    for score, pct in top5:
        bar = '█' * int(pct * 2)
        print(f"     {score:<8} {bar:<20} {pct}%")

    return {'home_xg': round(home_xg,2), 'away_xg': round(away_xg,2),
            'most_likely_score': top5[0][0], 'top5_scores': top5}


# ==============================
# 17) PREDICT NEXT 5 MATCHES (ฟังก์ชันหลัก)
# ==============================

def predict_next_5_matches(team, fixtures=None):
    """
    ทำนาย 5 แมตช์ข้างหน้าของทีม
    
    fixtures: list of dict ระบุตารางแข่งจริง (ถ้าไม่ใส่จะเดาเอง)
    ตัวอย่าง:
        fixtures = [
            {'HomeTeam': 'Arsenal',  'AwayTeam': 'Chelsea'},
            {'HomeTeam': 'Liverpool','AwayTeam': 'Arsenal'},
        ]
    """
    print(f"\n{'#'*62}")
    print(f"  🔮  วิเคราะห์ทีม: {team.upper()}")
    print(f"{'#'*62}")

    # ── ผลย้อนหลัง 5 แมตช์ ──
    last5_df = get_last_5_results(team)

    # ── หา 5 แมตช์ข้างหน้า ──
    if fixtures:
        # ใช้ตารางแข่งที่ user ระบุมา
        next5 = [f for f in fixtures
                 if f['HomeTeam'] == team or f['AwayTeam'] == team][:5]
        print(f"  ✅ ใช้ตารางแข่งที่ระบุ ({len(next5)} นัด)")
    else:
        # fallback: เดาจาก remaining_fixtures (เรียงตามตัวอักษร ไม่มีวันที่)
        next5 = [f for f in remaining_fixtures
                 if f['HomeTeam'] == team or f['AwayTeam'] == team][:5]
        print(f"  ⚠️  ไม่ได้ระบุตารางแข่ง → ใช้การเดาอัตโนมัติ (อาจไม่ตรงวันจริง)")

    if not next5:
        print(f"\n⚠️  ไม่พบแมตช์ข้างหน้าของ {team}")
        return None

    print(f"\n{'='*62}")
    print(f"  🔮  5 แมตช์ข้างหน้า: {team}")
    print(f"{'='*62}")

    predictions = []
    for i, match in enumerate(next5, 1):
        home_team_m = match['HomeTeam']
        away_team_m = match['AwayTeam']
        is_home     = (home_team_m == team)
        opponent    = away_team_m if is_home else home_team_m
        venue_th    = 'เหย้า' if is_home else 'เยือน'

        print(f"\n  นัดที่ {i}  |  {home_team_m}  vs  {away_team_m}  ({venue_th})")
        print(f"  {'─'*58}")

        result_pred = predict_match(home_team_m, away_team_m)
        score_pred  = predict_score(home_team_m, away_team_m)

        if result_pred and score_pred:
            if is_home:
                win_pct  = result_pred['Home Win']
                draw_pct = result_pred['Draw']
                loss_pct = result_pred['Away Win']
                outcome  = result_pred['Prediction']
            else:
                win_pct  = result_pred['Away Win']
                draw_pct = result_pred['Draw']
                loss_pct = result_pred['Home Win']
                flip     = {'Home Win': 'Away Win', 'Away Win': 'Home Win', 'Draw': 'Draw'}
                outcome  = flip.get(result_pred['Prediction'], result_pred['Prediction'])

            is_win  = (is_home and outcome == 'Home Win') or (not is_home and outcome == 'Away Win')
            is_draw = outcome == 'Draw'
            result_th = f"✅ {team} ชนะ" if is_win else ("🟡 เสมอ" if is_draw else f"❌ {team} แพ้")

            print(f"\n  📌 ผลที่น่าจะเป็น : {result_th}")
            print(f"  📊 ชนะ {win_pct}%  |  เสมอ {draw_pct}%  |  แพ้ {loss_pct}%")
            print(f"  🎯 สกอร์คาด      : {score_pred['most_likely_score']}")

            predictions.append({
                'match_no': i, 'home': home_team_m, 'away': away_team_m,
                'venue': venue_th, 'opponent': opponent,
                'win_pct': win_pct, 'draw_pct': draw_pct, 'loss_pct': loss_pct,
                'predicted_result': outcome,
                'predicted_score': score_pred['most_likely_score'],
            })

    # ── ตารางสรุป ──
    print(f"\n{'#'*62}")
    print(f"  📋  สรุป 5 แมตช์ข้างหน้า: {team}")
    print(f"{'#'*62}")
    print(f"  {'นัด':<5} {'คู่แข่ง':<24} {'สนาม':<7} {'ชนะ%':<8} {'เสมอ%':<8} {'แพ้%':<8} {'สกอร์คาด'}")
    print(f"  {'─'*68}")
    for p in predictions:
        print(f"  {p['match_no']:<5} {p['opponent']:<24} {p['venue']:<7} "
              f"{p['win_pct']:<8} {p['draw_pct']:<8} {p['loss_pct']:<8} {p['predicted_score']}")
    print(f"{'#'*62}\n")

    return {'next_5': predictions, 'last_5': last5_df}


# ==============================
# 18) ดึงตารางแข่งจริงจาก football-data.org API
# ==============================
# ✅ วิธีขอ API Key ฟรี:
#    1. ไปที่ https://www.football-data.org/client/register
#    2. สมัครฟรี (ใช้อีเมลสมัคร)
#    3. เช็คอีเมล จะได้ API key มา
#    4. วาง key ตรง API_KEY ด้านล่าง

import requests

API_KEY = "745c5b802b204590bfa05c093f00bd43"   # ← วาง key ที่ได้จาก football-data.org ตรงนี้

# แผนที่ชื่อทีม API → ชื่อในข้อมูลเรา
TEAM_NAME_MAP = {
    "Arsenal FC":                  "Arsenal",
    "Aston Villa FC":              "Aston Villa",
    "AFC Bournemouth":             "Bournemouth",
    "Brentford FC":                "Brentford",
    "Brighton & Hove Albion FC":   "Brighton",
    "Burnley FC":                  "Burnley",
    "Chelsea FC":                  "Chelsea",
    "Crystal Palace FC":           "Crystal Palace",
    "Everton FC":                  "Everton",
    "Fulham FC":                   "Fulham",
    "Leeds United FC":             "Leeds",
    "Liverpool FC":                "Liverpool",
    "Manchester City FC":          "Man City",
    "Manchester United FC":        "Man United",
    "Newcastle United FC":         "Newcastle",
    "Nottingham Forest FC":        "Nott'm Forest",
    "Sunderland AFC":              "Sunderland",
    "Tottenham Hotspur FC":        "Tottenham",
    "West Ham United FC":          "West Ham",
    "Wolverhampton Wanderers FC":  "Wolves",
}

def normalize(name):
    return TEAM_NAME_MAP.get(name, name)


def fetch_fixtures_from_api(target_team, num_matches=5):
    """
    ดึงตารางแข่ง Premier League ที่ยังไม่แข่งจาก football-data.org
    - PL competition id = PL
    - ดึงเฉพาะ SCHEDULED (ยังไม่แข่ง)
    """
    if API_KEY == "YOUR_API_KEY_HERE":
        print("  ❌ ยังไม่ได้ใส่ API Key!")
        print("  👉 สมัครฟรีที่ https://www.football-data.org/client/register")
        print("  👉 แล้วแก้ API_KEY = 'your_key_here' ในโค้ด")
        return None

    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}
    params  = {"status": "SCHEDULED"}          # เฉพาะนัดที่ยังไม่แข่ง

    try:
        print(f"  🌐 ดึงข้อมูลจาก football-data.org API...")
        r = requests.get(url, headers=headers, params=params, timeout=10)

        if r.status_code == 401:
            print("  ❌ API Key ไม่ถูกต้อง หรือยังไม่ activate")
            return None
        if r.status_code == 429:
            print("  ❌ เกิน rate limit (ฟรี = 10 req/min) — รอสักครู่แล้วลองใหม่")
            return None
        r.raise_for_status()

        data     = r.json()
        matches  = data.get("matches", [])
        print(f"  ✅ ดึงได้ {len(matches)} นัดที่ยังไม่แข่ง")

        # แปลงชื่อและกรองเฉพาะทีมที่ต้องการ
        all_fixtures = []
        for m in matches:
            home = normalize(m["homeTeam"]["name"])
            away = normalize(m["awayTeam"]["name"])
            date = m["utcDate"][:10]   # YYYY-MM-DD
            all_fixtures.append({
                "HomeTeam": home,
                "AwayTeam": away,
                "Date":     date,
            })

        # กรองเฉพาะทีมที่ต้องการ เรียงตามวันที่
        team_fixtures = [
            f for f in all_fixtures
            if f["HomeTeam"] == target_team or f["AwayTeam"] == target_team
        ][:num_matches]

        if not team_fixtures:
            print(f"  ❌ ไม่พบนัดของ '{target_team}'")
            print(f"  ชื่อทีมที่ API รู้จัก: {sorted(set([f['HomeTeam'] for f in all_fixtures]))}")
            return None

        print(f"\n  📅 ตารางแข่ง {num_matches} นัดข้างหน้าของ {target_team}:")
        print(f"  {'นัด':<5} {'วันที่':<14} {'เหย้า':<22} {'เยือน':<22} {'สนาม'}")
        print(f"  {'─'*65}")
        for i, f in enumerate(team_fixtures, 1):
            venue = "เหย้า" if f["HomeTeam"] == target_team else "เยือน"
            print(f"  {i:<5} {f['Date']:<14} {f['HomeTeam']:<22} {f['AwayTeam']:<22} {venue}")

        return team_fixtures

    except requests.exceptions.ConnectionError:
        print("  ❌ เชื่อมต่ออินเทอร์เน็ตไม่ได้")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def update_season_csv_from_api():
    """
    ดึงตารางแข่ง PL ทั้งหมด (แข่งแล้ว + ยังไม่แข่ง)
    จาก API แล้วเขียนทับ season 2025.csv วันที่แม่น 100%
    """
    from datetime import datetime, timedelta

    url     = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}

    try:
        print("\n" + "="*55)
        print("  📥  อัปเดต season 2025.csv จาก API...")
        r = requests.get(url, headers=headers,
                         params={"season": "2025"}, timeout=15)
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

            rows.append({
                "Date":     date_str,
                "HomeTeam": normalize(m["homeTeam"]["name"]),
                "AwayTeam": normalize(m["awayTeam"]["name"]),
                "FTHG":     hg,
                "FTAG":     ag,
                "FTR":      ftr,
            })

        df_new = pd.DataFrame(rows)
        played   = len(df_new[df_new["FTHG"] != ""])
        upcoming = len(df_new[df_new["FTHG"] == ""])
        df_new.to_csv("data_set/season 2025.csv", index=False)
        print(f"  ✅ แข่งแล้ว {played} นัด | ยังไม่แข่ง {upcoming} นัด")
        print(f"  💾 บันทึก → data_set/season 2025.csv")
        print("="*55)
        return df_new

    except requests.exceptions.ConnectionError:
        print("  ❌ ไม่มีอินเทอร์เน็ต — ใช้ข้อมูลเดิม")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# alias
def fetch_all_pl_fixtures():
    return update_season_csv_from_api()


# ==============================
# 18) predict_with_api
# ==============================

def predict_with_api(team, num_matches=5):
    SEP = '=' * 62
    print()
    print(SEP)
    print('  🔮  ทำนาย ' + str(num_matches) + ' แมตช์ข้างหน้า: ' + team)
    print(SEP)
    fixtures = fetch_fixtures_from_api(team, num_matches)
    if fixtures:
        predict_next_5_matches(team, fixtures=fixtures)
    else:
        print('  ⚠️  fallback: ใช้การเดาอัตโนมัติ')
        predict_next_5_matches(team)


# ==============================
# 19) ตารางแข่ง PL N นัดถัดไป + ทำนายทุกแมตช์
# ==============================

def show_next_pl_fixtures(num_matches=5):
    """แสดงตารางแข่ง PL N นัดถัดไปพร้อมทำนายผลทุกแมตช์"""

    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ ยังไม่ได้ใส่ API Key!")
        return

    SEP  = "=" * 65

    url     = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}
    params  = {"status": "SCHEDULED"}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        matches = r.json().get("matches", [])
        matches = sorted(matches, key=lambda x: x["utcDate"])[:num_matches]

        if not matches:
            print("  ⚠️  ไม่พบแมตช์ที่ยังไม่แข่ง")
            return

        from datetime import datetime, timedelta

        # ── แสดงตาราง ──
        print()
        print(SEP)
        print(f"  📅  ตารางแข่ง Premier League {num_matches} นัดถัดไป")
        print(SEP)
        print(f"  {'นัด':<5} {'วันที่':<14} {'เวลา(TH)':<11} {'เหย้า':<22} {'เยือน'}")
        print("  " + "-" * 60)

        upcoming = []
        for i, m in enumerate(matches, 1):
            home = normalize(m["homeTeam"]["name"])
            away = normalize(m["awayTeam"]["name"])
            utc_dt   = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            th_dt    = utc_dt + timedelta(hours=7)
            date_str = th_dt.strftime("%d/%m/%Y")
            time_str = th_dt.strftime("%H:%M")
            print(f"  {i:<5} {date_str:<14} {time_str:<11} {home:<22} {away}")
            upcoming.append({"HomeTeam": home, "AwayTeam": away,
                             "Date": date_str, "Time": time_str})

        # ── ทำนายทุกแมตช์ ──
        print()
        print(SEP)
        print(f"  🤖  ผลทำนาย {num_matches} นัดถัดไป")
        print(SEP)
        print(f"  {'นัด':<5} {'เหย้า':<20} {'vs':^4} {'เยือน':<20} "
              f"{'ชนะ%':>7} {'เสมอ%':>7} {'แพ้%':>7}  {'สกอร์'}")
        print("  " + "-" * 75)

        teams_ok = set(match_df["HomeTeam"].tolist() + match_df["AwayTeam"].tolist())

        for i, f in enumerate(upcoming, 1):
            home, away = f["HomeTeam"], f["AwayTeam"]
            if home not in teams_ok or away not in teams_ok:
                print(f"  {i:<5} {home:<20} {'vs':^4} {away:<20}  ⚠️ ไม่มีข้อมูล")
                continue

            r_pred = predict_match(home, away)
            s_pred = predict_score(home, away)

            if r_pred and s_pred:
                hw    = r_pred["Home Win"]
                dr    = r_pred["Draw"]
                aw    = r_pred["Away Win"]
                pred  = r_pred["Prediction"]
                icon  = "🏠" if pred == "Home Win" else ("🤝" if pred == "Draw" else "✈️")
                score = s_pred["most_likely_score"]
                print(f"  {i:<5} {home:<20} {'vs':^4} {away:<20} "
                      f"{hw:>7} {dr:>7} {aw:>7}  {icon} {score}")

        print("  " + "-" * 75)
        print("  🏠 เหย้าชนะ  🤝 เสมอ  ✈️ เยือนชนะ")
        print(SEP)
        print()
        return upcoming

    except requests.exceptions.ConnectionError:
        print("  ❌ เชื่อมต่ออินเทอร์เน็ตไม่ได้")
    except Exception as e:
        print(f"  ❌ Error: {e}")


# ==============================
# 🚀 เรียกใช้งาน
# ==============================

# ── STEP 1: อัปเดตตารางแข่งจาก API (วันที่แม่น 100%) ──
# รันทุกครั้งเพื่อให้ข้อมูลล่าสุด
update_season_csv_from_api()

# ── STEP 2: ทำนายทีมที่ต้องการ ──
predict_with_api("Arsenal")
# predict_with_api("Liverpool")
# predict_with_api("Man City")
# predict_with_api("Chelsea")
# predict_with_api("Aston Villa")

# ── แสดงตารางแข่ง PL ถัดไปพร้อมทำนาย ──
show_next_pl_fixtures(5)    # 5 นัดถัดไป
# show_next_pl_fixtures(10)  # 10 นัด
# show_next_pl_fixtures(20)  # 20 นัด


# ==============================
# 20) FULL SUMMARY REPORT
# ==============================

def print_full_summary():
    SEP  = "=" * 65
    LINE = "─" * 65

    print()
    print("█" * 65)
    print("  📊  FOOTBALL AI — FULL SUMMARY REPORT")
    print(f"  🗓️  วันที่รายงาน: {TODAY.date()}  |  ข้อมูลถึง: {data['Date'].max().date()}")
    print("█" * 65)

    # ── 1. ข้อมูลโดยรวม ──────────────────────────────────────────
    print()
    print(SEP)
    print("  📁  1. ข้อมูลที่ใช้เทรน")
    print(SEP)
    total_seasons = data['Date'].dt.year.nunique()
    teams_count   = data['HomeTeam'].nunique()
    print(f"  • แมตช์ทั้งหมด    : {len(data):,} นัด ({total_seasons} ฤดูกาล)")
    print(f"  • จำนวนทีม        : {teams_count} ทีม")
    print(f"  • ช่วงเวลา        : {data['Date'].min().date()} → {data['Date'].max().date()}")
    print(f"  • แมตช์เทรน (80%) : {len(train):,} นัด")
    print(f"  • แมตช์เทสต์(20%) : {len(test):,} นัด")
    print(f"  • Features ที่ใช้  : {len(FEATURES)} ตัว")

    # ── 2. Model Performance ──────────────────────────────────────
    print()
    print(SEP)
    print("  🤖  2. ประสิทธิภาพโมเดล (Ensemble: LR + RF + XGB)")
    print(SEP)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(f"  • Accuracy บน Test Set  : {acc}%")

    cm = confusion_matrix(y_test, y_pred)
    labels = ['Away Win', 'Draw', 'Home Win']
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>14}", end="")
    for l in labels:
        print(f"  {l:>10}", end="")
    print()
    for i, label in enumerate(labels):
        print(f"  {'Actual ':>7}{label:>9}  ", end="")
        for j in range(3):
            print(f"  {cm[i][j]:>10}", end="")
        print()

    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    print(f"\n  {'ผลลัพธ์':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {LINE}")
    for label in labels:
        r = report[label]
        print(f"  {label:<15} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f} {int(r['support']):>10}")
    print(f"  {LINE}")
    print(f"  {'Accuracy':<15} {'':>10} {'':>10} {report['accuracy']:>10.2f} {int(report['macro avg']['support']):>10}")

    # ── 3. Elo Rating Top 10 ──────────────────────────────────────
    print()
    print(SEP)
    print("  🏆  3. Elo Rating ล่าสุด (Top 10)")
    print(SEP)
    elo_sorted = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  {'#':<5} {'ทีม':<25} {'Elo':>8}  {'Bar'}")
    print(f"  {LINE}")
    max_elo = elo_sorted[0][1]
    for rank, (team, elo_val) in enumerate(elo_sorted, 1):
        bar = '█' * int((elo_val / max_elo) * 20)
        marker = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"{rank:<2} "))
        print(f"  {marker}   {team:<25} {round(elo_val):>8}  {bar}")

    # ── 4. ตาราง Season 2025-26 สรุป ─────────────────────────────
    print()
    print(SEP)
    print("  📋  4. ตารางคาดการณ์สิ้นฤดูกาล Season 2025-26")
    print(SEP)
    print(f"  {'#':<5} {'ทีม':<22} {'คะแนนจริง':>10} {'คะแนนทำนาย':>12} {'รวม':>7}  {'สถานะ'}")
    print(f"  {LINE}")
    final_sorted = final_table.sort_values('FinalPoints', ascending=False)
    for rank, (team, row) in enumerate(final_sorted.iterrows(), 1):
        if rank <= 4:
            status = "🔴 Champions League"
        elif rank <= 6:
            status = "🟠 Europa / Conf."
        elif rank >= 18:
            status = "🟡 ตกชั้น"
        else:
            status = ""
        print(f"  {rank:<5} {team:<22} {int(row['RealPoints']):>10} {int(row['PredictedPoints']):>12} {int(row['FinalPoints']):>7}  {status}")
    print(f"  {LINE}")
    print(f"  🔴 Top 4 = UEFA CL  |  🟠 Top 5-6 = Europa/Conf.  |  🟡 18-20 = ตกชั้น")

    # ── 5. สถิติทั่วไปของดาต้า ────────────────────────────────────
    print()
    print(SEP)
    print("  📈  5. สถิติน่าสนใจจากข้อมูล")
    print(SEP)
    valid = data.dropna(subset=['FTHG', 'FTAG'])
    home_wins  = (valid['FTHG'] > valid['FTAG']).sum()
    draws      = (valid['FTHG'] == valid['FTAG']).sum()
    away_wins  = (valid['FTHG'] < valid['FTAG']).sum()
    total_v    = len(valid)
    avg_goals  = (valid['FTHG'] + valid['FTAG']).mean()
    avg_home   = valid['FTHG'].mean()
    avg_away   = valid['FTAG'].mean()

    print(f"  • เหย้าชนะ      : {home_wins:,} นัด ({home_wins/total_v*100:.1f}%)")
    print(f"  • เสมอ          : {draws:,} นัด ({draws/total_v*100:.1f}%)")
    print(f"  • เยือนชนะ      : {away_wins:,} นัด ({away_wins/total_v*100:.1f}%)")
    print(f"  • เฉลี่ยประตู/นัด: {avg_goals:.2f} ประตู  (เหย้า {avg_home:.2f} | เยือน {avg_away:.2f})")

    # ทีมที่ยิงได้มากที่สุด
    goals_scored = valid.groupby('HomeTeam')['FTHG'].sum() + valid.groupby('AwayTeam')['FTAG'].sum()
    goals_conceded = valid.groupby('HomeTeam')['FTAG'].sum() + valid.groupby('AwayTeam')['FTHG'].sum()
    top_scorer   = goals_scored.idxmax()
    top_conceded = goals_conceded.idxmax()
    print(f"  • ทีมยิงมากสุด  : {top_scorer} ({int(goals_scored[top_scorer])} ประตู)")
    print(f"  • ทีมเสียมากสุด : {top_conceded} ({int(goals_conceded[top_conceded])} ประตู)")

    # ── 6. สรุปโมเดล & คำแนะนำ ───────────────────────────────────
    print()
    print(SEP)
    print("  💡  6. สรุปและคำแนะนำ")
    print(SEP)
    print(f"  • โมเดล Ensemble (LR + RF + XGB) ทำได้ {acc}% accuracy")
    print(f"  • ทำนาย Home Win ได้ดีที่สุด (F1 ≈ {report['Home Win']['f1-score']:.2f})")
    print(f"  • ทำนาย Draw ได้ยากที่สุด (F1 ≈ {report['Draw']['f1-score']:.2f}) — เป็นปัญหาปกติของ ML ฟุตบอล")
    print(f"  • ใช้ {len(FEATURES)} features: Rolling form, Elo, H2H, CS rate, Scoring rate")
    print(f"  • ข้อแนะนำ: เพิ่มข้อมูล (injury, weather, referee) จะช่วยเพิ่ม accuracy ได้")

    print()
    print("█" * 65)
    print("  ✅  END OF REPORT")
    print("█" * 65)
    print()


# ==============================
# PHASE 2 — COMPETITION GRADE
# ==============================

# ──────────────────────────────────────────────────────────────
# P2-1) MONTE CARLO SEASON SIMULATION (1,000 รอบ)
#       + Top4 / Relegation probability
# ──────────────────────────────────────────────────────────────

def run_monte_carlo(n_simulations=1000, verbose=True):
    """
    จำลองฤดูกาลที่เหลือ n_simulations รอบ โดยสุ่มผลตาม
    probability ของโมเดล แล้วรวมกับคะแนนจริงที่ทำแล้ว

    Returns:
        dict  {team: {'top4': %, 'top6': %, 'relegation': %, 'mean_pts': float, 'std_pts': float}}
    """
    if final_table is None:
        print("❌ กรุณาเรียก run_season_simulation() ก่อน")
        return None

    SEP  = "=" * 65
    LINE = "─" * 65

    if verbose:
        print()
        print(SEP)
        print(f"  🎲  MONTE CARLO SEASON SIMULATION  ({n_simulations:,} รอบ)")
        print(SEP)
        print(f"  กำลังจำลอง {len(remaining_fixtures)} แมตช์ที่เหลือ × {n_simulations:,} รอบ ...")

    # สร้าง feature matrix ของ remaining fixtures ครั้งเดียว
    if not remaining_fixtures:
        if verbose:
            print("  ℹ️  ฤดูกาลจบแล้ว ไม่มีแมตช์ที่เหลือ")
        return None

    future_rows = []
    for match in remaining_fixtures:
        home, away = match['HomeTeam'], match['AwayTeam']
        h = get_latest_features(home, is_home=True)
        a = get_latest_features(away, is_home=False)
        h_elo = final_elo.get(home, 1500)
        a_elo = final_elo.get(away, 1500)
        h2h_rows = match_df[(match_df['HomeTeam'] == home) & (match_df['AwayTeam'] == away)]
        h2h_rate = h2h_rows['H2H_HomeWinRate'].iloc[-1] if len(h2h_rows) > 0 else 0.33
        future_rows.append({
            'HomeTeam': home, 'AwayTeam': away,
            'Diff_Pts': h['Pts5'] - a['Pts5'],
            'Diff_GF':  h['GF5']  - a['GF5'],
            'Diff_GA':  h['GA5']  - a['GA5'],
            'Diff_Win': h['Win5'] - a['Win5'],
            'Diff_CS':  h['CS5']  - a['CS5'],
            'Diff_Streak': h['Streak3'] - a['Streak3'],
            'Diff_Elo':    h_elo - a_elo,
            'Diff_Scored': h['Scored5'] - a['Scored5'],
            'H2H_HomeWinRate': h2h_rate,
            'H_GF5': h['GF5'],     'H_GA5': h['GA5'],
            'H_Pts5': h['Pts5'],   'H_Streak3': h['Streak3'],
            'H_CS5': h['CS5'],     'H_Scored5': h['Scored5'],
            'A_GF5': a['GF5'],     'A_GA5': a['GA5'],
            'A_Pts5': a['Pts5'],   'A_Streak3': a['Streak3'],
            'A_CS5': a['CS5'],     'A_Scored5': a['Scored5'],
            'H_Elo_norm': h_elo / 1500,
            'A_Elo_norm': a_elo / 1500,
            'Elo_ratio':  h_elo / (a_elo + 1),
        })

    future_df = pd.DataFrame(future_rows)
    X_future_sc = scaler.transform(future_df[FEATURES])

    # ดึง probability ทุกแมตช์ครั้งเดียว shape (n_matches, 3)
    # class order: 0=Away Win, 1=Draw, 2=Home Win
    proba_matrix = ensemble.predict_proba(X_future_sc)   # (n_matches, 3)

    all_teams = list(final_table.index)
    real_pts  = {t: int(final_table.loc[t, 'RealPoints']) for t in all_teams}

    # ตัวนับ
    counts = {t: {'top4': 0, 'top6': 0, 'relegation': 0, 'pts_sum': 0.0, 'pts_sq': 0.0}
              for t in all_teams}

    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        sim_pts = dict(real_pts)  # เริ่มต้นจากคะแนนจริง

        for idx, match in enumerate(remaining_fixtures):
            home, away = match['HomeTeam'], match['AwayTeam']
            p_away, p_draw, p_home = proba_matrix[idx]
            # normalize เพื่อแก้ floating point precision
            probs = np.array([p_away, p_draw, p_home], dtype=np.float64)
            probs /= probs.sum()

            # สุ่มผล
            outcome = rng.choice([0, 1, 2], p=probs)
            if outcome == 2:   sim_pts[home] += 3
            elif outcome == 1: sim_pts[home] += 1; sim_pts[away] += 1
            else:              sim_pts[away] += 3

        # เรียงอันดับในรอบนี้
        ranked = sorted(sim_pts.items(), key=lambda x: x[1], reverse=True)
        for rank, (team, pts) in enumerate(ranked, 1):
            if rank <= 4:  counts[team]['top4'] += 1
            if rank <= 6:  counts[team]['top6'] += 1
            if rank >= 18: counts[team]['relegation'] += 1
            counts[team]['pts_sum'] += pts
            counts[team]['pts_sq']  += pts ** 2

    # คำนวณสถิติ
    results = {}
    for t in all_teams:
        c = counts[t]
        mean = c['pts_sum'] / n_simulations
        std  = ((c['pts_sq'] / n_simulations) - mean ** 2) ** 0.5
        results[t] = {
            'top4':       round(c['top4']       / n_simulations * 100, 1),
            'top6':       round(c['top6']       / n_simulations * 100, 1),
            'relegation': round(c['relegation'] / n_simulations * 100, 1),
            'mean_pts':   round(mean, 1),
            'std_pts':    round(std,  1),
        }

    if not verbose:
        return results

    # ── แสดงผล Top 4 / Relegation ──────────────────────────────
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_pts'], reverse=True)

    print(f"\n  {'Team':<22} {'Mean Pts':>9} {'±Std':>6} {'Top4%':>7} {'Top6%':>7} {'Rel%':>7}  {'Bar (Top4)'}")
    print(f"  {LINE}")

    for team, r in sorted_results:
        bar_top4 = '█' * int(r['top4'] / 5)   # 1 block = 5%
        bar_rel  = '▓' * int(r['relegation'] / 5)
        bar      = bar_top4 if r['top4'] >= r['relegation'] else bar_rel
        color_t4  = "🔴" if r['top4']       >= 60 else ("🟡" if r['top4']       >= 20 else "  ")
        color_rel = "🟡" if r['relegation'] >= 60 else ("⚠️ " if r['relegation'] >= 20 else "  ")
        print(f"  {team:<22} {r['mean_pts']:>9} {r['std_pts']:>6} "
              f"{color_t4}{r['top4']:>5}%  {r['top6']:>6}%  "
              f"{color_rel}{r['relegation']:>4}%  {bar}")

    print(f"  {LINE}")

    # ── Top 4 Champion Odds ──────────────────────────────────────
    print(f"\n  🏆  TITLE RACE (ชนะเลิศ: จบอันดับ 1)")
    print(f"  {LINE}")
    title_counts = {t: 0 for t in all_teams}
    rng2 = np.random.default_rng(99)
    for _ in range(n_simulations):
        sim_pts = dict(real_pts)
        for idx, match in enumerate(remaining_fixtures):
            home, away = match['HomeTeam'], match['AwayTeam']
            p_away, p_draw, p_home = proba_matrix[idx]
            probs = np.array([p_away, p_draw, p_home], dtype=np.float64)
            probs /= probs.sum()
            outcome = rng2.choice([0, 1, 2], p=probs)
            if outcome == 2:   sim_pts[home] += 3
            elif outcome == 1: sim_pts[home] += 1; sim_pts[away] += 1
            else:              sim_pts[away] += 3
        champion = max(sim_pts, key=sim_pts.get)
        title_counts[champion] += 1

    title_sorted = sorted(title_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    for team, cnt in title_sorted:
        pct = round(cnt / n_simulations * 100, 1)
        bar = '█' * int(pct / 2)
        if pct > 0:
            print(f"  {'🥇 ' if pct == max(c for _, c in title_sorted[:1]) / n_simulations * 100 else '   '}"
                  f"{team:<22} {bar:<30} {pct}%")

    print(f"\n  ✅ Monte Carlo เสร็จสิ้น ({n_simulations:,} simulations)")
    print(SEP)
    return results


# ──────────────────────────────────────────────────────────────
# P2-2) DRAW CALIBRATION ANALYSIS
#       วัดว่าโมเดลทำนาย Draw ดีแค่ไหน vs ความจริง
# ──────────────────────────────────────────────────────────────

def analyze_draw_calibration():
    """
    แบ่ง predicted Draw probability เป็น bins
    แล้วดูว่า actual draw rate ใน bin นั้นเป็นเท่าไร
    หากเส้น calibration ชิดเส้น y=x แสดงว่าโมเดล well-calibrated
    """
    from sklearn.calibration import calibration_curve

    SEP  = "=" * 65
    LINE = "─" * 65

    print()
    print(SEP)
    print("  📐  DRAW CALIBRATION ANALYSIS")
    print(SEP)

    # ดึง probability ของ Draw (class 1) บน test set
    draw_proba = ensemble.predict_proba(X_test_sc)[:, 1]
    actual_draw = (y_test == 1).astype(int).values

    # Calibration curve (5 bins)
    n_bins = 8
    fraction_of_positives, mean_predicted_value = calibration_curve(
        actual_draw, draw_proba, n_bins=n_bins, strategy='quantile'
    )

    print(f"\n  Predicted%   Actual%    Diff     Calibration Bar")
    print(f"  {LINE}")

    total_brier = 0.0
    for pred_p, act_p in zip(mean_predicted_value, fraction_of_positives):
        diff  = act_p - pred_p
        bar_pred = '█' * int(pred_p * 30)
        bar_act  = '░' * int(act_p  * 30)
        sign  = "+" if diff >= 0 else "-"
        flag  = "✅" if abs(diff) < 0.05 else ("⚠️ " if abs(diff) < 0.10 else "❌")
        print(f"  {pred_p*100:>8.1f}%   {act_p*100:>6.1f}%   {sign}{abs(diff)*100:>4.1f}%  {flag}  "
              f"pred:{bar_pred:<15} act:{bar_act:<15}")
        total_brier += (pred_p - act_p) ** 2

    # Brier Score (lower = better, 0.25 = no-skill baseline)
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(actual_draw, draw_proba)
    brier_baseline = brier_score_loss(actual_draw, np.full_like(draw_proba, actual_draw.mean()))

    print(f"\n  {LINE}")
    print(f"  📊 Brier Score (Draw)  : {brier:.4f}  (ยิ่งต่ำยิ่งดี)")
    print(f"  📊 Baseline Brier      : {brier_baseline:.4f}  (โมเดลสุ่มตาม base rate)")
    skill = (1 - brier / brier_baseline) * 100
    print(f"  📊 Brier Skill Score   : {skill:.1f}%  {'✅ ดีกว่า baseline' if skill > 0 else '❌ แย่กว่า baseline'}")

    # Actual draw rate vs predicted
    avg_pred_draw = draw_proba.mean() * 100
    avg_act_draw  = actual_draw.mean() * 100
    print(f"\n  📊 Avg Predicted Draw% : {avg_pred_draw:.1f}%")
    print(f"  📊 Actual Draw Rate    : {avg_act_draw:.1f}%")
    bias = avg_pred_draw - avg_act_draw
    print(f"  📊 Systematic Bias     : {bias:+.1f}%  "
          f"({'ทำนาย Draw มากเกินจริง' if bias > 2 else 'ทำนาย Draw น้อยเกินจริง' if bias < -2 else 'Bias ต่ำ ✅'})")

    # Suggestion
    print(f"\n  💡 คำแนะนำ:")
    if abs(bias) > 5:
        print(f"  • Recalibrate ด้วย Platt Scaling หรือ Isotonic Regression")
    if brier > brier_baseline:
        print(f"  • Draw prediction ยังแย่กว่า baseline — ลอง SMOTE หรือ cost-sensitive learning")
    else:
        print(f"  • โมเดล Draw ดีกว่า baseline แต่ยังมีช่องทางปรับปรุง")
        print(f"  • เพิ่ม feature เฉพาะ Draw เช่น: ทีมที่เสมอบ่อย, เกมคู่ปรับ, Head-to-Head draw rate")

    print(SEP)
    return {'brier': brier, 'brier_baseline': brier_baseline, 'skill': skill, 'bias': bias}


# ──────────────────────────────────────────────────────────────
# P2-3) SHAP ANALYSIS — วิเคราะห์ feature สำคัญ
# ──────────────────────────────────────────────────────────────

def run_shap_analysis(max_display=15):
    """
    ใช้ TreeExplainer บน XGBoost ใน ensemble
    วิเคราะห์ว่า feature ไหนส่งผลต่อการทำนายมากที่สุด
    แสดงผลแบบ ASCII bar chart (3 class: Away Win, Draw, Home Win)
    """
    try:
        import shap
    except ImportError:
        print("❌ กรุณาติดตั้ง shap ก่อน: pip install shap")
        return None

    SEP  = "=" * 65
    LINE = "─" * 65

    print()
    print(SEP)
    print("  🔍  SHAP FEATURE IMPORTANCE ANALYSIS")
    print(f"  ใช้ XGBoost (ใน Ensemble) + TreeSHAP บน Test Set ({len(X_test)} แมตช์)")
    print(SEP)

    # ดึง XGB จาก VotingClassifier
    xgb_model = None
    for name, estimator in ensemble.estimators:
        if name == 'xgb':
            xgb_model = estimator
            break

    # ใช้ fitted version จาก estimators_
    if xgb_model is None:
        print("❌ ไม่พบ XGB ใน ensemble")
        return None

    # ดึง fitted model จาก estimators_
    xgb_fitted = None
    for (name, _), fitted in zip(ensemble.estimators, ensemble.estimators_):
        if name == 'xgb':
            xgb_fitted = fitted
            break

    print("  กำลังคำนวณ SHAP values ...")
    explainer   = shap.TreeExplainer(xgb_fitted)
    shap_raw    = explainer.shap_values(X_test_sc)

    # XGBoost รุ่นเก่า → list of 3 arrays (n_samples, n_features)
    # XGBoost รุ่นใหม่ → single 3D array (n_samples, n_features, n_classes)
    if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        # (n_samples, n_features, n_classes) → list of (n_samples, n_features)
        shap_values = [shap_raw[:, :, i] for i in range(shap_raw.shape[2])]
    elif isinstance(shap_raw, list):
        shap_values = shap_raw
    else:
        # 2D array (binary-like fallback) → wrap in list
        shap_values = [shap_raw]
    # shap_values: list of 3 arrays (one per class), each shape (n_test, n_features)

    class_names = ['Away Win', 'Draw', 'Home Win']

    # ── Mean |SHAP| per feature per class ──────────────────────
    # รวม 3 class เป็น global importance
    mean_abs_shap = np.zeros(len(FEATURES))
    for cls_shap in shap_values:
        mean_abs_shap += np.abs(cls_shap).mean(axis=0)
    mean_abs_shap /= 3

    # เรียงลำดับ
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]

    print(f"\n  📊  Global Feature Importance (Mean |SHAP|, average across 3 classes)")
    print(f"  {'#':<4} {'Feature':<22} {'SHAP':>8}  {'Bar (relative importance)'}")
    print(f"  {LINE}")

    max_shap = mean_abs_shap[sorted_idx[0]]
    for rank, idx in enumerate(sorted_idx, 1):
        feat  = FEATURES[idx]
        val   = mean_abs_shap[idx]
        bar   = '█' * int(val / max_shap * 30)
        pct   = val / mean_abs_shap.sum() * 100
        print(f"  {rank:<4} {feat:<22} {val:>8.4f}  {bar:<30} ({pct:.1f}%)")

    # ── Per-Class Top 5 ─────────────────────────────────────────
    print(f"\n  📋  Top 5 Features แยกตาม Class")
    print(f"  {LINE}")
    for cls_idx, cls_name in enumerate(class_names):
        cls_shap = np.abs(shap_values[cls_idx]).mean(axis=0)
        top5_idx = np.argsort(cls_shap)[::-1][:5]
        icon = "✈️ " if cls_idx == 0 else ("🤝" if cls_idx == 1 else "🏠")
        print(f"\n  {icon}  {cls_name}")
        for r, i in enumerate(top5_idx, 1):
            bar = '█' * int(cls_shap[i] / cls_shap[top5_idx[0]] * 20)
            print(f"      {r}. {FEATURES[i]:<22} {cls_shap[i]:.4f}  {bar}")

    # ── Direction Analysis: ค่า SHAP บวก/ลบ ─────────────────────
    print(f"\n  🧭  Direction Analysis — Top 5 Features สำหรับ 🏠 Home Win")
    print(f"  {'Feature':<22} {'Mean SHAP':>10}  {'Direction'}")
    print(f"  {LINE}")
    hw_shap     = shap_values[2]   # class 2 = Home Win
    mean_signed = hw_shap.mean(axis=0)
    top5_hw     = np.argsort(np.abs(mean_signed))[::-1][:8]
    for i in top5_hw:
        direction = "➕ ช่วยให้เหย้าชนะ" if mean_signed[i] > 0 else "➖ ลดโอกาสเหย้าชนะ"
        print(f"  {FEATURES[i]:<22} {mean_signed[i]:>10.4f}  {direction}")

    # ── Insight Summary ──────────────────────────────────────────
    top1_feat = FEATURES[sorted_idx[0]]
    top2_feat = FEATURES[sorted_idx[1]]
    top3_feat = FEATURES[sorted_idx[2]]
    print(f"\n  💡 SHAP Insights:")
    print(f"  • Feature สำคัญที่สุด: {top1_feat}, {top2_feat}, {top3_feat}")
    elo_features = [FEATURES[i] for i in sorted_idx if 'Elo' in FEATURES[i]]
    form_features = [FEATURES[i] for i in sorted_idx if 'Pts' in FEATURES[i] or 'GF' in FEATURES[i]]
    if elo_features:
        print(f"  • Elo features ติด top: {', '.join(elo_features[:3])}")
    if form_features:
        print(f"  • Form features ติด top: {', '.join(form_features[:3])}")
    print(f"  • หากต้องการเพิ่ม accuracy: เน้นเพิ่มข้อมูลที่เกี่ยวกับ '{top1_feat}' และ '{top2_feat}'")

    print(SEP)
    return {'mean_abs_shap': mean_abs_shap, 'sorted_idx': sorted_idx,
            'shap_values': shap_values, 'features': FEATURES}


# ──────────────────────────────────────────────────────────────
# P2-4) PHASE 2 FULL REPORT
# ──────────────────────────────────────────────────────────────

def run_phase2(n_simulations=1000):
    """รัน Phase 2 ทั้งหมด"""
    print()
    print("█" * 65)
    print("  🚀  PHASE 2 — COMPETITION GRADE ANALYSIS")
    print("█" * 65)

    # 1. Monte Carlo
    mc_results = run_monte_carlo(n_simulations=n_simulations, verbose=True)

    # 2. Draw Calibration
    draw_stats = analyze_draw_calibration()

    # 3. SHAP
    shap_results = run_shap_analysis(max_display=15)

    # ── Phase 2 Summary ─────────────────────────────────────────
    SEP = "=" * 65
    print()
    print(SEP)
    print("  📋  PHASE 2 — SUMMARY")
    print(SEP)

    if mc_results:
        # Top 4 ที่น่าจะเกิดขึ้น
        top4_sorted = sorted(mc_results.items(), key=lambda x: x[1]['top4'], reverse=True)[:6]
        print(f"\n  🔴  Top 4 Probability (จาก {n_simulations:,} simulations)")
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

    print()
    print(SEP)
    print("  ✅  PHASE 2 COMPLETE")
    print(SEP)
    print()

    return {'monte_carlo': mc_results, 'draw_cal': draw_stats, 'shap': shap_results}


# ==============================
# 🚀 เรียกใช้งาน
# ==============================

# ── STEP 1: อัปเดต CSV จาก API ก่อนเสมอ (ข้อมูลล่าสุด 100%) ──
update_season_csv_from_api()

# ── STEP 2: จำลองฤดูกาลด้วยข้อมูลล่าสุด ──
run_season_simulation()

# ── STEP 3: ทำนายทีมที่ต้องการ ──
predict_with_api("Arsenal")
# predict_with_api("Liverpool")
# predict_with_api("Man City")
# predict_with_api("Chelsea")
# predict_with_api("Aston Villa")

# ── STEP 4: แสดงตารางแข่ง PL ถัดไปพร้อมทำนาย ──
show_next_pl_fixtures(5)    # 5 นัดถัดไป
# show_next_pl_fixtures(10)  # 10 นัด
# show_next_pl_fixtures(20)  # 20 นัด

# ── STEP 5: สรุปทั้งหมด ──
print_full_summary()

# ── STEP 6: Phase 2 — Competition Grade Analysis ──
# Monte Carlo 1000 รอบ + Draw Calibration + SHAP
# (ต้องติดตั้ง shap ก่อน: pip install shap)
run_phase2(n_simulations=1000)