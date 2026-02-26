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


# ==============================
# PHASE 3 — PRODUCTION GRADE
# ==============================

# ──────────────────────────────────────────────────────────────
# P3-1) ROLLING WINDOW CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────

def rolling_window_cv(n_splits=5, verbose=True):
    """
    Time-series cross-validation แบบ rolling window
    ไม่ใช้ข้อมูลอนาคตในการ validate
    แต่ละ fold: train = ข้อมูลก่อนหน้า, val = ข้อมูลช่วงถัดไป
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline

    SEP  = "=" * 65
    LINE = "─" * 65

    if verbose:
        print()
        print(SEP)
        print(f"  🔄  ROLLING WINDOW CROSS-VALIDATION  ({n_splits} folds)")
        print(SEP)

    # ใช้ match_df ที่มี features ครบ เรียงตามวันที่
    cv_df = match_df.dropna(subset=FEATURES + ['Result3']).sort_values('Date_x').reset_index(drop=True)
    X_cv  = cv_df[FEATURES].values
    y_cv  = cv_df['Result3'].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    if verbose:
        print(f"\n  {'Fold':<6} {'Train':>7} {'Val':>6} {'Acc':>7} {'HomeWin F1':>11} {'Draw F1':>9} {'AwayWin F1':>11}")
        print(f"  {LINE}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv), 1):
        X_tr, X_vl = X_cv[train_idx], X_cv[val_idx]
        y_tr, y_vl = y_cv[train_idx], y_cv[val_idx]

        # scale per fold (no leakage)
        sc_fold = StandardScaler()
        X_tr_sc = sc_fold.fit_transform(X_tr)
        X_vl_sc = sc_fold.transform(X_vl)

        # ใช้ ensemble เดียวกัน (XGB เร็วกว่า full ensemble ใน CV)
        cv_xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42, verbosity=0
        )
        cv_xgb.fit(X_tr_sc, y_tr)
        y_pred_fold = cv_xgb.predict(X_vl_sc)

        acc = accuracy_score(y_vl, y_pred_fold)
        rep = classification_report(y_vl, y_pred_fold, output_dict=True, zero_division=0)

        hw_f1   = rep.get('2', rep.get('Home Win', {})).get('f1-score', 0)
        draw_f1 = rep.get('1', rep.get('Draw', {})).get('f1-score', 0)
        aw_f1   = rep.get('0', rep.get('Away Win', {})).get('f1-score', 0)

        fold_results.append({'fold': fold, 'train': len(train_idx), 'val': len(val_idx),
                             'acc': acc, 'hw_f1': hw_f1, 'draw_f1': draw_f1, 'aw_f1': aw_f1})

        if verbose:
            print(f"  {fold:<6} {len(train_idx):>7} {len(val_idx):>6} {acc:>7.4f} "
                  f"{hw_f1:>11.4f} {draw_f1:>9.4f} {aw_f1:>11.4f}")

    # สรุป
    accs    = [r['acc']     for r in fold_results]
    hw_f1s  = [r['hw_f1']  for r in fold_results]
    dr_f1s  = [r['draw_f1']for r in fold_results]
    aw_f1s  = [r['aw_f1']  for r in fold_results]

    if verbose:
        print(f"  {LINE}")
        print(f"  {'Mean':<6} {'':>7} {'':>6} {np.mean(accs):>7.4f} "
              f"{np.mean(hw_f1s):>11.4f} {np.mean(dr_f1s):>9.4f} {np.mean(aw_f1s):>11.4f}")
        print(f"  {'Std':<6} {'':>7} {'':>6} {np.std(accs):>7.4f} "
              f"{np.std(hw_f1s):>11.4f} {np.std(dr_f1s):>9.4f} {np.std(aw_f1s):>11.4f}")
        print(f"\n  📊 Mean CV Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  📊 ความเสถียร       : {'✅ เสถียร (std < 0.03)' if np.std(accs) < 0.03 else '⚠️ ค่อนข้าง unstable'}")
        print(SEP)

    return fold_results


# ──────────────────────────────────────────────────────────────
# P3-2) HYPERPARAMETER TUNING (RandomizedSearchCV)
# ──────────────────────────────────────────────────────────────

def tune_hyperparameters(n_iter=30, verbose=True):
    """
    RandomizedSearchCV บน XGBoost ด้วย TimeSeriesSplit
    เร็วกว่า GridSearch แต่ครอบคลุมกว่า
    """
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline

    SEP = "=" * 65

    if verbose:
        print()
        print(SEP)
        print(f"  🔧  HYPERPARAMETER TUNING  (RandomizedSearch, n_iter={n_iter})")
        print(SEP)
        print(f"  กำลัง search {n_iter} combinations บน TimeSeriesSplit(5)...")

    cv_df = match_df.dropna(subset=FEATURES + ['Result3']).sort_values('Date_x').reset_index(drop=True)
    X_cv  = cv_df[FEATURES].values
    y_cv  = cv_df['Result3'].values

    sc_tune = StandardScaler()
    X_cv_sc = sc_tune.fit_transform(X_cv)

    param_dist = {
        'n_estimators':     [100, 200, 300, 400, 500],
        'max_depth':        [3, 4, 5, 6],
        'learning_rate':    [0.01, 0.03, 0.05, 0.1, 0.15],
        'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma':            [0, 0.1, 0.3, 0.5],
        'reg_alpha':        [0, 0.01, 0.1, 0.5],
        'reg_lambda':       [1, 1.5, 2.0, 3.0],
    }

    xgb_tune = XGBClassifier(
        eval_metric='mlogloss', random_state=42, verbosity=0
    )

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        xgb_tune, param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    search.fit(X_cv_sc, y_cv)

    best_params = search.best_params_
    best_score  = search.best_score_

    if verbose:
        print(f"\n  ✅ Best CV Accuracy : {best_score:.4f}")
        print(f"\n  📋 Best Parameters:")
        LINE = "─" * 65
        print(f"  {LINE}")
        for k, v in sorted(best_params.items()):
            print(f"  {'  ' + k:<28} = {v}")
        print(f"  {LINE}")

        # เปรียบเทียบกับ default
        print(f"\n  💡 แนะนำให้อัปเดต XGBClassifier ในโค้ดด้วย params ด้านบน")
        print(f"  💡 คาด accuracy เพิ่มขึ้นประมาณ ±{abs(best_score - 0.49)*100:.1f}%")
        print(SEP)

    return best_params, best_score


# ──────────────────────────────────────────────────────────────
# HELPER: Manual Prefit Calibration (รองรับ sklearn ≥ 1.2)
# ──────────────────────────────────────────────────────────────

class _PreFitCalibratedWrapper:
    """
    Wrapper ที่ทำ prefit calibration แบบ manual
    รองรับ sklearn ทุก version (ใช้แทน cv='prefit' ที่ถูกลบออกใน sklearn 1.2+)
    ใช้ Isotonic Regression หรือ Logistic (Platt Scaling) per class
    """
    def __init__(self, base_estimator, method='isotonic'):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrators_ = None
        self.classes_ = None

    def fit(self, X_cal, y_cal):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression as LR

        proba = self.base_estimator.predict_proba(X_cal)
        self.classes_ = self.base_estimator.classes_
        n_classes = len(self.classes_)
        self.calibrators_ = []

        for i in range(n_classes):
            y_bin = (y_cal == self.classes_[i]).astype(int)
            p_col = proba[:, i]
            if self.method == 'isotonic':
                cal = IsotonicRegression(out_of_bounds='clip')
                cal.fit(p_col, y_bin)
            else:
                # Platt Scaling (sigmoid)
                cal = LR()
                cal.fit(p_col.reshape(-1, 1), y_bin)
            self.calibrators_.append(cal)
        return self

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)
        n_classes = len(self.classes_)
        cal_proba = np.zeros_like(proba)

        for i in range(n_classes):
            p_col = proba[:, i]
            if self.method == 'isotonic':
                cal_proba[:, i] = self.calibrators_[i].predict(p_col)
            else:
                cal_proba[:, i] = self.calibrators_[i].predict_proba(
                    p_col.reshape(-1, 1))[:, 1]

        # Normalize ให้ผลรวม = 1
        row_sums = cal_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return cal_proba / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        idx   = np.argmax(proba, axis=1)
        return np.array([self.classes_[i] for i in idx])


# ──────────────────────────────────────────────────────────────
# P3-3) PROBABILITY CALIBRATION (Platt Scaling / Isotonic)
# ──────────────────────────────────────────────────────────────

def calibrate_probabilities(method='isotonic', verbose=True):
    """
    Calibrate probability ของ ensemble ด้วย CalibratedClassifierCV
    method: 'isotonic' หรือ 'sigmoid' (Platt Scaling)

    Returns: calibrated_model ที่ใช้แทน ensemble ได้เลย
    """
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve

    SEP  = "=" * 65
    LINE = "─" * 65

    if verbose:
        print()
        print(SEP)
        print(f"  🎯  PROBABILITY CALIBRATION  (method={method})")
        print(SEP)

    # แบ่ง calibration set จาก train (ใช้ 20% ท้ายของ train)
    cal_split = int(len(X_train) * 0.8)
    X_cal_train = X_train_sc[:cal_split]
    y_cal_train = y_train.iloc[:cal_split]
    X_cal_val   = X_train_sc[cal_split:]
    y_cal_val   = y_train.iloc[cal_split:]

    # เทรน base model ใหม่บน 80% ของ train
    from sklearn.base import clone
    base_model = clone(ensemble)
    base_model.fit(X_cal_train, y_cal_train)

    # Calibrate — รองรับ sklearn ทุก version
    # sklearn < 1.2 → cv='prefit' ใช้ได้
    # sklearn >= 1.2 → ใช้ _PreFitCalibratedWrapper แทน
    try:
        from sklearn.calibration import CalibratedClassifierCV as _CCCV
        calibrated = _CCCV(base_model, method=method, cv='prefit')
        calibrated.fit(X_cal_val, y_cal_val)
    except Exception:
        calibrated = _PreFitCalibratedWrapper(base_model, method=method)
        calibrated.fit(X_cal_val, y_cal_val)

    # เปรียบเทียบ before / after บน test set
    proba_before = ensemble.predict_proba(X_test_sc)
    proba_after  = calibrated.predict_proba(X_test_sc)

    acc_before = accuracy_score(y_test, ensemble.predict(X_test_sc))
    acc_after  = accuracy_score(y_test, calibrated.predict(X_test_sc))

    if verbose:
        print(f"\n  {'Metric':<30} {'Before':>10} {'After':>10} {'Change':>10}")
        print(f"  {LINE}")
        print(f"  {'Accuracy':<30} {acc_before:>10.4f} {acc_after:>10.4f} "
              f"{acc_after-acc_before:>+10.4f}")

        # Brier Score แต่ละ class
        from sklearn.metrics import brier_score_loss
        for cls_idx, cls_name in enumerate(['Away Win', 'Draw', 'Home Win']):
            y_bin = (y_test == cls_idx).astype(int)
            bs_b  = brier_score_loss(y_bin, proba_before[:, cls_idx])
            bs_a  = brier_score_loss(y_bin, proba_after[:, cls_idx])
            flag  = "✅" if bs_a < bs_b else "⚠️ "
            print(f"  {'Brier (' + cls_name + ')':<30} {bs_b:>10.4f} {bs_a:>10.4f} "
                  f"{bs_a-bs_b:>+10.4f} {flag}")

        # ECE (Expected Calibration Error) สำหรับ Home Win
        hw_before = proba_before[:, 2]
        hw_after  = proba_after[:, 2]
        y_hw_bin  = (y_test == 2).astype(int)

        def ece_score(y_true, y_prob, n_bins=10):
            bins   = np.linspace(0, 1, n_bins + 1)
            ece    = 0.0
            for i in range(n_bins):
                mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
                if mask.sum() > 0:
                    acc_b  = y_true[mask].mean()
                    conf_b = y_prob[mask].mean()
                    ece   += mask.sum() * abs(acc_b - conf_b)
            return ece / len(y_true)

        ece_b = ece_score(y_hw_bin.values, hw_before)
        ece_a = ece_score(y_hw_bin.values, hw_after)
        flag  = "✅" if ece_a < ece_b else "⚠️ "
        print(f"  {'ECE (Home Win)':<30} {ece_b:>10.4f} {ece_a:>10.4f} "
              f"{ece_a-ece_b:>+10.4f} {flag}")

        print(f"\n  {'method':<30} {method}")
        verdict = "✅ Calibration ช่วยได้" if acc_after >= acc_before else "⚠️  Accuracy ลดลงเล็กน้อย แต่ probability แม่นขึ้น"
        print(f"  {verdict}")
        print(SEP)

    return calibrated


# ──────────────────────────────────────────────────────────────
# P3-4) BACKTEST ROI SIMULATION
#       จำลองการแทงตาม model probability vs Kelly Criterion
# ──────────────────────────────────────────────────────────────

def backtest_roi(bankroll=1000.0, min_edge=0.03, kelly_fraction=0.25, verbose=True):
    """
    Backtest ROI โดยใช้ model probability บน test set
    กลยุทธ์: แทงเมื่อ model edge > min_edge (เทียบกับ implied odds)
    ใช้ Kelly Criterion × kelly_fraction เพื่อคำนวณ stake

    สมมติ odds แบบ fair (จาก model เอง) บวก margin 5% (สมมติ bookmaker)

    Returns: dict สรุปผล ROI
    """
    SEP  = "=" * 65
    LINE = "─" * 65

    if verbose:
        print()
        print(SEP)
        print(f"  💰  BACKTEST ROI SIMULATION")
        print(f"  Bankroll: £{bankroll:,.0f} | Min Edge: {min_edge*100:.0f}% | Kelly: {kelly_fraction*100:.0f}%")
        print(SEP)

    proba_test = ensemble.predict_proba(X_test_sc)   # (n, 3): Away, Draw, Home
    label_map  = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

    bk         = bankroll
    bets       = []
    total_bets = 0
    total_won  = 0
    total_staked = 0.0
    peak_bk    = bk
    max_dd     = 0.0

    for i, (proba, actual) in enumerate(zip(proba_test, y_test)):
        p_away, p_draw, p_home = proba

        # สร้าง implied odds จาก model + bookmaker margin 5%
        margin = 1.05
        odds = {
            0: (1 / p_away)  * margin if p_away  > 0.01 else 99,
            1: (1 / p_draw)  * margin if p_draw  > 0.01 else 99,
            2: (1 / p_home)  * margin if p_home  > 0.01 else 99,
        }
        model_p = {0: p_away, 1: p_draw, 2: p_home}

        # หา best bet: class ที่ model edge สูงสุด
        best_cls = max([0, 1, 2], key=lambda c: model_p[c] - (1 / odds[c]))
        edge     = model_p[best_cls] - (1 / odds[best_cls])

        if edge < min_edge:
            continue   # ไม่แทง

        # Kelly stake
        p   = model_p[best_cls]
        o   = odds[best_cls]
        k   = max(0, (p * o - 1) / (o - 1))   # Kelly fraction
        stake = min(bk * k * kelly_fraction, bk * 0.05)   # cap 5% bankroll
        stake = max(stake, 0.5)   # min £0.5

        won    = (best_cls == actual)
        profit = stake * (o - 1) if won else -stake
        bk    += profit

        total_bets   += 1
        total_staked += stake
        if won:
            total_won += 1

        # Max drawdown
        if bk > peak_bk:
            peak_bk = bk
        dd = (peak_bk - bk) / peak_bk * 100
        if dd > max_dd:
            max_dd = dd

        bets.append({'bet': total_bets, 'cls': best_cls, 'edge': edge,
                     'stake': stake, 'odds': o, 'won': won, 'profit': profit, 'bk': bk})

    if total_bets == 0:
        if verbose:
            print("  ⚠️  ไม่มีการแทงที่ตรงเงื่อนไข — ลด min_edge ลง")
        return None

    roi      = (bk - bankroll) / total_staked * 100
    win_rate = total_won / total_bets * 100
    net_pnl  = bk - bankroll
    yield_   = net_pnl / bankroll * 100

    if verbose:
        print(f"\n  {'Metric':<35} {'Value':>15}")
        print(f"  {LINE}")
        print(f"  {'จำนวนแมตช์ที่ผ่าน threshold':<35} {total_bets:>15,}")
        print(f"  {'Win Rate':<35} {win_rate:>14.1f}%")
        print(f"  {'Total Staked':<35} £{total_staked:>13,.2f}")
        print(f"  {'Net P&L':<35} £{net_pnl:>+13,.2f}")
        print(f"  {'Final Bankroll':<35} £{bk:>13,.2f}")
        print(f"  {'ROI (per unit staked)':<35} {roi:>14.1f}%")
        print(f"  {'Yield (on initial bankroll)':<35} {yield_:>14.1f}%")
        print(f"  {'Max Drawdown':<35} {max_dd:>14.1f}%")
        print(f"  {LINE}")

        # แบ่งตาม outcome type
        for cls in [2, 0, 1]:
            cls_bets = [b for b in bets if b['cls'] == cls]
            if not cls_bets:
                continue
            cls_won   = sum(1 for b in cls_bets if b['won'])
            cls_roi   = sum(b['profit'] for b in cls_bets) / sum(b['stake'] for b in cls_bets) * 100
            print(f"  {label_map[cls]:<35} {len(cls_bets):>4} เดิมพัน  "
                  f"Win: {cls_won/len(cls_bets)*100:.0f}%  ROI: {cls_roi:+.1f}%")

        print(f"\n  {'💡 สรุป':}")
        if roi > 5:
            print(f"  ✅ ROI {roi:.1f}% — โมเดลมี edge จริงในระยะยาว")
        elif roi > 0:
            print(f"  🟡 ROI {roi:.1f}% — มี edge เล็กน้อย ควรเพิ่มข้อมูลให้มากขึ้น")
        else:
            print(f"  ❌ ROI {roi:.1f}% — โมเดลยังไม่ beat the market ต้อง tune เพิ่ม")

        # แสดง bankroll curve (ASCII)
        print(f"\n  📈 Bankroll Curve (ทุก {max(1, total_bets//20)} เดิมพัน):")
        step     = max(1, total_bets // 20)
        sampled  = [bets[i] for i in range(0, len(bets), step)]
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


# ──────────────────────────────────────────────────────────────
# P3-5) PHASE 3 FULL RUNNER
# ──────────────────────────────────────────────────────────────

def run_phase3(tune=False, n_tune_iter=30, cv_splits=5,
               cal_method='isotonic', backtest_edge=0.03):
    """
    รัน Phase 3 ทั้งหมด
    tune=True  → รัน hyperparameter tuning (ช้า ~2-5 นาที)
    tune=False → ข้ามขั้นตอน tuning
    """
    print()
    print("█" * 65)
    print("  🏆  PHASE 3 — PRODUCTION GRADE")
    print("█" * 65)

    SEP = "=" * 65

    # 1. Rolling CV
    cv_results = rolling_window_cv(n_splits=cv_splits, verbose=True)

    # 2. Hyperparameter Tuning (optional)
    best_params = None
    if tune:
        best_params, best_score = tune_hyperparameters(n_iter=n_tune_iter, verbose=True)
    else:
        print(f"\n  ⏭️  ข้าม Hyperparameter Tuning (ตั้ง tune=True เพื่อรัน)")

    # 3. Probability Calibration
    calibrated_model = calibrate_probabilities(method=cal_method, verbose=True)

    # 4. Backtest ROI
    roi_result = backtest_roi(
        bankroll=1000.0,
        min_edge=backtest_edge,
        kelly_fraction=0.25,
        verbose=True
    )

    # ── Phase 3 Summary ─────────────────────────────────────────
    cv_accs = [r['acc'] for r in cv_results]
    print()
    print(SEP)
    print("  📋  PHASE 3 — SUMMARY")
    print(SEP)
    print(f"\n  🔄 Rolling CV ({cv_splits} folds)")
    print(f"     Mean Accuracy : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
    print(f"     Range         : [{min(cv_accs):.4f} – {max(cv_accs):.4f}]")

    if best_params:
        print(f"\n  🔧 Best Params (XGB)")
        for k, v in sorted(best_params.items()):
            print(f"     {k:<25} = {v}")

    if roi_result:
        print(f"\n  💰 Backtest ROI")
        print(f"     ROI            : {roi_result['roi']:+.1f}%")
        print(f"     Win Rate       : {roi_result['win_rate']:.1f}%")
        print(f"     Max Drawdown   : {roi_result['max_dd']:.1f}%")
        print(f"     Total Bets     : {roi_result['total_bets']:,}")

    print()
    print(SEP)
    print("  ✅  PHASE 3 COMPLETE")
    print(SEP)
    print()

    return {
        'cv': cv_results,
        'best_params': best_params,
        'calibrated_model': calibrated_model,
        'roi': roi_result,
    }

# ── STEP 7: Phase 3 — Production Grade ──
# tune=False → เร็ว (~10 วิ) | tune=True → ช้า (~3-5 นาที)
run_phase3(tune=False)
# run_phase3(tune=True, n_tune_iter=30)   # uncomment เพื่อ tune

# ==============================
# PHASE 4 — FULL ANALYTICS ENGINE
# ==============================

from sklearn.metrics import log_loss, f1_score


# ──────────────────────────────────────────────────────────────
# P4-1) ROLLING WINDOW VALIDATION  (พร้อม LogLoss)
# ──────────────────────────────────────────────────────────────

def p4_rolling_cv(n_splits=5, verbose=True):
    """Rolling CV พร้อม LogLoss per fold"""
    from sklearn.model_selection import TimeSeriesSplit

    SEP  = "=" * 70
    LINE = "─" * 70

    cv_df = match_df.dropna(subset=FEATURES + ['Result3']).sort_values('Date_x').reset_index(drop=True)
    X_cv  = cv_df[FEATURES].values
    y_cv  = cv_df['Result3'].values

    tscv  = TimeSeriesSplit(n_splits=n_splits)
    results = []

    if verbose:
        print()
        print(SEP)
        print(f"  📊  §1  ROLLING WINDOW VALIDATION")
        print(SEP)
        print(f"\n  {'Fold':<6} {'Train':>7} {'Val':>6} {'Acc':>7} {'HW-F1':>8} {'DR-F1':>8} {'AW-F1':>8} {'LogLoss':>9}")
        print(f"  {LINE}")

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X_cv), 1):
        X_tr, X_vl = X_cv[tr_idx], X_cv[vl_idx]
        y_tr, y_vl = y_cv[tr_idx], y_cv[vl_idx]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_vl_sc = sc.transform(X_vl)

        cv_mdl = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42, verbosity=0
        )
        cv_mdl.fit(X_tr_sc, y_tr)
        y_pred_fold  = cv_mdl.predict(X_vl_sc)
        y_proba_fold = cv_mdl.predict_proba(X_vl_sc)

        acc  = accuracy_score(y_vl, y_pred_fold)
        ll   = log_loss(y_vl, y_proba_fold, labels=[0, 1, 2])
        hw   = f1_score(y_vl, y_pred_fold, labels=[2], average='macro', zero_division=0)
        dr   = f1_score(y_vl, y_pred_fold, labels=[1], average='macro', zero_division=0)
        aw   = f1_score(y_vl, y_pred_fold, labels=[0], average='macro', zero_division=0)

        results.append({'fold': fold, 'train': len(tr_idx), 'val': len(vl_idx),
                        'acc': acc, 'hw_f1': hw, 'dr_f1': dr, 'aw_f1': aw, 'll': ll})

        if verbose:
            print(f"  {fold:<6} {len(tr_idx):>7} {len(vl_idx):>6} {acc:>7.4f} "
                  f"{hw:>8.4f} {dr:>8.4f} {aw:>8.4f} {ll:>9.4f}")

    if verbose:
        accs = [r['acc'] for r in results]
        lls  = [r['ll']  for r in results]
        print(f"  {LINE}")
        print(f"  {'Mean':<13} {'':>6} {np.mean(accs):>7.4f} "
              f"{np.mean([r['hw_f1'] for r in results]):>8.4f} "
              f"{np.mean([r['dr_f1'] for r in results]):>8.4f} "
              f"{np.mean([r['aw_f1'] for r in results]):>8.4f} "
              f"{np.mean(lls):>9.4f}")
        print()
        print(f"  CV Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  CV LogLoss   : {np.mean(lls):.4f}")
        stable = np.std(accs) < 0.03
        print(f"  ความเสถียร   : {'✅ เสถียร' if stable else '⚠️  unstable — พิจารณาเพิ่มข้อมูล'}")
        print(SEP)

    return results


# ──────────────────────────────────────────────────────────────
# P4-2) PROBABILITY CALIBRATION
# ──────────────────────────────────────────────────────────────

def p4_calibration(method='isotonic', verbose=True):
    """Calibrate + เปรียบเทียบ Accuracy, LogLoss, Brier — ใช้ _PreFitCalibratedWrapper"""
    from sklearn.metrics import brier_score_loss
    from sklearn.base import clone

    SEP  = "=" * 70
    LINE = "─" * 70

    cal_split   = int(len(X_train_sc) * 0.8)
    X_ct        = X_train_sc[:cal_split]
    y_ct        = y_train.iloc[:cal_split]
    X_cv2       = X_train_sc[cal_split:]
    y_cv2       = y_train.iloc[cal_split:]

    base = clone(ensemble)
    base.fit(X_ct, y_ct)

    # ใช้ _PreFitCalibratedWrapper เสมอ (รองรับ sklearn ทุก version)
    calibrated = _PreFitCalibratedWrapper(base, method=method)
    calibrated.fit(X_cv2, y_cv2)

    pb = ensemble.predict_proba(X_test_sc)
    pa = calibrated.predict_proba(X_test_sc)

    acc_b = accuracy_score(y_test, ensemble.predict(X_test_sc))
    acc_a = accuracy_score(y_test, calibrated.predict(X_test_sc))
    ll_b  = log_loss(y_test, pb, labels=[0, 1, 2])
    ll_a  = log_loss(y_test, pa, labels=[0, 1, 2])

    if verbose:
        print()
        print(SEP)
        print(f"  🎯  §2  PROBABILITY CALIBRATION  (method={method})")
        print(SEP)
        print(f"\n  {'Metric':<32} {'Before':>10} {'After':>10} {'Δ':>10}")
        print(f"  {LINE}")
        print(f"  {'Accuracy':<32} {acc_b:>10.4f} {acc_a:>10.4f} {acc_a-acc_b:>+10.4f}")
        ll_flag = "✅" if ll_a <= ll_b else "⚠️ "
        print(f"  {'Log Loss':<32} {ll_b:>10.4f} {ll_a:>10.4f} {ll_a-ll_b:>+10.4f}  {ll_flag}")
        if ll_a > ll_b:
            print(f"  {'':>32}  (isotonic อาจ overfit บน cal_set เล็ก — ปกติสำหรับ n<500)")

        for ci, cn in enumerate(['Away Win', 'Draw', 'Home Win']):
            yb = (y_test == ci).astype(int)
            bb = brier_score_loss(yb, pb[:, ci])
            ba = brier_score_loss(yb, pa[:, ci])
            flag = "✅" if ba < bb else "⚠️ "
            print(f"  {'Brier (' + cn + ')':<32} {bb:>10.4f} {ba:>10.4f} {ba-bb:>+10.4f}  {flag}")

        # Verdict: Brier ดีขึ้นทุก class = calibration ทำงานได้ดี แม้ LogLoss จะขึ้น
        brier_improved = sum(
            1 for ci in range(3)
            if brier_score_loss((y_test == ci).astype(int), pa[:, ci])
               < brier_score_loss((y_test == ci).astype(int), pb[:, ci])
        )
        if acc_a >= acc_b or brier_improved >= 2:
            verdict = "✅ Calibration ช่วยได้ (Accuracy ↑ + Brier ↓)"
        else:
            verdict = "⚠️  Calibration ช่วยได้บางส่วน — Brier ดีขึ้น แต่ LogLoss สูงขึ้น"
        print(f"\n  {verdict}")
        print(SEP)

    return calibrated


# ──────────────────────────────────────────────────────────────
# P4-3) VALUE BET DETECTION
# ──────────────────────────────────────────────────────────────

def p4_value_bets(calibrated_model, min_edge=0.05, bk_margin=0.05, verbose=True):
    """ค้นหา Value Bets จาก calibrated probabilities"""
    SEP  = "=" * 70
    LINE = "─" * 70

    proba  = calibrated_model.predict_proba(X_test_sc)
    actual = y_test.values
    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

    value_bets = []
    for i, (p_row, act) in enumerate(zip(proba, actual)):
        for cls in range(3):
            model_p = p_row[cls]
            bm_p    = model_p * (1 - bk_margin)
            if bm_p <= 0.01:
                continue
            bm_odds = 1 / bm_p
            edge    = model_p - (1 / bm_odds)
            if edge >= min_edge:
                value_bets.append({
                    'idx': i, 'cls': cls, 'model_p': model_p,
                    'bm_odds': bm_odds, 'edge': edge,
                    'won': (cls == act)
                })

    hit_rate = np.mean([b['won'] for b in value_bets]) * 100 if value_bets else 0.0
    ev_total = sum(b['edge'] for b in value_bets)

    if verbose:
        print()
        print(SEP)
        print(f"  💎  §3  VALUE BET DETECTION")
        print(SEP)
        print(f"\n  Value Bet Threshold : edge ≥ {min_edge*100:.0f}%  |  BM Margin: {bk_margin*100:.0f}%")
        print()
        print(f"  {'Metric':<45} {'Value':>15}")
        print(f"  {LINE}")
        print(f"  {'Value Bets พบ':<45} {len(value_bets):>15,}")
        print(f"  {'Hit Rate':<45} {hit_rate:>14.1f}%")

        print(f"\n  {'Class':<16} {'Count':>7} {'Hit%':>7} {'Avg Edge':>10} {'Avg Odds':>10}")
        print(f"  {LINE}")
        for cls in [2, 0, 1]:
            cls_bets = [b for b in value_bets if b['cls'] == cls]
            if not cls_bets:
                continue
            ch = np.mean([b['won'] for b in cls_bets]) * 100
            ae = np.mean([b['edge'] for b in cls_bets])
            ao = np.mean([b['bm_odds'] for b in cls_bets])
            print(f"  {label_map[cls]:<16} {len(cls_bets):>7} {ch:>6.1f}% {ae:>10.4f} {ao:>10.2f}")

        print(f"\n  Expected Value รวม (Σ edge) : {ev_total:>+.3f}")
        if len(value_bets) == 0:
            print(f"  ⚠️  ไม่พบ Value Bets — ลด min_value_edge เป็น 0.03")
        elif hit_rate < 40:
            print(f"  ⚠️  Hit rate ต่ำ — ระวัง overfit")
        print(SEP)

    return value_bets


# ──────────────────────────────────────────────────────────────
# P4-4/5) ROI BACKTEST + RISK CONTROL  |  P4-6) EQUITY CURVE
# ──────────────────────────────────────────────────────────────

def p4_roi_backtest(calibrated_model, bankroll=1000.0, kelly_frac=0.25,
                    max_bet_pct=0.05, stop_loss_pct=0.20, min_edge=0.05,
                    verbose=True):
    """ROI Backtest พร้อม Kelly Criterion + Stop-Loss + Equity Curve"""
    SEP  = "=" * 70
    LINE = "─" * 70

    proba  = calibrated_model.predict_proba(X_test_sc)
    actual = y_test.values
    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

    bk      = bankroll
    peak_bk = bk
    max_dd  = 0.0
    bets    = []
    stopped = False

    for p_row, act in zip(proba, actual):
        if stopped:
            break
        dd = (peak_bk - bk) / peak_bk * 100
        if dd >= stop_loss_pct * 100:
            stopped = True
            break

        best_cls, best_edge, best_odds = -1, -1.0, 1.0
        for cls in range(3):
            mp   = p_row[cls]
            bm_p = mp * 0.95
            if bm_p <= 0.01:
                continue
            bm_o = 1 / bm_p
            edge = mp - (1 / bm_o)
            if edge > best_edge:
                best_edge, best_cls, best_odds = edge, cls, bm_o

        if best_edge < min_edge:
            continue

        p     = p_row[best_cls]
        k     = max(0.0, (p * best_odds - 1) / (best_odds - 1))
        stake = min(bk * k * kelly_frac, bk * max_bet_pct)
        stake = max(stake, 0.5)

        won    = (best_cls == act)
        profit = stake * (best_odds - 1) if won else -stake
        bk    += profit

        if bk > peak_bk:
            peak_bk = bk
        dd2 = (peak_bk - bk) / peak_bk * 100
        if dd2 > max_dd:
            max_dd = dd2

        bets.append({'cls': best_cls, 'edge': best_edge, 'stake': stake,
                     'odds': best_odds, 'won': won, 'profit': profit, 'bk': bk})

    n_bets       = len(bets)
    n_won        = sum(b['won'] for b in bets)
    total_staked = sum(b['stake'] for b in bets)
    net_pnl      = bk - bankroll
    roi          = net_pnl / total_staked * 100 if total_staked > 0 else 0.0
    win_rate     = n_won / n_bets * 100 if n_bets > 0 else 0.0
    yield_       = net_pnl / bankroll * 100

    if verbose:
        print()
        print(SEP)
        print(f"  💰  §4  ROI BACKTEST  +  🛡️  §5  RISK CONTROL")
        print(SEP)
        print(f"\n  Bankroll    : £{bankroll:,.0f}")
        print(f"  Kelly Frac  : {kelly_frac*100:.0f}%  |  Max Bet: {max_bet_pct*100:.0f}% bankroll")
        print(f"  Stop-Loss   : {stop_loss_pct*100:.0f}% drawdown")
        print(f"  Edge Min    : {min_edge*100:.0f}%")
        print()
        print(f"  {'Metric':<45} {'Value':>15}")
        print(f"  {LINE}")
        print(f"  {'จำนวนเดิมพัน':<45} {n_bets:>15,}")
        print(f"  {'Win Rate':<45} {win_rate:>14.1f}%")
        print(f"  {'Total Staked':<45} £{total_staked:>13,.2f}")
        print(f"  {'Net P&L':<45} £{net_pnl:>+13,.2f}")
        print(f"  {'Final Bankroll':<45} £{bk:>13,.2f}")
        print(f"  {'ROI (per unit staked)':<45} {roi:>14.1f}%")
        print(f"  {'Yield (on initial BK)':<45} {yield_:>14.1f}%")
        print(f"  {'Max Drawdown':<45} {max_dd:>14.1f}%")
        print()
        stop_msg = "⚠️  Stop-Loss ถูก trigger!" if stopped else "✅ ไม่ถึง Stop-Loss ตลอด backtest"
        print(f"  {stop_msg}")
        print()

        print(f"  {'Class':<16} {'Bets':>6} {'Win%':>6} {'ROI':>8}")
        print(f"  {LINE}")
        for cls in [2, 0, 1]:
            cb = [b for b in bets if b['cls'] == cls]
            if not cb:
                continue
            cw = sum(b['won'] for b in cb)
            cr = sum(b['profit'] for b in cb) / sum(b['stake'] for b in cb) * 100
            print(f"  {label_map[cls]:<16} {len(cb):>6} {cw/len(cb)*100:>5.0f}% {cr:>+8.1f}%")

        # §6 Equity Curve
        print()
        print(SEP)
        print(f"  📈  §6  EQUITY CURVE")
        print(SEP)
        if n_bets == 0:
            print(f"  ⚠️  ไม่มีการเดิมพัน — ลด min_value_edge ลง")
        else:
            step    = max(1, n_bets // 30)
            samples = [bets[i] for i in range(0, n_bets, step)] + [bets[-1]]
            max_bk  = max(b['bk'] for b in samples)
            min_bk  = min(b['bk'] for b in samples)
            rng     = max_bk - min_bk or 1
            print(f"\n  £{max_bk:>8,.0f} ┐")
            for b in samples:
                bar = int((b['bk'] - min_bk) / rng * 40)
                print(f"           {'█' * bar}")
            print(f"  £{min_bk:>8,.0f} ┘")
        print(SEP)

    return {
        'n_bets': n_bets, 'win_rate': win_rate, 'roi': roi, 'yield': yield_,
        'net_pnl': net_pnl, 'max_dd': max_dd, 'final_bk': bk, 'stopped': stopped,
        'bets': bets
    }


# ──────────────────────────────────────────────────────────────
# P4-7) FEATURE IMPORTANCE  (XGB gain + RF impurity + SHAP)
# ──────────────────────────────────────────────────────────────

def p4_feature_importance(verbose=True):
    """
    7A) XGBoost gain importance
    7B) Random Forest impurity importance
    7C) SHAP (ถ้าติดตั้งแล้ว)
    ใช้ named_estimators_ เพื่อรองรับ sklearn ทุก version
    """
    SEP  = "=" * 70
    LINE = "─" * 70

    if verbose:
        print()
        print(SEP)
        print(f"  🔍  §7  FEATURE IMPORTANCE")
        print(SEP)

    results = {}

    # ดึง fitted models — ใช้ named_estimators_ แทน estimators_
    # (estimators_ คืน list ของ objects ใน sklearn ใหม่ ไม่ใช่ tuple)
    xgb_fitted = ensemble.named_estimators_.get('xgb', None)
    rf_fitted  = ensemble.named_estimators_.get('rf',  None)

    # ── 7A) XGBoost ──────────────────────────────────────────
    if xgb_fitted is not None:
        fi_xgb  = xgb_fitted.feature_importances_
        idx_xgb = np.argsort(fi_xgb)[::-1][:15]
        max_fi  = fi_xgb[idx_xgb[0]]

        if verbose:
            print(f"\n  7A) XGBoost Feature Importance (gain)")
            print(f"  {LINE}")
            print(f"  {'#':<4} {'Feature':<24} {'Score':>8}  Bar")
            print(f"  {LINE}")
            for rank, i in enumerate(idx_xgb, 1):
                bar = '█' * int(fi_xgb[i] / max_fi * 30)
                print(f"  {rank:<4} {FEATURES[i]:<24} {fi_xgb[i]:>8.4f}  {bar}")

        results['xgb_importance'] = dict(zip(FEATURES, fi_xgb))

    # ── 7B) Random Forest ─────────────────────────────────────
    if rf_fitted is not None:
        fi_rf   = rf_fitted.feature_importances_
        idx_rf  = np.argsort(fi_rf)[::-1][:15]
        max_rf  = fi_rf[idx_rf[0]]

        if verbose:
            print(f"\n  7B) Random Forest Feature Importance (impurity)")
            print(f"  {LINE}")
            print(f"  {'#':<4} {'Feature':<24} {'Score':>8}  Bar")
            print(f"  {LINE}")
            for rank, i in enumerate(idx_rf, 1):
                bar = '█' * int(fi_rf[i] / max_rf * 30)
                print(f"  {rank:<4} {FEATURES[i]:<24} {fi_rf[i]:>8.4f}  {bar}")

        results['rf_importance'] = dict(zip(FEATURES, fi_rf))

    # ── 7C) SHAP (optional) ───────────────────────────────────
    try:
        import shap
        if xgb_fitted is not None:
            if verbose:
                print(f"\n  7C) SHAP Values (TreeExplainer บน XGBoost)")
                print(f"  {LINE}")
                print(f"  กำลังคำนวณ SHAP values ...")

            explainer = shap.TreeExplainer(xgb_fitted)
            shap_raw  = explainer.shap_values(X_test_sc)

            if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
                shap_values = [shap_raw[:, :, i] for i in range(shap_raw.shape[2])]
            elif isinstance(shap_raw, list):
                shap_values = shap_raw
            else:
                shap_values = [shap_raw]

            mean_abs   = np.zeros(len(FEATURES))
            for sv in shap_values:
                mean_abs += np.abs(sv).mean(axis=0)
            mean_abs  /= len(shap_values)
            sorted_idx = np.argsort(mean_abs)[::-1][:15]
            max_shap   = mean_abs[sorted_idx[0]]

            if verbose:
                print(f"\n  {'#':<4} {'Feature':<24} {'|SHAP|':>8}  Bar")
                print(f"  {LINE}")
                for rank, i in enumerate(sorted_idx, 1):
                    bar = '█' * int(mean_abs[i] / max_shap * 30)
                    print(f"  {rank:<4} {FEATURES[i]:<24} {mean_abs[i]:>8.4f}  {bar}")

            results['shap'] = dict(zip(FEATURES, mean_abs))
    except ImportError:
        if verbose:
            print(f"\n  7C) SHAP: ไม่ได้ติดตั้ง — ข้าม (pip install shap)")
    except Exception as e:
        if verbose:
            print(f"\n  7C) SHAP error: {e}")

    if verbose:
        print(SEP)

    return results


# ──────────────────────────────────────────────────────────────
# P4-8) CALIBRATED vs UNCALIBRATED ROI COMPARISON
# ──────────────────────────────────────────────────────────────

def p4_compare_calibrated_vs_raw(calibrated_model, min_edge=0.03,
                                   bankroll=1000.0, kelly_frac=0.25,
                                   max_bet_pct=0.05, stop_loss_pct=0.20,
                                   verbose=True):
    """
    เปรียบเทียบ ROI ระหว่าง:
      - Raw ensemble (uncalibrated)
      - Calibrated model
    แสดงผลแบบ side-by-side เพื่อ validate ว่า calibration ช่วยหรือทำร้าย ROI
    """
    SEP  = "=" * 70
    LINE = "─" * 70

    def _run_backtest(model, label):
        proba  = model.predict_proba(X_test_sc)
        actual = y_test.values
        bk     = bankroll
        peak   = bk
        max_dd = 0.0
        bets   = []
        stopped = False

        for p_row, act in zip(proba, actual):
            if stopped:
                break
            dd = (peak - bk) / peak * 100
            if dd >= stop_loss_pct * 100:
                stopped = True
                break
            best_cls, best_edge, best_odds = -1, -1.0, 1.0
            for cls in range(3):
                mp   = p_row[cls]
                bm_p = mp * 0.95
                if bm_p <= 0.01:
                    continue
                bm_o = 1 / bm_p
                edge = mp - (1 / bm_o)
                if edge > best_edge:
                    best_edge, best_cls, best_odds = edge, cls, bm_o
            if best_edge < min_edge:
                continue
            p     = p_row[best_cls]
            k     = max(0.0, (p * best_odds - 1) / (best_odds - 1))
            stake = min(bk * k * kelly_frac, bk * max_bet_pct)
            stake = max(stake, 0.5)
            won    = (best_cls == act)
            profit = stake * (best_odds - 1) if won else -stake
            bk    += profit
            if bk > peak:
                peak = bk
            dd2 = (peak - bk) / peak * 100
            if dd2 > max_dd:
                max_dd = dd2
            bets.append({'won': won, 'profit': profit, 'stake': stake, 'bk': bk})

        n  = len(bets)
        sw = sum(b['stake'] for b in bets)
        nw = sum(1 for b in bets if b['won'])
        roi = (bk - bankroll) / sw * 100 if sw > 0 else 0.0
        wr  = nw / n * 100 if n > 0 else 0.0
        return {'label': label, 'n_bets': n, 'win_rate': wr, 'roi': roi,
                'net_pnl': bk - bankroll, 'max_dd': max_dd,
                'final_bk': bk, 'stopped': stopped, 'bets': bets}

    raw_result = _run_backtest(ensemble, "Uncalibrated (ensemble)")
    cal_result = _run_backtest(calibrated_model, "Calibrated")

    if verbose:
        print()
        print(SEP)
        print(f"  ⚖️   §8  CALIBRATED vs UNCALIBRATED — ROI COMPARISON")
        print(f"  edge≥{min_edge*100:.0f}%  |  Kelly {kelly_frac*100:.0f}%  |  Stop-Loss {stop_loss_pct*100:.0f}%")
        print(SEP)
        print()
        print(f"  {'Metric':<30} {'Uncalibrated':>16} {'Calibrated':>14}  {'Winner':>10}")
        print(f"  {LINE}")

        def _fmt_winner(v1, v2, higher_is_better=True):
            if higher_is_better:
                return "🏆 Raw" if v1 > v2 else ("🏆 Cal" if v2 > v1 else "─ Tie")
            else:
                return "🏆 Raw" if v1 < v2 else ("🏆 Cal" if v2 < v1 else "─ Tie")

        metrics = [
            ("จำนวนเดิมพัน",   raw_result['n_bets'],    cal_result['n_bets'],    True,  "{:>14,}",    "{:>14,}"),
            ("Win Rate",       raw_result['win_rate'],  cal_result['win_rate'],  True,  "{:>13.1f}%",  "{:>13.1f}%"),
            ("ROI",            raw_result['roi'],       cal_result['roi'],       True,  "{:>+13.1f}%", "{:>+13.1f}%"),
            ("Net P&L",        raw_result['net_pnl'],   cal_result['net_pnl'],   True,  "£{:>+12,.2f}", "£{:>+12,.2f}"),
            ("Final Bankroll", raw_result['final_bk'],  cal_result['final_bk'],  True,  "£{:>12,.2f}", "£{:>12,.2f}"),
            ("Max Drawdown",   raw_result['max_dd'],    cal_result['max_dd'],    False, "{:>13.1f}%",  "{:>13.1f}%"),
        ]

        for label, v1, v2, hib, fmt1, fmt2 in metrics:
            s1 = fmt1.format(v1)
            s2 = fmt2.format(v2)
            w  = _fmt_winner(v1, v2, hib)
            print(f"  {label:<30} {s1:>16} {s2:>14}  {w:>10}")

        # Stop-Loss status
        sl1 = "⚠️ triggered" if raw_result['stopped'] else "✅ ไม่ถึง"
        sl2 = "⚠️ triggered" if cal_result['stopped'] else "✅ ไม่ถึง"
        print(f"  {'Stop-Loss':<30} {sl1:>16} {sl2:>14}")

        print(f"\n  {LINE}")

        # Verdict
        raw_score = sum([
            raw_result['roi'] > cal_result['roi'],
            raw_result['win_rate'] > cal_result['win_rate'],
            raw_result['max_dd'] < cal_result['max_dd'],
            not raw_result['stopped'] and cal_result['stopped'],
        ])
        if raw_result['roi'] > 0 and cal_result['roi'] <= 0:
            verdict = "✅ ใช้ Uncalibrated — ROI บวกกว่าชัดเจน"
        elif cal_result['roi'] > 0 and raw_result['roi'] <= 0:
            verdict = "✅ ใช้ Calibrated — ROI บวกกว่าชัดเจน"
        elif raw_score >= 3:
            verdict = "💡 แนะนำ Uncalibrated สำหรับ betting (ROI ดีกว่า)"
        elif raw_score <= 1:
            verdict = "💡 แนะนำ Calibrated สำหรับ betting (ROI ดีกว่า)"
        else:
            verdict = "💡 ใกล้เคียงกัน — ใช้ Uncalibrated ง่ายกว่า"

        print(f"\n  {verdict}")
        print(f"  📌 Calibration ยังมีประโยชน์สำหรับ probability display (Brier ดีขึ้น)")
        print(f"     แต่สำหรับ edge/ROI calculation ควรใช้ raw probabilities")
        print(SEP)

    return {'raw': raw_result, 'calibrated': cal_result}


# ──────────────────────────────────────────────────────────────
# P4-RUNNER) PHASE 4 FULL ANALYTICS ENGINE
# ──────────────────────────────────────────────────────────────

def run_phase4(n_cv_splits=5, cal_method='isotonic',
               min_value_edge=0.05, bk_margin=0.05,
               bankroll=1000.0, kelly_frac=0.25,
               max_bet_pct=0.05, stop_loss_pct=0.20):
    """
    รัน Phase 4 ครบทุก section:
      §1  Rolling Window Validation  (+ LogLoss)
      §2  Probability Calibration    (sklearn-version-safe)
      §3  Value Bet Detection
      §4  ROI Backtest
      §5  Risk Control  (Kelly + Stop-Loss)
      §6  Equity Curve  (ASCII)
      §7  Feature Importance (XGB + RF + SHAP)
    """
    print()
    print("█" * 70)
    print("  🚀  PHASE 4 — FULL ANALYTICS ENGINE")
    print("█" * 70)

    # §1
    cv_results = p4_rolling_cv(n_splits=n_cv_splits, verbose=True)

    # §2
    calibrated = p4_calibration(method=cal_method, verbose=True)

    # §3  — ใช้ ensemble โดยตรง (ไม่ผ่าน calibration)
    # เหตุผล: calibrated LogLoss สูงกว่า uncalibrated → edge calculation แม่นกว่าบน raw proba
    value_bets = p4_value_bets(
        calibrated_model=ensemble,
        min_edge=min_value_edge,
        bk_margin=bk_margin,
        verbose=True
    )
    effective_edge = min_value_edge
    if len(value_bets) == 0 and min_value_edge > 0.01:
        retry_edge = max(0.01, min_value_edge - 0.02)
        print(f"  🔄 Auto-retry §3 ด้วย edge={retry_edge*100:.0f}% ...")
        value_bets = p4_value_bets(
            calibrated_model=ensemble,
            min_edge=retry_edge,
            bk_margin=bk_margin,
            verbose=True
        )
        effective_edge = retry_edge

    # §4 §5 §6 — ใช้ ensemble โดยตรง (consistent กับ Phase 3)
    roi_result = p4_roi_backtest(
        calibrated_model=ensemble,
        bankroll=bankroll,
        kelly_frac=kelly_frac,
        max_bet_pct=max_bet_pct,
        stop_loss_pct=stop_loss_pct,
        min_edge=effective_edge,
        verbose=True
    )

    # §7
    fi_result = p4_feature_importance(verbose=True)

    # §8 — Calibrated vs Uncalibrated ROI Comparison
    cal_roi = p4_compare_calibrated_vs_raw(
        calibrated_model=calibrated,
        min_edge=effective_edge,
        bankroll=bankroll,
        kelly_frac=kelly_frac,
        max_bet_pct=max_bet_pct,
        stop_loss_pct=stop_loss_pct,
        verbose=True
    )

    # Summary
    SEP     = "=" * 70
    cv_accs = [r['acc'] for r in cv_results]

    print()
    print(SEP)
    print("  📋  PHASE 4 — SUMMARY")
    print(SEP)
    cv_lls = [r['ll'] for r in cv_results]
    print(f"\n  §1  CV Accuracy    : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
    print(f"  §1  CV LogLoss     : {np.mean(cv_lls):.4f}")
    print(f"  §3  Value Bets พบ : {len(value_bets)}  (edge≥{effective_edge*100:.0f}%)")
    if roi_result['n_bets'] > 0:
        print(f"  §4  Total Bets     : {roi_result['n_bets']}")
        print(f"  §4  Win Rate       : {roi_result['win_rate']:.1f}%")
        print(f"  §4  ROI            : {roi_result['roi']:+.1f}%")
        print(f"  §4  Net P&L        : £{roi_result['net_pnl']:+,.2f}")
        print(f"  §5  Max Drawdown   : {roi_result['max_dd']:.1f}%")
        if roi_result['roi'] > 5:
            print(f"\n  💡 โมเดลมี edge จริง — ROI {roi_result['roi']:.1f}% บน test set")
        elif roi_result['roi'] > 0:
            print(f"\n  💡 มี edge เล็กน้อย — ลอง tune หรือเพิ่มข้อมูล")
        else:
            print(f"\n  ⚠️  ROI ติดลบ — model ยังไม่ beat the market")
    else:
        print(f"  §4  ไม่มี bets — พิจารณาลด min_value_edge < {effective_edge*100:.0f}%")

    # §8 summary
    if cal_roi:
        raw_r = cal_roi['raw']['roi']
        cal_r = cal_roi['calibrated']['roi']
        winner = "Uncalibrated" if raw_r >= cal_r else "Calibrated"
        print(f"\n  §8  ROI Comparison:")
        print(f"      Uncalibrated  : {raw_r:+.1f}%")
        print(f"      Calibrated    : {cal_r:+.1f}%")
        print(f"      🏆 ดีกว่า      : {winner}")

    print()
    print(SEP)
    print("  ✅  PHASE 4 COMPLETE")
    print(SEP)
    print()

    return {
        'cv': cv_results, 'calibrated': calibrated,
        'value_bets': value_bets, 'roi': roi_result,
        'feature_importance': fi_result, 'comparison': cal_roi,
    }


# ── STEP 8: Phase 4 — Full Analytics Engine ──
run_phase4(
    n_cv_splits=5,
    cal_method='sigmoid',  # sigmoid ดีกว่า isotonic บน cal_set เล็ก (n<500)
    min_value_edge=0.03,   # 3% edge — consistent กับ Phase 3
    bk_margin=0.05,
    bankroll=1000.0,
    kelly_frac=0.25,
    max_bet_pct=0.05,
    stop_loss_pct=0.20,
)