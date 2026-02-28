"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI v9.0 — FEATURE ENGINEERING                    ║
║   Data loading, Elo, rolling features, match-level features  ║
╚══════════════════════════════════════════════════════════════╝
"""

from src.config import *

# ══════════════════════════════════════════════════════════════
# 1) LOAD ALL DATA
# ══════════════════════════════════════════════════════════════

def load_data():
    print("Current directory:", os.getcwd())
    files = glob.glob(f"{DATA_DIR}/*.csv")
    # กรองเฉพาะไฟล์ที่ไม่ใช่ backup
    files = [f for f in files if 'backup' not in f.lower()]
    print("Loaded files:", files)

    df_list = [pd.read_csv(f) for f in files]
    data = pd.concat(df_list, ignore_index=True).copy()
    data['MatchID'] = data.index
    # Robust date parsing for mixed CSV formats:
    # 1) day-first pass, 2) fallback month-first for remaining NaT rows.
    date_dayfirst = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    if date_dayfirst.isna().any():
        date_monthfirst = pd.to_datetime(data.loc[date_dayfirst.isna(), 'Date'], dayfirst=False, errors='coerce')
        date_dayfirst.loc[date_dayfirst.isna()] = date_monthfirst
    data['Date'] = date_dayfirst
    data = data.sort_values('Date').reset_index(drop=True)

    # ✅ กรองเกมที่ยังไม่มีผล (FTR = NaN) — เกมในอนาคตที่ยังไม่ได้แข่ง
    if 'FTR' in data.columns:
        before = len(data)
        data = data[data['FTR'].notna()].reset_index(drop=True)
        removed = before - len(data)
        if removed > 0:
            print(f"✅ กรองเกมที่ยังไม่มีผลออก: {removed} เกม (เหลือ {len(data)} เกมที่มีผลจริง)")

    season_series = np.where(data['Date'].dt.month >= 8, data['Date'].dt.year, data['Date'].dt.year - 1)
    season_count = int(pd.Series(season_series).dropna().nunique())
    recommended_seasons = int(os.getenv("MIN_RECOMMENDED_SEASONS", "8"))

    print("\n===== DATA INFO =====")
    print("Total matches:", len(data))
    print("Date range:", data['Date'].min(), "→", data['Date'].max())
    print(f"Detected seasons: {season_count}  (recommended >= {recommended_seasons})")
    if season_count < recommended_seasons:
        print(
            f"⚠️  History is shorter than recommendation: {season_count} seasons. "
            "Model variance may be high."
        )
    return data


# ══════════════════════════════════════════════════════════════
# 2) xG & ODDS DETECTION
# ══════════════════════════════════════════════════════════════

def detect_xg_columns(data):
    _xg_home_col = next((c for c in data.columns if c.lower() in ['homexg','hxg','home_xg','xgh']), None)
    _xg_away_col = next((c for c in data.columns if c.lower() in ['awayxg','axg','away_xg','xga','xgaway']), None)

    if _xg_home_col is None and 'HomeXG' in data.columns: _xg_home_col = 'HomeXG'
    if _xg_away_col is None and 'AwayXG' in data.columns: _xg_away_col = 'AwayXG'

    xg_available = (_xg_home_col is not None and _xg_away_col is not None and
                    data[_xg_home_col].notna().sum() > 200)

    if xg_available:
        data['_HomeXG'] = pd.to_numeric(data[_xg_home_col], errors='coerce')
        data['_AwayXG'] = pd.to_numeric(data[_xg_away_col], errors='coerce')
        print(f"✅ xG columns found: {_xg_home_col}/{_xg_away_col}  "
              f"({data['_HomeXG'].notna().sum()} valid rows)")
    else:
        data['_HomeXG'] = np.nan
        data['_AwayXG'] = np.nan
        print("⚠️  xG columns NOT found — xG features will be skipped")

    return data, xg_available


def detect_odds_columns(data):
    def _find_odds_col(data, candidates):
        for c in candidates:
            if c in data.columns and pd.to_numeric(data[c], errors='coerce').notna().sum() > 200:
                return c
        return None

    if not USE_MARKET_FEATURES:
        data['_ImpH'] = np.nan; data['_ImpD'] = np.nan
        data['_ImpA'] = np.nan; data['_Overround'] = np.nan
        print("ℹ️  USE_MARKET_FEATURES=False — ปิดการใช้ market odds ทั้งหมด")
        return data, False

    _odds_h = _find_odds_col(data, ['B365H','BbAvH','PSH','WHH','MaxH','AvgH'])
    _odds_d = _find_odds_col(data, ['B365D','BbAvD','PSD','WHD','MaxD','AvgD'])
    _odds_a = _find_odds_col(data, ['B365A','BbAvA','PSA','WHA','MaxA','AvgA'])
    odds_available = all(x is not None for x in [_odds_h, _odds_d, _odds_a])

    if odds_available:
        data['_OddsH'] = pd.to_numeric(data[_odds_h], errors='coerce')
        data['_OddsD'] = pd.to_numeric(data[_odds_d], errors='coerce')
        data['_OddsA'] = pd.to_numeric(data[_odds_a], errors='coerce')
        _raw_h = 1 / data['_OddsH']; _raw_d = 1 / data['_OddsD']; _raw_a = 1 / data['_OddsA']
        _total = (_raw_h + _raw_d + _raw_a).replace(0, np.nan)
        data['_ImpH'] = _raw_h / _total
        data['_ImpD'] = _raw_d / _total
        data['_ImpA'] = _raw_a / _total
        data['_Overround'] = (_raw_h + _raw_d + _raw_a) - 1
        print(f"✅ Betting odds found: {_odds_h}/{_odds_d}/{_odds_a}")
    else:
        data['_ImpH'] = np.nan; data['_ImpD'] = np.nan
        data['_ImpA'] = np.nan; data['_Overround'] = np.nan
        print("⚠️  Betting odds NOT found — market features will be skipped")

    return data, odds_available


# ══════════════════════════════════════════════════════════════
# 3) ELO RATING — Dynamic K-factor
# ══════════════════════════════════════════════════════════════

def compute_elo(data, k_base=32, base=1500):
    elo = {}; elo_home = {}; elo_away = {}
    home_elo_before = []; away_elo_before = []
    home_elo_h_before = []; away_elo_a_before = []

    for _, row in data.iterrows():
        home = row['HomeTeam']; away = row['AwayTeam']
        hg = row['FTHG'];       ag = row['FTAG']

        if home not in elo:      elo[home]      = base
        if away not in elo:      elo[away]       = base
        if home not in elo_home: elo_home[home]  = base
        if away not in elo_away: elo_away[away]  = base

        home_elo_before.append(elo[home])
        away_elo_before.append(elo[away])
        home_elo_h_before.append(elo_home.get(home, base))
        away_elo_a_before.append(elo_away.get(away, base))

        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_away = 1 - exp_home

        if pd.isna(hg) or pd.isna(ag): continue
        hg, ag = int(hg), int(ag)

        if hg > ag:    score_home, score_away = 1.0, 0.0
        elif hg < ag:  score_home, score_away = 0.0, 1.0
        else:          score_home, score_away = 0.5, 0.5

        goal_diff = abs(hg - ag)
        k = k_base * (1 + 0.1 * min(goal_diff, 5))

        elo[home] += k * (score_home - exp_home)
        elo[away] += k * (score_away - exp_away)
        elo_home[home] = elo_home.get(home, base) + (k * 0.5) * (score_home - exp_home)
        elo_away[away] = elo_away.get(away, base) + (k * 0.5) * (score_away - exp_away)

    data = data.copy()
    data['Home_Elo']   = home_elo_before
    data['Away_Elo']   = away_elo_before
    data['Home_Elo_H'] = home_elo_h_before
    data['Away_Elo_A'] = away_elo_a_before
    data['Elo_Diff']   = data['Home_Elo'] - data['Away_Elo']
    return data, elo, elo_home, elo_away


# ══════════════════════════════════════════════════════════════
# 4) ROLLING FEATURES
# ══════════════════════════════════════════════════════════════

def rolling_shift(df, col, window=5):
    return (df.groupby('Team')[col].rolling(window).mean()
            .shift(1).reset_index(level=0, drop=True))

def ewm_shift(df, col, span=5):
    return (df.groupby('Team')[col]
            .apply(lambda x: x.ewm(span=span, adjust=False).mean().shift(1))
            .reset_index(level=0, drop=True))

def rolling_std_shift(df, col, window=5):
    return (df.groupby('Team')[col].rolling(window).std()
            .shift(1).reset_index(level=0, drop=True))


def build_team_df(data, xg_available):
    home_df = data[['MatchID','Date','HomeTeam','FTHG','FTAG',
                    'Home_Elo','Away_Elo','Home_Elo_H','Away_Elo_A','Elo_Diff',
                    '_HomeXG','_AwayXG']].copy()
    home_df.columns = ['MatchID','Date','Team','GF','GA',
                       'Own_Elo','Opp_Elo','Own_Elo_HA','Opp_Elo_HA','Elo_Diff','xGF','xGA']
    home_df['Home'] = 1

    away_df = data[['MatchID','Date','AwayTeam','FTAG','FTHG',
                    'Away_Elo','Home_Elo','Away_Elo_A','Home_Elo_H','Elo_Diff',
                    '_AwayXG','_HomeXG']].copy()
    away_df.columns = ['MatchID','Date','Team','GF','GA',
                       'Own_Elo','Opp_Elo','Own_Elo_HA','Opp_Elo_HA','Elo_Diff','xGF','xGA']
    away_df['Home'] = 0

    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df = team_df.sort_values(['Team','Date']).reset_index(drop=True)

    team_df['Win']    = (team_df['GF'] > team_df['GA']).astype(int)
    team_df['Draw']   = (team_df['GF'] == team_df['GA']).astype(int)
    team_df['Loss']   = (team_df['GF'] < team_df['GA']).astype(int)
    team_df['Points'] = team_df['Win']*3 + team_df['Draw']
    team_df['CS']     = (team_df['GA'] == 0).astype(int)
    team_df['Scored'] = (team_df['GF'] > 0).astype(int)
    team_df['GD']     = team_df['GF'] - team_df['GA']

    # Standard rolling
    team_df['GF_last5']      = rolling_shift(team_df, 'GF')
    team_df['GA_last5']      = rolling_shift(team_df, 'GA')
    team_df['GD_last5']      = rolling_shift(team_df, 'GD')
    team_df['Points_last5']  = rolling_shift(team_df, 'Points')
    team_df['Win_last5']     = rolling_shift(team_df, 'Win')
    team_df['Draw_last5']    = rolling_shift(team_df, 'Draw')
    team_df['CS_last5']      = rolling_shift(team_df, 'CS')
    team_df['Scored_last5']  = rolling_shift(team_df, 'Scored')
    team_df['Streak3']       = rolling_shift(team_df, 'Points', window=3)

    # EWM
    team_df['GF_ewm5']       = ewm_shift(team_df, 'GF')
    team_df['GA_ewm5']       = ewm_shift(team_df, 'GA')
    team_df['Pts_ewm5']      = ewm_shift(team_df, 'Points')
    team_df['GD_ewm5']       = ewm_shift(team_df, 'GD')

    # Longer window
    team_df['Points_last10'] = rolling_shift(team_df, 'Points', window=10)
    team_df['GF_last10']     = rolling_shift(team_df, 'GF', window=10)
    team_df['GD_std5']       = rolling_std_shift(team_df, 'GD')

    # Days rest
    team_df['DaysRest']      = team_df.groupby('Team')['Date'].diff().dt.days.fillna(7)
    team_df['DaysRest_lag']  = team_df.groupby('Team')['DaysRest'].shift(0)

    # xG rolling
    if xg_available:
        team_df['xGF_last5']  = rolling_shift(team_df, 'xGF')
        team_df['xGA_last5']  = rolling_shift(team_df, 'xGA')
        team_df['xGF_ewm']    = ewm_shift(team_df, 'xGF')
        team_df['xGA_ewm']    = ewm_shift(team_df, 'xGA')
        team_df['xGD_last5']  = team_df['xGF_last5'] - team_df['xGA_last5']
        team_df['xG_overperf']= rolling_shift(team_df, 'GF') - rolling_shift(team_df, 'xGF')
        team_df['xGF_slope']  = (team_df['xGF_ewm'] - rolling_shift(team_df, 'xGF', window=10)) / 0.5
        print("✅ xG rolling features computed (Phase 1 ACTIVE 🔥)")
    else:
        for col in ['xGF_last5','xGA_last5','xGF_ewm','xGA_ewm',
                    'xGD_last5','xG_overperf','xGF_slope']:
            team_df[col] = np.nan

    team_df = team_df.dropna(subset=['GF_last5','GA_last5','Points_last5'])
    print(f"✅ Rolling + EWM features computed: {len(team_df)} rows")
    return team_df


# ══════════════════════════════════════════════════════════════
# 5) SEQUENTIAL TEAM STATS (No Leakage)
# ══════════════════════════════════════════════════════════════

def compute_sequential_team_stats(data):
    home_wins = {}; home_games = {}
    away_wins = {}; away_games = {}
    home_draws = {}; home_g2 = {}
    hw_rates = []; aw_rates = []; hd_rates = []

    for _, row in data.sort_values('Date').iterrows():
        home = row['HomeTeam']; away = row['AwayTeam']
        hg = row['FTHG'];       ag = row['FTAG']

        hw_rates.append(home_wins.get(home, 0) / max(home_games.get(home, 1), 1))
        aw_rates.append(away_wins.get(away, 0) / max(away_games.get(away, 1), 1))
        hd_rates.append(home_draws.get(home, 0) / max(home_g2.get(home, 1), 1))

        if pd.isna(hg) or pd.isna(ag): continue
        hg, ag = int(hg), int(ag)

        home_games[home] = home_games.get(home, 0) + 1
        away_games[away] = away_games.get(away, 0) + 1
        home_g2[home]    = home_g2.get(home, 0) + 1

        if hg > ag:   home_wins[home]  = home_wins.get(home, 0) + 1
        elif ag > hg: away_wins[away]  = away_wins.get(away, 0) + 1
        else:         home_draws[home] = home_draws.get(home, 0) + 1

    data = data.sort_values('Date').reset_index(drop=True).copy()
    data['_HomeWinRate_seq']  = hw_rates
    data['_AwayWinRate_seq']  = aw_rates
    data['_HomeDrawRate_seq'] = hd_rates
    return data


def build_static_stats(valid_data):
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

    return home_stats, away_stats, draw_stats_home


# ══════════════════════════════════════════════════════════════
# 6) MERGE TO MATCH LEVEL + ADVANCED FEATURES
# ══════════════════════════════════════════════════════════════

def build_match_df(team_df, data, odds_available, xg_available):
    h = team_df[team_df['Home'] == 1].copy().rename(columns={
        'Team':'HomeTeam','GF_last5':'H_GF5','GA_last5':'H_GA5','GD_last5':'H_GD5',
        'Points_last5':'H_Pts5','Win_last5':'H_Win5','Draw_last5':'H_Draw5',
        'CS_last5':'H_CS5','Scored_last5':'H_Scored5','Streak3':'H_Streak3',
        'GF_ewm5':'H_GF_ewm','GA_ewm5':'H_GA_ewm','Pts_ewm5':'H_Pts_ewm','GD_ewm5':'H_GD_ewm',
        'Points_last10':'H_Pts10','GF_last10':'H_GF10','GD_std5':'H_GD_std',
        'DaysRest_lag':'H_DaysRest','Own_Elo':'H_Elo','Own_Elo_HA':'H_Elo_Home',
        'xGF_last5':'H_xGF5','xGA_last5':'H_xGA5','xGF_ewm':'H_xGF_ewm',
        'xGA_ewm':'H_xGA_ewm','xGD_last5':'H_xGD5','xG_overperf':'H_xG_overperf',
        'xGF_slope':'H_xGF_slope',
    })
    a = team_df[team_df['Home'] == 0].copy().rename(columns={
        'Team':'AwayTeam','GF_last5':'A_GF5','GA_last5':'A_GA5','GD_last5':'A_GD5',
        'Points_last5':'A_Pts5','Win_last5':'A_Win5','Draw_last5':'A_Draw5',
        'CS_last5':'A_CS5','Scored_last5':'A_Scored5','Streak3':'A_Streak3',
        'GF_ewm5':'A_GF_ewm','GA_ewm5':'A_GA_ewm','Pts_ewm5':'A_Pts_ewm','GD_ewm5':'A_GD_ewm',
        'Points_last10':'A_Pts10','GF_last10':'A_GF10','GD_std5':'A_GD_std',
        'DaysRest_lag':'A_DaysRest','Own_Elo':'A_Elo','Own_Elo_HA':'A_Elo_Away',
        'xGF_last5':'A_xGF5','xGA_last5':'A_xGA5','xGF_ewm':'A_xGF_ewm',
        'xGA_ewm':'A_xGA_ewm','xGD_last5':'A_xGD5','xG_overperf':'A_xG_overperf',
        'xGF_slope':'A_xGF_slope',
    })

    match_df = pd.merge(h, a, on='MatchID')
    actual_goals = data[['MatchID','FTHG','FTAG']].copy()
    match_df = match_df.merge(actual_goals, on='MatchID', how='left')

    if odds_available:
        odds_df = data[['MatchID','_ImpH','_ImpD','_ImpA','_Overround']].copy()
        match_df = match_df.merge(odds_df, on='MatchID', how='left')
    else:
        match_df['_ImpH'] = np.nan; match_df['_ImpD'] = np.nan
        match_df['_ImpA'] = np.nan; match_df['_Overround'] = np.nan

    print(f"✅ Matches after feature engineering: {len(match_df)}")
    return match_df


def add_advanced_features(match_df, xg_available, odds_available):
    # Difference features
    match_df['Diff_Pts']      = match_df['H_Pts5']    - match_df['A_Pts5']
    match_df['Diff_GF']       = match_df['H_GF5']     - match_df['A_GF5']
    match_df['Diff_GA']       = match_df['H_GA5']     - match_df['A_GA5']
    match_df['Diff_GD']       = match_df['H_GD5']     - match_df['A_GD5']
    match_df['Diff_Win']      = match_df['H_Win5']    - match_df['A_Win5']
    match_df['Diff_CS']       = match_df['H_CS5']     - match_df['A_CS5']
    match_df['Diff_Streak']   = match_df['H_Streak3'] - match_df['A_Streak3']
    match_df['Diff_Elo']      = match_df['H_Elo']     - match_df['A_Elo']
    match_df['Diff_Scored']   = match_df['H_Scored5'] - match_df['A_Scored5']
    match_df['Diff_Pts_ewm']  = match_df['H_Pts_ewm'] - match_df['A_Pts_ewm']
    match_df['Diff_GF_ewm']   = match_df['H_GF_ewm']  - match_df['A_GF_ewm']
    match_df['Diff_GD_ewm']   = match_df['H_GD_ewm']  - match_df['A_GD_ewm']

    # Momentum
    match_df['H_Momentum']    = match_df['H_Pts5']    - match_df['H_Pts10'] / 2
    match_df['A_Momentum']    = match_df['A_Pts5']    - match_df['A_Pts10'] / 2
    match_df['Diff_Momentum'] = match_df['H_Momentum'] - match_df['A_Momentum']

    # Days rest
    match_df['Diff_DaysRest'] = match_df['H_DaysRest'] - match_df['A_DaysRest']
    match_df['H_DaysRest']    = match_df['H_DaysRest'].clip(1, 21)
    match_df['A_DaysRest']    = match_df['A_DaysRest'].clip(1, 21)

    # Draw-specific
    match_df['Diff_Draw5']    = match_df['H_Draw5']   - match_df['A_Draw5']
    match_df['Combined_GF']   = match_df['H_GF5']     + match_df['A_GF5']
    match_df['Combined_GF_ewm'] = match_df['H_GF_ewm'] + match_df['A_GF_ewm']
    match_df['Mean_GD_std']   = (match_df['H_GD_std'].fillna(2) + match_df['A_GD_std'].fillna(2)) / 2

    # Elo
    match_df['H_Elo_norm']     = match_df['H_Elo']      / 1500
    match_df['A_Elo_norm']     = match_df['A_Elo']      / 1500
    match_df['Elo_ratio']      = match_df['H_Elo']      / (match_df['A_Elo'] + 1)
    match_df['H_Elo_Home_norm']= match_df['H_Elo_Home'] / 1500
    match_df['A_Elo_Away_norm']= match_df['A_Elo_Away'] / 1500

    # Seasonal
    match_df['Month']       = match_df['Date_x'].dt.month
    match_df['SeasonPhase'] = match_df['Month'].map(
        lambda m: 1 if m in [8,9,10] else (2 if m in [11,12,1,2] else 3))

    # S4: Deep features
    match_df['H_Form_slope']   = (match_df['H_Pts_ewm'] - match_df['H_Pts10'] / 2) / (match_df['H_GD_std'].fillna(1) + 0.5)
    match_df['A_Form_slope']   = (match_df['A_Pts_ewm'] - match_df['A_Pts10'] / 2) / (match_df['A_GD_std'].fillna(1) + 0.5)
    match_df['Diff_Form_slope']= match_df['H_Form_slope'] - match_df['A_Form_slope']
    match_df['H_HomeAdvantage']= match_df['H_Elo_Home'] / (match_df['H_Elo'] + 1)
    match_df['A_AwayPenalty']  = match_df['A_Elo_Away'] / (match_df['A_Elo'] + 1)
    match_df['Venue_edge']     = match_df['H_HomeAdvantage'] - match_df['A_AwayPenalty']
    match_df['H_AttackIdx']    = match_df['H_GF_ewm'] / (match_df['A_GA_ewm'].clip(0.3) + 0.01)
    match_df['A_AttackIdx']    = match_df['A_GF_ewm'] / (match_df['H_GA_ewm'].clip(0.3) + 0.01)
    match_df['Diff_AttackIdx'] = match_df['H_AttackIdx'] - match_df['A_AttackIdx']
    match_df['H_DefStr']       = match_df['H_CS5'] / (match_df['H_GA5'].clip(0.1) + 0.1)
    match_df['A_DefStr']       = match_df['A_CS5'] / (match_df['A_GA5'].clip(0.1) + 0.1)
    match_df['Diff_DefStr']    = match_df['H_DefStr'] - match_df['A_DefStr']
    match_df['Elo_closeness']  = 1 / (np.abs(match_df['Diff_Elo']) + 50)
    match_df['Form_closeness'] = 1 / (np.abs(match_df['Diff_Pts_ewm']) + 0.5)
    match_df['Draw_likelihood']= match_df['Elo_closeness'] * match_df['Form_closeness'] * match_df['Mean_GD_std'].clip(0.1)
    print("✅ Deep Feature Engineering (S4) computed")

    # v9 STEP 1: Draw-focused features
    match_df['Abs_Elo_diff'] = np.abs(match_df['Diff_Elo'])
    if xg_available and 'Diff_xGF_ewm' in match_df.columns:
        match_df['Abs_xGF_diff'] = np.abs(match_df['Diff_xGF_ewm'])
    else:
        match_df['Abs_xGF_diff'] = np.abs(match_df['Diff_GF_ewm'])
    match_df['GF_balance']  = 1 / (match_df['Abs_xGF_diff'] + 0.3)
    match_df['GA_balance']  = 1 / (np.abs(match_df['H_GA_ewm'] - match_df['A_GA_ewm']) + 0.3)
    if xg_available and 'H_xGF_ewm' in match_df.columns and 'A_xGF_ewm' in match_df.columns:
        _xg_sum = match_df['H_xGF_ewm'] + match_df['A_xGF_ewm']
        match_df['xG_tightness'] = 1 / (match_df['Abs_xGF_diff'] + 0.3) / (_xg_sum.clip(0.5) + 0.5)
    else:
        match_df['xG_tightness'] = match_df['GF_balance']
    match_df['Draw_EloXForm']    = match_df['Elo_closeness'] * match_df['Form_closeness']
    match_df['Late_season_draw'] = (match_df['SeasonPhase'] == 3).astype(int) * \
                                   (1 - np.abs(match_df['Elo_ratio'] - 1).clip(0, 0.3) / 0.3)
    match_df['Combined_GF_ewm']  = match_df['H_GF_ewm'] + match_df['A_GF_ewm']
    print("✅ Draw-focused features (v9 STEP 1): +8 features")

    # xG match-level
    if xg_available:
        match_df['Diff_xGF']         = match_df['H_xGF5']      - match_df['A_xGF5']
        match_df['Diff_xGA']         = match_df['H_xGA5']      - match_df['A_xGA5']
        match_df['Diff_xGD']         = match_df['H_xGD5']      - match_df['A_xGD5']
        match_df['Diff_xGF_ewm']     = match_df['H_xGF_ewm']   - match_df['A_xGF_ewm']
        match_df['Diff_xG_overperf'] = match_df['H_xG_overperf'] - match_df['A_xG_overperf']
        match_df['Diff_xGF_slope']   = match_df['H_xGF_slope']   - match_df['A_xGF_slope']
        match_df['H_xAttackIdx']     = match_df['H_xGF_ewm'] / (match_df['A_xGA_ewm'].clip(0.3) + 0.01)
        match_df['A_xAttackIdx']     = match_df['A_xGF_ewm'] / (match_df['H_xGA_ewm'].clip(0.3) + 0.01)
        match_df['Diff_xAttackIdx']  = match_df['H_xAttackIdx'] - match_df['A_xAttackIdx']
        print("✅ xG match-level features computed")

    # Market features
    if USE_MARKET_FEATURES and odds_available:
        match_df['Mkt_ImpH']    = match_df['_ImpH']
        match_df['Mkt_ImpD']    = match_df['_ImpD']
        match_df['Mkt_ImpA']    = match_df['_ImpA']
        match_df['Mkt_Spread']  = match_df['_ImpH'] - match_df['_ImpA']
        match_df['Mkt_DrawPrem']= match_df['_ImpD'] - 0.26
        match_df['Mkt_Overround']= match_df['_Overround']
    else:
        for col in ['Mkt_ImpH','Mkt_ImpD','Mkt_ImpA','Mkt_Spread','Mkt_DrawPrem','Mkt_Overround']:
            match_df[col] = np.nan

    return match_df


# ══════════════════════════════════════════════════════════════
# 7) H2H + TARGET
# ══════════════════════════════════════════════════════════════

def compute_h2h(data):
    h2h_home_wins = {}; h2h_draws = {}; h2h_total = {}
    h2h_rates = []; h2h_draw_rates = []

    for _, row in data.sort_values('Date_x').iterrows():
        home = row['HomeTeam']; away = row['AwayTeam']
        key  = tuple(sorted([home, away]))
        total = max(h2h_total.get(key, 0), 1)
        h2h_rates.append(h2h_home_wins.get((home, away), 0) / total)
        h2h_draw_rates.append(h2h_draws.get(key, 0) / total)

        if key not in h2h_total:           h2h_total[key] = 0
        if (home, away) not in h2h_home_wins: h2h_home_wins[(home, away)] = 0
        if key not in h2h_draws:           h2h_draws[key] = 0
        h2h_total[key] += 1
        if row['Win_x'] == 1:   h2h_home_wins[(home, away)] += 1
        if row['Draw_x'] == 1:  h2h_draws[key] += 1

    return h2h_rates, h2h_draw_rates


def get_result(row):
    if row['Win_x'] == 1:    return 2
    elif row['Draw_x'] == 1: return 1
    else:                    return 0


# ══════════════════════════════════════════════════════════════
# 8) FEATURE LIST
# ══════════════════════════════════════════════════════════════

BASE_FEATURES = [
    'Diff_Elo', 'Elo_ratio', 'H_Elo_norm', 'A_Elo_norm',
    'H_Elo_Home_norm', 'A_Elo_Away_norm',
    'H_GF5', 'H_GA5', 'H_Pts5', 'H_Streak3', 'H_CS5', 'H_Scored5',
    'A_GF5', 'A_GA5', 'A_Pts5', 'A_Streak3', 'A_CS5', 'A_Scored5',
    'Diff_Pts', 'Diff_GF', 'Diff_GA', 'Diff_GD',
    'Diff_Win', 'Diff_CS', 'Diff_Streak', 'Diff_Scored',
    'H_GF_ewm', 'H_GA_ewm', 'H_Pts_ewm',
    'A_GF_ewm', 'A_GA_ewm', 'A_Pts_ewm',
    'Diff_Pts_ewm', 'Diff_GF_ewm', 'Diff_GD_ewm',
    'Diff_Momentum',
    'H_Draw5', 'A_Draw5', 'Diff_Draw5',
    'H2H_DrawRate', 'Combined_GF', 'Mean_GD_std',
    'H2H_HomeWinRate',
    'HomeWinRate', 'AwayWinRate', 'HomeDrawRate',
    'H_DaysRest', 'A_DaysRest', 'Diff_DaysRest',
    'Month', 'SeasonPhase',
    'H_Form_slope', 'A_Form_slope', 'Diff_Form_slope',
    'H_HomeAdvantage', 'A_AwayPenalty', 'Venue_edge',
    'H_AttackIdx', 'A_AttackIdx', 'Diff_AttackIdx',
    'H_DefStr', 'A_DefStr', 'Diff_DefStr',
    'Elo_closeness', 'Form_closeness', 'Draw_likelihood',
    'Abs_Elo_diff', 'Abs_xGF_diff',
    'GF_balance', 'GA_balance',
    'xG_tightness', 'Draw_EloXForm',
    'Late_season_draw', 'Combined_GF_ewm',
]

XG_FEATURES = [
    'H_xGF5', 'H_xGA5', 'H_xGD5', 'H_xGF_ewm', 'H_xGA_ewm',
    'H_xG_overperf', 'H_xGF_slope',
    'A_xGF5', 'A_xGA5', 'A_xGD5', 'A_xGF_ewm', 'A_xGA_ewm',
    'A_xG_overperf', 'A_xGF_slope',
    'Diff_xGF', 'Diff_xGA', 'Diff_xGD', 'Diff_xGF_ewm',
    'Diff_xG_overperf', 'Diff_xGF_slope',
    'H_xAttackIdx', 'A_xAttackIdx', 'Diff_xAttackIdx',
]

MKT_FEATURES = [
    'Mkt_ImpH', 'Mkt_ImpD', 'Mkt_ImpA',
    'Mkt_Spread', 'Mkt_DrawPrem', 'Mkt_Overround',
]

LOW_IMPORTANCE_FEATURES = [
    'H_CS5', 'A_CS5', 'Diff_CS', 'Diff_Scored',
    'H_Draw5', 'H2H_DrawRate', 'H2H_HomeWinRate',
]


def build_feature_list(match_df, xg_available, odds_available):
    features = BASE_FEATURES.copy()

    if xg_available:
        features += [f for f in XG_FEATURES if f in match_df.columns]
        print(f"✅ Phase 1 xG: +{len([f for f in XG_FEATURES if f in match_df.columns])} features")

    if USE_MARKET_FEATURES and odds_available:
        features += [f for f in MKT_FEATURES if f in match_df.columns]
        print(f"✅ Phase 2 Market: +{len([f for f in MKT_FEATURES if f in match_df.columns])} features")

    # กรองเฉพาะ columns ที่มีจริง
    features = [f for f in features if f in match_df.columns]

    # Prune low-importance
    features_pruned = [f for f in features if f not in LOW_IMPORTANCE_FEATURES]
    print(f"✅ Features v5.0: {len(features)} ตัว  "
          f"(xG={'✅' if xg_available else '❌'}  Market={'✅' if odds_available else '❌'})")
    print(f"✅ Features v5.1 (pruned): {len(features_pruned)} ตัว  (-{len(features)-len(features_pruned)} low-importance)")

    return features_pruned


def get_season(date):
    if pd.isna(date): return np.nan
    return date.year if date.month >= 8 else date.year - 1


# ══════════════════════════════════════════════════════════════
# 9) MAIN PIPELINE — run all steps and return globals
# ══════════════════════════════════════════════════════════════

def _validate_form_sanity(match_df, top_n=5, draw_thresh=0.7):
    """
    ตรวจจับทีมที่มี form ผิดปกติ (เช่น เสมอ/แพ้ต่อเนื่องผิดธรรมชาติ)
    พิมพ์ warning แต่ไม่แก้ข้อมูล — เพราะอาจเป็นข้อมูลจริง
    """
    issues = []
    for team in match_df['HomeTeam'].unique():
        h_rows = match_df[match_df['HomeTeam'] == team].sort_values('Date_x').tail(top_n)
        a_rows = match_df[match_df['AwayTeam'] == team].sort_values('Date_x').tail(top_n)
        all_rows = pd.concat([
            h_rows[['Date_x','H_Draw5']].rename(columns={'H_Draw5':'Draw5','Date_x':'Date'}),
            a_rows[['Date_x','A_Draw5']].rename(columns={'A_Draw5':'Draw5','Date_x':'Date'}),
        ]).sort_values('Date').tail(top_n)
        if len(all_rows) >= top_n:
            draw_rate = all_rows['Draw5'].iloc[-1]
            if draw_rate >= draw_thresh:
                issues.append((team, round(draw_rate, 2)))
    if issues:
        print(f"\n  ⚠️  FORM SANITY — ทีมที่มี draw rate สูงผิดปกติ (last 5):")
        for team, dr in sorted(issues, key=lambda x: -x[1]):
            print(f"     {team:<22} draw_rate={dr:.0%} — อาจเป็น data จริง หรือ API issue")


def run_feature_pipeline():
    # Load
    data = load_data()
    data, XG_AVAILABLE  = detect_xg_columns(data)
    data, ODDS_AVAILABLE = detect_odds_columns(data)

    # Elo
    data, final_elo, final_elo_home, final_elo_away = compute_elo(data)
    print("✅ Dynamic Elo computed")

    # Sequential stats
    data = compute_sequential_team_stats(data)
    print("✅ Sequential team stats computed (No Leakage!)")

    # Static stats
    valid_data = data.dropna(subset=['FTHG','FTAG'])
    home_stats, away_stats, draw_stats_home = build_static_stats(valid_data)

    # Team rolling features
    team_df = build_team_df(data, XG_AVAILABLE)

    # Match-level
    match_df = build_match_df(team_df, data, ODDS_AVAILABLE, XG_AVAILABLE)
    match_df = add_advanced_features(match_df, XG_AVAILABLE, ODDS_AVAILABLE)

    # H2H
    match_df = match_df.sort_values('Date_x').reset_index(drop=True)
    h2h_win_rates, h2h_draw_rates = compute_h2h(match_df)
    match_df['H2H_HomeWinRate'] = h2h_win_rates
    match_df['H2H_DrawRate']    = h2h_draw_rates
    print("✅ H2H (Win + Draw rate) computed")

    # Sequential stats merge
    seq_stats = data[['MatchID', '_HomeWinRate_seq', '_AwayWinRate_seq', '_HomeDrawRate_seq']].copy()
    match_df = match_df.merge(seq_stats, on='MatchID', how='left')
    match_df['HomeWinRate']  = match_df['_HomeWinRate_seq'].fillna(0.45)
    match_df['AwayWinRate']  = match_df['_AwayWinRate_seq'].fillna(0.30)
    match_df['HomeDrawRate'] = match_df['_HomeDrawRate_seq'].fillna(0.25)
    match_df.drop(columns=['_HomeWinRate_seq','_AwayWinRate_seq','_HomeDrawRate_seq'], inplace=True)
    print("✅ Sequential (leak-free) stats merged into match_df")

    # Target
    match_df['Result3'] = match_df.apply(get_result, axis=1)

    # Feature list
    FEATURES = build_feature_list(match_df, XG_AVAILABLE, ODDS_AVAILABLE)

    # Core features (>= 95% non-NaN)
    core_threshold = 0.95
    core_stats = match_df[FEATURES].notna().mean()
    CORE_FEATURES = [f for f in FEATURES if core_stats.get(f, 0) >= core_threshold]
    if len(CORE_FEATURES) < 20:
        CORE_FEATURES = FEATURES.copy()
    else:
        print(f"✅ CORE_FEATURES selected: {len(CORE_FEATURES)} / {len(FEATURES)} (>= {core_threshold*100:.0f}% non-NaN)")

    match_df_clean = match_df.dropna(subset=CORE_FEATURES + ['Result3']).reset_index(drop=True)
    print(f"✅ match_df_clean rows after CORE_FEATURES dropna: {len(match_df_clean)} จากทั้งหมด {len(match_df)}")

    # Form sanity check
    try:
        _validate_form_sanity(match_df_clean)
    except Exception:
        pass

    # Bootstrap Elo สำหรับ promoted teams ที่ยังไม่มี Elo
    from src.config import NEW_TEAMS_BOOTSTRAPPED
    for team, info in NEW_TEAMS_BOOTSTRAPPED.items():
        if team not in final_elo:
            final_elo[team]      = info.get('elo', 1450)
            final_elo_home[team] = info.get('elo', 1450)
            final_elo_away[team] = info.get('elo', 1450)
            print(f"  ℹ️  Bootstrap Elo for {team}: {info.get('elo', 1450)}")

    return {
        'data':             data,
        'match_df':         match_df,
        'match_df_clean':   match_df_clean,
        'FEATURES':         FEATURES,
        'CORE_FEATURES':    CORE_FEATURES,
        'XG_AVAILABLE':     XG_AVAILABLE,
        'ODDS_AVAILABLE':   ODDS_AVAILABLE,
        'final_elo':        final_elo,
        'final_elo_home':   final_elo_home,
        'final_elo_away':   final_elo_away,
        'home_stats':       home_stats,
        'away_stats':       away_stats,
        'draw_stats_home':  draw_stats_home,
    }