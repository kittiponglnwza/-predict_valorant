"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI v9.0 — PREDICT                                ║
║   predict_match, predict_score, season sim, API fixtures    ║
╚══════════════════════════════════════════════════════════════╝
"""

from src.config import *
from src.model import (
    predict_2stage, apply_thresholds, suppress_draw_proba,
    blend_ml_poisson_dynamic, poisson_win_draw_loss,
    TwoStageEnsemble,
)

import requests
from datetime import datetime, timedelta


# ══════════════════════════════════════════════════════════════
# GET LATEST FEATURES FOR A TEAM
# ══════════════════════════════════════════════════════════════

def get_latest_features(team, is_home, ctx):
    """ดึง features ล่าสุดของทีม — bootstrap สำหรับทีมใหม่"""
    from src.config import NEW_TEAMS_BOOTSTRAPPED, get_bootstrap_features
    # ใช้ match_df (ไม่ filter) เพื่อให้ได้ข้อมูลล่าสุดจริง
    # match_df_clean ตัดหลายแมตช์ออกเพราะ NaN → form อาจเก่า
    match_df = ctx.get('match_df', ctx['match_df_clean'])
    final_elo = ctx['final_elo']

    # ── ทีมที่เพิ่งเลื่อนชั้น — ไม่มี historical data ────────────────
    if team in NEW_TEAMS_BOOTSTRAPPED:
        feats = get_bootstrap_features(team)
        feats['Elo_HA'] = final_elo.get(team, feats['Elo_HA'])
        return feats

    def _safe(row, col, fallback):
        v = row.get(col, np.nan) if isinstance(row, dict) else getattr(row, col, np.nan)
        try:
            v = float(v)
        except Exception:
            v = np.nan
        return fallback if pd.isna(v) else v

    if is_home:
        rows = match_df[match_df['HomeTeam'] == team].sort_values('Date_x')
        if len(rows) > 0:
            last = rows.iloc[-1]
            return {
                'GF5':        _safe(last, 'H_GF5', 1.5),
                'GA5':        _safe(last, 'H_GA5', 1.5),
                'GD5':        _safe(last, 'H_GD5', 0.0),
                'Pts5':       _safe(last, 'H_Pts5', 1.5),
                'Streak3':    _safe(last, 'H_Streak3', 1.5),
                'Win5':       _safe(last, 'H_Win5', 0.5),
                'Draw5':      _safe(last, 'H_Draw5', 0.25),
                'CS5':        _safe(last, 'H_CS5', 0.2),
                'Scored5':    _safe(last, 'H_Scored5', 0.6),
                'GF_ewm':     _safe(last, 'H_GF_ewm', 1.5),
                'GA_ewm':     _safe(last, 'H_GA_ewm', 1.5),
                'Pts_ewm':    _safe(last, 'H_Pts_ewm', 1.5),
                'GD_ewm':     _safe(last, 'H_GD_ewm', 0.0),
                'Pts10':      _safe(last, 'H_Pts10', 1.5),
                'DaysRest':   _safe(last, 'H_DaysRest', 7),
                'GD_std':     _safe(last, 'H_GD_std', 1.5),
                'Elo_HA':     _safe(last, 'H_Elo_Home', 1500),
                'xGF5':       _safe(last, 'H_xGF5', np.nan),
                'xGA5':       _safe(last, 'H_xGA5', np.nan),
                'xGD5':       _safe(last, 'H_xGD5', np.nan),
                'xGF_ewm':    _safe(last, 'H_xGF_ewm', np.nan),
                'xGA_ewm':    _safe(last, 'H_xGA_ewm', np.nan),
                'xG_overperf':_safe(last, 'H_xG_overperf', np.nan),
                'xGF_slope':  _safe(last, 'H_xGF_slope', np.nan),
            }
    else:
        rows = match_df[match_df['AwayTeam'] == team].sort_values('Date_x')
        if len(rows) > 0:
            last = rows.iloc[-1]
            return {
                'GF5':        _safe(last, 'A_GF5', 1.5),
                'GA5':        _safe(last, 'A_GA5', 1.5),
                'GD5':        _safe(last, 'A_GD5', 0.0),
                'Pts5':       _safe(last, 'A_Pts5', 1.5),
                'Streak3':    _safe(last, 'A_Streak3', 1.5),
                'Win5':       _safe(last, 'A_Win5', 0.5),
                'Draw5':      _safe(last, 'A_Draw5', 0.25),
                'CS5':        _safe(last, 'A_CS5', 0.2),
                'Scored5':    _safe(last, 'A_Scored5', 0.6),
                'GF_ewm':     _safe(last, 'A_GF_ewm', 1.5),
                'GA_ewm':     _safe(last, 'A_GA_ewm', 1.5),
                'Pts_ewm':    _safe(last, 'A_Pts_ewm', 1.5),
                'GD_ewm':     _safe(last, 'A_GD_ewm', 0.0),
                'Pts10':      _safe(last, 'A_Pts10', 1.5),
                'DaysRest':   _safe(last, 'A_DaysRest', 7),
                'GD_std':     _safe(last, 'A_GD_std', 1.5),
                'Elo_HA':     _safe(last, 'A_Elo_Away', 1500),
                'xGF5':       _safe(last, 'A_xGF5', np.nan),
                'xGA5':       _safe(last, 'A_xGA5', np.nan),
                'xGD5':       _safe(last, 'A_xGD5', np.nan),
                'xGF_ewm':    _safe(last, 'A_xGF_ewm', np.nan),
                'xGA_ewm':    _safe(last, 'A_xGA_ewm', np.nan),
                'xG_overperf':_safe(last, 'A_xG_overperf', np.nan),
                'xGF_slope':  _safe(last, 'A_xGF_slope', np.nan),
            }

    # Default fallback (ไม่มีข้อมูลเลย)
    return {
        'GF5': 1.5, 'GA5': 1.5, 'GD5': 0.0, 'Pts5': 1.5,
        'Streak3': 1.5, 'Win5': 0.5, 'Draw5': 0.25, 'CS5': 0.2,
        'Scored5': 0.6, 'GF_ewm': 1.5, 'GA_ewm': 1.5,
        'Pts_ewm': 1.5, 'GD_ewm': 0.0, 'Pts10': 1.5,
        'DaysRest': 7, 'GD_std': 1.5, 'Elo_HA': 1500,
        'xGF5': np.nan, 'xGA5': np.nan, 'xGD5': np.nan,
        'xGF_ewm': np.nan, 'xGA_ewm': np.nan,
        'xG_overperf': np.nan, 'xGF_slope': np.nan,
    }


# ══════════════════════════════════════════════════════════════
# BUILD MATCH ROW (feature vector สำหรับทำนาย)
# ══════════════════════════════════════════════════════════════

def build_match_row(home_team, away_team, ctx, match_date=None):
    """สร้าง feature row สำหรับทำนายแมตช์"""
    match_df_clean = ctx['match_df_clean']
    final_elo      = ctx['final_elo']
    final_elo_home = ctx['final_elo_home']
    final_elo_away = ctx['final_elo_away']
    home_stats     = ctx['home_stats']
    away_stats     = ctx['away_stats']
    draw_stats_home= ctx['draw_stats_home']
    XG_AVAILABLE   = ctx['XG_AVAILABLE']

    if match_date is None:
        match_date = TODAY
    month        = match_date.month if hasattr(match_date, 'month') else TODAY.month
    season_phase = 1 if month in [8,9,10] else (2 if month in [11,12,1,2] else 3)

    h = get_latest_features(home_team, is_home=True,  ctx=ctx)
    a = get_latest_features(away_team, is_home=False, ctx=ctx)

    h_elo      = final_elo.get(home_team, 1500)
    a_elo      = final_elo.get(away_team, 1500)
    h_elo_home = final_elo_home.get(home_team, 1500)
    a_elo_away = final_elo_away.get(away_team, 1500)

    # H2H
    h2h_rows = match_df_clean[
        (match_df_clean['HomeTeam'] == home_team) &
        (match_df_clean['AwayTeam'] == away_team)
    ]
    h2h_rate      = h2h_rows['H2H_HomeWinRate'].iloc[-1] if len(h2h_rows) > 0 else 0.33
    h2h_draw_rate = h2h_rows['H2H_DrawRate'].iloc[-1]    if len(h2h_rows) > 0 else 0.25

    # Static win/draw rates
    h_st = home_stats[home_stats['Team'] == home_team]
    a_st = away_stats[away_stats['Team'] == away_team]
    d_st = draw_stats_home[draw_stats_home['Team'] == home_team]
    home_win_rate  = h_st['HomeWinRate'].values[0]  if len(h_st) > 0 else 0.45
    away_win_rate  = a_st['AwayWinRate'].values[0]  if len(a_st) > 0 else 0.30
    home_draw_rate = d_st['HomeDrawRate'].values[0] if len(d_st) > 0 else 0.25

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
        # Form
        'H_GF5': h['GF5'],    'H_GA5': h['GA5'],    'H_Pts5': h['Pts5'],
        'H_Streak3': h['Streak3'], 'H_CS5': h['CS5'], 'H_Scored5': h['Scored5'],
        'A_GF5': a['GF5'],    'A_GA5': a['GA5'],    'A_Pts5': a['Pts5'],
        'A_Streak3': a['Streak3'], 'A_CS5': a['CS5'], 'A_Scored5': a['Scored5'],
        # Diffs
        'Diff_Pts':    h['Pts5']    - a['Pts5'],
        'Diff_GF':     h['GF5']     - a['GF5'],
        'Diff_GA':     h['GA5']     - a['GA5'],
        'Diff_GD':     h['GD5']     - a['GD5'],
        'Diff_Win':    h['Win5']    - a['Win5'],
        'Diff_CS':     h['CS5']     - a['CS5'],
        'Diff_Streak': h['Streak3'] - a['Streak3'],
        'Diff_Scored': h['Scored5'] - a['Scored5'],
        # EWM
        'H_GF_ewm': h['GF_ewm'], 'H_GA_ewm': h['GA_ewm'], 'H_Pts_ewm': h['Pts_ewm'],
        'A_GF_ewm': a['GF_ewm'], 'A_GA_ewm': a['GA_ewm'], 'A_Pts_ewm': a['Pts_ewm'],
        'Diff_Pts_ewm': h['Pts_ewm'] - a['Pts_ewm'],
        'Diff_GF_ewm':  h['GF_ewm']  - a['GF_ewm'],
        'Diff_GD_ewm':  h['GD_ewm']  - a['GD_ewm'],
        # Momentum
        'Diff_Momentum': momentum_h - momentum_a,
        # Draw features
        'H_Draw5': h['Draw5'], 'A_Draw5': a['Draw5'],
        'Diff_Draw5':     h['Draw5']  - a['Draw5'],
        'H2H_DrawRate':   h2h_draw_rate,
        'Combined_GF':    h['GF5']    + a['GF5'],
        'Mean_GD_std':    (h['GD_std'] + a['GD_std']) / 2,
        # H2H
        'H2H_HomeWinRate': h2h_rate,
        # Strength
        'HomeWinRate':  home_win_rate,
        'AwayWinRate':  away_win_rate,
        'HomeDrawRate': home_draw_rate,
        # Days Rest
        'H_DaysRest':   min(h['DaysRest'], 21),
        'A_DaysRest':   min(a['DaysRest'], 21),
        'Diff_DaysRest':min(h['DaysRest'], 21) - min(a['DaysRest'], 21),
        # Seasonal
        'Month':       month,
        'SeasonPhase': season_phase,
        # Deep Features (S4)
        'H_Form_slope':   (h['Pts_ewm'] - h['Pts10']/2) / (h['GD_std'] + 0.5),
        'A_Form_slope':   (a['Pts_ewm'] - a['Pts10']/2) / (a['GD_std'] + 0.5),
        'Diff_Form_slope':((h['Pts_ewm'] - h['Pts10']/2) / (h['GD_std'] + 0.5) -
                           (a['Pts_ewm'] - a['Pts10']/2) / (a['GD_std'] + 0.5)),
        'H_HomeAdvantage': h_elo_home / (h_elo + 1),
        'A_AwayPenalty':   a_elo_away / (a_elo + 1),
        'Venue_edge':      h_elo_home / (h_elo + 1) - a_elo_away / (a_elo + 1),
        'H_AttackIdx':     h['GF_ewm'] / (max(a['GA_ewm'], 0.3) + 0.01),
        'A_AttackIdx':     a['GF_ewm'] / (max(h['GA_ewm'], 0.3) + 0.01),
        'Diff_AttackIdx':  h['GF_ewm'] / (max(a['GA_ewm'], 0.3) + 0.01) -
                           a['GF_ewm'] / (max(h['GA_ewm'], 0.3) + 0.01),
        'H_DefStr':       h['CS5'] / (max(h['GA5'], 0.1) + 0.1),
        'A_DefStr':       a['CS5'] / (max(a['GA5'], 0.1) + 0.1),
        'Diff_DefStr':    h['CS5'] / (max(h['GA5'], 0.1) + 0.1) -
                          a['CS5'] / (max(a['GA5'], 0.1) + 0.1),
        'Elo_closeness':  1 / (abs(h_elo - a_elo) + 50),
        'Form_closeness': 1 / (abs(h['Pts_ewm'] - a['Pts_ewm']) + 0.5),
        'Draw_likelihood':1 / (abs(h_elo - a_elo) + 50) *
                          1 / (abs(h['Pts_ewm'] - a['Pts_ewm']) + 0.5) *
                          max((h['GD_std'] + a['GD_std']) / 2, 0.1),
        # xG features
        'H_xGF5':        h.get('xGF5', np.nan),
        'H_xGA5':        h.get('xGA5', np.nan),
        'H_xGD5':        h.get('xGD5', np.nan),
        'H_xGF_ewm':     h.get('xGF_ewm', np.nan),
        'H_xGA_ewm':     h.get('xGA_ewm', np.nan),
        'H_xG_overperf': h.get('xG_overperf', np.nan),
        'H_xGF_slope':   h.get('xGF_slope', np.nan),
        'A_xGF5':        a.get('xGF5', np.nan),
        'A_xGA5':        a.get('xGA5', np.nan),
        'A_xGD5':        a.get('xGD5', np.nan),
        'A_xGF_ewm':     a.get('xGF_ewm', np.nan),
        'A_xGA_ewm':     a.get('xGA_ewm', np.nan),
        'A_xG_overperf': a.get('xG_overperf', np.nan),
        'A_xGF_slope':   a.get('xGF_slope', np.nan),
        'Diff_xGF':      h.get('xGF5', np.nan)   - a.get('xGF5', np.nan)   if XG_AVAILABLE else np.nan,
        'Diff_xGA':      h.get('xGA5', np.nan)   - a.get('xGA5', np.nan)   if XG_AVAILABLE else np.nan,
        'Diff_xGD':      h.get('xGD5', np.nan)   - a.get('xGD5', np.nan)   if XG_AVAILABLE else np.nan,
        'Diff_xGF_ewm':  h.get('xGF_ewm', np.nan) - a.get('xGF_ewm', np.nan) if XG_AVAILABLE else np.nan,
        'Diff_xG_overperf': h.get('xG_overperf', np.nan) - a.get('xG_overperf', np.nan) if XG_AVAILABLE else np.nan,
        'Diff_xGF_slope':   h.get('xGF_slope', np.nan) - a.get('xGF_slope', np.nan) if XG_AVAILABLE else np.nan,
        'H_xAttackIdx':  h.get('xGF_ewm', np.nan) / (max(a.get('xGA_ewm', 0.3) or 0.3, 0.3) + 0.01) if XG_AVAILABLE else np.nan,
        'A_xAttackIdx':  a.get('xGF_ewm', np.nan) / (max(h.get('xGA_ewm', 0.3) or 0.3, 0.3) + 0.01) if XG_AVAILABLE else np.nan,
        'Diff_xAttackIdx': (h.get('xGF_ewm', np.nan) / (max(a.get('xGA_ewm', 0.3) or 0.3, 0.3) + 0.01) -
                            a.get('xGF_ewm', np.nan) / (max(h.get('xGA_ewm', 0.3) or 0.3, 0.3) + 0.01)) if XG_AVAILABLE else np.nan,
    }

    # Draw-focused features (STEP 1 v9)
    diff_elo    = row['Diff_Elo']
    row['Abs_Elo_diff'] = abs(diff_elo)

    if XG_AVAILABLE:
        _xgf_h = h.get('xGF_ewm', np.nan)
        _xgf_a = a.get('xGF_ewm', np.nan)
        if not (pd.isna(_xgf_h) or pd.isna(_xgf_a)):
            _abs_xgf = abs(_xgf_h - _xgf_a)
            _xg_sum  = _xgf_h + _xgf_a
        else:
            _abs_xgf = abs(h['GF_ewm'] - a['GF_ewm'])
            _xg_sum  = None
    else:
        _abs_xgf = abs(h['GF_ewm'] - a['GF_ewm'])
        _xg_sum  = None

    row['Abs_xGF_diff']   = _abs_xgf
    row['GF_balance']     = 1 / (_abs_xgf + 0.3)
    row['GA_balance']     = 1 / (abs(h['GA_ewm'] - a['GA_ewm']) + 0.3)
    row['xG_tightness']   = (1 / (_abs_xgf + 0.3) / (max(_xg_sum, 0.5) + 0.5)
                              if _xg_sum is not None else row['GF_balance'])
    row['Draw_EloXForm']  = row['Elo_closeness'] * row['Form_closeness']
    row['Late_season_draw'] = int(season_phase == 3) * min(home_draw_rate + 0.1, 1.0)
    row['Combined_GF_ewm']= h['GF_ewm'] + a['GF_ewm']

    # ── Market features — ใช้ Elo-implied prob แทน odds จริง (ไม่มี odds อนาคต) ──
    h_elo_prob = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
    a_elo_prob = 1 / (1 + 10 ** ((h_elo - a_elo) / 400))
    d_elo_prob = max(0.0, 1.0 - h_elo_prob - (a_elo_prob * 0.85))
    _total = h_elo_prob + d_elo_prob + (a_elo_prob * 0.85)
    if _total > 0:
        mkt_h = h_elo_prob / _total
        mkt_d = d_elo_prob / _total
        mkt_a = (a_elo_prob * 0.85) / _total
    else:
        mkt_h, mkt_d, mkt_a = 0.45, 0.25, 0.30

    row['Mkt_ImpH']     = mkt_h
    row['Mkt_ImpD']     = mkt_d
    row['Mkt_ImpA']     = mkt_a
    row['Mkt_Spread']   = mkt_h - mkt_a
    row['Mkt_DrawPrem'] = mkt_d - (mkt_h * 0.5 + mkt_a * 0.5)
    row['Mkt_Overround']= 0.05

    return row


# ══════════════════════════════════════════════════════════════
# PREDICT SINGLE MATCH
# ══════════════════════════════════════════════════════════════

def predict_match(home_team, away_team, ctx, match_date=None):
    """ทำนายผลแมตช์ — คืน dict พร้อม print ผล"""
    match_df_clean      = ctx['match_df_clean']
    FEATURES            = ctx['FEATURES']
    scaler              = ctx['scaler']
    stage1_cal          = ctx['stage1_cal']
    stage2_cal          = ctx['stage2_cal']
    final_elo           = ctx['final_elo']
    OPT_T_HOME          = ctx['OPT_T_HOME']
    OPT_T_DRAW          = ctx['OPT_T_DRAW']
    DRAW_SUPPRESS_FACTOR= ctx['DRAW_SUPPRESS_FACTOR']
    POISSON_HYBRID_READY= ctx['POISSON_HYBRID_READY']
    best_alpha          = ctx['best_alpha']
    home_poisson_model  = ctx['home_poisson_model']
    away_poisson_model  = ctx['away_poisson_model']
    poisson_scaler      = ctx['poisson_scaler']
    poisson_features_used = ctx['poisson_features_used']

    from src.config import NEW_TEAMS_BOOTSTRAPPED
    teams_in_data = set(match_df_clean['HomeTeam'].tolist() + match_df_clean['AwayTeam'].tolist())
    all_known = teams_in_data | set(NEW_TEAMS_BOOTSTRAPPED.keys())
    if home_team not in all_known:
        print(f"❌ ไม่พบทีม '{home_team}' ในข้อมูลและ bootstrap list"); return None
    if away_team not in all_known:
        print(f"❌ ไม่พบทีม '{away_team}' ในข้อมูลและ bootstrap list"); return None
    if home_team in NEW_TEAMS_BOOTSTRAPPED:
        print(f"  ℹ️  {home_team}: ใช้ bootstrap stats (ทีมใหม่/เลื่อนชั้น)")
    if away_team in NEW_TEAMS_BOOTSTRAPPED:
        print(f"  ℹ️  {away_team}: ใช้ bootstrap stats (ทีมใหม่/เลื่อนชั้น)")

    row = build_match_row(home_team, away_team, ctx, match_date)
    # Keep inference robust when training features evolve (e.g., market/xG additions).
    X = pd.DataFrame([row]).reindex(columns=FEATURES, fill_value=0).fillna(0)
    X_sc = scaler.transform(X)

    # 2-Stage ML prediction
    proba_ml = predict_2stage(X_sc, stage1_cal, stage2_cal)[0]

    # Poisson Hybrid blend
    if POISSON_HYBRID_READY:
        try:
            pf_row = pd.DataFrame([row]).reindex(columns=poisson_features_used, fill_value=0).fillna(0)
            pf_sc  = poisson_scaler.transform(pf_row)
            hxg    = float(np.clip(home_poisson_model.predict(pf_sc)[0], 0.3, 6.0))
            axg    = float(np.clip(away_poisson_model.predict(pf_sc)[0], 0.3, 6.0))
            ph, pd_, pa = poisson_win_draw_loss(hxg, axg)
            proba_pois  = np.array([pa, pd_, ph])
            elo_diff    = abs(row.get('Diff_Elo', 0))
            proba = blend_ml_poisson_dynamic(
                proba_ml.reshape(1,-1),
                proba_pois.reshape(1,-1),
                elo_diffs=[elo_diff],
                base_alpha=best_alpha
            )
            proba = suppress_draw_proba(proba, draw_factor=DRAW_SUPPRESS_FACTOR)[0]
            actual_alpha = float(np.clip(best_alpha - 0.2 + 0.4 * min(elo_diff/400, 1), 0.1, 0.9))
            model_tag    = f"Hybrid α={actual_alpha:.2f} 🔥"
        except Exception:
            proba     = proba_ml
            hxg = axg = None
            model_tag = "2-Stage ML"
    else:
        proba     = proba_ml
        hxg = axg = None
        model_tag = "2-Stage ML"

    pred      = apply_thresholds(proba.reshape(1, -1), OPT_T_HOME, OPT_T_DRAW)[0]
    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    h_elo     = final_elo.get(home_team, 1500)
    a_elo     = final_elo.get(away_team, 1500)

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
        bar = '█' * int(pct / 100 * bar_chars)
        print(f"  {label}: {bar:<28} {pct}%")
    print(f"{'─'*52}")
    print(f"  🎯 Prediction: {result['Prediction']}")
    print(f"{'='*52}")
    return result


# ══════════════════════════════════════════════════════════════
# PREDICT SCORE (Poisson)
# ══════════════════════════════════════════════════════════════

def predict_score(home_team, away_team, ctx):
    """ทำนาย xG และสกอร์ที่น่าจะเป็นด้วย Poisson"""
    match_df_clean    = ctx['match_df_clean']
    POISSON_MODEL_READY = ctx['POISSON_MODEL_READY']
    home_poisson_model  = ctx['home_poisson_model']
    away_poisson_model  = ctx['away_poisson_model']
    poisson_scaler      = ctx['poisson_scaler']
    poisson_features_used = ctx['poisson_features_used']
    data              = ctx['data']

    from src.config import NEW_TEAMS_BOOTSTRAPPED
    teams_in_data = set(match_df_clean['HomeTeam'].tolist() + match_df_clean['AwayTeam'].tolist())
    all_known = teams_in_data | set(NEW_TEAMS_BOOTSTRAPPED.keys())
    if home_team not in all_known or away_team not in all_known:
        return None

    if POISSON_MODEL_READY:
        row    = build_match_row(home_team, away_team, ctx)
        pf_row = pd.DataFrame([row]).reindex(columns=poisson_features_used, fill_value=0).fillna(0)
        pf_sc  = poisson_scaler.transform(pf_row)
        home_xg = float(np.clip(home_poisson_model.predict(pf_sc)[0], 0.3, 6.0))
        away_xg = float(np.clip(away_poisson_model.predict(pf_sc)[0], 0.3, 6.0))
        xg_source = "Poisson Regression 🔥"
    else:
        h = get_latest_features(home_team, True,  ctx)
        a = get_latest_features(away_team, False, ctx)
        lg = data.dropna(subset=['FTHG'])['FTHG'].mean()
        home_xg = max(0.3, min((h['GF_ewm']/lg) * (a['GA_ewm']/lg) * lg, 6.0))
        away_xg = max(0.3, min((a['GF_ewm']/lg) * (h['GA_ewm']/lg) * lg, 6.0))
        xg_source = "Ratio Method"

    score_probs = {}
    for hg in range(8):
        for ag in range(8):
            from scipy.stats import poisson as scipy_poisson
            score_probs[f"{hg}-{ag}"] = round(
                scipy_poisson.pmf(hg, home_xg) * scipy_poisson.pmf(ag, away_xg) * 100, 2
            )
    top5 = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
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
# LAST 5 RESULTS (Live API Data Version)
# ══════════════════════════════════════════════════════════════

def get_last_5_results(team, ctx):
    import pandas as pd
    import os, glob as _glob
    from src.config import DATA_DIR

    # ── โหลดทุก CSV ใน DATA_DIR (เหมือน load_data) เพื่อให้ได้ข้อมูลล่าสุด ──
    # ไม่ hardcode ชื่อไฟล์ เพราะชื่ออาจเป็น "season 2025.csv" หรือ "season_2025.csv"
    try:
        csv_files = [f for f in _glob.glob(os.path.join(DATA_DIR, "*.csv"))
                     if 'backup' not in f.lower()]
        if csv_files:
            dfs = []
            for f in csv_files:
                try:
                    _df = pd.read_csv(f)
                    _df['FTHG'] = pd.to_numeric(_df['FTHG'], errors='coerce')
                    _df['FTAG'] = pd.to_numeric(_df['FTAG'], errors='coerce')
                    _df['Date'] = pd.to_datetime(_df['Date'], dayfirst=True, errors='coerce')
                    dfs.append(_df)
                except Exception:
                    pass
            data = pd.concat(dfs, ignore_index=True)
            data = data.drop_duplicates(subset=['Date','HomeTeam','AwayTeam'], keep='last')
            data = data.sort_values('Date').reset_index(drop=True)
            print(f"  ✅  โหลดข้อมูลสด {len(data)} แมตช์จาก {len(csv_files)} ไฟล์")
        else:
            # fallback: ใช้ ctx['data']
            data = ctx['data'].copy()
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data['FTHG'] = pd.to_numeric(data['FTHG'], errors='coerce')
            data['FTAG'] = pd.to_numeric(data['FTAG'], errors='coerce')
    except Exception as _e:
        print(f"  ⚠️  โหลด CSV ล้มเหลว: {_e} — ใช้ ctx['data'] แทน")
        data = ctx['data'].copy()
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['FTHG'] = pd.to_numeric(data['FTHG'], errors='coerce')
        data['FTAG'] = pd.to_numeric(data['FTAG'], errors='coerce')

    valid = data.dropna(subset=['FTHG', 'FTAG']).copy()

    # ดึงฟอร์มตอนเป็นทีมเหย้า
    hm = valid[valid['HomeTeam'] == team][['Date','HomeTeam','AwayTeam','FTHG','FTAG']].copy()
    hm['Venue'] = 'H'; hm['GF'] = hm['FTHG']; hm['GA'] = hm['FTAG']; hm['Opponent'] = hm['AwayTeam']
    
    # ดึงฟอร์มตอนเป็นทีมเยือน
    am = valid[valid['AwayTeam'] == team][['Date','HomeTeam','AwayTeam','FTHG','FTAG']].copy()
    am['Venue'] = 'A'; am['GF'] = am['FTAG']; am['GA'] = am['FTHG']; am['Opponent'] = am['HomeTeam']

    # นำมารวมกันแล้วเรียงวันที่จากปัจจุบันย้อนกลับไปในอดีต
    all_m = pd.concat([hm, am]).sort_values('Date', ascending=False).head(5)

    def rl(r):
        if r['GF'] > r['GA']: return 'W'
        elif r['GF'] == r['GA']: return 'D'
        else: return 'L'
    all_m['Result'] = all_m.apply(rl, axis=1)
    icon_map = {'W': '✅ ชนะ', 'D': '🟡 เสมอ', 'L': '❌ แพ้'}

    # แสดงผลใน Terminal
    print(f"\n{'='*58}\n  📋  5 แมตช์ล่าสุดของ {team}\n{'='*58}")
    print(f"  {'วันที่':<13} {'คู่แข่ง':<22} {'สนาม':<6} {'สกอร์':<10} {'ผล'}")
    print(f"  {'─'*55}")
    for _, row in all_m.iterrows():
        ds = row['Date'].strftime('%d/%m/%Y') if pd.notna(row['Date']) else 'N/A'
        sc = f"{int(row['GF'])}-{int(row['GA'])}"
        venue_th = 'เหย้า' if row['Venue'] == 'H' else 'เยือน'
        print(f"  {ds:<13} {str(row['Opponent']):<22} {venue_th:<6} {sc:<10} {icon_map[row['Result']]}")
    print(f"{'='*58}")
    
    return all_m


# ══════════════════════════════════════════════════════════════
# UPDATE SEASON CSV FROM API
# ══════════════════════════════════════════════════════════════

def update_season_csv_from_api():
    url     = "https://api.football-data.org/v4/competitions/PL/matches"
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
            rows.append({
                "Date":     date_str,
                "HomeTeam": normalize_team_name(m["homeTeam"]["name"]),
                "AwayTeam": normalize_team_name(m["awayTeam"]["name"]),
                "FTHG": hg, "FTAG": ag, "FTR": ftr,
            })

        df_new   = pd.DataFrame(rows)
        played   = len(df_new[df_new["FTHG"] != ""])
        upcoming = len(df_new[df_new["FTHG"] == ""])
        df_new.to_csv(f"{DATA_DIR}/season 2025.csv", index=False)
        print(f"  ✅ แข่งแล้ว {played} นัด | ยังไม่แข่ง {upcoming} นัด")
        print(f"  💾 บันทึก → {DATA_DIR}/season 2025.csv")
        print("="*55)
        return df_new
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# RUN SEASON SIMULATION
# ══════════════════════════════════════════════════════════════

def run_season_simulation(ctx):
    """จำลองฤดูกาลที่เหลือ — คืน ctx ที่มี final_table และ remaining_fixtures"""
    FEATURES = ctx['FEATURES']
    scaler   = ctx['scaler']
    ensemble = ctx['ensemble']

    season_file = pd.read_csv(f"{DATA_DIR}/season 2025.csv")
    season_file['Date'] = pd.to_datetime(season_file['Date'], dayfirst=True, errors='coerce')
    played = season_file.dropna(subset=['FTHG','FTAG']).copy()
    played = played[played['Date'] <= TODAY]

    season_teams = list(set(
        season_file['HomeTeam'].tolist() + season_file['AwayTeam'].tolist()
    ))
    played_pairs     = set(zip(played['HomeTeam'], played['AwayTeam']))
    remaining        = [
        {'HomeTeam': h, 'AwayTeam': a}
        for h in season_teams for a in season_teams
        if h != a and (h, a) not in played_pairs
    ]
    unplayed_df = pd.DataFrame(remaining) if remaining else pd.DataFrame()

    print(f"\n📅 วันนี้: {TODAY.date()}")
    print(f"✅ แมตช์แข่งแล้ว:    {len(played)} นัด")
    print(f"⏳ แมตช์ยังไม่แข่ง: {len(unplayed_df)} นัด")
    print(f"   รวม: {len(played) + len(unplayed_df)} นัด")

    # ตารางคะแนนจริง
    real_table = {}
    for _, row in played.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        hg, ag = int(row['FTHG']), int(row['FTAG'])
        for t in [home, away]:
            if t not in real_table: real_table[t] = 0
        if hg > ag:   real_table[home] += 3
        elif hg < ag: real_table[away] += 3
        else:         real_table[home] += 1; real_table[away] += 1

    real_table_df = pd.DataFrame.from_dict(real_table, orient='index', columns=['RealPoints'])

    # ทำนายแมตช์ที่เหลือ
    pred_table = {}
    print(f"🤖 ทำนาย {len(remaining)} แมตช์ที่เหลือ")

    if remaining:
        future_rows = []
        for match in remaining:
            row = build_match_row(match['HomeTeam'], match['AwayTeam'], ctx)
            row['HomeTeam'] = match['HomeTeam']
            row['AwayTeam'] = match['AwayTeam']
            future_rows.append(row)
        future_df = pd.DataFrame(future_rows)
        X_future_df = future_df.reindex(columns=FEATURES, fill_value=0).fillna(0)
        X_future  = scaler.transform(X_future_df)
        future_df['Pred'] = ensemble.predict(X_future)

        for _, row in future_df.iterrows():
            home, away, pred = row['HomeTeam'], row['AwayTeam'], row['Pred']
            for t in [home, away]:
                if t not in pred_table: pred_table[t] = 0
            if pred == 2:   pred_table[home] += 3
            elif pred == 1: pred_table[home] += 1; pred_table[away] += 1
            else:           pred_table[away] += 3

    pred_table_df = pd.DataFrame.from_dict(pred_table, orient='index', columns=['PredictedPoints'])
    final_table   = real_table_df.join(pred_table_df, how='left').fillna(0)
    final_table['PredictedPoints'] = final_table['PredictedPoints'].astype(int)
    final_table['FinalPoints']     = final_table['RealPoints'] + final_table['PredictedPoints']
    final_table.index.name = 'Team'

    # ── แสดงตารางคะแนนจริง
    real_sorted = final_table.sort_values('RealPoints', ascending=False)
    played_count = len(played) // max(len(season_teams), 1)
    print(f"\n{'='*58}\n  📊  ตารางคะแนนจริง ณ ปัจจุบัน\n{'='*58}")
    for rank, (team, row) in enumerate(real_sorted.iterrows(), 1):
        status = "🔴 CL" if rank<=4 else ("🟠 Euro" if rank<=6 else ("🟡 Rel" if rank>=18 else ""))
        print(f"  {rank:<4} {team:<22} {int(row['RealPoints']):>5}  {status}")

    # ── แสดงตารางคาดการณ์
    final_sorted = final_table.sort_values('FinalPoints', ascending=False)
    print(f"\n{'='*62}\n  🔮  ตารางคาดการณ์สิ้นฤดูกาล\n{'='*62}")
    for rank, (team, row) in enumerate(final_sorted.iterrows(), 1):
        status = "🔴 CL" if rank<=4 else ("🟠 Euro" if rank<=6 else ("🟡 Rel" if rank>=18 else ""))
        try:
            real_rank = list(real_sorted.index).index(team) + 1
        except ValueError:
            real_rank = rank
        arrow = "▲" if rank < real_rank else ("▼" if rank > real_rank else "─")
        print(f"  {rank:<4} {team:<22} {int(row['RealPoints']):>9} "
              f"{int(row['PredictedPoints']):>10} {int(row['FinalPoints']):>8}  {arrow} {status}")

    # อัปเดต ctx
    ctx['final_table']        = final_table
    ctx['remaining_fixtures'] = remaining
    return ctx


# ══════════════════════════════════════════════════════════════
# FETCH FIXTURES FROM API
# ══════════════════════════════════════════════════════════════

def fetch_fixtures_from_api(target_team, num_matches=5):
    url     = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}
    try:
        print(f"  🌐 ดึงข้อมูลจาก football-data.org API...")
        r = requests.get(url, headers=headers, params={"status": "SCHEDULED"}, timeout=10)
        if r.status_code == 401: print("  ❌ API Key ไม่ถูกต้อง"); return None
        if r.status_code == 429: print("  ❌ Rate limit"); return None
        r.raise_for_status()

        matches = r.json().get("matches", [])
        all_fixtures = []
        for m in matches:
            utc_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            th_dt  = utc_dt + timedelta(hours=7)
            all_fixtures.append({
                "HomeTeam": normalize_team_name(m["homeTeam"]["name"]),
                "AwayTeam": normalize_team_name(m["awayTeam"]["name"]),
                "Date":     th_dt.strftime("%Y-%m-%d"),
                "DateObj":  th_dt,
            })

        team_fixtures = [
            f for f in all_fixtures
            if f["HomeTeam"] == target_team or f["AwayTeam"] == target_team
        ][:num_matches]

        if not team_fixtures:
            print(f"  ❌ ไม่พบนัดของ '{target_team}'"); return None
        return team_fixtures
    except requests.exceptions.ConnectionError:
        print("  ❌ เชื่อมต่ออินเทอร์เน็ตไม่ได้"); return None
    except Exception as e:
        print(f"  ❌ Error: {e}"); return None


# ══════════════════════════════════════════════════════════════
# PREDICT NEXT 5 MATCHES FOR A TEAM
# ══════════════════════════════════════════════════════════════

def predict_next_5_matches(team, ctx, fixtures=None):
    remaining_fixtures = ctx.get('remaining_fixtures', [])

    print(f"\n{'='*62}\n  🔮  ทำนาย 5 แมตช์ข้างหน้า: {team}\n{'='*62}")
    get_last_5_results(team, ctx)

    if fixtures:
        next5 = [f for f in fixtures if f['HomeTeam'] == team or f['AwayTeam'] == team][:5]
    else:
        next5 = [f for f in remaining_fixtures if f['HomeTeam'] == team or f['AwayTeam'] == team][:5]

    if not next5:
        print(f"\n⚠️  ไม่พบแมตช์ข้างหน้าของ {team}"); return None

    predictions = []
    for i, match in enumerate(next5, 1):
        hm, aw   = match['HomeTeam'], match['AwayTeam']
        is_home  = (hm == team)
        opponent = aw if is_home else hm
        venue_th = 'เหย้า' if is_home else 'เยือน'
        match_date = pd.Timestamp(match.get('DateObj', TODAY))

        print(f"\n  นัดที่ {i}  |  {hm}  vs  {aw}  ({venue_th})")
        print(f"  {'─'*58}")
        r_pred = predict_match(hm, aw, ctx, match_date)
        s_pred = predict_score(hm, aw, ctx)

        if r_pred and s_pred:
            if is_home:
                win_pct, draw_pct, loss_pct = r_pred['Home Win'], r_pred['Draw'], r_pred['Away Win']
                outcome = r_pred['Prediction']
            else:
                win_pct, draw_pct, loss_pct = r_pred['Away Win'], r_pred['Draw'], r_pred['Home Win']
                flip = {'Home Win': 'Away Win', 'Away Win': 'Home Win', 'Draw': 'Draw'}
                outcome = flip.get(r_pred['Prediction'], r_pred['Prediction'])

            is_win  = (is_home and outcome=='Home Win') or (not is_home and outcome=='Away Win')
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

    # Summary table
    print(f"\n{'='*68}\n  📋  สรุป 5 แมตช์ข้างหน้า: {team}\n{'='*68}")
    print(f"  {'นัด':<5} {'คู่แข่ง':<24} {'สนาม':<7} {'ชนะ%':<8} {'เสมอ%':<8} {'แพ้%':<8} {'สกอร์คาด'}")
    print(f"  {'─'*68}")
    for p in predictions:
        print(f"  {p['match_no']:<5} {p['opponent']:<24} {p['venue']:<7} "
              f"{p['win_pct']:<8} {p['draw_pct']:<8} {p['loss_pct']:<8} {p['predicted_score']}")
    return predictions


# ══════════════════════════════════════════════════════════════
# PREDICT WITH API (main entry point สำหรับทีม)
# ══════════════════════════════════════════════════════════════

def predict_with_api(team, ctx, num_matches=5):
    SEP = '=' * 62
    print(f"\n{SEP}\n  🔮  ทำนาย {num_matches} แมตช์ข้างหน้า: {team}\n{SEP}")
    fixtures = fetch_fixtures_from_api(team, num_matches)
    if fixtures:
        predict_next_5_matches(team, ctx, fixtures=fixtures)
    else:
        print('  ⚠️  fallback: ใช้การเดาอัตโนมัติ')
        predict_next_5_matches(team, ctx)


# ══════════════════════════════════════════════════════════════
# GET PL STANDINGS FROM API (ตารางคะแนนจริงจาก football-data.org)
# ══════════════════════════════════════════════════════════════

def get_pl_standings_from_api(season: int = None):
    """
    ดึงตารางคะแนน Premier League จาก football-data.org API
    season: ปี เช่น 2024 = ฤดูกาล 2024/25, None = ฤดูกาลปัจจุบัน
    คืน list of dict หรือ None ถ้าดึงไม่ได้
    """
    url = "https://api.football-data.org/v4/competitions/PL/standings"
    headers = {"X-Auth-Token": API_KEY}
    params = {}
    if season is not None:
        params["season"] = season

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        standings_raw = data.get("standings", [])
        # standings มี 3 type: TOTAL, HOME, AWAY — เอา TOTAL
        total = next((s for s in standings_raw if s.get("type") == "TOTAL"), None)
        if not total:
            return None

        rows = []
        for entry in total.get("table", []):
            team_name = entry.get("team", {}).get("name", "Unknown")
            rows.append({
                "pos":       entry.get("position", 0),
                "Club":      normalize_team_name(team_name),
                "MP":        entry.get("playedGames", 0),
                "W":         entry.get("won", 0),
                "D":         entry.get("draw", 0),
                "L":         entry.get("lost", 0),
                "GF":        entry.get("goalsFor", 0),
                "GA":        entry.get("goalsAgainst", 0),
                "GD":        entry.get("goalDifference", 0),
                "PTS":       entry.get("points", 0),
                "Form":      entry.get("form", "") or "",
            })

        rows.sort(key=lambda x: x["pos"])
        return rows

    except requests.exceptions.ConnectionError:
        print("  ❌ get_pl_standings: เชื่อมต่อ API ไม่ได้")
        return None
    except Exception as e:
        print(f"  ❌ get_pl_standings error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# SHOW NEXT PL FIXTURES (ตารางแข่งและทำนาย 5 นัดถัดไป)
# ══════════════════════════════════════════════════════════════

def show_next_pl_fixtures(ctx, num_matches=5):
    match_df_clean = ctx['match_df_clean']
    SEP = "=" * 65
    url     = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": API_KEY}
    try:
        from datetime import timezone
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_utc   = datetime.now(timezone.utc)
        r = requests.get(url, headers=headers, params={
            "status": "SCHEDULED",
            "dateFrom": today_utc,
        }, timeout=10)
        r.raise_for_status()
        matches = r.json().get("matches", [])
        # กรองเฉพาะที่ utcDate >= ตอนนี้จริงๆ ป้องกัน edge case
        matches = [
            m for m in matches
            if datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")) >= now_utc
        ]
        matches = sorted(matches, key=lambda x: x["utcDate"])[:num_matches]
        if not matches:
            print("  ⚠️  ไม่พบแมตช์"); return

        print(f"\n{SEP}\n  📅  ตารางแข่ง Premier League {num_matches} นัดถัดไป\n{SEP}")
        print(f"  {'นัด':<5} {'วันที่':<14} {'เวลา(TH)':<11} {'เหย้า':<22} {'เยือน'}")
        print("  " + "-"*60)

        upcoming = []
        for i, m in enumerate(matches, 1):
            home   = normalize_team_name(m["homeTeam"]["name"])
            away   = normalize_team_name(m["awayTeam"]["name"])
            home_id = m["homeTeam"].get("id")
            away_id = m["awayTeam"].get("id")
            home_logo = m["homeTeam"].get("crest") or (f"https://crests.football-data.org/{home_id}.png" if home_id else None)
            away_logo = m["awayTeam"].get("crest") or (f"https://crests.football-data.org/{away_id}.png" if away_id else None)
            utc_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            th_dt  = utc_dt + timedelta(hours=7)
            ds, ts = th_dt.strftime("%d/%m/%Y"), th_dt.strftime("%H:%M")
            print(f"  {i:<5} {ds:<14} {ts:<11} {home:<22} {away}")
            upcoming.append({
                "HomeTeam": home,
                "AwayTeam": away,
                "Date": ds,
                "Time": ts,
                "HomeID": home_id,
                "AwayID": away_id,
                "HomeLogo": home_logo,
                "AwayLogo": away_logo,
            })

        from src.config import NEW_TEAMS_BOOTSTRAPPED
        teams_ok = set(match_df_clean["HomeTeam"].tolist() + match_df_clean["AwayTeam"].tolist()) | set(NEW_TEAMS_BOOTSTRAPPED.keys())
        print(f"\n{SEP}\n  🤖  ผลทำนาย {num_matches} นัดถัดไป\n{SEP}")
        print(f"  {'นัด':<5} {'เหย้า':<20} {'vs':^4} {'เยือน':<20} "
              f"{'ชนะ%':>7} {'เสมอ%':>7} {'แพ้%':>7}  {'สกอร์'}")
        print("  " + "-"*75)

        for i, f in enumerate(upcoming, 1):
            home, away = f["HomeTeam"], f["AwayTeam"]
            if home not in teams_ok or away not in teams_ok:
                print(f"  {i:<5} {home:<20} {'vs':^4} {away:<20}  ⚠️ ไม่มีข้อมูล")
                continue
            r_pred = predict_match(home, away, ctx)
            s_pred = predict_score(home, away, ctx)
            if r_pred and s_pred:
                hw   = r_pred["Home Win"]; dr = r_pred["Draw"]; aw = r_pred["Away Win"]
                pred = r_pred["Prediction"]
                icon = "🏠" if pred == "Home Win" else ("🤝" if pred == "Draw" else "✈️")
                print(f"  {i:<5} {home:<20} {'vs':^4} {away:<20} "
                      f"{hw:>7} {dr:>7} {aw:>7}  {icon} {s_pred['most_likely_score']}")

        print("  " + "-"*75)
        print("  🏠 เหย้าชนะ  🤝 เสมอ  ✈️ เยือนชนะ")
        print(SEP)
        return upcoming
    except requests.exceptions.ConnectionError:
        print("  ❌ เชื่อมต่ออินเทอร์เน็ตไม่ได้")
    except Exception as e:
        print(f"  ❌ Error: {e}")


# ══════════════════════════════════════════════════════════════
# PRINT FULL SUMMARY REPORT
# ══════════════════════════════════════════════════════════════

def print_full_summary(ctx):
    data           = ctx['data']
    FEATURES       = ctx['FEATURES']
    train          = ctx['train']
    test           = ctx['test']
    final_elo      = ctx['final_elo']
    final_table    = ctx.get('final_table')
    y_test         = ctx['y_test']
    y_pred_final   = ctx['y_pred_final']
    proba_hybrid   = ctx['proba_hybrid']
    OPT_T_HOME     = ctx['OPT_T_HOME']
    OPT_T_DRAW     = ctx['OPT_T_DRAW']

    SEP  = "=" * 65
    LINE = "─" * 65
    print("\n" + "█"*65)
    print("  📊  FOOTBALL AI v9.0 — FULL SUMMARY REPORT")
    print(f"  🗓️  วันที่: {TODAY.date()}  |  ข้อมูลถึง: {data['Date'].max().date()}")
    print("█"*65)

    # 1. Data
    print(f"\n{SEP}\n  📁  1. ข้อมูล\n{SEP}")
    print(f"  • แมตช์ทั้งหมด : {len(data):,} นัด")
    print(f"  • ช่วงเวลา     : {data['Date'].min().date()} → {data['Date'].max().date()}")
    print(f"  • Train / Test : {len(train):,} / {len(test):,} นัด")
    print(f"  • Features     : {len(FEATURES)} ตัว")

    # 2. Model performance
    print(f"\n{SEP}\n  🤖  2. ผลโมเดล\n{SEP}")
    acc = round(accuracy_score(y_test, y_pred_final) * 100, 2)
    print(f"  • Accuracy (Test) : {acc}%")
    print(f"  • Thresholds      : t_home={OPT_T_HOME:.3f}  t_draw={OPT_T_DRAW:.3f}")
    cm = confusion_matrix(y_test, y_pred_final)
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_final,
                                 target_names=['Away Win','Draw','Home Win']))

    # 3. Elo Top 10
    print(f"\n{SEP}\n  🏆  3. Elo Rating ล่าสุด (Top 10)\n{SEP}")
    elo_sorted = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (team, elo_val) in enumerate(elo_sorted, 1):
        bar = '█' * int(elo_val / elo_sorted[0][1] * 20)
        print(f"  {rank:<4} {team:<25} {round(elo_val):>8}  {bar}")

    # 4. Season table
    if final_table is not None:
        print(f"\n{SEP}\n  📋  4. ตารางคาดการณ์สิ้นฤดูกาล\n{SEP}")
        for rank, (team, row) in enumerate(
            final_table.sort_values('FinalPoints', ascending=False).iterrows(), 1
        ):
            status = ("🔴 CL" if rank<=4 else
                      "🟠 Euro" if rank<=6 else
                      "🟡 ตกชั้น" if rank>=18 else "")
            print(f"  {rank:<5} {team:<22} แต้มจริง:{int(row['RealPoints']):>5} "
                  f"AI:{int(row['PredictedPoints']):>5} รวม:{int(row['FinalPoints']):>5}  {status}")

    # 5. สถิติ
    print(f"\n{SEP}\n  📈  5. สถิติ\n{SEP}")
    valid = data.dropna(subset=['FTHG','FTAG'])
    hw = (valid['FTHG'] > valid['FTAG']).sum()
    d  = (valid['FTHG'] == valid['FTAG']).sum()
    aw = (valid['FTHG'] < valid['FTAG']).sum()
    t  = len(valid)
    print(f"  • เหย้าชนะ : {hw:,} ({hw/t*100:.1f}%)  "
          f"เสมอ : {d:,} ({d/t*100:.1f}%)  "
          f"เยือนชนะ : {aw:,} ({aw/t*100:.1f}%)")
    print(f"  • เฉลี่ยประตู/นัด : {(valid['FTHG']+valid['FTAG']).mean():.2f}")

    print("\n" + "█"*65 + "\n  ✅  END OF REPORT\n" + "█"*65)