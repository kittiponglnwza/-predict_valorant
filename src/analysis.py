"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI v9.0 — ANALYSIS                               ║
║   Monte Carlo, Backtest ROI, Rolling CV, Walk-Forward, SHAP  ║
╚══════════════════════════════════════════════════════════════╝
"""

from src.config import *
from src.model import (predict_2stage, apply_thresholds, suppress_draw_proba,
                   blend_ml_poisson_dynamic, blend_ml_poisson,
                   optimize_blend_alpha, optimize_thresholds,
                   build_2stage_model, get_cv_lgbm_params,
                   apply_smote, build_poisson_proba_for_test,
                   poisson_win_draw_loss, TwoStageEnsemble)
from src.predict import build_match_row


# ══════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════

def run_monte_carlo(ctx, n_simulations=1000, verbose=True):
    final_table        = ctx.get('final_table')
    remaining_fixtures = ctx.get('remaining_fixtures', [])
    scaler             = ctx['scaler']
    ensemble           = ctx['ensemble']
    FEATURES           = ctx['FEATURES']

    if final_table is None:
        print("❌ กรุณาเรียก run_season_simulation() ก่อน"); return None
    if not remaining_fixtures:
        print("  ℹ️  ฤดูกาลจบแล้ว"); return None

    SEP = "=" * 65
    if verbose:
        print(f"\n{SEP}\n  🎲  MONTE CARLO SEASON SIMULATION  ({n_simulations:,} รอบ)\n{SEP}")
        print(f"  กำลังจำลอง {len(remaining_fixtures)} แมตช์ × {n_simulations:,} รอบ ...")

    future_rows = []
    for match in remaining_fixtures:
        row = build_match_row(match['HomeTeam'], match['AwayTeam'], ctx)
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    # ── Guard: reindex ให้ตรง FEATURES เสมอ เติม 0 ถ้าขาด ──
    future_df = future_df.reindex(columns=FEATURES, fill_value=0).fillna(0)

    X_future_sc  = scaler.transform(future_df)
    proba_matrix = ensemble.predict_proba(X_future_sc)

    all_teams = list(final_table.index)
    real_pts  = {t: int(final_table.loc[t, 'RealPoints']) for t in all_teams}
    counts    = {t: {'top4': 0, 'top6': 0, 'relegation': 0, 'pts_sum': 0.0} for t in all_teams}
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

    results = {}
    for team in all_teams:
        c = counts[team]
        results[team] = {
            'top4_pct':       round(c['top4'] / n_simulations * 100, 1),
            'top6_pct':       round(c['top6'] / n_simulations * 100, 1),
            'relegation_pct': round(c['relegation'] / n_simulations * 100, 1),
            'expected_pts':   round(c['pts_sum'] / n_simulations, 1),
        }

    if verbose:
        sorted_res = sorted(results.items(), key=lambda x: x[1]['expected_pts'], reverse=True)
        print(f"\n  {'ทีม':<22} {'คาดคะแน':>9} {'Top4%':>7} {'Top6%':>7} {'ตกชั้น%':>9}")
        print(f"  {'─'*60}")
        for team, r in sorted_res:
            flag = "🔴" if r['top4_pct'] > 50 else ("🟡" if r['relegation_pct'] > 30 else "")
            print(f"  {team:<22} {r['expected_pts']:>9} {r['top4_pct']:>7} "
                  f"{r['top6_pct']:>7} {r['relegation_pct']:>9} {flag}")
        print(f"  {'─'*60}")
        print(f"  🔴 > 50% CL  |  🟡 > 30% ตกชั้น  (n={n_simulations:,})")
        print(SEP)

    return results


# ══════════════════════════════════════════════════════════════
# BACKTEST ROI (Kelly Criterion)
# ══════════════════════════════════════════════════════════════

def backtest_roi(ctx, bankroll=1000.0, min_edge=0.03, kelly_fraction=0.15,
                 max_exposure=0.05, verbose=True):
    proba_test = ctx['proba_hybrid']
    y_test     = ctx['y_test']
    test       = ctx['test']
    SEP = "=" * 65

    if verbose:
        print(f"\n{SEP}\n  💰  KELLY CRITERION BACKTEST (v3.0)\n"
              f"  Bankroll: £{bankroll:,.0f} | Min Edge: {min_edge*100:.0f}% | "
              f"Kelly: {kelly_fraction*100:.0f}% | Max: {max_exposure*100:.0f}%/bet\n{SEP}")

    label_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    bk = bankroll; bets = []; total_staked = 0.0; peak_bk = bk; max_dd = 0.0

    for i, (proba, actual) in enumerate(zip(proba_test, y_test)):
        p_away, p_draw, p_home = proba
        margin = 1.05
        odds = {
            0: (1/p_away) * margin if p_away > 0.01 else 99,
            1: (1/p_draw) * margin if p_draw > 0.01 else 99,
            2: (1/p_home) * margin if p_home > 0.01 else 99,
        }
        model_p = {0: p_away, 1: p_draw, 2: p_home}
        best_cls = max([0,1,2], key=lambda c: model_p[c] - (1/odds[c]))
        edge     = model_p[best_cls] - (1/odds[best_cls])

        if edge < min_edge: continue

        p = model_p[best_cls]; o = odds[best_cls]
        kelly_full = (p * o - 1) / (o - 1)
        stake = min(bk * kelly_full * kelly_fraction, bk * max_exposure)
        stake = max(stake, 0.5)

        won   = (best_cls == actual)
        profit = stake * (o-1) if won else -stake
        bk   += profit; total_staked += stake
        if bk > peak_bk: peak_bk = bk
        dd = (peak_bk - bk) / peak_bk * 100
        if dd > max_dd: max_dd = dd
        bets.append({'cls': best_cls, 'edge': edge, 'stake': stake,
                     'odds': o, 'won': won, 'profit': profit, 'bk': bk})

    if not bets:
        print("  ⚠️  ไม่มีการแทงที่ตรงเงื่อนไข"); return None

    total_bets = len(bets); total_won = sum(1 for b in bets if b['won'])
    roi      = (bk - bankroll) / total_staked * 100
    win_rate = total_won / total_bets * 100

    if verbose:
        print(f"\n  จำนวนแมตช์ที่ผ่าน threshold : {total_bets:,}")
        print(f"  Win Rate                    : {win_rate:.1f}%")
        print(f"  Net P&L                     : £{bk - bankroll:>+,.2f}")
        print(f"  ROI (per unit staked)        : {roi:.1f}%")
        print(f"  Max Drawdown                 : {max_dd:.1f}%")
        for cls in [2, 0, 1]:
            cls_bets = [b for b in bets if b['cls'] == cls]
            if not cls_bets: continue
            cls_roi = sum(b['profit'] for b in cls_bets) / sum(b['stake'] for b in cls_bets) * 100
            print(f"  {label_map[cls]:<12} {len(cls_bets):>4} bets  "
                  f"Win:{sum(1 for b in cls_bets if b['won'])/len(cls_bets)*100:.0f}%  "
                  f"ROI:{cls_roi:+.1f}%")
        verdict = "✅ ROI > 5% — edge จริง" if roi>5 else ("🟡 edge เล็กน้อย" if roi>0 else "❌ ยังไม่ beat market")
        print(f"\n  💡 {verdict}")
        print(SEP)

    return {'roi': roi, 'win_rate': win_rate, 'net_pnl': bk-bankroll,
            'total_bets': total_bets, 'max_dd': max_dd, 'final_bk': bk}


# ══════════════════════════════════════════════════════════════
# ROLLING WINDOW CV
# ══════════════════════════════════════════════════════════════

def rolling_window_cv(ctx, n_splits=5, verbose=True):
    match_df_clean = ctx['match_df_clean']
    FEATURES       = ctx['FEATURES']
    best_lgbm_params = ctx.get('best_lgbm_params', {})
    SEP = "=" * 65

    if verbose:
        print(f"\n{SEP}\n  🔄  ROLLING WINDOW CV ({n_splits} folds)\n{SEP}")

    cv_df = match_df_clean.sort_values('Date_x').reset_index(drop=True)
    tscv  = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(cv_df), 1):
        tr = cv_df.iloc[tr_idx]; te = cv_df.iloc[te_idx]
        X_tr = tr[FEATURES].fillna(0).values; y_tr = tr['Result3'].values
        X_te = te[FEATURES].fillna(0).values; y_te = te['Result3'].values

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)

        X_tr_sm, y_tr_sm = apply_smote(X_tr_sc, y_tr)
        s1, s2 = build_2stage_model(X_tr_sm, y_tr_sm, best_lgbm_params)
        proba = predict_2stage(X_te_sc, s1, s2)
        proba = suppress_draw_proba(proba)
        t_home, t_draw, _ = optimize_thresholds(proba, y_te, n_steps=30,
                                                 t_draw_range=(0.18, 0.40),
                                                 t_home_range=(0.30, 0.55))
        preds = apply_thresholds(proba, t_home, t_draw)
        acc     = accuracy_score(y_te, preds)
        draw_f1 = f1_score(y_te, preds, labels=[1], average='macro', zero_division=0)

        if verbose:
            print(f"  Fold {fold}: acc={acc:.4f}  draw_f1={draw_f1:.4f}  "
                  f"(train={len(tr)}, test={len(te)})")
        results.append({'fold': fold, 'acc': acc, 'draw_f1': draw_f1,
                        'test_size': len(te)})

    if verbose:
        accs = [r['acc'] for r in results]
        print(f"\n  Mean Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Mean Draw F1  : {np.mean([r['draw_f1'] for r in results]):.4f}")
        print(SEP)

    return results


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD CV (season-by-season)
# ══════════════════════════════════════════════════════════════

def walk_forward_season_cv(ctx, verbose=True):
    from src.features import get_season
    match_df_clean   = ctx['match_df_clean']
    FEATURES         = ctx['FEATURES']
    best_lgbm_params = ctx.get('best_lgbm_params', {})
    SEP = "=" * 65

    df = match_df_clean.copy()
    df['Season'] = df['Date_x'].apply(get_season)
    seasons = sorted(df['Season'].dropna().unique())

    if verbose:
        print(f"\n{SEP}\n  🏆  WALK-FORWARD SEASON CV\n{SEP}")

    results = []
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season   = seasons[i]
        tr = df[df['Season'].isin(train_seasons)]
        te = df[df['Season'] == test_season]
        if len(te) < 50: continue

        X_tr = tr[FEATURES].fillna(0).values; y_tr = tr['Result3'].values
        X_te = te[FEATURES].fillna(0).values; y_te = te['Result3'].values

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
        X_tr_sm, y_tr_sm = apply_smote(X_tr_sc, y_tr)
        s1, s2 = build_2stage_model(X_tr_sm, y_tr_sm, best_lgbm_params)
        proba  = predict_2stage(X_te_sc, s1, s2)
        proba  = suppress_draw_proba(proba)
        t_home, t_draw, _ = optimize_thresholds(proba, y_te, n_steps=30,
                                                 t_draw_range=(0.18, 0.40),
                                                 t_home_range=(0.30, 0.55))
        preds  = apply_thresholds(proba, t_home, t_draw)
        acc    = accuracy_score(y_te, preds)
        draw_f1= f1_score(y_te, preds, labels=[1], average='macro', zero_division=0)

        if verbose:
            print(f"  Train {sorted(train_seasons)} → Test {test_season}: "
                  f"acc={acc:.4f}  draw_f1={draw_f1:.4f}  (n={len(te)})")
        results.append({'season': test_season, 'acc': acc, 'draw_f1': draw_f1, 'test_size': len(te)})

    if verbose and results:
        wf_accs  = [r['acc'] for r in results]
        wf_total = sum(r['test_size'] for r in results)
        wf_w     = sum(r['acc']*r['test_size'] for r in results) / wf_total
        print(f"\n  Mean Accuracy : {np.mean(wf_accs):.4f} ± {np.std(wf_accs):.4f}")
        print(f"  Weighted Acc  : {wf_w:.4f}  (pooled {wf_total} matches)")
        print(SEP)

    return results


# ══════════════════════════════════════════════════════════════
# DRAW CALIBRATION
# ══════════════════════════════════════════════════════════════

def analyze_draw_calibration(ctx, verbose=True):
    proba_hybrid = ctx['proba_hybrid']
    y_test       = ctx['y_test']
    SEP = "=" * 65

    if verbose:
        print(f"\n{SEP}\n  📐  DRAW CALIBRATION ANALYSIS\n{SEP}")

    draw_proba  = proba_hybrid[:, 1]
    draw_actual = (y_test == 1).astype(int)

    bins = np.linspace(0, 1, 11)
    for i in range(len(bins)-1):
        mask = (draw_proba >= bins[i]) & (draw_proba < bins[i+1])
        if mask.sum() < 5: continue
        pred_mean = draw_proba[mask].mean()
        actual_mean = draw_actual[mask].mean()
        n = mask.sum()
        bar = '█' * int(actual_mean * 20)
        if verbose:
            print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]  "
                  f"pred={pred_mean:.3f}  actual={actual_mean:.3f}  n={n:<4}  {bar}")

    bias = (draw_proba.mean() - draw_actual.mean()) * 100
    if verbose:
        print(f"\n  Overall Draw bias: {bias:+.1f}%  "
              f"(predicted={draw_proba.mean():.1%}  actual={draw_actual.mean():.1%})")
        print(f"  {'✅ Well calibrated' if abs(bias) < 5 else '⚠️ Bias detected — consider re-tuning suppression factor'}")
        print(SEP)

    return {'bias': bias, 'predicted_rate': draw_proba.mean(), 'actual_rate': draw_actual.mean()}


# ══════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE (SHAP / LightGBM)
# ══════════════════════════════════════════════════════════════

def run_feature_importance(ctx, max_display=20):
    FEATURES   = ctx['FEATURES']
    stage1_cal = ctx['stage1_cal']
    X_test_sc  = ctx['X_test_sc']
    SEP = "=" * 65
    print(f"\n{SEP}\n  📊  FEATURE IMPORTANCE\n{SEP}")

    try:
        base_model = stage1_cal.calibrated_classifiers_[0].estimator if hasattr(stage1_cal, 'calibrated_classifiers_') else stage1_cal
        if LGBM_AVAILABLE and hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_
            indices = np.argsort(importances)[::-1][:max_display]
            print(f"\n  LightGBM Feature Importance (Stage 1 - Draw detection):")
            for rank, idx in enumerate(indices, 1):
                bar = '█' * int(importances[idx] / max(importances) * 30)
                print(f"  {rank:<4} {FEATURES[idx]:<28} {importances[idx]:>8.0f}  {bar}")
    except Exception as e:
        print(f"  ⚠️  Feature importance ไม่พร้อม: {e}")

    if SHAP_AVAILABLE:
        try:
            sample = X_test_sc[:min(200, len(X_test_sc))]
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(sample)
            if isinstance(shap_values, list):
                sv = shap_values[1]  # Draw class
            else:
                sv = shap_values
            mean_abs = np.abs(sv).mean(axis=0)
            indices_shap = np.argsort(mean_abs)[::-1][:max_display]
            print(f"\n  SHAP Feature Importance (Draw class):")
            for rank, idx in enumerate(indices_shap, 1):
                bar = '█' * int(mean_abs[idx] / max(mean_abs) * 30)
                print(f"  {rank:<4} {FEATURES[idx]:<28} {mean_abs[idx]:>8.4f}  {bar}")
        except Exception as e:
            print(f"  ⚠️  SHAP ไม่พร้อม: {e}")

    print(SEP)


# ══════════════════════════════════════════════════════════════
# PHASE 2 (Monte Carlo + Draw Calibration + Feature Importance)
# ══════════════════════════════════════════════════════════════

def run_phase2(ctx, n_simulations=1000):
    print(f"\n{'█'*65}\n  🎲  PHASE 2 — Monte Carlo + Calibration\n{'█'*65}")
    mc_results = run_monte_carlo(ctx, n_simulations=n_simulations)
    analyze_draw_calibration(ctx)
    run_feature_importance(ctx)
    return mc_results


# ══════════════════════════════════════════════════════════════
# PHASE 3 (Rolling CV + Walk-Forward + ROI)
# ══════════════════════════════════════════════════════════════

def run_phase3(ctx):
    print(f"\n{'█'*65}\n  🏆  PHASE 3 — CV + ROI Backtest\n{'█'*65}")
    cv_results = rolling_window_cv(ctx, n_splits=5)
    wf_results = walk_forward_season_cv(ctx)
    roi_result = backtest_roi(ctx, bankroll=1000.0, min_edge=0.03, kelly_fraction=0.15)

    SEP = "=" * 65
    print(f"\n{SEP}\n  📋  PHASE 3 — SUMMARY\n{SEP}")

    if wf_results:
        wf_accs   = [r['acc'] for r in wf_results]
        wf_total  = sum(r['test_size'] for r in wf_results)
        wf_w      = sum(r['acc']*r['test_size'] for r in wf_results) / wf_total
        print(f"\n  🏆 Walk-Forward CV (PRIMARY)")
        print(f"     Mean Accuracy   : {np.mean(wf_accs):.4f} ± {np.std(wf_accs):.4f}")
        print(f"     Weighted Acc    : {wf_w:.4f}  (pooled {wf_total} matches)")

    if cv_results:
        cv_accs = [r['acc'] for r in cv_results]
        print(f"\n  🔄 Rolling CV (5 folds)")
        print(f"     Mean Accuracy : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
        print(f"     Mean Draw F1  : {np.mean([r['draw_f1'] for r in cv_results]):.4f}")

    if roi_result:
        verdict = "✅ edge จริง" if roi_result['roi']>5 else ("🟡 edge เล็กน้อย" if roi_result['roi']>0 else "❌ ยังไม่ beat market")
        print(f"\n  💰 Backtest ROI: {roi_result['roi']:+.1f}%  {verdict}")
        print(f"     Win Rate: {roi_result['win_rate']:.1f}%  |  Max DD: {roi_result['max_dd']:.1f}%  |  Bets: {roi_result['total_bets']:,}")

    print(f"\n{SEP}\n  ✅  PHASE 3 COMPLETE\n{SEP}\n")
    return {'cv': cv_results, 'walk_forward': wf_results, 'roi': roi_result}

# ══════════════════════════════════════════════════════════════
# MARKET ABLATION TEST
# ══════════════════════════════════════════════════════════════

def run_market_ablation_test(ctx, verbose=True):
    """
    เปรียบเทียบ model ที่มีและไม่มี market/xG features
    เพื่อพิสูจน์ว่า edge มาจาก football intelligence จริง
    """
    train    = ctx['train']
    test     = ctx['test']
    FEATURES = ctx['FEATURES']
    best_lgbm_params = ctx.get('best_lgbm_params', {})
    OPT_T_HOME       = ctx['OPT_T_HOME']
    OPT_T_DRAW       = ctx['OPT_T_DRAW']
    DRAW_SUPPRESS_FACTOR = ctx['DRAW_SUPPRESS_FACTOR']
    SEP  = "=" * 65

    if verbose:
        print(f"\n{SEP}\n  🔬  MARKET ABLATION TEST\n{SEP}")

    if not LGBM_AVAILABLE:
        print("  ⚠️  LightGBM ไม่พร้อม — ข้าม ablation test"); return None

    MKT_FEAT = ['Mkt_ImpH','Mkt_ImpD','Mkt_ImpA',
                'Mkt_Spread','Mkt_DrawPrem','Mkt_Overround']
    XG_FEAT  = [f for f in FEATURES if any(k in f for k in
                ['xGF','xGA','xGD','xG_over','xAttack','xGF_slope'])]

    configs = {
        'Full features':       FEATURES,
        'No Market':           [f for f in FEATURES if f not in MKT_FEAT],
        'No xG':               [f for f in FEATURES if f not in XG_FEAT],
        'Base only (Elo+Form)':[f for f in FEATURES if f not in MKT_FEAT and f not in XG_FEAT],
    }

    results = {}
    if verbose:
        print(f"\n  {'Config':<28} {'Features':>10} {'Accuracy':>10} {'Draw F1':>9}")
        print(f"  {'─'*60}")

    for name, feat_list in configs.items():
        if len(feat_list) < 5: continue
        try:
            X_tr = train[feat_list].fillna(0).values
            X_te = test[feat_list].fillna(0).values
            y_tr = train['Result3'].values
            y_te = test['Result3'].values

            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)

            mdl = lgb.LGBMClassifier(**get_cv_lgbm_params(best_lgbm_params))
            mdl.fit(X_tr_sc, y_tr)

            proba = mdl.predict_proba(X_te_sc)
            proba = suppress_draw_proba(proba, draw_factor=DRAW_SUPPRESS_FACTOR)
            t_h, t_d, _ = optimize_thresholds(proba, y_te, n_steps=20,
                                               t_home_range=(max(0.15, OPT_T_HOME-0.05),
                                                             min(0.55, OPT_T_HOME+0.05)),
                                               t_draw_range=(max(0.15, OPT_T_DRAW-0.05),
                                                             min(0.40, OPT_T_DRAW+0.05)))
            preds   = apply_thresholds(proba, t_h, t_d)
            acc     = accuracy_score(y_te, preds)
            draw_f1 = f1_score(y_te, preds, labels=[1], average='macro', zero_division=0)
            results[name] = {'acc': acc, 'draw_f1': draw_f1, 'n_features': len(feat_list)}

            marker = ''
            if name != 'Full features' and 'Full features' in results:
                drop   = results['Full features']['acc'] - acc
                marker = f'  (drop {drop:+.1%})'
            if verbose:
                print(f"  {name:<28} {len(feat_list):>10} {acc:>10.1%} {draw_f1:>9.3f}{marker}")
        except Exception as e:
            if verbose: print(f"  {name:<28}  Error: {e}")

    if verbose and len(results) >= 2:
        full_acc  = results.get('Full features', {}).get('acc', 0)
        nomkt_acc = results.get('No Market', {}).get('acc', 0)
        mkt_contrib = full_acc - nomkt_acc
        print(f"\n  Market contribution: {mkt_contrib:+.1%}")
        verdict = ("✅ Football intelligence แท้" if mkt_contrib < 0.02 else
                   "🟡 Market ช่วย moderate" if mkt_contrib < 0.05 else
                   "⚠️  Edge มาจาก Market เป็นหลัก")
        print(f"  {verdict}")
        print(SEP)

    return results


# ══════════════════════════════════════════════════════════════
# REGIME DETECTION — Form Clustering
# ══════════════════════════════════════════════════════════════

def detect_team_regimes(team, ctx, n_regimes=3, verbose=True):
    """ตรวจจับ form regime ของทีมด้วย K-Means clustering"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler as _MMS
    match_df_clean = ctx['match_df_clean']
    SEP  = "=" * 65

    # ดึงข้อมูล team จาก match_df_clean (ทั้ง home และ away)
    h_rows = match_df_clean[match_df_clean['HomeTeam'] == team][
        ['Date_x','H_Pts5','H_GF5','H_GA5','H_GD5','H_Pts_ewm','H_GD_ewm']
    ].rename(columns={'Date_x':'Date','H_Pts5':'Pts5','H_GF5':'GF5','H_GA5':'GA5',
                      'H_GD5':'GD5','H_Pts_ewm':'Pts_ewm','H_GD_ewm':'GD_ewm'})
    a_rows = match_df_clean[match_df_clean['AwayTeam'] == team][
        ['Date_x','A_Pts5','A_GF5','A_GA5','A_GD5','A_Pts_ewm','A_GD_ewm']
    ].rename(columns={'Date_x':'Date','A_Pts5':'Pts5','A_GF5':'GF5','A_GA5':'GA5',
                      'A_GD5':'GD5','A_Pts_ewm':'Pts_ewm','A_GD_ewm':'GD_ewm'})
    team_rows = pd.concat([h_rows, a_rows]).sort_values('Date').dropna()

    if len(team_rows) < n_regimes * 5:
        if verbose: print(f"  ⚠️  ข้อมูลไม่พอสำหรับ {team}")
        return None

    feat_cols = ['Pts5','GF5','GA5','GD5','Pts_ewm','GD_ewm']
    X = team_rows[feat_cols].fillna(0).values
    X_norm = _MMS().fit_transform(X)

    km     = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    labels = km.fit_predict(X_norm)
    team_rows = team_rows.copy()
    team_rows['Regime'] = labels

    # Label clusters by avg points
    cluster_pts = {c: team_rows[team_rows['Regime']==c]['Pts5'].mean() for c in range(n_regimes)}
    sorted_c    = sorted(cluster_pts, key=cluster_pts.get)
    names3 = ['❄️  Cold Form','🟡 Average Form','🔥 Hot Form']
    regime_names = {sorted_c[i]: names3[i] for i in range(min(n_regimes, 3))}
    team_rows['RegimeName'] = team_rows['Regime'].map(regime_names)

    current_regime = team_rows.iloc[-1]['RegimeName']
    current_pts5   = team_rows.iloc[-1]['Pts5']

    if verbose:
        print(f"\n{SEP}\n  🧠  REGIME DETECTION: {team.upper()}\n{SEP}")
        print(f"\n  Current Form Regime : {current_regime}")
        print(f"  Points (last 5)     : {current_pts5:.2f}")
        print(f"\n  {'Regime':<22} {'Matches':>8} {'Avg Pts':>9} {'Avg GF':>8} {'Avg GA':>8}")
        print(f"  {'─'*55}")
        for c in range(n_regimes):
            s = team_rows[team_rows['Regime'] == c]
            print(f"  {regime_names.get(c,''):<22} {len(s):>8} "
                  f"{s['Pts5'].mean():>9.2f} {s['GF5'].mean():>8.2f} {s['GA5'].mean():>8.2f}")

        recent = team_rows.tail(10)
        print(f"\n  📅 Last 10 matches:")
        icon_map = {'❄️  Cold Form':'❄','🟡 Average Form':'●','🔥 Hot Form':'🔥'}
        print("  ", end="")
        for _, r in recent.iterrows():
            print(icon_map.get(r['RegimeName'],'?'), end=" ")
        print(f"\n  ❄ Cold  ● Average  🔥 Hot\n{SEP}")

    return {
        'team':           team,
        'current_regime': current_regime,
        'current_pts5':   current_pts5,
    }


# ══════════════════════════════════════════════════════════════
# ANALYZE LEAGUE REGIMES (top N teams)
# ══════════════════════════════════════════════════════════════

def analyze_league_regimes(ctx, top_n=6):
    """ดู form regime ของ top N ทีม (ตาม Elo) พร้อมกัน"""
    final_elo = ctx['final_elo']
    SEP = "=" * 65
    print(f"\n{SEP}\n  🧠  LEAGUE-WIDE REGIME SUMMARY\n{SEP}")
    top_teams = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"  {'Team':<22} {'Current Regime':<22} {'Pts/5':>7} {'Elo':>7}")
    print(f"  {'─'*60}")
    results = {}
    for team, elo in top_teams:
        r = detect_team_regimes(team, ctx, verbose=False)
        if r:
            print(f"  {team:<22} {r['current_regime']:<22} "
                  f"{r['current_pts5']:>7.2f} {elo:>7.0f}")
            results[team] = r
    print(SEP)
    return results