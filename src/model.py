"""
╔══════════════════════════════════════════════════════════════╗
║   FOOTBALL AI v9.0 — MODEL TRAINING                         ║
║   2-Stage LightGBM, Poisson, Calibration, Thresholds        ║
╚══════════════════════════════════════════════════════════════╝
"""

from src.config import *
from src.features import get_season

# ══════════════════════════════════════════════════════════════
# TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════

def split_train_test(match_df_clean, FEATURES):
    match_df_clean = match_df_clean.copy()
    match_df_clean['Season'] = match_df_clean['Date_x'].apply(get_season)

    season_counts     = match_df_clean.groupby('Season').size()
    completed_seasons = season_counts[season_counts >= 200].index.tolist()
    completed_seasons = sorted([s for s in completed_seasons if s < get_season(TODAY)])

    if len(completed_seasons) >= 2:
        TEST_SEASON  = completed_seasons[-1]
        TRAIN_SEASON = completed_seasons[:-1]
        train = match_df_clean[match_df_clean['Season'].isin(TRAIN_SEASON)]
        test  = match_df_clean[match_df_clean['Season'] == TEST_SEASON]
        print(f"\n✅ Season-Based Split:")
        print(f"   Train seasons : {sorted(TRAIN_SEASON)}  ({len(train)} matches)")
        print(f"   Test season   : {TEST_SEASON}           ({len(test)} matches)")
    else:
        sorted_df = match_df_clean.sort_values('Date_x').reset_index(drop=True)
        split_idx = int(len(sorted_df) * 0.8)
        train = sorted_df.iloc[:split_idx].copy()
        test  = sorted_df.iloc[split_idx:].copy()
        print(f"\n⚠️  Season-based split ใช้ไม่ได้ — fallback เป็น index 80/20")
        print(f"   Train matches : {len(train)}")
        print(f"   Test  matches : {len(test)}")

    X_train = train[FEATURES].fillna(0)
    y_train = train['Result3']
    X_test  = test[FEATURES].fillna(0)
    y_test  = test['Result3']

    print(f"\nTrain: {len(train)}  |  Test: {len(test)}")
    return train, test, X_train, y_train, X_test, y_test


# ══════════════════════════════════════════════════════════════
# OPTUNA TUNING
# ══════════════════════════════════════════════════════════════

def tune_lgbm_optuna(X_tr, y_tr, n_trials=40, timeout=120):
    if not OPTUNA_AVAILABLE or not LGBM_AVAILABLE:
        return {}

    from sklearn.model_selection import StratifiedKFold
    print(f"\n🔥 S3: Optuna LightGBM Tuning (max {n_trials} trials / {timeout//60} min)...")

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 15, 63),
            'max_depth':         trial.suggest_int('max_depth', 3, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'objective': 'multiclass', 'metric': 'multi_logloss',
            'num_class': 3, 'class_weight': 'balanced',
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in cv.split(X_tr, y_tr):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr[val_idx], y_tr[val_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            p = m.predict_proba(X_tr[val_idx])
            scores.append(log_loss(y_tr[val_idx], p))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params
    print(f"  ✅ Optuna best log_loss: {study.best_value:.4f}  (trials={len(study.trials)})")
    print(f"  Best params: n_est={best.get('n_estimators')}, "
          f"lr={best.get('learning_rate'):.3f}, "
          f"leaves={best.get('num_leaves')}, depth={best.get('max_depth')}")
    return best


def get_cv_lgbm_params(best_lgbm_params=None):
    if best_lgbm_params is None:
        best_lgbm_params = {}
    base = {
        'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5,
        'num_leaves': 25, 'min_child_samples': 15, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'objective': 'multiclass',
        'metric': 'multi_logloss', 'num_class': 3,
        'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    optuna_keys = ['learning_rate', 'max_depth', 'num_leaves', 'n_estimators',
                   'min_child_samples', 'subsample', 'colsample_bytree',
                   'reg_alpha', 'reg_lambda']
    base.update({k: v for k, v in best_lgbm_params.items() if k in optuna_keys})
    return base


# ══════════════════════════════════════════════════════════════
# SMOTE
# ══════════════════════════════════════════════════════════════

def apply_smote(X_tr, y_tr):
    if not SMOTE_AVAILABLE:
        print("  ⚠️  SMOTE ไม่พร้อม — ใช้ class_weight แทน")
        return X_tr, y_tr

    counts = np.bincount(y_tr)
    target_draw = int(max(counts) * 0.50)
    if target_draw <= counts[1]:
        print(f"  ℹ️  Draw ({counts[1]}) มากพอแล้ว — ข้าม SMOTE")
        return X_tr, y_tr

    sm = SMOTE(sampling_strategy={1: target_draw},
               k_neighbors=min(5, counts[1]-1), random_state=42)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    print(f"  ✅ SMOTE: Draw {counts[1]} → {np.bincount(y_res)[1]}")
    return X_res, y_res


# ══════════════════════════════════════════════════════════════
# 2-STAGE MODEL
# ══════════════════════════════════════════════════════════════

def build_2stage_model(X_train_smote, y_train_smote, best_lgbm_params):
    print("\n🔧 Building v4.0 — 2-Stage Model...")

    y_train_draw   = (y_train_smote == 1).astype(int)
    y_train_nodraw = y_train_smote[y_train_smote != 1]
    X_train_nodraw = X_train_smote[y_train_smote != 1]

    if LGBM_AVAILABLE:
        stage1_params = {**{
            'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 5,
            'num_leaves': 25, 'min_child_samples': 15, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'class_weight': {0: 1, 1: 2.5},
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }, **{k: v for k, v in best_lgbm_params.items() if k in [
            'learning_rate', 'max_depth', 'num_leaves', 'min_child_samples',
            'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'n_estimators'
        ]}}
        stage1_model = lgb.LGBMClassifier(**stage1_params)
        stage2_model = lgb.LGBMClassifier(**{**stage1_params, 'class_weight': 'balanced'})
        print("  Stage 1 (Draw vs Not): LightGBM 🔥  [draw_weight=2.5]")
        print("  Stage 2 (Home vs Away): LightGBM 🔥")
    else:
        stage1_model = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                  learning_rate=0.05, subsample=0.8, random_state=42)
        stage2_model = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                  learning_rate=0.05, subsample=0.8, random_state=42)
        print("  Stage 1 (Draw vs Not): GBT")
        print("  Stage 2 (Home vs Away): GBT")

    stage1_model.fit(X_train_smote, y_train_draw)
    print("  ✅ Stage 1 trained")

    y_train_nodraw_bin = (y_train_nodraw == 2).astype(int)
    stage2_model.fit(X_train_nodraw, y_train_nodraw_bin)
    print("  ✅ Stage 2 trained")

    print("  🎯 Calibrating stages (isotonic)...")
    try:
        stage1_cal = CalibratedClassifierCV(stage1_model, method='isotonic', cv=3)
        stage1_cal.fit(X_train_smote, y_train_draw)
        stage2_cal = CalibratedClassifierCV(stage2_model, method='isotonic', cv=3)
        stage2_cal.fit(X_train_nodraw, y_train_nodraw_bin)
        print("  ✅ Calibration done")
    except Exception as e:
        print(f"  ⚠️  Calibration failed: {e} — using raw models")
        stage1_cal = stage1_model
        stage2_cal = stage2_model

    return stage1_cal, stage2_cal


def predict_2stage(X, stage1_cal, stage2_cal):
    p_draw_stage1   = stage1_cal.predict_proba(X)[:, 1]
    p_notdraw       = 1 - p_draw_stage1
    p_home_given_nd = stage2_cal.predict_proba(X)[:, 1]
    p_away_given_nd = 1 - p_home_given_nd

    p_home_win = p_notdraw * p_home_given_nd
    p_away_win = p_notdraw * p_away_given_nd
    p_draw     = p_draw_stage1

    total = p_home_win + p_draw + p_away_win
    p_home_win /= total; p_draw /= total; p_away_win /= total

    return np.column_stack([p_away_win, p_draw, p_home_win])


# ══════════════════════════════════════════════════════════════
# POISSON MODEL
# ══════════════════════════════════════════════════════════════

def poisson_win_draw_loss(home_xg, away_xg, max_goals=8):
    p_home_win = p_draw = p_away_win = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson.pmf(hg, home_xg) * poisson.pmf(ag, away_xg)
            if hg > ag:    p_home_win += p
            elif hg == ag: p_draw     += p
            else:          p_away_win += p
    total = p_home_win + p_draw + p_away_win
    return p_home_win/total, p_draw/total, p_away_win/total


def build_poisson_model(train, match_df_clean):
    print("\n🔥 Building Poisson Goal Model...")
    poisson_features = [
        'Diff_Elo', 'H_GF_ewm', 'H_GA_ewm', 'A_GF_ewm', 'A_GA_ewm',
        'H_Pts_ewm', 'A_Pts_ewm', 'H_Elo_norm', 'A_Elo_norm',
        'HomeWinRate', 'AwayWinRate',
    ]
    pf_available = [f for f in poisson_features if f in match_df_clean.columns]
    train_p = train[pf_available].fillna(0)

    if 'FTHG' in train.columns and train['FTHG'].notna().sum() > 100:
        y_home_goals = train['FTHG'].fillna(train['FTHG'].median()).clip(0, 8)
        y_away_goals = train['FTAG'].fillna(train['FTAG'].median()).clip(0, 8)
        print("  Using actual FTHG/FTAG for Poisson target")
    else:
        y_home_goals = train['H_GF5'].fillna(1.3).clip(0, 5)
        y_away_goals = train['A_GF5'].fillna(1.1).clip(0, 5)
        print("  Using rolling average as Poisson target (fallback)")

    sc_p = StandardScaler()
    train_p_sc = sc_p.fit_transform(train_p)

    home_poisson = PoissonRegressor(alpha=0.5, max_iter=500)
    away_poisson = PoissonRegressor(alpha=0.5, max_iter=500)
    home_poisson.fit(train_p_sc, y_home_goals.clip(0, 8))
    away_poisson.fit(train_p_sc, y_away_goals.clip(0, 8))
    print("✅ Poisson Goal Model trained")

    return home_poisson, away_poisson, sc_p, pf_available


def build_poisson_proba_for_test(test_df, poisson_features, poisson_scaler,
                                  home_poisson_model, away_poisson_model):
    pf_avail = [f for f in poisson_features if f in test_df.columns]
    X_pois   = test_df[pf_avail].fillna(0)
    X_pois_sc = poisson_scaler.transform(X_pois)
    home_xg_arr = np.clip(home_poisson_model.predict(X_pois_sc), 0.3, 6.0)
    away_xg_arr = np.clip(away_poisson_model.predict(X_pois_sc), 0.3, 6.0)
    proba_poisson = np.zeros((len(test_df), 3))
    for i, (hxg, axg) in enumerate(zip(home_xg_arr, away_xg_arr)):
        ph, pd_, pa = poisson_win_draw_loss(hxg, axg)
        proba_poisson[i] = [pa, pd_, ph]
    return proba_poisson


# ══════════════════════════════════════════════════════════════
# HYBRID BLEND + THRESHOLD
# ══════════════════════════════════════════════════════════════

def blend_ml_poisson_dynamic(ml_proba, poisson_proba, elo_diffs, base_alpha=0.5):
    elo_diffs    = np.array(elo_diffs)
    elo_norm     = np.clip(elo_diffs / 400.0, 0, 1)
    dynamic_alpha = np.clip(base_alpha - 0.2 + 0.4 * elo_norm, 0.1, 0.9)
    blended = np.zeros_like(ml_proba, dtype=float)
    for i in range(ml_proba.shape[0]):
        blended[i] = dynamic_alpha[i] * ml_proba[i] + (1 - dynamic_alpha[i]) * poisson_proba[i]
    row_sums = blended.sum(axis=1, keepdims=True)
    return blended / np.where(row_sums > 0, row_sums, 1)


def blend_ml_poisson(ml_proba, poisson_proba, alpha=0.6):
    blended  = alpha * ml_proba + (1 - alpha) * poisson_proba
    row_sums = blended.sum(axis=1, keepdims=True)
    return blended / np.where(row_sums > 0, row_sums, 1)


def optimize_blend_alpha(ml_proba, poisson_proba, y_true, alphas=None):
    if alphas is None:
        alphas = np.arange(0.3, 1.01, 0.05)
    best_alpha, best_score = 0.6, 0.0
    for a in alphas:
        blended = blend_ml_poisson(ml_proba, poisson_proba, alpha=a)
        preds   = np.argmax(blended, axis=1)
        score   = f1_score(y_true, preds, average='macro', zero_division=0)
        if score > best_score:
            best_score = score; best_alpha = a
    return best_alpha, best_score


def suppress_draw_proba(proba, draw_factor=0.92):
    suppressed      = proba.copy()
    suppressed[:, 1] *= draw_factor
    row_sums        = suppressed.sum(axis=1, keepdims=True)
    return suppressed / np.where(row_sums > 0, row_sums, 1)


def optimize_thresholds(proba, y_true, n_steps=50,
                        t_home_range=(0.15, 0.55), t_draw_range=(0.15, 0.55)):
    best_f1 = 0.0; best_t = (0.33, 0.33)
    thresholds_home = np.linspace(t_home_range[0], t_home_range[1], n_steps)
    thresholds_draw = np.linspace(t_draw_range[0], t_draw_range[1], n_steps)

    for t_draw in thresholds_draw:
        for t_home in thresholds_home:
            preds = []
            for row in proba:
                p_away, p_draw, p_home = row
                if p_draw >= t_draw:   preds.append(1)
                elif p_home >= t_home: preds.append(2)
                else:                  preds.append(0)
            score = f1_score(y_true, preds, average='macro', zero_division=0)
            if score > best_f1:
                best_f1 = score; best_t = (t_home, t_draw)
    return best_t[0], best_t[1], best_f1


def apply_thresholds(proba, t_home, t_draw):
    preds = []
    for row in proba:
        p_away, p_draw, p_home = row
        if p_draw >= t_draw:   preds.append(1)
        elif p_home >= t_home: preds.append(2)
        else:                  preds.append(0)
    return np.array(preds)


# ══════════════════════════════════════════════════════════════
# WRAPPER CLASS
# ══════════════════════════════════════════════════════════════

class TwoStageEnsemble:
    """Wrapper ให้ใช้ได้เหมือน sklearn classifier"""
    def __init__(self, stage1, stage2, t_home=0.33, t_draw=0.33):
        self.stage1  = stage1
        self.stage2  = stage2
        self.t_home  = t_home
        self.t_draw  = t_draw

    def predict_proba(self, X):
        return predict_2stage(X, self.stage1, self.stage2)

    def predict(self, X):
        return apply_thresholds(self.predict_proba(X), self.t_home, self.t_draw)


# ══════════════════════════════════════════════════════════════
# SAVE / LOAD MODEL
# ══════════════════════════════════════════════════════════════

def save_model(bundle):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"✅ Model v9 saved → {MODEL_PATH}")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════

def run_training_pipeline(match_df_clean, FEATURES, home_stats, away_stats,
                           final_elo, final_elo_home, final_elo_away):
    # Split
    train, test, X_train, y_train, X_test, y_test = split_train_test(match_df_clean, FEATURES)

    # Scale
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Optuna tune
    best_lgbm_params = tune_lgbm_optuna(X_train_sc, y_train.values)

    # SMOTE
    print("\n🔥 S5: Applying SMOTE for Draw class...")
    X_train_smote, y_train_smote = apply_smote(X_train_sc, y_train.values)

    # 2-Stage model
    stage1_cal, stage2_cal = build_2stage_model(X_train_smote, y_train_smote, best_lgbm_params)

    # Evaluate 2-stage raw
    proba_2stage      = predict_2stage(X_test_sc, stage1_cal, stage2_cal)
    y_pred_2stage_raw = np.argmax(proba_2stage, axis=1)
    acc_2stage        = accuracy_score(y_test, y_pred_2stage_raw)
    print(f"\n===== 2-STAGE RAW RESULTS =====")
    print(f"Accuracy: {round(acc_2stage*100, 2)}%")
    print(classification_report(y_test, y_pred_2stage_raw,
                                 target_names=['Away Win','Draw','Home Win']))

    # Poisson model
    try:
        home_poisson_model, away_poisson_model, poisson_scaler, poisson_features_used = \
            build_poisson_model(train, match_df_clean)
        POISSON_MODEL_READY = True
    except Exception as e:
        print(f"⚠️  Poisson model failed: {e}")
        POISSON_MODEL_READY = False
        home_poisson_model = away_poisson_model = poisson_scaler = None
        poisson_features_used = []

    # Hybrid blend
    print("\n🔥 PHASE 3: Building Poisson Hybrid Blend...")
    POISSON_HYBRID_READY = False
    best_alpha   = 0.5
    proba_hybrid = proba_2stage.copy()

    if POISSON_MODEL_READY:
        try:
            poisson_proba_test = build_poisson_proba_for_test(
                test, poisson_features_used, poisson_scaler,
                home_poisson_model, away_poisson_model)
            best_alpha, best_blend_f1 = optimize_blend_alpha(
                proba_2stage, poisson_proba_test, y_test.values)
            test_elo_diffs = np.abs(test['Diff_Elo'].fillna(0).values)
            proba_hybrid = blend_ml_poisson_dynamic(
                proba_2stage, poisson_proba_test,
                elo_diffs=test_elo_diffs, base_alpha=best_alpha)
            POISSON_HYBRID_READY = True
            avg_alpha = np.clip(best_alpha - 0.2 + 0.4 * np.clip(test_elo_diffs / 400, 0, 1), 0.1, 0.9).mean()
            print(f"  ✅ Poisson Hybrid (Dynamic α): base={best_alpha:.2f}  avg_α={avg_alpha:.2f}  macro F1={best_blend_f1:.4f}")
        except Exception as e:
            print(f"  ⚠️  Poisson hybrid failed: {e} — using ML-only")

    # Threshold optimization
    print("\n🔥 S6: Optimizing prediction thresholds...")

    # ── Adaptive Draw Suppression ─────────────────────────────────────────
    # คำนวณ bias จริงก่อน แล้ว suppress ให้พอดี
    _draw_raw_mean  = proba_hybrid[:, 1].mean()
    _draw_actual    = (y_test == 1).mean()
    _raw_bias       = (_draw_raw_mean - _draw_actual)

    if _raw_bias > 0.02:
        # suppress เพื่อลด bias: factor = actual_rate / predicted_rate (capped)
        adaptive_factor = float(np.clip(_draw_actual / (_draw_raw_mean + 1e-6), 0.70, 0.96))
        print(f"  📐 Adaptive suppression: raw_bias={_raw_bias*100:+.1f}%  → factor={adaptive_factor:.3f}")
    else:
        adaptive_factor = 0.95  # bias น้อย suppress เบาๆ
        print(f"  📐 Draw bias ต่ำ ({_raw_bias*100:+.1f}%) → factor={adaptive_factor:.3f}")

    DRAW_SUPPRESS_FACTOR = round(adaptive_factor, 3)
    proba_hybrid = suppress_draw_proba(proba_hybrid, draw_factor=DRAW_SUPPRESS_FACTOR)
    print(f"  🔧 Draw suppression applied (factor={DRAW_SUPPRESS_FACTOR})")

    OPT_T_HOME, OPT_T_DRAW, best_macro_f1 = optimize_thresholds(
        proba_hybrid, y_test,
        t_draw_range=(0.15, 0.40), t_home_range=(0.30, 0.55))
    print(f"  Optimal t_home={OPT_T_HOME:.3f}  t_draw={OPT_T_DRAW:.3f}")
    print(f"  Best macro F1 = {best_macro_f1:.4f}")

    # Draw calibration check
    _draw_pred_mean = proba_hybrid[:, 1].mean()
    _draw_actual    = (y_test == 1).mean()
    _draw_bias      = (_draw_pred_mean - _draw_actual) * 100
    print(f"  📐 Draw calibration check: predicted={_draw_pred_mean:.1%}  "
          f"actual={_draw_actual:.1%}  bias={_draw_bias:+.1f}%  "
          f"{'✅ improved' if abs(_draw_bias) < 8 else '⚠️ still biased'}")

    # Final evaluation
    y_pred_final = apply_thresholds(proba_hybrid, OPT_T_HOME, OPT_T_DRAW)
    acc_final    = accuracy_score(y_test, y_pred_final)
    hybrid_tag   = "2-Stage + Poisson Hybrid 🔥" if POISSON_HYBRID_READY else "2-Stage ML only"
    print(f"\n===== v5.0 FINAL RESULTS ({hybrid_tag}) =====")
    print(f"Accuracy : {round(acc_final*100, 2)}%  (ML-only 2-stage: {round(acc_2stage*100,2)}%)")
    if POISSON_HYBRID_READY:
        print(f"Hybrid gain: {round((acc_final - acc_2stage)*100, 2)}%")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_final)}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_final, target_names=['Away Win','Draw','Home Win']))

    # Fallback single-stage
    print("\n🔧 Building fallback single-stage ensemble (for CV + backtest)...")
    if LGBM_AVAILABLE:
        lgbm_clf = lgb.LGBMClassifier(**get_cv_lgbm_params(best_lgbm_params))
    else:
        lgbm_clf = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                              learning_rate=0.05, random_state=42)
    lgbm_clf.fit(X_train_sc, y_train)
    print("  ✅ Fallback single-stage trained")

    print("🎯 Applying Isotonic Calibration (single-stage fallback)...")
    try:
        calibrated_single = CalibratedClassifierCV(lgbm_clf, method='isotonic', cv=3)
        calibrated_single.fit(X_train_sc, y_train)
        acc_cal = accuracy_score(y_test, calibrated_single.predict(X_test_sc))
        print(f"Single-stage Calibrated Accuracy: {round(acc_cal*100, 2)}%")
    except Exception as e:
        print(f"⚠️  Calibration skipped: {e}")
        calibrated_single = lgbm_clf

    # Ensemble wrapper
    ensemble = TwoStageEnsemble(stage1_cal, stage2_cal, OPT_T_HOME, OPT_T_DRAW)

    # Save
    model_bundle = {
        'model':                ensemble,
        'stage1':               stage1_cal,
        'stage2':               stage2_cal,
        'fallback_single':      calibrated_single,
        'scaler':               scaler,
        'features':             FEATURES,
        'elo':                  final_elo,
        'elo_home':             final_elo_home,
        'elo_away':             final_elo_away,
        'teams':                list(final_elo.keys()),
        'home_stats':           home_stats,
        'away_stats':           away_stats,
        'opt_t_home':           OPT_T_HOME,
        'opt_t_draw':           OPT_T_DRAW,
        'draw_suppress_factor': DRAW_SUPPRESS_FACTOR,
        'poisson_hybrid_ready': POISSON_HYBRID_READY,
        'poisson_alpha':        best_alpha if POISSON_HYBRID_READY else 0.6,
        'poisson_model_home':   home_poisson_model if POISSON_MODEL_READY else None,
        'poisson_model_away':   away_poisson_model if POISSON_MODEL_READY else None,
        'poisson_scaler':       poisson_scaler if POISSON_MODEL_READY else None,
        'poisson_features':     poisson_features_used if POISSON_MODEL_READY else [],
        'version':              '9.0',
    }
    save_model(model_bundle)

    return {
        'train': train, 'test': test,
        'X_train_sc': X_train_sc, 'X_test_sc': X_test_sc,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler, 'ensemble': ensemble,
        'stage1_cal': stage1_cal, 'stage2_cal': stage2_cal,
        'calibrated_single': calibrated_single,
        'proba_hybrid': proba_hybrid, 'proba_2stage': proba_2stage,
        'y_pred_final': y_pred_final,
        'OPT_T_HOME': OPT_T_HOME, 'OPT_T_DRAW': OPT_T_DRAW,
        'DRAW_SUPPRESS_FACTOR': DRAW_SUPPRESS_FACTOR,
        'POISSON_HYBRID_READY': POISSON_HYBRID_READY,
        'POISSON_MODEL_READY':  POISSON_MODEL_READY,
        'best_alpha':           best_alpha,
        'home_poisson_model':   home_poisson_model,
        'away_poisson_model':   away_poisson_model,
        'poisson_scaler':       poisson_scaler,
        'poisson_features_used': poisson_features_used,
        'best_lgbm_params':     best_lgbm_params,
    }