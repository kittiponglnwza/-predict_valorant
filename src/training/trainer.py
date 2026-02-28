from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.config import CATBOOST_AVAILABLE, CatBoostClassifier, LGBM_AVAILABLE, lgb
from src.training.evaluator import evaluate_classifier


def build_stabilize_model(draw_weight: float = 1.0, engine: str = "auto"):
    if engine == "auto":
        engine = "lgbm" if LGBM_AVAILABLE else "gbt"

    if engine == "lgbm" and LGBM_AVAILABLE:
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            # [PATCH-1] ลด n_estimators + เพิ่ม regularization เพื่อลด overfitting
            n_estimators=250,
            learning_rate=0.04,
            max_depth=4,          # ลดจาก 6 → 4
            num_leaves=20,        # ลดจาก 31 → 20
            subsample=0.7,        # ลดจาก 0.8 → 0.7
            colsample_bytree=0.7, # ลดจาก 0.8 → 0.7
            min_child_samples=30, # เพิ่ม regularization
            reg_alpha=0.1,        # L1
            reg_lambda=1.0,       # L2
            # [STEP 2+3] away=1.25 ลด Home bias
            class_weight={0: 1.25, 1: float(draw_weight), 2: 1.0},
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    if engine == "catboost" and CATBOOST_AVAILABLE:
        return CatBoostClassifier(
            loss_function="MultiClass",
            # [PATCH-1] ลด iterations + เพิ่ม regularization
            iterations=300,       # ลดจาก 450 → 300
            depth=4,              # ลดจาก 6 → 4
            learning_rate=0.04,
            l2_leaf_reg=5.0,      # เพิ่ม L2
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            # [STEP 2+3] away=1.25 ลด Home bias
            class_weights=[1.25, float(draw_weight), 1.0],
        )

    return GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )


def align_proba_to_labels(
    proba: np.ndarray,
    model_classes: np.ndarray,
    labels: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(labels)), dtype=float)
    class_to_idx = {int(c): i for i, c in enumerate(model_classes)}
    for out_idx, label in enumerate(labels):
        if label in class_to_idx:
            out[:, out_idx] = proba[:, class_to_idx[label]]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


def apply_thresholds(proba: np.ndarray, t_home: float, t_draw: float) -> np.ndarray:
    """
    [STEP 2] Symmetric decision rule — ไม่ให้ Away เป็น default fallback
    Logic:
      1. p_home >= t_home → Home
      2. p_away >= t_home → Away  (symmetric threshold, ไม่ใช้ค่าต่างกัน)
      3. p_draw >= t_draw → Draw  (check หลังสุด: predict draw เฉพาะเมื่อ confident)
      4. else             → argmax (ไม่มี away-bias)
    """
    proba = np.asarray(proba)
    preds = np.empty(len(proba), dtype=int)
    for i, row in enumerate(proba):
        p_away, p_draw, p_home = float(row[0]), float(row[1]), float(row[2])
        if p_home >= t_home:
            preds[i] = 2
        elif p_away >= t_home:
            preds[i] = 0
        elif p_draw >= t_draw:
            preds[i] = 1
        else:
            preds[i] = int(np.argmax(row))
    return preds


def optimize_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_steps: int = 31,
    t_home_range: tuple[float, float] = (0.30, 0.70),
    t_draw_range: tuple[float, float] = (0.08, 0.65),
    min_recall: dict[int, float] | None = None,
    objective: str = "macro_f1",
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    constraints = min_recall or {}

    best_primary = -1.0
    best_secondary = -1.0
    best_draw_gap = float("inf")
    best_t_home = 0.5
    best_t_draw = 0.33
    true_draw_rate = float(np.mean(y_true == 1))

    home_grid = np.linspace(t_home_range[0], t_home_range[1], n_steps)
    draw_grid = np.linspace(t_draw_range[0], t_draw_range[1], n_steps)

    for t_draw in draw_grid:
        for t_home in home_grid:
            pred = apply_thresholds(proba, t_home=t_home, t_draw=t_draw)
            recalls = recall_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)
            label_to_recall = {0: float(recalls[0]), 1: float(recalls[1]), 2: float(recalls[2])}

            if constraints:
                valid = True
                for label, min_val in constraints.items():
                    if label_to_recall.get(label, 0.0) < float(min_val):
                        valid = False
                        break
                if not valid:
                    continue

            acc = float(accuracy_score(y_true, pred))
            macro_f1 = float(f1_score(y_true, pred, average="macro", zero_division=0))
            draw_gap = abs(float(np.mean(pred == 1)) - true_draw_rate)

            if objective == "accuracy":
                primary = acc
                secondary = macro_f1
            else:
                primary = macro_f1
                secondary = acc

            if primary > best_primary:
                best_primary = primary
                best_secondary = secondary
                best_draw_gap = float(draw_gap)
                best_t_home = float(t_home)
                best_t_draw = float(t_draw)
            elif abs(primary - best_primary) < 1e-12 and secondary > best_secondary:
                best_secondary = secondary
                best_draw_gap = float(draw_gap)
                best_t_home = float(t_home)
                best_t_draw = float(t_draw)
            elif (
                abs(primary - best_primary) < 1e-12
                and abs(secondary - best_secondary) < 1e-12
                and draw_gap < best_draw_gap
            ):
                best_draw_gap = float(draw_gap)
                best_t_home = float(t_home)
                best_t_draw = float(t_draw)

    if best_primary < 0 and constraints:
        return optimize_thresholds(
            proba=proba,
            y_true=y_true,
            n_steps=n_steps,
            t_home_range=t_home_range,
            t_draw_range=t_draw_range,
            min_recall=None,
            objective=objective,
        )

    return best_t_home, best_t_draw, float(best_primary)


def _prepare_xy(
    train_df,
    eval_df,
    features,
    target_col: str = "Result3",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = train_df[features].fillna(0)
    y_train = train_df[target_col].astype(int).values
    x_eval = eval_df[features].fillna(0)
    y_eval = eval_df[target_col].astype(int).values

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_eval_sc = scaler.transform(x_eval)
    return x_train_sc, y_train, x_eval_sc, y_eval


def _fit_estimator(
    base,
    x_train_sc: np.ndarray,
    y_train: np.ndarray,
    draw_weight: float,
    use_sigmoid_calibration: bool,
):
    sample_weight = np.where(y_train == 1, float(draw_weight), 1.0)

    if use_sigmoid_calibration:
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        try:
            calibrated.fit(x_train_sc, y_train)
            return calibrated, True
        except TypeError:
            calibrated.fit(x_train_sc, y_train, sample_weight=sample_weight)
            return calibrated, True
        except Exception:
            pass

    try:
        if isinstance(base, GradientBoostingClassifier):
            base.fit(x_train_sc, y_train, sample_weight=sample_weight)
        else:
            base.fit(x_train_sc, y_train)
    except TypeError:
        base.fit(x_train_sc, y_train)

    return base, False


def _fit_probability_sources(
    x_train_sc: np.ndarray,
    y_train: np.ndarray,
    eval_df,
    x_eval_sc: np.ndarray,
    draw_weight: float,
    use_sigmoid_calibration: bool,
    labels: tuple[int, ...] = (0, 1, 2),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    proba_by_source: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"calibrated": False, "sources": []}

    candidates: list[tuple[str, Any]] = []
    if LGBM_AVAILABLE:
        candidates.append(("lgbm", build_stabilize_model(draw_weight=draw_weight, engine="lgbm")))
    if CATBOOST_AVAILABLE:
        candidates.append(("catboost", build_stabilize_model(draw_weight=draw_weight, engine="catboost")))
    if not candidates:
        candidates.append(("gbt", build_stabilize_model(draw_weight=draw_weight, engine="gbt")))

    for name, base_model in candidates:
        fitted, calibrated = _fit_estimator(
            base=base_model,
            x_train_sc=x_train_sc,
            y_train=y_train,
            draw_weight=draw_weight,
            use_sigmoid_calibration=use_sigmoid_calibration,
        )
        metadata["calibrated"] = bool(metadata["calibrated"] or calibrated)
        proba_raw = fitted.predict_proba(x_eval_sc)
        proba = align_proba_to_labels(proba_raw, np.asarray(fitted.classes_), labels=labels)
        proba_by_source[name] = proba
        metadata["sources"].append(name)

    poisson_proba = _build_poisson_probabilities(eval_df, labels=labels)
    if poisson_proba is not None:
        proba_by_source["poisson"] = poisson_proba
        metadata["sources"].append("poisson")

    return proba_by_source, metadata


def _poisson_wdl_single(home_lambda: float, away_lambda: float, max_goals: int = 8) -> tuple[float, float, float]:
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for hg in range(max_goals + 1):
        p_hg = poisson.pmf(hg, home_lambda)
        for ag in range(max_goals + 1):
            p = p_hg * poisson.pmf(ag, away_lambda)
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    if total <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return p_home / total, p_draw / total, p_away / total


def _build_poisson_probabilities(eval_df, labels: tuple[int, ...] = (0, 1, 2)) -> np.ndarray | None:
    needed = ["H_GF_ewm", "H_GA_ewm", "A_GF_ewm", "A_GA_ewm", "Diff_Elo"]
    if any(col not in eval_df.columns for col in needed):
        return None

    tmp = eval_df[needed].copy()
    for col in needed:
        tmp[col] = tmp[col].replace([np.inf, -np.inf], np.nan)
        tmp[col] = tmp[col].fillna(tmp[col].median())
        tmp[col] = tmp[col].fillna(0.0)

    elo_adj = np.clip(tmp["Diff_Elo"].values / 400.0, -0.6, 0.6)
    home_lambda = 0.58 * tmp["H_GF_ewm"].values + 0.42 * tmp["A_GA_ewm"].values
    away_lambda = 0.58 * tmp["A_GF_ewm"].values + 0.42 * tmp["H_GA_ewm"].values
    home_lambda = np.clip(home_lambda * (1.0 + 0.08 * elo_adj) + 0.08, 0.2, 4.0)
    away_lambda = np.clip(away_lambda * (1.0 - 0.08 * elo_adj) + 0.08, 0.2, 4.0)

    out = np.zeros((len(tmp), len(labels)), dtype=float)
    for i, (lh, la) in enumerate(zip(home_lambda, away_lambda)):
        p_home, p_draw, p_away = _poisson_wdl_single(float(lh), float(la))
        # Labels are (away, draw, home)
        out[i, 0] = p_away
        out[i, 1] = p_draw
        out[i, 2] = p_home
    return out


def _normalize_weights(weights: dict[str, float], source_names: list[str]) -> dict[str, float]:
    cleaned = {name: float(weights.get(name, 0.0)) for name in source_names}
    total = sum(max(v, 0.0) for v in cleaned.values())
    if total <= 0:
        equal = 1.0 / len(source_names)
        return {name: equal for name in source_names}
    return {name: max(v, 0.0) / total for name, v in cleaned.items()}


def _generate_weight_candidates(source_names: list[str]) -> list[dict[str, float]]:
    source_names = sorted(source_names)
    if len(source_names) == 1:
        return [{source_names[0]: 1.0}]

    candidates: list[dict[str, float]] = []
    has_poisson = "poisson" in source_names

    if has_poisson:
        non_poisson = [s for s in source_names if s != "poisson"]
        # [STEP 4] เพิ่ม poisson น้อยลง (0.05, 0.10) เน้น lgbm-heavy blend
        for w_p in (0.00, 0.05, 0.10, 0.15, 0.20):
            rem = 1.0 - w_p
            if len(non_poisson) == 1:
                candidates.append({non_poisson[0]: rem, "poisson": w_p})
            elif len(non_poisson) == 2:
                # [STEP 4] เพิ่ม lgbm-heavy fracs (0.70, 0.80) ถ้า lgbm อยู่ใน non_poisson
                base_fracs = [0.35, 0.50, 0.65]
                extra_fracs: list[float] = []
                np_sorted = sorted(non_poisson)  # catboost < lgbm alphabetically
                if "lgbm" in np_sorted and "catboost" in np_sorted:
                    lgbm_idx = np_sorted.index("lgbm")
                    # frac คือ weight ของ non_poisson[0] (catboost)
                    # lgbm-heavy = catboost frac ต่ำ
                    extra_fracs = [0.20, 0.30] if lgbm_idx == 1 else [0.70, 0.80]
                for frac in base_fracs + extra_fracs:
                    candidates.append(
                        {
                            non_poisson[0]: rem * frac,
                            non_poisson[1]: rem * (1.0 - frac),
                            "poisson": w_p,
                        }
                    )
            else:
                eq = rem / len(non_poisson)
                row = {name: eq for name in non_poisson}
                row["poisson"] = w_p
                candidates.append(row)
    else:
        if len(source_names) == 2:
            a, b = source_names
            for w_a in (0.20, 0.35, 0.50, 0.65, 0.80):
                candidates.append({a: w_a, b: 1.0 - w_a})
        else:
            eq = 1.0 / len(source_names)
            candidates.append({name: eq for name in source_names})

    # Also evaluate single-source candidates.
    for name in source_names:
        row = {s: 0.0 for s in source_names}
        row[name] = 1.0
        candidates.append(row)

    deduped: list[dict[str, float]] = []
    seen = set()
    for row in candidates:
        norm = _normalize_weights(row, source_names)
        key = tuple((k, round(v, 4)) for k, v in sorted(norm.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(norm)
    return deduped


def _blend_probabilities(proba_by_source: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    source_names = sorted(proba_by_source.keys())
    norm_weights = _normalize_weights(weights, source_names)
    out = np.zeros_like(next(iter(proba_by_source.values())))
    for name in source_names:
        out += proba_by_source[name] * float(norm_weights[name])
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


def _default_min_recall(selection_metric: str) -> dict[int, float]:
    # [STEP 2+FIX] away=0.25 บังคับจับ Away, draw=0.08 ป้องกัน zero-draw collapse
    # home=0.25 คุมไม่ให้ over-predict Home
    return {0: 0.25, 1: 0.08, 2: 0.25}


def run_baseline_fold(
    train_df,
    eval_df,
    features,
    target_col: str = "Result3",
    labels: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    x_train_sc, y_train, x_eval_sc, y_eval = _prepare_xy(train_df, eval_df, features, target_col)
    sources, _ = _fit_probability_sources(
        x_train_sc=x_train_sc,
        y_train=y_train,
        eval_df=eval_df,
        x_eval_sc=x_eval_sc,
        draw_weight=1.0,
        use_sigmoid_calibration=False,
        labels=labels,
    )

    preferred = "lgbm" if "lgbm" in sources else next(iter(sources.keys()))
    proba = sources[preferred]
    metrics = evaluate_classifier(y_eval, proba, labels=labels)
    return {
        "metrics": metrics,
        "draw_weight": 1.0,
        "thresholds": None,
        "sigmoid_calibrated": False,
        "blend_weights": {preferred: 1.0},
        "sources": list(sources.keys()),
        "selection_metric": "accuracy",
    }


def tune_fold_on_val(
    train_df,
    eval_df,
    features,
    target_col: str = "Result3",
    labels: tuple[int, ...] = (0, 1, 2),
    # [FIX] draw_weight ต้องสูงพอที่ softmax output draw proba เป็น max บ้าง
    # draw_weight=1.0 ทำให้ draw proba ต่ำกว่า home/away เสมอ → argmax ไม่ช่วย
    # ค่า 1.5–2.5 ทำให้ model "เห็น" draw มากพอ แต่ไม่ overfit
    draw_weight_candidates: tuple[float, ...] = (1.5, 2.0, 2.5),
    use_sigmoid_options: tuple[bool, ...] = (True, False),
    selection_metric: str = "macro_f1",
    # [STEP 1] min_recall=None — accuracy-only mode ไม่บังคับ class constraint
    min_recall: dict[int, float] | None = None,
) -> dict[str, Any]:
    x_train_sc, y_train, x_eval_sc, y_eval = _prepare_xy(train_df, eval_df, features, target_col)
    # [FIX] บังคับ draw_recall ≥ 0.05 — ป้องกัน model predict draw=0 ตลอด
    # constraint เล็กน้อยนี้บังคับ optimizer ให้หา threshold ที่ predict draw บ้าง
    # ถ้า accuracy ลดแค่ 0.001–0.002 แต่ได้ draw prediction กลับมา = worth it
    recall_constraints: dict[int, float] = {1: 0.05}

    best: dict[str, Any] | None = None
    trial_rows: list[dict[str, Any]] = []

    for use_sigmoid in use_sigmoid_options:
        for dw in draw_weight_candidates:
            proba_sources, meta = _fit_probability_sources(
                x_train_sc=x_train_sc,
                y_train=y_train,
                eval_df=eval_df,
                x_eval_sc=x_eval_sc,
                draw_weight=float(dw),
                use_sigmoid_calibration=bool(use_sigmoid),
                labels=labels,
            )

            weight_candidates = _generate_weight_candidates(list(proba_sources.keys()))
            for blend_weights in weight_candidates:
                proba = _blend_probabilities(proba_sources, blend_weights)
                t_home, t_draw, tuned_primary = optimize_thresholds(
                    proba=proba,
                    y_true=y_eval,
                    # [STEP 2] t_home symmetric — ค้นหา threshold ที่ดีที่สุด
                    # t_home ใช้กับทั้ง Home และ Away (symmetric decision rule)
                    # t_draw สำหรับ Draw เท่านั้น (argmax fallback ช่วย)
                    t_home_range=(0.33, 0.58),
                    t_draw_range=(0.26, 0.45),
                    min_recall=recall_constraints,
                    objective="accuracy",
                )
                pred_tuned = apply_thresholds(proba, t_home=t_home, t_draw=t_draw)
                metrics = evaluate_classifier(y_eval, proba, labels=labels, y_pred=pred_tuned)
                # [PATCH-4] เลือก best โดย accuracy เป็น primary, macro_f1 เป็น secondary
                score_primary = float(metrics["accuracy"])

                row = {
                    "draw_weight": float(dw),
                    "use_sigmoid_calibration": bool(meta["calibrated"]),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "log_loss": float(metrics["log_loss"]),
                    "t_home": float(t_home),
                    "t_draw": float(t_draw),
                    "selection_metric": selection_metric,
                    # [PATCH-5] selection_score ใช้ accuracy
                    "selection_score": float(metrics["accuracy"]),
                    "blend_weights": {k: float(v) for k, v in blend_weights.items()},
                    "sources": list(proba_sources.keys()),
                }
                trial_rows.append(row)

                if best is None:
                    best = {
                        "metrics": metrics,
                        "draw_weight": float(dw),
                        "thresholds": {"t_home": float(t_home), "t_draw": float(t_draw)},
                        "sigmoid_calibrated": bool(meta["calibrated"]),
                        "selection_metric": selection_metric,
                        "selection_score": float(score_primary),
                        "tuned_primary_score": float(tuned_primary),
                        "blend_weights": {k: float(v) for k, v in blend_weights.items()},
                        "sources": list(proba_sources.keys()),
                    }
                    continue

                best_primary = float(best["selection_score"])
                curr_macro = float(metrics["macro_f1"])
                best_macro = float(best["metrics"]["macro_f1"])
                curr_ll = float(metrics["log_loss"])
                best_ll = float(best["metrics"]["log_loss"])

                if score_primary > best_primary:
                    best = {
                        "metrics": metrics,
                        "draw_weight": float(dw),
                        "thresholds": {"t_home": float(t_home), "t_draw": float(t_draw)},
                        "sigmoid_calibrated": bool(meta["calibrated"]),
                        "selection_metric": selection_metric,
                        "selection_score": float(score_primary),
                        "tuned_primary_score": float(tuned_primary),
                        "blend_weights": {k: float(v) for k, v in blend_weights.items()},
                        "sources": list(proba_sources.keys()),
                    }
                elif abs(score_primary - best_primary) < 1e-12 and curr_macro > best_macro:
                    best = {
                        "metrics": metrics,
                        "draw_weight": float(dw),
                        "thresholds": {"t_home": float(t_home), "t_draw": float(t_draw)},
                        "sigmoid_calibrated": bool(meta["calibrated"]),
                        "selection_metric": selection_metric,
                        "selection_score": float(score_primary),
                        "tuned_primary_score": float(tuned_primary),
                        "blend_weights": {k: float(v) for k, v in blend_weights.items()},
                        "sources": list(proba_sources.keys()),
                    }
                elif (
                    abs(score_primary - best_primary) < 1e-12
                    and abs(curr_macro - best_macro) < 1e-12
                    and curr_ll < best_ll
                ):
                    best = {
                        "metrics": metrics,
                        "draw_weight": float(dw),
                        "thresholds": {"t_home": float(t_home), "t_draw": float(t_draw)},
                        "sigmoid_calibrated": bool(meta["calibrated"]),
                        "selection_metric": selection_metric,
                        "selection_score": float(score_primary),
                        "tuned_primary_score": float(tuned_primary),
                        "blend_weights": {k: float(v) for k, v in blend_weights.items()},
                        "sources": list(proba_sources.keys()),
                    }

    assert best is not None
    best["draw_weight_trials"] = trial_rows
    return best


def run_improved_with_fixed_settings(
    train_df,
    eval_df,
    features,
    draw_weight: float,
    t_home: float,
    t_draw: float,
    use_sigmoid_calibration: bool = True,
    target_col: str = "Result3",
    labels: tuple[int, ...] = (0, 1, 2),
    blend_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    x_train_sc, y_train, x_eval_sc, y_eval = _prepare_xy(train_df, eval_df, features, target_col)
    proba_sources, meta = _fit_probability_sources(
        x_train_sc=x_train_sc,
        y_train=y_train,
        eval_df=eval_df,
        x_eval_sc=x_eval_sc,
        draw_weight=float(draw_weight),
        use_sigmoid_calibration=bool(use_sigmoid_calibration),
        labels=labels,
    )

    source_names = sorted(proba_sources.keys())
    if blend_weights:
        weights = _normalize_weights({k: float(v) for k, v in blend_weights.items()}, source_names)
    else:
        equal = 1.0 / len(source_names)
        weights = {name: equal for name in source_names}

    proba = _blend_probabilities(proba_sources, weights)
    pred_tuned = apply_thresholds(proba, t_home=float(t_home), t_draw=float(t_draw))
    metrics = evaluate_classifier(y_eval, proba, labels=labels, y_pred=pred_tuned)
    return {
        "metrics": metrics,
        "draw_weight": float(draw_weight),
        "thresholds": {"t_home": float(t_home), "t_draw": float(t_draw)},
        "sigmoid_calibrated": bool(meta["calibrated"]),
        "blend_weights": {k: float(v) for k, v in weights.items()},
        "sources": list(source_names),
    }


def get_fold_probabilities(
    train_df,
    eval_df,
    features,
    draw_weight: float,
    use_sigmoid_calibration: bool = True,
    target_col: str = "Result3",
    labels: tuple[int, ...] = (0, 1, 2),
    blend_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    x_train_sc, y_train, x_eval_sc, y_eval = _prepare_xy(train_df, eval_df, features, target_col)
    proba_sources, meta = _fit_probability_sources(
        x_train_sc=x_train_sc,
        y_train=y_train,
        eval_df=eval_df,
        x_eval_sc=x_eval_sc,
        draw_weight=float(draw_weight),
        use_sigmoid_calibration=bool(use_sigmoid_calibration),
        labels=labels,
    )

    source_names = sorted(proba_sources.keys())
    if blend_weights:
        weights = _normalize_weights({k: float(v) for k, v in blend_weights.items()}, source_names)
    else:
        equal = 1.0 / len(source_names)
        weights = {name: equal for name in source_names}

    proba = _blend_probabilities(proba_sources, weights)
    return {
        "y_true": y_eval,
        "proba": proba,
        "sigmoid_calibrated": bool(meta["calibrated"]),
        "sources": list(source_names),
        "blend_weights": {k: float(v) for k, v in weights.items()},
    }


def run_single_fold(
    train_df,
    eval_df,
    features,
    target_col: str = "Result3",
    labels: tuple[int, ...] = (0, 1, 2),
):
    # Backward-compatible wrapper: returns baseline-only behavior.
    return run_baseline_fold(
        train_df=train_df,
        eval_df=eval_df,
        features=features,
        target_col=target_col,
        labels=labels,
    )["metrics"]