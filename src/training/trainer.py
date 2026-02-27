from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

from src.config import LGBM_AVAILABLE, lgb
from src.training.evaluator import evaluate_classifier


def build_stabilize_model(draw_weight: float = 1.0):
    if LGBM_AVAILABLE:
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight={0: 1.0, 1: float(draw_weight), 2: 1.0},
            random_state=42,
            n_jobs=-1,
            verbose=-1,
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
    preds = []
    for row in proba:
        p_away, p_draw, p_home = row
        if p_draw >= t_draw:
            preds.append(1)
        elif p_home >= t_home:
            preds.append(2)
        else:
            preds.append(0)
    return np.asarray(preds, dtype=int)


def optimize_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_steps: int = 41,
    t_home_range: tuple[float, float] = (0.30, 0.65),
    t_draw_range: tuple[float, float] = (0.10, 0.60),
    min_recall: dict[int, float] | None = None,
) -> tuple[float, float, float]:
    best_f1 = -1.0
    best_balance = -1.0
    best_draw_gap = float("inf")
    best_t_home = 0.5
    best_t_draw = 0.33
    home_grid = np.linspace(t_home_range[0], t_home_range[1], n_steps)
    draw_grid = np.linspace(t_draw_range[0], t_draw_range[1], n_steps)
    y_true = np.asarray(y_true).astype(int)
    true_draw_rate = float(np.mean(y_true == 1))
    constraints = min_recall or {}

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
            score = f1_score(y_true, pred, average="macro", zero_division=0)
            balance = min(label_to_recall[0], label_to_recall[2])
            draw_gap = abs(float(np.mean(pred == 1)) - true_draw_rate)
            if score > best_f1:
                best_f1 = float(score)
                best_balance = float(balance)
                best_draw_gap = float(draw_gap)
                best_t_home = float(t_home)
                best_t_draw = float(t_draw)
            elif abs(score - best_f1) < 1e-12 and balance > best_balance:
                best_balance = float(balance)
                best_draw_gap = float(draw_gap)
                best_t_home = float(t_home)
                best_t_draw = float(t_draw)
            elif (
                abs(score - best_f1) < 1e-12
                and abs(balance - best_balance) < 1e-12
                and draw_gap < best_draw_gap
            ):
                best_draw_gap = float(draw_gap)
                best_t_home = float(t_home)
                best_t_draw = float(t_draw)

    if best_f1 < 0 and constraints:
        # Fallback if constraints are too strict for this fold.
        return optimize_thresholds(
            proba=proba,
            y_true=y_true,
            n_steps=n_steps,
            t_home_range=t_home_range,
            t_draw_range=t_draw_range,
            min_recall=None,
        )

    return best_t_home, best_t_draw, best_f1


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


def _fit_model(
    x_train_sc: np.ndarray,
    y_train: np.ndarray,
    draw_weight: float,
    use_sigmoid_calibration: bool,
):
    base = build_stabilize_model(draw_weight=draw_weight)
    if use_sigmoid_calibration:
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        try:
            if LGBM_AVAILABLE:
                calibrated.fit(x_train_sc, y_train)
            else:
                sw = np.where(y_train == 1, float(draw_weight), 1.0)
                calibrated.fit(x_train_sc, y_train, sample_weight=sw)
            return calibrated, True
        except Exception:
            pass

    if LGBM_AVAILABLE:
        base.fit(x_train_sc, y_train)
    else:
        sw = np.where(y_train == 1, float(draw_weight), 1.0)
        base.fit(x_train_sc, y_train, sample_weight=sw)
    return base, False


def run_baseline_fold(
    train_df,
    eval_df,
    features,
    target_col: str = "Result3",
    labels: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    x_train_sc, y_train, x_eval_sc, y_eval = _prepare_xy(train_df, eval_df, features, target_col)

    model = build_stabilize_model(draw_weight=1.0)
    if LGBM_AVAILABLE:
        model.fit(x_train_sc, y_train)
    else:
        model.fit(x_train_sc, y_train)

    proba_raw = model.predict_proba(x_eval_sc)
    proba = align_proba_to_labels(proba_raw, np.asarray(model.classes_), labels=labels)
    metrics = evaluate_classifier(y_eval, proba, labels=labels)
    return {
        "metrics": metrics,
        "draw_weight": 1.0,
        "thresholds": None,
        "sigmoid_calibrated": False,
    }


def tune_fold_on_val(
    train_df,
    eval_df,
    features,
    target_col: str = "Result3",
    labels: tuple[int, ...] = (0, 1, 2),
    draw_weight_candidates: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    use_sigmoid_options: tuple[bool, ...] = (False, True),
) -> dict[str, Any]:
    x_train_sc, y_train, x_eval_sc, y_eval = _prepare_xy(train_df, eval_df, features, target_col)

    best: dict[str, Any] | None = None
    trial_rows: list[dict[str, float]] = []

    for use_sigmoid in use_sigmoid_options:
        for dw in draw_weight_candidates:
            model, calibrated_used = _fit_model(
                x_train_sc,
                y_train,
                draw_weight=float(dw),
                use_sigmoid_calibration=bool(use_sigmoid),
            )
            proba_raw = model.predict_proba(x_eval_sc)
            proba = align_proba_to_labels(proba_raw, np.asarray(model.classes_), labels=labels)

            t_home, t_draw, tuned_f1 = optimize_thresholds(
                proba,
                y_eval,
                min_recall={0: 0.15, 2: 0.20},
            )
            pred_tuned = apply_thresholds(proba, t_home=t_home, t_draw=t_draw)
            metrics = evaluate_classifier(y_eval, proba, labels=labels, y_pred=pred_tuned)

            row = {
                "draw_weight": float(dw),
                "use_sigmoid_calibration": bool(calibrated_used),
                "macro_f1": float(metrics["macro_f1"]),
                "log_loss": float(metrics["log_loss"]),
                "t_home": float(t_home),
                "t_draw": float(t_draw),
            }
            trial_rows.append(row)

            if best is None:
                best = {
                    "metrics": metrics,
                    "draw_weight": float(dw),
                    "thresholds": {"t_home": float(t_home), "t_draw": float(t_draw)},
                    "sigmoid_calibrated": bool(calibrated_used),
                    "tuned_macro_f1": float(tuned_f1),
                }
                continue

            curr_f1 = float(metrics["macro_f1"])
            best_f1 = float(best["metrics"]["macro_f1"])
            if curr_f1 > best_f1:
                best = {
                    "metrics": metrics,
                    "draw_weight": float(dw),
                    "thresholds": {"t_home": float(t_home), "t_draw": float(t_draw)},
                    "sigmoid_calibrated": bool(calibrated_used),
                    "tuned_macro_f1": float(tuned_f1),
                }
            elif abs(curr_f1 - best_f1) < 1e-12 and float(metrics["log_loss"]) < float(best["metrics"]["log_loss"]):
                best = {
                    "metrics": metrics,
                    "draw_weight": float(dw),
                    "thresholds": {"t_home": float(t_home), "t_draw": float(t_draw)},
                    "sigmoid_calibrated": bool(calibrated_used),
                    "tuned_macro_f1": float(tuned_f1),
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
) -> dict[str, Any]:
    x_train_sc, y_train, x_eval_sc, y_eval = _prepare_xy(train_df, eval_df, features, target_col)
    model, calibrated_used = _fit_model(
        x_train_sc,
        y_train,
        draw_weight=float(draw_weight),
        use_sigmoid_calibration=bool(use_sigmoid_calibration),
    )
    proba_raw = model.predict_proba(x_eval_sc)
    proba = align_proba_to_labels(proba_raw, np.asarray(model.classes_), labels=labels)
    pred_tuned = apply_thresholds(proba, t_home=float(t_home), t_draw=float(t_draw))
    metrics = evaluate_classifier(y_eval, proba, labels=labels, y_pred=pred_tuned)
    return {
        "metrics": metrics,
        "draw_weight": float(draw_weight),
        "thresholds": {"t_home": float(t_home), "t_draw": float(t_draw)},
        "sigmoid_calibrated": bool(calibrated_used),
    }


def get_fold_probabilities(
    train_df,
    eval_df,
    features,
    draw_weight: float,
    use_sigmoid_calibration: bool = True,
    target_col: str = "Result3",
    labels: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    x_train_sc, y_train, x_eval_sc, y_eval = _prepare_xy(train_df, eval_df, features, target_col)
    model, calibrated_used = _fit_model(
        x_train_sc,
        y_train,
        draw_weight=float(draw_weight),
        use_sigmoid_calibration=bool(use_sigmoid_calibration),
    )
    proba_raw = model.predict_proba(x_eval_sc)
    proba = align_proba_to_labels(proba_raw, np.asarray(model.classes_), labels=labels)
    return {
        "y_true": y_eval,
        "proba": proba,
        "sigmoid_calibrated": bool(calibrated_used),
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
