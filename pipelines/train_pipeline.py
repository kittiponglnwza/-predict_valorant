from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.features import run_feature_pipeline
from src.training.splitters import (
    add_season_column,
    build_stabilize_folds,
    split_by_seasons,
    validate_fold_coverage,
)
from src.training.trainer import (
    get_fold_probabilities,
    optimize_thresholds,
    run_baseline_fold,
    run_improved_with_fixed_settings,
    tune_fold_on_val,
)


REPORT_PATH = Path("artifacts/reports/stabilize_backtest_report.json")


def _metric_delta(after: float, before: float) -> float:
    return float(after - before)


def _select_global_model_config(val_results: list[dict]) -> dict:
    # Aggregate all (draw_weight, sigmoid_flag) trials across validation folds.
    by_cfg: dict[tuple[float, bool], dict[str, list[float]]] = {}
    for fold_res in val_results:
        trials = fold_res["after"]["draw_weight_trials"]
        for row in trials:
            dw = float(row["draw_weight"])
            use_sigmoid = bool(row.get("use_sigmoid_calibration", False))
            key = (dw, use_sigmoid)
            by_cfg.setdefault(key, {"macro_f1": [], "log_loss": []})
            by_cfg[key]["macro_f1"].append(float(row["macro_f1"]))
            by_cfg[key]["log_loss"].append(float(row["log_loss"]))

    best_dw = 1.0
    best_sigmoid = False
    best_f1 = -1.0
    best_ll = float("inf")
    for (dw, use_sigmoid), stats in sorted(by_cfg.items()):
        mean_f1 = float(np.mean(stats["macro_f1"]))
        mean_ll = float(np.mean(stats["log_loss"]))
        if mean_f1 > best_f1:
            best_dw = dw
            best_sigmoid = use_sigmoid
            best_f1 = mean_f1
            best_ll = mean_ll
        elif abs(mean_f1 - best_f1) < 1e-12 and mean_ll < best_ll:
            best_dw = dw
            best_sigmoid = use_sigmoid
            best_f1 = mean_f1
            best_ll = mean_ll
    return {
        "draw_weight": float(best_dw),
        "use_sigmoid_calibration": bool(best_sigmoid),
    }


def _build_global_thresholds(
    match_df_clean,
    features,
    val_folds,
    draw_weight: float,
    use_sigmoid_calibration: bool,
) -> dict:
    pooled_proba = []
    pooled_true = []
    for fold in val_folds:
        train_df, eval_df = split_by_seasons(match_df_clean, fold, season_col="Season")
        fold_prob = get_fold_probabilities(
            train_df=train_df,
            eval_df=eval_df,
            features=features,
            draw_weight=draw_weight,
            use_sigmoid_calibration=use_sigmoid_calibration,
        )
        pooled_proba.append(fold_prob["proba"])
        pooled_true.append(fold_prob["y_true"])

    proba_all = np.vstack(pooled_proba)
    y_all = np.concatenate(pooled_true)
    t_home, t_draw, tuned_macro_f1 = optimize_thresholds(
        proba_all,
        y_all,
        min_recall={0: 0.20, 2: 0.20},
    )
    return {
        "t_home": float(t_home),
        "t_draw": float(t_draw),
        "pooled_val_macro_f1": float(tuned_macro_f1),
        "n_val_samples": int(len(y_all)),
    }


def _print_fold(before_metrics: dict, after_metrics: dict, after: dict, comparison: dict) -> None:
    print(
        f"BEFORE: logloss={before_metrics['log_loss']:.4f}  "
        f"brier={before_metrics['brier_score']:.4f}  "
        f"macro_f1={before_metrics['macro_f1']:.4f}"
    )
    print(f"        class_recall={before_metrics['class_wise_recall']}")
    print(
        f"AFTER : logloss={after_metrics['log_loss']:.4f}  "
        f"brier={after_metrics['brier_score']:.4f}  "
        f"macro_f1={after_metrics['macro_f1']:.4f}"
    )
    print(f"        class_recall={after_metrics['class_wise_recall']}")
    print(
        f"        tuned draw_weight={after['draw_weight']:.2f}"
        + (
            f"  thresholds=(home={after['thresholds']['t_home']:.3f}, draw={after['thresholds']['t_draw']:.3f})"
            if after["thresholds"] is not None
            else ""
        )
        + f"  sigmoid={after['sigmoid_calibrated']}"
    )
    print(
        "        deltas: "
        f"macro_f1={comparison['macro_f1_delta']:+.4f}  "
        f"logloss={comparison['log_loss_delta']:+.4f}  "
        f"draw_recall={comparison['draw_recall_delta']:+.4f}"
    )


def main():
    print("=== STABILIZE: rolling-origin backtest (2020-2025) ===")

    feat = run_feature_pipeline()
    match_df_clean = add_season_column(feat["match_df_clean"], date_col="Date_x")
    features = feat["FEATURES"]

    folds = build_stabilize_folds()
    validate_fold_coverage(match_df_clean, folds, season_col="Season")

    val_folds = [f for f in folds if f.mode == "val"]
    holdout_fold = next(f for f in folds if f.mode == "test")

    report = {"folds": [], "summary": {}, "selected_settings": {}}

    # 1) Validation folds: tune draw-weight loop + thresholds on val
    for fold in val_folds:
        train_df, eval_df = split_by_seasons(match_df_clean, fold, season_col="Season")
        before = run_baseline_fold(train_df, eval_df, features)
        after = tune_fold_on_val(train_df, eval_df, features)

        before_metrics = before["metrics"]
        after_metrics = after["metrics"]
        comparison = {
            "macro_f1_delta": _metric_delta(after_metrics["macro_f1"], before_metrics["macro_f1"]),
            "log_loss_delta": _metric_delta(after_metrics["log_loss"], before_metrics["log_loss"]),
            "brier_delta": _metric_delta(after_metrics["brier_score"], before_metrics["brier_score"]),
            "draw_recall_delta": _metric_delta(
                after_metrics["class_wise_recall"]["Draw"],
                before_metrics["class_wise_recall"]["Draw"],
            ),
        }

        fold_res = {
            "name": fold.name,
            "mode": fold.mode,
            "train_seasons": list(fold.train_seasons),
            "eval_season": fold.eval_season,
            "n_train": int(len(train_df)),
            "n_eval": int(len(eval_df)),
            "settings_source": "val_fold_tuning",
            "before": before,
            "after": after,
            "comparison": comparison,
        }
        report["folds"].append(fold_res)

        print(f"\n[{fold.name}] train={list(fold.train_seasons)} -> {fold.mode}={fold.eval_season}")
        _print_fold(before_metrics, after_metrics, after, comparison)

    # 2) Build global settings from validation only (safe for holdout)
    global_cfg = _select_global_model_config([f for f in report["folds"] if f["mode"] == "val"])
    global_thresholds = _build_global_thresholds(
        match_df_clean=match_df_clean,
        features=features,
        val_folds=val_folds,
        draw_weight=global_cfg["draw_weight"],
        use_sigmoid_calibration=global_cfg["use_sigmoid_calibration"],
    )
    report["selected_settings"]["global_from_validation"] = {
        "draw_weight": global_cfg["draw_weight"],
        "use_sigmoid_calibration": global_cfg["use_sigmoid_calibration"],
        "t_home": global_thresholds["t_home"],
        "t_draw": global_thresholds["t_draw"],
        "pooled_val_macro_f1": global_thresholds["pooled_val_macro_f1"],
        "n_val_samples": global_thresholds["n_val_samples"],
    }

    # 3) Final holdout: use only selected validation settings
    train_df, eval_df = split_by_seasons(match_df_clean, holdout_fold, season_col="Season")
    before = run_baseline_fold(train_df, eval_df, features)
    after = run_improved_with_fixed_settings(
        train_df=train_df,
        eval_df=eval_df,
        features=features,
        draw_weight=global_cfg["draw_weight"],
        t_home=global_thresholds["t_home"],
        t_draw=global_thresholds["t_draw"],
        use_sigmoid_calibration=global_cfg["use_sigmoid_calibration"],
    )

    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    comparison = {
        "macro_f1_delta": _metric_delta(after_metrics["macro_f1"], before_metrics["macro_f1"]),
        "log_loss_delta": _metric_delta(after_metrics["log_loss"], before_metrics["log_loss"]),
        "brier_delta": _metric_delta(after_metrics["brier_score"], before_metrics["brier_score"]),
        "draw_recall_delta": _metric_delta(
            after_metrics["class_wise_recall"]["Draw"],
            before_metrics["class_wise_recall"]["Draw"],
        ),
    }

    holdout_result = {
        "name": holdout_fold.name,
        "mode": holdout_fold.mode,
        "train_seasons": list(holdout_fold.train_seasons),
        "eval_season": holdout_fold.eval_season,
        "n_train": int(len(train_df)),
        "n_eval": int(len(eval_df)),
        "settings_source": "aggregated_from_validation",
        "before": before,
        "after": after,
        "comparison": comparison,
    }
    report["folds"].append(holdout_result)

    print(
        f"\n[{holdout_fold.name}] train={list(holdout_fold.train_seasons)}"
        f" -> {holdout_fold.mode}={holdout_fold.eval_season}"
    )
    _print_fold(before_metrics, after_metrics, after, comparison)

    val_results = [f for f in report["folds"] if f["mode"] == "val"]
    holdout = next(f for f in report["folds"] if f["mode"] == "test")
    report["summary"] = {
        "avg_val_macro_f1_before": float(np.mean([f["before"]["metrics"]["macro_f1"] for f in val_results])),
        "avg_val_macro_f1_after": float(np.mean([f["after"]["metrics"]["macro_f1"] for f in val_results])),
        "avg_val_log_loss_before": float(np.mean([f["before"]["metrics"]["log_loss"] for f in val_results])),
        "avg_val_log_loss_after": float(np.mean([f["after"]["metrics"]["log_loss"] for f in val_results])),
        "avg_val_draw_recall_before": float(
            np.mean([f["before"]["metrics"]["class_wise_recall"]["Draw"] for f in val_results])
        ),
        "avg_val_draw_recall_after": float(
            np.mean([f["after"]["metrics"]["class_wise_recall"]["Draw"] for f in val_results])
        ),
        "final_holdout_macro_f1_before": float(holdout["before"]["metrics"]["macro_f1"]),
        "final_holdout_macro_f1_after": float(holdout["after"]["metrics"]["macro_f1"]),
        "final_holdout_log_loss_before": float(holdout["before"]["metrics"]["log_loss"]),
        "final_holdout_log_loss_after": float(holdout["after"]["metrics"]["log_loss"]),
        "final_holdout_draw_recall_before": float(holdout["before"]["metrics"]["class_wise_recall"]["Draw"]),
        "final_holdout_draw_recall_after": float(holdout["after"]["metrics"]["class_wise_recall"]["Draw"]),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {REPORT_PATH}")
    print(f"Summary: {report['summary']}")
    print("Run complete.")


if __name__ == "__main__":
    main()
