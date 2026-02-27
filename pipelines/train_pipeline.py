from __future__ import annotations

import json
import os
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
SELECTION_METRIC = os.getenv("STABILIZE_SELECTION_METRIC", "accuracy").strip().lower()
MIN_RECALL_POLICY = {
    0: float(os.getenv("STABILIZE_MIN_RECALL_AWAY", "0.08")),
    1: float(os.getenv("STABILIZE_MIN_RECALL_DRAW", "0.10")),
    2: float(os.getenv("STABILIZE_MIN_RECALL_HOME", "0.08")),
}
RECOMMENDED_MIN_SEASONS = int(os.getenv("MIN_RECOMMENDED_SEASONS", "8"))


def _metric_delta(after: float, before: float) -> float:
    return float(after - before)


def _set_market_mode(use_market_features: bool) -> None:
    import src.config as cfg
    import src.features as feat_mod

    cfg.USE_MARKET_FEATURES = bool(use_market_features)
    feat_mod.USE_MARKET_FEATURES = bool(use_market_features)


def _run_feature_pipeline_for_mode(use_market_features: bool):
    _set_market_mode(use_market_features)
    feat = run_feature_pipeline()
    feat["requested_market_mode"] = bool(use_market_features)
    return feat


def _history_status(match_df_clean) -> dict:
    seasons = sorted(match_df_clean["Season"].dropna().astype(int).unique().tolist())
    return {
        "seasons": seasons,
        "n_seasons": int(len(seasons)),
        "recommended_min_seasons": int(RECOMMENDED_MIN_SEASONS),
        "meets_recommendation": bool(len(seasons) >= RECOMMENDED_MIN_SEASONS),
    }


def _select_global_model_config(val_results: list[dict], selection_metric: str) -> dict:
    by_cfg: dict[tuple[float, bool], dict[str, list[float]]] = {}
    for fold_res in val_results:
        trials = fold_res["after"]["draw_weight_trials"]
        for row in trials:
            dw = float(row["draw_weight"])
            use_sigmoid = bool(row.get("use_sigmoid_calibration", False))
            key = (dw, use_sigmoid)
            by_cfg.setdefault(key, {"selection": [], "macro_f1": [], "log_loss": [], "accuracy": []})
            by_cfg[key]["selection"].append(float(row["selection_score"]))
            by_cfg[key]["accuracy"].append(float(row.get("accuracy", 0.0)))
            by_cfg[key]["macro_f1"].append(float(row["macro_f1"]))
            by_cfg[key]["log_loss"].append(float(row["log_loss"]))

    best_dw = 1.0
    best_sigmoid = True
    best_selection = -1.0
    best_macro = -1.0
    best_ll = float("inf")
    best_acc = -1.0

    for (dw, use_sigmoid), stats in sorted(by_cfg.items()):
        mean_selection = float(np.mean(stats["selection"]))
        mean_macro = float(np.mean(stats["macro_f1"]))
        mean_ll = float(np.mean(stats["log_loss"]))
        mean_acc = float(np.mean(stats["accuracy"]))

        if mean_selection > best_selection:
            best_dw = dw
            best_sigmoid = use_sigmoid
            best_selection = mean_selection
            best_macro = mean_macro
            best_ll = mean_ll
            best_acc = mean_acc
        elif abs(mean_selection - best_selection) < 1e-12 and mean_macro > best_macro:
            best_dw = dw
            best_sigmoid = use_sigmoid
            best_selection = mean_selection
            best_macro = mean_macro
            best_ll = mean_ll
            best_acc = mean_acc
        elif (
            abs(mean_selection - best_selection) < 1e-12
            and abs(mean_macro - best_macro) < 1e-12
            and mean_ll < best_ll
        ):
            best_dw = dw
            best_sigmoid = use_sigmoid
            best_selection = mean_selection
            best_macro = mean_macro
            best_ll = mean_ll
            best_acc = mean_acc

    return {
        "draw_weight": float(best_dw),
        "use_sigmoid_calibration": bool(best_sigmoid),
        "selection_metric": selection_metric,
        "mean_selection_score": float(best_selection),
        "mean_accuracy": float(best_acc),
    }


def _select_global_blend_weights(val_results: list[dict]) -> dict[str, float]:
    source_totals: dict[str, float] = {}
    source_counts: dict[str, float] = {}
    for fold_res in val_results:
        after = fold_res["after"]
        w = after.get("blend_weights", {})
        weight_factor = float(fold_res.get("n_eval", 1))
        for name, val in w.items():
            source_totals[name] = source_totals.get(name, 0.0) + float(val) * weight_factor
            source_counts[name] = source_counts.get(name, 0.0) + weight_factor

    if not source_totals:
        return {}

    avg = {k: source_totals[k] / max(source_counts.get(k, 1.0), 1e-12) for k in source_totals}
    total = sum(max(v, 0.0) for v in avg.values())
    if total <= 0:
        eq = 1.0 / len(avg)
        return {k: eq for k in sorted(avg)}
    return {k: float(max(v, 0.0) / total) for k, v in sorted(avg.items())}


def _build_global_thresholds(
    match_df_clean,
    features,
    val_folds,
    draw_weight: float,
    use_sigmoid_calibration: bool,
    blend_weights: dict[str, float],
    selection_metric: str,
    min_recall: dict[int, float],
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
            blend_weights=blend_weights,
        )
        pooled_proba.append(fold_prob["proba"])
        pooled_true.append(fold_prob["y_true"])

    proba_all = np.vstack(pooled_proba)
    y_all = np.concatenate(pooled_true)
    t_home, t_draw, tuned_score = optimize_thresholds(
        proba_all,
        y_all,
        t_home_range=(0.35, 0.65),
        t_draw_range=(0.12, 0.45),
        min_recall=min_recall,
        objective=selection_metric,
    )
    return {
        "t_home": float(t_home),
        "t_draw": float(t_draw),
        "pooled_val_selection_score": float(tuned_score),
        "selection_metric": selection_metric,
        "n_val_samples": int(len(y_all)),
    }


def _print_fold(before_metrics: dict, after_metrics: dict, after: dict, comparison: dict) -> None:
    print(
        f"BEFORE: acc={before_metrics['accuracy']:.4f}  "
        f"logloss={before_metrics['log_loss']:.4f}  "
        f"brier={before_metrics['brier_score']:.4f}  "
        f"macro_f1={before_metrics['macro_f1']:.4f}"
    )
    print(f"        class_recall={before_metrics['class_wise_recall']}")
    print(
        f"AFTER : acc={after_metrics['accuracy']:.4f}  "
        f"logloss={after_metrics['log_loss']:.4f}  "
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
        + f"  blend={after.get('blend_weights', {})}"
    )
    print(
        "        deltas: "
        f"acc={comparison['accuracy_delta']:+.4f}  "
        f"macro_f1={comparison['macro_f1_delta']:+.4f}  "
        f"logloss={comparison['log_loss_delta']:+.4f}  "
        f"draw_recall={comparison['draw_recall_delta']:+.4f}"
    )


def _run_profile(feat: dict, profile_name: str) -> dict:
    match_df_clean = add_season_column(feat["match_df_clean"], date_col="Date_x")
    features = feat["FEATURES"]
    history = _history_status(match_df_clean)

    folds = build_stabilize_folds()
    validate_fold_coverage(match_df_clean, folds, season_col="Season")
    val_folds = [f for f in folds if f.mode == "val"]
    holdout_fold = next(f for f in folds if f.mode == "test")

    report = {
        "profile": {
            "name": profile_name,
            "requested_market_mode": bool(feat.get("requested_market_mode", False)),
            "odds_available": bool(feat.get("ODDS_AVAILABLE", False)),
            "n_features": int(len(features)),
            "selection_metric": SELECTION_METRIC,
        },
        "history": history,
        "folds": [],
        "summary": {},
        "selected_settings": {},
    }

    for fold in val_folds:
        train_df, eval_df = split_by_seasons(match_df_clean, fold, season_col="Season")
        before = run_baseline_fold(train_df, eval_df, features)
        after = tune_fold_on_val(
            train_df=train_df,
            eval_df=eval_df,
            features=features,
            selection_metric=SELECTION_METRIC,
            min_recall=MIN_RECALL_POLICY,
        )

        before_metrics = before["metrics"]
        after_metrics = after["metrics"]
        comparison = {
            "accuracy_delta": _metric_delta(after_metrics["accuracy"], before_metrics["accuracy"]),
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

        print(f"\n[{profile_name}/{fold.name}] train={list(fold.train_seasons)} -> {fold.mode}={fold.eval_season}")
        _print_fold(before_metrics, after_metrics, after, comparison)

    val_results = [f for f in report["folds"] if f["mode"] == "val"]
    global_cfg = _select_global_model_config(val_results=val_results, selection_metric=SELECTION_METRIC)
    global_blend = _select_global_blend_weights(val_results)
    global_thresholds = _build_global_thresholds(
        match_df_clean=match_df_clean,
        features=features,
        val_folds=val_folds,
        draw_weight=global_cfg["draw_weight"],
        use_sigmoid_calibration=global_cfg["use_sigmoid_calibration"],
        blend_weights=global_blend,
        selection_metric=SELECTION_METRIC,
        min_recall=MIN_RECALL_POLICY,
    )
    report["selected_settings"]["global_from_validation"] = {
        "draw_weight": global_cfg["draw_weight"],
        "use_sigmoid_calibration": global_cfg["use_sigmoid_calibration"],
        "blend_weights": global_blend,
        "t_home": global_thresholds["t_home"],
        "t_draw": global_thresholds["t_draw"],
        "selection_metric": global_thresholds["selection_metric"],
        "pooled_val_selection_score": global_thresholds["pooled_val_selection_score"],
        "n_val_samples": global_thresholds["n_val_samples"],
    }

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
        blend_weights=global_blend,
    )

    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    comparison = {
        "accuracy_delta": _metric_delta(after_metrics["accuracy"], before_metrics["accuracy"]),
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
        f"\n[{profile_name}/{holdout_fold.name}] train={list(holdout_fold.train_seasons)}"
        f" -> {holdout_fold.mode}={holdout_fold.eval_season}"
    )
    _print_fold(before_metrics, after_metrics, after, comparison)

    holdout = next(f for f in report["folds"] if f["mode"] == "test")
    report["summary"] = {
        "avg_val_accuracy_before": float(np.mean([f["before"]["metrics"]["accuracy"] for f in val_results])),
        "avg_val_accuracy_after": float(np.mean([f["after"]["metrics"]["accuracy"] for f in val_results])),
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
        "final_holdout_accuracy_before": float(holdout["before"]["metrics"]["accuracy"]),
        "final_holdout_accuracy_after": float(holdout["after"]["metrics"]["accuracy"]),
        "final_holdout_macro_f1_before": float(holdout["before"]["metrics"]["macro_f1"]),
        "final_holdout_macro_f1_after": float(holdout["after"]["metrics"]["macro_f1"]),
        "final_holdout_log_loss_before": float(holdout["before"]["metrics"]["log_loss"]),
        "final_holdout_log_loss_after": float(holdout["after"]["metrics"]["log_loss"]),
        "final_holdout_draw_recall_before": float(holdout["before"]["metrics"]["class_wise_recall"]["Draw"]),
        "final_holdout_draw_recall_after": float(holdout["after"]["metrics"]["class_wise_recall"]["Draw"]),
        "selection_metric": SELECTION_METRIC,
    }

    report["metric_views"] = {
        "decision_quality": {
            "metric": "accuracy",
            "avg_val_before": report["summary"]["avg_val_accuracy_before"],
            "avg_val_after": report["summary"]["avg_val_accuracy_after"],
            "holdout_before": report["summary"]["final_holdout_accuracy_before"],
            "holdout_after": report["summary"]["final_holdout_accuracy_after"],
        },
        "balance_quality": {
            "metric": "macro_f1",
            "avg_val_before": report["summary"]["avg_val_macro_f1_before"],
            "avg_val_after": report["summary"]["avg_val_macro_f1_after"],
            "holdout_before": report["summary"]["final_holdout_macro_f1_before"],
            "holdout_after": report["summary"]["final_holdout_macro_f1_after"],
        },
        "probability_quality": {
            "metric": "log_loss",
            "avg_val_before": report["summary"]["avg_val_log_loss_before"],
            "avg_val_after": report["summary"]["avg_val_log_loss_after"],
            "holdout_before": report["summary"]["final_holdout_log_loss_before"],
            "holdout_after": report["summary"]["final_holdout_log_loss_after"],
        },
    }
    return report


def _is_market_profile_usable(feat: dict) -> bool:
    if not bool(feat.get("ODDS_AVAILABLE", False)):
        return False
    return any(str(col).startswith("Mkt_") for col in feat.get("FEATURES", []))


def main():
    print("=== STABILIZE: rolling-origin backtest (2020-2025) ===")
    print(f"Selection metric: {SELECTION_METRIC}  (min_recall={MIN_RECALL_POLICY})")

    profiles: dict[str, dict] = {}

    print("\n--- Profile: no_market ---")
    feat_no_market = _run_feature_pipeline_for_mode(use_market_features=False)
    profiles["no_market"] = _run_profile(feat_no_market, profile_name="no_market")

    print("\n--- Profile: with_market ---")
    feat_with_market = _run_feature_pipeline_for_mode(use_market_features=True)
    if _is_market_profile_usable(feat_with_market):
        profiles["with_market"] = _run_profile(feat_with_market, profile_name="with_market")
    else:
        print("Market profile skipped: odds columns/features unavailable.")

    selected_name = max(
        profiles.keys(),
        key=lambda name: float(profiles[name]["summary"]["avg_val_accuracy_after"]),
    )
    selected = profiles[selected_name]

    profile_comparison = {}
    for name, rep in profiles.items():
        sm = rep["summary"]
        profile_comparison[name] = {
            "avg_val_accuracy_after": sm["avg_val_accuracy_after"],
            "avg_val_macro_f1_after": sm["avg_val_macro_f1_after"],
            "avg_val_log_loss_after": sm["avg_val_log_loss_after"],
            "requested_market_mode": rep["profile"]["requested_market_mode"],
            "odds_available": rep["profile"]["odds_available"],
            "n_features": rep["profile"]["n_features"],
        }

    final_report = {
        "selected_profile": selected_name,
        "profile_comparison": profile_comparison,
        "profile": selected["profile"],
        "history": selected["history"],
        "folds": selected["folds"],
        "summary": selected["summary"],
        "metric_views": selected["metric_views"],
        "selected_settings": selected["selected_settings"],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"\nSelected profile: {selected_name}")
    print(f"Report saved: {REPORT_PATH}")
    print(f"Summary: {final_report['summary']}")
    print("Run complete.")


if __name__ == "__main__":
    main()
