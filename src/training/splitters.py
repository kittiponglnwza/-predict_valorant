from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.features import get_season


@dataclass(frozen=True)
class TimeFold:
    name: str
    train_seasons: tuple[int, ...]
    eval_season: int
    mode: str  # "val" or "test"


def add_season_column(df: pd.DataFrame, date_col: str = "Date_x") -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    out = df.copy()
    out["Season"] = out[date_col].apply(get_season).astype("Int64")
    bad_rows = int(out["Season"].isna().sum())
    if bad_rows > 0:
        out = out.dropna(subset=["Season"]).copy()
        out["Season"] = out["Season"].astype(int)
    return out


def build_stabilize_folds() -> list[TimeFold]:
    # Phase STABILIZE plan (fixed by design)
    return [
        TimeFold("fold_1", (2020, 2021, 2022), 2023, "val"),
        TimeFold("fold_2", (2020, 2021, 2022, 2023), 2024, "val"),
        TimeFold("final_holdout", (2020, 2021, 2022, 2023, 2024), 2025, "test"),
    ]


def validate_fold_coverage(
    df: pd.DataFrame,
    folds: Iterable[TimeFold],
    season_col: str = "Season",
) -> None:
    available = set(df[season_col].dropna().astype(int).unique().tolist())
    for fold in folds:
        missing_train = [s for s in fold.train_seasons if s not in available]
        missing_eval = fold.eval_season not in available
        if missing_train:
            raise ValueError(f"{fold.name}: missing train seasons {missing_train}")
        if missing_eval:
            raise ValueError(f"{fold.name}: missing eval season {fold.eval_season}")


def split_by_seasons(
    df: pd.DataFrame,
    fold: TimeFold,
    season_col: str = "Season",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Time-safe split: no shuffling, strict season boundaries.
    train_df = df[df[season_col].isin(fold.train_seasons)].sort_values("Date_x").copy()
    eval_df = df[df[season_col] == fold.eval_season].sort_values("Date_x").copy()
    if train_df.empty or eval_df.empty:
        raise ValueError(f"{fold.name}: empty train/eval split.")
    return train_df, eval_df
