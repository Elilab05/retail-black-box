"""Leakage-aware 7-day-ahead direct supervised regression vs seasonal naive.

Each prediction made at the END of asof_date is for asof_date + 7 days.
Training labels are cut off BEFORE the first evaluation as-of date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .experiments import wape

HORIZON = 7


def prepare_supervised(t: pd.DataFrame, s: pd.DataFrame, c: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    category_values = sorted(t["category"].unique())
    idx = pd.MultiIndex.from_product(
        [sorted(s["store_id"].unique()), category_values, sorted(c["date"].unique())],
        names=["store_id", "category", "date"],
    )
    panel = t.set_index(["store_id", "category", "date"])[["amount_total", "total_transactions"]].reindex(idx).reset_index()
    panel = panel.sort_values(["store_id", "category", "date"]).reset_index(drop=True)
    key = panel.groupby(["store_id", "category"], sort=False, observed=True)
    panel["target"] = key["amount_total"].shift(-horizon)
    panel["target_date"] = panel["date"] + pd.Timedelta(days=horizon)
    panel["asof_amount"] = panel["amount_total"]
    panel["lag7_amount"] = key["amount_total"].shift(7)
    panel["lag14_amount"] = key["amount_total"].shift(14)
    panel["rolling7_mean"] = key["amount_total"].transform(lambda x: x.rolling(7, min_periods=4).mean())
    panel["rolling28_mean"] = key["amount_total"].transform(lambda x: x.rolling(28, min_periods=14).mean())
    panel = panel.merge(s[["store_id", "store_format", "region", "size_sqm", "num_checkouts"]], on="store_id", validate="many_to_one")
    future_cal = c[["date", "is_payday", "is_weekend", "is_navidad_season", "is_buen_fin", "is_semana_santa"]].rename(columns={"date": "target_date"})
    panel = panel.merge(future_cal, on="target_date", how="left", validate="many_to_one")
    panel["target_weekday"] = panel["target_date"].dt.dayofweek
    panel["target_month"] = panel["target_date"].dt.month
    for col in ("store_id", "category", "store_format", "region"):
        panel[col] = panel[col].astype("category")
    return panel


FEATURES = [
    "store_id", "category", "store_format", "region", "size_sqm", "num_checkouts",
    "asof_amount", "lag7_amount", "lag14_amount", "rolling7_mean", "rolling28_mean",
    "is_payday", "is_weekend", "is_navidad_season", "is_buen_fin", "is_semana_santa",
    "target_weekday", "target_month",
]


def evaluate_7d(t: pd.DataFrame, s: pd.DataFrame, c: pd.DataFrame, train_label_end: str = "2024-01-24", test_start: str = "2024-02-01") -> tuple[dict, pd.DataFrame]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("Install ML extras: pip install -e '.[ml]'") from exc

    panel = prepare_supervised(t, s, c)
    train_label_end = pd.Timestamp(train_label_end)
    test_start = pd.Timestamp(test_start)
    earliest_test_asof = test_start - pd.Timedelta(days=HORIZON)
    if train_label_end >= earliest_test_asof:
        raise ValueError("Training label cutoff must PRECEDE earliest evaluation as-of day to prevent leakage")
    valid = panel["target"].notna() & panel["asof_amount"].notna() & panel["target_date"].notna()
    train = panel.loc[valid & (panel["target_date"] <= train_label_end)].copy()
    test = panel.loc[valid & (panel["target_date"] >= test_start) & (panel["target_date"] <= c["date"].max())].copy()
    if train.empty or test.empty:
        raise ValueError("Training or evaluation slice empty; inspect temporal cutoffs.")
    for col in ["is_payday", "is_weekend", "is_navidad_season", "is_buen_fin", "is_semana_santa"]:
        train[col] = train[col].astype(float)
        test[col] = test[col].astype(float)
    model = lgb.LGBMRegressor(n_estimators=220, learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=40, verbosity=-1, random_state=42, n_jobs=4)
    model.fit(train[FEATURES], train["target"], categorical_feature=["store_id", "category", "store_format", "region"])
    test["predicted_ml"] = np.maximum(model.predict(test[FEATURES]), 0)
    test["predicted_seasonal_naive"] = test["asof_amount"].clip(lower=0)
    result = test[["target_date", "date", "store_id", "category", "target", "predicted_ml", "predicted_seasonal_naive"]].rename(columns={"target_date": "date_target", "date": "date_asof", "target": "actual"}).copy()
    overall_ml = wape(result["actual"], result["predicted_ml"])
    overall_baseline = wape(result["actual"], result["predicted_seasonal_naive"])
    by_category = {}
    for cat, frame in result.groupby("category", observed=True):
        by_category[str(cat)] = {"wape_ml": wape(frame["actual"], frame["predicted_ml"]), "wape_seasonal_naive": wape(frame["actual"], frame["predicted_seasonal_naive"]), "rows": len(frame)}
    metrics = {
        "task": "7_day_ahead_daily_store_category_revenue_forecast",
        "horizon_days": HORIZON,
        "training_labels_available_through": str(train_label_end.date()),
        "first_evaluation_asof_date": str(earliest_test_asof.date()),
        "evaluation_target_start": str(test_start.date()),
        "evaluation_target_end": str(result["date_target"].max().date()),
        "n_train": len(train),
        "n_test": len(test),
        "wape_ml": overall_ml,
        "wape_seasonal_naive": overall_baseline,
        "relative_wape_improvement": (overall_baseline - overall_ml) / overall_baseline if overall_baseline else None,
        "model_wins": bool(overall_ml < overall_baseline),
        "per_category": by_category,
        "caveats": [
            "This is a fixed-cutoff historical backtest, not a future prediction.",
            "Features use as-of history and known future calendar flags, never future realized sales.",
            "Seasonal baseline uses observed sales exactly seven days before target.",
            "Target dates with missing actuals and as-of dates with missing sales are excluded.",
            "A single February holdout is not evidence of robust performance across seasons.",
        ],
    }
    return metrics, result
