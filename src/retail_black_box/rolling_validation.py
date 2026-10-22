"""Expanding-window monthly backtests for the seven-day-ahead demand model.

No evaluation target occurs in training. At the start of each target month,
training labels stop eight days earlier, preceding the first test as-of date.
Each month has a newly fitted model and tests only dates inside that month.
"""
from __future__ import annotations

import pandas as pd
from .forecast import evaluate_7d, HORIZON


def summarize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Group per-period diagnostics from individual forecasts (no metric averaging)."""
    required = {"period", "actual", "predicted_ml", "predicted_seasonal_naive"}
    if missing := required - set(pred.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    rows = []
    for period, g in pred.groupby("period", sort=True):
        denom = g["actual"].abs().sum()
        actual_sum = g["actual"].sum()
        ml_err = (g["predicted_ml"] - g["actual"])
        base_err = (g["predicted_seasonal_naive"] - g["actual"])
        w_ml = float(ml_err.abs().sum() / denom) if denom > 0 else float("nan")
        w_base = float(base_err.abs().sum() / denom) if denom > 0 else float("nan")
        rows.append({
            "period": str(period), "n_test": len(g),
            "wape_ml": w_ml, "wape_baseline": w_base,
            "improvement_relative": (w_base - w_ml) / w_base if w_base > 0 else float("nan"),
            "bias_ml": float(ml_err.sum() / actual_sum) if actual_sum != 0 else float("nan"),
            "sum_abs_actual": float(denom),
            "sum_abs_error_ml": float(ml_err.abs().sum()),
            "sum_abs_error_baseline": float(base_err.abs().sum()),
        })
    return pd.DataFrame(rows)


def run_rolling_backtests(t: pd.DataFrame, s: pd.DataFrame, c: pd.DataFrame,
                          start: str = "2023-06", end: str = "2024-02") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one fixed-cutoff monthly backtest per month, without crossing train/test time."""
    periods = pd.period_range(start, end, freq="M")
    if len(periods) == 0:
        raise ValueError("Empty rolling period range")
    all_predictions = []
    for period in periods:
        first = period.start_time.normalize()
        last = period.end_time.normalize()
        if last > c["date"].max():
            raise ValueError(f"Evaluation period {period} exceeds available calendar")
        cutoff = first - pd.Timedelta(days=HORIZON + 1)
        # Slice through each month's end, so the original eval function
        # cannot accidentally test the later months.
        cal = c.loc[c["date"] <= last]
        transactions = t.loc[t["date"] <= last]
        _, frame = evaluate_7d(transactions, s, cal,
                                train_label_end=str(cutoff.date()),
                                test_start=str(first.date()))
        frame["period"] = str(period)
        all_predictions.append(frame)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_predictions(predictions)
    return summary, predictions
