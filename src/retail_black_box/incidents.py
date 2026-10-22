"""Detect full store outages; estimate expected unobserved sales (not lost revenue)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .data import store_day_coverage


def baseline_for_store_category(
    history: pd.DataFrame, store_id: str, category: str, date: pd.Timestamp, lookback_days: int = 42
) -> dict:
    """Only use observations strictly BEFORE the prediction date."""
    date = pd.Timestamp(date)
    selection = history.loc[
        (history["store_id"] == store_id)
        & (history["category"] == category)
        & (history["date"] < date)
        & (history["date"] >= date - pd.Timedelta(days=lookback_days))
        & (history["date"].dt.dayofweek == date.dayofweek)
        & history["amount_total"].notna(),
        "amount_total",
    ]
    method = "same_weekday_last_42d"
    if len(selection) < 2:
        selection = history.loc[
            (history["store_id"] == store_id)
            & (history["category"] == category)
            & (history["date"] < date)
            & (history["date"] >= date - pd.Timedelta(days=28))
            & history["amount_total"].notna(),
            "amount_total",
        ]
        method = "recent_28d_fallback"
    if len(selection) < 2:
        return {"estimate": np.nan, "p10": np.nan, "p90": np.nan, "support": len(selection), "method": "insufficient_history"}
    return {
        "estimate": float(selection.median()),
        "p10": float(selection.quantile(0.10)),
        "p90": float(selection.quantile(0.90)),
        "support": int(len(selection)),
        "method": method,
    }


def reconstruct_store_days(
    t: pd.DataFrame, missing: pd.DataFrame, categories: list[str] | None = None
) -> pd.DataFrame:
    """Reconstruct hypothetical *expected* sales, never assert sales really occurred."""
    cats = categories if categories is not None else sorted(t["category"].unique().tolist())
    # Pre-index once for efficient repeat lookups. Historical timeseries remain unchanged.
    by_pair = {(sid, cat): frame for (sid, cat), frame in t.groupby(["store_id", "category"], sort=False)}
    rows = []
    for item in missing[["store_id", "date"]].itertuples(index=False):
        for cat in cats:
            frame = by_pair.get((item.store_id, cat))
            if frame is None:
                continue
            result = baseline_for_store_category(frame, item.store_id, cat, pd.Timestamp(item.date))
            rows.append({"date": pd.Timestamp(item.date), "store_id": item.store_id, "category": cat, **result})
    return pd.DataFrame(rows, columns=["date", "store_id", "category", "estimate", "p10", "p90", "support", "method"])


def identify_incidents(t: pd.DataFrame, s: pd.DataFrame, c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = store_day_coverage(t, s, c)
    missing = grid.loc[grid["category_rows"] == 0, ["date", "store_id", "store_format", "region"]].copy()
    if missing.empty:
        return missing.assign(estimate_mxn=pd.Series(dtype=float)), pd.DataFrame()
    estimates = reconstruct_store_days(t, missing)
    per_day = estimates.groupby(["store_id", "date"], as_index=False).agg(
        estimate_mxn=("estimate", lambda v: v.sum(min_count=1)),
        p10_mxn=("p10", lambda v: v.sum(min_count=1)),
        p90_mxn=("p90", lambda v: v.sum(min_count=1)),
        categories_estimated=("estimate", "count"),
    )
    result = missing.merge(per_day, on=["store_id", "date"], how="left", validate="one_to_one")
    result["status"] = "DATA_COVERAGE_GAP"
    result["diagnosis"] = "All expected product categories missing; investigate POS/ingestion before inferring sales impact."
    return result.sort_values(["store_id", "date"]), estimates


def simulate_masked_day(t: pd.DataFrame, store_id: str, date: pd.Timestamp) -> pd.DataFrame:
    """Reconstruct a complete observed store-day using strictly earlier records."""
    day = t.loc[(t["store_id"] == store_id) & (t["date"] == date)]
    if day.empty:
        raise ValueError("Only observed store-days can be used for a masking experiment.")
    predicted = reconstruct_store_days(t, pd.DataFrame([{"store_id": store_id, "date": pd.Timestamp(date)}]), categories=sorted(day["category"].unique()))
    actual = day[["date", "store_id", "category", "amount_total"]].rename(columns={"amount_total": "actual"})
    return predicted.merge(actual, on=["date", "store_id", "category"], validate="one_to_one")
