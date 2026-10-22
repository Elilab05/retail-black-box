"""Pure data transformations used by the dashboard; independently testable."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_wape(actual: pd.Series, predicted: pd.Series) -> float:
    """Weighted absolute percentage error, excluding missing pairs."""
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna()
    denom = float(a[valid].abs().sum())
    return float((a[valid] - p[valid]).abs().sum() / denom) if denom > 0 else float("nan")


def scope_incidents(
    incidents: pd.DataFrame,
    estimates_by_category: pd.DataFrame,
    store_ids: list[str],
    category: str | None = None,
) -> pd.DataFrame:
    """Recalculate incident financial proxy at selected scope (MXN, not millions).

    This estimates unobserved *volume*, not lost sales.
    """
    cols = ["date", "store_id", "region", "estimate_mxn", "categories_estimated", "status"]
    selected = incidents.loc[incidents["store_id"].isin(store_ids)].copy()
    if selected.empty:
        return pd.DataFrame(columns=cols)
    if category is not None:
        by_cat = estimates_by_category.loc[
            (estimates_by_category["store_id"].isin(store_ids)) &
            (estimates_by_category["category"] == category),
            ["date", "store_id", "estimate"],
        ].rename(columns={"estimate": "category_estimate"})
        selected = selected.drop(columns=["estimate_mxn"]).merge(
            by_cat, on=["date", "store_id"], how="left", validate="one_to_one"
        ).rename(columns={"category_estimate": "estimate_mxn"})
        selected["categories_estimated"] = selected["estimate_mxn"].notna().astype(int)
    return selected.sort_values("estimate_mxn", ascending=False).reset_index(drop=True)


def selection_quality_summary(filtered: pd.DataFrame, flags: pd.DataFrame) -> dict:
    """KPIs calculated on the same filtered row set."""
    return {
        "observed_rows": int(len(filtered)),
        "observed_amount_mxn": float(filtered["amount_total"].sum()),
        "mismatched_count": int(flags["transaction_count_mismatch"].sum()),
        "mismatched_ticket": int(flags["ticket_mismatch"].sum()),
        "mismatched_amount": int(flags["amount_mismatch"].sum()),
    }


def coverage_fraction(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Observed interval coverage; diagnostic only, not proof of calibration."""
    mask = actual.notna() & lower.notna() & upper.notna()
    if not mask.any():
        return float("nan")
    return float(((actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask])).mean())
