"""Mask-and-recover experiment: auditable estimates on observed historical data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .data import store_day_coverage
from .incidents import reconstruct_store_days


def wape(actual: pd.Series, prediction: pd.Series) -> float:
    valid = actual.notna() & prediction.notna()
    denominator = float(actual.loc[valid].abs().sum())
    return float((actual.loc[valid] - prediction.loc[valid]).abs().sum() / denominator) if denominator else float("nan")


def masked_outage_experiment(
    t: pd.DataFrame, s: pd.DataFrame, c: pd.DataFrame, n_days: int = 120, seed: int = 42
) -> tuple[dict, pd.DataFrame]:
    """Sample complete observed store-days and pretend they are absent.

    Strict past-only prediction means the masked values are never used for features.
    This evaluates estimation accuracy, NOT detection sensitivity for true outages.
    """
    coverage = store_day_coverage(t, s, c)
    earliest = pd.Timestamp(t["date"].min()) + pd.Timedelta(days=70)
    complete = coverage[(coverage["category_rows"] == t["category"].nunique()) & (coverage["date"] >= earliest)]
    if complete.empty:
        raise ValueError("No complete store-days suitable for masked outage evaluation.")
    rng = np.random.default_rng(seed)
    selected = complete.iloc[rng.choice(len(complete), size=min(n_days, len(complete)), replace=False)]
    estimated = reconstruct_store_days(t, selected[["store_id", "date"]])
    actual = t[["store_id", "date", "category", "amount_total"]].rename(columns={"amount_total": "actual"})
    result = estimated.merge(actual, on=["store_id", "date", "category"], how="left", validate="one_to_one")
    valid = result["actual"].notna() & result["estimate"].notna()
    covered = valid & result["actual"].between(result["p10"], result["p90"], inclusive="both")
    metrics = {
        "experiment": "historical_complete_store_days_masked_as_missing",
        "seed": seed,
        "requested_store_days": n_days,
        "sampled_store_days": int(len(selected)),
        "category_rows": int(len(result)),
        "evaluable_rows": int(valid.sum()),
        "wape": wape(result["actual"], result["estimate"]),
        "mae_mxn": float((result.loc[valid, "actual"] - result.loc[valid, "estimate"]).abs().mean()),
        "empirical_p10_p90_coverage": float(covered.sum() / valid.sum()) if valid.any() else None,
        "warning": "Empirical p10/p90 from 2-6 samples are NOT calibrated prediction intervals; this is a feasibility experiment.",
    }
    return metrics, result
