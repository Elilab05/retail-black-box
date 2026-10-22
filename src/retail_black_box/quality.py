"""Explicit financial and record integrity checks, without automated data repair."""
from __future__ import annotations

import pandas as pd
from .data import store_day_coverage


def quality_flags(t: pd.DataFrame) -> pd.DataFrame:
    x = t[["date", "store_id", "category", "total_transactions", "cash_transactions", "card_transactions", "amount_total", "amount_cash", "amount_card", "avg_ticket"]].copy()
    known_count = x["cash_transactions"].notna() & x["card_transactions"].notna()
    x["transaction_count_mismatch"] = known_count & ((x["cash_transactions"] + x["card_transactions"] - x["total_transactions"]).abs() > 0.5)
    known_amount = x["amount_cash"].notna() & x["amount_card"].notna()
    x["amount_mismatch"] = known_amount & ((x["amount_cash"] + x["amount_card"] - x["amount_total"]).abs() > 1.0)
    known_ticket = x["avg_ticket"].notna() & (x["total_transactions"] > 0)
    x["ticket_mismatch"] = known_ticket & ((x["avg_ticket"] - x["amount_total"] / x["total_transactions"].where(x["total_transactions"] > 0)).abs() > 1.0)
    return x


def quality_report(t: pd.DataFrame, s: pd.DataFrame, c: pd.DataFrame) -> dict:
    flags = quality_flags(t)
    coverage = store_day_coverage(t, s, c)
    categories = t["category"].nunique()
    missing_store_days = coverage[coverage["category_rows"] == 0]
    absent_cells = (categories - coverage["category_rows"]).clip(lower=0).sum()
    return {
        "transactions_rows": int(len(t)),
        "stores": int(len(s)),
        "categories": int(categories),
        "calendar_days": int(len(c)),
        "expected_store_category_days": int(len(s) * len(c) * categories),
        "absent_store_category_days": int(absent_cells),
        "missing_whole_store_days": int(len(missing_store_days)),
        "partial_store_days": int(((coverage["category_rows"] > 0) & (coverage["category_rows"] < categories)).sum()),
        "duplicate_transaction_keys": int(t.duplicated(["date", "store_id", "category"]).sum()),
        "mismatched_transaction_count": int(flags["transaction_count_mismatch"].sum()),
        "mismatched_avg_ticket_gt_1_mxn": int(flags["ticket_mismatch"].sum()),
        "mismatched_payment_amount_gt_1_mxn": int(flags["amount_mismatch"].sum()),
        "null_cash_transaction_count": int(t["cash_transactions"].isna().sum()),
        "null_cash_amount_count": int(t["amount_cash"].isna().sum()),
        "null_avg_ticket_count": int(t["avg_ticket"].isna().sum()),
        "null_replenishment_signal_count": int(t["replenishment_signal"].isna().sum()) if "replenishment_signal" in t else None,
        "date_start": str(c["date"].min().date()),
        "date_end": str(c["date"].max().date()),
        "caveats": [
            "A missing store-day is not evidence of zero actual sales.",
            "Amount-payment reconciliation can pass even when count reconciliation fails.",
            "No causal or revenue-loss claim is justified by these checks alone.",
        ],
    }
