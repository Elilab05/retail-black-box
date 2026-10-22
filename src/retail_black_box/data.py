"""Strict loading and uniqueness validation; input CSVs are immutable."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = {
    "transactions": {"date", "store_id", "category", "total_transactions", "cash_transactions", "card_transactions", "amount_total", "amount_cash", "amount_card", "avg_ticket"},
    "stores": {"store_id", "store_format", "region"},
    "calendar": {"date", "is_payday", "is_weekend", "is_navidad_season", "is_buen_fin", "is_semana_santa"},
}


def load_data(directory: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = Path(directory)
    frames = {}
    for name in ("transactions", "stores", "calendar"):
        path = p / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Copy the supplied challenge CSVs into data/.")
        frame = pd.read_csv(path, parse_dates=["date"] if name != "stores" else None)
        missing = REQUIRED[name] - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
        frames[name] = frame
    t, s, c = (frames[k] for k in ("transactions", "stores", "calendar"))
    for name, df, cols in [("transactions", t, ["date", "store_id", "category"]), ("stores", s, ["store_id"]), ("calendar", c, ["date"])]:
        if df.duplicated(cols).any():
            raise ValueError(f"Duplicate keys in {name}: {cols}; resolve before modeling.")
    if not t["store_id"].isin(s["store_id"]).all():
        raise ValueError("Transactions reference store IDs absent from stores.csv")
    if not t["date"].isin(c["date"]).all():
        raise ValueError("Transactions reference dates absent from calendar.csv")
    return t, s, c


def store_day_coverage(t: pd.DataFrame, s: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Expected store-date grid; zero rows do NOT mean zero sales."""
    expected = pd.MultiIndex.from_product(
        [s["store_id"].unique(), c["date"].sort_values().unique()], names=["store_id", "date"]
    )
    n = t.groupby(["store_id", "date"], observed=True).size().rename("category_rows")
    return n.reindex(expected, fill_value=0).reset_index().merge(
        s[["store_id", "store_format", "region"]], on="store_id", how="left", validate="many_to_one"
    )
