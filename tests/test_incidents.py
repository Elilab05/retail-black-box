import pandas as pd
import pytest
from retail_black_box.incidents import baseline_for_store_category, reconstruct_store_days, simulate_masked_day


def make_history():
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    return pd.DataFrame({"date": dates, "store_id": "A", "category": "Food", "amount_total": [float(100 + x) for x in range(70)]})


def test_reconstruction_never_reads_future_or_masked_values():
    d = make_history()
    date = pd.Timestamp("2024-02-12")
    base = baseline_for_store_category(d, "A", "Food", date)
    altered = d.copy()
    altered.loc[altered.date >= date, "amount_total"] = 99_999_999
    changed = baseline_for_store_category(altered, "A", "Food", date)
    assert base == changed
    assert base["support"] > 1


def test_reconstruction_insufficient_history_explicit():
    d = make_history()
    result = baseline_for_store_category(d, "A", "Food", pd.Timestamp("2023-12-15"))
    assert result["method"] == "insufficient_history"
    assert pd.isna(result["estimate"])


def test_single_day_replay_uses_known_values_only_as_labels():
    d = make_history()
    evaluated = simulate_masked_day(d, "A", pd.Timestamp("2024-02-15"))
    assert len(evaluated) == 1
    assert evaluated["actual"].iloc[0] > 0
    assert evaluated["estimate"].iloc[0] < evaluated["actual"].iloc[0]
    with pytest.raises(ValueError):
        simulate_masked_day(d, "OTHER", pd.Timestamp("2024-02-15"))


def test_batch_missing_rows_shape():
    d = make_history()
    selected = pd.DataFrame([{"date": pd.Timestamp("2024-02-15"), "store_id": "A"}])
    r = reconstruct_store_days(d, selected)
    assert len(r) == 1
    assert set(["p10", "p90", "support", "method"]).issubset(r.columns)
