import pandas as pd
from retail_black_box.forecast import prepare_supervised


def test_forecast_features_are_aligned_to_asof_and_future_target():
    days = pd.date_range("2024-01-01", periods=35, freq="D")
    t = pd.DataFrame({"date": days, "store_id": "A", "category": "Food", "amount_total": list(range(35)), "total_transactions": list(range(35))})
    s = pd.DataFrame({"store_id": ["A"], "store_format": ["Express"], "region": ["Centro"], "size_sqm": [100], "num_checkouts": [2]})
    c = pd.DataFrame({"date": days, "is_payday": False, "is_weekend": False, "is_navidad_season": False, "is_buen_fin": False, "is_semana_santa": False})
    panel = prepare_supervised(t, s, c)
    row = panel.loc[panel["date"] == pd.Timestamp("2024-01-16")].iloc[0]
    assert row["asof_amount"] == 15
    assert row["lag7_amount"] == 8
    assert row["target"] == 22
    assert row["target_date"] == pd.Timestamp("2024-01-23")
