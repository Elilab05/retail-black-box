import numpy as np
import pandas as pd

from retail_black_box.dashboard_metrics import compute_wape, scope_incidents, selection_quality_summary


def test_wape_recomputes_after_filter():
    actual = pd.Series([100, 100, 0])
    predicted = pd.Series([90, 60, 10])
    assert compute_wape(actual, predicted) == 0.30
    assert compute_wape(actual.iloc[:1], predicted.iloc[:1]) == 0.10
    assert np.isnan(compute_wape(pd.Series([0]), pd.Series([5])))


def test_incident_scoping_by_store_and_category():
    incidents = pd.DataFrame({
        "date": pd.to_datetime(["2024-02-10", "2024-02-11"]),
        "store_id": ["S1", "S2"], "region": ["N", "S"],
        "estimate_mxn": [300, 800], "categories_estimated": [2, 2], "status": ["GAP", "GAP"]
    })
    detail = pd.DataFrame({
        "date": pd.to_datetime(["2024-02-10", "2024-02-10", "2024-02-11"]),
        "store_id": ["S1", "S1", "S2"], "category": ["A", "B", "A"], "estimate": [100, 200, 800]
    })
    all_s1 = scope_incidents(incidents, detail, ["S1"])
    assert all_s1["estimate_mxn"].sum() == 300
    cat_s1 = scope_incidents(incidents, detail, ["S1"], "A")
    assert cat_s1["estimate_mxn"].sum() == 100
    assert int(cat_s1["categories_estimated"].iloc[0]) == 1
    assert scope_incidents(incidents, detail, ["S3"]).empty


def test_quality_summary_is_scoped_to_input():
    df = pd.DataFrame({"amount_total": [20, 30]})
    flags = pd.DataFrame({"transaction_count_mismatch": [True, False], "ticket_mismatch": [False, True], "amount_mismatch": [False, False]})
    assert selection_quality_summary(df, flags) == {
        "observed_rows": 2, "observed_amount_mxn": 50., "mismatched_count": 1, "mismatched_ticket": 1, "mismatched_amount": 0,
    }
