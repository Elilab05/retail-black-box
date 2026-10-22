import pandas as pd
from retail_black_box.quality import quality_report, quality_flags
from retail_black_box.data import store_day_coverage


def fixtures():
    days = pd.to_datetime(["2024-01-01", "2024-01-02"])
    t = pd.DataFrame([
        {"date": days[0], "store_id": "A", "category": "Beverage", "total_transactions": 10, "cash_transactions": 4, "card_transactions": 6, "amount_total": 100., "amount_cash": 40., "amount_card": 60., "avg_ticket": 10., "replenishment_signal": 1.},
        {"date": days[0], "store_id": "A", "category": "Food", "total_transactions": 10, "cash_transactions": 4, "card_transactions": 5, "amount_total": 100., "amount_cash": 40., "amount_card": 60., "avg_ticket": 11.5, "replenishment_signal": 1.},
    ])
    s = pd.DataFrame([{"store_id": "A", "store_format": "Express", "region": "Norte"}])
    c = pd.DataFrame({"date": days})
    return t, s, c


def test_coverage_absence_not_zero():
    t, s, c = fixtures()
    cov = store_day_coverage(t, s, c)
    assert cov.loc[cov.date == pd.Timestamp("2024-01-02"), "category_rows"].iloc[0] == 0
    r = quality_report(t, s, c)
    assert r["missing_whole_store_days"] == 1
    assert r["absent_store_category_days"] == 2
    assert r["mismatched_transaction_count"] == 1
    assert r["mismatched_avg_ticket_gt_1_mxn"] == 1
    assert r["mismatched_payment_amount_gt_1_mxn"] == 0


def test_flags_do_not_change_transaction_amount():
    t, _, _ = fixtures()
    original = t.copy(deep=True)
    q = quality_flags(t)
    assert int(q.transaction_count_mismatch.sum()) == 1
    pd.testing.assert_frame_equal(t, original)
