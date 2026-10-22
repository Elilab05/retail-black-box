"""Regression checks for past-only inputs in both analytical workflows."""
import pandas as pd
import pytest
from retail_black_box.forecast import prepare_supervised, FEATURES
from retail_black_box.incidents import baseline_for_store_category


def test_future_target_mutation_does_not_change_asof_features():
    days = pd.date_range('2024-01-01', periods=40, freq='D')
    t = pd.DataFrame({
        'date': days, 'store_id': 'S1', 'category': 'A',
        'amount_total': [float(i + 1) * 100 for i in range(40)],
        'total_transactions': [10] * 40,
    })
    s = pd.DataFrame({'store_id': ['S1'], 'store_format': ['Urban'],
                      'region': ['Centro'], 'size_sqm': [100], 'num_checkouts': [4]})
    c = pd.DataFrame({'date': days, 'is_payday': 0, 'is_weekend': 0,
                      'is_navidad_season': 0, 'is_buen_fin': 0, 'is_semana_santa': 0})
    before = prepare_supervised(t, s, c)
    prediction_date = pd.Timestamp('2024-01-20')
    target_day = prediction_date + pd.Timedelta(days=7)
    t.loc[t['date'] == target_day, 'amount_total'] = 1e10
    after = prepare_supervised(t, s, c)
    first = before.loc[before['date'] == prediction_date].iloc[0]
    second = after.loc[after['date'] == prediction_date].iloc[0]
    pd.testing.assert_series_equal(first[FEATURES], second[FEATURES], check_names=False)
    assert first['target'] != second['target']


def test_incident_baseline_ignores_target_and_future_records():
    days = pd.date_range('2024-01-01', periods=60, freq='D')
    t = pd.DataFrame({'date': days, 'store_id': 'S1', 'category': 'A',
                      'amount_total': [float(i + 1) * 100 for i in range(60)]})
    day = pd.Timestamp('2024-02-12')
    baseline = baseline_for_store_category(t, 'S1', 'A', day)
    t.loc[t['date'] >= day, 'amount_total'] = 1e10
    replay = baseline_for_store_category(t, 'S1', 'A', day)
    assert baseline['estimate'] == pytest.approx(replay['estimate'])
    assert baseline['p10'] == pytest.approx(replay['p10'])
    assert baseline['p90'] == pytest.approx(replay['p90'])
    assert baseline['support'] >= 2
