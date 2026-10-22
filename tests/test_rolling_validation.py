import pandas as pd
import pytest
from retail_black_box.rolling_validation import summarize_predictions


def test_summary_aggregates_absolute_errors_and_signed_bias():
    predictions = pd.DataFrame({
        "period": ["2023-06", "2023-06", "2023-07"],
        "actual": [100., 200., 100.],
        "predicted_ml": [80., 240., 110.],
        "predicted_seasonal_naive": [50., 250., 80.],
    })
    out = summarize_predictions(predictions).set_index("period")
    assert out.loc["2023-06", "wape_ml"] == pytest.approx(60/300)
    assert out.loc["2023-06", "wape_baseline"] == pytest.approx(100/300)
    assert out.loc["2023-06", "bias_ml"] == pytest.approx(20/300)
    assert out.loc["2023-06", "improvement_relative"] == pytest.approx(.4)
    assert out.loc["2023-07", "n_test"] == 1


def test_summary_validates_schema():
    with pytest.raises(ValueError):
        summarize_predictions(pd.DataFrame({"period": ["2023-06"]}))
