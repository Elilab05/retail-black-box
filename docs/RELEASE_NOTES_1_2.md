# Release 1.2 — Multi-period robustness validation

## Main change
Added an **expanding-window, nine-month backtest** from June 2023 through February 2024, with a newly fitted LightGBM model before each target month. The evaluation horizon remains seven days. Model labels stop eight days before the start of the target month so that the first test as-of date is strictly after the training label cutoff. The source CSVs remain unchanged.

## Results (chain-level, WAPE weighted by actual revenue)
- LightGBM: **24.83%**
- Seasonal naive: **32.01%**
- LightGBM beats the baseline in **all nine tested months**, but errors are severe around changing seasonal sales regimes.
- November 2023: WAPE **36.98%**, signed bias approximately **−29.3%**.
- January 2024: WAPE **54.20%**, signed bias approximately **+52.1%**.

These are **historical backtests**, not prospective predictions. One year-end holiday season is insufficient to establish stable holiday accuracy, and selected monthly folds share historical training data. The January/November performance warns against relying on the point prediction alone.

## Application changes
The Forecast page now displays a monthly WAPE comparison, filtered by store, region and category, plus monthly signed bias and a risk notice. The original February forecast and its 17.29% result remain unchanged. The monthly overall metric uses summed absolute error divided by summed absolute actual revenue, not the arithmetic average of monthly WAPEs.

## Reproduce

```bash
python -m pip install -e '.[all]'
retail-black-box rolling --start-month 2023-06 --end-month 2024-02
pytest -q
```

The command generates `outputs/rolling_predictions.csv` and `outputs/rolling_validation.csv`. Both precomputed files are bundled in the upload package so the Streamlit dashboard can render without training models during a page load.

**Provenance:** all values were computed with original challenge CSVs, lightgbm 4.6.0, fixed seed 42 and the existing model configuration. Prior model evaluations should be repeated in the candidate's submission environment for exact reproducibility.
