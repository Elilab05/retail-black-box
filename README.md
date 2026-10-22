# Retail Black Box

**Detect · Diagnose · Prioritize** — retail decision intelligence built on verifiable operational evidence.

> **Technical exercise:** multi-region Mexican retailer. The objective is not simply to predict demand, but to avoid treating missing sales records as true zero-revenue days and to quantify the reliability of a proposed diagnostic workflow.

## What it does

1. **Data Reliability:** validates complete coverage by `(date, store_id, category)`, cash/card reconciliation in counts and amounts, and average tickets. No silent imputation or dataset mutation.
2. **Incident Explorer:** detects full store-days with absent records across every category. Estimates the **sales volume one might have expected to observe**, without alleging business revenue loss.
3. **Replay & Stress Test:** hides an *observed* store-day and reconstructs it using only earlier dates. Measures error on known holdout values.
4. **Forecast Audit:** compares a seven-day-ahead global LightGBM revenue predictor against the same-weekday seasonal naive baseline using a leakage-aware historical split. Does **not** automatically invent future promotions.
5. **Dashboard:** interactive Streamlit + Plotly application with store, region and category filters; historical forecast curves; one-day experiment and diagnostics.

## Results reproduced on the supplied dataset

| Check / test | Observed result |
|---|---:|
| Transaction records (store × category × day) | 203,958 |
| Stores / categories / calendar days | 80 / 6 / 425 |
| Missing `(date, store, category)` tuples | 42 |
| Completely absent store-days | 7 (3 stores) |
| Mismatches: count `cash + card != total` (when both observed) | 38,295 |
| Ticket average differences exceeding MXN $1 | 39,786 |
| Cash/card amount differences exceeding MXN $1 (where both available) | 0 |
| Estimated **unobserved** volume on 7 missing store-days | MXN $3,715,061.51 |
| Masked-outage experiment: 120 complete store-days, seed 42 | WAPE **22.76%** |
| Forecast LightGBM: 7-day-ahead WAPE, Feb 2024 | **17.29%** |
| Forecast seasonal naive: identical evaluation subset | **24.58%** |

**Interpretation:** estimated unobserved volume is not confirmed lost revenue. Forecast improvement (~29.7% relative WAPE reduction) is a historical single-month result, not a production SLA. Short historical quantiles `p10/p90` are not calibrated prediction intervals. Compare experiments with source outputs under `outputs/`.

## Run locally

From this repository root, with Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -e '.[all]'
```

Copy **the three original challenge files** into `data/`:

```
data/transactions.csv
data/stores.csv
data/calendar.csv
```

Then execute:

```bash
pytest -q
retail-black-box all --samples 120 --seed 42
streamlit run app.py
```

Output files appear under `outputs/`:

- `quality_report.json`
- `incidents.csv`, `incident_estimates_by_category.csv`
- `backtest_report.json`, `backtest_rows.csv`
- `forecast_report.json`, `forecast_predictions.csv`

For a lightweight first run without LightGBM: `retail-black-box all --skip-ml`. On macOS, if LightGBM reports an OpenMP library error, install OpenMP using `brew install libomp`, then retry.

## Experiment validity

- Unit of analysis: sales per store, category, calendar day.
- Outage definition: **all 6 expected categories absent** for a store-day; a partial-category gap is separately counted.
- Reconstructed value: median of the same weekday over the **preceding** 42 days (fallback: recent 28-day median when fewer than 2 same-weekday samples). Robust but relatively simple.
- Sampled masking: deterministic RNG seed **42**, selects 120 originally complete store-days after a 70-day warm-up; estimate only from data prior to each masked day.
- Forecasting: direct target `amount_total(t + 7)` using store/category attributes, observed sales/features no later than `t`, and deterministic calendar features for `t + 7`. Training labels through **2024-01-24**, first test as-of **2024-01-25**, evaluation target **February 2024**. Seasonal-naive baseline uses `amount_total(t)` for target `t + 7`.
- Outage detection itself is a deterministic coverage check. The masking experiment measures **reconstruction error**, not true outage detection precision/recall.
- `replenishment_signal` is deliberately excluded from modeling: the dictionary describes it as derived from observed demand, making its real-world feature availability ambiguous.

## Design choices & known limitations

See [`PROCESS.md`](PROCESS.md) for methodology, AI assistance disclosure and workflow and [`docs/DECISIONS.md`](docs/DECISIONS.md) for interpretation, risk and follow-up work.

This repository **does not publish the provided CSV datasets by default**: `.gitignore` excludes `data/*.csv`, because redistribution rights were not specified. The handoff archive includes the files for local execution. If publishing the raw dataset is explicitly authorized by the challenge owner, remove the ignore rule and document the authorization. Tests use synthetic fixtures, so CI runs without source data.

## Repository outline

```text
retail-black-box/
├── app.py                       # Streamlit dashboard
├── src/retail_black_box/
│   ├── data.py                  # Strict ingest, full store-day coverage
│   ├── quality.py               # Auditable financial/data integrity
│   ├── incidents.py             # Past-only outage estimates
│   ├── experiments.py           # Masked-day backtest
│   ├── forecast.py              # Direct 7-day-ahead ML vs baseline
│   └── cli.py                   # Reproducible entry point
├── tests/                       # Synthetic unit tests
├── docs/                        # Source dictionary and choices
├── .github/workflows/tests.yml # Dataset-free CI
├── PROCESS.md                   # Mandatory methodology / AI disclosure
└── data/                        # Supplied CSVs (not committed by default)
```

## Public GitHub publication

After checking the code and approving whether source CSVs may be shared:

```bash
git init -b main
git add .
git commit -m "Initial retail incident intelligence proof of concept"
gh auth login                  # Only if not logged into GitHub CLI
# gh repo create retail-black-box --public --source=. --remote=origin --push
```

The `gh repo create` line is intentionally commented: publishing a public repository is an explicit action requiring your GitHub account and review. The original challenge deadline is **48 hours from receipt**, not from repository creation.

Alternatively, when `gh` is installed and authenticated, run `bash scripts/publish_github.sh` to confirm explicit public publication. Confirm the data rights and repository name before proceeding.
