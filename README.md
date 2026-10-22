# Retail Black Box

**Live dashboard:** https://retail-black-box.streamlit.app/  
**Source:** https://github.com/Elilab05/retail-black-box

**Detect · Diagnose · Prioritize** — retail decision intelligence built on verifiable operational evidence.

> **Technical exercise:** multi-region Mexican retailer. The objective is not simply to predict demand, but to avoid treating missing sales records as true zero-revenue days and to quantify the reliability of a proposed diagnostic workflow.

## What it does

1. **Data Reliability:** validates complete coverage by `(date, store_id, category)`, cash/card reconciliation in counts and amounts, and average tickets. No silent imputation or dataset mutation.
2. **Incident Explorer:** detects full store-days with absent records across every category. Estimates the **sales volume one might have expected to observe**, without alleging business revenue loss.
3. **Replay & Stress Test:** hides an *observed* store-day and reconstructs it using only earlier dates. Measures error on known holdout values.
4. **Forecast Audit:** compares a seven-day-ahead global LightGBM revenue predictor against the same-weekday seasonal naive baseline using a leakage-aware historical split. Does **not** automatically invent future promotions.
5. **Dashboard:** interactive Streamlit + Plotly application with store, region and category filters; scoped headline KPIs and WAPE; historical forecasts; one-day replay; incident triage with historical evidence.

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

## Version 1.1 improvements

- Global KPIs replaced with filters-consistent observed volume, records, data integrity and expected volume proxy.
- Case-by-case incident investigation with temporal evidence, historical support and suggested verification steps.
- Forecast WAPE and relative improvement recalculated for selected store/region/category.
- Historical masking replay, transparent uncertainty caveats and scoped CSV exports.

See [`docs/RELEASE_NOTES_1_1.md`](docs/RELEASE_NOTES_1_1.md).

## Run locally

From this repository root, with Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -e '.[all]'
```

The three original challenge files are included in `data/`:

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

**Windows (PowerShell, without activating the virtual environment):**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m retail_black_box.cli all --samples 120 --seed 42
.\.venv\Scripts\python.exe -m retail_black_box.cli rolling --start-month 2023-06 --end-month 2024-02
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The nine-month rolling validation is a separate command from `all`; it takes longer because nine models are trained. `outputs/` must contain the generated forecast and rolling prediction files to display all sections. The datasets and outputs uploaded through GitHub's web UI may be tracked even though `.gitignore` excludes **new untracked** CSVs by default; check that the files are present in the repository after cloning. Do not interpret a `.gitignore` rule as proof that an already-committed file is private.

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

The published version used for the Streamlit demonstration includes the three **original** dataset CSV files supplied with the technical exercise to fulfill the requirement for a **public, complete, reproducible solution**. The original instructions did not explicitly grant redistribution rights or prohibit sharing the supplied files; this publication is a candidate judgment, **not** a claim of an express license. Source data is aggregated at store/category/day level and no direct personal identifiers appear in the supplied schema. Remove the data or request written authorization if the challenge owner states a restriction. Automated tests use synthetic fixtures.

## Repository outline

```text
retail-black-box/
├── app.py                       # Streamlit decision dashboard
├── src/retail_black_box/
│   ├── dashboard_metrics.py     # Filter-aware KPI / WAPE computations
│   ├── data.py                  # Strict ingest, full store-day coverage
│   ├── quality.py               # Auditable financial/data integrity
│   ├── incidents.py             # Past-only outage estimates
│   ├── experiments.py           # Masked-day backtest
│   ├── forecast.py              # Direct 7-day-ahead ML vs baseline
│   ├── rolling_validation.py    # Expanding-window monthly audits
│   └── cli.py                   # Reproducible entry point
├── tests/                       # Synthetic unit tests
├── docs/                        # Source dictionary and choices
├── .github/workflows/tests.yml # Dataset-free CI
├── PROCESS.md                   # Mandatory methodology / AI disclosure
├── data/                        # Original challenge CSVs (included)
└── outputs/                     # Reproducible reports and predictions
```


## Version 1.2 — Seasonal robustness (nine backtests)

An expanding-window backtest was conducted for every month from June 2023 through February 2024, updating the model before each month and forecasting seven days ahead. Over all **131,484** evaluated store/category/day predictions, LightGBM achieved **24.83% WAPE** versus **32.01%** for the seasonal-naive baseline (weighted over all actual values). It outperformed the baseline in all nine months, but had substantial forecast errors and signed bias during November and January, a limitation now surfaced in the dashboard.

To regenerate the complete monthly evidence, run `retail-black-box rolling --start-month 2023-06 --end-month 2024-02` and then restart Streamlit. See [`docs/RELEASE_NOTES_1_2.md`](docs/RELEASE_NOTES_1_2.md).

## Public GitHub publication

For publishing subsequent revisions after confirming the treatment of source CSVs:

```bash
git init -b main
git add .
git commit -m "Initial retail incident intelligence proof of concept"
gh auth login                  # Only if not logged into GitHub CLI
# gh repo create retail-black-box --public --source=. --remote=origin --push
```

The `gh repo create` line is intentionally commented: publishing a public repository is an explicit action requiring your GitHub account and review. The original challenge deadline is **48 hours from receipt**, not from repository creation.

Alternatively, when `gh` is installed and authenticated, run `bash scripts/publish_github.sh` to confirm explicit public publication. Confirm the data rights and repository name before proceeding.


## Submission-quality notes (final review)

- February category metrics displayed below the monthly robustness section **refer only to February 2024**, not the aggregate of all nine months.
- Twelve original automated tests and two additional temporal guardrail tests are available after applying the audit patch.
- The masked-day reconstruction check measures **estimation error**, not incident detection precision/recall. Incident root causes and verified revenue losses are unknown.
- The complete project was checked by recreating derived reports from the supplied CSVs. This does not replace an external clean-environment check from a fresh GitHub clone.
