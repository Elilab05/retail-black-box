# PROCESS — Technical challenge development record

## 1. Goal and selection rationale

The exercise allows the candidate to select any business problem. Chosen question:

**How can retail teams avoid acting on misleading sales signals when operational records are missing or inconsistent, and how accurately can the missing volume be estimated from earlier history?**

This leads to a deterministic coverage monitor, honest diagnostics, retrospective reconstruction experiments, and a separate forecasting baseline/ML comparison. A polished visualization without quantified experiments was deliberately not treated as sufficient.

## 2. Data understanding and audit

Read the three given CSV files and dictionary; preserve raw records. Confirm key uniqueness, time span, store/category count and the expected store-date-category grid. Reconcile payment methods at **count** level separately from **amount** level. Recompute ticket only on usable rows; tolerate MXN $1 to distinguish material inconsistencies from rounding. Flag missingness without replacing unknown with zero.

Found 42 missing tuples across 7 complete store-days (stores STR_045, STR_055, STR_064). Mismatch counts and results are computed programmatically via `retail-black-box quality` rather than hardcoded in the pipeline. These patterns call for validation of POS/ingestion; raw data alone do not establish the operational root cause.

## 3. Approach

**Outage estimates:** same-store, same-category, same-weekday historical median using the 42 days *strictly preceding* each target; fallback to 28-day trailing median when necessary. Save point estimate, reference sample count, and sample p10/p90. These are empirical dispersion summaries, not calibrated 80% confidence/prediction intervals.

**Offline evaluation:** randomly select 120 actually recorded complete store-days with at least 70 days prior history (fixed seed 42). Mask the target day *logically*, recompute using strictly earlier records, and compare to observed totals using WAPE/MAE. This tests estimation error on observed days; real outages may be nonrandom, so external validity is limited.

**Forecast:** global LightGBM predicts revenue of `(store, category, t+7)` from observations known at or before `t` and preannounced calendar flags for `t+7`. No as-of training record uses a label unavailable by the first evaluation as-of day. Compare with a seasonal naive prediction. Fixed training label cutoff: Jan 24, 2024; held-out target: February 2024. Future realized promotion, demand-derived replenishment signal, and actual future transaction counts are not included.

## 4. Execution and tests

```bash
python -m pip install -e '.[all]'
pytest -q
retail-black-box all --samples 120 --seed 42
streamlit run app.py
```

Results produced during initial run:

- 203,958 observed transactions; 80 stores, 6 categories, 425 calendar days.
- 42 absent store/category/date records in 7 missing store-days.
- 38,295 transaction-count mismatches and 39,786 avg-ticket differences above MXN $1; no amount discrepancies above MXN $1 when amount fields exist.
- Expected recorded volume for missing days: MXN $3,715,061.51 (not confirmed revenue loss).
- Reconstruction WAPE: 22.76% on 120 sampled complete store-days, seed 42.
- 7-day-ahead forecast: LightGBM WAPE 17.29% vs seasonal naive WAPE 24.58% on the February 2024 historical evaluation subset.

All numbers should be recalculated after modifying inputs or code. Generated reports under `outputs/` are excluded from public commits by default.

## 5. AI tooling declaration

**AI used: ChatGPT (OpenAI GPT-6), October 2026.** AI helped with solution brainstorming, architectural framing, implementation of Python utilities, tests, Streamlit interface, documentation and command-line validation. Computations were actually run against the challenge CSVs; reported metrics were taken from generated outputs rather than invented. AI-produced code requires candidate review and independent technical judgment before final submission. No outside scraping or undisclosed data source was used.

Candidate responsibilities before delivery: review data permissions, execute setup/tests in a clean environment, confirm the metrics and rationale, inspect code and GitHub history, and present known limitations clearly. Do not describe experiments not executed as completed.

## 6. Limitations and pending work

- Short 14-month history: only one main year-end season; limited generalization.
- Faults in payment counts cannot automatically be repaired using amounts.
- Potential observability bias: truly missing days may have a different data-generating process from randomly masked days.
- No verified inventory or actual lost-sales outcome; avoid direct financial impact claims.
- Empirical p10/p90 lack calibration; conformal calibration would require out-of-time validation.
- Single historical forecast holdout; assess rolling-origin evaluation and subgroup fairness before any production recommendation.
- Promotion flags may be unknown ahead of time; deliberately omitted from future-known feature set.
- Deployment, monitoring service, business-action A/B test and causal attribution are outside the current proof of concept.

## 7. Reproducibility and data handling

Source CSVs are not modified. CLI has deterministic sampling and fixed modeling seeds. GitHub Actions runs fixture-based tests without access to source CSVs. Raw files are ignored in Git history by default pending redistribution authorization; evaluators with the challenge package can restore them inside `data/`.

## 8. Added historical robustness audit (release 1.2)

Nine separate expanding-window evaluations cover June 2023 through February 2024. Every fold trains with labels strictly prior to its earliest evaluation as-of date. A new model is fitted before each target month and predicts the daily revenue seven days forward, using only history through each as-of date plus deterministic target-date calendar features. The complete rolling prediction records are kept for category/store-filtered WAPE and signed bias.

Across 131,484 evaluated rows, WAPE weighted over all absolute actual sales is 24.83% for LightGBM, 32.01% for seasonal naive. All nine months favor the model, but November (36.98% WAPE; about −29.3% signed bias) and January (54.20%; about +52.1% bias) expose severe seasonal regime sensitivity. These negative findings are disclosed in the app and release notes.

This is historical validation, not a guarantee of production accuracy; holiday transitions have only one year of representation. The fold results are reproducible with `retail-black-box rolling` and source CSVs.
