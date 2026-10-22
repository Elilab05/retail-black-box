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

All numbers should be recalculated after modifying inputs or code. Generated outputs are reproducible, and precomputed files were uploaded for the public Streamlit demo. `.gitignore` only affects newly untracked files; verify tracked artifacts after cloning.

## 5. AI tooling declaration

**AI used: Was used as a supporting tool during the development process, assisting with technical exploration, implementation, and refinement of the solution.

## 6. Limitations and pending work

- Short 14-month history: only one main year-end season; limited generalization.
- Faults in payment counts cannot automatically be repaired using amounts.
- Potential observability bias: truly missing days may have a different data-generating process from randomly masked days.
- No verified inventory or actual lost-sales outcome; avoid direct financial impact claims.
- Empirical p10/p90 lack calibration; conformal calibration would require out-of-time validation.
- Nine historical monthly expanding-window backtests were run (June 2023–February 2024). Despite winning all nine months against the selected baseline, the model has large errors at seasonal transitions. More years and independent evaluations are needed before a production recommendation.
- Promotion flags may be unknown ahead of time; deliberately omitted from future-known feature set.
- Deployment, monitoring service, business-action A/B test and causal attribution are outside the current proof of concept.


## 7. Added historical robustness audit (release 1.2)

Nine separate expanding-window evaluations cover June 2023 through February 2024. Every fold trains with labels strictly prior to its earliest evaluation as-of date. A new model is fitted before each target month and predicts the daily revenue seven days forward, using only history through each as-of date plus deterministic target-date calendar features. The complete rolling prediction records are kept for category/store-filtered WAPE and signed bias.

Across 131,484 evaluated rows, WAPE weighted over all absolute actual sales is 24.83% for LightGBM, 32.01% for seasonal naive. All nine months favor the model, but November (36.98% WAPE; about −29.3% signed bias) and January (54.20%; about +52.1% bias) expose severe seasonal regime sensitivity. These negative findings are disclosed in the app and release notes.

This is historical validation, not a guarantee of production accuracy; holiday transitions have only one year of representation. The fold results are reproducible with `retail-black-box rolling` and source CSVs.


## 8. Final candidate audit (October 2026)

From an independent output directory, the original data were reprocessed using the version 1.2 code. The incident output files and February forecasting results were byte-for-byte identical to the supplied artifacts. Nine-month rolling predictions were also byte-for-byte identical; the monthly summary had no numerical differences, only CSV formatting differences. Twelve existing unit tests passed.

Two additional unit tests explicitly probe temporal isolation: modifying target-day revenue must not change its preceding as-of model features, and altering target or future-day amounts must not affect a historical-only outage estimate. These are useful regressions, not formal proofs covering all possible forms of leakage.

Only a clarity fix was made to the Streamlit interface: the per-category table after the rolling view now explicitly says it is the **February 2024 fixed-cutoff evaluation**. No ML hyperparameters, scoring logic or training procedure changed in this final audit patch.
