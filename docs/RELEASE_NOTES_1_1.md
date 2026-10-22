# Retail Black Box 1.1 — dashboard review

## What changed
- All four headline KPIs now respect region, store and category filters. The unchanged, whole-chain quality report remains accessible separately.
- Incident triage ranks **missing coverage** by expected unobserved volume, not asserted sales loss. A selectable case opens a temporal evidence trace and category-level method/support details.
- Forecast WAPE (LightGBM and seasonal naive), improvement and sample counts are computed for the active selection, rather than showing chain-wide numbers for a single store.
- Replay is preserved, and now reports a WAPE for the selected simulated incident. The stored empirical p10/p90 coverage is clearly labeled uncalibrated.
- CSV export is available for incident queues and scoped evaluation predictions.
- Independent, data-only tests cover KPI scope, category-scope incident amounts and WAPE.

## What has NOT changed
- Original CSVs, estimation routines, training pipeline, and historical model output are unchanged.
- Historical holdout remains **February 2024 only**. This is not a live prediction or a calibrated probabilistic forecast.
- The historical reconstructed volume is **not** lost revenue or a cash recovery estimate.

## Deployment
Replace only `app.py`, and add `src/retail_black_box/dashboard_metrics.py`, `tests/test_dashboard_metrics.py` and this changelog. Streamlit Community Cloud typically redeploys commits to `main` automatically. The three input CSVs and output result files already present in GitHub are not replaced.
