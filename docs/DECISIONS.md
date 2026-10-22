# Key design decisions and defensibility

## Why not forecasting alone?

The problem permits any business solution. A retail model cannot be trusted if missing days are quietly treated as zero sales. This project pairs data reliability with historical forecasting rather than promising inventory optimization without inventory ground truth.

## Why the 42 missing records matter

The complete panel should contain `80 stores × 425 days × 6 categories = 204,000` rows; only 203,958 are observed. All 42 missing records are concentrated in 7 whole store-days at 3 stores. This suggests full coverage gaps worth investigating, not a random sprinkling of item-level zeros.

## Diagnosing vs claiming causes

A full day absent across all categories is consistent with an ingestion interruption, a point-of-sale connectivity fault or a source-system problem, among other causes. It is **not proof** of any of them. Likewise, the forecasted value is not a measurement of revenue lost.

## Two independent quality channels

The counts of cash/card transactions can disagree with total transactions while the values of cash/card amounts still sum to the total sales amount. The interface reports these as separate checks. Whether discrepancies arise from data quality, reporting logic or definitional differences remains to be investigated.

## Error metric interpretation

`WAPE = sum(abs(y - yhat)) / sum(abs(y))`. WAPE makes aggregate retail scale intuitive but can hide small-store mistakes. Future evaluation should add per-store WAPE, bias, weekend vs weekday cohorts and confidence/prediction-interval coverage.

## Causal validity and action rules

- Start with an incident hypothesis, not an alleged root cause.
- Present observations, estimate method, sample support and uncertainty caveat.
- Escalate to review source availability before triggering promotions, staffing changes or replenishment.
- Do not claim the model reduces stockouts or improves profit without experimental measurement.

## Security and publishing

No personal identifiers appear in the supplied aggregate schema. The public deliverable requests a complete, reproducible solution, so this submission includes the three original CSVs; that is a judgment about the exercise, not proof of redistribution rights. Remove on challenge-owner request. GitHub CI uses synthetic fixture data.
