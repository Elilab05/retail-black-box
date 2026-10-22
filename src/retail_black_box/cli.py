"""Reproducible command line experiments (no mutation of source CSVs)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_data
from .quality import quality_report
from .incidents import identify_incidents
from .experiments import masked_outage_experiment


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Retail Black Box - audits and historical experiments")
    p.add_argument("command", choices=["quality", "incidents", "backtest", "forecast", "rolling", "all"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--samples", type=int, default=120, help="Masked complete store-days")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-ml", action="store_true", help="Only relevant to 'all'")
    p.add_argument("--start-month", default="2023-06", help="First month for rolling validation")
    p.add_argument("--end-month", default="2024-02", help="Last month for rolling validation")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t, s, c = load_data(args.data_dir)
    if args.command in ["quality", "all"]:
        report = quality_report(t, s, c)
        save_json(out / "quality_report.json", report)
        print({k: v for k, v in report.items() if isinstance(v, int)})
    if args.command in ["incidents", "all"]:
        incident_days, category_details = identify_incidents(t, s, c)
        incident_days.to_csv(out / "incidents.csv", index=False)
        category_details.to_csv(out / "incident_estimates_by_category.csv", index=False)
        print(f"Incidents: {len(incident_days)} store-days, estimated volume {incident_days['estimate_mxn'].sum():,.2f} MXN")
    if args.command in ["backtest", "all"]:
        report, rows = masked_outage_experiment(t, s, c, n_days=args.samples, seed=args.seed)
        save_json(out / "backtest_report.json", report)
        rows.to_csv(out / "backtest_rows.csv", index=False)
        print(f"Masked outage WAPE: {report['wape']:.2%}")
    if args.command == "rolling":
        from .rolling_validation import run_rolling_backtests
        summary, predictions = run_rolling_backtests(t, s, c, args.start_month, args.end_month)
        summary.to_csv(out / "rolling_validation.csv", index=False)
        predictions.to_csv(out / "rolling_predictions.csv", index=False)
        total_actual = summary["sum_abs_actual"].sum()
        ml = summary["sum_abs_error_ml"].sum() / total_actual
        base = summary["sum_abs_error_baseline"].sum() / total_actual
        print(f"Rolling WAPE: ML {ml:.2%}, baseline {base:.2%}; {len(summary)} month(s) saved")
    if args.command == "forecast" or (args.command == "all" and not args.skip_ml):
        from .forecast import evaluate_7d
        report, rows = evaluate_7d(t, s, c)
        save_json(out / "forecast_report.json", report)
        rows.to_csv(out / "forecast_predictions.csv", index=False)
        print(f"Forecast WAPE model: {report['wape_ml']:.2%} | seasonal naive: {report['wape_seasonal_naive']:.2%}")


if __name__ == "__main__":
    main()

