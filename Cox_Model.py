"""National time-varying Cox models, extracted from Step3_Cox_Event.py.

This is the adjusted specification present in the source. Its correspondence to
the accompanying historical results must be checked; see README.md.
No individual-level input data are distributed with this script.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

THRESHOLDS = [
    "Rain_50mm_Heat_35C", "Rain_90_Heat_95", "Rain_95_Heat_90",
    "Rain_95_Heat_95", "Rain_99_Heat_95",
]
EXPOSURES = ["H_R", "R_H", "RHin", "Multiple", "Binary"]
COLUMNS = [
    "HR", "HR_lower", "HR_upper", "p", "n_events", "n_individuals",
    "Threshold", "Lag", "Lag_Num", "Exposure_Type", "HR_String", "Sig",
]

def fit_cox_model(file_path):
    """Fit the historical adjusted model; return None for the source-defined exclusions/failures."""
    try:
        df = pd.read_csv(file_path)
        df = df.rename(columns={'exposure_status': 'Exposure', 'trueage': 'Age'})
        if 'a1' in df.columns:
            df['Female'] = df['a1'].map({1: 0, 2: 1})
        elif 'sex' in df.columns:
            df['Female'] = df['sex']
        else:
            return None
        if 'residenc' in df.columns:
            df['Is_Urban'] = (df['residenc'] == 1).astype(int)
        else:
            df['Is_Urban'] = 0
        adl_cols = [f'e{i}' for i in range(1, 7)]
        iadl_cols = [f'e{i}' for i in range(7, 15)]
        valid_adl = [c for c in adl_cols if c in df.columns]
        valid_iadl = [c for c in iadl_cols if c in df.columns]
        df['ADL'] = df[valid_adl].sum(axis=1) if valid_adl else 0
        df['IADL'] = df[valid_iadl].sum(axis=1) if valid_iadl else 0
        cols_needed = ['id', 'event', 'tstart', 'tstop', 'Exposure', 'Age', 'Female', 'Is_Urban', 'ADL', 'IADL', 'prov']
        missing_cols = [c for c in cols_needed if c not in df.columns]
        if missing_cols:
            return None
        df_model = df[cols_needed].dropna()
        if len(df_model) == 0:
            return None
        n_events = int(df_model['event'].sum())
        n_individuals = df_model['id'].nunique()
        if n_events < 10:
            return None
        ctv = CoxTimeVaryingFitter()
        ctv.fit(df_model, id_col='id', event_col='event', start_col='tstart', stop_col='tstop', strata=['prov'], formula='Exposure + Age + Female + Is_Urban + ADL + IADL', show_progress=False)
        if 'Exposure' not in ctv.summary.index:
            return None
        coef = ctv.summary.loc['Exposure', 'coef']
        se = ctv.summary.loc['Exposure', 'se(coef)']
        p = ctv.summary.loc['Exposure', 'p']
        hr = np.exp(coef)
        hr_lower = np.exp(coef - 1.96 * se)
        hr_upper = np.exp(coef + 1.96 * se)
        if np.isnan(hr) or np.isinf(hr) or hr > 100 or (hr < 0.01):
            return None
        if hr_upper > 20:
            hr_upper = 20
        if hr_lower < 0.05:
            hr_lower = 0.05
        return {'HR': hr, 'HR_lower': hr_lower, 'HR_upper': hr_upper, 'p': p, 'n_events': n_events, 'n_individuals': n_individuals}
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of authorized individual-level Cox input CSVs.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output CSV path; existing files are never overwritten.")
    parser.add_argument("--threshold", choices=THRESHOLDS)
    parser.add_argument("--gap", type=int, choices=range(1, 8))
    parser.add_argument("--event", choices=EXPOSURES)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new path; do not overwrite the supplied results.")
    if not args.input_dir.is_dir():
        parser.error("Input directory does not exist.")
    thresholds = [args.threshold] if args.threshold else THRESHOLDS
    gaps = [args.gap] if args.gap else list(range(1, 8))
    exposures = [args.event] if args.event else EXPOSURES
    results, failed = [], []
    for threshold in thresholds:
        for gap in gaps:
            for exposure in exposures:
                path = args.input_dir / f"cox_data_{threshold}_Gap_{gap}days_{exposure}.csv"
                if not path.exists():
                    path = args.input_dir / f"cox_data_{threshold}_Gap_{gap}day_{exposure}.csv"
                if not path.exists():
                    failed.append((threshold, gap, exposure, "missing input"))
                    continue
                print(f"Fitting {threshold}, D{gap}, {exposure}", flush=True)
                result = fit_cox_model(path)
                if result is None:
                    failed.append((threshold, gap, exposure, "source routine skipped or failed"))
                    continue
                result.update({
                    "Threshold": threshold, "Lag": f"D{gap}", "Lag_Num": gap,
                    "Exposure_Type": exposure, "HR_String": f"{result['HR']:.2f}",
                    "Sig": "***" if result["p"] < 0.001 else "**" if result["p"] < 0.01 else "*" if result["p"] < 0.05 else "",
                })
                results.append(result)
    if failed:
        for item in failed:
            print("FAILED/SKIPPED:", item)
        raise SystemExit("No output written: the requested grid was incomplete. Inspect source/data before retrying.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as handle:
        pd.DataFrame(results, columns=COLUMNS).to_csv(handle, index=False)
    print(f"Saved {len(results)} model estimates to {args.output}")


if __name__ == "__main__":
    main()
