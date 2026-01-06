"""
Scan a folder of patient .txt files, compute predicted risk, assign LOW/MEDIUM/HIGH,
and save results to outputs/examples_by_riskband.csv.

Usage (from project root):
  py find_examples_by_band.py
  py find_examples_by_band.py --max_files 500
  py find_examples_by_band.py --low 0.10 --high 0.219
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib

# --- Make /scripts importable no matter where you run from ---
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore


def build_feature_row(feats: dict, feature_columns: list[str]) -> tuple[pd.DataFrame, int, int, list[str]]:
    """Return X (1 row, ordered), present_count, missing_count, missing_feature_names."""
    X = pd.DataFrame([feats])

    # Ensure all expected columns exist (use np.nan, not pd.NA)
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_columns]

    # Coerce to numeric (anything non-numeric becomes NaN)
    X = X.where(pd.notna(X), np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    present = int(X.notna().sum(axis=1).iloc[0])
    total = int(X.shape[1])
    missing = total - present
    missing_names = [c for c in feature_columns if pd.isna(X.iloc[0][c])]

    return X, present, missing, missing_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/raw/set-a", help="Folder containing patient .txt files")
    ap.add_argument("--bundle", type=str, default="outputs/model_bundle.joblib", help="Path to saved model bundle")
    ap.add_argument("--low", type=float, default=0.10, help="LOW threshold: prob < low")
    ap.add_argument("--high", type=float, default=None, help="HIGH threshold override (default: bundle default_threshold)")
    ap.add_argument("--max_files", type=int, default=300, help="How many files to scan")
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    bundle_path = PROJECT_ROOT / args.bundle

    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_columns = bundle["feature_columns"]

    # Thresholds
    thr_high = args.high
    if thr_high is None:
        thr_high_raw = bundle.get("default_threshold", 0.5)
        thr_high = float(thr_high_raw) if thr_high_raw is not None else 0.5

    thr_low = float(args.low)

    print(f"Using thresholds: LOW < {thr_low:.2f}, MEDIUM [{thr_low:.2f}..{thr_high:.3f}), HIGH >= {thr_high:.3f}\n")

    starter_vars = bundle.get("starter_vars")
    if starter_vars is None:
        print("WARNING: Bundle does not store starter_vars (feature engineering drift possible).")
    else:
        print(f"Bundle starter_vars loaded: {len(starter_vars)} variables\n")

    files = sorted(data_dir.glob("*.txt"))[: args.max_files]
    if not files:
        raise RuntimeError(f"No .txt files found in {data_dir}")

    rows = []
    band_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    for f in files:
        try:
            long_df = load_patient_long(f)
            feats = summarise_patient(long_df)

            X_row, present, missing, missing_names = build_feature_row(feats, feature_columns)
            X_imp = pd.DataFrame(imputer.transform(X_row), columns=feature_columns)

            prob = float(model.predict_proba(X_imp)[:, 1][0])

            if prob >= thr_high:
                band = "HIGH"
            elif prob >= thr_low:
                band = "MEDIUM"
            else:
                band = "LOW"

            band_counts[band] += 1

            rows.append({
                "patient_file": f.name,
                "prob": prob,
                "risk_band": band,
                "features_present": present,
                "features_missing": missing,
                "missing_features_first10": ";".join(missing_names[:10]),
            })

        except Exception as e:
            rows.append({
                "patient_file": f.name,
                "prob": np.nan,
                "risk_band": "ERROR",
                "features_present": np.nan,
                "features_missing": np.nan,
                "missing_features_first10": f"{type(e).__name__}: {e}",
            })

    df_out = pd.DataFrame(rows).sort_values(["risk_band", "prob"], ascending=[True, False])

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "examples_by_riskband.csv"
    df_out.to_csv(out_path, index=False)

    print("Band counts:", band_counts)
    print(f"Saved: {out_path}")

    # Print one example per band (if available) for quick sanity-check
    print("\nOne example per band (if found):")
    for band in ["LOW", "MEDIUM", "HIGH"]:
        sub = df_out[df_out["risk_band"] == band].dropna(subset=["prob"])
        if len(sub) > 0:
            r = sub.iloc[0]
            print(f"  {band}: {r['patient_file']}   prob={r['prob']:.6f}")
        else:
            print(f"  {band}: (none found)")


if __name__ == "__main__":
    main()
