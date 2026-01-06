"""
CLI predictor: load a saved model bundle and predict from a single patient text file.

Usage:
  python predict_patient_file.py --patient_file data/raw/set-a/132539.txt
  python predict_patient_file.py --patient_file data/raw/set-a/132539.txt --threshold 0.223
  python predict_patient_file.py --patient_file data/raw/set-a/132539.txt --bundle outputs/model_bundle.joblib
"""
import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Ensure imports work regardless of where you run this from
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent  # assumes this file sits in project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# If you ever run from inside /scripts, this also helps
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Now these should work if scripts/ has __init__.py (package style)
from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore


def safe_float(x, default: float = 0.5) -> float:
    """Convert thresholds safely (handles None, pd.NA/NaN, strings)."""
    try:
        if x is None:
            return default
        # pd.isna handles NaN and pandas NAType
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient_file", type=str, required=True, help="Path to a single patient .txt file")
    ap.add_argument("--bundle", type=str, default="outputs/model_bundle.joblib", help="Saved model bundle path")
    ap.add_argument("--threshold", type=float, default=None, help="Override decision threshold (optional)")
    args = ap.parse_args()

    patient_path = Path(args.patient_file)
    bundle_path = Path(args.bundle)

    if not patient_path.exists():
        raise FileNotFoundError(f"Patient file not found: {patient_path}")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")

    # -------------------------
    # Load model bundle
    # -------------------------
    bundle = joblib.load(bundle_path)

    starter_vars = bundle.get("starter_vars")
    if starter_vars is None:
        print("WARNING: This bundle does not store starter_vars. Feature engineering drift is possible.")
    else:
        print(f"Bundle starter_vars loaded: {len(starter_vars)} variables")

    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_columns = bundle["feature_columns"]

    # Threshold: CLI override > bundle default_threshold > 0.5
    thr_raw = args.threshold if args.threshold is not None else bundle.get("default_threshold", 0.5)
    thr = safe_float(thr_raw, default=0.5)

    # -------------------------
    # Load + featurise patient
    # -------------------------
    long_df = load_patient_long(patient_path)
    feats = summarise_patient(long_df)

    X = pd.DataFrame([feats])

    # Ensure all expected columns exist, and in the right order
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_columns]

    present = int(X.notna().sum(axis=1).iloc[0])
    total = int(X.shape[1])
    print(f"Features present: {present}/{total}")
    missing_cols = X.columns[X.isna().iloc[0]].tolist()
    print(f"Missing features (imputed): {len(missing_cols)}")
    print("First 10 missing:", missing_cols[:10])

    # Coerce to numeric; anything non-numeric becomes NaN (safe for sklearn)
    X = X.apply(pd.to_numeric, errors="coerce")

    # -------------------------
    # Impute + predict
    # -------------------------
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)

    prob = float(model.predict_proba(X_imp)[:, 1][0])
    pred = int(prob >= thr)

    # --- 3-band risk labelling (simple portfolio-friendly version) ---
    low_thr = 0.10
    high_thr = thr  # your calibrated threshold (e.g., 0.219)

    if prob >= high_thr:
        band = "HIGH RISK"
    elif prob >= low_thr:
        band = "MEDIUM RISK"
    else:
        band = "LOW RISK"

    print(f"Patient file: {patient_path.name}")
    print(f"Predicted mortality risk (prob): {prob:.6f}")
    print(f"Threshold (HIGH): {high_thr:.3f}")
    print(f"Threshold (MEDIUM): {low_thr:.2f}")
    print(f"Risk band: {band}")
    print(f"Binary prediction (>= HIGH threshold): {'HIGH RISK (1)' if pred == 1 else 'LOW RISK (0)'}")
    print(f"Action flag (>= HIGH threshold): {'TRIGGER (1)' if pred == 1 else 'NO TRIGGER (0)'}")


if __name__ == "__main__":
    main()
