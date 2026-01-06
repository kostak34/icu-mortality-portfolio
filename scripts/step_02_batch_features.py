# scripts/step_02_batch_features.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# Imports that work BOTH:
# - in Streamlit Cloud (package imports)
# - when you run locally in PyCharm/terminal (direct script run)
# -----------------------------
try:
    # Preferred: package import (works in Streamlit Cloud)
    from scripts.step_01_load_raw import (
        ROOT,
        SET_A_DIR,
        OUTCOMES_PATH,
        load_outcomes,
        load_patient_long,
        infer_patient_id_from_filename,
    )
except ModuleNotFoundError:
    # Fallback: direct import (sometimes needed when running file directly)
    from step_01_load_raw import (  # type: ignore
        ROOT,
        SET_A_DIR,
        OUTCOMES_PATH,
        load_outcomes,
        load_patient_long,
        infer_patient_id_from_filename,
    )

# -----------------------------
# Feature configuration
# -----------------------------
STARTER_VARS = [
    "HR", "RespRate", "NIMAP", "NISysABP", "NIDiasABP",  # vitals/bp
    "GCS", "Na", "K", "Mg", "Creatinine", "BUN", "Platelets", "HCT",  # labs
    "HCO3", "Glucose", "SaO2", "FiO2", "Lactate", "pH",  # gases/metabolic
    "MechVent",  # special (binary)
]

MAX_PATIENTS = 500  # change to None / remove cap if you want full run


def summarise_patient(long_df: pd.DataFrame) -> dict:
    """Return a small feature dict for a single patient's long-format df."""
    feat: dict = {}
    pnames = long_df["Parameter"].astype(str)

    for var in STARTER_VARS:
        sub = long_df.loc[pnames.str.upper() == var.upper(), "Value"]

        if sub.empty:
            feat[f"{var}_was_measured"] = 0
            feat[f"{var}_mean"] = pd.NA
            feat[f"{var}_last"] = pd.NA
            continue

        feat[f"{var}_was_measured"] = 1

        # last value by time ordering
        subw = long_df.loc[pnames.str.upper() == var.upper(), ["Time", "Value"]].sort_values("Time")
        last_val = subw["Value"].iloc[-1]

        # numeric mean if numeric; otherwise attempt coerce
        if pd.api.types.is_numeric_dtype(subw["Value"]):
            mean_val = subw["Value"].mean()
        else:
            mean_val = pd.to_numeric(subw["Value"], errors="coerce").mean()

        feat[f"{var}_mean"] = mean_val
        feat[f"{var}_last"] = last_val

        # special handling for MechVent (binary)
        if var.lower() == "mechvent":
            mv = pd.to_numeric(subw["Value"], errors="coerce")
            feat["MechVent_prop_on"] = float(mv.mean()) if mv.notna().any() else pd.NA
            feat["MechVent_last"] = float(mv.iloc[-1]) if mv.notna().any() else pd.NA

        # FiO2 scaling check (if present and numeric > 1.5, convert %→fraction)
        if var.lower() == "fio2" and pd.notna(mean_val):
            s = pd.to_numeric(subw["Value"], errors="coerce")
            if s.notna().any() and s.max() > 1.5:
                s = s / 100.0
                feat[f"{var}_mean"] = float(s.mean())
                feat[f"{var}_last"] = float(s.iloc[-1])

    return feat


def main():
    raw_files = sorted(Path(SET_A_DIR).glob("*.txt"))
    outcomes = load_outcomes(OUTCOMES_PATH)

    rows = []
    count = 0
    skipped = 0

    for f in raw_files:
        if MAX_PATIENTS is not None and count >= MAX_PATIENTS:
            break

        try:
            df_long = load_patient_long(f)
        except Exception:
            skipped += 1
            continue

        pid = infer_patient_id_from_filename(f)
        feats = summarise_patient(df_long)
        feats["patient_id"] = pid
        rows.append(feats)
        count += 1

    if not rows:
        raise RuntimeError("No patients parsed for feature building.")

    feat_df = pd.DataFrame(rows)

    # Merge labels; drop unlabeled
    ds = feat_df.merge(outcomes, how="inner", on="patient_id").dropna(subset=["outcome"]).reset_index(drop=True)

    # Save
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        processed_path = processed_dir / "mini_features.parquet"
        ds.to_parquet(processed_path, index=False)
        print(f"Built mini feature table: {ds.shape[0]} patients x {ds.shape[1]} columns")
        print(ds.head(3).T.head(20))
        print(f"Skipped files (unparsable): {skipped}")
        print(f"Saved to: {processed_path}")
    except Exception as e:
        print(f"Parquet save failed ({e}). Falling back to CSV.")
        processed_path = processed_dir / "mini_features.csv"
        ds.to_csv(processed_path, index=False)
        print(f"Built mini feature table: {ds.shape[0]} patients x {ds.shape[1]} columns")
        print(ds.head(3).T.head(20))
        print(f"Skipped files (unparsable): {skipped}")
        print(f"Saved to: {processed_path}")


if __name__ == "__main__":
    main()
