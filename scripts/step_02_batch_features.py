# scripts/step_02_batch_features.py
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from step_01_load_raw import (
    ROOT, SET_A_DIR, OUTCOMES_PATH,
    load_outcomes, load_patient_long, infer_patient_id_from_filename
)

# pick a small, extubation-relevant starter set (present in many files)
STARTER_VARS = [
    "HR", "RespRate", "NIMAP", "NISysABP", "NIDiasABP",  # vitals/bp
    "GCS", "Na", "K", "Mg", "Creatinine", "BUN", "Platelets", "HCT",  # labs
    "HCO3", "Glucose", "SaO2", "FiO2", "Lactate", "pH",  # gases/metabolic (may be missing in many)
    "MechVent",  # special (binary)
]

MAX_PATIENTS = None  # keep it small for the first pass

def summarise_patient(long_df: pd.DataFrame) -> dict:
    """Return a small feature dict for a single patient's long-format df."""
    feat = {}
    pnames = long_df["Parameter"].astype(str)

    for var in STARTER_VARS:
        sub = long_df.loc[pnames.str.upper() == var.upper(), "Value"]

        if sub.empty:
            feat[f"{var}_was_measured"] = 0
            feat[f"{var}_mean"] = np.nan
            feat[f"{var}_last"] = np.nan
            continue

        feat[f"{var}_was_measured"] = 1

        # last value by time ordering
        subw = long_df.loc[pnames.str.upper() == var.upper(), ["Time", "Value"]].sort_values("Time")
        last_val = subw["Value"].iloc[-1]

        # numeric mean if numeric; otherwise NA
        if pd.api.types.is_numeric_dtype(subw["Value"]):
            mean_val = subw["Value"].mean()
        else:
            # try to coerce to numeric in case mixed types
            mean_val = pd.to_numeric(subw["Value"], errors="coerce").mean()

        feat[f"{var}_mean"] = mean_val
        feat[f"{var}_last"] = last_val

        # special handling for MechVent (binary)
        if var.lower() == "mechvent":
            # proportion of time on vent (approximate as mean of 0/1 values)
            mv = pd.to_numeric(subw["Value"], errors="coerce")
            feat["MechVent_prop_on"] = float(mv.mean()) if mv.notna().any() else pd.NA
            feat["MechVent_last"] = float(mv.iloc[-1]) if mv.notna().any() else pd.NA

        # FiO2 scaling check (if present and numeric > 1.5, convert %→fraction)
        if var.lower() == "fio2" and pd.notna(mean_val):
            # recompute mean/last in fraction if appears in percent
            s = pd.to_numeric(subw["Value"], errors="coerce")
            if s.notna().any() and s.max() > 1.5:
                s = s / 100.0
                feat[f"{var}_mean"] = float(s.mean())
                feat[f"{var}_last"] = float(s.iloc[-1])

    return feat

if __name__ == "__main__":
    # Collect files (optionally cap via MAX_PATIENTS)
    raw_files = sorted(SET_A_DIR.glob("*.txt"))
    if isinstance(MAX_PATIENTS, int) and MAX_PATIENTS > 0:
        raw_files = raw_files[:MAX_PATIENTS]

    print(f"Found {len(raw_files)} patient files. Beginning feature extraction...\n")

    outcomes = load_outcomes(OUTCOMES_PATH)

    rows = []
    skipped = []

    for f in tqdm(raw_files, desc="Processing patients", unit="file"):
        try:
            df_long = load_patient_long(f)
        except Exception:
            skipped.append(f.name)
            continue

        pid = infer_patient_id_from_filename(f)
        feats = summarise_patient(df_long)
        feats["patient_id"] = pid
        rows.append(feats)

    if not rows:
        raise RuntimeError("No patients parsed for feature building.")

    feat_df = pd.DataFrame(rows)
    # Merge labels; drop unlabeled
    ds = feat_df.merge(outcomes, how="inner", on="patient_id").dropna(subset=["outcome"]).reset_index(drop=True)

    # --- save feature table (with fallback to CSV if Parquet fails) ---
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Choose a filename that reflects scope
    out_name = "features_full.parquet" if not MAX_PATIENTS else f"features_{len(ds)}.parquet"

    try:
        processed_path = processed_dir / out_name
        ds.to_parquet(processed_path, index=False)
        print(f"Built feature table: {ds.shape[0]} patients x {ds.shape[1]} columns")
        print(ds.head(3).T.head(20))  # quick peek
        print(f"Skipped files (unparsable): {len(skipped)}")
        if skipped:
            print("Examples:", skipped[:5])
        print(f"Saved to: {processed_path}")
    except Exception as e:
        print(f"Parquet save failed ({e}). Falling back to CSV.")
        out_name_csv = out_name.replace(".parquet", ".csv")
        processed_path = processed_dir / out_name_csv
        ds.to_csv(processed_path, index=False)
        print(f"Built feature table: {ds.shape[0]} patients x {ds.shape[1]} columns")
        print(ds.head(3).T.head(20))  # quick peek
        print(f"Skipped files (unparsable): {len(skipped)}")
        if skipped:
            print("Examples:", skipped[:5])
        print(f"Saved to: {processed_path}")
