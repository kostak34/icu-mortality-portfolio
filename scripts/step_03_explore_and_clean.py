# scripts/step_03_explore_and_clean.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from step_01_load_raw import ROOT

# --- load the processed dataset ---
processed_dir = ROOT / "data" / "processed"
parquet_path = processed_dir / "mini_features.parquet"
csv_path = processed_dir / "mini_features.csv"

processed_dir = ROOT / "data" / "processed"

# Prefer the full feature table if available
full_path = processed_dir / "features_full.parquet"
mini_path = processed_dir / "mini_features.parquet"

if full_path.exists():
    df = pd.read_parquet(full_path)
    print("Loaded FULL feature table.")
elif mini_path.exists():
    df = pd.read_parquet(mini_path)
    print("Loaded MINI feature table (small subset).")
else:
    raise FileNotFoundError("No processed feature table found. Run step_02_batch_features first.")

print(f"Loaded feature table: {df.shape[0]} patients x {df.shape[1]} columns\n")
print(df['outcome'].value_counts(dropna=False))


# --- peek at structure ---
print(df.dtypes.head(10))
print("\nPreview of first few rows:")
print(df.head(3))

# --- check for missingness ---
missing = df.isna().mean().sort_values(ascending=False)
print("\nMissingness (fraction of NaN per column, top 15):")
print(missing.head(15))

plt.figure(figsize=(10,5))
missing.head(30).plot(kind='bar')
plt.title("Fraction of missing values (top 30 features)")
plt.tight_layout()
plt.show()

# --- basic numeric summary ---
numeric_cols = df.select_dtypes(include=[np.number]).columns
summary = df[numeric_cols].describe().T
summary["missing_fraction"] = df[numeric_cols].isna().mean()
print("\nNumeric feature summary (first 15 rows):")
print(summary.head(15))

# --- simple cleaning rules ---
# Drop columns that are >80% missing
keep_cols = missing[missing < 0.8].index
df_clean = df[keep_cols].copy()

# Impute remaining numeric NaNs with median
num_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[num_cols] = df_clean[num_cols].apply(lambda col: col.fillna(col.median()))

# Confirm cleaning results
print(f"\nAfter cleaning: {df_clean.shape[1]} columns remain.")
print(f"Any NaNs left? {df_clean.isna().sum().sum()}")

# --- save cleaned dataset ---
processed_dir = ROOT / "data" / "processed"

# Name the cleaned file based on what we loaded
if (processed_dir / "features_full.parquet").exists():
    clean_path = processed_dir / "features_full_clean.parquet"
else:
    clean_path = processed_dir / "mini_features_clean.parquet"

df_clean.to_parquet(clean_path, index=False)
print(f"Saved cleaned feature table to: {clean_path}")

