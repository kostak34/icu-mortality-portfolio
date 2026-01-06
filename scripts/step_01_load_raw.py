# scripts/step_01_load_raw.py
from pathlib import Path
import pandas as pd

# --- set up consistent project root and paths ---
ROOT = Path(__file__).resolve().parents[1]   # this points one level up from /scripts
RAW_DIR = ROOT / "data" / "raw"
SET_A_DIR = RAW_DIR / "set-a"
OUTCOMES_PATH = RAW_DIR / "outcomes_a.txt"
  # rename if your file is outcomes_a.txt

# -------- tiny helpers ----------
def read_table_flexible(path: Path) -> pd.DataFrame:
    """
    Try a few common separators to read txt/csv with unknown delimiter.
    Returns the first successful parse with >= 2 columns.
    """
    for sep in [",", "\t", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    raise ValueError(f"Could not parse file with common separators: {path}")

def load_outcomes(path: Path) -> pd.DataFrame:
    # First try comma-separated (as you observed)
    try:
        df = pd.read_csv(path, sep=",", engine="python")
    except Exception:
        # Fallback: any whitespace (robust to mixed spaces/tabs)
        df = pd.read_csv(path, sep=r"\s+", engine="python")

    # Normalise column names
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    # Identify id/label columns (common names in this dataset)
    id_col = next((c for c in cols if c in ["recordid", "patientid", "id"]), None)
    label_col = next((c for c in cols if c in ["in-hospital_death", "inhospital_death", "outcome", "death"]), None)

    if id_col is None or label_col is None:
        raise ValueError(f"Could not find id/label columns in outcomes file. Found: {cols}")

    out = df[[id_col, label_col]].copy()
    out.columns = ["patient_id", "outcome"]

    # Force types and drop junk
    out["patient_id"] = pd.to_numeric(out["patient_id"], errors="coerce").astype("Int64")
    out["outcome"] = pd.to_numeric(out["outcome"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["patient_id", "outcome"]).reset_index(drop=True)
    return out

def load_patient_long(path: Path) -> pd.DataFrame:
    """
    Robust loader for a single patient's time-series file.
    Handles encodings, header/no-header, commas/tabs/spaces, and time formats (numeric or HH:MM).
    Returns DataFrame with columns: Time (int, minutes), Parameter (str), Value (float or str).
    """
    import os, re
    import pandas as pd

    if not path.exists() or os.path.getsize(path) == 0:
        raise ValueError(f"Empty or missing file: {path}")

    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        # Standardise first 3 columns
        if df.shape[1] >= 3:
            df = df.iloc[:, :3]
        df.columns = ["Time", "Parameter", "Value"]

        # Drop comment/header-like rows
        df = df[~df["Time"].astype(str).str.startswith("#")]
        df = df[df["Parameter"].astype(str).str.lower() != "parameter"]

        # --- Convert Time column ---
        def convert_time_to_minutes(t):
            s = str(t).strip()
            # Case 1: already numeric (e.g. 0, 30, 120)
            if s.replace(".", "", 1).isdigit():
                return float(s)
            # Case 2: HH:MM format
            if ":" in s:
                try:
                    h, m = s.split(":")
                    return int(h) * 60 + int(m)
                except Exception:
                    return None
            # Otherwise skip
            return None

        df["Time"] = df["Time"].apply(convert_time_to_minutes)
        df = df.dropna(subset=["Time"])
        df["Time"] = df["Time"].astype(int)

        # Coerce Value numeric when possible
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        # Clean Parameter text
        df["Parameter"] = df["Parameter"].astype(str).str.strip()
        df = df[(df["Parameter"] != "") & df["Parameter"].notna()]

        if df.empty:
            raise ValueError("empty after cleaning")
        return df.reset_index(drop=True)

    # Try a few parsing strategies
    encodings = ["utf-8", "latin1", "cp1252"]
    for enc in encodings:
        for header in [None, 0]:
            try:
                df = pd.read_csv(
                    path,
                    sep=None, engine="python", header=header,
                    on_bad_lines="skip", encoding=enc
                )
                return _clean_df(df)
            except Exception:
                continue

    # Manual fallback if all else fails
    rows = []
    text = path.read_bytes().decode("utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("time"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            parts = re.split(r"\t+|\s{2,}", line)
        if len(parts) < 3:
            continue
        t, par, val = parts[0].strip(), parts[1].strip(), ",".join(parts[2:]).strip()
        rows.append((t, par, val))

    if not rows:
        raise ValueError(f"Parsed but empty/invalid structure: {path}")

    df = pd.DataFrame(rows, columns=["Time", "Parameter", "Value"])
    return _clean_df(df)

def infer_patient_id_from_filename(path: Path) -> int:
    # e.g., data/raw/set_a/132020.txt -> 132020
    return int(path.stem)

# --- add near the top (after imports/paths) ---
from typing import Optional, List

def find_first_good_file(dirpath) -> tuple[Path, pd.DataFrame, List[str]]:
    bad_samples = []
    for f in sorted(dirpath.glob("*.txt")):
        try:
            df = load_patient_long(f)
            return f, df, bad_samples
        except Exception as e:
            # keep a few errors for visibility, then move on
            if len(bad_samples) < 5:
                bad_samples.append(f"{f.name}: {str(e).splitlines()[0]}")
            continue
    raise RuntimeError("No parsable patient files found.")

# --- replace your current 'pick one file' block in __main__ with this ---
if __name__ == "__main__":
    print("Loading outcomes…")
    outcomes = load_outcomes(OUTCOMES_PATH)
    print(outcomes.head())
    print(f"Outcomes shape: {outcomes.shape}\n")

    print("Scanning for a parsable patient file…")
    example_file, patient_long, bad_notes = find_first_good_file(SET_A_DIR)
    if bad_notes:
        print("Some files failed to parse (showing up to 5):")
        for b in bad_notes:
            print("  -", b)

    pid = infer_patient_id_from_filename(example_file)
    print(f"\nInspecting patient file: {example_file.name} (patient_id={pid})")
    print(patient_long.head(10))
    print(f"\nRows in this patient file: {len(patient_long)}")
    print("Unique parameters (first 20):",
          sorted(patient_long['Parameter'].astype(str).unique())[:20])

    # FiO2 quick stats
    fio2 = patient_long.loc[patient_long["Parameter"].str.lower()=="fio2", "Value"]
    if not fio2.empty and pd.api.types.is_numeric_dtype(fio2):
        mx = fio2.max()
        mn = fio2.min()
        print(f"\nFiO2 quick stats — min: {mn}, max: {mx}")
        if mx > 1.5:
            print("Note: FiO2 likely in percent; we’ll convert to 0–1 later.")
        else:
            print("Note: FiO2 appears to be in fraction (0–1).")
    else:
        print("\nNo numeric FiO2 present in this patient file (that’s fine).")