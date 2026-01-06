# app/app.py
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Path setup (robust imports)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project root (one level above /app)
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Import your existing functions
# These modules must exist in /scripts with these function names.
from step_01_load_raw import load_patient_long  # type: ignore
from step_02_batch_features import summarise_patient  # type: ignore

DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_MEDIUM_THRESHOLD = 0.10  # your “medium risk” threshold


@st.cache_resource
def load_bundle(bundle_path: Path) -> dict:
    bundle = joblib.load(bundle_path)
    return bundle


def align_and_impute(feats: dict, feature_columns: list[str], imputer) -> tuple[pd.DataFrame, int, list[str]]:
    """
    Build a one-row dataframe in training feature order,
    report missing features, and return imputed frame.
    """
    X = pd.DataFrame([feats])

    # Ensure all expected columns exist
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # Keep ONLY training columns in the correct order
    X = X[feature_columns]

    # Missing feature reporting (pre-imputation)
    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())
    missing_count = int(missing_mask.sum())

    # Convert any pd.NA weirdness → np.nan and coerce to numeric
    X = X.where(pd.notna(X), np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Impute
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)

    return X_imp, present_count, missing_features


def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")

st.title("ICU Mortality Risk Demo (Portfolio App)")
st.caption("Upload a single patient `.txt` file and get a calibrated probability + risk band.")

with st.expander("What this is (and isn’t)"):
    st.write(
        """
        This is a **portfolio demo** using a model trained on a public ICU dataset.
        It is **not** a validated clinical device. It’s meant to show end-to-end ML delivery:
        parsing → features → imputation → prediction → thresholding → risk bands.
        """
    )

# Bundle path input
bundle_path_str = st.text_input("Model bundle path", value=str(DEFAULT_BUNDLE_PATH))
bundle_path = Path(bundle_path_str)

if not bundle_path.exists():
    st.error(f"Bundle not found: {bundle_path}")
    st.stop()

bundle = load_bundle(bundle_path)

# Pull items from bundle
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.5))
thr_medium = float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))

starter_vars = bundle.get("starter_vars", None)

st.subheader("Loaded bundle")
col1, col2, col3 = st.columns(3)
col1.metric("High-risk threshold", f"{thr_high:.3f}")
col2.metric("Medium threshold", f"{thr_medium:.3f}")
col3.metric("Features expected", f"{len(feature_columns)}")

if starter_vars is None:
    st.warning("Bundle does not store `starter_vars` (feature engineering drift risk).")
else:
    st.info(f"Bundle starter_vars loaded: {len(starter_vars)} variables")

st.divider()

uploaded = st.file_uploader("Upload patient file (.txt)", type=["txt"])

if uploaded is None:
    st.stop()

# Save uploaded file to a temp location because your loader expects a path
with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = Path(tmp.name)

st.write(f"**File received:** `{uploaded.name}`")

try:
    long_df = load_patient_long(tmp_path)
    feats = summarise_patient(long_df)

    X_imp, present_count, missing_features = align_and_impute(feats, feature_columns, imputer)

    prob = float(model.predict_proba(X_imp)[:, 1][0])
    band = risk_band(prob, thr_medium, thr_high)
    binary_pred = int(prob >= thr_high)

    st.subheader("Prediction")
    st.metric("Predicted mortality risk (probability)", f"{prob:.6f}")

    band_color = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[band]
    st.write(f"**Risk band:** {band_color} **{band} RISK**")
    st.write(f"**Binary prediction (>= HIGH threshold):** {'HIGH RISK (1)' if binary_pred==1 else 'LOW RISK (0)'}")

    st.subheader("Feature completeness")
    missing_count = len(missing_features)
    st.write(f"**Features present:** {present_count}/{len(feature_columns)}")
    st.write(f"**Missing features (imputed):** {missing_count}")

    if missing_count > 0:
        with st.expander("Show missing features"):
            st.write(missing_features)

    # Optional: show a preview of engineered features (safe + helpful)
    with st.expander("Show engineered features (raw one-row, pre-imputation)"):
        X_raw = pd.DataFrame([feats])
        st.dataframe(X_raw)

except Exception as e:
    st.error("Something went wrong while predicting.")
    st.exception(e)
finally:
    # Clean up temp file
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass
